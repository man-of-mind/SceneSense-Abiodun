#!/usr/bin/env python3
"""Fit the queue/transport model on FIT cells and evaluate the frozen gates.

Model
-----
A standard queueing form: a fixed per-frame overhead plus a transmission term
proportional to the bytes that must clear the UE queue ahead of and including
this frame.

    latency_ns = overhead_ns + slope_ns_per_byte * (backlog + bytes)

`overhead_ns >= 0` and `slope_ns_per_byte > 0` are enforced, so predicted
latency is strictly increasing in both bytes and backlog by construction, and
it interpolates rather than binning either.  Both coefficients are made
non-increasing in the MCS bin by weighted pool-adjacent-violators, which
encodes the physical prior that a higher MCS is never slower.

Shape constraints make the monotonicity and MCS-direction checks structural
rather than empirical.  They do NOT make the predictive gates easier: held-out
next-backlog error, transport-latency error and deadline calibration are all
still measured against the frozen thresholds.

An earlier form without the overhead term (pure `bytes / rate`) was rejected
on FIT diagnosis: at zero backlog a 6.5 kB frame still takes ~39 ms, which a
proportional-only model cannot represent.

Binning and back-off are **not invented here**.  `BACKLOG_EDGES`, `MCS_EDGES`,
`MIN_BIN_SUPPORT` and the back-off order are imported unchanged from the
already-frozen `ue_mcs_backlog_run4_analysis_v1` registration.

Fit uses FIT whole cells only.  VALIDATION cells never influence a bin, a
back-off step, a scale or a model choice.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.ue_mcs_backlog_run4_analysis_v1 import contract as FROZEN

from . import contract as C


ANALYSIS_SCHEMA = "scenesense.production_queue_capture_analysis.v1"

# Imported unchanged from the frozen registration.
BACKLOG_EDGES = FROZEN.BACKLOG_EDGES
MCS_EDGES = FROZEN.MCS_EDGES
MIN_BIN_SUPPORT = FROZEN.MIN_BIN_SUPPORT
BACKOFF_ORDER = FROZEN.BACKOFF_ORDER_WITH_MCS
BACKOFF_ORDER_WITHOUT_MCS = FROZEN.BACKOFF_ORDER_WITHOUT_MCS

CYCLE_SECONDS = C.DURATION_NS / 1e9


class AnalysisError(RuntimeError):
    """A structural expectation about the parsed evidence failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    return max(0, bisect.bisect_right(list(edges), value) - 1)


def backlog_bin(value: float) -> int:
    return _bin_index(float(value), BACKLOG_EDGES)


def mcs_bin(value: float) -> int:
    return _bin_index(float(value), MCS_EDGES)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _pava_non_increasing(
    values: Sequence[float], weights: Sequence[float],
) -> list[float]:
    """Weighted pool-adjacent-violators for a non-increasing sequence."""
    blocks: list[list[float]] = []          # [value, weight]
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight)])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            value_b, weight_b = blocks.pop()
            value_a, weight_a = blocks.pop()
            total = weight_a + weight_b
            blocks.append([(value_a * weight_a + value_b * weight_b) / total,
                           total])
    out: list[float] = []
    for value, weight in blocks:
        out.extend([value] * int(round(weight)))
    return out


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least squares with overhead >= 0 and slope > 0 enforced."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    if slope <= 0.0 or intercept < 0.0:
        # Refit through the origin, which keeps both constraints feasible.
        denominator = sum(x * x for x in xs)
        slope = (sum(x * y for x, y in zip(xs, ys)) / denominator
                 if denominator > 0 else 0.0)
        intercept = 0.0
    require(slope > 0.0, "fitted slope must be strictly positive")
    return max(0.0, intercept), slope


@dataclass(frozen=True)
class LinearLatencyModel:
    """latency = overhead + slope * (backlog + bytes), shape constrained."""

    table: Mapping[tuple, dict[str, Any]]
    global_entry: Mapping[str, Any]
    support: Mapping[str, Any]
    use_mcs: bool

    def _entry(self, mcs: float | None) -> tuple[Mapping[str, Any], str]:
        if self.use_mcs and mcs is not None:
            key = (mcs_bin(mcs),)
            entry = self.table.get(key)
            if entry is not None and entry["count"] >= MIN_BIN_SUPPORT:
                return entry, f"MCS_BIN{key}"
        return self.global_entry, "GLOBAL"

    def predict_latency_ns(
        self, *, backlog: float, bytes_on_wire: int, mcs: float | None,
    ) -> tuple[float, str]:
        entry, source = self._entry(mcs)
        return (entry["overhead_ns"]
                + entry["slope_ns_per_byte"] * (backlog + bytes_on_wire)), source

    def service_rate_bps(self, mcs: float | None) -> float:
        entry, _ = self._entry(mcs)
        return 1e9 / entry["slope_ns_per_byte"]

    def predict_next_backlog(
        self, *, backlog: float, ingress_bytes: int, mcs: float | None,
    ) -> tuple[float, str]:
        entry, source = self._entry(mcs)
        served = self.service_rate_bps(mcs) * CYCLE_SECONDS
        return max(0.0, backlog + ingress_bytes - served), source


def fit_rate_model(fit_frames: Sequence[Mapping[str, Any]],
                   *, use_mcs: bool = True) -> LinearLatencyModel:
    buckets: dict[int, list[tuple[float, float]]] = {}
    everything: list[tuple[float, float]] = []
    for row in fit_frames:
        if not usable_for_rate(row):
            continue
        backlog = float(row["pre_action_rlc_backlog_bytes"])
        x = backlog + row["udp_application_bytes"]
        y = float(row["transport_latency_ns"])
        everything.append((x, y))
        if use_mcs and row["prior_ul_mcs"] is not None:
            buckets.setdefault(mcs_bin(row["prior_ul_mcs"]), []).append((x, y))
    require(bool(everything), "no usable FIT rows to fit the latency model")

    gx = [item[0] for item in everything]
    gy = [item[1] for item in everything]
    g_overhead, g_slope = _fit_line(gx, gy)
    global_entry = {"count": len(everything), "overhead_ns": g_overhead,
                    "slope_ns_per_byte": g_slope}

    ordered_bins = sorted(bucket for bucket, rows in buckets.items()
                          if len(rows) >= MIN_BIN_SUPPORT)
    raw: dict[int, tuple[float, float, int]] = {}
    for bucket in ordered_bins:
        rows = buckets[bucket]
        overhead, slope = _fit_line([item[0] for item in rows],
                                    [item[1] for item in rows])
        raw[bucket] = (overhead, slope, len(rows))

    # Higher MCS is never slower: both coefficients non-increasing in the bin.
    table: dict[tuple, dict[str, Any]] = {}
    if ordered_bins:
        weights = [raw[bucket][2] for bucket in ordered_bins]
        slopes = _pava_non_increasing(
            [raw[bucket][1] for bucket in ordered_bins], weights)
        overheads = _pava_non_increasing(
            [raw[bucket][0] for bucket in ordered_bins], weights)
        cursor = 0
        for bucket, weight in zip(ordered_bins, weights):
            table[(bucket,)] = {
                "count": raw[bucket][2],
                "overhead_ns": overheads[cursor],
                "slope_ns_per_byte": slopes[cursor],
                "raw_overhead_ns": raw[bucket][0],
                "raw_slope_ns_per_byte": raw[bucket][1],
                "implied_rate_mbps": 1e9 / slopes[cursor] * 8 / 1e6,
            }
            cursor += weight

    backlogs = [float(row["pre_action_rlc_backlog_bytes"])
                for row in fit_frames
                if row["pre_action_rlc_backlog_bytes"] is not None]
    return LinearLatencyModel(
        table=table, global_entry=global_entry, use_mcs=use_mcs,
        support={
            "fit_rows_used": len(everything),
            "fit_rows_total": len(fit_frames),
            "min_bytes": min(r["udp_application_bytes"] for r in fit_frames),
            "max_bytes": max(r["udp_application_bytes"] for r in fit_frames),
            "min_backlog": min(backlogs), "max_backlog": max(backlogs),
        })


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int_or_none(value: str) -> int | None:
    return int(value) if value not in ("", "None") else None


def load_rows(parsed_dir: Path) -> tuple[list[dict], list[dict]]:
    frames = []
    for row in _load(parsed_dir / "frames.csv"):
        frames.append({
            "cell_id": row["cell_id"], "partition": row["partition"],
            "tier_audit_only": row["tier_audit_only"],
            "profile_id_audit_only": row["profile_id_audit_only"],
            "frame_index": int(row["frame_index"]),
            "action_id": int(row["action_id"]),
            "total_transmitted_bytes": int(row["total_transmitted_bytes"]),
            "udp_application_bytes": int(row["udp_application_bytes"]),
            "pre_action_rlc_backlog_bytes":
                _int_or_none(row["pre_action_rlc_backlog_bytes"]),
            "prior_ul_mcs": _int_or_none(row["prior_ul_mcs"]),
            "transport_latency_ns": _int_or_none(row["transport_latency_ns"]),
            "terminal_outcome": row["terminal_outcome"],
            "complete": row["complete"] == "True",
            "rlc_ingress_bytes": int(row["rlc_ingress_bytes"]),
            "rlc_service_bytes": int(row["rlc_service_bytes"]),
            "pdcp_ingress_bytes": int(row["pdcp_ingress_bytes"]),
        })
    cycles = []
    for row in _load(parsed_dir / "cycles.csv"):
        cycles.append({
            "cell_id": row["cell_id"], "partition": row["partition"],
            "tier_audit_only": row["tier_audit_only"],
            "cycle_start_index": int(row["cycle_start_index"]),
            "pre_action_rlc_backlog_bytes":
                _int_or_none(row["pre_action_rlc_backlog_bytes"]),
            "prior_ul_mcs": _int_or_none(row["prior_ul_mcs"]),
            "measured_rlc_ingress_bytes":
                int(row["measured_rlc_ingress_bytes"]),
            "measured_rlc_service_bytes":
                int(row["measured_rlc_service_bytes"]),
            "observed_next_backlog_bytes":
                _int_or_none(row["observed_next_backlog_bytes"]),
            "pair_total_transmitted_bytes":
                int(row["pair_total_transmitted_bytes"]),
            "decision_terminal_outcome": row["decision_terminal_outcome"],
        })
    return frames, cycles


def usable_for_rate(row: Mapping[str, Any]) -> bool:
    """Only completed, non-inverted frames carry a measurable rate.

    This selection is used ONLY to fit the rate.  It is never used to report
    latency or deadline outcomes: those are always evaluated on the full sent
    population, which is what the anti-bias rule requires.
    """
    return (row["terminal_outcome"] in ("COMPLETE_WITHIN_DEADLINE",
                                        "COMPLETE_AFTER_DEADLINE")
            and row["transport_latency_ns"] is not None
            and row["transport_latency_ns"] > 0
            and row["pre_action_rlc_backlog_bytes"] is not None)


def _nmae(predicted: Sequence[float], observed: Sequence[float]) -> float:
    denominator = sum(abs(value) for value in observed)
    if denominator <= 0:
        return math.inf
    return sum(abs(p - o) for p, o in zip(predicted, observed)) / denominator


def evaluate(parsed_dir: Path) -> dict[str, Any]:
    frames, cycles = load_rows(parsed_dir)
    parse_report = json.loads(
        (parsed_dir / "PARSE_REPORT.json").read_text(encoding="utf-8"))
    fit_frames = [row for row in frames if row["partition"] == C.FIT]
    val_frames = [row for row in frames if row["partition"] == C.VALIDATION]
    fit_cycles = [row for row in cycles if row["partition"] == C.FIT]
    val_cycles = [row for row in cycles if row["partition"] == C.VALIDATION]
    require(bool(fit_frames) and bool(val_frames),
            "both partitions must be populated")

    model = fit_rate_model(fit_frames, use_mcs=True)
    model_no_mcs = fit_rate_model(fit_frames, use_mcs=False)

    results: dict[str, Any] = {}

    # -- gate 1: completeness ------------------------------------------
    results["gate_1"] = {
        "key": "COMPLETE_AND_SEALED_CAPTURE",
        "cells": len({row["cell_id"] for row in frames}),
        "frames": len(frames), "cycles": len(cycles),
        "passed": (len(frames) == C.EXPECTED_RAW_FRAMES
                   and len(cycles) == C.EXPECTED_PRIMARY_CYCLES
                   and len({row["cell_id"] for row in frames})
                   == C.EXPECTED_CELLS),
    }

    # -- gate 2: same-domain closure seal ------------------------------
    seals = [cell["closure_seal"] for cell in parse_report["cells"]]
    seal_failures = []
    for cell, seal in zip(parse_report["cells"], seals):
        problems = []
        if not seal["wire_bytes_conserved"]:
            problems.append("wire bytes not conserved sender->receiver")
        if not seal["payload_bytes_conserved"]:
            problems.append("payload bytes not conserved sender->receiver")
        if seal["datagrams_dropped_at_socket"] != 0:
            problems.append("sender dropped datagrams at the socket")
        if seal["unexplained_layer_residual_bytes"] != 0:
            problems.append("unexplained sender->PDCP->RLC residual")
        if not seal["observed_ip_fragmentation"]:
            problems.append("IP fragmentation was not observed")
        if seal["frames_with_exactly_one_terminal"] != C.FRAMES_PER_CELL:
            problems.append("not exactly one terminal per sent frame")
        if not seal["terminal_outcomes_registered"]:
            problems.append("unregistered terminal outcome")
        if problems:
            seal_failures.append({"cell_id": cell["cell_id"],
                                  "problems": problems})
    results["gate_2"] = {
        "key": "SAME_DOMAIN_CLOSURE_SEAL",
        "cells_sealed": len(seals),
        "requirements": list(C.CLOSURE_SEAL_REQUIREMENTS),
        "rlc_minus_pdcp_bytes_per_sdu":
            sorted({seal["rlc_minus_pdcp_bytes_per_sdu"] for seal in seals}),
        "total_unexplained_residual_bytes":
            sum(seal["unexplained_layer_residual_bytes"] for seal in seals),
        "failures": seal_failures,
        "passed": not seal_failures and len(seals) == C.EXPECTED_CELLS,
    }

    # -- gate 3: causal coverage ---------------------------------------
    backlog_cov = sum(1 for row in frames
                      if row["pre_action_rlc_backlog_bytes"] is not None)
    mcs_cov = sum(1 for row in frames if row["prior_ul_mcs"] is not None)
    results["gate_3"] = {
        "key": "CAUSAL_INPUT_COVERAGE",
        "backlog_coverage": backlog_cov / len(frames),
        "ul_mcs_coverage": mcs_cov / len(frames),
        "passed": backlog_cov == len(frames) and mcs_cov == len(frames),
    }

    # -- gate 9: boundary integrity ------------------------------------
    inversions = sum(1 for row in frames
                     if row["terminal_outcome"] == "EXCLUDED_INFRASTRUCTURE_FAULT")
    results["gate_9"] = {
        "key": "BOUNDARY_INTEGRITY",
        "excluded_inversions": inversions,
        "fraction": inversions / len(frames),
        "policy": C.BOUNDARY_INVERSION_POLICY,
        "passed": inversions / len(frames) <= C.MAX_BOUNDARY_INVERSION_FRACTION,
    }

    # -- gate 4: held-out next backlog ---------------------------------
    predicted, observed, persistence = [], [], []
    skipped = 0
    for row in val_cycles:
        backlog = row["pre_action_rlc_backlog_bytes"]
        target = row["observed_next_backlog_bytes"]
        if backlog is None or target is None:
            skipped += 1
            continue
        value, _ = model.predict_next_backlog(
            backlog=float(backlog),
            ingress_bytes=row["measured_rlc_ingress_bytes"],
            mcs=row["prior_ul_mcs"])
        predicted.append(value)
        observed.append(float(target))
        persistence.append(float(backlog))
    model_nmae = _nmae(predicted, observed)
    persistence_nmae = _nmae(persistence, observed)
    improvement = ((persistence_nmae - model_nmae) / persistence_nmae
                   if persistence_nmae not in (0.0, math.inf) else 0.0)
    results["gate_4"] = {
        "key": "VALIDATION_NEXT_BACKLOG_ERROR",
        "n": len(predicted), "skipped": skipped,
        "model_nmae": model_nmae, "persistence_nmae": persistence_nmae,
        "improvement_over_persistence": improvement,
        "passed": (model_nmae <= 0.10 and improvement >= 0.20),
    }

    # -- gate 5: held-out transport latency ----------------------------
    errors = []
    for row in val_frames:
        if not usable_for_rate(row):
            continue
        value, _ = model.predict_latency_ns(
            backlog=float(row["pre_action_rlc_backlog_bytes"]),
            bytes_on_wire=row["udp_application_bytes"],
            mcs=row["prior_ul_mcs"])
        errors.append(abs(value - row["transport_latency_ns"]) / 1e6)
    results["gate_5"] = {
        "key": "VALIDATION_TRANSPORT_LATENCY_ERROR",
        "n": len(errors),
        "p50_error_ms": _percentile(errors, 0.50),
        "p95_error_ms": _percentile(errors, 0.95),
        "passed": (bool(errors)
                   and _percentile(errors, 0.50) <= 17.0
                   and _percentile(errors, 0.95) <= 34.0),
    }

    # -- gate 6: deadline outcome on the FULL sent population ----------
    def deadline_scores(active: LinearLatencyModel) -> dict[str, Any]:
        brier_terms, false_success, predicted_success = [], 0, 0
        counted = 0
        for row in val_frames:
            if row["terminal_outcome"] == "EXCLUDED_INFRASTRUCTURE_FAULT":
                continue
            if row["pre_action_rlc_backlog_bytes"] is None:
                continue
            counted += 1
            actual = 1.0 if row["terminal_outcome"] == "COMPLETE_WITHIN_DEADLINE" \
                else 0.0
            value, _ = active.predict_latency_ns(
                backlog=float(row["pre_action_rlc_backlog_bytes"]),
                bytes_on_wire=row["udp_application_bytes"],
                mcs=row["prior_ul_mcs"])
            probability = 1.0 if value <= C.REWARD_DEADLINE_NS else 0.0
            brier_terms.append((probability - actual) ** 2)
            if probability >= 0.5:
                predicted_success += 1
                if actual == 0.0:
                    false_success += 1
        return {
            "n": counted,
            "brier": statistics.fmean(brier_terms) if brier_terms else math.nan,
            "predicted_success": predicted_success,
            "false_success": false_success,
            "false_success_rate": (false_success / predicted_success
                                   if predicted_success else 0.0),
        }

    with_mcs = deadline_scores(model)
    without_mcs = deadline_scores(model_no_mcs)
    results["gate_6"] = {
        "key": "VALIDATION_DEADLINE_OUTCOME_CALIBRATION",
        "population": "ALL_SENT_FRAMES_NOT_SURVIVORS_ONLY",
        **with_mcs,
        "passed": (with_mcs["false_success_rate"] <= 0.05
                   and with_mcs["brier"] <= 0.15),
    }

    # -- gate 7: MCS non-harm and direction ----------------------------
    degradation = with_mcs["brier"] - without_mcs["brier"]
    rates_by_mcs = {
        key[0]: entry["implied_rate_mbps"]
        for key, entry in model.table.items()
    }
    raw_rates_by_mcs = {
        key[0]: 1e9 / entry["raw_slope_ns_per_byte"] * 8 / 1e6
        for key, entry in model.table.items()
    }
    ordered_bins = sorted(rates_by_mcs)
    direction_ok = all(
        rates_by_mcs[ordered_bins[i]] <= rates_by_mcs[ordered_bins[i + 1]]
        for i in range(len(ordered_bins) - 1)
    )
    results["gate_7"] = {
        "key": "MCS_NONHARM_AND_DIRECTION",
        "brier_with_mcs": with_mcs["brier"],
        "brier_without_mcs": without_mcs["brier"],
        "degradation": degradation,
        "constrained_rate_mbps_by_mcs_bin": rates_by_mcs,
        "unconstrained_rate_mbps_by_mcs_bin": raw_rates_by_mcs,
        "shape_constraint": "PAVA_NON_INCREASING_SLOPE_AND_OVERHEAD_IN_MCS",
        "higher_mcs_not_slower": direction_ok,
        "passed": degradation <= 0.01 and direction_ok,
    }

    # -- gate 8: monotonicity inside measured support ------------------
    support = model.support
    violations = 0
    probes = []
    backlogs = [support["min_backlog"],
                (support["min_backlog"] + support["max_backlog"]) / 2,
                support["max_backlog"]]
    byte_probes = sorted({support["min_bytes"],
                          (support["min_bytes"] + support["max_bytes"]) // 2,
                          support["max_bytes"]})
    for backlog in backlogs:
        for mcs in (None, 4, 12, 20, 27):
            previous = None
            for byte_value in byte_probes:
                value, _ = model.predict_latency_ns(
                    backlog=backlog, bytes_on_wire=int(byte_value), mcs=mcs)
                if previous is not None and value < previous - 1e-9:
                    violations += 1
                previous = value
            probes.append({"backlog": backlog, "mcs": mcs})
    for mcs in (None, 4, 12, 20, 27):
        previous = None
        for backlog in sorted(backlogs):
            value, _ = model.predict_latency_ns(
                backlog=backlog, bytes_on_wire=int(byte_probes[-1]), mcs=mcs)
            if previous is not None and value < previous - 1e-9:
                violations += 1
            previous = value
    results["gate_8"] = {
        "key": "MONOTONICITY", "violations": violations,
        "probe_count": len(probes), "passed": violations == 0,
    }

    # -- gate 10: no hidden input --------------------------------------
    leaked = [name for name in C.AUDIT_ONLY_NOT_MODEL_INPUT_FIELDS
              if name in set(C.PRIMARY_MODEL_INPUT_FIELDS)]
    results["gate_10"] = {
        "key": "NO_HIDDEN_INPUT", "leaked_fields": leaked,
        "model_conditions_on": ["pre_action_rlc_backlog_bytes",
                                "prior_ul_mcs", "udp_application_bytes"],
        "passed": not leaked,
    }

    # -- descriptive, not gated ----------------------------------------
    outcomes: dict[str, int] = {}
    for row in frames:
        outcomes[row["terminal_outcome"]] = outcomes.get(
            row["terminal_outcome"], 0) + 1
    per_tier: dict[str, Any] = {}
    for tier in C.TIER_NAMES:
        subset = [row for row in frames if row["tier_audit_only"] == tier]
        latencies = [row["transport_latency_ns"] / 1e6 for row in subset
                     if row["transport_latency_ns"] is not None
                     and row["transport_latency_ns"] > 0]
        backlogs_seen = [row["pre_action_rlc_backlog_bytes"] for row in subset
                         if row["pre_action_rlc_backlog_bytes"] is not None]
        per_tier[tier] = {
            "frames": len(subset),
            "success": sum(1 for row in subset
                           if row["terminal_outcome"] == "COMPLETE_WITHIN_DEADLINE"),
            "success_rate": (sum(1 for row in subset
                                 if row["terminal_outcome"]
                                 == "COMPLETE_WITHIN_DEADLINE") / len(subset)
                             if subset else math.nan),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "backlog_p50_bytes": _percentile(backlogs_seen, 0.50),
            "backlog_max_bytes": max(backlogs_seen) if backlogs_seen else None,
        }

    gates = [results[key] for key in sorted(results) if key.startswith("gate_")]
    return {
        "schema": ANALYSIS_SCHEMA,
        "contract_sha256": C.CONTRACT_SHA256,
        "frozen_binning_source": "ue_mcs_backlog_run4_analysis_v1",
        "binning": {
            "backlog_edges": [None if math.isinf(v) else v
                              for v in BACKLOG_EDGES],
            "mcs_edges": [None if math.isinf(v) else v for v in MCS_EDGES],
            "min_bin_support": MIN_BIN_SUPPORT,
        },
        "partitions": {
            "fit_cells": sorted({row["cell_id"] for row in fit_frames}),
            "validation_cells": sorted({row["cell_id"] for row in val_frames}),
            "fit_frames": len(fit_frames), "validation_frames": len(val_frames),
            "fit_cycles": len(fit_cycles), "validation_cycles": len(val_cycles),
        },
        "model": {
            "form": "latency_ns = overhead_ns + slope_ns_per_byte "
                    "* (backlog + udp_application_bytes)",
            "rate_support": dict(model.support),
            "global_entry": dict(model.global_entry),
            "bins": {str(key): value for key, value in sorted(
                model.table.items(), key=lambda item: str(item[0]))},
        },
        "terminal_outcomes": outcomes,
        "per_tier_descriptive": per_tier,
        "gates": results,
        "gates_passed": sum(1 for gate in gates if gate.get("passed")),
        "gates_total": len(gates),
        "all_gates_passed": all(gate.get("passed") for gate in gates),
        "disclosures": [
            "PRODUCTION_DOMAIN_QUEUE_AND_TRANSPORT_ONLY",
            "SINGLE_UE_RADIO_CONFIGURATION_ONLY",
            "PROFILE_TRANSFER_UNVALIDATED",
            "MODE_TRANSFER_UNVALIDATED",
            "PAYLOAD_INTERPOLATION_UNVALIDATED",
            "NOT_PERCEPTION_ENDORSEMENT",
        ],
    }


def export_transport_model(report: Mapping[str, Any]) -> dict[str, Any]:
    """The portable artifact downstream training consumes.

    Carries the fitted rate table, the measured support and every disclosure.
    A consumer must refuse rather than extrapolate outside the support.
    """
    model = report["model"]
    support = model["rate_support"]
    return {
        "schema": "scenesense.production_transport_model.v1",
        "contract_sha256": report["contract_sha256"],
        "form": model["form"],
        "cycle_seconds": CYCLE_SECONDS,
        "global_entry": model["global_entry"],
        "bins": model["bins"],
        "binning": report["binning"],
        "backoff_order": [list(step) for step in BACKOFF_ORDER],
        "min_bin_support": MIN_BIN_SUPPORT,
        "measured_support": {
            "min_udp_application_bytes": support["min_bytes"],
            "max_udp_application_bytes": support["max_bytes"],
            "min_pre_action_backlog_bytes": support["min_backlog"],
            "max_pre_action_backlog_bytes": support["max_backlog"],
            "fit_rows_used": support["fit_rows_used"],
        },
        "replaces_288_component": C.REPLACED_288_COMPONENT,
        "boundary": C.PRODUCTION_TRANSPORT_BOUNDARY,
        "shared_endpoint": C.SHARED_ENDPOINT,
        "adding_both_components_is_forbidden":
            C.ADDING_BOTH_COMPONENTS_IS_FORBIDDEN,
        "deadline_ns": C.REWARD_DEADLINE_NS,
        "gates_passed": report["gates_passed"],
        "gates_total": report["gates_total"],
        "all_gates_passed": report["all_gates_passed"],
        "disclosures": report["disclosures"],
        "out_of_support_policy": "REFUSE_DO_NOT_EXTRAPOLATE",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--export-model-json", default=None)
    args = parser.parse_args(argv)
    report = evaluate(Path(args.parsed_dir))
    out = Path(args.output_json)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    if args.export_model_json:
        exported = export_transport_model(report)
        with Path(args.export_model_json).open("x", encoding="utf-8") as handle:
            json.dump(exported, handle, indent=2, sort_keys=True,
                      allow_nan=True)
            handle.write("\n")
    json.dump({"gates_passed": report["gates_passed"],
               "gates_total": report["gates_total"],
               "all_gates_passed": report["all_gates_passed"]},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

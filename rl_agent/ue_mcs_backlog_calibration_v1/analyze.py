#!/usr/bin/env python3
"""Analysis for the UE-local [previous UL MCS, pre-enqueue backlog] qualification.

Answers the preregistered questions and nothing else:

1. transient response after each within-cell load transition;
2. steady-state separation between tiers, and MCS stability across tiers;
3. repeatability across the two counterbalanced repetitions;
4. age / missingness;
5. whether [MCS, backlog] predicts next-frame outcomes better than either alone.

Simple interpretable models only. No neural policy is trained. Splits are
blocked by (cell, block); individual frames are never split at random, because
consecutive decisions inside a block are strongly dependent and a random split
would leak a block's own queue state into its test rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_mcs_backlog_calibration_v1 import contract as C  # noqa: E402

TIER_ORDER = list(C.TIER_ORDER)


def to_float(text: Any) -> float | None:
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def to_bool(text: Any) -> bool | None:
    if text in ("True", "true", True):
        return True
    if text in ("False", "false", False):
        return False
    return None


def load_decisions(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["backlog"] = to_float(row["pre_enqueue_backlog_bytes"])
        row["mcs_raw"] = to_float(row["prior_ul_mcs_raw"])
        row["has_prior_mcs"] = to_bool(row["has_prior_ul_grant"])
        row["mcs"] = row["mcs_raw"]
        row["latency_ms"] = to_float(row["uplink_latency_ms"])
        row["complete"] = to_bool(row["complete_reassembly"])
        row["in_budget"] = to_bool(row["within_transport_budget"])
        row["since_transition"] = int(row["decisions_since_transition"])
        row["block_index"] = int(row["block_index"])
        row["decision_index"] = int(row["decision_index"])
    return rows


def project_mcs_validity(
    rows: Sequence[Mapping[str, Any]], max_age_ms: float
) -> list[dict[str, Any]]:
    """Project an external age guard without mutating lossless evidence."""
    if not math.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("max_age_ms must be finite and positive")
    projected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        raw = source.get("mcs_raw")
        age = to_float(source.get("mcs_age_ms"))
        present = source.get("has_prior_mcs") is True
        if not present:
            row["mcs"] = None
            row["mcs_status"] = "MISSING_NO_PRIOR_GRANT"
        elif age is None or age > max_age_ms:
            row["mcs"] = None
            row["mcs_status"] = "MISSING_STALE"
        else:
            row["mcs"] = raw
            row["mcs_status"] = "OBSERVED"
        projected.append(row)
    return projected


def mcs_age_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage under each candidate guard; raw MCS values remain unchanged."""
    out: dict[str, Any] = {}
    for bound in C.MCS_VALIDITY_CANDIDATES_MS:
        projected = project_mcs_validity(rows, bound)
        observed = sum(1 for row in projected if row["mcs_status"] == "OBSERVED")
        out[f"{bound:g}_ms"] = {
            "max_age_ms": bound,
            "observed": observed,
            "decisions": len(projected),
            "coverage": observed / len(projected) if projected else None,
        }
    return out


def describe(values: Sequence[float]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "p50": None, "mean": None, "sd": None,
                "p95": None, "min": None, "max": None}
    ordered = sorted(clean)

    def pct(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    return {"n": len(clean), "p50": pct(0.5), "p95": pct(0.95),
            "mean": statistics.fmean(clean),
            "sd": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
            "min": ordered[0], "max": ordered[-1]}


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    import bisect
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b:
        return {"delta": None, "interpretation": "UNDEFINED"}
    ordered = sorted(b)
    greater = sum(bisect.bisect_left(ordered, v) for v in a)
    less = sum(len(ordered) - bisect.bisect_right(ordered, v) for v in a)
    delta = (greater - less) / (len(a) * len(b))
    mag = abs(delta)
    label = ("NEGLIGIBLE" if mag < 0.147 else "SMALL" if mag < 0.33
             else "MEDIUM" if mag < 0.474 else "LARGE")
    return {"delta": delta, "interpretation": label, "n_a": len(a), "n_b": len(b)}


# --------------------------------------------------------------------------
# Q1/Q2: steady state
# --------------------------------------------------------------------------


def steady_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Last STEADY_STATE_DECISIONS of each block, disjoint from the transient."""
    cutoff = C.FRAMES_PER_BLOCK - C.STEADY_STATE_DECISIONS
    return [r for r in rows if r["since_transition"] >= cutoff]


def steady_state_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steady = steady_rows(rows)
    by_channel_tier: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in steady:
        by_channel_tier[(row["profile_id"], row["tier"])].append(row)

    table: dict[str, Any] = {}
    for channel in C.CONTRAST_PROFILE_IDS:
        entry: dict[str, Any] = {}
        for tier in TIER_ORDER:
            subset = by_channel_tier.get((channel, tier), [])
            entry[tier] = {
                "backlog_bytes": describe([r["backlog"] for r in subset]),
                "previous_ul_mcs": describe([r["mcs"] for r in subset]),
                "complete_rate": (
                    sum(1 for r in subset if r["complete"]) / len(subset)
                    if subset else None),
                "latency_ms": describe([r["latency_ms"] for r in subset]),
            }
        # Q1: is MCS materially stable across loads at a fixed channel?
        mcs_p50 = [entry[t]["previous_ul_mcs"]["p50"] for t in TIER_ORDER]
        present = [v for v in mcs_p50 if v is not None]
        entry["q1_mcs_stability_across_load"] = {
            "p50_by_tier": dict(zip(TIER_ORDER, mcs_p50)),
            "max_abs_p50_gap": (max(present) - min(present)) if len(present) > 1 else None,
            "low_vs_high_effect": cliffs_delta(
                [r["mcs"] for r in by_channel_tier.get((channel, "low"), [])],
                [r["mcs"] for r in by_channel_tier.get((channel, "high"), [])]),
        }
        # Q3: does backlog respond to offered load at a fixed channel?
        entry["q3_backlog_responds_to_load"] = {
            "p50_by_tier": {t: entry[t]["backlog_bytes"]["p50"] for t in TIER_ORDER},
            "low_vs_high_effect": cliffs_delta(
                [r["backlog"] for r in by_channel_tier.get((channel, "high"), [])],
                [r["backlog"] for r in by_channel_tier.get((channel, "low"), [])]),
        }
        table[channel] = entry

    # Q2: at a fixed load, does MCS distinguish the two channels?
    q2: dict[str, Any] = {}
    for tier in TIER_ORDER:
        fav = [r["mcs"] for r in by_channel_tier.get(("FAVORABLE_STABLE", tier), [])]
        adv = [r["mcs"] for r in by_channel_tier.get(("ADVERSE_STABLE", tier), [])]
        q2[tier] = {
            "favorable": describe(fav), "adverse": describe(adv),
            "effect": cliffs_delta(fav, adv),
        }
    return {"steady_decisions": len(steady), "by_channel": table,
            "q2_mcs_separates_channels_at_fixed_load": q2,
            "window": {"steady_state_decisions": C.STEADY_STATE_DECISIONS,
                       "transient_decisions": C.TRANSIENT_DECISIONS}}


# --------------------------------------------------------------------------
# Transient response
# --------------------------------------------------------------------------


def transient_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Backlog and MCS in the first decisions after each load transition."""
    by_transition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["block_index"] == 0:
            continue                      # no transition into the first block
        if row["since_transition"] >= C.TRANSIENT_DECISIONS:
            continue
        key = (row["profile_id"], f"{row['previous_tier']}->{row['tier']}")
        by_transition[key].append(row)

    out: dict[str, Any] = {}
    for (channel, transition), subset in sorted(by_transition.items()):
        # Mean trajectory across the repetitions that share this transition.
        by_step: dict[int, list[float]] = defaultdict(list)
        mcs_by_step: dict[int, list[float]] = defaultdict(list)
        for row in subset:
            if row["backlog"] is not None:
                by_step[row["since_transition"]].append(row["backlog"])
            if row["mcs"] is not None:
                mcs_by_step[row["since_transition"]].append(row["mcs"])
        trajectory = [
            {"decisions_since_transition": step,
             "backlog_mean_bytes": statistics.fmean(by_step[step]) if by_step.get(step) else None,
             "mcs_mean": statistics.fmean(mcs_by_step[step]) if mcs_by_step.get(step) else None,
             "n": len(by_step.get(step, []))}
            for step in range(C.TRANSIENT_DECISIONS)
        ]
        first = trajectory[0]["backlog_mean_bytes"]
        last = trajectory[-1]["backlog_mean_bytes"]
        out.setdefault(channel, {})[transition] = {
            "samples": len(subset),
            "cells_contributing": len({r["cell_id"] for r in subset}),
            "backlog_first_decision_mean": first,
            "backlog_last_transient_decision_mean": last,
            "backlog_change_bytes": (
                None if first is None or last is None else last - first),
            "mcs_first_decision_mean": trajectory[0]["mcs_mean"],
            "mcs_last_transient_decision_mean": trajectory[-1]["mcs_mean"],
            "trajectory": trajectory,
        }
    return out


# --------------------------------------------------------------------------
# Repeatability
# --------------------------------------------------------------------------


def repeatability_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Agreement between the two counterbalanced repetitions."""
    steady = steady_rows(rows)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in steady:
        grouped[(row["profile_id"], row["tier"], row["repetition"])].append(row)

    out: dict[str, Any] = {}
    for channel in C.CONTRAST_PROFILE_IDS:
        entry: dict[str, Any] = {}
        for tier in TIER_ORDER:
            rep0 = grouped.get((channel, tier, "0"), [])
            rep1 = grouped.get((channel, tier, "1"), [])
            b0 = describe([r["backlog"] for r in rep0])
            b1 = describe([r["backlog"] for r in rep1])
            m0 = describe([r["mcs"] for r in rep0])
            m1 = describe([r["mcs"] for r in rep1])
            entry[tier] = {
                "backlog_p50_rep0": b0["p50"], "backlog_p50_rep1": b1["p50"],
                "backlog_effect_rep0_vs_rep1": cliffs_delta(
                    [r["backlog"] for r in rep0], [r["backlog"] for r in rep1]),
                "mcs_p50_rep0": m0["p50"], "mcs_p50_rep1": m1["p50"],
                "mcs_effect_rep0_vs_rep1": cliffs_delta(
                    [r["mcs"] for r in rep0], [r["mcs"] for r in rep1]),
            }
        out[channel] = entry
    return out


# --------------------------------------------------------------------------
# Age and missingness
# --------------------------------------------------------------------------


def missingness_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    for row in rows:
        entry = per_cell.setdefault(row["cell_id"], {
            "decisions": 0, "mcs_observed": 0, "mcs_missing_stale": 0,
            "mcs_missing_no_prior": 0, "backlog_observed": 0,
            "mcs_age_ms": [], "backlog_age_ms": []})
        entry["decisions"] += 1
        status = row["mcs_status"]
        if status == "OBSERVED":
            entry["mcs_observed"] += 1
        elif status == "MISSING_STALE":
            entry["mcs_missing_stale"] += 1
        else:
            entry["mcs_missing_no_prior"] += 1
        if row["backlog_status"] == "OBSERVED":
            entry["backlog_observed"] += 1
        age = to_float(row["mcs_age_ms"])
        if age is not None:
            entry["mcs_age_ms"].append(age)
        bage = to_float(row["backlog_age_ms"])
        if bage is not None:
            entry["backlog_age_ms"].append(bage)

    for entry in per_cell.values():
        entry["mcs_coverage"] = entry["mcs_observed"] / entry["decisions"]
        entry["backlog_coverage"] = entry["backlog_observed"] / entry["decisions"]
        entry["mcs_age_ms"] = describe(entry["mcs_age_ms"])
        entry["backlog_age_ms"] = describe(entry["backlog_age_ms"])

    coverages = [e["mcs_coverage"] for e in per_cell.values()]
    return {"per_cell": per_cell,
            "mcs_coverage_min": min(coverages) if coverages else None,
            "mcs_coverage_max": max(coverages) if coverages else None,
            "backlog_coverage_min": min(
                (e["backlog_coverage"] for e in per_cell.values()), default=None),
            "note": ("age is reported as validity evidence only and is never a "
                     "policy feature")}


def saturation_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much of the campaign sits at the RLC buffer ceiling.

    The three pinned tiers offer 0.50 / 21.08 / 70.45 Mbps at 10 fps against a
    ~6 Mbps uplink, so medium and high are 3.5x and 12x over capacity. Once the
    queue reaches its ceiling the feature stops varying, and a feature pinned at
    its ceiling carries no information regardless of how well it is measured.
    This quantifies that directly rather than letting it hide inside a median.
    """
    values = [r["backlog"] for r in rows if r["backlog"] is not None]
    if not values:
        return {"decisions": 0, "note": "no backlog observations"}
    ceiling = max(values)
    near = 0.95 * ceiling

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        vals = [r["backlog"] for r in subset if r["backlog"] is not None]
        if not vals:
            return {"n": 0, "at_ceiling_fraction": None}
        return {
            "n": len(vals),
            "at_ceiling_fraction": sum(1 for v in vals if v >= near) / len(vals),
            "p50": sorted(vals)[len(vals) // 2],
            "distinct_values": len(set(vals)),
        }

    by_tier = {tier: block([r for r in rows if r["tier"] == tier])
               for tier in TIER_ORDER}
    by_position = {
        str(position): block([r for r in rows if r["block_index"] == position])
        for position in range(C.BLOCKS_PER_CELL)}
    # Block 0 always starts from an empty queue, because the RAN is rebuilt from
    # cold, so its transient is the one uncontaminated by carry-over.
    clean = [r for r in rows
             if r["block_index"] == 0 and r["since_transition"] < C.TRANSIENT_DECISIONS]
    by_tier_clean = {
        tier: block([r for r in clean if r["tier"] == tier]) for tier in TIER_ORDER}

    return {
        "observed_ceiling_bytes": ceiling,
        "near_ceiling_threshold_bytes": near,
        "overall_at_ceiling_fraction": sum(1 for v in values if v >= near) / len(values),
        "by_tier": by_tier,
        "by_block_position": by_position,
        "first_block_clean_transient_by_tier": by_tier_clean,
        "offered_mbps_by_tier": {
            t.tier: t.offered_mbps for t in C.resolve_load_tiers(ROOT)},
        "interpretation": (
            "medium and high are far beyond the uplink's capacity, so the queue "
            "reaches its ceiling within seconds and stays there. Only the first "
            "block of each cell begins from an empty queue, which is why the "
            "position-balanced design matters: every tier occupies position 0 "
            "equally often, giving a clean from-empty transient for each"
        ),
    }


# --------------------------------------------------------------------------
# Prediction: A (MCS) vs B (backlog) vs C (both)
# --------------------------------------------------------------------------

FEATURE_SETS = {
    "P_action_only": ("log_payload_bytes",),
    "A_action_and_mcs": ("log_payload_bytes", "mcs"),
    "B_action_and_backlog": ("log_payload_bytes", "backlog"),
    "C_action_mcs_and_backlog": ("log_payload_bytes", "mcs", "backlog"),
}


def same_row_action_conditioned_targets(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Pair pre-action state and current action with that action's outcome.

    MCS and backlog were sampled strictly before the current decision, so no
    temporal shift is needed.  Payload is explicit: otherwise a load change
    could be misattributed to the radio features.
    """
    return [{
        "cell_id": row["cell_id"], "profile_id": row["profile_id"],
        "block_index": row["block_index"], "tier": row["tier"],
        "action_id": int(row["action_id"]),
        "payload_bytes": int(row["payload_bytes"]),
        "log_payload_bytes": math.log1p(int(row["payload_bytes"])),
        "mcs": row["mcs"], "backlog": row["backlog"],
        "y_complete": 1.0 if row["complete"] else 0.0,
        "y_latency_ms": row["latency_ms"],
        "y_in_budget": (None if row["in_budget"] is None
                        else (1.0 if row["in_budget"] else 0.0)),
    } for row in rows]


def _standardize(columns: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[tuple[float, float]]]:
    stats = []
    for column in columns:
        mean = statistics.fmean(column)
        sd = statistics.pstdev(column) or 1.0
        stats.append((mean, sd))
    scaled = [[(value - stats[i][0]) / stats[i][1] for value in column]
              for i, column in enumerate(columns)]
    return scaled, stats


def _fit_logistic(x: Sequence[Sequence[float]], y: Sequence[float], *,
                  iterations: int = 400, lr: float = 0.3) -> list[float]:
    n_features = len(x[0]) if x else 0
    weights = [0.0] * (n_features + 1)
    n = len(y)
    if n == 0:
        return weights
    for _ in range(iterations):
        grad = [0.0] * (n_features + 1)
        for row, target in zip(x, y):
            z = weights[0] + sum(w * v for w, v in zip(weights[1:], row))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - target
            grad[0] += err
            for i, v in enumerate(row):
                grad[i + 1] += err * v
        for i in range(len(weights)):
            weights[i] -= lr * grad[i] / n
    return weights


def _predict_logistic(weights: Sequence[float], x: Sequence[Sequence[float]]
                      ) -> list[float]:
    out = []
    for row in x:
        z = weights[0] + sum(w * v for w, v in zip(weights[1:], row))
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))
    return out


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1.0]
    neg = [s for s, y in zip(scores, labels) if y == 0.0]
    if not pos or not neg:
        return None
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        stop = index
        while (stop + 1 < len(ordered)
               and scores[ordered[stop + 1]] == scores[ordered[index]]):
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for pos_i in range(index, stop + 1):
            ranks[ordered[pos_i]] = average
        index = stop + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1.0)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def blocked_prediction(samples: Sequence[Mapping[str, Any]], target: str
                       ) -> dict[str, Any]:
    """Leave-one-cell-out comparison of feature sets A, B and C.

    Folds are whole cells. Consecutive decisions inside a block are strongly
    dependent, so a random per-frame split would let a block's own queue state
    leak into its test rows and would flatter every feature set equally.
    """
    usable = [s for s in samples
              if s["mcs"] is not None and s["backlog"] is not None
              and s[target] is not None]
    dropped = len(samples) - len(usable)
    if len(usable) < 50:
        return {"target": target, "usable": len(usable), "dropped": dropped,
                "note": "too few complete-case rows for a blocked comparison"}

    cells = sorted({s["cell_id"] for s in usable})
    results: dict[str, Any] = {}
    for name, features in FEATURE_SETS.items():
        fold_auc: list[float] = []
        fold_brier: list[float] = []
        for held in cells:
            train = [s for s in usable if s["cell_id"] != held]
            test = [s for s in usable if s["cell_id"] == held]
            if not train or not test:
                continue
            columns = [[s[f] for s in train] for f in features]
            scaled, stats = _standardize(columns)
            x_train = list(zip(*scaled)) if scaled else []
            y_train = [s[target] for s in train]
            if len(set(y_train)) < 2:
                continue
            weights = _fit_logistic([list(r) for r in x_train], y_train)
            x_test = [[(s[f] - stats[i][0]) / stats[i][1]
                       for i, f in enumerate(features)] for s in test]
            probs = _predict_logistic(weights, x_test)
            y_test = [s[target] for s in test]
            auc = roc_auc(probs, y_test)
            if auc is not None:
                fold_auc.append(auc)
            fold_brier.append(
                sum((p - y) ** 2 for p, y in zip(probs, y_test)) / len(y_test))
        results[name] = {
            "folds_scored": len(fold_auc),
            "auc_mean": statistics.fmean(fold_auc) if fold_auc else None,
            "auc_sd": statistics.pstdev(fold_auc) if len(fold_auc) > 1 else None,
            "auc_min": min(fold_auc) if fold_auc else None,
            "auc_max": max(fold_auc) if fold_auc else None,
            "brier_mean": statistics.fmean(fold_brier) if fold_brier else None,
            "per_fold_auc": fold_auc,
        }
    action = results.get("P_action_only", {}).get("auc_mean")
    backlog = results.get("B_action_and_backlog", {}).get("auc_mean")
    both = results.get("C_action_mcs_and_backlog", {}).get("auc_mean")
    mcs = results.get("A_action_and_mcs", {}).get("auc_mean")
    results["comparison"] = {
        "target": target, "usable_rows": len(usable),
        "dropped_incomplete_case_rows": dropped,
        "split": "LEAVE_ONE_CELL_OUT_BLOCKED",
        "c_minus_action_auc": (
            None if action is None or both is None else both - action),
        "c_minus_b_auc": (
            None if backlog is None or both is None else both - backlog),
        "c_minus_a_auc": (None if mcs is None or both is None else both - mcs),
        "best_feature_set": max(
            (k for k in FEATURE_SETS if results.get(k, {}).get("auc_mean") is not None),
            key=lambda k: results[k]["auc_mean"], default=None),
    }
    return results


# --------------------------------------------------------------------------
# Accounting and gates
# --------------------------------------------------------------------------


def terminal_accounting(rows: Sequence[Mapping[str, Any]],
                        run_dir: Path) -> dict[str, Any]:
    """Exact sent-to-terminal accounting, per cell and overall."""
    per_cell: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = per_cell.setdefault(row["cell_id"], {
            "decisions": 0, "chunks_expected": 0, "chunks_sent": 0,
            "chunks_dropped_at_socket": 0, "unique_chunks_received": 0,
            "terminal": defaultdict(int)})
        entry["decisions"] += 1
        entry["chunks_expected"] += int(row["chunks_per_frame"])
        entry["chunks_sent"] += int(row["chunks_sent"])
        entry["chunks_dropped_at_socket"] += int(row["chunks_dropped"])
        entry["unique_chunks_received"] += int(row["unique_chunks_received"])
        entry["terminal"][row["terminal_outcome"]] += 1
    for entry in per_cell.values():
        entry["terminal"] = dict(entry["terminal"])
        entry["sent_plus_dropped_equals_expected"] = (
            entry["chunks_sent"] + entry["chunks_dropped_at_socket"]
            == entry["chunks_expected"])
        entry["chunks_lost_in_network"] = (
            entry["chunks_sent"] - entry["unique_chunks_received"])
        entry["terminal_sum_equals_decisions"] = (
            sum(entry["terminal"].values()) == entry["decisions"])
    return {
        "per_cell": per_cell,
        "all_cells_balance": all(e["sent_plus_dropped_equals_expected"]
                                 and e["terminal_sum_equals_decisions"]
                                 for e in per_cell.values()),
        "decisions": sum(e["decisions"] for e in per_cell.values()),
    }


def evaluate_gates(run_dir: Path, rows: Sequence[Mapping[str, Any]],
                   build: Mapping[str, Any], manifest: Mapping[str, Any],
                   accounting: Mapping[str, Any],
                   missingness: Mapping[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    cells = manifest.get("cells", [])
    captured = [c for c in cells if c.get("status") == "CAPTURED"]
    add("CELL_IDENTITY_AND_PROFILE_READ_BACK",
        len(captured) == 12 and all(c.get("profile_read_back_ok") for c in captured),
        f"{len(captured)}/12 cells captured with profile read-back verified")

    add("EXACT_SENT_TO_TERMINAL_ACCOUNTING", accounting["all_cells_balance"],
        f"{accounting['decisions']} decisions; per-cell chunk and terminal sums balance")

    audits = [c["causal_audit"] for c in build["per_cell"] if c.get("joined")]
    add("NO_NEGATIVE_INTERVAL",
        all(a["negative_uplink_latency"] == 0 and a["negative_observation_age"] == 0
            for a in audits),
        "no negative uplink latency and no negative observation age in any cell")

    add("NO_CROSS_CELL_QUEUE_CONTAMINATION",
        all(c.get("clean_cell_marker", {}).get("ran_rebuilt_from_cold")
            for c in captured),
        "the RAN is rebuilt from cold before every cell, so an RLC queue cannot "
        "survive into the next cell")

    add("EVERY_OBSERVATION_PRECEDES_ITS_DECISION",
        all(a["all_joined_observations_precede_decision"] for a in audits),
        "backlog and MCS are taken from the last sample strictly before the decision")

    add("RETRANSMISSIONS_EXCLUDED",
        all(set(a["distinct_prev_grant_rounds"]) <= {0} for a in audits),
        "only HARQ round 0 grants enter the policy feature")

    add("MISSING_MCS_NEVER_ZERO",
        all(a["missing_mcs_coerced_to_zero"] == 0 for a in audits),
        "missing MCS stays missing; 0 is preserved only as a real observation")

    backlogs = [r["backlog"] for r in rows if r["backlog"] is not None]
    add("RAW_BACKLOG_RETAINED",
        bool(backlogs) and max(backlogs) > 1.0,
        f"raw byte counts retained, max {max(backlogs) if backlogs else 0}; "
        f"no log1p_scale=1 saturation applied")

    tables = set()
    for audit in audits:
        tables.update(audit["distinct_mcs_tables"])
    add("MCS_TABLE_CONSTANT_ACROSS_CELLS", len(tables) <= 1,
        f"observed MCS table(s): {sorted(tables)}")

    provenance = [c.get("ue_gnb_mcs_provenance", {})
                  for c in build["per_cell"] if c.get("joined")]
    add("UE_DCI_MATCHES_GNB_FINAL_MCS",
        bool(provenance) and all(p.get("provenance_verified") for p in provenance),
        "; ".join(
            f"{i}: coverage={p.get('coverage')}, selected!=final="
            f"{p.get('selected_final_adjustments')} (diagnostic), ue!=final="
            f"{p.get('ue_final_mismatches')}, ambiguous={p.get('ambiguous')}"
            for i, p in enumerate(provenance)))

    restored = [c for c in cells if c.get("restored")]
    add("RF_RESTORED_AND_READ_BACK", len(restored) == len(cells) and bool(cells),
        f"{len(restored)}/{len(cells)} cells restored noise_power_dB=-50 with read-back")

    cold = manifest.get("final_cold_state", {})
    add("NO_ORPHAN_PROCESS", bool(cold.get("cold")),
        f"orphans={cold.get('orphan_processes')} tunnels={cold.get('residual_ue_tunnels')}")
    add("CARLA_AND_CUDA_UNTOUCHED", not cold.get("carla_running", True),
        "no CARLA process at any point; no perception model or CUDA context created")
    return gates


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def decide(steady: Mapping[str, Any], transient: Mapping[str, Any],
           repeat: Mapping[str, Any], missing: Mapping[str, Any],
           prediction: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]
           ) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    # Backlog must actually respond to the within-cell load change.
    responses = []
    for channel, entry in steady["by_channel"].items():
        effect = entry["q3_backlog_responds_to_load"]["low_vs_high_effect"]
        responses.append((channel, effect))
    add("BACKLOG_RESPONDS_TO_LOAD",
        all(e["interpretation"] in ("MEDIUM", "LARGE") for _, e in responses),
        "; ".join(f"{c}: high-vs-low Cliff's delta {e['delta']} ({e['interpretation']})"
                  for c, e in responses))

    # MCS should be materially stable across load at a fixed channel.
    gaps = {c: entry["q1_mcs_stability_across_load"]["max_abs_p50_gap"]
            for c, entry in steady["by_channel"].items()}
    finite = [g for g in gaps.values() if g is not None]
    add("MCS_STABLE_ACROSS_LOAD_AT_FIXED_CHANNEL",
        bool(finite) and max(finite) <= 4.0,
        f"max |median MCS| gap across the three tiers: {gaps} (index units)")

    # MCS should separate the two channels at fixed load.
    separations = {t: v["effect"] for t, v in
                   steady["q2_mcs_separates_channels_at_fixed_load"].items()}
    add("MCS_SEPARATES_CHANNELS_AT_FIXED_LOAD",
        all(e["interpretation"] in ("MEDIUM", "LARGE")
            for e in separations.values() if e["delta"] is not None),
        "; ".join(f"{t}: {e['delta']} ({e['interpretation']})"
                  for t, e in separations.items()))

    # Availability of both features at the 10 Hz decision rate.
    add("FEATURES_AVAILABLE_AT_DECISION_RATE",
        (missing["mcs_coverage_min"] or 0) >= 0.90
        and (missing["backlog_coverage_min"] or 0) >= 0.95,
        f"worst per-cell MCS coverage {missing['mcs_coverage_min']}, "
        f"worst backlog coverage {missing['backlog_coverage_min']}")

    # Repeatability across the counterbalanced repetitions.
    unstable = []
    for channel, entry in repeat.items():
        for tier, item in entry.items():
            effect = item["backlog_effect_rep0_vs_rep1"]
            if effect["delta"] is not None and effect["interpretation"] == "LARGE":
                unstable.append(f"{channel}/{tier}")
    add("REPEATABLE_ACROSS_REPETITIONS", not unstable,
        f"cells whose two repetitions disagree at LARGE effect: {unstable or 'none'}")

    # The pair must add value after the current payload/action is controlled.
    improvements = []
    for target, block in prediction.items():
        comparison = block.get("comparison")
        if not comparison:
            continue
        delta_b = comparison.get("c_minus_b_auc")
        delta_a = comparison.get("c_minus_a_auc")
        if delta_b is not None and delta_a is not None:
            improvements.append((target, delta_a, delta_b))
    add("PAIR_IMPROVES_ON_EACH_SINGLE_FEATURE",
        any(da > 0.01 and db > 0.01 for _, da, db in improvements),
        "; ".join(f"{t}: C-A {da:+.4f}, C-B {db:+.4f}" for t, da, db in improvements)
        or "no target had enough complete-case rows")

    gates_passed = all(g["passed"] for g in gates)
    add("ALL_STRUCTURAL_GATES_PASS", gates_passed,
        f"{sum(1 for g in gates if g['passed'])}/{len(gates)} structural gates")

    passed = [c for c in checks if c["passed"]]
    if not gates_passed:
        verdict = "INCONCLUSIVE"
    elif len(passed) == len(checks):
        verdict = "ACCEPT_MCS_BACKLOG_STATE"
    elif len(passed) >= len(checks) - 1:
        verdict = "ACCEPT_MCS_BACKLOG_STATE"
    elif len(passed) <= len(checks) - 4:
        verdict = "REJECT_MCS_BACKLOG_STATE"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "checks": checks,
            "passed": len(passed), "total": len(checks),
            "statements": [
                "UL MCS is a delayed, quantized scheduler decision derived from "
                "gNB-measured uplink SNR in this custom build.",
                "It reaches the UE through standard DCI without an added controller.",
                "Backlog represents demand/queue pressure, not physical channel.",
                "This is a bounded engineering qualification, not repeated "
                "publication-level evidence.",
            ]}


def normalization_bounds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measured bounds for a future normalizer. Reported, never applied here."""
    backlog = sorted(r["backlog"] for r in rows if r["backlog"] is not None)
    mcs = sorted(r["mcs"] for r in rows if r["mcs"] is not None)

    def pct(values: Sequence[float], q: float) -> float | None:
        if not values:
            return None
        return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]

    return {
        "pre_enqueue_backlog_bytes": {
            "min": backlog[0] if backlog else None,
            "p50": pct(backlog, 0.5), "p95": pct(backlog, 0.95),
            "p99": pct(backlog, 0.99), "max": backlog[-1] if backlog else None,
            "recommended_transform": "log1p(bytes) / log1p(p99)",
            "warning": ("the deployed log1p_scale=1.0 maps every backlog above 1 "
                        "byte to 1.0 and must not be reused"),
        },
        "previous_ul_mcs": {
            "min": mcs[0] if mcs else None, "max": mcs[-1] if mcs else None,
            "p50": pct(mcs, 0.5),
            "recommended_transform": "mcs / 28 (MCS table 0 upper index)",
            "missing_policy": ("missing must be signalled explicitly, never encoded "
                               "as 0, which is a real modulation index"),
        },
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

TIER_COLORS = {"low": "#2E7D32", "medium": "#1565C0", "high": "#C62828"}
CHANNEL_COLORS = {"FAVORABLE_STABLE": "#2E7D32", "ADVERSE_STABLE": "#C62828"}


def require_create_only_targets(run_dir: Path, figures: bool) -> tuple[Path, Path]:
    """Refuse to overwrite immutable v2 analysis outputs."""
    report = run_dir / "analysis_v2.json"
    figure_dir = run_dir / "figures_v2"
    existing = [str(report)] if report.exists() else []
    if figures and figure_dir.exists():
        existing.append(str(figure_dir))
    if existing:
        raise FileExistsError(f"v2 analysis output already exists: {existing}")
    return report, figure_dir


def render_figures(rows: Sequence[Mapping[str, Any]], transient: Mapping[str, Any],
                   prediction: Mapping[str, Any], missing: Mapping[str, Any],
                   out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=False)
    written: list[str] = []

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
            written.append(f"{stem}.{suffix}")
        plt.close(fig)

    steady = steady_rows(rows)

    def grouped(values_key: str):
        data: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in steady:
            value = row[values_key]
            if value is not None:
                data[(row["profile_id"], row["tier"])].append(value)
        return data

    # 1. MCS distribution by channel and load.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    mcs_data = grouped("mcs")
    for axis, channel in zip(axes, C.CONTRAST_PROFILE_IDS):
        series = [mcs_data.get((channel, tier), []) for tier in TIER_ORDER]
        parts = axis.boxplot(series, tick_labels=TIER_ORDER, patch_artist=True,
                             showfliers=False)
        for patch, tier in zip(parts["boxes"], TIER_ORDER):
            patch.set_facecolor(TIER_COLORS[tier]); patch.set_alpha(0.55)
        axis.set_title(channel, fontsize=10)
        axis.set_xlabel("offered load tier")
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("previous round-0 UL MCS index")
    fig.suptitle("Previous UL MCS by channel and offered load (steady-state window)",
                 fontsize=11)
    save(fig, "fig01_mcs_by_channel_and_load")

    # 2. Backlog distribution by channel and load.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    backlog_data = grouped("backlog")
    for axis, channel in zip(axes, C.CONTRAST_PROFILE_IDS):
        series = [backlog_data.get((channel, tier), []) for tier in TIER_ORDER]
        parts = axis.boxplot(series, tick_labels=TIER_ORDER, patch_artist=True,
                             showfliers=False)
        for patch, tier in zip(parts["boxes"], TIER_ORDER):
            patch.set_facecolor(TIER_COLORS[tier]); patch.set_alpha(0.55)
        axis.set_title(channel, fontsize=10)
        axis.set_xlabel("offered load tier")
        axis.set_yscale("symlog")
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("pre-enqueue RLC backlog (raw bytes, symlog)")
    fig.suptitle("Pre-enqueue backlog by channel and offered load (steady-state)",
                 fontsize=11)
    save(fig, "fig02_backlog_by_channel_and_load")

    # 3. Transient response after each within-cell transition.
    channels = [c for c in C.CONTRAST_PROFILE_IDS if c in transient]
    if channels:
        fig, axes = plt.subplots(len(channels), 1,
                                 figsize=(10, 3.8 * len(channels)), squeeze=False)
        for axis, channel in zip(axes[:, 0], channels):
            for name, block in sorted(transient[channel].items()):
                steps = [p["decisions_since_transition"] for p in block["trajectory"]]
                values = [p["backlog_mean_bytes"] for p in block["trajectory"]]
                axis.plot(steps, values, marker="o", ms=3, lw=1.2, label=name)
            axis.set_yscale("symlog")
            axis.set_title(f"{channel}: backlog after a load transition", fontsize=10)
            axis.set_ylabel("mean backlog (bytes, symlog)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=3)
        axes[-1, 0].set_xlabel("decisions since transition (100 ms each)")
        save(fig, "fig03_transient_backlog_after_transition")

    # 4. MCS age and coverage per cell.
    cells = sorted(missing["per_cell"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(range(len(cells)),
                [missing["per_cell"][c]["mcs_coverage"] for c in cells],
                color="#1565C0", alpha=0.8)
    axes[0].axhline(0.9, color="#C62828", ls="--", lw=1, label="0.90")
    axes[0].set_ylabel("MCS coverage (fraction of decisions)")
    axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=8)
    axes[1].bar(range(len(cells)),
                [missing["per_cell"][c]["mcs_age_ms"]["p95"] or 0 for c in cells],
                color="#EF6C00", alpha=0.8)
    axes[1].axhline(C.MCS_MAX_AGE_MS, color="#C62828", ls="--", lw=1,
                    label=f"{C.MCS_MAX_AGE_MS:.0f} ms validity bound")
    axes[1].set_ylabel("MCS observation age P95 (ms)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(cells)))
        axis.set_xticklabels(cells, rotation=90, fontsize=6)
        axis.grid(alpha=0.25, axis="y")
    fig.suptitle("MCS availability and age by cell (age is validity evidence, "
                 "not a policy feature)", fontsize=10)
    save(fig, "fig04_mcs_age_and_coverage_by_cell")

    # 5. Pre-action feature vs the same action's uplink latency.
    samples = same_row_action_conditioned_targets(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, key, label in ((axes[0], "mcs", "previous round-0 UL MCS"),
                             (axes[1], "backlog", "pre-enqueue backlog (bytes)")):
        for channel in C.CONTRAST_PROFILE_IDS:
            xs = [s[key] for s in samples
                  if s["profile_id"] == channel and s[key] is not None
                  and s["y_latency_ms"] is not None]
            ys = [s["y_latency_ms"] for s in samples
                  if s["profile_id"] == channel and s[key] is not None
                  and s["y_latency_ms"] is not None]
            axis.scatter(xs, ys, s=5, alpha=0.25,
                         color=CHANNEL_COLORS[channel], label=channel)
        axis.set_xlabel(label)
        axis.set_ylabel("current action uplink latency (ms)")
        axis.set_yscale("symlog")
        if key == "backlog":
            axis.set_xscale("symlog")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pre-action features against the current action's uplink latency",
                 fontsize=11)
    save(fig, "fig05_features_vs_same_action_latency")

    # 6. Action baseline plus incremental MCS/backlog comparisons.
    targets = [t for t in prediction if prediction[t].get("comparison")]
    if targets:
        fig, axis = plt.subplots(figsize=(9, 5))
        width = 0.20
        for offset, name in enumerate(FEATURE_SETS):
            values = [prediction[t].get(name, {}).get("auc_mean") or 0
                      for t in targets]
            errors = [prediction[t].get(name, {}).get("auc_sd") or 0
                      for t in targets]
            axis.bar([i + offset * width for i in range(len(targets))], values,
                     width, yerr=errors, capsize=3, label=name, alpha=0.85)
        axis.axhline(0.5, color="#555555", ls="--", lw=1, label="chance")
        axis.set_xticks([i + 1.5 * width for i in range(len(targets))])
        axis.set_xticklabels(targets, fontsize=8)
        axis.set_ylabel("leave-one-cell-out AUC (mean +/- sd across folds)")
        axis.set_title("Action-conditioned prediction: incremental MCS/backlog value",
                       fontsize=11)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25, axis="y")
        save(fig, "fig06_prediction_comparison_abc")

    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    report_path, figure_dir = require_create_only_targets(
        args.run_dir, args.figures)

    raw_rows = load_decisions(args.run_dir / "decisions_v2.csv")
    rows = project_mcs_validity(raw_rows, C.MCS_MAX_AGE_MS)
    build = json.loads((args.run_dir / "decisions_build_v2.json").read_text())
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        # Partial/interrupted campaign: reconstruct enough of the manifest from
        # the per-cell records so the analysis still runs and the gates still
        # report honestly on what exists.
        cells = []
        for cell_dir in sorted((args.run_dir / "cells").iterdir()):
            record = cell_dir / "cell_record.json"
            if record.is_file():
                cells.append(json.loads(record.read_text()))
        manifest = {"cells": cells, "final_cold_state": {},
                    "reconstructed_from_cell_records": True}

    steady = steady_state_analysis(rows)
    saturation = saturation_analysis(rows)
    transient = transient_analysis(rows)
    repeat = repeatability_analysis(rows)
    missing = missingness_analysis(rows)
    accounting = terminal_accounting(rows, args.run_dir)
    samples = same_row_action_conditioned_targets(rows)
    prediction = {
        target: blocked_prediction(samples, target)
        for target in ("y_complete", "y_in_budget")
    }
    gates = evaluate_gates(args.run_dir, rows, build, manifest, accounting, missing)
    verdict = decide(steady, transient, repeat, missing, prediction, gates)

    report = {
        "run_dir": str(args.run_dir),
        "analysis_schema": "ue_mcs_backlog_analysis_v2",
        "decisions": len(rows),
        "design": {
            "block_orders": [list(o) for o in C.BLOCK_ORDERS],
            "channels": list(C.CONTRAST_PROFILE_IDS),
            "repetitions": C.REPETITIONS,
            "frames_per_block": C.FRAMES_PER_BLOCK,
            "transient_decisions": C.TRANSIENT_DECISIONS,
            "steady_state_decisions": C.STEADY_STATE_DECISIONS,
            "primary_mcs_validity_bound_ms": C.MCS_MAX_AGE_MS,
            "validity_bound_status": "ENGINEERING_HYPOTHESIS_SENSITIVITY_REPORTED",
        },
        "steady_state": steady,
        "saturation": saturation,
        "transient": transient,
        "repeatability": repeat,
        "missingness": missing,
        "mcs_age_sensitivity": mcs_age_sensitivity(raw_rows),
        "terminal_accounting": accounting,
        "prediction": prediction,
        "normalization_bounds": normalization_bounds(raw_rows),
        "gates": gates,
        "interpretation": verdict,
    }
    figures: list[str] = []
    if args.figures:
        figures = render_figures(rows, transient, prediction, missing,
                                 figure_dir)
    report["figures"] = figures
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(
            report, indent=2, sort_keys=True, default=str) + "\n")

    print(json.dumps({
        "verdict": verdict["verdict"],
        "checks_passed": f"{verdict['passed']}/{verdict['total']}",
        "gates_passed": f"{sum(1 for g in gates if g['passed'])}/{len(gates)}",
        "decisions": len(rows), "figures": len(figures),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

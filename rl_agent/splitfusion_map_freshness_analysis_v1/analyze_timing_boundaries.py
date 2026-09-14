#!/usr/bin/env python3
"""Add a controller-facing timing boundary to the 288-cell freshness study.

The immutable campaign measured physical map age from camera capture.  That is
the correct safety/freshness clock, but it includes preparation performed
before an action can take effect.  This analysis preserves that clock and adds
a second clock starting at ``capture_started_ns``: the entry to seven-channel
tensor assembly immediately before the selected action's UE path.

The UE result loop sampled ``edge_result_received_ns`` (UE monotonic clock) and
``feature_received_at`` (host wall clock) at the same receive event.  Their
robust global median offset bridges ``capture_started_ns`` to wall time.  The
bridge is measured and audited; no latency constant is guessed or subtracted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288 as source,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288_final_v3 as final_v3,
)
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_288 as prior
from rl_agent.splitfusion_map_freshness_analysis_v1.queue_models import (
    QueueOutcome,
    QueuePolicy,
    QueueSimulation,
    simulate_queue_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "scenesense.splitfusion.map_freshness_dual_clock_analysis.v1"
TERMINAL = "SPLITFUSION_MAP_FRESHNESS_DUAL_CLOCK_ANALYSIS_COMPLETE"
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_capture_vs_action_clock_v4"
)
PRIOR_ANALYSIS = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260913_action_network_freshness_v2"
)
PROFILE_ORDER = prior.PROFILE_ORDER
BUDGETS_MS = prior.BUDGETS_MS
BUDGETS_NS = prior.BUDGETS_NS
POLICY_ORDER = prior.POLICY_ORDER


class DualClockError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DualClockError(message)


def wall_seconds_to_ns(value: str | float) -> int:
    return int(
        (Decimal(str(value)) * Decimal(1_000_000_000)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return None if not data else float(statistics.median(data))


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def integer_median(values: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in values)
    require(bool(ordered), "integer median of empty input")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"column drift: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def collect_clock_bridge(
    cells: Sequence[Mapping[str, str]],
) -> tuple[int, dict[str, Any], dict[str, str]]:
    offsets: list[int] = []
    cell_anchor_counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    first_perf: int | None = None
    last_perf: int | None = None
    first_offsets: list[int] = []
    last_offsets: list[int] = []

    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        digest = source._attempt_manifest_hash(attempt, "per_frame_metrics.csv")
        hashes[cell["cell_id"]] = digest
        local: list[tuple[int, int]] = []
        for row in source._sent_rows(attempt):
            try:
                perf_ns = int(row["edge_result_received_ns"])
                wall_ns = wall_seconds_to_ns(row["feature_received_at"])
            except (KeyError, TypeError, ValueError):
                continue
            local.append((perf_ns, wall_ns - perf_ns))
        cell_anchor_counts[cell["cell_id"]] = len(local)
        for perf_ns, offset_ns in local:
            offsets.append(offset_ns)
            if first_perf is None or perf_ns < first_perf:
                first_perf = perf_ns
            if last_perf is None or perf_ns > last_perf:
                last_perf = perf_ns
        if number % 48 == 0:
            print(f"clock-bridge pass: {number}/288 cells", flush=True)

    require(len(offsets) == 344_177, "dual-clock anchor total drift")
    bridge_ns = integer_median(offsets)
    deviations_ms = [abs(value - bridge_ns) / 1e6 for value in offsets]

    # Robust endpoint drift is calculated from the earliest/latest 10,000
    # anchors.  It is insensitive to the rare scheduling delay between the two
    # adjacent clock calls.
    timed: list[tuple[int, int]] = []
    for cell in cells:
        attempt = source._attempt(cell)
        for row in source._sent_rows(attempt):
            try:
                perf_ns = int(row["edge_result_received_ns"])
                wall_ns = wall_seconds_to_ns(row["feature_received_at"])
            except (KeyError, TypeError, ValueError):
                continue
            timed.append((perf_ns, wall_ns - perf_ns))
    timed.sort()
    endpoint_count = min(10_000, max(1, len(timed) // 4))
    first_offsets = [value for _, value in timed[:endpoint_count]]
    last_offsets = [value for _, value in timed[-endpoint_count:]]
    drift_ms = (integer_median(last_offsets) - integer_median(first_offsets)) / 1e6

    audit = {
        "bridge_definition": "feature_received_at_wall_ns - edge_result_received_ns",
        "bridge_application": "action_start_wall_ns = capture_started_ns + robust_global_median_offset_ns",
        "same_event_pair": True,
        "same_physical_host": True,
        "anchor_count": len(offsets),
        "cells_with_anchors": sum(value > 0 for value in cell_anchor_counts.values()),
        "cells_without_anchors": sum(value == 0 for value in cell_anchor_counts.values()),
        "global_median_offset_ns": bridge_ns,
        "absolute_deviation_ms_p50": percentile(deviations_ms, 0.50),
        "absolute_deviation_ms_p95": percentile(deviations_ms, 0.95),
        "absolute_deviation_ms_p99": percentile(deviations_ms, 0.99),
        "absolute_deviation_ms_p999": percentile(deviations_ms, 0.999),
        "absolute_deviation_ms_max": max(deviations_ms),
        "robust_first_to_last_offset_drift_ms": drift_ms,
        "endpoint_anchor_count": endpoint_count,
        "interpretation": (
            "The median bridge is process-independent because perf_counter_ns "
            "uses the host monotonic clock. Rare positive deviations reflect "
            "preemption between the adjacent monotonic and wall-clock calls."
        ),
    }
    require(float(audit["absolute_deviation_ms_p99"]) < 0.01, "clock bridge p99 unstable")
    require(abs(drift_ms) < 0.01, "clock bridge endpoint drift")
    return bridge_ns, audit, hashes


def map_age_metrics(
    simulation: QueueSimulation,
    useful: Sequence[QueueOutcome],
    reference_ns: Mapping[int, int],
) -> dict[str, Any]:
    start = simulation.observation_start_ns
    end = simulation.observation_end_ns
    duration = end - start
    require(duration > 0, "non-positive observation duration")
    installed = [
        item
        for item in useful
        if item.install_ns is not None and int(item.install_ns) < end
    ]
    output: dict[str, Any] = {
        "map_age_ms_when_available": None,
        "map_available_fraction": 0.0,
    }
    for budget in BUDGETS_MS:
        output[f"fresh_map_time_ms_le_{budget}_fraction"] = 0.0
    if not installed:
        return output

    first_install = max(start, int(installed[0].install_ns))
    available_ns = max(0, end - first_install)
    output["map_available_fraction"] = available_ns / duration
    area_ns2 = 0.0
    fresh_ns = {budget: 0 for budget in BUDGETS_NS}
    for index, item in enumerate(installed):
        interval_start = max(first_install, int(item.install_ns))
        interval_end = (
            min(end, int(installed[index + 1].install_ns))
            if index + 1 < len(installed)
            else end
        )
        if interval_end <= interval_start:
            continue
        reference = int(reference_ns[item.frame.sequence_id])
        age_start = interval_start - reference
        require(age_start >= 0, "installed update predates its action boundary")
        interval = interval_end - interval_start
        area_ns2 += float(age_start) * interval + 0.5 * float(interval) ** 2
        for budget in BUDGETS_NS:
            fresh_ns[budget] += max(
                0,
                min(interval_end, reference + budget) - interval_start,
            )
    output["map_age_ms_when_available"] = (
        area_ns2 / available_ns / 1e6 if available_ns else None
    )
    for budget_ms, budget_ns in zip(BUDGETS_MS, BUDGETS_NS):
        output[f"fresh_map_time_ms_le_{budget_ms}_fraction"] = (
            fresh_ns[budget_ns] / duration
        )
    return output


def dual_clock_summary(
    simulation: QueueSimulation,
    action_start_ns: Mapping[int, int],
    prep_ms: Sequence[float],
    input_frames: int,
    label: str,
) -> dict[str, Any]:
    useful = prior.useful_installations(simulation.outcomes)
    observed = [
        item
        for item in useful
        if item.install_ns is not None
        and int(item.install_ns) < simulation.observation_end_ns
    ]
    service_ms = [
        (int(item.install_ns) - int(action_start_ns[item.frame.sequence_id])) / 1e6
        for item in observed
    ]
    require(
        all(value >= 0 for value in service_ms),
        f"{label}: negative action-service latency: "
        f"minimum={min(service_ms, default=0.0):.6f} ms",
    )
    action_map = map_age_metrics(simulation, useful, action_start_ns)
    output: dict[str, Any] = {
        "capture_to_action_start_ms_count": len(prep_ms),
        "capture_to_action_start_ms_median": median(prep_ms),
        "capture_to_action_start_ms_p95": percentile(prep_ms, 0.95),
        "capture_to_action_start_ms_p99": percentile(prep_ms, 0.99),
        "action_service_install_latency_ms_count": len(service_ms),
        "action_service_install_latency_ms_median": median(service_ms),
        "action_service_install_latency_ms_p95": percentile(service_ms, 0.95),
        "action_clock_map_age_ms_when_available": action_map[
            "map_age_ms_when_available"
        ],
        "action_clock_map_available_fraction": action_map["map_available_fraction"],
    }
    for budget in BUDGETS_MS:
        timely = sum(value <= budget for value in service_ms)
        output[f"action_service_installs_ms_le_{budget}"] = timely
        output[f"action_service_timely_yield_ms_le_{budget}"] = ratio(
            timely, input_frames
        )
        output[f"action_clock_fresh_map_time_ms_le_{budget}_fraction"] = (
            action_map[f"fresh_map_time_ms_le_{budget}_fraction"]
        )
    return output


def enforce_imputed_arrival_causality(
    frames: Sequence[Any],
    rows: Sequence[Mapping[str, str]],
    bridge_ns: int,
    label: str,
) -> tuple[list[Any], int, int]:
    """Ensure only imputed edge arrivals follow UE transmission start.

    ``ue_prepare_finished_ns`` is recorded after encoding and immediately
    before transmission. ``send_finished_ns`` is deliberately not used as the
    floor because edge reception can overlap the multi-datagram host send loop.
    """

    require(len(frames) == len(rows), f"{label}: frame/row count drift")
    corrected: list[Any] = []
    floored = 0
    floor_delta_ns = 0
    for frame, row in zip(frames, rows):
        if frame.arrival_ns is None:
            corrected.append(frame)
            continue
        transmission_start_ns = int(row["ue_prepare_finished_ns"]) + bridge_ns
        if row.get("edge_receipt_wall_s") not in (None, ""):
            require(
                int(frame.arrival_ns) + 1_000_000 >= transmission_start_ns,
                f"{label} frame {frame.frame_id}: observed edge arrival "
                f"precedes transmission start by "
                f"{(transmission_start_ns - int(frame.arrival_ns)) / 1e6:.6f} ms",
            )
            corrected.append(frame)
            continue
        corrected_arrival = max(int(frame.arrival_ns), transmission_start_ns)
        if corrected_arrival != int(frame.arrival_ns):
            floored += 1
            floor_delta_ns += corrected_arrival - int(frame.arrival_ns)
            frame = replace(frame, arrival_ns=corrected_arrival)
        corrected.append(frame)
    return corrected, floored, floor_delta_ns


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def block(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        duration = sum(float(item["map_observation_duration_s"]) for item in items)
        sent = sum(int(item["input_frames"]) for item in items)
        output: dict[str, Any] = {
            "cells": len(items),
            "sent": sent,
            "useful_installs": sum(int(item["useful_installs_during_observation"]) for item in items),
            "capture_to_action_start_ms_cell_median": median(
                item["capture_to_action_start_ms_median"] for item in items
            ),
            "capture_to_action_start_ms_p95_cell_median": median(
                item["capture_to_action_start_ms_p95"] for item in items
            ),
            "physical_install_aoi_ms_cell_median": median(
                item["physical_install_aoi_ms_median"]
                for item in items
                if item["physical_install_aoi_ms_median"] is not None
            ),
            "action_service_install_latency_ms_cell_median": median(
                item["action_service_install_latency_ms_median"]
                for item in items
                if item["action_service_install_latency_ms_median"] is not None
            ),
            "physical_map_aoi_ms_cell_median": median(
                item["physical_map_aoi_ms_when_available"]
                for item in items
                if item["physical_map_aoi_ms_when_available"] is not None
            ),
            "action_clock_map_age_ms_cell_median": median(
                item["action_clock_map_age_ms_when_available"]
                for item in items
                if item["action_clock_map_age_ms_when_available"] is not None
            ),
            "prior_physical_install_aoi_ms_cell_median": median(
                item["prior_physical_install_aoi_ms_median"]
                for item in items
                if item["prior_physical_install_aoi_ms_median"] is not None
            ),
            "prior_physical_map_aoi_ms_cell_median": median(
                item["prior_physical_map_aoi_ms_when_available"]
                for item in items
                if item["prior_physical_map_aoi_ms_when_available"] is not None
            ),
            "imputed_arrivals_floored_at_transmission_start": sum(
                int(item["imputed_arrivals_floored_at_transmission_start"])
                for item in items
            ),
        }
        for budget in BUDGETS_MS:
            physical = sum(
                float(item[f"physical_fresh_map_time_ms_le_{budget}_fraction"])
                * float(item["map_observation_duration_s"])
                for item in items
            )
            action = sum(
                float(item[f"action_clock_fresh_map_time_ms_le_{budget}_fraction"])
                * float(item["map_observation_duration_s"])
                for item in items
            )
            output[f"physical_fresh_map_time_ms_le_{budget}_fraction"] = physical / duration
            output[f"action_clock_fresh_map_time_ms_le_{budget}_fraction"] = action / duration
            output[f"action_service_timely_yield_ms_le_{budget}"] = ratio(
                sum(int(item[f"action_service_installs_ms_le_{budget}"]) for item in items),
                sent,
            )
            output[f"prior_physical_fresh_map_time_ms_le_{budget}_fraction"] = (
                sum(
                    float(item[f"prior_physical_fresh_map_time_ms_le_{budget}_fraction"])
                    * float(item["map_observation_duration_s"])
                    for item in items
                )
                / duration
            )
        return output

    result: dict[str, Any] = {}
    for policy in POLICY_ORDER:
        selected = [row for row in rows if row["queue_policy"] == policy.value]
        result[policy.value] = {
            "overall": block(selected),
            "by_network_profile": {
                profile: block(
                    [row for row in selected if row["network_profile"] == profile]
                )
                for profile in PROFILE_ORDER
            },
        }
    return result


def winner_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest = [
        row
        for row in rows
        if row["queue_policy"] == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value
    ]
    output: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        candidates = [row for row in latest if row["network_profile"] == profile]
        require(len(candidates) == 72, f"{profile}: not 72 latest-only actions")
        for budget in BUDGETS_MS:
            for clock, freshness_key in (
                ("PHYSICAL_CAPTURE_CLOCK", f"physical_fresh_map_time_ms_le_{budget}_fraction"),
                ("ACTION_START_CLOCK", f"action_clock_fresh_map_time_ms_le_{budget}_fraction"),
            ):
                for objective, quality_key in (
                    ("RAW_FRESHNESS", None),
                    ("PERSON_F1_X_FRESHNESS", "val_canonical_person_f1"),
                    ("VEHICLE_F1_X_FRESHNESS", "val_vehicle_f1"),
                ):
                    def score(row: Mapping[str, Any]) -> float:
                        quality = 1.0 if quality_key is None else float(row[quality_key] or 0.0)
                        return quality * float(row[freshness_key])

                    best = max(candidates, key=lambda row: (score(row), -float(row["median_feature_bytes"])))
                    output.append(
                        {
                            "network_profile": profile,
                            "freshness_budget_ms": budget,
                            "clock": clock,
                            "objective": objective,
                            "action_id": int(best["action_id"]),
                            "profile_id": best["profile_id"],
                            "family": best["family"],
                            "quantizer": best["quantizer"],
                            "q": best["q"],
                            "median_feature_bytes": best["median_feature_bytes"],
                            "fresh_map_fraction": best[freshness_key],
                            "validation_quality": (
                                None if quality_key is None else best[quality_key]
                            ),
                            "objective_score": score(best),
                        }
                    )
    return output


def eligibility_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest = [
        row
        for row in rows
        if row["queue_policy"] == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value
    ]
    output: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        candidates = [row for row in latest if row["network_profile"] == profile]
        for budget in BUDGETS_MS:
            for threshold in (0.10, 0.25, 0.50, 0.75):
                for clock, key in (
                    ("PHYSICAL_CAPTURE_CLOCK", f"physical_fresh_map_time_ms_le_{budget}_fraction"),
                    ("ACTION_START_CLOCK", f"action_clock_fresh_map_time_ms_le_{budget}_fraction"),
                ):
                    eligible = [
                        int(row["action_id"])
                        for row in candidates
                        if float(row[key]) >= threshold
                    ]
                    output.append(
                        {
                            "network_profile": profile,
                            "freshness_budget_ms": budget,
                            "minimum_fresh_map_fraction": threshold,
                            "clock": clock,
                            "eligible_action_count": len(eligible),
                            "eligible_action_ids": "|".join(map(str, eligible)),
                        }
                    )
    return output


def report(document: Mapping[str, Any], winners: Sequence[Mapping[str, Any]]) -> str:
    latest = document["aggregate"][QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    overall = latest["overall"]
    lines = [
        "# SplitFusion 288-cell dual-clock freshness analysis",
        "",
        "The completed 288-cell measurements remain immutable. This offline",
        "analysis reports two different, complementary clocks:",
        "",
        "- **Physical capture clock:** camera capture to map installation. This is",
        "  the actual age of information and remains the safety/freshness metric.",
        "- **Action-start clock:** entry to seven-channel tensor assembly to map",
        "  installation. This removes work completed before the selected action",
        "  can influence the system and is the controller-attributable metric.",
        "",
        "The action-start clock does not replace physical AoI. In particular, a",
        "map may satisfy an action-service budget while still be too old physically.",
        "",
        "## Clock bridge",
        "",
        f"The bridge uses {document['clock_bridge']['anchor_count']:,} same-event",
        "UE receive timestamps. Its absolute offset deviation is",
        f"{document['clock_bridge']['absolute_deviation_ms_p99']:.6f} ms at p99;",
        f"robust campaign drift is {document['clock_bridge']['robust_first_to_last_offset_drift_ms']:.6f} ms.",
        "",
        "## Strict latest-only aggregate",
        "",
        "| Quantity | Prior capture-clock replay | Corrected capture clock | Corrected action-start clock |",
        "|---|---:|---:|---:|",
        f"| Median cell install latency | {overall['prior_physical_install_aoi_ms_cell_median']:.1f} ms | {overall['physical_install_aoi_ms_cell_median']:.1f} ms | {overall['action_service_install_latency_ms_cell_median']:.1f} ms |",
        f"| Median cell time-weighted map age | {overall['prior_physical_map_aoi_ms_cell_median']:.1f} ms | {overall['physical_map_aoi_ms_cell_median']:.1f} ms | {overall['action_clock_map_age_ms_cell_median']:.1f} ms |",
    ]
    for budget in BUDGETS_MS:
        lines.append(
            f"| Route time with map age ≤{budget} ms | "
            f"{100 * overall[f'prior_physical_fresh_map_time_ms_le_{budget}_fraction']:.2f}% | "
            f"{100 * overall[f'physical_fresh_map_time_ms_le_{budget}_fraction']:.2f}% | "
            f"{100 * overall[f'action_clock_fresh_map_time_ms_le_{budget}_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Preparation before the action boundary is measured, not guessed:",
            f"median {overall['capture_to_action_start_ms_cell_median']:.1f} ms and",
            f"cell-median p95 {overall['capture_to_action_start_ms_p95_cell_median']:.1f} ms.",
            f"The causal correction delayed {overall['imputed_arrivals_floored_at_transmission_start']:,}",
            "imputed arrivals that the prior replay had placed before their own",
            "measured UE transmission-start boundary (encoding complete, before",
            "the first datagram). Observed arrivals were not changed. The later",
            "send-loop completion is not a causal lower bound because edge receipt",
            "can overlap the host's multi-datagram send loop.",
            "",
            "## FIFO versus strict latest-only after correction",
            "",
            "| Queue policy | Physical map AoI | Action-clock map age | Physical fresh ≤200 | Action-clock fresh ≤200 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICY_ORDER:
        row = document["aggregate"][policy.value]["overall"]
        lines.append(
            f"| {policy.value} | {row['physical_map_aoi_ms_cell_median']:.1f} ms | "
            f"{row['action_clock_map_age_ms_cell_median']:.1f} ms | "
            f"{100 * row['physical_fresh_map_time_ms_le_200_fraction']:.2f}% | "
            f"{100 * row['action_clock_fresh_map_time_ms_le_200_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Strict latest-only by network profile",
            "",
            "| Profile | Capture→action start | Physical map AoI | Action-clock map age | Physical fresh ≤200 | Action-clock fresh ≤200 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in PROFILE_ORDER:
        row = latest["by_network_profile"][profile]
        lines.append(
            f"| {profile} | {row['capture_to_action_start_ms_cell_median']:.1f} ms | "
            f"{row['physical_map_aoi_ms_cell_median']:.1f} ms | "
            f"{row['action_clock_map_age_ms_cell_median']:.1f} ms | "
            f"{100 * row['physical_fresh_map_time_ms_le_200_fraction']:.2f}% | "
            f"{100 * row['action_clock_fresh_map_time_ms_le_200_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Raw-freshness action winners under strict latest-only",
            "",
            "| Profile | Budget | Capture-clock winner | Capture fresh | Action-clock winner | Action-clock fresh |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in PROFILE_ORDER:
        for budget in BUDGETS_MS:
            selected = [
                row
                for row in winners
                if row["network_profile"] == profile
                and int(row["freshness_budget_ms"]) == budget
                and row["objective"] == "RAW_FRESHNESS"
            ]
            physical = next(row for row in selected if row["clock"] == "PHYSICAL_CAPTURE_CLOCK")
            action = next(row for row in selected if row["clock"] == "ACTION_START_CLOCK")
            lines.append(
                f"| {profile} | {budget} ms | {physical['action_id']} | "
                f"{100 * float(physical['fresh_map_fraction']):.2f}% | "
                f"{action['action_id']} | {100 * float(action['fresh_map_fraction']):.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Policy interpretation",
            "",
            "- Reward physical map freshness/utility; do not reward the action clock",
            "  as though it were the age of the sensed world.",
            "- Give the policy the input age at decision time. That state tells it",
            "  how much of the freshness budget preparation has already consumed.",
            "- Use action-service latency for action attribution and diagnosis. It",
            "  prevents fixed CARLA/sensor work from being blamed on compression.",
            "- The 100 ms source cadence is a sampling interval, not a latency term.",
            "- Both FIFO and strict latest-only reuse the same measured transport",
            "  surface and final edge calibration; this remains a counterfactual",
            "  queue replay, not a new live campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output already exists: {output}")
    prior_hashes = prior.verify_manifest(PRIOR_ANALYSIS)
    prior_rows = read_csv(PRIOR_ANALYSIS / "action_network_policy_freshness.csv")
    prior_by_key = {
        (row["cell_id"], row["queue_policy"]): row for row in prior_rows
    }
    require(len(prior_by_key) == 576, "prior analysis does not contain 576 unique rows")

    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    require(len(cells) == 288, "campaign table is not 288 cells")
    bridge_ns, clock_audit, per_frame_hashes = collect_clock_bridge(cells)
    quality = source._quality_by_action()
    provenance = final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        pool_hashes,
    ) = source._build_empirical_pools(cells)
    require(pool_hashes == per_frame_hashes, "per-frame hash verification drift")

    rows: list[dict[str, Any]] = []
    prep_values_all: list[float] = []
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = sorted(source._sent_rows(attempt), key=lambda row: float(row["capture_wall_s"]))
        action_starts: dict[int, int] = {}
        prep_ms: list[float] = []
        for sequence, row in enumerate(sent):
            capture_ns = wall_seconds_to_ns(row["capture_wall_s"])
            action_start_ns = int(row["capture_started_ns"]) + bridge_ns
            delay_ms = (action_start_ns - capture_ns) / 1e6
            require(0 <= delay_ms < 1_000, f"{cell['cell_id']}: invalid capture-to-action delay")
            action_starts[sequence] = action_start_ns
            prep_ms.append(delay_ms)
        prep_values_all.extend(prep_ms)
        family = cell["family"]
        frames, counters = source._candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=calibration[family],
            publication_samples=publication_samples[family],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get(
                (int(cell["action_id"]), cell["network_profile"]), ()
            ),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        require(len(frames) == len(sent), f"{cell['cell_id']}: frame count drift")
        corrected_frames, floored, floor_delta_ns = enforce_imputed_arrival_causality(
            frames, sent, bridge_ns, cell["cell_id"]
        )
        feature_bytes = [frame.feature_bytes for frame in frames]
        for policy in POLICY_ORDER:
            prior_simulation = simulate_queue_policy(
                frames,
                policy=policy,
                observation_tail_ns=prior.OBSERVATION_TAIL_NS,
            )
            prior_physical = prior.summarize_simulation(
                prior_simulation, input_frames=len(frames)
            )
            previous = prior_by_key[(cell["cell_id"], policy.value)]
            for key in (
                "useful_installs_during_observation",
                "useful_install_aoi_ms_median",
                "time_weighted_map_aoi_ms_when_available",
                "fresh_map_time_ms_le_150_fraction",
                "fresh_map_time_ms_le_200_fraction",
                "fresh_map_time_ms_le_250_fraction",
            ):
                left = prior_physical[key]
                right = source._f(previous.get(key, ""))
                if left is None or right is None:
                    require(left is None and right is None, f"{cell['cell_id']} {key}: null drift")
                else:
                    require(abs(float(left) - float(right)) < 1e-9, f"{cell['cell_id']} {key}: prior result drift")
            simulation = simulate_queue_policy(
                corrected_frames,
                policy=policy,
                observation_tail_ns=prior.OBSERVATION_TAIL_NS,
            )
            physical = prior.summarize_simulation(
                simulation, input_frames=len(corrected_frames)
            )
            action = dual_clock_summary(
                simulation,
                action_starts,
                prep_ms,
                len(frames),
                f"{cell['cell_id']} {policy.value}",
            )
            row: dict[str, Any] = {
                "cell_id": cell["cell_id"],
                "action_id": int(cell["action_id"]),
                "profile_id": cell["profile_id"],
                "network_profile": cell["network_profile"],
                "family": cell["family"],
                "quantizer": cell["quantizer"],
                "q": float(cell["q"]),
                "q_e4": int(cell["q_e4"]),
                "keep_count": int(cell["keep_count"]),
                "median_feature_bytes": median(feature_bytes),
                "queue_policy": policy.value,
                "input_frames": len(frames),
                "useful_installs_during_observation": physical["useful_installs_during_observation"],
                "map_observation_duration_s": physical["map_observation_duration_s"],
                "prior_physical_install_aoi_ms_median": prior_physical["useful_install_aoi_ms_median"],
                "prior_physical_install_aoi_ms_p95": prior_physical["useful_install_aoi_ms_p95"],
                "prior_physical_map_aoi_ms_when_available": prior_physical["time_weighted_map_aoi_ms_when_available"],
                "physical_install_aoi_ms_median": physical["useful_install_aoi_ms_median"],
                "physical_install_aoi_ms_p95": physical["useful_install_aoi_ms_p95"],
                "physical_map_aoi_ms_when_available": physical["time_weighted_map_aoi_ms_when_available"],
                "physical_map_available_fraction": physical["map_available_fraction"],
                "imputed_arrivals_floored_at_transmission_start": floored,
                "imputed_arrival_floor_total_ms": floor_delta_ns / 1e6,
                **action,
                **quality[int(cell["action_id"])],
                **counters,
                "source_per_frame_sha256": per_frame_hashes[cell["cell_id"]],
            }
            for budget in BUDGETS_MS:
                prior_fraction = float(
                    prior_physical[f"fresh_map_time_ms_le_{budget}_fraction"]
                )
                physical_fraction = float(physical[f"fresh_map_time_ms_le_{budget}_fraction"])
                action_fraction = float(row[f"action_clock_fresh_map_time_ms_le_{budget}_fraction"])
                row[f"prior_physical_fresh_map_time_ms_le_{budget}_fraction"] = prior_fraction
                row[f"physical_fresh_map_time_ms_le_{budget}_fraction"] = physical_fraction
                row[f"action_minus_physical_fresh_map_fraction_ms_le_{budget}"] = action_fraction - physical_fraction
                for cls, field in (
                    ("person", "val_canonical_person_f1"),
                    ("vehicle", "val_vehicle_f1"),
                ):
                    q = float(row[field] or 0.0)
                    row[f"physical_{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"] = q * physical_fraction
                    row[f"action_clock_{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"] = q * action_fraction
            rows.append(row)
        if number % 24 == 0:
            print(f"dual-clock simulation: {number}/288 cells", flush=True)

    require(len(rows) == 576, "dual-clock table is not 288 x 2")
    aggregate = aggregate_rows(rows)
    winners = winner_rows(rows)
    eligibility = eligibility_rows(rows)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "OFFLINE_COUNTERFACTUAL_NOT_LIVE_REMEASUREMENT",
        "budgets_ms": list(BUDGETS_MS),
        "primary_queue_policy": QueuePolicy.LATEST_ONLY_NO_EXPIRY.value,
        "physical_freshness_clock": "capture_wall_s",
        "action_attribution_clock": "capture_started_ns bridged to WALL_HOST",
        "action_boundary_semantics": "entry to seven-channel tensor assembly before UE action dispatch",
        "clock_bridge": clock_audit,
        "capture_to_action_start_all_sent_ms": {
            "count": len(prep_values_all),
            "median": median(prep_values_all),
            "p95": percentile(prep_values_all, 0.95),
            "p99": percentile(prep_values_all, 0.99),
            "minimum": min(prep_values_all),
            "maximum": max(prep_values_all),
        },
        "aggregate": aggregate,
        "provenance": {
            "prior_analysis_manifest_sha256": prior.sha256(PRIOR_ANALYSIS / "artifact_manifest.json"),
            "prior_analysis_artifacts_verified": len(prior_hashes),
            "prior_analysis_rows_reproduced": 576,
            "source_per_frame_hashes_verified": len(per_frame_hashes),
            "source_verification": provenance,
            "implementation": {
                str(Path(__file__).resolve().relative_to(ROOT)): prior.sha256(Path(__file__).resolve()),
            },
        },
        "interpretation_limits": [
            "physical capture-clock AoI remains the map safety/freshness quantity",
            "action-start age is an attribution and controller-service quantity, not physical scene age",
            "the action boundary is tensor-assembly entry; tensor-assembly completion was not timestamped separately",
            "rare delay between adjacent bridge clock calls is suppressed by a robust global median",
            "queue policies are offline counterfactuals over measured transport and calibrated final edge service",
            "no 288-cell live measurement was rerun or modified",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "action_network_dual_clock_freshness.csv", rows)
    write_csv(output / "top_actions_by_clock_profile_budget.csv", winners)
    write_csv(output / "eligible_actions_by_clock_profile_budget.csv", eligibility)
    atomic_json(output / "analysis.json", document)
    atomic_text(output / "REPORT.md", report(document, winners))
    return document


def finalize(output: Path, document: Mapping[str, Any]) -> None:
    primary = (
        "action_network_dual_clock_freshness.csv",
        "top_actions_by_clock_profile_budget.csv",
        "eligible_actions_by_clock_profile_budget.csv",
        "analysis.json",
        "REPORT.md",
    )
    hashes = {name: prior.sha256(output / name) for name in primary}
    atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": hashes},
    )
    atomic_text(
        output / TERMINAL,
        json.dumps(
            {
                "schema": f"{SCHEMA}.terminal",
                "status": document["status"],
                "analysis_sha256": hashes["analysis.json"],
                "dual_clock_table_sha256": hashes["action_network_dual_clock_freshness.csv"],
            },
            sort_keys=True,
        )
        + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = run(args.output.resolve())
    finalize(args.output.resolve(), document)
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

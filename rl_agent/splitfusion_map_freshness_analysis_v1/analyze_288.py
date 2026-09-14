#!/usr/bin/env python3
"""Join action quality to post-optimization queue/freshness counterfactuals.

This analysis never mutates the completed campaign.  It verifies and reuses
the measured 288-cell capture/arrival surface, applies the final qualified
edge-service calibration, and changes only the edge queue discipline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288 as source,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import (
    counterfactual_288_final_v3 as final_v3,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.queue_models import (
    QueueOutcome,
    QueuePolicy,
    QueueReason,
    QueueSimulation,
    simulate_queue_policy,
)


ROOT = Path(__file__).resolve().parents[2]
BUDGETS_MS = (150, 200, 250)
BUDGETS_NS = tuple(value * 1_000_000 for value in BUDGETS_MS)
OBSERVATION_TAIL_NS = 500_000_000
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "FADE_RECOVERY",
    "ADVERSE_STABLE",
)
POLICY_ORDER = (
    QueuePolicy.FIFO_NO_DISCARD,
    QueuePolicy.LATEST_ONLY_NO_EXPIRY,
)
SCHEMA = "scenesense.splitfusion.map_freshness_policy_analysis.v2"
TERMINAL = "SPLITFUSION_MAP_FRESHNESS_POLICY_ANALYSIS_COMPLETE"
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260913_action_network_freshness_v2"
)
FINAL_COUNTERFACTUAL = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260911_final_v3_predicted_horizon_288"
)


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return None if not data else float(statistics.median(data))


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def distribution_fields(prefix: str, values: Iterable[float]) -> dict[str, Any]:
    """Return the registered empirical quantiles without fitting a model."""

    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    output: dict[str, Any] = {
        f"{prefix}_count": len(data),
        f"{prefix}_minimum": None if not data else data[0],
        f"{prefix}_maximum": None if not data else data[-1],
    }
    for label, probability in (
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
    ):
        output[f"{prefix}_{label}"] = percentile(data, probability)
    return output


def verify_manifest(root: Path, name: str = "artifact_manifest.json") -> dict[str, str]:
    path = root / name
    require(path.is_file(), f"manifest absent: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    hashes = document.get("sha256") or document.get("files")
    require(isinstance(hashes, dict) and hashes, f"invalid manifest: {path}")
    for relative, expected in hashes.items():
        artifact = root / relative
        require(artifact.is_file(), f"manifest artifact absent: {artifact}")
        require(sha256(artifact) == expected, f"manifest hash drift: {artifact}")
    return {str(key): str(value) for key, value in hashes.items()}


def useful_installations(outcomes: Sequence[QueueOutcome]) -> list[QueueOutcome]:
    installed = sorted(
        (
            item
            for item in outcomes
            if item.reason is QueueReason.RESULT_PUBLISHED
            and item.install_ns is not None
        ),
        key=lambda item: (int(item.install_ns), item.frame.sequence_id),
    )
    useful: list[QueueOutcome] = []
    newest_capture = -1
    for item in installed:
        if item.frame.capture_ns > newest_capture:
            useful.append(item)
            newest_capture = item.frame.capture_ns
    return useful


def map_process_metrics(
    simulation: QueueSimulation,
    useful: Sequence[QueueOutcome],
) -> dict[str, Any]:
    start = simulation.observation_start_ns
    end = simulation.observation_end_ns
    duration = end - start
    require(duration > 0, "map observation duration is not positive")
    relevant = [
        item for item in useful if item.install_ns is not None and item.install_ns < end
    ]
    output: dict[str, Any] = {
        "map_observation_duration_s": duration / 1e9,
        "first_useful_install_delay_from_observation_start_ms": None,
        "map_available_fraction": 0.0,
        "time_weighted_map_aoi_ms_when_available": None,
    }
    for budget in BUDGETS_MS:
        output[f"fresh_map_time_ms_le_{budget}_fraction"] = 0.0
        output[f"soft_map_freshness_tau_{budget}_mean"] = 0.0
    if not relevant:
        return output

    first_install = max(start, int(relevant[0].install_ns))
    output["first_useful_install_delay_from_observation_start_ms"] = (
        first_install - start
    ) / 1e6
    available_ns = max(0, end - first_install)
    output["map_available_fraction"] = available_ns / duration
    area_ns2 = 0.0
    fresh_ns = {budget: 0 for budget in BUDGETS_NS}
    soft_area = {budget: 0.0 for budget in BUDGETS_NS}

    for index, item in enumerate(relevant):
        interval_start = max(first_install, int(item.install_ns))
        interval_end = (
            min(end, int(relevant[index + 1].install_ns))
            if index + 1 < len(relevant)
            else end
        )
        if interval_end <= interval_start:
            continue
        capture = item.frame.capture_ns
        age_start = interval_start - capture
        age_end = interval_end - capture
        interval = interval_end - interval_start
        area_ns2 += float(age_start) * interval + 0.5 * float(interval) ** 2
        for budget in BUDGETS_NS:
            crossing = capture + budget
            fresh_ns[budget] += max(
                0, min(interval_end, crossing) - interval_start
            )
            soft_area[budget] += float(budget) * (
                math.exp(-max(0, age_start) / budget)
                - math.exp(-max(0, age_end) / budget)
            )

    output["time_weighted_map_aoi_ms_when_available"] = (
        area_ns2 / available_ns / 1e6 if available_ns else None
    )
    for budget_ms, budget_ns in zip(BUDGETS_MS, BUDGETS_NS):
        output[f"fresh_map_time_ms_le_{budget_ms}_fraction"] = (
            fresh_ns[budget_ns] / duration
        )
        output[f"soft_map_freshness_tau_{budget_ms}_mean"] = (
            soft_area[budget_ns] / duration
        )
    return output


def summarize_simulation(
    simulation: QueueSimulation,
    *,
    input_frames: int,
) -> dict[str, Any]:
    outcomes = list(simulation.outcomes)
    installed = [
        item
        for item in outcomes
        if item.reason is QueueReason.RESULT_PUBLISHED
        and item.install_ns is not None
    ]
    useful = useful_installations(outcomes)
    observed_useful = [
        item
        for item in useful
        if item.install_ns is not None
        and item.install_ns < simulation.observation_end_ns
    ]
    queue_ms = [
        item.queue_wait_ns / 1e6
        for item in outcomes
        if item.queue_wait_ns is not None
    ]
    install_ms = [
        item.install_aoi_ns / 1e6
        for item in installed
        if item.install_aoi_ns is not None
    ]
    useful_install_ms = [
        item.install_aoi_ns / 1e6
        for item in observed_useful
        if item.install_aoi_ns is not None
    ]
    useful_ids = {item.frame.sequence_id for item in observed_useful}
    gap_ms = [
        (int(right.install_ns) - int(left.install_ns)) / 1e6
        for left, right in zip(observed_useful, observed_useful[1:])
    ]
    reasons = {
        reason.value: sum(item.reason is reason for item in outcomes)
        for reason in QueueReason
    }
    output: dict[str, Any] = {
        "input_frames": input_frames,
        "edge_scheduler_input_frames": sum(
            item.frame.arrival_ns is not None for item in outcomes
        ),
        "eventual_installs_after_full_drain": len(installed),
        "useful_installs_during_observation": len(observed_useful),
        "rate_useful_install_per_sent": ratio(len(observed_useful), input_frames),
        "install_aoi_ms_median": median(install_ms),
        "install_aoi_ms_p95": percentile(install_ms, 0.95),
        "useful_install_aoi_ms_median": median(useful_install_ms),
        "useful_install_aoi_ms_p95": percentile(useful_install_ms, 0.95),
        "inter_useful_install_gap_ms_median": median(gap_ms),
        "inter_useful_install_gap_ms_p95": percentile(gap_ms, 0.95),
        "queue_wait_ms_median": median(queue_ms),
        "queue_wait_ms_p95": percentile(queue_ms, 0.95),
        "queue_wait_ms_max": max(queue_ms) if queue_ms else None,
        "compute_queue_high_water": simulation.compute_queue_high_water,
        "publication_queue_high_water": simulation.publication_queue_high_water,
        "feature_bytes_charged": sum(item.frame.feature_bytes for item in outcomes),
        "feature_bytes_without_useful_install": sum(
            item.frame.feature_bytes
            for item in outcomes
            if item.frame.sequence_id not in useful_ids
        ),
        "compute_ms_charged": sum(item.compute_spent_ns for item in outcomes) / 1e6,
        "publication_ms_charged": (
            sum(item.publication_spent_ns for item in outcomes) / 1e6
        ),
        **distribution_fields("queue_wait_ms", queue_ms),
        **distribution_fields("useful_install_aoi_ms", useful_install_ms),
        **distribution_fields("inter_useful_install_gap_ms", gap_ms),
        **{f"reason_{key}": value for key, value in reasons.items()},
        **map_process_metrics(simulation, useful),
    }
    for budget in BUDGETS_MS:
        timely = sum(value <= budget for value in useful_install_ms)
        gap_compliant = sum(value <= budget for value in gap_ms)
        output[f"useful_installs_ms_le_{budget}"] = timely
        output[f"timely_useful_install_yield_ms_le_{budget}"] = ratio(
            timely, input_frames
        )
        output[f"inter_update_gap_ms_le_{budget}_fraction"] = ratio(
            gap_compliant, len(gap_ms)
        )
    require(sum(reasons.values()) == input_frames, "terminal accounting drift")
    return output


def row_for_policy(
    *,
    cell: Mapping[str, str],
    policy: QueuePolicy,
    simulation: QueueSimulation,
    summary: Mapping[str, Any],
    counters: Mapping[str, int],
    quality: Mapping[str, Any],
    feature_bytes: Sequence[int],
    per_frame_sha256: str,
) -> dict[str, Any]:
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
        "queue_policy": policy.value,
        "median_feature_bytes": median(feature_bytes),
        "source_per_frame_sha256": per_frame_sha256,
        "observation_tail_ms": OBSERVATION_TAIL_NS / 1e6,
        "distribution_contract": {
            "method": "exact empirical nearest-rank quantiles per action-network-policy cell",
            "probabilities": [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            "metrics": [
                "edge queue wait",
                "useful capture-to-install AoI",
                "inter-useful-install gap",
            ],
            "freshness_thresholds_ms": list(BUDGETS_MS),
            "parametric_fit": False,
        },
        **counters,
        **quality,
        **summary,
    }
    for budget in BUDGETS_MS:
        fresh = float(row[f"fresh_map_time_ms_le_{budget}_fraction"])
        yield_value = float(
            row[f"timely_useful_install_yield_ms_le_{budget}"] or 0.0
        )
        for cls, quality_field in (
            ("vehicle", "val_vehicle_f1"),
            ("person", "val_canonical_person_f1"),
        ):
            q_value = finite(row.get(quality_field)) or 0.0
            row[f"{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"] = q_value * fresh
            row[f"{cls}_f1_x_timely_update_yield_ms_le_{budget}"] = (
                q_value * yield_value
            )
    require(
        row["edge_scheduler_input_frames"] == row["measured_edge_admissions"],
        f"{cell['cell_id']} {policy.value}: admission total drift",
    )
    return row


def aggregate_policy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def block(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        sent = sum(int(item["input_frames"]) for item in items)
        duration_s = sum(float(item["map_observation_duration_s"]) for item in items)
        output: dict[str, Any] = {
            "cells": len(items),
            "sent": sent,
            "edge_admissions": sum(
                int(item["edge_scheduler_input_frames"]) for item in items
            ),
            "eventual_installs_after_full_drain": sum(
                int(item["eventual_installs_after_full_drain"]) for item in items
            ),
            "useful_installs_during_observation": sum(
                int(item["useful_installs_during_observation"]) for item in items
            ),
            "rate_useful_install_per_sent": ratio(
                sum(int(item["useful_installs_during_observation"]) for item in items),
                sent,
            ),
            "queue_wait_ms_cell_median": median(
                float(item["queue_wait_ms_median"])
                for item in items
                if item["queue_wait_ms_median"] is not None
            ),
            "queue_wait_ms_p95_cell_median": median(
                float(item["queue_wait_ms_p95"])
                for item in items
                if item["queue_wait_ms_p95"] is not None
            ),
            "queue_wait_ms_max": max(
                (
                    float(item["queue_wait_ms_max"])
                    for item in items
                    if item["queue_wait_ms_max"] is not None
                ),
                default=None,
            ),
            "compute_queue_high_water_max": max(
                int(item["compute_queue_high_water"]) for item in items
            ),
            "useful_install_aoi_ms_cell_median": median(
                float(item["useful_install_aoi_ms_median"])
                for item in items
                if item["useful_install_aoi_ms_median"] is not None
            ),
            "map_aoi_ms_cell_median": median(
                float(item["time_weighted_map_aoi_ms_when_available"])
                for item in items
                if item["time_weighted_map_aoi_ms_when_available"] is not None
            ),
            "map_available_fraction_duration_weighted": ratio(
                sum(
                    float(item["map_available_fraction"])
                    * float(item["map_observation_duration_s"])
                    for item in items
                ),
                duration_s,
            ),
            "superseded_compute": sum(
                int(item["reason_SUPERSEDED_PENDING_COMPUTE"]) for item in items
            ),
            "superseded_publication": sum(
                int(item["reason_SUPERSEDED_PENDING_PUBLICATION"]) for item in items
            ),
        }
        for budget in BUDGETS_MS:
            output[f"timely_useful_install_yield_ms_le_{budget}"] = ratio(
                sum(int(item[f"useful_installs_ms_le_{budget}"]) for item in items),
                sent,
            )
            output[f"fresh_map_time_ms_le_{budget}_fraction"] = ratio(
                sum(
                    float(item[f"fresh_map_time_ms_le_{budget}_fraction"])
                    * float(item["map_observation_duration_s"])
                    for item in items
                ),
                duration_s,
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


def pareto_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["queue_policy"] == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value
    ]
    output: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        profile_rows = [row for row in selected if row["network_profile"] == profile]
        require(len(profile_rows) == 72, f"{profile}: not 72 strict-latest rows")
        for budget in BUDGETS_MS:
            for cls in ("vehicle", "person"):
                score_key = f"{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"
                for candidate in profile_rows:
                    payload = float(candidate["median_feature_bytes"])
                    score = float(candidate[score_key])
                    dominated = any(
                        float(other["median_feature_bytes"]) <= payload
                        and float(other[score_key]) >= score
                        and (
                            float(other["median_feature_bytes"]) < payload
                            or float(other[score_key]) > score
                        )
                        for other in profile_rows
                        if other is not candidate
                    )
                    output.append(
                        {
                            "network_profile": profile,
                            "freshness_budget_ms": budget,
                            "class_name": cls,
                            "action_id": int(candidate["action_id"]),
                            "profile_id": candidate["profile_id"],
                            "family": candidate["family"],
                            "quantizer": candidate["quantizer"],
                            "q": candidate["q"],
                            "median_feature_bytes": payload,
                            "validation_f1": candidate[
                                "val_vehicle_f1"
                                if cls == "vehicle"
                                else "val_canonical_person_f1"
                            ],
                            "fresh_map_fraction": candidate[
                                f"fresh_map_time_ms_le_{budget}_fraction"
                            ],
                            "quality_weighted_fresh_map_score": score,
                            "pareto": int(not dominated),
                        }
                    )
    return output


def network_sensitivity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest = [
        row
        for row in rows
        if row["queue_policy"] == QueuePolicy.LATEST_ONLY_NO_EXPIRY.value
    ]
    output: list[dict[str, Any]] = []
    for action in range(72):
        action_rows = [row for row in latest if int(row["action_id"]) == action]
        require(len(action_rows) == 4, f"action {action}: not four network rows")
        for budget in BUDGETS_MS:
            fresh_key = f"fresh_map_time_ms_le_{budget}_fraction"
            for cls in ("vehicle", "person"):
                score_key = f"{cls}_f1_x_fresh_map_fraction_ms_le_{budget}"
                worst = min(action_rows, key=lambda row: float(row[score_key]))
                best = max(action_rows, key=lambda row: float(row[score_key]))
                output.append(
                    {
                        "action_id": action,
                        "profile_id": action_rows[0]["profile_id"],
                        "family": action_rows[0]["family"],
                        "quantizer": action_rows[0]["quantizer"],
                        "q": action_rows[0]["q"],
                        "freshness_budget_ms": budget,
                        "class_name": cls,
                        "worst_network_profile": worst["network_profile"],
                        "best_network_profile": best["network_profile"],
                        "worst_quality_weighted_fresh_map_score": worst[score_key],
                        "best_quality_weighted_fresh_map_score": best[score_key],
                        "network_score_range": float(best[score_key])
                        - float(worst[score_key]),
                        "worst_fresh_map_fraction": worst[fresh_key],
                        "best_fresh_map_fraction": best[fresh_key],
                    }
                )
    return output


def aggregate_error_rows(
    groups: Mapping[tuple[Any, ...], Mapping[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        values = groups[key]
        tp = int(values["tp"])
        fp = int(values["fp"])
        fn = int(values["fn"])
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        source_errors = values["source"]
        aligned_errors = values["aligned"]
        deltas = values["delta"]
        rows.append(
            {
                **dict(zip(columns, key)),
                "rows": int(values["rows"]),
                "rows_with_both_localization_errors": len(deltas),
                "source_time_precision_micro": precision,
                "source_time_recall_micro": recall,
                "source_time_f1_micro": f1,
                "source_time_xy_error_m_median": median(source_errors),
                "aligned_retrieval_xy_error_m_median": median(aligned_errors),
                "aligned_minus_source_xy_error_m_median": median(deltas),
                "aligned_minus_source_xy_error_m_p95": percentile(deltas, 0.95),
            }
        )
    return rows


def aligned_localization_analysis(
    cells: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    band_groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "tp": 0, "fp": 0, "fn": 0, "source": [], "aligned": [], "delta": []}
    )
    action_groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "tp": 0, "fp": 0, "fn": 0, "source": [], "aligned": [], "delta": []}
    )
    verified = 0
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        source._attempt_manifest_hash(attempt, "per_frame_metrics.csv")
        source._attempt_manifest_hash(attempt, "perception_metrics.csv")
        verified += 2
        per_frame = {
            int(row["frame_id"]): finite(row.get("install_aoi_ms"))
            for row in source._sent_rows(attempt)
        }
        for row in source._read_csv(attempt / "perception_metrics.csv"):
            frame_id = int(row["frame_id"])
            aoi = per_frame.get(frame_id)
            if aoi is None:
                continue
            if aoi <= 150:
                band = "LE_150"
            elif aoi <= 200:
                band = "GT_150_LE_200"
            elif aoi <= 250:
                band = "GT_200_LE_250"
            elif aoi <= 500:
                band = "GT_250_LE_500"
            else:
                band = "GT_500"
            cls = str(row["class_name"])
            keys = (
                (cell["network_profile"], cls, band),
                (int(cell["action_id"]), cell["network_profile"], cls),
            )
            for group, key in ((band_groups, keys[0]), (action_groups, keys[1])):
                target = group[key]
                target["rows"] += 1
                for count in ("tp", "fp", "fn"):
                    value = finite(row.get(count))
                    target[count] += 0 if value is None else int(value)
                source_error = finite(row.get("source_time_world_xy_error_m"))
                aligned_error = finite(row.get("aligned_world_xy_error_m"))
                if source_error is not None:
                    target["source"].append(source_error)
                if aligned_error is not None:
                    target["aligned"].append(aligned_error)
                if source_error is not None and aligned_error is not None:
                    target["delta"].append(aligned_error - source_error)
        if number % 24 == 0:
            print(f"aligned-localization pass: {number}/288 cells", flush=True)
    return (
        aggregate_error_rows(
            band_groups, ("network_profile", "class_name", "install_aoi_band_ms")
        ),
        aggregate_error_rows(
            action_groups, ("action_id", "network_profile", "class_name")
        ),
        {
            "attempt_files_hash_verified": verified,
            "interpretation": (
                "source-time detection metrics and near-install exact-record-retrieval "
                "localization check; aligned matching support may differ from source matching"
            ),
        },
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    columns = list(rows[0])
    require(all(list(row) == columns for row in rows), f"column drift: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, document: Any) -> None:
    atomic_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def report(document: Mapping[str, Any]) -> str:
    aggregate = document["aggregate"]
    lines = [
        "# SplitFusion action–network–freshness policy analysis",
        "",
        "This is an offline counterfactual analysis of the immutable 288-cell",
        "campaign. Measured captures, payloads, complete reassemblies, edge",
        "admissions and the final qualified edge-service calibration are held",
        "fixed. Queue discipline is the only experimental factor.",
        "",
        "The budgets are 150, 200 and 250 ms. A 100 ms budget is not used as",
        "the primary analysis because the source itself produces frames every",
        "100 ms; the historical 100 ms figure remains provenance only.",
        "",
        "## Queue-policy comparison",
        "",
        "| policy | queue median | queue p95 | max queue | useful install/sent | map AoI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_ORDER:
        item = aggregate[policy.value]["overall"]
        lines.append(
            f"| {policy.value} | {item['queue_wait_ms_cell_median']:.1f} ms | "
            f"{item['queue_wait_ms_p95_cell_median']:.1f} ms | "
            f"{item['queue_wait_ms_max']:.1f} ms | "
            f"{item['rate_useful_install_per_sent']:.4f} | "
            f"{item['map_aoi_ms_cell_median']:.1f} ms |"
        )
    lines.extend(["", "## Strict latest-only by network profile", ""])
    header = "| profile | useful install/sent | map AoI | " + " | ".join(
        f"map fresh ≤{budget} ms" for budget in BUDGETS_MS
    ) + " |"
    lines.extend([header, "|---|---:|---:|---:|---:|---:|"])
    latest = aggregate[QueuePolicy.LATEST_ONLY_NO_EXPIRY.value]
    for profile in PROFILE_ORDER:
        item = latest["by_network_profile"][profile]
        fields = " | ".join(
            f"{100 * item[f'fresh_map_time_ms_le_{budget}_fraction']:.2f}%"
            for budget in BUDGETS_MS
        )
        lines.append(
            f"| {profile} | {item['rate_useful_install_per_sent']:.4f} | "
            f"{item['map_aoi_ms_cell_median']:.1f} ms | {fields} |"
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- `FIFO_NO_DISCARD` drains every admitted frame, including work that",
            "  completes after the route observation window. It is a backlog",
            "  baseline, not the recommended map scheduler.",
            "- `LATEST_ONLY_NO_EXPIRY` never interrupts active CUDA work. While it",
            "  runs, each newer arrival replaces the single pending frame. The",
            "  newest pending frame is always selected next, regardless of how",
            "  long it waited; there is no 25 ms expiry.",
            "- Pre-edge transport failures and measured admission rejections are",
            "  identical under both policies.",
            "- Fresh-map fractions count the initial no-map interval as not fresh.",
            "  Time-weighted map AoI is conditional on a map being available and is",
            "  therefore reported separately from map availability.",
            "- Quality-weighted freshness is a decision surrogate, not a claim that",
            "  network conditions change the intrinsic validation accuracy.",
            "- The aligned localization check is secondary: aligned truth is sampled",
            "  during exact-record retrieval after ACK, and its matching support can",
            "  differ from the source-time matches.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output exists: {output}")
    final_hashes = verify_manifest(FINAL_COUNTERFACTUAL)
    provenance = final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    require(len(cells) == 288, "campaign table does not contain 288 cells")
    quality = source._quality_by_action()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        source_hashes,
    ) = source._build_empirical_pools(cells)

    rows: list[dict[str, Any]] = []
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = source._sent_rows(attempt)
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
        feature_bytes = [item.feature_bytes for item in frames]
        for policy in POLICY_ORDER:
            simulation = simulate_queue_policy(
                frames,
                policy=policy,
                observation_tail_ns=OBSERVATION_TAIL_NS,
            )
            summary = summarize_simulation(
                simulation,
                input_frames=len(frames),
            )
            rows.append(
                row_for_policy(
                    cell=cell,
                    policy=policy,
                    simulation=simulation,
                    summary=summary,
                    counters=counters,
                    quality=quality[int(cell["action_id"])],
                    feature_bytes=feature_bytes,
                    per_frame_sha256=source_hashes[cell["cell_id"]],
                )
            )
        if number % 24 == 0:
            print(f"queue-policy pass: {number}/288 cells", flush=True)

    require(len(rows) == 576, "policy table is not 288 cells x 2 policies")
    pareto = pareto_rows(rows)
    sensitivity = network_sensitivity_rows(rows)
    aligned_bands, aligned_actions, aligned_provenance = (
        aligned_localization_analysis(cells)
    )
    aggregate = aggregate_policy_rows(rows)
    document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "OFFLINE_COUNTERFACTUAL_NOT_LIVE_REMEASUREMENT",
        "budgets_ms": list(BUDGETS_MS),
        "queue_policies": [policy.value for policy in POLICY_ORDER],
        "primary_policy": QueuePolicy.LATEST_ONLY_NO_EXPIRY.value,
        "observation_tail_ms": OBSERVATION_TAIL_NS / 1e6,
        "provenance": {
            "final_counterfactual_manifest_sha256": sha256(
                FINAL_COUNTERFACTUAL / "artifact_manifest.json"
            ),
            "final_counterfactual_artifacts_verified": len(final_hashes),
            "source_verification": provenance,
            "source_per_frame_hashes_verified": len(source_hashes),
            "aligned_localization": aligned_provenance,
            "implementation": {
                str(Path(__file__).resolve().relative_to(ROOT)): sha256(
                    Path(__file__).resolve()
                ),
                str(
                    (Path(__file__).with_name("queue_models.py"))
                    .resolve()
                    .relative_to(ROOT)
                ): sha256(Path(__file__).with_name("queue_models.py")),
            },
        },
        "family_calibration": calibration,
        "aggregate": aggregate,
        "interpretation_limits": [
            "counterfactual queue replay, not live remeasurement",
            "measured per-cell reassembly and admission totals remain fixed",
            "missing admitted identities/timestamps retain deterministic imputations",
            "family service calibration is extrapolated from one live anchor",
            "quality-weighted freshness uses frozen action validation quality",
            "aligned localization truth is sampled during post-ACK exact retrieval",
        ],
    }

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "action_network_policy_freshness.csv", rows)
    write_csv(output / "quality_freshness_pareto.csv", pareto)
    write_csv(output / "action_network_sensitivity.csv", sensitivity)
    write_csv(output / "aligned_localization_by_profile_aoi_band.csv", aligned_bands)
    write_csv(output / "aligned_localization_by_action_profile.csv", aligned_actions)
    atomic_json(output / "analysis.json", document)
    atomic_text(output / "REPORT.md", report(document))
    return document


def finalize(output: Path, document: Mapping[str, Any]) -> None:
    primary = (
        "action_network_policy_freshness.csv",
        "quality_freshness_pareto.csv",
        "action_network_sensitivity.csv",
        "aligned_localization_by_profile_aoi_band.csv",
        "aligned_localization_by_action_profile.csv",
        "analysis.json",
        "REPORT.md",
    )
    hashes = {name: sha256(output / name) for name in primary}
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
                "policy_table_sha256": hashes[
                    "action_network_policy_freshness.csv"
                ],
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
    output = args.output.resolve()
    document = run(output)
    finalize(output, document)
    print(json.dumps(document["aggregate"], indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

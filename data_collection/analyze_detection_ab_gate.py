#!/usr/bin/env python3
"""Analyze the pre-registered three-arm detection-quality CARLA smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from data_collection.reconcile_detection_coverage import (
    denominator_rows,
    mark_matches,
    prepare_inputs,
)
from data_collection.rescore_policy_corpus_freshness import _truthy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "data_collection/configs/detection_ab_gate_v1.yaml"
ARMS = (
    "arm1_5k_nms4_top80",
    "arm2_200k_nms4_top80",
    "arm3_200k_nms2_top120",
)
EXPECTED_CONFIG = {
    "arm1_5k_nms4_top80": (5000, 4, 80),
    "arm2_200k_nms4_top80": (200000, 4, 80),
    "arm3_200k_nms2_top120": (200000, 2, 120),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {directory}, found {len(matches)}")
    return matches[0]


def _pct(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")


def _paired_block_bootstrap_lift(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    replicates: int,
    block_length: int,
    seed: int,
) -> Dict[str, object]:
    """Return a paired percentage-point lift and moving-block 95% CI.

    Frames are paired by within-run sequence index. Circular moving blocks retain
    short temporal runs of detections instead of treating every video frame as
    independent.
    """

    left = np.asarray(baseline, dtype=np.int8)
    right = np.asarray(candidate, dtype=np.int8)
    if len(left) != len(right):
        raise ValueError("paired coverage arrays must have equal length")
    if len(left) == 0:
        return {
            "paired_rows": 0,
            "baseline_matched_rows": 0,
            "candidate_matched_rows": 0,
            "candidate_only_rows": 0,
            "baseline_only_rows": 0,
            "lift_pp": float("nan"),
            "ci95_lower_pp": float("nan"),
            "ci95_upper_pp": float("nan"),
        }
    differences = right - left
    rng = np.random.default_rng(int(seed))
    n = len(differences)
    width = max(1, min(int(block_length), n))
    block_count = int(math.ceil(n / width))
    samples = np.empty(max(1, int(replicates)), dtype=np.float64)
    offsets = np.arange(width, dtype=int)
    for index in range(len(samples)):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[index] = float(differences[indices].mean()) * 100.0
    return {
        "paired_rows": n,
        "baseline_matched_rows": int(left.sum()),
        "candidate_matched_rows": int(right.sum()),
        "candidate_only_rows": int(((left == 0) & (right == 1)).sum()),
        "baseline_only_rows": int(((left == 1) & (right == 0)).sum()),
        "lift_pp": float(differences.mean()) * 100.0,
        "ci95_lower_pp": float(np.quantile(samples, 0.025)),
        "ci95_upper_pp": float(np.quantile(samples, 0.975)),
    }


def _longest_true_dwell(mask: Sequence[bool], timestamps: Sequence[float]) -> float:
    if not len(mask):
        return 0.0
    values = np.asarray(mask, dtype=bool)
    times = np.asarray(timestamps, dtype=float)
    finite_deltas = np.diff(times)
    finite_deltas = finite_deltas[np.isfinite(finite_deltas) & (finite_deltas > 0)]
    nominal_dt = float(np.median(finite_deltas)) if len(finite_deltas) else 0.0
    longest = 0.0
    start = None
    previous = None
    for index, enabled in enumerate(values):
        contiguous = (
            previous is None
            or nominal_dt <= 0.0
            or times[index] - times[previous] <= 1.5 * nominal_dt
        )
        if enabled:
            if start is None:
                start = index
            elif not contiguous:
                longest = max(longest, times[previous] - times[start] + nominal_dt)
                start = index
        elif not enabled:
            if start is not None:
                longest = max(longest, times[previous] - times[start] + nominal_dt)
            start = None
        previous = index
    if start is not None and previous is not None:
        longest = max(longest, times[previous] - times[start] + nominal_dt)
    return float(longest)


def _trajectory_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    thresholds: Mapping[str, float],
) -> Dict[str, object]:
    count = min(len(baseline), len(candidate))
    if count == 0:
        return {"aligned_rows": 0, "pair_valid": False, "failure": "empty_target_trajectory"}
    left = baseline.iloc[:count].reset_index(drop=True)
    right = candidate.iloc[:count].reset_index(drop=True)
    row_delta = abs(len(baseline) - len(candidate)) / max(len(baseline), len(candidate))
    agreement = float((left["in_scope"].to_numpy() == right["in_scope"].to_numpy()).mean())
    distance_delta = float(
        np.nanmedian(np.abs(left["distance_m"].to_numpy() - right["distance_m"].to_numpy()))
    )
    projection_delta = np.hypot(
        left["projected_x"].to_numpy() - right["projected_x"].to_numpy(),
        left["projected_y"].to_numpy() - right["projected_y"].to_numpy(),
    )
    projection_delta_median = float(np.nanmedian(projection_delta))
    checks = {
        "target_row_delta": row_delta
        <= float(thresholds["maximum_target_row_delta_fraction"]),
        "in_scope_sequence": agreement
        >= float(thresholds["minimum_in_scope_sequence_agreement"]),
        "distance_sequence": distance_delta
        <= float(thresholds["maximum_median_distance_delta_m"]),
        "projection_sequence": projection_delta_median
        <= float(thresholds["maximum_median_projection_delta_px"]),
    }
    return {
        "aligned_rows": count,
        "target_row_delta_fraction": row_delta,
        "in_scope_sequence_agreement": agreement,
        "median_abs_distance_delta_m": distance_delta,
        "median_projection_delta_px": projection_delta_median,
        **{f"check_{name}": passed for name, passed in checks.items()},
        "pair_valid": all(checks.values()),
        "failure": "" if all(checks.values()) else ",".join(
            name for name, passed in checks.items() if not passed
        ),
    }


def _target_actor_id(
    target_class: str,
    run_manifest: Mapping[str, object],
    gt: pd.DataFrame,
) -> int:
    if target_class == "vehicle":
        tracked = run_manifest.get("tracked_lead")
        diagnostic = run_manifest.get("experiment3_target")
        if isinstance(tracked, dict):
            return int(tracked["actor_id"])
        if isinstance(diagnostic, dict):
            return int(diagnostic["actor_id"])
        raise RuntimeError("vehicle arm is missing controlled-target provenance")
    pedestrians = gt[gt["class_name"] == "pedestrian"]
    counts = pedestrians.groupby("actor_id").size().sort_values(ascending=False)
    if counts.empty:
        raise RuntimeError("pedestrian arm has no pedestrian target rows")
    return int(counts.index[0])


def _load_run(
    record: Mapping[str, object],
    thresholds: Mapping[str, float],
    timing: Mapping[str, float],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    run_dir = Path(str(record["run_dir"])).resolve()
    gt, predictions, _, _ = prepare_inputs(run_dir)
    metrics = pd.read_csv(_single(run_dir / "streams", "*_metrics.csv"))
    resolved_path = _single(run_dir / "manifests", "*_resolved_config.json")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    run_manifest_path = _single(run_dir / "manifests", "*_manifest.json")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    target_class = str(record["target_class"])
    arm = str(record["ab_arm"])
    actor_id = _target_actor_id(target_class, run_manifest, gt)
    target = gt[(gt["actor_id"].astype(int) == actor_id) & (gt["class_name"] == target_class)].copy()
    target = target.sort_values(["carla_timestamp", "frame_id"]).reset_index(drop=True)
    target["in_scope"] = _truthy(target["in_camera_frustum"]) & (
        pd.to_numeric(target["distance_m"], errors="coerce")
        <= float(thresholds["headline_range_m"])
    )
    metric_index = metrics[["frame_id", "carla_timestamp"]].copy()
    metric_index["sequence_index"] = np.arange(len(metric_index), dtype=int)
    eligible = denominator_rows(target, "current_in_frustum_le25", 854, 480)
    marked = mark_matches(eligible, predictions)
    marked = marked.merge(
        metric_index[["frame_id", "sequence_index"]], on="frame_id", how="left"
    ).sort_values("sequence_index").reset_index(drop=True)
    matched_frames = marked.loc[marked["matched"], "frame_id"].nunique()

    trajectory = target.merge(metric_index, on="frame_id", how="left", suffixes=("", "_metric"))
    trajectory = trajectory.sort_values("sequence_index", na_position="last").reset_index(drop=True)
    trajectory["in_scope"] = _truthy(trajectory["in_camera_frustum"]) & (
        pd.to_numeric(trajectory["distance_m"], errors="coerce")
        <= float(thresholds["headline_range_m"])
    )

    received = _truthy(metrics["result_received"])
    wait = pd.to_numeric(metrics["camera_frame_wait_ms"], errors="coerce").dropna()
    expected_pps, expected_nms, expected_topk = EXPECTED_CONFIG[arm]
    config_checks = {
        "pps": int(resolved.get("radar_points_per_second", -1)) == expected_pps,
        "rasterizer": str(resolved.get("radar_rasterizer")) == "fast",
        "nms": int(resolved.get("object_nms_radius_px", -1)) == expected_nms,
        "topk": int(resolved.get("topk_objects", -1)) == expected_topk,
    }
    processed_min = math.ceil(
        float(timing["minimum_processed_fraction"]) * int(record.get("requested_frames", 80))
    )
    run_checks = {
        "collector_status": str(record.get("status", ""))
        in {"complete", "complete_with_teardown_warning"},
        "actor_cleanup": bool(record.get("actor_cleanup_pass", False)),
        "processed_frames": len(metrics) >= processed_min,
        "result_received": _pct(received.sum(), len(metrics))
        >= float(timing["minimum_result_received_pct"]),
        "camera_wait_median": bool(len(wait))
        and float(wait.median()) <= float(timing["median_max_ms"]),
        "camera_wait_p95": bool(len(wait))
        and float(wait.quantile(0.95)) <= float(timing["p95_max_ms"]),
        "eligible_target_rows": len(eligible)
        >= int(thresholds["minimum_target_eligible_rows"]),
        **{f"config_{name}": passed for name, passed in config_checks.items()},
    }

    radar_points = pd.to_numeric(metrics["radar_projected_points"], errors="coerce")
    def numeric_metric(name: str) -> pd.Series:
        if name not in metrics:
            return pd.Series(np.nan, index=metrics.index, dtype=float)
        return pd.to_numeric(metrics[name], errors="coerce")

    pre_topk = numeric_metric("decode_pre_topk_above_threshold_count")
    post_nms = numeric_metric("decode_post_topk_nms_count")
    saturated = numeric_metric("decode_topk_saturated")
    diagnostics_present = numeric_metric("decode_diagnostics_present")
    run_checks["decoder_diagnostics"] = bool(
        len(pre_topk)
        and pre_topk.notna().all()
        and post_nms.notna().all()
        and (diagnostics_present == 1).all()
    )
    target_speed = numeric_metric("tracked_target_speed_mps")
    if target_class != "vehicle":
        target_speed = pd.Series(np.nan, index=metrics.index)
    metric_fast = (
        metrics[["frame_id", "carla_timestamp"]]
        .assign(target_speed_mps=target_speed)
        .merge(target[["frame_id", "in_scope"]], on="frame_id", how="left")
    )
    fast_mask = metric_fast["in_scope"].fillna(False).astype(bool) & (
        metric_fast["target_speed_mps"] >= float(thresholds["fast_target_speed_min_mps"])
    )
    fast_dwell = _longest_true_dwell(
        fast_mask.to_numpy(), metric_fast["carla_timestamp"].to_numpy()
    )
    summary = {
        "episode_id": record["episode_id"],
        "target_class": target_class,
        "arm": arm,
        "seed": int(record["seed"]),
        "target_actor_id": actor_id,
        "processed_frames": len(metrics),
        "result_received_pct": _pct(received.sum(), len(metrics)),
        "camera_wait_median_ms": float(wait.median()) if len(wait) else float("nan"),
        "camera_wait_p95_ms": float(wait.quantile(0.95)) if len(wait) else float("nan"),
        "radar_points_per_second": int(resolved["radar_points_per_second"]),
        "radar_rasterizer": resolved["radar_rasterizer"],
        "object_nms_radius_px": int(resolved["object_nms_radius_px"]),
        "topk_objects": int(resolved["topk_objects"]),
        "radar_projected_points_p50": float(radar_points.median()),
        "decode_pre_topk_max": float(pre_topk.max()) if pre_topk.notna().any() else float("nan"),
        "decode_post_topk_nms_max": float(post_nms.max())
        if post_nms.notna().any()
        else float("nan"),
        "decode_topk_saturated_frames": int((saturated == 1).sum())
        if saturated.notna().any()
        else 0,
        "target_rows": len(target),
        "eligible_target_rows": len(eligible),
        "matched_target_rows": int(marked["matched"].sum()),
        "target_object_row_coverage_pct": _pct(marked["matched"].sum(), len(marked)),
        "eligible_target_frames": int(marked["frame_id"].nunique()),
        "matched_target_frames": int(matched_frames),
        "target_frame_coverage_pct": _pct(matched_frames, marked["frame_id"].nunique()),
        "target_speed_median_mps": float(target_speed.median())
        if target_speed.notna().any()
        else float("nan"),
        "fast_in_scope_dwell_s": fast_dwell,
        **{f"check_{name}": passed for name, passed in run_checks.items()},
        "run_valid": all(run_checks.values()),
        "run_failures": "" if all(run_checks.values()) else ",".join(
            name for name, passed in run_checks.items() if not passed
        ),
        "resolved_config_sha256": _sha256(resolved_path),
        "run_manifest_sha256": _sha256(run_manifest_path),
    }
    return summary, trajectory, marked


def _range_rows(marked: pd.DataFrame, target_class: str, arm: str) -> Iterable[Dict[str, object]]:
    labels = ("0-5", "5-10", "10-15", "15-20", "20-25")
    bins = pd.cut(
        marked["distance_m"],
        bins=(0, 5, 10, 15, 20, 25),
        labels=labels,
        include_lowest=True,
    )
    for label in labels:
        group = marked[bins == label]
        yield {
            "target_class": target_class,
            "arm": arm,
            "range_bin_m": label,
            "eligible_rows": len(group),
            "matched_rows": int(group["matched"].sum()),
            "coverage_pct": _pct(group["matched"].sum(), len(group)),
        }


def _report(
    output_dir: Path,
    summary: pd.DataFrame,
    pairing: pd.DataFrame,
    ranges: pd.DataFrame,
    gates: Mapping[str, object],
    batch_dir: Path,
) -> None:
    status = str(gates["status"])
    lines = [
        "# Detection A/B Gate Report",
        "",
        f"Status: **{status}**  ",
        f"Batch: `{batch_dir}`  ",
        "Protocol: matched controlled vehicle/pedestrian, fast rasterizer in all arms.",
        "",
        "## Headline gates",
        "",
        f"- Gate 1 detection: **{'PASS' if gates['gate1_pass'] else 'FAIL'}**",
        f"- Vehicle gate: **{'PASS' if gates['vehicle_gate_pass'] else 'FAIL'}**",
        f"- Pedestrian gate: **{'PASS' if gates['pedestrian_gate_pass'] else 'FAIL'}**",
        f"- Gate 2 controlled fast-in-view realization: **{'PASS' if gates['gate2_pass'] else 'FAIL'}**",
        f"- Arm-1 vehicle low baseline: {gates['arm1_vehicle_coverage_pct']:.2f}%",
        f"- Arm-3 vehicle coverage: {gates['arm3_vehicle_coverage_pct']:.2f}%",
        f"- Arm-2 vehicle lift over Arm 1: {gates['arm2_vehicle_lift_pp']:.2f} pp "
        f"(95% CI {gates['arm2_vehicle_ci95_lower_pp']:.2f}, {gates['arm2_vehicle_ci95_upper_pp']:.2f})",
        f"- Arm-3 vehicle lift over Arm 1: {gates['arm3_vehicle_lift_pp']:.2f} pp "
        f"(95% CI {gates['arm3_vehicle_ci95_lower_pp']:.2f}, {gates['arm3_vehicle_ci95_upper_pp']:.2f})",
        f"- Arm-3 pedestrian coverage: {gates['arm3_pedestrian_coverage_pct']:.2f}%",
        f"- Arm-2 pedestrian lift over Arm 1: {gates['arm2_pedestrian_lift_pp']:.2f} pp "
        f"(95% CI {gates['arm2_pedestrian_ci95_lower_pp']:.2f}, {gates['arm2_pedestrian_ci95_upper_pp']:.2f})",
        f"- Arm-3 pedestrian lift over Arm 1: {gates['arm3_pedestrian_lift_pp']:.2f} pp "
        f"(95% CI {gates['arm3_pedestrian_ci95_lower_pp']:.2f}, {gates['arm3_pedestrian_ci95_upper_pp']:.2f})",
        f"- Arm-3 fast in-scope dwell: {gates['arm3_fast_dwell_s']:.2f} s",
        f"- Top-80 saturation observed in matched scenes: **{'YES' if gates['top80_saturation_observed'] else 'NO'}**",
        "",
        "## Per-arm target coverage and validity",
        "",
        summary[
            [
                "target_class", "arm", "eligible_target_rows", "target_object_row_coverage_pct",
                "target_frame_coverage_pct", "radar_projected_points_p50",
                "decode_pre_topk_max", "decode_post_topk_nms_max",
                "decode_topk_saturated_frames",
                "camera_wait_median_ms", "camera_wait_p95_ms", "result_received_pct",
                "fast_in_scope_dwell_s", "run_valid", "run_failures",
            ]
        ].to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Matched-trajectory checks",
        "",
        pairing.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Decoder saturation interpretation",
        "",
        str(gates["saturation_interpretation"]),
        "",
        "## Decision",
        "",
    ]
    if status == "PASS_GATE_1_2":
        lines.append(
            "The corrected detector recipe and controlled fast-in-view realization passed. "
            "The gated chain may proceed to one versioned corrected corpus collection."
        )
    else:
        failures = ", ".join(str(value) for value in gates["failures"])
        lines.append(
            f"Stop the overnight chain. Failed checks: {failures}. Do not start the corrected corpus."
        )
    lines.extend(
        [
            "",
            "Detailed range results are in `coverage_by_range.csv`; exact inputs and artifact hashes "
            "are in `gate_manifest.json`.",
        ]
    )
    (output_dir / "DETECTION_AB_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(batch_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Tuple[Path, Dict[str, object]]:
    batch_dir = batch_dir.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    thresholds = config["detection_ab_gate"]
    timing = config["timing_gate"]
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["runs"]
    expected = {(class_name, arm) for class_name in ("vehicle", "pedestrian") for arm in ARMS}
    observed = {(str(row.get("target_class")), str(row.get("ab_arm"))) for row in records}
    if observed != expected:
        raise RuntimeError(f"unexpected A/B cells: expected {sorted(expected)}, found {sorted(observed)}")

    summaries = []
    trajectories: Dict[Tuple[str, str], pd.DataFrame] = {}
    marked_targets: Dict[Tuple[str, str], pd.DataFrame] = {}
    range_rows = []
    for record in records:
        row, trajectory, marked = _load_run(record, thresholds, timing)
        summaries.append(row)
        key = (str(record["target_class"]), str(record["ab_arm"]))
        trajectories[key] = trajectory
        marked_targets[key] = marked
        range_rows.extend(_range_rows(marked, key[0], key[1]))
    summary = pd.DataFrame(summaries).sort_values(["target_class", "arm"])
    ranges = pd.DataFrame(range_rows)

    pair_rows = []
    for target_class in ("vehicle", "pedestrian"):
        baseline = trajectories[(target_class, ARMS[0])]
        for arm in ARMS[1:]:
            comparison = _trajectory_comparison(
                baseline, trajectories[(target_class, arm)], thresholds
            )
            pair_rows.append(
                {"target_class": target_class, "baseline_arm": ARMS[0], "candidate_arm": arm, **comparison}
            )
    pairing = pd.DataFrame(pair_rows)

    bootstrap_replicates = int(thresholds.get("paired_bootstrap_replicates", 10000))
    bootstrap_block = int(thresholds.get("paired_bootstrap_block_length_frames", 5))
    bootstrap_seed = int(thresholds.get("paired_bootstrap_seed", 20260811))
    lift_rows = []
    for class_index, target_class in enumerate(("vehicle", "pedestrian")):
        baseline = marked_targets[(target_class, ARMS[0])][
            ["sequence_index", "matched"]
        ].rename(columns={"matched": "baseline_matched"})
        for arm_index, arm in enumerate(ARMS[1:], start=1):
            candidate = marked_targets[(target_class, arm)][
                ["sequence_index", "matched"]
            ].rename(columns={"matched": "candidate_matched"})
            paired = baseline.merge(candidate, on="sequence_index", how="inner")
            result = _paired_block_bootstrap_lift(
                paired["baseline_matched"].astype(bool).to_numpy(),
                paired["candidate_matched"].astype(bool).to_numpy(),
                replicates=bootstrap_replicates,
                block_length=bootstrap_block,
                seed=bootstrap_seed + 100 * class_index + arm_index,
            )
            lift_rows.append(
                {
                    "target_class": target_class,
                    "baseline_arm": ARMS[0],
                    "candidate_arm": arm,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_block_length_frames": bootstrap_block,
                    **result,
                }
            )
    lifts = pd.DataFrame(lift_rows)

    indexed = summary.set_index(["target_class", "arm"])
    vehicle1 = indexed.loc[("vehicle", ARMS[0])]
    vehicle2 = indexed.loc[("vehicle", ARMS[1])]
    vehicle3 = indexed.loc[("vehicle", ARMS[2])]
    pedestrian1 = indexed.loc[("pedestrian", ARMS[0])]
    pedestrian2 = indexed.loc[("pedestrian", ARMS[1])]
    pedestrian3 = indexed.loc[("pedestrian", ARMS[2])]
    arm1_coverage = float(vehicle1["target_object_row_coverage_pct"])
    arm3_coverage = float(vehicle3["target_object_row_coverage_pct"])
    arm3_lift = arm3_coverage - arm1_coverage
    arm3_dwell = float(vehicle3["fast_in_scope_dwell_s"])
    lift_indexed = lifts.set_index(["target_class", "candidate_arm"])
    vehicle2_lift = lift_indexed.loc[("vehicle", ARMS[1])]
    vehicle3_lift = lift_indexed.loc[("vehicle", ARMS[2])]
    pedestrian2_lift = lift_indexed.loc[("pedestrian", ARMS[1])]
    pedestrian3_lift = lift_indexed.loc[("pedestrian", ARMS[2])]
    pedestrian3_coverage = float(pedestrian3["target_object_row_coverage_pct"])

    density_checks = []
    for target_class in ("vehicle", "pedestrian"):
        p1 = float(indexed.loc[(target_class, ARMS[0]), "radar_projected_points_p50"])
        p2 = float(indexed.loc[(target_class, ARMS[1]), "radar_projected_points_p50"])
        p3 = float(indexed.loc[(target_class, ARMS[2]), "radar_projected_points_p50"])
        density_checks.extend(
            [
                (f"{target_class}_arm2_density_ratio", p1 > 0 and p2 / p1 >= float(thresholds["minimum_density_ratio_vs_5k"])),
                (f"{target_class}_arm3_density_ratio", p1 > 0 and p3 / p1 >= float(thresholds["minimum_density_ratio_vs_5k"])),
                (
                    f"{target_class}_200k_density_match",
                    max(p2, p3) > 0
                    and abs(p2 - p3) / max(p2, p3)
                    <= float(thresholds["maximum_200k_density_ratio_delta_fraction"]),
                ),
            ]
        )
    validity_checks = [
        ("all_runs_valid", bool(summary["run_valid"].all())),
        ("all_pairs_valid", bool(pairing["pair_valid"].all())),
        *density_checks,
        (
            "arm1_vehicle_is_low_baseline",
            arm1_coverage <= float(thresholds["arm1_vehicle_low_baseline_max_pct"]),
        ),
    ]
    vehicle_checks = [
        (
            "arm3_vehicle_coverage",
            arm3_coverage >= float(thresholds["arm3_vehicle_coverage_min_pct"]),
        ),
        (
            "arm3_vehicle_lift",
            arm3_lift >= float(thresholds["arm3_vehicle_lift_min_pp"]),
        ),
        (
            "arm3_vehicle_lift_ci_lower",
            float(vehicle3_lift["ci95_lower_pp"])
            > float(thresholds.get("paired_ci_lower_min_pp", 0.0)),
        ),
    ]
    pedestrian_checks = [
        (
            "arm3_pedestrian_coverage",
            pedestrian3_coverage
            >= float(thresholds["arm3_pedestrian_coverage_min_pct"]),
        ),
        (
            "arm3_pedestrian_lift_ci_lower",
            float(pedestrian3_lift["ci95_lower_pp"])
            > float(thresholds.get("paired_ci_lower_min_pp", 0.0)),
        ),
    ]
    vehicle_gate_pass = all(passed for _, passed in vehicle_checks)
    pedestrian_gate_pass = all(passed for _, passed in pedestrian_checks)
    gate1_pass = (
        all(passed for _, passed in validity_checks)
        and vehicle_gate_pass
        and pedestrian_gate_pass
    )
    gate2_checks = [
        (
            "arm3_fast_target_speed",
            float(vehicle3["target_speed_median_mps"])
            >= float(thresholds["fast_target_speed_min_mps"]),
        ),
        (
            "arm3_fast_target_dwell",
            arm3_dwell >= float(thresholds["fast_target_dwell_min_s"]),
        ),
    ]
    gate2_pass = all(passed for _, passed in gate2_checks)
    all_checks = validity_checks + vehicle_checks + pedestrian_checks + gate2_checks
    top80_saturation_observed = bool(
        (summary["decode_pre_topk_max"] >= 80).fillna(False).any()
    )
    if top80_saturation_observed:
        saturation_interpretation = (
            "At least one matched scene exceeded 80 above-threshold pre-top-k candidates; "
            "the Arm-2/Arm-3 NMS/top-k contrast is interpretable only for those saturated frames."
        )
    else:
        saturation_interpretation = (
            "No matched scene saturated top-80. Arm 2 approximately equalling Arm 3 is therefore "
            "expected; retain NMS-2/top-120 as the validated conservative default without claiming "
            "that this smoke measured an NMS benefit."
        )
    gates: Dict[str, object] = {
        "status": "PASS_GATE_1_2" if gate1_pass and gate2_pass else "FAIL_HOLD",
        "gate1_pass": gate1_pass,
        "vehicle_gate_pass": vehicle_gate_pass,
        "pedestrian_gate_pass": pedestrian_gate_pass,
        "gate2_pass": gate2_pass,
        "arm1_vehicle_coverage_pct": arm1_coverage,
        "arm3_vehicle_coverage_pct": arm3_coverage,
        "arm2_vehicle_lift_pp": float(vehicle2_lift["lift_pp"]),
        "arm2_vehicle_ci95_lower_pp": float(vehicle2_lift["ci95_lower_pp"]),
        "arm2_vehicle_ci95_upper_pp": float(vehicle2_lift["ci95_upper_pp"]),
        "arm3_vehicle_lift_pp": float(vehicle3_lift["lift_pp"]),
        "arm3_vehicle_ci95_lower_pp": float(vehicle3_lift["ci95_lower_pp"]),
        "arm3_vehicle_ci95_upper_pp": float(vehicle3_lift["ci95_upper_pp"]),
        "arm1_pedestrian_coverage_pct": float(pedestrian1["target_object_row_coverage_pct"]),
        "arm2_pedestrian_coverage_pct": float(pedestrian2["target_object_row_coverage_pct"]),
        "arm3_pedestrian_coverage_pct": pedestrian3_coverage,
        "arm2_pedestrian_lift_pp": float(pedestrian2_lift["lift_pp"]),
        "arm2_pedestrian_ci95_lower_pp": float(pedestrian2_lift["ci95_lower_pp"]),
        "arm2_pedestrian_ci95_upper_pp": float(pedestrian2_lift["ci95_upper_pp"]),
        "arm3_pedestrian_lift_pp": float(pedestrian3_lift["lift_pp"]),
        "arm3_pedestrian_ci95_lower_pp": float(pedestrian3_lift["ci95_lower_pp"]),
        "arm3_pedestrian_ci95_upper_pp": float(pedestrian3_lift["ci95_upper_pp"]),
        "arm3_fast_dwell_s": arm3_dwell,
        "top80_saturation_observed": top80_saturation_observed,
        "saturation_interpretation": saturation_interpretation,
        "checks": {name: passed for name, passed in all_checks},
        "failures": [name for name, passed in all_checks if not passed],
    }

    output_dir = batch_dir / "gate_analysis" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    pairing.to_csv(output_dir / "trajectory_pairing.csv", index=False)
    lifts.to_csv(output_dir / "paired_coverage_lifts.csv", index=False)
    ranges.to_csv(output_dir / "coverage_by_range.csv", index=False)
    _report(output_dir, summary, pairing, ranges, gates, batch_dir)
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    gate_manifest = {
        "schema": "detection_ab_gate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **gates,
        "inputs": {
            "batch_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        },
        "artifacts": artifacts,
    }
    (output_dir / "gate_manifest.json").write_text(
        json.dumps(gate_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    output_dir, gates = analyze(args.batch_dir, args.config)
    print(output_dir)
    print(gates["status"])
    if gates["status"] != "PASS_GATE_1_2":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconcile live-corpus direct coverage with offline validated object recall."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.rescore_policy_corpus_freshness import (
    _greedy_prediction_matches,
    _normalize_class,
    _sha256,
    _single_csv,
    _truthy,
)

DEFAULT_NEW_BATCH = (
    REPO_ROOT
    / "data_collection/experiments/policy_corpus_v1/20260811_002551_full"
)
DEFAULT_OLD_MANIFEST = (
    REPO_ROOT / "rl_agent/policy/experiments/pilot/20260810_205757/manifest.json"
)
DEFAULT_OFFLINE_METRICS = (
    REPO_ROOT
    / "experiments/ae_integrated_20260710/sweeps_permodel_zstd"
    / "noae__uint8__roi0.0/metrics/test_fusion_evaluation_metrics.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments/ae_integrated_20260710/noae_baseline/checkpoints"
    / "mprime_joint_noae/best.pt"
)
RANGE_EDGES_M = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
SCORE_MIN = 0.20
ASSOCIATION_GATE_M = 5.0
HEADLINE_RANGE_M = 25.0
OFFLINE_MIN_AREA_PX = 24.0


def _safe_pct(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def _find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} under {directory}, found {len(matches)}")
    return matches[0]


def _records(new_batch: Path, old_manifest: Path) -> List[Dict[str, object]]:
    old = json.loads(old_manifest.read_text(encoding="utf-8"))
    records: List[Dict[str, object]] = []
    selected_episode_ids = set()
    for item in old["selected_replays"]:
        gt_path = REPO_ROOT / str(item["ground_truth_path"])
        selected_episode_ids.add(str(item["episode_id"]))
        records.append(
            {
                "corpus": "old_policy_replays",
                "episode_id": str(item["episode_id"]),
                "scenario_family": str(item["scenario_family"]),
                "scenario_variant": str(item["run_group"]),
                "split": str(item["split"]),
                "run_dir": gt_path.parent.parent,
            }
        )
    registry = pd.read_csv(old_manifest.parent / "replay_registry.csv")
    speed_sweep_groups = {
        "speedsweep_slow",
        "speedsweep_normal",
        "speedsweep_fast",
        "speedsweep_veryfast",
        "speedsweep_s22",
        "speedsweep_s30",
    }
    for item in registry.to_dict(orient="records"):
        episode_id = str(item["episode_id"])
        run_group = str(item["run_group"])
        if episode_id in selected_episode_ids:
            continue
        if run_group in speed_sweep_groups:
            corpus = "old_validated_speed_sweeps"
        elif episode_id in {"ACC_normal", "ACC_fast", "ACC_veryfast"}:
            corpus = "old_fresh_acc_200k"
        else:
            continue
        gt_path = REPO_ROOT / str(item["ground_truth_path"])
        records.append(
            {
                "corpus": corpus,
                "episode_id": episode_id,
                "scenario_family": str(item["scenario_family"]),
                "scenario_variant": run_group,
                "split": str(item["split"]),
                "run_dir": gt_path.parent.parent,
            }
        )
    new = json.loads((new_batch / "batch_manifest.json").read_text(encoding="utf-8"))
    for item in new["runs"]:
        records.append(
            {
                "corpus": "new_policy_corpus_v1",
                "episode_id": str(item["episode_id"]),
                "scenario_family": str(item["scenario_family"]),
                "scenario_variant": str(item.get("scenario_variant", item["scenario_family"])),
                "split": str(item["split"]),
                "run_dir": Path(str(item["run_dir"])),
            }
        )
    return records


def prepare_inputs(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    gt_path = _single_csv(run_dir, "_object_ground_truth.csv")
    prediction_path = _single_csv(run_dir, "_object_predictions.csv")
    gt = pd.read_csv(gt_path)
    predictions = pd.read_csv(prediction_path)
    gt["class_name"] = gt["class_name"].map(_normalize_class)
    gt["world_x"] = pd.to_numeric(gt["origin_x"], errors="coerce")
    gt["world_y"] = pd.to_numeric(gt["origin_y"], errors="coerce")
    gt["distance_m"] = pd.to_numeric(gt["distance_m"], errors="coerce")
    predictions["class_name"] = predictions["class_name"].map(_normalize_class)
    score = pd.to_numeric(
        predictions.get("score", pd.Series(1.0, index=predictions.index)),
        errors="coerce",
    )
    predictions = predictions[score >= SCORE_MIN].copy()
    return gt, predictions, gt_path, prediction_path


def denominator_rows(
    gt: pd.DataFrame,
    denominator: str,
    camera_width: int,
    camera_height: int,
) -> pd.DataFrame:
    selected = gt[
        _truthy(gt["in_camera_frustum"])
        & (gt["distance_m"] <= HEADLINE_RANGE_M)
        & gt["world_x"].notna()
        & gt["world_y"].notna()
    ].copy()
    if denominator == "current_in_frustum_le25":
        return selected
    if denominator != "offline_visibility_proxy_le25":
        raise ValueError(f"unknown denominator: {denominator}")
    center_x = pd.to_numeric(selected["projected_x"], errors="coerce")
    center_y = pd.to_numeric(selected["projected_y"], errors="coerce")
    x1 = pd.to_numeric(selected["bbox_x1"], errors="coerce").clip(0, camera_width)
    y1 = pd.to_numeric(selected["bbox_y1"], errors="coerce").clip(0, camera_height)
    x2 = pd.to_numeric(selected["bbox_x2"], errors="coerce").clip(0, camera_width)
    y2 = pd.to_numeric(selected["bbox_y2"], errors="coerce").clip(0, camera_height)
    clipped_area = (x2 - x1).clip(lower=0) * (y2 - y1).clip(lower=0)
    return selected[
        center_x.between(0, camera_width, inclusive="left")
        & center_y.between(0, camera_height, inclusive="left")
        & (clipped_area >= OFFLINE_MIN_AREA_PX)
    ].copy()


def mark_matches(eligible: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = eligible.copy()
    if eligible.empty:
        eligible["matched"] = pd.Series(dtype=bool)
        return eligible
    matches = _greedy_prediction_matches(eligible, predictions, ASSOCIATION_GATE_M)
    matched_keys = set()
    if not matches.empty:
        matched_keys = set(
            zip(
                matches["actor_id"].astype(int),
                matches["class_name"].astype(str),
                matches["timestamp"].astype(float),
            )
        )
    eligible["matched"] = [
        (int(row.actor_id), str(row.class_name), float(row.carla_timestamp)) in matched_keys
        for row in eligible.itertuples()
    ]
    return eligible


def summarize_rows(
    marked: pd.DataFrame,
    metadata: Mapping[str, object],
    denominator: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_name in ("pedestrian", "vehicle"):
        group = marked[marked["class_name"] == class_name]
        matched_frames = group.loc[group["matched"], "frame_id"].nunique()
        rows.append(
            {
                **{key: metadata[key] for key in (
                    "corpus", "episode_id", "scenario_family", "scenario_variant", "split"
                )},
                "denominator": denominator,
                "class_name": class_name,
                "eligible_gt_rows": int(len(group)),
                "matched_rows": int(group["matched"].sum()),
                "eligible_frames": int(group["frame_id"].nunique()),
                "matched_frames": int(matched_frames),
                "direct_object_row_coverage_pct": _safe_pct(group["matched"].sum(), len(group)),
                "direct_frame_coverage_pct": _safe_pct(
                    matched_frames, group["frame_id"].nunique()
                ),
            }
        )
    return rows


def summarize_ranges(
    marked: pd.DataFrame,
    metadata: Mapping[str, object],
    denominator: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    labels = [
        f"{RANGE_EDGES_M[index]:g}-{RANGE_EDGES_M[index + 1]:g}"
        for index in range(len(RANGE_EDGES_M) - 1)
    ]
    marked = marked.copy()
    marked["range_bin_m"] = pd.cut(
        marked["distance_m"],
        bins=RANGE_EDGES_M,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    for class_name in ("pedestrian", "vehicle"):
        class_group = marked[marked["class_name"] == class_name]
        for label in labels:
            group = class_group[class_group["range_bin_m"] == label]
            matched_frames = group.loc[group["matched"], "frame_id"].nunique()
            rows.append(
                {
                    "corpus": metadata["corpus"],
                    "episode_id": metadata["episode_id"],
                    "denominator": denominator,
                    "class_name": class_name,
                    "range_bin_m": label,
                    "eligible_gt_rows": int(len(group)),
                    "matched_rows": int(group["matched"].sum()),
                    "eligible_frames": int(group["frame_id"].nunique()),
                    "matched_frames": int(matched_frames),
                }
            )
    return rows


def pooled_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    keys = ["corpus", "denominator", "class_name"]
    pooled = per_run.groupby(keys, as_index=False)[
        ["eligible_gt_rows", "matched_rows", "eligible_frames", "matched_frames"]
    ].sum()
    pooled["direct_object_row_coverage_pct"] = 100.0 * pooled["matched_rows"] / pooled[
        "eligible_gt_rows"
    ].replace(0, np.nan)
    pooled["direct_frame_coverage_pct"] = 100.0 * pooled["matched_frames"] / pooled[
        "eligible_frames"
    ].replace(0, np.nan)
    return pooled


def pooled_ranges(per_run_range: pd.DataFrame) -> pd.DataFrame:
    keys = ["corpus", "denominator", "class_name", "range_bin_m"]
    pooled = per_run_range.groupby(keys, as_index=False, observed=False)[
        ["eligible_gt_rows", "matched_rows", "eligible_frames", "matched_frames"]
    ].sum()
    pooled["direct_object_row_coverage_pct"] = 100.0 * pooled["matched_rows"] / pooled[
        "eligible_gt_rows"
    ].replace(0, np.nan)
    pooled["direct_frame_coverage_pct"] = 100.0 * pooled["matched_frames"] / pooled[
        "eligible_frames"
    ].replace(0, np.nan)
    return pooled


def timeout_audit(
    run_dir: Path,
    metadata: Mapping[str, object],
    current_eligible: pd.DataFrame,
) -> List[Dict[str, object]]:
    metrics_path = _find_single(run_dir / "streams", "*_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    received = _truthy(metrics["result_received"])
    object_count = pd.to_numeric(metrics["object_count"], errors="coerce").fillna(0)
    timeout_frames = set(metrics.loc[~received, "frame_id"].astype(int))
    empty_success_frames = set(metrics.loc[received & (object_count == 0), "frame_id"].astype(int))
    rows: List[Dict[str, object]] = []
    for class_name in ("all", "pedestrian", "vehicle"):
        group = (
            current_eligible
            if class_name == "all"
            else current_eligible[current_eligible["class_name"] == class_name]
        )
        timeout_gt = group[group["frame_id"].astype(int).isin(timeout_frames)]
        empty_gt = group[group["frame_id"].astype(int).isin(empty_success_frames)]
        rows.append(
            {
                "corpus": metadata["corpus"],
                "episode_id": metadata["episode_id"],
                "class_name": class_name,
                "processed_frames": int(len(metrics)),
                "result_timeout_frames": int((~received).sum()),
                "result_received_pct": _safe_pct(received.sum(), len(metrics)),
                "successful_empty_result_frames": int((received & (object_count == 0)).sum()),
                "eligible_gt_rows": int(len(group)),
                "eligible_gt_rows_on_timeout_frames": int(len(timeout_gt)),
                "eligible_gt_frames_on_timeout_frames": int(timeout_gt["frame_id"].nunique()),
                "eligible_gt_rows_on_successful_empty_frames": int(len(empty_gt)),
                "eligible_gt_frames_on_successful_empty_frames": int(
                    empty_gt["frame_id"].nunique()
                ),
                "metrics_sha256": _sha256(metrics_path),
            }
        )
    return rows


def config_audit(
    run_dir: Path,
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    config_path = _find_single(run_dir / "manifests", "*_resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics_path = _find_single(run_dir / "streams", "*_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    received = _truthy(metrics["result_received"])
    return {
        "corpus": metadata["corpus"],
        "episode_id": metadata["episode_id"],
        "scenario_family": metadata["scenario_family"],
        "checkpoint": str(config.get("fusion_checkpoint", "")),
        "camera_width": config.get("camera_width", 854),
        "camera_height": config.get("camera_height", 480),
        "radar_points_per_second": config.get("radar_points_per_second"),
        "radar_rasterizer": config.get("radar_rasterizer") or "legacy/default",
        "radar_temporal_window_frames": config.get("radar_temporal_window_frames"),
        "quantization_mode": config.get("quantization_mode"),
        "entropy_coder": config.get("entropy_coder"),
        "live_decode_score_threshold": config.get("object_score_threshold"),
        "analysis_score_threshold": SCORE_MIN,
        "object_nms_radius_px": config.get("object_nms_radius_px"),
        "topk_objects": config.get("topk_objects"),
        "front_device": config.get("front_device"),
        "back_device": config.get("back_device"),
        "result_timeout_s": config.get("result_timeout"),
        "radar_projected_points_p50": float(
            pd.to_numeric(metrics["radar_projected_points"], errors="coerce").median()
        ),
        "result_received_pct": _safe_pct(received.sum(), len(metrics)),
        "resolved_config_sha256": _sha256(config_path),
    }


def aggregate_timeout(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "processed_frames",
        "result_timeout_frames",
        "successful_empty_result_frames",
        "eligible_gt_rows",
        "eligible_gt_rows_on_timeout_frames",
        "eligible_gt_frames_on_timeout_frames",
        "eligible_gt_rows_on_successful_empty_frames",
        "eligible_gt_frames_on_successful_empty_frames",
    ]
    pooled = frame.groupby(["corpus", "class_name"], as_index=False)[metric_columns].sum()
    pooled["result_received_pct"] = 100.0 * (
        pooled["processed_frames"] - pooled["result_timeout_frames"]
    ) / pooled["processed_frames"].replace(0, np.nan)
    return pooled


def write_manifest(output_dir: Path, inputs: Mapping[str, Path]) -> None:
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": "detection_reconciliation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANALYSIS_COMPLETE",
        "constants": {
            "headline_range_m": HEADLINE_RANGE_M,
            "prediction_score_min": SCORE_MIN,
            "association_gate_m": ASSOCIATION_GATE_M,
            "range_edges_m": list(RANGE_EDGES_M),
            "offline_visibility_proxy_min_area_px": OFFLINE_MIN_AREA_PX,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()
        },
        "artifacts": artifacts,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def run(
    new_batch: Path = DEFAULT_NEW_BATCH,
    old_manifest: Path = DEFAULT_OLD_MANIFEST,
    offline_metrics_path: Path = DEFAULT_OFFLINE_METRICS,
    output_dir: Path | None = None,
) -> Path:
    new_batch = new_batch.resolve()
    old_manifest = old_manifest.resolve()
    offline_metrics_path = offline_metrics_path.resolve()
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = (
            REPO_ROOT / "data_collection/experiments/detection_reconciliation" / timestamp
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    per_run_rows: List[Dict[str, object]] = []
    per_range_rows: List[Dict[str, object]] = []
    timeout_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    input_hash_rows: List[Dict[str, object]] = []
    for metadata in _records(new_batch, old_manifest):
        run_dir = Path(str(metadata["run_dir"])).resolve()
        gt, predictions, gt_path, prediction_path = prepare_inputs(run_dir)
        config = config_audit(run_dir, metadata)
        config_rows.append(config)
        input_hash_rows.append(
            {
                "corpus": metadata["corpus"],
                "episode_id": metadata["episode_id"],
                "ground_truth_path": str(gt_path),
                "ground_truth_sha256": _sha256(gt_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
            }
        )
        current_eligible: pd.DataFrame | None = None
        for denominator in (
            "current_in_frustum_le25",
            "offline_visibility_proxy_le25",
        ):
            eligible = denominator_rows(
                gt,
                denominator,
                int(config.get("camera_width", 854) or 854),
                int(config.get("camera_height", 480) or 480),
            )
            marked = mark_matches(eligible, predictions)
            if denominator == "current_in_frustum_le25":
                current_eligible = eligible
            per_run_rows.extend(summarize_rows(marked, metadata, denominator))
            per_range_rows.extend(summarize_ranges(marked, metadata, denominator))
        if current_eligible is None:
            raise AssertionError("current denominator was not evaluated")
        timeout_rows.extend(timeout_audit(run_dir, metadata, current_eligible))

    per_run = pd.DataFrame(per_run_rows)
    per_range = pd.DataFrame(per_range_rows)
    timeouts = pd.DataFrame(timeout_rows)
    configs = pd.DataFrame(config_rows)
    hashes = pd.DataFrame(input_hash_rows)
    pooled = pooled_summary(per_run)
    range_pooled = pooled_ranges(per_range)
    timeout_pooled = aggregate_timeout(timeouts)
    offline = json.loads(offline_metrics_path.read_text(encoding="utf-8"))
    offline_summary = {
        "profile": "noae__uint8__roi0.0",
        "checkpoint": str(offline["checkpoint"]),
        "checkpoint_sha256": _sha256(DEFAULT_CHECKPOINT),
        "samples": int(offline["samples"]),
        "object_recall": float(offline["learned_object_recall"]),
        "pedestrian_recall": float(offline["learned_person_object_recall"]),
        "vehicle_recall": float(offline["learned_vehicle_object_recall"]),
        "score_threshold": SCORE_MIN,
        "nms_radius_px": 2,
        "topk_objects": 120,
        "match_gate_m": ASSOCIATION_GATE_M,
        "max_gt_distance_m": 40.0,
        "gt_min_area_px": OFFLINE_MIN_AREA_PX,
        "dataset": "fusion_training_data/moving_ego_pps200000_merged_8loops_stride2",
        "radar_points_per_second": 200000,
    }

    per_run.to_csv(output_dir / "coverage_per_run.csv", index=False)
    pooled.to_csv(output_dir / "coverage_pooled.csv", index=False)
    per_range.to_csv(output_dir / "coverage_by_range_per_run.csv", index=False)
    range_pooled.to_csv(output_dir / "coverage_by_range_pooled.csv", index=False)
    timeouts.to_csv(output_dir / "timeout_audit_per_run.csv", index=False)
    timeout_pooled.to_csv(output_dir / "timeout_audit_pooled.csv", index=False)
    configs.to_csv(output_dir / "live_config_audit.csv", index=False)
    hashes.to_csv(output_dir / "input_hashes.csv", index=False)
    (output_dir / "offline_reference.json").write_text(
        json.dumps(offline_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_manifest(
        output_dir,
        {
            "new_batch_manifest": new_batch / "batch_manifest.json",
            "old_policy_manifest": old_manifest,
            "offline_metrics": offline_metrics_path,
            "checkpoint": DEFAULT_CHECKPOINT,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-batch", type=Path, default=DEFAULT_NEW_BATCH)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--offline-metrics", type=Path, default=DEFAULT_OFFLINE_METRICS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(run(args.new_batch, args.old_manifest, args.offline_metrics, args.output_dir))


if __name__ == "__main__":
    main()

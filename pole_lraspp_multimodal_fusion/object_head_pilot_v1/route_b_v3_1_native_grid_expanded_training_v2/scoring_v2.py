#!/usr/bin/env python3
"""Registered native-grid validation scoring and staged decision gates."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"


def _load_native_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_native_grid_registered_evaluator_v1", NATIVE_PACKAGE / "evaluate_v1.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load the registered native-grid evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_NATIVE: Any | None = None


def native_evaluator() -> Any:
    # Training imports the regular fusion package, while the frozen evaluator needs
    # the repository namespace package. Load it only in the dedicated scoring
    # subprocess, never in the training process.
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = _load_native_evaluator()
    return _NATIVE


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def score_primary(
    experiment: Path, prediction_root: Path, checkpoint: Path,
    checkpoint_sha256: str, epoch: int,
) -> dict[str, Any]:
    native = native_evaluator()
    manifest = native.read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    if len(frame_ids) != 3345 or len(set(frame_ids)) != 3345:
        raise RuntimeError("registered validation frame count/uniqueness drift")
    inference = json.loads((prediction_root / "inference_manifest.json").read_text())
    detections = prediction_root / "detections.csv"
    if native.sha256(detections) != inference["detections_sha256"]:
        raise RuntimeError(f"detection hash drift at epoch {epoch}")
    if inference["checkpoint_sha256"] != checkpoint_sha256:
        raise RuntimeError(f"checkpoint provenance drift at epoch {epoch}")
    if inference["inference_pass_count"] != 1 or inference["native_object_grid"] != [108, 192]:
        raise RuntimeError(f"inference contract drift at epoch {epoch}")
    predictions, missing = native.load_predictions(detections)
    if missing:
        raise RuntimeError(f"missing/nonfinite prediction fields at epoch {epoch}: {missing[:5]}")
    gt, _states = native.load_gt(experiment, "v010")
    segmentation = native.score_segmentation(
        experiment, "v010", frame_ids, prediction_root,
        prediction_root / "segmentation_manifest.csv",
    )
    scored = {
        f"{threshold:.2f}": native.score_arm(
            experiment=experiment, contract="v010", frame_ids=frame_ids,
            predictions=predictions, gt=gt, threshold=threshold,
            ignore_cache={},
        )
        for threshold in (0.20, 0.02)
    }
    metrics = native.flatten(scored["0.20"], scored["0.02"], segmentation)
    taxonomy = native.run_taxonomy(experiment, frame_ids, predictions, gt, {})
    duplicate_fp = taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"]
    heatmap_miss = taxonomy["person_fn_at_0_02"]["counts"]["HEATMAP_CENTER_MISS"]
    return {
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_root": str(prediction_root),
        "prediction_set_sha256": inference["prediction_set_sha256"],
        "detections_sha256": inference["detections_sha256"],
        "inference_wall_seconds": inference["wall_seconds"],
        "inference_peak_allocated_mib": inference["peak_allocated_mib"],
        "inference_peak_reserved_mib": inference["peak_reserved_mib"],
        "metrics": metrics,
        "primary_v010": scored,
        "segmentation_v010": segmentation,
        "taxonomy_v010": taxonomy,
        "vehicle_duplicate_fp": duplicate_fp,
        "person_heatmap_center_miss": heatmap_miss,
        "all_metrics_finite": all_finite(metrics),
    }


def score_sensitivity(experiment: Path, prediction_root: Path) -> dict[str, Any]:
    native = native_evaluator()
    manifest = native.read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    predictions, missing = native.load_predictions(prediction_root / "detections.csv")
    if missing:
        raise RuntimeError(f"selected sensitivity predictions invalid: {missing[:5]}")
    gt, _states = native.load_gt(experiment, "v025")
    scored = {
        f"{threshold:.2f}": native.score_arm(
            experiment=experiment, contract="v025", frame_ids=frame_ids,
            predictions=predictions, gt=gt, threshold=threshold,
            ignore_cache={},
        )
        for threshold in (0.20, 0.02)
    }
    at_020, at_002 = scored["0.20"], scored["0.02"]
    flat = {
        "vehicle_precision": at_020["classes"]["vehicle"]["precision"],
        "vehicle_recall": at_020["classes"]["vehicle"]["recall"],
        "vehicle_f1": at_020["classes"]["vehicle"]["f1"],
        "vehicle_xy_mae_m": at_020["classes"]["vehicle"]["xy_mae_m"],
        "vehicle_recall_002": at_002["classes"]["vehicle"]["recall"],
        "person_precision": at_020["classes"]["person"]["precision"],
        "person_recall": at_020["classes"]["person"]["recall"],
        "person_f1": at_020["classes"]["person"]["f1"],
        "person_xy_mae_m": at_020["classes"]["person"]["xy_mae_m"],
        "person_recall_002": at_002["classes"]["person"]["recall"],
    }
    return {"flat": flat, "thresholds": scored, "all_metrics_finite": all_finite(flat)}


def epoch10_gate(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline"]
    return {
        "all_metrics_finite": bool(record["all_metrics_finite"]),
        "vehicle_f1_ge_baseline_minus_0_02": metric["vehicle_f1"] >= baseline["vehicle_f1"] - 0.02,
        "person_f1_ge_baseline_minus_0_02": metric["person_f1"] >= baseline["person_f1"] - 0.02,
        "vehicle_xy_le_baseline_plus_0_10m": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + 0.10,
        "person_xy_le_baseline_plus_0_10m": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + 0.10,
        "vehicle_iou_ge_baseline_minus_0_01": metric["vehicle_iou"] >= baseline["vehicle_iou"] - 0.01,
        "person_iou_ge_baseline_minus_0_01": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - 0.01,
        "foreground_iou_ge_baseline_minus_0_01": metric["foreground_miou"] >= baseline["foreground_miou"] - 0.01,
        "duplicate_fp_le_baseline_plus_20pct": record["vehicle_duplicate_fp"] <= 1.2 * baseline["vehicle_duplicate_fp"],
        "heatmap_miss_le_baseline_plus_20pct": record["person_heatmap_center_miss"] <= 1.2 * baseline["person_heatmap_center_miss"],
    }


def epoch20_gate(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline"]
    baseline_mean = (baseline["vehicle_f1"] + baseline["person_f1"]) / 2.0
    path_a = (
        metric["mean_class_f1"] >= baseline_mean + 0.02
        and metric["person_f1"] >= baseline["person_f1"]
    )
    path_b = (
        metric["person_f1"] >= baseline["person_f1"] + 0.03
        and metric["vehicle_f1"] >= baseline["vehicle_f1"] - 0.01
    )
    return {
        "all_metrics_finite": bool(record["all_metrics_finite"]),
        "vehicle_f1_nonregression": metric["vehicle_f1"] >= baseline["vehicle_f1"] - 0.02,
        "person_f1_nonregression": metric["person_f1"] >= baseline["person_f1"] - 0.02,
        "duplicate_fp_no_catastrophic_increase": record["vehicle_duplicate_fp"] <= 1.2 * baseline["vehicle_duplicate_fp"],
        "heatmap_miss_no_catastrophic_increase": record["person_heatmap_center_miss"] <= 1.2 * baseline["person_heatmap_center_miss"],
        "progress_path_a_or_b": path_a or path_b,
        "person_low_score_recall_or_precision_progress": (
            metric["person_recall_002"] >= baseline["person_recall_002"] + 0.02
            or metric["person_precision"] >= baseline["person_precision"] + 0.03
        ),
        "vehicle_xy_le_baseline_plus_0_05m": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + 0.05,
        "person_xy_le_baseline_plus_0_05m": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + 0.05,
        "vehicle_iou_ge_baseline_minus_0_01": metric["vehicle_iou"] >= baseline["vehicle_iou"] - 0.01,
        "person_iou_ge_baseline_minus_0_01": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - 0.01,
        "foreground_iou_ge_baseline_minus_0_01": metric["foreground_miou"] >= baseline["foreground_miou"] - 0.01,
    }


def primary_eligibility(record: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline"]
    return {
        "all_metrics_finite": bool(record["all_metrics_finite"]),
        "vehicle_f1_ge_baseline_minus_0_01": metric["vehicle_f1"] >= baseline["vehicle_f1"] - 0.01,
        "person_f1_ge_baseline_minus_0_01": metric["person_f1"] >= baseline["person_f1"] - 0.01,
        "vehicle_recall_ge_baseline_minus_0_01": metric["vehicle_recall"] >= baseline["vehicle_recall"] - 0.01,
        "person_recall_ge_baseline_minus_0_01": metric["person_recall"] >= baseline["person_recall"] - 0.01,
        "vehicle_xy_le_baseline_plus_0_05m": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + 0.05,
        "person_xy_le_baseline_plus_0_05m": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + 0.05,
        "vehicle_iou_ge_baseline_minus_0_01": metric["vehicle_iou"] >= baseline["vehicle_iou"] - 0.01,
        "person_iou_ge_baseline_minus_0_01": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - 0.01,
        "foreground_iou_ge_baseline_minus_0_01": metric["foreground_miou"] >= baseline["foreground_miou"] - 0.01,
    }


def sensitivity_no_reversal(sensitivity: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    metric, baseline = sensitivity["flat"], config["baseline_v025"]
    return {
        "all_metrics_finite": bool(sensitivity["all_metrics_finite"]),
        "vehicle_f1_ge_baseline_minus_0_01": metric["vehicle_f1"] >= baseline["vehicle_f1"] - 0.01,
        "person_f1_ge_baseline_minus_0_01": metric["person_f1"] >= baseline["person_f1"] - 0.01,
        "vehicle_xy_no_worse_than_baseline": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"],
        "person_xy_no_worse_than_baseline": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"],
    }


def rank_key(record: dict[str, Any]) -> tuple[float, float, float, int, int]:
    metric = record["metrics"]
    return (
        -metric["mean_class_f1"],
        -metric["minimum_class_recall"],
        metric["mean_xy_mae_m"],
        int(record["vehicle_duplicate_fp"]),
        int(record["epoch"]),
    )


def service_targets(record: dict[str, Any]) -> dict[str, bool]:
    metric = record["metrics"]
    return {
        "vehicle_precision_ge_0_80": metric["vehicle_precision"] >= 0.80,
        "vehicle_recall_ge_0_85": metric["vehicle_recall"] >= 0.85,
        "person_precision_ge_0_80": metric["person_precision"] >= 0.80,
        "person_recall_ge_0_80": metric["person_recall"] >= 0.80,
        "vehicle_xy_mae_le_1_0m": metric["vehicle_xy_mae_m"] <= 1.0,
        "person_xy_mae_le_1_2m": metric["person_xy_mae_m"] <= 1.2,
        "vehicle_iou_ge_0_85": metric["vehicle_iou"] >= 0.85,
        "person_box_mask_iou_ge_0_50": metric["person_box_mask_iou"] >= 0.50,
        "foreground_miou_ge_0_675": metric["foreground_miou"] >= 0.675,
    }


def material_gain(
    record: dict[str, Any], config: dict[str, Any],
    primary_guards: dict[str, bool], sensitivity_guards: dict[str, bool],
) -> dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline"]
    baseline_mean = (baseline["vehicle_f1"] + baseline["person_f1"]) / 2.0
    return {
        "mean_class_f1_ge_baseline_plus_0_05": metric["mean_class_f1"] >= baseline_mean + 0.05,
        "vehicle_f1_ge_baseline_plus_0_02": metric["vehicle_f1"] >= baseline["vehicle_f1"] + 0.02,
        "person_f1_ge_baseline_plus_0_05": metric["person_f1"] >= baseline["person_f1"] + 0.05,
        "person_recall_002_ge_baseline_plus_0_05": metric["person_recall_002"] >= baseline["person_recall_002"] + 0.05,
        "vehicle_duplicate_fp_le_979": record["vehicle_duplicate_fp"] <= 979,
        "person_heatmap_center_miss_le_685": record["person_heatmap_center_miss"] <= 685,
        "no_primary_localization_or_segmentation_regression": all(primary_guards.values()),
        "no_v025_sensitivity_reversal": all(sensitivity_guards.values()),
    }

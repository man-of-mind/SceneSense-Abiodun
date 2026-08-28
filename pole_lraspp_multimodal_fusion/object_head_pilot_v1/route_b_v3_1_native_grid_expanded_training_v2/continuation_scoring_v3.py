#!/usr/bin/env python3
"""Continuation scoring with diagnostic dimension/yaw errors on frozen matches."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from scoring_v2 import score_primary, score_sensitivity

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
CLEAN_SCORER = PACKAGE_ROOT.parent / "route_b_v3_1_clean_base_v1/score_contract_v1.py"


def _load_clean_scorer() -> Any:
    spec = importlib.util.spec_from_file_location("continuation_dimension_yaw_scorer_v3", CLEAN_SCORER)
    if spec is None or spec.loader is None:
        raise ImportError("unable to load frozen dimension/yaw scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extended_detection(
    experiment: Path, prediction_root: Path, contract: str, threshold: float = 0.20
) -> dict[str, Any]:
    scorer = _load_clean_scorer()
    manifest = scorer.read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    if len(frame_ids) != 3345 or len(set(frame_ids)) != 3345:
        raise RuntimeError("dimension/yaw validation denominator drift")
    inference = json.loads((prediction_root / "inference_manifest.json").read_text())
    result = scorer.score_detection(
        experiment=experiment,
        contract=contract,
        frame_ids=frame_ids,
        predictions=scorer.load_predictions(prediction_root / "detections.csv"),
        gt=scorer.load_gt(experiment, contract),
        threshold=threshold,
        prediction_size=tuple(int(value) for value in inference["input_size"]),
    )
    return result


def _reconcile_core(base: dict[str, Any], extended: dict[str, Any]) -> None:
    for class_name in ("vehicle", "person"):
        old = base["classes"][class_name]
        new = extended["classes"][class_name]
        for key in ("eligible_gt", "tp", "fp", "fn", "ignored_predictions"):
            if int(old[key]) != int(new[key]):
                raise RuntimeError(f"dimension/yaw scorer denominator drift: {class_name}/{key}")
        for key in ("precision", "recall", "f1", "xy_mae_m"):
            if not math.isclose(float(old[key]), float(new[key]), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"dimension/yaw scorer metric drift: {class_name}/{key}")


def primary_with_errors(
    experiment: Path, prediction_root: Path, checkpoint: Path,
    checkpoint_sha256: str, epoch: int,
) -> dict[str, Any]:
    result = score_primary(experiment, prediction_root, checkpoint, checkpoint_sha256, epoch)
    extended = _extended_detection(experiment, prediction_root, "v010")
    _reconcile_core(result["primary_v010"]["0.20"], extended)
    errors: dict[str, Any] = {}
    for class_name in ("vehicle", "person"):
        values = extended["classes"][class_name]
        for key in ("dimension_mae_m", "yaw_mae_deg"):
            result["metrics"][f"{class_name}_{key}"] = values[key]
            result["primary_v010"]["0.20"]["classes"][class_name][key] = values[key]
            errors[f"{class_name}_{key}"] = values[key]
    result["dimension_yaw_v010"] = errors
    result["dimension_yaw_core_reconciled"] = True
    return result


def sensitivity_with_errors(experiment: Path, prediction_root: Path) -> dict[str, Any]:
    result = score_sensitivity(experiment, prediction_root)
    extended = _extended_detection(experiment, prediction_root, "v025")
    _reconcile_core(result["thresholds"]["0.20"], extended)
    for class_name in ("vehicle", "person"):
        values = extended["classes"][class_name]
        result["flat"][f"{class_name}_dimension_mae_m"] = values["dimension_mae_m"]
        result["flat"][f"{class_name}_yaw_mae_deg"] = values["yaw_mae_deg"]
        result["thresholds"]["0.20"]["classes"][class_name].update({
            "dimension_mae_m": values["dimension_mae_m"],
            "yaw_mae_deg": values["yaw_mae_deg"],
        })
    result["denominators"] = {
        class_name: int(extended["classes"][class_name]["eligible_gt"])
        for class_name in ("vehicle", "person")
    }
    result["dimension_yaw_core_reconciled"] = True
    return result


def baseline_primary_with_errors(
    experiment: Path, amended_baseline: Path, prediction_root: Path,
) -> dict[str, Any]:
    amended = json.loads(amended_baseline.read_text())
    metrics = dict(amended["amended"]["v010"]["flat"])
    extended = _extended_detection(experiment, prediction_root, "v010")
    _reconcile_core(amended["amended"]["v010"]["thresholds"]["0.20"], extended)
    for class_name in ("vehicle", "person"):
        metrics[f"{class_name}_dimension_mae_m"] = extended["classes"][class_name]["dimension_mae_m"]
        metrics[f"{class_name}_yaw_mae_deg"] = extended["classes"][class_name]["yaw_mae_deg"]
    taxonomy = amended["amended_taxonomy"]
    return {
        "label": "amended_baseline",
        "selection_order": 0,
        "epoch": None,
        "checkpoint": amended["checkpoint"],
        "checkpoint_sha256": amended["checkpoint_sha256"],
        "prediction_root": str(prediction_root),
        "metrics": metrics,
        "primary_v010": amended["amended"]["v010"]["thresholds"],
        "segmentation_v010": amended["amended"]["v010"]["segmentation"],
        "taxonomy_v010": taxonomy,
        "vehicle_duplicate_fp": taxonomy["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"],
        "person_heatmap_center_miss": taxonomy["person_fn_at_0_02"]["counts"]["HEATMAP_CENTER_MISS"],
        "all_metrics_finite": True,
        "checkpoint_state_integrity": True,
        "dimension_yaw_core_reconciled": True,
    }

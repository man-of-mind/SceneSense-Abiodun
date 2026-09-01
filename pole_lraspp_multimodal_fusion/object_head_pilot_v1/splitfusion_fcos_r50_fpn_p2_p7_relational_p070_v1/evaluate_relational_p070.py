from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.runtime import (
    FROZEN_CHECKPOINT,
    sha256,
)

from .contract import (
    CANONICAL_PERSON_THRESHOLD,
    DEPLOYMENT_LOGIT_BIAS,
    FROZEN_CHECKPOINT_SHA256,
    HISTORICAL_STATUS,
    ROOT,
    SELECTOR_CHECKPOINT_SHA256,
    load_locked_config,
)

INFERENCE_SENTINEL = "RELATIONAL_P070_INFERENCE_COMPLETE\n"
INFERENCE_SCHEMA = "splitfusion_fcos_relational_p070_inference_v1"
VALIDATION_FRAMES = 3_345
P070_PERSON_TARGET = 0.70
ACCEPTED_SERVICE_CANDIDATE = {
    "person_precision": 0.730673,
    "person_recall": 0.600465,
}


def validate_prediction_directory(prediction_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Validate only a complete, immutable relational-p070 prediction contract."""
    locked = load_locked_config()
    prediction = Path(prediction_dir).resolve(strict=True)
    if not prediction.is_dir():
        raise RuntimeError("relational-p070 prediction path is not a directory")
    complete = prediction / "INFERENCE_COMPLETE"
    if not complete.is_file() or complete.read_text(encoding="utf-8") != INFERENCE_SENTINEL:
        raise RuntimeError("relational-p070 prediction directory is incomplete")
    manifest_path = prediction / "inference_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("relational-p070 inference manifest is missing")
    inference = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(inference, dict):
        raise RuntimeError("relational-p070 inference manifest is not an object")
    calibration = locked["calibration"]
    runtime = locked["runtime"]
    objective = locked["objective"]
    if (inference.get("schema") != INFERENCE_SCHEMA
            or inference.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or inference.get("selector_checkpoint_sha256") != SELECTOR_CHECKPOINT_SHA256
            or inference.get("historical_selector_status_unchanged")
            != objective["historical_0_80_status_preserved"]
            or inference.get("historical_selector_status_unchanged") != HISTORICAL_STATUS
            or inference.get("revised_objective") != {
                "precision": objective["precision_minimum"],
                "recall": objective["recall_minimum"],
            }
            or float(inference.get("deployment_bias", 0.0))
            != float(calibration["deployment_logit_bias"])
            or float(inference.get("deployment_threshold", -1.0))
            != float(calibration["deployment_threshold"])
            or float(inference.get("deployment_threshold", -1.0))
            != CANONICAL_PERSON_THRESHOLD
            or int(inference.get("validation_frames", -1)) != VALIDATION_FRAMES
            or int(inference.get("inference_pass_count", -1)) != 1
            or inference.get("candidate_creation") is not runtime["candidate_creation"]
            or inference.get("nms_rerun") is not runtime["nms_rerun"]
            or inference.get("candidate_order") != "original_post_nms"
            or inference.get("prediction_index") != "original_post_nms"
            or inference.get("consolidation_is_feature_only")
            is not runtime["consolidation_is_feature_only"]
            or inference.get("vehicle_behavior") != runtime["vehicle_policy"]
            or inference.get("geometry_changed") is not runtime["geometry_changed"]
            or inference.get("segmentation_changed") is not runtime["segmentation_changed"]):
        raise RuntimeError("relational-p070 inference contract drift")

    detections = prediction / "detections.csv"
    segmentation_manifest = prediction / "segmentation_manifest.csv"
    if not detections.is_file() or not segmentation_manifest.is_file():
        raise RuntimeError("relational-p070 prediction hashes cannot be verified")
    detection_hash = sha256(detections)
    segmentation_hash = sha256(segmentation_manifest)
    prediction_set_hash = hashlib.sha256((detection_hash + segmentation_hash).encode()).hexdigest()
    if (inference.get("detections_sha256") != detection_hash
            or inference.get("segmentation_manifest_sha256") != segmentation_hash
            or inference.get("prediction_set_sha256") != prediction_set_hash):
        raise RuntimeError("relational-p070 prediction artifact SHA-256 mismatch")
    return prediction, inference


def p070_service_decision(canonical_service: Mapping[str, Any]) -> dict[str, Any]:
    """Change only the two person targets; preserve the other seven gate rows."""
    canonical_targets = canonical_service.get("targets")
    if not isinstance(canonical_targets, Mapping) or len(canonical_targets) != 9:
        raise RuntimeError("frozen canonical nine-gate service contract drift")
    person_names = {"person_precision", "person_recall"}
    if not person_names.issubset(canonical_targets):
        raise RuntimeError("frozen canonical person gates are missing")
    targets = copy.deepcopy(dict(canonical_targets))
    for name in person_names:
        row = targets[name]
        if not isinstance(row, dict) or row.get("direction") != "higher":
            raise RuntimeError("frozen canonical person-gate shape drift")
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = float("nan")
        finite = math.isfinite(value)
        if (float(row.get("target", -1.0)) != 0.80
                or bool(row.get("passed")) != (finite and value >= 0.80)):
            raise RuntimeError("original canonical 0.80 person-gate result drift")
        row.update({
            "target": P070_PERSON_TARGET,
            "passed": finite and value >= P070_PERSON_TARGET,
            "attainment_ratio": value / P070_PERSON_TARGET if finite else None,
        })
        if not finite:
            row["value"] = None
    for name, row in canonical_targets.items():
        if name not in person_names and targets[name] != row:
            raise RuntimeError(f"supplemental p070 decision changed canonical gate: {name}")
    ratios = [row.get("attainment_ratio") for row in targets.values()]
    return {
        "schema": "splitfusion_fcos_relational_p070_service_decision_v1",
        "supplemental_to_unchanged_canonical_nine_gate_result": True,
        "person_precision_target": P070_PERSON_TARGET,
        "person_recall_target": P070_PERSON_TARGET,
        "other_seven_targets_unchanged": True,
        "targets": targets,
        "pass_count": sum(bool(row.get("passed")) for row in targets.values()),
        "all_pass": all(bool(row.get("passed")) for row in targets.values()),
        "minimum_attainment_ratio": (
            min(float(value) for value in ratios) if all(value is not None for value in ratios)
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen canonical v0.10 evaluation of one completed relational-p070 pass",
    )
    parser.add_argument("--prediction-dir", required=True, type=Path)
    args = parser.parse_args()
    prediction, _inference = validate_prediction_directory(args.prediction_dir)
    output = prediction / "evaluation_v010.json"
    if output.exists():
        raise FileExistsError(f"create-only evaluation output already exists: {output}")
    checkpoint = FROZEN_CHECKPOINT.resolve(strict=True)
    if sha256(checkpoint) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen epoch-26 checkpoint SHA-256 mismatch")

    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.base_runtime import load_base

    base = load_base()
    config = base.common.load_json(base.common.CONFIG_PATH)
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    scoring = base.evaluate.load_scoring()
    base.evaluate.install_undefined_localization_adapter(scoring)
    result = scoring.score_primary(
        dataset_root, prediction, checkpoint, FROZEN_CHECKPOINT_SHA256, 26,
    )
    result["service"] = base.evaluate.service(result)
    if len(result["service"]["targets"]) != 9:
        raise RuntimeError("frozen canonical nine-gate service contract drift")
    result["class_detail"] = {
        threshold: result["primary_v010"][threshold]["classes"]
        for threshold in ("0.20", "0.02")
    }
    result["p070_service_decision"] = p070_service_decision(result["service"])
    result["accepted_service_candidate_comparison"] = {
        "schema": "splitfusion_fcos_accepted_service_candidate_person_comparison_v1",
        "person_precision": ACCEPTED_SERVICE_CANDIDATE["person_precision"],
        "person_recall": ACCEPTED_SERVICE_CANDIDATE["person_recall"],
    }
    base.common.atomic_json(output, result, overwrite=False)
    print(json.dumps({
        "evaluation": str(output),
        "metrics": result["metrics"],
        "canonical_service": result["service"],
        "p070_service_decision": result["p070_service_decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

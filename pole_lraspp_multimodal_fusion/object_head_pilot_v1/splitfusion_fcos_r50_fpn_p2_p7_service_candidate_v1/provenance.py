from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    CANONICAL_SCORE_THRESHOLD,
    validate_configuration,
)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
LOCKED_CONFIG_PATH = PACKAGE / "locked_config.json"
FROZEN_CHECKPOINT_SHA256 = "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"
PERSON_RESULT_SHA256 = "a1bb8b2b7062abc2d0ef4c5cbc715154c5a4e9f1da64e050547de14c56bdddde"
PERSON_RESULT_RELATIVE_PATH = Path("experiments/person_instance_consolidation_v1/feasibility_result.json")
PERSON_RULE = {
    "grid_index": 27,
    "semantic_support_threshold": 0.10,
    "group_box_iou_threshold": 0.20,
}
VEHICLE_INTERVAL = {"lower_exclusive": 0.392763704, "upper_inclusive": 0.649181962}
VEHICLE_BASE_THRESHOLD = 0.5224518340619145
VEHICLE_LOGIT_BIAS = -1.476162131187961
VEHICLE_CLAMP_EPSILON = 1e-6
CANONICAL_THRESHOLD = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LockedConfiguration:
    person_result_path: Path
    person_result_sha256: str

    @property
    def person_rule(self) -> dict[str, float | int]:
        return dict(PERSON_RULE)

    @property
    def vehicle_calibration(self) -> dict[str, Any]:
        return {
            "train_only_feasible_interval": dict(VEHICLE_INTERVAL),
            "base_score_operating_threshold": VEHICLE_BASE_THRESHOLD,
            "logit_bias": VEHICLE_LOGIT_BIAS,
            "clamp_epsilon": VEHICLE_CLAMP_EPSILON,
            "canonical_score_threshold": CANONICAL_THRESHOLD,
            "arithmetic": "sigmoid(logit(clamp_fp32(base_score,1e-6,1-1e-6))+logit_bias_fp32)",
        }


def _validate_locked_file(config: dict[str, Any]) -> None:
    base = config.get("base", {})
    person = config.get("person", {})
    vehicle = config.get("vehicle", {})
    behavior = config.get("behavior", {})
    if (config.get("schema") != "splitfusion_fcos_service_candidate_locked_config_v1"
            or base != {
                "checkpoint": (
                    "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/"
                    "20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt"
                ),
                "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
                "epoch": 26,
            }
            or person != {
                "canonical_score_threshold": CANONICAL_THRESHOLD,
                "feasibility_result": str(PERSON_RESULT_RELATIVE_PATH),
                "feasibility_result_sha256": PERSON_RESULT_SHA256,
                **PERSON_RULE,
                "implementation": (
                    "splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core."
                    "retained_detection_indices"
                ),
                "required_status": "holdout_feasible",
                "source_commit": "a7b759b9e9565555340d3ea6175cb8283c73a687",
            }
            or vehicle != {
                "arithmetic": "sigmoid(logit(clamp_fp32(base_score,1e-6,1-1e-6))+logit_bias_fp32)",
                "base_score_operating_threshold": VEHICLE_BASE_THRESHOLD,
                "canonical_score_threshold": CANONICAL_THRESHOLD,
                "clamp_epsilon": VEHICLE_CLAMP_EPSILON,
                "logit_bias": VEHICLE_LOGIT_BIAS,
                "train_only_feasible_interval": VEHICLE_INTERVAL,
            }
            or behavior != {
                "candidate_creation": False,
                "candidate_order": "original_post_nms",
                "geometry_changed": False,
                "nms_rerun": False,
                "person_retained_fields_and_scores_changed": False,
                "segmentation_changed": False,
                "vehicle_candidates_filtered": False,
                "vehicle_non_score_fields_changed": False,
            }):
        raise RuntimeError("locked service-candidate configuration drift")
    validate_configuration(PERSON_RULE)
    if CANONICAL_SCORE_THRESHOLD != CANONICAL_THRESHOLD:
        raise RuntimeError("reviewed person canonical threshold drift")


def _validate_person_result(
    result: dict[str, Any], *, actual_sha256: str, expected_sha256: str,
) -> None:
    selected = result.get("selected_fit")
    holdout = result.get("holdout")
    if actual_sha256 != expected_sha256:
        raise RuntimeError("person consolidation feasibility-result SHA-256 mismatch")
    if (result.get("schema") != "splitfusion_fcos_person_instance_consolidation_result_v1"
            or result.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or result.get("status") != "holdout_feasible"
            or int(result.get("grid_configuration_count", -1)) != 36
            or int(result.get("holdout_evaluations", -1)) != 1
            or result.get("validation_or_test_accessed") is not False
            or not isinstance(selected, dict)
            or not isinstance(holdout, dict)
            or any(selected.get(name) != value for name, value in PERSON_RULE.items())
            or any(holdout.get(name) != value for name, value in PERSON_RULE.items())
            or float(selected.get("precision", -1.0)) < 0.80
            or float(selected.get("recall", -1.0)) < 0.80
            or float(holdout.get("precision", -1.0)) < 0.80
            or float(holdout.get("recall", -1.0)) < 0.80):
        raise RuntimeError("person consolidation feasibility result is not the locked feasible configuration")


def load_locked_configuration(result_path: Path | None = None) -> LockedConfiguration:
    config = json.loads(LOCKED_CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_locked_file(config)
    path = (ROOT / PERSON_RESULT_RELATIVE_PATH if result_path is None else Path(result_path)).resolve(strict=True)
    actual_hash = sha256(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    _validate_person_result(result, actual_sha256=actual_hash, expected_sha256=PERSON_RESULT_SHA256)
    return LockedConfiguration(person_result_path=path, person_result_sha256=actual_hash)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
LOCKED_CONFIG_PATH = PACKAGE / "locked_config.json"
BASE_SERVICE = (
    ROOT
    / "pole_lraspp_multimodal_fusion/object_head_pilot_v1"
    / "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_contract() -> dict[str, Any]:
    config = json.loads(LOCKED_CONFIG_PATH.read_text(encoding="utf-8"))
    base = config.get("base_service", {})
    person = config.get("person_output_filter", {})
    vehicle = config.get("vehicle_behavior", {})
    qualification = config.get("train_qualification", {})
    if not (
        config.get("schema")
        == "splitfusion_fcos_person_p025_service_candidate_locked_config_v1"
        and base.get("package") == "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1"
        and base.get("person_consolidation_score_threshold") == 0.20
        and person == {
            "score_threshold": 0.25,
            "comparison": "score_fp32 >= 0.25",
            "applied_after_base_service": True,
            "scores_changed": False,
            "non_score_fields_changed": False,
            "candidate_order_changed": False,
        }
        and vehicle == {"filtered": False, "reordered": False, "fields_changed": False}
        and qualification.get("avo_threshold") == 0.65
        and qualification.get("required_terminal") == "PERSON_P025_TRAIN_HOLDOUT_QUALIFIED"
        and config.get("decision")
        == "proposed_deployment_candidate_awaiting_final_acceptance"
        and config.get("approved_p020_service_automatically_replaced") is False
    ):
        raise RuntimeError("locked p025 service-candidate configuration drift")
    if sha256(BASE_SERVICE / "runtime.py") != base.get("runtime_sha256"):
        raise RuntimeError("base p020 service runtime SHA-256 drift")
    if sha256(BASE_SERVICE / "locked_config.json") != base.get("locked_config_sha256"):
        raise RuntimeError("base p020 service locked configuration SHA-256 drift")

    qualification_path = (ROOT / str(qualification["path"])).resolve(strict=True)
    if sha256(qualification_path) != qualification.get("sha256"):
        raise RuntimeError("p025 train qualification SHA-256 drift")
    result = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not (
        result.get("schema")
        == "splitfusion_fcos_person_p025_train_holdout_qualification_v1"
        and result.get("terminal") == qualification["required_terminal"]
        and result.get("qualified") is True
        and result.get("phase2_authorized") is True
        and result.get("validation_accessed") is False
        and result.get("test_accessed") is False
        and result.get("thresholds_evaluated") == [0.20, 0.25]
        and result.get("avo_threshold") == 0.65
        and all(result.get("qualification_gates", {}).values())
    ):
        raise RuntimeError("p025 train qualification is not eligible for deployment wrapping")
    return config

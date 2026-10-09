from __future__ import annotations

import argparse
import json
from pathlib import Path

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.runtime import (
    FROZEN_CHECKPOINT,
    sha256,
)

from .provenance import FROZEN_CHECKPOINT_SHA256, ROOT, load_locked_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen nine-gate evaluation of one service-candidate inference pass")
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    locked = load_locked_configuration()
    prediction = args.prediction_dir.resolve(strict=True)
    complete = prediction / "INFERENCE_COMPLETE"
    if (not complete.is_file()
            or complete.read_text(encoding="utf-8") != "SERVICE_CANDIDATE_INFERENCE_COMPLETE\n"):
        raise RuntimeError("service-candidate prediction directory is incomplete")
    inference = json.loads((prediction / "inference_manifest.json").read_text(encoding="utf-8"))
    if (inference.get("schema") != "splitfusion_fcos_service_candidate_inference_v1"
            or inference.get("checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or inference.get("person_feasibility_result_sha256") != locked.person_result_sha256
            or inference.get("person_configuration") != locked.person_rule
            or inference.get("vehicle_calibration") != locked.vehicle_calibration
            or float(inference.get("derived_threshold", -1.0)) != 0.20
            or int(inference.get("inference_pass_count", -1)) != 1
            or inference.get("candidate_set") != "person_consolidated_vehicle_unfiltered"
            or inference.get("candidate_creation") is not False
            or inference.get("nms_rerun") is not False
            or inference.get("candidate_order") != "original_post_nms"
            or inference.get("prediction_index") != "original_post_nms"
            or inference.get("person_retained_fields_and_scores_changed") is not False
            or inference.get("vehicle_candidates_filtered") is not False
            or inference.get("vehicle_non_score_fields_changed") is not False
            or inference.get("vehicle_scores_calibrated") is not True
            or inference.get("geometry_changed") is not False
            or inference.get("segmentation_changed") is not False):
        raise RuntimeError("service-candidate inference contract drift")
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
        raise RuntimeError("frozen nine-gate service contract drift")
    result["class_detail"] = {
        threshold: result["primary_v010"][threshold]["classes"] for threshold in ("0.20", "0.02")
    }
    output = args.output.resolve() if args.output is not None else prediction / "evaluation_v010.json"
    base.common.atomic_json(output, result, overwrite=False)
    print(json.dumps({"evaluation": str(output), "metrics": result["metrics"],
                      "service": result["service"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

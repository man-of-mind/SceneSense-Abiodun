from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import FROZEN_CHECKPOINT, FROZEN_CHECKPOINT_SHA256, ROOT, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one completed refined prediction directory")
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prediction = args.prediction_dir.resolve(strict=True)
    if not (prediction / "INFERENCE_COMPLETE").is_file():
        raise RuntimeError("refined prediction directory is incomplete")
    inference = json.loads((prediction / "inference_manifest.json").read_text(encoding="utf-8"))
    if (inference.get("schema") != "splitfusion_fcos_candidate_quality_inference_v1"
            or float(inference.get("derived_threshold", -1.0)) != 0.20
            or inference.get("checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256):
        raise RuntimeError("refined inference contract drift")
    checkpoint = FROZEN_CHECKPOINT.resolve(strict=True)
    if sha256(checkpoint) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen checkpoint SHA-256 mismatch")

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
    print(json.dumps({"evaluation": str(output), "metrics": result["metrics"], "service": result["service"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

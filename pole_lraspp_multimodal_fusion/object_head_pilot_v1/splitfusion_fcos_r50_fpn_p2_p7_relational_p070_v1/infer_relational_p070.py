from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.runtime import (
    load_frozen_runtime,
    require_device,
    sha256,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    PersonRoIDescriptor,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from .contract import CANONICAL_PERSON_THRESHOLD, load_revised_selector
from .runtime import apply_relational_p070_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Future single-pass revised-p070 validation inference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    relational = load_revised_selector(device)
    frozen = load_frozen_runtime(device)
    if frozen.checkpoint_sha256 != relational.base_checkpoint_sha256:
        raise RuntimeError("frozen base runtime/checkpoint contract mismatch")
    extractor = PersonRoIDescriptor().to(device).eval()
    dataset = frozen.base.data.InferenceDataset(frozen.dataset_root, "val")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    segmentation_dir = output / "segmentation"
    segmentation_dir.mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"
    segmentation_rows: list[dict[str, object]] = []
    base_count = retained_count = vehicle_count = person_count = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frozen.base.infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                fused, row, calibration = dataset[index]
                fused = fused.unsqueeze(0).to(device)
                calibration_device = {name: value.to(device) for name, value in calibration.items()}
                outputs = frozen.model(fused, dense=False)
                base = frozen.model.postprocess(outputs, [calibration_device])[0]
                detections, original_indices = apply_relational_p070_policy(
                    outputs, base, relational, extractor,
                )
                writer.writerows(combined_records(frozen.base, row, detections, original_indices))
                base_count += len(base["scores"])
                retained_count += len(detections["scores"])
                vehicle_count += int((detections["labels_internal"].long() != 1).sum())
                person_count += int((detections["labels_internal"].long() == 1).sum())

                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                labels = F.interpolate(
                    outputs["semantic_logits"].float(), size=source_hw,
                    mode="bilinear", align_corners=False,
                ).argmax(1)[0].cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"
                path = output / relative
                if not cv2.imwrite(str(path), labels):
                    raise RuntimeError(f"failed segmentation write {path}")
                segmentation_rows.append({
                    "sample_id": row["sample_id"], "prediction_path": str(relative),
                    "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(path),
                })
                if (index + 1) % 500 == 0:
                    print(json.dumps({"validation_frames": index + 1,
                                      "base_candidates": base_count,
                                      "retained_candidates": retained_count}), flush=True)

    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"),
        )
        writer.writeheader()
        writer.writerows(segmentation_rows)
    detection_hash = sha256(detections_path)
    segmentation_hash = sha256(segmentation_manifest)
    manifest = {
        "schema": "splitfusion_fcos_relational_p070_inference_v1",
        "base_checkpoint_sha256": relational.base_checkpoint_sha256,
        "selector_checkpoint_sha256": relational.selector_checkpoint_sha256,
        "historical_selector_status_unchanged": relational.historical_status,
        "revised_objective": {"precision": 0.70, "recall": 0.70},
        "deployment_bias": relational.deployment_bias,
        "deployment_threshold": CANONICAL_PERSON_THRESHOLD,
        "validation_frames": len(dataset),
        "inference_pass_count": 1,
        "candidate_creation": False,
        "nms_rerun": False,
        "candidate_order": "original_post_nms",
        "prediction_index": "original_post_nms",
        "consolidation_is_feature_only": True,
        "vehicle_behavior": "bit_exact_service_candidate_v1",
        "geometry_changed": False,
        "segmentation_changed": False,
        "base_detection_predictions": base_count,
        "detection_predictions": retained_count,
        "retained_vehicle_predictions": vehicle_count,
        "retained_person_predictions": person_count,
        "detections_sha256": detection_hash,
        "segmentation_manifest_sha256": segmentation_hash,
        "prediction_set_sha256": hashlib.sha256(
            (detection_hash + segmentation_hash).encode(),
        ).hexdigest(),
        "depth_paths_opened": 0,
        "semantic_gt_paths_opened": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "external_detection_fields": list(frozen.base.infer.FIELDS),
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "INFERENCE_COMPLETE").write_text(
        "RELATIONAL_P070_INFERENCE_COMPLETE\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "base_candidates": base_count,
                      "retained_candidates": retained_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

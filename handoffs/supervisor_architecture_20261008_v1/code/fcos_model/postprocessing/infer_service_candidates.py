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

from .provenance import FROZEN_CHECKPOINT_SHA256, load_locked_configuration
from .runtime import apply_combined_service_policy, combined_records


def main() -> int:
    parser = argparse.ArgumentParser(description="One-pass validation inference for locked service candidates")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    locked = load_locked_configuration()
    device = require_device(args.device)
    runtime = load_frozen_runtime(device)
    if runtime.checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("combined runtime is not bound to frozen epoch 26")
    dataset = runtime.base.data.InferenceDataset(runtime.dataset_root, "val")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    segmentation_dir = output / "segmentation"
    segmentation_dir.mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"
    segmentation_rows: list[dict[str, object]] = []
    base_count = retained_count = vehicle_count = person_count = 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime.base.infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                fused, row, calibration = dataset[index]
                fused = fused.unsqueeze(0).to(device)
                calibration_device = {name: value.to(device) for name, value in calibration.items()}
                outputs = runtime.model(fused, dense=False)
                base_detections = runtime.model.postprocess(outputs, [calibration_device])[0]
                detections, original_indices = apply_combined_service_policy(outputs, base_detections)
                writer.writerows(combined_records(runtime.base, row, detections, original_indices))
                base_count += len(base_detections["scores"])
                retained_count += len(detections["scores"])
                vehicle_count += int((detections["labels_internal"].long() != 1).sum())
                person_count += int((detections["labels_internal"].long() == 1).sum())

                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                semantic_labels = F.interpolate(
                    outputs["semantic_logits"].float(), size=source_hw, mode="bilinear", align_corners=False,
                ).argmax(1)[0]
                labels_np = semantic_labels.cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"
                path = output / relative
                if not cv2.imwrite(str(path), labels_np):
                    raise RuntimeError(f"failed segmentation write {path}")
                segmentation_rows.append({
                    "sample_id": row["sample_id"], "prediction_path": str(relative),
                    "width": labels_np.shape[1], "height": labels_np.shape[0], "sha256": sha256(path),
                })
                if (index + 1) % 500 == 0:
                    print(json.dumps({"validation_frames": index + 1, "base_candidates": base_count,
                                      "retained_candidates": retained_count}), flush=True)

    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    detection_hash = sha256(detections_path)
    segmentation_hash = sha256(segmentation_manifest)
    prediction_set_hash = hashlib.sha256((detection_hash + segmentation_hash).encode()).hexdigest()
    manifest = {
        "schema": "splitfusion_fcos_service_candidate_inference_v1",
        "base_checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "person_feasibility_result": str(locked.person_result_path),
        "person_feasibility_result_sha256": locked.person_result_sha256,
        "person_configuration": locked.person_rule,
        "vehicle_calibration": locked.vehicle_calibration,
        "validation_frames": len(dataset),
        "inference_pass_count": 1,
        "score_floor": 0.0,
        "base_candidate_score_floor": 0.02,
        "derived_threshold": 0.20,
        "base_post_nms_candidates_per_image": 100,
        "candidate_set": "person_consolidated_vehicle_unfiltered",
        "candidate_creation": False,
        "nms_rerun": False,
        "candidate_order": "original_post_nms",
        "prediction_index": "original_post_nms",
        "person_retained_fields_and_scores_changed": False,
        "vehicle_candidates_filtered": False,
        "vehicle_non_score_fields_changed": False,
        "vehicle_scores_calibrated": True,
        "geometry_changed": False,
        "segmentation_changed": False,
        "candidate_identity": ["image", "level", "flattened_point", "internal_class"],
        "native_object_grid": [108, 192],
        "fpn_levels": list(runtime.base.model.LEVELS),
        "base_detection_predictions": base_count,
        "detection_predictions": retained_count,
        "retained_vehicle_predictions": vehicle_count,
        "retained_person_predictions": person_count,
        "detections_sha256": detection_hash,
        "segmentation_manifest_sha256": segmentation_hash,
        "prediction_set_sha256": prediction_set_hash,
        "depth_paths_opened": 0,
        "depth_labels_used": False,
        "semantic_gt_paths_opened": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
        "external_detection_fields": list(runtime.base.infer.FIELDS),
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "INFERENCE_COMPLETE").write_text("SERVICE_CANDIDATE_INFERENCE_COMPLETE\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "base_candidates": base_count,
                      "retained_candidates": retained_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

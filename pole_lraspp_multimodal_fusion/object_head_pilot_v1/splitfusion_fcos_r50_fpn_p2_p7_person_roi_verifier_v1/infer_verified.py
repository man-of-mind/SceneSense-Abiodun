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

from .runtime import FROZEN_CHECKPOINT_SHA256, load_frozen_runtime, require_device, sha256
from .verifier import (
    VERIFIER_ARCHITECTURE,
    PersonRoIDescriptor,
    PersonVerifier,
    apply_person_refinement,
    fp16_round_trip_roi_descriptors,
)


def load_verifier(path: Path, device: torch.device) -> tuple[PersonVerifier, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != "splitfusion_fcos_person_roi_verifier_v1"
            or checkpoint.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or checkpoint.get("architecture") != VERIFIER_ARCHITECTURE
            or int(checkpoint.get("training", {}).get("epochs", -1)) != 5
            or checkpoint.get("status") != "train_feasible"
            or checkpoint.get("validation_allowed") is not True
            or checkpoint.get("validation_or_test_accessed") is not False):
        raise RuntimeError("verifier is not holdout-feasible or its checkpoint contract drifted")
    calibration_bias = float(checkpoint["holdout"]["calibration_bias"])
    if not np.isfinite(calibration_bias):
        raise FloatingPointError("non-finite person calibration bias")
    head = PersonVerifier()
    head.load_state_dict(checkpoint["verifier"], strict=True)
    head.to(device).eval()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in head.parameters()):
        raise FloatingPointError("non-finite verifier checkpoint parameter")
    return head, calibration_bias


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen-base inference with person-only ROI verification")
    parser.add_argument("--verifier-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    verifier_path = args.verifier_checkpoint.resolve(strict=True)
    verifier, calibration_bias = load_verifier(verifier_path, device)
    extractor = PersonRoIDescriptor().to(device).eval()
    runtime = load_frozen_runtime(device)
    dataset = runtime.base.data.InferenceDataset(runtime.dataset_root, "val")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    segmentation_dir = output / "segmentation"
    segmentation_dir.mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"
    segmentation_rows: list[dict[str, object]] = []
    detection_count = person_count = 0
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
                descriptors, scalars, person_indices = extractor(outputs, base_detections)
                descriptors = fp16_round_trip_roi_descriptors(descriptors)
                features = torch.cat((descriptors, scalars), dim=1)
                delta = verifier(features)
                detections = apply_person_refinement(
                    base_detections, delta, calibration_bias=calibration_bias,
                )
                if len(detections["scores"]) != len(base_detections["scores"]):
                    raise RuntimeError("person verifier changed the original candidate set")
                for prediction_index in range(len(detections["scores"])):
                    writer.writerow(runtime.base.infer.record(detections, row, prediction_index))
                detection_count += len(detections["scores"])
                person_count += person_indices.numel()

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
                    "sample_id": row["sample_id"],
                    "prediction_path": str(relative),
                    "width": labels_np.shape[1],
                    "height": labels_np.shape[0],
                    "sha256": sha256(path),
                })
                if (index + 1) % 500 == 0:
                    print(json.dumps({"validation_frames": index + 1, "detections": detection_count,
                                      "person_candidates": person_count}), flush=True)

    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    detection_hash = sha256(detections_path)
    segmentation_hash = sha256(segmentation_manifest)
    prediction_set_hash = hashlib.sha256((detection_hash + segmentation_hash).encode()).hexdigest()
    manifest = {
        "schema": "splitfusion_fcos_person_roi_verifier_inference_v1",
        "base_checkpoint": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "verifier_checkpoint": str(verifier_path),
        "verifier_checkpoint_sha256": sha256(verifier_path),
        "calibration_bias": calibration_bias,
        "validation_frames": len(dataset),
        "inference_pass_count": 1,
        "score_floor": 0.02,
        "derived_threshold": 0.20,
        "topk_per_level": 1000,
        "nms_iou": 0.60,
        "detections_per_image": 100,
        "candidate_set": "unchanged_post_original_nms",
        "candidate_creation": False,
        "boxes_and_geometry_changed": False,
        "vehicle_fields_and_scores_changed": False,
        "segmentation_changed": False,
        "candidate_identity": ["image", "level", "flattened_point", "internal_class"],
        "native_object_grid": [108, 192],
        "fpn_levels": list(runtime.base.model.LEVELS),
        "detection_predictions": detection_count,
        "person_candidates_verified": person_count,
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
    (output / "INFERENCE_COMPLETE").write_text("PERSON_ROI_VERIFIER_INFERENCE_COMPLETE\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "detections": detection_count,
                      "person_candidates": person_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIG_PATH, ROOT, atomic_json, atomic_text, load_json, named_tensor_hash, sha256, utc_now
from data import InferenceDataset
from model import CLASS_NAMES, LEVELS, build_model

FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "internal_class", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "size_x", "size_y", "size_z",
    "yaw_sin", "yaw_cos", "parked_score", "radar_support_score", "center_x_px", "center_y_px",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "fpn_level", "level_index", "point_index",
    "candidate_identity", "physical_ray_x_px", "physical_ray_y_px", "actor_forward_depth_m", "depth_bin",
    "depth_residual",
)


def record(result: dict[str, torch.Tensor], row: dict[str, str], index: int) -> dict[str, Any]:
    box = result["boxes"][index].double(); label = int(result["labels_internal"][index])
    local, world = result["local_xyz"][index].double(), result["world_xyz"][index].double()
    dimensions, yaw = result["dimensions"][index].double(), result["yaw"][index].double()
    uv = result["physical_uv"][index].double(); identity = result["candidate_identity"][index]
    values = {
        "sample_id": row["sample_id"], "frame_id": row["frame_id"], "prediction_index": index,
        "class_name": CLASS_NAMES[label], "internal_class": label, "score": float(result["scores"][index]),
        "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
        "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
        "size_x": float(dimensions[0]), "size_y": float(dimensions[1]), "size_z": float(dimensions[2]),
        "yaw_sin": float(yaw[0]), "yaw_cos": float(yaw[1]), "parked_score": 0.0,
        "radar_support_score": 0.0, "center_x_px": float((box[0] + box[2]) / 2),
        "center_y_px": float((box[1] + box[3]) / 2), "bbox_x0": float(box[0]), "bbox_y0": float(box[1]),
        "bbox_x1": float(box[2]), "bbox_y1": float(box[3]), "fpn_level": LEVELS[int(identity[1])],
        "level_index": int(identity[1]), "point_index": int(identity[2]),
        "candidate_identity": ":".join(str(int(value)) for value in identity),
        "physical_ray_x_px": float(uv[0]), "physical_ray_y_px": float(uv[1]),
        "actor_forward_depth_m": float(result["depth"][index]), "depth_bin": int(result["depth_bin"][index]),
        "depth_residual": float(result["depth_residual"][index]),
    }
    if not all(math.isfinite(float(value)) for value in values.values() if isinstance(value, (int, float))):
        raise FloatingPointError(f"nonfinite scored prediction {row['sample_id']} {index}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int, choices=(3, 8, 16, 22, 26))
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "TRAINING_COMPLETE").is_file(): raise RuntimeError("validation forbidden before epoch 26 completion")
    config = load_json(CONFIG_PATH); priors = load_json(experiment / "TRAIN_ONLY_PRIORS.json")
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    checkpoint_path = experiment / f"checkpoints/epoch_{args.epoch:03d}.pt"
    checkpoint_hash = sha256(checkpoint_path); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != args.epoch or checkpoint["config_sha256"] != sha256(CONFIG_PATH):
        raise RuntimeError("checkpoint provenance drift")
    device = torch.device("cuda:0"); model, _ = build_model(priors, device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    dataset = InferenceDataset(dataset_root, "val")
    output = experiment / f"predictions/epoch_{args.epoch:03d}"; output.mkdir(parents=True, exist_ok=False)
    segmentation_dir = output / "segmentation"; segmentation_dir.mkdir()
    atomic_json(output / "INFERENCE_STARTED.json", {"created_utc": utc_now(), "epoch": args.epoch,
                                                       "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                                                       "score_floor": 0.02, "validation_frames": len(dataset),
                                                       "depth_paths_opened": 0, "semantic_gt_paths_opened": 0}, overwrite=False)
    detections_path = output / "detections.csv"; segmentation_manifest = output / "segmentation_manifest.csv"
    segmentation_rows, detection_count = [], 0; started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                fused, row, calibration = dataset[index]
                fused = fused.unsqueeze(0).to(device, non_blocking=True)
                calibration_gpu = {name: value.to(device) for name, value in calibration.items()}
                outputs = model(fused, dense=False)
                detections = model.postprocess(outputs, [calibration_gpu])[0]
                for prediction_index in range(len(detections["scores"])):
                    writer.writerow(record(detections, row, prediction_index))
                detection_count += len(detections["scores"])
                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                labels = F.interpolate(outputs["semantic_logits"].float(), size=source_hw,
                                       mode="bilinear", align_corners=False).argmax(1)[0]
                labels_np = labels.cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"; path = output / relative
                if not cv2.imwrite(str(path), labels_np): raise RuntimeError(f"failed segmentation write {path}")
                segmentation_rows.append({"sample_id": row["sample_id"], "prediction_path": str(relative),
                                          "width": labels_np.shape[1], "height": labels_np.shape[0], "sha256": sha256(path)})
                if (index + 1) % 500 == 0:
                    print(json.dumps({"epoch": args.epoch, "validation_frames_complete": index + 1,
                                      "detections": detection_count}), flush=True)
    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader(); writer.writerows(segmentation_rows)
    inference = {"schema": "splitfusion_fcos_inference_manifest_v1", "created_utc": utc_now(),
                 "epoch": args.epoch, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                 "checkpoint_model_state_sha256": named_tensor_hash(checkpoint["model"].items()),
                 "validation_frames": len(dataset), "inference_pass_count": 1, "score_floor": 0.02,
                 "derived_threshold": 0.20, "topk_per_level": 1000, "nms_iou": 0.60,
                 "detections_per_image": 100, "world_nms": False, "native_object_grid": [108, 192],
                 "fpn_levels": list(LEVELS), "candidate_identity": ["image", "level", "flattened_point", "internal_class"],
                 "detection_predictions": detection_count, "detections_sha256": sha256(detections_path),
                 "segmentation_manifest_sha256": sha256(segmentation_manifest),
                 "prediction_set_sha256": hashlib.sha256((sha256(detections_path) + sha256(segmentation_manifest)).encode()).hexdigest(),
                 "raw_transport_bytes": 22020096, "transport_dtype": "float32", "transport_codec": "identity",
                 "depth_paths_opened": 0, "depth_labels_used": False, "semantic_gt_paths_opened": 0,
                 "wall_seconds": time.monotonic() - started,
                 "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                 "external_detection_fields": list(FIELDS)}
    atomic_json(output / "inference_manifest.json", inference, overwrite=False)
    atomic_text(output / "INFERENCE_COMPLETE", "ONE_SCORE_FLOOR_0_02_PASS_COMPLETE\n", overwrite=False)
    print(json.dumps({"epoch": args.epoch, "detections": detection_count,
                      "wall_seconds": inference["wall_seconds"], "sha256": inference["prediction_set_sha256"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())

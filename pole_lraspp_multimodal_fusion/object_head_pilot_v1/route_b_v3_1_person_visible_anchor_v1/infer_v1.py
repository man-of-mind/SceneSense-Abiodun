#!/usr/bin/env python3
"""Exactly one floor-0.02 validation inference pass per designated checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

from common_v1 import sha256, utc_now, write_json_x, write_text_x  # noqa: E402
from decode_v1 import TOPK_PER_CLASS, decode_all  # noqa: E402
from model_v1 import build_model  # noqa: E402
from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import parse_matrix  # noqa: E402

FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1",
)
SCORE_FLOOR = 0.02
RANGE_M = 40.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    if sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("candidate checkpoint SHA mismatch")
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA runtime unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["resolved_config"]
    if (checkpoint["numerical_policy"] != "full_fp32"
            or checkpoint["grad_scaler_enabled"] is not False
            or sha256(experiment / "RESOLVED_CONFIG.json") != checkpoint["resolved_config_sha256"]):
        raise RuntimeError("candidate checkpoint numerical/config registration drift")
    device = torch.device("cuda")
    model = build_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]),
        depth_bounds_m=tuple(config["person_private"]["depth_bounds_m"]), device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    dataset = experiment / "dataset"
    rows = [row for row in read_manifest(dataset / "manifest.csv") if row["split"] == "val"]
    if len(rows) != 3345 or len({row["sample_id"] for row in rows}) != 3345:
        raise RuntimeError("validation population drift")
    if any("canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"]
           for row in rows):
        raise RuntimeError("locked test reference in inference")
    output = experiment / "predictions" / args.tag
    output.mkdir(parents=True, exist_ok=False); (output / "segmentation").mkdir()
    write_json_x(output / "inference_started.json", {
        "created_utc": utc_now(), "tag": args.tag, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": args.checkpoint_sha256, "checkpoint_epoch": checkpoint["epoch"],
        "validation_frames": len(rows), "score_floor": SCORE_FLOOR,
        "inference_pass_number_for_checkpoint": 1,
    })
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    detections_path = output / "detections.csv"
    segmentation_rows: list[dict[str, Any]] = []
    prediction_count = vehicle_count = person_count = 0
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                fused, output_hw, _original = load_fused_tensor(
                    row, dataset, tuple(config["model_size_wh"]), device,
                )
                outputs = model(fused, feature_drop_fraction=0.0)
                if not (torch.isfinite(outputs["out"]).all().item()
                        and torch.isfinite(outputs["object"]).all().item()
                        and all(torch.isfinite(value).all().item()
                                for value in outputs["person_private"].values())):
                    raise RuntimeError(f"nonfinite inference output: {row['sample_id']}")
                seg_logits = F.interpolate(
                    outputs["out"], size=output_hw, mode="bilinear", align_corners=False,
                )
                labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"
                path = output / relative
                if not cv2.imwrite(str(path), labels):
                    raise RuntimeError(f"segmentation write failed: {row['sample_id']}")
                segmentation_rows.append({
                    "sample_id": row["sample_id"], "prediction_path": str(relative),
                    "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(path),
                })
                matrix = parse_matrix(row["camera_matrix_json"])
                predictions: list[dict[str, float]] = []
                if matrix is not None:
                    sx = config["model_size_wh"][0] / float(row["camera_width"])
                    sy = config["model_size_wh"][1] / float(row["camera_height"])
                    intrinsic = np.asarray([
                        [float(row["camera_fx"]) * sx, 0.0, float(row["camera_cx"]) * sx],
                        [0.0, float(row["camera_fy"]) * sy, float(row["camera_cy"]) * sy],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    predictions = decode_all(
                        outputs, camera_matrix=matrix, intrinsic_model=intrinsic,
                        score_threshold=SCORE_FLOOR,
                        offset_scales=config["resolved_offset_scales"],
                        depth_bounds_m=config["person_private"]["depth_bounds_m"],
                        topk=int(config["training"]["topk_per_class"]),
                        model_size=config["model_size_wh"],
                    )
                    camera = np.asarray(matrix)[:3, 3]
                    predictions = [prediction for prediction in predictions if math.hypot(
                        float(prediction["world_x"]) - camera[0],
                        float(prediction["world_y"]) - camera[1],
                    ) <= RANGE_M]
                for prediction_index, prediction in enumerate(predictions):
                    if not all(math.isfinite(float(prediction[field])) for field in (
                        "score", "world_x", "world_y", "world_z", "local_x", "local_y",
                        "local_z", "size_x", "size_y", "size_z", "center_x_px", "center_y_px",
                    )):
                        raise RuntimeError(f"nonfinite decoded prediction: {row['sample_id']}")
                    writer.writerow({
                        "sample_id": row["sample_id"], "frame_id": row["frame_id"],
                        "prediction_index": prediction_index,
                        **{field: prediction.get(field, "") for field in FIELDS
                           if field not in {"sample_id", "frame_id", "prediction_index"}},
                    })
                    vehicle_count += int(prediction["class_name"] == "vehicle")
                    person_count += int(prediction["class_name"] == "person")
                prediction_count += len(predictions)
                if index % 500 == 0:
                    print(f"[visible anchor inference {args.tag}] {index}/{len(rows)}", flush=True)
    segmentation_manifest = output / "segmentation_manifest.csv"
    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        fields = ("sample_id", "prediction_path", "width", "height", "sha256")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(segmentation_rows)
    manifest = {
        "schema": "route_b_v3_1_person_visible_anchor_inference_v1",
        "created_utc": utc_now(), "tag": args.tag, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": args.checkpoint_sha256, "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_frames": len(rows), "inference_pass_count": 1,
        "one_pass_supplies_thresholds": [0.20, 0.02], "score_floor": SCORE_FLOOR,
        "topk_per_class": TOPK_PER_CLASS, "native_object_grid": [108, 192],
        "native_local_maximum_kernel": 3, "world_nms": "none",
        "input_size": list(config["model_size_wh"]), "numerical_policy": "full_fp32",
        "private_fp16_used": False, "q": 0, "ae": False,
        "transported_bundle": ["low", "high"],
        "tail_calibration_metadata": ["camera_intrinsics", "camera_to_world"],
        "tail_raw_sensor_side_channels": [], "external_detection_fields": list(FIELDS),
        "detection_predictions": prediction_count, "vehicle_detection_rows": vehicle_count,
        "person_detection_rows": person_count, "detections_sha256": sha256(detections_path),
        "segmentation_manifest_sha256": sha256(segmentation_manifest),
        "prediction_set_sha256": hashlib.sha256(
            (sha256(detections_path) + sha256(segmentation_manifest)).encode("ascii")
        ).hexdigest(),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    write_json_x(output / "inference_manifest.json", manifest)
    write_text_x(output / "INFERENCE_COMPLETE", "ONE_FLOOR_0_02_INFERENCE_PASS_COMPLETE\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

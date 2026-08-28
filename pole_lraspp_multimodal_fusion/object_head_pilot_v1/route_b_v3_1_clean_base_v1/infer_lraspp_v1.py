#!/usr/bin/env python3
"""Run one create-only clean LR-ASPP validation inference pass at score floor 0.02."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
if str(FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSION_ROOT))

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor  # noqa: E402
from pole_lraspp_multimodal_fusion.model import OBJECT_HEAD_CHANNELS, build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import decode_objects, parse_matrix  # noqa: E402


BASE_CONFIG = FUSION_ROOT / "object_head_pilot_v1/configs/route_b_noae_precision_pilot_v1.yaml"
FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_model(checkpoint_path: Path, device: torch.device):
    config = load_config(BASE_CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cfg = config["training"]
    object_cfg = config.get("object_heads", {})
    fusion_cfg = config.get("fusion", {})
    num_classes = int(train_cfg.get("num_classes", 3))
    input_size = tuple(int(value) for value in checkpoint.get("input_size", train_cfg["input_size"]))
    object_names = tuple(checkpoint.get("object_class_names") or object_cfg["object_classes"])
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=int(checkpoint.get("radar_channels") or fusion_cfg.get("radar_channels", 4)),
        pretrained=False,
        object_channels=int(checkpoint.get("object_channels") or object_cfg.get("output_channels", OBJECT_HEAD_CHANNELS)),
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(checkpoint.get("fuse_low_into_object_head")) or bool(object_cfg.get("fuse_low_feature", False)),
        head_arch=str(checkpoint.get("object_head_arch") or object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(checkpoint.get("object_use_coordconv")) or bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(checkpoint.get("object_head_depth") or object_cfg.get("head_depth", 2)),
        predict_bbox2d=bool(checkpoint.get("object_predict_bbox2d")) or bool(object_cfg.get("predict_bbox2d", False)),
        use_groundplane_prior=bool(checkpoint.get("object_use_groundplane_prior")) or bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(checkpoint.get("object_groundplane_params") or object_cfg.get("groundplane_params", {})),
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model, input_size, object_names


def run(experiment: Path, checkpoint_path: Path, expected_sha: str, tag: str) -> int:
    dataset = experiment / "dataset"
    rows = [row for row in read_manifest(dataset / "manifest.csv") if row.get("split") == "val"]
    if len(rows) != 3345 or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("v3.1 validation manifest count/uniqueness failure")
    if any("canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"] for row in rows):
        raise RuntimeError("locked payload reference in inference input")
    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != expected_sha:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = experiment / "predictions" / tag
    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    write_json_x(output / "inference_started.json", {
        "tag": tag, "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "validation_frames": len(rows), "q": 0.0, "score_floor": 0.02,
    })
    device = torch.device("cuda")
    checkpoint, model, input_size, class_names = load_model(checkpoint_path, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    detections = output / "detections.csv"
    segmentation_rows: list[dict[str, Any]] = []
    prediction_count = 0
    with detections.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
        writer.writeheader()
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                fused, output_hw, _original_size = load_fused_tensor(row, dataset, input_size, device)
                outputs = model(fused, feature_drop_fraction=0.0)
                logits = F.interpolate(outputs["out"], size=output_hw, mode="bilinear", align_corners=False)
                labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                seg_rel = Path("segmentation") / f"{row['sample_id']}.png"
                seg_path = output / seg_rel
                if seg_path.exists() or not cv2.imwrite(str(seg_path), labels):
                    raise RuntimeError(f"create-only segmentation write failed: {row['sample_id']}")
                segmentation_rows.append({
                    "sample_id": row["sample_id"], "prediction_path": str(seg_rel),
                    "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(seg_path),
                })
                matrix = parse_matrix(row["camera_matrix_json"])
                predictions = [] if matrix is None else decode_objects(
                    outputs["object"], camera_matrix=matrix, topk=120, score_threshold=0.02,
                    nms_radius_px=2, object_class_names=class_names,
                    predict_bbox2d=bool(checkpoint.get("object_predict_bbox2d")),
                )
                if matrix is not None:
                    camera = np.asarray(matrix)[:3, 3]
                    predictions = [
                        prediction for prediction in predictions
                        if math.hypot(float(prediction["world_x"]) - camera[0], float(prediction["world_y"]) - camera[1]) <= 40.0
                    ]
                for pred_index, prediction in enumerate(predictions):
                    for key in ("score", "world_x", "world_y", "world_z", "size_x", "size_y", "size_z"):
                        if not math.isfinite(float(prediction[key])):
                            raise RuntimeError(f"nonfinite prediction {key}: {row['sample_id']}")
                    writer.writerow({
                        "sample_id": row["sample_id"], "frame_id": row["frame_id"],
                        "prediction_index": pred_index,
                        **{field: prediction.get(field, "") for field in FIELDS if field not in {"sample_id", "frame_id", "prediction_index"}},
                    })
                prediction_count += len(predictions)
                if index % 500 == 0:
                    print(f"[inference {tag}] {index}/{len(rows)}", flush=True)
    seg_manifest = output / "segmentation_manifest.csv"
    with seg_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    combined_hash = hashlib.sha256((sha256(detections) + sha256(seg_manifest)).encode("ascii")).hexdigest()
    result = {
        "schema": "route_b_v3_1_lraspp_inference_v1", "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation_frames": len(rows),
        "inference_pass_count": 1, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "input_size": list(input_size), "q": 0.0, "score_floor": 0.02,
        "topk": 120, "image_nms_radius_px": 2, "prediction_range_m": 40.0,
        "detection_predictions": prediction_count, "detections_sha256": sha256(detections),
        "segmentation_manifest_sha256": sha256(seg_manifest), "prediction_set_sha256": combined_hash,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }
    write_json_x(output / "inference_manifest.json", result)
    (output / "INFERENCE_COMPLETE").write_text("INFERENCE_COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    return run(args.experiment.resolve(), args.checkpoint, args.checkpoint_sha256, args.tag)


if __name__ == "__main__":
    raise SystemExit(main())

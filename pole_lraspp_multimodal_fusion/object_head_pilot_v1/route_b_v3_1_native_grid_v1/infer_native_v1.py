#!/usr/bin/env python3
"""One create-only native-grid validation inference pass at score floor 0.02.

The detection CSV schema, the segmentation output and the manifest are byte-identical
in shape to the v3.1 clean-base pass, so the external object record and the downstream
spatial-map contract are unchanged. Only the decoder geometry differs.
"""

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

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for _path in (str(ROOT), str(FUSION_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.common import read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.evaluate_fusion import load_fused_tensor  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import parse_matrix  # noqa: E402
from decode_v1 import (  # noqa: E402
    TOPK_PER_CLASS, decode_native_objects,
)
from model_v1 import (  # noqa: E402
    NATIVE_GRID, build_native_grid_model,
)

# Unchanged v3.1 external detection schema.
FIELDS = (
    "sample_id", "frame_id", "prediction_index", "class_name", "score",
    "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked_score",
    "radar_support_score", "center_x_px", "center_y_px", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1",
)
SCORE_FLOOR = 0.02
RANGE_M = 40.0


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    object_cfg = config["object_heads"]
    model = build_native_grid_model(
        num_classes=int(config["training"].get("num_classes", 3)),
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model, tuple(checkpoint["input_size"]), tuple(object_cfg["object_classes"])


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
        "validation_frames": len(rows), "q": 0.0, "score_floor": SCORE_FLOOR,
    })

    device = torch.device("cuda")
    checkpoint, model, input_size, class_names = load_model(checkpoint_path, device)
    native_stride = int(checkpoint["native_stride"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    detections = output / "detections.csv"
    segmentation_rows: list[dict[str, Any]] = []
    prediction_count = 0
    observed_grid: list[int] = []

    with detections.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
        writer.writeheader()
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                fused, output_hw, _original = load_fused_tensor(row, dataset, input_size, device)
                outputs = model(fused, feature_drop_fraction=0.0)
                if not observed_grid:
                    observed_grid = list(outputs["object"].shape[-2:])
                    if observed_grid != [NATIVE_GRID[1], NATIVE_GRID[0]]:
                        raise RuntimeError(f"object grid {observed_grid} is not native {NATIVE_GRID}")
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
                predictions = [] if matrix is None else decode_native_objects(
                    outputs["object"], camera_matrix=matrix, score_threshold=SCORE_FLOOR,
                    topk=TOPK_PER_CLASS, stride=native_stride,
                    model_size=input_size, object_class_names=class_names,
                )
                if matrix is not None:
                    camera = np.asarray(matrix)[:3, 3]
                    predictions = [
                        item for item in predictions
                        if math.hypot(float(item["world_x"]) - camera[0],
                                      float(item["world_y"]) - camera[1]) <= RANGE_M
                    ]
                for prediction_index, prediction in enumerate(predictions):
                    for key in ("score", "world_x", "world_y", "world_z", "size_x", "size_y", "size_z"):
                        if not math.isfinite(float(prediction[key])):
                            raise RuntimeError(f"nonfinite prediction {key}: {row['sample_id']}")
                    writer.writerow({
                        "sample_id": row["sample_id"], "frame_id": row["frame_id"],
                        "prediction_index": prediction_index,
                        **{field: prediction.get(field, "") for field in FIELDS
                           if field not in {"sample_id", "frame_id", "prediction_index"}},
                    })
                prediction_count += len(predictions)
                if index % 500 == 0:
                    print(f"[inference {tag}] {index}/{len(rows)}", flush=True)

    seg_manifest = output / "segmentation_manifest.csv"
    with seg_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    combined = hashlib.sha256((sha256(detections) + sha256(seg_manifest)).encode("ascii")).hexdigest()
    result = {
        "schema": "route_b_v3_1_native_grid_inference_v1", "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation_frames": len(rows),
        "inference_pass_count": 1, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "input_size": list(input_size), "q": 0.0, "score_floor": SCORE_FLOOR,
        "native_stride": native_stride, "native_object_grid": observed_grid,
        "topk_per_class": TOPK_PER_CLASS, "local_maximum_kernel": 3,
        "image_nms_radius_px": None, "world_nms": "none", "prediction_range_m": RANGE_M,
        "detection_predictions": prediction_count, "detections_sha256": sha256(detections),
        "segmentation_manifest_sha256": sha256(seg_manifest), "prediction_set_sha256": combined,
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

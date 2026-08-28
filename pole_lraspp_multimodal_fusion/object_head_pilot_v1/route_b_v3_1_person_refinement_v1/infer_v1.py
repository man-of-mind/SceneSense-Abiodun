#!/usr/bin/env python3
"""One create-only validation pass for a person-refinement checkpoint."""

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
for path in (str(PACKAGE_ROOT), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from person_decode_v1 import TOPK_PER_CLASS, decode_all  # noqa: E402
from person_model_v1 import build_model  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError("person-refinement checkpoint SHA mismatch")
    if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
        raise RuntimeError("required /usr/bin/python3 CUDA environment unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, registration = checkpoint["config"], checkpoint["registration"]
    design = config["person_design"]
    device = torch.device("cuda")
    model = build_model(
        radar_channels=int(checkpoint["radar_channels"]),
        hidden_channels=int(checkpoint["object_hidden_channels"]),
        head_depth=int(checkpoint["object_head_depth"]),
        person_hidden=int(design["hidden_channels"]),
        group_norm_groups=int(design["group_norm_groups"]),
        range_bins=int(design["range_bins"]), device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = experiment / "dataset"
    rows = [row for row in read_manifest(dataset / "manifest.csv") if row.get("split") == "val"]
    if len(rows) != 3345 or len({row["sample_id"] for row in rows}) != 3345:
        raise RuntimeError("validation count/uniqueness drift")
    if any("canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"] for row in rows):
        raise RuntimeError("locked test reference in inference rows")
    output = experiment / "predictions" / args.tag
    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    write_json_x(output / "inference_started.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "validation_frames": len(rows), "score_floor": SCORE_FLOOR,
    })
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    detections_path = output / "detections.csv"
    segmentation_rows: list[dict[str, Any]] = []
    prediction_count = 0
    vehicle_rows = 0
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                fused, output_hw, _original = load_fused_tensor(
                    row, dataset, tuple(config["registered_input_size"]), device,
                )
                outputs = model(fused, feature_drop_fraction=0.0)
                seg_logits = F.interpolate(outputs["out"], size=output_hw, mode="bilinear", align_corners=False)
                labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                seg_rel = Path("segmentation") / f"{row['sample_id']}.png"
                seg_path = output / seg_rel
                if not cv2.imwrite(str(seg_path), labels):
                    raise RuntimeError(f"segmentation write failed: {row['sample_id']}")
                segmentation_rows.append({
                    "sample_id": row["sample_id"], "prediction_path": str(seg_rel),
                    "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(seg_path),
                })
                matrix = parse_matrix(row["camera_matrix_json"])
                predictions: list[dict[str, float]] = []
                if matrix is not None:
                    scale_x = config["registered_input_size"][0] / float(row["camera_width"])
                    scale_y = config["registered_input_size"][1] / float(row["camera_height"])
                    intrinsic = np.asarray([
                        [float(row["camera_fx"]) * scale_x, 0.0, float(row["camera_cx"]) * scale_x],
                        [0.0, float(row["camera_fy"]) * scale_y, float(row["camera_cy"]) * scale_y],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    predictions = decode_all(
                        outputs, camera_matrix=matrix, intrinsic_model=intrinsic,
                        range_edges=registration["range_bins"]["edges_m"],
                        offset_caps=design["projected_offset_cap_grid_xy"],
                        score_threshold=SCORE_FLOOR,
                        model_size=config["registered_input_size"],
                    )
                    camera = np.asarray(matrix)[:3, 3]
                    predictions = [value for value in predictions if math.hypot(
                        float(value["world_x"]) - camera[0], float(value["world_y"]) - camera[1]
                    ) <= RANGE_M]
                for prediction_index, prediction in enumerate(predictions):
                    if not all(math.isfinite(float(prediction[key])) for key in (
                        "score", "world_x", "world_y", "world_z", "size_x", "size_y", "size_z"
                    )):
                        raise RuntimeError(f"nonfinite prediction: {row['sample_id']}")
                    writer.writerow({
                        "sample_id": row["sample_id"], "frame_id": row["frame_id"],
                        "prediction_index": prediction_index,
                        **{field: prediction.get(field, "") for field in FIELDS
                           if field not in {"sample_id", "frame_id", "prediction_index"}},
                    })
                    vehicle_rows += int(prediction["class_name"] == "vehicle")
                prediction_count += len(predictions)
                if index % 500 == 0:
                    print(f"[person refinement inference {args.tag}] {index}/{len(rows)}", flush=True)
    seg_manifest = output / "segmentation_manifest.csv"
    with seg_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height", "sha256"))
        writer.writeheader()
        writer.writerows(segmentation_rows)
    result = {
        "schema": "route_b_v3_1_person_refinement_inference_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]), "validation_frames": len(rows),
        "inference_pass_count": 1, "one_pass_supplies_thresholds": [0.20, 0.02],
        "score_floor": SCORE_FLOOR, "topk_per_class": TOPK_PER_CLASS,
        "input_size": list(config["registered_input_size"]),
        "native_object_grid": [108, 192],
        "native_local_maximum_kernel": 3, "world_nms": "none", "q": 0, "ae": False,
        "transported_bundle": ["low", "high"], "raw_tail_side_channels": [],
        "external_detection_fields": list(FIELDS), "detection_predictions": prediction_count,
        "vehicle_detection_rows": vehicle_rows,
        "detections_sha256": sha256(detections_path),
        "segmentation_manifest_sha256": sha256(seg_manifest),
        "prediction_set_sha256": hashlib.sha256(
            (sha256(detections_path) + sha256(seg_manifest)).encode("ascii")
        ).hexdigest(),
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    write_json_x(output / "inference_manifest.json", result)
    (output / "INFERENCE_COMPLETE").write_text("ONE_PERSON_REFINEMENT_INFERENCE_PASS_COMPLETE\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import CONFIG_PATH, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from data import InferenceDataset
from decode import (EXTERNAL_FIELDS, camera_matrix_from_row, decode_geometry, external_record,
                    intrinsic_from_row)
from model import build_model, freeze_bn_running_state


def latency(model: torch.nn.Module, value: torch.Tensor) -> dict[str, float]:
    def measure(function) -> tuple[float, float]:
        for _ in range(50): function()
        torch.cuda.synchronize()
        values = []
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); function(); end.record(); torch.cuda.synchronize()
            values.append(start.elapsed_time(end))
        return float(np.median(values)), float(np.percentile(values, 95))
    with torch.inference_mode():
        bundle = model.encode_front(value)
        front = measure(lambda: model.encode_front(value))
        tail = measure(lambda: model.decode_tail(bundle, dense=False))
        end_to_end = measure(lambda: model(value, dense=False))
    return {
        "front_median_ms": front[0], "front_p95_ms": front[1],
        "tail_median_ms": tail[0], "tail_p95_ms": tail[1],
        "end_to_end_median_ms": end_to_end[0], "end_to_end_p95_ms": end_to_end[1],
        "warmup_iterations": 50, "measured_iterations": 200,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int, choices=(10, 20, 30, 40))
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    if not (experiment / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("fixed validation begins only after complete 40-epoch training")
    config = load_json(CONFIG_PATH)
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"]
    if len(rows) != 3345 or len({row["sample_id"] for row in rows}) != 3345:
        raise RuntimeError("validation population drift")
    checkpoint_path = experiment / f"checkpoints/epoch_{args.epoch:03d}.pt"
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != args.epoch or checkpoint["resolved_config_sha256"] != sha256(CONFIG_PATH):
        raise RuntimeError("checkpoint provenance drift")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    model, _ = build_model(Path(config["pretrained"]["path"]), device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval(); freeze_bn_running_state(model)
    dataset = InferenceDataset(dataset_root, rows)
    output = experiment / f"predictions/epoch_{args.epoch:03d}"
    output.mkdir(parents=True, exist_ok=False); (output / "segmentation").mkdir()
    write_json_x(output / "INFERENCE_STARTED.json", {
        "created_utc": utc_now(), "epoch": args.epoch, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "score_floor": 0.02,
        "depth_argument": False, "dense_readout": False,
    })
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"
    segmentation_rows = []; detection_count = 0
    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXTERNAL_FIELDS); writer.writeheader()
        with torch.inference_mode():
            for index in range(len(dataset)):
                fused, row = dataset[index]
                fused = fused.unsqueeze(0).to(device, non_blocking=True)
                outputs = model(fused, dense=False)
                finite = torch.isfinite(outputs["out"]).all()
                for branch in outputs["objects"].values():
                    for value in branch.values(): finite &= torch.isfinite(value).all()
                if not finite.item(): raise RuntimeError(f"nonfinite inference {row['sample_id']}")
                source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                labels = F.interpolate(outputs["out"], size=source_hw, mode="bilinear", align_corners=False)
                labels = labels.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                relative = Path("segmentation") / f"{row['sample_id']}.png"
                path = output / relative
                if not cv2.imwrite(str(path), labels): raise RuntimeError(f"segmentation write {path}")
                segmentation_rows.append({"sample_id": row["sample_id"], "prediction_path": str(relative),
                                          "width": labels.shape[1], "height": labels.shape[0], "sha256": sha256(path)})
                records = decode_geometry(outputs, model.depth_anchors, model.depth_delta,
                                          camera_matrix_from_row(row), intrinsic_from_row(row), 0.02, 120)
                matrix = camera_matrix_from_row(row); camera_xy = matrix[:2, 3]
                records = [record for record in records if math.hypot(
                    float(record["world_x"]) - camera_xy[0], float(record["world_y"]) - camera_xy[1]) <= 40.0]
                for prediction_index, record in enumerate(records):
                    writer.writerow(external_record(record, row["sample_id"], row["frame_id"], prediction_index))
                detection_count += len(records)
                if (index + 1) % 500 == 0:
                    print(f"[inference epoch {args.epoch}] {index + 1}/{len(dataset)}", flush=True)
    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        fields = ("sample_id", "prediction_path", "width", "height", "sha256")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(segmentation_rows)
    real_input, _row = dataset[0]; real_input = real_input.unsqueeze(0).to(device)
    latency_report = latency(model, real_input)
    with torch.inference_mode(): bundle = model.encode_front(real_input)
    raw_bytes = sum(value.numel() * value.element_size() for value in bundle.values())
    serialized_bytes = len(b"".join(value.cpu().contiguous().numpy().tobytes() for value in bundle.values()))
    manifest = {
        "schema": "route_b_v3_1_depth_aware_lraspp_inference_v1", "created_utc": utc_now(),
        "epoch": args.epoch, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "validation_frames": len(rows), "inference_pass_count": 1, "score_floor": 0.02,
        "derived_threshold": 0.20, "topk_per_class": 120, "native_object_grid": [108, 192],
        "native_local_maximum_kernel": 3, "world_nms": "none", "compression": "identity_disabled",
        "dense_readout": "disabled", "depth_paths_opened": 0, "depth_labels_used": False,
        "transported_bundle": ["low", "high"],
        "transport_shapes": {name: list(value.shape) for name, value in bundle.items()},
        "transport_dtypes": {name: str(value.dtype) for name, value in bundle.items()},
        "raw_transport_bytes": raw_bytes, "identity_serialized_transport_bytes": serialized_bytes,
        "detection_predictions": detection_count, "detections_sha256": sha256(detections_path),
        "segmentation_manifest_sha256": sha256(segmentation_manifest),
        "prediction_set_sha256": hashlib.sha256((sha256(detections_path) + sha256(segmentation_manifest)).encode()).hexdigest(),
        "latency": latency_report, "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "external_detection_fields": list(EXTERNAL_FIELDS),
    }
    write_json_x(output / "inference_manifest.json", manifest)
    write_text_x(output / "INFERENCE_COMPLETE", "ONE_FLOOR_0_02_INFERENCE_PASS_COMPLETE\n")
    print(json.dumps({"epoch": args.epoch, "predictions": detection_count,
                      "wall_seconds": manifest["wall_seconds"], "latency": latency_report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

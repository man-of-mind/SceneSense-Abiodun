#!/usr/bin/env python3
"""Frozen validation evaluator for epochs 4, 8 and 12 only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, HERE.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    class_iou_from_confusion,
    read_manifest,
    update_confusion,
)
from pole_lraspp_multimodal_fusion.evaluate_fusion import yaw_error_deg  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    greedy_match_predictions,
    load_object_boxes,
    parse_matrix,
    valid_localization_objects,
)

from dataset_v1 import RouteBFasterRCNNDataset, detection_collate  # noqa: E402
from model_v1 import build_model, records_from_detections  # noqa: E402
from train_v1 import sha256, write_json_create  # noqa: E402


THRESHOLDS = (0.20, 0.02)
CLASSES = ("vehicle", "person")
FIXED_CONTRACT = {
    "validation_only": True,
    "range_gate_m": 40.0,
    "minimum_gt_area_px": 12.0,
    "primary_score": 0.20,
    "diagnostic_score": 0.02,
    "world_match_m": 3.0,
    "class_aware_world_matching": True,
    "box_nms": "torchvision detector-native class-aware NMS",
    "box_nms_iou": 0.5,
    "detections_per_image": 100,
    "score_floor": 0.02,
    "threshold_grid": False,
}


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def gt_box_input(obj: Dict[str, float], sx: float, sy: float) -> Tuple[float, float, float, float]:
    cx, cy = float(obj["center_x"]) * sx, float(obj["center_y"]) * sy
    bw, bh = float(obj["bbox_w"]) * sx, float(obj["bbox_h"]) * sy
    return (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)


def greedy_box_match(preds, gts, sx: float, sy: float, minimum_iou: float = 0.5):
    candidates = []
    for pred_index, pred in enumerate(preds):
        pbox = (pred["bbox_x0"], pred["bbox_y0"], pred["bbox_x1"], pred["bbox_y1"])
        for gt_index, gt in enumerate(gts):
            value = box_iou(pbox, gt_box_input(gt, sx, sy))
            if value >= minimum_iou:
                candidates.append((-value, pred_index, gt_index, value))
    candidates.sort()
    used_pred, used_gt, matches = set(), set(), []
    for _neg, pred_index, gt_index, value in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, value))
    return matches


def average_precision(records: List[Tuple[float, int]], gt_count: int) -> float:
    if gt_count <= 0 or not records:
        return 0.0
    records = sorted(records, key=lambda value: -value[0])
    tp = np.cumsum([value[1] for value in records], dtype=np.float64)
    fp = np.cumsum([1 - value[1] for value in records], dtype=np.float64)
    recall = tp / float(gt_count)
    precision = tp / np.maximum(1.0, tp + fp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def metric_bucket():
    return {
        "tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0,
        "xy": [], "dimension": [], "yaw": [], "world_matched_box_iou": [],
    }


def summarize_bucket(bucket: Dict, frames: int) -> Dict[str, float]:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "xy_mae_m": mean(bucket["xy"]),
        "dimension_mae_m": mean(bucket["dimension"]),
        "yaw_mae_deg": mean(bucket["yaw"]),
        "world_matched_box_iou_mean": mean(bucket["world_matched_box_iou"]),
        "duplicate_fp": bucket["duplicate_fp"],
        "duplicate_fp_per_frame": bucket["duplicate_fp"] / max(1, frames),
        "fp_per_frame": fp / max(1, frames),
    }


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False, input_size=tuple(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return checkpoint, model


def evaluate_one(path: Path, loader, rows, object_rows, output_dir: Path, device: torch.device) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint, model = load_checkpoint_model(path, device)
    width, height = map(int, checkpoint["input_size"])
    accum = {threshold: {name: metric_bucket() for name in CLASSES} for threshold in THRESHOLDS}
    box_accum = {
        threshold: {
            name: {"gt": 0, "pred": 0, "tp50": 0, "matched_ious": [], "ap_records": []}
            for name in CLASSES
        }
        for threshold in THRESHOLDS
    }
    confusion = np.zeros((3, 3), dtype=np.float64)
    csv_rows = {threshold: [] for threshold in THRESHOLDS}
    processed = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for rgb, radar, _targets, metadata in loader:
            rgb_device = [value.to(device, non_blocking=True) for value in rgb]
            radar_device = [value.to(device, non_blocking=True) for value in radar]
            output = model(rgb_device, radar_device)
            segmentation = output["segmentation"].argmax(dim=1).cpu().numpy().astype(np.int64)
            for batch_index, meta in enumerate(metadata):
                row = rows[int(meta["index"])]
                mask = _targets[batch_index]["segmentation"].numpy().astype(np.int64)
                update_confusion(confusion, segmentation[batch_index], mask, 3)
                matrix = parse_matrix(meta["camera_matrix_json"])
                if matrix is None:
                    all_predictions = []
                    camera_center = np.zeros(3)
                else:
                    all_predictions = records_from_detections(
                        output["detections"][batch_index], matrix, meta["camera_yaw_deg"]
                    )
                    camera_center = np.asarray(matrix)[:3, 3]
                    all_predictions = [
                        pred for pred in all_predictions
                        if math.hypot(pred["world_x"] - camera_center[0], pred["world_y"] - camera_center[1]) <= 40.0
                    ]
                gt_objects = valid_localization_objects(
                    object_rows.get(row["sample_id"], []),
                    image_width=int(meta["original_width"]),
                    image_height=int(meta["original_height"]),
                    min_area_px=12.0,
                    object_class_names=CLASSES,
                    max_distance_m=40.0,
                )
                sx = width / float(meta["original_width"])
                sy = height / float(meta["original_height"])
                for threshold in THRESHOLDS:
                    predictions = [pred for pred in all_predictions if pred["score"] >= threshold]
                    matches = greedy_match_predictions(predictions, gt_objects, max_distance_m=3.0, class_aware=True)
                    matched_pred = {pred_index for pred_index, _gt_index, _distance in matches}
                    matched_gt = {gt_index for _pred_index, gt_index, _distance in matches}
                    for pred_index, gt_index, distance in matches:
                        pred, gt = predictions[pred_index], gt_objects[gt_index]
                        bucket = accum[threshold][gt["class_name"]]
                        bucket["tp"] += 1
                        bucket["xy"].append(float(distance))
                        bucket["dimension"].append(float(np.mean(np.abs(
                            np.array([pred["size_x"], pred["size_y"], pred["size_z"]])
                            - np.array([gt["size_x"], gt["size_y"], gt["size_z"]])
                        ))))
                        bucket["yaw"].append(float(yaw_error_deg(pred, gt)))
                        pbox = (pred["bbox_x0"], pred["bbox_y0"], pred["bbox_x1"], pred["bbox_y1"])
                        bucket["world_matched_box_iou"].append(box_iou(pbox, gt_box_input(gt, sx, sy)))
                        csv_rows[threshold].append({
                            "sample_id": row["sample_id"], "status": "tp", "class_name": gt["class_name"],
                            "score": pred["score"], "xy_error_m": distance,
                        })
                    for pred_index, pred in enumerate(predictions):
                        if pred_index in matched_pred:
                            continue
                        bucket = accum[threshold][pred["class_name"]]
                        bucket["fp"] += 1
                        duplicate = any(
                            other_index != pred_index
                            and other["class_name"] == pred["class_name"]
                            and other["score"] > pred["score"]
                            and math.hypot(other["world_x"] - pred["world_x"], other["world_y"] - pred["world_y"]) <= 3.0
                            for other_index, other in enumerate(predictions)
                        )
                        bucket["duplicate_fp"] += int(duplicate)
                        csv_rows[threshold].append({
                            "sample_id": row["sample_id"], "status": "fp", "class_name": pred["class_name"],
                            "score": pred["score"], "xy_error_m": "",
                        })
                    for gt_index, gt in enumerate(gt_objects):
                        if gt_index not in matched_gt:
                            accum[threshold][gt["class_name"]]["fn"] += 1
                            csv_rows[threshold].append({
                                "sample_id": row["sample_id"], "status": "fn", "class_name": gt["class_name"],
                                "score": "", "xy_error_m": "",
                            })

                    for class_name in CLASSES:
                        class_predictions = [pred for pred in predictions if pred["class_name"] == class_name]
                        class_gt = [gt for gt in gt_objects if gt["class_name"] == class_name]
                        box_matches = greedy_box_match(class_predictions, class_gt, sx, sy, minimum_iou=0.5)
                        matched_box_pred = {pred_index for pred_index, _gt_index, _value in box_matches}
                        box_bucket = box_accum[threshold][class_name]
                        box_bucket["gt"] += len(class_gt)
                        box_bucket["pred"] += len(class_predictions)
                        box_bucket["tp50"] += len(box_matches)
                        box_bucket["matched_ious"].extend(value for _pred, _gt, value in box_matches)
                        box_bucket["ap_records"].extend(
                            (float(pred["score"]), int(pred_index in matched_box_pred))
                            for pred_index, pred in enumerate(class_predictions)
                        )
                processed += 1
            if processed % 800 < len(rgb):
                print(f"[eval {path.stem}] {processed}/{len(rows)}", flush=True)

    miou, ious, pixel_accuracy = class_iou_from_confusion(confusion)
    result = {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "epoch": int(checkpoint["epoch"]),
        "frames": len(rows),
        "fixed_contract": FIXED_CONTRACT,
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "segmentation": {
            "background_iou": float(ious[0]),
            "vehicle_iou": float(ious[1]),
            "person_box_mask_iou": float(ious[2]),
            "miou": float(miou),
            "pixel_accuracy": float(pixel_accuracy),
            "person_label": "projected-box-mask IoU; not silhouette IoU",
        },
        "by_threshold": {},
    }
    for threshold in THRESHOLDS:
        tag = f"{threshold:.2f}"
        classes = {name: summarize_bucket(accum[threshold][name], len(rows)) for name in CLASSES}
        overall = metric_bucket()
        for name in CLASSES:
            for key in ("tp", "fp", "fn", "duplicate_fp"):
                overall[key] += accum[threshold][name][key]
            for key in ("xy", "dimension", "yaw", "world_matched_box_iou"):
                overall[key].extend(accum[threshold][name][key])
        boxes = {}
        for name in CLASSES:
            bucket = box_accum[threshold][name]
            boxes[name] = {
                "gt": bucket["gt"], "predictions": bucket["pred"], "tp_iou50": bucket["tp50"],
                "precision_iou50": bucket["tp50"] / max(1, bucket["pred"]),
                "recall_iou50": bucket["tp50"] / max(1, bucket["gt"]),
                "mean_iou_of_iou50_matches": float(np.mean(bucket["matched_ious"])) if bucket["matched_ious"] else float("nan"),
                "ap50_score_floor_0.02": average_precision(bucket["ap_records"], bucket["gt"]),
            }
        result["by_threshold"][tag] = {"classes": classes, "overall": summarize_bucket(overall, len(rows)), "box_diagnostics": boxes}
        with (output_dir / f"detections_s{int(threshold * 100):03d}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "status", "class_name", "score", "xy_error_m"])
            writer.writeheader()
            writer.writerows(csv_rows[threshold])
    write_json_create(output_dir / "metrics.json", result)
    del model, checkpoint
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs=3, required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    checkpoints = [path.resolve(strict=True) for path in args.checkpoints]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    epochs = [int(payload["epoch"]) for payload in payloads]
    if epochs != [4, 8, 12]:
        raise SystemExit(f"authorized evaluation epochs are exactly [4, 8, 12], got {epochs}")
    input_sizes = {tuple(payload["input_size"]) for payload in payloads}
    del payloads
    if len(input_sizes) != 1:
        raise SystemExit(f"checkpoint input-size mismatch: {input_sizes}")
    experiment_dir = args.experiment_dir.resolve()
    dataset_dir = experiment_dir / "dataset"
    rows_all = read_manifest(dataset_dir / "manifest.csv")
    if any(row.get("split") == "test" for row in rows_all):
        raise SystemExit("locked test present; refusing evaluation")
    rows = [row for row in rows_all if row.get("split") == "val"]
    if len(rows) != 3588:
        raise SystemExit(f"expected 3588 validation rows, got {len(rows)}")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    width, height = next(iter(input_sizes))
    dataset = RouteBFasterRCNNDataset(dataset_dir, rows, object_rows, (width, height), training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=2,
        collate_fn=detection_collate,
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for checkpoint in checkpoints:
        output_dir = root / checkpoint.stem
        print(f"[eval] {checkpoint} -> {output_dir}", flush=True)
        results[checkpoint.stem] = evaluate_one(checkpoint, loader, rows, object_rows, output_dir, device)
    write_json_create(root / "all_epochs_metrics.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


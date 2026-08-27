#!/usr/bin/env python3
"""Phase D: fixed validation of recovery-v2 epochs 4/8/12 on the canonical 3,588-frame denominator.

ONE inference pass per checkpoint at score floor 0.02. The three operating points were registered
before this run (postprocessing_qualification_v1_20260827/offline_score_v1.py ARMS). The grid is
NOT extended after seeing results. The locked test split is absent and is never opened.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import common_v2 as C
from pole_lraspp_multimodal_fusion.common import class_iou_from_confusion, update_confusion
from pole_lraspp_multimodal_fusion.evaluate_fusion import yaw_error_deg
from pole_lraspp_multimodal_fusion.object_targets import (
    greedy_match_predictions, load_object_boxes, parse_matrix, valid_localization_objects,
)
from dataset_v1 import RouteBFasterRCNNDataset, detection_collate
from model_v1 import records_from_detections
from model_patch_v2 import load_recovery_checkpoint
from split_runtime_adapter_v1 import reconstruct_image_list

WORLD_M, RANGE_M, MIN_AREA_PX, SCORE_FLOOR = 3.0, 40.0, 12.0, 0.02
ARMS = {
    "RAW_PRIMARY": {"thresholds": {"vehicle": 0.20, "person": 0.20}, "world_nms": None},
    "RECALL_PRESERVING_WORLD_NMS": {"thresholds": {"vehicle": 0.20, "person": 0.20},
                                    "world_nms": {"vehicle": 2.0, "person": 1.0}},
    "CALIBRATED_WORLD_NMS": {"thresholds": {"vehicle": 0.89, "person": 0.95},
                             "world_nms": {"vehicle": 2.0, "person": 1.0}},
    "DIAGNOSTIC_CEILING_S002": {"thresholds": {"vehicle": 0.02, "person": 0.02}, "world_nms": None},
}
GATES = {
    "vehicle": {"precision": (0.80, "min"), "recall": (0.85, "min"), "xy_mae_m": (1.0, "max")},
    "person": {"precision": (0.80, "min"), "recall": (0.80, "min"), "xy_mae_m": (1.2, "max")},
}
SEG_GATES = {"vehicle_iou": (0.85, "min"), "person_box_mask_iou": (0.50, "min"), "miou": (0.80, "min")}
COLLISION_FIELDS = ("collision", "impact", "ttc", "hazard_window", "collision_window")


def box_iou(left, right) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(1e-9, la + ra - inter)


def bucket():
    return {"tp": 0, "fp": 0, "fn": 0, "duplicate_fp": 0, "xy": [], "dim": [], "yaw": []}


def summarize(item: Dict, frames: int) -> Dict[str, float]:
    tp, fp, fn = item["tp"], item["fp"], item["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2.0 * precision * recall / max(1e-9, precision + recall),
            "xy_mae_m": mean(item["xy"]), "dimension_mae_m": mean(item["dim"]),
            "yaw_mae_deg": mean(item["yaw"]), "duplicate_fp": item["duplicate_fp"],
            "duplicate_fp_per_frame": item["duplicate_fp"] / max(1, frames),
            "fp_per_frame": fp / max(1, frames)}


def world_nms_keep(predictions: List[Dict], radii: Dict[str, float]) -> List[int]:
    order = sorted(range(len(predictions)), key=lambda i: -predictions[i]["score"])
    dead, keep = set(), []
    for i in order:
        if i in dead:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in dead or predictions[j]["class_name"] != predictions[i]["class_name"]:
                continue
            if math.hypot(predictions[j]["world_x"] - predictions[i]["world_x"],
                          predictions[j]["world_y"] - predictions[i]["world_y"]) <= radii[predictions[i]["class_name"]]:
                dead.add(j)
    return sorted(keep)


def score_arm(predictions_by_frame, gts_by_frame, frame_ids, thresholds, world_nms) -> Dict:
    accum = {name: bucket() for name in C.CLASSES}
    suppressed = {name: 0 for name in C.CLASSES}
    suppressed_would_be_tp = {name: 0 for name in C.CLASSES}
    for sample_id in frame_ids:
        candidates = [p for p in predictions_by_frame.get(sample_id, [])
                      if p["score"] >= thresholds[p["class_name"]]]
        gts = gts_by_frame.get(sample_id, [])
        kept = candidates
        if world_nms is not None and candidates:
            keep_idx = world_nms_keep(candidates, world_nms)
            keep_set = set(keep_idx)
            if len(keep_set) != len(candidates):
                pre_tp = {pi for pi, _gi, _d in greedy_match_predictions(
                    candidates, gts, max_distance_m=WORLD_M, class_aware=True)}
                for index, prediction in enumerate(candidates):
                    if index not in keep_set:
                        suppressed[prediction["class_name"]] += 1
                        suppressed_would_be_tp[prediction["class_name"]] += int(index in pre_tp)
            kept = [candidates[i] for i in keep_idx]
        matches = greedy_match_predictions(kept, gts, max_distance_m=WORLD_M, class_aware=True)
        matched_pred = {pi for pi, _gi, _d in matches}
        matched_gt = {gi for _pi, gi, _d in matches}
        for pred_index, gt_index, distance in matches:
            prediction, gt = kept[pred_index], gts[gt_index]
            item = accum[gt["class_name"]]
            item["tp"] += 1
            item["xy"].append(float(distance))
            item["dim"].append(float(np.mean(np.abs(
                np.array([prediction["size_x"], prediction["size_y"], prediction["size_z"]])
                - np.array([gt["size_x"], gt["size_y"], gt["size_z"]])))))
            item["yaw"].append(float(yaw_error_deg(prediction, gt)))
        for pred_index, prediction in enumerate(kept):
            if pred_index in matched_pred:
                continue
            item = accum[prediction["class_name"]]
            item["fp"] += 1
            item["duplicate_fp"] += int(any(
                other_index != pred_index and other["class_name"] == prediction["class_name"]
                and other["score"] > prediction["score"]
                and math.hypot(other["world_x"] - prediction["world_x"],
                               other["world_y"] - prediction["world_y"]) <= WORLD_M
                for other_index, other in enumerate(kept)))
        for gt_index, gt in enumerate(gts):
            if gt_index not in matched_gt:
                accum[gt["class_name"]]["fn"] += 1
    frames = len(frame_ids)
    classes = {name: summarize(accum[name], frames) for name in C.CLASSES}
    return {"classes": classes,
            "mean_class_f1": float(np.mean([classes[n]["f1"] for n in C.CLASSES])),
            "suppressed": suppressed, "suppressed_would_be_tp": suppressed_would_be_tp}


def collision_window_support(rows) -> Dict[str, object]:
    columns = sorted(rows[0].keys()) if rows else []
    hits = [c for c in columns for f in COLLISION_FIELDS if f in c.lower()]
    return {"supported": bool(hits), "matching_manifest_columns": sorted(set(hits)),
            "status": "EVALUATED" if hits else "UNSUPPORTED",
            "note": "dataset/manifest.csv carries no collision/impact/TTC/hazard-window field, so the "
                    "collision-window-excluded stratum cannot be formed without new collection. "
                    "Reported as unevaluable rather than approximated; the criterion is UNVERIFIED, "
                    "not passed." if not hits else ""}


def evaluate_one(path: Path, loader, rows, object_rows, device, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload, model, frozen_box_head = load_recovery_checkpoint(path, device)
    width, height = map(int, payload["input_size"])
    roi_heads = model.detector.roi_heads
    predictions_by_frame, gts_by_frame, frame_ids = defaultdict(list), {}, []
    confusion = np.zeros((3, 3), dtype=np.float64)
    proposal_best_iou = {name: [] for name in C.CLASSES}
    proposal_counts = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    processed = 0
    with torch.inference_mode():
        for rgb, radar, targets, metadata in loader:
            bundle = model.encode_front([v.to(device, non_blocking=True) for v in rgb],
                                        [v.to(device, non_blocking=True) for v in radar])
            features, radar_features = bundle["rgb_fpn"], bundle["radar_fpn"]
            image_sizes = list(bundle["image_sizes"])
            image_list = reconstruct_image_list(tuple(bundle["image_batch_shape"]), image_sizes,
                                                features["0"].device, dtype=features["0"].dtype)
            proposals, _ = model.detector.rpn(image_list, features, None)
            detections, _ = roi_heads(features, proposals, image_sizes, None)
            seg_logits = model.segmentation_decoder(
                features, (bundle["image_batch_shape"][-2], bundle["image_batch_shape"][-1]))
            boxes = [item["boxes"] for item in detections]
            if sum(int(b.shape[0]) for b in boxes):
                # FROZEN box_head -> radar ROI localization path preserved bit-exactly
                visual = frozen_box_head(roi_heads.box_roi_pool(features, boxes, image_sizes))
                radar_embed = model.radar_roi_embed(model.radar_roi_pool(radar_features, boxes, image_sizes))
                fields = model.roi_localization_head(torch.cat([visual, radar_embed], dim=1))
            else:
                fields = features["0"].new_zeros((0, 10))
            offset = 0
            for item in detections:
                count = int(item["boxes"].shape[0])
                chunk = fields[offset:offset + count]
                offset += count
                item["local_xyz"] = chunk[:, 0:3]
                item["dimensions"] = torch.nn.functional.softplus(chunk[:, 3:6])
                item["local_yaw_sincos"] = torch.nn.functional.normalize(chunk[:, 6:8], dim=1)
                item["parked_score"] = torch.sigmoid(chunk[:, 8])
                item["radar_support_score"] = torch.sigmoid(chunk[:, 9])
            segmentation = seg_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            for batch_index, meta in enumerate(metadata):
                row = rows[int(meta["index"])]
                sample_id = str(row["sample_id"])
                frame_ids.append(sample_id)
                update_confusion(confusion, segmentation[batch_index],
                                 targets[batch_index]["segmentation"].numpy().astype(np.int64), 3)
                matrix = parse_matrix(meta["camera_matrix_json"])
                gt_objects = valid_localization_objects(
                    object_rows.get(sample_id, []),
                    image_width=int(meta["original_width"]), image_height=int(meta["original_height"]),
                    min_area_px=MIN_AREA_PX, object_class_names=C.CLASSES, max_distance_m=RANGE_M)
                gts_by_frame[sample_id] = gt_objects
                if matrix is None:
                    continue
                records = records_from_detections(detections[batch_index], matrix, meta["camera_yaw_deg"])
                centre = np.asarray(matrix)[:3, 3]
                predictions_by_frame[sample_id] = [
                    record for record in records
                    if math.hypot(record["world_x"] - centre[0], record["world_y"] - centre[1]) <= RANGE_M]
                # class-agnostic RPN proposal recall (diagnostic evidence, not a hard ceiling for
                # the 3 m world metric)
                proposal = proposals[batch_index].detach().cpu().numpy()
                proposal_counts.append(int(proposal.shape[0]))
                sx = width / float(meta["original_width"])
                sy = height / float(meta["original_height"])
                for gt in gt_objects:
                    cx, cy = float(gt["center_x"]) * sx, float(gt["center_y"]) * sy
                    bw, bh = float(gt["bbox_w"]) * sx, float(gt["bbox_h"]) * sy
                    target_box = (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)
                    best = 0.0
                    if proposal.shape[0]:
                        ix0 = np.maximum(proposal[:, 0], target_box[0])
                        iy0 = np.maximum(proposal[:, 1], target_box[1])
                        ix1 = np.minimum(proposal[:, 2], target_box[2])
                        iy1 = np.minimum(proposal[:, 3], target_box[3])
                        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
                        area_p = np.clip(proposal[:, 2] - proposal[:, 0], 0, None) * np.clip(proposal[:, 3] - proposal[:, 1], 0, None)
                        area_g = max(0.0, bw) * max(0.0, bh)
                        best = float(np.max(inter / np.maximum(1e-9, area_p + area_g - inter)))
                    proposal_best_iou[gt["class_name"]].append(best)
                processed += 1
            if processed % 800 < len(rgb):
                print(f"[eval {path.stem}] {processed}/{len(rows)}", flush=True)

    miou, ious, pixel_accuracy = class_iou_from_confusion(confusion)
    arms = {name: score_arm(predictions_by_frame, gts_by_frame, frame_ids,
                            spec["thresholds"], spec["world_nms"]) for name, spec in ARMS.items()}
    proposal_recall = {
        name: {f"iou{threshold}": float(np.mean(np.asarray(values) >= threshold)) if values else float("nan")
               for threshold in (0.3, 0.5, 0.7)}
        for name, values in proposal_best_iou.items()}
    for name in C.CLASSES:
        proposal_recall[name]["eligible_gt"] = len(proposal_best_iou[name])
    result = {
        "checkpoint": str(path.resolve()), "checkpoint_sha256": C.sha256(path),
        "epoch": int(payload["epoch"]), "frames": len(rows),
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "predictions_persisted": int(sum(len(v) for v in predictions_by_frame.values())),
        "eligible_gt": {n: sum(1 for g in sum(gts_by_frame.values(), []) if g["class_name"] == n) for n in C.CLASSES},
        "segmentation": {"background_iou": float(ious[0]), "vehicle_iou": float(ious[1]),
                         "person_box_mask_iou": float(ious[2]), "miou": float(miou),
                         "pixel_accuracy": float(pixel_accuracy),
                         "person_label": "projected-box-mask IoU; not silhouette IoU"},
        "rpn_proposal_recall_class_agnostic": proposal_recall,
        "proposals_per_frame_mean": float(np.mean(proposal_counts)) if proposal_counts else 0.0,
        "final_detection_recall_ceiling_score002": {
            n: arms["DIAGNOSTIC_CEILING_S002"]["classes"][n]["recall"] for n in C.CLASSES},
        "arms": arms,
    }
    result["gate_evaluation"] = {}
    for arm_name, arm in arms.items():
        checks, passed = {}, True
        for class_name, gates in GATES.items():
            for metric, (bound, direction) in gates.items():
                value = float(arm["classes"][class_name][metric])
                ok = (value >= bound) if direction == "min" else (value <= bound)
                if math.isnan(value):
                    ok = False
                checks[f"{class_name}_{metric}"] = {"value": value, "bound": bound,
                                                   "direction": direction, "pass": bool(ok)}
                passed = passed and ok
        for metric, (bound, direction) in SEG_GATES.items():
            value = float(result["segmentation"][metric])
            ok = value >= bound if direction == "min" else value <= bound
            checks[f"segmentation_{metric}"] = {"value": value, "bound": bound,
                                               "direction": direction, "pass": bool(ok)}
            passed = passed and ok
        result["gate_evaluation"][arm_name] = {"checks": checks, "all_gates_pass": bool(passed)}
    C.write_json_create(output_dir / "metrics.json", result)
    del model, frozen_box_head, payload
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=3, required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    checkpoints = [p.resolve(strict=True) for p in args.checkpoints]
    epochs = [int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) for p in checkpoints]
    if epochs != [4, 8, 12]:
        raise SystemExit(f"authorized evaluation epochs are exactly [4, 8, 12], got {epochs}")
    splits = C.load_split_rows()
    rows = splits["val"]
    object_rows = load_object_boxes(C.DATASET_DIR / "object_boxes.csv")
    dataset = RouteBFasterRCNNDataset(C.DATASET_DIR, rows, object_rows, (1024, 576), training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=2,
                        collate_fn=detection_collate)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for checkpoint in checkpoints:
        print(f"[eval] {checkpoint}", flush=True)
        results[checkpoint.stem] = evaluate_one(checkpoint, loader, rows, object_rows, device,
                                                root / checkpoint.stem)
    winners = [(name, arm) for name, result in results.items()
               for arm, gate in result["gate_evaluation"].items() if gate["all_gates_pass"]]
    terminal = ("FRCNN_BASE_VALIDATION_CANDIDATE_READY_FOR_LOCKED_TEST" if winners
                else "FRCNN_BASE_RECOVERY_FAILED_FINAL")
    summary = {
        "terminal": terminal, "passing_operating_points": winners,
        "registered_operating_points": ARMS, "gates": {"detection": GATES, "segmentation": SEG_GATES},
        "candidate_operating_points_evaluated": 3 * 3,
        "grid_extended_after_results": False,
        "collision_window_sensitivity": collision_window_support(rows),
        "locked_test": "absent and unopened",
        "results": results,
    }
    C.write_json_create(root / "all_epochs_metrics.json", summary)
    (root / "TERMINAL_VERDICT.txt").write_text(terminal + "\n", encoding="utf-8")
    print(f"TERMINAL {terminal}", flush=True)
    C.notify("Route B Faster R-CNN recovery v2 validation", terminal,
             root / "EVALUATION_COMPLETION_NOTIFICATION.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

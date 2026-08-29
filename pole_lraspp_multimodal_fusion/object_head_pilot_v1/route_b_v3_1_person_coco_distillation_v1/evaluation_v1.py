#!/usr/bin/env python3
"""Frozen evaluation extensions and exact v3.1 distillation decision gates."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
MATCH_RADIUS_M = 3.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _bin_label(value: float, edges: Sequence[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if float(left) <= value < float(right):
            return f"[{left:g},{right:g})"
    return f"[{edges[-2]:g},{edges[-1]:g})"


def _summarize(values: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label, bucket in values.items():
        eligible = int(bucket["eligible_gt"])
        matched = int(bucket["matched"])
        errors = list(bucket["xy_errors_m"])
        result[label] = {
            "eligible_gt": eligible, "tp": matched, "fn": eligible - matched,
            "recall": matched / max(1, eligible),
            "xy_mae_m": sum(errors) / len(errors) if errors else None,
        }
    return result


def person_slices(
    experiment: Path, prediction_root: Path, config: Dict[str, Any], *, threshold: float = 0.20,
) -> Dict[str, Any]:
    """Recall/XY slices using the exact frozen 3 m class-aware assignment."""
    manifest = _read_csv(experiment / "dataset/manifest.csv")
    frame_ids = [row["sample_id"] for row in manifest if row["split"] == "val"]
    metadata_rows = _read_csv(experiment / "contracts/v010/val/object_boxes.csv")
    predictions: Dict[str, list[Dict[str, Any]]] = {}
    for row in _read_csv(prediction_root / "detections.csv"):
        item = {"class_name": row["class_name"], "score": float(row["score"]),
                "world_x": float(row["world_x"]), "world_y": float(row["world_y"])}
        if not all(math.isfinite(float(item[key])) for key in ("score", "world_x", "world_y")):
            raise RuntimeError(f"nonfinite slice prediction: {row['sample_id']}")
        predictions.setdefault(row["sample_id"], []).append(item)
    gt: Dict[str, list[Dict[str, Any]]] = {}
    for row in metadata_rows:
        item = {"class_name": row["label"], "world_x": float(row["object_world_x"]),
                "world_y": float(row["object_world_y"]),
                "area_px": float(row["gt_bbox_area_px"]),
                "source_identity": row["source_identity"]}
        gt.setdefault(row["sample_id"], []).append(item)
    metadata = {row["source_identity"]: row for row in metadata_rows if row["label"] == "person"}
    bins = config["evaluation"]["person_breakdown_bins"]
    area_edges = [float(value) for value in bins["box_area_px"]]
    distance_edges = [float(value) for value in bins["distance_m"]]
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
        "box_area_px": { _bin_label(left, area_edges): {"eligible_gt": 0, "matched": 0, "xy_errors_m": []}
                         for left in area_edges[:-1]},
        "distance_m": { _bin_label(left, distance_edges): {"eligible_gt": 0, "matched": 0, "xy_errors_m": []}
                        for left in distance_edges[:-1]},
        "radar_support": {label: {"eligible_gt": 0, "matched": 0, "xy_errors_m": []}
                          for label in ("supported", "unsupported")},
    }
    metadata_misses = 0
    for sample_id in frame_ids:
        targets = list(gt.get(sample_id, []))
        frame_predictions = [item for item in predictions.get(sample_id, [])
                             if float(item["score"]) >= float(threshold)]
        candidates = []
        for pred_index, prediction in enumerate(frame_predictions):
            for gt_index, target in enumerate(targets):
                if prediction["class_name"] != target["class_name"]:
                    continue
                distance = math.hypot(float(prediction["world_x"])-float(target["world_x"]),
                                      float(prediction["world_y"])-float(target["world_y"]))
                if distance <= MATCH_RADIUS_M:
                    candidates.append((distance, pred_index, gt_index))
        used_predictions: set[int] = set(); used_gt: set[int] = set(); pred_to_gt: Dict[int, int] = {}
        for _distance, pred_index, gt_index in sorted(candidates):
            if pred_index in used_predictions or gt_index in used_gt:
                continue
            used_predictions.add(pred_index); used_gt.add(gt_index); pred_to_gt[pred_index] = gt_index
        gt_to_pred = {gt_index: pred_index for pred_index, gt_index in pred_to_gt.items()}
        for gt_index, target in enumerate(targets):
            if str(target["class_name"]) != "person":
                continue
            row = metadata.get(str(target["source_identity"]))
            if row is None:
                metadata_misses += 1
                continue
            area = float(target["area_px"])
            distance = float(row["gt_distance_m"])
            radar_label = "supported" if float(row.get("radar_support_points", "0") or 0) > 0 else "unsupported"
            labels = {
                "box_area_px": _bin_label(area, area_edges),
                "distance_m": _bin_label(distance, distance_edges),
                "radar_support": radar_label,
            }
            for dimension, label in labels.items():
                bucket = buckets[dimension][label]
                bucket["eligible_gt"] += 1
                if gt_index in used_gt:
                    bucket["matched"] += 1
                    prediction = frame_predictions[gt_to_pred[gt_index]]
                    bucket["xy_errors_m"].append(math.hypot(
                        float(prediction["world_x"]) - float(target["world_x"]),
                        float(prediction["world_y"]) - float(target["world_y"]),
                    ))
    return {
        "threshold": float(threshold), "matching": "fixed_class_aware_greedy_nearest_3m",
        "box_area_px": _summarize(buckets["box_area_px"]),
        "distance_m": _summarize(buckets["distance_m"]),
        "radar_support": _summarize(buckets["radar_support"]),
        "metadata_misses": metadata_misses,
    }


def baseline_deltas(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    names = (
        "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_recall_002",
        "vehicle_xy_mae_m", "vehicle_iou", "person_precision", "person_recall",
        "person_f1", "person_recall_002", "person_xy_mae_m", "person_box_mask_iou",
        "foreground_miou",
    )
    return {name: float(metrics[name]) - float(baseline[name]) for name in names}


def eligibility_gates(
    record: Mapping[str, Any], config: Mapping[str, Any], *, baseline_vehicle_count: int,
) -> Dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline_reference_epoch40"]
    limits = config["eligibility_limits"]
    vehicle = config["vehicle_non_regression"]
    vehicle_count = int(metric["vehicle_tp"]) + int(metric["vehicle_fp"])
    relative_count = abs(vehicle_count - int(baseline_vehicle_count)) / max(1, int(baseline_vehicle_count))
    return {
        "all_metrics_finite": bool(record["all_metrics_finite"]),
        "vehicle_f1_eligibility": metric["vehicle_f1"] >= baseline["vehicle_f1"] - limits["class_f1_below_baseline"],
        "person_f1_eligibility": metric["person_f1"] >= baseline["person_f1"] - limits["class_f1_below_baseline"],
        "vehicle_recall_eligibility": metric["vehicle_recall"] >= baseline["vehicle_recall"] - limits["class_recall_below_baseline"],
        "person_recall_eligibility": metric["person_recall"] >= baseline["person_recall"] - limits["class_recall_below_baseline"],
        "vehicle_xy_eligibility": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + limits["class_xy_mae_above_baseline_m"],
        "person_xy_eligibility": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + limits["class_xy_mae_above_baseline_m"],
        "vehicle_iou_eligibility": metric["vehicle_iou"] >= baseline["vehicle_iou"] - limits["class_iou_below_baseline"],
        "person_iou_eligibility": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - limits["class_iou_below_baseline"],
        "vehicle_f1_non_regression": metric["vehicle_f1"] >= baseline["vehicle_f1"] + vehicle["vehicle_f1_delta_min"],
        "vehicle_recall_non_regression": metric["vehicle_recall"] >= baseline["vehicle_recall"] + vehicle["vehicle_recall_delta_min"],
        "vehicle_xy_non_regression": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + vehicle["vehicle_xy_delta_max_m"],
        "vehicle_iou_non_regression": metric["vehicle_iou"] >= baseline["vehicle_iou"] + vehicle["vehicle_iou_delta_min"],
        "vehicle_detection_count_non_regression": relative_count <= vehicle["vehicle_detection_count_relative_tolerance"],
    }


def material_gain_gates(record: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    metric, baseline = record["metrics"], config["baseline_reference_epoch40"]
    design = config["material_gain"]
    a = design["A"]
    b = design["B"]
    gates_a = {
        "person_f1_delta": metric["person_f1"] - baseline["person_f1"] >= a["person_f1_delta_min"],
        "person_recall_delta": metric["person_recall"] - baseline["person_recall"] >= a["person_recall_delta_min"],
        "person_precision_delta": metric["person_precision"] - baseline["person_precision"] >= a["person_precision_delta_min"],
        "person_xy_improvement": baseline["person_xy_mae_m"] - metric["person_xy_mae_m"] >= a["person_xy_improvement_min_m"],
    }
    gates_b = {
        "person_f1_delta": metric["person_f1"] - baseline["person_f1"] >= b["person_f1_delta_min"],
        "person_xy_improvement": baseline["person_xy_mae_m"] - metric["person_xy_mae_m"] >= b["person_xy_improvement_min_m"],
        "person_iou_delta": metric["person_box_mask_iou"] - baseline["person_box_mask_iou"] >= b["person_iou_delta_min"],
    }
    return {"A": gates_a, "A_pass": all(gates_a.values()),
            "B": gates_b, "B_pass": all(gates_b.values()),
            "pass": all(gates_a.values()) or all(gates_b.values())}


def service_gates(record: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, bool]:
    metric, target = record["metrics"], config["service_targets"]
    return {
        "vehicle_precision": metric["vehicle_precision"] >= target["vehicle_precision_min"],
        "vehicle_recall": metric["vehicle_recall"] >= target["vehicle_recall_min"],
        "person_precision": metric["person_precision"] >= target["person_precision_min"],
        "person_recall": metric["person_recall"] >= target["person_recall_min"],
        "vehicle_xy_mae": metric["vehicle_xy_mae_m"] <= target["vehicle_xy_mae_max_m"],
        "person_xy_mae": metric["person_xy_mae_m"] <= target["person_xy_mae_max_m"],
        "vehicle_iou": metric["vehicle_iou"] >= target["vehicle_iou_min"],
        "person_box_mask_iou": metric["person_box_mask_iou"] >= target["person_box_mask_iou_min"],
        "foreground_miou": metric["foreground_miou"] >= target["foreground_miou_min"],
    }


def teacher_adoption_guard(record: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    metric, baseline = record["metrics"], config["baseline_reference_epoch40"]
    return (metric["person_precision"] < config["teacher_adoption_guard"]["person_precision_floor"]
            and metric["person_recall"] > baseline["person_recall"])


def catastrophic_gates(record: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, bool]:
    metric, baseline = record["metrics"], config["baseline_reference_epoch40"]
    limits = config["catastrophic_limits"]
    return {
        "vehicle_f1": metric["vehicle_f1"] >= baseline["vehicle_f1"] - limits["class_f1_below_baseline"],
        "person_f1": metric["person_f1"] >= baseline["person_f1"] - limits["class_f1_below_baseline"],
        "vehicle_xy": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + limits["class_xy_mae_above_baseline_m"],
        "person_xy": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + limits["class_xy_mae_above_baseline_m"],
        "vehicle_iou": metric["vehicle_iou"] >= baseline["vehicle_iou"] - limits["class_iou_below_baseline"],
        "person_iou": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - limits["class_iou_below_baseline"],
    }


def rank_key(record: Mapping[str, Any]) -> tuple[float, float, float, int, int]:
    """Lineage-fixed eligible-candidate order used by expanded native-grid training."""
    metric = record["metrics"]
    return (-float(metric["mean_class_f1"]), -float(metric["minimum_class_recall"]),
            float(metric["mean_xy_mae_m"]), int(record["vehicle_duplicate_fp"]), int(record["epoch"]))


def nondominated(records: Iterable[Mapping[str, Any]]) -> list[int]:
    rows = list(records)
    result = []
    for candidate in rows:
        c = candidate["metrics"]
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            o = other["metrics"]
            weak = (o["person_f1"] >= c["person_f1"] and o["person_recall"] >= c["person_recall"]
                    and o["person_xy_mae_m"] <= c["person_xy_mae_m"]
                    and o["person_box_mask_iou"] >= c["person_box_mask_iou"])
            strict = (o["person_f1"] > c["person_f1"] or o["person_recall"] > c["person_recall"]
                      or o["person_xy_mae_m"] < c["person_xy_mae_m"]
                      or o["person_box_mask_iou"] > c["person_box_mask_iou"])
            if weak and strict:
                dominated = True
                break
        if not dominated:
            result.append(int(candidate["epoch"]))
    return sorted(result)

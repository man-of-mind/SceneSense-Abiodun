#!/usr/bin/env python3
"""Pure policy functions for the authorized epoch-10 continuation."""

from __future__ import annotations

import math
from typing import Any, Mapping


METRIC_KEYS = (
    "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_recall_002",
    "vehicle_xy_mae_m", "person_precision", "person_recall", "person_f1",
    "person_recall_002", "person_xy_mae_m", "vehicle_iou",
    "person_box_mask_iou", "foreground_miou",
)


def all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def service_targets(record: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, bool]:
    metric = record["metrics"]
    target = contract["service_targets"]
    return {
        "vehicle_precision_ge_0_80": metric["vehicle_precision"] >= target["vehicle_precision_min"],
        "vehicle_recall_ge_0_85": metric["vehicle_recall"] >= target["vehicle_recall_min"],
        "person_precision_ge_0_80": metric["person_precision"] >= target["person_precision_min"],
        "person_recall_ge_0_80": metric["person_recall"] >= target["person_recall_min"],
        "vehicle_xy_mae_le_1_0m": metric["vehicle_xy_mae_m"] <= target["vehicle_xy_mae_max_m"],
        "person_xy_mae_le_1_2m": metric["person_xy_mae_m"] <= target["person_xy_mae_max_m"],
        "vehicle_iou_ge_0_85": metric["vehicle_iou"] >= target["vehicle_iou_min"],
        "person_box_mask_iou_ge_0_50": metric["person_box_mask_iou"] >= target["person_box_mask_iou_min"],
        "foreground_miou_ge_0_675": metric["foreground_miou"] >= target["foreground_miou_min"],
    }


def eligibility(
    record: Mapping[str, Any], baseline: Mapping[str, float], contract: Mapping[str, Any]
) -> dict[str, bool]:
    metric = record["metrics"]
    limit = contract["eligibility_limits"]
    return {
        "all_metrics_finite": bool(record.get("all_metrics_finite", all_finite(metric))),
        "vehicle_f1_ge_baseline_minus_0_01": metric["vehicle_f1"] >= baseline["vehicle_f1"] - limit["class_f1_below_baseline"],
        "person_f1_ge_baseline_minus_0_01": metric["person_f1"] >= baseline["person_f1"] - limit["class_f1_below_baseline"],
        "vehicle_recall_ge_baseline_minus_0_01": metric["vehicle_recall"] >= baseline["vehicle_recall"] - limit["class_recall_below_baseline"],
        "person_recall_ge_baseline_minus_0_01": metric["person_recall"] >= baseline["person_recall"] - limit["class_recall_below_baseline"],
        "vehicle_xy_le_baseline_plus_0_05m": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + limit["class_xy_mae_above_baseline_m"],
        "person_xy_le_baseline_plus_0_05m": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + limit["class_xy_mae_above_baseline_m"],
        "vehicle_iou_ge_baseline_minus_0_01": metric["vehicle_iou"] >= baseline["vehicle_iou"] - limit["class_iou_below_baseline"],
        "person_iou_ge_baseline_minus_0_01": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - limit["class_iou_below_baseline"],
    }


def catastrophic_regression(
    record: Mapping[str, Any], baseline: Mapping[str, float], contract: Mapping[str, Any]
) -> dict[str, bool]:
    """Return pass/fail guards; a false value is catastrophic at epoch 20/30."""
    metric = record["metrics"]
    limit = contract["catastrophic_limits"]
    return {
        "all_metrics_finite": bool(record.get("all_metrics_finite", all_finite(metric))),
        "vehicle_f1_ge_baseline_minus_0_03": metric["vehicle_f1"] >= baseline["vehicle_f1"] - limit["class_f1_below_baseline"],
        "person_f1_ge_baseline_minus_0_03": metric["person_f1"] >= baseline["person_f1"] - limit["class_f1_below_baseline"],
        "vehicle_xy_le_baseline_plus_0_15m": metric["vehicle_xy_mae_m"] <= baseline["vehicle_xy_mae_m"] + limit["class_xy_mae_above_baseline_m"],
        "person_xy_le_baseline_plus_0_15m": metric["person_xy_mae_m"] <= baseline["person_xy_mae_m"] + limit["class_xy_mae_above_baseline_m"],
        "vehicle_iou_ge_baseline_minus_0_02": metric["vehicle_iou"] >= baseline["vehicle_iou"] - limit["class_iou_below_baseline"],
        "person_iou_ge_baseline_minus_0_02": metric["person_box_mask_iou"] >= baseline["person_box_mask_iou"] - limit["class_iou_below_baseline"],
        "checkpoint_state_integrity": bool(record.get("checkpoint_state_integrity", True)),
    }


def rank_key(record: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[float, ...]:
    metric = record["metrics"]
    services = service_targets(record, contract)
    return (
        -sum(services.values()),
        -min(float(metric["vehicle_recall"]), float(metric["person_recall"])),
        -(float(metric["vehicle_f1"]) + float(metric["person_f1"])) / 2.0,
        (float(metric["vehicle_xy_mae_m"]) + float(metric["person_xy_mae_m"])) / 2.0,
        int(record["vehicle_duplicate_fp"]),
        int(record["selection_order"]),
    )


def decorate(record: dict[str, Any], baseline: Mapping[str, float], contract: Mapping[str, Any]) -> dict[str, Any]:
    metric = record["metrics"]
    metric["minimum_class_recall"] = min(metric["vehicle_recall"], metric["person_recall"])
    metric["mean_class_f1"] = (metric["vehicle_f1"] + metric["person_f1"]) / 2.0
    metric["mean_xy_mae_m"] = (metric["vehicle_xy_mae_m"] + metric["person_xy_mae_m"]) / 2.0
    record["service_targets"] = service_targets(record, contract)
    record["service_target_count"] = sum(record["service_targets"].values())
    record["eligibility_gates"] = eligibility(record, baseline, contract)
    record["eligible"] = all(record["eligibility_gates"].values())
    return record

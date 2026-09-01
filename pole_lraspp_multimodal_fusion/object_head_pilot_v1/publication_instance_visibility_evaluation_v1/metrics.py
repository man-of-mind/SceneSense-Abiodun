"""Registered metric families prepared for future publication predictions."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def visibility_range_views(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = protocol["visibility"]["thresholds"]
    bands = protocol["evaluation"]["strata"]["range_m"]
    return [
        {"visibility_minimum": float(threshold), "range_m": [float(lo), float(hi)]}
        for threshold in thresholds for lo, hi in bands
    ]


def greedy_world_xy_match(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    radius_m: float = 3.0,
) -> list[tuple[int, int, float]]:
    candidates = []
    for pi, prediction in enumerate(predictions):
        for gi, target in enumerate(targets):
            if prediction["class_name"] != target["class_name"]:
                continue
            distance = math.hypot(
                float(prediction["world_x"]) - float(target["world_x"]),
                float(prediction["world_y"]) - float(target["world_y"]),
            )
            if distance <= radius_m:
                candidates.append((distance, pi, gi))
    used_p, used_g, output = set(), set(), []
    for distance, pi, gi in sorted(candidates):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); output.append((pi, gi, distance))
    return output


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def iou_thresholds() -> list[float]:
    return [round(0.50 + 0.05 * index, 2) for index in range(10)]


def segmentation_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    ignore_value: int = 255,
) -> dict[str, float]:
    pred, truth = np.asarray(prediction), np.asarray(target)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("segmentation shape mismatch")
    valid = truth != ignore_value
    values = {}
    for class_name, label in (("vehicle", 1), ("person", 2)):
        intersection = int(np.count_nonzero(valid & (pred == label) & (truth == label)))
        union = int(np.count_nonzero(valid & ((pred == label) | (truth == label))))
        values[f"{class_name}_pixel_iou"] = intersection / max(1, union)
    values["foreground_miou"] = (
        values["vehicle_pixel_iou"] + values["person_pixel_iou"]
    ) / 2.0
    return values

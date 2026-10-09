#!/usr/bin/env python3
"""Deployable prediction-only postprocessors for the Route B v3.1 LR-ASPP object head.

The single registered candidate is a vehicle-only world-XY NMS at 2.0 m. It reads
prediction fields exclusively: no ground truth, no ignore masks, no contract state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

VEHICLE_WORLD_NMS_RADIUS_M = 2.0
NMS_CLASSES = ("vehicle",)


def vehicle_world_nms(
    predictions: Sequence[Mapping[str, Any]], radius_m: float = VEHICLE_WORLD_NMS_RADIUS_M
) -> list[Mapping[str, Any]]:
    """Greedy score-ordered world-XY NMS applied to vehicle predictions only.

    Person predictions pass through untouched and never suppress or get suppressed.
    Suppression depends only on ``class_name``, ``score``, ``world_x`` and ``world_y``.
    """
    ordered = sorted(
        range(len(predictions)),
        key=lambda index: (
            -float(predictions[index]["score"]),
            str(predictions[index]["class_name"]),
            index,
        ),
    )
    kept_indices: list[int] = []
    kept_vehicles: list[tuple[float, float]] = []
    for index in ordered:
        prediction = predictions[index]
        if str(prediction["class_name"]) not in NMS_CLASSES:
            kept_indices.append(index)
            continue
        x, y = float(prediction["world_x"]), float(prediction["world_y"])
        if any(math.hypot(x - kx, y - ky) <= radius_m for kx, ky in kept_vehicles):
            continue
        kept_vehicles.append((x, y))
        kept_indices.append(index)
    kept = set(kept_indices)
    return [prediction for index, prediction in enumerate(predictions) if index in kept]


def apply_arm(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], arm: str
) -> dict[str, list[Mapping[str, Any]]]:
    """Return a per-frame prediction mapping for one registered evaluation arm."""
    if arm == "RAW_FIXED_DECODER":
        return {sample_id: list(values) for sample_id, values in grouped.items()}
    if arm == "VEHICLE_WORLD_NMS_2M":
        return {
            sample_id: vehicle_world_nms(list(values), VEHICLE_WORLD_NMS_RADIUS_M)
            for sample_id, values in grouped.items()
        }
    raise ValueError(f"unregistered arm: {arm}")

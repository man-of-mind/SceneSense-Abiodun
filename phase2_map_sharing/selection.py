"""Deployable recipient-specific object selection baselines."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Mapping, Tuple

from .schemas import EgoState, MapObjectObservation


def select_recipient_hazards(
    objects: Iterable[MapObjectObservation],
    recipient_state: EgoState,
    *,
    capture_at_s: float,
    horizon_s: float,
    confidence_floor: float,
    safety_radius_m_by_class: Mapping[str, float],
) -> Tuple[MapObjectObservation, ...]:
    """Select objects whose observed kinematics threaten one named recipient.

    This baseline uses only causal object estimates and the recipient state. It
    does not use CARLA identity, future ground truth, or a global saliency flag.
    ``hazard_score`` is a transparent normalized proximity score, not a learned
    probability.
    """

    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    if float(capture_at_s) + 1e-12 < recipient_state.timestamp_s:
        raise ValueError("recipient state cannot come from the future of contribution capture")
    ego_dt = float(capture_at_s) - recipient_state.timestamp_s
    ego_x = recipient_state.x_m + recipient_state.vx_mps * ego_dt
    ego_y = recipient_state.y_m + recipient_state.vy_mps * ego_dt
    selected = []
    for obj in objects:
        if obj.confidence < confidence_floor:
            continue
        obj_dt = float(capture_at_s) - obj.observed_at_s
        obj_x = obj.x_m + obj.vx_mps * obj_dt
        obj_y = obj.y_m + obj.vy_mps * obj_dt
        rx, ry = obj_x - ego_x, obj_y - ego_y
        rvx = obj.vx_mps - recipient_state.vx_mps
        rvy = obj.vy_mps - recipient_state.vy_mps
        speed_sq = rvx * rvx + rvy * rvy
        if speed_sq <= 1e-12:
            t_closest = 0.0
        else:
            t_closest = max(0.0, min(float(horizon_s), -(rx * rvx + ry * rvy) / speed_sq))
        distance = math.hypot(rx + rvx * t_closest, ry + rvy * t_closest)
        radius = float(safety_radius_m_by_class.get(obj.class_name.strip().lower(), 2.5))
        if distance > radius:
            continue
        score = max(0.0, min(1.0, 1.0 - distance / max(radius, 1e-12)))
        selected.append(replace(obj, hazard_score=score))
    return tuple(selected)

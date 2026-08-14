"""Evaluation-only warning-to-truth association.

This module is deliberately outside the runtime map engine.  CARLA identity may
be present here for scoring, but it can never enter a MapContribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .schemas import WarningEvent


@dataclass(frozen=True)
class TruthTrajectory:
    truth_id: str
    class_name: str
    x_m: float
    y_m: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    state_at_s: float = 0.0
    safety_hazard: bool = False

    def position_at(self, timestamp_s: float) -> tuple[float, float]:
        dt = float(timestamp_s) - self.state_at_s
        return self.x_m + self.vx_mps * dt, self.y_m + self.vy_mps * dt


@dataclass(frozen=True)
class WarningTruthMatch:
    canonical_track_id: str
    truth_id: Optional[str]
    distance_m: Optional[float]
    safety_hazard: bool


def match_warning_to_truth(
    warning: WarningEvent,
    truth: Iterable[TruthTrajectory],
    *,
    gate_m: float = 3.0,
) -> WarningTruthMatch:
    candidates = []
    for item in truth:
        if item.class_name.strip().lower() != warning.class_name.strip().lower():
            continue
        tx, ty = item.position_at(warning.warning_at_s)
        distance = math.hypot(warning.object_x_m - tx, warning.object_y_m - ty)
        if distance <= gate_m:
            candidates.append((distance, item.truth_id, item))
    if not candidates:
        return WarningTruthMatch(warning.canonical_track_id, None, None, False)
    distance, _, item = min(candidates)
    return WarningTruthMatch(
        warning.canonical_track_id,
        item.truth_id,
        float(distance),
        bool(item.safety_hazard),
    )

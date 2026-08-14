"""Measured-profile rate-distortion helpers for the Task-C baselines."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .catalog import Action
from .shield import profile_quality


def supported_upper_hull(points: Iterable[tuple[str, float, float]]) -> tuple[str, ...]:
    """Return profiles supported by max(U - lambda*C), lambda >= 0.

    Equal-cost and weakly dominated points are removed first.  The remaining
    points form the increasing-utility Pareto staircase; the monotone-slope
    scan then removes non-supported staircase points.
    """

    best_at_cost: dict[float, tuple[str, float, float]] = {}
    for profile_id, cost, utility in points:
        value = (str(profile_id), float(cost), float(utility))
        incumbent = best_at_cost.get(value[1])
        if incumbent is None or (value[2], value[0]) > (incumbent[2], incumbent[0]):
            best_at_cost[value[1]] = value
    pareto = []
    best_utility = float("-inf")
    for value in sorted(best_at_cost.values(), key=lambda item: (item[1], item[0])):
        if value[2] > best_utility + 1e-12:
            pareto.append(value)
            best_utility = value[2]

    def slope(left: tuple[str, float, float], right: tuple[str, float, float]) -> float:
        return (right[2] - left[2]) / (right[1] - left[1])

    hull: list[tuple[str, float, float]] = []
    for value in pareto:
        while len(hull) >= 2 and slope(hull[-2], hull[-1]) <= slope(hull[-1], value) + 1e-15:
            hull.pop()
        hull.append(value)
    return tuple(item[0] for item in hull)


def supported_action_profiles(
    actions: Sequence[Action], reward_config: Mapping[str, object]
) -> tuple[str, ...]:
    seen: dict[str, Action] = {}
    for action in actions:
        if action.mode == "SPLIT" and action.profile_id is not None:
            seen.setdefault(action.profile_id, action)
    return supported_upper_hull(
        (
            profile_id,
            action.payload_kib,
            profile_quality(action, reward_config).normalized_utility,
        )
        for profile_id, action in seen.items()
    )


def lagrangian_dual_bound(
    points: Sequence[tuple[str, float, float]], budget: float
) -> tuple[float, float]:
    """Return min-lambda dual upper bound and a deterministic minimizer."""

    lambdas = {0.0}
    for left in points:
        for right in points:
            denominator = left[1] - right[1]
            if abs(denominator) <= 1e-15:
                continue
            value = (left[2] - right[2]) / denominator
            if value >= 0.0:
                lambdas.add(float(value))
    scored = []
    for value in lambdas:
        upper = value * budget + max(utility - value * cost for _, cost, utility in points)
        scored.append((upper, value))
    return min(scored, key=lambda item: (item[0], item[1]))

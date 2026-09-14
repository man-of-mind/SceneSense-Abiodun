#!/usr/bin/env python3
"""Small deterministic unit checks for analysis semantics."""

from __future__ import annotations

import math
from types import SimpleNamespace

from rl_agent.splitfusion_supervisor_analysis_v1.build_analysis import (
    pareto_ids,
    percentile,
    quality_score,
    useful_outcomes,
)


def test_quality_is_conservative_geometric_mean() -> None:
    overlap, centroid_rms_m, centroid_score, localization, combined = quality_score(
        {
            "val_segmentation_miou": 0.81,
            "val_vehicle_iou": 0.64,
            "val_person_box_mask_iou": 0.25,
            "val_vehicle_xy_mae_m": 0.6,
            "val_canonical_person_xy_mae_m": 0.8,
        }
    )
    assert abs(overlap - 0.4) < 1e-12
    assert abs(centroid_rms_m - math.sqrt(0.5)) < 1e-12
    assert abs(centroid_score - math.exp(-math.sqrt(0.5))) < 1e-12
    assert abs(localization - math.sqrt(overlap * centroid_score)) < 1e-12
    assert abs(combined - math.sqrt(0.81 * localization)) < 1e-12


def test_percentile_is_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 0.99) == 99.0


def test_pareto_minimizes_latency_and_maximizes_quality() -> None:
    rows = [
        {"action_id": 1, "x": 10.0, "combined_quality": 0.5},
        {"action_id": 2, "x": 20.0, "combined_quality": 0.4},
        {"action_id": 3, "x": 30.0, "combined_quality": 0.8},
    ]
    assert pareto_ids(rows, "x") == {1, 3}


def outcome(sequence: int, capture: int, install: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        frame=SimpleNamespace(sequence_id=sequence, capture_ns=capture),
        install_ns=install,
    )


def test_useful_installations_drop_older_late_map() -> None:
    result = SimpleNamespace(
        observation_end_ns=1_000,
        outcomes=(
            outcome(0, 100, 400),
            outcome(1, 300, 500),
            outcome(2, 200, 600),
            outcome(3, 700, None),
        ),
    )
    assert [item.frame.sequence_id for item in useful_outcomes(result)] == [0, 1]


if __name__ == "__main__":
    test_quality_is_conservative_geometric_mean()
    test_percentile_is_nearest_rank()
    test_pareto_minimizes_latency_and_maximizes_quality()
    test_useful_installations_drop_older_late_map()
    print("SUPERVISOR_ANALYSIS_UNIT_TEST_PASS")

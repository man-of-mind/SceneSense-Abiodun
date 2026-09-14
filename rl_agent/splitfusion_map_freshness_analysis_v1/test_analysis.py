from __future__ import annotations

import math

from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageFrame,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.analyze_288 import (
    BUDGETS_MS,
    distribution_fields,
    summarize_simulation,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.queue_models import (
    QueuePolicy,
    simulate_queue_policy,
)


def _frame(sequence: int, capture_ms: int) -> TwoStageFrame:
    return TwoStageFrame(
        frame_id=sequence,
        sequence_id=sequence,
        capture_ns=capture_ms * 1_000_000,
        arrival_ns=capture_ms * 1_000_000,
        compute_ns=20_000_000,
        publication_ns=5_000_000,
        post_publication_install_ns=5_000_000,
        feature_bytes=100,
    )


def test_registered_budgets_exclude_one_frame_period() -> None:
    assert BUDGETS_MS == (150, 200, 250)


def test_map_freshness_counts_initial_unavailable_time_as_not_fresh() -> None:
    frames = [_frame(0, 0), _frame(1, 100)]
    simulation = simulate_queue_policy(
        frames,
        policy=QueuePolicy.FIFO_NO_DISCARD,
        observation_tail_ns=500_000_000,
    )
    summary = summarize_simulation(simulation, input_frames=2)

    assert summary["first_useful_install_delay_from_observation_start_ms"] == 30.0
    assert math.isclose(summary["map_available_fraction"], 570 / 600)
    assert math.isclose(
        summary["fresh_map_time_ms_le_150_fraction"], 220 / 600
    )
    assert math.isclose(
        summary["fresh_map_time_ms_le_200_fraction"], 270 / 600
    )
    assert math.isclose(
        summary["fresh_map_time_ms_le_250_fraction"], 320 / 600
    )
    assert summary["timely_useful_install_yield_ms_le_150"] == 1.0


def test_distribution_is_exact_nearest_rank_not_a_fitted_model() -> None:
    result = distribution_fields("delay_ms", [40, 10, 30, 20])
    assert result == {
        "delay_ms_count": 4,
        "delay_ms_minimum": 10.0,
        "delay_ms_maximum": 40.0,
        "delay_ms_p10": 10.0,
        "delay_ms_p25": 10.0,
        "delay_ms_p50": 20.0,
        "delay_ms_p75": 30.0,
        "delay_ms_p90": 40.0,
        "delay_ms_p95": 40.0,
        "delay_ms_p99": 40.0,
    }


if __name__ == "__main__":
    test_registered_budgets_exclude_one_frame_period()
    test_map_freshness_counts_initial_unavailable_time_as_not_fresh()
    test_distribution_is_exact_nearest_rank_not_a_fitted_model()
    print("analysis metric tests: PASS")

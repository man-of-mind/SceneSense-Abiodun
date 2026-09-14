from __future__ import annotations

from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageFrame,
)
from rl_agent.splitfusion_map_freshness_analysis_v1.queue_models import (
    QueuePolicy,
    QueueReason,
    simulate_queue_policy,
)


def frame(sequence: int, arrival_ms: int, compute_ms: int = 70) -> TwoStageFrame:
    return TwoStageFrame(
        frame_id=sequence,
        sequence_id=sequence,
        capture_ns=sequence * 1_000_000,
        arrival_ns=arrival_ms * 1_000_000,
        compute_ns=compute_ms * 1_000_000,
        publication_ns=5_000_000,
        post_publication_install_ns=10_000_000,
        feature_bytes=100,
    )


def test_latest_replaces_all_pending_but_not_active() -> None:
    frames = [frame(0, 0), frame(1, 10), frame(2, 20), frame(3, 30)]
    result = simulate_queue_policy(
        frames, policy=QueuePolicy.LATEST_ONLY_NO_EXPIRY
    )
    reasons = [item.reason for item in result.outcomes]
    assert reasons == [
        QueueReason.RESULT_PUBLISHED,
        QueueReason.SUPERSEDED_PENDING_COMPUTE,
        QueueReason.SUPERSEDED_PENDING_COMPUTE,
        QueueReason.RESULT_PUBLISHED,
    ]
    assert result.outcomes[3].compute_start_ns == 70_000_000
    assert result.compute_queue_high_water == 1


def test_fifo_drains_every_admitted_frame_in_order() -> None:
    frames = [frame(0, 0), frame(1, 10), frame(2, 20), frame(3, 30)]
    result = simulate_queue_policy(frames, policy=QueuePolicy.FIFO_NO_DISCARD)
    assert all(
        item.reason is QueueReason.RESULT_PUBLISHED for item in result.outcomes
    )
    assert [item.compute_start_ns for item in result.outcomes] == [
        0,
        70_000_000,
        140_000_000,
        210_000_000,
    ]
    assert result.compute_queue_high_water == 3


def test_latest_has_no_fixed_wait_expiry() -> None:
    frames = [frame(0, 0, compute_ms=100), frame(1, 30, compute_ms=10)]
    result = simulate_queue_policy(
        frames, policy=QueuePolicy.LATEST_ONLY_NO_EXPIRY
    )
    assert result.outcomes[1].reason is QueueReason.RESULT_PUBLISHED
    assert result.outcomes[1].queue_wait_ns == 70_000_000


if __name__ == "__main__":
    test_latest_replaces_all_pending_but_not_active()
    test_fifo_drains_every_admitted_frame_in_order()
    test_latest_has_no_fixed_wait_expiry()
    print("queue model tests: PASS")

from __future__ import annotations

import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageConfig,
    TwoStageFrame,
    TwoStageReason,
    simulate_two_stage,
)


MS = 1_000_000


def frame(
    sequence: int,
    arrival_ms: int | None,
    *,
    capture_ms: int | None = None,
    compute_ms: int = 100,
    publication_ms: int = 10,
    install_ms: int = 5,
    pre_scheduler_reason: TwoStageReason | None = None,
) -> TwoStageFrame:
    capture = sequence * 10 if capture_ms is None else capture_ms
    return TwoStageFrame(
        frame_id=100 + sequence,
        sequence_id=sequence,
        capture_ns=capture * MS,
        arrival_ns=None if arrival_ms is None else arrival_ms * MS,
        compute_ns=compute_ms * MS,
        publication_ns=publication_ms * MS,
        post_publication_install_ns=install_ms * MS,
        feature_bytes=1000 + sequence,
        pre_scheduler_reason=pre_scheduler_reason,
    )


class TwoStageSimulatorTest(unittest.TestCase):
    def test_latest_pending_keeps_active_and_newest(self) -> None:
        frames = [
            frame(0, 0, capture_ms=0),
            frame(1, 20, capture_ms=20),
            frame(2, 40, capture_ms=40),
            frame(3, 60, capture_ms=60),
        ]
        result = simulate_two_stage(
            frames, config=TwoStageConfig(queue_wait_budget_ns=None)
        )
        reasons = {item.frame.sequence_id: item.reason for item in result.outcomes}
        self.assertEqual(reasons[0], TwoStageReason.RESULT_PUBLISHED)
        self.assertEqual(reasons[1], TwoStageReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons[2], TwoStageReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons[3], TwoStageReason.RESULT_PUBLISHED)

    def test_25ms_is_expiry_not_hold(self) -> None:
        immediate = simulate_two_stage(
            [frame(0, 0, capture_ms=0, compute_ms=5)],
            config=TwoStageConfig(queue_wait_budget_ns=25 * MS),
        ).outcomes[0]
        self.assertEqual(immediate.compute_start_ns, 0)

        frames = [
            frame(0, 0, capture_ms=0, compute_ms=100),
            frame(1, 60, capture_ms=60, compute_ms=5),
        ]
        result = simulate_two_stage(
            frames, config=TwoStageConfig(queue_wait_budget_ns=25 * MS)
        )
        self.assertEqual(
            result.outcomes[1].reason,
            TwoStageReason.QUEUE_WAIT_BUDGET_EXCEEDED,
        )

    def test_compute_and_publication_overlap(self) -> None:
        frames = [
            frame(0, 0, capture_ms=0, compute_ms=50, publication_ms=80),
            frame(1, 60, capture_ms=60, compute_ms=50, publication_ms=10),
        ]
        outcomes = simulate_two_stage(
            frames, config=TwoStageConfig(queue_wait_budget_ns=None)
        ).outcomes
        first, second = outcomes
        self.assertLess(second.compute_start_ns, first.publication_finish_ns)

    def test_publication_slot_is_latest_only(self) -> None:
        frames = [
            frame(0, 0, capture_ms=0, compute_ms=10, publication_ms=200),
            frame(1, 20, capture_ms=20, compute_ms=10),
            frame(2, 40, capture_ms=40, compute_ms=10),
        ]
        outcomes = simulate_two_stage(
            frames, config=TwoStageConfig(queue_wait_budget_ns=None)
        ).outcomes
        self.assertEqual(
            outcomes[1].reason, TwoStageReason.SUPERSEDED_PUBLICATION_PENDING
        )
        self.assertEqual(outcomes[2].reason, TwoStageReason.RESULT_PUBLISHED)

    def test_transport_and_terminal_accounting(self) -> None:
        frames = [
            frame(0, None, capture_ms=0),
            frame(1, 20, capture_ms=20, compute_ms=10),
        ]
        result = simulate_two_stage(
            frames, config=TwoStageConfig(queue_wait_budget_ns=25 * MS)
        )
        summary = result.summary()
        self.assertEqual(len(result.outcomes), 2)
        self.assertEqual(summary["reason_counts"]["TRANSPORT_INCOMPLETE"], 1)
        self.assertEqual(summary["ack_installed_frames"], 1)
        self.assertEqual(
            summary["feature_bytes_charged"], sum(item.feature_bytes for item in frames)
        )

    def test_measured_pre_queue_rejection_is_distinct(self) -> None:
        rejected = frame(
            0,
            None,
            capture_ms=0,
            pre_scheduler_reason=TwoStageReason.MEASURED_PRE_QUEUE_REJECTION,
        )
        result = simulate_two_stage(
            [rejected], config=TwoStageConfig(queue_wait_budget_ns=25 * MS)
        )
        self.assertEqual(
            result.outcomes[0].reason,
            TwoStageReason.MEASURED_PRE_QUEUE_REJECTION,
        )

    def test_horizon_blocks_stale_arrival(self) -> None:
        result = simulate_two_stage(
            [frame(0, 600, capture_ms=0, compute_ms=10)],
            config=TwoStageConfig(queue_wait_budget_ns=25 * MS),
        )
        self.assertEqual(
            result.outcomes[0].reason,
            TwoStageReason.PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1.simulator import (
    QueuePolicy,
    SimulationConfig,
    SimulationFrame,
    SimulationReason,
    simulate,
)
from rl_agent.splitfusion_edge_freshness_scheduler_v1.run_sweep import (
    _evenly_selected_indices,
    _linear_interpolate,
    _run_policies,
)


MS = 1_000_000


def frame(sequence: int, arrival_ms: int, service_ms: int = 100) -> SimulationFrame:
    return SimulationFrame(
        frame_id=100 + sequence,
        sequence_id=sequence,
        capture_ns=arrival_ms * MS,
        arrival_ns=arrival_ms * MS,
        service_ns=service_ms * MS,
        feature_bytes=sequence * 100,
    )


class SimulatorTest(unittest.TestCase):
    def test_latest_slot_runs_frame_one_then_frame_four(self) -> None:
        frames = (
            frame(1, 0),
            frame(2, 20),
            frame(3, 40),
            frame(4, 60),
        )
        result = simulate(
            frames,
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, None, None),
        )
        reasons = {outcome.frame.sequence_id: outcome.reason for outcome in result.outcomes}
        self.assertEqual(reasons[1], SimulationReason.INSTALLED)
        self.assertEqual(reasons[2], SimulationReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons[3], SimulationReason.SUPERSEDED_PENDING)
        self.assertEqual(reasons[4], SimulationReason.INSTALLED)

    def test_fifo_drains_every_frame(self) -> None:
        result = simulate(
            (frame(1, 0), frame(2, 20), frame(3, 40), frame(4, 60)),
            config=SimulationConfig(QueuePolicy.FIFO, None, None),
        )
        self.assertTrue(
            all(outcome.reason is SimulationReason.INSTALLED for outcome in result.outcomes)
        )

    def test_zero_wait_discards_work_that_did_not_find_idle_worker(self) -> None:
        result = simulate(
            (frame(1, 0), frame(2, 20), frame(3, 40), frame(4, 60)),
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, 0, None),
        )
        reasons = {outcome.frame.sequence_id: outcome.reason for outcome in result.outcomes}
        self.assertEqual(reasons[1], SimulationReason.INSTALLED)
        self.assertEqual(reasons[4], SimulationReason.QUEUE_WAIT_BUDGET_EXCEEDED)

    def test_processing_horizon_is_checked_at_arrival_start_and_install(self) -> None:
        late_arrival = SimulationFrame(101, 1, 0, 60 * MS, 10 * MS, 100)
        arrival_result = simulate(
            (late_arrival,),
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, None, 50 * MS),
        )
        self.assertEqual(
            arrival_result.outcomes[0].reason,
            SimulationReason.PROCESSING_HORIZON_EXPIRED_AT_ARRIVAL,
        )

        install_result = simulate(
            (frame(1, 0, service_ms=60),),
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, None, 50 * MS),
        )
        self.assertEqual(
            install_result.outcomes[0].reason,
            SimulationReason.PROCESSING_HORIZON_EXPIRED_BEFORE_INSTALL,
        )

    def test_summary_reconciles_and_reports_aoi(self) -> None:
        result = simulate(
            (frame(1, 0, 10), frame(2, 100, 10), frame(3, 200, 10)),
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, None, 500 * MS),
        )
        summary = result.summary()
        self.assertEqual(summary["input_frames"], 3)
        self.assertEqual(summary["installed_frames"], 3)
        self.assertEqual(summary["reason_counts"]["INSTALLED"], 3)
        self.assertEqual(summary["install_aoi_ms_median"], 10.0)
        self.assertIsNotNone(summary["time_weighted_map_aoi_ms"])

    def test_imputation_helpers_are_deterministic(self) -> None:
        self.assertEqual(_linear_interpolate([(0, 10), (100, 30)], 50), 20)
        self.assertEqual(_linear_interpolate([(0, 10), (100, 30)], -1), 10)
        self.assertEqual(_linear_interpolate([(0, 10), (100, 30)], 101), 30)
        self.assertEqual(_evenly_selected_indices(4, 2), {1, 3})

    def test_policy_rows_keep_transmitted_denominator(self) -> None:
        rows = _run_policies(
            [frame(1, 0, 10), frame(2, 100, 10)], transmitted_frames=3
        )
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertEqual(row["input_frames"], 2)
            self.assertEqual(row["transmitted_frames"], 3)
            self.assertEqual(row["transport_incomplete_frames"], 1)
            self.assertAlmostEqual(
                row["installed_per_transmitted"],
                row["installed_frames"] / 3,
            )


if __name__ == "__main__":
    unittest.main()

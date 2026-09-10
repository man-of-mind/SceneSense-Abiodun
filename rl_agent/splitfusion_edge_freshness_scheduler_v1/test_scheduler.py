from __future__ import annotations

import unittest

from rl_agent.splitfusion_edge_freshness_scheduler_v1.scheduler import (
    FrameTicket,
    FreshnessScheduler,
    OutcomeAccounting,
    OutcomeClass,
    Stage,
    TerminalReason,
)


MS = 1_000_000


def ticket(sequence: int, *, arrival_ms: int | None = None) -> FrameTicket:
    arrival = sequence * 100 if arrival_ms is None else arrival_ms
    return FrameTicket(
        run_id="run",
        cell_id="cell",
        stream_id="ue-1",
        frame_id=1000 + sequence,
        sequence_id=sequence,
        action_id=71,
        capture_timestamp_ns=arrival * MS,
        edge_arrival_timestamp_ns=arrival * MS,
        feature_bytes=6464,
    )


class FreshnessSchedulerTest(unittest.TestCase):
    def test_pending_replacement_is_explicit_and_not_transport_failure(self) -> None:
        scheduler = FreshnessScheduler(processing_horizon_ns=500 * MS)
        first, second = ticket(1), ticket(2)
        self.assertTrue(scheduler.offer(first, now_ns=100 * MS).admitted)
        admission = scheduler.offer(second, now_ns=200 * MS)
        self.assertTrue(admission.admitted)
        self.assertEqual(len(admission.feedback), 1)
        feedback = admission.feedback[0]
        self.assertEqual(feedback.reason, TerminalReason.SUPERSEDED_PENDING)
        self.assertEqual(feedback.replacing_frame_id, second.frame_id)
        credit = feedback.agent_credit()
        self.assertTrue(credit.intentional_freshness_drop)
        self.assertFalse(credit.count_as_transport_failure)
        self.assertFalse(credit.installation_utility_eligible)
        self.assertEqual(credit.charge_feature_bytes, first.feature_bytes)

    def test_started_work_can_stop_at_boundary_but_running_kernel_is_not_preempted(self) -> None:
        scheduler = FreshnessScheduler(processing_horizon_ns=500 * MS)
        first, second = ticket(1), ticket(2)
        scheduler.offer(first, now_ns=100 * MS)
        self.assertEqual(scheduler.take(now_ns=101 * MS).ticket, first)
        scheduler.offer(second, now_ns=200 * MS)
        feedback = scheduler.stage_gate(
            first,
            stage=Stage.BEFORE_PUBLICATION,
            now_ns=230 * MS,
            compute_spent_ns=129 * MS,
        )
        self.assertIsNotNone(feedback)
        assert feedback is not None
        self.assertEqual(
            feedback.reason, TerminalReason.SUPERSEDED_BEFORE_PUBLICATION
        )
        self.assertEqual(feedback.compute_spent_ns, 129 * MS)
        self.assertEqual(feedback.queue_wait_ns, 1 * MS)
        self.assertEqual(scheduler.take(now_ns=231 * MS).ticket, second)

    def test_stale_inflight_frame_cannot_be_installed_without_stage_gate(self) -> None:
        scheduler = FreshnessScheduler(processing_horizon_ns=500 * MS)
        first, second = ticket(1), ticket(2)
        scheduler.offer(first, now_ns=100 * MS)
        self.assertEqual(scheduler.take(now_ns=101 * MS).ticket, first)
        scheduler.offer(second, now_ns=200 * MS)
        with self.assertRaisesRegex(ValueError, "supersession gate"):
            scheduler.installed(
                first, now_ns=220 * MS, compute_spent_ns=119 * MS
            )

    def test_ten_ms_is_a_sweep_parameter_not_a_default(self) -> None:
        no_budget = FreshnessScheduler(processing_horizon_ns=500 * MS)
        candidate = ticket(1)
        no_budget.offer(candidate, now_ns=100 * MS)
        self.assertEqual(no_budget.take(now_ns=111 * MS).ticket, candidate)

        ten_ms = FreshnessScheduler(
            processing_horizon_ns=500 * MS, queue_wait_budget_ns=10 * MS
        )
        candidate = ticket(1)
        ten_ms.offer(candidate, now_ns=100 * MS)
        result = ten_ms.take(now_ns=111 * MS)
        self.assertIsNone(result.ticket)
        self.assertEqual(
            result.feedback[0].reason,
            TerminalReason.QUEUE_WAIT_BUDGET_EXCEEDED,
        )
        self.assertEqual(
            result.feedback[0].outcome_class, OutcomeClass.EXPIRED_WORK
        )

    def test_out_of_order_arrival_cannot_displace_fresher_work(self) -> None:
        scheduler = FreshnessScheduler(processing_horizon_ns=500 * MS)
        newer, older = ticket(2), ticket(1, arrival_ms=201)
        scheduler.offer(newer, now_ns=200 * MS)
        admission = scheduler.offer(older, now_ns=201 * MS)
        self.assertFalse(admission.admitted)
        self.assertEqual(
            admission.feedback[0].reason, TerminalReason.OUT_OF_ORDER_ARRIVAL
        )
        self.assertEqual(scheduler.take(now_ns=202 * MS).ticket, newer)

    def test_accounting_keeps_install_and_supersession_separate(self) -> None:
        scheduler = FreshnessScheduler(processing_horizon_ns=500 * MS)
        first, second = ticket(1), ticket(2)
        scheduler.offer(first, now_ns=100 * MS)
        displaced = scheduler.offer(second, now_ns=200 * MS).feedback[0]
        taken = scheduler.take(now_ns=201 * MS).ticket
        assert taken is not None
        installed = scheduler.installed(
            taken, now_ns=350 * MS, compute_spent_ns=149 * MS
        )
        accounting = OutcomeAccounting()
        accounting.add(displaced)
        accounting.add(installed)
        summary = accounting.summary()
        self.assertEqual(summary["terminal_frames"], 2)
        self.assertEqual(summary["installed_frames"], 1)
        self.assertEqual(summary["intentional_freshness_drops"], 1)
        self.assertEqual(summary["transport_failures"], 0)
        self.assertEqual(summary["raw_installed_over_terminal"], 0.5)
        self.assertEqual(summary["feature_bytes_charged"], 2 * 6464)
        self.assertEqual(summary["wasted_feature_bytes"], 6464)


if __name__ == "__main__":
    unittest.main()

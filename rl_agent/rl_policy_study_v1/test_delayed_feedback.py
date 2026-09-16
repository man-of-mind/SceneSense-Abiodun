#!/usr/bin/env python3
"""Focused tests for tail-only, delayed PPO feedback accounting."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rl_agent.rl_policy_study_v1.delayed_feedback import (
    FeedbackContractError,
    PendingTicket,
    PendingTransitionLedger,
    QualitySource,
    RewardWeights,
    TerminalFeedback,
    TerminalOutcome,
    TicketKey,
    transition_reward,
)


QUALITY_VERSION = "splitfusion-288-quality-v1"
QUALITY_CATALOG_SHA256 = "a" * 64


def ticket(
    frame: int,
    action: int = 71,
    *,
    previous_action: int | None = 71,
    payload_bytes: int = 100,
    compute_ms: float = 10.0,
) -> PendingTicket:
    service_start = frame * 111.111
    return PendingTicket(
        key=TicketKey("session", "ue-1", frame, action),
        decision_epoch=frame,
        opened_at_ms=service_start + 30.0,
        service_started_at_ms=service_start,
        previous_action_id=previous_action,
        segmentation_quality_anchor=0.7,
        localization_quality_anchor=0.8,
        quality_source=QualitySource.FROZEN_ACTION_VALIDATION,
        quality_version=QUALITY_VERSION,
        quality_catalog_sha256=QUALITY_CATALOG_SHA256,
        payload_bytes=payload_bytes,
        compute_ms_charged=compute_ms,
        old_log_probability=-1.0,
        old_value=0.0,
        incoming_lstm_h=(0.0,),
        incoming_lstm_c=(0.0,),
        deadline_ms=140.0,
    )


def terminal_feedback(
    item: PendingTicket,
    outcome: TerminalOutcome,
    latency_ms: float,
    *,
    payload_bytes: int | None = None,
    compute_ms: float | None = None,
    replacement_frame_id: int | None = None,
) -> TerminalFeedback:
    return TerminalFeedback(
        key=item.key,
        outcome=outcome,
        observed_at_ms=item.service_started_at_ms + latency_ms,
        payload_bytes_charged=(
            item.payload_bytes if payload_bytes is None else payload_bytes
        ),
        compute_ms_charged=(
            item.compute_ms_charged if compute_ms is None else compute_ms
        ),
        replacement_frame_id=replacement_frame_id,
    )


def tail_feedback(
    item: PendingTicket,
    latency_ms: float,
    *,
    segmentation: float | None = None,
    localization: float | None = None,
    payload_bytes: int | None = None,
    compute_ms: float | None = None,
) -> TerminalFeedback:
    return TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED,
        observed_at_ms=item.service_started_at_ms + latency_ms,
        payload_bytes_charged=(
            item.payload_bytes if payload_bytes is None else payload_bytes
        ),
        compute_ms_charged=(
            item.compute_ms_charged if compute_ms is None else compute_ms
        ),
        feedback_latency_ms=latency_ms,
        deadline_ms=item.deadline_ms,
        deadline_met=latency_ms <= item.deadline_ms,
        segmentation_quality=(
            item.segmentation_quality_anchor
            if segmentation is None
            else segmentation
        ),
        localization_quality=(
            item.localization_quality_anchor
            if localization is None
            else localization
        ),
        quality_source=item.quality_source,
        quality_version=item.quality_version,
        quality_catalog_sha256=item.quality_catalog_sha256,
        quality_available=True,
    )


class DelayedTailFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = RewardWeights(
            maximum_payload_bytes=1000,
            maximum_compute_ms=100.0,
        )

    def test_missing_next_frame_feedback_is_pending_not_lost(self) -> None:
        ledger = PendingTransitionLedger()
        first = ticket(1)
        ledger.open(first)
        state = ledger.observation(
            session_id="session",
            ue_id="ue-1",
            current_frame_id=2,
            now_ms=first.service_started_at_ms + 130.0,
        )
        self.assertEqual(state["pending_count"], 1)
        self.assertEqual(state["deadline_missed_pending_count"], 0)
        self.assertEqual(state["oldest_pending_frame_lag"], 1)
        self.assertEqual(ledger.terminal_prefix(), [])

    def test_deadline_crossing_is_pending_miss_not_transport_failure(self) -> None:
        ledger = PendingTransitionLedger()
        item = ticket(1)
        ledger.open(item)
        at_deadline = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=2,
            now_ms=item.service_started_at_ms + 140.0,
        )
        after_deadline = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=2,
            now_ms=item.service_started_at_ms + 140.001,
        )
        self.assertEqual(at_deadline["deadline_missed_pending_count"], 0)
        self.assertEqual(after_deadline["deadline_missed_pending_count"], 1)
        self.assertEqual(ledger.pending_count, 1)

    def test_out_of_order_tail_feedback_waits_for_contiguous_prefix(self) -> None:
        ledger = PendingTransitionLedger()
        first, second = ticket(1), ticket(2)
        ledger.open(first)
        ledger.open(second)
        ledger.close(tail_feedback(second, 120.0))
        self.assertEqual(ledger.terminal_prefix(), [])
        state = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=3,
            now_ms=second.service_started_at_ms + 121.0,
        )
        self.assertEqual(state["latest_tail_frame_lag"], 1)
        self.assertEqual(state["latest_tail_action_id"], 71)
        self.assertEqual(state["latest_feedback_latency_ms"], 120.0)
        self.assertAlmostEqual(state["latest_feedback_latency_ratio"], 120.0 / 140.0)
        self.assertEqual(state["latest_segmentation_quality"], 0.7)
        self.assertTrue(state["latest_quality_is_frozen_anchor"])
        self.assertFalse(state["latest_quality_is_live_proxy"])
        self.assertTrue(state["latest_deadline_met"])
        self.assertEqual(state["recent_terminal_count"], 1)
        # f2 is on-time, while f1 is already overdue and still pending.
        self.assertEqual(state["recent_on_time_tail_rate"], 0.5)
        self.assertEqual(state["recent_deadline_missed_pending_rate"], 0.5)
        ledger.close(tail_feedback(first, 139.0))
        self.assertEqual(
            [closed[0].key.frame_id for closed in ledger.terminal_prefix()], [1, 2]
        )

    def test_observation_reconstructs_historical_cutoff_without_future_leak(self) -> None:
        ledger = PendingTransitionLedger()
        first, second = ticket(1), ticket(2)
        ledger.open(first)
        ledger.open(second)
        event = tail_feedback(first, 180.0)
        ledger.close(event)
        before_second_opens = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=2,
            now_ms=second.opened_at_ms - 0.001,
        )
        self.assertEqual(before_second_opens["pending_count"], 1)
        self.assertFalse(before_second_opens["has_tail_feedback"])
        before_feedback = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=2,
            now_ms=event.observed_at_ms - 0.001,
        )
        self.assertEqual(before_feedback["pending_count"], 2)
        self.assertFalse(before_feedback["has_tail_feedback"])
        after_feedback = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=2,
            now_ms=event.observed_at_ms,
        )
        self.assertEqual(after_feedback["pending_count"], 1)
        self.assertTrue(after_feedback["has_tail_feedback"])
        self.assertEqual(after_feedback["recent_late_tail_rate"], 1.0)

    def test_late_old_ack_cannot_regress_latest_completed_frame(self) -> None:
        ledger = PendingTransitionLedger()
        first, second = ticket(1), ticket(2)
        ledger.open(first)
        ledger.open(second)
        ledger.close(tail_feedback(second, 100.0))
        ledger.close(tail_feedback(first, 300.0))
        state = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=3,
            now_ms=first.service_started_at_ms + 301.0,
        )
        self.assertEqual(state["latest_tail_frame_lag"], 1)
        self.assertEqual(state["latest_feedback_latency_ms"], 100.0)
        self.assertLess(state["time_since_last_terminal_event_ms"], 2.0)

    def test_no_feedback_fallback_contains_no_raw_frame_or_clock(self) -> None:
        state = PendingTransitionLedger().observation(
            session_id="session", ue_id="ue-1", current_frame_id=999,
            now_ms=123456.0,
        )
        self.assertFalse(state["has_tail_feedback"])
        self.assertEqual(state["latest_tail_frame_lag"], 0)
        self.assertEqual(state["time_since_latest_tail_feedback_ms"], 0.0)

    def test_duplicate_is_idempotent_and_conflict_is_fatal(self) -> None:
        ledger = PendingTransitionLedger()
        item = ticket(1)
        event = tail_feedback(item, 120.0)
        ledger.open(item)
        self.assertTrue(ledger.close(event))
        self.assertFalse(ledger.close(event))
        conflict = terminal_feedback(item, TerminalOutcome.REASSEMBLY_FAILED, 120.0)
        with self.assertRaises(FeedbackContractError):
            ledger.close(conflict)

    def test_missing_feedback_cannot_be_called_loss(self) -> None:
        ledger = PendingTransitionLedger()
        item = ticket(1)
        ledger.open(item)
        with self.assertRaises(FeedbackContractError):
            ledger.expire_after_reconciliation(
                item.key,
                observed_at_ms=item.service_started_at_ms + 700.0,
                durable_ledger_proves_absent=False,
            )

    def test_exact_140_ms_is_on_time_and_later_is_penalized(self) -> None:
        item = ticket(1)
        on_time_reward = transition_reward(item, tail_feedback(item, 140.0), self.weights)
        late_reward = transition_reward(item, tail_feedback(item, 140.001), self.weights)
        self.assertGreater(on_time_reward, late_reward)
        self.assertGreater(on_time_reward - late_reward, 0.49)

    def test_same_quality_more_latency_reduces_reward_before_deadline(self) -> None:
        item = ticket(1)
        fast = transition_reward(item, tail_feedback(item, 100.0), self.weights)
        slow = transition_reward(item, tail_feedback(item, 130.0), self.weights)
        self.assertGreater(fast, slow)

    def test_quality_requires_exact_version_hash_and_frozen_anchor(self) -> None:
        item = ticket(1)
        changed_events = (
            replace(tail_feedback(item, 120.0), quality_version="other"),
            replace(tail_feedback(item, 120.0), quality_catalog_sha256="b" * 64),
            replace(tail_feedback(item, 120.0), segmentation_quality=0.71),
            replace(tail_feedback(item, 120.0), feedback_latency_ms=119.0),
            replace(
                tail_feedback(item, 120.0),
                deadline_met="true",  # type: ignore[arg-type]
            ),
            replace(
                tail_feedback(item, 120.0),
                quality_available=1,  # type: ignore[arg-type]
            ),
            replace(
                tail_feedback(item, 120.0),
                quality_source=QualitySource.LABELLED_ENVIRONMENT,
            ),
        )
        for changed in changed_events:
            with self.subTest(changed=changed):
                with self.assertRaises(FeedbackContractError):
                    transition_reward(item, changed, self.weights)

    def test_unregistered_outcome_does_not_remove_open_ticket(self) -> None:
        ledger = PendingTransitionLedger()
        item = ticket(1)
        ledger.open(item)
        invalid = replace(
            terminal_feedback(item, TerminalOutcome.REASSEMBLY_FAILED, 80.0),
            outcome="NOT_REGISTERED",  # type: ignore[arg-type]
        )
        with self.assertRaises(FeedbackContractError):
            ledger.close(invalid)
        self.assertEqual(ledger.pending_count, 1)

    def test_previous_action_and_realized_resources_are_frozen(self) -> None:
        switched = ticket(1, action=70, previous_action=71, payload_bytes=10)
        unchanged = ticket(1, action=70, previous_action=70, payload_bytes=10)
        scalar_costs = replace(
            self.weights,
            beta_bytes=0.1,
            beta_compute=0.1,
            beta_action_switch=0.01,
        )
        event_switched = tail_feedback(
            switched, 100.0, payload_bytes=500, compute_ms=50.0
        )
        event_unchanged = replace(event_switched, key=unchanged.key)
        switch_reward = transition_reward(switched, event_switched, scalar_costs)
        unchanged_reward = transition_reward(unchanged, event_unchanged, scalar_costs)
        self.assertAlmostEqual(
            unchanged_reward - switch_reward, scalar_costs.beta_action_switch
        )
        planned_charge = tail_feedback(
            switched, 100.0, payload_bytes=10,
            compute_ms=switched.compute_ms_charged,
        )
        self.assertGreater(
            transition_reward(switched, planned_charge, scalar_costs), switch_reward
        )

    def test_one_action_per_frame_and_monotone_per_ue_order(self) -> None:
        ledger = PendingTransitionLedger()
        ledger.open(ticket(1))
        with self.assertRaises(FeedbackContractError):
            ledger.open(ticket(1, action=70))
        with self.assertRaises(FeedbackContractError):
            ledger.open(ticket(0))

    def test_registered_deadline_and_finite_open_time_are_enforced(self) -> None:
        with self.assertRaisesRegex(FeedbackContractError, "140-ms"):
            PendingTransitionLedger().open(replace(ticket(1), deadline_ms=150.0))
        with self.assertRaisesRegex(FeedbackContractError, "timestamp"):
            PendingTransitionLedger().open(
                replace(ticket(1), opened_at_ms=float("inf"))
            )

    def test_rejected_first_ticket_does_not_bind_ledger_identity(self) -> None:
        ledger = PendingTransitionLedger()
        invalid = replace(ticket(1), opened_at_ms=float("inf"))
        with self.assertRaises(FeedbackContractError):
            ledger.open(invalid)
        valid = replace(
            ticket(1),
            key=TicketKey("other-session", "ue-2", 1, 71),
        )
        ledger.open(valid)
        self.assertEqual(ledger.pending_count, 1)

    def test_recent_rates_use_all_terminal_outcomes(self) -> None:
        ledger = PendingTransitionLedger()
        items = [ticket(frame) for frame in range(1, 5)]
        for item in items:
            ledger.open(item)
        events = [
            tail_feedback(items[0], 100.0),
            tail_feedback(items[1], 160.0),
            terminal_feedback(items[2], TerminalOutcome.REASSEMBLY_FAILED, 80.0),
            terminal_feedback(
                items[3], TerminalOutcome.SUPERSEDED_PENDING, 70.0,
                replacement_frame_id=5,
            ),
        ]
        ledger.close_many(events)
        state = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=5,
            now_ms=max(event.observed_at_ms for event in events),
        )
        self.assertEqual(state["recent_terminal_count"], 4)
        self.assertEqual(state["last_terminal_frame_lag"], 1)
        self.assertEqual(state["last_terminal_action_id"], 71)
        self.assertEqual(state["recent_on_time_tail_rate"], 0.25)
        self.assertEqual(state["recent_late_tail_rate"], 0.25)
        self.assertEqual(state["recent_service_failure_rate"], 0.25)
        self.assertEqual(state["recent_superseded_rate"], 0.25)
        self.assertEqual(state["recent_deadline_miss_rate"], 0.25)

    def test_deadline_miss_rate_uses_strict_boundary_for_failure_outcomes(self) -> None:
        ledger = PendingTransitionLedger()
        on_boundary, after_boundary = ticket(1), ticket(2)
        ledger.open(on_boundary)
        ledger.open(after_boundary)
        ledger.close(
            terminal_feedback(
                on_boundary, TerminalOutcome.REASSEMBLY_FAILED, 140.0
            )
        )
        ledger.close(
            terminal_feedback(
                after_boundary, TerminalOutcome.REASSEMBLY_FAILED, 140.001
            )
        )
        state = ledger.observation(
            session_id="session", ue_id="ue-1", current_frame_id=3,
            now_ms=after_boundary.service_started_at_ms + 141.0,
        )
        self.assertEqual(state["recent_deadline_miss_rate"], 0.5)

    def test_supersession_is_not_transport_failure(self) -> None:
        item = ticket(1)
        superseded = terminal_feedback(
            item, TerminalOutcome.SUPERSEDED_PENDING, 80.0,
            replacement_frame_id=2,
        )
        failed = terminal_feedback(item, TerminalOutcome.REASSEMBLY_FAILED, 80.0)
        self.assertGreater(
            transition_reward(item, superseded, self.weights),
            transition_reward(item, failed, self.weights),
        )

    def test_structural_execution_failure_cannot_become_reward_sample(self) -> None:
        item = ticket(1)
        structural = terminal_feedback(
            item, TerminalOutcome.ACTION_EXECUTION_FAILED, 80.0
        )
        with self.assertRaisesRegex(FeedbackContractError, "structural"):
            transition_reward(item, structural, self.weights)


if __name__ == "__main__":
    unittest.main()

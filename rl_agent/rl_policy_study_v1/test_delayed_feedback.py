#!/usr/bin/env python3
"""Focused tests for delayed, out-of-order PPO feedback accounting."""

from __future__ import annotations

from rl_agent.rl_policy_study_v1.delayed_feedback import (
    FeedbackContractError,
    PendingTicket,
    PendingTransitionLedger,
    RewardWeights,
    TerminalFeedback,
    TerminalOutcome,
    TicketKey,
    transition_reward,
)


def ticket(frame: int, action: int = 71) -> PendingTicket:
    return PendingTicket(
        key=TicketKey("session", "ue-1", frame, action),
        decision_epoch=frame,
        opened_at_ms=frame * 100.0,
        capture_at_ms=frame * 100.0,
        payload_bytes=100,
        compute_ms_charged=10.0,
        old_log_probability=-1.0,
        old_value=0.0,
        incoming_lstm_h=(0.0,),
        incoming_lstm_c=(0.0,),
    )


def feedback(item: PendingTicket, at_ms: float) -> TerminalFeedback:
    return TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.MAP_INSTALLED,
        observed_at_ms=at_ms,
        install_latency_ms=80.0,
        installed_quality=0.8,
    )


def test_missing_next_frame_feedback_is_pending() -> None:
    ledger = PendingTransitionLedger()
    first = ticket(1)
    second = ticket(2)
    ledger.open(first)
    state = ledger.observation(session_id="session", ue_id="ue-1", current_frame_id=2, now_ms=200.0)
    assert state["pending_count"] == 1
    assert state["oldest_pending_frame_lag"] == 1
    ledger.open(second)
    assert ledger.pending_count == 2
    assert ledger.terminal_prefix() == []


def test_out_of_order_feedback_waits_for_contiguous_prefix() -> None:
    ledger = PendingTransitionLedger()
    first, second = ticket(1), ticket(2)
    ledger.open(first)
    ledger.open(second)
    ledger.close(feedback(second, 300.0))
    assert ledger.terminal_prefix() == []
    state = ledger.observation(session_id="session", ue_id="ue-1", current_frame_id=3, now_ms=310.0)
    assert state["latest_installed_frame_lag"] == 1
    ledger.close(feedback(first, 320.0))
    assert [item[0].key.frame_id for item in ledger.terminal_prefix()] == [1, 2]


def test_duplicate_is_idempotent_and_conflict_is_fatal() -> None:
    ledger = PendingTransitionLedger()
    item = ticket(1)
    event = feedback(item, 200.0)
    ledger.open(item)
    assert ledger.close(event)
    assert not ledger.close(event)
    conflict = TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.REASSEMBLY_FAILED,
        observed_at_ms=200.0,
    )
    try:
        ledger.close(conflict)
    except FeedbackContractError:
        pass
    else:
        raise AssertionError("conflicting feedback was accepted")


def test_missing_feedback_cannot_be_called_loss() -> None:
    ledger = PendingTransitionLedger()
    item = ticket(1)
    ledger.open(item)
    try:
        ledger.expire_after_reconciliation(
            item.key, observed_at_ms=700.0, durable_ledger_proves_absent=False
        )
    except FeedbackContractError:
        pass
    else:
        raise AssertionError("unreconciled missing feedback expired")


def test_supersession_is_not_transport_failure() -> None:
    item = ticket(1)
    superseded = TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.SUPERSEDED_PENDING,
        observed_at_ms=250.0,
        replacement_frame_id=2,
    )
    failed = TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.REASSEMBLY_FAILED,
        observed_at_ms=250.0,
    )
    weights = RewardWeights(
        maximum_payload_bytes=1000,
        maximum_compute_ms=100.0,
    )
    assert transition_reward(item, superseded, weights, previous_action_id=71) > transition_reward(item, failed, weights, previous_action_id=71)


def test_installed_quality_is_smoothly_latency_discounted() -> None:
    item = ticket(1)
    weights = RewardWeights(
        maximum_payload_bytes=1000,
        maximum_compute_ms=100.0,
        latency_tau_ms=250.0,
    )
    fresh = feedback(item, 200.0)
    stale = TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.MAP_INSTALLED,
        observed_at_ms=500.0,
        install_latency_ms=380.0,
        installed_quality=0.8,
    )
    assert transition_reward(item, fresh, weights, previous_action_id=71) > transition_reward(item, stale, weights, previous_action_id=71)


if __name__ == "__main__":
    test_missing_next_frame_feedback_is_pending()
    test_out_of_order_feedback_waits_for_contiguous_prefix()
    test_duplicate_is_idempotent_and_conflict_is_fatal()
    test_missing_feedback_cannot_be_called_loss()
    test_supersession_is_not_transport_failure()
    test_installed_quality_is_smoothly_latency_discounted()
    print("DELAYED_FEEDBACK_TEST_PASS")

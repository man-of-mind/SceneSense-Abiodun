"""Tests for the Phase-3b one-ticket action-hold and feedback state machine.

Every clock reading is an injected exact nanosecond integer; nothing sleeps and
nothing reads a wall clock, so each test is deterministic.

Ten test methods:

1.  ``test_normal_path_opens_reuses_and_closes_exactly_once``
2.  ``test_early_feedback_holds_gate_until_the_second_tensor``
3.  ``test_variable_duration_hold_records_the_exact_tensor_count``
4.  ``test_deadline_boundary_min_hold_and_late_orphan``
5.  ``test_identity_failures_are_rejected_without_state_mutation``
6.  ``test_duplicate_feedback_is_idempotent_and_conflicts_fail_closed``
7.  ``test_sequence_rules_and_frame_ids_as_validation_metadata``
8.  ``test_bounded_terminal_history_eviction_is_deterministic``
9.  ``test_controller_schema_descriptor_is_immutable_and_hash_bound``
10. ``test_terminal_classes_stay_distinguishable_and_unscored``
11. ``test_malformed_inputs_and_record_guards_fail_closed``
12. ``test_min_hold_timestamp_is_stamped_once_at_k_min`` (phase 3b.1)
13. ``test_frame_non_readmission_is_session_lifetime`` (phase 3b.1)
14. ``test_completed_ticket_timestamp_and_terminal_algebra`` (phase 3b.1)
15. ``test_controller_issues_gap_tolerant_exact_ticket_lineage`` (phase 4a.2;
    including a pre-decision genesis proof)

The schema hash in test 9 is recomputed with a locally written canonicalizer
rather than by calling the module's own helper, so the test does not merely
restate the implementation.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Mapping, Optional

from . import action_contract as ac
from . import reward_ticket_controller as rtc
from . import transaction_identity as ti

SESSION = "3f263fce-cc44-476e-93b5-19d09d439471"
OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
LINEAGE = "6d5ef476-4ea5-4bf0-9db6-65c15ac06936"

MS = 1_000_000
T0 = 1_000_000_000
B = rtc.B_REWARD_DEADLINE_NS
HEX_A = "a" * 64


def _plain(value: Any) -> Any:
    """Locally written flattener for read-only mappings/tuples to JSON types."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _independent_canonical_bytes(payload: Any) -> bytes:
    """A locally written canonicalizer, independent of the module under test."""
    return json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _independent_sha256(payload: Any) -> str:
    """Hash of the locally recomputed canonical bytes."""
    return hashlib.sha256(_independent_canonical_bytes(payload)).hexdigest()


class RewardTicketControllerTest(unittest.TestCase):
    """Phase-3b gate semantics, terminal classification and fail-closed rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()

    # -- helpers ----------------------------------------------------------- #

    def _action(self, mode_id: int = 0, q_e4: int = 3000) -> ti.ExecutedActionIdentity:
        executable = self.contract.resolve(mode_id, q_e4 / ac.Q_E4_SCALE)
        return ti.ExecutedActionIdentity.from_executable_action(
            executable, self.contract
        )

    def _controller(self, **kwargs: Any) -> rtc.RewardTicketController:
        kwargs.setdefault("controller_lineage_uuid", LINEAGE)
        return rtc.RewardTicketController(SESSION, **kwargs)

    def _message(
        self,
        action: ti.ExecutedActionIdentity,
        *,
        decision_seq: int = 0,
        reward_tensor_seq: int = 0,
        carla_frame_id: int = 100,
        session_uuid: str = SESSION,
        status: rtc.FeedbackTerminalStatus = rtc.FeedbackTerminalStatus.REWARD_FINAL,
    ) -> rtc.RewardFeedbackMessage:
        return rtc.RewardFeedbackMessage(
            identity=ti.RewardFeedbackIdentity(
                session_uuid=session_uuid,
                decision_seq=decision_seq,
                reward_tensor_seq=reward_tensor_seq,
                carla_frame_id=carla_frame_id,
                action=action,
            ),
            terminal_status=status,
        )

    def _complete_two_tensor_ticket(
        self,
        controller: rtc.RewardTicketController,
        *,
        decision_seq: int,
        first_tensor_seq: int,
        first_frame_id: int,
        opened_ns: int,
        action: Optional[ti.ExecutedActionIdentity] = None,
    ) -> rtc.CompletedTicket:
        """Open, reuse once and close one ticket with exact feedback."""
        action = action if action is not None else self._action()
        controller.open_decision(
            decision_seq=decision_seq,
            tensor_seq=first_tensor_seq,
            carla_frame_id=first_frame_id,
            action=action,
            now_ns=opened_ns,
        )
        controller.reuse_held_action(
            tensor_seq=first_tensor_seq + 1,
            carla_frame_id=first_frame_id + 1,
            now_ns=opened_ns + 50 * MS,
        )
        outcome = controller.submit_feedback(
            self._message(
                action,
                decision_seq=decision_seq,
                reward_tensor_seq=first_tensor_seq,
                carla_frame_id=first_frame_id,
            ),
            opened_ns + 80 * MS,
        )
        self.assertEqual(
            outcome.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        assert outcome.completed_ticket is not None
        return outcome.completed_ticket

    # ------------------------------------------------------------------ 1 -- #

    def test_normal_path_opens_reuses_and_closes_exactly_once(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=5, q_e4=5000)

        self.assertIs(controller.state, rtc.ControllerState.READY)
        self.assertTrue(controller.gate_available)

        first = controller.open_decision(
            decision_seq=0,
            tensor_seq=0,
            carla_frame_id=100,
            action=action,
            now_ns=T0,
        )
        self.assertIs(
            first.disposition, rtc.AdmissionDisposition.OPENED_NEW_DECISION
        )
        self.assertTrue(first.reward_requested)
        self.assertEqual(first.governed_tensor_index, 1)
        self.assertEqual(first.deadline_ns, T0 + B)
        self.assertIs(first.state_before, rtc.ControllerState.READY)
        self.assertIs(first.state_after, rtc.ControllerState.OPEN_UNRESOLVED)
        self.assertIsNone(first.completed_ticket)
        # the admitted envelope is a Phase-2 serializable record
        self.assertEqual(first.envelope.action, action)
        self.assertTrue(first.envelope.canonical_sha256())

        # the gate is now closed to new decisions
        self.assertFalse(controller.gate_available)

        second = controller.reuse_held_action(
            tensor_seq=1, carla_frame_id=101, now_ns=T0 + 100 * MS
        )
        self.assertIs(
            second.disposition, rtc.AdmissionDisposition.REUSED_HELD_ACTION
        )
        self.assertFalse(second.reward_requested)
        self.assertEqual(second.decision_seq, 0)
        self.assertEqual(second.governed_tensor_index, 2)
        # the exact same reconciled action identity, not an equal rebuild
        self.assertIs(second.action, action)
        self.assertIs(second.state_after, rtc.ControllerState.OPEN_UNRESOLVED)
        self.assertIsNone(second.completed_ticket)

        message = self._message(action, decision_seq=0, reward_tensor_seq=0,
                                carla_frame_id=100)
        outcome = controller.submit_feedback(message, T0 + 150 * MS)
        self.assertIs(
            outcome.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        self.assertTrue(outcome.accepted)
        self.assertIs(outcome.state_after, rtc.ControllerState.CLOSED)
        self.assertEqual(outcome.matched, "active")

        closed = outcome.completed_ticket
        assert closed is not None
        self.assertEqual(closed.hold_duration_tensors, 2)
        self.assertEqual(closed.tensor_seqs, (0, 1))
        self.assertEqual(closed.governed_frame_ids, (100, 101))
        self.assertEqual(closed.reward_tensor_seq, 0)
        self.assertEqual(closed.reward_carla_frame_id, 100)
        self.assertIs(closed.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT)
        self.assertEqual(closed.learning_disposition, "included")
        self.assertEqual(closed.resolution_ns, T0 + 150 * MS)
        self.assertEqual(closed.feedback_latency_ns, 150 * MS)
        # the k_min instant is the tensor-2 admission, not the closure
        self.assertEqual(closed.min_hold_satisfied_ns, T0 + 100 * MS)
        self.assertEqual(closed.closed_ns, T0 + 150 * MS)
        self.assertLess(closed.min_hold_satisfied_ns, closed.closed_ns)
        self.assertFalse(closed.timed_out)
        self.assertEqual(closed.accepted_feedback_sha256, message.canonical_sha256())
        # the completed-ticket summary serializes only through a reconciled action
        payload = closed.to_canonical_dict()
        self.assertEqual(payload["hold_duration_tensors"], 2)
        self.assertEqual(payload["reward_deadline_ns"], 200_000_000)
        self.assertEqual(payload["minimum_hold_tensors"], 2)
        self.assertNotIn("scalar_reward", payload)
        self.assertEqual(
            payload["executed_action"], action.to_canonical_dict()
        )
        # the hold re-proves the Phase-2 invariants
        self.assertTrue(closed.hold.reward_tensor.reward_requested)
        self.assertEqual(closed.hold.tensor_count, 2)

        # closing happened exactly once: a resubmission has no second effect
        again = controller.submit_feedback(message, T0 + 160 * MS)
        self.assertIs(again.disposition, rtc.FeedbackDisposition.DUPLICATE_IGNORED)
        self.assertEqual(controller.completed_count, 1)

        # the next future frame may open the next decision
        self.assertTrue(controller.gate_available)
        third = controller.open_decision(
            decision_seq=1,
            tensor_seq=2,
            carla_frame_id=102,
            action=self._action(mode_id=6, q_e4=9000),
            now_ns=T0 + 200 * MS,
        )
        self.assertIs(
            third.disposition, rtc.AdmissionDisposition.OPENED_NEW_DECISION
        )
        self.assertIs(third.state_before, rtc.ControllerState.CLOSED)
        self.assertTrue(third.reward_requested)

    # ------------------------------------------------------------------ 2 -- #

    def test_early_feedback_holds_gate_until_the_second_tensor(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=1, q_e4=0)
        controller.open_decision(
            decision_seq=0,
            tensor_seq=0,
            carla_frame_id=200,
            action=action,
            now_ns=T0,
        )

        early = controller.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=200),
            T0 + 10 * MS,
        )
        self.assertIs(
            early.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        self.assertIs(
            early.state_after, rtc.ControllerState.RESOLVED_WAITING_MIN_HOLD
        )
        self.assertIsNone(early.completed_ticket)
        self.assertEqual(controller.completed_count, 0)

        # the gate is unavailable even though the reward is already resolved
        self.assertFalse(controller.gate_available)
        status = controller.observe(T0 + 20 * MS)
        self.assertFalse(status.gate_available)
        self.assertFalse(status.min_hold_satisfied)
        self.assertEqual(status.governed_tensor_count, 1)
        self.assertIs(
            status.pending_terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT
        )

        with self.assertRaises(rtc.GateUnavailableError):
            controller.open_decision(
                decision_seq=1,
                tensor_seq=1,
                carla_frame_id=201,
                action=self._action(mode_id=2, q_e4=7000),
                now_ns=T0 + 20 * MS,
            )

        # admit_frame must not consult the actor while the ticket is held
        def _must_not_run() -> ti.ExecutedActionIdentity:  # pragma: no cover
            raise AssertionError("the actor was invoked during an action hold")

        second = controller.admit_frame(
            tensor_seq=1,
            carla_frame_id=201,
            now_ns=T0 + 30 * MS,
            next_decision_seq=1,
            select_action=_must_not_run,
        )
        self.assertIs(
            second.disposition, rtc.AdmissionDisposition.REUSED_HELD_ACTION
        )
        self.assertFalse(second.reward_requested)
        self.assertIs(second.action, action)
        self.assertEqual(second.decision_seq, 0)
        self.assertIs(second.state_after, rtc.ControllerState.CLOSED)
        closed = second.completed_ticket
        assert closed is not None
        self.assertEqual(closed.hold_duration_tensors, 2)
        self.assertIs(closed.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT)
        self.assertEqual(closed.resolution_ns, T0 + 10 * MS)
        self.assertEqual(closed.min_hold_satisfied_ns, T0 + 30 * MS)

        # only the following future frame can open a new decision
        selected = self._action(mode_id=3, q_e4=9800)
        third = controller.admit_frame(
            tensor_seq=2,
            carla_frame_id=202,
            now_ns=T0 + 40 * MS,
            next_decision_seq=1,
            select_action=lambda: rtc.PolicyDecisionSelection(
                action=selected,
                policy_decision_trace_sha256=HEX_A,
            ),
        )
        self.assertIs(
            third.disposition, rtc.AdmissionDisposition.OPENED_NEW_DECISION
        )
        self.assertEqual(third.decision_seq, 1)
        self.assertTrue(third.reward_requested)
        self.assertIs(third.action, selected)
        controller.reuse_held_action(
            tensor_seq=3,
            carla_frame_id=203,
            now_ns=T0 + 50 * MS,
        )
        final = controller.submit_feedback(
            self._message(
                selected,
                decision_seq=1,
                reward_tensor_seq=2,
                carla_frame_id=202,
            ),
            T0 + 60 * MS,
        ).completed_ticket
        assert final is not None
        self.assertEqual(final.policy_decision_trace_sha256, HEX_A)

    # ------------------------------------------------------------------ 3 -- #

    def test_variable_duration_hold_records_the_exact_tensor_count(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=11, q_e4=9800)
        controller.open_decision(
            decision_seq=4,
            tensor_seq=40,
            carla_frame_id=300,
            action=action,
            now_ns=T0,
        )

        for index, (tensor_seq, frame_id, offset) in enumerate(
            ((41, 301, 40 * MS), (42, 302, 80 * MS), (43, 303, 120 * MS)),
            start=2,
        ):
            result = controller.reuse_held_action(
                tensor_seq=tensor_seq, carla_frame_id=frame_id,
                now_ns=T0 + offset,
            )
            self.assertIs(
                result.disposition, rtc.AdmissionDisposition.REUSED_HELD_ACTION
            )
            self.assertFalse(result.reward_requested)
            self.assertIs(result.action, action)
            self.assertEqual(result.governed_tensor_index, index)
            self.assertIs(
                result.state_after, rtc.ControllerState.OPEN_UNRESOLVED
            )
            self.assertIsNone(result.completed_ticket)

        outcome = controller.submit_feedback(
            self._message(action, decision_seq=4, reward_tensor_seq=40,
                          carla_frame_id=300),
            T0 + 160 * MS,
        )
        closed = outcome.completed_ticket
        assert closed is not None
        self.assertEqual(closed.hold_duration_tensors, 4)
        self.assertEqual(closed.tensor_seqs, (40, 41, 42, 43))
        self.assertEqual(closed.governed_frame_ids, (300, 301, 302, 303))
        # k_min was reached by the second tensor and later reuses never moved it
        self.assertEqual(closed.min_hold_satisfied_ns, T0 + 40 * MS)
        self.assertEqual(closed.closed_ns, T0 + 160 * MS)
        self.assertEqual(closed.reward_tensor_seq, 40)
        # every governed tensor after the first reused the action without a
        # second reward request
        self.assertEqual(
            [member.reward_requested for member in closed.hold.tensors],
            [True, False, False, False],
        )
        self.assertEqual(
            {member.action for member in closed.hold.tensors}, {action}
        )

    # ------------------------------------------------------------------ 4 -- #

    def test_deadline_boundary_min_hold_and_late_orphan(self) -> None:
        # (a) the boundary is inclusive: receipt at exactly opened_ns + B closes
        controller = self._controller()
        action = self._action(mode_id=7, q_e4=7000)
        controller.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=400,
            action=action, now_ns=T0,
        )
        controller.reuse_held_action(
            tensor_seq=1, carla_frame_id=401, now_ns=T0 + 100 * MS
        )
        at_boundary = controller.observe(T0 + B)
        self.assertIs(at_boundary.state, rtc.ControllerState.OPEN_UNRESOLVED)
        self.assertEqual(at_boundary.remaining_ns, 0)
        self.assertIsNone(at_boundary.completed_ticket)
        outcome = controller.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=400),
            T0 + B,
        )
        self.assertIs(
            outcome.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        assert outcome.completed_ticket is not None
        self.assertEqual(outcome.completed_ticket.feedback_latency_ns, B)

        # (b) one nanosecond past the boundary is a timeout
        late = self._controller()
        late.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=410,
            action=action, now_ns=T0,
        )
        late.reuse_held_action(
            tensor_seq=1, carla_frame_id=411, now_ns=T0 + 100 * MS
        )
        expired = late.observe(T0 + B + 1)
        timed_out = expired.completed_ticket
        assert timed_out is not None
        self.assertIs(expired.state, rtc.ControllerState.CLOSED)
        self.assertIs(timed_out.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT)
        self.assertTrue(timed_out.timed_out)
        self.assertIsNone(timed_out.resolution_ns)
        self.assertIsNone(timed_out.feedback_latency_ns)
        self.assertIsNone(timed_out.accepted_feedback_sha256)
        self.assertEqual(
            timed_out.learning_disposition,
            "censored_pending_post_run_reconciliation",
        )
        self.assertEqual(timed_out.hold_duration_tensors, 2)
        # a timeout closes at the expiry observation but the k_min instant is
        # still the tensor-2 admission
        self.assertEqual(timed_out.min_hold_satisfied_ns, T0 + 100 * MS)
        self.assertEqual(timed_out.closed_ns, T0 + B + 1)
        orphan = late.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=410),
            T0 + B + 5 * MS,
        )
        self.assertIs(orphan.disposition, rtc.FeedbackDisposition.LATE_ORPHAN)
        self.assertFalse(orphan.accepted)
        self.assertEqual(orphan.matched, "terminal_history")
        self.assertIs(orphan.state_before, orphan.state_after)

        # (c) a timeout before tensor 2 never releases the gate early
        short = self._controller()
        short.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=420,
            action=action, now_ns=T0,
        )
        pending = short.observe(T0 + B + 1)
        self.assertIs(
            pending.state, rtc.ControllerState.TIMED_OUT_WAITING_MIN_HOLD
        )
        self.assertFalse(pending.gate_available)
        self.assertIsNone(pending.completed_ticket)
        self.assertEqual(short.completed_count, 0)
        self.assertIs(
            pending.pending_terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        with self.assertRaises(rtc.GateUnavailableError):
            short.open_decision(
                decision_seq=1, tensor_seq=1, carla_frame_id=421,
                action=action, now_ns=T0 + B + 2,
            )
        forced = short.reuse_held_action(
            tensor_seq=1, carla_frame_id=421, now_ns=T0 + B + 10 * MS
        )
        self.assertFalse(forced.reward_requested)
        self.assertIs(forced.action, action)
        released = forced.completed_ticket
        assert released is not None
        self.assertEqual(released.hold_duration_tensors, 2)
        self.assertIs(
            released.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        # here the k_min-satisfying reuse *is* the release, so they coincide
        self.assertEqual(released.min_hold_satisfied_ns, T0 + B + 10 * MS)
        self.assertEqual(released.closed_ns, T0 + B + 10 * MS)

        # (d) a timed-out decision can never attach to a newer decision
        newer_action = self._action(mode_id=8, q_e4=3000)
        short.open_decision(
            decision_seq=1, tensor_seq=2, carla_frame_id=422,
            action=newer_action, now_ns=T0 + B + 20 * MS,
        )
        before = short.snapshot()
        stale = short.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=420),
            T0 + B + 30 * MS,
        )
        self.assertIs(stale.disposition, rtc.FeedbackDisposition.LATE_ORPHAN)
        self.assertEqual(stale.decision_seq, 0)
        self.assertIs(short.state, rtc.ControllerState.OPEN_UNRESOLVED)
        after = short.snapshot()
        self.assertEqual(before.held_decision_seq, 1)
        self.assertEqual(after.held_decision_seq, 1)
        self.assertIsNone(after.pending_terminal_class)
        self.assertEqual(after.governed_tensor_count, 1)

        # (e) the deadline releases the gate inside admit_frame, and the frame
        # that observed the expiry -- not yet admitted anywhere -- is the future
        # frame that opens the next decision.
        routed = self._controller()
        routed.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=430,
            action=action, now_ns=T0,
        )
        routed.reuse_held_action(
            tensor_seq=1, carla_frame_id=431, now_ns=T0 + 50 * MS
        )
        picked = self._action(mode_id=0, q_e4=9800)
        opened = routed.admit_frame(
            tensor_seq=2,
            carla_frame_id=432,
            now_ns=T0 + B + 1,
            next_decision_seq=1,
            select_action=lambda: picked,
        )
        self.assertIs(
            opened.disposition, rtc.AdmissionDisposition.OPENED_NEW_DECISION
        )
        self.assertEqual(opened.decision_seq, 1)
        self.assertTrue(opened.reward_requested)
        self.assertIs(opened.action, picked)
        # the ticket the deadline closed is reported on the same admission
        released_by_clock = opened.completed_ticket
        assert released_by_clock is not None
        self.assertEqual(released_by_clock.decision_seq, 0)
        self.assertIs(
            released_by_clock.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        self.assertEqual(released_by_clock.hold_duration_tensors, 2)

    # ------------------------------------------------------------------ 5 -- #

    def test_identity_failures_are_rejected_without_state_mutation(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=2, q_e4=5000)
        other_action = self._action(mode_id=9, q_e4=9000)
        controller.open_decision(
            decision_seq=3, tensor_seq=30, carla_frame_id=500,
            action=action, now_ns=T0,
        )
        controller.reuse_held_action(
            tensor_seq=31, carla_frame_id=501, now_ns=T0 + 20 * MS
        )

        now = T0 + 50 * MS
        controller.observe(now)
        baseline = controller.snapshot()

        cases = (
            (
                "wrong session",
                self._message(action, decision_seq=3, reward_tensor_seq=30,
                              carla_frame_id=500, session_uuid=OTHER_SESSION),
                rtc.FeedbackDisposition.REJECTED_WRONG_SESSION,
            ),
            (
                "wrong decision_seq",
                self._message(action, decision_seq=4, reward_tensor_seq=30,
                              carla_frame_id=500),
                rtc.FeedbackDisposition.REJECTED_UNKNOWN_DECISION,
            ),
            (
                "ungoverned reward tensor_seq",
                self._message(action, decision_seq=3, reward_tensor_seq=99,
                              carla_frame_id=500),
                rtc.FeedbackDisposition.REJECTED_WRONG_REWARD_TENSOR_SEQ,
            ),
            (
                "governed but non-reward tensor",
                self._message(action, decision_seq=3, reward_tensor_seq=31,
                              carla_frame_id=501),
                rtc.FeedbackDisposition.REJECTED_NON_REWARD_TENSOR,
            ),
            (
                "wrong carla frame",
                self._message(action, decision_seq=3, reward_tensor_seq=30,
                              carla_frame_id=777),
                rtc.FeedbackDisposition.REJECTED_WRONG_CARLA_FRAME,
            ),
            (
                "wrong action identity",
                self._message(other_action, decision_seq=3, reward_tensor_seq=30,
                              carla_frame_id=500),
                rtc.FeedbackDisposition.REJECTED_ACTION_MISMATCH,
            ),
        )

        for label, message, expected in cases:
            with self.subTest(case=label):
                outcome = controller.submit_feedback(message, now)
                self.assertIs(outcome.disposition, expected)
                self.assertTrue(outcome.rejected)
                self.assertFalse(outcome.accepted)
                self.assertIs(outcome.state_before, outcome.state_after)
                self.assertIsNone(outcome.terminal_class)
                self.assertIsNone(outcome.completed_ticket)
                # nothing about the live ticket changed
                self.assertEqual(controller.snapshot(), baseline)
                self.assertEqual(controller.completed_count, 0)

        # a non-record submission and an unreconciled action both fail closed
        with self.assertRaises(rtc.RewardTicketControllerError):
            controller.submit_feedback({"decision_seq": 3}, now)  # type: ignore[arg-type]
        self.assertEqual(controller.snapshot(), baseline)

        keep, drop = ac.keep_drop_counts(5000)
        fabricated = ti.ExecutedActionIdentity(
            execution_mode=ac.EXECUTION_MODE,
            mode_id=2,
            family=self.contract.mode(2).family,
            quantizer=self.contract.mode(2).quantizer,
            q_e4=5000,
            keep_count=keep,
            drop_count=drop,
        )
        self.assertFalse(fabricated.is_catalog_reconciled)
        fresh = self._controller()
        with self.assertRaises(ti.UnreconciledActionIdentityError):
            fresh.open_decision(
                decision_seq=0, tensor_seq=0, carla_frame_id=1,
                action=fabricated, now_ns=T0,
            )
        self.assertIs(fresh.state, rtc.ControllerState.READY)

        # the exact message still closes the untouched ticket
        good = controller.submit_feedback(
            self._message(action, decision_seq=3, reward_tensor_seq=30,
                          carla_frame_id=500),
            now,
        )
        self.assertIs(
            good.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        assert good.completed_ticket is not None
        self.assertEqual(good.completed_ticket.hold_duration_tensors, 2)

    # ------------------------------------------------------------------ 6 -- #

    def test_duplicate_feedback_is_idempotent_and_conflicts_fail_closed(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=4, q_e4=9000)
        controller.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=600,
            action=action, now_ns=T0,
        )
        controller.reuse_held_action(
            tensor_seq=1, carla_frame_id=601, now_ns=T0 + 30 * MS
        )
        accepted = self._message(action, decision_seq=0, reward_tensor_seq=0,
                                 carla_frame_id=600)
        conflicting = self._message(
            action, decision_seq=0, reward_tensor_seq=0, carla_frame_id=600,
            status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
        )
        # identical fields hash identically; a differing payload does not
        self.assertEqual(
            accepted.canonical_sha256(),
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=600).canonical_sha256(),
        )
        self.assertNotEqual(
            accepted.canonical_sha256(), conflicting.canonical_sha256()
        )

        closed = controller.submit_feedback(accepted, T0 + 60 * MS)
        self.assertIs(
            closed.disposition, rtc.FeedbackDisposition.ACCEPTED_RESOLVED
        )
        self.assertEqual(controller.completed_count, 1)
        baseline = controller.snapshot()

        duplicate = controller.submit_feedback(accepted, T0 + 70 * MS)
        self.assertIs(
            duplicate.disposition, rtc.FeedbackDisposition.DUPLICATE_IGNORED
        )
        self.assertFalse(duplicate.accepted)
        self.assertIs(
            duplicate.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT
        )
        self.assertIsNone(duplicate.completed_ticket)
        self.assertEqual(controller.completed_count, 1)
        self.assertIs(duplicate.state_before, duplicate.state_after)

        conflict = controller.submit_feedback(conflicting, T0 + 80 * MS)
        self.assertIs(
            conflict.disposition,
            rtc.FeedbackDisposition.REJECTED_CONFLICTING_DUPLICATE,
        )
        self.assertTrue(conflict.rejected)
        self.assertEqual(controller.completed_count, 1)
        # the recorded terminal classification is unchanged
        retained = controller.completed_tickets[-1]
        self.assertIs(
            retained.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT
        )
        self.assertEqual(
            retained.accepted_feedback_sha256, accepted.canonical_sha256()
        )
        self.assertIs(controller.state, baseline.state)

        # the same rules apply to a ticket resolved but still waiting on k_min
        waiting = self._controller()
        waiting.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=610,
            action=action, now_ns=T0,
        )
        early = self._message(action, decision_seq=0, reward_tensor_seq=0,
                              carla_frame_id=610)
        waiting.submit_feedback(early, T0 + 5 * MS)
        self.assertIs(
            waiting.state, rtc.ControllerState.RESOLVED_WAITING_MIN_HOLD
        )
        held_before = waiting.snapshot()
        self.assertIs(
            waiting.submit_feedback(early, T0 + 6 * MS).disposition,
            rtc.FeedbackDisposition.DUPLICATE_IGNORED,
        )
        conflict_while_waiting = waiting.submit_feedback(
            self._message(
                action, decision_seq=0, reward_tensor_seq=0, carla_frame_id=610,
                status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
            ),
            T0 + 7 * MS,
        )
        self.assertIs(
            conflict_while_waiting.disposition,
            rtc.FeedbackDisposition.REJECTED_CONFLICTING_DUPLICATE,
        )
        self.assertIs(
            waiting.state, rtc.ControllerState.RESOLVED_WAITING_MIN_HOLD
        )
        self.assertEqual(
            waiting.snapshot().pending_terminal_class,
            held_before.pending_terminal_class,
        )

    # ------------------------------------------------------------------ 7 -- #

    def test_sequence_rules_and_frame_ids_as_validation_metadata(self) -> None:
        controller = self._controller()
        action = self._action(mode_id=0, q_e4=0)
        # carla_frame_id is validation metadata: it may move in any direction
        # while tensor_seq -- the only chronology -- strictly increases.
        controller.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=900,
            action=action, now_ns=T0,
        )
        controller.reuse_held_action(
            tensor_seq=1, carla_frame_id=400, now_ns=T0 + 10 * MS
        )
        controller.reuse_held_action(
            tensor_seq=2, carla_frame_id=650, now_ns=T0 + 20 * MS
        )
        outcome = controller.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=900),
            T0 + 30 * MS,
        )
        closed = outcome.completed_ticket
        assert closed is not None
        self.assertEqual(closed.tensor_seqs, (0, 1, 2))
        self.assertEqual(closed.governed_frame_ids, (900, 400, 650))
        # the reward tensor is the minimum tensor_seq, not the minimum frame id
        self.assertEqual(closed.reward_tensor_seq, 0)
        self.assertEqual(closed.reward_carla_frame_id, 900)

        controller.open_decision(
            decision_seq=1, tensor_seq=3, carla_frame_id=901,
            action=action, now_ns=T0 + 40 * MS,
        )

        # reused and non-monotonic tensor_seq are both refused
        for bad_seq in (3, 2, 0):
            with self.subTest(tensor_seq=bad_seq):
                with self.assertRaises(rtc.SequenceOrderError):
                    controller.reuse_held_action(
                        tensor_seq=bad_seq, carla_frame_id=902,
                        now_ns=T0 + 50 * MS,
                    )
        for bad_type in (True, 4.0, "4", None):
            with self.subTest(tensor_seq=bad_type):
                with self.assertRaises(rtc.SequenceOrderError):
                    controller.reuse_held_action(
                        tensor_seq=bad_type,  # type: ignore[arg-type]
                        carla_frame_id=902,
                        now_ns=T0 + 50 * MS,
                    )
        # a tensor is never re-attributed to another decision
        with self.assertRaises(rtc.SequenceOrderError):
            controller.reuse_held_action(
                tensor_seq=4, carla_frame_id=902, now_ns=T0 + 50 * MS,
                expect_decision_seq=0,
            )
        # a frame already governed by the active hold cannot be re-admitted
        with self.assertRaises(rtc.FrameReadmissionError):
            controller.reuse_held_action(
                tensor_seq=4, carla_frame_id=901, now_ns=T0 + 50 * MS
            )
        # nor one already governed by a retained closed decision
        with self.assertRaises(rtc.FrameReadmissionError):
            controller.reuse_held_action(
                tensor_seq=4, carla_frame_id=400, now_ns=T0 + 50 * MS
            )
        self.assertEqual(controller.snapshot().governed_tensor_count, 1)

        controller.reuse_held_action(
            tensor_seq=4, carla_frame_id=902, now_ns=T0 + 60 * MS
        )
        controller.submit_feedback(
            self._message(action, decision_seq=1, reward_tensor_seq=3,
                          carla_frame_id=901),
            T0 + 70 * MS,
        )
        self.assertTrue(controller.gate_available)

        # reused and decreasing decision_seq are both refused
        for bad_decision in (1, 0):
            with self.subTest(decision_seq=bad_decision):
                with self.assertRaises(rtc.SequenceOrderError):
                    controller.open_decision(
                        decision_seq=bad_decision, tensor_seq=5,
                        carla_frame_id=903, action=action, now_ns=T0 + 80 * MS,
                    )
        # and a monotonic clock is a precondition, not an ordering key
        with self.assertRaises(rtc.ClockRegressionError):
            controller.open_decision(
                decision_seq=2, tensor_seq=5, carla_frame_id=903,
                action=action, now_ns=T0 + 60 * MS,
            )
        self.assertIs(controller.state, rtc.ControllerState.CLOSED)
        opened = controller.open_decision(
            decision_seq=2, tensor_seq=5, carla_frame_id=903,
            action=action, now_ns=T0 + 80 * MS,
        )
        self.assertEqual(opened.decision_seq, 2)

    # ------------------------------------------------------------------ 8 -- #

    def test_bounded_terminal_history_eviction_is_deterministic(self) -> None:
        controller = self._controller(max_terminal_history=2)
        self.assertEqual(controller.max_terminal_history, 2)
        action = self._action(mode_id=10, q_e4=3000)

        first = self._complete_two_tensor_ticket(
            controller, decision_seq=0, first_tensor_seq=0,
            first_frame_id=1000, opened_ns=T0, action=action,
        )
        self.assertEqual(controller.retained_decision_seqs, (0,))
        self._complete_two_tensor_ticket(
            controller, decision_seq=1, first_tensor_seq=2,
            first_frame_id=1010, opened_ns=T0 + 300 * MS, action=action,
        )
        self.assertEqual(controller.retained_decision_seqs, (0, 1))
        self._complete_two_tensor_ticket(
            controller, decision_seq=2, first_tensor_seq=4,
            first_frame_id=1020, opened_ns=T0 + 600 * MS, action=action,
        )
        # eviction is oldest-first and therefore in decision_seq order
        self.assertEqual(controller.retained_decision_seqs, (1, 2))
        self.assertEqual(controller.completed_count, 3)
        self.assertEqual(
            [t.decision_seq for t in controller.completed_tickets], [1, 2]
        )

        # a newer decision is now active
        newer = self._action(mode_id=0, q_e4=5000)
        controller.open_decision(
            decision_seq=3, tensor_seq=6, carla_frame_id=1030,
            action=newer, now_ns=T0 + 900 * MS,
        )
        controller.observe(T0 + 910 * MS)
        baseline = controller.snapshot()

        evicted_feedback = self._message(
            action, decision_seq=first.decision_seq,
            reward_tensor_seq=first.reward_tensor_seq,
            carla_frame_id=first.reward_carla_frame_id,
        )
        outcome = controller.submit_feedback(evicted_feedback, T0 + 910 * MS)
        self.assertIs(
            outcome.disposition, rtc.FeedbackDisposition.REJECTED_UNKNOWN_DECISION
        )
        self.assertEqual(outcome.decision_seq, 0)
        self.assertIsNone(outcome.terminal_class)
        self.assertIsNone(outcome.completed_ticket)
        # the evicted decision's feedback could not touch the active ticket
        self.assertEqual(controller.snapshot(), baseline)
        self.assertIs(controller.state, rtc.ControllerState.OPEN_UNRESOLVED)
        self.assertEqual(controller.completed_count, 3)
        self.assertEqual(controller.retained_decision_seqs, (1, 2))

        # a still-retained decision is still classified exactly
        retained = controller.completed_tickets[0]
        still_known = controller.submit_feedback(
            self._message(
                action, decision_seq=retained.decision_seq,
                reward_tensor_seq=retained.reward_tensor_seq,
                carla_frame_id=retained.reward_carla_frame_id,
            ),
            T0 + 920 * MS,
        )
        self.assertIs(
            still_known.disposition, rtc.FeedbackDisposition.DUPLICATE_IGNORED
        )
        self.assertEqual(still_known.matched, "terminal_history")

        # exact identity is still enforced against a closed decision
        wrong_tensor = controller.submit_feedback(
            self._message(
                action, decision_seq=retained.decision_seq,
                reward_tensor_seq=retained.reward_tensor_seq + 50,
                carla_frame_id=retained.reward_carla_frame_id,
            ),
            T0 + 930 * MS,
        )
        self.assertIs(
            wrong_tensor.disposition,
            rtc.FeedbackDisposition.REJECTED_WRONG_REWARD_TENSOR_SEQ,
        )
        self.assertEqual(wrong_tensor.matched, "terminal_history")
        self.assertIs(wrong_tensor.state_before, wrong_tensor.state_after)
        self.assertEqual(controller.completed_count, 3)

        with self.assertRaises(rtc.RewardTicketControllerError):
            self._controller(max_terminal_history=0)
        with self.assertRaises(rtc.RewardTicketControllerError):
            self._controller(max_terminal_history=True)

    # ------------------------------------------------------------------ 9 -- #

    def test_controller_schema_descriptor_is_immutable_and_hash_bound(self) -> None:
        descriptor = rtc.CONTROLLER_SCHEMA_DESCRIPTOR
        self.assertIsInstance(descriptor, MappingProxyType)
        self.assertEqual(
            rtc.CONTROLLER_SCHEMA_ID,
            "splitfusion_hybrid_sac_reward_ticket_controller_v1",
        )
        self.assertEqual(rtc.CONTROLLER_SCHEMA_VERSION, 4)
        self.assertIn("pre-decision genesis proof", descriptor["revision_note"])
        self.assertEqual(
            rtc.CONTROLLER_SCHEMA_SHA256, _independent_sha256(descriptor)
        )
        self.assertEqual(len(rtc.CONTROLLER_SCHEMA_SHA256), 64)
        int(rtc.CONTROLLER_SCHEMA_SHA256, 16)

        # immutable at every depth
        with self.assertRaises(TypeError):
            descriptor["version"] = 2  # type: ignore[index]
        self.assertIsInstance(descriptor["constants"], MappingProxyType)
        with self.assertRaises(TypeError):
            descriptor["constants"]["reward_deadline_ns"] = 1  # type: ignore[index]
        self.assertIsInstance(descriptor["dependencies"], MappingProxyType)
        with self.assertRaises(TypeError):
            descriptor["dependencies"]["transaction_identity"][  # type: ignore[index]
                "schema_sha256"
            ] = "x"
        self.assertIsInstance(descriptor["states"], tuple)
        self.assertIsInstance(descriptor["transitions"], MappingProxyType)

        # the frozen constants are bound into the hashed descriptor
        self.assertEqual(rtc.B_REWARD_DEADLINE_NS, 200_000_000)
        self.assertEqual(rtc.K_MIN_TENSORS, 2)
        self.assertEqual(rtc.K_MIN_TENSORS, ti.MINIMUM_HOLD_TENSORS)
        constants = descriptor["constants"]
        self.assertEqual(constants["reward_deadline_ns"], rtc.B_REWARD_DEADLINE_NS)
        self.assertEqual(constants["minimum_hold_tensors"], rtc.K_MIN_TENSORS)
        self.assertEqual(constants["outstanding_tickets_max"], 1)

        # the phase-3b.1 clauses are part of the hashed contract
        self.assertIn("session", descriptor["frame_readmission"])
        self.assertIn("evicted", descriptor["frame_readmission"])
        self.assertIn("exactly once", descriptor["min_hold_timestamp"])
        self.assertIn("never overwritten", descriptor["min_hold_timestamp"])
        # the concurrency contract is stated explicitly and is hash-bound
        self.assertIn("not thread-safe", descriptor["concurrency"])
        self.assertIn("serialize", descriptor["concurrency"])
        self.assertIn("event loop", descriptor["concurrency"])
        for text in (
            rtc.__doc__ or "", rtc.RewardTicketController.__doc__ or ""
        ):
            self.assertIn("not thread-safe", text)
            self.assertIn("serialize", text)

        # the dependency schema ids and hashes are bound exactly
        dependencies = descriptor["dependencies"]
        self.assertEqual(
            dependencies["transaction_identity"]["schema_id"], ti.SCHEMA_ID
        )
        self.assertEqual(
            dependencies["transaction_identity"]["schema_sha256"],
            ti.SCHEMA_SHA256,
        )
        self.assertEqual(
            dependencies["transaction_identity"]["schema_version"],
            ti.SCHEMA_VERSION,
        )
        self.assertEqual(
            dependencies["executed_action_identity"]["schema_id"],
            ti.ACTION_IDENTITY_SCHEMA_ID,
        )
        self.assertEqual(
            dependencies["executed_action_identity"]["schema_sha256"],
            ti.ACTION_IDENTITY_SCHEMA_SHA256,
        )
        self.assertEqual(
            dependencies["action_catalog"]["sha256"], ac.CATALOG_SHA256
        )
        self.assertEqual(
            dependencies["action_catalog"]["schema"], ac.CATALOG_SCHEMA
        )

        # the declared transition table is the one the machine enforces
        self.assertEqual(
            set(descriptor["states"]),
            {state.value for state in rtc.ControllerState},
        )
        self.assertEqual(
            set(descriptor["events"]),
            {event.value for event in rtc.TicketEvent},
        )
        self.assertEqual(
            dict(descriptor["transitions"]),
            {
                f"{state.value}|{event.value}": tuple(
                    target.value for target in targets
                )
                for (state, event), targets in rtc._ALLOWED_TRANSITIONS.items()
            },
        )
        self.assertEqual(
            dict(descriptor["terminal_classes"]),
            {
                terminal.value: rtc.TERMINAL_LEARNING_DISPOSITION[terminal]
                for terminal in rtc.TerminalClass
            },
        )

        # every record leaving the module carries the same schema binding
        controller = self._controller()
        action = self._action()
        closed = self._complete_two_tensor_ticket(
            controller, decision_seq=0, first_tensor_seq=0,
            first_frame_id=1, opened_ns=T0, action=action,
        )
        for payload in (
            closed.to_canonical_dict(),
            self._message(action, carla_frame_id=1).to_canonical_dict(),
        ):
            self.assertEqual(
                payload["controller_schema_id"], rtc.CONTROLLER_SCHEMA_ID
            )
            self.assertEqual(
                payload["controller_schema_sha256"], rtc.CONTROLLER_SCHEMA_SHA256
            )
            self.assertEqual(
                payload["controller_schema_version"],
                rtc.CONTROLLER_SCHEMA_VERSION,
            )
        self.assertEqual(
            closed.canonical_sha256(), _independent_sha256(closed.to_canonical_dict())
        )

    # ----------------------------------------------------------------- 10 -- #

    def test_terminal_classes_stay_distinguishable_and_unscored(self) -> None:
        # no scalar reward, discount or gamma**d is produced in this phase
        for name in ("scalar_reward", "reward", "gamma", "discount"):
            self.assertFalse(
                hasattr(rtc.CompletedTicket, name),
                f"CompletedTicket must not expose {name} in Phase 3b",
            )

        # a proven action-path failure is an action outcome, not an instrument fault
        failure_ctl = self._controller()
        action = self._action(mode_id=3, q_e4=0)
        failure_ctl.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1100,
            action=action, now_ns=T0,
        )
        failure_ctl.reuse_held_action(
            tensor_seq=1, carla_frame_id=1101, now_ns=T0 + 20 * MS
        )
        failed = failure_ctl.submit_feedback(
            self._message(
                action, decision_seq=0, reward_tensor_seq=0,
                carla_frame_id=1100,
                status=rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE,
            ),
            T0 + 40 * MS,
        ).completed_ticket
        assert failed is not None
        self.assertIs(failed.terminal_class, rtc.TerminalClass.ACTION_PATH_FAILURE)
        self.assertEqual(
            failed.learning_disposition,
            "included_registered_negative_service_reward",
        )
        self.assertFalse(failed.timed_out)
        self.assertEqual(failed.feedback_latency_ns, 40 * MS)

        # an instrument fault is excluded and is its own terminal class
        fault_ctl = self._controller()
        fault_ctl.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1200,
            action=action, now_ns=T0,
        )
        fault_ctl.reuse_held_action(
            tensor_seq=1, carla_frame_id=1201, now_ns=T0 + 20 * MS
        )
        outcome = fault_ctl.record_infrastructure_fault(
            decision_seq=0, now_ns=T0 + 30 * MS, detail="evaluator crashed"
        )
        self.assertIs(
            outcome.disposition,
            rtc.FeedbackDisposition.ACCEPTED_INFRASTRUCTURE_FAULT,
        )
        self.assertTrue(outcome.accepted)
        self.assertIsNone(outcome.message_sha256)
        faulted = outcome.completed_ticket
        assert faulted is not None
        self.assertIs(
            faulted.terminal_class,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        )
        self.assertEqual(
            faulted.learning_disposition,
            "excluded_reported_as_experimental_failure",
        )
        self.assertIsNone(faulted.resolution_ns)
        self.assertFalse(faulted.timed_out)
        # all four terminal classes are distinct labels with distinct handling
        self.assertEqual(len(set(rtc.TERMINAL_LEARNING_DISPOSITION.values())), 4)

        # the same k_min hold applies to a fault before tensor 2
        early_fault = self._controller()
        early_fault.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1300,
            action=action, now_ns=T0,
        )
        pending = early_fault.record_infrastructure_fault(
            decision_seq=0, now_ns=T0 + 5 * MS, detail="mount unavailable"
        )
        self.assertIs(
            pending.state_after, rtc.ControllerState.FAULTED_WAITING_MIN_HOLD
        )
        self.assertIsNone(pending.completed_ticket)
        self.assertFalse(early_fault.gate_available)
        # a second terminal classification is impossible: it fails closed
        with self.assertRaises(rtc.IllegalTransitionError):
            early_fault.record_infrastructure_fault(
                decision_seq=0, now_ns=T0 + 6 * MS, detail="again"
            )
        self.assertIs(
            early_fault.state, rtc.ControllerState.FAULTED_WAITING_MIN_HOLD
        )
        matching = early_fault.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=1300),
            T0 + 7 * MS,
        )
        self.assertIs(matching.disposition, rtc.FeedbackDisposition.LATE_ORPHAN)
        released = early_fault.reuse_held_action(
            tensor_seq=1, carla_frame_id=1301, now_ns=T0 + 10 * MS
        ).completed_ticket
        assert released is not None
        self.assertEqual(released.hold_duration_tensors, 2)
        self.assertIs(
            released.terminal_class,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        )

        # faults are never re-attributed and never invented
        idle = self._controller()
        with self.assertRaises(rtc.NoOpenTicketError):
            idle.record_infrastructure_fault(
                decision_seq=0, now_ns=T0, detail="no ticket"
            )
        with self.assertRaises(rtc.NoOpenTicketError):
            idle.reuse_held_action(
                tensor_seq=0, carla_frame_id=1, now_ns=T0
            )
        idle.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1400,
            action=action, now_ns=T0,
        )
        with self.assertRaises(rtc.SequenceOrderError):
            idle.record_infrastructure_fault(
                decision_seq=1, now_ns=T0 + MS, detail="wrong decision"
            )
        with self.assertRaises(rtc.RewardTicketControllerError):
            idle.record_infrastructure_fault(
                decision_seq=0, now_ns=T0 + MS, detail=""
            )
        self.assertIs(idle.state, rtc.ControllerState.OPEN_UNRESOLVED)
        self.assertIsNone(idle.snapshot().pending_terminal_class)


    # ----------------------------------------------------------------- 11 -- #

    def test_malformed_inputs_and_record_guards_fail_closed(self) -> None:
        action = self._action(mode_id=1, q_e4=3000)

        # the session identifier is validated, never normalized
        for bad_session in (
            None, 42, "not-a-uuid", SESSION.upper(), SESSION.replace("-", ""),
            "{" + SESSION + "}",
        ):
            with self.subTest(session=bad_session):
                with self.assertRaises(rtc.RewardTicketControllerError):
                    rtc.RewardTicketController(
                        bad_session,  # type: ignore[arg-type]
                        controller_lineage_uuid=LINEAGE,
                    )
        for bad_lineage in (
            None,
            42,
            "not-a-uuid",
            LINEAGE.upper(),
            LINEAGE.replace("-", ""),
        ):
            with self.subTest(controller_lineage_uuid=bad_lineage):
                with self.assertRaises(rtc.RewardTicketControllerError):
                    self._controller(
                        controller_lineage_uuid=bad_lineage  # type: ignore[arg-type]
                    )

        controller = self._controller()

        # the clock must be an exact non-negative int
        for bad_clock in (None, 1.0, True, "0", -1):
            with self.subTest(now_ns=bad_clock):
                with self.assertRaises(rtc.ClockRegressionError):
                    controller.observe(bad_clock)  # type: ignore[arg-type]
        self.assertIsNone(controller.last_observed_ns)

        # identifiers must be exact non-negative ints
        for field, kwargs in (
            ("decision_seq", {"decision_seq": -1, "tensor_seq": 0,
                              "carla_frame_id": 0}),
            ("tensor_seq", {"decision_seq": 0, "tensor_seq": -1,
                            "carla_frame_id": 0}),
            ("carla_frame_id", {"decision_seq": 0, "tensor_seq": 0,
                                "carla_frame_id": -1}),
        ):
            with self.subTest(field=field):
                with self.assertRaises(rtc.SequenceOrderError):
                    controller.open_decision(
                        action=action, now_ns=T0, **kwargs  # type: ignore[arg-type]
                    )

        # the action must be a Phase-2 identity record
        with self.assertRaises(rtc.RewardTicketControllerError):
            controller.open_decision(
                decision_seq=0, tensor_seq=0, carla_frame_id=0,
                action="ae32/u4",  # type: ignore[arg-type]
                now_ns=T0,
            )
        # and admit_frame needs a real actor
        with self.assertRaises(rtc.RewardTicketControllerError):
            controller.admit_frame(
                tensor_seq=0, carla_frame_id=0, now_ns=T0,
                next_decision_seq=0,
                select_action=None,  # type: ignore[arg-type]
            )
        self.assertIs(controller.state, rtc.ControllerState.READY)

        # the feedback message rejects a wrong identity type or status type
        identity = ti.RewardFeedbackIdentity(
            session_uuid=SESSION, decision_seq=0, reward_tensor_seq=0,
            carla_frame_id=1, action=action,
        )
        with self.assertRaises(rtc.RewardTicketControllerError):
            rtc.RewardFeedbackMessage(
                identity={"decision_seq": 0},  # type: ignore[arg-type]
                terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
            )
        with self.assertRaises(rtc.RewardTicketControllerError):
            rtc.RewardFeedbackMessage(
                identity=identity,
                terminal_status="REWARD_FINAL",  # type: ignore[arg-type]
            )
        message = rtc.RewardFeedbackMessage(
            identity=identity,
            terminal_status=rtc.FeedbackTerminalStatus.REWARD_FINAL,
        )
        self.assertEqual(message.session_uuid, SESSION)
        self.assertEqual(message.decision_seq, 0)
        self.assertEqual(message.reward_tensor_seq, 0)
        self.assertEqual(message.carla_frame_id, 1)
        self.assertIs(message.action, action)
        self.assertIs(message.terminal_class, rtc.TerminalClass.REWARD_FINAL_EXACT)
        self.assertEqual(
            message.canonical_bytes(), _independent_canonical_bytes(
                message.to_canonical_dict()
            )
        )

        # a completed ticket refuses a hold that disagrees with its own identity
        closed = self._complete_two_tensor_ticket(
            controller, decision_seq=0, first_tensor_seq=0,
            first_frame_id=1, opened_ns=T0, action=action,
        )
        self.assertEqual(closed.canonical_bytes(), _independent_canonical_bytes(
            closed.to_canonical_dict()
        ))
        valid = dict(
            session_uuid=closed.session_uuid,
            decision_seq=closed.decision_seq,
            hold=closed.hold,
            terminal_class=closed.terminal_class,
            opened_ns=closed.opened_ns,
            deadline_ns=closed.deadline_ns,
            closed_ns=closed.closed_ns,
            resolution_ns=closed.resolution_ns,
            accepted_feedback_sha256=closed.accepted_feedback_sha256,
            min_hold_satisfied_ns=closed.min_hold_satisfied_ns,
            controller_lineage_uuid=closed.controller_lineage_uuid,
            lineage_ordinal=closed.lineage_ordinal,
            predecessor_completed_ticket_sha256=(
                closed.predecessor_completed_ticket_sha256
            ),
        )
        for label, override in (
            ("hold type", {"hold": (closed.hold,)}),
            ("terminal type", {"terminal_class": "REWARD_FINAL_EXACT"}),
            ("session mismatch", {"session_uuid": OTHER_SESSION}),
            ("decision mismatch", {"decision_seq": closed.decision_seq + 1}),
        ):
            with self.subTest(case=label):
                with self.assertRaises(rtc.RewardTicketControllerError):
                    rtc.CompletedTicket(**{**valid, **override})
        # the valid combination still constructs
        self.assertEqual(
            rtc.CompletedTicket(**valid).canonical_sha256(),
            closed.canonical_sha256(),
        )

        # convenience accessors agree with the envelope they wrap
        opened = controller.open_decision(
            decision_seq=1, tensor_seq=2, carla_frame_id=9,
            action=action, now_ns=T0 + 500 * MS,
        )
        self.assertIs(opened.transaction, opened.envelope.transaction)
        self.assertEqual(opened.tensor_seq, 2)
        self.assertEqual(opened.carla_frame_id, 9)
        self.assertEqual(controller.session_uuid, SESSION)
        self.assertEqual(controller.last_observed_ns, T0 + 500 * MS)

    # ----------------------------------------------------------------- 12 -- #

    def test_min_hold_timestamp_is_stamped_once_at_k_min(self) -> None:
        """Phase 3b.1: the k_min instant is an admission, never a closure."""
        action = self._action(mode_id=2, q_e4=3000)

        # (a) normal feedback: stamped at tensor 2, closed later by feedback
        normal = self._controller()
        normal.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1,
            action=action, now_ns=T0,
        )
        self.assertIsNone(normal.snapshot().min_hold_satisfied_ns)
        normal.reuse_held_action(
            tensor_seq=1, carla_frame_id=2, now_ns=T0 + 40 * MS
        )
        self.assertEqual(normal.snapshot().min_hold_satisfied_ns, T0 + 40 * MS)
        self.assertTrue(normal.snapshot().min_hold_satisfied)
        closed = normal.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=1),
            T0 + 120 * MS,
        ).completed_ticket
        assert closed is not None
        self.assertEqual(closed.min_hold_satisfied_ns, T0 + 40 * MS)
        self.assertEqual(closed.closed_ns, T0 + 120 * MS)
        self.assertEqual(closed.resolution_ns, T0 + 120 * MS)
        # feedback did not overwrite the admission instant
        self.assertNotEqual(closed.min_hold_satisfied_ns, closed.closed_ns)
        self.assertEqual(
            closed.to_canonical_dict()["min_hold_satisfied_ns"], T0 + 40 * MS
        )

        # (b) early feedback: the releasing reuse is itself the k_min instant
        early = self._controller()
        early.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=11,
            action=action, now_ns=T0,
        )
        early.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=11),
            T0 + 10 * MS,
        )
        # accepting feedback while below k_min must not stamp anything
        self.assertIsNone(early.snapshot().min_hold_satisfied_ns)
        early_closed = early.reuse_held_action(
            tensor_seq=1, carla_frame_id=12, now_ns=T0 + 60 * MS
        ).completed_ticket
        assert early_closed is not None
        self.assertEqual(early_closed.min_hold_satisfied_ns, T0 + 60 * MS)
        self.assertEqual(early_closed.closed_ns, T0 + 60 * MS)
        self.assertEqual(early_closed.resolution_ns, T0 + 10 * MS)
        # the resolution precedes the hold being satisfied, which is legal
        self.assertLess(
            early_closed.resolution_ns, early_closed.min_hold_satisfied_ns
        )

        # (c) timeout after two tensors: stamped at tensor 2, closed at expiry
        timeout = self._controller()
        timeout.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=21,
            action=action, now_ns=T0,
        )
        timeout.reuse_held_action(
            tensor_seq=1, carla_frame_id=22, now_ns=T0 + 30 * MS
        )
        expired = timeout.observe(T0 + B + 1).completed_ticket
        assert expired is not None
        self.assertIs(expired.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT)
        self.assertEqual(expired.min_hold_satisfied_ns, T0 + 30 * MS)
        self.assertEqual(expired.closed_ns, T0 + B + 1)
        # neither the deadline nor the closure supplied the stamp
        self.assertNotEqual(expired.min_hold_satisfied_ns, expired.deadline_ns)
        self.assertNotEqual(expired.min_hold_satisfied_ns, expired.closed_ns)

        # (d) reuse beyond k_min never moves the stamp
        long_hold = self._controller()
        long_hold.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=31,
            action=action, now_ns=T0,
        )
        long_hold.reuse_held_action(
            tensor_seq=1, carla_frame_id=32, now_ns=T0 + 20 * MS
        )
        for tensor_seq, frame_id, offset in (
            (2, 33, 40 * MS), (3, 34, 60 * MS), (4, 35, 80 * MS),
        ):
            with self.subTest(tensor_seq=tensor_seq):
                long_hold.reuse_held_action(
                    tensor_seq=tensor_seq, carla_frame_id=frame_id,
                    now_ns=T0 + offset,
                )
                self.assertEqual(
                    long_hold.snapshot().min_hold_satisfied_ns, T0 + 20 * MS
                )
        long_closed = long_hold.submit_feedback(
            self._message(action, decision_seq=0, reward_tensor_seq=0,
                          carla_frame_id=31),
            T0 + 100 * MS,
        ).completed_ticket
        assert long_closed is not None
        self.assertEqual(long_closed.hold_duration_tensors, 5)
        self.assertEqual(long_closed.min_hold_satisfied_ns, T0 + 20 * MS)

        # (e) an infrastructure fault below k_min also leaves the stamp to the
        # releasing admission
        faulted = self._controller()
        faulted.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=41,
            action=action, now_ns=T0,
        )
        faulted.record_infrastructure_fault(
            decision_seq=0, now_ns=T0 + 5 * MS, detail="evaluator fault"
        )
        self.assertIsNone(faulted.snapshot().min_hold_satisfied_ns)
        fault_closed = faulted.reuse_held_action(
            tensor_seq=1, carla_frame_id=42, now_ns=T0 + 70 * MS
        ).completed_ticket
        assert fault_closed is not None
        self.assertEqual(fault_closed.min_hold_satisfied_ns, T0 + 70 * MS)
        self.assertIsNone(fault_closed.resolution_ns)

    # ----------------------------------------------------------------- 13 -- #

    def test_frame_non_readmission_is_session_lifetime(self) -> None:
        """Phase 3b.1: a frame stays refused after its ticket is evicted."""
        controller = self._controller(max_terminal_history=1)
        action = self._action(mode_id=6, q_e4=5000)

        first = self._complete_two_tensor_ticket(
            controller, decision_seq=0, first_tensor_seq=0,
            first_frame_id=10, opened_ns=T0, action=action,
        )
        self.assertEqual(controller.retained_decision_seqs, (0,))
        self._complete_two_tensor_ticket(
            controller, decision_seq=1, first_tensor_seq=2,
            first_frame_id=12, opened_ns=T0 + 300 * MS, action=action,
        )
        # the bounded history has evicted the first ticket entirely
        self.assertEqual(controller.retained_decision_seqs, (1,))
        self.assertEqual(controller.completed_count, 2)
        self.assertEqual(first.governed_frame_ids, (10, 11))
        self.assertNotIn(
            first.decision_seq,
            [t.decision_seq for t in controller.completed_tickets],
        )
        # its feedback is no longer classifiable ...
        self.assertIs(
            controller.submit_feedback(
                self._message(
                    action, decision_seq=first.decision_seq,
                    reward_tensor_seq=first.reward_tensor_seq,
                    carla_frame_id=first.reward_carla_frame_id,
                ),
                T0 + 600 * MS,
            ).disposition,
            rtc.FeedbackDisposition.REJECTED_UNKNOWN_DECISION,
        )
        # ... but its frames can still never be re-admitted
        for evicted_frame in first.governed_frame_ids:
            with self.subTest(frame=evicted_frame, path="open"):
                with self.assertRaises(rtc.FrameReadmissionError):
                    controller.open_decision(
                        decision_seq=2, tensor_seq=4,
                        carla_frame_id=evicted_frame,
                        action=action, now_ns=T0 + 610 * MS,
                    )
        self.assertIs(controller.state, rtc.ControllerState.CLOSED)
        self.assertTrue(controller.gate_available)

        # the same holds on the reuse path
        controller.open_decision(
            decision_seq=2, tensor_seq=4, carla_frame_id=14,
            action=action, now_ns=T0 + 620 * MS,
        )
        for evicted_frame in first.governed_frame_ids:
            with self.subTest(frame=evicted_frame, path="reuse"):
                with self.assertRaises(rtc.FrameReadmissionError):
                    controller.reuse_held_action(
                        tensor_seq=5, carla_frame_id=evicted_frame,
                        now_ns=T0 + 630 * MS,
                    )
        self.assertEqual(controller.snapshot().governed_tensor_count, 1)

        # a frame id is recorded only after a *successful* admission: a frame
        # rejected on an unrelated rule stays admissible afterwards
        with self.assertRaises(rtc.SequenceOrderError):
            controller.reuse_held_action(
                tensor_seq=4, carla_frame_id=99, now_ns=T0 + 640 * MS
            )
        accepted = controller.reuse_held_action(
            tensor_seq=5, carla_frame_id=99, now_ns=T0 + 650 * MS
        )
        self.assertEqual(accepted.carla_frame_id, 99)
        self.assertEqual(accepted.governed_tensor_index, 2)
        # and now that it succeeded, it is refused like any other
        with self.assertRaises(rtc.FrameReadmissionError):
            controller.reuse_held_action(
                tensor_seq=6, carla_frame_id=99, now_ns=T0 + 660 * MS
            )

    # ----------------------------------------------------------------- 14 -- #

    def test_completed_ticket_timestamp_and_terminal_algebra(self) -> None:
        """Phase 3b.1: the hardened completed-ticket invariants."""
        controller = self._controller()
        action = self._action(mode_id=3, q_e4=7000)
        closed = self._complete_two_tensor_ticket(
            controller, decision_seq=0, first_tensor_seq=0,
            first_frame_id=1, opened_ns=T0, action=action,
        )
        valid = dict(
            session_uuid=closed.session_uuid,
            decision_seq=closed.decision_seq,
            hold=closed.hold,
            terminal_class=closed.terminal_class,
            opened_ns=closed.opened_ns,
            deadline_ns=closed.deadline_ns,
            closed_ns=closed.closed_ns,
            resolution_ns=closed.resolution_ns,
            accepted_feedback_sha256=closed.accepted_feedback_sha256,
            min_hold_satisfied_ns=closed.min_hold_satisfied_ns,
            controller_lineage_uuid=closed.controller_lineage_uuid,
            lineage_ordinal=closed.lineage_ordinal,
            predecessor_completed_ticket_sha256=(
                closed.predecessor_completed_ticket_sha256
            ),
        )
        # the real record satisfies the whole algebra
        self.assertEqual(closed.deadline_ns, closed.opened_ns + B)
        self.assertLessEqual(closed.opened_ns, closed.min_hold_satisfied_ns)
        self.assertLessEqual(closed.min_hold_satisfied_ns, closed.closed_ns)
        self.assertEqual(
            rtc.CompletedTicket(**valid).canonical_sha256(),
            closed.canonical_sha256(),
        )

        digest = closed.accepted_feedback_sha256
        assert digest is not None
        cases = (
            # deadline must be exactly opened_ns + B
            ("deadline short", {"deadline_ns": closed.opened_ns + B - 1}),
            ("deadline long", {"deadline_ns": closed.opened_ns + B + 1}),
            ("deadline zero", {"deadline_ns": 0}),
            # opened <= min_hold <= closed
            ("min hold before open",
             {"min_hold_satisfied_ns": closed.opened_ns - 1}),
            ("min hold after close",
             {"min_hold_satisfied_ns": closed.closed_ns + 1}),
            # feedback-resolved classes need both feedback fields
            ("resolved without resolution_ns", {"resolution_ns": None}),
            ("resolved without digest", {"accepted_feedback_sha256": None}),
            ("resolution before open",
             {"resolution_ns": closed.opened_ns - 1}),
            ("resolution after close",
             {"resolution_ns": closed.closed_ns + 1,
              "closed_ns": closed.closed_ns}),
            # timeout/fault classes must carry neither
            ("timeout with resolution",
             {"terminal_class": rtc.TerminalClass.FEEDBACK_TIMEOUT}),
            ("fault with digest",
             {"terminal_class": rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED}),
            # a timeout can only close after the deadline
            ("timeout closing before deadline",
             {"terminal_class": rtc.TerminalClass.FEEDBACK_TIMEOUT,
              "resolution_ns": None, "accepted_feedback_sha256": None}),
            # format validation
            ("float opened_ns", {"opened_ns": float(closed.opened_ns)}),
            ("negative opened_ns", {"opened_ns": -1}),
            ("bool decision_seq", {"decision_seq": True}),
            ("negative decision_seq", {"decision_seq": -1}),
            ("bool closed_ns", {"closed_ns": True}),
            ("digest too short", {"accepted_feedback_sha256": digest[:63]}),
            ("digest uppercase", {"accepted_feedback_sha256": digest.upper()}),
            ("digest non-hex",
             {"accepted_feedback_sha256": "z" * 64}),
            ("digest not a str", {"accepted_feedback_sha256": 1}),
            ("session not canonical",
             {"session_uuid": closed.session_uuid.upper()}),
        )
        for label, override in cases:
            with self.subTest(case=label):
                with self.assertRaises(rtc.RewardTicketControllerError):
                    rtc.CompletedTicket(**{**valid, **override})

        # a real timeout ticket is the positive control for the other branch
        timeout_ctl = self._controller()
        timeout_ctl.open_decision(
            decision_seq=0, tensor_seq=0, carla_frame_id=1,
            action=action, now_ns=T0,
        )
        timeout_ctl.reuse_held_action(
            tensor_seq=1, carla_frame_id=2, now_ns=T0 + 10 * MS
        )
        expired = timeout_ctl.observe(T0 + B + 1).completed_ticket
        assert expired is not None
        self.assertIsNone(expired.resolution_ns)
        self.assertIsNone(expired.accepted_feedback_sha256)
        self.assertGreater(expired.closed_ns, expired.deadline_ns)
        timeout_valid = dict(
            session_uuid=expired.session_uuid,
            decision_seq=expired.decision_seq,
            hold=expired.hold,
            terminal_class=expired.terminal_class,
            opened_ns=expired.opened_ns,
            deadline_ns=expired.deadline_ns,
            closed_ns=expired.closed_ns,
            resolution_ns=None,
            accepted_feedback_sha256=None,
            min_hold_satisfied_ns=expired.min_hold_satisfied_ns,
            controller_lineage_uuid=expired.controller_lineage_uuid,
            lineage_ordinal=expired.lineage_ordinal,
            predecessor_completed_ticket_sha256=(
                expired.predecessor_completed_ticket_sha256
            ),
        )
        self.assertEqual(
            rtc.CompletedTicket(**timeout_valid).canonical_sha256(),
            expired.canonical_sha256(),
        )
        for label, override in (
            ("timeout gains a digest",
             {"accepted_feedback_sha256": digest}),
            ("timeout gains a resolution",
             {"resolution_ns": expired.opened_ns + 1}),
            ("reward class without feedback fields",
             {"terminal_class": rtc.TerminalClass.REWARD_FINAL_EXACT}),
            # feedback accepted past the deadline is a LATE_ORPHAN and can
            # never be the resolution of a closed ticket
            ("resolution past the deadline",
             {"terminal_class": rtc.TerminalClass.REWARD_FINAL_EXACT,
              "resolution_ns": expired.deadline_ns + 1,
              "accepted_feedback_sha256": digest}),
        ):
            with self.subTest(case=label):
                with self.assertRaises(rtc.RewardTicketControllerError):
                    rtc.CompletedTicket(**{**timeout_valid, **override})

    def test_controller_issues_gap_tolerant_exact_ticket_lineage(self) -> None:
        """Adjacency follows completion order, never ``decision_seq - 1``."""
        controller = self._controller()
        self.assertIsNone(controller.genesis_proof)
        genesis = controller.authorize_episode_start(
            first_decision_seq=5,
            first_tensor_seq=10,
            first_carla_frame_id=100,
            state_observed_ns=T0,
        )
        # Issued and serializable before any decision or feedback exists.
        self.assertEqual(controller.completed_count, 0)
        self.assertIs(controller.state, rtc.ControllerState.READY)
        self.assertTrue(genesis.is_attested)
        self.assertEqual(genesis.session_uuid, SESSION)
        self.assertEqual(genesis.controller_lineage_uuid, LINEAGE)
        self.assertEqual(
            genesis.to_canonical_dict()["controller_schema_sha256"],
            rtc.CONTROLLER_SCHEMA_SHA256,
        )
        forged_genesis = rtc.ControllerGenesisProof(
            session_uuid=SESSION,
            controller_lineage_uuid=LINEAGE,
            first_decision_seq=5,
            first_tensor_seq=10,
            first_carla_frame_id=100,
            state_observed_ns=T0,
        )
        self.assertFalse(forged_genesis.is_attested)
        with self.assertRaises(rtc.RewardTicketControllerError):
            forged_genesis.to_canonical_dict()

        action = self._action(mode_id=3, q_e4=5000)
        first = self._complete_two_tensor_ticket(
            controller,
            decision_seq=5,
            first_tensor_seq=10,
            first_frame_id=100,
            opened_ns=T0,
            action=action,
        )
        second = self._complete_two_tensor_ticket(
            controller,
            decision_seq=19,
            first_tensor_seq=20,
            first_frame_id=200,
            opened_ns=T0 + 500 * MS,
            action=action,
        )
        self.assertTrue(first.lineage_is_attested)
        self.assertTrue(second.lineage_is_attested)
        self.assertEqual(first.controller_lineage_uuid, LINEAGE)
        self.assertEqual(second.controller_lineage_uuid, LINEAGE)
        self.assertEqual(first.lineage_ordinal, 0)
        self.assertIsNone(first.predecessor_completed_ticket_sha256)
        self.assertEqual(second.lineage_ordinal, 1)
        self.assertEqual(
            second.predecessor_completed_ticket_sha256,
            first.canonical_sha256(),
        )

        # Copying all visible fields is still not controller provenance.
        forged = rtc.CompletedTicket(
            session_uuid=second.session_uuid,
            decision_seq=second.decision_seq,
            hold=second.hold,
            terminal_class=second.terminal_class,
            opened_ns=second.opened_ns,
            deadline_ns=second.deadline_ns,
            closed_ns=second.closed_ns,
            resolution_ns=second.resolution_ns,
            accepted_feedback_sha256=second.accepted_feedback_sha256,
            min_hold_satisfied_ns=second.min_hold_satisfied_ns,
            controller_lineage_uuid=second.controller_lineage_uuid,
            lineage_ordinal=second.lineage_ordinal,
            predecessor_completed_ticket_sha256=(
                second.predecessor_completed_ticket_sha256
            ),
            controller_genesis_proof_sha256=(
                second.controller_genesis_proof_sha256
            ),
        )
        self.assertFalse(forged.lineage_is_attested)
        with self.assertRaises(rtc.RewardTicketControllerError):
            forged.require_lineage_attested()
        with self.assertRaises(rtc.RewardTicketControllerError):
            replace(second, closed_ns=second.closed_ns + 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

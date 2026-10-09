"""Bounded stdlib correctness checks; no producer, network, GPU, or timings."""

from dataclasses import FrozenInstanceError, replace
import threading
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1.transactions import (
    CapacityExceeded, FramePlan, HistoryState, PlanKey, PlanKind, ProtocolError,
    RetiredTransaction, RunInvalid, TentativeResult, TransactionController,
    TransactionError, WireBinding,
)


def plan(index=1, *, reserve=128, kind=PlanKind.ACCEPTED):
    key = PlanKey(index, ("complete-plan-identity-%s" % index).encode())
    return FramePlan(key, WireBinding(key, ("wire-ticket-tensor-%s" % index).encode()),
                     kind, b"captured-input", reserve)


def controller(**kwargs):
    options = dict(max_jobs=4, max_bytes=1024, max_history_bytes=128)
    options.update(kwargs)
    return TransactionController(**options)


def validate(result):
    if not result.numerical_payload.startswith(b"valid-"):
        raise ValueError("fixture numerical admission rejection")


class TransactionTests(unittest.TestCase):
    def finish_computation(self, ctl, ticket):
        lease = ctl.claim_next()
        self.assertEqual(lease.ticket, ticket)
        result = TentativeResult.for_lease(lease, lease.base_history.payload + b"H", b"valid-numerical")
        self.assertTrue(ctl.complete_computation(lease, result))
        return lease, result

    def ready(self, ctl, current_plan):
        ticket = ctl.admit(current_plan)
        lease, result = self.finish_computation(ctl, ticket)
        ctl.validate_computation(ticket, validate)
        ctl.confirm_send(ticket, current_plan.expected_wire)
        ready = ctl.take_ready()
        self.assertEqual(ready.ticket, ticket)
        return ticket, lease, result

    def drain_frame(self, ctl, ticket, *, archive_first=False):
        if archive_first:
            ctl.complete_archive(ticket)
        ctl.complete_scoring(ticket)
        ctl.confirm_feedback(ticket)
        if not archive_first:
            ctl.complete_archive(ticket)

    def test_computation_then_send_is_provisional_and_commits_once(self):
        ctl = controller()
        current = plan()
        ticket = ctl.admit(current)
        lease, result = self.finish_computation(ctl, ticket)
        self.assertFalse(ctl.complete_computation(lease, result))
        self.assertIsNone(ctl.take_ready())
        ctl.validate_computation(ticket, validate)
        self.assertFalse(ctl.validate_computation(ticket, validate))
        self.assertEqual(ctl.snapshot().history.payload, b"")
        self.assertIsNone(ctl.claim_next())
        ctl.confirm_send(ticket, current.expected_wire)
        self.assertFalse(ctl.confirm_send(ticket, current.expected_wire))
        self.assertEqual(ctl.snapshot().history.payload, b"H")
        self.assertEqual(ctl.take_ready().result, result)
        self.assertIsNone(ctl.take_ready())
        self.drain_frame(ctl, ticket)
        self.assertEqual(ctl.snapshot().committed_frames, 1)

    def test_send_then_computation_still_waits_for_validation(self):
        ctl = controller()
        current = plan()
        ticket = ctl.admit(current)
        ctl.confirm_send(ticket, current.expected_wire)
        self.finish_computation(ctl, ticket)
        self.assertEqual(ctl.snapshot().history_version, 0)
        self.assertIsNone(ctl.take_ready())
        ctl.validate_computation(ticket, validate)
        self.assertEqual(ctl.take_ready().history_version, 1)
        self.drain_frame(ctl, ticket, archive_first=True)
        self.assertTrue(ctl.wait_drained())

    def test_excluded_populations_never_reserve_or_advance_history(self):
        ctl = controller()
        for kind in (PlanKind.HOLD, PlanKind.FALLBACK, PlanKind.DISPLACED):
            self.assertIsNone(ctl.admit(plan(999, kind=kind)))
        self.assertTrue(ctl.snapshot().drained)
        self.assertEqual(ctl.snapshot().reserved_bytes, 0)
        self.assertEqual(ctl.claim_next(), None)
        self.assertIsNotNone(ctl.admit(plan(1)))

    def test_next_history_lease_waits_for_prior_commit(self):
        ctl = controller()
        first, second = plan(1), plan(2)
        t1, t2 = ctl.admit(first), ctl.admit(second)
        ctl.confirm_send(t2, second.expected_wire)
        self.finish_computation(ctl, t1)
        ctl.validate_computation(t1, validate)
        self.assertIsNone(ctl.claim_next())
        ctl.confirm_send(t1, first.expected_wire)
        lease, _ = self.finish_computation(ctl, t2)
        self.assertEqual(lease.base_history.payload, b"H")
        self.assertEqual(lease.history_version, 1)
        ctl.validate_computation(t2, validate)
        self.assertEqual(ctl.snapshot().history.payload, b"HH")
        self.assertEqual(ctl.take_ready().ticket, t1)
        self.assertEqual(ctl.take_ready().ticket, t2)
        self.drain_frame(ctl, t2)
        self.drain_frame(ctl, t1)
        self.assertTrue(ctl.wait_drained())

    def test_exact_duplicate_plan_idempotent_conflict_invalidates(self):
        ctl = controller()
        current = plan()
        ticket = ctl.admit(current)
        self.assertEqual(ticket, ctl.admit(current))
        self.assertEqual(ctl.snapshot().pending_jobs, 1)
        with self.assertRaises(ProtocolError):
            ctl.admit(replace(current, input_payload=b"different"))
        self.assertFalse(ctl.snapshot().run_valid)
        ctl.cancel(ticket)
        self.assertTrue(ctl.wait_drained())

    def test_regressing_frame_rejected_without_evicting(self):
        ctl = controller()
        ticket = ctl.admit(plan(8))
        with self.assertRaises(ProtocolError):
            ctl.admit(plan(7))
        self.assertEqual(ctl.snapshot().pending_jobs, 1)
        ctl.cancel(ticket)

    def test_foreign_wire_binding_invalidates_without_history(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        self.finish_computation(ctl, ticket)
        ctl.validate_computation(ticket, validate)
        with self.assertRaises(ProtocolError):
            ctl.confirm_send(ticket, WireBinding(ticket.plan_key, b"foreign-wire-ticket"))
        self.assertEqual(ctl.snapshot().history_version, 0)
        ctl.cancel(ticket)
        self.assertTrue(ctl.wait_drained())

    def test_conflicting_result_duplicate_is_explicit(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        lease, result = self.finish_computation(ctl, ticket)
        with self.assertRaises(ProtocolError):
            ctl.complete_computation(lease, replace(result, next_history=HistoryState.from_bytes(b"different")))
        ctl.cancel(ticket)
        self.assertFalse(ctl.snapshot().run_valid)

    def test_unsent_cancel_discards_tentative_history(self):
        ctl = controller()
        t1 = ctl.admit(plan(1))
        t2 = ctl.admit(plan(2))
        self.finish_computation(ctl, t1)
        ctl.validate_computation(t1, validate)
        ctl.cancel(t1)
        self.assertEqual(ctl.snapshot().history.payload, b"")
        lease = ctl.claim_next()
        self.assertEqual(lease.ticket, t2)
        self.assertEqual(lease.base_history.payload, b"")
        ctl.cancel(t2)
        self.assertFalse(ctl.wait_drained())
        ctl.acknowledge_cancelled(lease)
        self.assertTrue(ctl.wait_drained())
        self.assertTrue(ctl.snapshot().run_valid)

    def test_cancel_keeps_reservation_until_worker_releases(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        lease = ctl.claim_next()
        ctl.cancel(ticket)
        self.assertFalse(ctl.cancel(ticket))
        self.assertEqual(ctl.snapshot().reserved_bytes, 128)
        self.assertIsNone(ctl.claim_next())
        result = TentativeResult.for_lease(lease, b"must-not-commit", b"valid-result")
        self.assertFalse(ctl.complete_computation(lease, result))
        self.assertEqual(ctl.snapshot().reserved_bytes, 0)
        self.assertEqual(ctl.snapshot().history.payload, b"")
        with self.assertRaises(RetiredTransaction):
            ctl.complete_computation(lease, result)

    def test_failed_send_after_provisional_validation_never_commits(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        self.finish_computation(ctl, ticket)
        ctl.validate_computation(ticket, validate)
        ctl.send_failed(ticket, "fixture network failure")
        self.assertEqual(ctl.snapshot().history_version, 0)
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertTrue(ctl.wait_drained())
        with self.assertRaises(RunInvalid):
            ctl.admit(plan(2))

    def test_failed_send_during_worker_retains_reservation(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        lease = ctl.claim_next()
        ctl.send_failed(ticket, "failed")
        self.assertEqual(ctl.snapshot().pending_jobs, 1)
        ctl.acknowledge_cancelled(lease)
        self.assertEqual(ctl.snapshot().reserved_bytes, 0)

    def test_worker_failure_sticky_and_releases(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        lease = ctl.claim_next()
        ctl.worker_failed(lease, "fixture exception")
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertEqual(ctl.snapshot().history_version, 0)
        self.assertTrue(ctl.wait_drained())
        with self.assertRaises(RunInvalid):
            ctl.claim_next()

    def test_capacity_limits_include_pending_archive_no_eviction(self):
        for limits in ((1, 1024), (4, 128)):
            with self.subTest(limits=limits):
                ctl = TransactionController(max_jobs=limits[0], max_bytes=limits[1], max_history_bytes=128)
                ticket, _, _ = self.ready(ctl, plan())
                ctl.complete_scoring(ticket)
                ctl.confirm_feedback(ticket)
                with self.assertRaises(CapacityExceeded):
                    ctl.admit(plan(2))
                self.assertEqual(ctl.snapshot().pending_jobs, 1)
                self.assertEqual(ctl.snapshot().reserved_bytes, 128)
                ctl.complete_archive(ticket)
                self.assertTrue(ctl.wait_drained())
                self.assertFalse(ctl.snapshot().run_valid)

    def test_computed_bytes_cannot_exceed_reservation(self):
        ctl = controller()
        ticket = ctl.admit(plan(reserve=70))
        lease = ctl.claim_next()
        result = TentativeResult.for_lease(lease, b"H", b"valid-result-too-large")
        with self.assertRaises(CapacityExceeded):
            ctl.complete_computation(lease, result)
        self.assertTrue(ctl.wait_drained())
        self.assertFalse(ctl.snapshot().run_valid)

    def test_history_capacity_is_independently_bounded(self):
        ctl = TransactionController(max_jobs=2, max_bytes=1024, max_history_bytes=1)
        ticket = ctl.admit(plan())
        lease = ctl.claim_next()
        with self.assertRaises(CapacityExceeded):
            ctl.complete_computation(lease, TentativeResult.for_lease(lease, b"HH", b"valid-result"))
        self.assertTrue(ctl.wait_drained())

    def test_identity_and_retained_base_history_count_toward_capacity(self):
        with self.assertRaises(ValueError):
            plan(reserve=14)
        ctl = controller(initial_history=b"H" * 80)
        ticket = ctl.admit(plan(reserve=128))
        with self.assertRaises(CapacityExceeded):
            ctl.claim_next()
        self.assertTrue(ctl.wait_drained())
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertEqual(ctl.snapshot().history.payload, b"H" * 80)

    def test_late_archive_failure_preserves_emitted_feedback(self):
        ctl = controller()
        ticket, _, _ = self.ready(ctl, plan())
        ctl.complete_scoring(ticket)
        ctl.confirm_feedback(ticket)
        self.assertEqual(ctl.snapshot().feedback_emitted, 1)
        ctl.archive_failed(ticket, "disk full")
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertEqual(ctl.snapshot().feedback_emitted, 1)
        self.assertEqual(ctl.snapshot().history_version, 1)
        self.assertTrue(ctl.wait_drained())

    def test_archive_first_retains_until_scoring_and_feedback_complete(self):
        ctl = controller()
        ticket, _, _ = self.ready(ctl, plan())
        self.assertTrue(ctl.complete_archive(ticket))
        self.assertFalse(ctl.complete_archive(ticket))
        self.assertEqual(ctl.snapshot().reserved_bytes, 128)
        self.assertTrue(ctl.complete_scoring(ticket))
        self.assertFalse(ctl.complete_scoring(ticket))
        self.assertFalse(ctl.wait_drained())
        ctl.confirm_feedback(ticket)
        self.assertTrue(ctl.wait_drained())

    def test_feedback_duplicates_and_scoring_failure(self):
        ctl = controller()
        ticket, _, _ = self.ready(ctl, plan())
        ctl.complete_scoring(ticket)
        self.assertTrue(ctl.confirm_feedback(ticket))
        self.assertFalse(ctl.confirm_feedback(ticket))
        ctl.complete_archive(ticket)
        ctl2 = controller()
        ticket2, _, _ = self.ready(ctl2, plan())
        ctl2.scoring_failed(ticket2, "cannot score fixture")
        ctl2.complete_archive(ticket2)
        self.assertTrue(ctl2.wait_drained())
        self.assertFalse(ctl2.snapshot().run_valid)
        self.assertEqual(ctl2.snapshot().feedback_emitted, 0)

    def test_validator_outside_lock_and_cancel_during_validation(self):
        ctl = controller()
        ticket = ctl.admit(plan())
        self.finish_computation(ctl, ticket)
        entered, resume = threading.Event(), threading.Event()
        results, failures = [], []

        def blocking_validator(result):
            # Calling snapshot inside the callback proves it does not hold the
            # controller's non-reentrant lock.
            self.assertEqual(ctl.snapshot().history_version, 0)
            entered.set()
            if not resume.wait(2):
                raise AssertionError("test validation resume timed out")
            validate(result)

        def worker():
            try:
                results.append(ctl.validate_computation(ticket, blocking_validator))
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            ctl.cancel(ticket)
            self.assertFalse(ctl.wait_drained())
            self.assertEqual(ctl.snapshot().reserved_bytes, 128)
        finally:
            resume.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, [False])
        self.assertTrue(ctl.wait_drained())
        self.assertEqual(ctl.snapshot().history_version, 0)

    def test_concurrent_send_confirmations_and_validation_commit_once(self):
        ctl = controller()
        current = plan()
        ticket = ctl.admit(current)
        self.finish_computation(ctl, ticket)
        start = threading.Barrier(5, timeout=2)
        failures = []

        def contender(index):
            try:
                start.wait()
                if index % 2:
                    ctl.confirm_send(ticket, current.expected_wire)
                else:
                    ctl.validate_computation(ticket, validate)
            except BaseException as exc:
                failures.append(exc)

        threads = [threading.Thread(target=contender, args=(index,), daemon=True) for index in range(4)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(2)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(ctl.snapshot().history_version, 1)
        self.assertIsNotNone(ctl.take_ready())
        self.assertIsNone(ctl.take_ready())
        self.drain_frame(ctl, ticket)
        self.assertTrue(ctl.wait_drained())

    def test_validation_rejects_boolean_shortcut_and_numerical_failure(self):
        for validator in (lambda result: True, lambda result: (_ for _ in ()).throw(ValueError("bad"))):
            with self.subTest(validator=validator):
                ctl = controller()
                ticket = ctl.admit(plan())
                self.finish_computation(ctl, ticket)
                ctl.confirm_send(ticket, plan().expected_wire)
                with self.assertRaises(ValueError):
                    ctl.validate_computation(ticket, validator)
                self.assertFalse(ctl.snapshot().run_valid)
                self.assertEqual(ctl.snapshot().history_version, 0)
                self.assertTrue(ctl.wait_drained())

    def test_close_drains_without_dropping_and_stale_ticket_explicit(self):
        ctl = controller()
        ticket, _, _ = self.ready(ctl, plan())
        ctl.close()
        self.assertFalse(ctl.wait_drained())
        with self.assertRaises(TransactionError):
            ctl.admit(plan(2))
        self.drain_frame(ctl, ticket)
        snapshot = ctl.snapshot()
        self.assertTrue(snapshot.closed)
        self.assertTrue(snapshot.drained)
        self.assertEqual(snapshot.reserved_bytes, 0)
        self.assertEqual(snapshot.released_frames, 1)
        with self.assertRaises(RetiredTransaction):
            ctl.confirm_feedback(ticket)

    def test_retired_archive_loss_report_still_invalidates_run(self):
        ctl = controller()
        ticket, _, _ = self.ready(ctl, plan())
        self.drain_frame(ctl, ticket)
        with self.assertRaises(RetiredTransaction):
            ctl.archive_failed(ticket, "acknowledged archive disappeared")
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertEqual(ctl.snapshot().feedback_emitted, 1)

    def test_immutable_storage_and_digest_validation(self):
        with self.assertRaises(ValueError):
            PlanKey(1, bytearray(b"mutable"))
        with self.assertRaises(ValueError):
            HistoryState(b"H", "wrong")
        with self.assertRaises(FrozenInstanceError):
            plan().key.frame_index = 7


if __name__ == "__main__":
    unittest.main()

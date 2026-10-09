"""Offline correctness fixtures for the bounded identity/event join."""

from dataclasses import FrozenInstanceError
import hashlib
import threading
import unittest

from . import contract as C
from .join import (
    BoundedTargetPredictionJoin, JoinCapacityExceeded, JoinClosed, JoinInvalid,
    JoinProtocolError, PairAlreadyTaken, UnknownIdentity,
)
from .test_contract import fixture


PREDICTION = b"opaque fixture prediction; external scorer must validate this"
PREDICTION_SHA = hashlib.sha256(PREDICTION).hexdigest()


def admitted_target(*, ticket=0, history="f"):
    metadata, seg, obj, expected = fixture()
    metadata["policy_identity"]["ticket_seq"] = ticket
    expected["expected_policy_identity"]["ticket_seq"] = ticket
    metadata["history_after_sha256"] = history * 64
    return C.admit_target(metadata=metadata, segmentation=seg, object_ignore=obj, **expected)


def reservation(target, prediction=PREDICTION):
    return (len(target.identity_bytes) + len(target.metadata_bytes)
            + len(target.segmentation_bytes) + len(target.object_ignore_bytes) + len(prediction))


class JoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = admitted_target()
        cls.equal_target = admitted_target()
        cls.other_ticket = admitted_target(ticket=1)
        cls.conflicting_target = admitted_target(history="a")

    def new_join(self, *, reserve=None, max_jobs=2, max_bytes=None):
        reserve = reservation(self.target) if reserve is None else reserve
        join = BoundedTargetPredictionJoin(max_jobs=max_jobs, max_bytes=max_bytes or 2 * reserve)
        self.assertTrue(join.register(self.target.identity_bytes, reserved_bytes=reserve))
        return join

    def put_both(self, join):
        join.put_target(self.target)
        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)

    def test_target_then_prediction_preserves_exact_immutable_storage(self):
        join = self.new_join()
        self.assertTrue(join.put_target(self.target))
        self.assertIsNone(join.wait_take(self.target.identity_bytes, timeout=0))
        self.assertTrue(join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA))
        pair = join.wait_take(self.target.identity_bytes, timeout=0)
        self.assertIs(pair.target, self.target)
        self.assertIs(pair.target.segmentation_bytes, self.target.segmentation_bytes)
        self.assertIs(pair.target.metadata, self.target.metadata)
        self.assertIs(pair.prediction_bytes, PREDICTION)
        with self.assertRaises(FrozenInstanceError):
            pair.prediction_bytes = b"mutated"
        with self.assertRaises(ValueError):
            pair.target.segmentation.setflags(write=True)
        join.release(pair.identity_bytes)
        self.assertTrue(join.snapshot().drained)

    def test_prediction_then_target_both_required(self):
        join = self.new_join()
        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
        self.assertIsNone(join.wait_take(self.target.identity_bytes, timeout=0))
        join.put_target(self.target)
        pair = join.wait_take(self.target.identity_bytes, timeout=0)
        self.assertEqual(pair.prediction_sha256, PREDICTION_SHA)
        join.release(pair.identity_bytes)

    def test_identical_active_duplicates_are_idempotent_including_after_take(self):
        join = self.new_join()
        self.assertFalse(join.register(self.target.identity_bytes, reserved_bytes=reservation(self.target)))
        self.put_both(join)
        before = join.snapshot()
        self.assertFalse(join.put_target(self.equal_target))
        self.assertFalse(join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA))
        pair = join.wait_take(self.target.identity_bytes, timeout=0)
        self.assertFalse(join.put_target(self.equal_target))
        self.assertFalse(join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA))
        self.assertEqual(join.snapshot().reserved_bytes, before.reserved_bytes)
        self.assertEqual(join.snapshot().retained_payload_bytes, before.retained_payload_bytes)
        self.assertTrue(join.snapshot().run_valid)
        join.release(pair.identity_bytes)

    def test_conflicting_target_does_not_replace_existing_target(self):
        join = self.new_join()
        join.put_target(self.target)
        self.assertEqual(self.target.identity_bytes, self.conflicting_target.identity_bytes)
        before = join.snapshot().retained_payload_bytes
        with self.assertRaises(JoinProtocolError):
            join.put_target(self.conflicting_target)
        self.assertEqual(join.snapshot().retained_payload_bytes, before)
        self.assertFalse(join.snapshot().run_valid)
        with self.assertRaises(JoinInvalid):
            join.wait_take(self.target.identity_bytes, timeout=0)
        join.abandon(self.target.identity_bytes, "conflicting source publication")

    def test_conflicting_prediction_does_not_replace_existing_payload(self):
        join = self.new_join()
        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
        changed = b"different"
        with self.assertRaises(JoinProtocolError):
            join.put_prediction(self.target.identity_bytes, changed, hashlib.sha256(changed).hexdigest())
        self.assertFalse(join.snapshot().run_valid)
        join.abandon(self.target.identity_bytes, "conflicting prediction")

    def test_exact_identity_includes_ticket_not_only_frame(self):
        self.assertEqual(self.target.metadata["capture_identity"]["frame_id"],
                         self.other_ticket.metadata["capture_identity"]["frame_id"])
        self.assertNotEqual(self.target.identity_bytes, self.other_ticket.identity_bytes)
        join = self.new_join()
        with self.assertRaises(UnknownIdentity):
            join.put_target(self.other_ticket)
        self.assertEqual(join.snapshot().pending_jobs, 1)
        self.assertEqual(join.snapshot().retained_payload_bytes, len(self.target.identity_bytes))
        join.abandon(self.target.identity_bytes, "foreign target rejected")

    def test_no_arrival_auto_registers_a_frame(self):
        for target_arrival in (True, False):
            with self.subTest(target_arrival=target_arrival):
                join = BoundedTargetPredictionJoin(max_jobs=1, max_bytes=reservation(self.target))
                with self.assertRaises(UnknownIdentity):
                    if target_arrival:
                        join.put_target(self.target)
                    else:
                        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                self.assertTrue(join.snapshot().drained)
                self.assertFalse(join.snapshot().run_valid)

    def test_unadmitted_target_and_mutable_identity_or_prediction_rejected(self):
        for operation in (lambda j: j.put_target({"already_validated": True}),
                          lambda j: j.put_prediction(self.target.identity_bytes, bytearray(PREDICTION), PREDICTION_SHA),
                          lambda j: j.put_prediction(bytearray(self.target.identity_bytes), PREDICTION, PREDICTION_SHA),
                          lambda j: j.register(memoryview(self.target.identity_bytes), reserved_bytes=reservation(self.target))):
            with self.subTest(operation=operation):
                join = self.new_join()
                with self.assertRaises(JoinProtocolError):
                    operation(join)
                self.assertFalse(join.snapshot().run_valid)
                self.assertEqual(join.snapshot().retained_payload_bytes, len(self.target.identity_bytes))
                join.abandon(self.target.identity_bytes, "malformed ownership handoff")

    def test_prediction_checksum_is_required_but_not_numerical_admission(self):
        join = self.new_join()
        with self.assertRaises(JoinProtocolError):
            join.put_prediction(self.target.identity_bytes, PREDICTION, "0" * 64)
        self.assertFalse(join.snapshot().run_valid)
        join.abandon(self.target.identity_bytes, "bad checksum")
        # Deliberately not a numerical prediction record; this handoff promises
        # bytes/integrity only, so the external scorer must reject it if needed.
        join2 = self.new_join()
        self.put_both(join2)
        self.assertEqual(join2.wait_take(self.target.identity_bytes, timeout=0).prediction_bytes, PREDICTION)
        join2.release(self.target.identity_bytes)

    def test_registration_job_and_byte_caps_never_evict(self):
        reserve = reservation(self.target)
        for max_jobs, max_bytes in ((1, 2 * reserve), (2, reserve)):
            with self.subTest(max_jobs=max_jobs, max_bytes=max_bytes):
                join = self.new_join(max_jobs=max_jobs, max_bytes=max_bytes)
                self.put_both(join)
                pair = join.wait_take(self.target.identity_bytes, timeout=0)
                with self.assertRaises(JoinCapacityExceeded):
                    join.register(self.other_ticket.identity_bytes, reserved_bytes=reservation(self.other_ticket))
                self.assertEqual(join.snapshot().pending_jobs, 1)
                self.assertEqual(join.snapshot().reserved_bytes, reserve)
                join.release(pair.identity_bytes)
                self.assertTrue(join.snapshot().drained)
                self.assertFalse(join.snapshot().run_valid)

    def test_slot_budget_checks_metadata_two_masks_prediction_and_identity(self):
        for prediction_first in (True, False):
            with self.subTest(prediction_first=prediction_first):
                join = self.new_join(reserve=reservation(self.target) - 1)
                if prediction_first:
                    join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                    with self.assertRaises(JoinCapacityExceeded):
                        join.put_target(self.target)
                else:
                    join.put_target(self.target)
                    with self.assertRaises(JoinCapacityExceeded):
                        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                self.assertFalse(join.snapshot().run_valid)
                self.assertLessEqual(join.snapshot().retained_payload_bytes, join.snapshot().reserved_bytes)
                join.abandon(self.target.identity_bytes, "reservation underestimated")
                self.assertEqual(join.snapshot().reserved_bytes, 0)
                self.assertEqual(join.snapshot().retained_payload_bytes, 0)

    def test_conflicting_reservation_and_impossible_minimum_rejected(self):
        join = self.new_join()
        with self.assertRaises(JoinProtocolError):
            join.register(self.target.identity_bytes, reserved_bytes=reservation(self.target) + 1)
        join.abandon(self.target.identity_bytes, "reservation conflict")
        join2 = BoundedTargetPredictionJoin(max_jobs=1, max_bytes=reservation(self.target))
        with self.assertRaises(JoinProtocolError):
            join2.register(self.target.identity_bytes, reserved_bytes=2 * C.MASK_BYTES)
        self.assertTrue(join2.snapshot().drained)

    def test_timeout_does_not_drop_target_prediction_or_reservation(self):
        for target_first in (True, False):
            with self.subTest(target_first=target_first):
                join = self.new_join()
                if target_first:
                    join.put_target(self.target)
                else:
                    join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                before = join.snapshot()
                self.assertIsNone(join.wait_take(self.target.identity_bytes, timeout=.001))
                self.assertEqual(join.snapshot(), before)
                if target_first:
                    join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                else:
                    join.put_target(self.target)
                self.assertIsNotNone(join.wait_take(self.target.identity_bytes, timeout=0))
                join.release(self.target.identity_bytes)

    def test_waiter_wakes_when_second_side_arrives(self):
        join = self.new_join()
        join.put_target(self.target)
        started = threading.Event()
        results, failures = [], []

        def consumer():
            started.set()
            try:
                results.append(join.wait_take(self.target.identity_bytes, timeout=2))
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=consumer, daemon=True)
        worker.start()
        self.assertTrue(started.wait(2))
        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].target, self.target)
        join.release(self.target.identity_bytes)

    def test_pair_taken_exactly_once_by_competing_consumers(self):
        join = self.new_join()
        self.put_both(join)
        barrier = threading.Barrier(3, timeout=2)
        results, failures = [], []

        def consumer():
            try:
                barrier.wait()
                results.append(join.wait_take(self.target.identity_bytes, timeout=2))
            except BaseException as exc:
                failures.append(exc)

        workers = [threading.Thread(target=consumer, daemon=True) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(2)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], PairAlreadyTaken)
        self.assertEqual(join.snapshot().pairs_taken, 1)
        self.assertTrue(join.snapshot().run_valid)
        join.release(self.target.identity_bytes)

    def test_release_retains_until_explicit_consumer_ack_and_rejects_late_arrivals(self):
        for target_arrival in (True, False):
            with self.subTest(target_arrival=target_arrival):
                join = self.new_join()
                self.put_both(join)
                pair = join.wait_take(self.target.identity_bytes, timeout=0)
                self.assertEqual(join.snapshot().reserved_bytes, reservation(self.target))
                join.release(pair.identity_bytes)
                self.assertEqual(join.snapshot().reserved_bytes, 0)
                self.assertEqual(join.snapshot().retained_payload_bytes, 0)
                with self.assertRaises(UnknownIdentity):
                    if target_arrival:
                        join.put_target(self.target)
                    else:
                        join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
                self.assertTrue(join.snapshot().drained)

    def test_release_before_pair_take_cannot_silently_drop_accepted_data(self):
        join = self.new_join()
        join.put_target(self.target)
        with self.assertRaises(JoinProtocolError):
            join.release(self.target.identity_bytes)
        self.assertEqual(join.snapshot().pending_jobs, 1)
        join.abandon(self.target.identity_bytes, "explicit failure cleanup")
        self.assertTrue(join.snapshot().drained)
        self.assertFalse(join.snapshot().run_valid)

    def test_close_wakes_waiter_and_retains_reservation_until_explicit_abandonment(self):
        join = self.new_join()
        join.put_target(self.target)
        started = threading.Event()
        failures = []

        def consumer():
            started.set()
            try:
                join.wait_take(self.target.identity_bytes, timeout=2)
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=consumer, daemon=True)
        worker.start()
        self.assertTrue(started.wait(2))
        join.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], JoinClosed)
        self.assertTrue(join.snapshot().closed)
        self.assertEqual(join.snapshot().pending_jobs, 1)
        self.assertEqual(join.snapshot().reserved_bytes, reservation(self.target))
        with self.assertRaises(JoinClosed):
            join.put_prediction(self.target.identity_bytes, PREDICTION, PREDICTION_SHA)
        with self.assertRaises(JoinClosed):
            join.register(self.other_ticket.identity_bytes, reserved_bytes=reservation(self.other_ticket))
        join.abandon(self.target.identity_bytes, "closed with missing prediction")
        self.assertTrue(join.snapshot().drained)

    def test_close_preserves_already_joined_pair_for_take_and_release(self):
        join = self.new_join()
        self.put_both(join)
        join.close()
        self.assertIsNotNone(join.wait_take(self.target.identity_bytes, timeout=0))
        join.release(self.target.identity_bytes)
        self.assertTrue(join.snapshot().run_valid)
        self.assertTrue(join.snapshot().drained)


if __name__ == "__main__":
    unittest.main()

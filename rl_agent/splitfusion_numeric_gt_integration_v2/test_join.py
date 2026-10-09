"""Actual capability join, both arrival orders and bounded concurrent wakeups."""
import hashlib
import threading
import unittest

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from . import ready as R, scoring as S, join as J
from .test_ready import fixture


RESERVE = 4 * 1024 * 1024


def inputs(scorer, *, label=0):
    payload, expected = fixture()
    target = R.admit_payload(payload, **expected)
    labels = np.full((720, 1280), label, np.uint8)
    records = []
    identity = dict(gt_identity=C.thaw(target.metadata["capture_identity"]),
        action_identity=C.thaw(target.metadata["action_identity"]),
        capture_timestamp_ns=target.metadata["capture_identity"]["capture_timestamp_ns"],
        prediction_digest=hashlib.sha256(labels.tobytes() + b"[]").hexdigest())
    prediction = S.admit_prediction(scorer, identity=identity, labels=labels, records=records,
        prediction_contract=C.thaw(target.metadata["prediction_contract"]))
    return target, prediction


class JoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = S.NumericalScorer()
        cls.target, cls.prediction = inputs(cls.scorer)

    def make(self, *, reserve=RESERVE):
        value = J.BoundedTargetPredictionJoin(self.scorer, max_jobs=2, max_bytes=2 * RESERVE)
        value.register(self.target.identity_bytes, reserved_bytes=reserve)
        return value

    def waiter(self, join):
        started = threading.Event()
        results = []
        def take():
            started.set()
            try:
                results.append(join.wait_take(self.target.identity_bytes, timeout=2))
            except Exception as error:
                results.append(error)
        thread = threading.Thread(target=take)
        thread.start()
        self.assertTrue(started.wait(1))
        return thread, results

    def test_both_arrival_orders_wake_same_registered_pair(self):
        for first in ("target", "prediction"):
            with self.subTest(first=first):
                value = self.make()
                thread, results = self.waiter(value)
                if first == "target":
                    value.put_target(self.target)
                else:
                    value.put_prediction(self.target.identity_bytes, self.prediction)
                self.assertIsNone(value.wait_take(self.target.identity_bytes, timeout=0))
                if first == "target":
                    value.put_prediction(self.target.identity_bytes, self.prediction)
                else:
                    value.put_target(self.target)
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(results), 1)
                pair = results[0]
                self.assertIs(pair.target, self.target)
                self.assertIs(pair.prediction, self.prediction)
                frame = S.join_admitted_inputs(self.scorer, target=pair.target, prediction=pair.prediction)
                self.assertEqual(self.scorer.score(frame).measurement["frame_id"], 10)
                value.release(self.target.identity_bytes)
                self.assertEqual(value.snapshot()["reserved_bytes"], 0)

    def test_timeout_keeps_identity_payload_and_reservation(self):
        value = self.make()
        value.put_target(self.target)
        before = value.snapshot()
        self.assertIsNone(value.wait_take(self.target.identity_bytes, timeout=.001))
        self.assertEqual(value.snapshot(), before)
        value.put_prediction(self.target.identity_bytes, self.prediction)
        self.assertIsNotNone(value.wait_take(self.target.identity_bytes, timeout=0))

    def test_close_wakes_waiter_and_retains_unmatched_target(self):
        value = self.make()
        value.put_target(self.target)
        thread, results = self.waiter(value)
        value.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(results[0], J.JoinClosed)
        self.assertEqual(value.snapshot()["pending_jobs"], 1)
        self.assertEqual(value.snapshot()["reserved_bytes"], RESERVE)
        value.abandon(self.target.identity_bytes, "closed without matching prediction")
        self.assertEqual(value.snapshot()["pending_jobs"], 0)
        self.assertFalse(value.snapshot()["valid"])

    def test_arrival_never_auto_registers(self):
        value = J.BoundedTargetPredictionJoin(self.scorer, max_jobs=1, max_bytes=RESERVE)
        with self.assertRaisesRegex(J.JoinProtocolError, "unregistered"):
            value.put_target(self.target)
        self.assertEqual(value.snapshot()["pending_jobs"], 0)

    def test_private_capabilities_and_exact_scorer_owner_required(self):
        value = self.make()
        with self.assertRaisesRegex(J.JoinProtocolError, "actual ready admission"):
            value.put_target({"already_validated": True})
        other = S.NumericalScorer()
        _, foreign = inputs(other)
        value = self.make()
        with self.assertRaisesRegex(J.JoinProtocolError, "scorer's actual admission"):
            value.put_prediction(self.target.identity_bytes, foreign)
        value = self.make()
        with self.assertRaises(J.JoinProtocolError):
            value.put_prediction(self.target.identity_bytes, b"already validated")

    def test_duplicate_identical_inputs_are_idempotent_not_second_work(self):
        value = self.make()
        self.assertFalse(value.register(self.target.identity_bytes, reserved_bytes=RESERVE))
        self.assertTrue(value.put_target(self.target))
        self.assertTrue(value.put_prediction(self.target.identity_bytes, self.prediction))
        before = value.snapshot()
        self.assertFalse(value.put_target(self.target))
        self.assertFalse(value.put_prediction(self.target.identity_bytes, self.prediction))
        self.assertEqual(value.snapshot(), before)
        value.wait_take(self.target.identity_bytes, timeout=0)
        with self.assertRaises(J.PairAlreadyTaken):
            value.wait_take(self.target.identity_bytes, timeout=0)

    def test_conflicting_admitted_prediction_invalidates_and_wakes_waiter(self):
        value = self.make()
        value.put_prediction(self.target.identity_bytes, self.prediction)
        thread, results = self.waiter(value)
        _, conflict = inputs(self.scorer, label=1)
        with self.assertRaisesRegex(J.JoinProtocolError, "conflicting duplicate prediction"):
            value.put_prediction(self.target.identity_bytes, conflict)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(results[0], J.JoinInvalid)
        self.assertEqual(value.snapshot()["pending_jobs"], 1)

    def test_payload_capacity_fault_does_not_evict_or_replace(self):
        value = self.make(reserve=len(self.target.identity_bytes) + 2 * C.MASK_BYTES)
        with self.assertRaisesRegex(J.JoinCapacityExceeded, "payload exceeds"):
            value.put_target(self.target)
        self.assertEqual(value.snapshot()["pending_jobs"], 1)
        self.assertEqual(value.snapshot()["retained_payload_bytes"], len(self.target.identity_bytes))
        value.abandon(self.target.identity_bytes, "reservation insufficient")

    def test_job_reservation_capacity_rejects_without_eviction(self):
        value = J.BoundedTargetPredictionJoin(self.scorer, max_jobs=1, max_bytes=2 * RESERVE)
        value.register(self.target.identity_bytes, reserved_bytes=RESERVE)
        next_identity = C._parse(self.target.identity_bytes)
        next_identity["capture_identity"]["frame_id"] += 1
        next_identity["action_identity"]["frame_id"] += 1
        with self.assertRaises(J.JoinCapacityExceeded):
            value.register(C.canonical(next_identity), reserved_bytes=RESERVE)
        self.assertEqual(value.snapshot()["pending_jobs"], 1)

    def test_retired_identity_cannot_be_reregistered_or_receive_late_data(self):
        for late in ("register", "prediction"):
            with self.subTest(late=late):
                value = self.make()
                value.put_target(self.target)
                value.put_prediction(self.target.identity_bytes, self.prediction)
                value.wait_take(self.target.identity_bytes, timeout=0)
                value.release(self.target.identity_bytes)
                with self.assertRaises(J.JoinProtocolError):
                    if late == "register":
                        value.register(self.target.identity_bytes, reserved_bytes=RESERVE)
                    else:
                        value.put_prediction(self.target.identity_bytes, self.prediction)
                self.assertEqual(value.snapshot()["pending_jobs"], 0)

    def test_full_executed_action_disagreement_rejected_before_pair(self):
        value = J.BoundedTargetPredictionJoin(self.scorer, max_jobs=1, max_bytes=RESERVE)
        key = C._parse(self.target.identity_bytes)
        key["action_identity"]["q_e4"] += 1
        identity = C.canonical(key)
        value.register(identity, reserved_bytes=RESERVE)
        with self.assertRaisesRegex(J.JoinProtocolError, "full identity/calibration"):
            value.put_prediction(identity, self.prediction)

    def test_release_without_take_is_not_implicit_cancellation(self):
        value = self.make()
        with self.assertRaisesRegex(J.JoinProtocolError, "untaken pair"):
            value.release(self.target.identity_bytes)
        self.assertEqual(value.snapshot()["pending_jobs"], 1)
        value.abandon(self.target.identity_bytes, "explicit failure cleanup")
        self.assertEqual(value.snapshot()["pending_jobs"], 0)

    def test_explicit_session_rollover_coexists_but_cannot_cross_join(self):
        value = self.make()
        second = C._parse(self.target.identity_bytes)
        second["capture_identity"]["frame_id"] = 11
        second["action_identity"].update(frame_id=11, session_uuid="explicit-next-session")
        second["frame_context"]["frame_id"] = 11
        second["plan_sha256"] = "e" * 64
        key = C.canonical(second)
        self.assertTrue(value.register(key, reserved_bytes=RESERVE))
        labels = np.zeros((720, 1280), np.uint8)
        identity = dict(gt_identity=second["capture_identity"], action_identity=second["action_identity"],
            capture_timestamp_ns=second["capture_identity"]["capture_timestamp_ns"],
            prediction_digest=hashlib.sha256(labels.tobytes() + b"[]").hexdigest())
        prediction = S.admit_prediction(self.scorer, identity=identity, labels=labels, records=[],
            prediction_contract=second["prediction_contract"])
        value.put_target(self.target)
        value.put_prediction(key, prediction)
        self.assertEqual(value.snapshot()["pending_jobs"], 2)
        self.assertIsNone(value.wait_take(key, timeout=0))
        self.assertIsNone(value.wait_take(self.target.identity_bytes, timeout=0))
        with self.assertRaisesRegex(J.JoinProtocolError, "full identity/calibration"):
            value.put_prediction(self.target.identity_bytes, prediction)


if __name__ == "__main__":
    unittest.main()

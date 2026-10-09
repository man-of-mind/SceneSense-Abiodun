"""Compose the actual candidate contracts offline, without a fake live claim.

Numerical fixtures are synthetic; these tests prove lifecycle/identity joining,
not producer geometry, real tracker cloning, prediction parsing or quality.
"""
import unittest

from . import contract as C
from .join import BoundedTargetPredictionJoin
from .test_contract import fixture
from .transactions import FramePlan, PlanKey, PlanKind, TentativeResult, TransactionController, WireBinding


class CompositionTests(unittest.TestCase):
    def prepare(self):
        meta, segmentation, ignore, expected = fixture()
        initial, next_history = b"accepted-history-0", b"accepted-history-1"
        meta.update(history_before_sha256=C.sha256(initial), history_after_sha256=C.sha256(next_history))
        target = C.admit_target(metadata=meta, segmentation=segmentation, object_ignore=ignore, **expected)
        identity = target.identity_bytes
        key = PlanKey(meta["capture_identity"]["frame_id"], identity)
        wire = WireBinding(key, C.canonical(dict(policy=meta["policy_identity"], plan=meta["plan_sha256"])))
        plan = FramePlan(key, wire, PlanKind.ACCEPTED, b"immutable-owned-source-handle", 4 * 1024 * 1024)
        ctl = TransactionController(max_jobs=2, max_bytes=8 * 1024 * 1024, max_history_bytes=1024,
                                    initial_history=initial)
        join = BoundedTargetPredictionJoin(max_jobs=2, max_bytes=4 * 1024 * 1024)
        ticket = ctl.admit(plan)
        join.register(identity, reserved_bytes=2 * 1024 * 1024)
        lease = ctl.claim_next()
        payload = C.encode_target(target, ready_raw_ns=100, ready_wall_ns=200)
        result = TentativeResult.for_lease(lease, next_history, payload)
        admitted = {}
        def validate(result):
            decoded = C.decode_target(result.numerical_payload, **expected).target
            C.require(decoded.identity_bytes == result.plan_key.canonical_identity, "plan/target identity differs")
            C.require(decoded.metadata["history_before_sha256"] == result.base_history_sha256
                and decoded.metadata["history_after_sha256"] == result.next_history.sha256,
                "target/history receipt differs")
            admitted["target"] = decoded
        return ctl, join, ticket, lease, result, plan, validate, admitted

    def test_both_send_orders_and_both_prediction_orders(self):
        for send_first in (True, False):
            for prediction_first in (True, False):
                with self.subTest(send_first=send_first, prediction_first=prediction_first):
                    ctl, join, ticket, lease, result, plan, validate, admitted = self.prepare()
                    prediction = b"opaque-prediction-fixture-not-a-validated-numerical-record"
                    if prediction_first:
                        join.put_prediction(plan.key.canonical_identity, prediction, C.sha256(prediction))
                        self.assertIsNone(join.wait_take(plan.key.canonical_identity, timeout=0))
                    if send_first:
                        ctl.confirm_send(ticket, plan.expected_wire)
                    ctl.complete_computation(lease, result)
                    ctl.validate_computation(ticket, validate)
                    if not send_first:
                        self.assertEqual(ctl.snapshot().history_version, 0)
                        self.assertIsNone(ctl.take_ready())
                        ctl.confirm_send(ticket, plan.expected_wire)
                    ready = ctl.take_ready()
                    self.assertEqual(ready.result, result)
                    join.put_target(admitted["target"])
                    if not prediction_first:
                        self.assertIsNone(join.wait_take(plan.key.canonical_identity, timeout=0))
                        join.put_prediction(plan.key.canonical_identity, prediction, C.sha256(prediction))
                    pair = join.wait_take(plan.key.canonical_identity, timeout=0)
                    self.assertEqual(pair.target.semantic_sha256, admitted["target"].semantic_sha256)
                    self.assertIsNone(ctl.take_ready())
                    ctl.complete_scoring(ticket)
                    ctl.confirm_feedback(ticket)
                    self.assertEqual(ctl.snapshot().pending_jobs, 1)
                    self.assertEqual(join.snapshot().pending_jobs, 1)
                    ctl.complete_archive(ticket)
                    join.release(plan.key.canonical_identity)
                    self.assertTrue(ctl.snapshot().drained and join.snapshot().drained)
                    self.assertEqual(ctl.snapshot().committed_frames, 1)
                    self.assertEqual(ctl.snapshot().feedback_emitted, 1)

    def test_unsent_failure_never_publishes_or_changes_history(self):
        ctl, join, ticket, lease, result, plan, validate, admitted = self.prepare()
        ctl.complete_computation(lease, result)
        ctl.validate_computation(ticket, validate)
        ctl.send_failed(ticket, "synthetic transport failure")
        join.abandon(plan.key.canonical_identity, "send failed before target publication")
        self.assertIsNone(ctl.take_ready())
        self.assertEqual(ctl.snapshot().history_version, 0)
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertTrue(ctl.snapshot().drained and join.snapshot().drained)

    def test_history_digest_mismatch_rejected_before_commit(self):
        ctl, join, ticket, lease, result, plan, validate, admitted = self.prepare()
        invalid = TentativeResult.for_lease(lease, b"foreign-history", result.numerical_payload)
        ctl.complete_computation(lease, invalid)
        ctl.confirm_send(ticket, plan.expected_wire)
        with self.assertRaises(C.ContractError):
            ctl.validate_computation(ticket, validate)
        join.abandon(plan.key.canonical_identity, "history receipt differs")
        self.assertEqual(ctl.snapshot().history_version, 0)
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertTrue(ctl.snapshot().drained)

    def test_late_archive_failure_does_not_retract_feedback(self):
        ctl, join, ticket, lease, result, plan, validate, admitted = self.prepare()
        ctl.confirm_send(ticket, plan.expected_wire)
        ctl.complete_computation(lease, result); ctl.validate_computation(ticket, validate)
        ctl.take_ready(); join.put_target(admitted["target"])
        prediction = b"opaque-fixture"
        join.put_prediction(plan.key.canonical_identity, prediction, C.sha256(prediction))
        join.wait_take(plan.key.canonical_identity, timeout=0)
        ctl.complete_scoring(ticket); ctl.confirm_feedback(ticket)
        ctl.archive_failed(ticket, "synthetic full disk")
        join.abandon(plan.key.canonical_identity, "archive failed after feedback")
        self.assertEqual(ctl.snapshot().feedback_emitted, 1)
        self.assertFalse(ctl.snapshot().run_valid)
        self.assertTrue(ctl.snapshot().drained and join.snapshot().drained)


if __name__ == "__main__":
    unittest.main()

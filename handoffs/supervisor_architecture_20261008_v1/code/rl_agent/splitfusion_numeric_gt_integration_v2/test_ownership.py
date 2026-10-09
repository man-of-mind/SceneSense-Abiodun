"""Real synthetic producer transactions; no server, GPU, or benchmark."""
from contextlib import contextmanager
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch

import carla
import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as F
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_dispatch_v1 import frame_context as FC
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS, payload as P
from rl_agent.splitfusion_numeric_gt_producer_v1.test_numeric_producer import ARGS, upright, frame_products
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from . import ownership as O, ready as R


RESERVE = 32 * 1024 * 1024


def plan_for(kwargs):
    capture = kwargs["identity"]
    action = dict(frame_id=capture["frame_id"], session_uuid="session", decision_seq=0,
        tensor_seq=1, ticket_seq=0, controller_lineage_sha256="1" * 64,
        execution_bundle_sha256="2" * 64, mode_id=11, q_e4=2999,
        anchor_action_id=None, reward_requested=True, frame_kind="POLICY_DECISION")
    return C.canonical(dict(capture=capture, action=action,
                            frame_context=PO.context_document(context_for(kwargs)))), action


@contextmanager
def owned_fixture(*, jobs=2, budget=2 * RESERVE):
    with F.SourceGeometryTests().fixture() as (directory, kwargs):
        kwargs = upright(kwargs)
        pose = carla.Transform(carla.Location(1.8, 0., 1.55), carla.Rotation(pitch=-4.))
        camera = np.asarray(pose.get_matrix())
        kwargs["camera_matrix"], kwargs["camera_inverse"] = camera, np.linalg.inv(camera)
        kwargs["rgb"]["camera_matrix"] = camera.tolist()
        for name in ("semantic_image", "depth_image"):
            kwargs[name].transform = pose
        producer = NS.NumericSourceProducer(static_catalogue_dir=directory / "catalogue", **ARGS)
        owner = O.ProducerOwner(producer, max_retained_jobs=jobs, max_retained_bytes=budget)
        try:
            yield owner, producer, kwargs
        finally:
            owner.close(abandon=True)


def context_for(kwargs):
    identity = kwargs["identity"]
    return FC.build_frame_context_v1(stream_id=identity["stream_id"], frame_id=identity["frame_id"],
        sequence_id=1, capture_timestamp_ns=identity["capture_timestamp_ns"],
        ego_world_x=0., ego_world_y=0., ego_world_z=0.,
        ego_world_pitch=0., ego_world_yaw=0., ego_world_roll=0.)


def ready_for(owner, token, kwargs, *, admit=True):
    staged = owner._staged_for_assembly(token)
    target, avo, payload = frame_products(staged, kwargs)
    owner.stage_products(token, target=target, avo=avo, payload=payload)
    plan, action = plan_for(kwargs)
    ready = R.admit_payload(payload, expected_capture_identity=kwargs["identity"],
        expected_action_identity=action, frame_context=context_for(kwargs),
        plan_sha256=C.sha256(plan), run_binding="4" * 64, producer_binding=payload.binding_sha256,
        prediction_contract=PA.native_prediction_contract().document())
    if admit:
        owner.set_ready(token, ready)
    return ready


def advance(kwargs):
    frame, stamp = kwargs["identity"]["frame_id"] + 1, kwargs["rgb"]["timestamp"] + .05
    kwargs["identity"].update(frame_id=frame, capture_timestamp_ns=int(stamp * 1e9))
    for name in ("semantic_image", "depth_image"):
        kwargs[name].frame, kwargs[name].timestamp = frame, stamp
    kwargs["rgb"].update(frame_id=frame, timestamp=stamp)
    kwargs["rgb"]["radar_callback"].update(frame_id=frame, timestamp=stamp)
    kwargs["frozen_world"].frozen_frame_id = frame
    kwargs["frozen_world"].frozen_timestamp_s = stamp
    kwargs["dynamic_snapshot_evidence"]["frame_id"] = frame


class ProducerOwnershipTests(unittest.TestCase):
    def compute(self, owner, kwargs, *, reserve=RESERVE):
        return owner.compute(plan_for(kwargs)[0], kwargs, reserved_bytes=reserve)

    def drain(self, owner, token):
        owner.complete_scoring(token)
        owner.confirm_feedback(token)
        owner.complete_archive(token)

    def test_numerical_admission_then_exact_send_commits_real_history_once(self):
        with owned_fixture() as (owner, producer, kwargs):
            before = producer.history_sha256()
            with patch.object(producer, "commit", wraps=producer.commit) as commit:
                token = self.compute(owner, kwargs)
                ready = ready_for(owner, token, kwargs)
                self.assertEqual(producer.processed_frames, 0)
                result = owner.confirm_send(token, plan_for(kwargs)[0])
                self.assertEqual(result["history_before_sha256"], before)
                self.assertEqual(result["history_after_sha256"], ready.metadata["history_after_sha256"])
                self.assertEqual(producer.history_sha256(), result["history_after_sha256"])
                self.assertIs(owner.ready_target(token), ready)
                owner.confirm_send(token, plan_for(kwargs)[0])
                self.assertEqual(commit.call_count, 1)
                self.drain(owner, token)
                self.assertEqual(owner.snapshot()["retained_jobs"], 0)
                self.assertTrue(owner.snapshot()["valid"])

    def test_send_confirmation_before_admission_cannot_commit_or_publish(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            self.assertIsNone(owner.confirm_send(token, plan_for(kwargs)[0]))
            self.assertEqual(producer.processed_frames, 0)
            ready_for(owner, token, kwargs)
            self.assertEqual(producer.processed_frames, 1)
            self.drain(owner, token)

    def test_cancel_rolls_back_real_tracker_conflicts_and_registration(self):
        with owned_fixture() as (owner, producer, kwargs):
            before = producer.history_sha256()
            owner.register_population([1, 2, 3], "e" * 64, before_frame_id=100)
            with patch.object(producer, "cancel", wraps=producer.cancel) as cancel:
                token = self.compute(owner, kwargs)
                self.assertEqual(owner._staged_for_assembly(token).population, frozenset((1, 2, 3)))
                owner.cancel(token, "send was cancelled before execution")
                self.assertEqual(cancel.call_count, 1)
                self.assertEqual(producer.history_sha256(), before)
                self.assertEqual(producer.tracker._ages, {})
                self.assertEqual(producer.conflict_state.frame_count, 0)
                self.assertEqual(producer.population, frozenset((1, 2)))
                self.assertIsNone(producer.pending_registration)
                advance(kwargs)
                owner.register_population([1, 2, 3], "f" * 64, before_frame_id=101)
                later = self.compute(owner, kwargs)
                ready_for(owner, later, kwargs)
                owner.confirm_send(later, plan_for(kwargs)[0])
                self.assertEqual(producer.population_evidence_sha256, "f" * 64)
                self.drain(owner, later)

    def test_hold_fallback_displaced_never_call_actual_producer(self):
        with owned_fixture() as (owner, producer, kwargs), patch.object(producer, "compute", wraps=producer.compute) as compute:
            for kind in (O.PlanKind.HOLD, O.PlanKind.FALLBACK, O.PlanKind.DISPLACED):
                with self.assertRaisesRegex(O.OwnershipError, "cannot enter"):
                    owner.compute(plan_for(kwargs)[0], kwargs, reserved_bytes=RESERVE, kind=kind)
            self.assertEqual(compute.call_count, 0)
            self.assertEqual(owner.snapshot()["reserved_bytes"], 0)
            self.assertTrue(owner.snapshot()["valid"])

    def test_exact_full_send_identity_not_only_frame_controls_commit(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            foreign = C._parse(plan_for(kwargs)[0])
            foreign["action"]["q_e4"] += 1
            with self.assertRaisesRegex(O.OwnershipError, "exact full plan"):
                owner.confirm_send(token, C.canonical(foreign))
            self.assertEqual(producer.processed_frames, 0)
            owner.cancel(token, "foreign send result")
            self.assertFalse(owner.snapshot()["valid"])

    def test_only_issued_ready_capability_can_unlock_commit(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            owner.confirm_send(token, plan_for(kwargs)[0])
            with self.assertRaisesRegex(ValueError, "private admitted"):
                owner.set_ready(token, {"already_validated": True})
            self.assertEqual(producer.processed_frames, 0)
            owner.cancel(token, "untrusted target rejected")

    def test_capacity_reservation_rejects_before_compute_and_never_evicts(self):
        with owned_fixture(jobs=1) as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            owner.confirm_send(token, plan_for(kwargs)[0])
            advance(kwargs)
            with patch.object(producer, "compute", wraps=producer.compute) as compute:
                with self.assertRaises(O.CapacityError):
                    self.compute(owner, kwargs)
                self.assertEqual(compute.call_count, 0)
            self.assertEqual(owner.snapshot()["retained_jobs"], 1)
            self.assertIsNotNone(owner._archive_for_owner(token))
            self.drain(owner, token)
            self.assertEqual(owner.snapshot()["reserved_bytes"], 0)

    def test_actual_payload_overflow_cancels_real_pending_transaction(self):
        with owned_fixture() as (owner, producer, kwargs):
            before = producer.history_sha256()
            with patch.object(producer, "cancel", wraps=producer.cancel) as cancel:
                with self.assertRaisesRegex(O.CapacityError, "payload exceeds"):
                    self.compute(owner, kwargs, reserve=1024)
                self.assertEqual(cancel.call_count, 1)
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(owner.snapshot()["retained_jobs"], 0)
            self.assertFalse(owner.snapshot()["valid"])

    def test_all_three_consumer_terminals_required_to_release(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            owner.confirm_send(token, plan_for(kwargs)[0])
            owner.complete_archive(token)
            owner.complete_scoring(token)
            self.assertEqual(owner.snapshot()["retained_jobs"], 1)
            owner.confirm_feedback(token)
            self.assertEqual(owner.snapshot()["retained_jobs"], 0)
            self.assertEqual(producer.processed_frames, 1)

    def test_late_archive_failure_invalidates_run_without_rewriting_history(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            owner.confirm_send(token, plan_for(kwargs)[0])
            history = producer.history_sha256()
            owner.complete_scoring(token)
            owner.confirm_feedback(token)
            owner.consumer_failed(token, consumer="archive", reason="disk write failed")
            self.assertFalse(owner.snapshot()["valid"])
            self.assertEqual(owner.snapshot()["retained_jobs"], 0)
            self.assertEqual(producer.history_sha256(), history)

    def test_process_lease_and_same_thread_private_access_enforced(self):
        with owned_fixture() as (owner, producer, kwargs):
            with self.assertRaisesRegex(O.OwnershipError, "process-wide"):
                O.ProducerOwner(producer, max_retained_jobs=1, max_retained_bytes=RESERVE)
            token = self.compute(owner, kwargs)
            errors = []
            def foreign():
                try:
                    owner._staged_for_assembly(token)
                except O.OwnershipError as error:
                    errors.append(str(error))
            thread = threading.Thread(target=foreign)
            thread.start(); thread.join(10)
            self.assertEqual(len(errors), 1)
            self.assertIn("foreign thread", errors[0])
            owner.cancel(token, "foreign borrowing attempt")
            self.assertFalse(owner.snapshot()["valid"])

    def test_retained_caller_array_and_population_aliases_are_detached(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            staged = owner._staged_for_assembly(token)
            camera, radar = staged.camera_matrix.copy(), staged.radar.copy()
            population = list(staged.dynamic_snapshot_evidence["active_population_ids"])
            kwargs["camera_matrix"][:] = 999
            kwargs["radar_world_xyz"][:] = -999
            kwargs["dynamic_snapshot_evidence"]["active_population_ids"].clear()
            np.testing.assert_array_equal(staged.camera_matrix, camera)
            np.testing.assert_array_equal(staged.radar, radar)
            self.assertEqual(staged.dynamic_snapshot_evidence["active_population_ids"], population)
            with self.assertRaises(ValueError):
                staged.camera_matrix.setflags(write=True)
            owner.cancel(token, "alias ownership test complete")

    def test_commit_postcondition_failure_is_failstop_without_cancel_retry(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            actual = producer.commit
            def broken(staged):
                actual(staged)
                raise NS.StagingError("injected final history failure")
            with patch.object(producer, "commit", side_effect=broken) as commit, \
                 patch.object(producer, "cancel", wraps=producer.cancel) as cancel:
                with self.assertRaisesRegex(NS.StagingError, "final history"):
                    owner.confirm_send(token, plan_for(kwargs)[0])
                with self.assertRaises(O.OwnershipError):
                    owner.confirm_send(token, plan_for(kwargs)[0])
                owner.close(abandon=True)
                self.assertEqual(commit.call_count, 1)
                self.assertEqual(cancel.call_count, 0)
            self.assertFalse(owner.snapshot()["valid"])
            with self.assertRaisesRegex(O.OwnershipError, "failed producer"):
                O.ProducerOwner(producer, max_retained_jobs=1, max_retained_bytes=RESERVE)

    def test_cancel_postcondition_failure_never_retries_cancel_or_commit(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            before = owner._staged_for_assembly(token).history_before_sha256
            actual = producer.cancel
            def broken(staged):
                actual(staged)
                raise NS.StagingError("injected cancellation postcondition failure")
            with patch.object(producer, "cancel", side_effect=broken) as cancel:
                with self.assertRaisesRegex(NS.StagingError, "cancellation postcondition"):
                    owner.cancel(token, "unsent cancellation")
                owner.close(abandon=True)
                self.assertEqual(cancel.call_count, 1)
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(owner.snapshot()["commit_calls"], 0)
            self.assertFalse(owner.snapshot()["valid"])

    def test_numerical_avo_must_match_owned_payload_before_ready(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            staged = owner._staged_for_assembly(token)
            target, avo, payload = frame_products(staged, kwargs)
            foreign = replace(avo, avo={**avo.avo, "threshold": .5})
            with self.assertRaisesRegex(O.OwnershipError, "AVO differs"):
                owner.stage_products(token, target=target, avo=foreign, payload=payload)
            self.assertEqual(producer.processed_frames, 0)
            owner.cancel(token, "AVO ownership mismatch")

    def test_actual_ready_context_must_equal_full_plan_not_only_pose_tolerance(self):
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            staged = owner._staged_for_assembly(token)
            target, avo, payload = frame_products(staged, kwargs)
            owner.stage_products(token, target=target, avo=avo, payload=payload)
            plan, action = plan_for(kwargs)
            context = context_for(kwargs)
            nearby = replace(context, ego_world=replace(context.ego_world, x=1e-7))
            foreign = R.admit_payload(payload, expected_capture_identity=kwargs["identity"],
                expected_action_identity=action, frame_context=nearby,
                plan_sha256=C.sha256(plan), run_binding="4" * 64,
                producer_binding=payload.binding_sha256,
                prediction_contract=PA.native_prediction_contract().document())
            with self.assertRaisesRegex(O.OwnershipError, "full producer plan"):
                owner.set_ready(token, foreign)
            self.assertEqual(producer.processed_frames, 0)
            owner.cancel(token, "context differs from exact planned context")

    def test_foreign_token_and_next_compute_while_pending_failstop(self):
        for failure in ("foreign", "pending"):
            with self.subTest(failure=failure), owned_fixture() as (owner, producer, kwargs):
                token = self.compute(owner, kwargs)
                with self.assertRaises(O.OwnershipError):
                    if failure == "foreign":
                        owner._staged_for_assembly(O.ProducerToken(token.sequence, token._owner))
                    else:
                        self.compute(owner, kwargs)
                owner.cancel(token, "ownership protocol failed")
                self.assertFalse(owner.snapshot()["valid"])

    def test_source_validation_failure_has_no_commit_or_cancel_of_nonexistent_stage(self):
        with owned_fixture() as (owner, producer, kwargs):
            before = producer.history_sha256()
            kwargs["radar_world_xyz"][0, 0] = np.nan
            with self.assertRaises(NS.SourceGeometryError):
                self.compute(owner, kwargs)
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(owner.snapshot()["cancel_calls"], 0)
            self.assertEqual(owner.snapshot()["commit_calls"], 0)
            self.assertEqual(owner.snapshot()["retained_jobs"], 0)
            self.assertFalse(owner.snapshot()["valid"])

    def test_close_cancels_unsent_pending_and_refuses_unfinished_accepted(self):
        with owned_fixture() as (owner, producer, kwargs):
            before = producer.history_sha256()
            self.compute(owner, kwargs)
            owner.close()
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(owner.snapshot()["cancel_calls"], 1)
        with owned_fixture() as (owner, producer, kwargs):
            token = self.compute(owner, kwargs)
            ready_for(owner, token, kwargs)
            owner.confirm_send(token, plan_for(kwargs)[0])
            with self.assertRaisesRegex(O.OwnershipError, "before scoring"):
                owner.close()
            self.assertFalse(owner.snapshot()["valid"])
            self.assertFalse(owner.snapshot()["closed"])
            self.drain(owner, token)
            owner.close()
            self.assertTrue(owner.snapshot()["closed"])


if __name__ == "__main__":
    unittest.main()

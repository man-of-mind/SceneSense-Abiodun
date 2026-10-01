"""CPU-only tests for the isolated B-variant live adapters."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import branch_evidence_v1 as B
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import operational_ack_v1 as A
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_operational_ack_v1 import (
    identity,
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def manifest(variant: L.ActorVariant) -> L.BActorManifestV1:
    order = L.expected_feature_order(variant)
    return L.BActorManifestV1(
        variant=variant,
        feature_order=order,
        feature_count=len(order),
        feature_schema_sha256=L.feature_schema_sha256(variant, order),
        actor_boundary_sha256=sha(f"{variant.value}-actor"),
        weights_file_sha256=sha(f"{variant.value}-weights"),
        selected_seed=43,
        selected_update=10_000,
    )


class FakeActor:
    def __init__(self, boundary: str, *, mode_id: int = 6, q_e4: int = 6784):
        self.boundary = boundary
        self.mode_id = mode_id
        self.q_e4 = q_e4
        self.calls: list[tuple[float, ...]] = []

    def act_on_vector(self, values):
        self.calls.append(values)
        return {
            "mode_id": self.mode_id,
            "q_e4": self.q_e4,
            "actor_boundary_sha256": self.boundary,
        }


class ActorBindingTest(unittest.TestCase):
    def test_exact_orders_are_distinct_and_roundtrip(self) -> None:
        from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
            frozen_actor_loader_v1 as actor_loader,
        )

        run4b = manifest(L.ActorVariant.RUN4B)
        run5b = manifest(L.ActorVariant.RUN5B)
        self.assertEqual(run4b.feature_count, 20)
        self.assertEqual(run5b.feature_count, 21)
        self.assertEqual(run5b.feature_order[:20], run4b.feature_order)
        self.assertEqual(run4b.feature_order[19],
                         "prev_operational_success")
        self.assertEqual(run4b.feature_order,
                         actor_loader.RUN4B_FEATURE_ORDER)
        self.assertEqual(run5b.feature_order,
                         actor_loader.RUN5B_FEATURE_ORDER)
        self.assertEqual(run5b.feature_order[-1],
                         "effective_external_ul_snr_proxy_scaled")
        self.assertNotEqual(run4b.feature_schema_sha256,
                            run5b.feature_schema_sha256)
        self.assertEqual(L.BActorManifestV1.from_mapping(run4b.as_dict()), run4b)

    def test_old_qperc_state_and_cross_variant_manifest_are_refused(self) -> None:
        base = manifest(L.ActorVariant.RUN4B)
        legacy_success_order = (*base.feature_order[:-1], "prev_success")
        with self.assertRaisesRegex(L.ActorBindingError, "order"):
            L.BActorManifestV1(
                variant=L.ActorVariant.RUN4B,
                feature_order=legacy_success_order,
                feature_count=len(legacy_success_order),
                feature_schema_sha256=L.feature_schema_sha256(
                    L.ActorVariant.RUN4B, legacy_success_order),
                actor_boundary_sha256=base.actor_boundary_sha256,
                weights_file_sha256=base.weights_file_sha256,
                selected_seed=base.selected_seed,
                selected_update=base.selected_update,
            )
        old_order = (*base.feature_order[:17], "prev_quality_qperc",
                     *base.feature_order[17:])
        with self.assertRaisesRegex(L.ActorBindingError, "order"):
            L.BActorManifestV1(
                variant=L.ActorVariant.RUN4B,
                feature_order=old_order,
                feature_count=len(old_order),
                feature_schema_sha256=L.feature_schema_sha256(
                    L.ActorVariant.RUN4B, old_order),
                actor_boundary_sha256=base.actor_boundary_sha256,
                weights_file_sha256=base.weights_file_sha256,
                selected_seed=43,
                selected_update=10_000,
            )
        raw = base.as_dict()
        raw["variant"] = L.ActorVariant.RUN5B.value
        with self.assertRaises(L.ActorBindingError):
            L.BActorManifestV1.from_mapping(raw)

    def test_actor_is_called_once_on_exact_finite_tuple(self) -> None:
        binding = manifest(L.ActorVariant.RUN4B)
        actor = FakeActor(binding.actor_boundary_sha256)
        adapter = L.BoundFrozenActorV1(
            actor, binding,
            observed_actor_boundary_sha256=binding.actor_boundary_sha256)
        values = tuple(index / 20 for index in range(20))
        decision = adapter.act(values)
        self.assertEqual((decision.mode_id, decision.q_e4), (6, 6784))
        self.assertEqual(actor.calls, [values])
        self.assertEqual(decision.feature_schema_sha256,
                         binding.feature_schema_sha256)

    def test_width_nonfinite_boundary_and_foreign_decision_fail_closed(self) -> None:
        binding = manifest(L.ActorVariant.RUN5B)
        actor = FakeActor(binding.actor_boundary_sha256)
        with self.assertRaisesRegex(L.ActorBindingError, "differs"):
            L.BoundFrozenActorV1(
                actor, binding, observed_actor_boundary_sha256=sha("other"))
        adapter = L.BoundFrozenActorV1(
            actor, binding,
            observed_actor_boundary_sha256=binding.actor_boundary_sha256)
        with self.assertRaisesRegex(L.ActorBindingError, "tuple"):
            adapter.act(tuple(0.0 for _ in range(20)))
        values = [0.0] * 21
        values[4] = float("nan")
        with self.assertRaisesRegex(L.ActorBindingError, "finite"):
            adapter.act(tuple(values))
        bad = FakeActor(binding.actor_boundary_sha256, mode_id=12)
        with self.assertRaisesRegex(L.ActorBindingError, "mode_id"):
            L.BoundFrozenActorV1(
                bad, binding,
                observed_actor_boundary_sha256=binding.actor_boundary_sha256,
            ).act(tuple(0.0 for _ in range(21)))


def phase6_inputs(*, reward_requested: bool = True, anchor: bool = False):
    row = identity()
    action_id = 40 if anchor else None
    profile_id = "split_ae64_uint8_q3000" if anchor else None
    envelope = SimpleNamespace(
        reward_requested=reward_requested,
        frame_id=row.frame_id,
        tensor_seq=row.tensor_seq,
        capture_timestamp_ns=row.capture_timestamp_ns,
        mode_id=row.mode_id,
        q_e4=row.q_e4,
        keep_count=row.keep_count,
        anchor_action_id=action_id,
        session_uuid=row.session_uuid,
        controller_lineage_sha256=row.controller_lineage_sha256,
        decision_seq=row.decision_seq,
        ticket_seq=row.ticket_seq,
        execution_bundle_sha256=row.execution_bundle_sha256,
    )
    context = SimpleNamespace(
        frame_id=row.frame_id,
        sequence_id=row.tensor_seq,
        capture_timestamp_ns=row.capture_timestamp_ns,
        stream_id=row.stream_id,
    )
    profile = SimpleNamespace(
        mode_id=row.mode_id,
        q_e4=row.q_e4,
        keep_count=row.keep_count,
        action_id=action_id,
        profile_id=profile_id,
        execution_bundle_sha256=row.execution_bundle_sha256,
    )
    return envelope, context, profile


class Phase6IdentityAdapterTest(unittest.TestCase):
    def test_off_anchor_and_anchor_identity_are_exact(self) -> None:
        for anchored in (False, True):
            envelope, context, profile = phase6_inputs(anchor=anchored)
            row = L.identity_from_phase6(
                run_id="run4b_live_001", cell_id="fade_recovery",
                envelope=envelope, context=context, profile=profile)
            self.assertEqual((row.frame_id, row.tensor_seq, row.mode_id, row.q_e4),
                             (envelope.frame_id, envelope.tensor_seq,
                              envelope.mode_id, envelope.q_e4))
            self.assertEqual(row.anchor_action_id, envelope.anchor_action_id)
            self.assertEqual(row.profile_id,
                             profile.profile_id if anchored else None)

    def test_hold_refused_for_ack_but_admitted_for_map_identity(self) -> None:
        envelope, context, profile = phase6_inputs(reward_requested=False)
        with self.assertRaisesRegex(L.EdgeDispatchError, "decision frame"):
            L.identity_from_phase6(
                run_id="run4b_live_001", cell_id="fade_recovery",
                envelope=envelope, context=context, profile=profile)
        row = L.identity_from_phase6(
            run_id="run4b_live_001", cell_id="fade_recovery",
            envelope=envelope, context=context, profile=profile,
            require_operational_ack=False)
        self.assertEqual(row.frame_id, envelope.frame_id)

    def test_prediction_bytes_use_the_shared_postrun_bundle_codec(self) -> None:
        from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
            postrun_artifact_v1 as artifact,
        )

        row = identity()
        mask = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
        payload = L.encode_prediction_tail_output(
            identity=row,
            objects=({"class_name": "vehicle", "world_x": 1.0},),
            semantic_mask=mask,
        )
        decoded = artifact.decode_bundle(
            payload, expected_kind=artifact.PREDICTION_KIND,
            expected_identity=row)
        self.assertEqual(decoded.objects[0]["class_name"], "vehicle")
        np.testing.assert_array_equal(decoded.semantic_mask, mask)

    def test_cross_input_drift_is_refused(self) -> None:
        envelope, context, profile = phase6_inputs()
        context.sequence_id += 1
        with self.assertRaisesRegex(L.EdgeDispatchError, "tensor_seq"):
            L.identity_from_phase6(
                run_id="run4b_live_001", cell_id="fade_recovery",
                envelope=envelope, context=context, profile=profile)


def succeeded(code: str) -> L.BranchCallbackResultV1:
    return L.BranchCallbackResultV1(
        status=B.BranchStatus.SUCCEEDED,
        detail_code=code,
        evidence_sha256=sha(code),
    )


class TailDispatchTest(unittest.TestCase):
    def test_ack_is_observed_before_either_independent_branch(self) -> None:
        events: list[str] = []

        def ack_sender(packet: bytes):
            self.assertIsInstance(A.decode_ack(packet), A.TailOutputAckV1)
            events.append("ack")

        def map_callback(work):
            events.append("map")
            return succeeded("MAP_PUBLISHED")

        def prediction_callback(work):
            events.append("prediction")
            return succeeded("PREDICTION_RETAINED")

        dispatch = L.TailOutputDispatchV1(
            ack_sender=ack_sender, queue_depth=2,
            map_callback=map_callback,
            prediction_callback=prediction_callback)
        self.assertFalse(any(thread.name.startswith("run4b5b-")
                             for thread in threading.enumerate()))
        dispatch.start()
        receipt = dispatch.dispatch_decision(
            identity(), b"usable-tail-output", tail_ready_monotonic_raw_ns=10)
        dispatch.drain()
        dispatch.stop()
        self.assertEqual(events[0], "ack")
        self.assertCountEqual(events[1:], ["map", "prediction"])
        self.assertTrue(receipt.map_enqueued and receipt.prediction_enqueued)
        self.assertIs(
            dispatch.coordinator.result(identity(), B.Branch.MAP).status,
            B.BranchStatus.SUCCEEDED)
        self.assertIs(
            dispatch.coordinator.result(identity(), B.Branch.EVALUATION).status,
            B.BranchStatus.SUCCEEDED)

    def test_map_failure_does_not_delay_ack_or_cancel_prediction(self) -> None:
        ack_packets: list[bytes] = []

        def broken_map(_work):
            raise RuntimeError("map endpoint unavailable")

        dispatch = L.TailOutputDispatchV1(
            ack_sender=ack_packets.append, queue_depth=2,
            map_callback=broken_map,
            prediction_callback=lambda _work: succeeded("PREDICTION_RETAINED"))
        dispatch.start()
        dispatch.dispatch_decision(
            identity(), b"usable-tail-output", tail_ready_monotonic_raw_ns=10)
        dispatch.drain()
        dispatch.stop()
        self.assertEqual(len(ack_packets), 1)
        self.assertIsInstance(A.decode_ack(ack_packets[0]).identity,
                              A.FrameActionIdentityV1)
        self.assertIs(
            dispatch.coordinator.result(identity(), B.Branch.MAP).status,
            B.BranchStatus.FAILED)
        self.assertIs(
            dispatch.coordinator.result(identity(), B.Branch.EVALUATION).status,
            B.BranchStatus.SUCCEEDED)
        self.assertEqual(len(dispatch.worker_errors), 1)

    def test_map_only_hold_sends_no_ack_and_never_enters_prediction_branch(self) -> None:
        ack_packets: list[bytes] = []
        map_calls: list[int] = []
        prediction_calls: list[int] = []
        dispatch = L.TailOutputDispatchV1(
            ack_sender=ack_packets.append, queue_depth=2,
            map_callback=lambda work: (
                map_calls.append(work.identity.frame_id)
                or succeeded("MAP_PUBLISHED")),
            prediction_callback=lambda work: (
                prediction_calls.append(work.identity.frame_id)
                or succeeded("PREDICTION_RETAINED")),
        )
        dispatch.start()
        self.assertTrue(dispatch.dispatch_map_only(
            identity(frame_id=1313, tensor_seq=15), b"held-tail-output",
            tail_ready_monotonic_raw_ns=11))
        dispatch.drain()
        dispatch.stop()
        self.assertEqual(ack_packets, [])
        self.assertEqual(map_calls, [1313])
        self.assertEqual(prediction_calls, [])
        self.assertEqual(len(dispatch.auxiliary_map_results), 1)

    def test_ack_sender_failure_occurs_before_any_branch_submission(self) -> None:
        calls: list[str] = []

        def fail_ack(_packet):
            calls.append("ack")
            raise OSError("downlink unavailable")

        dispatch = L.TailOutputDispatchV1(
            ack_sender=fail_ack, queue_depth=1,
            map_callback=lambda _work: calls.append("map") or succeeded("MAP_OK"),
            prediction_callback=lambda _work: (
                calls.append("prediction") or succeeded("PREDICTION_OK")),
        )
        dispatch.start()
        with self.assertRaises(OSError):
            dispatch.dispatch_decision(
                identity(), b"usable-tail-output", tail_ready_monotonic_raw_ns=10)
        dispatch.drain()
        dispatch.stop()
        self.assertEqual(calls, ["ack"])

    def test_bounded_map_queue_drop_cannot_retract_sent_ack(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        ack_packets: list[bytes] = []

        def blocked_map(_work):
            entered.set()
            self.assertTrue(release.wait(timeout=3.0))
            return succeeded("MAP_PUBLISHED")

        dispatch = L.TailOutputDispatchV1(
            ack_sender=ack_packets.append, queue_depth=1,
            map_callback=blocked_map,
            prediction_callback=lambda _work: succeeded("PREDICTION_RETAINED"))
        dispatch.start()
        rows = [
            identity(decision_seq=7 + index, ticket_seq=7 + index,
                     frame_id=1312 + index, tensor_seq=14 + index)
            for index in range(3)
        ]
        dispatch.dispatch_decision(rows[0], b"tail-0",
                                   tail_ready_monotonic_raw_ns=10)
        self.assertTrue(entered.wait(timeout=3.0))
        dispatch.dispatch_decision(rows[1], b"tail-1",
                                   tail_ready_monotonic_raw_ns=11)
        third = dispatch.dispatch_decision(rows[2], b"tail-2",
                                           tail_ready_monotonic_raw_ns=12)
        self.assertFalse(third.map_enqueued)
        self.assertEqual(len(ack_packets), 3)
        self.assertIs(dispatch.coordinator.result(rows[2], B.Branch.MAP).status,
                      B.BranchStatus.SKIPPED)
        release.set()
        dispatch.drain()
        dispatch.stop()

    def test_invalid_callback_result_is_failed_and_worker_continues(self) -> None:
        calls = 0

        def map_callback(_work):
            nonlocal calls
            calls += 1
            if calls == 1:
                return object()
            return succeeded("MAP_PUBLISHED")

        dispatch = L.TailOutputDispatchV1(
            ack_sender=lambda _packet: None, queue_depth=2,
            map_callback=map_callback,
            prediction_callback=lambda _work: succeeded("PREDICTION_RETAINED"))
        dispatch.start()
        rows = [identity(), identity(decision_seq=8, ticket_seq=8,
                                     frame_id=1313, tensor_seq=15)]
        for index, row in enumerate(rows):
            dispatch.dispatch_decision(
                row, f"tail-{index}".encode("ascii"),
                tail_ready_monotonic_raw_ns=10 + index)
        dispatch.drain()
        dispatch.stop()
        self.assertEqual(calls, 2)
        self.assertIs(dispatch.coordinator.result(
            rows[0], B.Branch.MAP).status, B.BranchStatus.FAILED)
        self.assertIs(dispatch.coordinator.result(
            rows[1], B.Branch.MAP).status, B.BranchStatus.SUCCEEDED)
        self.assertTrue(any("foreign result" in item
                            for item in dispatch.worker_errors))


class PredictionStoreAdapterTest(unittest.TestCase):
    def test_create_only_store_retains_the_exact_acknowledged_bytes(self) -> None:
        ack_packets: list[bytes] = []
        with tempfile.TemporaryDirectory() as directory:
            store = B.PredictionEvidenceStoreV1.create(
                Path(directory) / "predictions")
            callback = L.prediction_store_callback(store, clock=lambda: 91)
            dispatch = L.TailOutputDispatchV1(
                ack_sender=ack_packets.append, queue_depth=2,
                map_callback=lambda _work: succeeded("MAP_PUBLISHED"),
                prediction_callback=callback)
            dispatch.start()
            dispatch.dispatch_decision(
                identity(), b"exact-postrun-prediction",
                tail_ready_monotonic_raw_ns=80)
            dispatch.drain()
            dispatch.stop()
            records = store.verify_all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].prediction_sha256,
                             A.decode_ack(ack_packets[0]).tail_output_sha256)
            self.assertEqual(records[0].recorded_monotonic_raw_ns, 91)


if __name__ == "__main__":
    unittest.main()

"""Real evaluator, packet transport, pose and native prediction-archive tests.

Only model inference/live services are absent. The retained target fixture,
quality calculation, asynchronous evaluator, feedback, local broker and actual
close/preservation run. Synthetic fixtures are never live-qualified.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from rl_agent.splitfusion_live_parity_runtime_v7.container_check import _ticket_and_target
from rl_agent.splitfusion_live_parity_runtime_v1 import pose
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T, runtime as R
from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W, run4_map_protocol_v2 as MP
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from . import edge as H, broker as B, scoring_packet as P

BINDING = "b" * 64
ROOT = Path(__file__).resolve().parents[2]


class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gt_edge_packet_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gtdir = self.root / "original_fixture"
        self.ticket, self.expected = _ticket_and_target(self.gtdir)
        self.identity = dict(self.ticket.gt_identity)
        bundle = T.read_bundle(self.gtdir, expected_identity=self.identity, timeout_s=0.)
        self.prepared = bundle.prepared
        self.metadata = json.loads((bundle.directory / "inputs.json").read_bytes())
        self.avo = json.loads(LG.sidecar_path(self.gtdir, self.identity).read_bytes())
        self.avo.update(record_kind="AVO_SCORED_FRAME", native_shape=[720, 1280])
        for row in self.avo["avo"]["qualified"]:
            row["qualified"] = True
        self.packet = self.make_packet()
        self.socket = self.root / "gt.sock"
        self.profile = self.root / "profile"
        self.sent, self.received = [], threading.Event()
        self.worker = None
        self.addCleanup(self.stop_worker)

    def make_packet(self, metadata=None):
        return P.encode_packet(identity=self.identity, prepared=self.prepared,
            input_metadata=self.metadata if metadata is None else metadata,
            avo_document=self.avo, ready_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            ready_wall_ns=time.time_ns(), binding=BINDING)

    def stop_worker(self):
        if self.worker is not None and self.worker._thread.is_alive():
            self.worker.close(timeout_s=2.)

    def evaluator(self, timeout=.3):
        def send(payload):
            self.sent.append(payload)
            self.received.set()
        self.worker = E.Run4EvaluatorV2(spec=W.load_run4_quality_spec(ROOT), send=send,
            read_ground_truth=lambda **kw: G.read_ground_truth(self.gtdir, **kw),
            match_distance_m=3., gt_timeout_s=timeout, probe_timeout_s=.002,
            poll_interval_s=.001, evidence_dir=self.gtdir)
        self.worker.start()
        return self.worker

    def scope(self):
        return H.install_edge_handoff(socket_path=self.socket, binding=BINDING,
            max_frames=4, archive_directory=self.profile / "prediction_archive",
            publisher_uid=os.geteuid(), publisher_gid=os.getegid())

    def assert_success(self, state, closed):
        self.assertTrue(state["complete"])
        json.dumps(state, allow_nan=False)
        self.assertEqual(state["errors"], [])
        self.assertFalse(state["durability_claim"])
        self.assertTrue(state["publisher_credential_policy"]["explicitly_supplied"])
        self.assertEqual(state["publisher_credential_policy"], state["broker"]["credential_policy"])
        self.assertFalse(self.socket.exists())
        self.assertFalse(state["broker"]["worker_alive"])
        self.assertEqual(state["broker"]["admitted"], 1)
        self.assertEqual(state["broker"]["consumed"], 1)
        self.assertTrue(closed["parity_archive_complete"])
        self.assertTrue(closed["pose_archive_complete"])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.worker.records[0]["measurement"], self.expected.measurement)
        expected_feedback, expected_reason = W.quality_feedback(self.worker._spec,
            self.expected.measurement, self.ticket.envelope)
        self.assertEqual(self.sent[0], W.encode_feedback(expected_feedback, expected_reason))
        timing = self.worker.records[0]["timing"]
        self.assertEqual(timing["scoring_packet_sha256"], P.digest(self.packet))
        self.assertEqual(timing["target_bundle_sha256"], P.digest(self.packet))
        self.assertEqual(timing["target_reference_kind"], H.REFERENCE_KIND)
        self.assertEqual(timing["gt_transport_semantics"], H.TRANSPORT)
        self.assertEqual(timing["avo_sidecar_sha256"], P.digest(T._canonical(self.avo)))
        self.assertGreaterEqual(timing["packet_decode_completed_raw_ns"], timing["packet_admitted_raw_ns"])
        self.assertEqual(self.worker._gt_timeout, .3)
        self.assertEqual(self.worker._probe_timeout, .002)
        manifests = json.loads((self.profile / "prediction_archive/MANIFEST.json").read_bytes())
        self.assertEqual(manifests["frames"], 1)
        self.assertEqual(manifests["evaluated_frames"], 1)
        events = pose.read_archive(self.profile / "pose_archive")["events"]
        reconciled = [row for row in events if row["kind"] == "EDGE_TARGET_RECONCILED"]
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["gt_identity"], self.identity)
        self.assertEqual(reconciled[0]["scoring_packet_sha256"], P.digest(self.packet))
        self.assertEqual(reconciled[0]["target_reference_kind"], H.REFERENCE_KIND)

    def exercise_order(self, gt_first):
        originals = (R.install_edge_evaluation, T.read_bundle, pose._verified_target_camera,
            E.Run4EvaluatorV2._record_read, E.Run4EvaluatorV2._evaluate_with)
        with install_prediction_contract(), self.scope() as state, ExitStack() as stack:
            # Any accidental durable target/AVO read during live scoring fails.
            for owner, name in ((T, "read_bundle"), (T, "_read"), (LG, "read_avo_bundle")):
                stack.enter_context(mock.patch.object(owner, name,
                    side_effect=AssertionError("disk GT read on packet path")))
            worker = self.evaluator()
            if gt_first:
                answer = B.publish(self.socket, self.packet, binding=BINDING)
                worker.submit(self.ticket)
            else:
                worker.submit(self.ticket)
                self.assertFalse(self.received.wait(.015))
                answer = B.publish(self.socket, self.packet, binding=BINDING)
            self.assertEqual(answer["status"], "ADMITTED")
            self.assertTrue(self.received.wait(2.))
            closed = worker.close(timeout_s=2.)
        self.assertEqual(originals, (R.install_edge_evaluation, T.read_bundle, pose._verified_target_camera,
            E.Run4EvaluatorV2._record_read, E.Run4EvaluatorV2._evaluate_with))
        self.assert_success(state, closed)

    def test_gt_first_real_evaluator_exact_quality_and_no_disk_read(self):
        self.exercise_order(True)

    def test_prediction_first_real_nonblocking_probe_then_exact_quality(self):
        self.exercise_order(False)

    def test_missing_gt_uses_unchanged_local_watchdog_not_new_reward_deadline(self):
        with install_prediction_contract(), self.scope() as state:
            worker = self.evaluator(timeout=.025)
            worker.submit(self.ticket)
            self.assertTrue(self.received.wait(2.))
            closed = worker.close(timeout_s=2.)
        self.assertTrue(state["complete"])
        feedback, reason = W.decode_feedback(self.sent[0])
        self.assertEqual(reason, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
        self.assertEqual(feedback.kind, "EVALUATOR_FAULT")
        self.assertEqual(closed["gt_expired"], 1)
        self.assertEqual(state["reads"], [])

    def test_pose_camera_mismatch_fails_without_scoring_or_substitution(self):
        metadata = json.loads(json.dumps(self.metadata))
        inverse = json.loads(metadata["frame"]["camera_inverse_matrix_json"])
        inverse[0][3] += 5.
        metadata["frame"]["camera_inverse_matrix_json"] = json.dumps(inverse)
        packet = self.make_packet(metadata)
        with self.assertRaisesRegex(B.BrokerError, "close/drain/preserve"):
            with install_prediction_contract(), self.scope() as state:
                worker = self.evaluator()
                B.publish(self.socket, packet, binding=BINDING)
                worker.submit(self.ticket)
                self.assertTrue(self.received.wait(2.))
                worker.close(timeout_s=2.)
        self.assertFalse(state["complete"])
        self.assertTrue(state["pose_state"]["errors"])
        self.assertEqual(self.worker.records[0]["reason"], "EVALUATOR_EXCEPTION")
        self.assertIsNone(self.worker.records[0]["measurement"])

    def test_consume_once_and_broker_pending_is_not_silent_success(self):
        with self.assertRaisesRegex(B.BrokerError, "close/drain/preserve"):
            with install_prediction_contract(), self.scope() as state:
                worker = self.evaluator()
                B.publish(self.socket, self.packet, binding=BINDING)
                worker.close(timeout_s=2.)
        self.assertEqual(state["broker"]["pending_jobs"], 1)
        self.assertFalse(state["complete"])

    def test_scope_exception_restores_without_masking_primary_failure(self):
        old = (G.read_ground_truth, E.Run4EvaluatorV2._observe_components,
               E.Run4EvaluatorV2._record_read, E.Run4EvaluatorV2._evaluate_with,
               E.Run4EvaluatorV2.close, E.Run4EdgeProcessorV2.verify,
               PW.publish_ready_after_warmup)
        with self.assertRaisesRegex(RuntimeError, "injected outer failure"):
            with install_prediction_contract(), self.scope() as state:
                raise RuntimeError("injected outer failure")
        self.assertEqual(old, (G.read_ground_truth, E.Run4EvaluatorV2._observe_components,
            E.Run4EvaluatorV2._record_read, E.Run4EvaluatorV2._evaluate_with,
            E.Run4EvaluatorV2.close, E.Run4EdgeProcessorV2.verify, PW.publish_ready_after_warmup))
        self.assertFalse(self.socket.exists())
        self.assertFalse(state["complete"])

    def test_retained_pose_ready_guard_and_wire_identity_preserved(self):
        profile = SimpleNamespace(action_id=self.identity["action_id"], profile_id=self.identity["profile_id"])
        values = self.ticket.envelope, self.ticket.context, profile, MP.run4_identity(self.ticket.envelope)
        with mock.patch.object(E.Run4EdgeProcessorV2, "verify", return_value=values):
            with install_prediction_contract(), self.scope() as state:
                processor = E.Run4EdgeProcessorV2(contract=None, run_id=self.identity["run_id"],
                    cell_id=self.identity["cell_id"], service_deadline_s=.170, ack_timeout_s=.170,
                    compute=lambda **kwargs: None)
                PW.publish_ready_after_warmup(lambda: dict(completed=True, modes_warmed=list(range(12))), lambda: None)
                self.assertEqual(processor.verify(b"synthetic-wire-not-model-input"), values)
                worker = self.evaluator()
                B.publish(self.socket, self.packet, binding=BINDING)
                worker.submit(self.ticket)
                self.assertTrue(self.received.wait(2.))
                closed = worker.close(timeout_s=2.)
        self.assert_success(state, closed)
        poses = pose.read_archive(self.profile / "pose_archive")
        self.assertEqual(poses["manifest"]["counts"]["EDGE_READY"], 1)
        self.assertEqual(poses["manifest"]["counts"]["EDGE_WIRE"], 1)

    def test_live_wire_before_ready_retains_original_refusal(self):
        profile = SimpleNamespace(action_id=self.identity["action_id"], profile_id=self.identity["profile_id"])
        values = self.ticket.envelope, self.ticket.context, profile, MP.run4_identity(self.ticket.envelope)
        with mock.patch.object(E.Run4EdgeProcessorV2, "verify", return_value=values):
            with self.assertRaisesRegex(pose.PoseParityError, "before READY"):
                with install_prediction_contract(), self.scope() as state:
                    processor = E.Run4EdgeProcessorV2(contract=None, run_id=self.identity["run_id"],
                        cell_id=self.identity["cell_id"], service_deadline_s=.170, ack_timeout_s=.170,
                        compute=lambda **kwargs: None)
                    processor.verify(b"synthetic-before-ready")
        self.assertFalse(state["complete"])
        self.assertFalse(self.socket.exists())

    def test_exception_drains_running_evaluator_before_restoring_methods(self):
        original = E.Run4EvaluatorV2.__init__
        with self.assertRaisesRegex(RuntimeError, "injected after submit"):
            with install_prediction_contract(), self.scope() as state:
                worker = self.evaluator(timeout=.025)
                worker.submit(self.ticket)
                raise RuntimeError("injected after submit")
        self.assertFalse(worker._thread.is_alive())
        self.assertEqual(state["evaluator_close"]["gt_expired"], 1)
        self.assertTrue(state["prediction_state"]["flushed"])
        self.assertTrue(state["pose_state"]["flushed"])
        self.assertFalse(state["complete"])
        self.assertIs(E.Run4EvaluatorV2.__init__, original)
        json.dumps(state, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

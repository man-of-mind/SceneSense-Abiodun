"""Genuine collector/source-ownership/CPU-worker/archive lifecycle composition.

The retained fixture replaces live sensor RPC only. Two numerical processes,
immutable descriptor transfer, exact source ownership, GT computation, local
packet admission, durable archival and original collector preservation run.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import unittest

from rl_agent.splitfusion_live_parity_runtime_v7 import test_worker_integration as F
from rl_agent.splitfusion_live_parity_runtime_v1.entries import evidence_scope
from rl_agent.splitfusion_live_parity_repair_v1 import composition, target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import source_ownership as SO
from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as WF
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, runtime as R4
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from . import runtime as R, client as CL, contract as C, broker as B, scoring_packet as P


class RuntimeTests(unittest.TestCase):
    # Reuse the actual fixture without inheriting tests for the sealed V4 path.
    fixture = F.RealWorkerCompositionTests.fixture

    @contextmanager
    def composed(self, f, *, initialization_failure=False):
        clients, servers = [], []
        f.observation = {"state": {"clients": clients}}

        def factory(**kwargs):
            root = Path(kwargs["handoff_config"]["directory"])
            self.assertTrue(root.is_dir())
            self.assertEqual(root, f.attempt / "edge")
            # Base.__init__ creates edge_evidence_dir before the production
            # RemoteCollector._spawn_sensors starts this client.
            server = B.Server(root / C.SOCKET_NAME, binding=C.binding(), max_frames=100)
            servers.append(server)
            server.start()
            if initialization_failure:
                kwargs["expected_bootstrap_sources"] = {}
            client = CL.Client(**kwargs)
            clients.append(client)
            return client

        try:
            with SO.install_source_ownership(attempt_dir=f.attempt, max_frames=100) as ownership:
                with R.install_cpu_gt_worker(attempt_dir=f.attempt, max_frames=100,
                        expected_sources=W.sources(), expected_bootstrap_sources=C.sources(),
                        _client_factory=factory) as worker_scope:
                    with evidence_scope(f.attempt), composition.ue_scope(attempt_dir=f.attempt, max_frames=100):
                        f.owner = LG.build_live_gt_collector(f.Base)(attempt_dir=f.attempt)
                        yield ownership, worker_scope, clients, servers[0]
        finally:
            for client in clients:
                if client.process.poll() is None:
                    client.abort()
            for server in servers:
                self.assertFalse(server.close()["worker_alive"])

    def run_job(self, f):
        thread = f.start_job()
        thread.join(timeout=20.)
        self.assertFalse(thread.is_alive())
        return thread

    def test_two_real_frames_resolve_receipts_before_original_preservation(self):
        source_before = C.sources()
        with self.fixture() as f:
            packets = []
            with self.composed(f) as (ownership, scope, clients, server):
                client = clients[0]
                pid = client.process.pid
                self.assertIsNone(client.process.poll())
                self.assertEqual(client.ready["process_context"]["torch_threads"], 1)
                self.assertFalse(client.ready["process_context"]["cuda_initialized"])
                self.assertEqual(client.ready["process_context"]["environment"], C.ENVIRONMENT)
                for index in range(2):
                    if index:
                        WF.advance(f.kwargs, frame=101, replenish=True)
                    self.run_job(f)
                    self.assertEqual(f.task_errors, [])
                    record = f.owner._avo_results[-1]
                    self.assertIsNone(record["complete_target"]["manifest_sha256"])
                    self.assertTrue(record["complete_target"]["archive_pending"])
                    self.assertFalse(record["archive_receipt"]["durable"])
                    packet = P.decode_packet(server.take(f.kwargs["identity"]),
                        expected_identity=f.kwargs["identity"], binding=C.binding())
                    packets.append(packet)
                    self.assertEqual(packet["manifest_sha256"], record["scoring_handoff"]["packet_sha256"])
                    self.assertEqual(packet["prepared"].content_sha256,
                        record["complete_target"]["prepared_content_sha256"])
                    self.assertEqual(client.process.pid, pid)
                    self.assertEqual(f.owner._source_ownership.pending, {})
                self.assertEqual(f.owner._avo_sync.accepted, 2)
                self.assertFalse((f.attempt / "parity_complete_targets").exists())
                f.owner.finish()
                self.assertTrue(scope["pending_receipts_resolved_before_preservation"])
                self.assertEqual(scope["archive"]["completed"], 2)
                self.assertEqual(scope["archive"]["outstanding_jobs"], 0)
                self.assertEqual(scope["archive"]["child_returncode"], 0)
                self.assertIsNotNone(client.process.poll())
            self.assertTrue(scope["complete"])
            self.assertTrue(scope["joint_drained"])
            self.assertTrue(scope["preserved_before_teardown"])
            self.assertTrue(ownership["manifest"]["complete"])
            self.assertTrue(SO.verify_ownership_archive(f.attempt / "source_ownership",
                expected_identities=f.identities, expected_action_identities=f.actions)["passed"])
            summary = json.loads((f.attempt / "run5_live_avo_gt_summary.json").read_bytes())
            self.assertEqual(len(summary["records"]), 2)
            self.assertEqual(summary["rejected_records"], [])
            for record, packet in zip(summary["records"], packets):
                self.assertTrue(record["archive_receipt"]["durable"])
                self.assertNotIn("archive_pending", record["complete_target"])
                target = T.verify_full_inputs(f.attempt / "parity_complete_targets",
                    expected_identity=record["complete_identity"],
                    expected_manifest_sha256=record["complete_target"]["manifest_sha256"])
                self.assertTrue(target.native_inputs_verified)
                self.assertEqual(target.prepared.content_sha256, packet["prepared"].content_sha256)
                avo_path = f.attempt / "run5_live_avo_inputs" / Path(record["path"]).name
                self.assertEqual(json.loads(avo_path.read_bytes()), packet["avo_document"])
                self.assertEqual(hashlib.sha256(avo_path.read_bytes()).hexdigest(), packet["avo_sidecar_sha256"])
            stages = json.loads((f.attempt / "complete_gt_stage_timing/MANIFEST.json").read_bytes())
            self.assertEqual(stages["schema"], "splitfusion.gt_scoring_handoff.stage_timing.v1")
            self.assertFalse(stages["work_and_readiness_unchanged"])
            self.assertTrue(stages["scoring_ready_is_not_archive_durable"])
            self.assertEqual(stages["records"], 12)
            self.assertEqual((f.attempt / "complete_gt_stage_timing/stages.jsonl").read_bytes(),
                (f.attempt / "cpu_gt_worker/engine/complete_gt_stage_timing/stages.jsonl").read_bytes())
        self.assertEqual(C.sources(), source_before)

    def test_handled_second_frame_failure_preserves_first_and_rejection(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(R4.RuntimeErrorGT, "did not complete"):
                with self.composed(f) as (ownership, scope, clients, server):
                    self.run_job(f)
                    self.assertEqual(f.task_errors, [])
                    first = dict(f.kwargs["identity"])
                    server.take(first)
                    WF.advance(f.kwargs, frame=101, replenish=True)
                    f.kwargs["frozen_world"].population_evidence["registry_complete"] = False
                    self.run_job(f)
                    self.assertEqual(len(f.task_errors), 1)
                    self.assertIsInstance(f.task_errors[0], W.WorkerError)
                    self.assertEqual(len(f.owner._avo_results), 1)
                    self.assertEqual(len(f.owner._avo_rejected_results), 1)
                    self.assertEqual(server.stats()["admitted"], 1)
                    self.assertFalse(f.owner._source_ownership.pending)
                    f.owner.finish()
            self.assertFalse(scope["complete"])
            self.assertIsNotNone(clients[0].process.poll())
            summary = json.loads((f.attempt / "run5_live_avo_gt_summary.json").read_bytes())
            self.assertEqual(len(summary["records"]), 1)
            self.assertEqual(len(summary["rejected_records"]), 1)
            self.assertTrue(summary["records"][0]["archive_receipt"]["durable"])
            self.assertTrue(T.verify_full_inputs(f.attempt / "parity_complete_targets",
                expected_identity=first).native_inputs_verified)
            rejected = json.loads((f.attempt / "cpu_gt_worker/engine/rejected_101.json").read_bytes())
            self.assertFalse(rejected["scoring_packet_admitted"])
            self.assertFalse((f.attempt / "edge" / C.SOCKET_NAME).exists())

    def test_source_mismatch_stops_before_sensor_fixture(self):
        with self.fixture() as f:
            with self.assertRaises(W.WorkerError):
                with self.composed(f, initialization_failure=True):
                    self.fail("invalid bootstrap sources must refuse construction")
            self.assertNotIn("sensor_fixture_started", f.events)
            scope = json.loads((f.attempt / "GT_WORKER_SCOPE.json").read_bytes())
            self.assertFalse(scope["complete"])
            self.assertFalse((f.attempt / "edge" / C.SOCKET_NAME).exists())

    def test_second_archive_io_failure_salvages_first_without_false_success(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(R4.RuntimeErrorGT, "did not complete"):
                with self.composed(f) as (ownership, scope, clients, server):
                    self.run_job(f)
                    self.assertEqual(f.task_errors, [])
                    first = dict(f.kwargs["identity"])
                    server.take(first)
                    first_ready = T.bundle_path(f.attempt / "edge", first) / T.READY_NAME
                    until = time.monotonic() + 5.
                    while not first_ready.is_file() and time.monotonic() < until:
                        time.sleep(.005)
                    self.assertTrue(first_ready.is_file())
                    WF.advance(f.kwargs, frame=101, replenish=True)
                    second = dict(f.kwargs["identity"])
                    # Create-only file collision is an authentic archive-process
                    # IO failure after packet admission, not a mocked receipt.
                    collision = LG.sidecar_path(f.attempt / "edge", second)
                    collision.write_bytes(b"DELIBERATE_ARCHIVE_IO_FAULT_FIXTURE")
                    self.run_job(f)
                    server.take(second)
                    self.assertEqual(f.task_errors, [])
                    self.assertEqual(len(f.owner._avo_results), 2)
                    original_second = json.loads(W.canonical(f.owner._avo_results[-1]))
                    f.owner.finish()
            self.assertFalse(scope["complete"])
            self.assertTrue(scope["preserved_before_teardown"])
            self.assertTrue(scope["close_receipt_recovered_from_owned_evidence"])
            self.assertTrue(scope["archive"]["failed"])
            self.assertIsNotNone(clients[0].process.poll())
            proof = json.loads((f.attempt / "FAILED_ARCHIVE_PARTIAL_PRESERVATION.json").read_bytes())
            self.assertEqual(proof["status"], "FAILED")
            self.assertEqual(proof["original_counts"], dict(records=2, rejected_records=0))
            self.assertEqual(proof["preserved_counts"], dict(records=1, rejected_records=0))
            self.assertFalse(proof["durable_success_for_unresolved"])
            self.assertEqual(proof["unresolved"][0]["record"], original_second)
            self.assertTrue(proof["archive_writer_exit_verified"])
            summary = json.loads((f.attempt / "run5_live_avo_gt_summary.json").read_bytes())
            self.assertEqual([r["frame_id"] for r in summary["records"]], [first["frame_id"]])
            self.assertTrue(summary["records"][0]["archive_receipt"]["durable"])
            self.assertTrue(T.verify_full_inputs(f.attempt / "parity_complete_targets",
                expected_identity=first).native_inputs_verified)
            self.assertFalse(T.bundle_path(f.attempt / "parity_complete_targets", second).exists())
            self.assertEqual(collision.read_bytes(), b"DELIBERATE_ARCHIVE_IO_FAULT_FIXTURE")

    def test_inflight_abort_stops_child_before_source_ownership_closes(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(R4.RuntimeErrorGT, "has not drained"):
                with self.composed(f) as (ownership, scope, clients, server):
                    client = clients[0]
                    os.kill(client.process.pid, signal.SIGSTOP)  # Owned test worker only.
                    f.start_job()
                    deadline = time.monotonic() + 3.
                    while not client.lock.locked() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(client.lock.locked())
                    f.owner.finish()
            self.assertIsNotNone(client.process.poll())
            self.assertFalse(f.owner.evaluation_worker.is_alive())
            self.assertFalse(f.owner._source_ownership.pending)
            self.assertFalse(scope["complete"])
            recorded = json.loads((f.attempt / "source_ownership/MANIFEST.json").read_bytes())
            self.assertTrue(recorded["worker_drained"])
            self.assertFalse(recorded["complete"])


if __name__ == "__main__":
    unittest.main()

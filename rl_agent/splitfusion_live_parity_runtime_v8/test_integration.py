"""Real CPU handoff worker/archiver and actual v8 edge entry without services."""
from contextlib import contextmanager, ExitStack
import json
import importlib.util
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

from rl_agent.splitfusion_live_gt_handoff_v1 import runtime as HR, contract as H, broker as B, scoring_packet as P
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as WF
from rl_agent.splitfusion_live_parity_runtime_v7 import test_worker_integration as F
from rl_agent.splitfusion_live_parity_runtime_v7.container_check import _ticket_and_target
from rl_agent.splitfusion_live_parity_repair_v1 import composition, target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import source_ownership as SO
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from . import entries as E, contract as C, worker_observer as O


class WorkerIntegrationTests(unittest.TestCase):
    fixture = F.RealWorkerCompositionTests.fixture

    def test_two_real_frames_caps_archive_identity_and_genuine_close(self):
        with self.fixture() as f, ExitStack() as stack:
            from rl_agent.splitfusion_live_gt_optimization_v5.test_bootstrap import _retained_hook
            spec = importlib.util.spec_from_file_location("v8_retained_hook_fixture", _retained_hook())
            hook = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook)
            stack.enter_context(mock.patch.object(subprocess.Popen, "__init__", subprocess.Popen.__init__))
            hook.install_child_environment_restore({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
            servers = []
            try:
                with SO.install_source_ownership(attempt_dir=f.attempt, max_frames=C.BUDGET) as ownership:
                    with O.observe_cpu_worker(attempt_dir=f.attempt, max_frames=C.BUDGET,
                            expected_sources=W.sources(), expected_bootstrap_sources=H.sources(),
                            handoff_binding=H.binding(),
                            drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]) as observation:
                        f.observation = observation
                        def factory(**kwargs):
                            directory = Path(kwargs["handoff_config"]["directory"])
                            server = B.Server(directory / H.SOCKET_NAME, binding=H.binding(), max_frames=C.BUDGET)
                            servers.append(server.start())
                            return observation["factory"](**kwargs)
                        with HR.install_cpu_gt_worker(attempt_dir=f.attempt, max_frames=C.BUDGET,
                                expected_sources=W.sources(), expected_bootstrap_sources=H.sources(),
                                _client_factory=factory) as scope:
                            with E.evidence_scope(f.attempt), composition.ue_scope(attempt_dir=f.attempt, max_frames=C.BUDGET):
                                f.owner = LG.build_live_gt_collector(f.Base)(attempt_dir=f.attempt)
                                client = observation["state"]["clients"][0]
                                self.assertEqual(client.archive_startup["environment"], H.ENVIRONMENT)
                                self.assertEqual(client.archive_startup["parent_pid"], client.process.pid)
                                self.assertEqual(f.events[-1], "sensor_fixture_started")
                                for index in range(2):
                                    if index:
                                        WF.advance(f.kwargs, frame=101, replenish=True)
                                    f.start_job().join(timeout=20.)
                                    self.assertFalse(f.owner.evaluation_worker.is_alive())
                                    self.assertEqual(f.task_errors, [])
                                    servers[0].take(f.kwargs["identity"])
                                f.owner.finish()
                proof = E._worker_proof(f.attempt, scope, W.sources(), H.sources(), H.binding())
                self.assertTrue(proof["complete"], proof)
                self.assertEqual((scope["new_cpu_processes"], scope["new_queues"]), (2, 2))
                timeline = json.loads((f.attempt / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
                self.assertTrue(timeline["complete"])
                self.assertEqual(len(timeline["requests"]), 2)
                self.assertEqual(timeline["clients"][0]["close"]["timing"],
                    json.loads((f.attempt / "cpu_gt_worker/engine/HANDOFF_DRAIN.json").read_bytes()))
                self.assertEqual(timeline["clients"][0]["archive_startup"]["pid"], scope["archive"]["child_pid"])
                from .postflight_checks import verify_worker_scope
                self.assertTrue(verify_worker_scope(f.attempt, expected_sources=W.sources(),
                    outer_scope={"cpu_worker": proof})["passed"])
                for request in timeline["requests"]:
                    self.assertEqual(request["process_context"]["torch_threads"], 1)
                    self.assertFalse(request["process_context"]["cuda_initialized"])
                    self.assertEqual(len(request["rows"]), 6)
                saved = dict(scope, new_cpu_processes=1)
                self.assertFalse(E._worker_proof(f.attempt, saved, W.sources(), H.sources(), H.binding())["complete"])
            finally:
                for server in servers:
                    self.assertFalse(server.close()["worker_alive"])


class EdgeEntryIntegrationTests(unittest.TestCase):
    def test_actual_entry_scores_packet_and_preserves_full_close_after_broker_exit(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as ER
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as Wire
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            profile = root / "profiler"
            profile.mkdir()
            directory = root / "gt"
            ticket, expected = _ticket_and_target(directory)
            bundle = T.read_bundle(directory, expected_identity=ticket.gt_identity, timeout_s=0.)
            metadata = json.loads((bundle.directory / "inputs.json").read_bytes())
            avo = json.loads(LG.sidecar_path(directory, ticket.gt_identity).read_bytes())
            avo.update(record_kind="AVO_SCORED_FRAME", native_shape=[720, 1280])
            for row in avo["avo"]["qualified"]:
                row["qualified"] = True
            packet = P.encode_packet(identity=ticket.gt_identity, prepared=bundle.prepared,
                input_metadata=metadata, avo_document=avo, binding=H.binding(),
                ready_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), ready_wall_ns=time.time_ns())
            registration = dict(id=C.ID, transmitted_budget=C.BUDGET, edge_module=C.EDGE,
                live_metric_contract=C.metrics(), handoff_binding=H.binding(), bootstrap_sources=H.sources(),
                worker_sources=W.sources(), publisher_uid=os.getuid(), publisher_gid=os.getgid())
            registration["registration_sha256"] = C.canonical(registration)
            path = root / "registration.json"
            path.write_text(json.dumps(registration))
            socket = root / H.SOCKET_NAME
            parent = ModuleType("rl_agent.splitfusion_single_gpu_profiler_v1")
            parent.__path__ = []
            parent.edge_profiled_v1 = NS(profile_path=lambda _: profile / "events.jsonl")
            stack.enter_context(mock.patch.dict(sys.modules, {parent.__name__: parent}))
            stack.enter_context(mock.patch.object(E, "REGISTRATION", path))
            stack.enter_context(mock.patch.object(E, "SOCKET", socket))
            result = {}
            def retained(argv, *, max_frames):
                self.assertTrue(socket.exists())
                self.assertEqual(max_frames, 100)
                sent = []
                evaluator = ER.Run4EvaluatorV2(spec=Wire.load_run4_quality_spec(Path(__file__).resolve().parents[2]),
                    send=sent.append, read_ground_truth=lambda **kw: G.read_ground_truth(directory, **kw),
                    match_distance_m=3., gt_timeout_s=.02)
                evaluator._thread = NS(join=lambda timeout=None: None, is_alive=lambda: False)
                B.publish(socket, packet, binding=H.binding())
                with ExitStack() as guards:
                    for owner, name in ((T, "read_bundle"), (T, "_read"), (LG, "read_avo_bundle")):
                        guards.enter_context(mock.patch.object(owner, name,
                            side_effect=AssertionError("durable GT fallback used")))
                    evaluator.submit(ticket)
                    evaluator.evaluate(evaluator._queue.get_nowait())
                    result.update(evaluator.close())
                self.assertEqual(evaluator.records[0]["measurement"], expected.measurement)
                self.assertEqual(len(sent), 1)
                self.assertTrue(result["pose_archive_complete"])
                return 0
            stack.enter_context(mock.patch.object(E, "run_edge_only", retained))
            self.assertEqual(E.edge([]), 0)
            self.assertFalse(socket.exists())
            saved = json.loads((profile / "EDGE_HANDOFF_SCOPE.json").read_bytes())
            closed = json.loads((profile / "evaluator_close_result.json").read_bytes())
            self.assertTrue(saved["complete"])
            self.assertTrue(closed["pose_archive_complete"])
            self.assertEqual(saved["evaluator_close"], closed)
            self.assertEqual(saved["broker"]["admitted"], 1)
            self.assertEqual(saved["broker"]["consumed"], 1)
            self.assertFalse(saved["broker"]["worker_alive"])
            self.assertEqual(pose.read_archive(profile / "pose_archive")["manifest"]["counts"]["EDGE_TARGET_RECONCILED"], 1)


if __name__ == "__main__":
    unittest.main()

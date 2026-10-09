"""CPU-only real worker + retained composition/ownership/finalizer checks.

Only sensor RPC, the world-load catalogue callback and legacy object-GT rows
are fixtures. Complete-GT computation, immutable handoff, validation/fsync,
AVO, target/source publication, preservation and child lifecycle are real.
"""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W, runtime as R
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G, composition, target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2 import source_ownership as SO
from rl_agent.splitfusion_live_parity_repair_v2.postflight_checks import verify_gt_timing
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_object_gt_v2 as OG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_gt_priority_v2 as GP
from .worker_observer import observe_cpu_worker, ObservedClient
from . import contract as C, entries as E, postflight_checks as P


class RealWorkerCompositionTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with F.fixture() as (root, kwargs, radar), ExitStack() as stack:
            # Main research checkout intentionally lacks the retained profiler
            # package. Its registered worktree is read-only test input here;
            # after deployment the package is local and this branch is unused.
            import rl_agent
            repo = Path(__file__).resolve().parents[2]
            if not (repo / "rl_agent/splitfusion_single_gpu_profiler_v1").is_dir():
                retained = repo.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent"
                self.assertTrue((retained / "splitfusion_single_gpu_profiler_v1").is_dir())
                stack.enter_context(mock.patch.object(rl_agent, "__path__", [*rl_agent.__path__, str(retained)]))
            attempt = root / "live_cell"
            attempt.mkdir()
            events, task_errors, identities, actions = [], [], {}, {}
            original_factory = LG.build_live_gt_collector
            case = self

            class Catalogue:
                def __init__(self, directory):
                    self.directory = Path(directory)
                    shutil.copytree(root / "catalogue", self.directory)
                def require_world(self, world):
                    events.append("catalogue_checked")

            class Base:
                def __init__(self, *, attempt_dir):
                    self.attempt_dir = Path(attempt_dir)
                    self.edge_evidence_dir = self.attempt_dir / "edge"
                    self.edge_evidence_dir.mkdir()
                    self.world, self.ego = object(), NS(id=999)
                    self._quality_scene_lock = threading.Lock()
                    self._quality_scenes = {}
                    self.aligned_actor_tracker = object()
                    self.live = NS(last_datagram={}, _run4_identity=actions,
                        identity_for_frame=lambda frame: identities[frame])
                    self.evaluation_worker = NS(is_alive=lambda: False, join=lambda **kw: None)
                    self.worker = NS(is_alive=lambda: False)
                    self.evaluation_queue = OG.DeferringRewardPriorityGtQueueV2(
                        classify=self.classify, low_blocked=lambda: False,
                        on_skip=lambda frame, reason: None, maxsize_high=64, maxsize_low=64)
                    self._spawn_sensors()

                def _spawn_sensors(self):
                    events.append("sensor_fixture_started")
                def _on_rgb(self, image): pass
                def _on_semantic(self, image): pass
                def _on_radar(self, image): pass
                def _ground_truth(self, **kwargs):
                    return [{"legacy_fixture_frame": kwargs["frame_id"]}]
                def classify(self, item):
                    frame = item["frame_id"]
                    if frame not in actions or frame not in self.live.last_datagram:
                        raise GP.GtQueueError("unowned fixture frame")
                    return GP.HIGH if actions[frame]["reward_requested"] else GP.LOW
                def finish(self):
                    events.append("retained_base_finish")
                    return not self.evaluation_worker.is_alive()

            def no_sensor_rpc(base):
                real = original_factory(base)
                def spawn(owner):
                    return super(real, owner)._spawn_sensors()
                real._spawn_sensors = spawn
                return real

            stack.enter_context(mock.patch.object(G, "FreshCatalogue", Catalogue))
            stack.enter_context(mock.patch.object(LG, "build_live_gt_collector", no_sensor_rpc))
            state = NS(root=root, attempt=attempt, kwargs=kwargs, radar=radar, events=events,
                task_errors=task_errors, identities=identities, actions=actions, Base=Base,
                owner=None, observation=None)

            def start_job():
                owner = state.owner
                frame = kwargs["identity"]["frame_id"]
                identities[frame] = dict(kwargs["identity"])
                actions[frame] = {"frame_id": frame, "reward_requested": True, "mode_id": 11, "q_e4": 3000}
                owner.live.last_datagram[frame] = W.raw()
                owner._quality_scenes[frame] = kwargs["frozen_world"]
                owner._on_rgb(kwargs["rgb"]["_rgb_image"])
                owner._on_semantic(kwargs["semantic_image"])
                owner._on_depth(kwargs["depth_image"])
                owner._on_radar(kwargs["semantic_image"])
                ticket = dict(frame_id=frame, timestamp=kwargs["policy_timestamp_s"],
                    camera_matrix=kwargs["camera_matrix"], camera_inverse=kwargs["camera_inverse"],
                    radar_points={"world_xyz": radar}, scene=kwargs["frozen_world"])
                owner.evaluation_queue.put_nowait(ticket)
                def work():
                    item = owner.evaluation_queue.get_nowait()
                    try:
                        owner._ground_truth(frame_id=item["frame_id"], timestamp=item["timestamp"],
                            camera_matrix=item["camera_matrix"], camera_inverse=item["camera_inverse"],
                            radar_points=item["radar_points"], world=item["scene"])
                    except BaseException as exc:
                        task_errors.append(exc)
                    finally:
                        owner.evaluation_queue.task_done()
                thread = threading.Thread(target=work, name="retained-gt-fixture", daemon=True)
                owner.evaluation_worker = thread
                thread.start()
                return thread
            state.start_job = start_job
            try:
                yield state
            finally:
                if state.observation is not None:
                    for client in state.observation["state"]["clients"]:
                        process = getattr(client, "process", None)
                        if process is not None and process.poll() is None:
                            client.abort()
                if state.owner is not None:
                    state.owner.evaluation_worker.join(timeout=5.)
                case.assertFalse(torch.cuda.is_initialized())

    @contextmanager
    def composed(self, f, *, initialization_failure=False):
        with SO.install_source_ownership(attempt_dir=f.attempt, max_frames=C.BUDGET) as ownership:
            with observe_cpu_worker(attempt_dir=f.attempt, max_frames=C.BUDGET, expected_sources=W.sources(),
                    drain_check=lambda: ownership["inspect_drain"]()["worker_drained"]) as observation:
                f.observation = observation
                factory = observation["factory"]
                if initialization_failure:
                    def factory(**kwargs):
                        kwargs["expected_sources"] = {}
                        return ObservedClient(_observation=observation["state"], **kwargs)
                with R.install_cpu_gt_worker(attempt_dir=f.attempt, max_frames=C.BUDGET, expected_sources=W.sources(),
                        _client_factory=factory) as worker_scope:
                    with E.evidence_scope(f.attempt), composition.ue_scope(attempt_dir=f.attempt, max_frames=C.BUDGET):
                        f.owner = LG.build_live_gt_collector(f.Base)(attempt_dir=f.attempt)
                        yield ownership, worker_scope

    def test_two_real_jobs_preserve_identity_history_sources_and_joint_drain(self):
        with self.fixture() as f:
            with self.composed(f) as (ownership, scope):
                client = f.observation["state"]["clients"][0]
                first_pid = client.process.pid
                self.assertIsNone(client.process.poll())
                f.start_job().join(timeout=20.)
                self.assertFalse(f.owner.evaluation_worker.is_alive())
                self.assertEqual(f.task_errors, [])
                F.advance(f.kwargs, frame=101, replenish=True)
                f.start_job().join(timeout=20.)
                self.assertEqual(f.task_errors, [])
                self.assertEqual(client.process.pid, first_pid)
                self.assertEqual(f.owner._avo_sync.accepted, 2)
                self.assertEqual(f.owner._source_ownership.pending, {})
                f.owner.finish()
            self.assertTrue(scope["complete"])
            self.assertTrue(ownership["manifest"]["complete"])
            self.assertIsNotNone(client.process.poll())
            self.assertTrue(SO.verify_ownership_archive(f.attempt / "source_ownership",
                expected_identities=f.identities, expected_action_identities=f.actions)["passed"])
            self.assertEqual(verify_gt_timing(f.attempt / "complete_gt_stage_timing",
                expected_frames=f.identities)["stages"], 10)
            for record in f.owner._avo_results:
                verified = T.verify_full_inputs(f.attempt / "parity_complete_targets",
                    expected_identity=record["complete_identity"],
                    expected_manifest_sha256=record["complete_target"]["manifest_sha256"])
                self.assertTrue(verified.native_inputs_verified)
            timeline = json.loads((f.attempt / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
            self.assertTrue(timeline["complete"])
            self.assertEqual(len(timeline["requests"]), 2)
            for request in timeline["requests"]:
                self.assertEqual(request["worker"]["pid"], first_pid)
                self.assertFalse(request["process_context"]["cuda_initialized"])
                self.assertEqual(len(request["rows"]), 5)
                self.assertGreater(request["worker"]["thread_cpu_ns"], 0)
            outer = {"cpu_worker": E._worker_proof(f.attempt, scope, W.sources())}
            proof = P.verify_worker_scope(f.attempt, expected_sources=W.sources(), outer_scope=outer)
            self.assertTrue(proof["passed"])
            self.assertEqual(proof["requests"], 2)
            self.assertEqual(proof["genuine_child_stages"], 10)
            # Each tamper is applied to genuine successful output and restored;
            # no manufactured successful worker metadata is used by this test.
            timeline_path = f.attempt / "CPU_GT_PROCESS_TIMELINE.json"
            timeline_raw = timeline_path.read_bytes()
            changed = json.loads(timeline_raw)
            changed["requests"][0]["worker"]["pid"] += 1
            timeline_path.write_text(json.dumps(changed))
            outer["cpu_worker"]["timeline_sha256"] = C.sha(timeline_path)
            with self.assertRaisesRegex(ValueError, "execution clocks/PID"):
                P.verify_worker_scope(f.attempt, expected_sources=W.sources(), outer_scope=outer)
            timeline_path.write_bytes(timeline_raw)
            outer["cpu_worker"]["timeline_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "observer source/factory binding"):
                P.verify_worker_scope(f.attempt, expected_sources=W.sources(), outer_scope=outer)
            outer["cpu_worker"]["timeline_sha256"] = C.sha(timeline_path)
            targets_path = f.attempt / "parity_complete_targets/MANIFEST.json"
            targets_raw = targets_path.read_bytes()
            changed = json.loads(targets_raw)
            changed["frames"][0]["identity"]["frame_id"] += 999
            targets_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "IPC population differs"):
                P.verify_worker_scope(f.attempt, expected_sources=W.sources(), outer_scope=outer)
            targets_path.write_bytes(targets_raw)
            self.assertTrue(P.verify_worker_scope(f.attempt,
                expected_sources=W.sources(), outer_scope=outer)["passed"])

    def test_real_handled_geometry_rejection_keeps_original_path_and_sync(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"):
                with self.composed(f):
                    f.kwargs["frozen_world"].population_evidence["registry_complete"] = False
                    f.start_job().join(timeout=20.)
                    self.assertEqual(len(f.task_errors), 1)
                    self.assertIsInstance(f.task_errors[0], W.WorkerError)
                    self.assertEqual(f.owner._avo_sync.accepted, 1)
                    self.assertEqual(len(f.owner._avo_rejected_results), 1)
                    self.assertFalse(f.owner._source_ownership.pending)
                    f.owner.finish()
            rejected = f.attempt / "parity_rejected_targets" / T.bundle_path(f.attempt, f.kwargs["identity"]).name
            self.assertTrue((rejected / "REJECTION.json").is_file())
            summary = json.loads((f.attempt / "run5_live_avo_gt_summary.json").read_bytes())
            self.assertEqual(summary["source_synchronization"]["accepted"], 1)
            self.assertEqual(len(summary["rejected_records"]), 1)
            timeline = json.loads((f.attempt / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
            self.assertFalse(timeline["complete"])
            self.assertTrue(timeline["child_stopped"])
            self.assertEqual(timeline["requests"][0]["status"], "FAILED")
            self.assertEqual(len(timeline["requests"][0]["rows"]), 2)

    def test_real_child_initialization_failure_never_starts_sensor_fixture(self):
        with self.fixture() as f:
            with self.assertRaises(W.WorkerError):
                with self.composed(f, initialization_failure=True):
                    self.fail("foreign source binding must not initialize")
            self.assertNotIn("sensor_fixture_started", f.events)
            clients = f.observation["state"]["clients"]
            self.assertEqual(len(clients), 1)
            self.assertIsNotNone(clients[0].process.poll())
            scope = json.loads((f.attempt / "GT_WORKER_SCOPE.json").read_bytes())
            self.assertFalse(scope["complete"])

    def test_inflight_real_child_shutdown_aborts_before_ownership_snapshot(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(R.RuntimeErrorGT, "has not drained"):
                with self.composed(f):
                    client = f.observation["state"]["clients"][0]
                    os.kill(client.process.pid, signal.SIGSTOP)  # Owned test child only.
                    f.start_job()
                    deadline = time.monotonic() + 3.
                    while not client.lock.locked() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(client.lock.locked())
                    f.owner.finish()
            self.assertIsNotNone(client.process.poll())
            self.assertFalse(f.owner.evaluation_worker.is_alive())
            self.assertFalse(f.owner._source_ownership.pending)
            self.assertFalse((f.attempt / "complete_gt_stage_timing").exists())
            ownership = json.loads((f.attempt / "source_ownership/MANIFEST.json").read_bytes())
            self.assertTrue(ownership["worker_drained"])
            self.assertFalse(ownership["complete"])
            timeline = json.loads((f.attempt / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
            self.assertFalse(timeline["complete"])
            self.assertTrue(timeline["child_stopped"])


if __name__ == "__main__":
    unittest.main()

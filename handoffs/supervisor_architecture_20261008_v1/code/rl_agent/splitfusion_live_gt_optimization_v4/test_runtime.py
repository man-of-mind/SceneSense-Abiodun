"""Real retained collector/finalizer ordering, with a fake CPU process only."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from . import runtime as R, worker as W
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_parity_repair_v2.gt_timing import StageLog, install_gt_timing
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY


class _Thread:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.alive = False


class RuntimeCompositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events, self.clients = [], []
        self.failure = None
        case = self

        class Catalogue:
            def __init__(self, directory):
                self.directory = Path(directory)
                self.directory.mkdir()

            def require_world(self, world):
                case.events.append("catalogue_checked")

        class Base:
            def __init__(self, *, attempt_dir):
                self.attempt_dir = Path(attempt_dir)
                self.edge_evidence_dir = self.attempt_dir / "targets"
                self.edge_evidence_dir.mkdir()
                self.world, self.ego = object(), SimpleNamespace(id=1)
                self._quality_scene_lock = threading.Lock()
                self._quality_scenes = {}
                self.live = SimpleNamespace(last_datagram={}, _run4_identity={},
                    identity_for_frame=lambda frame: {"frame_id": frame, "session_id": "unit"})
                self.evaluation_worker = _Thread()
                self.evaluation_queue = SimpleNamespace(unfinished_tasks=0)
                self._spawn_sensors()

            def _spawn_sensors(self):
                case.events.append("sensors_started")

            def _ground_truth(self, **kwargs):
                case.events.append("base_ground_truth")
                return ["unchanged_legacy_rows"]

            def finish(self):
                case.events.append("retained_finish")
                if case.failure != "undrained":
                    self.evaluation_worker.alive = False
                return True

        class Client:
            def __init__(self, **kwargs):
                case.events.append("child_start")
                self.kwargs = kwargs
                self.root = Path(kwargs["attempt_dir"])
                (self.root / "engine").mkdir(parents=True)
                self.failed = self.closed = False
                self.count = 0
                self.lock = threading.Lock()
                self.process = SimpleNamespace(pid=987654, poll=lambda: None if not self.closed else int(self.failed))
                self.ready = {"cuda_initialized": False, "process_context": {"affinity": list(range(24))}}
                self.sync = None
                self.log = StageLog(kwargs["max_frames"])
                case.clients.append(self)

            def run(self, *, kwargs, sync_config, **rest):
                case.events.append("child_request")
                self.sync = self.sync or SY.LiveSyncEvidence(Path(sync_config["directory"]),
                    raw_bundle_limit=sync_config["raw_bundle_limit"], raw_byte_limit=sync_config["raw_byte_limit"])
                identity = kwargs["identity"]
                record = SY.source_alignment_record(identity=identity, **{name: kwargs[name] for name in
                    ("rgb", "semantic_image", "depth_image", "frozen_world", "policy_timestamp_s")})
                if case.failure == "reject":
                    record["sources"]["depth"]["timestamp"] += .1
                try:
                    self.sync.check_and_record(record)
                except SY.SourceSynchronizationError:
                    self._flush()
                    self.failed = self.closed = True
                    error = W.WorkerError("source rejected")
                    error.worker_result = dict(identity=identity, sync=self.sync.summary())
                    raise error
                for stage in ("avo_scoring_and_sidecar", "complete_source_build", "prepare_complete_target",
                              "publish_complete_target", "archive_auxiliary_sources"):
                    self.log.rows.append(dict(stage=stage, frame_id=identity["frame_id"],
                        start_raw_ns=1, end_raw_ns=2, wall_ns=1, thread_cpu_ns=1, completed=True))
                self.count += 1
                output = dict(complete_identity=identity, dummy_payload="exact")
                result = dict(identity=identity, record=output, sync=self.sync.summary(),
                    ipc={"frame_id": identity["frame_id"], "child_pid": self.process.pid})
                if case.failure == "after_target":
                    self._flush()
                    self.failed = self.closed = True
                    error = W.WorkerError("original source archive failed")
                    error.worker_result = result
                    error.avo_evidence_record = {"frame_id": identity["frame_id"], "error": "original source archive failed"}
                    raise error
                return result

            def _flush(self):
                if not self.log.closed:
                    self.log.flush(self.root / "engine/complete_gt_stage_timing", worker_drained=True)

            def close(self):
                case.events.append("child_close")
                if case.failure == "close_alive":
                    raise W.WorkerError("close failed with child alive")
                if case.failure != "lost_timing":
                    self._flush()
                self.closed = True
                if case.failure == "close_dead":
                    raise W.WorkerError("close failed after child exited")

            def abort(self):
                case.events.append("child_abort")
                self.failed = self.closed = True

        self.Base, self.Client = Base, Client
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(G, "FreshCatalogue", Catalogue))
        original_factory = LG.build_live_gt_collector

        def sensors_without_rpc(base):
            real = original_factory(base)
            def spawn(owner):
                return super(real, owner)._spawn_sensors()
            real._spawn_sensors = spawn  # Only real CARLA sensor RPC is replaced.
            return real
        self.stack.enter_context(patch.object(LG, "build_live_gt_collector", sensors_without_rpc))

        def preserve(records, directory, **kwargs):
            self.events.append("original_preserve")
            self.assertTrue(self.clients[0].closed)
            self.assertFalse(self.owner.evaluation_worker.is_alive())
            self.assertEqual(self.owner._avo_sync.accepted, len(records) + len(kwargs["rejected_records"]))
            if self.failure == "preserve_failed":
                raise OSError("original preserve failed")
            return dict(directory=str(directory), preserved=True)
        self.stack.enter_context(patch.object(LG, "preserve_avo_evidence", preserve))
        self.stack.enter_context(patch.object(G, "preserve_complete_targets",
            lambda records, destination: {"complete": True, "frames": len(records)}))

    def scope(self, **kwargs):
        return R.install_cpu_gt_worker(attempt_dir=self.root, max_frames=10,
            expected_sources=W.sources(), _client_factory=self.Client, **kwargs)

    def collector(self):
        self.owner = LG.build_live_gt_collector(self.Base)(attempt_dir=self.root)
        return self.owner

    def request(self, frame):
        owner = self.owner
        matrix = np.eye(4)
        image = SimpleNamespace(frame=frame, timestamp=float(frame), width=2, height=2,
            raw_data=b"\0"*16, transform=SimpleNamespace(get_matrix=lambda: matrix.tolist()))
        rgb = SY.sensor_metadata(image)
        rgb["_rgb_image"] = image
        for key in ("semantic_callback", "depth_callback", "radar_callback"):
            rgb[key] = SY.sensor_metadata(image)
        owner._avo_rgb[frame] = rgb
        owner._avo_semantic[frame] = (image, rgb["semantic_callback"])
        owner._avo_depth[frame] = (image, rgb["depth_callback"])
        owner._avo_radar[frame] = rgb["radar_callback"]
        owner._quality_scenes[frame] = SimpleNamespace(frozen_frame_id=frame, frozen_timestamp_s=float(frame))
        owner.live.last_datagram[frame] = W.raw()
        owner.live._run4_identity[frame] = {"reward_requested": True}
        return owner._ground_truth(frame_id=frame, timestamp=float(frame), camera_matrix=matrix,
            radar_points={"world_xyz": np.zeros((0, 3))}, world=owner.world)

    def test_actual_factory_and_finish_keep_order_and_cumulative_sync(self):
        with self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.assertLess(self.events.index("child_start"), self.events.index("sensors_started"))
                self.assertEqual(self.request(10), ["unchanged_legacy_rows"])
                self.request(12)
                owner.finish()
                self.assertLess(self.events.index("retained_finish"), self.events.index("child_close"))
                self.assertLess(self.events.index("child_close"), self.events.index("original_preserve"))
        self.assertTrue(state["complete"])
        self.assertEqual(state["child_timing"]["records"], 10)
        self.assertEqual(state["parent_ipc_records"][0]["stage"], "parent_snapshot_and_ipc")
        summary = json.loads((self.root / "run5_live_avo_gt_summary.json").read_text())
        self.assertEqual(summary["source_synchronization"]["accepted"], 2)
        self.assertEqual(summary["all_tick_sensor_barrier"]["rejections"]["records"], [])
        self.assertEqual(self.clients[0].kwargs["semantic_attempt_dir"], self.root)
        self.assertTrue((self.root / "run5_live_sync_evidence/frame_0000000010.observed.json").exists())

    def test_failure_mirrors_sync_and_retains_original_rejected_record(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.failure = "after_target"
                with self.assertRaises(W.WorkerError):
                    self.request(10)
                self.assertEqual(owner._avo_sync.accepted, 1)
                self.assertEqual(len(owner._avo_rejected_results), 1)
                owner.finish()
        self.assertFalse(state["complete"])
        self.assertTrue(state["joint_drained"])
        self.assertTrue(state["preserved_before_teardown"])

    def test_source_rejection_mirrors_real_durable_summary(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.failure = "reject"
                with self.assertRaises(W.WorkerError):
                    self.request(10)
                self.assertEqual(owner._avo_sync.rejected, 1)
                self.assertTrue((owner._avo_sync.directory / "frame_0000000010.rejected.json").is_file())
                self.assertEqual(owner._avo_barrier_rejections.rejected, 0)
                owner.finish()
        self.assertFalse(state["complete"])

    def test_undrained_worker_forbids_preserve_and_aborts_child(self):
        with self.assertRaises(R.RuntimeErrorGT):
            with self.scope() as state:
                with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                    owner = self.collector()
                    self.failure = "undrained"
                    owner.finish()
        self.assertFalse(state["complete"])
        self.assertNotIn("original_preserve", self.events)
        self.assertIn("child_abort", self.events)
        self.assertFalse((self.root / "complete_gt_stage_timing").exists())

    def test_setup_without_reward_cleans_up_without_false_complete(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                self.collector()
        self.assertFalse(state["complete"])
        self.assertIn("child_abort", self.events)
        self.assertIsNotNone(self.clients[0].process.poll())

    def test_zero_reward_normal_finish_preserves_empty_child_timing(self):
        with self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                self.collector().finish()
        self.assertTrue(state["complete"])
        self.assertEqual(state["child_timing"]["records"], 0)

    def test_existing_parent_timing_scope_is_refused(self):
        with install_gt_timing(attempt_dir=self.root, max_frames=10, drain_check=lambda: True):
            with self.assertRaisesRegex(R.RuntimeErrorGT, "parent complete-GT timing"):
                with self.scope():
                    self.fail("must not enter")

    def test_parent_timing_installed_late_is_also_refused(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "parent complete-GT timing"):
            with self.scope():
                with install_gt_timing(attempt_dir=self.root, max_frames=10, drain_check=lambda: True):
                    with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                        self.fail("must not enter")

    def test_merge_rejects_regressing_or_foreign_history(self):
        owner = SimpleNamespace(_avo_sync=SY.LiveSyncEvidence(self.root / "sync"))
        summary = owner._avo_sync.summary()
        summary.update(accepted=1, records=[{"frame_id": 10, "status": "accepted"}])
        R._merge_sync(owner, summary, frame=10, max_frames=10)
        summary["records"][0]["frame_id"] = 8
        with self.assertRaisesRegex(R.RuntimeErrorGT, "history"):
            R._merge_sync(owner, summary, frame=12, max_frames=10)

    def test_dead_child_missing_timing_does_not_discard_prior_targets(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.request(10)
                self.failure = "lost_timing"
                owner.finish()
        self.assertFalse(state["complete"])
        self.assertTrue(state["joint_drained"])
        self.assertTrue(state["preserved_before_teardown"])
        self.assertIsNone(state["child_timing"])
        self.assertIn("original_preserve", self.events)
        self.assertTrue((self.root / "run5_live_avo_gt_summary.json").is_file())
        self.assertFalse((self.root / "complete_gt_stage_timing").exists())

    def test_parent_barrier_rejection_remains_a_separate_untouched_stream(self):
        with self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                owner._avo_barrier_rejections.record_rejection({"identity": {"frame_id": 9}},
                    SY.SourceSynchronizationError("SOURCE_CALLBACK_NOT_COMPLETED", "depth"))
                parent_before = owner._avo_barrier_rejections.summary()
                self.request(10)
                owner.finish()
                self.assertEqual(parent_before, owner._avo_barrier_rejections.summary())
        summary = json.loads((self.root / "run5_live_avo_gt_summary.json").read_text())
        self.assertEqual(summary["source_synchronization"]["accepted"], 1)
        self.assertEqual(summary["source_synchronization"]["rejected"], 0)
        self.assertEqual(summary["all_tick_sensor_barrier"]["rejections"]["rejected"], 1)

    def test_inflight_ipc_forbids_retained_preservation(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "in flight"):
            with self.scope() as state:
                with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                    owner = self.collector()
                    self.clients[0].lock.acquire()
                    try:
                        owner.finish()
                    finally:
                        self.clients[0].lock.release()
        self.assertFalse(state["complete"])
        self.assertNotIn("original_preserve", self.events)

    def test_scope_evidence_write_failure_is_not_silently_successful(self):
        with patch.object(R, "_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                with self.scope():
                    pass

    def test_evidence_failure_does_not_replace_primary_exception(self):
        with patch.object(R, "_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(ValueError, "primary"):
                with self.scope():
                    raise ValueError("primary")

    def test_nested_scope_refused_without_replacing_outer_hooks(self):
        with self.scope():
            install = G.install_complete_gt
            with self.assertRaisesRegex(R.RuntimeErrorGT, "nested"):
                with self.scope():
                    self.fail("must not enter")
            self.assertIs(G.install_complete_gt, install)
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                self.collector().finish()

    def test_partial_sync_counter_before_failed_durable_write_preserved(self):
        owner = SimpleNamespace(_avo_sync=SY.LiveSyncEvidence(self.root / "sync"))
        partial = owner._avo_sync.summary()
        partial["accepted"] = 1
        with self.assertRaisesRegex(R.RuntimeErrorGT, "count"):
            R._merge_sync(owner, partial, frame=10, max_frames=10)
        R._merge_sync(owner, partial, frame=10, max_frames=10, allow_partial=True)
        self.assertEqual(owner._avo_sync.summary(), partial)

    def test_close_error_after_child_exit_preserves_but_fails_scope(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "did not complete"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.request(10)
                self.failure = "close_dead"
                owner.finish()
        self.assertIn("original_preserve", self.events)
        self.assertTrue(state["joint_drained"])
        self.assertTrue(state["preserved_before_teardown"])
        self.assertFalse(state["worker_close_succeeded"])
        self.assertFalse(state["complete"])

    def test_close_error_with_child_alive_blocks_copy_then_aborts(self):
        with self.assertRaisesRegex(R.RuntimeErrorGT, "has not exited"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.request(10)
                self.failure = "close_alive"
                owner.finish()
        self.assertNotIn("original_preserve", self.events)
        self.assertIn("child_abort", self.events)
        self.assertFalse(state["joint_drained"])
        self.assertFalse(state["preserved_before_teardown"])
        self.assertFalse((self.root / "complete_gt_stage_timing").exists())
        self.assertIsNotNone(self.clients[0].process.poll())

    def test_original_preservation_error_never_claims_preserved(self):
        with self.assertRaisesRegex(OSError, "original preserve failed"), self.scope() as state:
            with G.install_complete_gt(attempt_dir=self.root, max_frames=10):
                owner = self.collector()
                self.request(10)
                self.failure = "preserve_failed"
                owner.finish()
        self.assertTrue(state["joint_drained"])
        self.assertTrue(state["preservation_started"])
        self.assertFalse(state["preserved_before_teardown"])
        self.assertFalse(state["complete"])


if __name__ == "__main__":
    unittest.main()

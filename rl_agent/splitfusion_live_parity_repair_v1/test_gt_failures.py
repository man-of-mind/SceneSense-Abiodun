"""Post-AVO fault preservation through the actual existing GT-worker composition."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from . import gt_bridge as B, source_geometry as S, target_transport as T
from . import test_source_geometry as TS


class InjectedStageFailure(RuntimeError):
    pass


class GroundTruthFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        B.bind_legacy_namespace()

    @contextmanager
    def worker_fixture(self):
        """Same real LG subclass composition used by test_gt_bridge, no workers started."""
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        fixture = TS.SourceGeometryTests()
        with fixture.fixture(pitch=0) as (root, args):
            attempt = root / "attempt"; attempt.mkdir()
            args["rgb"]["_rgb_image"] = args["semantic_image"]
            frozen = args["frozen_world"]
            actors = TS.Actors((TS.actor(1, "walker.pedestrian.0001", 20., -3.),
                                TS.actor(2, "vehicle.test", 20., 4.)))
            frozen.get_actors = lambda: actors
            frozen.population_evidence = dict(args["dynamic_snapshot_evidence"], registered_population_ids=[1, 2],
                population={"registered_actors": [1, 2]}, population_sha256=B.sha({"registered_actors": [1, 2]}))
            with B.install_complete_gt(attempt_dir=attempt, max_frames=2) as installed:
                shutil.copytree(root / "catalogue", installed["catalogue"].directory)
                class Base:
                    def _ground_truth(self, **kwargs): return []
                cls = LG.build_live_gt_collector(Base)
                obj = cls.__new__(cls)
                obj._complete_producer = None
                obj._quality_scene_lock = threading.Lock()
                obj._quality_scenes = {100: frozen}
                obj.live = SimpleNamespace(_run4_identity={100: {"reward_requested": True}}, last_datagram={100: 1},
                    identity_for_frame=lambda frame: args["identity"])
                obj.ego = SimpleNamespace(id=999)
                obj.edge_evidence_dir = root / "edge"; obj.edge_evidence_dir.mkdir()
                obj._avo_results, obj._avo_rejected_results = [], []
                obj._avo_sync = None
                obj._avo_sources = mock.Mock(return_value=(args["rgb"], args["semantic_image"], args["depth_image"]))
                kwargs = dict(frame_id=100, timestamp=5., camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                              radar_points={"world_xyz": args["radar_world_xyz"]})
                yield root, attempt, args, obj, kwargs, LG

    @contextmanager
    def fail_stage(self, stage, identity):
        ident = threading.get_ident()
        def fail(*args, **kwargs):
            self.assertEqual(threading.get_ident(), ident)
            raise InjectedStageFailure(stage)
        if stage == "complete_source_build":
            with mock.patch.object(S.LiveCompleteTargetProducer, "build", side_effect=fail) as injected:
                yield injected
        elif stage == "prepare_target":
            with mock.patch.object(B.evaluation, "prepare_target", side_effect=fail) as injected:
                yield injected
        elif stage == "publish_target":
            original_write, original_payload = T.write_bundle, T._write_x
            def partial_write(directory, **kwargs):
                destination = T.bundle_path(directory, identity)
                def payload(path, value):
                    if Path(path) == destination / "raw_semantic.npy":
                        fail()
                    return original_payload(path, value)
                with mock.patch.object(T, "_write_x", side_effect=payload):
                    return original_write(directory, **kwargs)
            with mock.patch.object(T, "write_bundle", side_effect=partial_write) as injected:
                yield injected
        elif stage == "archive_sources":
            def partial_archive(directory, **kwargs):
                directory = Path(directory); directory.mkdir(exist_ok=False)
                T._write_x(directory / "PARTIAL_ARCHIVE.json", B.canonical({"stage": stage}))
                fail()
            with mock.patch.object(B, "write_source_archive", side_effect=partial_archive) as injected:
                yield injected
        else:
            raise AssertionError(stage)

    def exercise(self, stage):
        with self.worker_fixture() as (root, attempt, args, obj, kwargs, LG):
            with self.fail_stage(stage, args["identity"]) as injected, \
                    mock.patch.object(threading.Thread, "start", side_effect=AssertionError("new GT thread forbidden")) as thread_start, \
                    mock.patch.object(B.evaluation, "evaluate_prepared", side_effect=AssertionError("GT producer must not score")) as scorer:
                with self.assertRaisesRegex(InjectedStageFailure, stage) as caught:
                    obj._ground_truth(**kwargs)
                injected.assert_called_once()
                thread_start.assert_not_called(); scorer.assert_not_called()
            obj._avo_sources.assert_called_once_with(100)
            self.assertEqual(obj._avo_results, [])
            self.assertEqual(len(obj._avo_rejected_results), 1)
            rejected = obj._avo_rejected_results[0]
            self.assertEqual(rejected, caught.exception.avo_evidence_record)
            old_path = Path(rejected["path"])
            self.assertTrue(old_path.is_file())
            self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), rejected["sha256"])

            target = T.bundle_path(obj.edge_evidence_dir, args["identity"])
            failed = attempt / "parity_rejected_targets" / target.name
            rejection = json.loads((failed / "REJECTION.json").read_text())
            self.assertEqual(rejection["identity"], args["identity"])
            self.assertEqual(rejection["stage"], stage)
            self.assertFalse(rejection["live_qualified"])
            self.assertEqual(rejection["target_ready"], stage == "archive_sources")
            for name, digest in rejection["sha256"].items():
                self.assertEqual(hashlib.sha256((failed / name).read_bytes()).hexdigest(), digest)
            for name in ("rgb_bgra.npy", "depth_bgra.npy", "raw_semantic_bgra.npy", "radar_world_xyz.npy", "SOURCE_EVIDENCE.json"):
                self.assertIn(name, rejection["sha256"])
            expected_depth = np.frombuffer(args["depth_image"].raw_data, np.uint8).reshape(720, 1280, 4)
            np.testing.assert_array_equal(np.load(failed / "depth_bgra.npy", allow_pickle=False), expected_depth)
            np.testing.assert_array_equal(np.load(failed / "radar_world_xyz.npy", allow_pickle=False), args["radar_world_xyz"])

            if stage == "archive_sources":
                # Scoring inputs were genuinely complete before archival failed.
                # They are not retroactively erased or mislabeled incomplete;
                # the route still failed and qualification remains invalid.
                restored = T.verify_full_inputs(obj.edge_evidence_dir, expected_identity=args["identity"])
                self.assertFalse(restored.manifest["live_qualified"])
                self.assertTrue((failed / target.name / T.READY_NAME).is_file())
                self.assertTrue((failed / (target.name + ".sources") / "PARTIAL_ARCHIVE.json").is_file())
            else:
                self.assertFalse((target / T.READY_NAME).exists())
                with self.assertRaises(T.TargetNotReadyError):
                    T.read_bundle(obj.edge_evidence_dir, expected_identity=args["identity"], timeout_s=0)
                if stage == "publish_target":
                    self.assertTrue((failed / target.name / "inputs.json").is_file())
                    self.assertTrue((failed / target.name / "depth_bgra.npy").is_file())

            # Existing LG preservation must retain original AVO inputs even
            # when there are zero successful complete-target records.
            preservation = LG.preserve_avo_evidence(obj._avo_results, root / "preserved",
                                                    rejected_records=obj._avo_rejected_results)
            self.assertTrue(preservation["all_avo_inputs_preserved"])
            self.assertEqual(preservation["complete_targets"]["frames"], 0)
            preserved = root / "preserved" / old_path.name
            self.assertEqual(hashlib.sha256(preserved.read_bytes()).hexdigest(), rejected["sha256"])
            manifest = json.loads((root / "preserved" / "MANIFEST.json").read_text())
            self.assertIn(old_path.name, manifest["sha256"])
            # Hide this test's owned scratch root: preserved inputs/rejections
            # must be independent of its continued presence.
            obj.edge_evidence_dir.rename(root / "hidden_teardown_scratch")
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((root / "preserved" / name).read_bytes()).hexdigest(), digest)
            for name, digest in rejection["sha256"].items():
                self.assertEqual(hashlib.sha256((failed / name).read_bytes()).hexdigest(), digest)
            self.assertFalse(torch.cuda.is_initialized())

    def test_build_failure_retains_original_avo_and_native_sources(self):
        self.exercise("complete_source_build")

    def test_prepare_failure_retains_original_avo_and_native_sources(self):
        self.exercise("prepare_target")

    def test_partial_target_write_never_publishes_ready(self):
        self.exercise("publish_target")

    def test_archive_failure_after_ready_invalidates_qualification(self):
        self.exercise("archive_sources")

    def test_failed_next_frame_never_reuses_previous_source_evidence(self):
        from rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync import sensor_metadata
        with self.worker_fixture() as (root, attempt, args, obj, kwargs, LG):
            obj._ground_truth(**kwargs)
            self.assertEqual(len(obj._avo_results), 1)
            first = obj._avo_results[0]
            original_record = B.canonical(first)
            first_identity = dict(first["complete_identity"])
            originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in obj.edge_evidence_dir.rglob("*") if path.is_file()}
            self.assertEqual(obj._complete_producer.last_evidence["identity"]["frame_id"], 100)

            # A new immutable owned source, not a relabelled prior target.
            frame, stamp = 101, 5.05
            semantic = SimpleNamespace(**vars(args["semantic_image"]))
            depth = SimpleNamespace(**vars(args["depth_image"]))
            for image in (semantic, depth): image.frame, image.timestamp = frame, stamp
            altered = np.frombuffer(semantic.raw_data, np.uint8).copy().reshape(720, 1280, 4)
            altered[0, 0, 0] = 17
            semantic.raw_data = altered.tobytes()
            depth.raw_data = TS.encode_depth(21.).tobytes()
            rgb = sensor_metadata(semantic, receipt_raw_ns=2)
            rgb["_rgb_image"] = semantic
            rgb["radar_callback"] = sensor_metadata(semantic, receipt_raw_ns=2)
            population = json.loads(B.canonical(args["frozen_world"].population_evidence))
            population["frame_id"] = frame
            frozen = SimpleNamespace(**vars(args["frozen_world"]))
            frozen.frozen_frame_id, frozen.frozen_timestamp_s = frame, stamp
            frozen.population_evidence = population
            second_identity = dict(first_identity, frame_id=frame, capture_timestamp_ns=int(stamp*1e9))
            obj._quality_scenes[frame] = frozen
            obj.live._run4_identity[frame] = {"reward_requested": True}
            obj.live.last_datagram[frame] = 1
            obj.live.identity_for_frame = lambda _: second_identity
            obj._avo_sources.return_value = (rgb, semantic, depth)
            second_kwargs = dict(kwargs, frame_id=frame, timestamp=stamp)
            with self.fail_stage("complete_source_build", second_identity), \
                    mock.patch.object(threading.Thread, "start", side_effect=AssertionError("new GT thread forbidden")) as thread_start:
                with self.assertRaises(InjectedStageFailure): obj._ground_truth(**second_kwargs)
                thread_start.assert_not_called()

            self.assertEqual(len(obj._avo_results), 1)
            self.assertEqual(B.canonical(obj._avo_results[0]), original_record)
            self.assertEqual(len(obj._avo_rejected_results), 1)
            self.assertEqual(obj._complete_producer.last_evidence["identity"], first_identity)
            target = T.bundle_path(obj.edge_evidence_dir, second_identity)
            rejected_root = attempt / "parity_rejected_targets" / target.name
            source = json.loads((rejected_root / "SOURCE_EVIDENCE.json").read_text())
            self.assertEqual(source["identity"], second_identity)
            self.assertEqual(source["population"]["frame_id"], frame)
            self.assertTrue(source["incomplete_current_source"])
            self.assertEqual(source["failed_stage"], "complete_source_build")
            self.assertNotIn("dynamic_corners", source)  # never copies frame100's completed geometry
            rejection = json.loads((rejected_root / "REJECTION.json").read_text())
            self.assertEqual(rejection["identity"], second_identity)
            self.assertFalse(rejection["target_ready"])
            for name, digest in rejection["sha256"].items():
                self.assertEqual(hashlib.sha256((rejected_root / name).read_bytes()).hexdigest(), digest)
            np.testing.assert_array_equal(np.load(rejected_root / "depth_bgra.npy", allow_pickle=False), TS.encode_depth(21.))
            np.testing.assert_array_equal(np.load(rejected_root / "rgb_bgra.npy", allow_pickle=False), altered)
            with self.assertRaises(T.TargetNotReadyError):
                T.read_bundle(obj.edge_evidence_dir, expected_identity=second_identity, timeout_s=0)
            for path, digest in originals.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            restored = T.verify_full_inputs(obj.edge_evidence_dir, expected_identity=first_identity,
                expected_manifest_sha256=first["complete_target"]["manifest_sha256"])
            self.assertEqual(restored.prepared.frame_id, 100)
            preserved = LG.preserve_avo_evidence(obj._avo_results, root / "preserved_two_frames",
                rejected_records=obj._avo_rejected_results)
            self.assertEqual(preserved["complete_targets"]["frames"], 1)
            for record in [*obj._avo_results, *obj._avo_rejected_results]:
                kept = root / "preserved_two_frames" / Path(record["path"]).name
                self.assertEqual(hashlib.sha256(kept.read_bytes()).hexdigest(), record["sha256"])
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__": unittest.main()

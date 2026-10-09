"""No CARLA process/GPU: exercise exact live producer and edge consumption."""
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np

from . import edge_quality as E, live_gt as G, quality_adapter as Q


class Transform:
    def __init__(self, x=0):
        self.matrix = np.eye(4)
        self.matrix[0, 3] = x
        self.rotation = S(yaw=0., pitch=0., roll=0.)

    def get_matrix(self):
        return self.matrix.tolist()


class Actors(list):
    def filter(self, pattern):
        return [a for a in self if a.type_id.startswith("walker.pedestrian.")]


def frozen_world():
    transform = Transform(10.)
    actor = S(id=42, type_id="walker.pedestrian.1", bounding_box=S(
        location=S(x=0., y=0., z=0.), extent=S(x=.25, y=.25, z=.9)),
        get_transform=lambda: transform)
    return S(frozen_frame_id=10, frozen_timestamp_s=1., frozen_observed_raw_ns=1,
             get_actors=lambda: Actors([actor]))


def image(kind="depth", frame=10, timestamp=1.):
    bgra = np.zeros((200, 200, 4), dtype=np.uint8)
    if kind == "depth":
        value = int(round(9.8 / 1000. * 16777215))
        bgra[..., 2] = value & 255
        bgra[..., 1] = (value >> 8) & 255
        bgra[..., 0] = (value >> 16) & 255
    else:
        bgra[80:120, 98:103, 2] = 12
    return S(frame=frame, timestamp=timestamp, width=200, height=200,
             raw_data=bgra.tobytes(), transform=Transform())


def identity():
    return dict(run_id="r", cell_id="c", stream_id="s", frame_id=10,
                action_id=71, profile_id="p", capture_timestamp_ns=20)


def arguments():
    rgb = G.sensor_metadata(image("rgb"))
    rgb["_rgb_image"] = image("rgb")
    rgb["radar_callback"] = G.sensor_metadata(image())
    return dict(identity=identity(), rgb=rgb, semantic_image=image("semantic"), depth_image=image(),
                frozen_world=frozen_world(), camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                policy_timestamp_s=1.)


class LiveGtTests(unittest.TestCase):
    def test_depth_decoder_matches_frozen_decoder(self):
        from data_collection.route_b_publication_actor_volume_visibility_v1.run_audit import decode_depth_bgra
        img = image()
        expected = decode_depth_bgra(np.frombuffer(img.raw_data, dtype=np.uint8).reshape(200, 200, 4))
        np.testing.assert_array_equal(G.decode_depth(img), expected)

    def test_raw_mask_correction_and_no_input_mutation(self):
        img = image("semantic")
        before = img.raw_data
        self.assertEqual(G.corrected_semantic_image(img)[90, 100], 2)
        self.assertEqual(img.raw_data, before)

    def test_native_calibration(self):
        k = G.native_intrinsics(1280, 720)
        self.assertEqual(k[0, 2], 640)
        self.assertEqual(k[1, 2], 360)
        self.assertEqual(k[0, 0], k[1, 1])

    def test_frozen_geometry_contains_ids_and_unprojected(self):
        world = frozen_world()
        row = G.frozen_people_rows(world, frame_id=10, timestamp=1., camera_matrix=np.eye(4),
            camera_inverse=np.eye(4), intrinsics=G.native_intrinsics(200, 200), width=200, height=200)[0]
        self.assertEqual(row["gt_actor_id"], "42")
        self.assertEqual(row["object_world_x"], 10.)
        self.assertGreater(row["gt_bbox_area_px"], 12)
        world.frozen_frame_id = 11
        with self.assertRaises(Q.LiveGtContractError):
            G.frozen_people_rows(world, frame_id=10, timestamp=1., camera_matrix=np.eye(4),
                camera_inverse=np.eye(4), intrinsics=G.native_intrinsics(200, 200), width=200, height=200)

    def test_sidecar_exact_identity_and_replayable_sensor_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = G.write_avo_bundle(root, **arguments())
            record = json.loads(Path(result["path"]).read_text())
            self.assertEqual(record["identity"], identity())
            self.assertEqual(record["raw_pedestrian_rows"][0]["gt_actor_id"], "42")
            for payload in record["payloads"].values():
                self.assertEqual(hashlib.sha256((root / payload["filename"]).read_bytes()).hexdigest(), payload["sha256"])
            self.assertEqual(np.load(root / record["payloads"]["depth_bgra"]["filename"]).tobytes(), image().raw_data)
            gt = G.read_avo_bundle(root, expected_identity=identity(), timeout_s=.01)
            self.assertEqual(gt["avo"]["frame_id"], 10)
            self.assertEqual(gt["avo_sidecar_sha256"], result["sha256"])
            self.assertEqual(result["eligible"], 1)

    def test_native_inputs_and_all_actor_rows_written_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = G.score_live_avo_people
            def check(**kwargs):
                path = next(root.glob("*.avo_inputs.json"))
                document = json.loads(path.read_text())
                self.assertEqual(document["raw_pedestrian_rows"], kwargs["raw_actor_rows"])
                self.assertEqual(document["record_kind"], "AVO_PRE_SCORE_INPUTS")
                for payload in document["payloads"].values():
                    self.assertEqual(hashlib.sha256((root/payload["filename"]).read_bytes()).hexdigest(), payload["sha256"])
                return original(**kwargs)
            with patch.object(G, "score_live_avo_people", side_effect=check):
                record = G.write_avo_bundle(root, **arguments())
            final = json.loads(Path(record["path"]).read_text())
            inputs = root/final["prescore_input"]["filename"]
            self.assertEqual(hashlib.sha256(inputs.read_bytes()).hexdigest(), final["prescore_input"]["sha256"])
            self.assertLessEqual(json.loads(inputs.read_text())["write_raw_ns"], final["prescore_input_ready_raw_ns"])
            self.assertLessEqual(final["prescore_input_ready_raw_ns"], final["scoring_start_raw_ns"])

    def test_rejected_score_preserves_native_inputs_and_precise_offender(self):
        args = arguments()
        args["frozen_world"].get_actors()[0].get_transform().rotation.pitch = 3.5
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(Q.LiveGtContractError, "TILTED_SCOREABLE_TARGET") as error:
                G.write_avo_bundle(root/"scratch", **args)
            record = error.exception.avo_evidence_record
            rejected = json.loads(Path(record["path"]).read_text())
            self.assertEqual(rejected["geometry_diagnostics"]["offender"]["gt_actor_id"], "42")
            self.assertEqual(rejected["geometry_diagnostics"]["offender"]["pitch_deg"], 3.5)
            G.preserve_avo_evidence([], root/"retained", rejected_records=[record])
            manifest = json.loads((root/"retained"/"MANIFEST.json").read_text())
            self.assertEqual(len(manifest["sha256"]), 4)
            inputs = json.loads((root/"retained"/rejected["prescore_input"]["filename"]).read_text())
            self.assertEqual(inputs["raw_pedestrian_rows"][0]["object_pitch_deg"], 3.5)
            self.assertFalse(list((root/"scratch").glob("*.avo.json")))

    def test_missing_and_foreign_sidecars_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(Q.LiveGtContractError):
                G.read_avo_bundle(root, expected_identity=identity(), timeout_s=0)
            G.write_avo_bundle(root, **arguments())
            with self.assertRaises(Q.LiveGtContractError):
                G.read_avo_bundle(root, expected_identity={**identity(), "run_id": "other"}, timeout_s=.01)

    def test_preservation_survives_scratch_removal_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = G.write_avo_bundle(root / "scratch", **arguments())
            result = G.preserve_avo_evidence([record], root / "retained")
            self.assertEqual(result["files"], 4)
            manifest = json.loads((root / "retained" / "MANIFEST.json").read_text())
            for path in (root / "scratch").iterdir():
                path.unlink()
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((root / "retained" / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                G.preserve_avo_evidence([], root / "retained")

    def test_installed_edge_evaluator_keeps_wire_and_quality_formula(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as ER
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as GE
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import Pipeline, processor, SPEC
        from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality import evaluate_exact_quality
        processed = processor().process(Pipeline().step().wire, edge_timing={})
        ticket = processed.evaluation
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, :] = 1
        gt = dict(semantic=mask, objects=[dict(class_name="vehicle", world_x=1., world_y=2.)],
                  avo=dict(qualified=[], structural_ignored=[]), gt_ready_wall_ns=1,
                  avo_sidecar_sha256="a" * 64)
        sent = []
        with patch.object(E, "_INSTALLED", False), patch.object(GE, "read_ground_truth", GE.read_ground_truth), \
             patch.object(ER.Run4EvaluatorV2, "_evaluate_with", ER.Run4EvaluatorV2._evaluate_with), \
             patch.object(ER.Run4EvaluatorV2, "GT_COMPONENTS", ER.Run4EvaluatorV2.GT_COMPONENTS), \
             patch.object(W, "LIVE_MEASUREMENT_SEMANTICS", W.LIVE_MEASUREMENT_SEMANTICS):
            E.install_edge_quality_seams()
            evaluator = ER.Run4EvaluatorV2(spec=SPEC, send=sent.append, match_distance_m=3.,
                                           gt_timeout_s=.1, read_ground_truth=lambda **kw: gt)
            evaluator.evaluate(ticket)
            feedback, _ = W.decode_feedback(sent[0])
            measured = evaluator.records[0]["measurement"]
            self.assertEqual(feedback.q_perc, evaluate_exact_quality(SPEC, measured).q_perc)
            self.assertEqual(measured["avo_sidecar_sha256"], "a" * 64)
            self.assertEqual(evaluator.records[0]["measurement_semantics"], G.MEASUREMENT_SEMANTICS)
            evaluator.evaluate(ticket)
            self.assertEqual(len(sent), 1)

    def test_different_frame_timestamp_pose_fail_before_publication(self):
        for change in ({"depth_image": image(frame=11)}, {"depth_image": image(timestamp=1.01)},
                       {"semantic_image": image("semantic", frame=11)}, {"camera_matrix": np.diag([1, 1, 1, 2])}):
            with self.subTest(change=list(change)), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(Q.LiveGtContractError):
                    G.write_avo_bundle(Path(tmp), **{**arguments(), **change})
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_avo_person_matching_exact_including_ties(self):
        from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1.run_comparison import greedy_match
        rng = np.random.default_rng(88)
        for _ in range(50):
            pred = [dict(world_x=float(rng.integers(0, 10)), world_y=0.) for i in range(8)]
            gt = [dict(world_x=float(rng.integers(0, 10)), world_y=0.) for i in range(5)]
            for indices in (None, {1, 2, 4}):
                self.assertEqual(Q.greedy_match(pred, gt, indices), greedy_match(pred, gt, indices))

    def test_measurement_vehicle_unchanged_person_ignored(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        args = dict(frame_id=10, predicted_mask=np.array([[0, 1, 2]], dtype=np.uint8),
                    ground_truth_mask=np.array([[0, 1, 2]], dtype=np.uint8),
                    predictions=[dict(class_name="vehicle", world_x=1., world_y=0., score=.8),
                                 dict(class_name="person", world_x=10., world_y=0., score=.8)],
                    ground_truth_objects=[dict(class_name="vehicle", world_x=1., world_y=0.),
                                          dict(class_name="person", world_x=10., world_y=0.)], match_distance_m=3.)
        old = W.live_measurement(**args)
        value = E.measurement(**args, avo=dict(qualified=[dict(world_x=10., world_y=0., avo=.2)], structural_ignored=[]))
        for key in old:
            if not key.startswith("loc_person_"):
                self.assertEqual(value[key], old[key])
        self.assertEqual(value["loc_person_eligible_gt"], 0)
        self.assertEqual(value["loc_person_fp"], 0)
        self.assertEqual(value["loc_person_avo_ignored_predictions"], 1)

    def test_bounded_callback_and_source_consume(self):
        class Base:
            def __init__(self, **kwargs): pass
            def _on_rgb(self, image): pass
            def _on_semantic(self, image): pass
            def _on_radar(self, image): pass
        obj = G.build_live_gt_collector(Base)()
        for i in range(30):
            obj._on_depth(image(frame=i))
            obj._on_semantic(image("semantic", frame=i))
            obj._on_rgb(image(frame=i))
            obj._on_radar(image(frame=i))
        self.assertEqual(len(obj._avo_depth), 24)
        self.assertEqual(len(obj._avo_sources(29)), 3)
        self.assertNotIn(29, obj._avo_depth)

    def test_existing_worker_build_after_send_only(self):
        events = []
        class Base:
            def __init__(self, **kwargs):
                self._quality_scene_lock = threading.Lock()
                self._quality_scenes = {10: frozen_world()}
                self.live = S(_run4_identity={10: {"reward_requested": True}},
                    last_datagram=S(get=lambda frame: 1), identity_for_frame=lambda frame: identity())
                self.edge_evidence_dir = Path("unused")
            def _ground_truth(self, **kwargs):
                events.append("original")
                self._quality_scenes.pop(10)
                return []
        obj = G.build_live_gt_collector(Base)()
        obj._on_depth(image())
        obj._avo_put(obj._avo_semantic, 10, (image("semantic"), G.sensor_metadata(image("semantic"))))
        obj._avo_put(obj._avo_rgb, 10, arguments()["rgb"])
        with patch.object(G, "write_avo_bundle", side_effect=lambda *a, **kw: events.append("avo") or {}) as write:
            obj._ground_truth(frame_id=10, timestamp=1., camera_matrix=np.eye(4), camera_inverse=np.eye(4))
            self.assertEqual(events, ["original", "avo"])
            self.assertEqual(write.call_args.kwargs["frozen_world"].frozen_frame_id, 10)
        obj = G.build_live_gt_collector(Base)()
        obj.live.last_datagram.get = lambda frame: None
        with patch.object(G, "write_avo_bundle") as write, self.assertRaises(Q.LiveGtContractError):
            obj._ground_truth(frame_id=10)
        write.assert_not_called()

    def test_real_mro_aligned_diagnostic_never_reenters_avo(self):
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as A
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        # Real inheritance and both real wrappers, replacing only the costly
        # leaf geometry calculation and disk publication, not dispatch logic.
        cls = U.build_run4_collector_class(G.build_live_gt_collector(A.QualityPassiveSplitCollector))
        for aligned_first in (True, False):
            with self.subTest(aligned_first=aligned_first):
                obj = cls.__new__(cls)
                obj._quality_scene_lock = threading.Lock()
                obj._quality_scenes = {10: frozen_world()}
                obj.aligned_actor_tracker = object()
                obj._avo_results, obj._avo_sync = [], None
                obj.edge_evidence_dir = Path("unused")
                obj.live = S(_run4_identity={10: {"reward_requested": True}},
                    last_datagram=S(get=lambda frame: 1), identity_for_frame=lambda frame: identity(),
                    gt_log=S(rows_started=lambda *a: None, completed=lambda *a, **k: None))
                obj._run4_gate = S(close=lambda *a: None)
                args = arguments()
                source_reads = []
                obj._avo_sources = lambda frame: source_reads.append(frame) or (
                    args["rgb"], args["semantic_image"], args["depth_image"])
                shared = dict(frame_id=10, timestamp=1., camera_matrix=np.eye(4), camera_inverse=np.eye(4), radar_points={})
                source = {**shared, "world": obj._quality_scenes[10]}
                aligned = {**shared, "stationary_tracker": obj.aligned_actor_tracker}
                with patch.object(P.PassiveSplitCollector, "_ground_truth", return_value=[]) as leaf, \
                     patch.object(A, "write_object_ground_truth") as objects, \
                     patch.object(G, "write_avo_bundle", return_value={}) as avo:
                    for keywords in ((aligned, source) if aligned_first else (source, aligned)):
                        obj._ground_truth(**keywords)
                    self.assertEqual(leaf.call_count, 2)
                    self.assertEqual(objects.call_count, 1)
                    self.assertEqual(avo.call_count, 1)
                    self.assertEqual(source_reads, [10])
                    self.assertEqual(len(obj._avo_results), 1)
                    self.assertNotIn(10, obj._quality_scenes)

    def test_sensor_headers_all_equal_but_one_tick_ahead_of_policy_rejected(self):
        from .gt_source_sync import SourceSynchronizationError
        args = arguments()
        args["policy_timestamp_s"] = .95
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(SourceSynchronizationError, "SOURCE_TIMESTAMP_MISMATCH"):
            G.write_avo_bundle(Path(tmp), **args)

    def test_source_snapshot_timestamp_is_not_silently_relabelled(self):
        args = arguments()
        args["frozen_world"].frozen_timestamp_s = .95
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(Q.LiveGtContractError, "frozen_world"):
            G.write_avo_bundle(Path(tmp), **args)

    def test_timestamped_scene_source_preserves_actor_snapshot(self):
        class World:
            def __init__(self, actors, frozen_frame_id):
                self._actors, self.frozen_frame_id = actors, frozen_frame_id
        actors = object()
        class Source:
            def capture(self, snapshot):
                return World(actors, snapshot.frame)
        source = G.build_timestamped_scene_source(Source, World)()
        value = source.capture(S(frame=22, timestamp=S(elapsed_seconds=5.25)))
        self.assertIs(value._actors, actors)
        self.assertEqual(value.frozen_timestamp_s, 5.25)
        self.assertEqual(value.frozen_frame_id, 22)
        self.assertGreater(value.frozen_observed_raw_ns, 0)

    def test_rejection_metadata_precedes_guard_and_raw_retention_is_bounded(self):
        from .gt_source_sync import LiveSyncEvidence, SourceSynchronizationError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = LiveSyncEvidence(root / "diagnostics", raw_bundle_limit=1)
            for frame in (10, 11):
                args = arguments()
                args["identity"]["frame_id"] = frame
                args["policy_timestamp_s"] = .95
                with self.assertRaises(SourceSynchronizationError):
                    G.write_avo_bundle(root / "gt", **args, sync_evidence=recorder)
                observed = json.loads((root / "diagnostics" / f"frame_{frame:010d}.observed.json").read_text())
                self.assertEqual(observed["status"], "OBSERVED_BEFORE_VALIDATION")
                self.assertEqual(observed["sources"]["rgb"]["timestamp_hex"], float(1.).hex())
                self.assertEqual(observed["sources"]["policy_radar"]["timestamp_repr"], repr(.95))
            self.assertEqual(recorder.summary()["rejected"], 2)
            self.assertEqual(recorder.summary()["raw_bundles"], 1)
            self.assertEqual(len(list((root / "diagnostics").glob("*.bgra.npy"))), 3)
            self.assertFalse((root / "gt").exists())

    def test_sensor_metadata_captured_at_callback_not_later_mutation(self):
        class Base:
            def __init__(self, **kwargs): pass
            def _on_rgb(self, image): pass
            def _on_semantic(self, image): pass
            def _on_radar(self, image): pass
        obj = G.build_live_gt_collector(Base)()
        imgs = [image("rgb"), image("semantic"), image(), image()]
        for hook, value in zip((obj._on_rgb, obj._on_semantic, obj._on_depth, obj._on_radar), imgs):
            hook(value)
        imgs[2].timestamp = 1.05
        rgb, sem, dep = obj._avo_sources(10)
        self.assertEqual(rgb["depth_callback"]["timestamp"], 1.)
        self.assertEqual(dep.timestamp, 1.05)  # original strict check still rejects mutation

    def test_no_person_geometry_still_has_defined_empty_avo(self):
        args = arguments()
        args["frozen_world"].get_actors = lambda: Actors()
        with tempfile.TemporaryDirectory() as tmp:
            result = G.write_avo_bundle(Path(tmp), **args)
            self.assertEqual(result["eligible"], 0)


class AllTickBarrierTests(unittest.TestCase):
    def collector(self, **kwargs):
        class Base:
            def __init__(self, **kw):
                self.camera = S(id=7)
                self.queued = []
            def _on_rgb(self, value): pass
            def _on_semantic(self, value): pass
            def _on_radar(self, value): pass
            def on_world_tick(self, frame_id, route_tick):
                self.queued.append((frame_id, route_tick))
        return G.build_live_gt_collector(Base)(**kwargs)

    def snapshot(self, frame=10, timestamp=1., camera_x=0.):
        return S(frame=frame, timestamp=S(elapsed_seconds=timestamp),
                 find=lambda ident: S(get_transform=lambda: Transform(camera_x)) if ident == 7 else None)

    def callbacks(self, obj, frame=10, timestamp=1.):
        for hook, kind in ((obj._on_rgb, "rgb"), (obj._on_semantic, "semantic"),
                           (obj._on_depth, "depth"), (obj._on_radar, "radar")):
            hook(image(kind, frame, timestamp))

    def test_all_ticks_checked_before_original_queue_and_sources_retained(self):
        obj = self.collector()
        for tick in (1, 2, 3, 4):
            self.callbacks(obj, frame=9+tick, timestamp=float(tick))
            obj.world = S(get_snapshot=lambda tick=tick: self.snapshot(9+tick, float(tick)))
            obj.on_world_tick(9+tick, tick)
        self.assertEqual(len(obj._avo_tick_barrier_records), 4)
        self.assertEqual(obj.queued, [(10, 1), (11, 2), (12, 3), (13, 4)])
        self.assertTrue(all(r["outside_action_open"] for r in obj._avo_tick_barrier_records))
        self.assertEqual(len(obj._avo_sources(10)), 3)

    def test_equal_camera_headers_wrong_world_refused_before_queue(self):
        obj = self.collector()
        self.callbacks(obj, timestamp=1.05)
        obj.world = S(get_snapshot=lambda: self.snapshot())
        with self.assertRaisesRegex(Q.LiveGtContractError, "SOURCE_TIMESTAMP_MISMATCH"):
            obj.on_world_tick(10, 2)
        self.assertEqual(obj.queued, [])
        self.assertEqual(obj._avo_tick_barrier_records[0]["status"], "SOURCE_TIMESTAMP_MISMATCH")

    def test_owned_tick_pose_checked_even_when_camera_triple_agrees(self):
        obj = self.collector()
        self.callbacks(obj)
        with self.assertRaisesRegex(Q.LiveGtContractError, "SOURCE_OWNED_CAMERA_POSE_MISMATCH"):
            obj.wait_for_sensor_frame(10, snapshot=self.snapshot(camera_x=.1))

    def test_missing_callback_bounded_and_durable_before_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self.collector(attempt_dir=tmp)
            obj._on_rgb(image("rgb"))
            with self.assertRaisesRegex(Q.LiveGtContractError, "SOURCE_METADATA_MISSING"):
                obj.wait_for_sensor_frame(10, snapshot=self.snapshot(), timeout_s=.001)
            diagnostics = Path(tmp) / "run5_live_barrier_rejections"
            self.assertEqual(len(list(diagnostics.glob("*.observed.json"))), 1)
            self.assertEqual(len(list(diagnostics.glob("*.rejected.json"))), 1)
            self.assertEqual(len(list(diagnostics.glob("*.bgra.npy"))), 1)

    def test_received_header_does_not_count_as_completed_parent_callback(self):
        obj = self.collector()
        self.callbacks(obj)
        obj._avo_callbacks_complete[10].remove("rgb")
        with self.assertRaisesRegex(Q.LiveGtContractError, "SOURCE_CALLBACK_NOT_COMPLETED"):
            obj.wait_for_sensor_frame(10, snapshot=self.snapshot(), timeout_s=.001)

    def test_callback_and_evidence_bounds_fail_closed(self):
        obj = self.collector()
        for frame in range(30):
            self.callbacks(obj, frame=frame)
        self.assertEqual(len(obj._avo_callbacks_complete), 24)
        obj._avo_tick_barrier_limit = 0
        with self.assertRaisesRegex(Q.LiveGtContractError, "evidence bound exceeded"):
            obj.wait_for_sensor_frame(29, snapshot=self.snapshot(frame=29))

    def test_real_run4_quality_mro_barrier_and_leaf_called_once_every_tick(self):
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as A
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        cls = U.build_run4_collector_class(G.build_live_gt_collector(A.QualityPassiveSplitCollector))
        obj = cls.__new__(cls)
        template = self.collector()
        self.callbacks(template, 10, 1.)
        self.callbacks(template, 11, 1.05)
        obj.__dict__.update(template.__dict__)
        obj._run4_carla_trace = {"ticks": []}
        obj.live, obj.failures = S(infrastructure_fault=None), []
        obj._quality_scene_lock, obj._quality_scenes = threading.Lock(), {}
        obj.qualification_capture_limit, obj.sent = None, 0
        events = []
        obj.scene_source = S(capture=lambda snapshot: events.append("freeze") or
            A.QualityFrozenWorld([], frozen_frame_id=snapshot.frame))
        with patch.object(P.PassiveSplitCollector, "on_world_tick", side_effect=lambda *a: events.append("leaf")) as leaf:
            for frame, tick, stamp in ((10, 1, 1.), (11, 2, 1.05)):
                obj.world = S(get_snapshot=lambda: self.snapshot(frame, stamp))
                obj.on_world_tick(frame, tick)
            self.assertEqual(leaf.call_count, 2)
        self.assertEqual(events, ["freeze", "leaf", "leaf"])
        self.assertEqual(len(obj._avo_tick_barrier_records), 2)
        self.assertEqual(len(obj._run4_carla_trace["ticks"]), 2)


if __name__ == "__main__":
    unittest.main()

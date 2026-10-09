"""Real preparation/aggregation/wire/target/evaluator seams, CPU only."""
from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np

from . import pose as P
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge
gt_bridge.bind_legacy_namespace()
from rl_agent import ue_route_b_split_cell_adapter_v1 as AD
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import Pipeline, processor, _fresh_radio, _scene, MS
from rl_agent.splitfusion_live_parity_repair_v1 import runtime, target_transport as T, evaluation as V
from rl_agent.splitfusion_live_parity_repair_v1.test_target_transport import fixture
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A


class Transform:
    def __init__(self, matrix, pose=None):
        self.matrix = np.array(matrix, dtype=np.float64, copy=True)
        self.location = NS(**dict(zip("xyz", self.matrix[:3, 3])))
        self.rotation = NS(**dict(zip(("pitch", "yaw", "roll"), (pose or (0.,) * 6)[3:])))
    def get_matrix(self):
        return self.matrix.tolist()
    def get_inverse_matrix(self):
        return np.linalg.inv(self.matrix).tolist()


class Actor:
    def __init__(self, ident, matrix, pose=None):
        self.id = ident
        self.transform = Transform(matrix, pose)
    def get_transform(self):
        return self.transform
    def get_velocity(self):
        return NS(x=0., y=0., z=0.)


def sensors(frame=1001, stamp=1., ego=(1., 2., .1, 1., 90., 2.)):
    ego_matrix = P.carla_transform_matrix(ego)
    camera = ego_matrix @ P.carla_transform_matrix(AD.CAMERA_MOUNT)
    radar = ego_matrix @ P.carla_transform_matrix(AD.RADAR_MOUNT)
    actors = {7: Actor(7, camera), 8: Actor(8, radar), 9: Actor(9, ego_matrix, ego)}
    snapshot = NS(frame=frame, timestamp=NS(elapsed_seconds=stamp), find=actors.get)
    def image(matrix):
        return NS(frame=frame, timestamp=stamp, width=2, height=2,
                  transform=Transform(matrix), raw_data=bytes(16))
    rgb, semantic, depth = [image(camera) for _ in range(3)]
    measurement = NS(frame=frame, timestamp=stamp, transform=Transform(radar),
                     raw_data=np.asarray([[1., .0, .0, 20.]], dtype=np.float32).tobytes())
    return snapshot, actors, (rgb, semantic, depth, measurement)


class PreparationBase(AD.PassiveSplitCollector):
    """Keep the real _records_for and _process_token, fake only external I/O."""
    def __init__(self, **kwargs):
        from data_collection.radar_sweep_aggregator_v1 import RadarSweepAggregator
        from pole_lraspp_multimodal_fusion import radar_fusion
        from rl_agent.splitfusion_live_parity_repair_v1 import perception
        self.pipeline = Pipeline()
        self.stream_id = self.pipeline.pipe.stream_id
        snapshot, actors, self.sensor_values = sensors()
        self.snapshot = snapshot
        self.camera, self.radar, self.ego = (Actor(i, a.transform.matrix,
            (1., 2., .1, 1., 90., 2.) if i == 9 else None) for i, a in actors.items())
        self.world = NS(get_snapshot=lambda: snapshot)
        self.sensor_condition = threading.Condition()
        self.images, self.radars, self.semantic_images = OrderedDict(), OrderedDict(), OrderedDict()
        self.stop_event = threading.Event()
        self.aggregator = RadarSweepAggregator()
        self.aggregator_error = None
        self.qualification_capture_limit = None
        self.sent, self.dropped = 0, 0
        self.sent_frames = set()
        self.ack_timeout_s, self.service_deadline_s = 10., .170
        self.qualification_transmit_order = ()
        self.cell = {"action_id": 71}
        self.profile_activated = True
        self.scene_source = None
        self.model_size = (768, 432)
        self.intrinsics = perception.radar_projection_intrinsics()
        self.tracker = radar_fusion.StationaryTrackAccumulator()
        self.raster_kwargs, self.raster_result = None, None
        def build(**kw):
            self.raster_kwargs = dict(kw)
            self.raster_result = radar_fusion.build_radar_sample(**kw)
            return self.raster_result
        self.parked = NS(build_radar_sample=build)
        self.feedback = NS(register_capture=mock.Mock())
        self.segmentation_queue, self.evaluation_queue = queue.Queue(), queue.Queue()
        self.segmentation_evidence_errors, self.evaluation_errors = {}, {}
        self.transport_counters = NS(bump=mock.Mock())
        self.qualification_interframe_drain_s = 0.
        self.rows = []
        self.live = NS(submit=self.submit)
        self.prepared = None

    def _append_row(self, row):
        self.rows.append(row)

    def submit(self, **kwargs):
        self.submit_kwargs = kwargs
        pipe = self.pipeline
        pipe.h.host.advance(100 * MS)
        _fresh_radio(pipe.h)
        planned = pipe.pipe.plan(frame_id=kwargs["frame_id"], capture_wall_ns=kwargs["capture_timestamp_ns"],
            scene=_scene(pipe.h), carla_timestamp=kwargs["carla_timestamp"])
        self.prepared = pipe.pipe.materialize(planned, frame_id=kwargs["frame_id"],
            capture_wall_ns=kwargs["capture_timestamp_ns"], ego_pose=kwargs["ego_pose"],
            carla_timestamp=kwargs["carla_timestamp"], input_7ch=lambda: object())
        kwargs["on_commit"]()
        return {"sent": True}

    def feed(self):
        rgb, sem, depth, radar = self.sensor_values
        self._on_rgb(rgb)
        self._on_semantic(sem)
        self._on_depth(depth)
        for offset in (3, 2, 1, 0):
            matrix = np.asarray(radar.transform.get_matrix())
            matrix[0, 3] -= offset * .25
            self._on_radar(NS(frame=radar.frame-offset, timestamp=radar.timestamp-offset*.05,
                transform=Transform(matrix), raw_data=radar.raw_data))
        self.wait_for_sensor_frame(rgb.frame, snapshot=self.snapshot)

    def advance_live_actors(self):
        for actor in (self.camera, self.radar, self.ego):
            actor.transform.matrix[0, 3] += .3394622802734375
            actor.transform.location.x += .3394622802734375

    def process(self):
        return self._process_token({"frame_id": 1001, "scheduled_perf": time.perf_counter()})


def write_target(root, ticket, *, delta=0.):
    _, inputs, prepared = fixture()
    frame_id = ticket.context.frame_id
    camera = P.context_camera(ticket.context)
    camera[0, 3] += delta
    sample = f"live_test_frame{frame_id}"
    frame = dict(inputs.frame, frame_id=frame_id, sample_id=sample,
                 camera_inverse_matrix_json=json.dumps(np.linalg.inv(camera).tolist()))
    provenance = dict(inputs.provenance, capture_identity=dict(ticket.gt_identity), sample_id=sample)
    inputs = replace(inputs, frame=frame, provenance=provenance)
    inputs = replace(inputs, provenance=dict(provenance, input_content_sha256=A.input_content_sha256(inputs)))
    target = {**prepared.target, "provenance": inputs.provenance,
              "tables": {**prepared.target["tables"], "target_manifest.csv": [{"sample_id": sample}]}}
    prepared = V.restore_prepared_target(frame_id=frame_id, sample_id=sample,
        target=target, content_sha256=V._target_digest(frame_id, sample, target))
    descriptor = T.write_bundle(root, identity=ticket.gt_identity, inputs=inputs, prepared=prepared)
    return descriptor


class PoseTests(unittest.TestCase):
    def test_real_preparation_uses_owned_camera_radar_ego_after_live_advance(self):
        from pole_lraspp_multimodal_fusion import radar_fusion
        with tempfile.TemporaryDirectory() as root:
            with P.install_ue_pose(attempt_dir=Path(root), max_frames=2):
                host = LG.build_live_gt_collector(PreparationBase)()
                host.feed()
                source = host._pose_sources[1001]
                host.advance_live_actors()
                host.process()
                self.assertEqual(host.sent, 1)
                np.testing.assert_array_equal(host.raster_kwargs["sensor_matrix"], source.matrix("radar"))
                np.testing.assert_array_equal(host.raster_kwargs["camera_inverse_matrix"], np.linalg.inv(source.matrix("camera")))
                # The real aggregator's same window and reference transform are
                # an independent numerical oracle for motion compensation/P40.
                detections, meta = host.aggregator.window_detections(0,
                    sensor_inverse_matrix=np.linalg.inv(source.matrix("radar")), reference_timestamp_s=1.)
                np.testing.assert_array_equal(host.raster_kwargs["detections"], detections)
                self.assertEqual(meta["callback_frames"], [998, 999, 1000, 1001])
                np.testing.assert_array_equal(meta["raw_provenance"]["original_range_m"], [20.] * 4)
                expected = radar_fusion.build_radar_sample(**{**host.raster_kwargs,
                    "detections": detections, "tracker": radar_fusion.StationaryTrackAccumulator()})
                np.testing.assert_array_equal(host.raster_result[0], expected[0])
                _, context = W.unpack_sfd4(host.prepared.wire)
                self.assertEqual(context.ego_world.values(), tuple(source.document["ego_world"]))
                self.assertNotEqual(host.submit_kwargs["ego_pose"], context.ego_world.values())
                ticket = host.evaluation_queue.get_nowait()
                np.testing.assert_array_equal(ticket["camera_matrix"], source.matrix("camera"))
                # Unrelated callers still see current actor state.
                np.testing.assert_array_equal(AD.actor_world_matrix(host.camera), host.camera.transform.matrix)
            archive = P.read_archive(Path(root)/"pose_archive")
            self.assertEqual(archive["manifest"]["counts"], {"OWNED_SOURCE": 1, "UE_WIRE": 1, "UE_PREPARATION_DONE": 1})

    def test_scope_nests_with_complete_gt_factory_and_restores(self):
        original = LG.build_live_gt_collector
        with tempfile.TemporaryDirectory() as root:
            with P.install_ue_pose(attempt_dir=Path(root), max_frames=1):
                with gt_bridge.install_complete_gt(attempt_dir=Path(root), max_frames=1):
                    cls = LG.build_live_gt_collector(PreparationBase)
                    self.assertEqual([c.__name__ for c in cls.__mro__[:3]],
                                     ["CompleteCollector", "OwnedPoseCollector", "LiveAvoCollector"])
                    host = cls()
                    host.feed()
                    host.advance_live_actors()
                    host.process()
            self.assertIs(LG.build_live_gt_collector, original)

    def test_records_mutation_and_missing_owned_pose_fail_before_submit(self):
        for mutation in ("timestamp", "pose", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                with P.install_ue_pose(attempt_dir=Path(root), max_frames=1):
                    host = LG.build_live_gt_collector(PreparationBase)()
                    host.feed()
                    if mutation == "timestamp":
                        host.images[1001][0].timestamp += .05
                    elif mutation == "pose":
                        host.images[1001][0].transform.matrix[0, 3] += .3394622802734375
                    else:
                        host._pose_sources.clear()
                    with self.assertRaises(P.PoseParityError):
                        host.process()
                    self.assertIsNone(host.prepared)
                    self.assertIsNone(getattr(P._LOCAL, "binding", None))
                    with self.assertRaisesRegex(P.PoseParityError, "preparation failed"):
                        host.on_world_tick(1002, 2)
                self.assertFalse(P.read_archive(Path(root)/"pose_archive")["manifest"]["complete"])

    def test_snapshot_ego_and_radar_mismatch_not_inferred_from_rgb(self):
        for role in ("ego", "radar"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as root:
                with P.install_ue_pose(attempt_dir=Path(root), max_frames=1):
                    host = LG.build_live_gt_collector(PreparationBase)()
                    ident = getattr(host, role).id
                    actor = host.snapshot.find(ident)
                    actor.transform.matrix[0, 3] += .01
                    actor.transform.location.x += .01
                    with self.assertRaises(P.PoseParityError):
                        host.feed()
                self.assertFalse(P.read_archive(Path(root)/"pose_archive")["manifest"]["complete"])

    def test_thread_local_cleanup_and_materialize_without_source_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with P.install_ue_pose(attempt_dir=Path(root), max_frames=1):
                host = LG.build_live_gt_collector(PreparationBase)()
                host.feed()
                source = host._pose_sources[1001]
                host.advance_live_actors()
                found = []
                with P._bind_preparation(host, source, mock.Mock()):
                    thread = threading.Thread(target=lambda: found.append(AD.actor_world_matrix(host.camera)))
                    thread.start(); thread.join()
                    np.testing.assert_array_equal(AD.actor_world_matrix(host.camera), source.matrix("camera"))
                    with self.assertRaisesRegex(P.PoseParityError, "nested"):
                        with P._bind_preparation(host, source, mock.Mock()):
                            pass
                np.testing.assert_array_equal(found[0], host.camera.transform.matrix)
                with self.assertRaisesRegex(P.PoseParityError, "outside"):
                    Pipeline().step()

    def test_invalid_rigid_matrix_and_owned_copy(self):
        for value in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)):
            with self.assertRaises(P.PoseParityError):
                P._matrix(value, "test")
        value = np.eye(4)
        result = P._matrix(value, "test")
        value[0, 3] = .5
        self.assertEqual(result[0, 3], 0.)
        self.assertFalse(result.flags.writeable)


class EdgePoseTests(unittest.TestCase):
    def exercise(self, root, *, delta=0., tamper=False, ue_host=None):
        from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
        sent = []
        edge_path = root/"edge_pose"
        original_class = E.Run4EvaluatorV2
        with P.install_edge_pose(archive_directory=edge_path, max_frames=2) as state:
            PW.publish_ready_after_warmup(lambda: {"completed": True, "modes_warmed": list(range(12))}, lambda: None)
            prepared = ue_host.prepared if ue_host is not None else Pipeline().step()
            edge = processor()
            edge.verify(prepared.wire)  # real receive-loop admission then compute
            produced = edge.process(prepared.wire, edge_timing={})
            ticket = produced.evaluation
            target = write_target(root/"targets", ticket, delta=delta)
            if tamper:
                path = Path(target["directory"])/"inputs.json"
                path.write_bytes(path.read_bytes() + b"tampered")
            avo = {"avo": {"frame_id": ticket.context.frame_id, "qualified": [], "structural_ignored": []},
                "avo_sidecar_sha256": "b"*64, "avo_ready_wall_ns": time.time_ns()}
            with mock.patch.object(LG, "read_avo_bundle", return_value=avo):
                with runtime.install_edge_evaluation(archive_directory=root/"predictions", max_frames=2):
                    # Retained profiler may replace the module alias with a subclass.
                    class Profiled(original_class):
                        pass
                    E.Run4EvaluatorV2 = Profiled
                    try:
                        worker = Profiled(spec=W.load_run4_quality_spec(Path(__file__).resolve().parents[2]),
                            send=sent.append, read_ground_truth=lambda **kw: G.read_ground_truth(root/"targets", **kw),
                            match_distance_m=3., gt_timeout_s=.05, probe_timeout_s=.005)
                        worker.start()
                        with mock.patch.object(V, "evaluate_prepared", wraps=V.evaluate_prepared) as evaluate:
                            worker.submit(ticket)
                            result = worker.close()
                            calls = evaluate.call_count
                        self.assertFalse(result["worker_alive"])
                        self.assertTrue((edge_path/"MANIFEST.json").is_file())
                    finally:
                        E.Run4EvaluatorV2 = original_class
            self.assertTrue(state["flushed"])
        return P.read_archive(edge_path), ticket, result, calls, sent

    def test_real_wire_target_reader_evaluator_and_profiler_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, ticket, result, calls, sent = self.exercise(root)
            self.assertTrue(archive["manifest"]["complete"])
            self.assertEqual(archive["manifest"]["counts"],
                {"EDGE_READY": 1, "EDGE_WIRE": 1, "EDGE_VERIFY_REPEAT": 1, "EDGE_TARGET_RECONCILED": 1})
            self.assertEqual(calls, 1)
            self.assertEqual(len(sent), 1)
            self.assertTrue(result["pose_archive_complete"])
            event = next(row for row in archive["events"] if row["kind"] == "EDGE_TARGET_RECONCILED")
            self.assertEqual(event["context"], P.context_document(ticket.context))

    def test_target_mismatch_rejected_before_scoring_and_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, ticket, result, calls, sent = self.exercise(Path(temp), delta=.3394622802734375)
            self.assertFalse(archive["manifest"]["complete"])
            self.assertEqual(calls, 0)
            self.assertEqual(len(sent), 1)
            self.assertFalse(result["pose_archive_complete"])
            self.assertIn("edge wire/target camera pose mismatch", archive["manifest"]["errors"][0])
            feedback, reason = W.decode_feedback(sent[0])
            self.assertEqual(feedback.kind, "EVALUATOR_FAULT")
            self.assertIsNone(feedback.q_perc)
            self.assertEqual(reason, W.EvaluatorReason.EVALUATOR_EXCEPTION)

    def test_real_36_path_warmup_isolated_from_live_pose_budget(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import CONTRACT, FakeAE, FakeCodec, runtimes
        fake_runtime = NS(_counters=NS(frames_attempted=0), _context_session=object(), _detached_tail=object())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with P.install_edge_pose(archive_directory=root/"edge", max_frames=1) as state:
                ready = []
                report = PW.publish_ready_after_warmup(
                    lambda: PW.warm_edge(processor(), fake_runtime, CONTRACT, device="cpu",
                        encoders={f: FakeAE(f) for f in ("AE128", "AE64", "AE32")},
                        codec=FakeCodec(), sync=lambda: None),
                    lambda: ready.append(state["ready"]))
                self.assertEqual(ready, [False])
                self.assertEqual(len(report["paths"]), 36)
                self.assertEqual(state["warm_verified"], 72)
                self.assertTrue(state["ready"])
                # UE warmup uses continuous_ue.prepare, never FramePipeline.materialize.
                ue, _, _ = runtimes()
                with P.install_ue_pose(attempt_dir=root/"ue", max_frames=1):
                    self.assertTrue(PW.warm_ue(ue, CONTRACT, prepare_input=lambda *a: object(),
                                             sync=lambda: None)["completed"])
                frame = Pipeline().step().wire
                edge = processor()
                edge.verify(frame)
                edge.process(frame, edge_timing={})
                edge.verify(frame)  # terminal/repeated diagnostics are also idempotent
            counts = P.read_archive(root/"edge")["manifest"]["counts"]
            self.assertEqual(counts, {"EDGE_READY": 1, "EDGE_WIRE": 1, "EDGE_VERIFY_REPEAT": 2})
            self.assertEqual(P.read_archive(root/"ue"/"pose_archive")["manifest"]["frames"], 0)

    def test_failed_ready_and_conflicting_live_context_fail_closed(self):
        for mode in ("before_ready", "failed_ready", "conflicting_context", "budget"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                with P.install_edge_pose(archive_directory=Path(temp)/"edge", max_frames=1) as state:
                    edge = processor()
                    prepared = Pipeline().step()
                    if mode == "failed_ready":
                        with self.assertRaisesRegex(RuntimeError, "write_ready"):
                            PW.publish_ready_after_warmup(lambda: {"completed": True, "modes_warmed": list(range(12))},
                                mock.Mock(side_effect=RuntimeError("write_ready")))
                    elif mode != "before_ready":
                        PW.publish_ready_after_warmup(lambda: {"completed": True, "modes_warmed": list(range(12))}, lambda: None)
                        edge.verify(prepared.wire)
                    if mode == "conflicting_context":
                        envelope, context = W.unpack_sfd4(prepared.wire)
                        context = replace(context, ego_world=replace(context.ego_world, x=context.ego_world.x+.1))
                        prepared = replace(prepared, wire=W.pack_sfd4(envelope, context))
                    if mode == "budget":
                        pipe = Pipeline(); pipe.frame += 1
                        prepared = pipe.step()
                    with self.assertRaises(P.PoseParityError):
                        edge.verify(prepared.wire)

    def test_tampered_target_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive, ticket, result, calls, sent = self.exercise(Path(temp), tamper=True)
            self.assertEqual(calls, 0)
            self.assertEqual(archive["manifest"]["counts"].get("EDGE_TARGET_RECONCILED", 0), 0)

    def test_archive_cross_join_and_pose_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with P.install_ue_pose(attempt_dir=root/"ue", max_frames=2):
                host = LG.build_live_gt_collector(PreparationBase)()
                host.feed()
                host.advance_live_actors()
                host.process()
                archive, ticket, result, calls, sent = self.exercise(root, ue_host=host)
            expected = [{"frame_id": ticket.context.frame_id, "run4_identity": host.prepared.identity}]
            summary = P.verify_archives(root/"ue"/"pose_archive", root/"edge_pose",
                expected_transmitted=expected, expected_evaluated=expected)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["transmitted_frames"], 1)
            with self.assertRaises(P.PoseParityError):
                P.verify_archives(root/"ue"/"pose_archive", root/"edge_pose",
                    expected_transmitted=[], expected_evaluated=expected)
            evidence = root/"edge_pose"/"pose_evidence.jsonl"
            evidence.write_bytes(evidence.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(P.PoseParityError, "hash"):
                P.read_archive(root/"edge_pose")

    def test_create_only_archives_and_exception_restoration(self):
        old = (T.read_bundle, runtime.install_edge_evaluation, E.Run4EdgeProcessorV2.verify)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with P.install_edge_pose(archive_directory=root/"edge", max_frames=1):
                    raise RuntimeError("injected")
            self.assertFalse(P.read_archive(root/"edge")["manifest"]["complete"])
            self.assertEqual(old, (T.read_bundle, runtime.install_edge_evaluation, E.Run4EdgeProcessorV2.verify))
            with self.assertRaises(FileExistsError):
                with P.install_edge_pose(archive_directory=root/"edge", max_frames=1):
                    pass


if __name__ == "__main__":
    unittest.main()

"""CPU-only parity tests for the post-route GT materializer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import b_route_bridge_v3 as bridge
from . import branch_evidence_v1 as predictions
from . import operational_ack_v1 as operational
from . import postrun_artifact_v1 as artifacts
from . import postrun_evaluator_v1 as evaluator
from . import postrun_gt_materializer_v1 as materializer
from .test_postrun_evaluator_v1 import reward_spec


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _identity(frame: int) -> operational.FrameActionIdentityV1:
    return operational.FrameActionIdentityV1(
        run_id="run4b5b-materializer-test", cell_id="route-b",
        stream_id="ego",
        session_uuid="00000000-0000-4000-8000-000000000001",
        controller_lineage_sha256=_sha("lineage"), decision_seq=frame,
        ticket_seq=frame, frame_id=frame, tensor_seq=frame,
        capture_timestamp_ns=1_790_000_000_000_000_000 + frame,
        mode_id=11, q_e4=3000, keep_count=6800, anchor_action_id=67,
        profile_id="split_ae32_uint4_q3000",
        execution_bundle_sha256=_sha("bundle"),
    )


def _actor(actor_id: int, kind: str, x: float, y: float,
           extent: tuple[float, float, float]) -> bridge.PrimitiveActorV3:
    return bridge.PrimitiveActorV3(
        actor_id=actor_id, type_id=kind,
        bbox_location={"x": 0.0, "y": 0.0, "z": extent[2]},
        bbox_extent={"x": extent[0], "y": extent[1], "z": extent[2]},
        bbox_rotation={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        transform={"location": {"x": x, "y": y, "z": 0.0},
                   "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}},
        velocity={"x": 0.0, "y": 0.0, "z": 0.0},
    )


class _Origin:
    x = y = z = 0.0


class _Semantic:
    def __init__(self, frame: int) -> None:
        self.frame, self.timestamp, self.width, self.height = frame, 2.5, 8, 6
        value = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        value[1:4, 1:5, 2] = 7
        value[3:6, 6:8, 2] = 15
        self.raw_data = value.tobytes(order="C")


def _complete_spool(root: Path, frame: int = 17):
    spool = bridge.RawGroundTruthSpoolV3(root)
    identity = _identity(frame)
    scene = bridge.PrimitiveSceneV3((
        _actor(100, "vehicle.fixture", 10.0, 0.0, (2.0, 1.0, 0.75)),
        _actor(200, "walker.pedestrian.fixture", 12.0, 1.5,
               (0.3, 0.3, 0.9)),
    ))
    semantic = _Semantic(frame)
    spool.write_identity(identity)
    spool.write_scene(
        frame_id=frame, timestamp=2.5, scene=scene,
        camera_matrix=np.eye(4, dtype=np.float64),
        camera_inverse=np.eye(4, dtype=np.float64),
        camera_location=_Origin(),
        radar_world_xyz=np.asarray(
            [[10.0, 0.0, 0.75], [12.0, 1.5, 0.9]], dtype=np.float32),
    )
    spool.write_semantic(frame_id=frame, image=semantic)
    spool.seal()
    return identity, scene, semantic


def _direct(scene: bridge.PrimitiveSceneV3, semantic: _Semantic):
    carla, parked, valid_objects, route = (
        materializer._load_authoritative_modules())  # noqa: SLF001
    raw_scene = {"actors": [actor.to_dict() for actor in scene.actors]}
    rows = parked.build_object_rows(
        world=materializer._world(raw_scene, carla, route),  # noqa: SLF001
        ego_vehicle=materializer._Ego(),  # noqa: SLF001
        sample_base={"timestamp": 2.5, "frame_id": 17},
        camera_location=carla.Location(0.0, 0.0, 0.0),
        camera_matrix=np.eye(4, dtype=np.float64),
        camera_inverse_matrix=np.eye(4, dtype=np.float64),
        intrinsics=route.camera_intrinsics(768, 448, 120.0),
        width=768, height=448, max_distance_m=140.0,
        radar_world_xyz=np.asarray(
            [[10.0, 0.0, 0.75], [12.0, 1.5, 0.9]], dtype=np.float32),
        stationary_tracker=parked.ActorStationaryTracker(0.35, 5.0),
        include_pedestrians=True, radar_support_margin_m=1.0,
        radar_person_support_mode="radius", radar_person_support_radius_m=1.5,
        radar_person_support_z_down_m=0.5,
        radar_person_support_z_up_m=2.0,
    )
    objects = valid_objects(rows, image_width=768, image_height=448,
                            min_area_px=12.0, max_distance_m=40.0)
    return objects, route.semantic_gt_3class(semantic)


def _outcome(identity: operational.FrameActionIdentityV1, success: bool):
    opened = 10_000_000_000
    if success:
        return operational.OperationalOutcomeV1(
            identity=identity,
            terminal=operational.OperationalTerminal.SUCCESS,
            action_open_monotonic_raw_ns=opened,
            resolution_monotonic_raw_ns=opened + 100_000_000,
            observed_latency_ns=100_000_000, state_latency_ns=100_000_000,
            accepted_ack_sha256=_sha("ack"),
        )
    return operational.OperationalOutcomeV1(
        identity=identity, terminal=operational.OperationalTerminal.TIMEOUT,
        action_open_monotonic_raw_ns=opened,
        resolution_monotonic_raw_ns=opened + operational.FIRST_LATE_TICK_NS,
        observed_latency_ns=None, state_latency_ns=0,
        accepted_ack_sha256=None,
    )


class PostRouteMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_object_semantic_and_qperc_parity(self) -> None:
        identity, scene, semantic = _complete_spool(self.root / "raw")
        result = materializer.PostRouteGroundTruthMaterializerV1().materialize(
            spool_root=self.root / "raw", evidence_root=self.root / "gt")
        self.assertEqual((result.record_count, result.missing_count), (1, 0))
        record = artifacts.GroundTruthEvidenceStoreV1.open_existing(
            self.root / "gt").verify_all()[0]
        actual = artifacts.decode_bundle(
            (self.root / "gt" / record.artifact_relative_path).read_bytes(),
            expected_kind=artifacts.GROUND_TRUTH_KIND,
            expected_identity=identity,
        )
        expected_objects, expected_mask = _direct(scene, semantic)
        self.assertEqual(actual.objects, tuple(expected_objects))
        np.testing.assert_array_equal(actual.semantic_mask, expected_mask)

        direct_gt = artifacts.GroundTruthEvidenceStoreV1.create(
            self.root / "direct_gt")
        direct_gt.write(identity=identity, eligible_objects=expected_objects,
                        semantic_mask=expected_mask,
                        recorded_monotonic_raw_ns=1)
        prediction_store = predictions.PredictionEvidenceStoreV1.create(
            self.root / "predictions")
        payload = artifacts.encode_bundle(
            kind=artifacts.PREDICTION_KIND, identity=identity,
            objects=expected_objects, semantic_mask=expected_mask)
        publication = predictions.TailOutputBranchCoordinatorV1().publish(
            identity, payload)
        prediction_store.write(publication, payload,
                               recorded_monotonic_raw_ns=2)
        common = dict(
            prediction_root=self.root / "predictions",
            outcomes=[_outcome(identity, True)],
            payload_bytes_by_identity={identity.exact_sha256(): 100_000},
        )
        engine = evaluator.PostRunEvaluatorV1(reward_spec=reward_spec())
        observed = engine.evaluate(ground_truth_root=self.root / "gt",
                                   output_root=self.root / "eval_observed",
                                   **common)
        direct = engine.evaluate(ground_truth_root=self.root / "direct_gt",
                                 output_root=self.root / "eval_direct", **common)
        self.assertEqual(observed.frame_metrics_path.read_bytes(),
                         direct.frame_metrics_path.read_bytes())
        self.assertEqual(observed.object_metrics_path.read_bytes(),
                         direct.object_metrics_path.read_bytes())
        self.assertEqual(observed.summary_path.read_bytes(),
                         direct.summary_path.read_bytes())
        row = next(csv.DictReader(
            observed.frame_metrics_path.open(newline="", encoding="utf-8")))
        self.assertEqual((row["quality_defined"], float(row["q_perc"])),
                         ("1", 1.0))
        report = json.loads(result.report_path.read_text(encoding="ascii"))
        self.assertEqual(report["status"], "COMPLETE_WITH_ALL_GT")
        self.assertFalse(report["live_qperc_computed"])

    def test_missing_gt_is_reported_and_timeout_row_survives(self) -> None:
        identity = _identity(18)
        spool = bridge.RawGroundTruthSpoolV3(self.root / "raw")
        spool.write_identity(identity)
        spool.seal()
        result = materializer.PostRouteGroundTruthMaterializerV1().materialize(
            spool_root=self.root / "raw", evidence_root=self.root / "gt")
        self.assertEqual((result.identity_frame_count, result.record_count,
                          result.missing_count), (1, 0, 1))
        report = json.loads(result.report_path.read_text(encoding="ascii"))
        reasons = report["missing_identity_frames"][0]["reasons"]
        self.assertEqual(reasons, ["SCENE_NOT_CAPTURED", "SEMANTIC_NOT_CAPTURED"])
        predictions.PredictionEvidenceStoreV1.create(self.root / "predictions")
        evaluated = evaluator.PostRunEvaluatorV1(
            reward_spec=reward_spec()).evaluate(
                prediction_root=self.root / "predictions",
                ground_truth_root=self.root / "gt",
                outcomes=[_outcome(identity, False)],
                payload_bytes_by_identity={identity.exact_sha256(): 50_000},
                output_root=self.root / "evaluation")
        rows = list(csv.DictReader(
            evaluated.frame_metrics_path.open(newline="", encoding="utf-8")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["operational_terminal"], "TIMEOUT")
        self.assertEqual(rows[0]["ground_truth_present"], "0")
        self.assertEqual(rows[0]["evaluation_reward"], "-1.0")

    def test_tamper_and_create_only_refuse(self) -> None:
        _complete_spool(self.root / "raw", 19)
        path = self.root / "raw/semantic/0000000019.bgra"
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(materializer.RawSpoolIntegrityError,
                                    "byte count|digest"):
            materializer.PostRouteGroundTruthMaterializerV1().materialize(
                spool_root=self.root / "raw", evidence_root=self.root / "gt")
        self.assertFalse((self.root / "gt").exists())

        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _complete_spool(self.root / "raw", 20)
        worker = materializer.PostRouteGroundTruthMaterializerV1()
        worker.materialize(spool_root=self.root / "raw",
                           evidence_root=self.root / "gt")
        with self.assertRaises(materializer.MaterializationCreateOnlyError):
            worker.materialize(spool_root=self.root / "raw",
                               evidence_root=self.root / "gt")

    def test_live_ack_modules_do_not_import_materializer(self) -> None:
        package = Path(__file__).resolve().parent
        for name in ("operational_ack_v1.py", "live_adapters_v1.py",
                     "b_opportunity_processor_v1.py", "b_edge_service_v2.py"):
            self.assertNotIn("postrun_gt_materializer_v1",
                             (package / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

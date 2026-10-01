"""Portable CPU tests for b_route_bridge_v4.

This version avoids importing the full pinned Route-B module, whose import
requires ignored calibration artifacts not present in an isolated worktree.
The production seam restoration is instead checked structurally; the route
module's own integration suite covers its concrete import environment.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import b_route_bridge_v3 as V3
from . import b_route_bridge_v4 as B
from . import b_ue_process_v1 as U
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


def sha(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()


def identity(frame: int) -> A.FrameActionIdentityV1:
    return A.FrameActionIdentityV1(
        run_id="run", cell_id="cell", stream_id="ego",
        session_uuid="00000000-0000-4000-8000-000000000001",
        controller_lineage_sha256=sha("lineage"), decision_seq=frame,
        ticket_seq=frame, frame_id=frame, tensor_seq=frame,
        capture_timestamp_ns=frame + 1, mode_id=11, q_e4=3000,
        keep_count=6800, anchor_action_id=67,
        profile_id="split_ae32_uint4_q3000",
        execution_bundle_sha256=sha("bundle"))


class _Vec: x = 1.0; y = 2.0; z = 3.0
class _Rot: pitch = 4.0; yaw = 5.0; roll = 6.0
class _Transform: location = _Vec(); rotation = _Rot()
class _Bbox: location = _Vec(); extent = _Vec(); rotation = _Rot()
class _Actor: id = 7; type_id = "vehicle.test"; bounding_box = _Bbox()
class _SnapshotActor:
    def get_transform(self): return _Transform()
    def get_velocity(self): return _Vec()
class _Snapshot:
    def find(self, actor_id): return _SnapshotActor() if actor_id == 7 else None
class _World:
    def get_actors(self): return (_Actor(),)
class _Semantic:
    frame = 3; timestamp = 1.25; width = 2; height = 1
    raw_data = b"\x00\x01\x02\x03\x04\x05\x06\x07"


class BRouteBridgeV5Test(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_create_only_primitive_raw_spool(self):
        source = B.PrimitiveSceneSnapshotSourceV4(_World(), ego_id=99)
        source.refresh_static(force=True); scene = source.capture(_Snapshot())
        spool = B.RawGroundTruthSpoolV4(self.root / "raw")
        spool.write_identity(identity(3))
        spool.write_scene(frame_id=3, timestamp=1.25, scene=scene,
                          camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                          camera_location=_Vec(),
                          radar_world_xyz=np.zeros((0, 3), np.float32))
        spool.write_semantic(frame_id=3, image=_Semantic())
        manifest = spool.seal()
        self.assertEqual(manifest["counts"], {"identity": 1, "scene": 1, "semantic": 1})
        self.assertEqual((self.root / "raw/semantic/0000000003.bgra").read_bytes(),
                         _Semantic.raw_data)
        with self.assertRaisesRegex(V3.BRouteBridgeError, "already exists"):
            B.RawGroundTruthSpoolV4(self.root / "raw")

    def test_exact_300_closes_admission_before_route_wakes(self):
        attempts = []
        def processor(opportunity, _previous):
            return U.BTransmissionV1(identity(opportunity.frame_id),
                opportunity.action_open_monotonic_raw_ns, 10, True)
        def driver(bridge):
            for frame in range(301):
                attempts.append(frame)
                bridge.offer_prepared(B.RouteOpportunityV4(
                    frame, frame, frame + 1, 10_000 + frame, {"frame_id": frame}))
        bridge = B.BRouteBridgeV4(
            variant=L.ActorVariant.RUN4B, feature_schema_sha256=sha("schema"),
            actor_boundary_sha256=sha("actor"), processor=processor,
            route_driver=driver, raw_spool_root=self.root / "spool")
        for frame in range(300): bridge.transmit_next(frame, None)
        bridge.close()
        self.assertEqual(bridge.transmitted, 300)
        self.assertLessEqual(max(attempts), 300)

    def test_materialization_is_postroute_only(self):
        called = []
        bridge = B.BRouteBridgeV4(
            variant=L.ActorVariant.RUN4B, feature_schema_sha256=sha("schema"),
            actor_boundary_sha256=sha("actor"), processor=lambda *_: None,
            route_driver=lambda _bridge: None, raw_spool_root=self.root / "spool",
            postrun_materializer=lambda source, target: called.append((source, target)) or 7)
        with self.assertRaisesRegex(V3.BRouteBridgeError, "not sealed"):
            bridge.materialize_postroute(self.root / "gt")
        self.assertEqual(called, [])

    def test_live_workers_do_no_gt_conversion_or_quality(self):
        source = inspect.getsource(V3.build_b_collector_class).lower()
        for forbidden in ("._ground_truth(", "semantic_gt_3class(",
                          "evaluate_exact_quality(", "q_perc"):
            self.assertNotIn(forbidden, source)

    def test_seam_restoration_is_in_finally_and_covers_all_symbols(self):
        source = inspect.getsource(B.installed_b_route_seams)
        self.assertIn("finally:", source)
        for symbol in ("LivePilotCellRuntime", "PassiveSplitCollector",
                       "SceneSnapshotSource", "InstallFeedbackLedger"):
            self.assertGreaterEqual(source.count(symbol), 2)


if __name__ == "__main__": unittest.main()

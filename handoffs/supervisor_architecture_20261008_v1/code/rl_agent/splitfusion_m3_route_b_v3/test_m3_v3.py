"""M3.1 v3 completing repair (P3: GT-worker live context/source camera). CPU only, from the live worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v rl_agent.splitfusion_m3_route_b_v3.test_m3_v3
"""
import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from . import capacity as CAP
from . import contract as C
from . import pose_tolerance as PT
from .test_gc_live import Base, Stubs
from . import worker_entry as WE

ROOT = Path(__file__).resolve().parents[2]
V2_POSES = (ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v2/good_all/capture/cells/"
            "a71__favorable_stable/pose_archive/pose_evidence.jsonl")
V2_GT_REJECTED = 1.0009130875232586e-05


def ready():
    from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
    return R


class P3Tests(Base):
    def test_g1_only_the_live_context_line_changes(self):
        original, text = CAP.substituted_source("P3_GT_LIVE_CONTEXT_SOURCE_CAMERA")
        changed = [l for l in original.splitlines() if l not in text.splitlines()]
        self.assertEqual(len(changed), 1)
        self.assertIn('"live context/source camera"', changed[0])
        self.assertIn('PO._agree(inverse, _numeric_matrix(embedded, 4, "frame inverse"), "source frame inverse")', text)
        self.assertIn('"producer camera inverse differs"', text)
        self.assertIn('"source manifest camera location differs"', text)

    def test_g2_install_restore_on_ready(self):
        R = ready()
        before = R._frame_geometry
        R._M3_AGREE_RECONSTRUCTED = PT.agree_reconstructed
        installed = CAP.install(CAP.POSE_GT_WORKER)
        try:
            self.assertEqual(R._frame_geometry.__m3_capacity_substitution__, "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA")
            self.assertIs(R._frame_geometry.__globals__, vars(R))
        finally:
            installed.uninstall()
            del R._M3_AGREE_RECONSTRUCTED
        self.assertIs(R._frame_geometry, before)

    def test_g3_worker_entry_installs_p3_in_both_roles_and_restores(self):
        R = ready()
        before = R._frame_geometry
        for role, expected in (("prep", ["C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE",
                                         "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"]),
                               ("score", ["P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"])):
            seen = {}
            trace = self.tmp / role
            trace.mkdir()
            with Stubs() as st:
                def serve(fd, parent, cpu):
                    seen["sub"] = R._frame_geometry.__m3_capacity_substitution__
                    seen["agree"] = R._M3_AGREE_RECONSTRUCTED is PT.agree_reconstructed
                WE.main(self.argv(role, "control", trace=trace), serve=serve)
            doc = self.trace(role, trace)
            self.assertEqual(seen, dict(sub="P3_GT_LIVE_CONTEXT_SOURCE_CAMERA", agree=True))
            self.assertEqual([r["id"] for r in doc["m3_capacity"]["installed"]], expected)
            self.assertTrue(doc["m3_capacity"]["restored"])
            self.assertEqual(doc["m3_pose"]["rule"], C.POSE)
            self.assertIs(R._frame_geometry, before)
            self.assertFalse(hasattr(R, "_M3_AGREE_RECONSTRUCTED"))

    def test_g4_v2_archive_contains_the_rejected_value_and_the_rule_accepts_every_wire(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose as P
        rows = [json.loads(l) for l in V2_POSES.read_text().splitlines()]
        owned = {r["source"]["frame_id"]: r["source"] for r in rows if r["kind"] == "OWNED_SOURCE"}
        wires = [r for r in rows if r["kind"] == "UE_WIRE"]
        self.assertEqual(len(wires), 1232)
        reg = P.StaticCameraRegistry.audited()
        old_rejects, deltas = 0, []
        for w in wires:
            c = dict(w["context"])
            e = c.pop("ego_world")
            value = P.build_frame_context_v1(**c, **dict(zip(("ego_world_x", "ego_world_y", "ego_world_z",
                                                              "ego_world_pitch", "ego_world_yaw", "ego_world_roll"), e)))
            camera = np.array(owned[w["context"]["frame_id"]]["camera_matrix"])
            context_camera = P.context_camera(value, reg)
            deltas.append(PT.agree_reconstructed(camera, context_camera, "live context/source camera"))
            try:
                P._agree(camera, context_camera, "retained")
            except Exception:
                old_rejects += 1
        self.assertIn(V2_GT_REJECTED, deltas)                       # the exact value the GT worker rejected
        self.assertGreaterEqual(old_rejects, 1)
        self.assertLessEqual(max(deltas), C.POSE["max_tolerance_m"])


if __name__ == "__main__":
    unittest.main()

import ast
import csv
import json
import math
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_range_correction_v1 import apply as A
from rl_agent.splitfusion_range_correction_v1 import features as F

PKG = Path(__file__).resolve().parents[1]


def synthetic_frame(theta=0.3, origin=(10.0, -5.0, 2.0)):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    o = np.array(origin)
    recs = []
    for i, (lx, ly, lz) in enumerate([(12.0, 2.0, -0.5), (25.0, -4.0, 0.3), (7.0, 0.5, -1.0), (33.0, 6.0, 0.1)]):
        w = R @ np.array([lx, ly, lz]) + o
        recs.append({"class_name": "vehicle" if i % 2 == 0 else "person", "score": 0.5 + 0.1 * i,
                     "local_x": lx, "local_y": ly, "local_z": lz, "world_x": w[0], "world_y": w[1], "world_z": w[2],
                     "bbox_x0": 100.0 + 10 * i, "bbox_y0": 150.0, "bbox_x1": 180.0 + 10 * i, "bbox_y1": 220.0,
                     "level_index": 1, "depth_residual": 0.1, "prediction_index": str(i), "actor_forward_depth_m": lx})
    return recs, o


def params(intercept=0.1, coef_scale=0.0, enabled=True):
    n = len(F.FEATURE_NAMES)
    cls = {"enabled": enabled, "intercept": intercept, "coef": [coef_scale] * n, "mean": [0.0] * n, "std": [1.0] * n}
    return {"clip_log_scale": 0.3, "classes": {"vehicle": dict(cls), "person": dict(cls)}}


class TestRangeCorrection(unittest.TestCase):
    def test_origin_recovered_from_records(self):
        recs, o = synthetic_frame()
        self.assertLess(np.abs(A.frame_origin(recs) - o).max(), 1e-9)

    def test_bearing_preserved_lateral_unchanged(self):
        recs, o = synthetic_frame()
        out = A.correct_frame(recs, params(intercept=0.2))
        for r0, r1 in zip(recs, out):
            v0 = np.array([r0["world_x"], r0["world_y"]]) - o[:2]
            v1 = np.array([r1["world_x"], r1["world_y"]]) - o[:2]
            u = v0 / np.linalg.norm(v0)
            lateral = abs(u[0] * v1[1] - u[1] * v1[0])
            self.assertLess(lateral, 1e-9)                          # no lateral component added
            self.assertAlmostEqual(np.linalg.norm(v1) / np.linalg.norm(v0), math.exp(0.2), places=12)
            # image ray unchanged: local_y/local_x and local_z/local_x identical
            self.assertAlmostEqual(r1["local_y"] / r1["local_x"], r0["local_y"] / r0["local_x"], places=12)
            self.assertAlmostEqual(r1["local_z"] / r1["local_x"], r0["local_z"] / r0["local_x"], places=12)
            for k in ("score", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "class_name", "prediction_index"):
                self.assertEqual(r0[k], r1[k])

    def test_output_clip(self):
        recs, o = synthetic_frame()
        out = A.correct_frame(recs, params(intercept=5.0))
        self.assertAlmostEqual(out[0]["local_x"] / recs[0]["local_x"], math.exp(0.3), places=12)

    def test_identity_when_coefficients_zero(self):
        recs, _ = synthetic_frame()
        for p in (params(intercept=0.0, coef_scale=0.0), params(intercept=0.3, enabled=False),
                  {"clip_log_scale": 0.3, "classes": {}}):
            out = A.correct_frame(recs, p)
            for r0, r1 in zip(recs, out):
                self.assertEqual(r0, r1)  # bit-exact

    def test_inference_reads_only_record_fields(self):
        accessed = set()

        class Spy(dict):
            def __getitem__(self, k):
                accessed.add(k)
                return dict.__getitem__(self, k)

        recs, _ = synthetic_frame()
        spies = [Spy(r) for r in recs]
        A.correct_frame(spies, params(intercept=0.1, coef_scale=0.01))
        self.assertTrue(accessed <= set(F.INPUT_FIELDS) | {"actor_forward_depth_m"}, accessed)
        # minimal record (only the declared inputs) suffices
        minimal = [{k: r[k] for k in F.INPUT_FIELDS} for r in recs]
        A.correct_frame(minimal, params(intercept=0.1, coef_scale=0.01))

    def test_inference_modules_have_no_gt_or_io_access(self):
        for name in ("apply.py", "features.py"):
            src = (PKG / name).read_text()
            tree = ast.parse(src)
            mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            self.assertFalse(any(m and ("score" == m.split(".")[-1] or "fit" == m.split(".")[-1]) for m in mods), mods)
            for token in ("object_boxes", "actor_volume", "gt_distance", "import csv", "open(", "radar_tensor", "contracts/"):
                self.assertNotIn(token, src, f"{name} references {token}")

    def test_frozen_params_load_and_sha(self):
        frozen = PKG.parents[1] / "experiments/splitfusion_range_correction_v1/20261006_v1/fit/FROZEN_PARAMS.json"
        if not frozen.exists():
            self.skipTest("not yet frozen")
        digest = (frozen.parent / "FROZEN_PARAMS.sha256").read_text().split()[0]
        p = A.load_frozen(frozen, digest)
        self.assertFalse(p["validation_accessed"])
        with self.assertRaises(RuntimeError):
            A.load_frozen(frozen, "0" * 64)


class TestBaselineReproduction(unittest.TestCase):
    def test_original_counts_reproduced_exactly(self):
        from rl_agent.splitfusion_range_correction_v1 import score as S
        det = S.read_detections(S.DET)
        val = [r["sample_id"] for r in csv.DictReader((S.DS / "dataset/manifest.csv").open()) if r["split"] == "val"]
        vg = S.load_vehicle_gt("val"); avo = S.load_avo_table(S.AVO); st = S.load_val_structural()
        V = dict(tp=0, fn=0, fp=0, neu=0, w=0, e=0.0); P = dict(tp=0, fn=0, fp=0, neu=0, w=0, e=0.0)
        origin_fail = 0
        for sid in val:
            r = S.vehicle_frame(det.get(sid, []), vg.get(sid, []), S.audit_v1.load_ignore(S.DS, "v010", sid))
            for e in r["err"]:
                V["fn" if e is None else "tp"] += 1
                if e is not None: V["e"] += e; V["w"] += e <= 1.0
            V["fp"] += r["fp"]; V["neu"] += r["neutral"]
            r = S.person_frame(det.get(sid, []), avo.get(sid, []), st.get(sid, []))
            for e in r["err"]:
                P["fn" if e is None else "tp"] += 1
                if e is not None: P["e"] += e; P["w"] += e <= 1.2
            P["fp"] += r["fp"]; P["neu"] += r["neutral"]
            origin_fail += A.frame_origin(det.get(sid, [])) is None
        self.assertEqual((V["tp"], V["fn"], V["w"], V["fp"], V["neu"]), (8416, 1275, 7231, 618, 2411))
        self.assertAlmostEqual(V["e"] / V["tp"], 0.4786754207718958, places=12)
        self.assertEqual((P["tp"], P["fn"], P["w"], P["fp"], P["neu"]), (2052, 825, 1533, 862, 289))
        self.assertAlmostEqual(P["e"] / P["tp"], 0.8121813463526099, places=12)
        self.assertEqual(origin_fail, 0)


if __name__ == "__main__":
    unittest.main()

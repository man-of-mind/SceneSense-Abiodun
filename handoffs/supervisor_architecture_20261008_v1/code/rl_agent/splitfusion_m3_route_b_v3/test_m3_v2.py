"""M3.1 v2 repair checks (CPU only). Run from the live worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v rl_agent.splitfusion_m3_route_b_v3.test_m3_v2

R-POSE: archived v1 poses (2,412 owned sources, 1,204 wires), synthetic float32 boundary cases, float32-emulated CARLA
compositions, genuine errors rejected, call-site scoping, install/restore. R-CLASS: real Run-4 updates (decision and
hold frames) through the real Phase-6 install path and the frozen baseline normalizer.
"""
import inspect
import json
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

from . import capacity as CAP
from . import contract as C
from . import map_class as MC
from . import pose_tolerance as PT

ROOT = Path(__file__).resolve().parents[2]
V1_POSES = (ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261007_m3_route_b_v1/good_all/capture/cells/"
            "a71__favorable_stable/pose_archive/pose_evidence.jsonl")


def P():
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    return pose


def ulp(x):
    return float(np.spacing(np.float32(abs(x))))


def pose_matrix(x, y, z=0.05, yaw=37.0, pitch=0.3, roll=0.1):
    return P().carla_transform_matrix((x, y, z, pitch, yaw, roll))


class PoseArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [json.loads(line) for line in V1_POSES.read_text().splitlines()]
        cls.owned = {r["source"]["frame_id"]: r["source"] for r in rows if r["kind"] == "OWNED_SOURCE"}
        cls.wires = [r for r in rows if r["kind"] == "UE_WIRE"]
        cls.errors = [r for r in rows if r["kind"] == "ERROR"]

    def test_a1_all_accepted_owned_poses_pass_and_are_within_one_ulp(self):
        pose = P()
        reg = pose.StaticCameraRegistry.audited()
        cal = reg.resolve(*reg.identities[0])
        mount = pose.carla_transform_matrix(pose.RADAR_MOUNT)
        self.assertEqual(len(self.owned), 2412)
        ratios = []
        for s in self.owned.values():
            ego = np.array(s["ego_matrix"])
            for actual, expected, name in ((ego @ mount, s["radar_matrix"], "ego/mount radar"),
                                           (ego @ np.asarray(cal.camera_to_ego_values), s["camera_matrix"],
                                            "ego/mount RGB camera")):
                old = pose._agree(actual, expected, name)                 # the archived poses were accepted
                new = PT.agree_reconstructed(actual, expected, name)
                self.assertEqual(old, new)                                # same recorded overall delta
                e = np.array(expected)
                m = max(np.abs(actual[:3, 3]).max(), np.abs(e[:3, 3]).max())
                ratios.append(np.abs(actual[:3, 3] - e[:3, 3]).max() / ulp(m))
        self.assertLessEqual(max(ratios), 1.0)

    def test_a2_all_wire_source_camera_checks_reproduce_recorded_delta(self):
        pose = P()
        reg = pose.StaticCameraRegistry.audited()
        self.assertEqual(len(self.wires), 1204)
        for w in self.wires:
            c = dict(w["context"])
            e = c.pop("ego_world")
            value = pose.build_frame_context_v1(**c, **dict(zip(("ego_world_x", "ego_world_y", "ego_world_z",
                                                                  "ego_world_pitch", "ego_world_yaw", "ego_world_roll"), e)))
            source = self.owned[w["context"]["frame_id"]]
            delta = PT.agree_reconstructed(pose.context_camera(value, reg), source["camera_matrix"], "wire/source camera")
            self.assertEqual(delta, w["max_abs_matrix_delta"])

    def test_a3_rejected_frame_has_no_matrices_so_no_replay_is_claimed(self):
        self.assertEqual([(r["frame_id"], r["error"].split(":")[0]) for r in self.errors], [(3765, "PoseParityError")])
        self.assertNotIn(3765, self.owned)


class PoseSyntheticTests(unittest.TestCase):
    def check(self, actual, expected, name="synthetic"):
        return PT.agree_reconstructed(actual, expected, name)

    def test_s1_float32_boundaries_plus_minus_128_256(self):
        for c in (127.99, 128.0, 128.01, 255.99, 256.0, 256.01, 300.0, 511.9):
            for sign in (1, -1):
                for axis in (0, 1):
                    xy = [5.0, 5.0]
                    xy[axis] = sign * c
                    m = pose_matrix(*xy)
                    tol = max(1e-5, 2 * ulp(c))
                    ok, bad = m.copy(), m.copy()
                    ok[axis, 3] += math.copysign(0.95 * tol, -sign)            # toward origin: magnitude <= c
                    bad[axis, 3] += math.copysign(1.05 * tol, -sign)
                    self.check(ok, m)
                    with self.assertRaises(Exception):
                        self.check(bad, m)

    def test_s2_floor_below_64m_is_the_retained_1e5(self):
        m = pose_matrix(40.0, -20.0)
        for delta, accepted in ((9.9e-6, True), (1.01e-5, False)):
            a = m.copy()
            a[0, 3] += delta
            if accepted:
                self.check(a, m)
            else:
                with self.assertRaises(Exception):
                    self.check(a, m)

    def test_s3_float32_emulated_carla_composition_passes_new_and_breaks_old_beyond_128(self):
        pose = P()
        rng = np.random.default_rng(20261007)
        mount = pose.carla_transform_matrix(pose.RADAR_MOUNT)
        old_fail = {"<128": 0, ">=128": 0}
        for _ in range(4000):
            r = rng.uniform(5.0, 300.0)
            angle = rng.uniform(0, 2 * np.pi)
            x, y = np.float32(r * np.cos(angle)), np.float32(r * np.sin(angle))
            ego = pose_matrix(float(x), float(y), yaw=float(np.float32(rng.uniform(-180, 180))))
            reconstructed = ego @ mount
            carla = (ego.astype(np.float32) @ mount.astype(np.float32)).astype(np.float32).astype(np.float64)
            self.check(reconstructed, carla)
            try:
                pose._agree(reconstructed, carla, "old")
            except Exception:
                old_fail["<128" if max(abs(x), abs(y)) < 128 else ">=128"] += 1
        self.assertEqual(old_fail["<128"], 0)
        self.assertGreater(old_fail[">=128"], 0)                      # the retained bound is unreliable there

    def test_s4_genuine_errors_are_rejected(self):
        pose = P()
        m = pose_matrix(-92.9, 128.6)
        cases = []
        t = m.copy(); t[1, 3] += 1e-3; cases.append(("translation 1 mm", t))
        cases.append(("rotation 1e-4 rad", m @ pose.carla_transform_matrix((0, 0, 0, 0, math.degrees(1e-4), 0))))
        cases.append(("radar mount used as camera mount", m @ pose.carla_transform_matrix(pose.RADAR_MOUNT)))
        cases.append(("wrong actor 3 m away", pose_matrix(-92.9, 131.6)))
        h = m.copy(); h[3, 0] = 1e-3; cases.append(("homogeneous row", h))
        n = m.copy(); n[0, 3] = float("nan"); cases.append(("non-finite", n))
        for name, bad in cases:
            with self.assertRaises(Exception, msg=name):
                self.check(bad, m)
        far = pose_matrix(600.0, 0.0)
        with self.assertRaises(Exception):
            self.check(far, far)                                       # outside the registered coordinate range
        self.assertLessEqual(PT.translation_tolerance(512.0), C.POSE["max_tolerance_m"])
        with self.assertRaises(ValueError):
            PT.translation_tolerance(512.5)


class PoseScopeTests(unittest.TestCase):
    def test_c1_only_the_four_reconstructed_call_sites_change(self):
        original, text = CAP.substituted_source("P1_OWNED_POSE_RECONSTRUCTED")
        self.assertEqual(text.count("_M3_AGREE_RECONSTRUCTED("), 3)
        for strict in ('_agree(camera, owned["camera"].get_matrix(), "owned RGB camera")',
                       '_agree(radar, owned["radar"].get_matrix(), "owned radar")',
                       'f"{role} source frame/timestamp mismatch"', '"owned snapshot/frame mismatch"',
                       '"owned pose requires accepted all-sensor barrier"', "registry.resolve(*registry.identities[0])"):
            self.assertIn(strict, text)
        self.assertEqual([l for l in original.splitlines() if l not in text.splitlines()],
                         [l for l in original.splitlines() if "_agree(ego" in l])
        original2, text2 = CAP.substituted_source("P2_WIRE_SOURCE_CAMERA")
        self.assertEqual(text2.count("_M3_AGREE_RECONSTRUCTED("), 1)
        self.assertIn('_agree(image.transform.get_matrix(), source.matrix(role), f"preparation {role}")', text2)
        self.assertEqual(len([l for l in original2.splitlines() if l not in text2.splitlines()]), 1)

    def test_c2_install_restore_and_contextmanager_shape(self):
        pose = P()
        before = (pose.capture_owned_pose, pose.install_ue_pose)
        pose._M3_AGREE_RECONSTRUCTED = PT.agree_reconstructed
        installed = CAP.install(CAP.POSE)
        try:
            self.assertEqual(pose.capture_owned_pose.__m3_capacity_substitution__, "P1_OWNED_POSE_RECONSTRUCTED")
            self.assertEqual(pose.install_ue_pose.__m3_capacity_substitution__, "P2_WIRE_SOURCE_CAMERA")
            self.assertTrue(hasattr(pose.install_ue_pose, "__wrapped__"))           # still a contextmanager
            self.assertIs(inspect.unwrap(pose.install_ue_pose).__globals__, vars(pose))
        finally:
            installed.uninstall()
            del pose._M3_AGREE_RECONSTRUCTED
        self.assertEqual((pose.capture_owned_pose, pose.install_ue_pose), before)

    def test_c3_live_gt_and_producer_copy_checks_untouched(self):
        for module, needle in (("rl_agent.splitfusion_run5_live_avo_v1.live_gt", "any(delta > 1e-5 for delta in deltas.values())"),
                               ("rl_agent.splitfusion_numeric_gt_producer_v1.source", "rtol=0, atol=1e-5")):
            self.assertIn(needle, inspect.getsource(__import__(module, fromlist=["x"])))
        self.assertEqual({s["id"] for s in C.POSE_SUBSTITUTIONS},
                         {"P1_OWNED_POSE_RECONSTRUCTED", "P2_WIRE_SOURCE_CAMERA", "P3_GT_LIVE_CONTEXT_SOURCE_CAMERA"})


class CompositionTests(unittest.TestCase):
    def test_d1_child_capacity_scope_substitutes_before_install_ue_pose(self):
        import tempfile
        from . import entries
        pose = P()
        before = (pose.capture_owned_pose, pose.install_ue_pose, pose.PoseArchive.__init__)
        with tempfile.TemporaryDirectory() as tmp:
            with entries.install_ue_capacity(Path(tmp)) as installed:
                self.assertEqual([r["id"] for r in installed.records],
                                 ["C4_UE_TICKETS", "C5_UE_POSE_ARCHIVE", "P1_OWNED_POSE_RECONSTRUCTED", "P2_WIRE_SOURCE_CAMERA"])
                self.assertIs(pose._M3_AGREE_RECONSTRUCTED, PT.agree_reconstructed)
                self.assertEqual(pose.install_ue_pose.__m3_capacity_substitution__, "P2_WIRE_SOURCE_CAMERA")
            written = json.loads((Path(tmp) / "m3_route/POSE_TOLERANCE.json").read_text())
            self.assertEqual(written["rule"], C.POSE)
            self.assertTrue(json.loads((Path(tmp) / "m3_route/UE_CAPACITY_SUBSTITUTIONS.json").read_text())["restored"])
        self.assertEqual((pose.capture_owned_pose, pose.install_ue_pose, pose.PoseArchive.__init__), before)
        self.assertFalse(hasattr(pose, "_M3_AGREE_RECONSTRUCTED"))
        self.assertIn("POSE.install_ue_pose(attempt_dir=attempt", inspect.getsource(entries.child))

    def test_d2_map_trampoline_scope_sets_and_restores(self):
        import tempfile
        from rl_agent.splitfusion_single_gpu_profiler_v1 import seams_ue_v1 as SU
        from . import entries
        with tempfile.TemporaryDirectory() as tmp:
            with entries.install_map_trampoline(Path(tmp)):
                self.assertEqual(SU.MAP_SERVER, C.MAP_SERVER_TRAMPOLINE)
                self.assertTrue((ROOT / C.MAP_SERVER_TRAMPOLINE).is_file())
            self.assertTrue(json.loads((Path(tmp) / "m3_route/MAP_SERVER_SEAM.json").read_text())["restored"])
        self.assertEqual(SU.MAP_SERVER, C.RETAINED_MAP_SERVER)


class MapClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as ER
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import CONTRACT
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import FakeActor, Pipeline
        from rl_agent.splitfusion_single_gpu_profiler_v1.test_early_submit_v1 import fake_compute
        p = Pipeline(FakeActor(mode_id=9, q_e4=4321))
        wires = [p.step(), p.step()]
        proc = ER.Run4EdgeProcessorV2(contract=CONTRACT, run_id="run", cell_id="cell", service_deadline_s=0.1,
                                      ack_timeout_s=0.5, compute=fake_compute)
        cls.updates = [proc.process(w.wire, edge_timing={}).update for w in wires]
        cls.contract = CONTRACT

    def test_k1_convert_adds_only_type_on_a_copy(self):
        for u in self.updates:
            before = json.dumps(u, sort_keys=True)
            out = MC.convert(u)
            self.assertEqual(json.dumps(u, sort_keys=True), before)                  # input untouched
            self.assertEqual(len(out["records"]), len(u["records"]))
            for a, b in zip(u["records"], out["records"]):
                self.assertEqual({k: v for k, v in b.items() if k != "type"}, a)    # order/fields/scores/coords
                self.assertEqual(b["type"], {"vehicle": "Vehicle", "person": "Pedestrian"}[a["class_name"]])
            self.assertEqual({k: v for k, v in out.items() if k != "records"}, {k: v for k, v in u.items() if k != "records"})
        self.assertEqual({u["run4_identity"]["frame_kind"] for u in self.updates}, {"POLICY_DECISION", "POLICY_HOLD"})

    def test_k2_unknown_class_untyped_and_conflicts_refused(self):
        u = dict(self.updates[0], records=[dict(self.updates[0]["records"][0], class_name="bicycle")])
        self.assertNotIn("type", MC.convert(u)["records"][0])
        bad = dict(self.updates[0], records=[dict(self.updates[0]["records"][0], type="Pedestrian")])
        with self.assertRaises(MC.MapClassError):
            MC.convert(bad)
        source = inspect.getsource(MC)
        for gt_source in ("numeric_gt", "ground_truth", "actor_boxes", "gt_identity"):
            self.assertNotIn(gt_source, source)                     # labels come from the model records only

    def test_k3_real_install_path_keeps_class_in_installed_json(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_map_server_v2 as M6
        saved = M6.install
        try:
            MC.install_map_class_conversion()
            server = M6.install(self.contract)
            argv, sys.argv = sys.argv, ["map"]
            try:
                server.baseline.CONFIG = server.baseline.parse_args()
            finally:
                sys.argv = argv
            for u in self.updates:
                server.install_under_state_lock(u, time.time())
                record = server.baseline.installed_frame_history[(u["stream_id"], int(u["frame_id"]))]
                self.assertEqual([o["type"] for o in record["objects"]],
                                 [{"vehicle": "Vehicle", "person": "Pedestrian"}[r["class_name"]] for r in u["records"]])
                self.assertEqual([o["score"] for o in record["objects"]], [r["score"] for r in u["records"]])
        finally:
            M6.install = saved

    def test_k4_trampoline_keeps_the_process_marker(self):
        from rl_agent.splitfusion_single_gpu_profiler_v1 import runner_v1, sampler_v1
        self.assertIn("map_profiled_v1.py", C.MAP_SERVER_TRAMPOLINE)
        self.assertIn("map_profiled_v1.py", runner_v1.PROFILER_PROCESS_MARKERS)
        role = next(rule for name, rule in sampler_v1.ROLE_RULES if name == "map")
        self.assertTrue(role("python3", "python3 " + str(ROOT / C.MAP_SERVER_TRAMPOLINE)))


if __name__ == "__main__":
    unittest.main()

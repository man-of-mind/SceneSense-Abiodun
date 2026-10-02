"""Focused tests for the radar-preparation bundle seam (CPU unless noted)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import radar_bundle_v1 as RB
from . import runner_v1 as RUN
from . import threadcap_v1 as TC

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
EXPERIMENT_PATH = PACKAGE / "experiment_radar_bundle_v1.json"


class PackingTest(unittest.TestCase):
    def test_identity_resize_packing_is_bit_exact_against_production_on_cpu(self) -> None:
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base

        rng = np.random.default_rng(5)
        device = torch.device("cpu")
        for _ in range(4):
            frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            radar = (rng.random((4, RB.MODEL_HEIGHT, RB.MODEL_WIDTH)) * 2 - 1).astype(np.float32)
            reference = live_base._prepare_live_input(frame, radar, device)
            optimized = RB.prepare_live_input_identity_resize(frame, radar, device)
            self.assertTrue(torch.equal(reference, optimized))
            self.assertEqual((reference.shape, reference.dtype), (optimized.shape, optimized.dtype))
        # a non-model-size radar tensor still takes the production resize path
        radar_small = (rng.random((4, 224, 384))).astype(np.float32)
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.assertTrue(torch.equal(live_base._prepare_live_input(frame, radar_small, device),
                                    RB.prepare_live_input_identity_resize(frame, radar_small, device)))

    def test_packing_seam_requires_the_unwrapped_production_function(self) -> None:
        module = types.SimpleNamespace(_prepare_live_input=lambda f, r, d: None)
        module._prepare_live_input.__name__ = "_prepare_live_input"
        self.assertEqual(RB.install_packing_short_circuit(module)["packing_replaced"],
                         "_prepare_live_input")
        self.assertIs(module._prepare_live_input, RB.prepare_live_input_identity_resize)
        wrapped = types.SimpleNamespace(_prepare_live_input=lambda f, r, d: None)   # a wrapper
        with self.assertRaises(RB.RadarBundleError):
            RB.install_packing_short_circuit(wrapped)


class BuilderProxyTest(unittest.TestCase):
    def test_bundle_module_forwards_attributes_and_strips_profile_keys(self) -> None:
        from rl_agent.splitfusion_supervisor_analysis_v1 import profiled_sensor_stages as PS

        module = types.SimpleNamespace(radar_support_count=lambda: 7, CONST=3)
        seen = {}

        def fake_profiled(**kwargs):
            seen.update(kwargs)
            return np.zeros((4, 2, 2), np.float32), {"u": np.zeros(1)}, {
                "radar_points": 1.0, "profile_radar_total_ms": 9.0}

        original = PS.build_radar_sample_profiled
        PS.build_radar_sample_profiled = fake_profiled
        try:
            proxy = RB.BundleParkedModule(module, rasterizer="R")
            tensor, points, summary = proxy.build_radar_sample(detections=np.zeros((0, 4)), tracker=None)
            self.assertEqual(seen["rasterizer_override"], "R")
            self.assertEqual(summary, {"radar_points": 1.0})
            self.assertEqual((proxy.radar_support_count(), proxy.CONST, proxy.calls), (7, 3, 1))
            with self.assertRaises(RB.RadarBundleError):
                proxy.build_radar_sample(rasterizer_override="X")
        finally:
            PS.build_radar_sample_profiled = original

    def test_innermost_timer_is_found_under_the_profiler_proxy(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U

        from . import recorder_v1 as RV
        from . import seams_ue_v1 as SU

        module = types.SimpleNamespace(build_radar_sample=lambda **kw: ("t", {}, {}))
        timer = U._PreparationTimer(module, lambda real, **kw: real(**kw))
        outer = SU._MethodSpanProxy(timer, {"build_radar_sample": "ue.radar_rasterize"}, RV.active)
        self.assertIs(RB.innermost_preparation_timer(outer, U._PreparationTimer), timer)
        with self.assertRaises(RB.RadarBundleError):
            RB.innermost_preparation_timer(module, U._PreparationTimer)
        timer._parked = RB.BundleParkedModule(module, rasterizer=None)
        self.assertIs(outer._parked._bundle_module, module)          # Run-4 GT path still forwards

    def test_tracker_replacement_requires_empty_state_and_keeps_parameters(self) -> None:
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base
        from rl_agent.splitfusion_sensor_optimization_v1.optimized_stages import (
            SingleSortStationaryTrackAccumulator,
        )

        tracker = live_base.FastStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
        replaced = RB.optimized_tracker_like(tracker)
        self.assertIsInstance(replaced, SingleSortStationaryTrackAccumulator)
        self.assertEqual((replaced.stationary_velocity_mps, replaced.parked_threshold_s,
                          replaced.association_grid_m, replaced.max_stale_s), (0.35, 5.0, 1.5, 2.0))
        tracker.update(np.array([[1.0, 2.0, 0.0, 0.0]]), 0.1)
        with self.assertRaises(RB.RadarBundleError):
            RB.optimized_tracker_like(tracker)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_collector_seam_installs_outermost_and_refuses_double_install(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base

        from . import recorder_v1 as RV
        from . import seams_ue_v1 as SU

        module = types.SimpleNamespace(build_radar_sample=lambda **kw: None)

        class FakeBase:
            def __init__(self, **kwargs):
                self.live = SimpleNamespace(device=torch.device("cuda:0"))
                self.tracker = live_base.FastStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
                timer = U._PreparationTimer(module, lambda real, **kw: real(**kw))
                self.parked = SU._MethodSpanProxy(timer, {"build_radar_sample": "x"}, RV.active)

        pinned = types.SimpleNamespace(PassiveSplitCollector=FakeBase)
        with tempfile.TemporaryDirectory() as tmp:
            cls = RB.install_builder_and_tracker(pinned=pinned, phase6_ue=U, evidence_dir=Path(tmp),
                                                 record={"packing_replaced": "_prepare_live_input"})
            self.assertIs(pinned.PassiveSplitCollector, cls)
            collector = cls()
            snapshot = json.loads((Path(tmp) / RB.SNAPSHOT_NAME).read_text())
            self.assertEqual(snapshot["bundle"], "optimized")
            self.assertEqual(snapshot["components"], list(RB.COMPONENTS))
            self.assertEqual(snapshot["tracker"], "SingleSortStationaryTrackAccumulator")
            self.assertIsInstance(collector.parked._parked, RB.BundleParkedModule)
            with self.assertRaises(FileExistsError):                 # create-only snapshot
                cls()


class ArmsTest(unittest.TestCase):
    def test_arms_c_and_t_carry_the_cap_and_the_bundle_marker(self) -> None:
        base = {"PATH": "/bin"}
        for arm, bundle in (("C", "production"), ("T", "optimized")):
            env = TC.arm_environment(base, arm)
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["MKL_NUM_THREADS"], "1")
            self.assertEqual(env[TC.BUNDLE_ENV], bundle)
        restored, _ = TC.restored_environment(TC.arm_environment(base, "T"),
                                              {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
        self.assertEqual(restored, base)                                   # markers never leak

    def test_child_refuses_a_bundle_marker_that_contradicts_the_arm(self) -> None:
        restore = json.dumps({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
        script = """
            import json
            from rl_agent.splitfusion_single_gpu_profiler_v1 import threadcap_v1 as TC
            try:
                arm, _ = TC.child_arm(); print(json.dumps(["ok", arm]))
            except TC.ThreadCapError as exc:
                print(json.dumps(["refused", str(exc)[:50]]))
        """
        base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", *TC.CAP_VARIABLES)}
        cap = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", TC.RESTORE_ENV: restore}
        ok = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=str(ROOT), text=True,
                            capture_output=True, env={**base, **cap, TC.ARM_ENV: "T",
                                                      TC.BUNDLE_ENV: "optimized"}).stdout
        bad = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=str(ROOT), text=True,
                             capture_output=True, env={**base, **cap, TC.ARM_ENV: "C",
                                                       TC.BUNDLE_ENV: "optimized"}).stdout
        self.assertEqual(json.loads(ok.strip().splitlines()[-1])[:2], ["ok", "T"])
        self.assertEqual(json.loads(bad.strip().splitlines()[-1])[0], "refused")

    def test_e4_checks_the_recorded_bundle_against_the_arm(self) -> None:
        def cell(tmp: Path, arm: str, bundle: dict) -> dict:
            (tmp / TC.EVIDENCE_LEAF).mkdir(parents=True, exist_ok=True)
            env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            for stage, cap in (("PRE", None), ("POST", True)):
                (tmp / TC.EVIDENCE_LEAF / TC.UE_SETTINGS_NAME.format(stage=stage)).write_text(json.dumps(
                    {"stage": stage.lower(), "arm": arm, "environment": env, "cap_effective": cap,
                     "torch": {"intra_op_threads": 1, "inter_op_threads": 24},
                     "openmp_max_threads": {"libgomp.so.1": 1}}))
            (tmp / "radar_bundle").mkdir(exist_ok=True)
            (tmp / "radar_bundle" / RB.SNAPSHOT_NAME).write_text(json.dumps(bundle))
            return RUN.experiment_checks(cell=tmp, arm=arm, inventory={"scope_ok": True})["checks"]

        with tempfile.TemporaryDirectory() as tmp:
            good_t = cell(Path(tmp) / "t", "T", {"bundle": "optimized", "components": list(RB.COMPONENTS),
                                                  "tracker": "SingleSortStationaryTrackAccumulator"})
            self.assertTrue(all(good_t.values()), good_t)
            good_c = cell(Path(tmp) / "c", "C", {"bundle": "production", "components": []})
            self.assertTrue(all(good_c.values()), good_c)
            wrong = cell(Path(tmp) / "w", "T", {"bundle": "production", "components": []})
            self.assertFalse(wrong["E4_RADAR_BUNDLE_AS_REGISTERED"])

    @unittest.skipUnless(EXPERIMENT_PATH.is_file(), "bundle registration not sealed yet")
    def test_bundle_registration_is_sealed_and_bound(self) -> None:
        registration = RUN.load_qualification(EXPERIMENT_PATH)
        self.assertEqual(set(registration["arms"]), {"C", "T"})
        self.assertEqual(registration["arms"]["C"]["bundle"], "production")
        self.assertEqual(registration["arms"]["T"]["bundle"], "optimized")
        for arm in ("C", "T"):
            self.assertEqual(registration["arms"][arm]["env"], TC.TREATMENT)
            self.assertEqual(RUN.select_arm(registration, arm)["bundle"], TC.BUNDLES[arm])
        RUN.verify_capture_binding(registration)
        bound = RUN.verify_experiment_binding(registration)
        self.assertIn("offline_parity_result", bound)
        self.assertIn("optimized_stages", bound)


if __name__ == "__main__":
    unittest.main()

"""Focused tests: v2 single-buffer input is bit-identical to the installed v1 candidate, the corrected original and
the recorded live tensors; identical refusals; seam ordering. CPU tensors (CUDA parity is in prove.py).
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v rl_agent.splitfusion_ue_input_prep_v2.test_candidate
"""
import unittest

import numpy as np
import torch

from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
from rl_agent.splitfusion_ue_input_prep_v1 import candidate as V1
from rl_agent.splitfusion_ue_input_prep_v1 import test_candidate as T1

from . import candidate as V2


def outcome(fn, *args):
    try:
        return ("OK", fn(*args, "cpu").numpy().tobytes())
    except Exception as error:
        return (type(error).__name__, str(error))


class RecordedParity(unittest.TestCase):
    def test_all_49_recorded_frames_bit_exact(self):
        n = 0
        for frame, bgra, live in T1.recorded():
            bgr = T1.carla_image_to_bgr(bgra)
            radar = np.ascontiguousarray(live[0, 3:7, :432, :])
            out = V2.prepare_live_input_fused(bgr, radar, "cpu")
            self.assertEqual((tuple(out.shape), out.dtype, out.is_contiguous()), ((1, 7, 448, 768), torch.float32, True))
            raw = out.numpy().tobytes()
            self.assertEqual(raw, live.tobytes(), frame)
            self.assertEqual(raw, V1.prepare_live_input_bgr2rgb(bgr, radar, "cpu").numpy().tobytes(), frame)
            self.assertEqual(raw, P.prepare_live_input(bgr, radar, "cpu").numpy().tobytes(), frame)
            self.assertTrue((out[0, :, 432:] == 0).all() and not torch.signbit(out[0, :, 432:]).any())
            n += 1
        self.assertEqual(n, 49)


class SyntheticParity(unittest.TestCase):
    def test_random_extreme_and_non_contiguous_inputs(self):
        rng = np.random.default_rng(3)
        cases = []
        for i in range(12):
            bgr = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
            radar = (rng.standard_normal((4, 432, 768)) * (10.0 ** (i % 4))).astype(np.float32)
            cases.append((bgr, radar))
        cases.append((np.zeros((720, 1280, 3), np.uint8), np.zeros((4, 432, 768), np.float32)))
        cases.append((np.full((720, 1280, 3), 255, np.uint8), np.full((4, 432, 768), -0.0, np.float32)))
        big = rng.integers(0, 256, size=(720, 1280, 6), dtype=np.uint8)
        cases.append((big[:, :, ::2], np.asfortranarray(rng.standard_normal((4, 432, 768)).astype(np.float32))))
        for bgr, radar in cases:
            self.assertEqual(outcome(V2.prepare_live_input_fused, bgr, radar), outcome(V1.prepare_live_input_bgr2rgb, bgr, radar))

    def test_fresh_output_per_call(self):
        bgr = np.zeros((720, 1280, 3), np.uint8); radar = np.zeros((4, 432, 768), np.float32)
        a = V2.prepare_live_input_fused(bgr, radar, "cpu"); b = V2.prepare_live_input_fused(bgr, radar, "cpu")
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        before = a.clone(); V2.prepare_live_input_fused(bgr + 1, radar + 1, "cpu")
        self.assertTrue(torch.equal(a, before))


class Refusals(unittest.TestCase):
    def test_malformed_inputs_refused_identically(self):
        good_bgr = np.zeros((720, 1280, 3), np.uint8); good_radar = np.zeros((4, 432, 768), np.float32)
        nan = good_radar.copy(); nan[1, 2, 3] = np.nan
        inf = good_radar.copy(); inf[0, 0, 0] = np.inf
        bad = [(good_bgr.astype(np.float32), good_radar), (good_bgr[:, :, :2], good_radar), (good_bgr[:719], good_radar),
               (good_bgr.tolist(), good_radar), (None, good_radar), (good_bgr, nan), (good_bgr, inf),
               (good_bgr, good_radar.astype(np.float64)), (good_bgr, good_radar[:3]), (good_bgr, good_radar[:, :431]),
               (good_bgr, torch.zeros(4, 432, 768)), (np.zeros((720, 1280, 4), np.uint8), good_radar)]
        for bgr, radar in bad:
            got = outcome(V2.prepare_live_input_fused, bgr, radar)
            self.assertEqual(got, outcome(V1.prepare_live_input_bgr2rgb, bgr, radar))
            self.assertEqual(got[0], "PerceptionParityError")


class Seam(unittest.TestCase):
    def test_requires_v1_installed_and_restores(self):
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
        from rl_agent.splitfusion_live_parity_repair_v1.runtime import install_ue_perception
        bind_legacy_namespace()
        original = B._prepare_live_input
        with self.assertRaises(V2.CandidateSeamError):
            with V2.install_fused_candidate():
                pass
        with install_ue_perception(), V1.install_bgr2rgb_candidate():
            with V2.install_fused_candidate() as info:
                self.assertIs(B._prepare_live_input, V2.prepare_live_input_fused)
                self.assertTrue(info["enabled"])
            self.assertIs(B._prepare_live_input, V1.prepare_live_input_bgr2rgb)
        self.assertIs(B._prepare_live_input, original)
        self.assertEqual(V2.verify_v1_source(), V2.V1_SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()

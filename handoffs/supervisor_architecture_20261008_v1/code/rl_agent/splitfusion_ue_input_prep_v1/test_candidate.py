"""Regression tests: candidate is bit-identical to the corrected function and to recorded live tensors.

CPU only (CUDA_VISIBLE_DEVICES empty). Recorded frames: the 49-frame v11 treatment capture.
"""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
from rl_agent.splitfusion_ue_input_prep_v1 import candidate as C

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = (ROOT / "experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11/treatment/capture/cells/"
           "a71__favorable_stable")


def carla_image_to_bgr(bgra):
    # The unchanged ue_route_b_split_cell_adapter_v1.carla_image_to_bgr arithmetic, applied to the raw bytes.
    source = (ROOT / "rl_agent/ue_route_b_split_cell_adapter_v1.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "carla_image_to_bgr")
    ns = dict(np=np, Any=object)   # the source annotates image: Any
    exec(compile(ast.Module(body=[node], type_ignores=[]), "carla_image_to_bgr", "exec"), ns)

    class Image:
        raw_data, height, width = bgra.tobytes(), bgra.shape[0], bgra.shape[1]
    return ns["carla_image_to_bgr"](Image)


def recorded():
    sources = {}
    for d in sorted(CAPTURE.glob("parity_complete_targets/*.sources")):
        sources[json.loads((d / "MANIFEST.json").read_text())["identity"]["frame_id"]] = d / "rgb_bgra.npy"
    for path in sorted(CAPTURE.glob("paired_input_archive/*.npy")):
        frame = int(path.name.split("__frame")[1].split("__")[0])
        if frame in sources:
            yield frame, np.load(sources[frame]), np.load(path)


class RecordedParity(unittest.TestCase):
    def test_all_recorded_frames_bit_exact(self):
        n = 0
        for frame, bgra, live in recorded():
            frame_bgr = carla_image_to_bgr(bgra)
            radar = np.ascontiguousarray(live[0, 3:7, :432, :])
            original = P.prepare_live_input(frame_bgr, radar, "cpu").numpy()
            candidate = C.prepare_live_input_bgr2rgb(frame_bgr, radar, "cpu").numpy()
            self.assertEqual(candidate.tobytes(), original.tobytes(), frame)
            self.assertEqual(candidate.tobytes(), live.tobytes(), frame)
            self.assertEqual((candidate.shape, candidate.dtype), ((1, 7, 448, 768), np.float32))
            self.assertTrue((candidate[0, :, 432:, :] == 0).all())
            n += 1
        self.assertEqual(n, 49)


class SyntheticParity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.bgra = rng.integers(0, 256, (720, 1280, 4), dtype=np.uint8)
        self.radar = rng.standard_normal((4, 432, 768)).astype(np.float32)

    def same(self, frame, radar=None):
        radar = self.radar if radar is None else radar
        a = P.prepare_live_input(frame, radar, "cpu")
        b = C.prepare_live_input_bgr2rgb(frame, radar, "cpu")
        self.assertEqual(a.numpy().tobytes(), b.numpy().tobytes())
        self.assertEqual((a.device, a.dtype, a.shape), (b.device, b.dtype, b.shape))

    def test_layouts(self):
        view = self.bgra[:, :, :3]
        self.same(view.copy())
        self.same(view)                                   # non-contiguous
        self.same(np.asfortranarray(view))
        ro = view.copy(); ro.setflags(write=False)
        self.same(ro)

    def test_extremes(self):
        self.same(np.zeros((720, 1280, 3), np.uint8))
        self.same(np.full((720, 1280, 3), 255, np.uint8))

    def test_input_untouched(self):
        frame = self.bgra[:, :, :3].copy(); before = frame.tobytes()
        C.prepare_live_input_bgr2rgb(frame, self.radar, "cpu")
        self.assertEqual(frame.tobytes(), before)

    def test_same_rejections(self):
        bad = [(self.bgra[:, :, :3].astype(np.int16), self.radar), (self.bgra[:, :, :3][:719], self.radar),
               (self.bgra[:, :, :3].copy(), self.radar.astype(np.float64)),
               (self.bgra[:, :, :3].copy(), np.where(np.arange(4)[:, None, None] == 1, np.nan, self.radar).astype(np.float32)),
               (self.bgra[:, :, :3].copy(), np.where(np.arange(4)[:, None, None] == 2, np.inf, self.radar).astype(np.float32))]
        for frame, radar in bad:
            with self.assertRaises(P.PerceptionParityError) as a:
                P.prepare_live_input(frame, radar, "cpu")
            with self.assertRaises(P.PerceptionParityError) as b:
                C.prepare_live_input_bgr2rgb(frame, radar, "cpu")
            self.assertEqual(str(a.exception), str(b.exception))


class Seam(unittest.TestCase):
    def test_source_line_diff_is_exactly_one_line(self):
        original = (ROOT / C.ORIGINAL_PERCEPTION).read_text()
        start = original.index("    c = CONTRACT\n    if (not isinstance(frame_bgr")
        body_o = original[start:].split("\n\n\n")[0].splitlines()
        cand = Path(C.__file__).read_text()
        s = cand.index("    c = P.CONTRACT\n")
        body_c = cand[s:].split("\n\n\n")[0].splitlines()
        norm = lambda line: line.replace("P.CONTRACT", "CONTRACT").replace("P.PerceptionParityError", "PerceptionParityError")
        diff = [(a, b) for a, b in zip(body_o, body_c) if norm(b) != a]
        self.assertEqual(len(body_o), len(body_c))
        self.assertEqual(len(diff), 1)
        self.assertIn("np.ascontiguousarray(frame_bgr[:, :, ::-1])", diff[0][0])
        self.assertIn("cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)", diff[0][1])

    def test_pinned_original(self):
        self.assertEqual(C.verify_original_source(), C.ORIGINAL_PERCEPTION_SHA256)

    def test_install_requires_corrected_function_and_restores(self):
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
        legacy = B._prepare_live_input
        with self.assertRaises(C.CandidateSeamError):
            with C.install_bgr2rgb_candidate():
                pass
        self.assertIs(B._prepare_live_input, legacy)
        B._prepare_live_input = P.prepare_live_input            # what install_ue_perception() installs
        try:
            with C.install_bgr2rgb_candidate() as state:
                self.assertTrue(state["enabled"])
                self.assertIs(B._prepare_live_input, C.prepare_live_input_bgr2rgb)
            self.assertIs(B._prepare_live_input, P.prepare_live_input)
            with C.install_bgr2rgb_candidate(enabled=False) as state:
                self.assertFalse(state["enabled"])
                self.assertIs(B._prepare_live_input, P.prepare_live_input)
        finally:
            B._prepare_live_input = legacy


if __name__ == "__main__":
    unittest.main()

"""Three-pass checks against the literal frozen data reader and radar builder."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image
import torch

from . import perception as P

ROOT = Path(__file__).resolve().parents[2]


def load_literal_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def original_data_module():
    """Load the complete untouched module; only its import resolution is scoped."""
    path = ROOT / P.REFERENCE_DATA_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == P.REFERENCE_DATA_SHA256
    common = load_literal_module(path.with_name("common.py"), "_parity_original_common")
    with mock.patch.dict(sys.modules, {"common": common}):
        return load_literal_module(path, "_parity_original_fcos_data")


class PerceptionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reference = original_data_module()
        cls.radar = load_literal_module(ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/radar_fusion.py", "_parity_original_radar")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self):
        rng = np.random.default_rng(4317)
        return (rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
                rng.uniform(-1, 1, (4, 432, 768)).astype(np.float32))

    def original_input(self, bgr, radar):
        with tempfile.TemporaryDirectory(prefix="fcos-parity-") as directory:
            root = Path(directory); (root / "dataset").mkdir()
            Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1])).save(root / "dataset/rgb.png")
            np.save(root / "dataset/radar.npy", radar)
            row = {"sample_id": "synthetic", "frame_id": "1", "rgb_path": "rgb.png",
                   "radar_tensor_path": "radar.npy", "camera_width": "1280", "camera_height": "720",
                   "camera_fx": str(P.CONTRACT.native_fx), "camera_fy": str(P.CONTRACT.native_fy),
                   "camera_cx": "640", "camera_cy": "360", "camera_matrix_json": json.dumps(np.eye(4).tolist())}
            # The original constructor's dataset size gate is irrelevant to a
            # one-frame numerical fixture. Reader, normalization, concatenation,
            # padding and calibration methods are all untouched original code.
            with mock.patch.object(self.reference, "load_split_rows", return_value=[row]):
                dataset = self.reference.InferenceDataset(root)
            tensor, _row, calibration = dataset[0]
            return tensor.unsqueeze(0), calibration

    def test_pass1_source_pins(self):
        self.assertEqual(len(P.verify_reference_sources(ROOT)), 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_pass1_same_input_literal_reference(self):
        bgr, radar = self.fixture()
        patterns = [bgr, np.zeros_like(bgr), np.full_like(bgr, 255), bgr[:, ::-1],
                    np.broadcast_to(np.array([0, 127, 255], np.uint8), bgr.shape)]
        for pixels in patterns:
            with self.subTest(kind=str(pixels.strides)):
                expected, calibration = self.original_input(pixels, radar)
                actual = P.prepare_live_input(pixels, radar, "cpu")
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (1, 7, 448, 768))
                self.assertEqual(int(torch.count_nonzero(actual[..., -16:, :])), 0)
                P.require_edge_intrinsic_parity(calibration["intrinsic"])

    def test_pass1_inputs_are_not_mutated(self):
        bgr, radar = self.fixture(); a, b = bgr.copy(), radar.copy()
        P.prepare_live_input(bgr, radar, "cpu")
        np.testing.assert_array_equal(bgr, a); np.testing.assert_array_equal(radar, b)

    def test_pass2_bad_rgb_rejected(self):
        bgr, radar = self.fixture()
        for bad in [bgr.astype(np.float32), bgr[..., :2], bgr[:432, :768], bgr.tolist(), None]:
            with self.subTest(kind=type(bad)):
                with self.assertRaises(P.PerceptionParityError): P.prepare_live_input(bad, radar, "cpu")

    def test_pass2_bad_radar_rejected_no_silent_resize(self):
        bgr, radar = self.fixture()
        nan = radar.copy(); nan[0, 0, 0] = np.nan
        inf = radar.copy(); inf[1, 0, 0] = np.inf
        for bad in [radar.astype(np.float64), radar[:3], np.zeros((4,448,768),np.float32),
                    radar.transpose(1,2,0), nan, inf, None]:
            with self.subTest(kind=getattr(bad, "shape", None)):
                with self.assertRaises(P.PerceptionParityError): P.prepare_live_input(bgr, bad, "cpu")

    def test_pass2_padding_after_normalization_not_before(self):
        bgr, radar = self.fixture()
        actual = P.prepare_live_input(np.zeros_like(bgr), np.zeros_like(radar), "cpu")
        self.assertGreater(int(torch.count_nonzero(actual[:, :3, :432, :])), 0)
        self.assertEqual(int(torch.count_nonzero(actual[:, :, 432:, :])), 0)

    def test_pass2_calibration_448_and_nonfinite_rejected(self):
        wrong = P.content_intrinsics().copy(); wrong[1,2] = 224
        with self.assertRaises(P.PerceptionParityError): P.require_edge_intrinsic_parity(wrong)
        wrong[1,2] = np.nan
        with self.assertRaises(P.PerceptionParityError): P.require_edge_intrinsic_parity(wrong)

    def collector(self):
        k = P.content_intrinsics().copy(); k[1,2] = 224
        return SimpleNamespace(sensors=[], model_size=(768,448), intrinsics=k)

    def test_pass3_binding_before_workers_and_no_state_changes(self):
        obj = self.collector(); tracker = object(); obj.tracker = tracker
        report = P.bind_collector_geometry(obj)
        self.assertEqual(obj.model_size, (768,432)); self.assertEqual(obj.intrinsics[1,2],216)
        self.assertIs(obj.tracker, tracker); self.assertFalse(report["scene_descriptors_modified"])
        with self.assertRaises(P.PerceptionParityError): P.bind_collector_geometry(obj)
        late = self.collector(); late.worker = object()
        with self.assertRaises(P.PerceptionParityError): P.bind_collector_geometry(late)
        late = self.collector(); late.sensors = [object()]
        with self.assertRaises(P.PerceptionParityError): P.bind_collector_geometry(late)

    def test_pass3_radar_original_source_projection_and_rasterization(self):
        # Execute the literal original collection camera function, not a copy
        # of the repair's matrix arithmetic.
        source = ROOT / "carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py"
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="intrinsics_at")
        ns = {"np":np,"math":math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),"exec"),ns)
        reference_k = ns["intrinsics_at"](768,432,120)
        np.testing.assert_array_equal(reference_k,P.radar_projection_intrinsics())
        np.testing.assert_array_equal(reference_k.astype(np.float32),P.content_intrinsics(dtype=np.float32))
        obj = self.collector(); P.bind_collector_geometry(obj)
        detections = np.array([[0,0,10,0],[.1,.15,25,-2],[-.15,-.3,40,3]],np.float32)
        common = dict(detections=detections,sensor_matrix=np.eye(4),camera_inverse_matrix=np.eye(4),
                      frame_time_s=1.,max_range_m=120.,max_abs_velocity_mps=20.,parked_threshold_s=5.,
                      point_radius_px=4,rasterizer="fast")
        ref = self.radar.build_radar_sample(**common, camera_intrinsics=reference_k,width=768,height=432,
                                         tracker=self.radar.StationaryTrackAccumulator())
        actual = self.radar.build_radar_sample(**common,camera_intrinsics=obj.intrinsics,
            width=obj.model_size[0],height=obj.model_size[1],tracker=self.radar.StationaryTrackAccumulator())
        np.testing.assert_array_equal(actual[0],ref[0])
        for key in ref[1]: np.testing.assert_array_equal(actual[1][key],ref[1][key])
        self.assertEqual(actual[2],ref[2]); self.assertEqual(float(actual[1]["v"][0]),216.)

    def test_pass3_si_contract_and_input_unchanged(self):
        from rl_agent.splitfusion_hybrid_sac_v1.scene_descriptors import camera_spatial_information
        bgr, radar = self.fixture()
        def si():
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            return camera_spatial_information(cv2.cvtColor(cv2.resize(rgb,(768,448)),cv2.COLOR_RGB2GRAY))
        before=si(); P.prepare_live_input(bgr,radar,"cpu"); self.assertEqual(si(),before)
        self.assertIn("768x448",P.CONTRACT.scene_si_semantics)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

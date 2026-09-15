#!/usr/bin/env python3
"""End-to-end equivalence of the wired optimized path.

This exercises the same seams the live cell installs -- the profiled radar
builder with the CUDA rasterizer override and the single-sort tracker, and the
seven-channel packing with the identity-resize short circuit -- against the
unmodified production functions over a simulated multi-frame episode.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as live_base
from rl_agent.splitfusion_sensor_optimization_v1.optimized_stages import (
    CudaRadarRasterizer,
    SingleSortStationaryTrackAccumulator,
)
from rl_agent.splitfusion_supervisor_analysis_v1.profiled_sensor_stages import (
    build_radar_sample_profiled,
    prepare_live_input_profiled,
)

WIDTH, HEIGHT = 768, 448
MAX_RANGE_M, MAX_ABS_VELOCITY_MPS, PARKED_S, RADIUS = 120.0, 20.0, 5.0, 4


def _pose(t: float) -> np.ndarray:
    yaw = 0.15 * t
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0, 8.0 * t], [s, c, 0.0, 0.4 * t],
                     [0.0, 0.0, 1.0, 1.6], [0.0, 0.0, 0.0, 1.0]])


def _intrinsics() -> np.ndarray:
    f = WIDTH / (2.0 * math.tan(math.radians(120.0) / 2.0))
    return np.array([[f, 0.0, WIDTH / 2.0], [0.0, f, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def _detections(rng: np.random.Generator, count: int) -> np.ndarray:
    """A CARLA-shaped [altitude, azimuth, depth, velocity] window."""
    altitude = np.deg2rad(rng.uniform(-15.0, 15.0, count))
    azimuth = np.deg2rad(rng.uniform(-60.0, 60.0, count))
    depth = np.clip(rng.gamma(3.0, 12.0, count), 0.5, MAX_RANGE_M)
    velocity = np.where(rng.random(count) < 0.12, rng.normal(0.0, 6.0, count),
                        rng.normal(0.0, 0.08, count))
    return np.stack([altitude, azimuth, depth, velocity], axis=1).astype(np.float32)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class TestWiredPathEquivalence(unittest.TestCase):
    def test_radar_sample_episode_is_bit_exact(self) -> None:
        device = torch.device("cuda:0")
        override = CudaRadarRasterizer(device)
        optimized_tracker = SingleSortStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0)
        reference_tracker = live_base.FastStationaryTrackAccumulator(0.35, PARKED_S, 1.5, 2.0)
        intrinsics = _intrinsics()
        rng = np.random.default_rng(20260914)
        for step in range(25):
            detections = _detections(rng, int(rng.choice([0, 12, 20000, 40000])))
            sensor_matrix = _pose(0.1 * step)
            camera_inverse = np.linalg.inv(_pose(0.1 * step))
            common = dict(
                detections=detections, sensor_matrix=sensor_matrix,
                camera_inverse_matrix=camera_inverse, camera_intrinsics=intrinsics,
                width=WIDTH, height=HEIGHT, frame_time_s=0.1 * step,
                max_range_m=MAX_RANGE_M, max_abs_velocity_mps=MAX_ABS_VELOCITY_MPS,
                parked_threshold_s=PARKED_S, point_radius_px=RADIUS,
            )
            actual_tensor, actual_points, actual_summary = build_radar_sample_profiled(
                tracker=optimized_tracker, rasterizer="fast",
                rasterizer_override=override, **common
            )
            expected_tensor, expected_points, expected_summary = radar_fusion.build_radar_sample(
                tracker=reference_tracker, rasterizer="fast", **common
            )
            self.assertTrue(np.array_equal(expected_tensor, actual_tensor), f"tensor differs at step {step}")
            self.assertEqual(expected_tensor.dtype, actual_tensor.dtype)
            self.assertEqual(expected_tensor.shape, actual_tensor.shape)
            self.assertEqual(set(expected_points), set(actual_points))
            for name, expected in expected_points.items():
                self.assertTrue(np.array_equal(expected, actual_points[name]), f"evidence {name} differs at step {step}")
                self.assertEqual(expected.dtype, actual_points[name].dtype)
            for name, expected in expected_summary.items():
                self.assertEqual(expected, actual_summary[name], f"summary {name} differs at step {step}")

    def test_seven_channel_packing_is_bit_exact(self) -> None:
        device = torch.device("cuda:0")
        rng = np.random.default_rng(7)
        constants = (
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )
        for _ in range(6):
            frame_bgr = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            radar_tensor = (rng.random((4, HEIGHT, WIDTH)) * 2.0 - 1.0).astype(np.float32)
            reference = live_base._prepare_live_input(frame_bgr, radar_tensor, device)
            optimized, _ = prepare_live_input_profiled(
                frame_bgr, radar_tensor, device,
                normalization_constants=constants,
                skip_identity_radar_resize=True,
            )
            self.assertTrue(torch.equal(reference, optimized))
            self.assertEqual(reference.shape, optimized.shape)
            self.assertEqual(reference.dtype, optimized.dtype)
            self.assertEqual(int(optimized.shape[1]), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

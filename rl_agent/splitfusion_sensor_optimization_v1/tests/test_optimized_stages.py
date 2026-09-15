#!/usr/bin/env python3
"""Deterministic bit-exactness gates for the sensor-preparation optimizations.

Every assertion is exact equality against the production implementation.  No
tolerance is used anywhere in this file; an optimization that cannot reproduce
the production bit pattern is a failure, not a rounding difference.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion import radar_fusion
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    FastStationaryTrackAccumulator,
)
from rl_agent.splitfusion_sensor_optimization_v1.optimized_stages import (
    CudaRadarRasterizer,
    SingleSortStationaryTrackAccumulator,
    radar_channels_already_sized,
)

WIDTH, HEIGHT = 768, 448


def _rasterizer_case(rng: np.random.Generator, count: int, radius: int) -> dict:
    if count == 0:
        empty = np.zeros(0, dtype=np.float32)
        return dict(
            width=WIDTH, height=HEIGHT, u=empty, v=empty, depth_m=empty,
            velocity_mps=empty, stationary_age_s=empty,
            valid_mask=np.zeros(0, dtype=bool), max_range_m=120.0,
            max_abs_velocity_mps=20.0, parked_threshold_s=5.0,
            point_radius_px=radius,
        )
    mode = int(rng.integers(0, 4))
    if mode == 0:
        u = rng.uniform(-5.0, WIDTH + 5.0, count)
        v = rng.uniform(-5.0, HEIGHT + 5.0, count)
    elif mode == 1:
        u = np.rint(rng.uniform(0.0, WIDTH, count))
        v = np.rint(rng.uniform(0.0, HEIGHT, count))
    elif mode == 2:
        # Rounds to exactly width/height: the production border case.
        u = np.full(count, float(WIDTH) - 0.4)
        v = np.full(count, float(HEIGHT) - 0.4)
    else:
        u = np.zeros(count)
        v = np.zeros(count)
    depth = rng.gamma(3.0, 12.0, count).astype(np.float32)
    depth[rng.random(count) < 0.02] = np.nan
    depth[rng.random(count) < 0.02] = np.inf
    return dict(
        width=WIDTH, height=HEIGHT,
        u=u.astype(np.float32), v=v.astype(np.float32), depth_m=depth,
        # Exact +/- ties exercise the signed-velocity recombination.
        velocity_mps=rng.choice([-20.0, -6.0, -0.1, 0.0, 0.1, 6.0, 20.0], count).astype(np.float32),
        stationary_age_s=rng.choice([0.0, 2.5, 5.0, 15.0], count).astype(np.float32),
        valid_mask=rng.random(count) > 0.05,
        max_range_m=120.0, max_abs_velocity_mps=20.0, parked_threshold_s=5.0,
        point_radius_px=radius,
    )


class TestSingleSortStationaryTracker(unittest.TestCase):
    """The tracker must match ages *and* the retained track table, every frame."""

    def _episode(self, seed: int, frames: int = 40) -> None:
        rng = np.random.default_rng(seed)
        reference = FastStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
        candidate = SingleSortStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
        for step in range(frames):
            count = int(rng.choice([0, 1, 3, 200, 5000, 40000]))
            if count == 0:
                points = np.zeros((0, 4))
            else:
                # Collide many returns into few cells so within-cell ordering,
                # the first-moving reset and the age carry are all exercised.
                x = rng.integers(-40, 40, count) * 1.5 + rng.normal(0.0, 0.3, count)
                y = rng.integers(-40, 40, count) * 1.5 + rng.normal(0.0, 0.3, count)
                # 0.35 sits exactly on the stationary threshold boundary.
                speed = rng.choice([0.0, 0.1, 0.34, 0.35, 0.36, 3.0, -5.0], count)
                points = np.stack([x, y, np.full(count, 1.6), speed], axis=1)
            expected = reference.update(points, 0.1 * step)
            actual = candidate.update(points, 0.1 * step)
            self.assertTrue(np.array_equal(expected, actual), f"ages differ at seed {seed} step {step}")
            self.assertEqual(expected.dtype, actual.dtype)
            for name in ("_keys", "_ages", "_last_seen", "_x", "_y"):
                self.assertTrue(
                    np.array_equal(getattr(reference, name), getattr(candidate, name)),
                    f"track state {name} differs at seed {seed} step {step}",
                )

    def test_randomized_episodes_are_bit_exact(self) -> None:
        for seed in range(12):
            self._episode(seed)

    def test_stale_eviction_matches(self) -> None:
        reference = FastStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
        candidate = SingleSortStationaryTrackAccumulator(0.35, 5.0, 1.5, 2.0)
        rng = np.random.default_rng(1234)
        here = np.stack([
            rng.integers(0, 5, 400) * 1.5, rng.integers(0, 5, 400) * 1.5,
            np.full(400, 1.6), np.zeros(400),
        ], axis=1)
        far = np.stack([
            rng.integers(400, 405, 400) * 1.5, rng.integers(400, 405, 400) * 1.5,
            np.full(400, 1.6), np.zeros(400),
        ], axis=1)
        for step, points in enumerate([here] * 10 + [far] * 40 + [here] * 10):
            expected = reference.update(points, 0.5 * step)
            actual = candidate.update(points, 0.5 * step)
            self.assertTrue(np.array_equal(expected, actual))
            self.assertTrue(np.array_equal(reference._keys, candidate._keys))
            self.assertTrue(np.array_equal(reference._ages, candidate._ages))


@unittest.skipUnless(
    __import__("torch").cuda.is_available(), "CUDA rasterizer requires a CUDA device"
)
class TestCudaRadarRasterizer(unittest.TestCase):
    """The CUDA rasterizer must reproduce the production tensor bit for bit."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rasterizer = CudaRadarRasterizer("cuda:0")

    def test_matches_production_across_configurations(self) -> None:
        for seed in range(60):
            rng = np.random.default_rng(seed)
            count = int(rng.choice([0, 1, 7, 500, 20000, 40000, 60000]))
            radius = int(rng.choice([0, 1, 2, 4, 6]))
            case = _rasterizer_case(rng, count, radius)
            expected = radar_fusion.rasterize_radar_channels_fast(**case)
            actual = self.rasterizer(**case)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(
                np.array_equal(expected, actual),
                f"seed {seed} count {count} radius {radius}: "
                f"{int(np.sum(expected != actual))} differing elements",
            )

    def test_production_radius_and_density(self) -> None:
        """The deployed configuration: radius 4, ~40k returns, 768x448."""
        rng = np.random.default_rng(2026)
        for _ in range(10):
            case = _rasterizer_case(rng, 40000, 4)
            expected = radar_fusion.rasterize_radar_channels_fast(**case)
            self.assertTrue(np.array_equal(expected, self.rasterizer(**case)))

    def test_all_points_rejected(self) -> None:
        case = _rasterizer_case(np.random.default_rng(0), 100, 4)
        case["valid_mask"] = np.zeros(100, dtype=bool)
        expected = radar_fusion.rasterize_radar_channels_fast(**case)
        self.assertTrue(np.array_equal(expected, self.rasterizer(**case)))
        self.assertTrue(np.array_equal(expected, np.zeros((4, HEIGHT, WIDTH), np.float32)))

    def test_rejects_non_cuda_device(self) -> None:
        with self.assertRaises(ValueError):
            CudaRadarRasterizer("cpu")


class TestIdentityResizeGuard(unittest.TestCase):
    def test_identity_resize_is_bit_exact(self) -> None:
        import cv2

        rng = np.random.default_rng(5)
        tensor = (rng.random((4, HEIGHT, WIDTH)).astype(np.float32) * 3.0)
        self.assertTrue(radar_channels_already_sized(tensor, WIDTH, HEIGHT))
        for index, channel in enumerate(tensor):
            interpolation = cv2.INTER_NEAREST if index == 0 else cv2.INTER_LINEAR
            resized = cv2.resize(channel, (WIDTH, HEIGHT), interpolation=interpolation)
            self.assertTrue(np.array_equal(resized, channel))

    def test_rejects_mismatched_shape(self) -> None:
        self.assertFalse(
            radar_channels_already_sized(np.zeros((4, 100, 200), np.float32), WIDTH, HEIGHT)
        )
        self.assertFalse(radar_channels_already_sized(np.zeros((HEIGHT, WIDTH), np.float32), WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main(verbosity=2)

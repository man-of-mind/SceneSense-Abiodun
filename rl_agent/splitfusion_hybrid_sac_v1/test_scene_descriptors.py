"""Exact, fail-closed tests for the Phase-3a SI/P40 contract."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unittest
from types import MappingProxyType

import numpy as np

from . import scene_descriptors as sd


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _independent_sobel_si(luma: np.ndarray) -> float:
    """Pure-NumPy reference using the explicit 3x3 Sobel stencils."""

    source = luma.astype(np.float32)
    # np.pad(mode="reflect") matches OpenCV BORDER_REFLECT_101, which is
    # BORDER_DEFAULT for this Sobel operation.
    padded = np.pad(source, 1, mode="reflect")
    left = padded[:-2, :-2] + 2.0 * padded[1:-1, :-2] + padded[2:, :-2]
    right = padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
    top = padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
    bottom = padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    sobel_x = right - left
    sobel_y = bottom - top
    return float(np.std(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)))


class SceneDescriptorsTest(unittest.TestCase):
    def test_schema_is_immutable_and_hash_bound(self) -> None:
        self.assertEqual(
            sd.SCHEMA_ID, "splitfusion_hybrid_sac_scene_descriptors_v1"
        )
        self.assertEqual(sd.SCHEMA_VERSION, 1)
        self.assertIsInstance(sd.SCHEMA_DESCRIPTOR, MappingProxyType)
        with self.assertRaises(TypeError):
            sd.SCHEMA_DESCRIPTOR["version"] = 2  # type: ignore[index]
        with self.assertRaises(TypeError):
            sd.SCHEMA_DESCRIPTOR["camera_si"]["dtype"] = "float32"  # type: ignore[index]

        canonical = json.dumps(
            _thaw(sd.SCHEMA_DESCRIPTOR),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), sd.SCHEMA_SHA256)
        self.assertEqual(
            sd.SCHEMA_SHA256,
            "1bd469559f185c1faaafd254b6e742084ab6e5f66a594366b854799938b122b2",
        )

    def test_si_matches_independent_sobel_reference_exactly(self) -> None:
        yy, xx = np.indices(
            (sd.CAMERA_LUMA_HEIGHT, sd.CAMERA_LUMA_WIDTH), dtype=np.int32
        )
        luma = ((13 * xx + 29 * yy + ((xx // 17) % 2) * 91) % 256).astype(
            np.uint8
        )
        expected = _independent_sobel_si(luma)
        actual = sd.camera_spatial_information(luma)
        self.assertEqual(actual, expected)

        flat = np.full(luma.shape, 83, dtype=np.uint8)
        self.assertEqual(sd.camera_spatial_information(flat), 0.0)

    def test_si_rejects_wrong_type_shape_and_dtype(self) -> None:
        valid = np.zeros(
            (sd.CAMERA_LUMA_HEIGHT, sd.CAMERA_LUMA_WIDTH), dtype=np.uint8
        )
        for bad in (
            valid.tolist(),
            valid[:-1],
            valid[:, :, None],
            valid.astype(np.float32),
            valid.astype(np.uint16),
        ):
            with self.subTest(type=type(bad).__name__, shape=np.shape(bad)):
                with self.assertRaises(sd.InvalidCameraLumaError):
                    sd.camera_spatial_information(bad)

    def test_p40_matches_independent_reference_and_preserves_denominator(self) -> None:
        ranges = np.asarray([10.0, 20.0, 40.0, 80.0], dtype=np.float32)
        # Independent scalar computation: (0.75 + 0.50 + 0 + 0) / 4.
        self.assertEqual(sd.radar_proximity_p40(ranges), 0.3125)
        self.assertEqual(
            sd.radar_proximity_p40(np.asarray([40.0, 120.0], dtype=np.float64)),
            0.0,
        )
        near = np.asarray([1.0, 5.0, 10.0, 39.0, 80.0], dtype=np.float64)
        expected = sum(max(0.0, min(1.0, 1.0 - x / 40.0)) for x in near) / len(
            near
        )
        self.assertEqual(sd.radar_proximity_p40(near), expected)

    def test_radar_unavailable_is_distinct_and_never_zero_filled(self) -> None:
        for missing in (None, np.asarray([], dtype=np.float32)):
            with self.subTest(missing=missing):
                with self.assertRaises(sd.RadarUnavailableError):
                    sd.radar_proximity_p40(missing)

    def test_radar_malformed_values_fail_closed(self) -> None:
        malformed = (
            [10.0, 20.0],
            np.asarray([[10.0]], dtype=np.float32),
            np.asarray([10], dtype=np.int32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([math.nan], dtype=np.float32),
            np.asarray([math.inf], dtype=np.float64),
            np.asarray([120.0001], dtype=np.float64),
        )
        for ranges in malformed:
            with self.subTest(ranges=repr(ranges)):
                with self.assertRaises(sd.InvalidRadarRangesError):
                    sd.radar_proximity_p40(ranges)

    def test_combined_sample_is_immutable_schema_bound_and_validated(self) -> None:
        luma = np.zeros(
            (sd.CAMERA_LUMA_HEIGHT, sd.CAMERA_LUMA_WIDTH), dtype=np.uint8
        )
        sample = sd.compute_scene_descriptors(
            luma, np.asarray([10.0, 20.0, 80.0], dtype=np.float32)
        )
        self.assertEqual(sample.camera_si, 0.0)
        self.assertEqual(sample.radar_p40, (0.75 + 0.5 + 0.0) / 3.0)
        self.assertEqual(
            sample.as_record(),
            {
                "schema_id": sd.SCHEMA_ID,
                "schema_sha256": sd.SCHEMA_SHA256,
                "camera_si": 0.0,
                "radar_p40": (0.75 + 0.5 + 0.0) / 3.0,
            },
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            sample.camera_si = 1.0  # type: ignore[misc]

        invalid_values = (
            {"camera_si": -1.0, "radar_p40": 0.5},
            {"camera_si": math.nan, "radar_p40": 0.5},
            {"camera_si": 1.0, "radar_p40": -0.01},
            {"camera_si": 1.0, "radar_p40": 1.01},
            {"camera_si": 1.0, "radar_p40": math.inf},
            {"camera_si": True, "radar_p40": 0.5},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(sd.SceneDescriptorError):
                    sd.SceneDescriptorSample(**values)


if __name__ == "__main__":
    unittest.main()

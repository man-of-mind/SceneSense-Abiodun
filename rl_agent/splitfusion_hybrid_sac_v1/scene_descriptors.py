"""Pure in-memory camera/radar scene descriptors for Hybrid-SAC state v1.

This module freezes only the two production candidates selected for the first
policy state:

``camera_si``
    The standard deviation of the magnitude of the 3-by-3 Sobel gradients of
    an already-resized 768x448 uint8 luma image.

``radar_p40``
    ``mean(clip(1 - range_m / 40, 0, 1))`` over every valid return in the
    current, non-overlapping 100-ms radar sweep.  Returns beyond 40 m remain in
    the denominator and contribute zero.

There is deliberately no image decoding, resizing, RGB conversion, radar
window construction, file I/O, normalization for a neural policy, or fallback
action in this module.  Missing radar and malformed radar are different
fail-closed outcomes.  The caller's external runtime guard must select and log
the registered fallback action; neither case is silently encoded as P40=0.

The SI value is in luma-code gradient units and is *not* normalized to [0, 1].
Policy scaling is a later state-contract decision.  Importing this module does
not read a catalog or any other file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np

__all__ = [
    "SceneDescriptorError",
    "InvalidCameraLumaError",
    "RadarUnavailableError",
    "InvalidRadarRangesError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_DESCRIPTOR",
    "SCHEMA_SHA256",
    "CAMERA_LUMA_HEIGHT",
    "CAMERA_LUMA_WIDTH",
    "CAMERA_LUMA_DTYPE",
    "SOBEL_KERNEL_SIZE",
    "RADAR_SWEEP_DURATION_MS",
    "RADAR_SENSOR_RANGE_M",
    "P40_HORIZON_M",
    "SceneDescriptorSample",
    "camera_spatial_information",
    "radar_proximity_p40",
    "compute_scene_descriptors",
]


class SceneDescriptorError(ValueError):
    """Base class for scene-descriptor contract violations."""


class InvalidCameraLumaError(SceneDescriptorError):
    """The camera input does not satisfy the frozen luma contract."""


class RadarUnavailableError(SceneDescriptorError):
    """No current-sweep radar observation is available for P40.

    This is a runtime condition for the external radar-validity/fallback guard,
    not a numeric P40 value.
    """


class InvalidRadarRangesError(SceneDescriptorError):
    """Radar was present, but its range vector violates the frozen contract."""


SCHEMA_ID = "splitfusion_hybrid_sac_scene_descriptors_v1"
SCHEMA_VERSION = 1

CAMERA_LUMA_HEIGHT = 448
CAMERA_LUMA_WIDTH = 768
CAMERA_LUMA_DTYPE = "uint8"
SOBEL_KERNEL_SIZE = 3

RADAR_SWEEP_DURATION_MS = 100
RADAR_SENSOR_RANGE_M = 120.0
P40_HORIZON_M = 40.0


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


_SCHEMA_LITERAL = {
    "schema_id": SCHEMA_ID,
    "version": SCHEMA_VERSION,
    "scope": "pure in-memory scene descriptors; no policy scaling or fallback",
    "camera_si": {
        "input": {
            "semantic": "already-resized grayscale/luma",
            "shape_hw": [CAMERA_LUMA_HEIGHT, CAMERA_LUMA_WIDTH],
            "dtype": CAMERA_LUMA_DTYPE,
        },
        "definition": (
            "std(sqrt(Sobel_x(luma)^2 + Sobel_y(luma)^2)); "
            "OpenCV CV_32F, ksize=3, BORDER_DEFAULT"
        ),
        "normalization": "none; luma-code gradient units",
    },
    "radar_p40": {
        "input": {
            "semantic": "valid ranges from current non-overlapping radar sweep",
            "duration_ms": RADAR_SWEEP_DURATION_MS,
            "dtype": ["float32", "float64"],
            "shape": "non-empty one-dimensional vector",
            "valid_range_m": ["strictly greater than 0", RADAR_SENSOR_RANGE_M],
        },
        "definition": "mean(clip(1 - range_m / 40, 0, 1)) over all returns",
        "horizon_m": P40_HORIZON_M,
        "missingness": (
            "raise RadarUnavailableError; external guard chooses fallback; "
            "never substitute numeric zero"
        ),
        "invalidity": (
            "raise InvalidRadarRangesError; external guard chooses fallback; "
            "never discard invalid elements or substitute numeric zero"
        ),
    },
}

SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(_SCHEMA_LITERAL)
SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _thaw(SCHEMA_DESCRIPTOR),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _require_camera_luma(luma: Any) -> np.ndarray:
    if not isinstance(luma, np.ndarray):
        raise InvalidCameraLumaError(
            f"luma must be a numpy.ndarray, got {type(luma).__name__}"
        )
    expected_shape = (CAMERA_LUMA_HEIGHT, CAMERA_LUMA_WIDTH)
    if luma.shape != expected_shape:
        raise InvalidCameraLumaError(
            f"luma shape must be {expected_shape}, got {luma.shape}"
        )
    if luma.dtype != np.uint8:
        raise InvalidCameraLumaError(
            f"luma dtype must be uint8, got {luma.dtype}"
        )
    # Explicit even though uint8 is intrinsically finite: this makes the
    # fail-closed finite-value rule visible and robust if the dtype contract is
    # deliberately extended in a later schema.
    if not bool(np.all(np.isfinite(luma))):
        raise InvalidCameraLumaError("luma contains a non-finite value")
    return luma


def camera_spatial_information(luma: Any) -> float:
    """Compute registered Sobel-gradient dispersion from prepared luma.

    The caller supplies the already-resized 768x448 uint8 luma array.  This
    function performs no conversion, resizing, normalization or I/O.
    """

    checked = _require_camera_luma(luma)
    luma_f32 = checked.astype(np.float32, copy=False)
    sobel_x = cv2.Sobel(
        luma_f32,
        cv2.CV_32F,
        1,
        0,
        ksize=SOBEL_KERNEL_SIZE,
        borderType=cv2.BORDER_DEFAULT,
    )
    sobel_y = cv2.Sobel(
        luma_f32,
        cv2.CV_32F,
        0,
        1,
        ksize=SOBEL_KERNEL_SIZE,
        borderType=cv2.BORDER_DEFAULT,
    )
    magnitude = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    value = float(np.std(magnitude))
    if not math.isfinite(value) or value < 0.0:
        raise InvalidCameraLumaError(
            f"Sobel-gradient dispersion produced invalid value {value!r}"
        )
    return value


def _require_radar_ranges(range_m: Any) -> np.ndarray:
    if range_m is None:
        raise RadarUnavailableError("current 100-ms radar sweep is unavailable")
    if not isinstance(range_m, np.ndarray):
        raise InvalidRadarRangesError(
            f"range_m must be a numpy.ndarray, got {type(range_m).__name__}"
        )
    if range_m.ndim != 1:
        raise InvalidRadarRangesError(
            f"range_m must be one-dimensional, got shape {range_m.shape}"
        )
    if range_m.size == 0:
        raise RadarUnavailableError("current 100-ms radar sweep has no returns")
    if range_m.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise InvalidRadarRangesError(
            f"range_m dtype must be float32 or float64, got {range_m.dtype}"
        )

    ranges_f64 = range_m.astype(np.float64, copy=False)
    if not bool(np.all(np.isfinite(ranges_f64))):
        raise InvalidRadarRangesError("range_m contains NaN or infinity")
    if not bool(np.all(ranges_f64 > 0.0)):
        raise InvalidRadarRangesError("every range_m value must be greater than 0")
    if not bool(np.all(ranges_f64 <= RADAR_SENSOR_RANGE_M)):
        raise InvalidRadarRangesError(
            f"range_m exceeds the registered {RADAR_SENSOR_RANGE_M:g}-m sensor range"
        )
    return ranges_f64


def radar_proximity_p40(range_m: Any) -> float:
    """Compute range-weighted support inside the 40-m task horizon.

    The denominator is the number of *all* valid current-sweep returns.  A
    return at 10 m contributes 0.75; at 20 m, 0.5; and at or beyond 40 m,
    zero.  This is aggregate radar support, not an object count or detector.
    """

    ranges_f64 = _require_radar_ranges(range_m)
    contributions = np.clip(1.0 - ranges_f64 / P40_HORIZON_M, 0.0, 1.0)
    value = float(np.mean(contributions, dtype=np.float64))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InvalidRadarRangesError(f"P40 produced invalid value {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SceneDescriptorSample:
    """One immutable, valid SI/P40 observation for the policy-state builder."""

    camera_si: float
    radar_p40: float

    def __post_init__(self) -> None:
        for name, value in (
            ("camera_si", self.camera_si),
            ("radar_p40", self.radar_p40),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise SceneDescriptorError(
                    f"{name} must be a finite real scalar, got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise SceneDescriptorError(f"{name} must be finite, got {value!r}")
        if float(self.camera_si) < 0.0:
            raise SceneDescriptorError("camera_si cannot be negative")
        if not 0.0 <= float(self.radar_p40) <= 1.0:
            raise SceneDescriptorError("radar_p40 must lie in [0, 1]")

    def as_record(self) -> dict[str, Any]:
        """Return the canonical schema-bound primitive record."""

        return {
            "schema_id": SCHEMA_ID,
            "schema_sha256": SCHEMA_SHA256,
            "camera_si": float(self.camera_si),
            "radar_p40": float(self.radar_p40),
        }


def compute_scene_descriptors(
    luma: Any,
    current_sweep_range_m: Any,
) -> SceneDescriptorSample:
    """Compute a valid SI/P40 pair or fail closed for the external guard."""

    return SceneDescriptorSample(
        camera_si=camera_spatial_information(luma),
        radar_p40=radar_proximity_p40(current_sweep_range_m),
    )

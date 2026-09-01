"""Small CPU primitives for the registered z-buffer visibility equation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CARLA_MAX_DEPTH_M = 1000.0
CARLA_DEPTH_DENOMINATOR = 16_777_215
TAU_EMPTY_M = 0.02
TAU_MATCH_M = 0.02
TRANSFORM_TOLERANCE = 1e-4
WALKER_BONE_TOLERANCE = 1e-3


class ZBufferVisibilityError(RuntimeError):
    """Fail-closed z-buffer visibility error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_bgra(image: Any) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected = int(image.width) * int(image.height) * 4
    if raw.size != expected:
        raise ZBufferVisibilityError(f"image byte count {raw.size} != {expected}")
    return raw.reshape((int(image.height), int(image.width), 4)).copy()


def decode_depth_bgra(raw_bgra: np.ndarray) -> np.ndarray:
    """Decode CARLA's lossless 24-bit BGRA depth buffer into float64 metres."""
    raw = np.asarray(raw_bgra)
    if raw.ndim != 3 or raw.shape[2] != 4 or raw.dtype != np.uint8:
        raise ZBufferVisibilityError(
            f"depth image must be uint8 HxWx4 BGRA, got {raw.dtype} {raw.shape}"
        )
    code = raw[:, :, 2].astype(np.uint32)
    code += raw[:, :, 1].astype(np.uint32) << np.uint32(8)
    code += raw[:, :, 0].astype(np.uint32) << np.uint32(16)
    depth = code.astype(np.float64) * (CARLA_MAX_DEPTH_M / CARLA_DEPTH_DENOMINATOR)
    if not np.all(np.isfinite(depth)):
        raise ZBufferVisibilityError("decoded depth contains non-finite values")
    return depth


def _depth(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ZBufferVisibilityError(f"{label} must be a finite HxW depth array")
    return array


def compute_zbuffer_visibility(
    empty_depth_m: np.ndarray,
    actor_depth_m: np.ndarray,
    scene_depth_m: np.ndarray,
    *,
    tau_empty_m: float = TAU_EMPTY_M,
    tau_match_m: float = TAU_MATCH_M,
) -> dict[str, Any]:
    """Apply the exact registered A_i and V_i equations."""
    empty = _depth(empty_depth_m, "D_empty")
    actor = _depth(actor_depth_m, "D_actor")
    scene = _depth(scene_depth_m, "D_scene")
    if empty.shape != actor.shape or actor.shape != scene.shape:
        raise ZBufferVisibilityError(
            f"depth shape mismatch: {empty.shape}, {actor.shape}, {scene.shape}"
        )
    if float(tau_empty_m) != TAU_EMPTY_M or float(tau_match_m) != TAU_MATCH_M:
        raise ZBufferVisibilityError("registered 0.02 m tolerances may not be changed")
    support = actor + np.float64(TAU_EMPTY_M) < empty
    difference = np.abs(scene - actor)
    visible = support & (difference <= np.float64(TAU_MATCH_M))
    support_pixels = int(np.count_nonzero(support))
    if support_pixels <= 0:
        raise ZBufferVisibilityError("actor-only depth produced zero A_i pixels")
    visible_pixels = int(np.count_nonzero(visible))
    visibility = visible_pixels / support_pixels
    if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
        raise ZBufferVisibilityError(f"invalid visibility {visibility}")
    return {
        "support": support,
        "visible": visible,
        "depth_difference_m": difference,
        "support_pixels": support_pixels,
        "visible_pixels": visible_pixels,
        "visibility": float(visibility),
    }


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ZBufferVisibilityError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        raise ZBufferVisibilityError("mask IoU union is empty")
    value = int(np.count_nonzero(a & b)) / union
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ZBufferVisibilityError(f"invalid mask IoU {value}")
    return float(value)


def transform_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value.get_matrix(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ZBufferVisibilityError("invalid transform matrix")
    return matrix


def inverse_transform_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value.get_inverse_matrix(), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ZBufferVisibilityError("invalid inverse transform matrix")
    return matrix


def relative_transform_matrix(camera_transform: Any, actor_transform: Any) -> np.ndarray:
    return inverse_transform_matrix(camera_transform) @ transform_matrix(actor_transform)


def reproduce_transform_matrix(
    reference_camera_transform: Any, camera_relative_actor_matrix: np.ndarray
) -> np.ndarray:
    relative = np.asarray(camera_relative_actor_matrix, dtype=np.float64)
    if relative.shape != (4, 4) or not np.all(np.isfinite(relative)):
        raise ZBufferVisibilityError("invalid camera-relative actor matrix")
    return transform_matrix(reference_camera_transform) @ relative


def matrix_to_transform_payload(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ZBufferVisibilityError("invalid 4x4 transform")
    pitch = math.asin(max(-1.0, min(1.0, float(value[2, 0]))))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        yaw = math.atan2(float(value[1, 0]), float(value[0, 0]))
        roll = math.atan2(float(-value[2, 1]), float(value[2, 2]))
    else:
        yaw = math.atan2(float(-value[0, 1]), float(value[1, 1]))
        roll = 0.0
    return {
        "location": {
            "x": float(value[0, 3]),
            "y": float(value[1, 3]),
            "z": float(value[2, 3]),
        },
        "rotation": {
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
            "roll": math.degrees(roll),
        },
    }


def transform_payload(value: Any) -> dict[str, dict[str, float]]:
    return {
        "location": {axis: float(getattr(value.location, axis)) for axis in "xyz"},
        "rotation": {
            key: float(getattr(value.rotation, key)) for key in ("pitch", "yaw", "roll")
        },
    }


def write_json_x(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return sha256(path)


def write_png_x(path: Path, value: np.ndarray) -> str:
    import cv2

    array = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), array, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise ZBufferVisibilityError(f"failed to write {path}")
    recovered = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if recovered is None or not np.array_equal(recovered, array):
        raise ZBufferVisibilityError(f"PNG did not round-trip losslessly: {path}")
    return sha256(path)


def write_npy_x(path: Path, value: np.ndarray) -> str:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ZBufferVisibilityError(f"cannot persist non-finite array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    recovered = np.load(path, allow_pickle=False)
    if not np.array_equal(recovered, array):
        raise ZBufferVisibilityError(f"NPY did not round-trip losslessly: {path}")
    return sha256(path)

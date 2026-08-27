"""Frozen depth-visibility contract used by the Route B v3 collector.

This module is deliberately CARLA-independent so the projection, depth decode,
eligibility, and tier rules can be checked on tiny synthetic arrays before a
server is started.  It never consults semantic or instance walker tags and has
no low-support fallback.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


ALGORITHM_VERSION = "route_b_depth_visibility_interval_v1"
DEPTH_ENCODING = "CARLA BGRA uint8 encoded-depth, lossless PNG"
CARLA_MAX_DEPTH_M = 1000.0
DEPTH_TOLERANCE_M = 0.25
MAX_DISTANCE_M = 40.0
MIN_PROJECTED_AREA_PX = 12.0
MIN_MODEL_VISIBLE_PX = 12
VISIBLE_THRESHOLD = 0.10
CLEAR_THRESHOLD = 0.25

TIER_CLEAR = "clear"
TIER_MARGINAL = "marginal_or_heavily_occluded"
TIER_UNOBSERVABLE = "unobservable"


def decode_depth_bgra(raw_bgra: np.ndarray) -> np.ndarray:
    """Decode CARLA's 24-bit depth value from a BGRA uint8 image to metres."""
    raw = np.asarray(raw_bgra)
    if raw.ndim != 3 or raw.shape[2] != 4 or raw.dtype != np.uint8:
        raise ValueError(f"expected HxWx4 uint8 BGRA depth, got {raw.shape}/{raw.dtype}")
    values = raw.astype(np.float32, copy=False)
    blue, green, red = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    normalized = (
        red + green * 256.0 + blue * 256.0 * 256.0
    ) / (256.0 ** 3 - 1.0)
    return (CARLA_MAX_DEPTH_M * normalized).astype(np.float32, copy=False)


def depth_image_bgra(depth_image: Any) -> np.ndarray:
    """Return an owning HxWx4 BGRA array from a CARLA depth measurement."""
    return np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(
        int(depth_image.height), int(depth_image.width), 4
    ).copy()


def depth_is_plausible(depth_m: np.ndarray) -> bool:
    values = np.asarray(depth_m)
    return bool(
        values.ndim == 2
        and values.size > 0
        and np.all(np.isfinite(values))
        and float(values.min()) > 0.0
        and float(values.max()) <= CARLA_MAX_DEPTH_M
    )


def _nearest_model_count(mask: np.ndarray, model_width: int, model_height: int) -> int:
    height, width = mask.shape
    ys = np.floor(np.arange(model_height) * (height / float(model_height))).astype(np.int64)
    xs = np.floor(np.arange(model_width) * (width / float(model_width))).astype(np.int64)
    ys = np.clip(ys, 0, height - 1)
    xs = np.clip(xs, 0, width - 1)
    return int(np.count_nonzero(mask[np.ix_(ys, xs)]))


def visibility_tier(visible_fraction: float, model_visible_px: int) -> str:
    if int(model_visible_px) < MIN_MODEL_VISIBLE_PX or float(visible_fraction) < VISIBLE_THRESHOLD:
        return TIER_UNOBSERVABLE
    if float(visible_fraction) < CLEAR_THRESHOLD:
        return TIER_MARGINAL
    return TIER_CLEAR


def eligibility_flags(
    *, distance_m: float, projected_area_px: float,
    model_visible_px: int, visible_fraction: float,
) -> tuple[bool, bool]:
    geometry = (
        float(distance_m) <= MAX_DISTANCE_M
        and float(projected_area_px) >= MIN_PROJECTED_AREA_PX
    )
    pixel_support = int(model_visible_px) >= MIN_MODEL_VISIBLE_PX
    return (
        bool(geometry and pixel_support and float(visible_fraction) >= VISIBLE_THRESHOLD),
        bool(geometry and pixel_support and float(visible_fraction) >= CLEAR_THRESHOLD),
    )


def visibility_from_corners(
    corners_world: np.ndarray,
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    depth_m: np.ndarray,
    *,
    distance_m: float,
    width: int,
    height: int,
    model_width: int,
    model_height: int,
    tolerance_m: float = DEPTH_TOLERANCE_M,
) -> tuple[dict[str, Any], np.ndarray]:
    """Project eight AABB corners and return metrics plus a full-res visible mask.

    The returned mask contains only pixels inside the clipped projected box
    whose measured planar depth is within the actor's corner-derived near/far
    interval plus the fixed symmetric tolerance.  There is intentionally no
    fallback to a filled box or ellipse.
    """
    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"expected exactly eight actor AABB corners, got {corners.shape}")
    if np.asarray(depth_m).shape != (int(height), int(width)):
        raise ValueError(
            f"depth shape {np.asarray(depth_m).shape} != {(int(height), int(width))}"
        )
    homogeneous = np.concatenate([corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    camera = (np.asarray(camera_inverse, dtype=np.float64) @ homogeneous.T).T[:, :3]
    forward = camera[:, 0]
    in_front = forward > 0.05
    if not np.any(in_front):
        raise ValueError("actor AABB has no corner in front of camera")
    x = forward[in_front]
    y = camera[in_front, 1]
    z = camera[in_front, 2]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    u = matrix[0, 2] + (y / x) * matrix[0, 0]
    v = matrix[1, 2] - (z / x) * matrix[1, 1]

    ux0, uy0 = float(np.min(u)), float(np.min(v))
    ux1, uy1 = float(np.max(u)), float(np.max(v))
    unclipped_w, unclipped_h = max(0.0, ux1 - ux0), max(0.0, uy1 - uy0)
    unclipped_area = unclipped_w * unclipped_h
    cx0, cy0 = float(np.clip(ux0, 0.0, width)), float(np.clip(uy0, 0.0, height))
    cx1, cy1 = float(np.clip(ux1, 0.0, width)), float(np.clip(uy1, 0.0, height))
    clipped_w, clipped_h = max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)
    clipped_area = clipped_w * clipped_h
    if clipped_area <= 0.0:
        raise ValueError("actor projected box has no in-frame area")

    col0, row0 = max(0, int(math.floor(cx0))), max(0, int(math.floor(cy0)))
    col1 = min(int(width), max(col0 + 1, int(math.ceil(cx1))))
    row1 = min(int(height), max(row0 + 1, int(math.ceil(cy1))))
    roi = np.asarray(depth_m)[row0:row1, col0:col1]
    if roi.size == 0:
        raise ValueError("actor projected box produced an empty sampled ROI")

    near_m, far_m = float(np.min(x)), float(np.max(x))
    lower, upper = near_m - float(tolerance_m), far_m + float(tolerance_m)
    finite = np.isfinite(roi)
    consistent_roi = finite & (roi >= lower) & (roi <= upper)
    closer_roi = finite & (roi < lower)
    farther_roi = finite & (roi > upper)
    visible_mask = np.zeros((int(height), int(width)), dtype=bool)
    visible_mask[row0:row1, col0:col1] = consistent_roi
    native_visible_px = int(np.count_nonzero(consistent_roi))
    roi_px = int(roi.size)
    visible_fraction = native_visible_px / float(roi_px)
    model_visible_px = _nearest_model_count(visible_mask, int(model_width), int(model_height))
    eligible_visible, eligible_clear = eligibility_flags(
        distance_m=float(distance_m), projected_area_px=clipped_area,
        model_visible_px=model_visible_px, visible_fraction=visible_fraction,
    )
    metrics: dict[str, Any] = {
        "unclipped_bbox_x": ux0,
        "unclipped_bbox_y": uy0,
        "unclipped_bbox_w": unclipped_w,
        "unclipped_bbox_h": unclipped_h,
        "clipped_bbox_x": cx0,
        "clipped_bbox_y": cy0,
        "clipped_bbox_w": clipped_w,
        "clipped_bbox_h": clipped_h,
        "unclipped_projected_area_px": unclipped_area,
        "clipped_projected_area_px": clipped_area,
        "projected_box_in_frame_fraction": (
            clipped_area / unclipped_area if unclipped_area > 0.0 else 0.0
        ),
        "actor_near_depth_m": near_m,
        "actor_far_depth_m": far_m,
        "sampled_roi_px": roi_px,
        "native_visible_px": native_visible_px,
        "model_input_visible_px": model_visible_px,
        "visible_fraction": visible_fraction,
        "occluder_closer_fraction": float(np.count_nonzero(closer_roi)) / roi_px,
        "background_farther_fraction": float(np.count_nonzero(farther_roi)) / roi_px,
        "geometry_qualified_v2": bool(
            float(distance_m) <= MAX_DISTANCE_M and clipped_area >= MIN_PROJECTED_AREA_PX
        ),
        "eligible_visible_v010": eligible_visible,
        "eligible_clear_v025": eligible_clear,
        "visibility_tier": visibility_tier(visible_fraction, model_visible_px),
        "depth_tolerance_m": float(tolerance_m),
        "visibility_algorithm_version": ALGORITHM_VERSION,
    }
    return metrics, visible_mask


def reconstruct_consistent_mask(
    depth_m: np.ndarray, row: dict[str, Any], *, width: int, height: int,
) -> np.ndarray:
    """Rebuild one actor mask from a retained visibility row and raw depth."""
    x0 = max(0, int(math.floor(float(row["clipped_bbox_x"]))))
    y0 = max(0, int(math.floor(float(row["clipped_bbox_y"]))))
    x1 = min(int(width), max(x0 + 1, int(math.ceil(
        float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"])))))
    y1 = min(int(height), max(y0 + 1, int(math.ceil(
        float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"])))))
    mask = np.zeros((int(height), int(width)), dtype=bool)
    roi = np.asarray(depth_m)[y0:y1, x0:x1]
    tolerance = float(row["depth_tolerance_m"])
    mask[y0:y1, x0:x1] = (
        np.isfinite(roi)
        & (roi >= float(row["actor_near_depth_m"]) - tolerance)
        & (roi <= float(row["actor_far_depth_m"]) + tolerance)
    )
    return mask


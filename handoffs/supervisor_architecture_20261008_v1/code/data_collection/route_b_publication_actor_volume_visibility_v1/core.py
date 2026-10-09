"""Geometry core for the actor-volume pedestrian visibility metric.

The metric replaces the global depth-interval occupancy of
``route_b_depth_visibility_interval_v1`` with a per-actor oriented-bounding-box
containment test.  A depth pixel supports an actor only when the 3D point it
back-projects to lies inside that specific actor's oriented volume, so road and
ground pixels that merely happen to fall inside the actor's global near/far
depth interval can no longer be counted as actor support.

This module is deliberately I/O-free and CARLA-free so every rule can be
exercised on small synthetic arrays.  It never consults semantic masks,
instance masks, detections, learned masks, or human labels.

Coordinate conventions (identical to the frozen collector contract in
``data_collection/route_b_perception_v3/visibility_v1.py``):

  * World frame is CARLA's left-handed UE frame, metres.
  * ``camera_inverse`` is the 4x4 world -> sensor matrix, ``camera_matrix`` is
    its recorded 4x4 sensor -> world counterpart.
  * In the sensor frame ``x`` is forward (and is exactly what the CARLA depth
    camera encodes), ``y`` is right, ``z`` is up.
  * Forward projection:  ``u = cx + (y / x) * fx``,  ``v = cy - (z / x) * fy``.
  * Back projection of pixel centre ``(u, v)`` at planar depth ``d``:
        ``x = d``,  ``y = (u - cx) * d / fx``,  ``z = -(v - cy) * d / fy``.
  * An actor's oriented box is its recorded world bbox centre plus its recorded
    half extents, rotated by the recorded actor yaw about the world z axis
    (pitch and roll are zero for the pedestrian actors in this corpus; the
    reconstruction is verified against the recorded projected box).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

ALGORITHM_VERSION = "route_b_actor_volume_visibility_v1"

# --- locked metric constants -------------------------------------------------
ACTOR_VOLUME_TOLERANCE_M = 0.05
"""Fixed numerical containment tolerance applied to every oriented-box face."""

GROUND_REJECT_MARGIN_M = 0.03
"""Actor-local height above the box bottom plane below which points are ground."""

CARLA_MAX_DEPTH_M = 1000.0
"""Far plane of the CARLA depth encoding; pixels at the far plane are sky."""

# --- locked human visibility bands ------------------------------------------
BAND_NOT_OBSERVABLE = "not_observable_0_20"
BAND_HEAVY = "heavy_20_65"
BAND_PARTIAL = "partial_65_90"
BAND_BARE = "bare_90_100"
BAND_ORDER: tuple[str, ...] = (
    BAND_NOT_OBSERVABLE,
    BAND_HEAVY,
    BAND_PARTIAL,
    BAND_BARE,
)
BAND_EDGES: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.20, BAND_NOT_OBSERVABLE),
    (0.20, 0.65, BAND_HEAVY),
    (0.65, 0.90, BAND_PARTIAL),
    (0.90, 1.00, BAND_BARE),
)
BINARY_DECISION_THRESHOLD = 0.65
"""The >=0.65 'at least partially visible' operating point."""

_CLAMP_EPS = 1e-9


def band_for_score(score: float) -> str:
    """Map an automatic visibility score onto the frozen human band names."""
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"non-finite visibility score {score!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"visibility score {value!r} outside [0, 1]")
    for lower, upper, name in BAND_EDGES:
        if value >= lower and (value < upper or name == BAND_BARE):
            return name
    raise ValueError(f"unbanded visibility score {value!r}")


def clamp_unit(value: float, *, eps: float = _CLAMP_EPS) -> float:
    """Clamp only numerical round-off back into [0, 1]; reject real excursions."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite ratio {value!r}")
    if number < -eps or number > 1.0 + eps:
        raise ValueError(f"ratio {number!r} outside [0, 1] beyond round-off")
    return min(1.0, max(0.0, number))


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    """World-from-actor rotation for a yaw-only CARLA transform."""
    yaw = math.radians(float(yaw_deg))
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def bbox_corner_offsets(extent: Sequence[float]) -> np.ndarray:
    """Eight signed half-extent offsets in the actor-local bbox frame."""
    ex, ey, ez = (float(v) for v in extent)
    return np.asarray(
        [
            [sx * ex, sy * ey, sz * ez]
            for sx in (1.0, -1.0)
            for sy in (1.0, -1.0)
            for sz in (1.0, -1.0)
        ],
        dtype=np.float64,
    )


def oriented_box_corners(
    centre_world: Sequence[float], extent: Sequence[float], yaw_deg: float
) -> np.ndarray:
    """Reconstruct the eight world corners of an actor's oriented bounding box."""
    centre = np.asarray(centre_world, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"expected a 3-vector actor centre, got {centre.shape}")
    rotation = yaw_rotation(yaw_deg)
    return (rotation @ bbox_corner_offsets(extent).T).T + centre


def project_points(
    points_world: np.ndarray,
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    *,
    min_forward_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points; return (u, v, forward-depth) for in-front points."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 world points, got {points.shape}")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (np.asarray(camera_inverse, dtype=np.float64) @ homogeneous.T).T[:, :3]
    forward = camera[:, 0]
    in_front = forward > float(min_forward_m)
    if not np.any(in_front):
        raise ValueError("no point lies in front of the camera")
    x = forward[in_front]
    y = camera[in_front, 1]
    z = camera[in_front, 2]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    u = matrix[0, 2] + (y / x) * matrix[0, 0]
    v = matrix[1, 2] - (z / x) * matrix[1, 1]
    return u, v, x


def projected_boxes(
    corners_world: np.ndarray,
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, float]:
    """Clipped and unclipped axis-aligned projected boxes for one actor volume."""
    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"expected exactly eight actor corners, got {corners.shape}")
    u, v, x = project_points(corners, camera_inverse, intrinsics)
    ux0, uy0 = float(np.min(u)), float(np.min(v))
    ux1, uy1 = float(np.max(u)), float(np.max(v))
    unclipped_w, unclipped_h = max(0.0, ux1 - ux0), max(0.0, uy1 - uy0)
    cx0 = float(np.clip(ux0, 0.0, float(width)))
    cy0 = float(np.clip(uy0, 0.0, float(height)))
    cx1 = float(np.clip(ux1, 0.0, float(width)))
    cy1 = float(np.clip(uy1, 0.0, float(height)))
    clipped_w, clipped_h = max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)
    if clipped_w * clipped_h <= 0.0:
        raise ValueError("actor projected box has no in-frame area")
    return {
        "unclipped_bbox_x": ux0,
        "unclipped_bbox_y": uy0,
        "unclipped_bbox_w": unclipped_w,
        "unclipped_bbox_h": unclipped_h,
        "unclipped_projected_area_px": unclipped_w * unclipped_h,
        "clipped_bbox_x": cx0,
        "clipped_bbox_y": cy0,
        "clipped_bbox_w": clipped_w,
        "clipped_bbox_h": clipped_h,
        "clipped_projected_area_px": clipped_w * clipped_h,
        "actor_near_depth_m": float(np.min(x)),
        "actor_far_depth_m": float(np.max(x)),
    }


def truncation_from_boxes(clipped_area_px: float, unclipped_area_px: float) -> float:
    """Boundary truncation, kept strictly separate from occlusion."""
    unclipped = float(unclipped_area_px)
    if unclipped <= 0.0:
        raise ValueError("unclipped projected area must be positive")
    return clamp_unit(1.0 - float(clipped_area_px) / unclipped)


def roi_pixel_bounds(
    clipped: dict[str, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    """Integer pixel window covering the clipped projected box (collector rule)."""
    cx0, cy0 = float(clipped["clipped_bbox_x"]), float(clipped["clipped_bbox_y"])
    cx1 = cx0 + float(clipped["clipped_bbox_w"])
    cy1 = cy0 + float(clipped["clipped_bbox_h"])
    col0, row0 = max(0, int(math.floor(cx0))), max(0, int(math.floor(cy0)))
    col1 = min(int(width), max(col0 + 1, int(math.ceil(cx1))))
    row1 = min(int(height), max(row0 + 1, int(math.ceil(cy1))))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("clipped projected box produced an empty sampled ROI")
    return row0, row1, col0, col1


def back_project_roi(
    depth_m: np.ndarray,
    bounds: tuple[int, int, int, int],
    camera_matrix: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, np.ndarray]:
    """Back-project every valid depth pixel of the ROI into world coordinates."""
    row0, row1, col0, col1 = bounds
    roi = np.asarray(depth_m, dtype=np.float64)[row0:row1, col0:col1]
    if roi.size == 0:
        raise ValueError("empty ROI")
    rows, cols = np.mgrid[row0:row1, col0:col1]
    valid = np.isfinite(roi) & (roi > 0.0) & (roi < CARLA_MAX_DEPTH_M)
    depth = roi[valid]
    u = cols[valid].astype(np.float64) + 0.5
    v = rows[valid].astype(np.float64) + 0.5
    matrix = np.asarray(intrinsics, dtype=np.float64)
    cam_x = depth
    cam_y = (u - matrix[0, 2]) * depth / matrix[0, 0]
    cam_z = -(v - matrix[1, 2]) * depth / matrix[1, 1]
    camera_points = np.stack([cam_x, cam_y, cam_z, np.ones_like(depth)], axis=1)
    world = (np.asarray(camera_matrix, dtype=np.float64) @ camera_points.T).T[:, :3]
    return {
        "world": world,
        "u": u,
        "v": v,
        "row": rows[valid].astype(np.int64),
        "col": cols[valid].astype(np.int64),
        "depth_m": depth,
        "roi_px": int(roi.size),
        "valid_px": int(depth.size),
    }


def actor_local_points(
    world_points: np.ndarray, centre_world: Sequence[float], yaw_deg: float
) -> np.ndarray:
    """Rotate/translate world points into the actor's local oriented-box frame."""
    centre = np.asarray(centre_world, dtype=np.float64)
    rotation = yaw_rotation(yaw_deg)
    return (rotation.T @ (np.asarray(world_points, dtype=np.float64) - centre).T).T


def inside_actor_volume(
    local_points: np.ndarray,
    extent: Sequence[float],
    *,
    tolerance_m: float = ACTOR_VOLUME_TOLERANCE_M,
    ground_margin_m: float = GROUND_REJECT_MARGIN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inside_and_above_ground, volume_test_only) boolean masks."""
    local = np.asarray(local_points, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError(f"expected Nx3 actor-local points, got {local.shape}")
    half = np.asarray([float(v) for v in extent], dtype=np.float64)
    if np.any(half <= 0.0):
        raise ValueError(f"actor half extents must be positive, got {half}")
    in_volume = np.all(np.abs(local) <= half + float(tolerance_m), axis=1)
    above_ground = local[:, 2] > (-half[2] + float(ground_margin_m))
    return in_volume & above_ground, in_volume


def normalized_actor_distance(
    local_points: np.ndarray, extent: Sequence[float]
) -> np.ndarray:
    """Extent-normalised Euclidean distance from the actor-local box centre."""
    local = np.asarray(local_points, dtype=np.float64)
    half = np.asarray([float(v) for v in extent], dtype=np.float64)
    return np.linalg.norm(local / half, axis=1)


def visible_box_from_pixels(
    rows: np.ndarray, cols: np.ndarray
) -> tuple[float, float, float, float]:
    """Tight pixel-extent 2D box (x, y, w, h) enclosing the retained points.

    A retained pixel occupies the unit square ``[col, col + 1) x [row, row + 1)``
    in continuous image coordinates, so the enclosing box runs from the smallest
    retained pixel origin to one past the largest.
    """
    if rows.size == 0:
        raise ValueError("no retained points")
    x0, y0 = float(np.min(cols)), float(np.min(rows))
    x1, y1 = float(np.max(cols)) + 1.0, float(np.max(rows)) + 1.0
    return x0, y0, x1 - x0, y1 - y0


def intersect_boxes(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Intersection of two (x, y, w, h) boxes; zero width/height when disjoint."""
    ax0, ay0, aw, ah = (float(v) for v in a)
    bx0, by0, bw, bh = (float(v) for v in b)
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)


def assign_competing_pedestrians(
    world_points: np.ndarray,
    target_key: str,
    candidates: Iterable[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Deterministically award each depth point to one pedestrian volume.

    ``candidates`` are dicts with ``key`` (a stable identifier), ``centre``,
    ``extent`` and ``yaw_deg``.  A point is retained for the target only when
    the target owns it, i.e. no other pedestrian box that also contains the
    point has a strictly smaller normalised actor-local distance.  Exact ties
    are broken by the lexicographically smallest key so the rule is
    order-independent.
    """
    boxes = sorted(candidates, key=lambda box: str(box["key"]))
    if not any(str(box["key"]) == str(target_key) for box in boxes):
        raise ValueError(f"target {target_key!r} missing from candidate boxes")
    count = int(np.asarray(world_points).shape[0])
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_rank = np.full(count, -1, dtype=np.int64)
    contains: dict[str, np.ndarray] = {}
    for rank, box in enumerate(boxes):
        local = actor_local_points(world_points, box["centre"], box["yaw_deg"])
        retained, _ = inside_actor_volume(local, box["extent"])
        contains[str(box["key"])] = retained
        if not np.any(retained):
            continue
        distance = np.where(
            retained, normalized_actor_distance(local, box["extent"]), np.inf
        )
        better = distance < best_distance
        best_distance = np.where(better, distance, best_distance)
        best_rank = np.where(better, rank, best_rank)
    target_rank = next(
        rank for rank, box in enumerate(boxes) if str(box["key"]) == str(target_key)
    )
    owned = (best_rank == target_rank) & contains[str(target_key)]
    competing = sum(
        1
        for key, mask in contains.items()
        if key != str(target_key) and bool(np.any(mask & contains[str(target_key)]))
    )
    return {
        "owned": owned,
        "target_contains": contains[str(target_key)],
        "competing_actor_boxes": competing,
    }

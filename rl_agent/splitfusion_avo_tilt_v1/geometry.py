"""Tilt-aware actor-volume geometry for the live pedestrian AVO (candidate, versioned).

The frozen geometry (``data_collection/route_b_publication_actor_volume_visibility_v1``, hash-pinned in the
producer's bound originals and NOT edited) rotates every pedestrian box by yaw only, and the live adapter refuses any
scoreable pedestrian whose frozen pitch or roll exceeds 1e-6 degrees.  This module replaces only the box rotation:

* ``|pitch| <= 1e-6 and |roll| <= 1e-6`` (the unchanged pre-existing upright boundary): the frozen
  ``core.yaw_rotation(yaw)`` -- the identical arithmetic, so upright results are bit-identical;
* otherwise: the 3x3 rotation block of the FROZEN CARLA actor transform matrix (``transform.get_matrix()`` as
  captured on the UE side).  No Euler order is assumed: the matrix is validated as a proper rotation and against
  CARLA's own ``carla.Transform(Location(), Rotation(pitch, yaw, roll)).get_matrix()``.  This is the convention the
  producer's target path already uses for every actor (``actor_bbox_world_points``: ``M @ (bbox.location + offsets)``).

The single world-from-box rotation feeds every geometric step: corners and projected boxes (structural eligibility),
actor-local containment, the actor-local ground rule and competing-volume ownership (tilted competitors included).
AVO formula, 0.65 threshold, tolerances, ground margin, range limits and matching are unchanged.

Bounding-box rotation: the target path ignores ``bbox.rotation``; every live pedestrian box observed has zero
rotation.  A non-zero bbox rotation is refused (fail closed), never guessed.

Ground-rule envelope: the frozen ground rule is defined in the actor-local frame (height above the box bottom
plane, margin 0.03 m).  For a tilted box a horizontal world ground deviates from that plane across the half
footprint by ``|R[2,0]|*ex + |R[2,1]|*ey``.  Where that exceeds the frozen 0.03 m margin the rule no longer means
"reject the world ground", so such an actor is outside this version's envelope: it is refused if it is a target or
a possible competitor; a structurally ignored actor proven (bounding spheres) disjoint from every target is omitted
from competition exactly as the previous adapter did.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

GEOMETRY_VERSION = "route_b_actor_volume_visibility_v1__tilt_aware_candidate_v1"
UPRIGHT_TOLERANCE_DEG = 1e-6        # unchanged boundary of the supported yaw-only path
ROTATION_TOLERANCE = 1e-6           # proper-rotation and CARLA-consistency checks on the frozen matrix
PATH_UPRIGHT = "UPRIGHT_YAW_ONLY_FROZEN_CORE"
PATH_TILTED = "FROZEN_CARLA_ACTOR_MATRIX"


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise LiveGtContractError(f"nonfinite {name}")
    return number


def carla_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """CARLA's own Euler -> matrix conversion (libcarla ``Transform.get_matrix``), rotation block only."""
    import carla
    transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0),
                                carla.Rotation(pitch=float(pitch_deg), yaw=float(yaw_deg), roll=float(roll_deg)))
    return np.asarray(transform.get_matrix(), dtype=np.float64)[:3, :3]


def is_upright(pitch_deg: float, roll_deg: float) -> bool:
    return abs(float(pitch_deg)) <= UPRIGHT_TOLERANCE_DEG and abs(float(roll_deg)) <= UPRIGHT_TOLERANCE_DEG


def box_rotation(*, yaw_deg: float, pitch_deg: float, roll_deg: float, actor_rotation_matrix: Sequence[float],
                 bbox_rotation_deg: Sequence[float]) -> tuple[np.ndarray, str]:
    """World-from-box rotation and the path that produced it (fail closed on invalid evidence)."""
    yaw, pitch, roll = (_finite(v, name) for v, name in ((yaw_deg, "actor yaw"), (pitch_deg, "actor pitch"),
                                                         (roll_deg, "actor roll")))
    bbox = [_finite(v, "bbox rotation") for v in bbox_rotation_deg]
    if len(bbox) != 3 or any(v != 0.0 for v in bbox):
        raise LiveGtContractError("non-zero pedestrian bounding-box rotation is outside the tilt-aware geometry v1")
    if is_upright(pitch, roll):
        return core.yaw_rotation(yaw), PATH_UPRIGHT
    matrix = np.asarray(actor_rotation_matrix, dtype=np.float64)
    if matrix.shape != (9,) and matrix.shape != (3, 3):
        raise LiveGtContractError("frozen actor rotation matrix must have 9 values")
    rotation = np.ascontiguousarray(matrix.reshape(3, 3))
    if not np.all(np.isfinite(rotation)):
        raise LiveGtContractError("nonfinite frozen actor rotation matrix")
    if (float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))) > ROTATION_TOLERANCE
            or abs(float(np.linalg.det(rotation)) - 1.0) > ROTATION_TOLERANCE):
        raise LiveGtContractError("frozen actor rotation matrix is not a proper rotation")
    reference = carla_rotation_matrix(pitch, yaw, roll)
    if float(np.max(np.abs(rotation - reference))) > ROTATION_TOLERANCE:
        raise LiveGtContractError("frozen actor rotation matrix differs from its frozen pitch/yaw/roll")
    return rotation, PATH_TILTED


def footprint_tilt_m(rotation: np.ndarray, extent: Sequence[float]) -> float:
    """Height range of the box bottom face's half footprint along world z (0 for an upright box)."""
    r = np.asarray(rotation, dtype=np.float64)
    return float(abs(r[2, 0]) * float(extent[0]) + abs(r[2, 1]) * float(extent[1]))


def within_ground_envelope(rotation: np.ndarray, extent: Sequence[float]) -> bool:
    return footprint_tilt_m(rotation, extent) <= core.GROUND_REJECT_MARGIN_M


def oriented_box_corners(centre_world: Sequence[float], extent: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    """``core.oriented_box_corners`` with an explicit world-from-box rotation (identical arithmetic)."""
    centre = np.asarray(centre_world, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"expected a 3-vector actor centre, got {centre.shape}")
    return (np.asarray(rotation, dtype=np.float64) @ core.bbox_corner_offsets(extent).T).T + centre


def actor_local_points(world_points: np.ndarray, centre_world: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    """``core.actor_local_points`` with an explicit world-from-box rotation (identical arithmetic)."""
    centre = np.asarray(centre_world, dtype=np.float64)
    return (np.asarray(rotation, dtype=np.float64).T @ (np.asarray(world_points, dtype=np.float64) - centre).T).T


def assign_competing_pedestrians(world_points: np.ndarray, target_key: str,
                                 candidates: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    """``core.assign_competing_pedestrians`` with each candidate's own ``rotation`` (rule unchanged)."""
    boxes = sorted(candidates, key=lambda box: str(box["key"]))
    if not any(str(box["key"]) == str(target_key) for box in boxes):
        raise ValueError(f"target {target_key!r} missing from candidate boxes")
    count = int(np.asarray(world_points).shape[0])
    best_distance = np.full(count, np.inf, dtype=np.float64)
    best_rank = np.full(count, -1, dtype=np.int64)
    contains: dict[str, np.ndarray] = {}
    for rank, box in enumerate(boxes):
        local = actor_local_points(world_points, box["centre"], box["rotation"])
        retained, _ = core.inside_actor_volume(local, box["extent"])
        contains[str(box["key"])] = retained
        if not np.any(retained):
            continue
        distance = np.where(retained, core.normalized_actor_distance(local, box["extent"]), np.inf)
        better = distance < best_distance
        best_distance = np.where(better, distance, best_distance)
        best_rank = np.where(better, rank, best_rank)
    target_rank = next(rank for rank, box in enumerate(boxes) if str(box["key"]) == str(target_key))
    owned = (best_rank == target_rank) & contains[str(target_key)]
    competing = sum(1 for key, mask in contains.items()
                    if key != str(target_key) and bool(np.any(mask & contains[str(target_key)])))
    return {"owned": owned, "target_contains": contains[str(target_key)], "competing_actor_boxes": competing}


def score_actor_frame(*, depth_m: np.ndarray, camera_matrix: np.ndarray, camera_inverse: np.ndarray,
                      intrinsics: np.ndarray, width: int, height: int, target_key: str,
                      target_centre: Sequence[float], target_extent: Sequence[float], target_rotation: np.ndarray,
                      pedestrian_boxes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``scoring.score_actor_frame`` with the target's and competitors' explicit rotations.

    Formula, tolerances, ROI rasterisation, ownership and diagnostics are the frozen ones; only the version label
    differs (``algorithm_version`` = this geometry version).
    """
    corners = oriented_box_corners(target_centre, target_extent, target_rotation)
    boxes = core.projected_boxes(corners, camera_inverse, intrinsics, width=width, height=height)
    bounds = core.roi_pixel_bounds(boxes, width=width, height=height)
    roi = core.back_project_roi(depth_m, bounds, camera_matrix, intrinsics)
    assignment = assign_competing_pedestrians(roi["world"], target_key, pedestrian_boxes)
    owned = assignment["owned"]
    retained_count = int(np.count_nonzero(owned))
    clipped_area = float(boxes["clipped_projected_area_px"])
    result: dict[str, Any] = {
        "algorithm_version": GEOMETRY_VERSION,
        "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
        "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
        **boxes,
        "sampled_roi_px": roi["roi_px"],
        "valid_depth_px": roi["valid_px"],
        "retained_actor_point_count": retained_count,
        "competing_actor_boxes": int(assignment["competing_actor_boxes"]),
        "truncation": core.truncation_from_boxes(clipped_area, float(boxes["unclipped_projected_area_px"])),
    }
    if retained_count == 0:
        result.update({"visibility": 0.0, "visibility_band": core.band_for_score(0.0), "no_support": True,
                       "visible_bbox_x": float("nan"), "visible_bbox_y": float("nan"), "visible_bbox_w": 0.0,
                       "visible_bbox_h": 0.0, "visible_box_area_px": 0.0, "visible_box_raster_ratio": 0.0,
                       "degenerate_visible_box": False, "visible_box_height_ratio": 0.0,
                       "visible_box_width_ratio": 0.0, "actor_point_surface_occupancy": 0.0,
                       "visible_box_fill_ratio": 0.0, "retained_local_z_min_above_floor_m": float("nan"),
                       "retained_max_abs_local_excess_m": float("nan")})
        return result
    rows, cols = roi["row"][owned], roi["col"][owned]
    raster = core.visible_box_from_pixels(rows, cols)
    clipped_box = (float(boxes["clipped_bbox_x"]), float(boxes["clipped_bbox_y"]),
                   float(boxes["clipped_bbox_w"]), float(boxes["clipped_bbox_h"]))
    vx, vy, vw, vh = core.intersect_boxes(raster, clipped_box)
    visibility = core.clamp_unit(vw * vh / clipped_area)
    local = actor_local_points(roi["world"][owned], target_centre, target_rotation)
    half = np.asarray([float(v) for v in target_extent], dtype=np.float64)
    result.update({
        "visibility": visibility, "visibility_band": core.band_for_score(visibility), "no_support": False,
        "visible_bbox_x": vx, "visible_bbox_y": vy, "visible_bbox_w": vw, "visible_bbox_h": vh,
        "visible_box_area_px": vw * vh, "visible_box_raster_ratio": raster[2] * raster[3] / clipped_area,
        "degenerate_visible_box": bool(vw * vh <= 0.0),
        "visible_box_height_ratio": core.clamp_unit(vh / float(boxes["clipped_bbox_h"])),
        "visible_box_width_ratio": core.clamp_unit(vw / float(boxes["clipped_bbox_w"])),
        "actor_point_surface_occupancy": core.clamp_unit(retained_count / float(roi["roi_px"])),
        "visible_box_fill_ratio": core.clamp_unit(min(1.0, retained_count / (vw * vh)) if vw * vh > 0.0 else 0.0),
        "retained_local_z_min_above_floor_m": float(np.min(local[:, 2]) + half[2]),
        "retained_max_abs_local_excess_m": float(np.max(np.abs(local) - half[None, :])),
    })
    return result

"""Exact frame-local reuse and ROI-limited per-object work for one producer frame.

Each change is an orchestration change over the pinned numerics; every output value is computed by the same
arithmetic on the same operands:

1. ROI-limited visibility. ``visibility_from_corners`` (pinned ``route_b_perception_v3/visibility_v1``) builds a
   720x1280 mask that is zero outside the clipped box ROI and then counts it on the proven fixed 768x432 grid.
   ``visibility_core`` runs the identical ROI arithmetic and counts only the grid rows/columns inside the ROI.
   Pixels outside the ROI are zero, so the integer count is identical. The full mask is materialised (np.zeros +
   ROI assignment, exactly as the original) only where a caller consumes it.
2. ROI-limited static semantic counts: ``count(visible & class_mask)`` over the full image equals
   ``count(roi_visible & class_mask[roi])`` because ``visible`` is zero outside the ROI.
3. Lazy radar KD-tree: the original builds ``cKDTree(radar)`` for every frame but queries it only for static
   objects that reach row construction. The identical tree is built on first use, or never.
4. Frame-local ``actor_bbox_world_points``: a pure function of the frozen actor snapshot. It is called about
   2.3 times per actor per frame (row build, visibility, dynamic corners). It is computed once per actor per frame,
   and every caller gets fresh copies.
5. Conflict digests through the already qualified exact private conflict state
   (``splitfusion_live_gt_optimization_v1.source_geometry._conflict_namespace``: fast exact JSON normalisation;
   every validator, including the per-frame catalogue re-hash, is retained).
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import math

import numpy as np
from scipy.spatial import cKDTree

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S

GRID = (720, 1280, 432, 768)
_KEEP = {}


def _keep(n):
    """Rows/columns kept by the proven fixed nearest grid: index % 5 in {0, 1, 3} (V1S._fixed_grid_proof)."""
    keep = _KEEP.get(n)
    if keep is None:
        V1S._fixed_grid_proof()
        keep = np.isin(np.arange(n) % 5, (0, 1, 3))
        keep.flags.writeable = False
        _KEEP[n] = keep
    return keep


def visibility_core(vis, corners_world, camera_inverse, intrinsics, depth_m, *, distance_m, width, height,
                    model_width, model_height, tolerance_m=None):
    """Verbatim ``visibility_from_corners`` arithmetic; returns metrics, ROI bounds and the ROI visible mask."""
    if tolerance_m is None:
        tolerance_m = vis.DEPTH_TOLERANCE_M
    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"expected exactly eight actor AABB corners, got {corners.shape}")
    if np.asarray(depth_m).shape != (int(height), int(width)):
        raise ValueError(f"depth shape {np.asarray(depth_m).shape} != {(int(height), int(width))}")
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
    native_visible_px = int(np.count_nonzero(consistent_roi))
    roi_px = int(roi.size)
    visible_fraction = native_visible_px / float(roi_px)
    if (int(height), int(width), int(model_height), int(model_width)) == GRID:
        rows, cols = _keep(int(height))[row0:row1], _keep(int(width))[col0:col1]
        model_visible_px = int(np.count_nonzero(consistent_roi[rows][:, cols]))
    else:                                    # outside the proven geometry: the original full-mask count
        full = np.zeros((int(height), int(width)), dtype=bool)
        full[row0:row1, col0:col1] = consistent_roi
        model_visible_px = vis._nearest_model_count(full, int(model_width), int(model_height))
    eligible_visible, eligible_clear = vis.eligibility_flags(distance_m=float(distance_m),
        projected_area_px=clipped_area, model_visible_px=model_visible_px, visible_fraction=visible_fraction)
    metrics = {
        "unclipped_bbox_x": ux0, "unclipped_bbox_y": uy0, "unclipped_bbox_w": unclipped_w, "unclipped_bbox_h": unclipped_h,
        "clipped_bbox_x": cx0, "clipped_bbox_y": cy0, "clipped_bbox_w": clipped_w, "clipped_bbox_h": clipped_h,
        "unclipped_projected_area_px": unclipped_area, "clipped_projected_area_px": clipped_area,
        "projected_box_in_frame_fraction": (clipped_area / unclipped_area if unclipped_area > 0.0 else 0.0),
        "actor_near_depth_m": near_m, "actor_far_depth_m": far_m, "sampled_roi_px": roi_px,
        "native_visible_px": native_visible_px, "model_input_visible_px": model_visible_px,
        "visible_fraction": visible_fraction,
        "occluder_closer_fraction": float(np.count_nonzero(closer_roi)) / roi_px,
        "background_farther_fraction": float(np.count_nonzero(farther_roi)) / roi_px,
        "geometry_qualified_v2": bool(float(distance_m) <= vis.MAX_DISTANCE_M and clipped_area >= vis.MIN_PROJECTED_AREA_PX),
        "eligible_visible_v010": eligible_visible, "eligible_clear_v025": eligible_clear,
        "visibility_tier": vis.visibility_tier(visible_fraction, model_visible_px),
        "depth_tolerance_m": float(tolerance_m), "visibility_algorithm_version": vis.ALGORITHM_VERSION,
    }
    return metrics, (row0, row1, col0, col1), consistent_roi


def make_visibility_from_corners(vis):
    """Same signature/return as the original: (metrics, full-resolution visible mask)."""
    def visibility_from_corners(corners_world, camera_inverse, intrinsics, depth_m, *, distance_m, width, height,
                                model_width, model_height, tolerance_m=None):
        metrics, (row0, row1, col0, col1), consistent_roi = visibility_core(vis, corners_world, camera_inverse,
            intrinsics, depth_m, distance_m=distance_m, width=width, height=height, model_width=model_width,
            model_height=model_height, tolerance_m=tolerance_m)
        visible_mask = np.zeros((int(height), int(width)), dtype=bool)
        visible_mask[row0:row1, col0:col1] = consistent_roi
        return metrics, visible_mask
    return visibility_from_corners


def static_rows(bound, *, frame, boxes, geometry, depth_m, semantic, vehicle_mask, same_class, radar):
    """``splitfusion_numeric_gt_producer_v1.source.static_rows`` with ROI-limited counts and a lazy radar tree.

    Row dictionaries, order, numeric validation, failure counters and the summary are unchanged.
    """
    ns = bound.static
    vis = bound.visibility
    carla = ns["carla"]
    width, height = int(frame["camera_width"]), int(frame["camera_height"])
    if (width, height) != (1280, 720):
        raise RuntimeError(f"unexpected retained camera size: {(width, height)}")
    episode, sample_id = frame["experiment_id"], frame["sample_id"]
    if not ns["depth_is_plausible"](depth_m):
        raise RuntimeError(f"implausible retained depth: {sample_id}")
    if semantic is None or semantic.shape != (height, width):
        raise RuntimeError(f"invalid retained semantic tags: {sample_id}")
    radar_world = np.asarray(radar, dtype=np.float64)
    if radar_world.ndim != 2 or radar_world.shape[1] != 3 or not np.all(np.isfinite(radar_world)):
        raise RuntimeError(f"invalid retained radar world points: {sample_id}")
    tree_box = [None, False]          # [tree, built]

    def radar_tree():
        if not tree_box[1]:
            tree_box[0] = cKDTree(radar_world) if len(radar_world) else None
            tree_box[1] = True
        return tree_box[0]
    camera_inverse = np.asarray(json.loads(frame["camera_inverse_matrix_json"]), dtype=np.float64)
    intrinsics = ns["intrinsics_from_manifest"](frame)
    camera_center = np.asarray([float(frame[f"camera_{axis}"]) for axis in "xyz"], dtype=np.float64)
    class_tag = {name: int(getattr(carla.CityObjectLabel, name)) for name in ns["SEMANTIC_NAMES"]}
    actor_rows_by_sample = {sample_id: boxes}
    rows, ambiguity_rows = [], []
    projection_failures = Counter()
    frame_counts = {"sample_id": sample_id, "in_range": 0, "projected_in_frame": 0, "area_qualified": 0,
                    "eligible_v010": 0, "eligible_v025": 0, "unobservable_area_qualified": 0, "ambiguous": 0}
    for item in geometry:
        catalog, center, extent = item["row"], item["center"], item["extent"]
        distance = float(np.linalg.norm(center - camera_center))
        if distance > ns["MAX_RANGE_M"]:
            continue
        frame_counts["in_range"] += 1
        try:
            metrics, (r0, r1, c0, c1), visible_roi = visibility_core(vis, item["corners"], camera_inverse, intrinsics,
                depth_m, distance_m=distance, width=width, height=height, model_width=ns["MODEL_WIDTH"],
                model_height=ns["MODEL_HEIGHT"])
        except ValueError as exc:
            projection_failures[str(exc)] += 1
            continue
        frame_counts["projected_in_frame"] += 1
        sensor_center = (camera_inverse @ np.asarray([center[0], center[1], center[2], 1.0]))[:3]
        semantic_vehicle_px = int(np.count_nonzero(visible_roi & vehicle_mask[r0:r1, c0:c1]))
        semantic_same_px = int(np.count_nonzero(visible_roi & same_class(class_tag[catalog["semantic_class"]])[r0:r1, c0:c1]))
        native_visible_px = int(metrics["native_visible_px"])
        static_size = extent * 2.0
        ambiguities = ns["actor_ambiguities"](center, static_size, actor_rows_by_sample.get(sample_id, []))
        nearest = ambiguities[0] if ambiguities else None
        row = {
            "experiment_id": episode, "sample_id": sample_id, "frame_id": int(frame["frame_id"]),
            "timestamp": float(frame["timestamp"]),
            **{key: catalog[key] for key in (
                "environment_object_id", "carla_environment_object_id", "object_name", "semantic_class", "enabled",
                "map_name", "map_sha256", "bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m",
                "bbox_extent_x_m", "bbox_extent_y_m", "bbox_extent_z_m", "bbox_rotation_roll_deg",
                "bbox_rotation_pitch_deg", "bbox_rotation_yaw_deg")},
            "size_x_m": float(static_size[0]), "size_y_m": float(static_size[1]), "size_z_m": float(static_size[2]),
            "object_sensor_x_m": float(sensor_center[0]), "object_sensor_y_m": float(sensor_center[1]),
            "object_sensor_z_m": float(sensor_center[2]), "range_m": distance, **metrics,
            "semantic_vehicle_visible_px": semantic_vehicle_px, "semantic_same_class_visible_px": semantic_same_px,
            "semantic_vehicle_visible_fraction": (semantic_vehicle_px / native_visible_px if native_visible_px else 0.0),
            "semantic_same_class_visible_fraction": (semantic_same_px / native_visible_px if native_visible_px else 0.0),
            "radar_support_points": ns["radar_support_count"](radar_tree(), radar_world, center, extent, item["rotation"]),
            "radar_support_margin_m": ns["RADAR_MARGIN_M"],
            "possible_dynamic_static_duplicate": bool(ambiguities), "ambiguous_actor_count": len(ambiguities),
            "ambiguous_actor_ids": ";".join(str(match["gt_actor_id"]) for match in ambiguities),
            "nearest_ambiguous_actor_id": (nearest["gt_actor_id"] if nearest else ""),
            "nearest_ambiguous_center_distance_m": (nearest["center_distance_m"] if nearest else ""),
            "nearest_ambiguous_size_x_delta_m": (nearest["size_deltas_m"][0] if nearest else ""),
            "nearest_ambiguous_size_y_delta_m": (nearest["size_deltas_m"][1] if nearest else ""),
            "nearest_ambiguous_size_z_delta_m": (nearest["size_deltas_m"][2] if nearest else ""),
        }
        rows.append(row)
        if ambiguities:
            ambiguity_rows.append(row)
            frame_counts["ambiguous"] += 1
        if bool(metrics["geometry_qualified_v2"]):
            frame_counts["area_qualified"] += 1
            if metrics["visibility_tier"] == "unobservable":
                frame_counts["unobservable_area_qualified"] += 1
        if bool(metrics["eligible_visible_v010"]):
            frame_counts["eligible_v010"] += 1
        if bool(metrics["eligible_clear_v025"]):
            frame_counts["eligible_v025"] += 1
    numeric_fields = ("bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m", "bbox_extent_x_m", "bbox_extent_y_m",
        "bbox_extent_z_m", "size_x_m", "size_y_m", "size_z_m", "bbox_rotation_roll_deg", "bbox_rotation_pitch_deg",
        "bbox_rotation_yaw_deg", "object_sensor_x_m", "object_sensor_y_m", "object_sensor_z_m", "range_m",
        "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h", "clipped_projected_area_px",
        "visible_fraction", "model_input_visible_px", "semantic_vehicle_visible_px", "radar_support_points")
    for row in rows:
        for name in numeric_fields:
            ns["finite_float"](row[name], name)
        if row["visibility_algorithm_version"] != ns["ALGORITHM_VERSION"]:
            raise RuntimeError("visibility helper version drift")
    keys = ("in_range", "projected_in_frame", "area_qualified", "eligible_v010", "eligible_v025",
            "unobservable_area_qualified", "ambiguous")
    summary = {"projected_row_count": len(rows), "ambiguity_row_count": len(ambiguity_rows),
        "ambiguity_unique_static_objects": len({row["environment_object_id"] for row in ambiguity_rows}),
        "ambiguity_frames": len({row["sample_id"] for row in ambiguity_rows}),
        "projection_failures": dict(sorted(projection_failures.items())),
        "totals": {key: sum(int(r[key]) for r in [frame_counts]) for key in keys},
        "per_frame": {key: ns["numeric_summary"]([int(r[key]) for r in [frame_counts]]) for key in keys}}
    stats = dict(radar_tree_built=tree_box[1])
    return rows, ambiguity_rows, summary, stats


@contextmanager
def frame_bbox_cache(objects):
    """Install a per-frame memo of ``actor_bbox_world_points`` (fresh copies per call); always restored."""
    original = objects["actor_bbox_world_points"]
    memo = {}
    calls = [0, 0]

    def actor_bbox_world_points(actor):
        calls[0] += 1
        key = id(actor)
        hit = memo.get(key)
        if hit is None or hit[0] is not actor:
            center, corners = original(actor)
            hit = memo[key] = (actor, center, corners)
            calls[1] += 1
        return hit[1].copy(), hit[2].copy()
    objects["actor_bbox_world_points"] = actor_bbox_world_points
    try:
        yield calls
    finally:
        objects["actor_bbox_world_points"] = original

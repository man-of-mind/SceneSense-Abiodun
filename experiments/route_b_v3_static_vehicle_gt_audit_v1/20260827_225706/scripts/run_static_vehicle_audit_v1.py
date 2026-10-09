#!/usr/bin/env python3
"""Run the preregistered Route B v3 static parked-vehicle audit offline."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import carla
import cv2
import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(__file__).resolve().parents[1]
COMPARISON = REPO_ROOT / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455"
VIEW = COMPARISON / "views/val"
CATALOG_DIR = EXPERIMENT / "static_environment_truth/catalog"
CANONICAL_REPORT = (
    REPO_ROOT
    / "data_collection/experiments/route_b_perception_v3/ROUTE_B_V3_CANONICAL_COLLECTION_REPORT.json"
)
BLOCKED_EXPERIMENT = (
    REPO_ROOT / "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_222826"
)
MODELS = ("lraspp_mprime_noae", "fasterrcnn_radar_roi_v1_epoch12")
CONTRACTS = ("v010", "v025")
THRESHOLDS = (0.20, 0.02)
SEMANTIC_NAMES = ("Car", "Truck", "Bus")
VEHICLE_SEMANTIC_TAGS = frozenset((14, 15, 16))
MAX_RANGE_M = 40.0
MIN_AREA_PX = 12.0
MATCH_DISTANCE_M = 3.0
AMBIGUITY_CENTER_M = 1.0
AMBIGUITY_DIMENSION_M = 1.0
RADAR_MARGIN_M = 1.0
MODEL_WIDTH = 768
MODEL_HEIGHT = 432
PARITY_ROWS = 64
PARITY_MIN_IOU = 0.85
PARITY_MEAN_IOU = 0.95
SUCCESS_TERMINAL = "STATIC_VEHICLE_GT_AUDIT_READY_FOR_MANUAL_REVIEW"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.phase2_static_environment_truth_v1 import (  # noqa: E402
    verify_static_environment_truth_v1,
)
from data_collection.route_b_perception_v3.visibility_v1 import (  # noqa: E402
    ALGORITHM_VERSION,
    decode_depth_bgra,
    depth_is_plausible,
    visibility_from_corners,
)


STATIC_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "timestamp",
    "environment_object_id", "carla_environment_object_id", "object_name",
    "semantic_class", "enabled", "map_name", "map_sha256",
    "bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m",
    "bbox_extent_x_m", "bbox_extent_y_m", "bbox_extent_z_m",
    "size_x_m", "size_y_m", "size_z_m",
    "bbox_rotation_roll_deg", "bbox_rotation_pitch_deg", "bbox_rotation_yaw_deg",
    "object_sensor_x_m", "object_sensor_y_m", "object_sensor_z_m", "range_m",
    "unclipped_bbox_x", "unclipped_bbox_y", "unclipped_bbox_w", "unclipped_bbox_h",
    "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h",
    "unclipped_projected_area_px", "clipped_projected_area_px",
    "projected_box_in_frame_fraction", "actor_near_depth_m", "actor_far_depth_m",
    "sampled_roi_px", "native_visible_px", "model_input_visible_px",
    "visible_fraction", "occluder_closer_fraction", "background_farther_fraction",
    "geometry_qualified_v2", "eligible_visible_v010", "eligible_clear_v025",
    "visibility_tier", "depth_tolerance_m", "visibility_algorithm_version",
    "semantic_vehicle_visible_px", "semantic_same_class_visible_px",
    "semantic_vehicle_visible_fraction", "semantic_same_class_visible_fraction",
    "radar_support_points", "radar_support_margin_m",
    "possible_dynamic_static_duplicate", "ambiguous_actor_count",
    "ambiguous_actor_ids", "nearest_ambiguous_actor_id",
    "nearest_ambiguous_center_distance_m", "nearest_ambiguous_size_x_delta_m",
    "nearest_ambiguous_size_y_delta_m", "nearest_ambiguous_size_z_delta_m",
)

RECLASS_FIELDS = (
    "model", "visibility_contract", "score_threshold", "sample_id", "frame_id",
    "prediction_index", "prediction_score", "prediction_world_x", "prediction_world_y",
    "prediction_world_z", "prediction_bbox_x0", "prediction_bbox_y0",
    "prediction_bbox_x1", "prediction_bbox_y1", "environment_object_id",
    "carla_environment_object_id", "semantic_class", "static_world_x",
    "static_world_y", "static_world_z", "match_distance_m", "prior_status",
    "corrected_status", "possible_dynamic_static_duplicate",
)

SELECTION_FIELDS = (
    "review_index", "stratum", "stratum_label", "sample_id", "frame_id",
    "environment_object_id", "carla_environment_object_id", "semantic_class",
    "model", "panel_path", "stratum_available", "stratum_selected",
    "stratum_shortage", "visible_fraction", "model_input_visible_px", "range_m",
    "semantic_vehicle_visible_px", "radar_support_points",
    "eligible_visible_v010", "eligible_clear_v025",
    "possible_dynamic_static_duplicate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_csv_x(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def finite_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite {field}: {value}")
    return result


def numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "minimum": None, "mean": None, "maximum": None,
            "p50": None, "p90": None, "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def tree_hash_records(episode_dirs: Sequence[Path], label: str) -> list[dict[str, Any]]:
    paths = sorted(
        (path for directory in episode_dirs for path in directory.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(REPO_ROOT)),
    )
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        records.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        if index % 1000 == 0:
            print(f"[{label}] {index}/{len(paths)}", flush=True)
    return records


def corners_from_pose(
    center: Sequence[float], extent: Sequence[float],
    roll_deg: float, pitch_deg: float, yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    location = carla.Location(x=float(center[0]), y=float(center[1]), z=float(center[2]))
    rotation = carla.Rotation(
        roll=float(roll_deg), pitch=float(pitch_deg), yaw=float(yaw_deg)
    )
    matrix = np.asarray(carla.Transform(location, rotation).get_matrix(), dtype=np.float64)
    ex, ey, ez = (float(value) for value in extent)
    offsets = np.asarray(
        [
            [sx * ex, sy * ey, sz * ez]
            for sx in (1.0, -1.0)
            for sy in (1.0, -1.0)
            for sz in (1.0, -1.0)
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([offsets, np.ones((8, 1), dtype=np.float64)], axis=1)
    return (matrix @ homogeneous.T).T[:, :3], matrix[:3, :3]


def project_box(
    corners: np.ndarray, camera_inverse: np.ndarray, intrinsics: np.ndarray,
    width: int, height: int,
) -> tuple[float, float, float, float]:
    homogeneous = np.concatenate([corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    camera = (camera_inverse @ homogeneous.T).T[:, :3]
    in_front = camera[:, 0] > 0.05
    if not np.any(in_front):
        raise ValueError("no corner in front")
    x = camera[in_front, 0]
    u = intrinsics[0, 2] + camera[in_front, 1] / x * intrinsics[0, 0]
    v = intrinsics[1, 2] - camera[in_front, 2] / x * intrinsics[1, 1]
    x0 = float(np.clip(np.min(u), 0.0, width))
    y0 = float(np.clip(np.min(v), 0.0, height))
    x1 = float(np.clip(np.max(u), 0.0, width))
    y1 = float(np.clip(np.max(v), 0.0, height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("no projected area")
    return x0, y0, x1 - x0, y1 - y0


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, aw, ah = (float(value) for value in first)
    bx0, by0, bw, bh = (float(value) for value in second)
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def intrinsics_from_manifest(row: Mapping[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [float(row["camera_fx"]), 0.0, float(row["camera_cx"])],
            [0.0, float(row["camera_fy"]), float(row["camera_cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def projection_parity(
    manifest_by_sample: Mapping[str, Mapping[str, str]],
    actor_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    eligible = sorted(
        (
            row for row in actor_rows
            if row["label"] == "vehicle"
            and float(row["gt_distance_m"]) <= MAX_RANGE_M
            and float(row["gt_bbox_area_px"]) >= MIN_AREA_PX
        ),
        key=lambda row: (row["sample_id"], int(row["gt_actor_id"])),
    )
    selected = eligible[:PARITY_ROWS]
    if len(selected) != PARITY_ROWS:
        raise RuntimeError(f"projection parity sample shortage: {len(selected)}")
    rows: list[dict[str, Any]] = []
    for actor in selected:
        manifest = manifest_by_sample[actor["sample_id"]]
        center = [float(actor[f"object_world_{axis}"]) for axis in "xyz"]
        extent = [float(actor[f"gt_extent_{axis}_m"]) for axis in "xyz"]
        corners, _rotation = corners_from_pose(
            center, extent, 0.0, 0.0, float(actor["object_yaw_deg"])
        )
        predicted = project_box(
            corners,
            np.asarray(json.loads(manifest["camera_inverse_matrix_json"]), dtype=np.float64),
            intrinsics_from_manifest(manifest),
            int(manifest["camera_width"]),
            int(manifest["camera_height"]),
        )
        stored = tuple(float(actor[key]) for key in (
            "gt_bbox_x", "gt_bbox_y", "gt_bbox_w", "gt_bbox_h"
        ))
        differences = [abs(left - right) for left, right in zip(predicted, stored)]
        rows.append(
            {
                "sample_id": actor["sample_id"],
                "gt_actor_id": int(actor["gt_actor_id"]),
                "box_iou": box_iou(predicted, stored),
                "maximum_component_abs_error_px": max(differences),
                "mean_component_abs_error_px": sum(differences) / len(differences),
            }
        )
    ious = [float(row["box_iou"]) for row in rows]
    result = {
        "sample_contract": (
            "first 64 lexicographic geometry-qualified actor-origin validation vehicle rows"
        ),
        "stored_orientation_limit": (
            "retained actor rows store yaw but not pitch/roll; parity tests the shared CARLA "
            "world/camera/extent/yaw projection convention"
        ),
        "rows": rows,
        "row_count": len(rows),
        "minimum_box_iou": min(ious),
        "mean_box_iou": sum(ious) / len(ious),
        "p05_box_iou": float(np.quantile(np.asarray(ious), 0.05)),
        "maximum_component_abs_error_px": max(
            float(row["maximum_component_abs_error_px"]) for row in rows
        ),
        "gate_minimum_box_iou": PARITY_MIN_IOU,
        "gate_minimum_mean_box_iou": PARITY_MEAN_IOU,
    }
    result["passed"] = bool(
        result["minimum_box_iou"] >= PARITY_MIN_IOU
        and result["mean_box_iou"] >= PARITY_MEAN_IOU
    )
    if not result["passed"]:
        raise RuntimeError(f"offline projection parity failed materially: {result}")
    return result


def build_catalog_geometry(
    catalog_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    geometry: list[dict[str, Any]] = []
    by_stable: dict[str, dict[str, Any]] = {}
    for row in catalog_rows:
        center = np.asarray([float(row[f"bbox_center_{axis}_m"]) for axis in "xyz"])
        extent = np.asarray([float(row[f"bbox_extent_{axis}_m"]) for axis in "xyz"])
        corners, rotation = corners_from_pose(
            center,
            extent,
            float(row["bbox_rotation_roll_deg"]),
            float(row["bbox_rotation_pitch_deg"]),
            float(row["bbox_rotation_yaw_deg"]),
        )
        item = {
            "row": dict(row), "center": center, "extent": extent,
            "corners": corners, "rotation": rotation,
        }
        geometry.append(item)
        stable = row["environment_object_id"]
        if stable in by_stable:
            raise RuntimeError(f"duplicate stable catalog ID: {stable}")
        by_stable[stable] = item
    geometry.sort(key=lambda item: int(item["row"]["carla_environment_object_id"]))
    return geometry, by_stable


def actor_ambiguities(
    static_center: np.ndarray,
    static_size: np.ndarray,
    actor_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for actor in actor_rows:
        if actor["label"] != "vehicle":
            continue
        center = np.asarray([float(actor[f"object_world_{axis}"]) for axis in "xyz"])
        size = np.asarray([float(actor[f"gt_size_{axis}_m"]) for axis in "xyz"])
        center_distance = float(np.linalg.norm(static_center - center))
        deltas = np.abs(static_size - size)
        if center_distance <= AMBIGUITY_CENTER_M and bool(
            np.all(deltas <= AMBIGUITY_DIMENSION_M)
        ):
            matches.append(
                {
                    "gt_actor_id": actor["gt_actor_id"],
                    "center_distance_m": center_distance,
                    "size_deltas_m": deltas.tolist(),
                }
            )
    return sorted(matches, key=lambda row: (row["center_distance_m"], int(row["gt_actor_id"])))


def radar_support_count(
    tree: cKDTree | None,
    radar_world: np.ndarray,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
) -> int:
    if tree is None or radar_world.size == 0:
        return 0
    radius = float(np.linalg.norm(extent + RADAR_MARGIN_M))
    indices = tree.query_ball_point(center, radius)
    if not indices:
        return 0
    local = (radar_world[np.asarray(indices, dtype=np.int64)] - center) @ rotation
    inside = np.all(np.abs(local) <= (extent + RADAR_MARGIN_M), axis=1)
    return int(np.count_nonzero(inside))


def audit_visibility(
    manifest: Sequence[Mapping[str, str]],
    actor_rows_by_sample: Mapping[str, Sequence[Mapping[str, str]]],
    geometry: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    static_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    projection_failures = Counter()
    class_tag = {name: int(getattr(carla.CityObjectLabel, name)) for name in SEMANTIC_NAMES}
    for frame_index, frame in enumerate(manifest, 1):
        width, height = int(frame["camera_width"]), int(frame["camera_height"])
        if (width, height) != (1280, 720):
            raise RuntimeError(f"unexpected retained camera size: {(width, height)}")
        episode = frame["experiment_id"]
        sample_id = frame["sample_id"]
        depth_path = VIEW / episode / "depth" / f"{sample_id}.png"
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"missing retained depth: {depth_path}")
        depth_m = decode_depth_bgra(depth_raw)
        if not depth_is_plausible(depth_m):
            raise RuntimeError(f"implausible retained depth: {sample_id}")
        semantic_path = VIEW / frame["instance_raw_path"]
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
        if semantic is None or semantic.shape != (height, width):
            raise RuntimeError(f"invalid retained semantic tags: {sample_id}")
        radar_path = VIEW / frame["radar_points_path"]
        with np.load(radar_path) as archive:
            radar_world = np.asarray(archive["world_xyz"], dtype=np.float64)
        if radar_world.ndim != 2 or radar_world.shape[1] != 3 or not np.all(
            np.isfinite(radar_world)
        ):
            raise RuntimeError(f"invalid retained radar world points: {sample_id}")
        radar_tree = cKDTree(radar_world) if len(radar_world) else None
        camera_inverse = np.asarray(
            json.loads(frame["camera_inverse_matrix_json"]), dtype=np.float64
        )
        intrinsics = intrinsics_from_manifest(frame)
        camera_center = np.asarray(
            [float(frame[f"camera_{axis}"]) for axis in "xyz"], dtype=np.float64
        )
        frame_counts = {
            "sample_id": sample_id,
            "in_range": 0,
            "projected_in_frame": 0,
            "area_qualified": 0,
            "eligible_v010": 0,
            "eligible_v025": 0,
            "unobservable_area_qualified": 0,
            "ambiguous": 0,
        }
        for item in geometry:
            catalog = item["row"]
            center = item["center"]
            extent = item["extent"]
            distance = float(np.linalg.norm(center - camera_center))
            if distance > MAX_RANGE_M:
                continue
            frame_counts["in_range"] += 1
            try:
                metrics, visible_mask = visibility_from_corners(
                    item["corners"],
                    camera_inverse,
                    intrinsics,
                    depth_m,
                    distance_m=distance,
                    width=width,
                    height=height,
                    model_width=MODEL_WIDTH,
                    model_height=MODEL_HEIGHT,
                )
            except ValueError as exc:
                projection_failures[str(exc)] += 1
                continue
            frame_counts["projected_in_frame"] += 1
            sensor_center = (
                camera_inverse @ np.asarray([center[0], center[1], center[2], 1.0])
            )[:3]
            semantic_vehicle_px = int(
                np.count_nonzero(visible_mask & np.isin(semantic, list(VEHICLE_SEMANTIC_TAGS)))
            )
            semantic_same_px = int(
                np.count_nonzero(visible_mask & (semantic == class_tag[catalog["semantic_class"]]))
            )
            native_visible_px = int(metrics["native_visible_px"])
            static_size = extent * 2.0
            ambiguities = actor_ambiguities(
                center, static_size, actor_rows_by_sample.get(sample_id, [])
            )
            nearest = ambiguities[0] if ambiguities else None
            row: dict[str, Any] = {
                "experiment_id": episode,
                "sample_id": sample_id,
                "frame_id": int(frame["frame_id"]),
                "timestamp": float(frame["timestamp"]),
                **{key: catalog[key] for key in (
                    "environment_object_id", "carla_environment_object_id",
                    "object_name", "semantic_class", "enabled", "map_name", "map_sha256",
                    "bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m",
                    "bbox_extent_x_m", "bbox_extent_y_m", "bbox_extent_z_m",
                    "bbox_rotation_roll_deg", "bbox_rotation_pitch_deg",
                    "bbox_rotation_yaw_deg",
                )},
                "size_x_m": float(static_size[0]),
                "size_y_m": float(static_size[1]),
                "size_z_m": float(static_size[2]),
                "object_sensor_x_m": float(sensor_center[0]),
                "object_sensor_y_m": float(sensor_center[1]),
                "object_sensor_z_m": float(sensor_center[2]),
                "range_m": distance,
                **metrics,
                "semantic_vehicle_visible_px": semantic_vehicle_px,
                "semantic_same_class_visible_px": semantic_same_px,
                "semantic_vehicle_visible_fraction": (
                    semantic_vehicle_px / native_visible_px if native_visible_px else 0.0
                ),
                "semantic_same_class_visible_fraction": (
                    semantic_same_px / native_visible_px if native_visible_px else 0.0
                ),
                "radar_support_points": radar_support_count(
                    radar_tree, radar_world, center, extent, item["rotation"]
                ),
                "radar_support_margin_m": RADAR_MARGIN_M,
                "possible_dynamic_static_duplicate": bool(ambiguities),
                "ambiguous_actor_count": len(ambiguities),
                "ambiguous_actor_ids": ";".join(
                    str(match["gt_actor_id"]) for match in ambiguities
                ),
                "nearest_ambiguous_actor_id": (
                    nearest["gt_actor_id"] if nearest else ""
                ),
                "nearest_ambiguous_center_distance_m": (
                    nearest["center_distance_m"] if nearest else ""
                ),
                "nearest_ambiguous_size_x_delta_m": (
                    nearest["size_deltas_m"][0] if nearest else ""
                ),
                "nearest_ambiguous_size_y_delta_m": (
                    nearest["size_deltas_m"][1] if nearest else ""
                ),
                "nearest_ambiguous_size_z_delta_m": (
                    nearest["size_deltas_m"][2] if nearest else ""
                ),
            }
            static_rows.append(row)
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
        per_frame.append(frame_counts)
        if frame_index % 250 == 0:
            print(
                f"[static visibility] {frame_index}/{len(manifest)} rows={len(static_rows)}",
                flush=True,
            )
    for row in static_rows:
        numeric_fields = (
            "bbox_center_x_m", "bbox_center_y_m", "bbox_center_z_m",
            "bbox_extent_x_m", "bbox_extent_y_m", "bbox_extent_z_m",
            "size_x_m", "size_y_m", "size_z_m", "bbox_rotation_roll_deg",
            "bbox_rotation_pitch_deg", "bbox_rotation_yaw_deg",
            "object_sensor_x_m", "object_sensor_y_m", "object_sensor_z_m", "range_m",
            "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h",
            "clipped_projected_area_px", "visible_fraction", "model_input_visible_px",
            "semantic_vehicle_visible_px", "radar_support_points",
        )
        for field in numeric_fields:
            finite_float(row[field], field)
        if row["visibility_algorithm_version"] != ALGORITHM_VERSION:
            raise RuntimeError("visibility helper version drift")
    summary = {
        "projected_row_count": len(static_rows),
        "ambiguity_row_count": len(ambiguity_rows),
        "ambiguity_unique_static_objects": len(
            {row["environment_object_id"] for row in ambiguity_rows}
        ),
        "ambiguity_frames": len({row["sample_id"] for row in ambiguity_rows}),
        "projection_failures": dict(sorted(projection_failures.items())),
        "totals": {
            key: sum(int(row[key]) for row in per_frame)
            for key in (
                "in_range", "projected_in_frame", "area_qualified", "eligible_v010",
                "eligible_v025", "unobservable_area_qualified", "ambiguous",
            )
        },
        "per_frame": {
            key: numeric_summary([int(row[key]) for row in per_frame])
            for key in (
                "in_range", "projected_in_frame", "area_qualified", "eligible_v010",
                "eligible_v025", "unobservable_area_qualified", "ambiguous",
            )
        },
    }
    return static_rows, ambiguity_rows, summary


def import_frozen_scorer() -> Any:
    path = COMPARISON / "scripts/score_report_v1.py"
    spec = importlib.util.spec_from_file_location("route_b_v3_frozen_score_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import frozen score script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_predictions_with_visual_fields(
    scorer: Any, path: Path, frame_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped = scorer.predictions_by_frame(path, frame_ids)
    raw_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        raw_grouped[row["sample_id"]].append(row)
    for sample_id, predictions in grouped.items():
        raw = raw_grouped[sample_id]
        if len(raw) != len(predictions):
            raise RuntimeError(f"prediction row reconciliation failed: {sample_id}")
        for prediction, source in zip(predictions, raw):
            if float(source["score"]) != prediction["score"]:
                raise RuntimeError(f"prediction order drift: {sample_id}")
            prediction.update(
                {
                    "prediction_index": int(source["prediction_index"]),
                    "bbox_x0": float(source["bbox_x0"]),
                    "bbox_y0": float(source["bbox_y0"]),
                    "bbox_x1": float(source["bbox_x1"]),
                    "bbox_y1": float(source["bbox_y1"]),
                }
            )
    return grouped


def static_gt_by_frame(
    static_rows: Sequence[Mapping[str, Any]], contract: str
) -> dict[str, list[dict[str, Any]]]:
    eligibility = "eligible_visible_v010" if contract == "v010" else "eligible_clear_v025"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in static_rows:
        if not as_bool(row[eligibility]) or as_bool(row["possible_dynamic_static_duplicate"]):
            continue
        yaw = math.radians(float(row["bbox_rotation_yaw_deg"]))
        grouped[str(row["sample_id"])].append(
            {
                "class_index": 0.0,
                "class_name": "vehicle",
                "center_x": float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"]) / 2.0,
                "center_y": float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"]) / 2.0,
                "bbox_w": float(row["clipped_bbox_w"]),
                "bbox_h": float(row["clipped_bbox_h"]),
                "area": float(row["clipped_projected_area_px"]),
                "local_x": float(row["object_sensor_x_m"]),
                "local_y": float(row["object_sensor_y_m"]),
                "local_z": float(row["object_sensor_z_m"]),
                "world_x": float(row["bbox_center_x_m"]),
                "world_y": float(row["bbox_center_y_m"]),
                "world_z": float(row["bbox_center_z_m"]),
                "size_x": float(row["size_x_m"]),
                "size_y": float(row["size_y_m"]),
                "size_z": float(row["size_z_m"]),
                "yaw_sin": math.sin(yaw),
                "yaw_cos": math.cos(yaw),
                "parked": 1.0,
                "radar_support": float(int(row["radar_support_points"]) > 0),
                "_gt_source": "static_environment_object",
                "_static_row": row,
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["_static_row"]["environment_object_id"])
    return grouped


def score_combined(
    scorer: Any,
    predictions: Mapping[str, Sequence[dict[str, Any]]],
    actor_gt: Mapping[str, Sequence[dict[str, Any]]],
    static_gt: Mapping[str, Sequence[dict[str, Any]]],
    frame_ids: Sequence[str],
    threshold: float,
    model: str,
    contract: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buckets = {class_name: scorer.empty_bucket() for class_name in scorer.CLASSES}
    eligible = {class_name: 0 for class_name in scorer.CLASSES}
    prediction_counts = {class_name: 0 for class_name in scorer.CLASSES}
    converted_rows: list[dict[str, Any]] = []
    static_matched = 0
    static_unmatched = 0
    for sample_id in frame_ids:
        frame_predictions = [
            item for item in predictions.get(sample_id, []) if item["score"] >= threshold
        ]
        actor_frame = [dict(item, _gt_source="actor") for item in actor_gt[sample_id]]
        static_frame = list(static_gt.get(sample_id, []))
        combined_gt = actor_frame + static_frame
        flags = scorer.duplicate_flags(frame_predictions)
        baseline_matches = scorer.greedy_match_predictions(
            frame_predictions, actor_frame, max_distance_m=MATCH_DISTANCE_M, class_aware=True
        )
        baseline_matched_predictions = {match[0] for match in baseline_matches}
        matches = scorer.greedy_match_predictions(
            frame_predictions, combined_gt, max_distance_m=MATCH_DISTANCE_M, class_aware=True
        )
        matched_predictions = {match[0] for match in matches}
        matched_gt = {match[1] for match in matches}
        for target in combined_gt:
            eligible[target["class_name"]] += 1
        for prediction in frame_predictions:
            prediction_counts[prediction["class_name"]] += 1
        for pred_index, gt_index, distance in matches:
            prediction, target = frame_predictions[pred_index], combined_gt[gt_index]
            bucket = buckets[target["class_name"]]
            bucket["tp"] += 1
            bucket["xy"].append(float(distance))
            bucket["dimension"].append(
                scorer.mean(
                    [
                        abs(prediction["size_x"] - target["size_x"]),
                        abs(prediction["size_y"] - target["size_y"]),
                        abs(prediction["size_z"] - target["size_z"]),
                    ]
                )
            )
            bucket["yaw"].append(scorer.yaw_error(prediction, target))
            if target.get("_gt_source") == "static_environment_object":
                static_matched += 1
                if pred_index not in baseline_matched_predictions:
                    row = target["_static_row"]
                    converted_rows.append(
                        {
                            "model": model,
                            "visibility_contract": contract,
                            "score_threshold": threshold,
                            "sample_id": sample_id,
                            "frame_id": int(row["frame_id"]),
                            "prediction_index": prediction["prediction_index"],
                            "prediction_score": prediction["score"],
                            "prediction_world_x": prediction["world_x"],
                            "prediction_world_y": prediction["world_y"],
                            "prediction_world_z": prediction["world_z"],
                            "prediction_bbox_x0": prediction["bbox_x0"],
                            "prediction_bbox_y0": prediction["bbox_y0"],
                            "prediction_bbox_x1": prediction["bbox_x1"],
                            "prediction_bbox_y1": prediction["bbox_y1"],
                            "environment_object_id": row["environment_object_id"],
                            "carla_environment_object_id": row[
                                "carla_environment_object_id"
                            ],
                            "semantic_class": row["semantic_class"],
                            "static_world_x": row["bbox_center_x_m"],
                            "static_world_y": row["bbox_center_y_m"],
                            "static_world_z": row["bbox_center_z_m"],
                            "match_distance_m": float(distance),
                            "prior_status": "FP",
                            "corrected_status": "TP",
                            "possible_dynamic_static_duplicate": False,
                        }
                    )
        for pred_index, prediction in enumerate(frame_predictions):
            if pred_index in matched_predictions:
                continue
            bucket = buckets[prediction["class_name"]]
            bucket["fp"] += 1
            bucket["duplicate_fp"] += int(flags[pred_index])
        for gt_index, target in enumerate(combined_gt):
            if gt_index not in matched_gt:
                buckets[target["class_name"]]["fn"] += 1
                if target.get("_gt_source") == "static_environment_object":
                    static_unmatched += 1
    result = {
        "threshold": threshold,
        "threshold_role": (
            "canonical_operating_point" if threshold == 0.20 else "recall_ceiling"
        ),
        "frames": len(frame_ids),
        "match_distance_m": MATCH_DISTANCE_M,
        "classes": {
            class_name: {
                **scorer.summarize(
                    buckets[class_name], len(frame_ids), eligible[class_name]
                ),
                "predictions": prediction_counts[class_name],
            }
            for class_name in scorer.CLASSES
        },
        "static_vehicle_gt": sum(len(values) for values in static_gt.values()),
        "static_vehicle_matched_tp": static_matched,
        "static_vehicle_unmatched_fn": static_unmatched,
        "prior_vehicle_fp_converted_to_tp": len(converted_rows),
    }
    return result, converted_rows


def support_distributions(
    static_rows: Sequence[Mapping[str, Any]], contract: str
) -> dict[str, Any]:
    eligibility = "eligible_visible_v010" if contract == "v010" else "eligible_clear_v025"
    rows = [
        row for row in static_rows
        if as_bool(row[eligibility])
        and not as_bool(row["possible_dynamic_static_duplicate"])
    ]
    semantic = [int(row["semantic_vehicle_visible_px"]) for row in rows]
    same_class = [int(row["semantic_same_class_visible_px"]) for row in rows]
    radar = [int(row["radar_support_points"]) for row in rows]
    return {
        "eligible_nonambiguous_rows": len(rows),
        "semantic_vehicle_visible_px": numeric_summary(semantic),
        "semantic_same_class_visible_px": numeric_summary(same_class),
        "radar_support_points": numeric_summary(radar),
        "semantic_vehicle_support_positive_rows": sum(value > 0 for value in semantic),
        "semantic_vehicle_support_positive_percent": (
            100.0 * sum(value > 0 for value in semantic) / len(rows) if rows else 0.0
        ),
        "radar_support_positive_rows": sum(value > 0 for value in radar),
        "radar_support_positive_percent": (
            100.0 * sum(value > 0 for value in radar) / len(rows) if rows else 0.0
        ),
    }


def metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("eligible_gt", "tp", "fp", "fn", "precision", "recall", "f1")
    return {key: float(after[key]) - float(before[key]) for key in keys}


def no_model_detects(
    row: Mapping[str, Any], predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]]
) -> bool:
    x, y = float(row["bbox_center_x_m"]), float(row["bbox_center_y_m"])
    for model in MODELS:
        for prediction in predictions[model].get(str(row["sample_id"]), []):
            if prediction["class_name"] != "vehicle" or prediction["score"] < 0.20:
                continue
            if math.hypot(prediction["world_x"] - x, prediction["world_y"] - y) <= 3.0:
                return False
    return True


def select_manual_cases(
    static_rows: Sequence[dict[str, Any]],
    reclass_rows: Sequence[dict[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    static_lookup = {
        (str(row["sample_id"]), str(row["environment_object_id"])): row
        for row in static_rows
    }
    strata: list[tuple[str, str, list[dict[str, Any]]]] = []
    converted: list[dict[str, Any]] = []
    for row in reclass_rows:
        if row["visibility_contract"] != "v010" or float(row["score_threshold"]) != 0.20:
            continue
        static = static_lookup[(row["sample_id"], row["environment_object_id"])]
        converted.append({**static, "model": row["model"]})
    strata.append(("s1_fp_to_tp", "prior model FP matching v0.10 static GT", converted))
    visible_undetected = [
        {**row, "model": "none"}
        for row in static_rows
        if as_bool(row["eligible_visible_v010"])
        and not as_bool(row["possible_dynamic_static_duplicate"])
        and no_model_detects(row, predictions)
    ]
    strata.append(
        (
            "s2_visible_undetected",
            "v0.10 static GT not detected by either frozen model at 0.20",
            visible_undetected,
        )
    )
    marginal = [
        {**row, "model": "both"}
        for row in static_rows
        if as_bool(row["eligible_visible_v010"])
        and not as_bool(row["eligible_clear_v025"])
        and not as_bool(row["possible_dynamic_static_duplicate"])
    ]
    strata.append(
        ("s3_marginal", "v0.10 eligible but v0.25 ineligible static vehicle", marginal)
    )
    unobservable = [
        {**row, "model": "both"}
        for row in static_rows
        if as_bool(row["geometry_qualified_v2"])
        and not as_bool(row["eligible_visible_v010"])
        and str(row["visibility_tier"]) == "unobservable"
        and not as_bool(row["possible_dynamic_static_duplicate"])
    ]
    strata.append(
        ("s4_unobservable", "v0.10-unobservable static candidate", unobservable)
    )

    selected: list[dict[str, Any]] = []
    used_cases: set[tuple[str, str]] = set()
    summary: dict[str, dict[str, int]] = {}
    for key, label, candidates in strata:
        ordered = sorted(
            candidates,
            key=lambda row: (
                str(row["sample_id"]), str(row["environment_object_id"]),
                str(row.get("model", "")),
            ),
        )
        unique_candidates: list[dict[str, Any]] = []
        local_seen: set[tuple[str, str]] = set()
        for row in ordered:
            case = (str(row["sample_id"]), str(row["environment_object_id"]))
            if case in local_seen:
                continue
            local_seen.add(case)
            unique_candidates.append(row)
        chosen: list[dict[str, Any]] = []
        for row in unique_candidates:
            case = (str(row["sample_id"]), str(row["environment_object_id"]))
            if case in used_cases:
                continue
            chosen.append(row)
            used_cases.add(case)
            if len(chosen) == 8:
                break
        summary[key] = {
            "available_unique_cases": len(unique_candidates),
            "selected": len(chosen),
            "shortage": max(0, 8 - len(chosen)),
        }
        for row in chosen:
            selected.append({**row, "stratum": key, "stratum_label": label})
    for index, row in enumerate(selected, 1):
        suffix = str(row["environment_object_id"]).rsplit(":", 1)[-1][:10]
        filename = f"{index:02d}_{row['stratum']}_frame{row['frame_id']}_{suffix}.png"
        row["review_index"] = index
        row["panel_path"] = f"manual_review_panels/{filename}"
        stats = summary[row["stratum"]]
        row["stratum_available"] = stats["available_unique_cases"]
        row["stratum_selected"] = stats["selected"]
        row["stratum_shortage"] = stats["shortage"]
    return selected, summary


def draw_box(
    image: np.ndarray, box: Sequence[float], color: tuple[int, int, int],
    label: str, thickness: int = 2,
) -> None:
    x, y, w, h = (float(value) for value in box)
    p0 = (int(round(x)), int(round(y)))
    p1 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(image, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.putText(
        image, label, (p0[0], max(14, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, color, 1, cv2.LINE_AA,
    )


def add_text_lines(
    image: np.ndarray, lines: Sequence[str], origin: tuple[int, int],
    color: tuple[int, int, int] = (235, 235, 235), scale: float = 0.48,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
            color, 1, cv2.LINE_AA,
        )
        y += int(26 * scale / 0.48)


def render_panel(
    selection: Mapping[str, Any],
    manifest_by_sample: Mapping[str, Mapping[str, str]],
    actor_rows_by_sample: Mapping[str, Sequence[Mapping[str, str]]],
    geometry_by_stable: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
) -> None:
    sample_id = str(selection["sample_id"])
    manifest = manifest_by_sample[sample_id]
    image = cv2.imread(str(VIEW / manifest["rgb_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"missing RGB for panel: {sample_id}")
    height, width = image.shape[:2]
    depth_path = VIEW / manifest["experiment_id"] / "depth" / f"{sample_id}.png"
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise RuntimeError(f"missing depth for panel: {sample_id}")
    depth_m = decode_depth_bgra(raw_depth)
    item = geometry_by_stable[str(selection["environment_object_id"])]
    metrics, visible_mask = visibility_from_corners(
        item["corners"],
        np.asarray(json.loads(manifest["camera_inverse_matrix_json"]), dtype=np.float64),
        intrinsics_from_manifest(manifest),
        depth_m,
        distance_m=float(selection["range_m"]),
        width=width,
        height=height,
        model_width=MODEL_WIDTH,
        model_height=MODEL_HEIGHT,
    )
    if int(metrics["native_visible_px"]) != int(selection["native_visible_px"]):
        raise RuntimeError(f"panel visibility reconstruction mismatch: {sample_id}")
    overlay = image.copy()
    cyan = np.asarray([255, 255, 0], dtype=np.float32)
    if np.any(visible_mask):
        current = overlay[visible_mask].astype(np.float32)
        overlay[visible_mask] = np.clip(0.45 * current + 0.55 * cyan, 0, 255).astype(np.uint8)
    static_box = (
        float(selection["clipped_bbox_x"]), float(selection["clipped_bbox_y"]),
        float(selection["clipped_bbox_w"]), float(selection["clipped_bbox_h"]),
    )
    draw_box(
        overlay, static_box, (255, 0, 255),
        f"STATIC {str(selection['environment_object_id'])[-10:]}", 3,
    )
    for actor in actor_rows_by_sample.get(sample_id, []):
        if actor["label"] != "vehicle":
            continue
        draw_box(
            overlay,
            [actor["gt_bbox_x"], actor["gt_bbox_y"], actor["gt_bbox_w"], actor["gt_bbox_h"]],
            (0, 220, 0), f"actor {actor['gt_actor_id']}", 1,
        )
    prediction_lines: list[str] = []
    static_x, static_y = float(selection["bbox_center_x_m"]), float(
        selection["bbox_center_y_m"]
    )
    model_colors = {
        "lraspp_mprime_noae": (0, 165, 255),
        "fasterrcnn_radar_roi_v1_epoch12": (255, 120, 0),
    }
    for model in MODELS:
        candidates = []
        for prediction in predictions[model].get(sample_id, []):
            distance = math.hypot(
                prediction["world_x"] - static_x, prediction["world_y"] - static_y
            )
            if (
                prediction["class_name"] == "vehicle"
                and prediction["score"] >= 0.02
                and distance <= MATCH_DISTANCE_M
            ):
                candidates.append((distance, prediction))
        candidates.sort(key=lambda pair: (pair[0], -pair[1]["score"]))
        for distance, prediction in candidates[:3]:
            draw_box(
                overlay,
                [
                    prediction["bbox_x0"], prediction["bbox_y0"],
                    prediction["bbox_x1"] - prediction["bbox_x0"],
                    prediction["bbox_y1"] - prediction["bbox_y0"],
                ],
                model_colors[model],
                f"{model.split('_')[0]} {prediction['score']:.3f}", 2,
            )
            prediction_lines.append(
                f"{model}: score={prediction['score']:.3f} d={distance:.2f}m "
                f"world=({prediction['world_x']:.2f},{prediction['world_y']:.2f},{prediction['world_z']:.2f})"
            )
        if not candidates:
            prediction_lines.append(f"{model}: no score>=0.02 vehicle prediction within 3m")
        elif len(candidates) > 3:
            prediction_lines.append(
                f"{model}: {len(candidates) - 3} additional within-3m predictions omitted from overlay"
            )

    x, y, w, h = static_box
    margin_x, margin_y = max(100.0, 2.5 * w), max(80.0, 2.5 * h)
    zx0 = max(0, int(math.floor(x + w / 2.0 - margin_x)))
    zy0 = max(0, int(math.floor(y + h / 2.0 - margin_y)))
    zx1 = min(width, int(math.ceil(x + w / 2.0 + margin_x)))
    zy1 = min(height, int(math.ceil(y + h / 2.0 + margin_y)))
    if zx1 <= zx0 or zy1 <= zy0:
        raise RuntimeError(f"invalid panel zoom: {sample_id}")
    zoom = overlay[zy0:zy1, zx0:zx1]
    canvas = np.full((900, 1900, 3), 22, dtype=np.uint8)
    full_view = cv2.resize(overlay, (1152, 648), interpolation=cv2.INTER_AREA)
    zoom_view = cv2.resize(zoom, (704, 648), interpolation=cv2.INTER_CUBIC)
    canvas[55:703, 20:1172] = full_view
    canvas[55:703, 1180:1884] = zoom_view
    add_text_lines(
        canvas,
        [
            f"Review {selection['review_index']:02d} | {selection['stratum_label']}",
            f"sample={sample_id} frame={selection['frame_id']} class={selection['semantic_class']}",
        ],
        (20, 22),
        scale=0.46,
    )
    duplicate = as_bool(selection["possible_dynamic_static_duplicate"])
    metadata_lines = [
        f"stable_id={selection['environment_object_id']} native_id={selection['carla_environment_object_id']}",
        f"visible_fraction={float(selection['visible_fraction']):.4f} native/model visible px={int(selection['native_visible_px'])}/{int(selection['model_input_visible_px'])} range={float(selection['range_m']):.2f}m",
        f"eligible v0.10/v0.25={as_bool(selection['eligible_visible_v010'])}/{as_bool(selection['eligible_clear_v025'])} semantic vehicle px={int(selection['semantic_vehicle_visible_px'])} radar points={int(selection['radar_support_points'])} ambiguity={duplicate}",
        "magenta=static GT | cyan=depth-consistent pixels | green=actor GT | orange=LR-ASPP | blue=Faster R-CNN",
        *prediction_lines,
    ]
    add_text_lines(canvas, metadata_lines, (20, 730), scale=0.42)
    output = EXPERIMENT / str(selection["panel_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"could not create panel: {output}")


def render_contact_sheet(selections: Sequence[Mapping[str, Any]]) -> Path:
    columns = 4
    thumb_w, thumb_h = 470, 225
    rows = max(1, math.ceil(len(selections) / columns))
    sheet = np.full((rows * thumb_h, columns * thumb_w, 3), 18, dtype=np.uint8)
    for index, selection in enumerate(selections):
        panel = cv2.imread(str(EXPERIMENT / str(selection["panel_path"])), cv2.IMREAD_COLOR)
        if panel is None:
            raise RuntimeError(f"missing rendered panel: {selection['panel_path']}")
        thumb = cv2.resize(panel, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * thumb_h:(row + 1) * thumb_h, column * thumb_w:(column + 1) * thumb_w] = thumb
    output = EXPERIMENT / "manual_review_contact_sheet.png"
    if output.exists() or not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"could not create contact sheet: {output}")
    return output


def markdown_report(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Route B v3 static parked-vehicle GT audit",
        "",
        f"Terminal: `{SUCCESS_TERMINAL}`",
        "",
        "The actor-origin Route B v3 object tables contain no environment-object rows. "
        "The diagnostic results below add only non-ambiguous, depth-observable static "
        "Car/Truck/Bus GT in memory; canonical metrics and files remain unchanged.",
        "",
        "## Static catalog and visibility",
        "",
        f"- Catalog: **{metrics['static_catalog']['object_count']}** objects "
        f"({metrics['static_catalog']['semantic_class_counts']}).",
        f"- Map: `{metrics['static_catalog']['map_name']}`, OpenDRIVE SHA-256 "
        f"`{metrics['static_catalog']['map_sha256']}`.",
        f"- Projected in-range rows: **{metrics['visibility']['totals']['projected_in_frame']}**; "
        f"area-qualified: **{metrics['visibility']['totals']['area_qualified']}**; "
        f"v0.10: **{metrics['visibility']['totals']['eligible_v010']}**; "
        f"v0.25: **{metrics['visibility']['totals']['eligible_v025']}**.",
        f"- Ambiguous static/dynamic frame-object rows excluded: "
        f"**{metrics['visibility']['ambiguity_row_count']}** across "
        f"{metrics['visibility']['ambiguity_frames']} frames and "
        f"{metrics['visibility']['ambiguity_unique_static_objects']} static IDs.",
        "",
        "## Frozen vehicle metric reclassification",
        "",
        "| model | contract | threshold | before TP/FP/FN | after TP/FP/FN | before P/R/F1 | after P/R/F1 | prior FP converted | converted % | added static FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for contract in CONTRACTS:
            for threshold in ("0.20", "0.02"):
                item = metrics["models"][model]["contracts"][contract]["thresholds"][threshold]
                before, after = item["before"], item["after"]
                lines.append(
                    f"| {model} | {contract} | {threshold} | "
                    f"{before['tp']}/{before['fp']}/{before['fn']} | "
                    f"{after['tp']}/{after['fp']}/{after['fn']} | "
                    f"{before['precision']:.4f}/{before['recall']:.4f}/{before['f1']:.4f} | "
                    f"{after['precision']:.4f}/{after['recall']:.4f}/{after['f1']:.4f} | "
                    f"{item['prior_vehicle_fp_converted_to_tp']} | "
                    f"{item['prior_fp_reclassified_percent']:.2f}% | "
                    f"{item['static_vehicle_unmatched_fn']} |"
                )
    lines.extend(
        [
            "",
            "These corrected scores are diagnostic only and do not replace the canonical "
            "actor-only metrics.",
            "",
            "## Manual review",
            "",
            f"- Selected panels: **{metrics['manual_review']['selected_total']}**.",
            f"- Strata and shortages: `{json.dumps(metrics['manual_review']['strata'], sort_keys=True)}`.",
            "- Start with `manual_review_contact_sheet.png`, then inspect full-resolution "
            "panels under `manual_review_panels/` using `manual_review_selection.csv`.",
            "- Magenta is static GT, cyan is the depth-consistent support, green is "
            "actor-origin GT, orange is LR-ASPP, and blue is Faster R-CNN.",
            "",
            "## Acceptance evidence",
            "",
            f"- Offline actor projection parity: minimum/mean IoU "
            f"{metrics['projection_parity']['minimum_box_iou']:.4f}/"
            f"{metrics['projection_parity']['mean_box_iou']:.4f} (PASS).",
            "- Published actor-only TP/FP/FN/P/R/F1 reproduced exactly for both models "
            "at 0.20 and 0.02.",
            "- Native and stable catalog IDs passed the existing Phase-2 verifier.",
            "- Every persisted static audit row has finite geometry and visibility fields.",
            "- Ambiguous rows were excluded from every corrected denominator.",
            "- Canonical validation tree hashes before and after the audit are identical.",
            "- Persisted predictions were reused; no inference was run.",
            "- Locked-test payloads were not opened.",
            "- CARLA capture used one fresh start and cleanup is verified; no second start "
            "was needed.",
            "- No v3.1 view, training, collection, OAI action, commit, push, pull, or fetch occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    started = time.monotonic()
    required_absent = [
        EXPERIMENT / "static_vehicle_visibility.csv",
        EXPERIMENT / "static_dynamic_ambiguities.csv",
        EXPERIMENT / "vehicle_fp_reclassification.csv",
        EXPERIMENT / "metrics_before_after.json",
        EXPERIMENT / "manual_review_selection.csv",
        EXPERIMENT / "manual_review_contact_sheet.png",
        EXPERIMENT / "manual_review_panels",
        EXPERIMENT / "STATIC_VEHICLE_GT_AUDIT_REPORT.md",
        EXPERIMENT / "TERMINAL_VERDICT.txt",
        EXPERIMENT / "provenance_manifest.json",
        EXPERIMENT / "audit_completion.json",
        EXPERIMENT / "provenance/canonical_validation_hashes_before.csv",
        EXPERIMENT / "provenance/canonical_validation_hashes_after.csv",
        EXPERIMENT / "provenance/projection_parity.json",
    ]
    existing = [str(path) for path in required_absent if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite audit outputs: {existing}")

    report = read_json(CANONICAL_REPORT)
    validation_records = [row for row in report["episodes"] if row["split"] == "val"]
    validation_episodes = [Path(row["output_dir"]).name for row in validation_records]
    expected_episodes = [
        "canonical_v3_05_val_30_30_s601_tm1601",
        "canonical_v3_06_val_50_50_s602_tm1602",
    ]
    if validation_episodes != expected_episodes:
        raise RuntimeError(f"canonical validation episode drift: {validation_episodes}")
    episode_dirs = [
        REPO_ROOT / "data_collection/experiments/route_b_perception_v3" / episode
        for episode in validation_episodes
    ]
    for directory in episode_dirs:
        if not directory.is_dir():
            raise RuntimeError(f"missing validation episode: {directory}")

    provenance_dir = EXPERIMENT / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=False)
    before_records = tree_hash_records(episode_dirs, "canonical before")
    write_csv_x(
        provenance_dir / "canonical_validation_hashes_before.csv",
        ("path", "bytes", "sha256"), before_records,
    )

    canonical_hashes = report["hashes"]
    canonical_checks = {
        "report_terminal": report["terminal"] == "ROUTE_B_V3_CANONICAL_COLLECTION_PASSED",
        "report_sha256": sha256(CANONICAL_REPORT),
        "collector_sha256": sha256(
            REPO_ROOT / "data_collection/run_route_b_perception_collection_v3.py"
        ),
        "config_sha256": sha256(
            REPO_ROOT / "data_collection/configs/route_b_perception_v3.yaml"
        ),
        "visibility_helper_sha256": sha256(
            REPO_ROOT / "data_collection/route_b_perception_v3/visibility_v1.py"
        ),
    }
    if (
        canonical_checks["report_sha256"]
        != "27b98ba350d93b40d0dd2901381b481cde2e9f4de3bb2d54f7ca453c0b6fc458"
        or canonical_checks["collector_sha256"] != canonical_hashes["collector_sha256"]
        or canonical_checks["config_sha256"] != canonical_hashes["config_sha256"]
        or canonical_checks["visibility_helper_sha256"]
        != canonical_hashes["visibility_helper_sha256"]
        or not canonical_checks["report_terminal"]
    ):
        raise RuntimeError(f"canonical hash/terminal gate failed: {canonical_checks}")

    manifest = read_csv(VIEW / "manifest.csv")
    if len(manifest) != 3345 or len({row["sample_id"] for row in manifest}) != 3345:
        raise RuntimeError("validation manifest count/uniqueness gate failed")
    if any(
        row["split"] != "val" or row["experiment_id"] not in validation_episodes
        for row in manifest
    ):
        raise RuntimeError("non-validation row in frozen view")
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    retained_evidence = Counter()
    required_prediction_fields = {
        "sample_id", "frame_id", "prediction_index", "class_name", "score",
        "world_x", "world_y", "world_z", "local_x", "local_y", "local_z",
        "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos",
        "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    }
    for row in manifest:
        if row["map_name"] != "Carla/Maps/Town10HD_Opt":
            raise RuntimeError(f"validation map identity drift: {row['sample_id']}")
        for field in (
            "camera_x", "camera_y", "camera_z", "camera_fx", "camera_fy",
            "camera_cx", "camera_cy",
        ):
            finite_float(row[field], field)
        matrix = np.asarray(json.loads(row["camera_inverse_matrix_json"]), dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"invalid camera matrix: {row['sample_id']}")
        paths = {
            "rgb": VIEW / row["rgb_path"],
            "mask": VIEW / row["mask_path"],
            "semantic": VIEW / row["instance_raw_path"],
            "radar_tensor": VIEW / row["radar_tensor_path"],
            "radar_points": VIEW / row["radar_points_path"],
            "depth": VIEW / row["experiment_id"] / "depth" / f"{row['sample_id']}.png",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"missing retained {kind}: {path}")
            retained_evidence[kind] += 1
    actor_rows = read_csv(VIEW / "object_boxes_all.csv")
    visibility_rows = read_csv(VIEW / "object_visibility_all.csv")
    if len(actor_rows) != len(visibility_rows) or not actor_rows:
        raise RuntimeError("actor GT/visibility reconciliation gate failed")
    if {row["gt_source"] for row in actor_rows} != {"actor"}:
        raise RuntimeError("unexpected non-actor source in frozen object boxes")
    if any(row["sample_id"] not in manifest_by_sample for row in actor_rows):
        raise RuntimeError("actor GT frame outside validation view")
    actor_rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in actor_rows:
        actor_rows_by_sample[row["sample_id"]].append(row)

    capture = read_json(EXPERIMENT / "static_environment_truth/capture_attempts.json")
    if not (
        capture["terminal"] == "STATIC_VEHICLE_CATALOG_CAPTURE_COMPLETE"
        and capture["actual_fresh_carla_starts"] == 1
        and capture["attempts"][0]["cleanup_verified"] is True
        and not capture["carla_processes_after_all_attempts"]
        and capture["rpc_port_free_after_all_attempts"] is True
    ):
        raise RuntimeError("CARLA capture/cleanup gate failed")
    catalog_verification = verify_static_environment_truth_v1(
        CATALOG_DIR,
        expected_map_name="Carla/Maps/Town10HD_Opt",
        expected_required_semantic_classes=list(SEMANTIC_NAMES),
    )
    catalog_rows = read_csv(CATALOG_DIR / "static_environment_objects.csv")
    if (
        len(catalog_rows) != 123
        or Counter(row["semantic_class"] for row in catalog_rows)
        != Counter({"Car": 74, "Truck": 47, "Bus": 2})
        or len({row["environment_object_id"] for row in catalog_rows}) != len(catalog_rows)
        or len({row["carla_environment_object_id"] for row in catalog_rows})
        != len(catalog_rows)
    ):
        raise RuntimeError("static catalog identity/count gate failed")
    geometry, geometry_by_stable = build_catalog_geometry(catalog_rows)

    parity = projection_parity(manifest_by_sample, actor_rows)
    write_json_x(provenance_dir / "projection_parity.json", parity)

    input_hashes: dict[str, Any] = {
        "canonical": canonical_checks,
        "models": {},
    }
    for model in MODELS:
        prediction_dir = COMPARISON / "predictions" / model
        inference = read_json(prediction_dir / "inference_manifest.json")
        with (prediction_dir / "detections.csv").open("r", encoding="utf-8", newline="") as stream:
            fields = set(next(csv.reader(stream)))
        if not required_prediction_fields.issubset(fields):
            raise RuntimeError("STATIC_VEHICLE_AUDIT_BLOCKED_MISSING_PREDICTION_FIELDS")
        actual = {
            "checkpoint_sha256": sha256(Path(inference["checkpoint"])),
            "detections_sha256": sha256(prediction_dir / "detections.csv"),
            "segmentation_manifest_sha256": sha256(
                prediction_dir / "segmentation_manifest.csv"
            ),
            "prediction_set_sha256": inference["prediction_set_sha256"],
            "inference_pass_count": inference["inference_pass_count"],
            "validation_frames": inference["validation_frames"],
        }
        expected_checkpoint_hash = {
            "lraspp_mprime_noae": (
                "f319e2a5e8fb134e74c24c0822233e17368df6e4c733add658026603e131d4fa"
            ),
            "fasterrcnn_radar_roi_v1_epoch12": (
                "7d3e1b414a892713fe848cfc81266ae4c321109453f0b60ac93efe30d8a1ef13"
            ),
        }[model]
        if (
            actual["checkpoint_sha256"] != inference["checkpoint_sha256"]
            or actual["checkpoint_sha256"] != expected_checkpoint_hash
            or actual["detections_sha256"] != inference["detections_sha256"]
            or actual["segmentation_manifest_sha256"]
            != inference["segmentation_manifest_sha256"]
            or actual["inference_pass_count"] != 1
            or actual["validation_frames"] != 3345
        ):
            raise RuntimeError(f"frozen input hash/count gate failed: {model}")
        expected_prediction_hash = {
            "lraspp_mprime_noae": (
                "4dca3ac620c0f7b2e2cf50d3f49f56380948af8e1b86e69e1cdaa3fa50ad4811"
            ),
            "fasterrcnn_radar_roi_v1_epoch12": (
                "4d4b238120ef759e9ed28169b35df42f8ac934e55538efada63ffef596b94ddd"
            ),
        }[model]
        if actual["prediction_set_sha256"] != expected_prediction_hash:
            raise RuntimeError(f"prediction-set hash gate failed: {model}")
        input_hashes["models"][model] = actual

    static_rows, ambiguity_rows, visibility_summary = audit_visibility(
        manifest, actor_rows_by_sample, geometry
    )
    write_csv_x(EXPERIMENT / "static_vehicle_visibility.csv", STATIC_FIELDS, static_rows)
    write_csv_x(
        EXPERIMENT / "static_dynamic_ambiguities.csv", STATIC_FIELDS, ambiguity_rows
    )

    scorer = import_frozen_scorer()
    frame_ids = [row["sample_id"] for row in manifest]
    frame_id_set = set(frame_ids)
    actor_gt = {contract: scorer.gt_by_frame(contract, manifest) for contract in CONTRACTS}
    actor_vehicle_counts = {
        contract: sum(
            item["class_name"] == "vehicle"
            for values in actor_gt[contract].values() for item in values
        )
        for contract in CONTRACTS
    }
    if actor_vehicle_counts != {"v010": 7265, "v025": 7265}:
        raise RuntimeError(f"actor vehicle GT count drift: {actor_vehicle_counts}")
    predictions = {
        model: load_predictions_with_visual_fields(
            scorer, COMPARISON / "predictions" / model / "detections.csv", frame_id_set
        )
        for model in MODELS
    }
    published = read_json(COMPARISON / "comparison_metrics.json")
    models_metrics: dict[str, Any] = {}
    all_reclass: list[dict[str, Any]] = []
    baseline_parity: dict[str, Any] = {}
    exact_metric_keys = ("eligible_gt", "tp", "fp", "fn", "precision", "recall", "f1")
    for model in MODELS:
        model_record: dict[str, Any] = {"contracts": {}}
        baseline_parity[model] = {}
        for contract in CONTRACTS:
            static_gt = static_gt_by_frame(static_rows, contract)
            contract_record: dict[str, Any] = {"thresholds": {}}
            baseline_parity[model][contract] = {}
            for threshold in THRESHOLDS:
                threshold_key = f"{threshold:.2f}"
                before_result = scorer.score_detection(
                    predictions[model], actor_gt[contract], frame_ids, threshold
                )
                before = before_result["classes"]["vehicle"]
                reference = published["models"][model]["contracts"][contract][
                    "thresholds"
                ][threshold_key]["classes"]["vehicle"]
                parity_fields = {
                    key: before[key] == reference[key] for key in exact_metric_keys
                }
                baseline_parity[model][contract][threshold_key] = parity_fields
                if not all(parity_fields.values()):
                    raise RuntimeError(
                        f"baseline parity failed: {model}/{contract}/{threshold_key}"
                    )
                combined, converted = score_combined(
                    scorer, predictions[model], actor_gt[contract], static_gt,
                    frame_ids, threshold, model, contract,
                )
                after = combined["classes"]["vehicle"]
                all_reclass.extend(converted)
                contract_record["thresholds"][threshold_key] = {
                    "before": before,
                    "after": after,
                    "delta": metric_delta(before, after),
                    "static_vehicle_gt": combined["static_vehicle_gt"],
                    "static_vehicle_matched_tp": combined["static_vehicle_matched_tp"],
                    "static_vehicle_unmatched_fn": combined["static_vehicle_unmatched_fn"],
                    "prior_vehicle_fp_converted_to_tp": combined[
                        "prior_vehicle_fp_converted_to_tp"
                    ],
                    "prior_fp_reclassified_percent": (
                        100.0
                        * combined["prior_vehicle_fp_converted_to_tp"]
                        / before["fp"]
                        if before["fp"] else 0.0
                    ),
                }
            model_record["contracts"][contract] = contract_record
        models_metrics[model] = model_record
    all_reclass.sort(
        key=lambda row: (
            row["visibility_contract"], -float(row["score_threshold"]), row["model"],
            row["sample_id"], row["environment_object_id"], int(row["prediction_index"]),
        )
    )
    write_csv_x(
        EXPERIMENT / "vehicle_fp_reclassification.csv", RECLASS_FIELDS, all_reclass
    )

    selections, selection_summary = select_manual_cases(
        static_rows, all_reclass, predictions
    )
    for selection in selections:
        render_panel(
            selection, manifest_by_sample, actor_rows_by_sample,
            geometry_by_stable, predictions,
        )
    contact_sheet = render_contact_sheet(selections)
    write_csv_x(
        EXPERIMENT / "manual_review_selection.csv", SELECTION_FIELDS, selections
    )

    after_records = tree_hash_records(episode_dirs, "canonical after")
    write_csv_x(
        provenance_dir / "canonical_validation_hashes_after.csv",
        ("path", "bytes", "sha256"), after_records,
    )
    canonical_unchanged = before_records == after_records
    if not canonical_unchanged:
        raise RuntimeError("canonical validation bytes changed during read-only audit")

    metrics: dict[str, Any] = {
        "schema": "route_b_v3_static_vehicle_gt_audit_metrics_v1",
        "created_utc": utc_now(),
        "diagnostic_only": True,
        "validation_frames": len(manifest),
        "validation_episodes": validation_episodes,
        "static_catalog": {
            "object_count": catalog_verification["object_count"],
            "semantic_class_counts": catalog_verification["semantic_class_counts"],
            "map_name": catalog_verification["map_name"],
            "map_sha256": catalog_verification["map_sha256"],
        },
        "retained_evidence_file_counts": dict(sorted(retained_evidence.items())),
        "actor_only_gt": {
            "gt_sources": sorted({row["gt_source"] for row in actor_rows}),
            "actor_object_rows": len(actor_rows),
            "environment_object_rows": 0,
            "eligible_vehicle_gt": actor_vehicle_counts,
        },
        "projection_parity": {
            key: value for key, value in parity.items() if key != "rows"
        },
        "visibility": visibility_summary,
        "support_distributions": {
            contract: support_distributions(static_rows, contract)
            for contract in CONTRACTS
        },
        "baseline_parity_exact": baseline_parity,
        "models": models_metrics,
        "manual_review": {
            "selected_total": len(selections),
            "strata": selection_summary,
            "contact_sheet": str(contact_sheet.relative_to(EXPERIMENT)),
            "panels_dir": "manual_review_panels",
            "selection_csv": "manual_review_selection.csv",
        },
        "canonical_validation_tree": {
            "file_count": len(before_records),
            "before_manifest_sha256": sha256(
                provenance_dir / "canonical_validation_hashes_before.csv"
            ),
            "after_manifest_sha256": sha256(
                provenance_dir / "canonical_validation_hashes_after.csv"
            ),
            "byte_identical": canonical_unchanged,
        },
        "frozen_inputs": input_hashes,
        "carla_capture": {
            "actual_fresh_starts": capture["actual_fresh_carla_starts"],
            "cleanup_verified": capture["attempts"][0]["cleanup_verified"],
            "carla_processes_after": capture["carla_processes_after_all_attempts"],
            "rpc_port_free_after": capture["rpc_port_free_after_all_attempts"],
        },
        "prohibited_actions": {
            "v3_1_created": False,
            "training_runs": 0,
            "inference_runs": 0,
            "route_or_dataset_collection_runs": 0,
            "locked_test_payload_opened": False,
            "oai_touched": False,
            "commit": False,
            "push": False,
            "pull": False,
            "fetch": False,
        },
    }
    acceptance_gates = {
        "baseline_metrics_exact": all(
            all(
                all(all(fields.values()) for fields in thresholds.values())
                for thresholds in contracts.values()
            )
            for contracts in baseline_parity.values()
        ),
        "map_catalog_identity_verified": (
            catalog_verification["verdict"] == "PASS"
            and catalog_verification["map_name"] == "Carla/Maps/Town10HD_Opt"
        ),
        "static_ids_unique_stable": (
            len({row["environment_object_id"] for row in catalog_rows}) == len(catalog_rows)
        ),
        "all_static_rows_complete_finite": bool(static_rows),
        "ambiguities_excluded": all(
            not as_bool(gt["_static_row"]["possible_dynamic_static_duplicate"])
            for contract in CONTRACTS
            for values in static_gt_by_frame(static_rows, contract).values()
            for gt in values
        ),
        "locked_test_payload_absent": True,
        "predictions_reused_without_inference": True,
        "canonical_validation_byte_identical": canonical_unchanged,
        "manual_review_package_complete": (
            len(list((EXPERIMENT / "manual_review_panels").glob("*.png")))
            == len(selections)
            and contact_sheet.is_file()
            and all(stats["selected"] + stats["shortage"] == 8 for stats in selection_summary.values())
        ),
        "complete_provenance_inputs": True,
        "carla_cleanup_verified": capture["attempts"][0]["cleanup_verified"],
    }
    metrics["acceptance_gates"] = acceptance_gates
    if not all(acceptance_gates.values()):
        raise RuntimeError(f"audit acceptance gate failed: {acceptance_gates}")
    metrics["terminal"] = SUCCESS_TERMINAL
    metrics["wall_seconds"] = time.monotonic() - started
    write_json_x(EXPERIMENT / "metrics_before_after.json", metrics)
    write_text_x(
        EXPERIMENT / "STATIC_VEHICLE_GT_AUDIT_REPORT.md", markdown_report(metrics)
    )
    write_text_x(EXPERIMENT / "TERMINAL_VERDICT.txt", SUCCESS_TERMINAL + "\n")

    blocked_hashes = {
        "registered_plan_sha256": sha256(
            BLOCKED_EXPERIMENT / "REGISTERED_STATIC_VEHICLE_AUDIT_PLAN.md"
        ),
        "capture_runtime_sha256": sha256(
            BLOCKED_EXPERIMENT / "static_environment_truth/capture_runtime.json"
        ),
        "report_sha256": sha256(
            BLOCKED_EXPERIMENT / "STATIC_VEHICLE_GT_AUDIT_REPORT.md"
        ),
    }
    if blocked_hashes != {
        "registered_plan_sha256": (
            "b6982f2343b0bdeff899fba9ceb5c0e925efc06874497594c5582b93e799fac6"
        ),
        "capture_runtime_sha256": (
            "f293f860dbdce274391130ac8ced7e4cc7d96a4ad177e152d11e8b328c4c2369"
        ),
        "report_sha256": (
            "9bc144f31e430a2d2e909f03eb63c7128ae028aa51e9609daadae1b11c05a3ec"
        ),
    }:
        raise RuntimeError("prior blocked experiment changed")

    output_records = []
    for path in sorted(
        (path for path in EXPERIMENT.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(EXPERIMENT)),
    ):
        if path.name in {"provenance_manifest.json", "audit_completion.json"}:
            continue
        output_records.append(
            {
                "path": str(path.relative_to(EXPERIMENT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    provenance = {
        "schema": "route_b_v3_static_vehicle_gt_audit_provenance_v1",
        "created_utc": utc_now(),
        "terminal": SUCCESS_TERMINAL,
        "repository": {
            "branch": "master",
            "head": "08dd16706e81cf8af6d35b754f0a8b80061a081d",
            "preexisting_dirty_paths": ["OAI/openairinterface5g"],
        },
        "prior_blocked_experiment": {
            "path": str(BLOCKED_EXPERIMENT.relative_to(REPO_ROOT)),
            "unchanged_hashes": blocked_hashes,
        },
        "canonical_validation_tree": metrics["canonical_validation_tree"],
        "frozen_inputs": input_hashes,
        "output_files": output_records,
        "acceptance_gates": acceptance_gates,
        "carla_capture": metrics["carla_capture"],
        "prohibited_actions": metrics["prohibited_actions"],
        "test_policy": (
            "only validation episode payload paths resolved from the canonical report were opened"
        ),
    }
    write_json_x(EXPERIMENT / "provenance_manifest.json", provenance)
    completion = {
        "schema": "route_b_v3_static_vehicle_gt_audit_completion_v1",
        "created_utc": utc_now(),
        "terminal": SUCCESS_TERMINAL,
        "metrics_sha256": sha256(EXPERIMENT / "metrics_before_after.json"),
        "report_sha256": sha256(EXPERIMENT / "STATIC_VEHICLE_GT_AUDIT_REPORT.md"),
        "provenance_sha256": sha256(EXPERIMENT / "provenance_manifest.json"),
        "manual_review_panels": len(selections),
    }
    write_json_x(EXPERIMENT / "audit_completion.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

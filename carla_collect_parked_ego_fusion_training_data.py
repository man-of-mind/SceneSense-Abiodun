#!/usr/bin/env python3
"""Collect parked-ego RGB+radar fusion samples in the pole-training schema.

This is a saved-sample collector, not a split-inference runtime. It mounts RGB,
semantic segmentation, and radar sensors on a parked CARLA ego vehicle, then
writes a small dataset compatible with the existing fusion metadata layout:

  fusion_training_data/<experiment_id>/
    manifest.csv
    object_boxes.csv
    metadata.json
    rgb/<sample_id>.jpg
    masks/<sample_id>.png
    semantic_tags/<sample_id>.png
    radar_tensors/<sample_id>.npy
    radar_points/<sample_id>.npz

The goal is to start the parked-ego fine-tuning/retraining data path. It does
not train a model by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import carla_split_inference_udp_demo as od_demo
import carla_split_inference_udp_segmentation_trained_lraspp_demo as trained_seg_demo
import carla_split_inference_udp_segmentation_trained_lraspp_pole_client as pole_client
import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "pole_lraspp_multimodal_fusion"),
)
from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    append_manifest_rows,
    append_object_box_rows,
    carla_semantic_tags_to_training_mask,
    save_json,
    stable_split,
)
from pole_lraspp_multimodal_fusion.radar_fusion import (  # noqa: E402
    StationaryTrackAccumulator,
    build_radar_sample,
    radar_raw_to_alt_az_depth_velocity,
)


carla = trained_seg_demo.carla
cv2 = trained_seg_demo.cv2

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "fusion_training_data"
DEFAULT_EXPERIMENT_PREFIX = "parked_ego_fusion_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect parked-ego RGB/radar/GT samples for fusion fine-tuning."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where fusion training datasets are written.",
    )
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Dataset folder name. Defaults to timestamped parked_ego_fusion_training_*.",
    )
    parser.add_argument("--max-samples", type=int, default=120, help="Number of saved samples.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Save every Nth synchronized frame after warmup.",
    )
    parser.add_argument("--warmup-ticks", type=int, default=10)
    parser.add_argument("--sensor-timeout", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=10.0)
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument("--sync-world", dest="sync_world", action="store_true")
    sync_group.add_argument("--async-world", dest="sync_world", action="store_false")
    parser.set_defaults(sync_world=True)

    parser.add_argument("--camera-width", type=int, default=854)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument(
        "--model-input-width",
        type=int,
        default=768,
        help="Radar tensor width saved for fusion-model input.",
    )
    parser.add_argument(
        "--model-input-height",
        type=int,
        default=432,
        help="Radar tensor height saved for fusion-model input.",
    )

    parser.add_argument("--ego-vehicle-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-role-name", default="scenesense_fusion_training_ego")
    parser.add_argument("--ego-spawn-index", type=int, default=152)
    parser.add_argument("--ego-spawn-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-right-offset-m", type=float, default=3.0)
    parser.add_argument("--ego-spawn-z-offset-m", type=float, default=0.15)
    parser.add_argument("--ego-spawn-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--ego-freeze", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ego-camera-x", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_X)
    parser.add_argument("--ego-camera-y", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_Y)
    parser.add_argument("--ego-camera-z", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_Z)
    parser.add_argument("--ego-camera-pitch", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_PITCH)
    parser.add_argument("--ego-camera-yaw", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_YAW)
    parser.add_argument("--ego-camera-roll", type=float, default=fusion_runtime.DEFAULT_EGO_CAMERA_ROLL)
    parser.add_argument("--ego-radar-x", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_X)
    parser.add_argument("--ego-radar-y", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_Y)
    parser.add_argument("--ego-radar-z", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_Z)
    parser.add_argument("--ego-radar-pitch", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_PITCH)
    parser.add_argument("--ego-radar-yaw", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_YAW)
    parser.add_argument("--ego-radar-roll", type=float, default=fusion_runtime.DEFAULT_EGO_RADAR_ROLL)

    parser.add_argument("--radar-range", type=float, default=120.0)
    parser.add_argument("--radar-hfov", type=float, default=100.0)
    parser.add_argument("--radar-vfov", type=float, default=30.0)
    parser.add_argument("--radar-points-per-second", type=int, default=5000)
    parser.add_argument("--radar-max-velocity", type=float, default=20.0)
    parser.add_argument("--radar-raster-radius-px", type=int, default=2)
    parser.add_argument("--stationary-velocity-mps", type=float, default=0.35)
    parser.add_argument("--parked-threshold-s", type=float, default=5.0)
    parser.add_argument("--association-grid-m", type=float, default=1.5)
    parser.add_argument("--max-stale-s", type=float, default=2.0)
    parser.add_argument(
        "--radar-support-margin-m",
        type=float,
        default=1.0,
        help="Extra bbox margin used to count radar points supporting an object label.",
    )

    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=10)
    parser.add_argument("--spawn-radius", type=float, default=fusion_runtime.DEFAULT_SPAWN_RADIUS_METERS)
    parser.add_argument("--gt-max-distance-m", type=float, default=140.0)
    parser.add_argument(
        "--include-pedestrians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include pedestrian actor boxes in object_boxes.csv.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=23,
        help="Seed used by stable train/val/test split assignment.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.14)
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def matrix_json(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=float).tolist(), separators=(",", ":"))


def transform_payload(transform: "carla.Transform") -> Dict[str, Dict[str, float]]:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def semantic_tags_from_image(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (int(image.height), int(image.width), 4)
    )
    return arr[:, :, 2].copy()


def wait_for_measurement(
    measurement_queue: "queue.Queue[object]",
    minimum_frame: int,
    timeout: float,
) -> Optional[object]:
    deadline = time.time() + float(timeout)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0.0:
            return None
        try:
            measurement = measurement_queue.get(timeout=remaining)
        except queue.Empty:
            return None
        if int(getattr(measurement, "frame", -1)) < int(minimum_frame):
            continue
        return measurement


def bbox_corner_offsets(extent: "carla.Vector3D") -> np.ndarray:
    ex, ey, ez = float(extent.x), float(extent.y), float(extent.z)
    return np.asarray(
        [
            [ex, ey, ez],
            [ex, ey, -ez],
            [ex, -ey, ez],
            [ex, -ey, -ez],
            [-ex, ey, ez],
            [-ex, ey, -ez],
            [-ex, -ey, ez],
            [-ex, -ey, -ez],
        ],
        dtype=np.float64,
    )


def actor_bbox_world_points(actor: "carla.Actor") -> Tuple[np.ndarray, np.ndarray]:
    bbox = actor.bounding_box
    center_local = np.asarray(
        [bbox.location.x, bbox.location.y, bbox.location.z],
        dtype=np.float64,
    )
    local_points = center_local[None, :] + bbox_corner_offsets(bbox.extent)
    matrix = np.asarray(actor.get_transform().get_matrix(), dtype=np.float64)
    homo = np.concatenate([local_points, np.ones((local_points.shape[0], 1))], axis=1)
    corners_world = (matrix @ homo.T).T[:, :3]
    center_world = (matrix @ np.asarray([*center_local, 1.0], dtype=np.float64).T).T[:3]
    return center_world, corners_world


def project_world_points_to_bbox(
    corners_world: np.ndarray,
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> Optional[Dict[str, float]]:
    if corners_world.size == 0:
        return None
    homo = np.concatenate(
        [corners_world.astype(np.float64), np.ones((corners_world.shape[0], 1))],
        axis=1,
    )
    corners_cam = (camera_inverse_matrix @ homo.T).T[:, :3]
    depth = corners_cam[:, 0]
    in_front = depth > 0.05
    if not np.any(in_front):
        return None
    x = depth[in_front]
    y = corners_cam[in_front, 1]
    z = corners_cam[in_front, 2]
    u = intrinsics[0, 2] + (y / x) * intrinsics[0, 0]
    v = intrinsics[1, 2] - (z / x) * intrinsics[1, 1]
    x1 = float(np.clip(np.min(u), 0.0, float(width)))
    y1 = float(np.clip(np.min(v), 0.0, float(height)))
    x2 = float(np.clip(np.max(u), 0.0, float(width)))
    y2 = float(np.clip(np.max(v), 0.0, float(height)))
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return None
    return {
        "gt_bbox_x": x1,
        "gt_bbox_y": y1,
        "gt_bbox_w": bbox_w,
        "gt_bbox_h": bbox_h,
        "gt_bbox_area_px": bbox_w * bbox_h,
        "gt_center_x": x1 + bbox_w / 2.0,
        "gt_center_y": y1 + bbox_h / 2.0,
        "gt_depth_m": float(np.min(x)),
    }


def transform_world_to_actor_local(points_world: np.ndarray, actor: "carla.Actor") -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    homo = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1))],
        axis=1,
    )
    return (inverse @ homo.T).T[:, :3]


def radar_support_count(
    *,
    actor: "carla.Actor",
    radar_world_xyz: np.ndarray,
    margin_m: float,
) -> int:
    if radar_world_xyz.size == 0:
        return 0
    bbox = actor.bounding_box
    extent = bbox.extent
    center = np.asarray([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)
    local = transform_world_to_actor_local(radar_world_xyz, actor) - center[None, :]
    margin = max(0.0, float(margin_m))
    inside = (
        (np.abs(local[:, 0]) <= float(extent.x) + margin)
        & (np.abs(local[:, 1]) <= float(extent.y) + margin)
        & (np.abs(local[:, 2]) <= float(extent.z) + margin)
    )
    return int(np.count_nonzero(inside))


class ActorStationaryTracker:
    def __init__(self, stationary_velocity_mps: float, parked_threshold_s: float) -> None:
        self.stationary_velocity_mps = float(stationary_velocity_mps)
        self.parked_threshold_s = float(parked_threshold_s)
        self._ages: Dict[int, float] = {}
        self._last_time: Dict[int, float] = {}

    def update(self, actor: "carla.Actor", timestamp_s: float) -> Tuple[float, int, int]:
        actor_id = int(actor.id)
        velocity = actor.get_velocity()
        speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        last_time = self._last_time.get(actor_id, float(timestamp_s))
        dt = max(0.0, float(timestamp_s) - last_time)
        age = self._ages.get(actor_id, 0.0)
        stationary = int(speed <= self.stationary_velocity_mps)
        if stationary:
            age = min(self.parked_threshold_s * 3.0, age + dt)
        else:
            age = 0.0
        self._ages[actor_id] = age
        self._last_time[actor_id] = float(timestamp_s)
        return float(age), stationary, int(age >= self.parked_threshold_s)


def project_actor_to_object_row(
    *,
    actor: "carla.Actor",
    label: str,
    sample_base: Dict[str, object],
    camera_location: "carla.Location",
    camera_inverse_matrix: np.ndarray,
    camera_matrix: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_distance_m: float,
    radar_world_xyz: np.ndarray,
    stationary_tracker: ActorStationaryTracker,
    radar_support_margin_m: float,
) -> Optional[Dict[str, object]]:
    try:
        transform = actor.get_transform()
        bbox = actor.bounding_box
        center_world, corners_world = actor_bbox_world_points(actor)
        distance_m = float(actor.get_location().distance(camera_location))
    except RuntimeError:
        return None
    if float(max_distance_m) > 0.0 and distance_m > float(max_distance_m):
        return None
    projection = project_world_points_to_bbox(
        corners_world,
        camera_inverse_matrix,
        intrinsics,
        int(width),
        int(height),
    )
    if projection is None:
        return None

    sensor_center = (camera_inverse_matrix @ np.asarray([*center_world, 1.0], dtype=np.float64).T).T[:3]
    velocity = actor.get_velocity()
    speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
    stationary_age_s, stationary_label, parked_label = stationary_tracker.update(
        actor,
        float(sample_base["timestamp"]),
    )
    return {
        **sample_base,
        "label": label,
        "gt_actor_id": str(actor.id),
        "gt_source": "actor",
        "gt_actor_type_id": str(getattr(actor, "type_id", "")),
        **projection,
        "gt_distance_m": distance_m,
        "gt_extent_x_m": float(bbox.extent.x),
        "gt_extent_y_m": float(bbox.extent.y),
        "gt_extent_z_m": float(bbox.extent.z),
        "gt_size_x_m": float(bbox.extent.x) * 2.0,
        "gt_size_y_m": float(bbox.extent.y) * 2.0,
        "gt_size_z_m": float(bbox.extent.z) * 2.0,
        "object_world_x": float(center_world[0]),
        "object_world_y": float(center_world[1]),
        "object_world_z": float(center_world[2]),
        "object_sensor_x": float(sensor_center[0]),
        "object_sensor_y": float(sensor_center[1]),
        "object_sensor_z": float(sensor_center[2]),
        "object_yaw_deg": float(transform.rotation.yaw),
        "object_velocity_x_mps": float(velocity.x),
        "object_velocity_y_mps": float(velocity.y),
        "object_velocity_z_mps": float(velocity.z),
        "object_speed_mps": float(speed),
        "stationary_age_s": float(stationary_age_s),
        "stationary_label": int(stationary_label),
        "parked_label": int(parked_label),
        "radar_support_points": radar_support_count(
            actor=actor,
            radar_world_xyz=radar_world_xyz,
            margin_m=float(radar_support_margin_m),
        ),
    }


def build_object_rows(
    *,
    world: "carla.World",
    ego_vehicle: "carla.Actor",
    sample_base: Dict[str, object],
    camera_location: "carla.Location",
    camera_matrix: np.ndarray,
    camera_inverse_matrix: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_distance_m: float,
    radar_world_xyz: np.ndarray,
    stationary_tracker: ActorStationaryTracker,
    include_pedestrians: bool,
    radar_support_margin_m: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    patterns: List[Tuple[str, str]] = [("vehicle", "vehicle.*")]
    if include_pedestrians:
        patterns.append(("person", "walker.pedestrian.*"))
    actors = world.get_actors()
    for label, pattern in patterns:
        for actor in actors.filter(pattern):
            if int(actor.id) == int(ego_vehicle.id):
                continue
            row = project_actor_to_object_row(
                actor=actor,
                label=label,
                sample_base=sample_base,
                camera_location=camera_location,
                camera_inverse_matrix=camera_inverse_matrix,
                camera_matrix=camera_matrix,
                intrinsics=intrinsics,
                width=int(width),
                height=int(height),
                max_distance_m=float(max_distance_m),
                radar_world_xyz=radar_world_xyz,
                stationary_tracker=stationary_tracker,
                radar_support_margin_m=float(radar_support_margin_m),
            )
            if row is not None:
                rows.append(row)
    return rows


def prepare_dataset_dirs(dataset_dir: Path) -> Dict[str, Path]:
    dirs = {
        "rgb": dataset_dir / "rgb",
        "masks": dataset_dir / "masks",
        "semantic_tags": dataset_dir / "semantic_tags",
        "radar_tensors": dataset_dir / "radar_tensors",
        "radar_points": dataset_dir / "radar_points",
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_sample_files(
    *,
    dataset_dir: Path,
    dirs: Dict[str, Path],
    sample_id: str,
    image: "carla.Image",
    semantic_image: "carla.Image",
    radar_tensor: np.ndarray,
    radar_points: Dict[str, np.ndarray],
    jpeg_quality: int,
) -> Tuple[Dict[str, Path], np.ndarray]:
    rgb_path = dirs["rgb"] / f"{sample_id}.jpg"
    mask_path = dirs["masks"] / f"{sample_id}.png"
    semantic_tags_path = dirs["semantic_tags"] / f"{sample_id}.png"
    radar_tensor_path = dirs["radar_tensors"] / f"{sample_id}.npy"
    radar_points_path = dirs["radar_points"] / f"{sample_id}.npz"

    frame_bgr = od_demo.camera_image_to_bgr(image)
    cv2.imwrite(str(rgb_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    tags = semantic_tags_from_image(semantic_image)
    mask = carla_semantic_tags_to_training_mask(tags)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(semantic_tags_path), tags)

    np.save(radar_tensor_path, radar_tensor.astype(np.float32, copy=False))
    np.savez_compressed(radar_points_path, **radar_points)

    return (
        {
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "semantic_tags_path": semantic_tags_path,
            "radar_tensor_path": radar_tensor_path,
            "radar_points_path": radar_points_path,
        },
        mask,
    )


def build_manifest_row(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    experiment_id: str,
    sample_id: str,
    split: str,
    file_paths: Dict[str, Path],
    image: "carla.Image",
    semantic_image: "carla.Image",
    radar_measurement: "carla.RadarMeasurement",
    mask: np.ndarray,
    world: "carla.World",
    camera: "carla.Actor",
    radar: "carla.Actor",
    ego_vehicle: "carla.Actor",
    camera_matrix: np.ndarray,
    camera_inverse_matrix: np.ndarray,
    radar_matrix: np.ndarray,
    radar_inverse_matrix: np.ndarray,
    intrinsics_full: np.ndarray,
    radar_summary: Dict[str, float],
) -> Dict[str, object]:
    camera_transform = camera.get_transform()
    radar_transform = radar.get_transform()
    anchor_transform = ego_vehicle.get_transform()
    radar_to_camera = camera_inverse_matrix @ radar_matrix
    vehicle_pixels = int(np.count_nonzero(mask == 1))
    person_pixels = int(np.count_nonzero(mask == 2))
    return {
        "experiment_id": experiment_id,
        "sample_id": sample_id,
        "split": split,
        "rgb_path": relpath(file_paths["rgb_path"], dataset_dir),
        "mask_path": relpath(file_paths["mask_path"], dataset_dir),
        "instance_raw_path": relpath(file_paths["semantic_tags_path"], dataset_dir),
        "radar_tensor_path": relpath(file_paths["radar_tensor_path"], dataset_dir),
        "radar_points_path": relpath(file_paths["radar_points_path"], dataset_dir),
        "frame_id": int(image.frame),
        "radar_frame_id": int(getattr(radar_measurement, "frame", -1)),
        "timestamp": float(image.timestamp),
        "radar_timestamp": float(getattr(radar_measurement, "timestamp", image.timestamp)),
        "traffic_light_id": "",
        "traffic_light_opendrive_id": "",
        "map_name": str(world.get_map().name),
        "camera_x": float(camera_transform.location.x),
        "camera_y": float(camera_transform.location.y),
        "camera_z": float(camera_transform.location.z),
        "camera_pitch": float(camera_transform.rotation.pitch),
        "camera_yaw": float(camera_transform.rotation.yaw),
        "camera_roll": float(camera_transform.rotation.roll),
        "camera_fov": float(args.camera_fov),
        "camera_width": int(args.camera_width),
        "camera_height": int(args.camera_height),
        "camera_fx": float(intrinsics_full[0, 0]),
        "camera_fy": float(intrinsics_full[1, 1]),
        "camera_cx": float(intrinsics_full[0, 2]),
        "camera_cy": float(intrinsics_full[1, 2]),
        "camera_matrix_json": matrix_json(camera_matrix),
        "camera_inverse_matrix_json": matrix_json(camera_inverse_matrix),
        "radar_matrix_json": matrix_json(radar_matrix),
        "radar_inverse_matrix_json": matrix_json(radar_inverse_matrix),
        "radar_to_camera_matrix_json": matrix_json(radar_to_camera),
        "anchor_x": float(anchor_transform.location.x),
        "anchor_y": float(anchor_transform.location.y),
        "anchor_z": float(anchor_transform.location.z),
        "anchor_pitch": float(anchor_transform.rotation.pitch),
        "anchor_yaw": float(anchor_transform.rotation.yaw),
        "anchor_roll": float(anchor_transform.rotation.roll),
        "radar_horizontal_fov": float(args.radar_hfov),
        "radar_vertical_fov": float(args.radar_vfov),
        "radar_range_m": float(args.radar_range),
        "radar_points": int(radar_summary.get("radar_points", 0)),
        "radar_stationary_points": int(radar_summary.get("radar_stationary_points", 0)),
        "radar_parked_evidence_points": int(radar_summary.get("radar_parked_evidence_points", 0)),
        "traffic_density": int(args.npc_vehicles),
        "pedestrian_density": int(args.npc_pedestrians),
        "scenario_id": "parked_ego_training",
        "view_id": str(args.ego_spawn_index),
        "vehicle_pixels": vehicle_pixels,
        "person_pixels": person_pixels,
    }


def write_metadata(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    experiment_id: str,
    world: "carla.World",
    ego_vehicle: "carla.Actor",
    camera: "carla.Actor",
    radar: "carla.Actor",
) -> None:
    metadata = {
        "schema": "scenesense_parked_ego_fusion_training_data.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "description": (
            "Saved parked-ego RGB/radar/GT samples for fine-tuning the "
            "pole-trained RGB+radar fusion model."
        ),
        "world": str(world.get_map().name),
        "sample_count_requested": int(args.max_samples),
        "camera_resolution": [int(args.camera_width), int(args.camera_height)],
        "model_input_size": [int(args.model_input_width), int(args.model_input_height)],
        "ego_vehicle": {
            "actor_id": int(ego_vehicle.id),
            "type_id": str(getattr(ego_vehicle, "type_id", "")),
            "transform": transform_payload(ego_vehicle.get_transform()),
        },
        "camera": {
            "actor_id": int(camera.id),
            "relative_transform": {
                "x": float(args.ego_camera_x),
                "y": float(args.ego_camera_y),
                "z": float(args.ego_camera_z),
                "pitch": float(args.ego_camera_pitch),
                "yaw": float(args.ego_camera_yaw),
                "roll": float(args.ego_camera_roll),
            },
            "world_transform": transform_payload(camera.get_transform()),
            "fov": float(args.camera_fov),
        },
        "radar": {
            "actor_id": int(radar.id),
            "relative_transform": {
                "x": float(args.ego_radar_x),
                "y": float(args.ego_radar_y),
                "z": float(args.ego_radar_z),
                "pitch": float(args.ego_radar_pitch),
                "yaw": float(args.ego_radar_yaw),
                "roll": float(args.ego_radar_roll),
            },
            "world_transform": transform_payload(radar.get_transform()),
            "range_m": float(args.radar_range),
            "horizontal_fov": float(args.radar_hfov),
            "vertical_fov": float(args.radar_vfov),
            "points_per_second": int(args.radar_points_per_second),
        },
        "split_ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": max(0.0, 1.0 - float(args.train_ratio) - float(args.val_ratio)),
            "seed": int(args.split_seed),
        },
        "command_args": vars(args),
    }
    save_json(dataset_dir / "metadata.json", metadata)


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    experiment_id = str(args.experiment_id).strip() or f"{now_stamp()}_{DEFAULT_EXPERIMENT_PREFIX}"
    dataset_dir = Path(args.output_root).expanduser().resolve() / experiment_id
    dirs = prepare_dataset_dirs(dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    object_boxes_path = dataset_dir / "object_boxes.csv"
    split_ratios = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": max(0.0, 1.0 - float(args.train_ratio) - float(args.val_ratio)),
    }

    print(f"Dataset directory: {dataset_dir}")

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    try:
        traffic_manager.set_random_device_seed(int(args.seed))
    except RuntimeError:
        pass
    try:
        world.set_pedestrians_seed(int(args.seed))
    except Exception:
        pass

    original_settings = world.get_settings()
    actors: List["carla.Actor"] = []
    pedestrian_controllers: List["carla.Actor"] = []
    image_queue: "queue.Queue[object]" = queue.Queue(maxsize=4)
    semantic_queue: "queue.Queue[object]" = queue.Queue(maxsize=4)
    radar_queue: "queue.Queue[object]" = queue.Queue(maxsize=4)

    try:
        if bool(args.sync_world):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(0.1, float(args.fps))
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.tick()

        ego_vehicle = fusion_runtime._spawn_parked_ego_vehicle(world=world, args=args)
        actors.append(ego_vehicle)
        anchor_location = ego_vehicle.get_location()
        print(f"Parked ego: id={ego_vehicle.id}, type={ego_vehicle.type_id}")

        background_vehicles = pole_client.spawn_background_vehicles_near(
            client,
            world,
            traffic_manager,
            anchor_location,
            int(args.npc_vehicles),
            float(args.spawn_radius),
        )
        actors.extend(background_vehicles)
        print(f"Spawned background vehicles: {len(background_vehicles)}")

        pedestrians, pedestrian_controllers = pole_client.spawn_background_pedestrians_near(
            client,
            world,
            anchor_location,
            int(args.npc_pedestrians),
            float(args.spawn_radius),
        )
        actors.extend(pedestrians)
        actors.extend(pedestrian_controllers)
        print(f"Spawned pedestrians: {len(pedestrians)}")

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(int(args.camera_width)))
        camera_bp.set_attribute("image_size_y", str(int(args.camera_height)))
        camera_bp.set_attribute("fov", str(float(args.camera_fov)))
        camera_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
        semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        semantic_bp.set_attribute("image_size_x", str(int(args.camera_width)))
        semantic_bp.set_attribute("image_size_y", str(int(args.camera_height)))
        semantic_bp.set_attribute("fov", str(float(args.camera_fov)))
        semantic_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(float(args.radar_range)))
        radar_bp.set_attribute("horizontal_fov", str(float(args.radar_hfov)))
        radar_bp.set_attribute("vertical_fov", str(float(args.radar_vfov)))
        radar_bp.set_attribute("points_per_second", str(int(args.radar_points_per_second)))
        radar_bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))

        camera = world.spawn_actor(
            camera_bp,
            fusion_runtime._ego_camera_transform(args),
            attach_to=ego_vehicle,
        )
        semantic_camera = world.spawn_actor(
            semantic_bp,
            fusion_runtime._ego_camera_transform(args),
            attach_to=ego_vehicle,
        )
        radar = world.spawn_actor(
            radar_bp,
            fusion_runtime._ego_radar_transform(args),
            attach_to=ego_vehicle,
        )
        actors.extend([camera, semantic_camera, radar])
        camera.listen(lambda image: od_demo.put_latest(image_queue, image))
        semantic_camera.listen(lambda image: od_demo.put_latest(semantic_queue, image))
        radar.listen(lambda measurement: od_demo.put_latest(radar_queue, measurement))

        write_metadata(
            args=args,
            dataset_dir=dataset_dir,
            experiment_id=experiment_id,
            world=world,
            ego_vehicle=ego_vehicle,
            camera=camera,
            radar=radar,
        )

        tracker = StationaryTrackAccumulator(
            stationary_velocity_mps=float(args.stationary_velocity_mps),
            parked_threshold_s=float(args.parked_threshold_s),
            association_grid_m=float(args.association_grid_m),
            max_stale_s=float(args.max_stale_s),
        )
        actor_stationary_tracker = ActorStationaryTracker(
            stationary_velocity_mps=float(args.stationary_velocity_mps),
            parked_threshold_s=float(args.parked_threshold_s),
        )
        intrinsics_full = fusion_runtime.intrinsics_at(
            int(args.camera_width),
            int(args.camera_height),
            float(args.camera_fov),
        )
        intrinsics_input = fusion_runtime.intrinsics_at(
            int(args.model_input_width),
            int(args.model_input_height),
            float(args.camera_fov),
        )

        for _ in range(max(0, int(args.warmup_ticks))):
            if bool(args.sync_world):
                world.tick()
            else:
                time.sleep(1.0 / max(0.1, float(args.fps)))

        saved = 0
        attempts = 0
        while saved < int(args.max_samples):
            attempts += 1
            if bool(args.sync_world):
                frame_id = int(world.tick())
            else:
                time.sleep(1.0 / max(0.1, float(args.fps)))
                frame_id = 0
            if int(args.sample_stride) > 1 and attempts % int(args.sample_stride) != 0:
                continue

            image = od_demo.wait_for_camera_frame(
                image_queue,
                frame_id,
                float(args.sensor_timeout),
            )
            semantic_image = od_demo.wait_for_camera_frame(
                semantic_queue,
                frame_id,
                float(args.sensor_timeout),
            )
            radar_measurement = wait_for_measurement(
                radar_queue,
                frame_id,
                float(args.sensor_timeout),
            )
            if image is None or semantic_image is None or radar_measurement is None:
                print(f"Warning: missing synchronized sensors at frame {frame_id}; retrying.")
                continue

            camera_matrix = fusion_runtime.actor_world_matrix(camera)
            camera_inverse_matrix = fusion_runtime.actor_world_inverse_matrix(camera)
            radar_matrix = fusion_runtime.actor_world_matrix(radar)
            radar_inverse_matrix = fusion_runtime.actor_world_inverse_matrix(radar)
            detections = radar_raw_to_alt_az_depth_velocity(bytes(radar_measurement.raw_data))
            radar_tensor, radar_points, radar_summary = build_radar_sample(
                detections=detections,
                sensor_matrix=radar_matrix,
                camera_inverse_matrix=camera_inverse_matrix,
                camera_intrinsics=intrinsics_input,
                width=int(args.model_input_width),
                height=int(args.model_input_height),
                frame_time_s=float(getattr(radar_measurement, "timestamp", image.timestamp)),
                tracker=tracker,
                max_range_m=float(args.radar_range),
                max_abs_velocity_mps=float(args.radar_max_velocity),
                parked_threshold_s=float(args.parked_threshold_s),
                point_radius_px=int(args.radar_raster_radius_px),
            )

            sample_id = f"{experiment_id}_{saved:06d}_frame{int(image.frame)}"
            split = stable_split(sample_id, split_ratios, int(args.split_seed))
            file_paths, mask = save_sample_files(
                dataset_dir=dataset_dir,
                dirs=dirs,
                sample_id=sample_id,
                image=image,
                semantic_image=semantic_image,
                radar_tensor=radar_tensor,
                radar_points=radar_points,
                jpeg_quality=int(args.jpeg_quality),
            )
            manifest_row = build_manifest_row(
                args=args,
                dataset_dir=dataset_dir,
                experiment_id=experiment_id,
                sample_id=sample_id,
                split=split,
                file_paths=file_paths,
                image=image,
                semantic_image=semantic_image,
                radar_measurement=radar_measurement,
                mask=mask,
                world=world,
                camera=camera,
                radar=radar,
                ego_vehicle=ego_vehicle,
                camera_matrix=camera_matrix,
                camera_inverse_matrix=camera_inverse_matrix,
                radar_matrix=radar_matrix,
                radar_inverse_matrix=radar_inverse_matrix,
                intrinsics_full=intrinsics_full,
                radar_summary=radar_summary,
            )
            sample_base = {
                "experiment_id": experiment_id,
                "sample_id": sample_id,
                "frame_id": int(image.frame),
                "timestamp": float(image.timestamp),
                "traffic_light_id": "",
                "scenario_id": "parked_ego_training",
                "view_id": str(args.ego_spawn_index),
            }
            object_rows = build_object_rows(
                world=world,
                ego_vehicle=ego_vehicle,
                sample_base=sample_base,
                camera_location=camera.get_transform().location,
                camera_matrix=camera_matrix,
                camera_inverse_matrix=camera_inverse_matrix,
                intrinsics=intrinsics_full,
                width=int(args.camera_width),
                height=int(args.camera_height),
                max_distance_m=float(args.gt_max_distance_m),
                radar_world_xyz=np.asarray(radar_points["world_xyz"], dtype=np.float32),
                stationary_tracker=actor_stationary_tracker,
                include_pedestrians=bool(args.include_pedestrians),
                radar_support_margin_m=float(args.radar_support_margin_m),
            )
            append_manifest_rows(manifest_path, [manifest_row])
            append_object_box_rows(object_boxes_path, object_rows)
            saved += 1
            if saved == 1 or saved % 10 == 0 or saved >= int(args.max_samples):
                print(
                    f"Saved {saved}/{int(args.max_samples)} samples "
                    f"(frame={int(image.frame)}, objects={len(object_rows)}, "
                    f"vehicle_pixels={manifest_row['vehicle_pixels']}, "
                    f"person_pixels={manifest_row['person_pixels']})"
                )

        print(f"Done. Dataset: {dataset_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"Object boxes: {object_boxes_path}")
        return 0
    finally:
        for actor in reversed(actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                pass
        for controller in pedestrian_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if bool(args.sync_world):
            try:
                world.apply_settings(original_settings)
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

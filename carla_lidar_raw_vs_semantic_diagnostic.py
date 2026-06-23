#!/usr/bin/env python3
"""Paired raw-vs-semantic LiDAR diagnostic for SceneSense.

This script is intentionally separate from the supervisor's original LiDAR
visualizer. It spawns a raw CARLA LiDAR and a semantic CARLA LiDAR at the same
pose, records matched frames, and evaluates how many vehicle/person actors each
sensor can support through geometry alone versus simulator-provided semantic
tags/object ids.

Outputs are written under:
  lidar_diagnostic_runs/raw_vs_semantic_<timestamp>/

The raw LiDAR path uses CARLA actor boxes only for evaluation/labeling. It does
not use those boxes as a detector output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover - CARLA is available on the lab machines.
    carla = None


# CARLA 0.10 semantic IDs used by the existing SceneSense scripts. Some older
# CARLA examples use a different CityScapes ordering, so keep legacy aliases in
# the default tag filters rather than relying only on this display map.
CITYSCAPES_TAGS = {
    0: "unlabeled",
    1: "building",
    2: "fence",
    3: "other",
    4: "road_line",
    5: "road",
    6: "sidewalk",
    7: "vegetation",
    8: "traffic_light",
    9: "vegetation",
    10: "sky",
    11: "pedestrian_legacy",
    12: "pedestrian",
    13: "rider",
    14: "car",
    15: "truck",
    16: "bus",
    17: "train",
    18: "motorcycle",
    19: "static",
    20: "dynamic",
    21: "water",
    22: "other",
    23: "terrain",
    24: "pedestrian_legacy",
    25: "pedestrian_legacy",
}


@dataclass
class ActorBox:
    actor_id: int
    actor_type: str
    blueprint_id: str
    transform: "carla.Transform"
    bbox_location: np.ndarray
    bbox_rotation: "carla.Rotation"
    extent: np.ndarray
    location: np.ndarray
    center_world: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw CARLA LiDAR with semantic CARLA LiDAR in a matched scene."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--load-town", action="store_true")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--output-root", default="lidar_diagnostic_runs")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--warmup-ticks", type=int, default=30)
    parser.add_argument("--asynch", action="store_true", help="Do not force synchronous mode.")
    parser.add_argument("--preview", action="store_true", help="Show an RGB preview from the LiDAR/camera pose.")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=120.0)

    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-spawn-index", type=int, default=80)
    parser.add_argument("--ego-spawn-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-right-offset-m", type=float, default=7.0)
    parser.add_argument("--ego-spawn-z-offset-m", type=float, default=0.15)
    parser.add_argument("--ego-spawn-yaw-offset-deg", type=float, default=-28.414)

    parser.add_argument("--sensor-x", type=float, default=1.8)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-z", type=float, default=1.55)
    parser.add_argument("--sensor-pitch", type=float, default=-4.0)
    parser.add_argument("--sensor-yaw", type=float, default=0.0)
    parser.add_argument("--sensor-roll", type=float, default=0.0)

    parser.add_argument("--lidar-range", type=float, default=120.0)
    parser.add_argument("--lidar-upper-fov", type=float, default=15.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-15.0)
    parser.add_argument("--lidar-channels", type=int, default=64)
    parser.add_argument("--lidar-rotation-frequency", type=float, default=20.0)
    parser.add_argument("--lidar-pps", type=int, default=600000)
    parser.add_argument(
        "--lidar-sensor-tick",
        type=float,
        default=0.0,
        help="0 means every simulator tick. Use 0.05 for 20 Hz-style sampling.",
    )

    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=25)
    parser.add_argument("--spawn-radius", type=float, default=90.0)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--npc-vehicle-speed-difference-pct", type=float, default=20.0)
    parser.add_argument("--npc-pedestrian-max-speed-mps", type=float, default=1.1)
    parser.add_argument("--npc-pedestrian-cross-factor", type=float, default=0.5)
    parser.add_argument(
        "--pedestrian-placement",
        choices=("random", "front_sector", "sensor_crossing_line", "nav_crossing_line", "visible_nav"),
        default="random",
        help=(
            "Use front_sector to sample navigation points in view; use sensor_crossing_line "
            "to place walkers directly along a crossing line in the sensor FOV; "
            "use nav_crossing_line to choose valid navigation points nearest that line; "
            "use visible_nav to choose valid navigation points by camera projection/depth."
        ),
    )
    parser.add_argument("--pedestrian-front-min-distance-m", type=float, default=6.0)
    parser.add_argument("--pedestrian-front-max-distance-m", type=float, default=45.0)
    parser.add_argument("--pedestrian-front-lateral-m", type=float, default=18.0)
    parser.add_argument("--pedestrian-spawn-attempts", type=int, default=6000)
    parser.add_argument("--pedestrian-line-forward-m", type=float, default=24.0)
    parser.add_argument("--pedestrian-line-lateral-min-m", type=float, default=-16.0)
    parser.add_argument("--pedestrian-line-lateral-max-m", type=float, default=16.0)
    parser.add_argument("--pedestrian-line-z-m", type=float, default=-1.45)
    parser.add_argument("--pedestrian-line-spacing-m", type=float, default=2.0)
    parser.add_argument("--pedestrian-line-nearest-nav-max-m", type=float, default=12.0)
    parser.add_argument("--pedestrian-visible-min-depth-m", type=float, default=10.0)
    parser.add_argument("--pedestrian-visible-max-depth-m", type=float, default=50.0)
    parser.add_argument("--pedestrian-visible-u-margin-px", type=float, default=80.0)
    parser.add_argument("--pedestrian-visible-v-margin-px", type=float, default=80.0)
    parser.add_argument("--pedestrian-visible-target-depth-m", type=float, default=25.0)
    parser.add_argument(
        "--pedestrian-motion",
        choices=("ai_random", "cross_sensor"),
        default="ai_random",
        help="cross_sensor directly moves spawned walkers laterally through the sensor FOV.",
    )
    parser.add_argument("--pedestrian-cross-speed-mps", type=float, default=1.2)
    parser.add_argument("--pedestrian-cross-lateral-m", type=float, default=22.0)
    parser.add_argument("--pedestrian-cross-target-epsilon-m", type=float, default=1.2)

    parser.add_argument("--gt-max-distance-m", type=float, default=140.0)
    parser.add_argument("--bbox-margin-xy", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-up", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-down", type=float, default=0.70)
    parser.add_argument(
        "--person-association-mode",
        choices=("bbox", "radius"),
        default="radius",
        help=(
            "Geometry association for pedestrian actors in raw/semantic-tag modes. "
            "radius is more stable for sparse person point clouds than strict actor boxes."
        ),
    )
    parser.add_argument(
        "--person-association-radius-m",
        type=float,
        default=1.1,
        help="Ground-plane radius around the pedestrian actor used when --person-association-mode=radius.",
    )
    parser.add_argument(
        "--person-association-z-down-m",
        type=float,
        default=0.4,
        help="Meters below pedestrian actor origin included by radius association.",
    )
    parser.add_argument(
        "--person-association-z-up-m",
        type=float,
        default=5.0,
        help="Meters above pedestrian actor origin included by radius association.",
    )
    parser.add_argument("--min-vehicle-points", type=int, default=20)
    parser.add_argument("--min-person-points", type=int, default=10)
    parser.add_argument(
        "--semantic-ped-tags",
        default="4,12,24,25",
        help="Tags treated as person candidates for semantic-tag geometry mode.",
    )
    parser.add_argument(
        "--semantic-vehicle-tags",
        default="10,14,15,16",
        help="Tags treated as vehicle candidates for semantic-tag geometry mode.",
    )
    parser.add_argument("--sample-points-per-frame", type=int, default=400)
    parser.add_argument("--debug-every", type=int, default=0)
    return parser.parse_args()


def now_experiment_id() -> str:
    return f"raw_vs_semantic_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_int_set(text: str) -> set[int]:
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def rotation_matrix_from_carla_rotation(rot: "carla.Rotation") -> np.ndarray:
    roll = deg_to_rad(rot.roll)
    pitch = deg_to_rad(rot.pitch)
    yaw = deg_to_rad(rot.yaw)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def transform_points(points: np.ndarray, tf: "carla.Transform") -> np.ndarray:
    if points.size == 0:
        return points.reshape((-1, 3)).astype(np.float32)
    rot = rotation_matrix_from_carla_rotation(tf.rotation)
    loc = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float32)
    return points @ rot.T + loc


def inverse_transform_points(points_world: np.ndarray, tf: "carla.Transform") -> np.ndarray:
    if points_world.size == 0:
        return points_world.reshape((-1, 3)).astype(np.float32)
    rot = rotation_matrix_from_carla_rotation(tf.rotation)
    loc = np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float32)
    return (points_world - loc) @ rot


def offset_transform(
    base_tf: "carla.Transform",
    forward_m: float,
    right_m: float,
    z_m: float,
    yaw_offset_deg: float,
) -> "carla.Transform":
    yaw = deg_to_rad(base_tf.rotation.yaw)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    right = np.array([math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0], dtype=np.float32)
    loc = base_tf.location
    new_loc = carla.Location(
        x=float(loc.x + forward[0] * forward_m + right[0] * right_m),
        y=float(loc.y + forward[1] * forward_m + right[1] * right_m),
        z=float(loc.z + z_m),
    )
    new_rot = carla.Rotation(
        pitch=base_tf.rotation.pitch,
        yaw=base_tf.rotation.yaw + yaw_offset_deg,
        roll=base_tf.rotation.roll,
    )
    return carla.Transform(new_loc, new_rot)


def transform_matrix(tf: "carla.Transform") -> np.ndarray:
    return np.array(tf.get_matrix(), dtype=np.float32)


def sensor_world_matrix_from_ego(ego: "carla.Actor", args: argparse.Namespace) -> np.ndarray:
    sensor_tf = carla.Transform(
        carla.Location(x=args.sensor_x, y=args.sensor_y, z=args.sensor_z),
        carla.Rotation(pitch=args.sensor_pitch, yaw=args.sensor_yaw, roll=args.sensor_roll),
    )
    return transform_matrix(ego.get_transform()) @ transform_matrix(sensor_tf)


def sensor_ground_matrix_from_ego(ego: "carla.Actor", args: argparse.Namespace) -> np.ndarray:
    """Sensor-position frame with yaw only, used for placing controlled walkers.

    The actual camera/LiDAR can be pitched down. If we use that pitched frame to
    place pedestrians "forward" on the ground, the line can end up below the
    road. This horizontal frame keeps the same sensor origin/yaw but removes
    pitch/roll so local x/y mean ground-plane forward/lateral.
    """
    ego_tf = ego.get_transform()
    ego_matrix = transform_matrix(ego_tf)
    sensor_offset = np.array([args.sensor_x, args.sensor_y, args.sensor_z, 1.0], dtype=np.float32)
    sensor_world = ego_matrix @ sensor_offset
    return transform_matrix(
        carla.Transform(
            carla.Location(x=float(sensor_world[0]), y=float(sensor_world[1]), z=float(sensor_world[2])),
            carla.Rotation(pitch=0.0, yaw=float(ego_tf.rotation.yaw + args.sensor_yaw), roll=0.0),
        )
    )


def location_in_sensor_front_sector(
    location: "carla.Location",
    sensor_world_matrix: np.ndarray,
    min_distance_m: float,
    max_distance_m: float,
    lateral_m: float,
) -> bool:
    point = np.array([location.x, location.y, location.z, 1.0], dtype=np.float32)
    local = np.linalg.inv(sensor_world_matrix) @ point
    forward = float(local[0])
    lateral = float(local[1])
    return min_distance_m <= forward <= max_distance_m and abs(lateral) <= lateral_m


def sensor_local_to_world_location(
    sensor_world_matrix: np.ndarray,
    forward_m: float,
    lateral_m: float,
    z_m: float,
) -> "carla.Location":
    point = sensor_world_matrix @ np.array([forward_m, lateral_m, z_m, 1.0], dtype=np.float32)
    return carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2]))


def crossing_line_locations(
    sensor_world_matrix: np.ndarray,
    count: int,
    forward_m: float,
    lateral_min_m: float,
    lateral_max_m: float,
    z_m: float,
    spacing_m: float,
) -> List["carla.Location"]:
    if count <= 0:
        return []
    spacing = max(0.5, float(spacing_m))
    span = abs(lateral_max_m - lateral_min_m)
    slots_per_row = max(1, int(math.floor(span / spacing)) + 1)
    rows = max(1, int(math.ceil(count / slots_per_row)))
    row_offsets = np.linspace(-4.0, 4.0, rows, dtype=np.float32) if rows > 1 else np.array([0.0], dtype=np.float32)
    locations: List["carla.Location"] = []
    for row_offset in row_offsets:
        laterals = np.linspace(lateral_min_m, lateral_max_m, slots_per_row, dtype=np.float32)
        for lateral in laterals:
            locations.append(
                sensor_local_to_world_location(
                    sensor_world_matrix,
                    forward_m + float(row_offset),
                    float(lateral),
                    z_m,
                )
            )
            if len(locations) >= count:
                return locations
    return locations


def nearest_nav_locations_to_crossing_line(
    world: "carla.World",
    ego: "carla.Actor",
    sensor_world_matrix: np.ndarray,
    count: int,
    spawn_radius: float,
    forward_m: float,
    lateral_min_m: float,
    lateral_max_m: float,
    z_m: float,
    spacing_m: float,
    max_distance_m: float,
    attempts: int,
) -> List["carla.Location"]:
    targets = crossing_line_locations(
        sensor_world_matrix,
        count,
        forward_m,
        lateral_min_m,
        lateral_max_m,
        z_m,
        spacing_m,
    )
    if not targets:
        return []

    ego_loc = ego.get_location()
    inv_sensor = np.linalg.inv(sensor_world_matrix)
    candidates: List[Tuple[float, float, carla.Location]] = []
    max_attempts = max(attempts, count * 200)
    for _ in range(max_attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None or loc.distance(ego_loc) > spawn_radius:
            continue
        local = inv_sensor @ np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float32)
        forward = float(local[0])
        lateral = float(local[1])
        if forward < 0.0 or forward > max(forward_m + 45.0, 80.0):
            continue
        if abs(lateral) > max(abs(lateral_min_m), abs(lateral_max_m)) + 35.0:
            continue
        candidates.append((forward, lateral, loc))

    if not candidates:
        return []

    chosen: List[carla.Location] = []
    used: set[int] = set()
    for target in targets:
        best_i = -1
        best_dist = float("inf")
        for idx, (_forward, _lateral, loc) in enumerate(candidates):
            if idx in used:
                continue
            dist = math.hypot(float(loc.x - target.x), float(loc.y - target.y))
            if dist < best_dist:
                best_dist = dist
                best_i = idx
        if best_i >= 0 and best_dist <= max_distance_m:
            used.add(best_i)
            chosen.append(candidates[best_i][2])
        if len(chosen) >= count:
            break
    return chosen


def project_world_location(
    location: "carla.Location",
    camera_world_matrix: np.ndarray,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> Optional[Tuple[float, float, float]]:
    world_to_camera = np.linalg.inv(camera_world_matrix)
    point_world = np.array([location.x, location.y, location.z, 1.0], dtype=np.float32)
    point_camera = world_to_camera @ point_world
    depth = float(point_camera[0])
    if depth <= 0.2:
        return None
    k = camera_intrinsics(image_width, image_height, fov_deg)
    cam = np.array([float(point_camera[1]), float(-point_camera[2]), depth], dtype=np.float32)
    uvw = k @ cam
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth


def visible_nav_locations(
    world: "carla.World",
    ego: "carla.Actor",
    camera_world_matrix: np.ndarray,
    count: int,
    spawn_radius: float,
    image_width: int,
    image_height: int,
    fov_deg: float,
    min_depth_m: float,
    max_depth_m: float,
    u_margin_px: float,
    v_margin_px: float,
    target_depth_m: float,
    attempts: int,
) -> List["carla.Location"]:
    ego_loc = ego.get_location()
    candidates: List[Tuple[float, float, float, carla.Location]] = []
    max_attempts = max(attempts, count * 500)
    for _ in range(max_attempts):
        loc = world.get_random_location_from_navigation()
        if loc is None or loc.distance(ego_loc) > spawn_radius:
            continue
        projected = project_world_location(loc, camera_world_matrix, image_width, image_height, fov_deg)
        if projected is None:
            continue
        u, v, depth = projected
        if not (min_depth_m <= depth <= max_depth_m):
            continue
        if not (-u_margin_px <= u <= image_width + u_margin_px):
            continue
        if not (-v_margin_px <= v <= image_height + v_margin_px):
            continue
        # Prefer points near target depth and nearer the image center, with enough spacing below.
        center_penalty = abs(u - image_width / 2.0) / max(1.0, image_width / 2.0)
        score = abs(depth - target_depth_m) + 4.0 * center_penalty
        candidates.append((score, u, depth, loc))
    candidates.sort(key=lambda item: item[0])

    chosen: List[carla.Location] = []
    for _score, _u, _depth, loc in candidates:
        if all(loc.distance(existing) >= 1.5 for existing in chosen):
            chosen.append(loc)
        if len(chosen) >= count:
            break
    return chosen


def find_blueprint(blueprints: "carla.BlueprintLibrary", pattern: str) -> "carla.ActorBlueprint":
    matches = list(blueprints.filter(pattern))
    if not matches:
        raise RuntimeError(f"No blueprint matches {pattern!r}")
    return random.choice(matches)


def spawn_ego(world: "carla.World", args: argparse.Namespace) -> "carla.Vehicle":
    spawn_points = world.get_map().get_spawn_points()
    if args.ego_spawn_index < 0 or args.ego_spawn_index >= len(spawn_points):
        raise ValueError(
            f"ego spawn index {args.ego_spawn_index} outside available range 0..{len(spawn_points) - 1}"
        )
    bp = find_blueprint(world.get_blueprint_library(), args.ego_blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "scenesense_lidar_diag_ego")
    tf = offset_transform(
        spawn_points[args.ego_spawn_index],
        args.ego_spawn_forward_offset_m,
        args.ego_spawn_right_offset_m,
        args.ego_spawn_z_offset_m,
        args.ego_spawn_yaw_offset_deg,
    )
    ego = world.try_spawn_actor(bp, tf)
    if ego is None:
        raise RuntimeError(f"Failed to spawn ego at {tf}")
    ego.set_simulate_physics(False)
    return ego


def spawn_npc_vehicles(
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    ego: "carla.Actor",
    count: int,
    spawn_radius: float,
    speed_diff_pct: float,
) -> List["carla.Actor"]:
    if count <= 0:
        return []
    ego_loc = ego.get_location()
    blueprints = []
    for bp in world.get_blueprint_library().filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels"):
            try:
                if int(bp.get_attribute("number_of_wheels")) < 4:
                    continue
            except ValueError:
                pass
        blueprints.append(bp)
    if not blueprints:
        blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)
    actors: List["carla.Actor"] = []
    for tf in spawn_points:
        if len(actors) >= count:
            break
        if tf.location.distance(ego_loc) > spawn_radius or tf.location.distance(ego_loc) < 8.0:
            continue
        bp = random.choice(blueprints)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "scenesense_lidar_diag_npc")
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(actor, float(speed_diff_pct))
        actors.append(actor)
    return actors


def spawn_walkers(
    world: "carla.World",
    client: "carla.Client",
    ego: "carla.Actor",
    count: int,
    spawn_radius: float,
    max_speed_mps: float,
    crossing_factor: float,
    placement: str,
    sensor_world_matrix: np.ndarray,
    front_min_distance_m: float,
    front_max_distance_m: float,
    front_lateral_m: float,
    spawn_attempts: int,
    motion: str,
    line_forward_m: float,
    line_lateral_min_m: float,
    line_lateral_max_m: float,
    line_z_m: float,
    line_spacing_m: float,
    line_nearest_nav_max_m: float,
    visible_min_depth_m: float,
    visible_max_depth_m: float,
    visible_u_margin_px: float,
    visible_v_margin_px: float,
    visible_target_depth_m: float,
    camera_width: int,
    camera_height: int,
    camera_fov: float,
) -> Tuple[List["carla.Actor"], List["carla.Actor"]]:
    if count <= 0:
        return [], []
    blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    ego_loc = ego.get_location()
    spawn_locations: List[carla.Location] = []
    attempts = 0
    if placement == "sensor_crossing_line":
        spawn_locations = crossing_line_locations(
            sensor_world_matrix,
            count,
            line_forward_m,
            line_lateral_min_m,
            line_lateral_max_m,
            line_z_m,
            line_spacing_m,
        )
        attempts = len(spawn_locations)
    elif placement == "nav_crossing_line":
        spawn_locations = nearest_nav_locations_to_crossing_line(
            world,
            ego,
            sensor_world_matrix,
            count,
            spawn_radius,
            line_forward_m,
            line_lateral_min_m,
            line_lateral_max_m,
            line_z_m,
            line_spacing_m,
            line_nearest_nav_max_m,
            spawn_attempts,
        )
        attempts = max(spawn_attempts, count * 200)
    elif placement == "visible_nav":
        spawn_locations = visible_nav_locations(
            world,
            ego,
            sensor_world_matrix,
            count,
            spawn_radius,
            camera_width,
            camera_height,
            camera_fov,
            visible_min_depth_m,
            visible_max_depth_m,
            visible_u_margin_px,
            visible_v_margin_px,
            visible_target_depth_m,
            spawn_attempts,
        )
        attempts = max(spawn_attempts, count * 500)
    else:
        max_attempts = max(spawn_attempts, count * 80)
        while len(spawn_locations) < count and attempts < max_attempts:
            attempts += 1
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if loc.distance(ego_loc) > spawn_radius:
                continue
            if placement == "front_sector" and not location_in_sensor_front_sector(
                loc,
                sensor_world_matrix,
                front_min_distance_m,
                front_max_distance_m,
                front_lateral_m,
            ):
                continue
            spawn_locations.append(loc)

    if placement in ("front_sector", "sensor_crossing_line", "nav_crossing_line", "visible_nav") and len(spawn_locations) < count:
        print(
            f"Warning: {placement} walker placement found {len(spawn_locations)}/{count} "
            f"candidate nav points after {attempts} attempts."
        )

    walkers: List["carla.Actor"] = []
    controllers: List["carla.Actor"] = []
    for loc in spawn_locations:
        bp = random.choice(blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = world.try_spawn_actor(bp, carla.Transform(loc))
        if actor is None:
            continue
        walkers.append(actor)
        if motion == "ai_random":
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), actor)
            if controller is None:
                actor.destroy()
                walkers.pop()
                continue
            controllers.append(controller)

    world.set_pedestrians_cross_factor(float(crossing_factor))
    for controller in controllers:
        controller.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            controller.go_to_location(target)
        controller.set_max_speed(float(max_speed_mps))
    return walkers, controllers


def update_cross_sensor_walkers(
    walkers: Sequence["carla.Actor"],
    sensor_world_matrix: np.ndarray,
    target_state: Dict[int, float],
    speed_mps: float,
    lateral_m: float,
    epsilon_m: float,
) -> None:
    if not walkers:
        return
    inv_sensor = np.linalg.inv(sensor_world_matrix)
    for walker in walkers:
        if not walker.is_alive:
            continue
        loc = walker.get_location()
        local = inv_sensor @ np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float32)
        actor_id = int(walker.id)
        if actor_id not in target_state:
            target_state[actor_id] = -lateral_m if float(local[1]) >= 0.0 else lateral_m
        if abs(float(local[1]) - target_state[actor_id]) <= epsilon_m:
            target_state[actor_id] *= -1.0
        target_local = np.array([float(local[0]), target_state[actor_id], float(local[2]), 1.0], dtype=np.float32)
        target_world = sensor_world_matrix @ target_local
        direction = np.array([target_world[0] - loc.x, target_world[1] - loc.y, 0.0], dtype=np.float32)
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-3:
            continue
        direction /= norm
        walker.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(x=float(direction[0]), y=float(direction[1]), z=0.0),
                speed=float(speed_mps),
                jump=False,
            )
        )


def configure_lidar_bp(
    bp: "carla.ActorBlueprint",
    args: argparse.Namespace,
    sensor_tick: float,
) -> None:
    bp.set_attribute("range", str(args.lidar_range))
    bp.set_attribute("upper_fov", str(args.lidar_upper_fov))
    bp.set_attribute("lower_fov", str(args.lidar_lower_fov))
    bp.set_attribute("channels", str(args.lidar_channels))
    bp.set_attribute("rotation_frequency", str(args.lidar_rotation_frequency))
    bp.set_attribute("points_per_second", str(args.lidar_pps))
    bp.set_attribute("sensor_tick", str(sensor_tick))


def raw_lidar_to_array(data: "carla.LidarMeasurement") -> np.ndarray:
    arr = np.frombuffer(data.raw_data, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    return arr.reshape((-1, 4)).astype(np.float32)


def camera_image_to_bgr(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(deg_to_rad(fov_deg) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_actor_markers(
    actor_boxes: Sequence[ActorBox],
    camera_tf: "carla.Transform",
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> List[dict]:
    if not actor_boxes:
        return []
    k = camera_intrinsics(image_width, image_height, fov_deg)
    world_to_camera = np.array(camera_tf.get_inverse_matrix(), dtype=np.float32)
    markers: List[dict] = []
    for box in actor_boxes:
        point_world = np.array(
            [box.center_world[0], box.center_world[1], box.center_world[2], 1.0],
            dtype=np.float32,
        )
        point_camera = world_to_camera @ point_world
        # CARLA camera convention: x is depth, y is horizontal, z is vertical.
        depth = float(point_camera[0])
        if depth <= 0.2:
            continue
        cam = np.array([float(point_camera[1]), float(-point_camera[2]), depth], dtype=np.float32)
        uvw = k @ cam
        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])
        if -80.0 <= u <= image_width + 80.0 and -80.0 <= v <= image_height + 80.0:
            markers.append(
                {
                    "u": u,
                    "v": v,
                    "depth": depth,
                    "actor_type": box.actor_type,
                    "actor_id": box.actor_id,
                }
            )
    return markers


def draw_preview(
    image_bgr: np.ndarray,
    frame_row: dict,
    spawned_walkers: int,
    spawned_vehicles: int,
    preview_width: int,
    preview_height: int,
    actor_markers: Optional[Sequence[dict]] = None,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on lab environment.
        raise RuntimeError("OpenCV is required for --preview") from exc

    original_h, original_w = image_bgr.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if preview_width > 0 and preview_height > 0:
        scale_x = preview_width / float(original_w)
        scale_y = preview_height / float(original_h)
        image_bgr = cv2.resize(image_bgr, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    visible_person_markers = 0
    visible_vehicle_markers = 0
    for marker in actor_markers or []:
        u = int(round(float(marker["u"]) * scale_x))
        v = int(round(float(marker["v"]) * scale_y))
        if marker.get("actor_type") == "person":
            color = (255, 0, 255)
            prefix = "P"
            visible_person_markers += 1
            radius = 8
        else:
            color = (255, 210, 0)
            prefix = "V"
            visible_vehicle_markers += 1
            radius = 6
        if 0 <= u < image_bgr.shape[1] and 0 <= v < image_bgr.shape[0]:
            cv2.circle(image_bgr, (u, v), radius, color, 2, cv2.LINE_AA)
            label = f"{prefix}{marker.get('actor_id')} {float(marker.get('depth', 0.0)):.0f}m"
            cv2.putText(
                image_bgr,
                label,
                (u + 8, max(16, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                label,
                (u + 8, max(16, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    lines = [
        "Raw vs Semantic LiDAR diagnostic",
        f"raw pts: {frame_row.get('raw_points', 0)} | semantic pts: {frame_row.get('semantic_points', 0)}",
        f"actors in GT range: vehicles={frame_row.get('actor_vehicle_count', 0)} persons={frame_row.get('actor_person_count', 0)}",
        f"projected markers: vehicles={visible_vehicle_markers} persons={visible_person_markers}",
        f"spawned: vehicles={spawned_vehicles} walkers={spawned_walkers}",
        (
            "recall raw V/P: "
            f"{_fmt_metric(frame_row.get('raw_vehicle_recall'))}/"
            f"{_fmt_metric(frame_row.get('raw_person_recall'))}"
        ),
        (
            "recall semantic-id V/P: "
            f"{_fmt_metric(frame_row.get('semantic_id_vehicle_recall'))}/"
            f"{_fmt_metric(frame_row.get('semantic_id_person_recall'))}"
        ),
        "press q in this window to stop",
    ]
    y = 26
    for line in lines:
        cv2.putText(image_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25
    return image_bgr


def _fmt_metric(value) -> str:
    if value in ("", None):
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def semantic_lidar_to_arrays(data: "carla.SemanticLidarMeasurement") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: List[Tuple[float, float, float]] = []
    tags: List[int] = []
    obj_ids: List[int] = []
    for det in data:
        p = det.point
        points.append((float(p.x), float(p.y), float(p.z)))
        tag = 0
        for attr in ("object_tag", "semantic_tag", "tag"):
            if hasattr(det, attr):
                tag = int(getattr(det, attr))
                break
        obj_id = 0
        for attr in ("object_idx", "obj_idx", "object_id", "id", "idx"):
            if hasattr(det, attr):
                obj_id = int(getattr(det, attr))
                break
        tags.append(tag)
        obj_ids.append(obj_id)
    if not points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.array(points, dtype=np.float32),
        np.array(tags, dtype=np.int32),
        np.array(obj_ids, dtype=np.int32),
    )


def get_actor_boxes(world: "carla.World", ego: "carla.Actor", max_distance_m: float) -> List[ActorBox]:
    ego_loc = ego.get_location()
    boxes: List[ActorBox] = []
    for actor in world.get_actors():
        type_id = actor.type_id
        if actor.id == ego.id:
            continue
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
            continue
        loc = actor.get_location()
        if loc.distance(ego_loc) > max_distance_m:
            continue
        bbox = actor.bounding_box
        actor_type = "vehicle" if type_id.startswith("vehicle.") else "person"
        bbox_location = np.array([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float32)
        actor_matrix = np.array(actor.get_transform().get_matrix(), dtype=np.float32)
        bbox_center_h = np.array([bbox.location.x, bbox.location.y, bbox.location.z, 1.0], dtype=np.float32)
        center_world = (actor_matrix @ bbox_center_h)[:3]
        boxes.append(
            ActorBox(
                actor_id=int(actor.id),
                actor_type=actor_type,
                blueprint_id=type_id,
                transform=actor.get_transform(),
                bbox_location=bbox_location,
                bbox_rotation=bbox.rotation,
                extent=np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z], dtype=np.float32),
                location=np.array([loc.x, loc.y, loc.z], dtype=np.float32),
                center_world=center_world.astype(np.float32),
            )
        )
    return boxes


def points_inside_actor_box(
    points_world: np.ndarray,
    actor_box: ActorBox,
    margin_xy: float,
    margin_z_up: float,
    margin_z_down: float,
) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    pw_h = np.concatenate([points_world.astype(np.float32), ones], axis=1)
    inv_actor = np.array(actor_box.transform.get_inverse_matrix(), dtype=np.float32)
    actor_local = (inv_actor @ pw_h.T).T[:, :3]
    inv_bbox_rot = rotation_matrix_from_carla_rotation(actor_box.bbox_rotation).T
    local = (actor_local - actor_box.bbox_location.reshape(1, 3)) @ inv_bbox_rot.T
    ext = actor_box.extent
    inside_xy = (np.abs(local[:, 0]) <= ext[0] + margin_xy) & (
        np.abs(local[:, 1]) <= ext[1] + margin_xy
    )
    inside_z = (local[:, 2] <= ext[2] + margin_z_up) & (local[:, 2] >= -ext[2] - margin_z_down)
    return inside_xy & inside_z


def points_inside_person_radius(
    points_world: np.ndarray,
    actor_box: ActorBox,
    radius_m: float,
    z_down_m: float,
    z_up_m: float,
) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    origin = actor_box.location.astype(np.float32)
    dx = points_world[:, 0] - origin[0]
    dy = points_world[:, 1] - origin[1]
    dz = points_world[:, 2] - origin[2]
    radius = max(0.05, float(radius_m))
    inside_xy = (dx * dx + dy * dy) <= radius * radius
    inside_z = (dz >= -abs(float(z_down_m))) & (dz <= abs(float(z_up_m)))
    return inside_xy & inside_z


def points_associated_with_actor(
    points_world: np.ndarray,
    actor_box: ActorBox,
    args: argparse.Namespace,
) -> np.ndarray:
    if (
        actor_box.actor_type == "person"
        and getattr(args, "person_association_mode", "radius") == "radius"
    ):
        return points_inside_person_radius(
            points_world,
            actor_box,
            getattr(args, "person_association_radius_m", 1.1),
            getattr(args, "person_association_z_down_m", 0.4),
            getattr(args, "person_association_z_up_m", 2.4),
        )
    return points_inside_actor_box(
        points_world,
        actor_box,
        args.bbox_margin_xy,
        args.bbox_margin_z_up,
        args.bbox_margin_z_down,
    )


def evaluate_mode(
    mode: str,
    points_world: np.ndarray,
    actor_boxes: Sequence[ActorBox],
    args: argparse.Namespace,
    tags: Optional[np.ndarray] = None,
    obj_ids: Optional[np.ndarray] = None,
    ped_tags: Optional[set[int]] = None,
    veh_tags: Optional[set[int]] = None,
) -> List[dict]:
    rows: List[dict] = []
    for box in actor_boxes:
        min_points = args.min_vehicle_points if box.actor_type == "vehicle" else args.min_person_points
        if mode == "semantic_object_id":
            if obj_ids is None:
                selected = np.empty((0, 3), dtype=np.float32)
            else:
                selected = points_world[obj_ids == box.actor_id]
        else:
            candidate_points = points_world
            if mode == "semantic_tag_bbox" and tags is not None:
                tag_set = veh_tags if box.actor_type == "vehicle" else ped_tags
                candidate_points = points_world[np.isin(tags, list(tag_set or set()))]
            inside = points_associated_with_actor(candidate_points, box, args)
            selected = candidate_points[inside]

        point_count = int(selected.shape[0])
        hit = int(point_count >= min_points)
        if point_count > 0:
            centroid = selected.mean(axis=0)
            xy_error = float(np.linalg.norm(centroid[:2] - box.center_world[:2]))
            centroid_x, centroid_y, centroid_z = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
        else:
            xy_error = ""
            centroid_x = centroid_y = centroid_z = ""
        rows.append(
            {
                "mode": mode,
                "actor_type": box.actor_type,
                "actor_id": box.actor_id,
                "blueprint_id": box.blueprint_id,
                "point_count": point_count,
                "hit": hit,
                "xy_error_m": xy_error,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_z": centroid_z,
                "actor_x": float(box.center_world[0]),
                "actor_y": float(box.center_world[1]),
                "actor_z": float(box.center_world[2]),
            }
        )
    return rows


def summarize_actor_rows(rows: Sequence[dict]) -> dict:
    summary: Dict[str, Dict[str, dict]] = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        summary[mode] = {}
        for actor_type in ("vehicle", "person"):
            subset = [row for row in rows if row["mode"] == mode and row["actor_type"] == actor_type]
            total = len(subset)
            hits = sum(int(row["hit"]) for row in subset)
            xy_errors = [
                float(row["xy_error_m"])
                for row in subset
                if row["hit"] and row["xy_error_m"] not in ("", None)
            ]
            point_counts = [int(row["point_count"]) for row in subset]
            summary[mode][actor_type] = {
                "actor_observations": total,
                "hits": hits,
                "recall": float(hits / total) if total else None,
                "xy_error_mean_m": float(np.mean(xy_errors)) if xy_errors else None,
                "xy_error_median_m": float(np.median(xy_errors)) if xy_errors else None,
                "points_per_actor_mean": float(np.mean(point_counts)) if point_counts else None,
                "points_per_actor_median": float(np.median(point_counts)) if point_counts else None,
            }
    return summary


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if carla is None:
        raise SystemExit(
            "Could not import carla. Run this inside the CARLA PythonAPI environment."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)

    experiment_id = args.experiment_id or now_experiment_id()
    output_dir = Path(args.output_root) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.load_world(args.town) if args.load_town else client.get_world()
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_random_device_seed(int(args.seed))
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    original_settings = world.get_settings()
    actors_to_destroy: List["carla.Actor"] = []
    walker_controllers: List["carla.Actor"] = []
    raw_sensor = None
    semantic_sensor = None
    camera_sensor = None

    raw_queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()
    semantic_queue: "queue.Queue[carla.SemanticLidarMeasurement]" = queue.Queue()
    camera_queue: "queue.Queue[carla.Image]" = queue.Queue()
    walker_cross_targets: Dict[int, float] = {}

    try:
        if not args.asynch:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / float(args.fps)
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            world.tick()

        ego = spawn_ego(world, args)
        actors_to_destroy.append(ego)
        sensor_world_matrix = sensor_world_matrix_from_ego(ego, args)
        pedestrian_ground_matrix = sensor_ground_matrix_from_ego(ego, args)

        npc_vehicles = spawn_npc_vehicles(
            world,
            traffic_manager,
            ego,
            args.npc_vehicles,
            args.spawn_radius,
            args.npc_vehicle_speed_difference_pct,
        )
        actors_to_destroy.extend(npc_vehicles)

        walkers, controllers = spawn_walkers(
            world,
            client,
            ego,
            args.npc_pedestrians,
            args.spawn_radius,
            args.npc_pedestrian_max_speed_mps,
            args.npc_pedestrian_cross_factor,
            args.pedestrian_placement,
            pedestrian_ground_matrix,
            args.pedestrian_front_min_distance_m,
            args.pedestrian_front_max_distance_m,
            args.pedestrian_front_lateral_m,
            args.pedestrian_spawn_attempts,
            args.pedestrian_motion,
            args.pedestrian_line_forward_m,
            args.pedestrian_line_lateral_min_m,
            args.pedestrian_line_lateral_max_m,
            args.pedestrian_line_z_m,
            args.pedestrian_line_spacing_m,
            args.pedestrian_line_nearest_nav_max_m,
            args.pedestrian_visible_min_depth_m,
            args.pedestrian_visible_max_depth_m,
            args.pedestrian_visible_u_margin_px,
            args.pedestrian_visible_v_margin_px,
            args.pedestrian_visible_target_depth_m,
            args.camera_width,
            args.camera_height,
            args.camera_fov,
        )
        actors_to_destroy.extend(walkers)
        actors_to_destroy.extend(controllers)
        walker_controllers.extend(controllers)
        print(
            f"Spawned diagnostic actors: vehicles={len(npc_vehicles)} "
            f"walkers={len(walkers)} placement={args.pedestrian_placement} "
            f"motion={args.pedestrian_motion}"
        )

        sensor_tf = carla.Transform(
            carla.Location(x=args.sensor_x, y=args.sensor_y, z=args.sensor_z),
            carla.Rotation(pitch=args.sensor_pitch, yaw=args.sensor_yaw, roll=args.sensor_roll),
        )
        bp_lib = world.get_blueprint_library()
        sensor_tick = float(args.lidar_sensor_tick) if not args.asynch else 0.0
        if args.preview:
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(args.camera_width))
            camera_bp.set_attribute("image_size_y", str(args.camera_height))
            camera_bp.set_attribute("fov", str(args.camera_fov))
            camera_bp.set_attribute("sensor_tick", str(sensor_tick))
            camera_sensor = world.spawn_actor(camera_bp, sensor_tf, attach_to=ego)
            actors_to_destroy.append(camera_sensor)
            camera_sensor.listen(camera_queue.put)

        raw_bp = bp_lib.find("sensor.lidar.ray_cast")
        semantic_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        configure_lidar_bp(raw_bp, args, sensor_tick)
        configure_lidar_bp(semantic_bp, args, sensor_tick)

        raw_sensor = world.spawn_actor(raw_bp, sensor_tf, attach_to=ego)
        semantic_sensor = world.spawn_actor(semantic_bp, sensor_tf, attach_to=ego)
        actors_to_destroy.extend([raw_sensor, semantic_sensor])
        raw_sensor.listen(raw_queue.put)
        semantic_sensor.listen(semantic_queue.put)

        for _ in range(max(0, int(args.warmup_ticks))):
            if args.pedestrian_motion == "cross_sensor":
                update_cross_sensor_walkers(
                    walkers,
                    pedestrian_ground_matrix,
                    walker_cross_targets,
                    args.pedestrian_cross_speed_mps,
                    args.pedestrian_cross_lateral_m,
                    args.pedestrian_cross_target_epsilon_m,
                )
            world.tick() if not args.asynch else time.sleep(1.0 / float(args.fps))
            while not raw_queue.empty():
                raw_queue.get_nowait()
            while not semantic_queue.empty():
                semantic_queue.get_nowait()
            while not camera_queue.empty():
                camera_queue.get_nowait()

        frame_rows: List[dict] = []
        actor_rows: List[dict] = []
        raw_sample_rows: List[dict] = []
        semantic_sample_rows: List[dict] = []
        ped_tags = parse_int_set(args.semantic_ped_tags)
        veh_tags = parse_int_set(args.semantic_vehicle_tags)

        target_frames = int(round(float(args.duration_s) * float(args.fps)))
        start_wall = time.time()
        captured = 0

        for _ in range(target_frames):
            if args.pedestrian_motion == "cross_sensor":
                update_cross_sensor_walkers(
                    walkers,
                    pedestrian_ground_matrix,
                    walker_cross_targets,
                    args.pedestrian_cross_speed_mps,
                    args.pedestrian_cross_lateral_m,
                    args.pedestrian_cross_target_epsilon_m,
                )
            frame_id = int(world.tick()) if not args.asynch else -1
            if args.asynch:
                time.sleep(1.0 / float(args.fps))

            try:
                raw_data = raw_queue.get(timeout=2.0)
                sem_data = semantic_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # Drain to the most recent matched-ish samples if sensors produced multiple callbacks.
            while not raw_queue.empty():
                raw_data = raw_queue.get_nowait()
            while not semantic_queue.empty():
                sem_data = semantic_queue.get_nowait()
            camera_data = None
            if args.preview:
                try:
                    camera_data = camera_queue.get_nowait()
                    while not camera_queue.empty():
                        camera_data = camera_queue.get_nowait()
                except queue.Empty:
                    camera_data = None

            raw_arr = raw_lidar_to_array(raw_data)
            sem_points_sensor, sem_tags, sem_obj_ids = semantic_lidar_to_arrays(sem_data)

            raw_points_sensor = raw_arr[:, :3] if raw_arr.size else np.empty((0, 3), dtype=np.float32)
            raw_points_world = transform_points(raw_points_sensor, raw_sensor.get_transform())
            sem_points_world = transform_points(sem_points_sensor, semantic_sensor.get_transform())
            actor_boxes = get_actor_boxes(world, ego, args.gt_max_distance_m)

            frame_actor_rows: List[dict] = []
            frame_actor_rows.extend(evaluate_mode("raw_bbox", raw_points_world, actor_boxes, args))
            frame_actor_rows.extend(
                evaluate_mode(
                    "semantic_tag_bbox",
                    sem_points_world,
                    actor_boxes,
                    args,
                    tags=sem_tags,
                    ped_tags=ped_tags,
                    veh_tags=veh_tags,
                )
            )
            frame_actor_rows.extend(
                evaluate_mode(
                    "semantic_object_id",
                    sem_points_world,
                    actor_boxes,
                    args,
                    obj_ids=sem_obj_ids,
                )
            )
            for row in frame_actor_rows:
                row["frame"] = int(raw_data.frame)
                row["semantic_frame"] = int(sem_data.frame)
                actor_rows.append(row)

            mode_counts = summarize_actor_rows(frame_actor_rows)
            tag_counts = {
                str(int(tag)): int(count)
                for tag, count in zip(*np.unique(sem_tags, return_counts=True))
            } if sem_tags.size else {}

            raw_bytes = int(raw_points_sensor.shape[0] * 4 * 4)
            sem_bytes_est = int(sem_points_sensor.shape[0] * (3 * 4 + 2 * 4))
            frame_rows.append(
                frame_row := {
                    "frame": int(raw_data.frame),
                    "world_tick_frame": frame_id,
                    "semantic_frame": int(sem_data.frame),
                    "elapsed_wall_s": round(time.time() - start_wall, 4),
                    "raw_points": int(raw_points_sensor.shape[0]),
                    "semantic_points": int(sem_points_sensor.shape[0]),
                    "raw_bytes_est": raw_bytes,
                    "semantic_bytes_est": sem_bytes_est,
                    "actor_vehicle_count": sum(1 for box in actor_boxes if box.actor_type == "vehicle"),
                    "actor_person_count": sum(1 for box in actor_boxes if box.actor_type == "person"),
                    "raw_vehicle_recall": mode_counts.get("raw_bbox", {}).get("vehicle", {}).get("recall"),
                    "raw_person_recall": mode_counts.get("raw_bbox", {}).get("person", {}).get("recall"),
                    "semantic_tag_vehicle_recall": mode_counts.get("semantic_tag_bbox", {}).get("vehicle", {}).get("recall"),
                    "semantic_tag_person_recall": mode_counts.get("semantic_tag_bbox", {}).get("person", {}).get("recall"),
                    "semantic_id_vehicle_recall": mode_counts.get("semantic_object_id", {}).get("vehicle", {}).get("recall"),
                    "semantic_id_person_recall": mode_counts.get("semantic_object_id", {}).get("person", {}).get("recall"),
                    "semantic_tag_counts_json": json.dumps(tag_counts, sort_keys=True),
                }
            )

            if args.preview and camera_data is not None:
                try:
                    import cv2

                    actor_markers = project_actor_markers(
                        actor_boxes,
                        camera_sensor.get_transform(),
                        int(camera_data.width),
                        int(camera_data.height),
                        args.camera_fov,
                    )
                    preview = draw_preview(
                        camera_image_to_bgr(camera_data),
                        frame_row,
                        spawned_walkers=len(walkers),
                        spawned_vehicles=len(npc_vehicles),
                        preview_width=args.preview_width,
                        preview_height=args.preview_height,
                        actor_markers=actor_markers,
                    )
                    cv2.imshow("SceneSense raw-vs-semantic LiDAR diagnostic", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Preview requested stop via q.")
                        break
                except RuntimeError as exc:
                    print(f"Preview disabled: {exc}")
                    args.preview = False

            if args.sample_points_per_frame > 0:
                raw_n = min(args.sample_points_per_frame, raw_points_world.shape[0])
                if raw_n:
                    idx = np.random.choice(raw_points_world.shape[0], raw_n, replace=False)
                    for p in raw_points_world[idx]:
                        raw_sample_rows.append(
                            {
                                "frame": int(raw_data.frame),
                                "x": float(p[0]),
                                "y": float(p[1]),
                                "z": float(p[2]),
                            }
                        )
                sem_n = min(args.sample_points_per_frame, sem_points_world.shape[0])
                if sem_n:
                    idx = np.random.choice(sem_points_world.shape[0], sem_n, replace=False)
                    for p, tag, obj_id in zip(sem_points_world[idx], sem_tags[idx], sem_obj_ids[idx]):
                        semantic_sample_rows.append(
                            {
                                "frame": int(sem_data.frame),
                                "x": float(p[0]),
                                "y": float(p[1]),
                                "z": float(p[2]),
                                "tag": int(tag),
                                "tag_name": CITYSCAPES_TAGS.get(int(tag), "unknown"),
                                "object_id": int(obj_id),
                            }
                        )

            captured += 1
            if args.debug_every > 0 and captured % args.debug_every == 0:
                print(
                    f"captured={captured}/{target_frames} raw_pts={raw_points_sensor.shape[0]} "
                    f"sem_pts={sem_points_sensor.shape[0]} actors={len(actor_boxes)}"
                )

        write_csv(
            output_dir / "frame_metrics.csv",
            frame_rows,
            [
                "frame",
                "world_tick_frame",
                "semantic_frame",
                "elapsed_wall_s",
                "raw_points",
                "semantic_points",
                "raw_bytes_est",
                "semantic_bytes_est",
                "actor_vehicle_count",
                "actor_person_count",
                "raw_vehicle_recall",
                "raw_person_recall",
                "semantic_tag_vehicle_recall",
                "semantic_tag_person_recall",
                "semantic_id_vehicle_recall",
                "semantic_id_person_recall",
                "semantic_tag_counts_json",
            ],
        )
        write_csv(
            output_dir / "actor_metrics.csv",
            actor_rows,
            [
                "frame",
                "semantic_frame",
                "mode",
                "actor_type",
                "actor_id",
                "blueprint_id",
                "point_count",
                "hit",
                "xy_error_m",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "actor_x",
                "actor_y",
                "actor_z",
            ],
        )
        if raw_sample_rows:
            write_csv(output_dir / "raw_points_sample.csv", raw_sample_rows, ["frame", "x", "y", "z"])
        if semantic_sample_rows:
            write_csv(
                output_dir / "semantic_points_sample.csv",
                semantic_sample_rows,
                ["frame", "x", "y", "z", "tag", "tag_name", "object_id"],
            )

        summary = {
            "experiment_id": experiment_id,
            "output_dir": str(output_dir.resolve()),
            "captured_frames": captured,
            "settings": vars(args),
            "sensor": {
                "raw": "sensor.lidar.ray_cast",
                "semantic": "sensor.lidar.ray_cast_semantic",
                "same_transform": True,
                "lidar_range_m": args.lidar_range,
                "channels": args.lidar_channels,
                "points_per_second": args.lidar_pps,
                "sensor_tick": sensor_tick,
            },
            "spawned_actors": {
                "vehicles": len(npc_vehicles),
                "walkers": len(walkers),
                "pedestrian_placement": args.pedestrian_placement,
                "pedestrian_motion": args.pedestrian_motion,
                "pedestrian_frame": "sensor_yaw_ground_plane",
            },
            "modes": {
                "raw_bbox": "Raw LiDAR xyz points assigned to CARLA actor boxes for evaluation only.",
                "semantic_tag_bbox": "Semantic LiDAR points filtered by semantic tag, then assigned to actor boxes.",
                "semantic_object_id": "Semantic LiDAR points grouped by CARLA object id; this is oracle association.",
            },
            "overall": summarize_actor_rows(actor_rows),
        }
        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if raw_sensor is not None:
            raw_sensor.stop()
        if semantic_sensor is not None:
            semantic_sensor.stop()
        if camera_sensor is not None:
            camera_sensor.stop()
        if args.preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        for controller in walker_controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        for actor in reversed(actors_to_destroy):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        if not args.asynch:
            try:
                traffic_manager.set_synchronous_mode(False)
                world.apply_settings(original_settings)
            except RuntimeError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

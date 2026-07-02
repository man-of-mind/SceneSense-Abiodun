#!/usr/bin/env python3
"""Controlled CARLA radar-vs-pedestrian diagnostic.

This script isolates the radar sensor from the fusion model. It places one
pedestrian directly in front of an ego-mounted radar, then sweeps radar
points-per-second and pedestrian distance. The pedestrian can remain stationary
or walk across/toward the radar so we can measure whether CARLA radar returns
change with pedestrian motion, distance, point density, or physics settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover - CARLA env provides this.
    carla = None


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def amplitude_values_for_distances(args: argparse.Namespace, distances: Sequence[float]) -> List[float]:
    text = str(getattr(args, "walker_motion_amplitude_list_m", "") or "").strip()
    if not text:
        return [float(getattr(args, "walker_motion_amplitude_m", 0.0)) for _ in distances]
    values = parse_float_list(text)
    if len(values) != len(distances):
        raise SystemExit(
            "--walker-motion-amplitude-list-m must have the same number of entries as --distance-list-m "
            f"({len(values)} amplitudes for {len(distances)} distances)."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--load-town", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--output-root", default="abiodun/radar_pedestrian_diagnostic_runs")

    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--frames-per-condition", type=int, default=80)
    parser.add_argument("--pps-list", default="5000,12000,24000,60000")
    parser.add_argument("--distance-list-m", default="2,5,10,15,20,30,40,60")
    parser.add_argument(
        "--plot-min-distance-m",
        type=float,
        default=10.0,
        help="Minimum distance included in generated distance plots.",
    )
    parser.add_argument(
        "--walker-physics-mode",
        choices=("default", "on", "off", "both", "all"),
        default="default",
        help=(
            "default leaves the walker physics state untouched; on/off call "
            "set_simulate_physics explicitly; both runs on+off; all runs default+on+off."
        ),
    )

    parser.add_argument("--ego-spawn-index", type=int, default=80)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-z-offset-m", type=float, default=0.15)
    parser.add_argument("--walker-blueprint", default="walker.pedestrian.0001")
    parser.add_argument(
        "--walker-z-offset-m",
        type=float,
        default=0.0,
        help="Extra height above the estimated ground after adding the walker bbox half-height.",
    )
    parser.add_argument(
        "--walker-z-mode",
        choices=("feet", "bbox_center"),
        default="bbox_center",
        help=(
            "How to place the walker vertically. feet puts the actor origin at road/nav ground; "
            "bbox_center adds half the actor height before applying walker-z-offset-m."
        ),
    )
    parser.add_argument("--walker-lateral-m", type=float, default=0.0)
    parser.add_argument("--walker-face-radar", action="store_true", default=True)
    parser.add_argument("--no-walker-face-radar", dest="walker_face_radar", action="store_false")
    parser.add_argument(
        "--walker-motion-mode",
        choices=("stationary", "cross", "toward_away"),
        default="stationary",
        help=(
            "stationary keeps the walker fixed; cross walks left-right across the radar FOV; "
            "toward_away walks along the radar line of sight and reverses at the amplitude limits."
        ),
    )
    parser.add_argument(
        "--walker-motion-control",
        choices=("walker_control", "walker_control_nudge", "kinematic", "kinematic_cycle"),
        default="walker_control",
        help=(
            "walker_control uses CARLA WalkerControl and can get stuck on medians/bollards; "
            "walker_control_nudge uses WalkerControl for animation and only nudges on stalls; "
            "kinematic moves the actor by transform updates; kinematic_cycle forces a complete "
            "left/right deterministic path across each condition."
        ),
    )
    parser.add_argument(
        "--walker-motion-amplitude-m",
        type=float,
        default=3.0,
        help="Half-width of the moving-walker path around each target distance.",
    )
    parser.add_argument(
        "--walker-motion-amplitude-list-m",
        default="",
        help=(
            "Optional comma-separated half-widths for moving-walker paths. "
            "When provided, it must match --distance-list-m one-for-one and overrides "
            "--walker-motion-amplitude-m for each distance."
        ),
    )
    parser.add_argument(
        "--walker-motion-speed-mps",
        type=float,
        default=1.2,
        help="Walker control speed for moving diagnostic modes.",
    )
    parser.add_argument(
        "--walker-nudge-after-frames",
        type=int,
        default=8,
        help="For walker_control_nudge, nudge the actor after this many low-progress frames.",
    )
    parser.add_argument(
        "--walker-kinematic-cycle-count",
        type=float,
        default=1.0,
        help=(
            "For kinematic_cycle, number of full triangular left-right-left cycles per condition. "
            "Use 0.5 for a single left-to-right pass."
        ),
    )
    parser.add_argument(
        "--debug-placement",
        action="store_true",
        help="Print the walker world/sensor position whenever the diagnostic places it.",
    )
    parser.add_argument(
        "--debug-marker-life-s",
        type=float,
        default=2.0,
        help="Lifetime for CARLA debug marker drawn at the walker placement when --debug-placement is enabled.",
    )

    parser.add_argument("--radar-x", type=float, default=1.8)
    parser.add_argument("--radar-y", type=float, default=0.0)
    parser.add_argument("--radar-z", type=float, default=1.55)
    parser.add_argument("--radar-pitch", type=float, default=-4.0)
    parser.add_argument("--radar-yaw", type=float, default=0.0)
    parser.add_argument("--radar-roll", type=float, default=0.0)
    parser.add_argument("--radar-range", type=float, default=120.0)
    parser.add_argument("--radar-hfov", type=float, default=120.0)
    parser.add_argument("--radar-vfov", type=float, default=30.0)
    parser.add_argument("--radar-sensor-tick", type=float, default=0.0)

    parser.add_argument("--person-radius-m", type=float, default=1.5)
    parser.add_argument("--person-z-down-m", type=float, default=0.5)
    parser.add_argument("--person-z-up-m", type=float, default=2.0)
    parser.add_argument("--bbox-margin-m", type=float, default=0.35)
    parser.add_argument("--depth-window-m", type=float, default=1.0)
    parser.add_argument(
        "--cep-min-points-list",
        default="1,5,10,20,25",
        help=(
            "Comma-separated radar-point thresholds used when summarizing CEP. "
            "CEP is computed only on frames with at least this many associated pedestrian points."
        ),
    )
    parser.add_argument(
        "--cep-plot-min-points",
        type=int,
        default=10,
        help="Associated-point threshold used by generated CEP presentation plots.",
    )
    parser.add_argument(
        "--cep-control-valid-rate",
        type=float,
        default=0.8,
        help=(
            "Valid-frame-rate target for control-knob plots that choose the minimum PPS per distance."
        ),
    )

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-height", type=int, default=720)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=120.0)
    return parser.parse_args()


def experiment_id() -> str:
    return f"radar_ped_distance_pps_{time.strftime('%Y%m%d_%H%M%S')}"


def deg_to_rad(value: float) -> float:
    return float(value) * math.pi / 180.0


def transform_matrix(tf: "carla.Transform") -> np.ndarray:
    return np.asarray(tf.get_matrix(), dtype=np.float64)


def transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_xyz.astype(np.float64), np.ones((points_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (matrix @ homogeneous.T).T[:, :3]


def radar_measurement_to_arrays(measurement: "carla.RadarMeasurement", radar_tf: "carla.Transform") -> Dict[str, np.ndarray]:
    raw = np.frombuffer(bytes(measurement.raw_data), dtype=np.float32)
    if raw.size == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "local_xyz": np.zeros((0, 3), dtype=np.float64),
            "world_xyz": np.zeros((0, 3), dtype=np.float64),
            "velocity_mps": empty,
            "azimuth_rad": empty,
            "altitude_rad": empty,
            "depth_m": empty,
        }
    detections = raw.reshape((-1, 4)).astype(np.float64)
    velocity = detections[:, 0]
    azimuth = detections[:, 1]
    altitude = detections[:, 2]
    depth = detections[:, 3]
    x = depth * np.cos(altitude) * np.cos(azimuth)
    y = depth * np.cos(altitude) * np.sin(azimuth)
    z = depth * np.sin(altitude)
    local_xyz = np.stack([x, y, z], axis=1)
    world_xyz = transform_points(local_xyz, transform_matrix(radar_tf))
    return {
        "local_xyz": local_xyz,
        "world_xyz": world_xyz,
        "velocity_mps": velocity,
        "azimuth_rad": azimuth,
        "altitude_rad": altitude,
        "depth_m": depth,
    }


def forward_right_vectors(yaw_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    yaw = deg_to_rad(yaw_deg)
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    right = np.asarray([math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0), 0.0], dtype=np.float64)
    return forward, right


def vector_to_array(vector: object) -> np.ndarray:
    return np.asarray(
        [float(getattr(vector, "x", 0.0)), float(getattr(vector, "y", 0.0)), float(getattr(vector, "z", 0.0))],
        dtype=np.float64,
    )


def transform_forward_right(tf: "carla.Transform") -> Tuple[np.ndarray, np.ndarray]:
    try:
        forward = vector_to_array(tf.get_forward_vector())
        right = vector_to_array(tf.get_right_vector())
    except Exception:
        forward, right = forward_right_vectors(tf.rotation.yaw)
    forward[2] = 0.0
    right[2] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    right_norm = float(np.linalg.norm(right))
    if forward_norm > 1e-6:
        forward /= forward_norm
    if right_norm > 1e-6:
        right /= right_norm
    return forward, right


def radar_world_pose(ego: "carla.Actor", args: argparse.Namespace) -> "carla.Transform":
    sensor_tf = carla.Transform(
        carla.Location(x=float(args.radar_x), y=float(args.radar_y), z=float(args.radar_z)),
        carla.Rotation(
            pitch=float(args.radar_pitch),
            yaw=float(args.radar_yaw),
            roll=float(args.radar_roll),
        ),
    )
    ego_mat = transform_matrix(ego.get_transform())
    sensor_mat = transform_matrix(sensor_tf)
    world_mat = ego_mat @ sensor_mat
    loc = carla.Location(x=float(world_mat[0, 3]), y=float(world_mat[1, 3]), z=float(world_mat[2, 3]))
    # Translation is enough for placement; yaw uses ego yaw + radar yaw.
    rot = carla.Rotation(
        pitch=float(args.radar_pitch),
        yaw=float(ego.get_transform().rotation.yaw + args.radar_yaw),
        roll=float(args.radar_roll),
    )
    return carla.Transform(loc, rot)


def actor_bbox_center_world(actor: "carla.Actor") -> np.ndarray:
    bbox = actor.bounding_box
    actor_matrix = transform_matrix(actor.get_transform())
    center_h = np.asarray([bbox.location.x, bbox.location.y, bbox.location.z, 1.0], dtype=np.float64)
    return (actor_matrix @ center_h)[:3]


def point_to_actor_local(point_world: np.ndarray, actor: "carla.Actor") -> np.ndarray:
    inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    point_h = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    return (inverse @ point_h)[:3]


def point_to_transform_local(point_world: np.ndarray, tf: "carla.Transform") -> np.ndarray:
    inverse = np.asarray(tf.get_inverse_matrix(), dtype=np.float64)
    point_h = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    return (inverse @ point_h)[:3]


def place_walker(
    walker: "carla.Actor",
    ego: "carla.Actor",
    args: argparse.Namespace,
    distance_m: float,
    reference_tf: Optional["carla.Transform"] = None,
    lateral_offset_m: float = 0.0,
    forward_offset_m: float = 0.0,
    yaw_override_deg: Optional[float] = None,
) -> "carla.Transform":
    radar_tf = reference_tf if reference_tf is not None else radar_world_pose(ego, args)
    forward, right = transform_forward_right(radar_tf)
    origin = np.asarray([radar_tf.location.x, radar_tf.location.y, radar_tf.location.z], dtype=np.float64)
    location_xyz = (
        origin
        + forward * (float(distance_m) + float(forward_offset_m))
        + right * (float(args.walker_lateral_m) + float(lateral_offset_m))
    )
    ground_z = float(ego.get_location().z)
    try:
        waypoint = walker.get_world().get_map().get_waypoint(
            carla.Location(x=float(location_xyz[0]), y=float(location_xyz[1]), z=float(ground_z) + 2.0),
            project_to_road=True,
            lane_type=carla.LaneType.Any,
        )
        if waypoint is not None:
            ground_z = float(waypoint.transform.location.z)
    except Exception:
        pass
    half_height = float(getattr(walker.bounding_box.extent, "z", 0.9))
    if str(getattr(args, "walker_z_mode", "feet")) == "bbox_center":
        actor_z = ground_z + half_height + float(args.walker_z_offset_m)
    else:
        actor_z = ground_z + float(args.walker_z_offset_m)
    if yaw_override_deg is not None:
        yaw = float(yaw_override_deg)
    else:
        yaw = radar_tf.rotation.yaw + 180.0 if bool(args.walker_face_radar) else radar_tf.rotation.yaw
    tf = carla.Transform(
        carla.Location(
            x=float(location_xyz[0]),
            y=float(location_xyz[1]),
            z=float(actor_z),
        ),
        carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0),
    )
    walker.set_transform(tf)
    if bool(getattr(args, "debug_placement", False)):
        center = actor_bbox_center_world(walker)
        local = point_to_transform_local(center, radar_tf)
        extent = walker.bounding_box.extent
        try:
            world = walker.get_world()
            marker_life = max(0.05, float(getattr(args, "debug_marker_life_s", 2.0)))
            world.debug.draw_point(
                carla.Location(x=float(center[0]), y=float(center[1]), z=float(center[2])),
                size=0.25,
                color=carla.Color(255, 0, 255),
                life_time=marker_life,
            )
            world.debug.draw_string(
                tf.location + carla.Location(z=2.0),
                f"walker {distance_m:.1f}m",
                draw_shadow=True,
                color=carla.Color(255, 0, 255),
                life_time=marker_life,
            )
        except Exception:
            pass
        print(
            "Placed walker "
            f"target={distance_m:.1f}m world=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
            f"radar_local=({local[0]:.2f},{local[1]:.2f},{local[2]:.2f}) "
            f"tf_z={tf.location.z:.2f} ground_z={ground_z:.2f} "
            f"bbox_extent=({extent.x:.2f},{extent.y:.2f},{extent.z:.2f}) "
            f"z_mode={getattr(args, 'walker_z_mode', 'feet')}"
        )
    return tf


def reset_walker_motion(
    walker: "carla.Actor",
    ego: "carla.Actor",
    args: argparse.Namespace,
    distance_m: float,
    reference_tf: "carla.Transform",
    amplitude_m: Optional[float] = None,
) -> Dict[str, object]:
    mode = str(getattr(args, "walker_motion_mode", "stationary"))
    amplitude = max(
        0.0,
        float(getattr(args, "walker_motion_amplitude_m", 0.0) if amplitude_m is None else amplitude_m),
    )
    state: Dict[str, object] = {
        "mode": mode,
        "target_distance_m": float(distance_m),
        "direction": 1.0,
        "amplitude_m": amplitude,
        "coordinate_m": -amplitude,
        "cycle_frame_index": 0,
        "last_coordinate_m": None,
        "stuck_frames": 0,
    }
    if mode == "cross" and amplitude > 0.0:
        _forward, right = transform_forward_right(reference_tf)
        yaw = math.degrees(math.atan2(float(right[1]), float(right[0])))
        place_walker(
            walker,
            ego,
            args,
            float(distance_m),
            reference_tf=reference_tf,
            lateral_offset_m=-amplitude,
            yaw_override_deg=yaw,
        )
    elif mode == "toward_away" and amplitude > 0.0:
        forward, _right = transform_forward_right(reference_tf)
        yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0])))
        place_walker(
            walker,
            ego,
            args,
            float(distance_m),
            reference_tf=reference_tf,
            forward_offset_m=-amplitude,
            yaw_override_deg=yaw,
        )
    else:
        place_walker(walker, ego, args, float(distance_m), reference_tf=reference_tf)
        state["mode"] = "stationary"
    return state


def update_walker_motion(
    ego: "carla.Actor",
    walker: "carla.Actor",
    args: argparse.Namespace,
    reference_tf: "carla.Transform",
    state: Dict[str, object],
) -> None:
    mode = str(state.get("mode", "stationary"))
    if mode == "stationary":
        return
    amplitude = max(0.0, float(state.get("amplitude_m", 0.0)))
    speed = max(0.0, float(getattr(args, "walker_motion_speed_mps", 0.0)))
    if amplitude <= 0.0 or speed <= 0.0:
        return

    forward, right = transform_forward_right(reference_tf)
    control_mode = str(getattr(args, "walker_motion_control", "walker_control"))
    if control_mode == "kinematic_cycle":
        total_frames = max(1, int(getattr(args, "frames_per_condition", 1)) - 1)
        frame_index = int(state.get("cycle_frame_index", 0))
        cycle_count = max(0.0, float(getattr(args, "walker_kinematic_cycle_count", 1.0)))
        progress = min(1.0, max(0.0, float(frame_index) / float(total_frames)))
        phase = progress * cycle_count
        if cycle_count <= 0.5:
            # Single one-way pass: left edge to right edge.
            t = min(1.0, max(0.0, phase / max(cycle_count, 1e-6)))
            coordinate = -amplitude + 2.0 * amplitude * t
            direction = 1.0
        else:
            # Triangular wave: left edge -> right edge -> left edge.
            t = phase % 1.0
            if t <= 0.5:
                coordinate = -amplitude + 4.0 * amplitude * t
                direction = 1.0
            else:
                coordinate = amplitude - 4.0 * amplitude * (t - 0.5)
                direction = -1.0
        state["coordinate_m"] = coordinate
        state["direction"] = direction
        state["cycle_frame_index"] = frame_index + 1
        if mode == "cross":
            yaw = math.degrees(math.atan2(float(right[1] * direction), float(right[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                lateral_offset_m=coordinate,
                yaw_override_deg=yaw,
            )
            velocity_axis = right
        else:
            yaw = math.degrees(math.atan2(float(forward[1] * direction), float(forward[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                forward_offset_m=coordinate,
                yaw_override_deg=yaw,
            )
            velocity_axis = forward
        try:
            walker.set_target_velocity(
                carla.Vector3D(
                    x=float(velocity_axis[0] * direction * speed),
                    y=float(velocity_axis[1] * direction * speed),
                    z=0.0,
                )
            )
            walker.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(
                        x=float(velocity_axis[0] * direction),
                        y=float(velocity_axis[1] * direction),
                        z=0.0,
                    ),
                    speed=float(speed),
                    jump=False,
                )
            )
        except Exception:
            pass
        return

    if control_mode == "kinematic":
        dt = 1.0 / max(0.1, float(getattr(args, "fps", 10.0)))
        coordinate = float(state.get("coordinate_m", -amplitude))
        direction = float(state.get("direction", 1.0))
        coordinate += direction * speed * dt
        if coordinate >= amplitude:
            coordinate = amplitude
            direction = -1.0
        elif coordinate <= -amplitude:
            coordinate = -amplitude
            direction = 1.0
        state["coordinate_m"] = coordinate
        state["direction"] = direction
        if mode == "cross":
            yaw = math.degrees(math.atan2(float(right[1] * direction), float(right[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                lateral_offset_m=coordinate,
                yaw_override_deg=yaw,
            )
            velocity_axis = right
        else:
            yaw = math.degrees(math.atan2(float(forward[1] * direction), float(forward[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                forward_offset_m=coordinate,
                yaw_override_deg=yaw,
            )
            velocity_axis = forward
        try:
            walker.set_target_velocity(
                carla.Vector3D(
                    x=float(velocity_axis[0] * direction * speed),
                    y=float(velocity_axis[1] * direction * speed),
                    z=0.0,
                )
            )
            walker.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(
                        x=float(velocity_axis[0] * direction),
                        y=float(velocity_axis[1] * direction),
                        z=0.0,
                    ),
                    speed=float(speed),
                    jump=False,
                )
            )
        except Exception:
            pass
        return

    walker_center = actor_bbox_center_world(walker)
    walker_local = point_to_transform_local(walker_center, reference_tf)
    if mode == "cross":
        coordinate = float(walker_local[1]) - float(getattr(args, "walker_lateral_m", 0.0))
        axis = right
    else:
        coordinate = float(walker_local[0]) - float(state.get("target_distance_m", 0.0))
        axis = forward

    direction = float(state.get("direction", 1.0))
    if coordinate >= amplitude:
        direction = -1.0
    elif coordinate <= -amplitude:
        direction = 1.0
    state["direction"] = direction
    previous_coordinate = state.get("last_coordinate_m")
    if previous_coordinate is not None:
        progress = abs(float(coordinate) - float(previous_coordinate))
        if progress < max(0.01, float(speed) / max(1.0, float(getattr(args, "fps", 10.0))) * 0.15):
            state["stuck_frames"] = int(state.get("stuck_frames", 0)) + 1
        else:
            state["stuck_frames"] = 0
    state["last_coordinate_m"] = coordinate

    if control_mode == "walker_control_nudge" and int(state.get("stuck_frames", 0)) >= int(args.walker_nudge_after_frames):
        dt = 1.0 / max(0.1, float(getattr(args, "fps", 10.0)))
        next_coordinate = max(-amplitude, min(amplitude, coordinate + direction * speed * dt))
        if mode == "cross":
            yaw = math.degrees(math.atan2(float(right[1] * direction), float(right[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                lateral_offset_m=next_coordinate,
                yaw_override_deg=yaw,
            )
        else:
            yaw = math.degrees(math.atan2(float(forward[1] * direction), float(forward[0] * direction)))
            place_walker(
                walker,
                ego,
                args,
                float(state.get("target_distance_m", 0.0)),
                reference_tf=reference_tf,
                forward_offset_m=next_coordinate,
                yaw_override_deg=yaw,
            )
        state["coordinate_m"] = next_coordinate
        state["last_coordinate_m"] = next_coordinate
        state["stuck_frames"] = 0

    axis = axis * direction
    try:
        walker.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(x=float(axis[0]), y=float(axis[1]), z=0.0),
                speed=float(speed),
                jump=False,
            )
        )
    except Exception:
        # Some CARLA builds ignore direct walker control when physics are off.
        # The row-level motion fields make that visible in the resulting CSV.
        pass


def points_inside_person_radius(
    points_world: np.ndarray,
    walker: "carla.Actor",
    radius_m: float,
    z_down_m: float,
    z_up_m: float,
) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    loc = walker.get_location()
    dx = points_world[:, 0] - float(loc.x)
    dy = points_world[:, 1] - float(loc.y)
    dz = points_world[:, 2] - float(loc.z)
    return (
        (dx * dx + dy * dy <= float(radius_m) * float(radius_m))
        & (dz >= -abs(float(z_down_m)))
        & (dz <= abs(float(z_up_m)))
    )


def points_inside_actor_box(points_world: np.ndarray, actor: "carla.Actor", margin_m: float) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    bbox = actor.bounding_box
    inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=np.float64)
    homogeneous = np.concatenate(
        [points_world.astype(np.float64), np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    local = (inverse @ homogeneous.T).T[:, :3]
    local -= np.asarray([bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64)[None, :]
    margin = max(0.0, float(margin_m))
    return (
        (np.abs(local[:, 0]) <= float(bbox.extent.x) + margin)
        & (np.abs(local[:, 1]) <= float(bbox.extent.y) + margin)
        & (np.abs(local[:, 2]) <= float(bbox.extent.z) + margin)
    )


def camera_image_to_bgr(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))[:, :, :3].copy()


def camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = float(width) / (2.0 * math.tan(deg_to_rad(float(fov_deg)) / 2.0))
    return np.asarray(
        [[focal, 0.0, float(width) / 2.0], [0.0, focal, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_point_to_camera(
    point_world: np.ndarray,
    camera_tf: "carla.Transform",
    width: int,
    height: int,
    fov_deg: float,
) -> Tuple[float, float, float, bool]:
    local = point_to_transform_local(point_world, camera_tf)
    depth = float(local[0])
    if depth <= 0.05:
        return float("nan"), float("nan"), depth, False
    k = camera_intrinsics(width, height, fov_deg)
    u = float(k[0, 2] + (local[1] / depth) * k[0, 0])
    v = float(k[1, 2] - (local[2] / depth) * k[1, 1])
    visible = 0.0 <= u < float(width) and 0.0 <= v < float(height)
    return u, v, depth, visible


def draw_preview(image_bgr: np.ndarray, row: Dict[str, object], args: argparse.Namespace) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for --preview") from exc
    original_h, original_w = image_bgr.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    if args.preview_width > 0 and args.preview_height > 0:
        scale_x = float(args.preview_width) / float(original_w)
        scale_y = float(args.preview_height) / float(original_h)
        image_bgr = cv2.resize(
            image_bgr,
            (int(args.preview_width), int(args.preview_height)),
            interpolation=cv2.INTER_AREA,
        )
    marker_visible = False
    try:
        u = float(row.get("walker_camera_u", "nan"))
        v = float(row.get("walker_camera_v", "nan"))
        if math.isfinite(u) and math.isfinite(v):
            px = int(round(u * scale_x))
            py = int(round(v * scale_y))
            marker_visible = 0 <= px < image_bgr.shape[1] and 0 <= py < image_bgr.shape[0]
            color = (255, 0, 255) if marker_visible else (0, 180, 255)
            cv2.circle(image_bgr, (max(0, min(image_bgr.shape[1] - 1, px)), max(0, min(image_bgr.shape[0] - 1, py))), 12, color, 2, cv2.LINE_AA)
            cv2.putText(
                image_bgr,
                f"walker marker depth={float(row.get('walker_camera_depth_m', 0.0)):.1f}m",
                (max(8, min(image_bgr.shape[1] - 300, px + 12)), max(24, min(image_bgr.shape[0] - 8, py - 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_bgr,
                f"walker marker depth={float(row.get('walker_camera_depth_m', 0.0)):.1f}m",
                (max(8, min(image_bgr.shape[1] - 300, px + 12)), max(24, min(image_bgr.shape[0] - 8, py - 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
    except Exception:
        marker_visible = False
    lines = [
        "CARLA radar pedestrian PPS/distance diagnostic",
        (
            f"pps={row['radar_pps']} dist={float(row['target_distance_m']):.1f}m "
            f"motion={row.get('walker_motion_mode', 'stationary')}/{row.get('walker_motion_control', 'walker_control')} "
            f"physics={row['walker_physics']}"
        ),
        f"radar pts/frame={row['total_radar_points']} radius_pts={row['person_radius_points']} bbox_pts={row['person_bbox_points']}",
        f"depth_window_pts={row['depth_window_points']} nearest_depth={row['nearest_depth_m']}",
        (
            "walker local x/y/z="
            f"{float(row.get('walker_radar_x_m', float('nan'))):.1f}/"
            f"{float(row.get('walker_radar_y_m', float('nan'))):.1f}/"
            f"{float(row.get('walker_radar_z_m', float('nan'))):.1f}; "
            f"camera marker={'visible' if marker_visible else 'off-screen'}"
        ),
        "press q to stop",
    ]
    y = 28
    for line in lines:
        cv2.putText(image_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return image_bgr


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def radar_xy_estimate(
    points_local: np.ndarray,
    mask: np.ndarray,
    gt_x_m: float,
    gt_y_m: float,
    prefix: str,
) -> Dict[str, object]:
    selected = points_local[mask]
    if selected.size == 0:
        return {
            f"{prefix}_mean_x_m": "",
            f"{prefix}_mean_y_m": "",
            f"{prefix}_median_x_m": "",
            f"{prefix}_median_y_m": "",
            f"{prefix}_mean_xy_error_m": "",
            f"{prefix}_median_xy_error_m": "",
        }
    xy = selected[:, :2].astype(np.float64)
    mean_xy = np.mean(xy, axis=0)
    median_xy = np.median(xy, axis=0)
    gt_xy = np.asarray([float(gt_x_m), float(gt_y_m)], dtype=np.float64)
    mean_error = float(np.linalg.norm(mean_xy - gt_xy))
    median_error = float(np.linalg.norm(median_xy - gt_xy))
    return {
        f"{prefix}_mean_x_m": float(mean_xy[0]),
        f"{prefix}_mean_y_m": float(mean_xy[1]),
        f"{prefix}_median_x_m": float(median_xy[0]),
        f"{prefix}_median_y_m": float(median_xy[1]),
        f"{prefix}_mean_xy_error_m": mean_error,
        f"{prefix}_median_xy_error_m": median_error,
    }


def finite_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def summarize_condition(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    radius_counts = [int(row["person_radius_points"]) for row in rows]
    bbox_counts = [int(row["person_bbox_points"]) for row in rows]
    depth_counts = [int(row["depth_window_points"]) for row in rows]
    total_counts = [int(row["total_radar_points"]) for row in rows]
    nearest_depths = [
        float(row["nearest_depth_m"])
        for row in rows
        if row.get("nearest_depth_m") not in ("", None) and math.isfinite(float(row["nearest_depth_m"]))
    ]
    def rate(values: Sequence[int]) -> float:
        return float(sum(1 for value in values if value > 0) / len(values)) if values else float("nan")

    def threshold_rate(values: Sequence[int], threshold: int) -> float:
        return float(sum(1 for value in values if value >= threshold) / len(values)) if values else float("nan")

    return {
        "frames": len(rows),
        "radius_support_rate": rate(radius_counts),
        "radius_support_rate_ge5": threshold_rate(radius_counts, 5),
        "radius_support_rate_ge10": threshold_rate(radius_counts, 10),
        "bbox_support_rate": rate(bbox_counts),
        "bbox_support_rate_ge5": threshold_rate(bbox_counts, 5),
        "bbox_support_rate_ge10": threshold_rate(bbox_counts, 10),
        "depth_window_support_rate": rate(depth_counts),
        "mean_radius_points": float(mean(radius_counts)) if radius_counts else float("nan"),
        "median_radius_points": float(median(radius_counts)) if radius_counts else float("nan"),
        "mean_bbox_points": float(mean(bbox_counts)) if bbox_counts else float("nan"),
        "median_bbox_points": float(median(bbox_counts)) if bbox_counts else float("nan"),
        "mean_depth_window_points": float(mean(depth_counts)) if depth_counts else float("nan"),
        "mean_total_radar_points": float(mean(total_counts)) if total_counts else float("nan"),
        "median_nearest_depth_m": float(median(nearest_depths)) if nearest_depths else "",
    }


def summarize_cep_rows(
    rows: Sequence[Dict[str, object]],
    *,
    association: str,
    estimator: str,
    min_points: int,
) -> Dict[str, object]:
    count_field = f"person_{association}_points"
    error_field = f"{association}_{estimator}_xy_error_m"
    valid_errors: List[float] = []
    point_counts: List[int] = []
    for row in rows:
        try:
            count = int(row.get(count_field, 0))
        except (TypeError, ValueError):
            count = 0
        error = finite_float(row.get(error_field))
        if count >= int(min_points) and error is not None:
            point_counts.append(count)
            valid_errors.append(error)
    frames = len(rows)
    return {
        "frames": frames,
        "valid_frames": len(valid_errors),
        "valid_rate": float(len(valid_errors) / frames) if frames else float("nan"),
        "mean_points_in_valid_frames": float(mean(point_counts)) if point_counts else "",
        "median_points_in_valid_frames": float(median(point_counts)) if point_counts else "",
        "mean_xy_error_m": float(mean(valid_errors)) if valid_errors else "",
        "cep50_m": percentile(valid_errors, 50.0) if valid_errors else "",
        "cep90_m": percentile(valid_errors, 90.0) if valid_errors else "",
        "cep95_m": percentile(valid_errors, 95.0) if valid_errors else "",
    }


def write_summary_outputs(
    output_dir: Path,
    frame_rows: Sequence[Dict[str, object]],
    *,
    plot_min_distance_m: float = 10.0,
    cep_min_points: Sequence[int] = (1, 5, 10, 20, 25),
    cep_plot_min_points: int = 10,
    cep_control_valid_rate: float = 0.8,
) -> None:
    groups: Dict[Tuple[str, str, str, int, float, float], List[Dict[str, object]]] = {}
    for row in frame_rows:
        key = (
            str(row.get("walker_motion_mode", "stationary")),
            str(row.get("walker_motion_control", "walker_control")),
            str(row["walker_physics"]),
            int(row["radar_pps"]),
            float(row["target_distance_m"]),
            float(row.get("walker_motion_amplitude_m", 0.0)),
        )
        groups.setdefault(key, []).append(row)
    summary_rows: List[Dict[str, object]] = []
    for (motion, control, physics, pps, distance, amplitude), rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = summarize_condition(rows)
        summary_rows.append(
            {
                "walker_motion_mode": motion,
                "walker_motion_control": control,
                "walker_physics": physics,
                "radar_pps": pps,
                "target_distance_m": distance,
                "walker_motion_amplitude_m": amplitude,
                **summary,
            }
        )
    summary_fields = [
        "walker_motion_mode",
        "walker_motion_control",
        "walker_physics",
        "radar_pps",
        "target_distance_m",
        "walker_motion_amplitude_m",
        "frames",
        "radius_support_rate",
        "radius_support_rate_ge5",
        "radius_support_rate_ge10",
        "bbox_support_rate",
        "bbox_support_rate_ge5",
        "bbox_support_rate_ge10",
        "depth_window_support_rate",
        "mean_radius_points",
        "median_radius_points",
        "mean_bbox_points",
        "median_bbox_points",
        "mean_depth_window_points",
        "mean_total_radar_points",
        "median_nearest_depth_m",
    ]
    write_csv(output_dir / "summary_by_condition.csv", summary_rows, summary_fields)

    cep_rows: List[Dict[str, object]] = []
    for (motion, control, physics, pps, distance, amplitude), rows in sorted(groups.items(), key=lambda item: item[0]):
        for association in ("radius", "bbox"):
            for estimator in ("mean", "median"):
                for min_points in cep_min_points:
                    cep_rows.append(
                        {
                            "walker_motion_mode": motion,
                            "walker_motion_control": control,
                            "walker_physics": physics,
                            "radar_pps": pps,
                            "target_distance_m": distance,
                            "walker_motion_amplitude_m": amplitude,
                            "association": association,
                            "estimator": estimator,
                            "min_points": int(min_points),
                            **summarize_cep_rows(
                                rows,
                                association=association,
                                estimator=estimator,
                                min_points=int(min_points),
                            ),
                        }
                    )
    cep_fields = [
        "walker_motion_mode",
        "walker_motion_control",
        "walker_physics",
        "radar_pps",
        "target_distance_m",
        "walker_motion_amplitude_m",
        "association",
        "estimator",
        "min_points",
        "frames",
        "valid_frames",
        "valid_rate",
        "mean_points_in_valid_frames",
        "median_points_in_valid_frames",
        "mean_xy_error_m",
        "cep50_m",
        "cep90_m",
        "cep95_m",
    ]
    write_csv(output_dir / "cep_by_condition.csv", cep_rows, cep_fields)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None  # type: ignore
    if plt is not None and summary_rows:
        min_plot_distance = float(plot_min_distance_m)
        for physics in sorted({str(row["walker_physics"]) for row in summary_rows}):
            subset = [row for row in summary_rows if str(row["walker_physics"]) == physics]
            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted(
                    [
                        row
                        for row in subset
                        if int(row["radar_pps"]) == pps
                        and float(row["target_distance_m"]) >= min_plot_distance
                    ],
                    key=lambda row: float(row["target_distance_m"]),
                )
                if not pts:
                    continue
                ax.plot(
                    [float(row["target_distance_m"]) for row in pts],
                    [float(row["mean_radius_points"]) for row in pts],
                    marker="o",
                    linewidth=2,
                    label=f"{pps} PPS",
                )
            ax.set_title(f"Pedestrian radar points vs distance ({physics} physics)")
            ax.set_xlabel("Pedestrian distance from radar (m)")
            ax.set_ylabel("Mean radar points inside pedestrian radius")
            ax.set_xlim(left=min_plot_distance)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.savefig(output_dir / f"mean_person_radar_points_vs_distance_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"mean_person_radar_points_vs_distance_{physics}.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted(
                    [
                        row
                        for row in subset
                        if int(row["radar_pps"]) == pps
                        and float(row["target_distance_m"]) >= min_plot_distance
                    ],
                    key=lambda row: float(row["target_distance_m"]),
                )
                if not pts:
                    continue
                ax.plot(
                    [float(row["target_distance_m"]) for row in pts],
                    [float(row["radius_support_rate"]) for row in pts],
                    marker="o",
                    linewidth=2,
                    label=f"{pps} PPS",
                )
            ax.set_title(f"Pedestrian radar support rate vs distance ({physics} physics)")
            ax.set_xlabel("Pedestrian distance from radar (m)")
            ax.set_ylabel("Frames with >=1 pedestrian radar point")
            ax.set_xlim(left=min_plot_distance)
            ax.set_ylim(0.0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.savefig(output_dir / f"person_radar_support_rate_vs_distance_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"person_radar_support_rate_vs_distance_{physics}.pdf")
            plt.close(fig)

        for physics in sorted({str(row["walker_physics"]) for row in cep_rows}):
            subset = [
                row
                for row in cep_rows
                if str(row["walker_physics"]) == physics
                and str(row["association"]) == "bbox"
                and str(row["estimator"]) == "median"
                and int(row["min_points"]) == int(cep_plot_min_points)
            ]
            if not subset:
                continue
            fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted(
                    [
                        row
                        for row in subset
                        if int(row["radar_pps"]) == pps
                        and float(row["target_distance_m"]) >= min_plot_distance
                        and finite_float(row.get("cep50_m")) is not None
                    ],
                    key=lambda row: float(row["target_distance_m"]),
                )
                if not pts:
                    continue
                ax.plot(
                    [float(row["target_distance_m"]) for row in pts],
                    [float(row["cep50_m"]) for row in pts],
                    marker="o",
                    linewidth=2,
                    label=f"{pps} PPS",
                )
            ax.set_title(f"Radar-only pedestrian CEP50 vs distance ({physics} physics)")
            ax.set_xlabel("Pedestrian distance from radar (m)")
            ax.set_ylabel(f"CEP50 XY error (m), bbox median, >= {int(cep_plot_min_points)} pts")
            ax.set_xlim(left=min_plot_distance)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False, ncols=2)
            fig.savefig(output_dir / f"pedestrian_cep50_vs_distance_bbox_median_ge{int(cep_plot_min_points)}_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"pedestrian_cep50_vs_distance_bbox_median_ge{int(cep_plot_min_points)}_{physics}.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted(
                    [
                        row
                        for row in subset
                        if int(row["radar_pps"]) == pps
                        and float(row["target_distance_m"]) >= min_plot_distance
                        and finite_float(row.get("valid_rate")) is not None
                    ],
                    key=lambda row: float(row["target_distance_m"]),
                )
                if not pts:
                    continue
                ax.plot(
                    [float(row["target_distance_m"]) for row in pts],
                    [float(row["valid_rate"]) for row in pts],
                    marker="o",
                    linewidth=2,
                    label=f"{pps} PPS",
                )
            ax.set_title(f"Frames with enough pedestrian radar points ({physics} physics)")
            ax.set_xlabel("Pedestrian distance from radar (m)")
            ax.set_ylabel(f"Fraction of frames with >= {int(cep_plot_min_points)} bbox points")
            ax.set_xlim(left=min_plot_distance)
            ax.set_ylim(0.0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False, ncols=2)
            fig.savefig(output_dir / f"pedestrian_cep_valid_rate_bbox_median_ge{int(cep_plot_min_points)}_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"pedestrian_cep_valid_rate_bbox_median_ge{int(cep_plot_min_points)}_{physics}.pdf")
            plt.close(fig)

            target_rate = float(cep_control_valid_rate)
            rate_tag = int(round(target_rate * 100.0))
            for metric, metric_label in (("cep50_m", "CEP50"), ("cep90_m", "CEP90")):
                selected_distances: List[float] = []
                selected_pps: List[int] = []
                selected_metric: List[float] = []
                for distance in sorted({float(row["target_distance_m"]) for row in subset}):
                    if distance < min_plot_distance:
                        continue
                    candidates = []
                    for row in subset:
                        if float(row["target_distance_m"]) != distance:
                            continue
                        value = finite_float(row.get(metric))
                        valid_rate = finite_float(row.get("valid_rate"))
                        if value is None or valid_rate is None:
                            continue
                        if valid_rate >= target_rate:
                            candidates.append((int(row["radar_pps"]), value))
                    if not candidates:
                        continue
                    pps, value = sorted(candidates, key=lambda item: item[0])[0]
                    selected_distances.append(distance)
                    selected_pps.append(pps)
                    selected_metric.append(value)
                if not selected_distances:
                    continue
                fig, ax_pps = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
                ax_cep = ax_pps.twinx()
                ax_pps.plot(
                    selected_distances,
                    selected_pps,
                    marker="s",
                    linewidth=2.4,
                    color="#1f77b4",
                    label=f"Minimum PPS for >= {rate_tag}% useful frames",
                )
                ax_cep.plot(
                    selected_distances,
                    selected_metric,
                    marker="o",
                    linewidth=2.4,
                    color="#d62728",
                    label=f"{metric_label} at selected PPS",
                )
                ax_pps.set_title(
                    f"Radar PPS control knob vs pedestrian {metric_label} ({physics} physics)"
                )
                ax_pps.set_xlabel("Pedestrian distance from radar (m)")
                ax_pps.set_ylabel("Minimum radar PPS", color="#1f77b4")
                ax_cep.set_ylabel(f"{metric_label} XY error (m)", color="#d62728")
                ax_pps.tick_params(axis="y", labelcolor="#1f77b4")
                ax_cep.tick_params(axis="y", labelcolor="#d62728")
                ax_pps.set_xlim(left=min_plot_distance)
                ax_pps.grid(axis="y", alpha=0.25)
                lines_pps, labels_pps = ax_pps.get_legend_handles_labels()
                lines_cep, labels_cep = ax_cep.get_legend_handles_labels()
                ax_pps.legend(lines_pps + lines_cep, labels_pps + labels_cep, frameon=False, loc="upper left")
                filename = (
                    f"pedestrian_{metric_label.lower()}_control_knob_bbox_median_ge"
                    f"{int(cep_plot_min_points)}_vr{rate_tag}_{physics}"
                )
                fig.savefig(output_dir / f"{filename}.png", dpi=220)
                fig.savefig(output_dir / f"{filename}.pdf")
                plt.close(fig)


def spawn_ego(world: "carla.World", args: argparse.Namespace) -> "carla.Actor":
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points in CARLA map.")
    base_tf = spawn_points[int(args.ego_spawn_index) % len(spawn_points)]
    tf = carla.Transform(
        carla.Location(
            x=base_tf.location.x,
            y=base_tf.location.y,
            z=base_tf.location.z + float(args.ego_z_offset_m),
        ),
        base_tf.rotation,
    )
    library = world.get_blueprint_library()
    try:
        bp = library.find(str(args.ego_blueprint))
    except RuntimeError:
        matches = list(library.filter(str(args.ego_blueprint)))
        if not matches:
            matches = list(library.filter("vehicle.*"))
        if not matches:
            raise RuntimeError(f"No vehicle blueprint found for {args.ego_blueprint!r}")
        bp = random.choice(matches)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "scenesense_radar_ped_diag_ego")
    ego = world.try_spawn_actor(bp, tf)
    if ego is None:
        raise RuntimeError(f"Failed to spawn ego at spawn index {args.ego_spawn_index}")
    ego.set_simulate_physics(False)
    return ego


def spawn_walker(world: "carla.World", args: argparse.Namespace) -> "carla.Actor":
    library = world.get_blueprint_library()
    try:
        bp = library.find(str(args.walker_blueprint))
    except RuntimeError:
        matches = list(library.filter(str(args.walker_blueprint)))
        if not matches:
            matches = list(library.filter("walker.pedestrian.*"))
        if not matches:
            raise RuntimeError(f"No pedestrian blueprint found for {args.walker_blueprint!r}")
        bp = random.choice(matches)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "scenesense_radar_ped_diag_walker")
    actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(z=2.0)))
    if actor is None:
        # Fallback to any pedestrian blueprint.
        choices = list(library.filter("walker.pedestrian.*"))
        if not choices:
            raise RuntimeError("No pedestrian blueprints found.")
        bp = random.choice(choices)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(z=2.0)))
    if actor is None:
        raise RuntimeError("Failed to spawn pedestrian actor.")
    return actor


def spawn_radar(world: "carla.World", ego: "carla.Actor", args: argparse.Namespace, pps: int, radar_queue: "queue.Queue[object]") -> "carla.Actor":
    bp = world.get_blueprint_library().find("sensor.other.radar")
    bp.set_attribute("range", str(float(args.radar_range)))
    bp.set_attribute("horizontal_fov", str(float(args.radar_hfov)))
    bp.set_attribute("vertical_fov", str(float(args.radar_vfov)))
    bp.set_attribute("points_per_second", str(int(pps)))
    tick = float(args.radar_sensor_tick)
    if tick <= 0.0:
        tick = 1.0 / max(0.1, float(args.fps))
    bp.set_attribute("sensor_tick", str(tick))
    sensor_tf = carla.Transform(
        carla.Location(x=float(args.radar_x), y=float(args.radar_y), z=float(args.radar_z)),
        carla.Rotation(pitch=float(args.radar_pitch), yaw=float(args.radar_yaw), roll=float(args.radar_roll)),
    )
    radar = world.spawn_actor(bp, sensor_tf, attach_to=ego)
    radar.listen(lambda measurement: radar_queue.put(measurement))
    return radar


def spawn_camera(world: "carla.World", ego: "carla.Actor", args: argparse.Namespace, camera_queue: "queue.Queue[object]") -> Optional["carla.Actor"]:
    if not bool(args.preview):
        return None
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(int(args.camera_width)))
    bp.set_attribute("image_size_y", str(int(args.camera_height)))
    bp.set_attribute("fov", str(float(args.camera_fov)))
    bp.set_attribute("sensor_tick", str(1.0 / max(0.1, float(args.fps))))
    sensor_tf = carla.Transform(
        carla.Location(x=float(args.radar_x), y=float(args.radar_y), z=float(args.radar_z)),
        carla.Rotation(pitch=float(args.radar_pitch), yaw=float(args.radar_yaw), roll=float(args.radar_roll)),
    )
    camera = world.spawn_actor(bp, sensor_tf, attach_to=ego)
    camera.listen(lambda image: camera_queue.put(image))
    return camera


def set_walker_physics(walker: "carla.Actor", mode: str) -> str:
    if str(mode) == "default":
        return "default"
    try:
        enabled = str(mode) == "on"
        walker.set_simulate_physics(bool(enabled))
        return "on" if enabled else "off"
    except Exception as exc:
        return f"request_{mode}_failed:{exc}"


def main() -> int:
    args = parse_args()
    if carla is None:
        raise SystemExit("Could not import carla. Run inside the CARLA PythonAPI environment.")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    pps_values = parse_int_list(str(args.pps_list))
    distances = parse_float_list(str(args.distance_list_m))
    amplitudes = amplitude_values_for_distances(args, distances)
    if args.walker_physics_mode == "both":
        physics_modes = ["on", "off"]
    elif args.walker_physics_mode == "all":
        physics_modes = ["default", "on", "off"]
    else:
        physics_modes = [str(args.walker_physics_mode)]

    output_dir = Path(args.output_root) / (args.experiment_id or experiment_id())
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.load_world(str(args.town)) if bool(args.load_town) else client.get_world()

    original_settings = world.get_settings()
    actors: List["carla.Actor"] = []
    radar: Optional["carla.Actor"] = None
    camera: Optional["carla.Actor"] = None
    radar_queue: "queue.Queue[object]" = queue.Queue()
    camera_queue: "queue.Queue[object]" = queue.Queue()
    frame_rows: List[Dict[str, object]] = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / max(0.1, float(args.fps))
        world.apply_settings(settings)
        world.tick()

        ego = spawn_ego(world, args)
        actors.append(ego)
        walker = spawn_walker(world, args)
        actors.append(walker)
        camera = spawn_camera(world, ego, args, camera_queue)
        if camera is not None:
            actors.append(camera)

        for physics_mode in physics_modes:
            physics_status = set_walker_physics(walker, physics_mode)
            for pps in pps_values:
                if radar is not None:
                    radar.stop()
                    radar.destroy()
                    actors = [actor for actor in actors if actor is not radar]
                    radar = None
                while not radar_queue.empty():
                    radar_queue.get_nowait()
                radar = spawn_radar(world, ego, args, int(pps), radar_queue)
                actors.append(radar)
                world.tick()
                while not radar_queue.empty():
                    radar_queue.get_nowait()
                for distance_m, amplitude_m in zip(distances, amplitudes):
                    motion_state = reset_walker_motion(
                        walker,
                        ego,
                        args,
                        float(distance_m),
                        reference_tf=radar.get_transform(),
                        amplitude_m=float(amplitude_m),
                    )
                    for _ in range(max(0, int(args.warmup_frames))):
                        update_walker_motion(ego, walker, args, radar.get_transform(), motion_state)
                        world.tick()
                        while not radar_queue.empty():
                            radar_queue.get_nowait()
                        while not camera_queue.empty():
                            camera_queue.get_nowait()
                    if str(motion_state.get("mode", "stationary")) != "stationary":
                        # Warmup lets the sensors settle, but for moving-walker
                        # diagnostics we want the measured condition to begin at
                        # the path edge rather than halfway through the crossing.
                        motion_state = reset_walker_motion(
                            walker,
                            ego,
                            args,
                            float(distance_m),
                            reference_tf=radar.get_transform(),
                            amplitude_m=float(amplitude_m),
                        )

                    for local_frame in range(int(args.frames_per_condition)):
                        update_walker_motion(ego, walker, args, radar.get_transform(), motion_state)
                        world_tick = int(world.tick())
                        try:
                            measurement = radar_queue.get(timeout=2.0)
                        except queue.Empty:
                            continue
                        while not radar_queue.empty():
                            measurement = radar_queue.get_nowait()
                        arrays = radar_measurement_to_arrays(measurement, radar.get_transform())
                        world_xyz = arrays["world_xyz"]
                        local_xyz = arrays["local_xyz"]
                        radius_mask = points_inside_person_radius(
                            world_xyz,
                            walker,
                            float(args.person_radius_m),
                            float(args.person_z_down_m),
                            float(args.person_z_up_m),
                        )
                        bbox_mask = points_inside_actor_box(world_xyz, walker, float(args.bbox_margin_m))
                        depth = arrays["depth_m"]
                        depth_window_mask = np.abs(depth - float(distance_m)) <= float(args.depth_window_m)
                        nearest_depth = float(np.min(np.abs(depth - float(distance_m)))) if depth.size else ""
                        walker_center = actor_bbox_center_world(walker)
                        walker_radar = point_to_transform_local(walker_center, radar.get_transform())
                        radius_xy = radar_xy_estimate(
                            local_xyz,
                            radius_mask,
                            float(walker_radar[0]),
                            float(walker_radar[1]),
                            "radius",
                        )
                        bbox_xy = radar_xy_estimate(
                            local_xyz,
                            bbox_mask,
                            float(walker_radar[0]),
                            float(walker_radar[1]),
                            "bbox",
                        )
                        walker_camera_u = walker_camera_v = walker_camera_depth = float("nan")
                        walker_camera_visible = False
                        if camera is not None:
                            walker_camera_u, walker_camera_v, walker_camera_depth, walker_camera_visible = project_point_to_camera(
                                walker_center,
                                camera.get_transform(),
                                int(args.camera_width),
                                int(args.camera_height),
                                float(args.camera_fov),
                            )
                        actual_ground_distance = float(math.sqrt(float(walker_radar[0]) ** 2 + float(walker_radar[1]) ** 2))
                        motion_coordinate = (
                            float(walker_radar[1]) - float(args.walker_lateral_m)
                            if str(motion_state.get("mode", "stationary")) == "cross"
                            else float(walker_radar[0]) - float(distance_m)
                        )
                        mean_radius_velocity = (
                            float(np.mean(arrays["velocity_mps"][radius_mask]))
                            if arrays["velocity_mps"].size and np.any(radius_mask)
                            else ""
                        )
                        mean_abs_radius_velocity = (
                            float(np.mean(np.abs(arrays["velocity_mps"][radius_mask])))
                            if arrays["velocity_mps"].size and np.any(radius_mask)
                            else ""
                        )
                        row = {
                            "world_tick_frame": world_tick,
                            "radar_frame": int(getattr(measurement, "frame", -1)),
                            "local_condition_frame": int(local_frame),
                            "walker_motion_mode": str(motion_state.get("mode", "stationary")),
                            "walker_motion_control": str(getattr(args, "walker_motion_control", "walker_control")),
                            "walker_motion_coordinate_m": float(motion_coordinate),
                            "walker_motion_direction": float(motion_state.get("direction", 0.0)),
                            "walker_motion_speed_mps": float(args.walker_motion_speed_mps)
                            if str(motion_state.get("mode", "stationary")) != "stationary"
                            else 0.0,
                            "walker_motion_amplitude_m": float(motion_state.get("amplitude_m", 0.0)),
                            "walker_physics": physics_status,
                            "radar_pps": int(pps),
                            "target_distance_m": float(distance_m),
                            "actual_ground_distance_m": actual_ground_distance,
                            "actual_depth_m": float(walker_radar[0]),
                            "actual_lateral_m": float(walker_radar[1]),
                            "walker_actor_id": int(walker.id),
                            "total_radar_points": int(world_xyz.shape[0]),
                            "person_radius_points": int(np.count_nonzero(radius_mask)),
                            "person_bbox_points": int(np.count_nonzero(bbox_mask)),
                            "depth_window_points": int(np.count_nonzero(depth_window_mask)),
                            "nearest_depth_m": nearest_depth,
                            "walker_world_x": float(walker_center[0]),
                            "walker_world_y": float(walker_center[1]),
                            "walker_world_z": float(walker_center[2]),
                            "walker_radar_x_m": float(walker_radar[0]),
                            "walker_radar_y_m": float(walker_radar[1]),
                            "walker_radar_z_m": float(walker_radar[2]),
                            "walker_camera_u": float(walker_camera_u),
                            "walker_camera_v": float(walker_camera_v),
                            "walker_camera_depth_m": float(walker_camera_depth),
                            "walker_camera_visible": int(bool(walker_camera_visible)),
                            "mean_person_radius_depth_m": (
                                float(np.mean(depth[radius_mask])) if depth.size and np.any(radius_mask) else ""
                            ),
                            "mean_person_radius_velocity_mps": mean_radius_velocity,
                            "mean_abs_person_radius_velocity_mps": mean_abs_radius_velocity,
                            **radius_xy,
                            **bbox_xy,
                        }
                        frame_rows.append(row)

                        if bool(args.preview):
                            camera_data = None
                            try:
                                camera_data = camera_queue.get_nowait()
                                while not camera_queue.empty():
                                    camera_data = camera_queue.get_nowait()
                            except queue.Empty:
                                camera_data = None
                            if camera_data is not None:
                                try:
                                    import cv2

                                    preview = draw_preview(camera_image_to_bgr(camera_data), row, args)
                                    cv2.imshow("SceneSense radar pedestrian diagnostic", preview)
                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        raise KeyboardInterrupt
                                except RuntimeError as exc:
                                    print(f"Preview disabled: {exc}")
                                    args.preview = False

                print(f"Completed physics={physics_status} pps={pps}")

        frame_fields = [
            "world_tick_frame",
            "radar_frame",
            "local_condition_frame",
            "walker_motion_mode",
            "walker_motion_control",
            "walker_motion_coordinate_m",
            "walker_motion_direction",
            "walker_motion_speed_mps",
            "walker_motion_amplitude_m",
            "walker_physics",
            "radar_pps",
            "target_distance_m",
            "actual_ground_distance_m",
            "actual_depth_m",
            "actual_lateral_m",
            "walker_actor_id",
            "total_radar_points",
            "person_radius_points",
            "person_bbox_points",
            "depth_window_points",
            "nearest_depth_m",
            "mean_person_radius_depth_m",
            "mean_person_radius_velocity_mps",
            "mean_abs_person_radius_velocity_mps",
            "radius_mean_x_m",
            "radius_mean_y_m",
            "radius_median_x_m",
            "radius_median_y_m",
            "radius_mean_xy_error_m",
            "radius_median_xy_error_m",
            "bbox_mean_x_m",
            "bbox_mean_y_m",
            "bbox_median_x_m",
            "bbox_median_y_m",
            "bbox_mean_xy_error_m",
            "bbox_median_xy_error_m",
            "walker_world_x",
            "walker_world_y",
            "walker_world_z",
            "walker_radar_x_m",
            "walker_radar_y_m",
            "walker_radar_z_m",
            "walker_camera_u",
            "walker_camera_v",
            "walker_camera_depth_m",
            "walker_camera_visible",
        ]
        write_csv(output_dir / "frame_metrics.csv", frame_rows, frame_fields)
        write_summary_outputs(
            output_dir,
            frame_rows,
            plot_min_distance_m=float(args.plot_min_distance_m),
            cep_min_points=parse_int_list(str(args.cep_min_points_list)),
            cep_plot_min_points=int(args.cep_plot_min_points),
            cep_control_valid_rate=float(args.cep_control_valid_rate),
        )
        summary = {
            "experiment_id": output_dir.name,
            "output_dir": str(output_dir.resolve()),
            "settings": vars(args),
            "radar_public_model_note": (
                "CARLA exposes radar detections as velocity, azimuth, altitude, and depth. "
                "No material/pathloss tuning knobs are exposed by the sensor blueprint; this "
                "diagnostic empirically measures returned points for our simulator build."
            ),
            "frames": len(frame_rows),
            "pps_values": pps_values,
            "distances_m": distances,
            "walker_motion_amplitudes_m": amplitudes,
            "physics_modes": physics_modes,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        if frame_rows:
            write_csv(
                output_dir / "frame_metrics_partial.csv",
                frame_rows,
                [
                    "world_tick_frame",
                    "radar_frame",
                    "local_condition_frame",
                    "walker_physics",
                    "radar_pps",
                    "target_distance_m",
                    "walker_actor_id",
                    "total_radar_points",
                    "person_radius_points",
                    "person_bbox_points",
                    "depth_window_points",
                    "nearest_depth_m",
                    "mean_person_radius_depth_m",
                    "mean_person_radius_velocity_mps",
                    "walker_world_x",
                    "walker_world_y",
                    "walker_world_z",
                    "walker_radar_x_m",
                    "walker_radar_y_m",
                    "walker_radar_z_m",
                    "walker_camera_u",
                    "walker_camera_v",
                    "walker_camera_depth_m",
                    "walker_camera_visible",
                ],
            )
        print("Stopped by user.")
        return 130
    finally:
        if radar is not None:
            try:
                radar.stop()
            except RuntimeError:
                pass
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
        if bool(args.preview):
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        for actor in reversed(actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

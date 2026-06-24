#!/usr/bin/env python3
"""Controlled CARLA radar-vs-pedestrian diagnostic.

This script isolates the radar sensor from the fusion model. It places one
stationary pedestrian directly in front of an ego-mounted radar, then sweeps
radar points-per-second and pedestrian distance. The goal is to measure whether
CARLA radar returns disappear because of distance, point density, or pedestrian
physics settings.
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
) -> "carla.Transform":
    radar_tf = reference_tf if reference_tf is not None else radar_world_pose(ego, args)
    forward, right = transform_forward_right(radar_tf)
    origin = np.asarray([radar_tf.location.x, radar_tf.location.y, radar_tf.location.z], dtype=np.float64)
    location_xyz = origin + forward * float(distance_m) + right * float(args.walker_lateral_m)
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
        f"pps={row['radar_pps']} dist={float(row['target_distance_m']):.1f}m physics={row['walker_physics']}",
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

    return {
        "frames": len(rows),
        "radius_support_rate": rate(radius_counts),
        "bbox_support_rate": rate(bbox_counts),
        "depth_window_support_rate": rate(depth_counts),
        "mean_radius_points": float(mean(radius_counts)) if radius_counts else float("nan"),
        "median_radius_points": float(median(radius_counts)) if radius_counts else float("nan"),
        "mean_bbox_points": float(mean(bbox_counts)) if bbox_counts else float("nan"),
        "median_bbox_points": float(median(bbox_counts)) if bbox_counts else float("nan"),
        "mean_depth_window_points": float(mean(depth_counts)) if depth_counts else float("nan"),
        "mean_total_radar_points": float(mean(total_counts)) if total_counts else float("nan"),
        "median_nearest_depth_m": float(median(nearest_depths)) if nearest_depths else "",
    }


def write_summary_outputs(output_dir: Path, frame_rows: Sequence[Dict[str, object]]) -> None:
    groups: Dict[Tuple[str, int, float], List[Dict[str, object]]] = {}
    for row in frame_rows:
        key = (str(row["walker_physics"]), int(row["radar_pps"]), float(row["target_distance_m"]))
        groups.setdefault(key, []).append(row)
    summary_rows: List[Dict[str, object]] = []
    for (physics, pps, distance), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        summary = summarize_condition(rows)
        summary_rows.append(
            {
                "walker_physics": physics,
                "radar_pps": pps,
                "target_distance_m": distance,
                **summary,
            }
        )
    summary_fields = [
        "walker_physics",
        "radar_pps",
        "target_distance_m",
        "frames",
        "radius_support_rate",
        "bbox_support_rate",
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None  # type: ignore
    if plt is not None and summary_rows:
        for physics in sorted({str(row["walker_physics"]) for row in summary_rows}):
            subset = [row for row in summary_rows if str(row["walker_physics"]) == physics]
            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted([row for row in subset if int(row["radar_pps"]) == pps], key=lambda row: float(row["target_distance_m"]))
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
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.savefig(output_dir / f"mean_person_radar_points_vs_distance_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"mean_person_radar_points_vs_distance_{physics}.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            for pps in sorted({int(row["radar_pps"]) for row in subset}):
                pts = sorted([row for row in subset if int(row["radar_pps"]) == pps], key=lambda row: float(row["target_distance_m"]))
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
            ax.set_ylim(0.0, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False)
            fig.savefig(output_dir / f"person_radar_support_rate_vs_distance_{physics}.png", dpi=220)
            fig.savefig(output_dir / f"person_radar_support_rate_vs_distance_{physics}.pdf")
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
                for distance_m in distances:
                    place_walker(walker, ego, args, float(distance_m), reference_tf=radar.get_transform())
                    for _ in range(max(0, int(args.warmup_frames))):
                        world.tick()
                        while not radar_queue.empty():
                            radar_queue.get_nowait()
                        while not camera_queue.empty():
                            camera_queue.get_nowait()

                    for local_frame in range(int(args.frames_per_condition)):
                        world_tick = int(world.tick())
                        try:
                            measurement = radar_queue.get(timeout=2.0)
                        except queue.Empty:
                            continue
                        while not radar_queue.empty():
                            measurement = radar_queue.get_nowait()
                        arrays = radar_measurement_to_arrays(measurement, radar.get_transform())
                        world_xyz = arrays["world_xyz"]
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
                        row = {
                            "world_tick_frame": world_tick,
                            "radar_frame": int(getattr(measurement, "frame", -1)),
                            "local_condition_frame": int(local_frame),
                            "walker_physics": physics_status,
                            "radar_pps": int(pps),
                            "target_distance_m": float(distance_m),
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
                            "mean_person_radius_velocity_mps": (
                                float(np.mean(arrays["velocity_mps"][radius_mask]))
                                if arrays["velocity_mps"].size and np.any(radius_mask)
                                else ""
                            ),
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
        ]
        write_csv(output_dir / "frame_metrics.csv", frame_rows, frame_fields)
        write_summary_outputs(output_dir, frame_rows)
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

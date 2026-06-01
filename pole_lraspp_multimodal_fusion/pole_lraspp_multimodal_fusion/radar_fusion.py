from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def radar_raw_to_alt_az_depth_velocity(raw_data: bytes) -> np.ndarray:
    """Return CARLA radar detections as [altitude, azimuth, depth, velocity]."""
    points = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4)
    if points.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    # CARLA raw order is velocity, azimuth, altitude, depth.
    return np.stack([points[:, 2], points[:, 1], points[:, 3], points[:, 0]], axis=1).astype(np.float32, copy=False)


def radar_spherical_to_local(detections: np.ndarray) -> np.ndarray:
    if detections.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    altitudes = detections[:, 0].astype(np.float64)
    azimuths = detections[:, 1].astype(np.float64)
    depths = detections[:, 2].astype(np.float64)
    x_local = depths * np.cos(altitudes) * np.cos(azimuths)
    y_local = depths * np.cos(altitudes) * np.sin(azimuths)
    z_local = depths * np.sin(altitudes)
    return np.stack([x_local, y_local, z_local], axis=1)


def transform_points(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homogeneous = np.concatenate([points_xyz.astype(np.float64), np.ones((len(points_xyz), 1), dtype=np.float64)], axis=1)
    return (matrix.astype(np.float64) @ homogeneous.T).T[:, :3]


def radar_spherical_to_world(detections: np.ndarray, sensor_matrix: np.ndarray) -> np.ndarray:
    local = radar_spherical_to_local(detections)
    world = transform_points(local, sensor_matrix)
    velocities = detections[:, 3:4].astype(np.float64) if detections.size else np.zeros((0, 1), dtype=np.float64)
    return np.concatenate([world, velocities], axis=1)


def world_to_camera_points(points_world: np.ndarray, camera_inverse_matrix: np.ndarray) -> np.ndarray:
    return transform_points(points_world, camera_inverse_matrix)


def project_camera_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_cam.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, np.zeros((0,), dtype=bool)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    in_front = x > 0.05
    u = np.zeros(points_cam.shape[0], dtype=np.float32)
    v = np.zeros(points_cam.shape[0], dtype=np.float32)
    depth = x.astype(np.float32)
    if np.any(in_front):
        u[in_front] = intrinsics[0, 2] + (y[in_front] / x[in_front]) * intrinsics[0, 0]
        v[in_front] = intrinsics[1, 2] - (z[in_front] / x[in_front]) * intrinsics[1, 1]
    return u, v, depth, in_front


@dataclass
class StationaryTrackAccumulator:
    stationary_velocity_mps: float = 0.35
    parked_threshold_s: float = 5.0
    association_grid_m: float = 1.5
    max_stale_s: float = 2.0

    def __post_init__(self) -> None:
        self._tracks: Dict[Tuple[int, int], Dict[str, float]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        scale = max(0.05, float(self.association_grid_m))
        return int(round(float(x) / scale)), int(round(float(y) / scale))

    def update(self, world_velocity_points: np.ndarray, frame_time_s: float) -> np.ndarray:
        if world_velocity_points.size == 0:
            return np.zeros((0,), dtype=np.float32)
        now = float(frame_time_s)
        ages = np.zeros((world_velocity_points.shape[0],), dtype=np.float32)
        seen = set()
        for idx, row in enumerate(world_velocity_points):
            x, y, velocity = float(row[0]), float(row[1]), float(row[3])
            key = self._key(x, y)
            seen.add(key)
            track = self._tracks.get(key, {"age_s": 0.0, "last_seen_s": now})
            dt = max(0.0, now - float(track.get("last_seen_s", now)))
            if abs(velocity) <= float(self.stationary_velocity_mps):
                track["age_s"] = min(float(self.parked_threshold_s) * 3.0, float(track.get("age_s", 0.0)) + dt)
            else:
                track["age_s"] = 0.0
            track["last_seen_s"] = now
            track["x"] = x
            track["y"] = y
            self._tracks[key] = track
            ages[idx] = float(track["age_s"])

        stale_after = max(float(self.max_stale_s), float(self.association_grid_m))
        for key in list(self._tracks):
            if key not in seen and now - float(self._tracks[key].get("last_seen_s", now)) > stale_after:
                del self._tracks[key]
        return ages


def rasterize_radar_channels(
    *,
    width: int,
    height: int,
    u: np.ndarray,
    v: np.ndarray,
    depth_m: np.ndarray,
    velocity_mps: np.ndarray,
    stationary_age_s: np.ndarray,
    valid_mask: np.ndarray,
    max_range_m: float,
    max_abs_velocity_mps: float,
    parked_threshold_s: float,
    point_radius_px: int = 2,
) -> np.ndarray:
    channels = np.zeros((4, int(height), int(width)), dtype=np.float32)
    if u.size == 0:
        return channels
    in_image = (
        valid_mask
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
        & np.isfinite(depth_m)
    )
    if not np.any(in_image):
        return channels
    radius = max(0, int(point_radius_px))
    max_range = max(1.0, float(max_range_m))
    max_velocity = max(0.1, float(max_abs_velocity_mps))
    parked_threshold = max(0.1, float(parked_threshold_s))
    for px_f, py_f, depth, velocity, age in zip(
        u[in_image], v[in_image], depth_m[in_image], velocity_mps[in_image], stationary_age_s[in_image]
    ):
        px = int(round(float(px_f)))
        py = int(round(float(py_f)))
        y0, y1 = max(0, py - radius), min(int(height), py + radius + 1)
        x0, x1 = max(0, px - radius), min(int(width), px + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        channels[0, y0:y1, x0:x1] = 1.0
        range_score = 1.0 - min(max(float(depth), 0.0), max_range) / max_range
        channels[1, y0:y1, x0:x1] = np.maximum(channels[1, y0:y1, x0:x1], range_score)
        vel_score = max(-1.0, min(1.0, float(velocity) / max_velocity))
        current = channels[2, y0:y1, x0:x1]
        channels[2, y0:y1, x0:x1] = np.where(np.abs(vel_score) > np.abs(current), vel_score, current)
        age_score = min(max(float(age), 0.0), parked_threshold) / parked_threshold
        channels[3, y0:y1, x0:x1] = np.maximum(channels[3, y0:y1, x0:x1], age_score)
    return channels


def build_radar_sample(
    *,
    detections: np.ndarray,
    sensor_matrix: np.ndarray,
    camera_inverse_matrix: np.ndarray,
    camera_intrinsics: np.ndarray,
    width: int,
    height: int,
    frame_time_s: float,
    tracker: StationaryTrackAccumulator,
    max_range_m: float,
    max_abs_velocity_mps: float,
    parked_threshold_s: float,
    point_radius_px: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    world_velocity = radar_spherical_to_world(detections, sensor_matrix)
    ages = tracker.update(world_velocity, frame_time_s)
    points_cam = world_to_camera_points(world_velocity[:, :3], camera_inverse_matrix) if world_velocity.size else np.zeros((0, 3), dtype=np.float64)
    u, v, depth, valid = project_camera_points(points_cam, camera_intrinsics)
    velocities = world_velocity[:, 3].astype(np.float32) if world_velocity.size else np.zeros((0,), dtype=np.float32)
    tensor = rasterize_radar_channels(
        width=width,
        height=height,
        u=u,
        v=v,
        depth_m=depth,
        velocity_mps=velocities,
        stationary_age_s=ages,
        valid_mask=valid,
        max_range_m=max_range_m,
        max_abs_velocity_mps=max_abs_velocity_mps,
        parked_threshold_s=parked_threshold_s,
        point_radius_px=point_radius_px,
    )
    points = {
        "world_xyz": world_velocity[:, :3].astype(np.float32) if world_velocity.size else np.zeros((0, 3), dtype=np.float32),
        "camera_xyz": points_cam.astype(np.float32) if points_cam.size else np.zeros((0, 3), dtype=np.float32),
        "velocity_mps": velocities.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "camera_depth_m": depth.astype(np.float32),
        "stationary_age_s": ages.astype(np.float32),
        "valid_projection": valid.astype(np.uint8),
    }
    stationary = np.abs(velocities) <= float(tracker.stationary_velocity_mps)
    parked = ages >= float(parked_threshold_s)
    summary = {
        "radar_points": float(detections.shape[0]),
        "radar_stationary_points": float(np.sum(stationary)),
        "radar_parked_evidence_points": float(np.sum(parked)),
    }
    return tensor, points, summary

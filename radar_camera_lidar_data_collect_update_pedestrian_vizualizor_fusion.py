#!/usr/bin/env python3
"""
CARLA 0.10.0 - Sensors-only script (NO spawning)

Option A IMPLEMENTED:
Keep ALL LiDAR points for geometry (do NOT discard points that fall outside RGB frustum)
Colorize ONLY the subset that projects into RGB; others get fallback color (or semantic color)
Semantic labels are also sampled where available; unlabeled points are kept by default
Voxel map integration for a more “recognizable” map
PLY export with xyz + rgb (+ optional label)

Run example (recommended):
  python3 fusion_semantic_rgb_voxel_map_v2.py \
    --fusion-map --use-semantic \
    --lidar-range 200 --lidar-upper-fov 30 --lidar-lower-fov -45 --lidar-channels 128 \
    --map-export-every 50 --voxel-size 0.20 \
    --drop-semantic-ids 4,10 \
    --ply-axis meshlab --debug-every 50

Notes:
- If --semantic-colorize is ON: colors come from semantic palette (best for interpretability).
- If --semantic-colorize is OFF: use photometric RGB where available, fallback color elsewhere.
- Semantic filtering applies ONLY to points with known semantic labels by default (unlabeled points are kept).
"""

import time
import carla
import argparse
import logging
import numpy as np
import os
import json
import math
import pygame
import threading
from collections import Counter, defaultdict

# -----------------------
# Globals for saving
# -----------------------
g_lidar_data_list = []
g_camera_data_list = []
g_pedestrian_detections = []
g_vehicle_detections = []

# -----------------------
# Globals for visualization
# -----------------------
g_vis_camera = None
g_vis_lidar_points = None  # {"points": Nx3 sensor, "tags": N, "obj_ids": N}
g_vis_ped_clusters = []
g_vis_veh_clusters = []
g_last_tag_counts = {}

# -----------------------
# Walker/Vehicle cache for bbox association
# -----------------------
g_walker_cache = []
g_vehicle_cache = []
g_cache_lock = threading.Lock()

# -----------------------
# Sensor refs
# -----------------------
g_camera_sensor = None       # instance segmentation (viz)
g_lidar_sensor = None        # semantic lidar
g_rgb_sensor = None          # rgb for fusion/color
g_semantic_sensor = None     # semantic seg for labels/filtering
g_radar_sensor = None        # radar optional dynamic filtering

# -----------------------
# Fusion buffers
# -----------------------
g_radar_buffer = {}
g_rgb_buffer = {}
g_sem_buffer = {}
g_buffer_lock = threading.Lock()

# -----------------------
# Voxel Map Accumulator (WORLD coords)
# -----------------------
g_voxel_map = {}  # (ix,iy,iz) -> {"n":int, "sum_xyz":(3,), "sum_rgb":(3,), "label_hist": dict}
g_voxel_lock = threading.Lock()

# -----------------------
# Semantic palette
# -----------------------
SEM_COLOR = defaultdict(lambda: (200, 200, 200))
SEM_COLOR.update({
    0: (0, 0, 0),          # unlabeled
    1: (70, 70, 70),       # building
    2: (100, 40, 40),      # fence
    3: (55, 90, 80),       # other
    4: (220, 20, 60),      # pedestrian
    5: (153, 153, 153),    # pole
    6: (157, 234, 50),     # road line
    7: (128, 64, 128),     # road
    8: (244, 35, 232),     # sidewalk
    9: (107, 142, 35),     # vegetation
    10: (0, 0, 142),       # vehicle
    11: (102, 102, 156),   # wall
    12: (220, 220, 0),     # traffic sign
    13: (70, 130, 180),    # sky
    14: (81, 0, 81),       # ground
    15: (150, 100, 100),   # bridge
    16: (230, 150, 140),   # rail track
    17: (180, 165, 180),   # guard rail
    18: (250, 170, 30),    # traffic light
    19: (110, 190, 160),   # static
    20: (170, 120, 50),    # dynamic
    21: (45, 60, 150),     # water
    22: (145, 170, 100),   # terrain
})

# -----------------------
# SemanticLidarDetection compatibility
# -----------------------
def _get_detection_tag(det) -> int:
    for attr in ("object_tag", "semantic_tag", "tag"):
        if hasattr(det, attr):
            try:
                return int(getattr(det, attr))
            except Exception:
                pass
    return 0

def _get_detection_obj_idx(det) -> int:
    for attr in ("object_idx", "obj_idx", "object_id", "id", "idx"):
        if hasattr(det, attr):
            try:
                return int(getattr(det, attr))
            except Exception:
                pass
    return 0

# -----------------------
# Math helpers
# -----------------------
def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def rotation_matrix_from_carla_rotation(rot: carla.Rotation) -> np.ndarray:
    roll = _deg2rad(rot.roll)
    pitch = _deg2rad(rot.pitch)
    yaw = _deg2rad(rot.yaw)

    cr = math.cos(roll);  sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw);   sy = math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)

def inv_rotation_matrix_from_carla_rotation(rot: carla.Rotation) -> np.ndarray:
    return rotation_matrix_from_carla_rotation(rot).T.astype(np.float32)

# -----------------------
# Camera projection helpers
# -----------------------
def get_camera_K(width, height, fov_deg):
    focal = width / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    K = np.identity(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def transform_points(points_xyz, transform: carla.Transform):
    if points_xyz.size == 0:
        return points_xyz
    m = np.array(transform.get_matrix(), dtype=np.float32)  # 4x4
    pts = np.concatenate(
        [points_xyz.astype(np.float32), np.ones((len(points_xyz), 1), dtype=np.float32)],
        axis=1
    )
    pw = (m @ pts.T).T
    return pw[:, :3]

def world_to_camera(points_world, camera_transform: carla.Transform):
    if points_world.size == 0:
        return points_world
    inv = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)  # 4x4
    pts = np.concatenate(
        [points_world.astype(np.float32), np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1
    )
    pc = (inv @ pts.T).T
    return pc[:, :3]

def project_points(points_cam, K, width, height):
    """
    CARLA camera coords from get_inverse_matrix:
      x forward, y right, z up
      u = cx + (y/x)*fx
      v = cy - (z/x)*fy
    Returns:
      ui, vi, idx_valid (indices into original points_cam)
    """
    if points_cam.size == 0:
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    front = x > 0.05
    if not np.any(front):
        return np.zeros((0,), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    idx_front = np.where(front)[0]
    x2 = x[front]; y2 = y[front]; z2 = z[front]

    uu = (K[0, 2] + (y2 / x2) * K[0, 0])
    vv = (K[1, 2] - (z2 / x2) * K[1, 1])

    ui = uu.astype(np.int32)
    vi = vv.astype(np.int32)

    valid = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = ui[valid]; vi = vi[valid]
    idx = idx_front[valid]
    return ui, vi, idx

# -----------------------
# Frame matching
# -----------------------
def get_nearest_packet(buffer: dict, frame: int, max_delta: int = 10):
    if not buffer:
        return None
    if frame in buffer:
        return buffer[frame]
    best_k, best_d = None, 10**9
    for k in buffer.keys():
        d = abs(int(k) - int(frame))
        if d <= max_delta and d < best_d:
            best_k, best_d = k, d
    return buffer.get(best_k, None) if best_k is not None else None

# -----------------------
# Radar helpers (optional)
# -----------------------
def radar_detections_to_points_sensor(radar_meas):
    pts, vel, azs, alts, rngs = [], [], [], [], []
    for d in radar_meas:
        az = float(getattr(d, "azimuth"))
        alt = float(getattr(d, "altitude"))
        r = float(getattr(d, "depth"))
        v = float(getattr(d, "velocity"))
        x = r * math.cos(alt) * math.cos(az)
        y = r * math.cos(alt) * math.sin(az)
        z = r * math.sin(alt)
        pts.append([x, y, z])
        vel.append(v); azs.append(az); alts.append(alt); rngs.append(r)
    if not pts:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32))
    return (np.array(pts, np.float32),
            np.array(vel, np.float32),
            np.array(azs, np.float32),
            np.array(alts, np.float32),
            np.array(rngs, np.float32))

def filter_dynamic_points_with_radar(points_world: np.ndarray,
                                     radar_points_world: np.ndarray,
                                     radar_vel: np.ndarray,
                                     vel_thresh: float = 1.0,
                                     assoc_radius: float = 1.5) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    if radar_points_world is None or radar_points_world.size == 0:
        return np.ones((points_world.shape[0],), dtype=bool)

    movers = np.abs(radar_vel) > vel_thresh
    rp = radar_points_world[movers]
    if rp.size == 0:
        return np.ones((points_world.shape[0],), dtype=bool)

    d2 = ((points_world[:, None, :] - rp[None, :, :]) ** 2).sum(axis=2)
    min_d = np.sqrt(d2.min(axis=1))
    return (min_d > assoc_radius)

# -----------------------
# Option A: FULL-set samplers (do NOT discard points)
# -----------------------
def sample_rgb_full(points_world: np.ndarray,
                    cam_tf: carla.Transform,
                    K: np.ndarray,
                    rgb: np.ndarray,
                    fallback_rgb=(80, 80, 80)):
    """
    Returns:
      rgb_full: Nx3 uint8 (fallback where not projectable)
      rgb_mask: N bool   (True where sampled from RGB image)
    """
    n = points_world.shape[0]
    rgb_full = np.zeros((n, 3), dtype=np.uint8)
    rgb_full[:] = np.array(fallback_rgb, dtype=np.uint8).reshape(1, 3)
    rgb_mask = np.zeros((n,), dtype=bool)

    if n == 0:
        return rgb_full, rgb_mask

    H, W = rgb.shape[0], rgb.shape[1]
    pts_cam = world_to_camera(points_world, cam_tf)
    ui, vi, idx = project_points(pts_cam, K, W, H)
    if idx.size == 0:
        return rgb_full, rgb_mask

    rgb_full[idx] = rgb[vi, ui, :].astype(np.uint8)
    rgb_mask[idx] = True
    return rgb_full, rgb_mask

def sample_semantic_full(points_world: np.ndarray,
                         cam_tf: carla.Transform,
                         K: np.ndarray,
                         sem_ids: np.ndarray,
                         unknown_label=0):
    """
    Returns:
      sem_full: N uint8 (unknown_label where not projectable)
      sem_mask: N bool  (True where semantic was sampled)
    """
    n = points_world.shape[0]
    sem_full = np.full((n,), np.uint8(unknown_label), dtype=np.uint8)
    sem_mask = np.zeros((n,), dtype=bool)

    if n == 0:
        return sem_full, sem_mask

    H, W = sem_ids.shape[0], sem_ids.shape[1]
    pts_cam = world_to_camera(points_world, cam_tf)
    ui, vi, idx = project_points(pts_cam, K, W, H)
    if idx.size == 0:
        return sem_full, sem_mask

    sem_full[idx] = sem_ids[vi, ui].astype(np.uint8)
    sem_mask[idx] = True
    return sem_full, sem_mask

# -----------------------
# Voxel integration
# -----------------------
def voxel_key(pts_world: np.ndarray, voxel_size: float):
    inv = 1.0 / max(1e-6, voxel_size)
    q = np.floor(pts_world * inv).astype(np.int32)
    return q

def integrate_voxels(pts_world: np.ndarray,
                     rgb_cols: np.ndarray,
                     sem_labels: np.ndarray,
                     voxel_size: float):
    if pts_world.size == 0:
        return

    keys = voxel_key(pts_world, voxel_size)

    buckets = defaultdict(list)
    for i, k in enumerate(keys):
        buckets[(int(k[0]), int(k[1]), int(k[2]))].append(i)

    with g_voxel_lock:
        for k, idxs in buckets.items():
            pts_k = pts_world[idxs]
            rgb_k = rgb_cols[idxs] if rgb_cols is not None and rgb_cols.size else None
            sem_k = sem_labels[idxs] if sem_labels is not None and sem_labels.size else None

            n_new = len(idxs)
            sum_xyz_new = pts_k.sum(axis=0).astype(np.float64)

            if k not in g_voxel_map:
                g_voxel_map[k] = {
                    "n": 0,
                    "sum_xyz": np.zeros((3,), dtype=np.float64),
                    "sum_rgb": np.zeros((3,), dtype=np.float64),
                    "label_hist": defaultdict(int)
                }

            v = g_voxel_map[k]
            v["n"] += n_new
            v["sum_xyz"] += sum_xyz_new

            if rgb_k is not None:
                v["sum_rgb"] += rgb_k.astype(np.float64).sum(axis=0)

            if sem_k is not None:
                for s in sem_k:
                    v["label_hist"][int(s)] += 1

def voxel_map_to_arrays(semantic_colorize: bool,
                        store_semantic_label: bool):
    with g_voxel_lock:
        items = list(g_voxel_map.items())

    if not items:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.uint8),
                np.zeros((0,), np.uint8))

    pts = np.zeros((len(items), 3), np.float32)
    cols = np.zeros((len(items), 3), np.uint8)
    labels = np.zeros((len(items),), np.uint8)

    for i, (_k, v) in enumerate(items):
        n = max(1, int(v["n"]))
        pts[i] = (v["sum_xyz"] / n).astype(np.float32)

        if v["label_hist"]:
            lab = max(v["label_hist"].items(), key=lambda kv: kv[1])[0]
        else:
            lab = 0
        labels[i] = np.uint8(lab)

        if semantic_colorize:
            cols[i] = np.array(SEM_COLOR[int(lab)], dtype=np.uint8)
        else:
            rgb = (v["sum_rgb"] / n) if n > 0 else np.zeros((3,), np.float64)
            cols[i] = np.clip(rgb, 0, 255).astype(np.uint8)

    if not store_semantic_label:
        labels = np.zeros((0,), np.uint8)

    return pts, cols, labels

# -----------------------
# PLY export
# -----------------------
def apply_ply_axis_transform(pts_world: np.ndarray, axis_mode: str):
    if pts_world.size == 0:
        return pts_world
    if axis_mode == "carla":
        return pts_world
    if axis_mode == "meshlab":
        out = pts_world.copy()
        out[:, 1] *= -1.0
        return out
    return pts_world

def write_ply_xyzrgb(path: str, pts: np.ndarray, cols: np.ndarray, labels: np.ndarray = None):
    n = pts.shape[0]
    has_label = (labels is not None and labels.size == n)

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if has_label:
        header_lines.append("property uchar label")
    header_lines.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        if has_label:
            for i in range(n):
                x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                r, g, b = int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2])
                lab = int(labels[i])
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b} {lab}\n")
        else:
            for i in range(n):
                x, y, z = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
                r, g, b = int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2])
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

def export_voxel_map_ply(out_dir: str,
                         frame: int,
                         axis_mode: str,
                         semantic_colorize: bool,
                         store_semantic_label: bool,
                         max_points: int):
    os.makedirs(out_dir, exist_ok=True)

    pts, cols, labels = voxel_map_to_arrays(
        semantic_colorize=semantic_colorize,
        store_semantic_label=store_semantic_label
    )
    if pts.shape[0] == 0:
        logging.info(f"[MAP] export skipped (empty voxel map) out_dir={out_dir} frame={frame}")
        return

    if max_points > 0 and pts.shape[0] > max_points:
        pts = pts[-max_points:]
        cols = cols[-max_points:]
        if labels.size:
            labels = labels[-max_points:]

    pts2 = apply_ply_axis_transform(pts, axis_mode)

    ply_path = os.path.join(out_dir, f"map_{frame:06d}.ply")
    write_ply_xyzrgb(ply_path, pts2, cols, labels if labels.size else None)
    logging.info(f"[MAP] exported {ply_path} (voxels={pts.shape[0]})")

# -----------------------
# Camera callbacks
# -----------------------
def camera_callback(image, run_dir, W, H):
    global g_vis_camera

    img_dir = os.path.join(run_dir, "output_instance_segmentation")
    raw_dir = os.path.join(run_dir, "output_camera_raw")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    png_path = os.path.join(img_dir, f"frame_{image.frame:06d}.png")
    image.save_to_disk(png_path)

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))  # BGRA

    npy_path = os.path.join(raw_dir, f"frame_{image.frame:06d}.npy")
    np.save(npy_path, arr)

    rgb = arr[:, :, :3][:, :, ::-1]
    vis = np.clip(rgb.astype(np.float32) * 10.0, 0, 255).astype(np.uint8)
    g_vis_camera = np.transpose(vis, (1, 0, 2))

    g_camera_data_list.append({
        "timestamp": int(image.frame),
        "image_file": os.path.join("output_instance_segmentation", f"frame_{image.frame:06d}.png"),
        "raw_file": os.path.join("output_camera_raw", f"frame_{image.frame:06d}.npy"),
        "width": int(image.width),
        "height": int(image.height),
        "fov": float(image.fov),
        "type": "instance_segmentation"
    })

def rgb_camera_callback(image, run_dir):
    frame = int(image.frame)

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))  # BGRA
    rgb = arr[:, :, :3][:, :, ::-1].copy()

    rgb_dir = os.path.join(run_dir, "output_rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    image.save_to_disk(os.path.join(rgb_dir, f"frame_{frame:06d}.png"))

    with g_buffer_lock:
        g_rgb_buffer[frame] = {"ts": frame, "rgb": rgb}
        if len(g_rgb_buffer) > 500:
            for k in sorted(g_rgb_buffer.keys())[:-250]:
                g_rgb_buffer.pop(k, None)

def semantic_camera_callback(image, run_dir):
    frame = int(image.frame)
    try:
        image.convert(carla.ColorConverter.Raw)
    except Exception:
        pass

    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))  # BGRA
    sem_id = arr[:, :, 2].copy()  # R channel holds class id in RAW

    sem_dir = os.path.join(run_dir, "output_semantic_raw")
    os.makedirs(sem_dir, exist_ok=True)
    np.save(os.path.join(sem_dir, f"frame_{frame:06d}.npy"), sem_id)

    with g_buffer_lock:
        g_sem_buffer[frame] = {"ts": frame, "sem": sem_id}
        if len(g_sem_buffer) > 500:
            for k in sorted(g_sem_buffer.keys())[:-250]:
                g_sem_buffer.pop(k, None)

# -----------------------
# Radar callback
# -----------------------
def radar_callback(radar_meas):
    frame = int(radar_meas.frame)
    if g_radar_sensor is None:
        return

    pts_sensor, vel, az, alt, rng = radar_detections_to_points_sensor(radar_meas)
    radar_tf = g_radar_sensor.get_transform()
    pts_world = transform_points(pts_sensor, radar_tf)

    with g_buffer_lock:
        g_radar_buffer[frame] = {
            "ts": frame,
            "points_world": pts_world,
            "vel": vel,
        }
        if len(g_radar_buffer) > 500:
            for k in sorted(g_radar_buffer.keys())[:-250]:
                g_radar_buffer.pop(k, None)

# -----------------------
# Cache refresh (detections overlay)
# -----------------------
def refresh_actor_cache(world, pattern: str):
    actors = world.get_actors().filter(pattern)
    out = []
    for a in actors:
        try:
            tf = a.get_transform()
            inv_actor = np.array(tf.get_inverse_matrix(), dtype=np.float32)
            bb = a.bounding_box
            bb_loc = np.array([bb.location.x, bb.location.y, bb.location.z], dtype=np.float32)
            bb_ext = np.array([bb.extent.x, bb.extent.y, bb.extent.z], dtype=np.float32)
            inv_bb_rot = inv_rotation_matrix_from_carla_rotation(bb.rotation)
            loc = tf.location
            a_loc = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
            out.append((int(a.id), inv_actor, bb_loc, bb_ext, inv_bb_rot, a_loc))
        except Exception:
            continue
    return out

def refresh_caches(world):
    global g_walker_cache, g_vehicle_cache
    walkers = refresh_actor_cache(world, "walker.pedestrian.*")
    vehicles = refresh_actor_cache(world, "vehicle.*")
    with g_cache_lock:
        g_walker_cache = walkers
        g_vehicle_cache = vehicles
    return len(walkers), len(vehicles)

def assign_points_to_cache_OBB(points_world: np.ndarray,
                               points_sensor: np.ndarray,
                               cache_list,
                               margin_xy: float,
                               margin_z_up: float,
                               margin_z_down: float,
                               max_actors_to_test: int = 5000):
    if points_world.size == 0 or not cache_list:
        return {}, {}

    if len(cache_list) > max_actors_to_test:
        cache_list = cache_list[:max_actors_to_test]

    ones = np.ones((points_world.shape[0], 1), dtype=np.float32)
    pw_h = np.concatenate([points_world.astype(np.float32), ones], axis=1)

    assigned = {}
    hit_counts = {}

    for (aid, inv_actor, bb_loc, bb_ext, inv_bb_rot, _actor_loc) in cache_list:
        pa = (inv_actor @ pw_h.T).T[:, :3]
        d = pa - bb_loc.reshape(1, 3)
        d2 = (inv_bb_rot @ d.T).T

        inside_xy = (np.abs(d2[:, 0]) <= (bb_ext[0] + margin_xy)) & (np.abs(d2[:, 1]) <= (bb_ext[1] + margin_xy))
        inside_z = (d2[:, 2] <= (bb_ext[2] + margin_z_up)) & (d2[:, 2] >= (-bb_ext[2] - margin_z_down))
        inside = inside_xy & inside_z

        c = int(np.count_nonzero(inside))
        if c > 0:
            assigned.setdefault(aid, []).append(points_sensor[inside])
            hit_counts[aid] = hit_counts.get(aid, 0) + c

    return assigned, hit_counts

# -----------------------
# LiDAR callback (includes Option A fusion)
# -----------------------
def lidar_callback(semantic_lidar,
                   ped_candidate_tags_set,
                   veh_candidate_tags_set,
                   min_ped_points,
                   min_veh_points,
                   sample_save_points,
                   bbox_margin_xy,
                   bbox_margin_z_up,
                   bbox_margin_z_down,
                   max_candidates_for_bbox,
                   debug_every,
                   debug_nearest_walker,
                   debug_walker_tag_hist,
                   # fusion args
                   do_fusion_map: bool,
                   map_downsample_stride: int,
                   map_export_every: int,
                   map_export_dir: str,
                   fusion_match_max_delta: int,
                   dynamic_filter_with_radar: bool,
                   radar_vel_thresh: float,
                   radar_assoc_radius: float,
                   # semantic fusion args
                   use_semantic: bool,
                   drop_semantic_ids: set,
                   keep_semantic_ids: set,
                   semantic_colorize: bool,
                   store_semantic_label: bool,
                   semantic_filter_mode: str,   # "unlabeled_keep" | "require_semantic"
                   voxel_size: float,
                   ply_axis: str,
                   max_map_points: int,
                   fallback_rgb: tuple):
    global g_vis_lidar_points, g_vis_ped_clusters, g_vis_veh_clusters, g_last_tag_counts
    global g_pedestrian_detections, g_vehicle_detections

    frame = int(semantic_lidar.frame)

    pts = []
    tags = []
    obj_ids = []
    tag_counts = {}

    for det in semantic_lidar:
        p = det.point
        pts.append([float(p.x), float(p.y), float(p.z)])
        t = _get_detection_tag(det)
        oid = _get_detection_obj_idx(det)
        tags.append(t)
        obj_ids.append(oid)
        tag_counts[t] = tag_counts.get(t, 0) + 1

    g_last_tag_counts = tag_counts

    if not pts:
        g_vis_lidar_points = None
        g_vis_ped_clusters = []
        g_vis_veh_clusters = []
        g_pedestrian_detections.append({"timestamp": frame, "pedestrians": []})
        g_vehicle_detections.append({"timestamp": frame, "vehicles": []})
        return

    pts = np.array(pts, dtype=np.float32)
    tags = np.array(tags, dtype=np.int32)
    obj_ids = np.array(obj_ids, dtype=np.int32)

    g_vis_lidar_points = {"points": pts, "tags": tags, "obj_ids": obj_ids}

    do_log = (debug_every > 0 and (frame % debug_every == 0))
    if do_log:
        top = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        logging.info(f"[STEP0 ingest] frame={frame} total_pts={len(pts)} top_tags={top}")

    # STEP1: candidates for overlay/detections
    ped_cand_mask = np.isin(tags, list(ped_candidate_tags_set))
    veh_cand_mask = np.isin(tags, list(veh_candidate_tags_set))

    ped_cand_pts_sensor = pts[ped_cand_mask]
    veh_cand_pts_sensor = pts[veh_cand_mask]

    if ped_cand_pts_sensor.shape[0] > max_candidates_for_bbox:
        idx = np.random.choice(ped_cand_pts_sensor.shape[0], size=max_candidates_for_bbox, replace=False)
        ped_cand_pts_sensor = ped_cand_pts_sensor[idx]
    if veh_cand_pts_sensor.shape[0] > max_candidates_for_bbox:
        idx = np.random.choice(veh_cand_pts_sensor.shape[0], size=max_candidates_for_bbox, replace=False)
        veh_cand_pts_sensor = veh_cand_pts_sensor[idx]

    # STEP2: caches
    with g_cache_lock:
        walker_cache = list(g_walker_cache)
        vehicle_cache = list(g_vehicle_cache)

    if g_lidar_sensor is None:
        g_vis_ped_clusters = []
        g_vis_veh_clusters = []
        g_pedestrian_detections.append({"timestamp": frame, "pedestrians": []})
        g_vehicle_detections.append({"timestamp": frame, "vehicles": []})
        return

    # STEP3: transform candidates to world
    lidar_tf = g_lidar_sensor.get_transform()
    ped_cand_pts_world = transform_points(ped_cand_pts_sensor, lidar_tf)
    veh_cand_pts_world = transform_points(veh_cand_pts_sensor, lidar_tf)

    # Optional debug stats
    if do_log and debug_nearest_walker and len(walker_cache) > 0 and ped_cand_pts_world.shape[0] > 0:
        walker_locs = np.stack([c[5] for c in walker_cache], axis=0)
        sample_n = min(300, ped_cand_pts_world.shape[0])
        idx = np.random.choice(ped_cand_pts_world.shape[0], size=sample_n, replace=False) if ped_cand_pts_world.shape[0] > sample_n else np.arange(sample_n)
        pw = ped_cand_pts_world[idx]
        dists = np.sqrt(((pw[:, None, :] - walker_locs[None, :, :]) ** 2).sum(axis=2))
        nearest = dists.min(axis=1)
        logging.info(f"[STEP3 nearest-walker] frame={frame} min={nearest.min():.2f} med={np.median(nearest):.2f} max={nearest.max():.2f}")

    if do_log and debug_walker_tag_hist and len(walker_cache) > 0:
        test_walkers = walker_cache[:min(20, len(walker_cache))]
        pts_world_all = transform_points(pts, lidar_tf)
        ones = np.ones((pts_world_all.shape[0], 1), dtype=np.float32)
        pw_h_all = np.concatenate([pts_world_all.astype(np.float32), ones], axis=1)
        hist = Counter()
        for (wid, inv_actor, bb_loc, bb_ext, _inv_bb_rot, _loc) in test_walkers:
            pa = (inv_actor @ pw_h_all.T).T[:, :3]
            d = pa - bb_loc.reshape(1, 3)
            inside = (np.abs(d[:, 0]) <= (bb_ext[0] + 1.0)) & (np.abs(d[:, 1]) <= (bb_ext[1] + 1.0)) & (np.abs(d[:, 2]) <= (bb_ext[2] + 1.0))
            if np.any(inside):
                for t in tags[inside]:
                    hist[int(t)] += 1
        logging.info(f"[STEP3b walker-tag-hist] frame={frame} {hist.most_common(10)}")

    # STEP4: bbox assign (OBB)
    ped_assigned, ped_hit_counts = assign_points_to_cache_OBB(
        points_world=ped_cand_pts_world,
        points_sensor=ped_cand_pts_sensor,
        cache_list=walker_cache,
        margin_xy=bbox_margin_xy,
        margin_z_up=bbox_margin_z_up,
        margin_z_down=bbox_margin_z_down
    )
    veh_assigned, veh_hit_counts = assign_points_to_cache_OBB(
        points_world=veh_cand_pts_world,
        points_sensor=veh_cand_pts_sensor,
        cache_list=vehicle_cache,
        margin_xy=bbox_margin_xy,
        margin_z_up=bbox_margin_z_up,
        margin_z_down=bbox_margin_z_down
    )

    # STEP5: cluster
    ped_clusters = []
    veh_clusters = []

    for wid, chunks in ped_assigned.items():
        arr = np.concatenate(chunks, axis=0)
        if arr.shape[0] < min_ped_points:
            continue
        ped_clusters.append({"actor_id": int(wid), "num_points": int(arr.shape[0]), "centroid_sensor": arr.mean(axis=0)})

    for vid, chunks in veh_assigned.items():
        arr = np.concatenate(chunks, axis=0)
        if arr.shape[0] < min_veh_points:
            continue
        veh_clusters.append({"actor_id": int(vid), "num_points": int(arr.shape[0]), "centroid_sensor": arr.mean(axis=0)})

    g_vis_ped_clusters = ped_clusters
    g_vis_veh_clusters = veh_clusters

    # Save detections JSON
    g_pedestrian_detections.append({"timestamp": frame, "pedestrians": [
        {"actor_id": int(c["actor_id"]), "num_points": int(c["num_points"]),
         "centroid_sensor": {"x": float(c["centroid_sensor"][0]), "y": float(c["centroid_sensor"][1]), "z": float(c["centroid_sensor"][2])},
         "method": "bbox_obb"} for c in ped_clusters
    ]})
    g_vehicle_detections.append({"timestamp": frame, "vehicles": [
        {"actor_id": int(c["actor_id"]), "num_points": int(c["num_points"]),
         "centroid_sensor": {"x": float(c["centroid_sensor"][0]), "y": float(c["centroid_sensor"][1]), "z": float(c["centroid_sensor"][2])},
         "method": "bbox_obb"} for c in veh_clusters
    ]})

    # Save sampled lidar points (debug)
    keep = min(len(pts), sample_save_points)
    idx = np.random.choice(len(pts), size=keep, replace=False) if len(pts) > keep else np.arange(len(pts))
    g_lidar_data_list.append({"timestamp": frame, "points": [
        {"x": float(pts[i, 0]), "y": float(pts[i, 1]), "z": float(pts[i, 2]), "tag": int(tags[i]), "id": int(obj_ids[i])}
        for i in idx
    ]})

    # =======================
    # FUSION / MAPPING (Option A)
    # =======================
    if do_fusion_map and g_rgb_sensor is not None and g_lidar_sensor is not None:
        pts_map_sensor = pts[::max(1, map_downsample_stride)]
        pts_world_map = transform_points(pts_map_sensor, lidar_tf)

        with g_buffer_lock:
            rgb_pkt = get_nearest_packet(g_rgb_buffer, frame, max_delta=fusion_match_max_delta)
            sem_pkt = get_nearest_packet(g_sem_buffer, frame, max_delta=fusion_match_max_delta) if use_semantic else None
            radar_pkt = get_nearest_packet(g_radar_buffer, frame, max_delta=fusion_match_max_delta) if dynamic_filter_with_radar else None

        if do_log:
            logging.info(
                f"[FUSION] frame={frame} rgb={'YES' if rgb_pkt else 'NO'} sem={'YES' if sem_pkt else 'NO'} "
                f"radar={'YES' if radar_pkt else 'NO'} in={pts_world_map.shape[0]}"
            )

        if rgb_pkt is None:
            if do_log:
                logging.info(f"[FUSION] frame={frame} skip (no RGB packet matched)")
        else:
            # 1) optional radar dynamic filtering
            pts_world_map2 = pts_world_map
            if dynamic_filter_with_radar and radar_pkt is not None:
                keep_static = filter_dynamic_points_with_radar(
                    pts_world_map2,
                    radar_pkt["points_world"],
                    radar_pkt["vel"],
                    vel_thresh=radar_vel_thresh,
                    assoc_radius=radar_assoc_radius
                )
                pts_world_map2 = pts_world_map2[keep_static]

            # 2) semantic sampling (FULL) and semantic filtering (does NOT depend on RGB!)
            sem_full = None
            sem_mask = None
            if use_semantic and sem_pkt is not None and g_semantic_sensor is not None:
                sem_img = sem_pkt["sem"]
                cam_tf_sem = g_semantic_sensor.get_transform()
                K_sem = get_camera_K(sem_img.shape[1], sem_img.shape[0], float(g_semantic_sensor.attributes.get("fov", "90")))

                sem_full, sem_mask = sample_semantic_full(
                    pts_world_map2, cam_tf_sem, K_sem, sem_img, unknown_label=0
                )

                if semantic_filter_mode == "require_semantic":
                    pts_world_map2 = pts_world_map2[sem_mask]
                    sem_full = sem_full[sem_mask]
                    sem_mask = np.ones((pts_world_map2.shape[0],), dtype=bool)

                # apply keep/drop only to points that have known semantic labels
                if sem_full is not None and sem_full.size == pts_world_map2.shape[0]:
                    m = np.ones((pts_world_map2.shape[0],), dtype=bool)
                    known = sem_mask if sem_mask is not None else np.ones_like(m)

                    if keep_semantic_ids:
                        m_known = np.isin(sem_full, list(keep_semantic_ids))
                        m = (~known) | m_known  # keep unlabeled, filter labeled
                    if drop_semantic_ids:
                        m_drop = np.isin(sem_full, list(drop_semantic_ids))
                        m = m & ((~known) | (~m_drop))  # keep unlabeled, drop labeled that match

                    pts_world_map2 = pts_world_map2[m]
                    sem_full = sem_full[m]
                    if sem_mask is not None:
                        sem_mask = sem_mask[m]

            # 3) RGB sampling (FULL) — THIS IS THE KEY OPTION A CHANGE
            rgb = rgb_pkt["rgb"]
            cam_tf_rgb = g_rgb_sensor.get_transform()
            K_rgb = get_camera_K(rgb.shape[1], rgb.shape[0], float(g_rgb_sensor.attributes.get("fov", "90")))

            rgb_full, rgb_mask = sample_rgb_full(
                pts_world_map2, cam_tf_rgb, K_rgb, rgb, fallback_rgb=fallback_rgb
            )

            # 4) choose colors: semantic_colorize overrides RGB where label is known
            if semantic_colorize and use_semantic and sem_full is not None and sem_full.size == pts_world_map2.shape[0]:
                cols = np.zeros_like(rgb_full)
                for i, lab in enumerate(sem_full):
                    cols[i] = np.array(SEM_COLOR[int(lab)], dtype=np.uint8)
            else:
                cols = rgb_full

            if do_log:
                rgb_cov = float(np.mean(rgb_mask)) if rgb_mask is not None and rgb_mask.size else 0.0
                logging.info(f"[FUSION] frame={frame} after_filters={pts_world_map2.shape[0]} rgb_coverage={rgb_cov*100:.1f}% voxel={voxel_size}")

            # 5) integrate voxels with FULL geometry
            if pts_world_map2.shape[0] > 0:
                integrate_voxels(
                    pts_world=pts_world_map2,
                    rgb_cols=cols,
                    sem_labels=sem_full if (use_semantic and sem_full is not None) else None,
                    voxel_size=voxel_size
                )

        # Export periodically
        if map_export_every > 0 and (frame % map_export_every == 0):
            export_voxel_map_ply(
                out_dir=map_export_dir,
                frame=frame,
                axis_mode=ply_axis,
                semantic_colorize=semantic_colorize,
                store_semantic_label=store_semantic_label,
                max_points=max_map_points
            )

# -----------------------
# Draw overlay (minimal)
# -----------------------
def draw_overlay(display, W, H, cam_fov,
                 ped_candidate_tags_set, veh_candidate_tags_set,
                 draw_stride, point_radius,
                 draw_points, draw_centroids,
                 ped_color=(255, 0, 0),
                 veh_color=(0, 0, 255)):
    if g_vis_lidar_points is None or g_camera_sensor is None or g_lidar_sensor is None:
        return

    K = get_camera_K(W, H, cam_fov)

    pts = g_vis_lidar_points["points"]
    tags = g_vis_lidar_points["tags"]

    lidar_tf = g_lidar_sensor.get_transform()
    cam_tf = g_camera_sensor.get_transform()

    pts_world = transform_points(pts, lidar_tf)
    pts_cam = world_to_camera(pts_world, cam_tf)

    ped_mask = np.isin(tags, list(ped_candidate_tags_set))
    veh_mask = np.isin(tags, list(veh_candidate_tags_set))

    if draw_points:
        ped_idx = np.where(ped_mask)[0][::max(1, draw_stride)]
        veh_idx = np.where(veh_mask)[0][::max(1, draw_stride)]

        ui, vi, _ = project_points(pts_cam[ped_idx], K, W, H)
        for u, v in zip(ui, vi):
            if point_radius <= 1:
                display.set_at((int(u), int(v)), ped_color)
            else:
                pygame.draw.circle(display, ped_color, (int(u), int(v)), int(point_radius), 0)

        ui2, vi2, _ = project_points(pts_cam[veh_idx], K, W, H)
        for u, v in zip(ui2, vi2):
            if point_radius <= 1:
                display.set_at((int(u), int(v)), veh_color)
            else:
                pygame.draw.circle(display, veh_color, (int(u), int(v)), int(point_radius), 0)

    if draw_centroids:
        for c in g_vis_ped_clusters[:4]:
            centroid_sensor = c["centroid_sensor"].reshape(1, 3)
            cw = transform_points(centroid_sensor, lidar_tf)
            cc = world_to_camera(cw, cam_tf)
            uu, vv, idx = project_points(cc, K, W, H)
            if idx.size == 1:
                pygame.draw.circle(display, (0, 255, 0), (int(uu[0]), int(vv[0])), 8, 2)

# -----------------------
# Sensor transform helper
# -----------------------
def _offset_spawn_transform(transform: carla.Transform,
                            forward_m: float,
                            right_m: float,
                            z_offset_m: float,
                            yaw_offset_deg: float) -> carla.Transform:
    yaw_rad = math.radians(float(transform.rotation.yaw))
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x) + forward_x * float(forward_m) + right_x * float(right_m),
            y=float(transform.location.y) + forward_y * float(forward_m) + right_y * float(right_m),
            z=float(transform.location.z) + float(z_offset_m),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw) + float(yaw_offset_deg),
            roll=float(transform.rotation.roll),
        ),
    )

def _sensor_transform_from_ego(ego_tf: carla.Transform,
                               sensor_x: float,
                               sensor_y: float,
                               sensor_z: float,
                               sensor_pitch: float,
                               sensor_yaw: float,
                               sensor_roll: float) -> carla.Transform:
    # The parked-ego fusion scripts attach the camera/radar in ego-local
    # coordinates: x forward, y right, z up. This diagnostic does not need to
    # spawn the ego vehicle, but it uses the same local pose convention.
    yaw_rad = math.radians(float(ego_tf.rotation.yaw))
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)
    return carla.Transform(
        carla.Location(
            x=float(ego_tf.location.x) + forward_x * float(sensor_x) + right_x * float(sensor_y),
            y=float(ego_tf.location.y) + forward_y * float(sensor_x) + right_y * float(sensor_y),
            z=float(ego_tf.location.z) + float(sensor_z),
        ),
        carla.Rotation(
            pitch=float(ego_tf.rotation.pitch) + float(sensor_pitch),
            yaw=float(ego_tf.rotation.yaw) + float(sensor_yaw),
            roll=float(ego_tf.rotation.roll) + float(sensor_roll),
        ),
    )

def choose_parked_ego_camera_transform(world,
                                       spawn_index: int,
                                       forward_offset_m: float,
                                       right_offset_m: float,
                                       z_offset_m: float,
                                       yaw_offset_deg: float,
                                       camera_x: float,
                                       camera_y: float,
                                       camera_z: float,
                                       camera_pitch: float,
                                       camera_yaw: float,
                                       camera_roll: float) -> carla.Transform:
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points are available for parked-ego camera placement.")
    index = int(spawn_index) % len(spawn_points)
    ego_tf = _offset_spawn_transform(
        spawn_points[index],
        forward_m=float(forward_offset_m),
        right_m=float(right_offset_m),
        z_offset_m=float(z_offset_m),
        yaw_offset_deg=float(yaw_offset_deg),
    )
    return _sensor_transform_from_ego(
        ego_tf,
        sensor_x=float(camera_x),
        sensor_y=float(camera_y),
        sensor_z=float(camera_z),
        sensor_pitch=float(camera_pitch),
        sensor_yaw=float(camera_yaw),
        sensor_roll=float(camera_roll),
    )

def choose_sensor_transform(world, sensor_z, pitch, yaw_add, fallback_x, fallback_y, fallback_z):
    traffic_light_actor = None
    for a in world.get_actors().filter("traffic.traffic_light"):
        traffic_light_actor = a
        break

    if traffic_light_actor:
        tl = traffic_light_actor.get_transform()
        sensor_loc = tl.location + carla.Location(z=sensor_z)
        sensor_rot = tl.rotation
        sensor_rot.pitch = pitch
        sensor_rot.yaw += yaw_add
        return carla.Transform(sensor_loc, sensor_rot)

    return carla.Transform(
        carla.Location(x=fallback_x, y=fallback_y, z=fallback_z),
        carla.Rotation(pitch=pitch, yaw=yaw_add)
    )

# -----------------------
# Main
# -----------------------
def _parse_int_set(s: str):
    s = s.strip()
    if not s:
        return set()
    return set(int(x.strip()) for x in s.split(",") if x.strip())

def _parse_rgb_triplet(s: str, default=(80, 80, 80)):
    s = s.strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return default
    try:
        r = int(parts[0]); g = int(parts[1]); b = int(parts[2])
        r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
        return (r, g, b)
    except Exception:
        return default

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)

    parser.add_argument("--asynch", action="store_true", help="Do NOT force sync mode; use if another script ticks.")
    parser.add_argument("--no-rendering", action="store_true")

    # placement
    parser.add_argument(
        "--placement-mode",
        default="traffic_light",
        choices=["traffic_light", "manual", "parked_ego_camera"],
        help=(
            "traffic_light: first traffic-light pole; manual: fallback x/y/z + pitch/yaw; "
            "parked_ego_camera: same spawn/offset/camera convention as the fusion parked-ego client."
        ),
    )
    parser.add_argument("--sensor-z", type=float, default=5.0)
    parser.add_argument("--sensor-pitch", type=float, default=-20.0)
    parser.add_argument("--sensor-yaw-add", type=float, default=90.0)
    parser.add_argument("--fallback-x", type=float, default=100.0)
    parser.add_argument("--fallback-y", type=float, default=0.0)
    parser.add_argument("--fallback-z", type=float, default=10.0)
    parser.add_argument("--ego-spawn-index", type=int, default=80)
    parser.add_argument("--ego-spawn-forward-offset-m", type=float, default=4.0)
    parser.add_argument("--ego-spawn-right-offset-m", type=float, default=7.0)
    parser.add_argument("--ego-spawn-z-offset-m", type=float, default=0.15)
    parser.add_argument("--ego-spawn-yaw-offset-deg", type=float, default=-28.414)
    parser.add_argument("--ego-camera-x", type=float, default=1.8)
    parser.add_argument("--ego-camera-y", type=float, default=0.0)
    parser.add_argument("--ego-camera-z", type=float, default=1.55)
    parser.add_argument("--ego-camera-pitch", type=float, default=-4.0)
    parser.add_argument("--ego-camera-yaw", type=float, default=0.0)
    parser.add_argument("--ego-camera-roll", type=float, default=0.0)

    # camera (instance seg - viz)
    parser.add_argument("--camera-w", type=int, default=800)
    parser.add_argument("--camera-h", type=int, default=600)
    parser.add_argument("--camera-fov", type=float, default=90.0)

    # RGB camera
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--rgb-w", type=int, default=800)
    parser.add_argument("--rgb-h", type=int, default=600)
    parser.add_argument("--rgb-fov", type=float, default=90.0)

    # Semantic camera
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--semantic-w", type=int, default=800)
    parser.add_argument("--semantic-h", type=int, default=600)
    parser.add_argument("--semantic-fov", type=float, default=90.0)

    # lidar
    parser.add_argument("--lidar-range", type=float, default=120.0)
    parser.add_argument("--lidar-upper-fov", type=float, default=15.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-15.0)
    parser.add_argument("--lidar-channels", type=int, default=64)
    parser.add_argument("--lidar-rotation-frequency", type=float, default=20.0)
    parser.add_argument("--lidar-pps", type=int, default=600000)
    parser.add_argument("--lidar-sensor-tick", type=float, default=0.05)

    # radar
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--radar-range", type=float, default=80.0)
    parser.add_argument("--radar-hfov", type=float, default=30.0)
    parser.add_argument("--radar-vfov", type=float, default=10.0)

    # tags for overlay/detections (semantic LiDAR tags)
    parser.add_argument("--ped-candidate-tags", default="12,24,25,4")
    parser.add_argument("--veh-candidate-tags", default="14,15,16")

    parser.add_argument("--min-ped-points", type=int, default=10)
    parser.add_argument("--min-veh-points", type=int, default=20)

    parser.add_argument("--bbox-margin-xy", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-up", type=float, default=0.35)
    parser.add_argument("--bbox-margin-z-down", type=float, default=0.70)
    parser.add_argument("--max-candidates-for-bbox", type=int, default=30000)

    parser.add_argument("--refresh-actors-every", type=float, default=0.5)

    # draw
    parser.add_argument("--draw-stride", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--no-draw-points", action="store_true")
    parser.add_argument("--no-draw-centroids", action="store_true")

    # logging
    parser.add_argument("--debug-every", type=int, default=50)
    parser.add_argument("--debug-nearest-walker", action="store_true")
    parser.add_argument("--debug-walker-tag-hist", action="store_true")

    # toggles
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")

    # fusion / mapping
    parser.add_argument("--fusion-map", action="store_true")
    parser.add_argument("--map-downsample-stride", type=int, default=2)
    parser.add_argument("--map-export-every", type=int, default=50)
    parser.add_argument("--map-export-dir", default="output_map_ply")
    parser.add_argument("--fusion-match-max-delta", type=int, default=10)

    # semantic filtering + output
    parser.add_argument("--drop-semantic-ids", default="4,10")
    parser.add_argument("--keep-semantic-ids", default="")
    parser.add_argument("--semantic-colorize", action="store_true")
    parser.add_argument("--store-semantic-label", action="store_true")
    parser.add_argument("--semantic-filter-mode", default="unlabeled_keep", choices=["unlabeled_keep", "require_semantic"],
                        help="How to treat points without semantic label (outside semantic camera frustum).")

    # voxel mapping
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--max-map-points", type=int, default=600000)

    # PLY axis mode
    parser.add_argument("--ply-axis", default="meshlab", choices=["carla", "meshlab"])

    # radar dynamic filtering
    parser.add_argument("--dynamic-filter-with-radar", action="store_true")
    parser.add_argument("--radar-vel-thresh", type=float, default=1.0)
    parser.add_argument("--radar-assoc-radius", type=float, default=1.5)

    # Option A fallback color
    parser.add_argument("--fallback-rgb", default="80,80,80",
                        help="Fallback RGB for points not visible in RGB camera (Option A). Example: 80,80,80")

    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    ped_candidate_tags_set = set(int(x.strip()) for x in args.ped_candidate_tags.split(",") if x.strip())
    veh_candidate_tags_set = set(int(x.strip()) for x in args.veh_candidate_tags.split(",") if x.strip())

    drop_semantic_ids = _parse_int_set(args.drop_semantic_ids)
    keep_semantic_ids = _parse_int_set(args.keep_semantic_ids)
    fallback_rgb = _parse_rgb_triplet(args.fallback_rgb, default=(80, 80, 80))

    run_dir = f"sensor_log_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)
    print("RUN_DIR:", os.path.abspath(run_dir))

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    sensor_actors = []
    synchronous_master = False

    pygame.init()
    W, H = args.camera_w, args.camera_h
    display = pygame.display.set_mode((W, H), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Fusion Option A: Full LiDAR Geometry + RGB/Semantic Coloring (Voxel PLY)")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    global g_camera_sensor, g_lidar_sensor, g_rgb_sensor, g_semantic_sensor, g_radar_sensor

    last_refresh = 0.0
    walkers_cached = 0
    vehicles_cached = 0

    try:
        world = client.get_world()

        settings = world.get_settings()
        if not args.asynch:
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = float(args.lidar_sensor_tick)
        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        if args.placement_mode == "parked_ego_camera":
            sensor_transform = choose_parked_ego_camera_transform(
                world,
                spawn_index=int(args.ego_spawn_index),
                forward_offset_m=float(args.ego_spawn_forward_offset_m),
                right_offset_m=float(args.ego_spawn_right_offset_m),
                z_offset_m=float(args.ego_spawn_z_offset_m),
                yaw_offset_deg=float(args.ego_spawn_yaw_offset_deg),
                camera_x=float(args.ego_camera_x),
                camera_y=float(args.ego_camera_y),
                camera_z=float(args.ego_camera_z),
                camera_pitch=float(args.ego_camera_pitch),
                camera_yaw=float(args.ego_camera_yaw),
                camera_roll=float(args.ego_camera_roll),
            )
        elif args.placement_mode == "manual":
            sensor_transform = carla.Transform(
                carla.Location(
                    x=float(args.fallback_x),
                    y=float(args.fallback_y),
                    z=float(args.fallback_z),
                ),
                carla.Rotation(
                    pitch=float(args.sensor_pitch),
                    yaw=float(args.sensor_yaw_add),
                    roll=0.0,
                ),
            )
        else:
            sensor_transform = choose_sensor_transform(
                world,
                sensor_z=args.sensor_z,
                pitch=args.sensor_pitch,
                yaw_add=args.sensor_yaw_add,
                fallback_x=args.fallback_x,
                fallback_y=args.fallback_y,
                fallback_z=args.fallback_z
            )

        print(
            "Sensor placement:",
            args.placement_mode,
            f"loc=({sensor_transform.location.x:.2f}, {sensor_transform.location.y:.2f}, {sensor_transform.location.z:.2f})",
            f"rot=(pitch={sensor_transform.rotation.pitch:.2f}, yaw={sensor_transform.rotation.yaw:.2f}, roll={sensor_transform.rotation.roll:.2f})",
        )

        walkers_cached, vehicles_cached = refresh_caches(world)
        print(f"Initial walkers in world: {walkers_cached} | vehicles: {vehicles_cached}")

        # Instance Segmentation Camera (viz)
        if not args.no_camera:
            cam_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
            cam_bp.set_attribute("image_size_x", str(W))
            cam_bp.set_attribute("image_size_y", str(H))
            cam_bp.set_attribute("fov", str(args.camera_fov))
            cam_bp.set_attribute("sensor_tick", str(args.lidar_sensor_tick if not args.asynch else 0.0))
            g_camera_sensor = world.spawn_actor(cam_bp, sensor_transform)
            g_camera_sensor.listen(lambda img: camera_callback(img, run_dir, W, H))
            sensor_actors.append(g_camera_sensor)
            print("Spawned InstanceSeg Camera:", g_camera_sensor.id)

        # RGB Camera
        if not args.no_rgb:
            rgb_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            rgb_bp.set_attribute("image_size_x", str(args.rgb_w))
            rgb_bp.set_attribute("image_size_y", str(args.rgb_h))
            rgb_bp.set_attribute("fov", str(args.rgb_fov))
            rgb_bp.set_attribute("sensor_tick", str(args.lidar_sensor_tick if not args.asynch else 0.0))
            g_rgb_sensor = world.spawn_actor(rgb_bp, sensor_transform)
            g_rgb_sensor.listen(lambda img: rgb_camera_callback(img, run_dir))
            sensor_actors.append(g_rgb_sensor)
            print("Spawned RGB Camera:", g_rgb_sensor.id)

        # Semantic Segmentation Camera
        if args.use_semantic:
            sem_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
            sem_bp.set_attribute("image_size_x", str(args.semantic_w))
            sem_bp.set_attribute("image_size_y", str(args.semantic_h))
            sem_bp.set_attribute("fov", str(args.semantic_fov))
            sem_bp.set_attribute("sensor_tick", str(args.lidar_sensor_tick if not args.asynch else 0.0))
            g_semantic_sensor = world.spawn_actor(sem_bp, sensor_transform)
            g_semantic_sensor.listen(lambda img: semantic_camera_callback(img, run_dir))
            sensor_actors.append(g_semantic_sensor)
            print("Spawned SemanticSeg Camera:", g_semantic_sensor.id)

        # Radar
        if not args.no_radar:
            radar_bp = world.get_blueprint_library().find("sensor.other.radar")
            radar_bp.set_attribute("range", str(args.radar_range))
            radar_bp.set_attribute("horizontal_fov", str(args.radar_hfov))
            radar_bp.set_attribute("vertical_fov", str(args.radar_vfov))
            radar_bp.set_attribute("sensor_tick", str(args.lidar_sensor_tick if not args.asynch else 0.0))
            g_radar_sensor = world.spawn_actor(radar_bp, sensor_transform)
            g_radar_sensor.listen(lambda data: radar_callback(data))
            sensor_actors.append(g_radar_sensor)
            print("Spawned Radar:", g_radar_sensor.id)

        # Semantic LiDAR
        if not args.no_lidar:
            lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
            lidar_bp.set_attribute("range", str(args.lidar_range))
            lidar_bp.set_attribute("upper_fov", str(args.lidar_upper_fov))
            lidar_bp.set_attribute("lower_fov", str(args.lidar_lower_fov))
            lidar_bp.set_attribute("channels", str(args.lidar_channels))
            lidar_bp.set_attribute("rotation_frequency", str(args.lidar_rotation_frequency))
            lidar_bp.set_attribute("points_per_second", str(args.lidar_pps))
            lidar_bp.set_attribute("sensor_tick", str(args.lidar_sensor_tick if not args.asynch else 0.0))

            g_lidar_sensor = world.spawn_actor(lidar_bp, sensor_transform)
            g_lidar_sensor.listen(lambda data: lidar_callback(
                data,
                ped_candidate_tags_set=ped_candidate_tags_set,
                veh_candidate_tags_set=veh_candidate_tags_set,
                min_ped_points=args.min_ped_points,
                min_veh_points=args.min_veh_points,
                sample_save_points=800,
                bbox_margin_xy=args.bbox_margin_xy,
                bbox_margin_z_up=args.bbox_margin_z_up,
                bbox_margin_z_down=args.bbox_margin_z_down,
                max_candidates_for_bbox=args.max_candidates_for_bbox,
                debug_every=args.debug_every,
                debug_nearest_walker=args.debug_nearest_walker,
                debug_walker_tag_hist=args.debug_walker_tag_hist,
                # fusion
                do_fusion_map=args.fusion_map,
                map_downsample_stride=max(1, args.map_downsample_stride),
                map_export_every=max(0, args.map_export_every),
                map_export_dir=os.path.join(run_dir, args.map_export_dir),
                fusion_match_max_delta=max(0, int(args.fusion_match_max_delta)),
                dynamic_filter_with_radar=args.dynamic_filter_with_radar,
                radar_vel_thresh=float(args.radar_vel_thresh),
                radar_assoc_radius=float(args.radar_assoc_radius),
                # semantic
                use_semantic=bool(args.use_semantic),
                drop_semantic_ids=drop_semantic_ids,
                keep_semantic_ids=keep_semantic_ids,
                semantic_colorize=bool(args.semantic_colorize),
                store_semantic_label=bool(args.store_semantic_label),
                semantic_filter_mode=str(args.semantic_filter_mode),
                voxel_size=float(args.voxel_size),
                ply_axis=str(args.ply_axis),
                max_map_points=int(args.max_map_points),
                fallback_rgb=fallback_rgb,
            ))
            sensor_actors.append(g_lidar_sensor)
            print("Spawned Semantic LiDAR:", g_lidar_sensor.id)

        print("\nRunning... Close window or Ctrl+C to stop.\n")
        print(f"Fusion map: {args.fusion_map} | voxel_size={args.voxel_size} | export_every={args.map_export_every} | ply_axis={args.ply_axis}")
        print(f"Option A: KEEP ALL LiDAR geometry. fallback_rgb={fallback_rgb}")
        print(f"Semantic: {args.use_semantic} | filter_mode={args.semantic_filter_mode} | drop={sorted(drop_semantic_ids)} | keep={sorted(keep_semantic_ids)}")
        print(f"Colorize: semantic_colorize={args.semantic_colorize} | store_semantic_label={args.store_semantic_label}")

        while True:
            if synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

            now = time.time()
            if now - last_refresh >= args.refresh_actors_every:
                walkers_cached, vehicles_cached = refresh_caches(world)
                last_refresh = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            display.fill((0, 0, 0))

            if g_vis_camera is not None:
                surface = pygame.surfarray.make_surface(g_vis_camera)
                display.blit(surface, (0, 0))

            draw_overlay(
                display, W, H, args.camera_fov,
                ped_candidate_tags_set, veh_candidate_tags_set,
                draw_stride=max(1, args.draw_stride),
                point_radius=max(1, args.point_radius),
                draw_points=(not args.no_draw_points),
                draw_centroids=(not args.no_draw_centroids),
            )

            fps = clock.get_fps()
            with g_voxel_lock:
                vox_n = len(g_voxel_map)
            top_tags = sorted(g_last_tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            hud = f"FPS:{fps:.1f} Walkers:{walkers_cached} Vehicles:{vehicles_cached} Voxels:{vox_n} TopTags:{top_tags}"
            display.blit(font.render(hud, True, (255, 255, 255)), (10, 10))

            pygame.display.flip()
            clock.tick(60)

    finally:
        pygame.quit()

        try:
            if synchronous_master:
                s = world.get_settings()
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                s.no_rendering_mode = False
                world.apply_settings(s)
        except Exception:
            pass

        print(f"\nStopping {len(sensor_actors)} sensors")
        for s in sensor_actors:
            try:
                s.stop()
            except Exception:
                pass

        print("Saving outputs...")
        try:
            with open(os.path.join(run_dir, "lidar_data.json"), "w") as f:
                json.dump(g_lidar_data_list, f)
            with open(os.path.join(run_dir, "camera_data.json"), "w") as f:
                json.dump(g_camera_data_list, f, indent=2)
            with open(os.path.join(run_dir, "pedestrian_detections.json"), "w") as f:
                json.dump(g_pedestrian_detections, f, indent=2)
            with open(os.path.join(run_dir, "vehicle_detections.json"), "w") as f:
                json.dump(g_vehicle_detections, f, indent=2)

            export_voxel_map_ply(
                out_dir=os.path.join(run_dir, "output_map_ply_final"),
                frame=999999,
                axis_mode=args.ply_axis,
                semantic_colorize=bool(args.semantic_colorize),
                store_semantic_label=bool(args.store_semantic_label),
                max_points=int(args.max_map_points),
            )

            print("Saved outputs to:", os.path.abspath(run_dir))
        except Exception as e:
            print("ERROR saving outputs:", repr(e))

        print("Destroying sensor actors only...")
        try:
            client.apply_batch([carla.command.DestroyActor(s.id) for s in sensor_actors])
        except Exception:
            pass

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting.")

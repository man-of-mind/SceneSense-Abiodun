#!/usr/bin/env python3
"""Bounded CARLA smoke: can synchronized depth give a trustworthy pedestrian
visibility / occlusion contract for a future Route B collection?

Diagnostic only. Nothing here is wired into the canonical v2 collector.
The algorithm, the tolerance and the eligibility rule are frozen in
``DEPTH_VISIBILITY_ALGORITHM_V1.md`` and were registered before this ran.

One stationary ego at the canonical Route B camera mounting, one RGB and one
colocated depth camera at the exact Route B resolution/FOV, a small pool of
static actors (no AI controllers, no traffic manager) teleported between
controlled visibility stages inside one fresh world.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_COLLECTION_ROOT = HERE.parent
REPO_ROOT = DATA_COLLECTION_ROOT.parent
for _p in (str(REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import carla  # noqa: E402

from data_collection.render_provenance_v1 import (  # noqa: E402
    RenderProvenanceError,
    assert_epic_rendering,
    render_provenance,
)

# --- frozen Route B geometry (see DEPTH_VISIBILITY_ALGORITHM_V1.md sec.1) ----
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 120.0
MODEL_INPUT_WIDTH = 768
MODEL_INPUT_HEIGHT = 432
EGO_CAMERA = dict(x=1.8, y=0.0, z=1.55, pitch=-4.0, yaw=0.0, roll=0.0)
WORLD_DELTA_S = 0.05  # 20 Hz, the v2 world tick
TIMESTAMP_TOLERANCE_S = 1e-4  # the established v2 synchronous tolerance

# --- frozen contract constants (registered before results) ------------------
DEPTH_TOLERANCE_M = 0.25
ELIGIBILITY_MAX_DISTANCE_M = 40.0
ELIGIBILITY_MIN_AREA_PX = 12.0
ELIGIBILITY_MIN_MODEL_INPUT_PX = 12
ELIGIBILITY_MIN_VISIBLE_FRACTION = 0.10
SENSITIVITY_FRACTIONS = (0.05, 0.20)

FRAMES_PER_STAGE = 3
SETTLE_TICKS = 8
CARLA_MAX_DEPTH_M = 1000.0


class SmokeRuntimeError(RuntimeError):
    """CARLA/sensor/cleanup failure that prevents a valid conclusion."""


# ---------------------------------------------------------------------------
# geometry helpers - byte-identical to the canonical Route B implementations
# ---------------------------------------------------------------------------
def intrinsics_at(width: int, height: int, fov_deg: float) -> np.ndarray:
    f = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return np.array(
        [[f, 0.0, float(width) / 2.0], [0.0, f, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def bbox_corner_offsets(extent: "carla.Vector3D") -> np.ndarray:
    ex, ey, ez = float(extent.x), float(extent.y), float(extent.z)
    return np.asarray(
        [[ex, ey, ez], [ex, ey, -ez], [ex, -ey, ez], [ex, -ey, -ez],
         [-ex, ey, ez], [-ex, ey, -ez], [-ex, -ey, ez], [-ex, -ey, -ez]],
        dtype=np.float64,
    )


def actor_bbox_world_points(actor: "carla.Actor") -> tuple[np.ndarray, np.ndarray]:
    bbox = actor.bounding_box
    center_local = np.asarray(
        [bbox.location.x, bbox.location.y, bbox.location.z], dtype=np.float64
    )
    local_points = center_local[None, :] + bbox_corner_offsets(bbox.extent)
    matrix = np.asarray(actor.get_transform().get_matrix(), dtype=np.float64)
    homo = np.concatenate([local_points, np.ones((local_points.shape[0], 1))], axis=1)
    corners_world = (matrix @ homo.T).T[:, :3]
    center_world = (matrix @ np.asarray([*center_local, 1.0], dtype=np.float64).T).T[:3]
    return center_world, corners_world


def decode_depth_m(depth_image: Any) -> np.ndarray:
    """Repository convention: cooperative_fusion/engine_gt_prototype.py."""
    raw = np.frombuffer(depth_image.raw_data, np.uint8).reshape(
        (depth_image.height, depth_image.width, 4)
    ).astype(np.float32)
    b, g, r = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
    norm = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return CARLA_MAX_DEPTH_M * norm


def rgb_array(image: Any) -> np.ndarray:
    raw = np.frombuffer(image.raw_data, np.uint8).reshape((image.height, image.width, 4))
    return raw[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB


def nearest_downsample_count(mask_full: np.ndarray) -> int:
    """Count of depth-consistent pixels at model input resolution (nearest, no
    area estimate)."""
    ys = (np.arange(MODEL_INPUT_HEIGHT) * (CAMERA_HEIGHT / MODEL_INPUT_HEIGHT)).astype(np.int64)
    xs = (np.arange(MODEL_INPUT_WIDTH) * (CAMERA_WIDTH / MODEL_INPUT_WIDTH)).astype(np.int64)
    return int(mask_full[np.ix_(ys, xs)].sum())


def visibility_metrics(
    actor: "carla.Actor",
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    depth_m: np.ndarray,
    camera_location: "carla.Location",
) -> dict[str, Any] | None:
    center_world, corners_world = actor_bbox_world_points(actor)
    distance_m = float(actor.get_location().distance(camera_location))
    homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    corners_cam = (camera_inverse @ homo.T).T[:, :3]
    depth = corners_cam[:, 0]
    in_front = depth > 0.05
    if not np.any(in_front):
        return None
    x = depth[in_front]
    y = corners_cam[in_front, 1]
    z = corners_cam[in_front, 2]
    u = intrinsics[0, 2] + (y / x) * intrinsics[0, 0]
    v = intrinsics[1, 2] - (z / x) * intrinsics[1, 1]
    x1 = float(np.clip(np.min(u), 0.0, float(CAMERA_WIDTH)))
    y1 = float(np.clip(np.min(v), 0.0, float(CAMERA_HEIGHT)))
    x2 = float(np.clip(np.max(u), 0.0, float(CAMERA_WIDTH)))
    y2 = float(np.clip(np.max(v), 0.0, float(CAMERA_HEIGHT)))
    box_w, box_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    near_m, far_m = float(np.min(x)), float(np.max(x))

    c0, r0 = int(math.floor(x1)), int(math.floor(y1))
    c1, r1 = int(math.ceil(x2)), int(math.ceil(y2))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(CAMERA_WIDTH, max(c1, c0 + 1)), min(CAMERA_HEIGHT, max(r1, r0 + 1))
    roi = depth_m[r0:r1, c0:c1]
    roi_px = int(roi.size)
    if roi_px == 0:
        return None
    lo, hi = near_m - DEPTH_TOLERANCE_M, far_m + DEPTH_TOLERANCE_M
    consistent = (roi >= lo) & (roi <= hi)
    closer = roi < lo
    farther = roi > hi
    consistent_px = int(consistent.sum())

    mask_full = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=bool)
    mask_full[r0:r1, c0:c1] = consistent

    return {
        "gt_distance_m": distance_m,
        "gt_bbox_x": x1, "gt_bbox_y": y1, "gt_bbox_w": box_w, "gt_bbox_h": box_h,
        "projected_area_px": box_w * box_h,
        "roi_px": roi_px,
        "near_m": near_m, "far_m": far_m,
        "depth_consistent_px": consistent_px,
        "visible_fraction": consistent_px / roi_px,
        "model_input_visible_px": nearest_downsample_count(mask_full),
        "occluder_closer_fraction": float(closer.sum()) / roi_px,
        "background_farther_fraction": float(farther.sum()) / roi_px,
        "object_world_x": float(center_world[0]),
        "object_world_y": float(center_world[1]),
        "object_world_z": float(center_world[2]),
        "_roi": (c0, r0, c1, r1),
        "_mask": mask_full,
    }


def eligible(row: dict[str, Any], min_fraction: float = ELIGIBILITY_MIN_VISIBLE_FRACTION) -> bool:
    return (
        row["gt_distance_m"] <= ELIGIBILITY_MAX_DISTANCE_M
        and row["projected_area_px"] >= ELIGIBILITY_MIN_AREA_PX
        and row["model_input_visible_px"] >= ELIGIBILITY_MIN_MODEL_INPUT_PX
        and row["visible_fraction"] >= min_fraction
    )


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------
PARK_FORWARD_M = -30.0  # out of the forward frustum when a stage does not use an actor


def ego_relative_location(
    ego_transform: "carla.Transform", forward_m: float, right_m: float, z_m: float
) -> "carla.Location":
    yaw = math.radians(float(ego_transform.rotation.yaw))
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -math.sin(yaw), math.cos(yaw)
    return carla.Location(
        x=float(ego_transform.location.x) + fx * forward_m + rx * right_m,
        y=float(ego_transform.location.y) + fy * forward_m + ry * right_m,
        z=float(z_m),
    )


class Smoke:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.output_dir)
        self.client: Any = None
        self.world: Any = None
        self.original_settings: Any = None
        self.ego: Any = None
        self.sensors: list[Any] = []
        self.pool: dict[str, Any] = {}
        self.pool_dz: dict[str, float] = {}
        self.queues = {"rgb": queue.Queue(), "depth": queue.Queue()}
        self.intrinsics = intrinsics_at(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV)
        self.rows: list[dict[str, Any]] = []
        self.sync_records: list[dict[str, Any]] = []
        self.seen_frames: list[int] = []
        self.evidence: dict[str, Any] = {}
        self.cleanup_state = "not_run"
        self.warnings: list[str] = []
        self.stage_manifest: list[dict[str, Any]] = []
        self.current_stage = "probe"
        self.ego_pose_probe: list[dict[str, Any]] = []

    # -- world ---------------------------------------------------------
    def connect(self) -> None:
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(120.0)
        # Explicit reload, exactly as run_route_b_density_loop.py does. A freshly
        # booted server has no populated episode settings, and get_settings()
        # raises bad_optional_access until a world is loaded.
        self.world = self.client.load_world("Town10HD_Opt", True)
        map_name = str(self.world.get_map().name)
        if "Town10HD_Opt" not in map_name:
            raise SmokeRuntimeError(f"unexpected map {map_name!r}; expected Town10HD_Opt")
        try:
            self.original_settings = self.world.get_settings()
            settings = self.world.get_settings()
        except RuntimeError:
            self.original_settings = None
            settings = carla.WorldSettings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = WORLD_DELTA_S
        settings.no_rendering_mode = False  # rendering must remain enabled
        self.world.apply_settings(settings)
        self.world.tick()

    def preflight_render(self) -> dict[str, Any]:
        provenance = render_provenance(
            self.world, self.client, port=self.args.port,
            camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT, camera_fov=CAMERA_FOV,
        )
        assert_epic_rendering(provenance)
        return provenance

    def spawn_ego_and_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        ego_bp = (bp_lib.filter("vehicle.lincoln.mkz") or bp_lib.filter("vehicle.*"))[0]
        ego = None
        for index in (self.args.ego_spawn_index,) + tuple(range(len(spawn_points))):
            ego = self.world.try_spawn_actor(ego_bp, spawn_points[index])
            if ego is not None:
                self.ego_spawn_index = int(index)
                break
        if ego is None:
            raise SmokeRuntimeError("could not spawn the stationary ego")
        self.ego = ego
        self.ego.apply_control(carla.VehicleControl(hand_brake=True))
        self.ego.set_simulate_physics(False)
        for _ in range(5):
            self.world.tick()

        camera_transform = carla.Transform(
            carla.Location(x=EGO_CAMERA["x"], y=EGO_CAMERA["y"], z=EGO_CAMERA["z"]),
            carla.Rotation(pitch=EGO_CAMERA["pitch"], yaw=EGO_CAMERA["yaw"], roll=EGO_CAMERA["roll"]),
        )
        for name, blueprint_id in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth")):
            bp = bp_lib.find(blueprint_id)
            bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
            bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
            bp.set_attribute("fov", str(CAMERA_FOV))
            bp.set_attribute("sensor_tick", "0.0")  # free-running, as in Route B v2
            sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.ego)
            sensor.listen(lambda item, key=name: self.queues[key].put(item))
            self.sensors.append(sensor)
        self.camera, self.depth_camera = self.sensors
        for _ in range(10):
            self.world.tick()
            self._drain_all()
        self._select_ego_pose(spawn_points)

    def _select_ego_pose(self, spawn_points: list[Any]) -> None:
        """Pick the first spawn pose whose forward corridor is actually open.

        Stages 1/2 need ~30 m of clear space ahead; a spawn point facing a wall
        would make the "clearly visible" stages meaningless. Measured from the
        depth camera itself, so it is a property of the rendered scene, not a
        guess about the map.
        """
        cx, cy = int(self.intrinsics[0, 2]), int(self.intrinsics[1, 2])
        candidates = [self.ego_spawn_index] + [
            i for i in range(min(len(spawn_points), 40)) if i != self.ego_spawn_index
        ]
        self.ego_pose_probe: list[dict[str, Any]] = []
        for index in candidates:
            self.ego.set_transform(spawn_points[index])
            for _ in range(4):
                self.world.tick()
            self._drain_all()
            _, _, _, depth_m, _ = self.capture()
            corridor = float(np.median(depth_m[cy - 30:cy + 30, cx - 150:cx + 150]))
            self.ego_pose_probe.append({"spawn_index": index, "median_forward_depth_m": corridor})
            if corridor >= 35.0:
                self.ego_spawn_index = int(index)
                return
        best = max(self.ego_pose_probe, key=lambda r: r["median_forward_depth_m"])
        self.ego_spawn_index = int(best["spawn_index"])
        self.ego.set_transform(spawn_points[self.ego_spawn_index])
        for _ in range(4):
            self.world.tick()
        self._drain_all()
        self.warnings.append(
            "no spawn pose reached a 35 m clear forward corridor; used the best available "
            f"({best['median_forward_depth_m']:.1f} m at index {self.ego_spawn_index})"
        )

    def _drain_all(self) -> None:
        for q in self.queues.values():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def spawn_pool(self) -> None:
        """Static actor pool. No AI controllers, no traffic manager, no autopilot."""
        bp_lib = self.world.get_blueprint_library()
        ego_tf = self.ego.get_transform()
        road_z = float(ego_tf.location.z)

        def pick_vehicle(*patterns: str) -> Any:
            for pattern in patterns:
                found = list(bp_lib.filter(pattern))
                if found:
                    return found[0]
            return list(bp_lib.filter("vehicle.*"))[0]

        walker_candidates = sorted(bp_lib.filter("walker.pedestrian.*"), key=lambda b: b.id)
        if not walker_candidates:
            raise SmokeRuntimeError("no walker.pedestrian.* blueprints in this build")
        specs = [
            # CARLA 0.10 blueprint ids (verified against the running server).
            ("occluder", pick_vehicle("vehicle.sprinter.mercedes", "vehicle.carlacola.actors",
                                      "vehicle.fuso.mitsubishi", "vehicle.*"), 12.0, 0.0),
            ("vehicle_a", pick_vehicle("vehicle.mini.cooper", "vehicle.*"), 16.0, -4.5),
            ("vehicle_b", pick_vehicle("vehicle.dodge.charger", "vehicle.*"), 22.0, 6.0),
            ("walker", walker_candidates[0], 10.0, 0.0),
        ]
        for name, bp, forward, right in specs:
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", f"depth_smoke_{name}")
            actor = None
            for z_try in (1.0, 1.6, 2.2, 3.0):
                location = ego_relative_location(ego_tf, forward, right, road_z + z_try)
                transform = carla.Transform(location, carla.Rotation(yaw=float(ego_tf.rotation.yaw) + 180.0))
                actor = self.world.try_spawn_actor(bp, transform)
                if actor is not None:
                    break
            if actor is None:
                raise SmokeRuntimeError(f"could not spawn pool actor {name} ({bp.id})")
            self.pool[name] = actor
        for _ in range(30):  # let everything settle onto the ground under physics
            self.world.tick()
        for name, actor in self.pool.items():
            self.pool_dz[name] = float(actor.get_transform().location.z) - road_z
            if hasattr(actor, "apply_control") and name != "walker":
                try:
                    actor.apply_control(carla.VehicleControl(hand_brake=True))
                except Exception:
                    pass
            actor.set_simulate_physics(False)  # deterministic teleport between stages
        self._drain_all()

    def place(self, placements: dict[str, tuple[float, float]]) -> None:
        ego_tf = self.ego.get_transform()
        road_z = float(ego_tf.location.z)
        for name, actor in self.pool.items():
            forward, right = placements.get(name, (PARK_FORWARD_M, 0.0))
            location = ego_relative_location(ego_tf, forward, right, road_z + self.pool_dz[name])
            actor.set_transform(
                carla.Transform(location, carla.Rotation(yaw=float(ego_tf.rotation.yaw) + 180.0))
            )
        for _ in range(SETTLE_TICKS):
            self.world.tick()
        self._drain_all()

    # -- synchronized capture ------------------------------------------
    def _wait_exact(self, name: str, frame_id: int) -> Any:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                item = self.queues[name].get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if int(item.frame) == int(frame_id):
                return item
            if int(item.frame) > int(frame_id):
                raise SmokeRuntimeError(
                    f"{name} sensor overshot frame {frame_id} (got {int(item.frame)}): out-of-order capture"
                )
        raise SmokeRuntimeError(f"{name} sensor never produced frame {frame_id}")

    def capture(self) -> tuple[int, Any, np.ndarray, np.ndarray, np.ndarray]:
        frame_id = int(self.world.tick())
        rgb_image = self._wait_exact("rgb", frame_id)
        depth_image = self._wait_exact("depth", frame_id)
        snapshot_ts = float(self.world.get_snapshot().timestamp.elapsed_seconds)
        timestamps = [float(rgb_image.timestamp), float(depth_image.timestamp), snapshot_ts]
        delta = max(timestamps) - min(timestamps)
        if frame_id in self.seen_frames:
            raise SmokeRuntimeError(f"duplicate sensor frame id {frame_id}")
        if self.seen_frames and frame_id <= self.seen_frames[-1]:
            raise SmokeRuntimeError(f"out-of-order sensor frame id {frame_id}")
        self.seen_frames.append(frame_id)
        rgb = rgb_array(rgb_image)
        depth_m = decode_depth_m(depth_image)
        self.sync_records.append({
            "stage": self.current_stage,
            "frame_id": frame_id,
            "rgb_frame_id": int(rgb_image.frame),
            "depth_frame_id": int(depth_image.frame),
            "frame_ids_identical": int(rgb_image.frame) == int(depth_image.frame) == frame_id,
            "rgb_timestamp_s": float(rgb_image.timestamp),
            "depth_timestamp_s": float(depth_image.timestamp),
            "world_timestamp_s": snapshot_ts,
            "timestamp_delta_s": delta,
            "rgb_nonempty": bool(rgb.std() > 0.0),
            "depth_nonempty": bool(depth_m.std() > 0.0),
            "depth_all_finite": bool(np.all(np.isfinite(depth_m))),
            "depth_min_m": float(depth_m.min()),
            "depth_max_m": float(depth_m.max()),
            "depth_plausible": bool(np.all(np.isfinite(depth_m))
                                    and float(depth_m.min()) >= 0.0
                                    and float(depth_m.max()) <= CARLA_MAX_DEPTH_M),
        })
        camera_inverse = np.asarray(
            self.camera.get_transform().get_inverse_matrix(), dtype=np.float64
        )
        return frame_id, rgb_image, rgb, depth_m, camera_inverse

    # -- stages ---------------------------------------------------------
    def probe_static_geometry(self) -> tuple[float, float] | None:
        """Deterministic placement behind static scene geometry: read the depth
        image once, pick a bearing whose static hit is 8-50 m, place the walker
        4 m beyond it along the same bearing."""
        self.place({})  # everything parked behind the ego
        _, _, _, depth_m, _ = self.capture()
        fx = self.intrinsics[0, 0]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        row = int(cy) + 40  # slightly below the horizon: facades, not sky
        best: tuple[float, float] | None = None
        for col in range(60, CAMERA_WIDTH - 60, 20):
            d = float(np.median(depth_m[row - 4:row + 5, col - 4:col + 5]))
            if not (8.0 <= d <= 50.0):
                continue
            bearing = math.atan2((col - cx) / fx, 1.0)
            forward = (d + 4.0) * math.cos(bearing)
            right = (d + 4.0) * math.sin(bearing)
            if abs(right) > 25.0:
                continue
            if best is None or d < best[0]:
                best = (d, forward, right)  # type: ignore[assignment]
        if best is None:
            return None
        return float(best[1]), float(best[2])  # type: ignore[index]

    def build_stages(self) -> list[dict[str, Any]]:
        """Stage placements. The occluder edge is computed from the occluder's
        real bounding box so the partial/heavy/full geometry is deterministic
        rather than guessed."""
        occ = self.pool["occluder"]
        walker = self.pool["walker"]
        occ_half_w = float(occ.bounding_box.extent.y)
        ped_half_w = float(walker.bounding_box.extent.y)
        occ_forward, ped_forward = 12.0, 14.5
        # lateral position, at the pedestrian range, of the occluder's silhouette edge
        edge_at_ped = occ_half_w * (ped_forward / occ_forward)
        partial_r = edge_at_ped + 1.6 * ped_half_w   # most of the body clear of the edge
        heavy_r = edge_at_ped - 0.6 * ped_half_w     # only a sliver clear of the edge
        self.geometry_notes = {
            "occluder_type_id": str(occ.type_id),
            "occluder_half_width_m": occ_half_w,
            "occluder_half_height_m": float(occ.bounding_box.extent.z),
            "walker_type_id": str(walker.type_id),
            "walker_half_width_m": ped_half_w,
            "occluder_forward_m": occ_forward,
            "pedestrian_forward_m": ped_forward,
            "occluder_edge_lateral_at_pedestrian_range_m": edge_at_ped,
            "partial_lateral_m": partial_r,
            "heavy_lateral_m": heavy_r,
        }
        stages: list[dict[str, Any]] = [
            {"stage": "S1_ped_visible_10m", "expected": "visible",
             "targets": {"walker": "person"},
             "placements": {"walker": (10.0, 0.0)}},
            {"stage": "S2_ped_visible_30m", "expected": "visible",
             "targets": {"walker": "person"},
             "placements": {"walker": (30.0, 0.0)}},
            {"stage": "S3_ped_partial_behind_vehicle", "expected": "partial",
             "targets": {"walker": "person", "occluder": "vehicle"},
             "placements": {"walker": (ped_forward, partial_r), "occluder": (occ_forward, 0.0)}},
            {"stage": "S4_ped_heavy_occluded", "expected": "heavy",
             "targets": {"walker": "person", "occluder": "vehicle"},
             "placements": {"walker": (ped_forward, heavy_r), "occluder": (occ_forward, 0.0)}},
            {"stage": "S5_ped_fully_occluded", "expected": "fully_occluded",
             "targets": {"walker": "person", "occluder": "vehicle"},
             "placements": {"walker": (ped_forward, 0.0), "occluder": (occ_forward, 0.0)}},
        ]
        static_placement = self.probe_static_geometry()
        if static_placement is not None:
            stages.append({
                "stage": "S6_ped_behind_static_geometry", "expected": "fully_occluded",
                "targets": {"walker": "person"},
                "placements": {"walker": static_placement},
            })
            self.geometry_notes["static_geometry_placement_forward_right_m"] = list(static_placement)
        else:
            self.warnings.append(
                "S6 skipped: no deterministic static-geometry hit in 8-50 m on the probe row"
            )
        stages.append({
            "stage": "S7_vehicle_controls", "expected": "control",
            "targets": {"vehicle_a": "vehicle", "vehicle_b": "vehicle", "occluder": "vehicle"},
            "placements": {"vehicle_a": (16.0, -4.5), "occluder": (9.0, 0.0),
                           "vehicle_b": (14.0, 0.0)},
            "target_expected": {"vehicle_a": "visible", "vehicle_b": "fully_occluded",
                                "occluder": "visible"},
        })
        return stages

    def run_stage(self, stage: dict[str, Any]) -> None:
        self.current_stage = stage["stage"]
        self.place(stage["placements"])
        camera_location = self.camera.get_transform().location
        for frame_index in range(FRAMES_PER_STAGE):
            frame_id, rgb_image, rgb, depth_m, camera_inverse = self.capture()
            frame_rows: list[dict[str, Any]] = []
            for actor_key, label in stage["targets"].items():
                actor = self.pool[actor_key]
                metrics = visibility_metrics(
                    actor, camera_inverse, self.intrinsics, depth_m, camera_location
                )
                if metrics is None:
                    self.warnings.append(
                        f"{stage['stage']}/{actor_key}: no valid projection at frame {frame_id}"
                    )
                    continue
                expected = stage.get("target_expected", {}).get(actor_key, stage["expected"])
                row = {
                    "stage": stage["stage"], "frame_index": frame_index, "frame_id": frame_id,
                    "actor_key": actor_key, "actor_id": int(actor.id),
                    "actor_type_id": str(actor.type_id), "label": label,
                    "expected_visibility": expected,
                    **{k: v for k, v in metrics.items() if not k.startswith("_")},
                    "eligible": bool(eligible(metrics)),
                    "eligible_at_0.05": bool(eligible(metrics, 0.05)),
                    "eligible_at_0.20": bool(eligible(metrics, 0.20)),
                }
                self.rows.append(row)
                frame_rows.append({**row, "_roi": metrics["_roi"], "_mask": metrics["_mask"]})
            if frame_index == 0:
                self.evidence[stage["stage"]] = {
                    "rgb": rgb, "depth_m": depth_m, "frame_id": frame_id, "rows": frame_rows,
                }
        self.stage_manifest.append({
            "stage": stage["stage"],
            "expected": stage["expected"],
            "targets": stage["targets"],
            "target_expected": stage.get("target_expected", {}),
            "placements_ego_relative_forward_right_m": {
                k: list(v) for k, v in stage["placements"].items()
            },
            "frames": FRAMES_PER_STAGE,
        })

    # -- cleanup --------------------------------------------------------
    def cleanup(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception as exc:  # noqa: BLE001
                records.append({"actor": "sensor", "id": int(sensor.id), "stop_error": str(exc)})
        for actor in list(self.sensors) + list(self.pool.values()) + ([self.ego] if self.ego else []):
            actor_id = int(actor.id)
            try:
                ok = bool(actor.destroy())
            except Exception as exc:  # noqa: BLE001
                ok, error = False, str(exc)
            else:
                error = ""
            records.append({"id": actor_id, "destroyed": ok, "error": error})
        try:
            if self.world is not None:
                restore = self.original_settings
                if restore is None:
                    restore = carla.WorldSettings()
                    restore.synchronous_mode = False
                    restore.no_rendering_mode = False
                self.world.apply_settings(restore)
                # POST-RUN FIX (applied 2026-08-27, after the 20260827_023930
                # artifacts were produced): actor destruction is only committed
                # on the next tick, so the residual scan below read a stale
                # actor list and reported 4 phantom survivors. The run itself
                # did clean up - verified independently against the live world.
                self.world.wait_for_tick(seconds=10.0)
        except Exception as exc:  # noqa: BLE001
            records.append({"actor": "world_settings", "destroyed": False, "error": str(exc)})
        alive = 0
        try:
            actors = self.world.get_actors()
            alive = sum(
                1 for a in actors
                if str(a.attributes.get("role_name", "")).startswith("depth_smoke_")
            )
        except Exception as exc:  # noqa: BLE001
            records.append({"actor": "residual_scan", "destroyed": False, "error": str(exc)})
        succeeded = all(r.get("destroyed", True) for r in records) and alive == 0
        self.cleanup_state = "succeeded" if succeeded else "failed"
        return {"succeeded": succeeded, "residual_smoke_actors": alive, "records": records}


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def write_contact_sheet(smoke: Smoke, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    stages = list(smoke.evidence.keys())
    fig, axes = plt.subplots(len(stages), 3, figsize=(16.5, 3.1 * len(stages)))
    if len(stages) == 1:
        axes = np.asarray([axes])
    for row_index, stage in enumerate(stages):
        payload = smoke.evidence[stage]
        rgb, depth_m, rows = payload["rgb"], payload["depth_m"], payload["rows"]

        ax = axes[row_index, 0]
        ax.imshow(rgb)
        for row in rows:
            colour = "magenta" if row["label"] == "person" else "cyan"
            ax.add_patch(Rectangle(
                (row["gt_bbox_x"], row["gt_bbox_y"]), row["gt_bbox_w"], row["gt_bbox_h"],
                fill=False, edgecolor=colour, linewidth=1.6,
            ))
            ax.text(row["gt_bbox_x"], max(12.0, row["gt_bbox_y"] - 6.0), row["actor_key"],
                    color=colour, fontsize=7)
        ax.set_title(f"{stage}  RGB + projected boxes  (frame {payload['frame_id']})", fontsize=8)
        ax.axis("off")

        ax = axes[row_index, 1]
        ax.imshow(np.clip(depth_m, 0.0, 60.0), cmap="viridis")
        ax.set_title("decoded depth (m, clipped 0-60)", fontsize=8)
        ax.axis("off")

        ax = axes[row_index, 2]
        overlay = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
        for row in rows:
            mask = row["_mask"]
            tint = (255, 0, 255) if row["label"] == "person" else (0, 255, 255)
            for channel in range(3):
                overlay[:, :, channel][mask] = tint[channel]
            c0, r0, c1, r1 = row["_roi"]
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                   edgecolor="white", linewidth=0.8, linestyle=":"))
        ax.imshow(overlay)
        caption = "\n".join(
            f"{r['actor_key']}[{r['expected_visibility']}] vf={r['visible_fraction']:.3f} "
            f"px@input={r['model_input_visible_px']} -> "
            f"{'ELIGIBLE' if r['eligible'] else 'REJECTED'}"
            for r in rows
        )
        ax.set_title("depth-consistent actor pixels", fontsize=8)
        ax.text(0.01, -0.03, caption, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", family="monospace")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def _per_stage_groups(sync: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in sync:
        groups.setdefault(record["stage"], []).append(record)
    return [g for key, g in groups.items() if key != "probe"]


def evaluate_gates(smoke: Smoke) -> dict[str, Any]:
    rows = smoke.rows
    sync = smoke.sync_records

    def stage_rows(stage_prefix: str, actor_key: str | None = None) -> list[dict[str, Any]]:
        return [
            r for r in rows
            if r["stage"].startswith(stage_prefix) and (actor_key is None or r["actor_key"] == actor_key)
        ]

    visible_rows = stage_rows("S1") + stage_rows("S2")
    occluded_rows = stage_rows("S5")
    partial_rows = stage_rows("S3", "walker")
    heavy_rows = stage_rows("S4", "walker")

    visible_vf = [r["visible_fraction"] for r in visible_rows if r["actor_key"] == "walker"]
    occluded_vf = [r["visible_fraction"] for r in occluded_rows if r["actor_key"] == "walker"]
    vf_visible_median = float(np.median(visible_vf)) if visible_vf else float("nan")
    vf_occluded_median = float(np.median(occluded_vf)) if occluded_vf else float("nan")
    separation_abs = vf_visible_median - vf_occluded_median
    separation_ratio = (
        vf_visible_median / vf_occluded_median if vf_occluded_median > 1e-9 else float("inf")
    )

    gates = {
        "G1_frame_ids_identical": all(r["frame_ids_identical"] for r in sync) and bool(sync),
        "G2_timestamp_delta_within_tolerance": all(
            r["timestamp_delta_s"] <= TIMESTAMP_TOLERANCE_S for r in sync
        ),
        "G3_no_missing_duplicate_or_out_of_order_frames": (
            len(set(smoke.seen_frames)) == len(smoke.seen_frames)
            and smoke.seen_frames == sorted(smoke.seen_frames)
            and all(
                [f["frame_id"] for f in group] == list(range(group[0]["frame_id"],
                                                             group[0]["frame_id"] + len(group)))
                for group in _per_stage_groups(sync)
            )
        ),
        "G4_images_non_empty": all(r["rgb_nonempty"] and r["depth_nonempty"] for r in sync),
        "G5_depth_finite_and_plausible": all(r["depth_plausible"] for r in sync),
        "G6_visible_materially_above_fully_occluded": bool(
            separation_abs >= 0.10 and separation_ratio >= 5.0
        ),
        "G7_fully_occluded_rejected": bool(occluded_rows) and all(
            not r["eligible"] for r in occluded_rows if r["actor_key"] == "walker"
        ),
        "G8_clearly_visible_accepted": bool(visible_rows) and all(
            r["eligible"] for r in visible_rows if r["actor_key"] == "walker"
        ),
        "G9_partial_accepted": bool(partial_rows) and all(r["eligible"] for r in partial_rows),
        "G10_cleanup_succeeded": smoke.cleanup_state == "succeeded",
    }
    return {
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "separation": {
            "visible_visible_fraction_median": vf_visible_median,
            "fully_occluded_visible_fraction_median": vf_occluded_median,
            "absolute_difference": separation_abs,
            "ratio": separation_ratio,
            "required_absolute_difference": 0.10,
            "required_ratio": 5.0,
        },
        "heavy_occluded_reported_truthfully": [
            {
                "frame_id": r["frame_id"],
                "visible_fraction": r["visible_fraction"],
                "model_input_visible_px": r["model_input_visible_px"],
                "occluder_closer_fraction": r["occluder_closer_fraction"],
                "eligible_at_registered_0.10": r["eligible"],
                "eligible_at_0.05": r["eligible_at_0.05"],
                "eligible_at_0.20": r["eligible_at_0.20"],
            }
            for r in heavy_rows
        ],
    }


def sensitivity_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stages: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stages.setdefault(f"{row['stage']}::{row['actor_key']}", []).append(row)
    table = []
    for key, group in sorted(stages.items()):
        stage, actor_key = key.split("::")
        table.append({
            "stage": stage,
            "actor_key": actor_key,
            "expected_visibility": group[0]["expected_visibility"],
            "frames": len(group),
            "gt_distance_m_mean": float(np.mean([r["gt_distance_m"] for r in group])),
            "projected_area_px_mean": float(np.mean([r["projected_area_px"] for r in group])),
            "visible_fraction_mean": float(np.mean([r["visible_fraction"] for r in group])),
            "model_input_visible_px_mean": float(np.mean([r["model_input_visible_px"] for r in group])),
            "occluder_closer_fraction_mean": float(
                np.mean([r["occluder_closer_fraction"] for r in group])
            ),
            "background_farther_fraction_mean": float(
                np.mean([r["background_farther_fraction"] for r in group])
            ),
            "eligible_registered_0.10": sum(1 for r in group if r["eligible"]),
            "eligible_at_0.05": sum(1 for r in group if r["eligible_at_0.05"]),
            "eligible_at_0.20": sum(1 for r in group if r["eligible_at_0.20"]),
        })
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ego-spawn-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--carla-launch-command", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # create-only
    started_wall = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")

    smoke = Smoke(args)
    terminal = "DEPTH_VISIBILITY_SMOKE_RUNTIME_FAILED"
    failure = ""
    cleanup_payload: dict[str, Any] = {"succeeded": False, "records": []}
    provenance: dict[str, Any] = {}
    verdict: dict[str, Any] = {}
    sim_start = sim_end = 0.0

    try:
        smoke.connect()
        provenance = smoke.preflight_render()
        sim_start = float(smoke.world.get_snapshot().timestamp.elapsed_seconds)
        smoke.spawn_ego_and_sensors()
        smoke.spawn_pool()
        stages = smoke.build_stages()
        for stage in stages:
            smoke.run_stage(stage)
        sim_end = float(smoke.world.get_snapshot().timestamp.elapsed_seconds)
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            cleanup_payload = smoke.cleanup()
        except Exception as exc:  # noqa: BLE001
            failure = failure or f"cleanup failure: {type(exc).__name__}: {exc}"

    resolved_config = {
        "schema": "route_b_depth_visibility_contract_smoke.v1",
        "started": started_iso,
        "camera_resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "camera_fov_deg": CAMERA_FOV,
        "model_input_size": [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT],
        "ego_camera_transform": EGO_CAMERA,
        "world_delta_s": WORLD_DELTA_S,
        "sensor_tick_s": 0.0,
        "sensor_tick_policy": "free-running, one capture per world tick (Route B v2 contract)",
        "depth_decode": "(R + G*256 + B*256^2)/(256^3-1) * 1000 m  [engine_gt_prototype.decode_depth_m]",
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "registered_eligibility_rule": {
            "max_distance_m": ELIGIBILITY_MAX_DISTANCE_M,
            "min_projected_area_px": ELIGIBILITY_MIN_AREA_PX,
            "min_model_input_visible_px": ELIGIBILITY_MIN_MODEL_INPUT_PX,
            "min_visible_fraction": ELIGIBILITY_MIN_VISIBLE_FRACTION,
            "registered_before_results": True,
        },
        "sensitivity_fractions_reported_only": list(SENSITIVITY_FRACTIONS),
        "frames_per_stage": FRAMES_PER_STAGE,
        "settle_ticks_between_stages": SETTLE_TICKS,
        "semantic_or_instance_tags_used": False,
        "traffic_manager": False,
        "walker_ai_controllers": False,
        "population_replenishment": False,
        "carla_launch_command": args.carla_launch_command,
        "render_provenance": provenance,
        "geometry_notes": getattr(smoke, "geometry_notes", {}),
        "ego_spawn_index": getattr(smoke, "ego_spawn_index", None),
        "ego_pose_corridor_probe": getattr(smoke, "ego_pose_probe", []),
    }
    (out_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, default=str))

    if smoke.rows:
        import csv as _csv
        field_names = [k for k in smoke.rows[0].keys()]
        with (out_dir / "per_frame_visibility_metrics.csv").open("w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(smoke.rows)
    (out_dir / "stage_manifest.json").write_text(
        json.dumps(smoke.stage_manifest, indent=2, default=str)
    )
    (out_dir / "frame_alignment_evidence.json").write_text(
        json.dumps(smoke.sync_records, indent=2, default=str)
    )

    if not failure and smoke.rows:
        verdict = evaluate_gates(smoke)
        # small provenance retention first: one RGB/depth pair per stage, so the
        # contact sheet can be rebuilt offline if plotting fails.
        prov_dir = out_dir / "provenance_frames"
        prov_dir.mkdir(exist_ok=True)
        import cv2
        for stage, payload in smoke.evidence.items():
            cv2.imwrite(str(prov_dir / f"{stage}_rgb.jpg"),
                        np.ascontiguousarray(payload["rgb"][:, :, ::-1]),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            np.savez_compressed(prov_dir / f"{stage}_depth_m.npz",
                                depth_m=payload["depth_m"].astype(np.float16),
                                frame_id=np.int64(payload["frame_id"]))
        contact_ok = True
        try:
            write_contact_sheet(smoke, out_dir / "contact_sheet.png")
        except Exception as exc:  # noqa: BLE001
            contact_ok = False
            smoke.warnings.append(f"contact sheet failed: {type(exc).__name__}: {exc}")
        verdict["contact_sheet_written"] = contact_ok
        runtime_gates = [
            "G1_frame_ids_identical", "G2_timestamp_delta_within_tolerance",
            "G3_no_missing_duplicate_or_out_of_order_frames", "G4_images_non_empty",
            "G5_depth_finite_and_plausible", "G10_cleanup_succeeded",
        ]
        contract_gates = [
            "G6_visible_materially_above_fully_occluded", "G7_fully_occluded_rejected",
            "G8_clearly_visible_accepted", "G9_partial_accepted",
        ]
        if not all(verdict["gates"][g] for g in runtime_gates):
            terminal = "DEPTH_VISIBILITY_SMOKE_RUNTIME_FAILED"
        elif not all(verdict["gates"][g] for g in contract_gates):
            terminal = "DEPTH_VISIBILITY_CONTRACT_FAILED"
        elif not contact_ok:
            terminal = "DEPTH_VISIBILITY_SMOKE_RUNTIME_FAILED"
        else:
            terminal = "DEPTH_VISIBILITY_SMOKE_READY_FOR_MANUAL_REVIEW"


    summary = {
        "schema": "route_b_depth_visibility_contract_smoke_summary.v1",
        "terminal": terminal,
        "failure": failure,
        "warnings": smoke.warnings,
        "wall_seconds": time.time() - started_wall,
        "simulated_seconds": max(0.0, sim_end - sim_start),
        "world_ticks_captured": len(smoke.seen_frames),
        "max_timestamp_delta_s": max((r["timestamp_delta_s"] for r in smoke.sync_records), default=None),
        "cleanup": cleanup_payload,
        "cleanup_state": smoke.cleanup_state,
        "per_stage_summary": sensitivity_table(smoke.rows),
        "verdict": verdict,
        "config": resolved_config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"terminal": terminal, "failure": failure,
                      "warnings": smoke.warnings,
                      "out_dir": str(out_dir)}, indent=2))
    return 0 if terminal == "DEPTH_VISIBILITY_SMOKE_READY_FOR_MANUAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())

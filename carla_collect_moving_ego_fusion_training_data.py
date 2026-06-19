#!/usr/bin/env python3
"""Collect moving-ego RGB+radar fusion samples.

This script is intentionally separate from
`carla_collect_parked_ego_fusion_training_data.py`. The parked collector remains
the stable static-view tool; this moving collector owns Traffic Manager
autopilot, route-progress logging, and visual route probes while reusing the
same dataset schema helpers.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PYTHONAPI_ROOT = Path(__file__).resolve().parents[2]
CARLA_AGENTS_ROOT = PYTHONAPI_ROOT / "carla"
if CARLA_AGENTS_ROOT.exists() and str(CARLA_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CARLA_AGENTS_ROOT))
PY_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"
for root in Path(__file__).resolve().parents[:7]:
    for site_packages in root.glob(f"**/lib/{PY_VERSION}/site-packages"):
        if not list(site_packages.glob("carla*.so")):
            continue
        if str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
        break

import carla_collect_parked_ego_fusion_training_data as parked

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception:  # pragma: no cover - depends on CARLA PythonAPI install layout.
    GlobalRoutePlanner = None


carla = parked.carla
cv2 = parked.cv2

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "fusion_training_data"
DEFAULT_EXPERIMENT_PREFIX = "moving_ego_fusion_training"

ROUTE_PROGRESS_FIELDS = (
    "frame_id",
    "timestamp_s",
    "elapsed_s",
    "saved_samples",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_yaw",
    "ego_speed_mps",
    "distance_traveled_m",
    "distance_to_start_m",
    "loop_count",
    "loop_elapsed_s",
    "loop_distance_m",
    "loop_completed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect moving-ego RGB/radar/GT samples for fusion fine-tuning."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--warmup-ticks", type=int, default=30)
    parser.add_argument("--sensor-timeout", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=10.0)
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument("--sync-world", dest="sync_world", action="store_true")
    sync_group.add_argument("--async-world", dest="sync_world", action="store_false")
    parser.set_defaults(sync_world=True)

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-window-name", default="SceneSense moving fusion collector")
    parser.add_argument("--preview-width", type=int, default=1440)
    parser.add_argument("--preview-height", type=int, default=810)

    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov", type=float, default=120.0)
    parser.add_argument("--model-input-width", type=int, default=768)
    parser.add_argument("--model-input-height", type=int, default=432)

    parser.add_argument("--ego-vehicle-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-role-name", default="scenesense_moving_fusion_ego")
    parser.add_argument("--ego-spawn-index", type=int, default=80)
    parser.add_argument("--ego-spawn-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-right-offset-m", type=float, default=0.0)
    parser.add_argument("--ego-spawn-z-offset-m", type=float, default=0.15)
    parser.add_argument("--ego-spawn-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--ego-freeze", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ego-autopilot-speed-difference-pct", type=float, default=35.0)
    parser.add_argument("--ego-follow-distance-m", type=float, default=18.0)
    parser.add_argument(
        "--ego-ignore-lights-pct",
        type=float,
        default=0.0,
        help=(
            "Traffic Manager percentage for ego traffic-light ignoring. "
            "Use 0 for realistic driving and 100 for SCAN-style continuous route probes."
        ),
    )
    parser.add_argument(
        "--ego-fixed-path-spawn-indices",
        default="",
        help=(
            "Optional comma-separated CARLA spawn indices for a pinned Traffic "
            "Manager route, for example 80,85,91,94,99,80. Leave empty to let "
            "Traffic Manager choose its own route."
        ),
    )
    parser.add_argument(
        "--ego-fixed-path-progress-csv",
        default="",
        help=(
            "Optional route_progress.csv from a previous good moving run. When "
            "set, the ego reuses those recorded x/y/z points as its pinned "
            "Traffic Manager path. This takes priority over spawn indices."
        ),
    )
    parser.add_argument("--ego-fixed-path-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ego-fixed-path-min-spacing-m", type=float, default=8.0)
    parser.add_argument("--ego-disable-lane-change", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ego-camera-x", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_X)
    parser.add_argument("--ego-camera-y", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_Y)
    parser.add_argument("--ego-camera-z", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_Z)
    parser.add_argument("--ego-camera-pitch", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_PITCH)
    parser.add_argument("--ego-camera-yaw", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_YAW)
    parser.add_argument("--ego-camera-roll", type=float, default=parked.fusion_runtime.DEFAULT_EGO_CAMERA_ROLL)
    parser.add_argument("--ego-radar-x", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_X)
    parser.add_argument("--ego-radar-y", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_Y)
    parser.add_argument("--ego-radar-z", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_Z)
    parser.add_argument("--ego-radar-pitch", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_PITCH)
    parser.add_argument("--ego-radar-yaw", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_YAW)
    parser.add_argument("--ego-radar-roll", type=float, default=parked.fusion_runtime.DEFAULT_EGO_RADAR_ROLL)

    parser.add_argument("--radar-range", type=float, default=120.0)
    parser.add_argument("--radar-hfov", type=float, default=120.0)
    parser.add_argument("--radar-vfov", type=float, default=30.0)
    parser.add_argument("--radar-points-per-second", type=int, default=5000)
    parser.add_argument("--radar-max-velocity", type=float, default=20.0)
    parser.add_argument("--radar-raster-radius-px", type=int, default=2)
    parser.add_argument("--stationary-velocity-mps", type=float, default=0.35)
    parser.add_argument("--parked-threshold-s", type=float, default=5.0)
    parser.add_argument("--association-grid-m", type=float, default=1.5)
    parser.add_argument("--max-stale-s", type=float, default=2.0)
    parser.add_argument("--radar-support-margin-m", type=float, default=1.0)

    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=25)
    parser.add_argument("--spawn-radius", type=float, default=95.0)
    parser.add_argument("--npc-vehicle-speed-difference-pct", type=float, default=35.0)
    parser.add_argument("--npc-pedestrian-max-speed-mps", type=float, default=0.9)
    parser.add_argument("--npc-pedestrian-cross-factor", type=float, default=0.5)
    parser.add_argument("--gt-max-distance-m", type=float, default=140.0)
    parser.add_argument("--include-pedestrians", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.14)

    parser.add_argument("--route-progress-every-s", type=float, default=1.0)
    parser.add_argument("--loop-return-radius-m", type=float, default=12.0)
    parser.add_argument("--loop-min-distance-m", type=float, default=250.0)
    parser.add_argument("--loop-min-elapsed-s", type=float, default=30.0)
    parser.add_argument("--stop-after-loops", type=int, default=0)
    parser.add_argument("--stop-on-stuck", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stuck-speed-threshold-mps", type=float, default=0.20)
    parser.add_argument("--stuck-timeout-s", type=float, default=20.0)
    parser.add_argument("--stuck-min-elapsed-s", type=float, default=20.0)
    parser.add_argument(
        "--stuck-ignore-traffic-light-waits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not treat normal red/yellow traffic-light waiting as a stuck condition.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def ego_speed_mps(actor: "carla.Actor") -> float:
    velocity = actor.get_velocity()
    return math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)


def ego_waiting_at_traffic_light(actor: "carla.Actor") -> bool:
    try:
        if not bool(actor.is_at_traffic_light()):
            return False
        traffic_light = actor.get_traffic_light()
        if traffic_light is None:
            return True
        state = traffic_light.get_state()
        return state in {carla.TrafficLightState.Red, carla.TrafficLightState.Yellow}
    except Exception:
        return False


def world_timestamp_s(world: "carla.World") -> float:
    try:
        return float(world.get_snapshot().timestamp.elapsed_seconds)
    except Exception:
        return float(time.time())


def parse_spawn_index_list(text: str) -> List[int]:
    cleaned = str(text or "").replace(";", ",").replace(" ", ",")
    values: List[int] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def copy_location(location: "carla.Location") -> "carla.Location":
    return carla.Location(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
    )


def _append_spaced_location(
    route: List["carla.Location"],
    location: "carla.Location",
    min_spacing_m: float,
) -> None:
    candidate = copy_location(location)
    if not route or route[-1].distance(candidate) >= float(min_spacing_m):
        route.append(candidate)


def build_fixed_tm_path_from_progress_csv(
    *,
    args: argparse.Namespace,
) -> List["carla.Location"]:
    progress_csv = str(args.ego_fixed_path_progress_csv or "").strip()
    if not progress_csv:
        return []

    path = Path(progress_csv).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Fixed route progress CSV not found: {path}")

    min_spacing_m = max(1.0, float(args.ego_fixed_path_min_spacing_m))
    route: List["carla.Location"] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                location = carla.Location(
                    x=float(row["ego_x"]),
                    y=float(row["ego_y"]),
                    z=float(row.get("ego_z", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            _append_spaced_location(route, location, min_spacing_m)

    if bool(args.ego_fixed_path_loop) and len(route) >= 2 and route[-1].distance(route[0]) > min_spacing_m:
        route.append(copy_location(route[0]))
    print(
        "Fixed Traffic Manager path: "
        f"source={path}, route_points={len(route)}, "
        f"min_spacing={min_spacing_m:.1f}m"
    )
    return route


def build_fixed_tm_path_from_spawn_indices(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> List["carla.Location"]:
    indices = parse_spawn_index_list(str(args.ego_fixed_path_spawn_indices))
    if not indices:
        return []

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("Cannot build fixed Traffic Manager path: no map spawn points found.")

    invalid = [idx for idx in indices if idx < 0 or idx >= len(spawn_points)]
    if invalid:
        raise ValueError(
            "Invalid --ego-fixed-path-spawn-indices values "
            f"{invalid}; available index range is 0..{len(spawn_points) - 1}."
        )

    if bool(args.ego_fixed_path_loop) and len(indices) >= 2 and indices[-1] != indices[0]:
        indices = list(indices) + [indices[0]]

    key_points = [spawn_points[idx].location for idx in indices]
    min_spacing_m = max(1.0, float(args.ego_fixed_path_min_spacing_m))
    route: List["carla.Location"] = []
    if len(key_points) < 2:
        return [copy_location(point) for point in key_points]

    if GlobalRoutePlanner is not None:
        planner = GlobalRoutePlanner(world.get_map(), min_spacing_m)
        for start, end in zip(key_points[:-1], key_points[1:]):
            trace = planner.trace_route(start, end)
            if not trace:
                _append_spaced_location(route, start, min_spacing_m)
                _append_spaced_location(route, end, min_spacing_m)
                continue
            for waypoint, _road_option in trace:
                _append_spaced_location(route, waypoint.transform.location, min_spacing_m)
            _append_spaced_location(route, end, min_spacing_m)
    else:
        for point in key_points:
            _append_spaced_location(route, point, min_spacing_m)

    print(
        "Fixed Traffic Manager path: "
        f"spawn_indices={indices}, route_points={len(route)}, "
        f"planner={'yes' if GlobalRoutePlanner is not None else 'no'}"
    )
    return route


def build_fixed_tm_path(
    *,
    world: "carla.World",
    args: argparse.Namespace,
) -> List["carla.Location"]:
    path_from_csv = build_fixed_tm_path_from_progress_csv(args=args)
    if path_from_csv:
        return path_from_csv
    return build_fixed_tm_path_from_spawn_indices(world=world, args=args)


class RouteProgressMonitor:
    def __init__(
        self,
        *,
        start_location: "carla.Location",
        progress_path: Path,
        return_radius_m: float,
        min_distance_m: float,
        min_elapsed_s: float,
        progress_every_s: float,
    ) -> None:
        self.start_location = carla.Location(
            x=float(start_location.x),
            y=float(start_location.y),
            z=float(start_location.z),
        )
        self.progress_path = progress_path
        self.return_radius_m = float(return_radius_m)
        self.min_distance_m = float(min_distance_m)
        self.min_elapsed_s = float(min_elapsed_s)
        self.progress_every_s = max(0.1, float(progress_every_s))
        self.start_timestamp_s: Optional[float] = None
        self.loop_start_timestamp_s: Optional[float] = None
        self.loop_start_distance_m = 0.0
        self.last_logged_timestamp_s: Optional[float] = None
        self.last_location: Optional["carla.Location"] = None
        self.distance_traveled_m = 0.0
        self.loop_count = 0
        self.has_left_start_zone = False
        self.last_loop_elapsed_s: Optional[float] = None
        self.last_loop_distance_m: Optional[float] = None
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=ROUTE_PROGRESS_FIELDS).writeheader()

    def distance_to_start(self, location: "carla.Location") -> float:
        return math.sqrt(
            float(location.x - self.start_location.x) ** 2
            + float(location.y - self.start_location.y) ** 2
            + float(location.z - self.start_location.z) ** 2
        )

    def update(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        ego_transform: "carla.Transform",
        speed_mps: float,
        saved_samples: int,
    ) -> bool:
        location = ego_transform.location
        if self.start_timestamp_s is None:
            self.start_timestamp_s = float(timestamp_s)
            self.loop_start_timestamp_s = float(timestamp_s)
        if self.last_location is not None:
            self.distance_traveled_m += math.sqrt(
                float(location.x - self.last_location.x) ** 2
                + float(location.y - self.last_location.y) ** 2
                + float(location.z - self.last_location.z) ** 2
            )
        self.last_location = carla.Location(
            x=float(location.x),
            y=float(location.y),
            z=float(location.z),
        )

        elapsed_s = float(timestamp_s) - float(self.start_timestamp_s)
        loop_start_ts = float(self.loop_start_timestamp_s or self.start_timestamp_s)
        loop_elapsed_s = float(timestamp_s) - loop_start_ts
        loop_distance_m = self.distance_traveled_m - self.loop_start_distance_m
        distance_to_start_m = self.distance_to_start(location)
        leave_radius_m = max(self.return_radius_m * 1.5, self.return_radius_m + 5.0)
        if distance_to_start_m >= leave_radius_m:
            self.has_left_start_zone = True

        loop_completed = False
        if (
            self.has_left_start_zone
            and distance_to_start_m <= self.return_radius_m
            and loop_distance_m >= self.min_distance_m
            and loop_elapsed_s >= self.min_elapsed_s
        ):
            self.loop_count += 1
            loop_completed = True
            self.has_left_start_zone = False
            self.last_loop_elapsed_s = loop_elapsed_s
            self.last_loop_distance_m = loop_distance_m
            self.loop_start_timestamp_s = float(timestamp_s)
            self.loop_start_distance_m = self.distance_traveled_m
            print(
                "[route] completed loop "
                f"{self.loop_count}: elapsed={loop_elapsed_s:.1f}s, "
                f"distance={loop_distance_m:.1f}m, saved_samples={saved_samples}"
            )

        should_log = loop_completed or self.last_logged_timestamp_s is None
        if self.last_logged_timestamp_s is not None:
            should_log = should_log or (
                float(timestamp_s) - float(self.last_logged_timestamp_s) >= self.progress_every_s
            )
        if should_log:
            self.last_logged_timestamp_s = float(timestamp_s)
            with self.progress_path.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=ROUTE_PROGRESS_FIELDS)
                writer.writerow(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_s": round(float(timestamp_s), 4),
                        "elapsed_s": round(float(elapsed_s), 4),
                        "saved_samples": int(saved_samples),
                        "ego_x": round(float(location.x), 4),
                        "ego_y": round(float(location.y), 4),
                        "ego_z": round(float(location.z), 4),
                        "ego_yaw": round(float(ego_transform.rotation.yaw), 4),
                        "ego_speed_mps": round(float(speed_mps), 4),
                        "distance_traveled_m": round(float(self.distance_traveled_m), 4),
                        "distance_to_start_m": round(float(distance_to_start_m), 4),
                        "loop_count": int(self.loop_count),
                        "loop_elapsed_s": round(float(loop_elapsed_s), 4),
                        "loop_distance_m": round(float(loop_distance_m), 4),
                        "loop_completed": int(loop_completed),
                    }
                )
        return loop_completed

    def summary(self) -> Dict[str, object]:
        return {
            "progress_path": str(self.progress_path),
            "start_location": {
                "x": float(self.start_location.x),
                "y": float(self.start_location.y),
                "z": float(self.start_location.z),
            },
            "return_radius_m": float(self.return_radius_m),
            "min_distance_m": float(self.min_distance_m),
            "min_elapsed_s": float(self.min_elapsed_s),
            "distance_traveled_m": float(self.distance_traveled_m),
            "loop_count": int(self.loop_count),
            "last_loop_elapsed_s": self.last_loop_elapsed_s,
            "last_loop_distance_m": self.last_loop_distance_m,
        }


def configure_ego_autopilot(
    *,
    ego_vehicle: "carla.Actor",
    traffic_manager: "carla.TrafficManager",
    args: argparse.Namespace,
    fixed_path: Sequence["carla.Location"],
) -> None:
    try:
        ego_vehicle.set_simulate_physics(True)
    except RuntimeError:
        pass
    try:
        ego_vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
        )
    except RuntimeError:
        pass
    try:
        ego_vehicle.set_autopilot(True, int(args.tm_port))
        traffic_manager.ignore_lights_percentage(
            ego_vehicle,
            max(0.0, min(100.0, float(args.ego_ignore_lights_pct))),
        )
        traffic_manager.vehicle_percentage_speed_difference(
            ego_vehicle,
            float(args.ego_autopilot_speed_difference_pct),
        )
        traffic_manager.distance_to_leading_vehicle(
            ego_vehicle,
            float(args.ego_follow_distance_m),
        )
        if bool(args.ego_disable_lane_change):
            try:
                traffic_manager.auto_lane_change(ego_vehicle, False)
            except Exception:
                pass
        if fixed_path:
            traffic_manager.set_path(ego_vehicle, list(fixed_path))
    except Exception:
        try:
            ego_vehicle.set_autopilot(True)
        except RuntimeError:
            pass
    print(
        "Ego autopilot: "
        f"speed_diff={float(args.ego_autopilot_speed_difference_pct):.1f}%, "
        f"follow_distance={float(args.ego_follow_distance_m):.1f}m, "
        f"ignore_lights={float(args.ego_ignore_lights_pct):.1f}%, "
        f"fixed_path_points={len(fixed_path)}"
    )


def destroy_actors_batch(client: "carla.Client", actors: Sequence["carla.Actor"]) -> None:
    commands = []
    seen_ids = set()
    for actor in reversed(list(actors)):
        try:
            actor_id = int(actor.id)
        except Exception:
            continue
        if actor_id in seen_ids:
            continue
        seen_ids.add(actor_id)
        commands.append(carla.command.DestroyActor(actor_id))
    if not commands:
        return
    try:
        client.apply_batch(commands)
    except RuntimeError as exc:
        print(f"Warning: batched actor cleanup failed: {exc}")


def configure_background_motion(
    *,
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    vehicles: Sequence["carla.Actor"],
    pedestrian_controllers: Sequence["carla.Actor"],
    args: argparse.Namespace,
) -> None:
    for vehicle in vehicles:
        try:
            traffic_manager.vehicle_percentage_speed_difference(
                vehicle,
                float(args.npc_vehicle_speed_difference_pct),
            )
        except RuntimeError:
            continue
    try:
        world.set_pedestrians_cross_factor(
            clamp(float(args.npc_pedestrian_cross_factor), 0.0, 1.0)
        )
    except RuntimeError:
        pass
    if float(args.npc_pedestrian_max_speed_mps) > 0.0:
        for controller in pedestrian_controllers:
            try:
                controller.set_max_speed(float(args.npc_pedestrian_max_speed_mps))
            except RuntimeError:
                continue
    print(
        "Background motion: "
        f"vehicle_speed_diff={float(args.npc_vehicle_speed_difference_pct):.1f}%, "
        f"ped_max_speed={float(args.npc_pedestrian_max_speed_mps):.2f}m/s, "
        f"ped_cross_factor={float(args.npc_pedestrian_cross_factor):.2f}"
    )


def draw_preview(
    *,
    args: argparse.Namespace,
    image: "carla.Image",
    ego_vehicle: "carla.Actor",
    saved: int,
    monitor: RouteProgressMonitor,
) -> bool:
    if not bool(args.preview):
        return False
    frame = parked.od_demo.camera_image_to_bgr(image)
    if int(args.preview_width) > 0 and int(args.preview_height) > 0:
        frame = cv2.resize(
            frame,
            (int(args.preview_width), int(args.preview_height)),
            interpolation=cv2.INTER_AREA,
        )
    transform = ego_vehicle.get_transform()
    lines = [
        f"Moving fusion collector | saved={saved}/{int(args.max_samples)}",
        f"frame={int(image.frame)} | speed={ego_speed_mps(ego_vehicle):.2f} m/s",
        "distance={:.1f} m | start_gap={:.1f} m | loops={}".format(
            float(monitor.distance_traveled_m),
            float(monitor.distance_to_start(transform.location)),
            int(monitor.loop_count),
        ),
        "press q or Esc to stop",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30
    window_name = str(args.preview_window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(args.preview_width), int(args.preview_height))
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27)


def write_moving_metadata(
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
        "schema": "scenesense_moving_ego_fusion_training_data.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "description": "Saved moving-ego RGB/radar/GT samples for fusion model training.",
        "world": str(world.get_map().name),
        "scenario_id": "moving_ego_training",
        "view_id": f"autopilot_spawn{int(args.ego_spawn_index)}",
        "sample_count_requested": int(args.max_samples),
        "camera_resolution": [int(args.camera_width), int(args.camera_height)],
        "model_input_size": [int(args.model_input_width), int(args.model_input_height)],
        "ego_vehicle": {
            "actor_id": int(ego_vehicle.id),
            "type_id": str(getattr(ego_vehicle, "type_id", "")),
            "transform": parked.transform_payload(ego_vehicle.get_transform()),
        },
        "ego_autopilot": {
            "speed_difference_pct": float(args.ego_autopilot_speed_difference_pct),
            "follow_distance_m": float(args.ego_follow_distance_m),
            "ignore_lights_pct": float(args.ego_ignore_lights_pct),
            "stuck_ignore_traffic_light_waits": bool(args.stuck_ignore_traffic_light_waits),
            "fixed_path_spawn_indices": parse_spawn_index_list(
                str(args.ego_fixed_path_spawn_indices)
            ),
            "fixed_path_progress_csv": str(args.ego_fixed_path_progress_csv),
            "fixed_path_loop": bool(args.ego_fixed_path_loop),
            "fixed_path_min_spacing_m": float(args.ego_fixed_path_min_spacing_m),
            "disable_lane_change": bool(args.ego_disable_lane_change),
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
            "world_transform": parked.transform_payload(camera.get_transform()),
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
            "world_transform": parked.transform_payload(radar.get_transform()),
            "range_m": float(args.radar_range),
            "horizontal_fov": float(args.radar_hfov),
            "vertical_fov": float(args.radar_vfov),
            "points_per_second": int(args.radar_points_per_second),
        },
        "background_motion": {
            "npc_vehicle_speed_difference_pct": float(args.npc_vehicle_speed_difference_pct),
            "npc_pedestrian_max_speed_mps": float(args.npc_pedestrian_max_speed_mps),
            "npc_pedestrian_cross_factor": float(args.npc_pedestrian_cross_factor),
        },
        "split_ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": max(0.0, 1.0 - float(args.train_ratio) - float(args.val_ratio)),
            "seed": int(args.split_seed),
        },
        "command_args": vars(args),
    }
    parked.save_json(dataset_dir / "metadata.json", metadata)


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    experiment_id = str(args.experiment_id).strip() or f"{parked.now_stamp()}_{DEFAULT_EXPERIMENT_PREFIX}"
    dataset_dir = Path(args.output_root).expanduser().resolve() / experiment_id
    dirs = parked.prepare_dataset_dirs(dataset_dir)
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
    traffic_manager.set_global_distance_to_leading_vehicle(float(args.ego_follow_distance_m))
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

        ego_vehicle = parked.fusion_runtime._spawn_parked_ego_vehicle(world=world, args=args)
        actors.append(ego_vehicle)
        anchor_location = ego_vehicle.get_location()
        print(f"Moving ego: id={ego_vehicle.id}, type={ego_vehicle.type_id}")

        background_vehicles = parked.pole_client.spawn_background_vehicles_near(
            client,
            world,
            traffic_manager,
            anchor_location,
            int(args.npc_vehicles),
            float(args.spawn_radius),
        )
        actors.extend(background_vehicles)
        print(f"Spawned background vehicles: {len(background_vehicles)}")

        pedestrians, pedestrian_controllers = parked.pole_client.spawn_background_pedestrians_near(
            client,
            world,
            anchor_location,
            int(args.npc_pedestrians),
            float(args.spawn_radius),
        )
        actors.extend(pedestrians)
        actors.extend(pedestrian_controllers)
        print(f"Spawned pedestrians: {len(pedestrians)}")

        configure_background_motion(
            world=world,
            traffic_manager=traffic_manager,
            vehicles=background_vehicles,
            pedestrian_controllers=pedestrian_controllers,
            args=args,
        )

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

        camera = world.spawn_actor(camera_bp, parked.fusion_runtime._ego_camera_transform(args), attach_to=ego_vehicle)
        semantic_camera = world.spawn_actor(semantic_bp, parked.fusion_runtime._ego_camera_transform(args), attach_to=ego_vehicle)
        radar = world.spawn_actor(radar_bp, parked.fusion_runtime._ego_radar_transform(args), attach_to=ego_vehicle)
        actors.extend([camera, semantic_camera, radar])
        camera.listen(lambda image: parked.od_demo.put_latest(image_queue, image))
        semantic_camera.listen(lambda image: parked.od_demo.put_latest(semantic_queue, image))
        radar.listen(lambda measurement: parked.od_demo.put_latest(radar_queue, measurement))

        write_moving_metadata(
            args=args,
            dataset_dir=dataset_dir,
            experiment_id=experiment_id,
            world=world,
            ego_vehicle=ego_vehicle,
            camera=camera,
            radar=radar,
        )

        for _ in range(max(0, int(args.warmup_ticks))):
            if bool(args.sync_world):
                world.tick()
            else:
                time.sleep(1.0 / max(0.1, float(args.fps)))

        fixed_tm_path = build_fixed_tm_path(world=world, args=args)
        configure_ego_autopilot(
            ego_vehicle=ego_vehicle,
            traffic_manager=traffic_manager,
            args=args,
            fixed_path=fixed_tm_path,
        )
        monitor = RouteProgressMonitor(
            start_location=ego_vehicle.get_location(),
            progress_path=dataset_dir / "route_progress.csv",
            return_radius_m=float(args.loop_return_radius_m),
            min_distance_m=float(args.loop_min_distance_m),
            min_elapsed_s=float(args.loop_min_elapsed_s),
            progress_every_s=float(args.route_progress_every_s),
        )

        tracker = parked.StationaryTrackAccumulator(
            stationary_velocity_mps=float(args.stationary_velocity_mps),
            parked_threshold_s=float(args.parked_threshold_s),
            association_grid_m=float(args.association_grid_m),
            max_stale_s=float(args.max_stale_s),
        )
        actor_stationary_tracker = parked.ActorStationaryTracker(
            stationary_velocity_mps=float(args.stationary_velocity_mps),
            parked_threshold_s=float(args.parked_threshold_s),
        )
        intrinsics_full = parked.fusion_runtime.intrinsics_at(
            int(args.camera_width),
            int(args.camera_height),
            float(args.camera_fov),
        )
        intrinsics_input = parked.fusion_runtime.intrinsics_at(
            int(args.model_input_width),
            int(args.model_input_height),
            float(args.camera_fov),
        )

        saved = 0
        attempts = 0
        stop_reason = "max_samples"
        stuck_started_at_s: Optional[float] = None
        while saved < int(args.max_samples):
            attempts += 1
            if bool(args.sync_world):
                frame_id = int(world.tick())
            else:
                time.sleep(1.0 / max(0.1, float(args.fps)))
                frame_id = 0

            timestamp_s = world_timestamp_s(world)
            speed_mps = ego_speed_mps(ego_vehicle)
            loop_completed = monitor.update(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                ego_transform=ego_vehicle.get_transform(),
                speed_mps=speed_mps,
                saved_samples=saved,
            )
            if loop_completed and int(args.stop_after_loops) > 0 and monitor.loop_count >= int(args.stop_after_loops):
                print(f"Stopping after detected loop_count={monitor.loop_count}.")
                stop_reason = f"loop_count_{monitor.loop_count}"
                break
            elapsed_s = 0.0
            if monitor.start_timestamp_s is not None:
                elapsed_s = float(timestamp_s) - float(monitor.start_timestamp_s)
            traffic_light_wait = bool(args.stuck_ignore_traffic_light_waits) and ego_waiting_at_traffic_light(
                ego_vehicle
            )
            if (
                bool(args.stop_on_stuck)
                and elapsed_s >= float(args.stuck_min_elapsed_s)
                and speed_mps <= float(args.stuck_speed_threshold_mps)
                and not traffic_light_wait
            ):
                if stuck_started_at_s is None:
                    stuck_started_at_s = float(timestamp_s)
                stuck_duration_s = float(timestamp_s) - float(stuck_started_at_s)
                if stuck_duration_s >= float(args.stuck_timeout_s):
                    stop_reason = f"stuck_{stuck_duration_s:.1f}s"
                    print(
                        "Stopping because ego appears stuck: "
                        f"speed={speed_mps:.2f}m/s for {stuck_duration_s:.1f}s."
                    )
                    break
            else:
                stuck_started_at_s = None
            if int(args.sample_stride) > 1 and attempts % int(args.sample_stride) != 0:
                continue

            image = parked.od_demo.wait_for_camera_frame(image_queue, frame_id, float(args.sensor_timeout))
            semantic_image = parked.od_demo.wait_for_camera_frame(semantic_queue, frame_id, float(args.sensor_timeout))
            radar_measurement = parked.wait_for_measurement(radar_queue, frame_id, float(args.sensor_timeout))
            if image is None or semantic_image is None or radar_measurement is None:
                print(f"Warning: missing synchronized sensors at frame {frame_id}; retrying.")
                continue

            camera_matrix = parked.fusion_runtime.actor_world_matrix(camera)
            camera_inverse_matrix = parked.fusion_runtime.actor_world_inverse_matrix(camera)
            radar_matrix = parked.fusion_runtime.actor_world_matrix(radar)
            radar_inverse_matrix = parked.fusion_runtime.actor_world_inverse_matrix(radar)
            detections = parked.radar_raw_to_alt_az_depth_velocity(bytes(radar_measurement.raw_data))
            radar_tensor, radar_points, radar_summary = parked.build_radar_sample(
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
            split = parked.stable_split(sample_id, split_ratios, int(args.split_seed))
            file_paths, mask = parked.save_sample_files(
                dataset_dir=dataset_dir,
                dirs=dirs,
                sample_id=sample_id,
                image=image,
                semantic_image=semantic_image,
                radar_tensor=radar_tensor,
                radar_points=radar_points,
                jpeg_quality=int(args.jpeg_quality),
            )
            manifest_row = parked.build_manifest_row(
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
            manifest_row["scenario_id"] = "moving_ego_training"
            manifest_row["view_id"] = f"autopilot_spawn{int(args.ego_spawn_index)}"
            sample_base = {
                "experiment_id": experiment_id,
                "sample_id": sample_id,
                "frame_id": int(image.frame),
                "timestamp": float(image.timestamp),
                "traffic_light_id": "",
                "scenario_id": "moving_ego_training",
                "view_id": f"autopilot_spawn{int(args.ego_spawn_index)}",
            }
            object_rows = parked.build_object_rows(
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
            stop_requested = draw_preview(
                args=args,
                image=image,
                ego_vehicle=ego_vehicle,
                saved=saved + 1,
                monitor=monitor,
            )
            parked.append_manifest_rows(manifest_path, [manifest_row])
            parked.append_object_box_rows(object_boxes_path, object_rows)
            saved += 1
            if saved == 1 or saved % 10 == 0 or saved >= int(args.max_samples):
                print(
                    f"Saved {saved}/{int(args.max_samples)} samples "
                    f"(frame={int(image.frame)}, objects={len(object_rows)}, "
                    f"vehicle_pixels={manifest_row['vehicle_pixels']}, "
                    f"person_pixels={manifest_row['person_pixels']})"
                )
            if stop_requested:
                print("Preview stop requested; ending collection early.")
                stop_reason = "preview_stop"
                break

        route_summary = monitor.summary()
        route_summary["stop_reason"] = stop_reason
        parked.save_json(dataset_dir / "route_summary.json", route_summary)
        print(f"Done. Dataset: {dataset_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"Object boxes: {object_boxes_path}")
        print(f"Route progress: {dataset_dir / 'route_progress.csv'}")
        print(f"Route loops detected: {monitor.loop_count}")
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
        destroy_actors_batch(client, actors)
        if bool(args.sync_world):
            try:
                world.apply_settings(original_settings)
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
        if bool(args.preview):
            try:
                cv2.destroyWindow(str(args.preview_window_name))
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

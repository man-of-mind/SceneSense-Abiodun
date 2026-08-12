#!/usr/bin/env python3
"""
Deterministic Physical AI CARLA scenario controller UI (v1).

This passive client attaches to the already-running CARLA server and already
loaded world.  It does not start the server, load/reload a map, change world
settings, or call world.tick()/wait_for_tick().  Sensor data and actor motion
therefore follow the existing master clock (for example generate_traffic.py).

The controller creates a reproducible scripted demo with ego vehicle and
pedestrian routes, deterministic NPC spawn choices, a bus/truck occluder,
ground-truth bounding boxes with LOS/NLOS overlap estimates, a front ego radar, and
selectable ego/traffic-light-pole RGB viewpoints.  To avoid the Vulkan memory
failure caused by allocating a camera for every pole, every view shares one
relocatable physical RGB sensor/render target.

Runtime controls
----------------
    W/S/A/D          move the selected manual ego actor
    Shift            pedestrian run modifier
    Space            vehicle handbrake / pedestrian jump
    Arrow keys       selected camera pitch/yaw
    Tab               cycle camera view
    F1                switch manual-control target
    B                 toggle ground-truth boxes
    R                 reset selected camera orientation
    Esc / Ctrl+Q      quit and clean up owned CARLA actors

Replay means a deterministic rebuild from the last launched seed/configuration;
it is not a recording of manual keyboard input.  Exact physics trajectories can
still vary if the external master clock or unrelated world actors differ.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import carla
import numpy as np

try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "pygame is required; install it in the CARLA Python environment"
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML is required; install it with 'python -m pip install pyyaml'"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_API_DIR = SCRIPT_DIR.parent
CARLA_AGENTS_DIR = PYTHON_API_DIR / "carla"
if str(CARLA_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_AGENTS_DIR))

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    GlobalRoutePlanner = None

import traffic_light_pole_camera_ui_client_v1 as pole_camera_ui


LOG = logging.getLogger("physical_ai_scenario_controller")
DEFAULT_CONFIG = SCRIPT_DIR / "physical_ai_scenario_config_v1.yaml"
DEFAULT_TRAFFIC_LIGHT_DATA = SCRIPT_DIR / "traffic_lights_data.json"

PANEL_WIDTH = 390
BOTTOM_HEIGHT = 205
COLOR_BG = (15, 19, 25)
COLOR_PANEL = (25, 31, 40)
COLOR_FIELD = (38, 46, 58)
COLOR_FIELD_HOVER = (49, 60, 75)
COLOR_TEXT = (235, 240, 247)
COLOR_MUTED = (153, 166, 183)
COLOR_ACCENT = (42, 157, 244)
COLOR_GREEN = (64, 205, 130)
COLOR_ORANGE = (255, 166, 59)
COLOR_RED = (238, 85, 85)
COLOR_BORDER = (71, 84, 101)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def copy_location(location: carla.Location) -> carla.Location:
    return carla.Location(x=location.x, y=location.y, z=location.z)


def copy_transform(transform: carla.Transform) -> carla.Transform:
    return carla.Transform(
        copy_location(transform.location),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        ),
    )


def local_to_world(
    parent_transform: carla.Transform,
    local_location: carla.Location,
) -> carla.Location:
    matrix = np.asarray(parent_transform.get_matrix(), dtype=np.float64)
    point = np.asarray(
        [local_location.x, local_location.y, local_location.z, 1.0],
        dtype=np.float64,
    )
    world_point = matrix @ point
    return carla.Location(
        x=float(world_point[0]),
        y=float(world_point[1]),
        z=float(world_point[2]),
    )


def actor_alive(actor: Optional[carla.Actor]) -> bool:
    if actor is None:
        return False
    try:
        return bool(actor.is_alive)
    except RuntimeError:
        return False


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a YAML mapping")
    for section in ("scenario", "camera", "radar", "traffic_manager", "ui"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError("configuration is missing mapping '{}'".format(section))
    scenario = payload["scenario"]
    for section in ("ego_vehicle", "ego_pedestrian", "occluder", "ground_truth"):
        if section not in scenario or not isinstance(scenario[section], dict):
            raise ValueError("scenario is missing mapping '{}'".format(section))
    if str(scenario["occluder"]["type"]).lower() not in {"none", "bus", "truck"}:
        raise ValueError("occluder.type must be none, bus, or truck")
    resolution = payload["camera"].get("resolution", [])
    if len(resolution) != 2 or min(int(resolution[0]), int(resolution[1])) <= 0:
        raise ValueError("camera.resolution must contain positive width and height")
    return payload


class LatestMailbox:
    """A one-item thread-safe mailbox: a producer never blocks or queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def push(self, value: Any) -> None:
        with self._lock:
            self._value = value

    def pop(self) -> Any:
        with self._lock:
            value = self._value
            self._value = None
        return value

    def clear(self) -> None:
        with self._lock:
            self._value = None


class RadarMailbox:
    """Stores only compact statistics from the newest radar measurement."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summary = {"frame": -1, "points": 0, "nearest_m": None}

    def push(self, measurement: carla.RadarMeasurement) -> None:
        nearest = None
        count = 0
        for detection in measurement:
            count += 1
            depth = float(detection.depth)
            nearest = depth if nearest is None else min(nearest, depth)
        with self._lock:
            self._summary = {
                "frame": int(measurement.frame),
                "points": count,
                "nearest_m": nearest,
            }

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._summary)

    def clear(self) -> None:
        with self._lock:
            self._summary = {"frame": -1, "points": 0, "nearest_m": None}


class ExperimentLogger:
    """Small structured event log plus resolved config and manifest."""

    def __init__(self, root: Path, config: Dict[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        name = str(config["scenario"].get("name", "physical_ai_demo"))
        self.experiment_id = "{}_{}".format(name, stamp)
        self.directory = root / self.experiment_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events_path = self.directory / "events.jsonl"
        with (self.directory / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        manifest = {
            "experiment_id": self.experiment_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": ["resolved_config.yaml", "events.jsonl"],
            "replay_semantics": "rebuild_from_seed_and_resolved_config",
        }
        with (self.directory / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def event(
        self,
        name: str,
        frame_id: int = -1,
        actor_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "experiment_id": self.experiment_id,
            "frame_id": int(frame_id),
            "timestamp": time.time(),
            "anchor_id": None,
            "modality": None,
            "event": name,
            "actor_id": actor_id,
            "details": details or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class CameraView:
    key: str
    label: str
    kind: str
    actor: Optional[carla.Actor]
    mount: carla.Location
    fixed_location: Optional[carla.Location]
    yaw: float
    pitch: float
    default_yaw: float
    default_pitch: float

    def reset(self) -> None:
        self.yaw = self.default_yaw
        self.pitch = self.default_pitch


class CameraDirector:
    """One physical RGB sensor shared by every ego and pole viewpoint."""

    def __init__(
        self,
        world: carla.World,
        config: Dict[str, Any],
        vehicle: carla.Vehicle,
        pedestrian: carla.Walker,
        pole_views: Sequence[pole_camera_ui.PoleCamera],
    ) -> None:
        self.world = world
        self.config = config
        self.mailbox = LatestMailbox()
        self.views: List[CameraView] = []
        camera_cfg = config["camera"]
        vehicle_cfg = config["scenario"]["ego_vehicle"]
        pedestrian_cfg = config["scenario"]["ego_pedestrian"]
        self.views.append(
            CameraView(
                key="ego_vehicle",
                label="Ego vehicle camera",
                kind="vehicle",
                actor=vehicle,
                mount=carla.Location(
                    x=float(vehicle_cfg["camera_mount"][0]),
                    y=float(vehicle_cfg["camera_mount"][1]),
                    z=float(vehicle_cfg["camera_mount"][2]),
                ),
                fixed_location=None,
                yaw=float(camera_cfg["vehicle_default_yaw_deg"]),
                pitch=float(camera_cfg["vehicle_default_pitch_deg"]),
                default_yaw=float(camera_cfg["vehicle_default_yaw_deg"]),
                default_pitch=float(camera_cfg["vehicle_default_pitch_deg"]),
            )
        )
        self.views.append(
            CameraView(
                key="ego_pedestrian",
                label="Ego pedestrian camera",
                kind="pedestrian",
                actor=pedestrian,
                mount=carla.Location(
                    x=float(pedestrian_cfg["camera_mount"][0]),
                    y=float(pedestrian_cfg["camera_mount"][1]),
                    z=float(pedestrian_cfg["camera_mount"][2]),
                ),
                fixed_location=None,
                yaw=float(camera_cfg["pedestrian_default_yaw_deg"]),
                pitch=float(camera_cfg["pedestrian_default_pitch_deg"]),
                default_yaw=float(camera_cfg["pedestrian_default_yaw_deg"]),
                default_pitch=float(camera_cfg["pedestrian_default_pitch_deg"]),
            )
        )
        for pole in pole_views:
            self.views.append(
                CameraView(
                    key="pole_{}".format(pole.pole.display_id),
                    label=pole.label,
                    kind="pole",
                    actor=pole.pole.actor,
                    mount=carla.Location(),
                    fixed_location=copy_location(pole.location),
                    yaw=float(pole.yaw),
                    pitch=float(pole.pitch),
                    default_yaw=float(pole.default_yaw),
                    default_pitch=float(pole.default_pitch),
                )
            )
        self.selected_index = 0
        self.sensor: Optional[carla.Sensor] = None
        self.listening = False
        self.last_transform = self.transform_for(self.selected)
        self._spawn_sensor()

    @property
    def selected(self) -> CameraView:
        return self.views[self.selected_index]

    def transform_for(self, view: CameraView) -> carla.Transform:
        if view.kind == "pole":
            return carla.Transform(
                copy_location(view.fixed_location),
                carla.Rotation(pitch=view.pitch, yaw=view.yaw),
            )
        if not actor_alive(view.actor):
            return self.last_transform
        parent = view.actor.get_transform()
        location = local_to_world(parent, view.mount)
        return carla.Transform(
            location,
            carla.Rotation(
                pitch=parent.rotation.pitch + view.pitch,
                yaw=parent.rotation.yaw + view.yaw,
                roll=parent.rotation.roll,
            ),
        )

    def _spawn_sensor(self) -> None:
        camera_cfg = self.config["camera"]
        width, height = map(int, camera_cfg["resolution"])
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(float(camera_cfg["fov_deg"])))
        blueprint.set_attribute("sensor_tick", str(float(camera_cfg["sensor_tick_s"])))
        if blueprint.has_attribute("gamma"):
            blueprint.set_attribute("gamma", str(float(camera_cfg["gamma"])))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "physical_ai_shared_rgb")
        self.sensor = self.world.spawn_actor(blueprint, self.last_transform)
        self.sensor.listen(self.mailbox.push)
        self.listening = True
        LOG.info("Spawned shared RGB camera actor %d", self.sensor.id)

    def select(self, index: int) -> None:
        if not self.views or self.sensor is None:
            return
        index %= len(self.views)
        if index == self.selected_index:
            return
        if self.listening:
            self.sensor.stop()
            self.listening = False
        self.selected_index = index
        self.mailbox.clear()
        self.update_transform(force=True)
        self.sensor.listen(self.mailbox.push)
        self.listening = True
        LOG.info("Selected shared camera view: %s", self.selected.label)

    def cycle(self, offset: int) -> None:
        self.select(self.selected_index + int(offset))

    def set_orientation(self, yaw: float, pitch: float) -> None:
        self.selected.yaw = clamp(yaw, -180.0, 180.0)
        self.selected.pitch = clamp(pitch, -90.0, 45.0)
        self.update_transform(force=True)

    def reset_orientation(self) -> None:
        self.selected.reset()
        self.update_transform(force=True)

    def update_transform(self, force: bool = False) -> None:
        if self.sensor is None or not actor_alive(self.sensor):
            return
        transform = self.transform_for(self.selected)
        self.last_transform = transform
        # Ego views move every frame; fixed pole views only move when changed.
        if force or self.selected.kind != "pole":
            self.sensor.set_transform(transform)

    def destroy(self) -> None:
        if self.sensor is None:
            return
        try:
            if self.listening:
                self.sensor.stop()
                self.listening = False
        except RuntimeError:
            pass
        self.mailbox.clear()
        try:
            if actor_alive(self.sensor):
                self.sensor.destroy()
        except RuntimeError:
            pass
        self.sensor = None


class ActorProjectionCache:
    """Refreshes overlay candidates at 2 Hz rather than doing an RPC per frame."""

    def __init__(self, world: carla.World) -> None:
        self.world = world
        self.actors: List[carla.Actor] = []
        self.next_refresh = 0.0

    def get(self) -> List[carla.Actor]:
        now = time.monotonic()
        if now >= self.next_refresh:
            actors = self.world.get_actors()
            self.actors = list(actors.filter("vehicle.*")) + list(
                actors.filter("walker.pedestrian.*")
            )
            self.next_refresh = now + 0.5
        return self.actors


def camera_calibration(width: int, height: int, fov_degrees: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    matrix = np.identity(3, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def project_actor_box(
    actor: carla.Actor,
    camera_transform: carla.Transform,
    calibration: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[pygame.Rect, float]]:
    try:
        vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    except RuntimeError:
        return None
    points = np.asarray([[p.x, p.y, p.z, 1.0] for p in vertices], dtype=np.float64)
    inverse = np.asarray(camera_transform.get_inverse_matrix(), dtype=np.float64)
    camera_points = (inverse @ points.T).T[:, :3]
    depths = camera_points[:, 0]
    visible = depths > 0.10
    if np.count_nonzero(visible) < 2:
        return None
    camera_points = camera_points[visible]
    depths = camera_points[:, 0]
    u = calibration[0, 2] + camera_points[:, 1] / depths * calibration[0, 0]
    v = calibration[1, 2] - camera_points[:, 2] / depths * calibration[1, 1]
    x1 = int(clamp(float(np.min(u)), 0, width - 1))
    x2 = int(clamp(float(np.max(u)), 0, width - 1))
    y1 = int(clamp(float(np.min(v)), 0, height - 1))
    y2 = int(clamp(float(np.max(v)), 0, height - 1))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return pygame.Rect(x1, y1, x2 - x1, y2 - y1), float(np.min(depths))


def intersection_fraction(subject: pygame.Rect, foreground: pygame.Rect) -> float:
    intersection = subject.clip(foreground)
    if subject.width <= 0 or subject.height <= 0:
        return 0.0
    return intersection.width * intersection.height / float(subject.width * subject.height)


def draw_ground_truth_boxes(
    surface: pygame.Surface,
    actors: Iterable[carla.Actor],
    camera_transform: carla.Transform,
    camera_width: int,
    camera_height: int,
    fov: float,
    max_distance: float,
    overlap_threshold: float,
    excluded_ids: Sequence[int],
    occluder_id: Optional[int],
    font: pygame.font.Font,
) -> None:
    calibration = camera_calibration(camera_width, camera_height, fov)
    projected = []
    excluded = set(int(item) for item in excluded_ids)
    for actor in actors:
        if int(actor.id) in excluded or not actor_alive(actor):
            continue
        result = project_actor_box(
            actor, camera_transform, calibration, camera_width, camera_height
        )
        if result is None:
            continue
        rect, depth = result
        if depth > max_distance:
            continue
        projected.append((depth, rect, actor))
    projected.sort(key=lambda item: item[0])

    foreground: List[Tuple[pygame.Rect, carla.Actor]] = []
    for depth, rect, actor in projected:
        nlos = any(
            intersection_fraction(rect, nearer_rect) >= overlap_threshold
            for nearer_rect, _ in foreground
        )
        color = COLOR_ORANGE if nlos else COLOR_GREEN
        kind = "PED" if actor.type_id.startswith("walker.") else "VEH"
        status = "NLOS*" if nlos else "LOS*"
        if occluder_id is not None and int(actor.id) == int(occluder_id):
            kind = "OCC"
        pygame.draw.rect(surface, color, rect, 2)
        label = "{} {} id={} {:.1f}m".format(status, kind, actor.id, depth)
        text_surface = font.render(label, True, COLOR_TEXT, COLOR_BG)
        surface.blit(text_surface, (rect.left, max(0, rect.top - text_surface.get_height())))
        foreground.append((rect, actor))


def image_to_surface(image: carla.Image) -> pygame.Surface:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    rgb = array[:, :, ::-1]
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


class ScenarioController:
    """Owns all actors created for one deterministic scenario run."""

    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        traffic_light_data: Path,
    ) -> None:
        self.client = client
        self.world = world
        self.map = world.get_map()
        self.traffic_light_data = traffic_light_data
        self.traffic_manager = None
        self.camera: Optional[CameraDirector] = None
        self.radar: Optional[carla.Sensor] = None
        self.radar_mailbox = RadarMailbox()
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.ego_pedestrian: Optional[carla.Walker] = None
        self.ego_walker_controller: Optional[carla.Actor] = None
        self.occluder: Optional[carla.Vehicle] = None
        self.npc_vehicles: List[carla.Vehicle] = []
        self.npc_walkers: List[carla.Walker] = []
        self.walker_controllers: List[carla.Actor] = []
        self.last_config: Optional[Dict[str, Any]] = None
        self.logger: Optional[ExperimentLogger] = None
        self.running = False
        self.status = "Ready: configure the scenario and press Start"
        self.server_frame = -1
        self.server_elapsed = 0.0
        self._tick_callback_id = self.world.on_tick(self._on_world_tick)
        self._projection_cache = ActorProjectionCache(world)
        self._route_planner = None
        self._vehicle_route: List[carla.Location] = []
        self._pedestrian_body_yaw = 0.0
        self._vehicle_steer = 0.0
        self.manual_target = "vehicle"
        self.gt_enabled = True

    def _on_world_tick(self, snapshot: carla.WorldSnapshot) -> None:
        self.server_frame = int(snapshot.frame)
        self.server_elapsed = float(snapshot.timestamp.elapsed_seconds)

    def _log_event(
        self, name: str, actor: Optional[carla.Actor] = None, **details: Any
    ) -> None:
        if self.logger is not None:
            self.logger.event(
                name,
                frame_id=self.server_frame,
                actor_id=None if actor is None else int(actor.id),
                details=details,
            )

    def _get_route_planner(self, resolution: float):
        if GlobalRoutePlanner is None:
            return None
        if self._route_planner is None:
            self._route_planner = GlobalRoutePlanner(self.map, float(resolution))
        return self._route_planner

    def _build_route(
        self,
        start: carla.Location,
        end: carla.Location,
        resolution: float,
    ) -> List[carla.Location]:
        planner = self._get_route_planner(resolution)
        if planner is None:
            LOG.warning("GlobalRoutePlanner unavailable; using direct route endpoints")
            return [copy_location(start), copy_location(end)]
        try:
            trace = planner.trace_route(start, end)
        except Exception as exc:
            LOG.warning("Route planner failed (%s); using direct route endpoints", exc)
            return [copy_location(start), copy_location(end)]
        locations = [copy_location(waypoint.transform.location) for waypoint, _ in trace]
        return locations or [copy_location(start), copy_location(end)]

    @staticmethod
    def _downsample_route(
        route: Sequence[carla.Location], spacing: float = 3.0
    ) -> List[carla.Location]:
        if not route:
            return []
        result = [copy_location(route[0])]
        for location in route[1:]:
            if result[-1].distance(location) >= spacing:
                result.append(copy_location(location))
        if result[-1].distance(route[-1]) > 1.0:
            result.append(copy_location(route[-1]))
        return result

    def _activate_scripted_ego_vehicle(
        self,
        route_path: Sequence[carla.Location],
        traffic_manager_port: int,
    ) -> None:
        """Use Traffic Manager for the v1 ego scripted-route implementation."""
        self.ego_vehicle.set_autopilot(True, int(traffic_manager_port))
        try:
            self.traffic_manager.auto_lane_change(self.ego_vehicle, False)
        except Exception:
            pass
        self.traffic_manager.set_path(self.ego_vehicle, list(route_path))

    @staticmethod
    def _set_blueprint_attribute(
        blueprint: carla.ActorBlueprint, name: str, value: str
    ) -> None:
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, value)

    def _vehicle_blueprints(self, blueprint_filter: str) -> List[carla.ActorBlueprint]:
        blueprints = sorted(
            self.world.get_blueprint_library().filter(blueprint_filter),
            key=lambda item: item.id,
        )
        four_wheel = []
        for blueprint in blueprints:
            if blueprint.has_attribute("number_of_wheels"):
                if int(blueprint.get_attribute("number_of_wheels")) != 4:
                    continue
            four_wheel.append(blueprint)
        if not four_wheel:
            raise RuntimeError("no four-wheel vehicle blueprints matched")
        return four_wheel

    def _walker_blueprints(self, blueprint_filter: str) -> List[carla.ActorBlueprint]:
        blueprints = sorted(
            self.world.get_blueprint_library().filter(blueprint_filter),
            key=lambda item: item.id,
        )
        if not blueprints:
            raise RuntimeError("no pedestrian blueprints matched")
        return blueprints

    def _prepare_blueprint(
        self,
        blueprint: carla.ActorBlueprint,
        rng: random.Random,
        role_name: str,
    ) -> carla.ActorBlueprint:
        self._set_blueprint_attribute(blueprint, "role_name", role_name)
        self._set_blueprint_attribute(blueprint, "is_invincible", "false")
        if blueprint.has_attribute("color"):
            values = sorted(blueprint.get_attribute("color").recommended_values)
            if values:
                blueprint.set_attribute("color", rng.choice(values))
        if blueprint.has_attribute("driver_id"):
            values = sorted(blueprint.get_attribute("driver_id").recommended_values)
            if values:
                blueprint.set_attribute("driver_id", rng.choice(values))
        return blueprint

    def _try_spawn_vehicle_at_index(
        self,
        blueprint: carla.ActorBlueprint,
        spawn_points: Sequence[carla.Transform],
        requested_index: int,
    ) -> Tuple[carla.Vehicle, int]:
        for offset in range(len(spawn_points)):
            index = (requested_index + offset) % len(spawn_points)
            actor = self.world.try_spawn_actor(blueprint, spawn_points[index])
            if actor is not None:
                if offset:
                    LOG.warning(
                        "Spawn index %d occupied; used deterministic fallback %d",
                        requested_index % len(spawn_points),
                        index,
                    )
                return actor, index
        raise RuntimeError("no free vehicle spawn point is available")

    def _navigation_locations(self, seed: int, count: int = 128) -> List[carla.Location]:
        try:
            self.world.set_pedestrians_seed(int(seed))
        except (AttributeError, RuntimeError):
            LOG.warning("CARLA build does not expose world.set_pedestrians_seed")
        locations: List[carla.Location] = []
        attempts = 0
        while len(locations) < count and attempts < count * 12:
            attempts += 1
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            if any(location.distance(existing) < 1.0 for existing in locations):
                continue
            locations.append(copy_location(location))
        if len(locations) < 2:
            raise RuntimeError("CARLA navigation mesh returned fewer than two points")
        return locations

    def _spawn_walker(
        self,
        blueprint: carla.ActorBlueprint,
        location: carla.Location,
        rng: random.Random,
        role_name: str,
    ) -> carla.Walker:
        self._prepare_blueprint(blueprint, rng, role_name)
        transform = carla.Transform(
            carla.Location(x=location.x, y=location.y, z=location.z + 0.5)
        )
        walker = self.world.try_spawn_actor(blueprint, transform)
        if walker is None:
            raise RuntimeError("pedestrian spawn location is occupied")
        return walker

    def _try_spawn_walker_at_index(
        self,
        blueprint: carla.ActorBlueprint,
        navigation: Sequence[carla.Location],
        requested_index: int,
        rng: random.Random,
        role_name: str,
    ) -> Tuple[carla.Walker, int]:
        """Try the requested deterministic nav point, then stable fallbacks."""
        for offset in range(len(navigation)):
            index = (requested_index + offset) % len(navigation)
            try:
                walker = self._spawn_walker(
                    blueprint,
                    navigation[index],
                    rng,
                    role_name,
                )
            except RuntimeError as exc:
                if str(exc) != "pedestrian spawn location is occupied":
                    raise
                continue
            if offset:
                LOG.warning(
                    "Pedestrian nav index %d occupied; used deterministic fallback %d",
                    requested_index % len(navigation),
                    index,
                )
            return walker, index
        raise RuntimeError("no free pedestrian navigation point is available")

    def _spawn_walker_controller(
        self,
        walker: carla.Walker,
        destination: carla.Location,
        speed: float,
    ) -> carla.Actor:
        blueprint = self.world.get_blueprint_library().find("controller.ai.walker")
        controller = self.world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=walker,
        )
        controller.start()
        controller.go_to_location(destination)
        controller.set_max_speed(float(speed))
        return controller

    def _select_occluder_blueprint(self, kind: str) -> carla.ActorBlueprint:
        library = self.world.get_blueprint_library()
        preferences = {
            "bus": [
                "vehicle.mitsubishi.fusorosa",
                "vehicle.mercedes.sprinter",
                "vehicle.volkswagen.t2_2021",
            ],
            "truck": [
                "vehicle.carlamotors.european_hgv",
                "vehicle.carlamotors.firetruck",
                "vehicle.carlamotors.carlacola",
                "vehicle.tesla.cybertruck",
            ],
        }
        available = {blueprint.id: blueprint for blueprint in library.filter("vehicle.*")}
        for blueprint_id in preferences[kind]:
            if blueprint_id in available:
                return available[blueprint_id]
        tokens = ("bus", "coach") if kind == "bus" else ("truck", "hgv", "firetruck")
        matches = sorted(
            [bp for bp in available.values() if any(token in bp.id for token in tokens)],
            key=lambda item: item.id,
        )
        if matches:
            return matches[0]
        raise RuntimeError("no suitable {} blueprint is installed".format(kind))

    def _spawn_occluder(
        self,
        config: Dict[str, Any],
        rng: random.Random,
    ) -> Optional[carla.Vehicle]:
        occluder_cfg = config["scenario"]["occluder"]
        kind = str(occluder_cfg["type"]).lower()
        if kind == "none":
            return None
        fraction = clamp(float(occluder_cfg["route_fraction"]), 0.0, 1.0)
        route_index = min(len(self._vehicle_route) - 1, round(fraction * (len(self._vehicle_route) - 1)))
        route_location = self._vehicle_route[route_index]
        waypoint = self.map.get_waypoint(
            route_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        base = copy_transform(waypoint.transform) if waypoint else carla.Transform(route_location)
        right = base.rotation.get_right_vector()
        lateral = float(occluder_cfg["lateral_offset_m"])
        base.location.x += right.x * lateral
        base.location.y += right.y * lateral
        base.location.z += 0.25
        blueprint = self._select_occluder_blueprint(kind)
        self._prepare_blueprint(blueprint, rng, "physical_ai_occluder")
        # Deterministic nearby attempts preserve the chosen route fraction.
        forward = base.rotation.get_forward_vector()
        for delta in (0.0, 4.0, -4.0, 8.0, -8.0):
            candidate = copy_transform(base)
            candidate.location.x += forward.x * delta
            candidate.location.y += forward.y * delta
            actor = self.world.try_spawn_actor(blueprint, candidate)
            if actor is None:
                continue
            actor.apply_control(carla.VehicleControl(hand_brake=True))
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                pass
            self._log_event(
                "occluder_spawned",
                actor,
                type=kind,
                route_fraction=fraction,
                lateral_offset_m=lateral,
            )
            return actor
        raise RuntimeError("unable to place {} occluder near route".format(kind))

    def _create_camera_director(
        self,
        config: Dict[str, Any],
        pole_views: Sequence[pole_camera_ui.PoleCamera],
    ) -> CameraDirector:
        """Factory hook so a subclass can install a different camera layout."""
        return CameraDirector(
            self.world,
            config,
            self.ego_vehicle,
            self.ego_pedestrian,
            pole_views,
        )

    def _build_pole_views(self, config: Dict[str, Any]) -> List[pole_camera_ui.PoleCamera]:
        records = pole_camera_ui.load_traffic_light_records(self.traffic_light_data)
        poles = pole_camera_ui.resolve_live_poles(self.world, records, 4.0)
        camera_cfg = config["camera"]
        namespace = argparse.Namespace(
            camera_x=0.0,
            camera_y=0.0,
            camera_height=float(camera_cfg["pole_height_m"]),
            initial_yaw_offset=float(camera_cfg["pole_yaw_offset_deg"]),
            initial_pitch=float(camera_cfg["pole_pitch_deg"]),
        )
        return pole_camera_ui.build_pole_camera_views(poles, namespace)

    def _spawn_radar(self, config: Dict[str, Any]) -> None:
        radar_cfg = config["radar"]
        if not bool(radar_cfg.get("enabled", True)):
            return
        blueprint = self.world.get_blueprint_library().find("sensor.other.radar")
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "physical_ai_ego_radar")
        attributes = {
            "range": radar_cfg["range_m"],
            "horizontal_fov": radar_cfg["horizontal_fov_deg"],
            "vertical_fov": radar_cfg["vertical_fov_deg"],
            "points_per_second": radar_cfg["points_per_second"],
            "sensor_tick": radar_cfg["sensor_tick_s"],
        }
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(name, str(value))
        transform = carla.Transform(carla.Location(x=1.5, z=1.0))
        self.radar = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.radar.listen(self.radar_mailbox.push)
        self._log_event("radar_spawned", self.radar, modality="radar")

    def start(self, config: Dict[str, Any]) -> None:
        self.cleanup(remove_tick_callback=False)
        frozen = copy.deepcopy(config)
        self.last_config = frozen
        scenario = frozen["scenario"]
        seed = int(scenario["seed"])
        rng = random.Random(seed)
        output_root = Path(frozen.get("logging", {}).get("output_root", "experiments"))
        if not output_root.is_absolute():
            output_root = SCRIPT_DIR / output_root
        self.logger = ExperimentLogger(output_root, frozen)
        self.gt_enabled = bool(scenario["ground_truth"]["enabled"])
        self.status = "Building deterministic scenario..."
        self._log_event("scenario_build_started", seed=seed)
        try:
            tm_cfg = frozen["traffic_manager"]
            self.traffic_manager = self.client.get_trafficmanager(int(tm_cfg["port"]))
            # This seeds CARLA Traffic Manager but never changes its clock mode.
            self.traffic_manager.set_random_device_seed(seed)

            spawn_points = list(self.map.get_spawn_points())
            if len(spawn_points) < 2:
                raise RuntimeError("loaded map exposes fewer than two vehicle spawn points")
            vehicle_cfg = scenario["ego_vehicle"]
            vehicle_blueprints = self._vehicle_blueprints(vehicle_cfg["blueprint_filter"])
            requested_ego_blueprint_id = vehicle_cfg.get("blueprint_id")
            if requested_ego_blueprint_id:
                requested_ego_blueprint_id = str(requested_ego_blueprint_id)
                ego_candidates = self._vehicle_blueprints(requested_ego_blueprint_id)
                ego_source_blueprint = next(
                    (
                        blueprint
                        for blueprint in ego_candidates
                        if str(blueprint.id) == requested_ego_blueprint_id
                    ),
                    None,
                )
                if ego_source_blueprint is None:
                    raise RuntimeError(
                        "requested ego vehicle blueprint is unavailable: {}".format(
                            requested_ego_blueprint_id
                        )
                    )
            else:
                ego_source_blueprint = rng.choice(vehicle_blueprints)
            ego_blueprint = self._prepare_blueprint(
                ego_source_blueprint, rng, "physical_ai_ego_vehicle"
            )
            requested_start = int(vehicle_cfg["start_spawn_index"]) % len(spawn_points)
            self.ego_vehicle, actual_start = self._try_spawn_vehicle_at_index(
                ego_blueprint, spawn_points, requested_start
            )
            end_index = int(vehicle_cfg["end_spawn_index"]) % len(spawn_points)
            self._vehicle_route = self._build_route(
                self.ego_vehicle.get_location(),
                spawn_points[end_index].location,
                float(vehicle_cfg["route_sampling_resolution_m"]),
            )
            route_path = self._downsample_route(self._vehicle_route)
            if bool(vehicle_cfg["scripted_route"]):
                self._activate_scripted_ego_vehicle(
                    route_path,
                    int(tm_cfg["port"]),
                )
            self._log_event(
                "ego_vehicle_spawned",
                self.ego_vehicle,
                requested_spawn_index=requested_start,
                actual_spawn_index=actual_start,
                end_spawn_index=end_index,
                blueprint_id=str(self.ego_vehicle.type_id),
                scripted=bool(vehicle_cfg["scripted_route"]),
                route_points=len(route_path),
            )

            navigation = self._navigation_locations(seed)
            pedestrian_cfg = scenario["ego_pedestrian"]
            walker_blueprints = self._walker_blueprints(
                pedestrian_cfg["blueprint_filter"]
            )
            ped_start = int(pedestrian_cfg["start_spawn_index"]) % len(navigation)
            ped_end = int(pedestrian_cfg["end_spawn_index"]) % len(navigation)
            if ped_end == ped_start:
                ped_end = (ped_start + 1) % len(navigation)
            ego_walker_blueprint = rng.choice(walker_blueprints)
            self.ego_pedestrian, actual_ped_start = self._try_spawn_walker_at_index(
                ego_walker_blueprint,
                navigation,
                ped_start,
                rng,
                "physical_ai_ego_pedestrian",
            )
            if ped_end == actual_ped_start:
                ped_end = (actual_ped_start + 1) % len(navigation)
            self._pedestrian_body_yaw = self.ego_pedestrian.get_transform().rotation.yaw
            if bool(pedestrian_cfg["scripted_route"]):
                self.ego_walker_controller = self._spawn_walker_controller(
                    self.ego_pedestrian,
                    navigation[ped_end],
                    float(pedestrian_cfg["scripted_speed_mps"]),
                )
                self.walker_controllers.append(self.ego_walker_controller)
            self._log_event(
                "ego_pedestrian_spawned",
                self.ego_pedestrian,
                start_navigation_index=actual_ped_start,
                requested_start_navigation_index=ped_start,
                actual_start_navigation_index=actual_ped_start,
                end_navigation_index=ped_end,
                scripted=bool(pedestrian_cfg["scripted_route"]),
            )

            # Reserve both ego actors before placing any occluder or NPC. This
            # prevents scenario-owned traffic from consuming an ego spawn.
            self.occluder = self._spawn_occluder(frozen, rng)

            reserved = {actual_start}
            candidate_indices = [
                i for i in range(len(spawn_points)) if i not in reserved
            ]
            rng.shuffle(candidate_indices)
            desired_vehicles = max(0, int(scenario["npc_vehicles"]))
            for index in candidate_indices:
                if len(self.npc_vehicles) >= desired_vehicles:
                    break
                blueprint = self._prepare_blueprint(
                    rng.choice(vehicle_blueprints), rng, "physical_ai_npc_vehicle"
                )
                actor = self.world.try_spawn_actor(blueprint, spawn_points[index])
                if actor is None:
                    continue
                actor.set_autopilot(True, int(tm_cfg["port"]))
                self.npc_vehicles.append(actor)

            desired_walkers = max(0, int(scenario["npc_pedestrians"]))
            available_nav = [
                i
                for i in range(len(navigation))
                if i not in {actual_ped_start, ped_end}
            ]
            rng.shuffle(available_nav)
            for index in available_nav:
                if len(self.npc_walkers) >= desired_walkers:
                    break
                walker = None
                try:
                    walker = self._spawn_walker(
                        rng.choice(walker_blueprints),
                        navigation[index],
                        rng,
                        "physical_ai_npc_pedestrian",
                    )
                    destination = navigation[rng.randrange(len(navigation))]
                    controller = self._spawn_walker_controller(
                        walker, destination, rng.uniform(1.2, 2.0)
                    )
                except RuntimeError:
                    if actor_alive(walker):
                        walker.destroy()
                    continue
                self.npc_walkers.append(walker)
                self.walker_controllers.append(controller)

            pole_views = self._build_pole_views(frozen)
            self.camera = self._create_camera_director(frozen, pole_views)
            self._spawn_radar(frozen)
            self.running = True
            self.status = (
                "Running: {} vehicles, {} pedestrians, {} shared RGB sensor"
            ).format(len(self.npc_vehicles), len(self.npc_walkers), 1)
            self._log_event(
                "scenario_started",
                npc_vehicles=len(self.npc_vehicles),
                npc_pedestrians=len(self.npc_walkers),
                camera_views=len(self.camera.views),
                shared_rgb_sensors=1,
            )
        except Exception:
            self._log_event("scenario_build_failed")
            self.cleanup(remove_tick_callback=False)
            raise

    def replay(self) -> None:
        if self.last_config is None:
            raise RuntimeError("no launched scenario is available to replay")
        frozen = copy.deepcopy(self.last_config)
        self.start(frozen)
        self.status = "Replay rebuilt from the last launched seed/config"

    def set_ground_truth(self, enabled: bool) -> None:
        self.gt_enabled = bool(enabled)
        self._log_event("ground_truth_toggled", enabled=self.gt_enabled)

    def update_controls(self, keys: Sequence[bool], delta_seconds: float) -> None:
        if not self.running or self.last_config is None:
            return
        delta_seconds = clamp(delta_seconds, 0.0, 0.1)
        scenario = self.last_config["scenario"]
        if self.manual_target == "vehicle":
            if bool(scenario["ego_vehicle"]["scripted_route"]):
                return
            if not actor_alive(self.ego_vehicle):
                return
            steer_axis = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
            target_steer = 0.65 * steer_axis
            self._vehicle_steer += clamp(
                target_steer - self._vehicle_steer,
                -2.8 * delta_seconds,
                2.8 * delta_seconds,
            )
            forward = bool(keys[pygame.K_w])
            backward = bool(keys[pygame.K_s])
            control = carla.VehicleControl(
                throttle=0.65 if forward or backward else 0.0,
                steer=float(self._vehicle_steer),
                # Hold the vehicle when manual control is selected but no
                # longitudinal key is pressed; otherwise a mode switch leaves
                # the Lincoln coasting toward the roadside.
                brake=0.0 if forward or backward else 0.35,
                hand_brake=bool(keys[pygame.K_SPACE]),
                reverse=backward and not forward,
            )
            self.ego_vehicle.apply_control(control)
        else:
            if bool(scenario["ego_pedestrian"]["scripted_route"]):
                return
            if not actor_alive(self.ego_pedestrian):
                return
            turn_axis = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
            self._pedestrian_body_yaw = (
                self._pedestrian_body_yaw + turn_axis * 90.0 * delta_seconds
            ) % 360.0
            move_axis = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
            pedestrian_cfg = scenario["ego_pedestrian"]
            running = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            speed = float(
                pedestrian_cfg["run_speed_mps"]
                if running
                else pedestrian_cfg["walk_speed_mps"]
            )
            direction = carla.Rotation(yaw=self._pedestrian_body_yaw).get_forward_vector()
            if move_axis < 0:
                direction = carla.Vector3D(x=-direction.x, y=-direction.y, z=0.0)
            control = carla.WalkerControl()
            control.direction = direction
            control.speed = speed if move_axis else (0.01 if turn_axis else 0.0)
            control.jump = bool(keys[pygame.K_SPACE])
            self.ego_pedestrian.apply_control(control)

    def update_camera_keys(self, keys: Sequence[bool], delta_seconds: float) -> None:
        if self.camera is None:
            return
        yaw_axis = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        pitch_axis = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        if yaw_axis or pitch_axis:
            self.camera.set_orientation(
                self.camera.selected.yaw + yaw_axis * 75.0 * delta_seconds,
                self.camera.selected.pitch + pitch_axis * 60.0 * delta_seconds,
            )
        self.camera.update_transform()

    def cleanup(self, remove_tick_callback: bool = False) -> None:
        was_running = self.running
        self.running = False
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.radar is not None:
            try:
                self.radar.stop()
            except RuntimeError:
                pass
            try:
                if actor_alive(self.radar):
                    self.radar.destroy()
            except RuntimeError:
                pass
            self.radar = None
        self.radar_mailbox.clear()
        for controller in self.walker_controllers:
            try:
                if actor_alive(controller):
                    controller.stop()
            except RuntimeError:
                pass
        for actor in reversed(
            self.walker_controllers
            + self.npc_walkers
            + self.npc_vehicles
            + [self.ego_pedestrian, self.occluder, self.ego_vehicle]
        ):
            if not actor_alive(actor):
                continue
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self.walker_controllers.clear()
        self.npc_walkers.clear()
        self.npc_vehicles.clear()
        self.ego_walker_controller = None
        self.ego_pedestrian = None
        self.occluder = None
        self.ego_vehicle = None
        self._vehicle_route.clear()
        if was_running:
            self._log_event("scenario_stopped")
        if remove_tick_callback and self._tick_callback_id is not None:
            try:
                self.world.remove_on_tick(self._tick_callback_id)
            except RuntimeError:
                pass
            self._tick_callback_id = None
        self.status = "Scenario reset; owned actors removed"


class Button:
    def __init__(self, label: str, rect: pygame.Rect, action) -> None:
        self.label = label
        self.rect = rect
        self.action = action
        self.enabled = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if (
            self.enabled
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.action()
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        color = COLOR_FIELD_HOVER if hovered and self.enabled else COLOR_FIELD
        if not self.enabled:
            color = (31, 36, 44)
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, 1, border_radius=5)
        text = font.render(self.label, True, COLOR_TEXT if self.enabled else COLOR_MUTED)
        screen.blit(text, text.get_rect(center=self.rect.center))


class Stepper:
    def __init__(
        self,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        y: int,
        integer: bool = True,
    ) -> None:
        self.label = label
        self.value = float(value)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.integer = integer
        self.minus = pygame.Rect(247, y, 36, 28)
        self.plus = pygame.Rect(337, y, 36, 28)
        self.value_rect = pygame.Rect(284, y, 52, 28)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        old = self.value
        if self.minus.collidepoint(event.pos):
            self.value = clamp(self.value - self.step, self.minimum, self.maximum)
        elif self.plus.collidepoint(event.pos):
            self.value = clamp(self.value + self.step, self.minimum, self.maximum)
        return old != self.value

    def get(self):
        return int(round(self.value)) if self.integer else float(self.value)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.blit(font.render(self.label, True, COLOR_TEXT), (18, self.minus.y + 5))
        for rect, label in ((self.minus, "-"), (self.plus, "+")):
            pygame.draw.rect(screen, COLOR_FIELD, rect, border_radius=4)
            pygame.draw.rect(screen, COLOR_BORDER, rect, 1, border_radius=4)
            glyph = font.render(label, True, COLOR_TEXT)
            screen.blit(glyph, glyph.get_rect(center=rect.center))
        pygame.draw.rect(screen, COLOR_BG, self.value_rect, border_radius=3)
        text_value = "{}".format(self.get()) if self.integer else "{:.2f}".format(self.value)
        text = font.render(text_value, True, COLOR_ACCENT)
        screen.blit(text, text.get_rect(center=self.value_rect.center))


class Toggle:
    def __init__(self, label: str, value: bool, y: int) -> None:
        self.label = label
        self.value = bool(value)
        self.rect = pygame.Rect(288, y, 85, 28)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.value = not self.value
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.blit(font.render(self.label, True, COLOR_TEXT), (18, self.rect.y + 5))
        pygame.draw.rect(
            screen, COLOR_GREEN if self.value else COLOR_FIELD, self.rect, border_radius=4
        )
        text = font.render("ON" if self.value else "OFF", True, COLOR_BG if self.value else COLOR_TEXT)
        screen.blit(text, text.get_rect(center=self.rect.center))


class CycleField:
    def __init__(self, label: str, value: str, choices: Sequence[str], y: int) -> None:
        self.label = label
        self.choices = list(choices)
        self.index = self.choices.index(str(value).lower())
        self.rect = pygame.Rect(247, y, 126, 28)

    @property
    def value(self) -> str:
        return self.choices[self.index]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.index = (self.index + 1) % len(self.choices)
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.blit(font.render(self.label, True, COLOR_TEXT), (18, self.rect.y + 5))
        pygame.draw.rect(screen, COLOR_FIELD, self.rect, border_radius=4)
        text = font.render(self.value.upper(), True, COLOR_ACCENT)
        screen.blit(text, text.get_rect(center=self.rect.center))


class Slider:
    def __init__(
        self,
        label: str,
        rect: pygame.Rect,
        minimum: float,
        maximum: float,
        value: float,
    ) -> None:
        self.label = label
        self.rect = rect
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.value = clamp(value, minimum, maximum)
        self.dragging = False

    def set(self, value: float) -> None:
        self.value = clamp(value, self.minimum, self.maximum)

    def handle_event(self, event: pygame.event.Event) -> bool:
        hit = self.rect.inflate(12, 28)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hit.collidepoint(event.pos):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            fraction = clamp((event.pos[0] - self.rect.left) / self.rect.width, 0.0, 1.0)
            new_value = self.minimum + fraction * (self.maximum - self.minimum)
            changed = not math.isclose(new_value, self.value)
            self.value = new_value
            return changed
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label = font.render("{}: {:+.1f} deg".format(self.label, self.value), True, COLOR_TEXT)
        screen.blit(label, (self.rect.left, self.rect.top - 28))
        pygame.draw.line(screen, COLOR_BORDER, self.rect.midleft, self.rect.midright, 6)
        fraction = (self.value - self.minimum) / (self.maximum - self.minimum)
        x = int(self.rect.left + fraction * self.rect.width)
        pygame.draw.line(screen, COLOR_ACCENT, self.rect.midleft, (x, self.rect.centery), 6)
        pygame.draw.circle(screen, COLOR_TEXT, (x, self.rect.centery), 9)


class ScenarioUI:
    def __init__(
        self,
        controller: ScenarioController,
        base_config: Dict[str, Any],
    ) -> None:
        self.controller = controller
        self.base_config = copy.deepcopy(base_config)
        window = base_config["ui"]["window_size"]
        self.width, self.height = int(window[0]), int(window[1])
        self.fps = int(base_config["ui"].get("fps", 60))
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CARLA Physical AI Scenario Controller v1")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 17)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        self.title_font = pygame.font.Font(pygame.font.get_default_font(), 23)
        self.clock = pygame.time.Clock()
        self.latest_surface: Optional[pygame.Surface] = None
        self.gt_overlay: Optional[pygame.Surface] = None
        self.next_gt_overlay_update = 0.0
        self.error_message = ""
        self._build_controls()

    def _build_controls(self) -> None:
        scenario = self.base_config["scenario"]
        self.seed = Stepper("Random seed", scenario["seed"], 0, 99999999, 1, 83)
        self.npc_vehicles = Stepper("NPC vehicles", scenario["npc_vehicles"], 0, 100, 1, 119)
        self.npc_pedestrians = Stepper("NPC pedestrians", scenario["npc_pedestrians"], 0, 200, 1, 155)
        vehicle = scenario["ego_vehicle"]
        self.vehicle_start = Stepper("Vehicle start index", vehicle["start_spawn_index"], 0, 999, 1, 209)
        self.vehicle_end = Stepper("Vehicle end index", vehicle["end_spawn_index"], 0, 999, 1, 245)
        self.vehicle_scripted = Toggle("Vehicle scripted route", vehicle["scripted_route"], 281)
        pedestrian = scenario["ego_pedestrian"]
        self.ped_start = Stepper("Pedestrian start index", pedestrian["start_spawn_index"], 0, 127, 1, 335)
        self.ped_end = Stepper("Pedestrian end index", pedestrian["end_spawn_index"], 0, 127, 1, 371)
        self.ped_scripted = Toggle("Pedestrian scripted route", pedestrian["scripted_route"], 407)
        occluder = scenario["occluder"]
        self.occluder = CycleField("Occluder type", occluder["type"], ["none", "bus", "truck"], 461)
        self.occluder_fraction = Stepper("Route fraction", occluder["route_fraction"], 0.0, 1.0, 0.05, 497, False)
        self.occluder_lateral = Stepper("Lateral offset (m)", occluder["lateral_offset_m"], -10.0, 10.0, 0.5, 533, False)
        self.ground_truth = Toggle("GT boxes + LOS estimate", scenario["ground_truth"]["enabled"], 580)
        self.config_controls = [
            self.seed, self.npc_vehicles, self.npc_pedestrians,
            self.vehicle_start, self.vehicle_end, self.vehicle_scripted,
            self.ped_start, self.ped_end, self.ped_scripted,
            self.occluder, self.occluder_fraction, self.occluder_lateral,
            self.ground_truth,
        ]
        self.start_button = Button("Start current config", pygame.Rect(18, 625, 355, 38), self._start)
        self.reset_button = Button("Reset", pygame.Rect(18, 673, 170, 38), self._reset)
        self.replay_button = Button("Replay last launch", pygame.Rect(203, 673, 170, 38), self._replay)
        self.target_button = Button("Control target: VEHICLE", pygame.Rect(18, 721, 355, 36), self._toggle_target)
        bottom_y = self.height - BOTTOM_HEIGHT
        self.prev_view = Button("< Previous view", pygame.Rect(PANEL_WIDTH + 25, bottom_y + 32, 145, 34), lambda: self._cycle_view(-1))
        self.next_view = Button("Next view >", pygame.Rect(self.width - 170, bottom_y + 32, 145, 34), lambda: self._cycle_view(1))
        self.reset_view = Button("Reset view", pygame.Rect(self.width - 170, bottom_y + 78, 145, 32), self._reset_view)
        slider_left = PANEL_WIDTH + 210
        slider_width = max(240, self.width - slider_left - 210)
        self.yaw_slider = Slider("Camera yaw", pygame.Rect(slider_left, bottom_y + 103, slider_width, 10), -180.0, 180.0, 0.0)
        self.pitch_slider = Slider("Camera pitch", pygame.Rect(slider_left, bottom_y + 160, slider_width, 10), -90.0, 45.0, 0.0)
        self.buttons = [self.start_button, self.reset_button, self.replay_button, self.target_button, self.prev_view, self.next_view, self.reset_view]

    def _current_config(self) -> Dict[str, Any]:
        config = copy.deepcopy(self.base_config)
        scenario = config["scenario"]
        scenario["seed"] = self.seed.get()
        scenario["npc_vehicles"] = self.npc_vehicles.get()
        scenario["npc_pedestrians"] = self.npc_pedestrians.get()
        scenario["ego_vehicle"]["start_spawn_index"] = self.vehicle_start.get()
        scenario["ego_vehicle"]["end_spawn_index"] = self.vehicle_end.get()
        scenario["ego_vehicle"]["scripted_route"] = self.vehicle_scripted.value
        scenario["ego_pedestrian"]["start_spawn_index"] = self.ped_start.get()
        scenario["ego_pedestrian"]["end_spawn_index"] = self.ped_end.get()
        scenario["ego_pedestrian"]["scripted_route"] = self.ped_scripted.value
        scenario["occluder"]["type"] = self.occluder.value
        scenario["occluder"]["route_fraction"] = self.occluder_fraction.get()
        scenario["occluder"]["lateral_offset_m"] = self.occluder_lateral.get()
        scenario["ground_truth"]["enabled"] = self.ground_truth.value
        return config

    def _guard(self, action) -> None:
        self.error_message = ""
        try:
            action()
        except Exception as exc:
            LOG.exception("UI action failed")
            self.error_message = str(exc)

    def _start(self) -> None:
        self._guard(lambda: self.controller.start(self._current_config()))
        self.ground_truth.value = self.controller.gt_enabled
        self.gt_overlay = None
        self._sync_camera_sliders()

    def _reset(self) -> None:
        self._guard(lambda: self.controller.cleanup(remove_tick_callback=False))
        self.latest_surface = None
        self.gt_overlay = None

    def _replay(self) -> None:
        self._guard(self.controller.replay)
        self.ground_truth.value = self.controller.gt_enabled
        self.gt_overlay = None
        self._sync_camera_sliders()

    def _toggle_target(self) -> None:
        self.controller.manual_target = (
            "pedestrian" if self.controller.manual_target == "vehicle" else "vehicle"
        )
        self.target_button.label = "Control target: {}".format(
            self.controller.manual_target.upper()
        )

    def _cycle_view(self, offset: int) -> None:
        if self.controller.camera is not None:
            self.controller.camera.cycle(offset)
            self.gt_overlay = None
            self._sync_camera_sliders()

    def _reset_view(self) -> None:
        if self.controller.camera is not None:
            self.controller.camera.reset_orientation()
            self._sync_camera_sliders()

    def _sync_camera_sliders(self) -> None:
        if self.controller.camera is None:
            return
        self.yaw_slider.set(self.controller.camera.selected.yaw)
        self.pitch_slider.set(self.controller.camera.selected.pitch)

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            if event.key == pygame.K_ESCAPE or (event.key == pygame.K_q and mods & pygame.KMOD_CTRL):
                return False
            if event.key == pygame.K_TAB:
                self._cycle_view(1)
            elif event.key == pygame.K_F1:
                self._toggle_target()
            elif event.key == pygame.K_b:
                self.ground_truth.value = not self.ground_truth.value
                self.controller.set_ground_truth(self.ground_truth.value)
            elif event.key == pygame.K_r:
                self._reset_view()
        for control in self.config_controls:
            changed = control.handle_event(event)
            if changed and control is self.ground_truth and self.controller.running:
                self.controller.set_ground_truth(self.ground_truth.value)
        for button in self.buttons:
            button.handle_event(event)
        changed = self.yaw_slider.handle_event(event)
        changed = self.pitch_slider.handle_event(event) or changed
        if changed and self.controller.camera is not None:
            self.controller.camera.set_orientation(
                self.yaw_slider.value, self.pitch_slider.value
            )
        return True

    def _draw_wrapped(
        self,
        text: str,
        x: int,
        y: int,
        width: int,
        color: Tuple[int, int, int],
        font: pygame.font.Font,
        max_lines: int = 3,
    ) -> int:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        for line in lines:
            self.screen.blit(font.render(line, True, color), (x, y))
            y += font.get_linesize()
        return y

    def _draw_panel(self) -> None:
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, 0, PANEL_WIDTH, self.height))
        self.screen.blit(self.title_font.render("Physical AI Scenario", True, COLOR_TEXT), (18, 15))
        self.screen.blit(self.small_font.render("Passive deterministic controller v1", True, COLOR_MUTED), (18, 45))
        self.screen.blit(self.small_font.render("ACTOR POPULATION", True, COLOR_ACCENT), (18, 66))
        self.screen.blit(self.small_font.render("EGO VEHICLE ROUTE", True, COLOR_ACCENT), (18, 191))
        self.screen.blit(self.small_font.render("EGO PEDESTRIAN ROUTE", True, COLOR_ACCENT), (18, 317))
        self.screen.blit(self.small_font.render("OCCLUSION", True, COLOR_ACCENT), (18, 443))
        for control in self.config_controls:
            control.draw(self.screen, self.font)
        for button in (self.start_button, self.reset_button, self.replay_button, self.target_button):
            button.draw(self.screen, self.font)
        mode = "running" if self.controller.running else "stopped"
        status_color = COLOR_GREEN if self.controller.running else COLOR_MUTED
        self.screen.blit(self.small_font.render("STATUS [{}]".format(mode), True, status_color), (18, 775))
        y = self._draw_wrapped(self.error_message or self.controller.status, 18, 798, 355, COLOR_RED if self.error_message else COLOR_TEXT, self.small_font, 3)
        self._draw_wrapped(
            "WASD motion | arrows camera | Shift run | F1 target | Tab view | B boxes",
            18, max(y + 5, 850), 355, COLOR_MUTED, self.small_font, 2,
        )

    def _draw_video(self) -> None:
        video_rect = pygame.Rect(
            PANEL_WIDTH,
            0,
            self.width - PANEL_WIDTH,
            self.height - BOTTOM_HEIGHT,
        )
        pygame.draw.rect(self.screen, COLOR_BG, video_rect)
        camera = self.controller.camera
        if camera is not None:
            image = camera.mailbox.pop()
            if image is not None:
                self.latest_surface = image_to_surface(image)
        if self.latest_surface is None:
            label = self.title_font.render(
                "Start a scenario to activate the shared RGB stream", True, COLOR_MUTED
            )
            self.screen.blit(label, label.get_rect(center=video_rect.center))
            return
        surface = self.latest_surface.copy()
        if camera is not None and self.controller.gt_enabled:
            ground_truth = self.controller.last_config["scenario"]["ground_truth"]
            now = time.monotonic()
            if self.gt_overlay is None or now >= self.next_gt_overlay_update:
                self.gt_overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                excluded_ids = [
                    actor.id
                    for actor in (self.controller.ego_vehicle, self.controller.ego_pedestrian)
                    if actor is not None
                ]
                draw_ground_truth_boxes(
                    self.gt_overlay,
                    self.controller._projection_cache.get(),
                    camera.last_transform,
                    surface.get_width(),
                    surface.get_height(),
                    float(self.controller.last_config["camera"]["fov_deg"]),
                    float(ground_truth["max_distance_m"]),
                    float(ground_truth["nlos_overlap_threshold"]),
                    excluded_ids,
                    None if self.controller.occluder is None else self.controller.occluder.id,
                    self.small_font,
                )
                self.next_gt_overlay_update = now + 0.10
            surface.blit(self.gt_overlay, (0, 0))
        else:
            self.gt_overlay = None
        scaled = pygame.transform.smoothscale(surface, video_rect.size)
        self.screen.blit(scaled, video_rect)
        overlay = pygame.Surface((video_rect.width, 38), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 155))
        self.screen.blit(overlay, video_rect.topleft)
        view_label = camera.selected.label if camera else "No camera"
        frame_label = "{} | master frame {} | GT {}".format(
            view_label,
            self.controller.server_frame,
            "ON" if self.controller.gt_enabled else "OFF",
        )
        self.screen.blit(self.font.render(frame_label, True, COLOR_TEXT), (video_rect.left + 12, 10))

    def _draw_bottom(self) -> None:
        top = self.height - BOTTOM_HEIGHT
        pygame.draw.rect(self.screen, COLOR_PANEL, (PANEL_WIDTH, top, self.width - PANEL_WIDTH, BOTTOM_HEIGHT))
        pygame.draw.line(self.screen, COLOR_BORDER, (PANEL_WIDTH, top), (self.width, top), 1)
        camera = self.controller.camera
        view_text = camera.selected.label if camera else "No active camera view"
        self.screen.blit(self.font.render("Selected: " + view_text, True, COLOR_TEXT), (PANEL_WIDTH + 190, top + 39))
        for button in (self.prev_view, self.next_view, self.reset_view):
            button.enabled = camera is not None
            button.draw(self.screen, self.font)
        self.yaw_slider.draw(self.screen, self.font)
        self.pitch_slider.draw(self.screen, self.font)
        radar = self.controller.radar_mailbox.get()
        nearest = "--" if radar["nearest_m"] is None else "{:.1f} m".format(radar["nearest_m"])
        radar_text = "Ego radar: {} pts | nearest {} | frame {}".format(
            radar["points"], nearest, radar["frame"]
        )
        self.screen.blit(self.small_font.render(radar_text, True, COLOR_MUTED), (PANEL_WIDTH + 25, top + 183))
        legend = "Green/orange = LOS*/NLOS* image-overlap estimate; boxes are CARLA ground truth"
        legend_surface = self.small_font.render(legend, True, COLOR_MUTED)
        self.screen.blit(legend_surface, (self.width - legend_surface.get_width() - 25, top + 183))

    def run(self) -> None:
        keep_running = True
        while keep_running:
            delta_seconds = self.clock.tick(self.fps) / 1000.0
            for event in pygame.event.get():
                if not self._handle_event(event):
                    keep_running = False
                    break
            keys = pygame.key.get_pressed()
            self.controller.update_controls(keys, delta_seconds)
            self.controller.update_camera_keys(keys, delta_seconds)
            self._sync_camera_sliders()
            self.screen.fill(COLOR_BG)
            self._draw_video()
            self._draw_panel()
            self._draw_bottom()
            pygame.display.flip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument("--timeout", type=float, default=10.0, help="RPC timeout seconds")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="scenario YAML")
    parser.add_argument(
        "--traffic-light-data",
        type=Path,
        default=DEFAULT_TRAFFIC_LIGHT_DATA,
        help="traffic-light metadata JSON",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_yaml_config(args.config.resolve())
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    LOG.info(
        "Connected passively to %s:%d; loaded map=%s. Master clock is untouched.",
        args.host,
        args.port,
        world.get_map().name,
    )
    controller = ScenarioController(
        client,
        world,
        args.traffic_light_data.resolve(),
    )
    ui = ScenarioUI(controller, config)
    try:
        ui.run()
    finally:
        controller.cleanup(remove_tick_callback=True)
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

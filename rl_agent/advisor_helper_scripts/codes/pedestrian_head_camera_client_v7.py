#!/usr/bin/env python3

"""
Manual pedestrian client with a head-mounted RGB camera.

The client connects to the world that is already loaded by a running CARLA
server.  It deliberately does not load a map, change world settings, or call
``world.tick()``.  With ``generate_traffic.py`` acting as the synchronous
master, the camera therefore captures one image per master simulation tick.

The pedestrian defaults to the requested Town10HD world coordinates
``x=70.99, y=60.97``.  Its ground height and initial heading are taken from
the nearest Sidewalk waypoint while the requested x/y values remain exact.
Pass ``--spawn-x`` and ``--spawn-y`` together to replace both coordinates;
pressing Y later returns the ego pedestrian to that same resolved transform
and recenters its camera.
Use ``--camera-height-reduction`` to lower the automatically derived head
height, or ``--camera-height``/``--camera-z`` to set an absolute mount height.
The optional top-down view follows this controlled ego pedestrian because this
client does not spawn a separately controllable ego vehicle.

By default, the client also spawns 30 car-only NPC vehicles on road spawn
points and enables stable Traffic Manager behavior for them: automatic lane
changes are disabled, cars remain centered, follow at 3 m, and target the
posted speed limit without a conservative reduction. This actor-specific speed
setting applies only to cars spawned by this client. Trucks, vans, motorcycles,
and bicycles are excluded using CARLA's ``base_type == "car"`` metadata. Pass
``--npc-vehicles 0`` to disable background traffic.

In a synchronous world, launch the clock-master process first. It must create
the selected Traffic Manager port, configure that manager as synchronous, and
call ``world.tick()``. This passive client never assumes either ownership.

Controls
--------
    W / S               walk forward / backward
    A / D               turn pedestrian left / right
    Hold Shift + W/S    run forward / backward
    Up / Down Arrow     camera pitch up / down
    Left / Right Arrow  camera yaw left / right
    R                   recenter the camera
    Y                   respawn at startup location and recenter camera
    B                   toggle vehicle/pedestrian ground-truth boxes
    U                   toggle ego-following top-down map
    Space               jump
    Esc / Q             quit

Example
-------
    python3 pedestrian_head_camera_client_v7.py --spawn-x 70.99 --spawn-y 60.97 \\
        --camera-height-reduction 0.30 --npc-vehicles 30 --tm-port 8000
"""

import argparse
import logging
import math
import random
import threading
import time
import uuid
from typing import List, Optional, Sequence, Tuple

import carla
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "pygame is required for live video and keyboard control. "
        "Install it in the CARLA Python environment with 'python3 -m pip install pygame'."
    ) from exc


LOG = logging.getLogger("pedestrian_head_camera")

# ==============================================================================
# PEDESTRIAN SPEED CONFIGURATION
# Edit these two values to change the normal walking and Shift-running speeds.
# They can also be overridden without editing the file by passing
# ``--walk-speed <m/s>`` and ``--run-speed <m/s>`` on the command line.
# ==============================================================================
DEFAULT_WALK_SPEED_MPS = 2.5
DEFAULT_RUN_SPEED_MPS = 5.0

# Fixed pedestrian spawn requested for this version. ``--spawn-x`` and
# ``--spawn-y`` can still override these defaults for reuse in another scene.
DEFAULT_SPAWN_X = 70.99
DEFAULT_SPAWN_Y = 60.97
DEFAULT_SPAWN_HEIGHT_OFFSET_M = 0.5
RESPAWN_OCCUPANCY_RADIUS_M = 1.0
RESPAWN_HOME_TOLERANCE_M = 0.10

# The camera height is relative to the walker actor origin. The default is
# derived from the selected pedestrian's bounding box at runtime.
DEFAULT_CAMERA_HEIGHT_REDUCTION_M = 0.0
MIN_CAMERA_MOUNT_HEIGHT_M = 0.10

# NPC car defaults. Car-only filtering remains mandatory even if the
# command-line blueprint pattern is broadened.
DEFAULT_NPC_CAR_COUNT = 30
DEFAULT_NPC_VEHICLE_FILTER = "vehicle.*"
DEFAULT_NPC_VEHICLE_GENERATION = "All"
DEFAULT_NPC_MIN_SPAWN_DISTANCE_M = 8.0
DEFAULT_NPC_FOLLOW_DISTANCE_M = 3.0
# Explicitly use the posted speed limit for this client's cars. Keeping this
# actor-specific 0% setting prevents a shared Traffic Manager's conservative
# global setting from being inherited without modifying other clients' cars.
DEFAULT_NPC_SPEED_DIFFERENCE_PERCENT = 0.0
DEFAULT_TRAFFIC_MANAGER_PORT = 8000
NPC_CAR_ROLE_PREFIX = "pedestrian_head_camera_npc_car"
NPC_HEALTH_CHECK_SIM_SECONDS = 5.0
NPC_HEALTH_CHECK_WALL_TIMEOUT_SECONDS = 8.0
NPC_HEALTH_MIN_DISPLACEMENT_M = 1.0
NPC_HEALTH_MIN_MOVING_RATIO = 0.5

# Local Pygame overlay colors. These do not modify CARLA's shared debug view.
VEHICLE_BOX_COLOR = (0, 190, 255)
PEDESTRIAN_BOX_COLOR = (255, 190, 0)
BOX_LABEL_COLOR = (245, 245, 245)
BOX_LABEL_BACKGROUND = (0, 0, 0)

# Ego-following top-down map configuration. The OpenCV colors are BGR values
# chosen to match the Physical AI scenario UI's RGB palette.
DEFAULT_TOPDOWN_ZOOM_RADIUS_M = 60.0
MIN_TOPDOWN_ZOOM_RADIUS_M = 1.0
MAX_TOPDOWN_ZOOM_RADIUS_M = 10000.0
TOPDOWN_MAP_REFRESH_HZ = 10.0
TOPDOWN_WAYPOINT_SPACING_M = 3.0
TOPDOWN_COLOR_BACKGROUND = (30, 23, 18)
TOPDOWN_COLOR_GRID = (61, 50, 43)
TOPDOWN_COLOR_BUILDING_FILL = (42, 42, 42)
TOPDOWN_COLOR_BUILDING_EDGE = (64, 64, 64)
TOPDOWN_COLOR_LANE_CENTERLINE = (85, 85, 85)
TOPDOWN_COLOR_VEHICLE = (220, 150, 72)
TOPDOWN_COLOR_PEDESTRIAN = (178, 195, 82)
TOPDOWN_COLOR_EGO = (68, 173, 255)

MIN_BUILDING_HEIGHT_M = 2.0
MIN_BUILDING_AREA_M2 = 20.0
MIN_BUILDING_VOLUME_M3 = 80.0
BUILDING_ROAD_PROXIMITY_M = 20.0
BUILDING_EDGE_SAMPLE_M = 5.0


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse WIDTHxHEIGHT command-line values."""
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "resolution must have the form WIDTHxHEIGHT, for example 1280x720"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def topdown_zoom_radius(value: str) -> float:
    """Parse a numerically safe ego-centered map radius."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if (
        not math.isfinite(parsed)
        or parsed < MIN_TOPDOWN_ZOOM_RADIUS_M
        or parsed > MAX_TOPDOWN_ZOOM_RADIUS_M
    ):
        raise argparse.ArgumentTypeError(
            "must be between {:.1f} and {:.1f} meters".format(
                MIN_TOPDOWN_ZOOM_RADIUS_M,
                MAX_TOPDOWN_ZOOM_RADIUS_M,
            )
        )
    return parsed


class LatestCameraFrame:
    """A thread-safe, one-frame mailbox used by the camera callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image = None

    def push(self, image: carla.Image) -> None:
        # Replacing the reference drops stale frames instead of blocking CARLA's
        # sensor callback thread when rendering falls behind.
        with self._lock:
            self._image = image

    def pop(self):
        with self._lock:
            image = self._image
            self._image = None
        return image


class RecentWorldSnapshots:
    """Keep a small, thread-safe frame-to-snapshot cache for box alignment."""

    def __init__(self, capacity: int = 8) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._snapshots = {}

    def push(self, snapshot) -> None:
        frame = int(snapshot.frame)
        with self._lock:
            self._snapshots[frame] = snapshot
            while len(self._snapshots) > self._capacity:
                oldest_frame = min(self._snapshots)
                del self._snapshots[oldest_frame]

    def get(self, frame: int):
        with self._lock:
            return self._snapshots.get(int(frame))

    def latest(self):
        with self._lock:
            if not self._snapshots:
                return None
            return self._snapshots[max(self._snapshots)]


class NpcTrafficHealthMonitor:
    """Perform one passive, snapshot-only check for stalled NPC traffic."""

    def __init__(
        self,
        actor_ids: Sequence[int],
        traffic_manager_port: int,
        synchronous_world: bool,
    ) -> None:
        self.actor_ids = tuple(sorted(set(int(value) for value in actor_ids)))
        self.traffic_manager_port = traffic_manager_port
        self.synchronous_world = synchronous_world
        self.initial_frame = None
        self.initial_sim_time = None
        self.initial_locations = {}
        self.initial_wall_time = time.monotonic()
        self.last_frame = None
        self.last_frame_wall_time = self.initial_wall_time
        self.finished = not self.actor_ids

    def update(self, snapshot) -> None:
        if self.finished or snapshot is None:
            return

        now = time.monotonic()
        current_frame = int(snapshot.frame)
        if self.last_frame is None or current_frame != self.last_frame:
            self.last_frame = current_frame
            self.last_frame_wall_time = now
        elif (
            now - self.last_frame_wall_time
            >= NPC_HEALTH_CHECK_WALL_TIMEOUT_SECONDS
        ):
            if self.synchronous_world:
                clock_guidance = "start or inspect the synchronous clock master"
            else:
                clock_guidance = "inspect the asynchronous CARLA server"
            LOG.warning(
                "NPC traffic health check: the CARLA frame stopped advancing; %s.",
                clock_guidance,
            )
            self.finished = True
            return

        if self.initial_frame is None:
            initial_locations = {}
            for actor_id in self.actor_ids:
                actor_snapshot = snapshot.find(actor_id)
                if actor_snapshot is not None:
                    initial_locations[actor_id] = (
                        actor_snapshot.get_transform().location
                    )
            missing_initial_ids = set(self.actor_ids) - set(initial_locations)
            initialization_timed_out = (
                now - self.initial_wall_time
                >= NPC_HEALTH_CHECK_WALL_TIMEOUT_SECONDS
            )
            if missing_initial_ids and not initialization_timed_out:
                return
            if not initial_locations:
                if initialization_timed_out:
                    LOG.warning(
                        "NPC traffic health check: 0/%d owned cars appeared in "
                        "world snapshots within %.1f seconds",
                        len(self.actor_ids),
                        NPC_HEALTH_CHECK_WALL_TIMEOUT_SECONDS,
                    )
                    self.finished = True
                return
            if missing_initial_ids:
                LOG.warning(
                    "NPC traffic health check: only %d/%d owned cars appeared "
                    "before the %.1f-second initialization timeout",
                    len(initial_locations),
                    len(self.actor_ids),
                    NPC_HEALTH_CHECK_WALL_TIMEOUT_SECONDS,
                )
            self.initial_frame = current_frame
            self.initial_sim_time = float(snapshot.timestamp.elapsed_seconds)
            self.initial_locations = initial_locations
            return

        sim_elapsed = (
            float(snapshot.timestamp.elapsed_seconds) - self.initial_sim_time
        )
        if sim_elapsed < NPC_HEALTH_CHECK_SIM_SECONDS:
            return

        sampled_count = 0
        moving_count = 0
        for actor_id, initial_location in self.initial_locations.items():
            actor_snapshot = snapshot.find(actor_id)
            if actor_snapshot is None:
                continue
            sampled_count += 1
            current_location = actor_snapshot.get_transform().location
            horizontal_displacement = math.hypot(
                current_location.x - initial_location.x,
                current_location.y - initial_location.y,
            )
            if horizontal_displacement >= NPC_HEALTH_MIN_DISPLACEMENT_M:
                moving_count += 1

        expected_count = len(self.actor_ids)
        missing_count = expected_count - sampled_count
        stationary_count = sampled_count - moving_count
        moving_ratio = moving_count / expected_count
        if moving_ratio < NPC_HEALTH_MIN_MOVING_RATIO:
            if self.synchronous_world:
                guidance = (
                    "verify that the clock-master process was launched first "
                    "and owns synchronous TM port {}"
                ).format(self.traffic_manager_port)
            else:
                guidance = (
                    "try an unused port such as --tm-port 8010 to rule out "
                    "stale shared-TM settings"
                )
            LOG.warning(
                "NPC traffic health check: %d/%d cars moved at least %.1f m "
                "in %.1f simulation seconds (%d stationary, %d missing); %s. "
                "Cars stopped at red lights or behind traffic may be legitimate.",
                moving_count,
                expected_count,
                NPC_HEALTH_MIN_DISPLACEMENT_M,
                sim_elapsed,
                stationary_count,
                missing_count,
                guidance,
            )
        else:
            LOG.info(
                "NPC traffic health check: %d/%d owned cars moved at least "
                "%.1f m in %.1f simulation seconds",
                moving_count,
                expected_count,
                NPC_HEALTH_MIN_DISPLACEMENT_M,
                sim_elapsed,
            )
        self.finished = True


class ActorProjectionCache:
    """Refresh vehicle/pedestrian handles at 2 Hz instead of every frame."""

    def __init__(self, world: carla.World, refresh_seconds: float = 0.5) -> None:
        self.world = world
        self.refresh_seconds = refresh_seconds
        self.actors: List[carla.Actor] = []
        self.next_refresh = 0.0

    def get(self) -> List[carla.Actor]:
        now = time.monotonic()
        if now >= self.next_refresh:
            actors = self.world.get_actors()
            self.actors = list(actors.filter("vehicle.*")) + list(
                actors.filter("walker.pedestrian.*")
            )
            self.next_refresh = now + self.refresh_seconds
        return self.actors

    def invalidate(self) -> None:
        """Force the next overlay pass to fetch current actor handles."""
        self.actors = []
        self.next_refresh = 0.0


def camera_calibration(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Build the pinhole intrinsic matrix for the RGB camera."""
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    calibration = np.identity(3, dtype=np.float64)
    calibration[0, 0] = calibration[1, 1] = focal
    calibration[0, 2] = width / 2.0
    calibration[1, 2] = height / 2.0
    return calibration


def project_actor_box(
    actor: carla.Actor,
    actor_transform: carla.Transform,
    world_to_camera: np.ndarray,
    calibration: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[pygame.Rect, float]]:
    """Project an actor's eight 3D bounding-box corners into a 2D envelope."""
    try:
        vertices = actor.bounding_box.get_world_vertices(actor_transform)
    except (AttributeError, RuntimeError):
        return None

    world_points = np.asarray(
        [[vertex.x, vertex.y, vertex.z, 1.0] for vertex in vertices],
        dtype=np.float64,
    )
    camera_points = (world_to_camera @ world_points.T).T[:, :3]

    # CARLA camera coordinates are x forward, y right, z up.
    depths = camera_points[:, 0]
    in_front = depths > 0.10
    if np.count_nonzero(in_front) < 2:
        return None
    camera_points = camera_points[in_front]
    depths = camera_points[:, 0]
    horizontal = (
        calibration[0, 2]
        + camera_points[:, 1] / depths * calibration[0, 0]
    )
    vertical = (
        calibration[1, 2]
        - camera_points[:, 2] / depths * calibration[1, 1]
    )

    if (
        float(np.max(horizontal)) < 0.0
        or float(np.min(horizontal)) > width - 1
        or float(np.max(vertical)) < 0.0
        or float(np.min(vertical)) > height - 1
    ):
        return None

    left = int(np.clip(np.min(horizontal), 0, width - 1))
    right = int(np.clip(np.max(horizontal), 0, width - 1))
    top = int(np.clip(np.min(vertical), 0, height - 1))
    bottom = int(np.clip(np.max(vertical), 0, height - 1))
    if right - left < 3 or bottom - top < 3:
        return None
    return pygame.Rect(left, top, right - left, bottom - top), float(np.min(depths))


def draw_ground_truth_boxes(
    surface: pygame.Surface,
    actors: Sequence[carla.Actor],
    camera_transform: carla.Transform,
    snapshot,
    calibration: np.ndarray,
    max_distance: float,
    excluded_ids: Sequence[int],
    font: pygame.font.Font,
) -> Tuple[int, int]:
    """Draw local vehicle/pedestrian GT overlays and return class counts."""
    width, height = surface.get_size()
    excluded = {int(actor_id) for actor_id in excluded_ids}
    camera_location = camera_transform.location
    world_to_camera = np.asarray(
        camera_transform.get_inverse_matrix(), dtype=np.float64
    )
    projected = []

    for actor in actors:
        try:
            actor_id = int(actor.id)
            if actor_id in excluded:
                continue
            actor_snapshot = snapshot.find(actor_id)
            if actor_snapshot is None:
                continue
            actor_transform = actor_snapshot.get_transform()
            if actor_transform.location.distance(camera_location) > max_distance:
                continue
            result = project_actor_box(
                actor,
                actor_transform,
                world_to_camera,
                calibration,
                width,
                height,
            )
        except (AttributeError, RuntimeError):
            continue
        if result is None:
            continue
        rect, depth = result
        is_pedestrian = str(actor.type_id).startswith("walker.pedestrian.")
        projected.append((depth, rect, actor_id, is_pedestrian))

    # Draw distant boxes first so nearer outlines and labels remain legible.
    projected.sort(key=lambda item: item[0], reverse=True)
    vehicle_count = 0
    pedestrian_count = 0
    for depth, rect, actor_id, is_pedestrian in projected:
        if is_pedestrian:
            color = PEDESTRIAN_BOX_COLOR
            label_kind = "PED"
            pedestrian_count += 1
        else:
            color = VEHICLE_BOX_COLOR
            label_kind = "VEH"
            vehicle_count += 1

        pygame.draw.rect(surface, color, rect, 2)
        label = font.render(
            "{} id={} {:.1f}m".format(label_kind, actor_id, depth),
            True,
            BOX_LABEL_COLOR,
            BOX_LABEL_BACKGROUND,
        )
        label_x = max(0, min(rect.left, width - label.get_width()))
        label_y = rect.top - label.get_height()
        if label_y < 0:
            label_y = min(height - label.get_height(), rect.top + 2)
        surface.blit(label, (label_x, label_y))

    return vehicle_count, pedestrian_count


def actor_footprint_points(
    actor: carla.Actor,
    actor_transform: carla.Transform,
) -> np.ndarray:
    """Return the actor bounding box's four ordered world-XY footprint points."""
    vertices = actor.bounding_box.get_world_vertices(actor_transform)
    points = np.asarray(
        [[float(vertex.x), float(vertex.y)] for vertex in vertices],
        dtype=np.float32,
    )
    points = np.unique(np.round(points, decimals=5), axis=0)
    if len(points) < 3:
        raise ValueError("actor bounding box has no drawable XY footprint")
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


class TopDownMapRenderer:
    """Physical-AI-style map centered on the controlled ego pedestrian."""

    def __init__(
        self,
        carla_world: carla.World,
        carla_map: carla.Map,
        zoom_radius_m: float,
        width: int = 960,
        height: int = 960,
        refresh_hz: float = TOPDOWN_MAP_REFRESH_HZ,
    ) -> None:
        self._world = carla_world
        self._map = carla_map
        self._zoom_radius_m = float(zoom_radius_m)
        if (
            not math.isfinite(self._zoom_radius_m)
            or self._zoom_radius_m < MIN_TOPDOWN_ZOOM_RADIUS_M
            or self._zoom_radius_m > MAX_TOPDOWN_ZOOM_RADIUS_M
        ):
            raise ValueError(
                "top-down zoom radius must be between {:.1f} and {:.1f} meters".format(
                    MIN_TOPDOWN_ZOOM_RADIUS_M,
                    MAX_TOPDOWN_ZOOM_RADIUS_M,
                )
            )
        refresh_hz = float(refresh_hz)
        if not math.isfinite(refresh_hz) or refresh_hz <= 0.0:
            raise ValueError("top-down refresh rate must be positive and finite")

        self._width = int(width)
        self._height = int(height)
        self._window_name = "CARLA Ego Pedestrian Top-Down Map"
        self._window_created = False
        self._last_refresh_time = None
        self._refresh_period_seconds = 1.0 / refresh_hz
        self._header_height = 58
        self._footer_height = 34
        self._margin = 28
        available_height = self._height - self._header_height - self._footer_height
        self._plot_size = min(self._width - 2 * self._margin, available_height)
        if self._plot_size < 64:
            raise ValueError("top-down map dimensions are too small")
        self._plot_left = (self._width - self._plot_size) // 2
        self._plot_top = self._header_height + (
            available_height - self._plot_size
        ) // 2
        self._plot_center_pixel = (self._plot_size - 1) / 2.0
        self._scale = (self._plot_size - 1) / (2.0 * self._zoom_radius_m)
        self._center_x = 0.0
        self._center_y = 0.0
        self._road_polylines = []
        self._building_footprints = []
        self.ready = cv2 is not None
        if self.ready:
            self._build_static_geometry()

    @staticmethod
    def _geometry_entry(points) -> Tuple[np.ndarray, Tuple[float, ...]]:
        points_array = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        bounds = (
            float(np.min(points_array[:, 0])),
            float(np.min(points_array[:, 1])),
            float(np.max(points_array[:, 0])),
            float(np.max(points_array[:, 1])),
        )
        return points_array, bounds

    @staticmethod
    def _smooth_polyline(points, passes: int = 2) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        for _ in range(max(0, int(passes))):
            if len(result) < 3:
                break
            smoothed = [result[0]]
            for first, second in zip(result, result[1:]):
                smoothed.append(0.75 * first + 0.25 * second)
                smoothed.append(0.25 * first + 0.75 * second)
            smoothed.append(result[-1])
            result = np.asarray(smoothed, dtype=np.float32)
        return result

    @classmethod
    def _build_road_polylines(
        cls,
        waypoints,
        sample_spacing: float,
    ):
        lane_samples = {}
        for waypoint in waypoints:
            try:
                if waypoint.lane_type != carla.LaneType.Driving:
                    continue
                key = (
                    int(waypoint.road_id),
                    int(waypoint.section_id),
                    int(waypoint.lane_id),
                )
                location = waypoint.transform.location
                lane_samples.setdefault(key, []).append(
                    (float(waypoint.s), float(location.x), float(location.y))
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue

        polylines = []
        minimum_separation = max(0.1, float(sample_spacing) * 0.25)
        for samples in lane_samples.values():
            samples.sort(key=lambda item: item[0])
            points = []
            for _, x_coord, y_coord in samples:
                if points:
                    separation = math.hypot(
                        x_coord - points[-1][0],
                        y_coord - points[-1][1],
                    )
                    if separation < minimum_separation:
                        continue
                points.append((x_coord, y_coord))
            if len(points) >= 2:
                smoothed = cls._smooth_polyline(points, passes=2)
                polylines.append(cls._geometry_entry(smoothed))
        return polylines

    @staticmethod
    def _building_footprint(bounding_box) -> np.ndarray:
        transform = carla.Transform(
            bounding_box.location,
            bounding_box.rotation,
        )
        extent = bounding_box.extent
        corners = []
        for x_coord, y_coord in (
            (extent.x, extent.y),
            (-extent.x, extent.y),
            (-extent.x, -extent.y),
            (extent.x, -extent.y),
        ):
            corner = transform.transform(
                carla.Location(
                    x=float(x_coord),
                    y=float(y_coord),
                    z=-float(extent.z),
                )
            )
            corners.append((float(corner.x), float(corner.y)))
        return np.asarray(corners, dtype=np.float32)

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        total = 0.0
        for current, following in zip(points, np.roll(points, -1, axis=0)):
            total += float(current[0]) * float(following[1])
            total -= float(current[1]) * float(following[0])
        return abs(total) * 0.5

    @staticmethod
    def _sample_polygon_edges(points: np.ndarray, spacing: float):
        samples = []
        if len(points) < 2:
            return samples
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            length = math.hypot(
                float(end[0] - start[0]),
                float(end[1] - start[1]),
            )
            steps = max(1, int(math.ceil(length / max(0.1, spacing))))
            for step in range(steps + 1):
                fraction = float(step) / float(steps)
                samples.append(
                    (
                        float(start[0] + (end[0] - start[0]) * fraction),
                        float(start[1] + (end[1] - start[1]) * fraction),
                    )
                )
        return samples

    def _build_building_footprints(self, road_locations: np.ndarray):
        try:
            environment_objects = self._world.get_environment_objects(
                carla.CityObjectLabel.Buildings
            )
        except Exception as exc:
            LOG.warning("Building footprints unavailable for top-down map: %s", exc)
            return []

        cell_size = BUILDING_ROAD_PROXIMITY_M
        road_grid = {}
        for x_coord, y_coord in road_locations:
            key = (
                math.floor(float(x_coord) / cell_size),
                math.floor(float(y_coord) / cell_size),
            )
            road_grid.setdefault(key, []).append(
                (float(x_coord), float(y_coord))
            )

        maximum_distance_squared = BUILDING_ROAD_PROXIMITY_M ** 2
        footprints = []
        for environment_object in environment_objects:
            try:
                bounding_box = environment_object.bounding_box
                footprint = self._building_footprint(bounding_box)
                area = self._polygon_area(footprint)
                height = float(bounding_box.extent.z) * 2.0
                if (
                    height < MIN_BUILDING_HEIGHT_M
                    or area < MIN_BUILDING_AREA_M2
                    or area * height < MIN_BUILDING_VOLUME_M3
                ):
                    continue

                close_to_road = not road_grid
                for sample_x, sample_y in self._sample_polygon_edges(
                    footprint,
                    BUILDING_EDGE_SAMPLE_M,
                ):
                    cell_x = math.floor(sample_x / cell_size)
                    cell_y = math.floor(sample_y / cell_size)
                    for offset_x in (-1, 0, 1):
                        for offset_y in (-1, 0, 1):
                            nearby = road_grid.get(
                                (cell_x + offset_x, cell_y + offset_y),
                                [],
                            )
                            for road_x, road_y in nearby:
                                dx = sample_x - road_x
                                dy = sample_y - road_y
                                if dx * dx + dy * dy <= maximum_distance_squared:
                                    close_to_road = True
                                    break
                            if close_to_road:
                                break
                        if close_to_road:
                            break
                    if close_to_road:
                        break
                if close_to_road:
                    footprints.append(self._geometry_entry(footprint))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        return footprints

    def _build_static_geometry(self) -> None:
        waypoints = list(
            self._map.generate_waypoints(TOPDOWN_WAYPOINT_SPACING_M)
        )
        if not waypoints:
            raise RuntimeError("Unable to build top-down map without waypoints")
        road_locations = np.asarray(
            [
                (
                    float(waypoint.transform.location.x),
                    float(waypoint.transform.location.y),
                )
                for waypoint in waypoints
            ],
            dtype=np.float32,
        )
        self._road_polylines = self._build_road_polylines(
            waypoints,
            TOPDOWN_WAYPOINT_SPACING_M,
        )
        self._building_footprints = self._build_building_footprints(
            road_locations
        )
        LOG.info(
            "Top-down map geometry: %d lane polylines, %d building footprints",
            len(self._road_polylines),
            len(self._building_footprints),
        )

    @staticmethod
    def _nice_grid_spacing(raw_spacing: float) -> float:
        if raw_spacing <= 0.0:
            return 10.0
        exponent = math.floor(math.log10(raw_spacing))
        fraction = raw_spacing / (10.0 ** exponent)
        if fraction <= 1.0:
            nice = 1.0
        elif fraction <= 2.0:
            nice = 2.0
        elif fraction <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10.0 ** exponent)

    @staticmethod
    def _bounds_intersect(first, second) -> bool:
        return not (
            first[2] < second[0]
            or first[0] > second[2]
            or first[3] < second[1]
            or first[1] > second[3]
        )

    def _visible_world_bounds(self) -> Tuple[float, float, float, float]:
        return (
            self._center_x - self._zoom_radius_m,
            self._center_y - self._zoom_radius_m,
            self._center_x + self._zoom_radius_m,
            self._center_y + self._zoom_radius_m,
        )

    def _world_xy_to_pixel(self, x_coord: float, y_coord: float) -> Tuple[int, int]:
        # Match the Physical AI map: CARLA +X is right and +Y is down.
        pixel_x = self._plot_center_pixel + (
            float(x_coord) - self._center_x
        ) * self._scale
        pixel_y = self._plot_center_pixel + (
            float(y_coord) - self._center_y
        ) * self._scale
        return int(round(pixel_x)), int(round(pixel_y))

    def _world_to_pixel(self, location: carla.Location) -> Tuple[int, int]:
        return self._world_xy_to_pixel(location.x, location.y)

    def _points_to_pixels(self, points: np.ndarray) -> np.ndarray:
        pixels = np.empty_like(points, dtype=np.float32)
        pixels[:, 0] = self._plot_center_pixel + (
            points[:, 0] - self._center_x
        ) * self._scale
        pixels[:, 1] = self._plot_center_pixel + (
            points[:, 1] - self._center_y
        ) * self._scale
        return np.rint(pixels).astype(np.int32)

    def _location_is_visible(self, location: carla.Location) -> bool:
        x_coord = float(location.x)
        y_coord = float(location.y)
        return (
            math.isfinite(x_coord)
            and math.isfinite(y_coord)
            and abs(x_coord - self._center_x) <= self._zoom_radius_m
            and abs(y_coord - self._center_y) <= self._zoom_radius_m
        )

    def _vehicle_footprint_in_view(
        self,
        actor: carla.Actor,
        actor_transform: carla.Transform,
    ) -> Optional[np.ndarray]:
        bounding_box = actor.bounding_box
        extent = bounding_box.extent
        coarse_margin = math.hypot(float(extent.x), float(extent.y)) + math.hypot(
            float(bounding_box.location.x),
            float(bounding_box.location.y),
        )
        location = actor_transform.location
        if (
            abs(float(location.x) - self._center_x)
            > self._zoom_radius_m + coarse_margin
            or abs(float(location.y) - self._center_y)
            > self._zoom_radius_m + coarse_margin
        ):
            return None
        footprint = actor_footprint_points(actor, actor_transform)
        footprint_bounds = self._geometry_entry(footprint)[1]
        if not self._bounds_intersect(
            footprint_bounds,
            self._visible_world_bounds(),
        ):
            return None
        return footprint

    def _draw_grid(self, image: np.ndarray) -> None:
        spacing = self._nice_grid_spacing(
            (2.0 * self._zoom_radius_m) / 8.0
        )
        bounds = self._visible_world_bounds()
        value = math.ceil(bounds[0] / spacing) * spacing
        while value <= bounds[2]:
            pixel_x, _ = self._world_xy_to_pixel(value, self._center_y)
            cv2.line(
                image,
                (pixel_x, 0),
                (pixel_x, self._plot_size - 1),
                TOPDOWN_COLOR_GRID,
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "{:.0f}".format(value),
                (pixel_x + 3, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (130, 135, 142),
                1,
                cv2.LINE_AA,
            )
            value += spacing

        value = math.ceil(bounds[1] / spacing) * spacing
        while value <= bounds[3]:
            _, pixel_y = self._world_xy_to_pixel(self._center_x, value)
            cv2.line(
                image,
                (0, pixel_y),
                (self._plot_size - 1, pixel_y),
                TOPDOWN_COLOR_GRID,
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "{:.0f}".format(value),
                (3, max(13, pixel_y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (130, 135, 142),
                1,
                cv2.LINE_AA,
            )
            value += spacing

    def _draw_static_map(self) -> np.ndarray:
        image = np.full(
            (self._plot_size, self._plot_size, 3),
            TOPDOWN_COLOR_BACKGROUND,
            dtype=np.uint8,
        )
        self._draw_grid(image)
        visible_bounds = self._visible_world_bounds()

        for points, bounds in self._building_footprints:
            if not self._bounds_intersect(bounds, visible_bounds):
                continue
            pixels = self._points_to_pixels(points)
            cv2.fillPoly(
                image,
                [pixels],
                TOPDOWN_COLOR_BUILDING_FILL,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                image,
                [pixels],
                True,
                TOPDOWN_COLOR_BUILDING_EDGE,
                1,
                lineType=cv2.LINE_AA,
            )

        for points, bounds in self._road_polylines:
            if not self._bounds_intersect(bounds, visible_bounds):
                continue
            cv2.polylines(
                image,
                [self._points_to_pixels(points)],
                False,
                TOPDOWN_COLOR_LANE_CENTERLINE,
                2,
                lineType=cv2.LINE_AA,
            )
        return image

    def _draw_vehicle(
        self,
        image: np.ndarray,
        actor: carla.Actor,
        actor_transform: carla.Transform,
        color,
        ego: bool = False,
        footprint: Optional[np.ndarray] = None,
    ) -> None:
        if footprint is None:
            footprint = actor_footprint_points(actor, actor_transform)
        footprint_pixels = self._points_to_pixels(footprint)
        cv2.fillPoly(image, [footprint_pixels], color, lineType=cv2.LINE_AA)
        cv2.polylines(
            image,
            [footprint_pixels],
            True,
            (235, 240, 247) if ego else color,
            2 if ego else 1,
            lineType=cv2.LINE_AA,
        )
        center = self._world_to_pixel(actor_transform.location)
        cv2.circle(
            image,
            center,
            3 if ego else 2,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
        forward_vector = actor_transform.get_forward_vector()
        heading_length = max(1.5, float(actor.bounding_box.extent.x) * 2.0)
        heading_location = carla.Location(
            x=actor_transform.location.x + forward_vector.x * heading_length,
            y=actor_transform.location.y + forward_vector.y * heading_length,
            z=actor_transform.location.z,
        )
        cv2.line(
            image,
            center,
            self._world_to_pixel(heading_location),
            (235, 240, 247) if ego else color,
            2 if ego else 1,
            lineType=cv2.LINE_AA,
        )

    def _draw_pedestrian(
        self,
        image: np.ndarray,
        actor_transform: carla.Transform,
        color,
        ego: bool = False,
    ) -> None:
        center = self._world_to_pixel(actor_transform.location)
        cv2.circle(image, center, 5 if ego else 4, (18, 23, 30), -1)
        cv2.circle(
            image,
            center,
            4 if ego else 3,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )

    def _draw_live_actors(
        self,
        image: np.ndarray,
        carla_world: carla.World,
        hero_actor: carla.Actor,
        hero_transform: carla.Transform,
    ) -> Tuple[int, int]:
        visible_vehicle_count = 0
        visible_pedestrian_count = 0
        hero_id = int(hero_actor.id)
        try:
            actors = carla_world.get_actors()
            vehicles = actors.filter("vehicle.*")
            pedestrians = actors.filter("walker.pedestrian.*")
        except RuntimeError:
            vehicles = []
            pedestrians = []

        for vehicle in vehicles:
            if int(vehicle.id) == hero_id:
                continue
            try:
                actor_transform = vehicle.get_transform()
                footprint = self._vehicle_footprint_in_view(
                    vehicle,
                    actor_transform,
                )
                if footprint is None:
                    continue
                self._draw_vehicle(
                    image,
                    vehicle,
                    actor_transform,
                    TOPDOWN_COLOR_VEHICLE,
                    footprint=footprint,
                )
                visible_vehicle_count += 1
            except (AttributeError, RuntimeError, ValueError):
                continue

        for pedestrian in pedestrians:
            if int(pedestrian.id) == hero_id:
                continue
            try:
                actor_transform = pedestrian.get_transform()
                if not self._location_is_visible(actor_transform.location):
                    continue
                self._draw_pedestrian(
                    image,
                    actor_transform,
                    TOPDOWN_COLOR_PEDESTRIAN,
                )
                visible_pedestrian_count += 1
            except (AttributeError, RuntimeError):
                continue

        if hero_actor.type_id.startswith("walker.pedestrian."):
            self._draw_pedestrian(
                image,
                hero_transform,
                TOPDOWN_COLOR_EGO,
                ego=True,
            )
            visible_pedestrian_count += 1
        else:
            self._draw_vehicle(
                image,
                hero_actor,
                hero_transform,
                TOPDOWN_COLOR_EGO,
                ego=True,
            )
            visible_vehicle_count += 1
        return visible_vehicle_count, visible_pedestrian_count

    def _draw_status(
        self,
        frame: np.ndarray,
        ego_location: carla.Location,
        visible_vehicle_count: int,
        visible_pedestrian_count: int,
    ) -> None:
        cv2.putText(
            frame,
            "Ego pedestrian top-down map | radius {:.1f} m".format(
                self._zoom_radius_m
            ),
            (self._plot_left, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (235, 240, 247),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "ego x={:.2f}  y={:.2f} | +X right, +Y down".format(
                float(ego_location.x),
                float(ego_location.y),
            ),
            (self._plot_left, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )

        legend_y = self._height - 12
        legend_entries = (
            ("EGO PED", TOPDOWN_COLOR_EGO),
            (
                "ALL VEHICLES {}".format(visible_vehicle_count),
                TOPDOWN_COLOR_VEHICLE,
            ),
            (
                "ALL PEDESTRIANS {}".format(visible_pedestrian_count),
                TOPDOWN_COLOR_PEDESTRIAN,
            ),
        )
        x_coord = self._plot_left
        for label, color in legend_entries:
            cv2.circle(frame, (x_coord + 5, legend_y - 4), 5, color, -1)
            cv2.putText(
                frame,
                label,
                (x_coord + 15, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (215, 220, 227),
                1,
                cv2.LINE_AA,
            )
            x_coord += 38 + len(label) * 8

    def render(
        self,
        carla_world: carla.World,
        hero_actor: carla.Actor,
    ) -> None:
        if not self.ready or hero_actor is None:
            return
        now = time.monotonic()
        if (
            self._last_refresh_time is not None
            and now - self._last_refresh_time < self._refresh_period_seconds
        ):
            return
        self._last_refresh_time = now

        try:
            hero_transform = hero_actor.get_transform()
        except RuntimeError:
            return
        self._center_x = float(hero_transform.location.x)
        self._center_y = float(hero_transform.location.y)

        plot_image = self._draw_static_map()
        vehicle_count, pedestrian_count = self._draw_live_actors(
            plot_image,
            carla_world,
            hero_actor,
            hero_transform,
        )
        frame = np.full(
            (self._height, self._width, 3),
            TOPDOWN_COLOR_BACKGROUND,
            dtype=np.uint8,
        )
        plot_bottom = self._plot_top + self._plot_size
        plot_right = self._plot_left + self._plot_size
        frame[
            self._plot_top:plot_bottom,
            self._plot_left:plot_right,
        ] = plot_image
        cv2.rectangle(
            frame,
            (self._plot_left, self._plot_top),
            (plot_right - 1, plot_bottom - 1),
            (75, 82, 92),
            1,
        )
        self._draw_status(
            frame,
            hero_transform.location,
            vehicle_count,
            pedestrian_count,
        )

        try:
            if not self._window_created:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name, self._width, self._height)
                self._window_created = True
            cv2.imshow(self._window_name, frame)
            cv2.waitKey(1)
        except cv2.error:
            self.close()
            self.ready = False

    def close(self) -> None:
        self._last_refresh_time = None
        if cv2 is None or not self._window_created:
            return
        try:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._window_created = False


def toggle_topdown_map(
    renderer,
    enabled: bool,
    world: carla.World,
    carla_map: carla.Map,
    zoom_radius: float,
):
    """Toggle the local map without mutating the CARLA world or its actors."""
    if enabled:
        if renderer is not None:
            renderer.close()
        LOG.info("Ego-pedestrian top-down map disabled")
        return renderer, False
    if cv2 is None:
        LOG.warning("Top-down map unavailable because OpenCV is not installed")
        return renderer, False
    if renderer is not None and not renderer.ready:
        renderer.close()
        renderer = None
    if renderer is None:
        try:
            renderer = TopDownMapRenderer(world, carla_map, zoom_radius)
        except Exception as exc:
            LOG.warning("Unable to initialize top-down map: %s", exc)
            return None, False
    enabled = bool(renderer.ready)
    if enabled:
        LOG.info("Ego-pedestrian top-down map enabled")
    return renderer, enabled


def horizontal_distance(first: carla.Location, second: carla.Location) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def copy_transform(transform: carla.Transform) -> carla.Transform:
    """Return a detached copy suitable for repeated actor spawns."""
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
        ),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        ),
    )


def respawn_target_blocker(
    world: carla.World,
    walker: carla.Walker,
    spawn_transform: carla.Transform,
) -> Optional[carla.Actor]:
    """Find another vehicle/walker occupying the configured respawn point."""
    target = spawn_transform.location
    actors = world.get_actors()
    for actor in actors:
        try:
            if int(actor.id) == int(walker.id):
                continue
            if not (
                actor.type_id.startswith("vehicle.")
                or actor.type_id.startswith("walker.")
            ):
                continue
            try:
                if actor.bounding_box.contains(target, actor.get_transform()):
                    return actor
            except (AttributeError, RuntimeError):
                pass
            if (
                horizontal_distance(actor.get_location(), target)
                < RESPAWN_OCCUPANCY_RADIUS_M
            ):
                return actor
        except (AttributeError, RuntimeError):
            continue
    return None


def fixed_sidewalk_transform(
    world: carla.World,
    spawn_x: float,
    spawn_y: float,
    spawn_z: Optional[float],
    spawn_yaw: Optional[float],
    sidewalk_tolerance: float,
) -> carla.Transform:
    """
    Build a spawn transform whose x/y coordinates remain exactly requested.

    The nearest Sidewalk waypoint supplies ground z and heading unless explicit
    overrides are provided.  It is used only as a reference; its x/y position
    does not replace ``spawn_x`` or ``spawn_y``.
    """
    carla_map = world.get_map()
    requested_location = carla.Location(
        x=spawn_x,
        y=spawn_y,
        z=0.0 if spawn_z is None else spawn_z,
    )
    sidewalk_waypoint = carla_map.get_waypoint(
        requested_location,
        project_to_road=True,
        lane_type=carla.LaneType.Sidewalk,
    )
    if sidewalk_waypoint is None:
        raise RuntimeError(
            "no Sidewalk waypoint was found near the requested spawn "
            "coordinates ({:.2f}, {:.2f})".format(spawn_x, spawn_y)
        )

    sidewalk_location = sidewalk_waypoint.transform.location
    distance_to_center = horizontal_distance(requested_location, sidewalk_location)
    half_lane_width = max(0.0, float(sidewalk_waypoint.lane_width) / 2.0)
    distance_outside_sidewalk = max(0.0, distance_to_center - half_lane_width)
    if distance_outside_sidewalk > sidewalk_tolerance:
        raise RuntimeError(
            "requested spawn coordinates ({:.2f}, {:.2f}) are {:.2f} m "
            "outside the nearest Sidewalk lane (tolerance {:.2f} m)".format(
                spawn_x,
                spawn_y,
                distance_outside_sidewalk,
                sidewalk_tolerance,
            )
        )

    spawn_location = carla.Location(
        x=spawn_x,
        y=spawn_y,
        z=(
            sidewalk_location.z + DEFAULT_SPAWN_HEIGHT_OFFSET_M
            if spawn_z is None
            else spawn_z
        ),
    )
    yaw = (
        sidewalk_waypoint.transform.rotation.yaw
        if spawn_yaw is None
        else spawn_yaw
    )
    return carla.Transform(spawn_location, carla.Rotation(yaw=yaw))


def spawn_pedestrian(
    world: carla.World,
    rng: random.Random,
    blueprint_filter: str,
    spawn_x: float,
    spawn_y: float,
    spawn_z: Optional[float],
    spawn_yaw: Optional[float],
    sidewalk_tolerance: float,
    resolved_spawn_transform: Optional[carla.Transform] = None,
    pedestrian_blueprint_id: Optional[str] = None,
) -> Tuple[carla.Walker, carla.Transform]:
    """Spawn one manually controlled walker at the fixed sidewalk location."""
    blueprint_library = world.get_blueprint_library()
    if pedestrian_blueprint_id is None:
        blueprints = list(blueprint_library.filter(blueprint_filter))
    else:
        try:
            blueprints = [blueprint_library.find(pedestrian_blueprint_id)]
        except (IndexError, RuntimeError) as exc:
            raise RuntimeError(
                "pedestrian blueprint {!r} is unavailable".format(
                    pedestrian_blueprint_id
                )
            ) from exc
    if not blueprints:
        raise RuntimeError(
            "no pedestrian blueprints matched {!r}".format(blueprint_filter)
        )

    spawn_transform = (
        fixed_sidewalk_transform(
            world,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            spawn_z=spawn_z,
            spawn_yaw=spawn_yaw,
            sidewalk_tolerance=sidewalk_tolerance,
        )
        if resolved_spawn_transform is None
        else copy_transform(resolved_spawn_transform)
    )
    blueprint = rng.choice(blueprints)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "manual_pedestrian")
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")

    walker = world.try_spawn_actor(blueprint, spawn_transform)
    if walker is not None:
        return walker, spawn_transform

    raise RuntimeError(
        "unable to spawn a pedestrian at ({:.2f}, {:.2f}, {:.2f}); "
        "the fixed location may be occupied".format(
            spawn_transform.location.x,
            spawn_transform.location.y,
            spawn_transform.location.z,
        )
    )


def npc_generation_number(value: str) -> Optional[int]:
    """Return a numeric blueprint generation, or None for an All selection."""
    if value.strip().lower() == "all":
        return None
    try:
        generation = int(value)
    except ValueError as exc:
        raise ValueError("NPC vehicle generation must be All, 1, 2, or 3") from exc
    if generation not in (1, 2, 3):
        raise ValueError("NPC vehicle generation must be All, 1, 2, or 3")
    return generation


def select_npc_car_blueprints(
    world: carla.World,
    blueprint_filter: str,
    generation: str,
) -> List[carla.ActorBlueprint]:
    """Select only four-wheel blueprints explicitly classified as cars."""
    requested_generation = npc_generation_number(generation)
    selected = []
    for blueprint in world.get_blueprint_library().filter(blueprint_filter):
        try:
            if not blueprint.has_attribute("base_type"):
                continue
            if blueprint.get_attribute("base_type").as_str() != "car":
                continue
            if blueprint.has_attribute("number_of_wheels"):
                if blueprint.get_attribute("number_of_wheels").as_int() != 4:
                    continue
            # A role name is required for exact, failure-safe ownership
            # reconciliation if a batch response is lost after server receipt.
            if not blueprint.has_attribute("role_name"):
                continue
            if requested_generation is not None:
                if not blueprint.has_attribute("generation"):
                    continue
                if blueprint.get_attribute("generation").as_int() != requested_generation:
                    continue
        except (RuntimeError, ValueError):
            continue
        selected.append(blueprint)
    return sorted(selected, key=lambda blueprint: blueprint.id)


def configure_npc_car_blueprint(
    blueprint: carla.ActorBlueprint,
    rng: random.Random,
    role_name: str,
) -> carla.ActorBlueprint:
    """Apply an owned role name plus deterministic paint/driver selections."""
    blueprint.set_attribute("role_name", role_name)
    for attribute_name in ("color", "driver_id"):
        if not blueprint.has_attribute(attribute_name):
            continue
        values = list(blueprint.get_attribute(attribute_name).recommended_values)
        if values:
            blueprint.set_attribute(attribute_name, rng.choice(values))
    return blueprint


def find_owned_npc_car_ids(world: carla.World, role_name: str) -> List[int]:
    """Find cars carrying this run's unique ownership role."""
    return sorted(
        int(actor.id)
        for actor in world.get_actors().filter("vehicle.*")
        if actor.attributes.get("role_name") == role_name
    )


def configure_npc_car_traffic_behavior(
    world: carla.World,
    traffic_manager,
    actor_ids: Sequence[int],
    role_name: str,
    follow_distance: float,
    speed_difference: float,
    enable_lane_changes: bool,
    auto_lights: bool,
) -> None:
    """Apply stable, per-actor behavior without changing shared TM globals."""
    requested_ids = sorted(set(int(actor_id) for actor_id in actor_ids))
    if not requested_ids:
        return

    actors = list(world.get_actors(requested_ids))
    actors_by_id = {int(actor.id): actor for actor in actors}
    missing_ids = [
        actor_id for actor_id in requested_ids if actor_id not in actors_by_id
    ]
    if missing_ids:
        raise RuntimeError(
            "unable to configure missing NPC car actor IDs: {}".format(
                missing_ids
            )
        )

    unexpected_ids = [
        actor_id
        for actor_id, actor in actors_by_id.items()
        if (
            not actor.type_id.startswith("vehicle.")
            or actor.attributes.get("role_name") != role_name
        )
    ]
    if unexpected_ids:
        raise RuntimeError(
            "refusing to configure non-owned NPC actor IDs: {}".format(
                sorted(unexpected_ids)
            )
        )

    for actor_id in requested_ids:
        actor = actors_by_id[actor_id]
        traffic_manager.auto_lane_change(actor, enable_lane_changes)
        traffic_manager.distance_to_leading_vehicle(actor, follow_distance)
        traffic_manager.vehicle_percentage_speed_difference(
            actor,
            speed_difference,
        )
        # Per-vehicle zero offset overrides a stale shared global lane offset.
        traffic_manager.vehicle_lane_offset(actor, 0.0)

        # Explicitly retain collision avoidance and traffic-rule compliance if
        # this port is shared with another client that changed TM defaults.
        traffic_manager.ignore_vehicles_percentage(actor, 0.0)
        traffic_manager.ignore_lights_percentage(actor, 0.0)
        traffic_manager.ignore_signs_percentage(actor, 0.0)
        traffic_manager.ignore_walkers_percentage(actor, 0.0)

        if not enable_lane_changes:
            traffic_manager.random_left_lanechange_percentage(actor, 0.0)
            traffic_manager.random_right_lanechange_percentage(actor, 0.0)
        traffic_manager.update_vehicle_lights(actor, auto_lights)

    LOG.info(
        "Configured %d NPC cars: lane_changes=%s, follow_distance=%.1f m, "
        "speed_difference=%+.1f%%, lane_offset=0.0 m, auto_lights=%s",
        len(requested_ids),
        "enabled" if enable_lane_changes else "disabled",
        follow_distance,
        speed_difference,
        auto_lights,
    )


def spawn_npc_cars(
    client: carla.Client,
    world: carla.World,
    traffic_manager,
    count: int,
    rng: random.Random,
    blueprint_filter: str,
    generation: str,
    exclusion_origin: carla.Location,
    min_spawn_distance: float,
    role_name: str,
) -> List[int]:
    """Batch-spawn owned car-only NPCs without advancing the CARLA clock."""
    if count <= 0:
        return []

    blueprint_library = world.get_blueprint_library()
    blueprints = select_npc_car_blueprints(
        world,
        blueprint_filter=blueprint_filter,
        generation=generation,
    )
    if not blueprints:
        raise RuntimeError(
            "no car-only vehicle blueprints matched filter {!r} and "
            "generation {!r}".format(blueprint_filter, generation)
        )

    spawn_points = sorted(
        (
            transform
            for transform in world.get_map().get_spawn_points()
            if transform.location.distance(exclusion_origin) >= min_spawn_distance
        ),
        key=lambda transform: (
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.yaw,
        ),
    )
    rng.shuffle(spawn_points)
    if len(spawn_points) < count:
        LOG.warning(
            "Requested %d NPC cars, but only %d road spawn points are at least "
            "%.1f m from the pedestrian",
            count,
            len(spawn_points),
            min_spawn_distance,
        )
    if not spawn_points:
        return []

    spawn_actor = carla.command.SpawnActor
    set_autopilot = carla.command.SetAutopilot
    traffic_manager_port = traffic_manager.get_port()
    actor_ids = set()
    errors = []
    next_spawn_point = 0

    # Retry occupied points using the remaining shuffled candidates. Spawning
    # and autopilot registration are separate batches so a chained-command
    # failure cannot hide a successfully created actor ID.
    while len(actor_ids) < count and next_spawn_point < len(spawn_points):
        needed = count - len(actor_ids)
        attempt_points = spawn_points[
            next_spawn_point : next_spawn_point + needed
        ]
        next_spawn_point += len(attempt_points)
        batch = []
        for transform in attempt_points:
            blueprint_id = rng.choice(blueprints).id
            blueprint = configure_npc_car_blueprint(
                blueprint_library.find(blueprint_id),
                rng,
                role_name,
            )
            batch.append(spawn_actor(blueprint, transform))

        previous_ids = set(actor_ids)
        response_lost = False
        try:
            # False is deliberate: this client never sends a synchronous tick
            # cue and therefore never competes with the existing clock master.
            responses = client.apply_batch_sync(batch, False)
        except RuntimeError as exc:
            # The server may have accepted actors before the RPC result was
            # lost. Recover them by this run's unique role before deciding
            # whether the batch failed completely.
            recovered_ids = set(find_owned_npc_car_ids(world, role_name))
            actor_ids.update(recovered_ids)
            if not recovered_ids - previous_ids:
                raise RuntimeError("NPC car spawn batch failed") from exc
            LOG.warning(
                "NPC spawn response was lost; recovered %d owned cars by role",
                len(recovered_ids - previous_ids),
            )
            response_lost = True
            responses = []

        for response in responses:
            if response.error:
                errors.append(response.error)
            else:
                actor_ids.add(int(response.actor_id))

        # Reconciliation also catches a missing response or a server-side
        # spawn that completed immediately before a transport error.
        actor_ids.update(find_owned_npc_car_ids(world, role_name))
        new_actor_ids = sorted(actor_ids - previous_ids)
        if new_actor_ids:
            autopilot_responses = client.apply_batch_sync(
                [
                    set_autopilot(actor_id, True, traffic_manager_port)
                    for actor_id in new_actor_ids
                ],
                False,
            )
            autopilot_errors = [
                response.error
                for response in autopilot_responses
                if response.error
            ]
            response_count_mismatch = (
                len(autopilot_responses) != len(new_actor_ids)
            )
            if autopilot_errors or response_count_mismatch:
                first_error = (
                    autopilot_errors[0]
                    if autopilot_errors
                    else "missing CARLA batch response"
                )
                failed_count = len(autopilot_errors)
                if response_count_mismatch:
                    failed_count += abs(
                        len(new_actor_ids) - len(autopilot_responses)
                    )
                raise RuntimeError(
                    "failed to enable Traffic Manager autopilot for {} owned "
                    "NPC car(s): {}".format(
                        failed_count,
                        first_error,
                    )
                )

        # Do not submit more spawn commands after an ambiguous response. Some
        # accepted actors could become visible after reconciliation, and a
        # retry could otherwise exceed the requested count.
        if response_lost:
            break

    if errors:
        LOG.warning(
            "%d NPC car spawn attempts failed and were retried; first error: %s",
            len(errors),
            errors[0],
        )
    spawn_log = LOG.info if len(actor_ids) == count else LOG.warning
    spawn_log(
        "Spawned %d/%d requested NPC cars on Traffic Manager port %d",
        len(actor_ids),
        count,
        traffic_manager_port,
    )
    return sorted(actor_ids)


def destroy_npc_cars(
    client: carla.Client,
    world: carla.World,
    actor_ids: Sequence[int],
    role_name: str,
) -> None:
    """Destroy only NPC cars created by this process, without a tick cue."""
    recorded_ids = set(int(actor_id) for actor_id in actor_ids)
    owned_ids = set()
    try:
        owned_ids.update(find_owned_npc_car_ids(world, role_name))
    except RuntimeError as exc:
        LOG.warning("Unable to reconcile owned NPC cars before cleanup: %s", exc)

    # Revalidate every stored ID against the unique role. This prevents an
    # unrelated actor from being deleted if another client reloads the world
    # and CARLA later reuses an actor ID before this process exits.
    for actor_id in recorded_ids - owned_ids:
        try:
            actor = world.get_actor(actor_id)
            if (
                actor is not None
                and actor.attributes.get("role_name") == role_name
            ):
                owned_ids.add(actor_id)
            elif actor is not None:
                LOG.warning(
                    "Skipping NPC cleanup for reused actor id=%d with role=%r",
                    actor_id,
                    actor.attributes.get("role_name"),
                )
        except RuntimeError as exc:
            LOG.warning(
                "Unable to validate recorded NPC actor id=%d: %s",
                actor_id,
                exc,
            )
    owned_ids = sorted(owned_ids)
    if not owned_ids:
        return

    failed_ids = []
    destroyed_count = 0
    try:
        responses = client.apply_batch_sync(
            [carla.command.DestroyActor(actor_id) for actor_id in owned_ids],
            False,
        )
        for index, actor_id in enumerate(owned_ids):
            if index >= len(responses) or responses[index].error:
                failed_ids.append(actor_id)
            else:
                destroyed_count += 1
    except RuntimeError as exc:
        LOG.warning("NPC car batch cleanup failed: %s", exc)
        failed_ids = owned_ids

    unconfirmed_ids = []
    for actor_id in failed_ids:
        try:
            actor = world.get_actor(actor_id)
            if actor is None or actor.destroy():
                destroyed_count += 1
            else:
                unconfirmed_ids.append(actor_id)
        except RuntimeError:
            unconfirmed_ids.append(actor_id)

    if unconfirmed_ids:
        LOG.warning(
            "Confirmed %d/%d owned NPC car deletions; unconfirmed actor IDs: %s",
            destroyed_count,
            len(owned_ids),
            unconfirmed_ids,
        )
    else:
        LOG.info("Destroyed all %d owned NPC cars", len(owned_ids))


def head_mount_location(
    walker: carla.Walker,
    camera_x: Optional[float],
    camera_z: Optional[float],
    camera_height_reduction: float,
) -> carla.Location:
    """Place the camera just in front of the walker's eyes/head."""
    bounds = walker.bounding_box
    default_x = bounds.location.x + bounds.extent.x + 0.05
    default_z = max(1.45, bounds.location.z + bounds.extent.z - 0.12)
    resolved_z = (
        camera_z
        if camera_z is not None
        else default_z - camera_height_reduction
    )
    if resolved_z <= MIN_CAMERA_MOUNT_HEIGHT_M:
        raise RuntimeError(
            "resolved camera height {:.3f} m must be above {:.2f} m; "
            "reduce --camera-height-reduction or use --camera-z".format(
                resolved_z,
                MIN_CAMERA_MOUNT_HEIGHT_M,
            )
        )
    return carla.Location(
        x=default_x if camera_x is None else camera_x,
        y=0.0,
        z=resolved_z,
    )


def spawn_rgb_camera(
    world: carla.World,
    walker: carla.Walker,
    mount_location: carla.Location,
    width: int,
    height: int,
    fov: float,
    gamma: float,
    frame_mailbox: LatestCameraFrame,
    cleanup_sink: Optional[
        List[Tuple[Optional[carla.Sensor], Optional[carla.Walker]]]
    ] = None,
) -> carla.Sensor:
    """Spawn a rigid, head-mounted camera that samples every simulation tick."""
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")
    if blueprint.has_attribute("gamma"):
        blueprint.set_attribute("gamma", str(gamma))

    camera = world.spawn_actor(
        blueprint,
        carla.Transform(mount_location),
        attach_to=walker,
        attachment_type=carla.AttachmentType.Rigid,
    )
    try:
        camera.listen(frame_mailbox.push)
    except BaseException:
        camera_cleanup_unconfirmed = False
        try:
            camera.stop()
        except RuntimeError:
            pass
        try:
            camera_cleanup_unconfirmed = not bool(camera.destroy())
        except RuntimeError:
            camera_cleanup_unconfirmed = True
        if cleanup_sink is not None and camera_cleanup_unconfirmed:
            cleanup_sink.append((camera, None))
        raise
    return camera


class PedestrianController:
    """Translate held keyboard keys into walker and camera controls."""

    def __init__(
        self,
        walker: carla.Walker,
        camera: carla.Sensor,
        mount_location: carla.Location,
        initial_yaw: float,
        walk_speed: float,
        run_speed: float,
        turn_rate: float,
        look_rate: float,
    ) -> None:
        self.walker = walker
        self.camera = camera
        self.mount_location = mount_location
        self.body_yaw = initial_yaw
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.walk_speed = walk_speed
        self.run_speed = run_speed
        self.turn_rate = turn_rate
        self.look_rate = look_rate
        self.current_speed = 0.0
        self.is_running = False
        self.last_walker_transform = walker.get_transform()

    @staticmethod
    def _pressed(keys, *key_codes: int) -> bool:
        return any(keys[key_code] for key_code in key_codes)

    def reset_camera(self) -> None:
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0

    def reset_while_at_spawn(self, configured_yaw: float) -> None:
        """Reset controls/view when replacement is unnecessary at home."""
        self.stop()
        self.body_yaw = configured_yaw
        self.current_speed = 0.0
        self.is_running = False
        self.reset_camera()
        self.last_walker_transform = self.walker.get_transform()
        self.camera.set_transform(self._camera_world_transform())

    def update(self, keys, delta_seconds: float) -> None:
        delta_seconds = min(max(delta_seconds, 0.0), 0.1)

        # WASD is reserved exclusively for pedestrian motion.
        turn_axis = int(self._pressed(keys, pygame.K_d)) - int(
            self._pressed(keys, pygame.K_a)
        )
        self.body_yaw = (
            self.body_yaw + turn_axis * self.turn_rate * delta_seconds
        ) % 360.0

        move_axis = int(self._pressed(keys, pygame.K_w)) - int(
            self._pressed(keys, pygame.K_s)
        )
        run_requested = self._pressed(
            keys, pygame.K_LSHIFT, pygame.K_RSHIFT
        )
        movement_speed = (
            self.run_speed if run_requested else self.walk_speed
        )

        heading = carla.Rotation(yaw=self.body_yaw).get_forward_vector()
        if move_axis < 0:
            heading = carla.Vector3D(x=-heading.x, y=-heading.y, z=0.0)

        control = carla.WalkerControl()
        control.direction = heading
        control.speed = movement_speed if move_axis else (0.01 if turn_axis else 0.0)
        control.jump = self._pressed(keys, pygame.K_SPACE)
        self.walker.apply_control(control)
        self.current_speed = control.speed
        self.is_running = bool(move_axis and run_requested)

        # Arrow keys control the head camera. Numpad 2/4/6/8 remain aliases.
        yaw_axis = int(
            self._pressed(keys, pygame.K_RIGHT, pygame.K_KP6)
        ) - int(
            self._pressed(keys, pygame.K_LEFT, pygame.K_KP4)
        )
        pitch_axis = int(
            self._pressed(keys, pygame.K_UP, pygame.K_KP8)
        ) - int(
            self._pressed(keys, pygame.K_DOWN, pygame.K_KP2)
        )
        self.camera_yaw = max(
            -90.0,
            min(
                90.0,
                self.camera_yaw + yaw_axis * self.look_rate * delta_seconds,
            ),
        )
        self.camera_pitch = max(
            -60.0,
            min(
                60.0,
                self.camera_pitch + pitch_axis * self.look_rate * delta_seconds,
            ),
        )

        self.last_walker_transform = self.walker.get_transform()
        self.camera.set_transform(self._camera_world_transform())

    def _camera_world_transform(self) -> carla.Transform:
        """
        Compose the relative head pose with the current walker transform.

        CARLA Actor.set_transform() takes a world transform.  Reapplying this
        composed pose keeps the dynamically rotated camera on the moving head,
        matching the approach used by the local manual-control clients.
        """
        parent = self.last_walker_transform
        parent_yaw_radians = math.radians(parent.rotation.yaw)
        cos_yaw = math.cos(parent_yaw_radians)
        sin_yaw = math.sin(parent_yaw_radians)

        relative = self.mount_location
        location = carla.Location(
            x=parent.location.x + relative.x * cos_yaw - relative.y * sin_yaw,
            y=parent.location.y + relative.x * sin_yaw + relative.y * cos_yaw,
            z=parent.location.z + relative.z,
        )
        rotation = carla.Rotation(
            pitch=parent.rotation.pitch + self.camera_pitch,
            yaw=parent.rotation.yaw + self.camera_yaw,
            roll=parent.rotation.roll,
        )
        return carla.Transform(location, rotation)

    def stop(self) -> None:
        if self.walker.is_alive:
            self.walker.apply_control(carla.WalkerControl())


def destroy_pedestrian_rig(
    controller: Optional[PedestrianController],
    camera: Optional[carla.Sensor],
    walker: Optional[carla.Walker],
    description: str,
) -> Tuple[Optional[carla.Sensor], Optional[carla.Walker]]:
    """Clean a rig and return actor handles whose deletion was unconfirmed."""
    remaining_camera = None
    remaining_walker = None
    if controller is not None:
        try:
            controller.stop()
        except RuntimeError:
            pass
    if camera is not None:
        camera_id = camera.id
        try:
            camera.stop()
        except RuntimeError:
            pass
        try:
            if camera.destroy():
                LOG.info("Destroyed %s RGB camera id=%d", description, camera_id)
            else:
                remaining_camera = camera
                LOG.warning(
                    "CARLA did not confirm destruction of %s RGB camera id=%d",
                    description,
                    camera_id,
                )
        except RuntimeError:
            remaining_camera = camera
    if walker is not None:
        walker_id = walker.id
        try:
            if walker.destroy():
                LOG.info("Destroyed %s pedestrian id=%d", description, walker_id)
            else:
                remaining_walker = walker
                LOG.warning(
                    "CARLA did not confirm destruction of %s pedestrian id=%d",
                    description,
                    walker_id,
                )
        except RuntimeError:
            remaining_walker = walker
    return remaining_camera, remaining_walker


def retry_retired_pedestrian_rigs(
    retired_rigs: List[
        Tuple[Optional[carla.Sensor], Optional[carla.Walker]]
    ],
) -> None:
    """Retry unconfirmed deletions and retain only actors still pending."""
    still_pending = []
    for index, (camera, walker) in enumerate(retired_rigs, start=1):
        remaining_rig = destroy_pedestrian_rig(
            None,
            camera,
            walker,
            "retired retry #{}".format(index),
        )
        if any(actor is not None for actor in remaining_rig):
            still_pending.append(remaining_rig)
    retired_rigs[:] = still_pending


def spawn_replacement_pedestrian_rig(
    world: carla.World,
    rng: random.Random,
    args: argparse.Namespace,
    width: int,
    height: int,
    spawn_transform: carla.Transform,
    pedestrian_blueprint_id: str,
    cleanup_sink: Optional[
        List[Tuple[Optional[carla.Sensor], Optional[carla.Walker]]]
    ] = None,
) -> Tuple[
    carla.Walker,
    carla.Sensor,
    PedestrianController,
    LatestCameraFrame,
    carla.Location,
]:
    """Create a complete replacement rig, cleaning partial actors on error."""
    new_walker = None
    new_camera = None
    try:
        new_walker, _ = spawn_pedestrian(
            world,
            rng,
            blueprint_filter=args.walker_filter,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_z=args.spawn_z,
            spawn_yaw=args.spawn_yaw,
            sidewalk_tolerance=args.sidewalk_tolerance,
            resolved_spawn_transform=spawn_transform,
            pedestrian_blueprint_id=pedestrian_blueprint_id,
        )
        new_mount_location = head_mount_location(
            new_walker,
            args.camera_x,
            args.camera_z,
            args.camera_height_reduction,
        )
        new_frame_mailbox = LatestCameraFrame()
        new_camera = spawn_rgb_camera(
            world,
            new_walker,
            new_mount_location,
            width,
            height,
            args.fov,
            args.gamma,
            new_frame_mailbox,
            cleanup_sink,
        )
        new_controller = PedestrianController(
            walker=new_walker,
            camera=new_camera,
            mount_location=new_mount_location,
            initial_yaw=spawn_transform.rotation.yaw,
            walk_speed=args.walk_speed,
            run_speed=args.run_speed,
            turn_rate=args.turn_rate,
            look_rate=args.look_rate,
        )
        return (
            new_walker,
            new_camera,
            new_controller,
            new_frame_mailbox,
            new_mount_location,
        )
    except BaseException:
        remaining_rig = destroy_pedestrian_rig(
            None,
            new_camera,
            new_walker,
            "partial replacement",
        )
        if (
            cleanup_sink is not None
            and any(actor is not None for actor in remaining_rig)
        ):
            cleanup_sink.append(remaining_rig)
        raise


def image_to_surface(image: carla.Image) -> pygame.Surface:
    """Convert CARLA BGRA bytes to an RGB pygame surface."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    expected_size = image.width * image.height * 4
    if array.size != expected_size:
        raise ValueError(
            "camera frame has {} bytes; expected {}".format(
                array.size, expected_size
            )
        )
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def draw_hud(
    display: pygame.Surface,
    font: pygame.font.Font,
    controller: PedestrianController,
    frame_id: Optional[int],
    client_fps: float,
    boxes_enabled: bool,
    box_counts: Tuple[int, int],
    npc_car_counts: Tuple[int, int],
    topdown_enabled: bool,
    topdown_radius: float,
) -> None:
    vehicle_boxes, pedestrian_boxes = box_counts
    spawned_npc_cars, requested_npc_cars = npc_car_counts
    lines = [
        "Pedestrian: W/S forward/back   A/D turn   Hold Shift + W/S to run",
        "Camera: Arrow Up/Down pitch   Arrow Left/Right yaw   R recenter",
        "Y: respawn/recenter at configured start   Space: jump   Esc/Q: quit",
        "Visuals: B boxes   U top-down map {} ({:.1f} m)".format(
            "ON" if topdown_enabled else "OFF",
            topdown_radius,
        ),
        "NPC cars: {}/{} on Traffic Manager autopilot".format(
            spawned_npc_cars,
            requested_npc_cars,
        ),
        "Boxes: {}   Vehicles: {}   Pedestrians: {}".format(
            "ON" if boxes_enabled else "OFF",
            vehicle_boxes if boxes_enabled else "--",
            pedestrian_boxes if boxes_enabled else "--",
        ),
        "Movement: {} at {:.2f} m/s   Walk: {:.2f}   Run: {:.2f}".format(
            "RUNNING" if controller.is_running else (
                "WALKING" if controller.current_speed > 0.01 else "STOPPED"
            ),
            controller.current_speed,
            controller.walk_speed,
            controller.run_speed,
        ),
        "CARLA frame: {}   Client: {:.0f} FPS   Camera yaw: {:+.1f}  pitch: {:+.1f}".format(
            "--" if frame_id is None else frame_id,
            client_fps,
            controller.camera_yaw,
            controller.camera_pitch,
        ),
    ]
    line_height = font.get_linesize()
    overlay = pygame.Surface(
        (display.get_width(), line_height * len(lines) + 12),
        pygame.SRCALPHA,
    )
    overlay.fill((0, 0, 0, 135))
    for index, line in enumerate(lines):
        text = font.render(line, True, (245, 245, 245))
        overlay.blit(text, (10, 6 + index * line_height))
    display.blit(overlay, (0, 0))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("-p", "--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="CARLA client timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(1280, 720),
        metavar="WIDTHxHEIGHT",
        help="display and camera resolution (default: 1280x720)",
    )
    parser.add_argument(
        "--topdown-zoom-radius",
        type=topdown_zoom_radius,
        default=DEFAULT_TOPDOWN_ZOOM_RADIUS_M,
        metavar="METERS",
        help=(
            "ego-pedestrian-centered map half-width/half-height in meters "
            "(range: 1-10000; default: %(default)s)"
        ),
    )
    parser.add_argument("--fov", type=float, default=90.0, help="camera FOV in degrees")
    parser.add_argument("--gamma", type=float, default=2.2, help="camera gamma")
    parser.add_argument(
        "--walker-filter",
        default="walker.pedestrian.*",
        help="pedestrian blueprint filter",
    )
    parser.add_argument("--seed", type=int, default=0, help="local random seed")
    parser.add_argument(
        "--spawn-x",
        type=float,
        default=None,
        help=(
            "pedestrian startup/respawn world x coordinate; requires "
            "--spawn-y (default: {:.2f})".format(DEFAULT_SPAWN_X)
        ),
    )
    parser.add_argument(
        "--spawn-y",
        type=float,
        default=None,
        help=(
            "pedestrian startup/respawn world y coordinate; requires "
            "--spawn-x (default: {:.2f})".format(DEFAULT_SPAWN_Y)
        ),
    )
    parser.add_argument(
        "--spawn-z",
        type=float,
        default=None,
        help="optional exact world z coordinate (default: Sidewalk z + 0.5 m)",
    )
    parser.add_argument(
        "--spawn-yaw",
        type=float,
        default=None,
        help="optional initial yaw in degrees (default: Sidewalk heading)",
    )
    parser.add_argument(
        "--sidewalk-tolerance",
        type=float,
        default=2.0,
        help="maximum permitted distance outside a Sidewalk lane in meters",
    )
    parser.add_argument(
        "--walk-speed",
        type=float,
        default=DEFAULT_WALK_SPEED_MPS,
        help="walking speed in m/s (default: %(default)s)",
    )
    parser.add_argument(
        "--run-speed",
        type=float,
        default=DEFAULT_RUN_SPEED_MPS,
        help="Shift-running speed in m/s (default: %(default)s)",
    )
    parser.add_argument(
        "--turn-rate",
        type=float,
        default=100.0,
        help="pedestrian turn rate in degrees/s",
    )
    parser.add_argument(
        "--look-rate",
        type=float,
        default=60.0,
        help="camera yaw/pitch rate in degrees/s",
    )
    parser.add_argument(
        "--camera-x",
        type=float,
        default=None,
        help="override forward head-camera offset in meters",
    )
    camera_height_group = parser.add_mutually_exclusive_group()
    camera_height_group.add_argument(
        "--camera-z",
        "--camera-height",
        dest="camera_z",
        type=float,
        default=None,
        metavar="METERS",
        help="absolute head-camera mount height above the walker origin",
    )
    camera_height_group.add_argument(
        "--camera-height-reduction",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT_REDUCTION_M,
        metavar="METERS",
        help=(
            "meters subtracted from the automatic head-camera height "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-n",
        "--npc-vehicles",
        "--number-of-vehicles",
        dest="npc_vehicles",
        type=int,
        default=DEFAULT_NPC_CAR_COUNT,
        metavar="COUNT",
        help="car-only NPC count; 0 disables NPC spawning (default: %(default)s)",
    )
    parser.add_argument(
        "--npc-vehicle-filter",
        default=DEFAULT_NPC_VEHICLE_FILTER,
        metavar="PATTERN",
        help="NPC vehicle blueprint pattern; cars-only is still enforced",
    )
    parser.add_argument(
        "--npc-vehicle-generation",
        default=DEFAULT_NPC_VEHICLE_GENERATION,
        metavar="GENERATION",
        help="NPC blueprint generation: All, 1, 2, or 3 (default: %(default)s)",
    )
    parser.add_argument(
        "--npc-seed",
        type=int,
        default=None,
        help="NPC spawn/blueprint seed (default: use --seed)",
    )
    parser.add_argument(
        "--npc-min-spawn-distance",
        type=float,
        default=DEFAULT_NPC_MIN_SPAWN_DISTANCE_M,
        metavar="METERS",
        help="minimum NPC road-spawn distance from pedestrian (default: %(default)s)",
    )
    parser.add_argument(
        "--npc-follow-distance",
        type=float,
        default=DEFAULT_NPC_FOLLOW_DISTANCE_M,
        metavar="METERS",
        help="per-NPC following distance (default: %(default)s)",
    )
    parser.add_argument(
        "--npc-speed-difference",
        type=float,
        default=DEFAULT_NPC_SPEED_DIFFERENCE_PERCENT,
        metavar="PERCENT",
        help=(
            "per-NPC percentage below the speed limit; 0 matches the limit "
            "and negative is faster "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--npc-enable-lane-changes",
        action="store_true",
        help="allow automatic NPC lane changes (disabled by default for stability)",
    )
    parser.add_argument(
        "--no-npc-auto-lights",
        action="store_false",
        dest="npc_auto_lights",
        default=True,
        help="disable automatic brake/indicator/headlight management",
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=DEFAULT_TRAFFIC_MANAGER_PORT,
        metavar="PORT",
        help="Traffic Manager port used for NPC autopilot (default: %(default)s)",
    )
    parser.add_argument(
        "--show-bboxes",
        action="store_true",
        help="start with vehicle/pedestrian ground-truth boxes enabled",
    )
    parser.add_argument(
        "--bbox-max-distance",
        type=float,
        default=90.0,
        metavar="METERS",
        help="maximum distance for ground-truth boxes (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if (args.spawn_x is None) != (args.spawn_y is None):
        parser.error("--spawn-x and --spawn-y must be provided together")
    if args.spawn_x is None:
        args.spawn_x = DEFAULT_SPAWN_X
        args.spawn_y = DEFAULT_SPAWN_Y

    coordinates = [args.spawn_x, args.spawn_y]
    if args.spawn_z is not None:
        coordinates.append(args.spawn_z)
    if args.spawn_yaw is not None:
        coordinates.append(args.spawn_yaw)
    if not all(math.isfinite(value) for value in coordinates):
        parser.error("spawn coordinates and yaw must be finite")
    camera_offsets = [args.camera_height_reduction]
    if args.camera_x is not None:
        camera_offsets.append(args.camera_x)
    if args.camera_z is not None:
        camera_offsets.append(args.camera_z)
    if not all(math.isfinite(value) for value in camera_offsets):
        parser.error("camera offsets and heights must be finite")
    if args.camera_height_reduction < 0.0:
        parser.error("--camera-height-reduction must be non-negative")
    if (
        args.camera_z is not None
        and args.camera_z <= MIN_CAMERA_MOUNT_HEIGHT_M
    ):
        parser.error(
            "--camera-z/--camera-height must be greater than {:.2f} m".format(
                MIN_CAMERA_MOUNT_HEIGHT_M
            )
        )
    if args.sidewalk_tolerance <= 0.0:
        parser.error("--sidewalk-tolerance must be positive")
    if args.walk_speed <= 0.0:
        parser.error("--walk-speed must be positive")
    if args.run_speed <= 0.0:
        parser.error("--run-speed must be positive")
    if args.run_speed <= args.walk_speed:
        parser.error("--run-speed must be greater than --walk-speed")
    if args.turn_rate <= 0.0 or args.look_rate <= 0.0:
        parser.error("--turn-rate and --look-rate must be positive")
    if args.npc_vehicles < 0:
        parser.error("--npc-vehicles must be non-negative")
    if not args.npc_vehicle_filter.strip():
        parser.error("--npc-vehicle-filter must not be empty")
    try:
        npc_generation_number(args.npc_vehicle_generation)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        not math.isfinite(args.npc_min_spawn_distance)
        or args.npc_min_spawn_distance < 0.0
    ):
        parser.error("--npc-min-spawn-distance must be finite and non-negative")
    if (
        not math.isfinite(args.npc_follow_distance)
        or args.npc_follow_distance <= 0.0
    ):
        parser.error("--npc-follow-distance must be finite and positive")
    if (
        not math.isfinite(args.npc_speed_difference)
        or not -100.0 <= args.npc_speed_difference < 100.0
    ):
        parser.error(
            "--npc-speed-difference must be finite and in [-100, 100)"
        )
    if not 1 <= args.tm_port <= 65535:
        parser.error("--tm-port must be between 1 and 65535")
    if args.bbox_max_distance <= 0.0:
        parser.error("--bbox-max-distance must be positive")
    if not 1.0 <= args.fov <= 179.0:
        parser.error("--fov must be between 1 and 179 degrees")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    width, height = args.resolution
    rng = random.Random(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    walker = None
    camera = None
    controller = None
    world = None
    carla_map = None
    snapshot_callback_id = None
    npc_vehicle_ids: List[int] = []
    retired_rigs: List[
        Tuple[Optional[carla.Sensor], Optional[carla.Walker]]
    ] = []
    npc_role_name = None
    npc_health_monitor = None
    topdown_renderer = None
    topdown_enabled = False
    pygame_initialized = False

    try:
        # get_world() attaches to the existing map.  Do not use load_world().
        world = client.get_world()
        carla_map = world.get_map()
        snapshot_buffer = RecentWorldSnapshots()
        snapshot_buffer.push(world.get_snapshot())
        snapshot_callback_id = world.on_tick(snapshot_buffer.push)
        projection_cache = ActorProjectionCache(world)
        settings = world.get_settings()
        LOG.info(
            "Connected to map %s; synchronous_mode=%s, fixed_delta_seconds=%s",
            carla_map.name,
            settings.synchronous_mode,
            settings.fixed_delta_seconds,
        )
        if settings.synchronous_mode:
            LOG.info(
                "Passive client mode: waiting for the existing master client "
                "(for example generate_traffic.py) to advance simulation ticks"
            )
        else:
            LOG.warning(
                "The loaded world is asynchronous. This client remains passive "
                "and will follow server-generated ticks."
            )

        walker, spawn_transform = spawn_pedestrian(
            world,
            rng,
            blueprint_filter=args.walker_filter,
            spawn_x=args.spawn_x,
            spawn_y=args.spawn_y,
            spawn_z=args.spawn_z,
            spawn_yaw=args.spawn_yaw,
            sidewalk_tolerance=args.sidewalk_tolerance,
        )
        mount_location = head_mount_location(
            walker,
            args.camera_x,
            args.camera_z,
            args.camera_height_reduction,
        )

        pygame.init()
        pygame.font.init()
        pygame_initialized = True
        display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA Pedestrian Head Camera v7")
        font = pygame.font.Font(pygame.font.get_default_font(), 18)
        box_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        display.fill((0, 0, 0))
        pygame.display.flip()

        frame_mailbox = LatestCameraFrame()
        camera = spawn_rgb_camera(
            world,
            walker,
            mount_location,
            width,
            height,
            args.fov,
            args.gamma,
            frame_mailbox,
            retired_rigs,
        )
        controller = PedestrianController(
            walker=walker,
            camera=camera,
            mount_location=mount_location,
            initial_yaw=spawn_transform.rotation.yaw,
            walk_speed=args.walk_speed,
            run_speed=args.run_speed,
            turn_rate=args.turn_rate,
            look_rate=args.look_rate,
        )

        if args.npc_vehicles > 0:
            npc_seed = args.seed if args.npc_seed is None else args.npc_seed
            npc_rng = random.Random(npc_seed)
            npc_role_name = "{}_{}".format(
                NPC_CAR_ROLE_PREFIX,
                uuid.uuid4().hex,
            )
            if settings.synchronous_mode:
                LOG.warning(
                    "Synchronous passive mode requires the already-running "
                    "clock master to own Traffic Manager port %d, configure "
                    "that manager as synchronous, and call world.tick()",
                    args.tm_port,
                )
            # The Traffic Manager may be shared with another client. Do not
            # change its synchronous mode, global spacing, speed, or RNG seed.
            traffic_manager = client.get_trafficmanager(args.tm_port)
            npc_vehicle_ids = spawn_npc_cars(
                client,
                world,
                traffic_manager,
                count=args.npc_vehicles,
                rng=npc_rng,
                blueprint_filter=args.npc_vehicle_filter,
                generation=args.npc_vehicle_generation,
                exclusion_origin=spawn_transform.location,
                min_spawn_distance=args.npc_min_spawn_distance,
                role_name=npc_role_name,
            )
            configure_npc_car_traffic_behavior(
                world,
                traffic_manager,
                npc_vehicle_ids,
                role_name=npc_role_name,
                follow_distance=args.npc_follow_distance,
                speed_difference=args.npc_speed_difference,
                enable_lane_changes=args.npc_enable_lane_changes,
                auto_lights=args.npc_auto_lights,
            )
            npc_health_monitor = NpcTrafficHealthMonitor(
                npc_vehicle_ids,
                traffic_manager_port=args.tm_port,
                synchronous_world=bool(settings.synchronous_mode),
            )
            LOG.info(
                "NPC cars-only filter=%r, generation=%s, seed=%d, role=%s",
                args.npc_vehicle_filter,
                args.npc_vehicle_generation,
                npc_seed,
                npc_role_name,
            )
            if not settings.synchronous_mode:
                LOG.info(
                    "Asynchronous world: if NPC motion remains irregular, retry "
                    "with an unused Traffic Manager port such as --tm-port 8010 "
                    "to rule out inherited shared-port settings"
                )
        else:
            LOG.info("NPC car spawning disabled with --npc-vehicles 0")

        LOG.info(
            "Spawned walker id=%d and RGB camera id=%d at (%.2f, %.2f, %.2f)",
            walker.id,
            camera.id,
            spawn_transform.location.x,
            spawn_transform.location.y,
            spawn_transform.location.z,
        )
        LOG.info(
            "Configured startup/respawn: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f; "
            "camera mount z=%.2f m",
            spawn_transform.location.x,
            spawn_transform.location.y,
            spawn_transform.location.z,
            spawn_transform.rotation.yaw,
            mount_location.z,
        )
        LOG.info(
            "The client never calls world.tick() and does not modify world settings"
        )
        LOG.info(
            "Pedestrian speeds: walk=%.2f m/s, Shift-run=%.2f m/s",
            args.walk_speed,
            args.run_speed,
        )
        LOG.info(
            "Ground-truth boxes start %s; press B to toggle them",
            "enabled" if args.show_bboxes else "disabled",
        )
        LOG.info(
            "Top-down map follows ego pedestrian; press U to toggle "
            "(radius %.1f m)",
            args.topdown_zoom_radius,
        )
        LOG.info(
            "Press Y to respawn the ego pedestrian at the configured startup "
            "coordinates"
        )

        clock = pygame.time.Clock()
        surface = None
        box_overlay = None
        box_counts = (0, 0)
        boxes_enabled = bool(args.show_bboxes)
        box_refresh_requested = boxes_enabled
        box_calibration = camera_calibration(width, height, args.fov)
        frame_id = None
        camera_frame_transform = None
        running = True

        while running:
            # A sleeping rate limiter keeps the UI responsive without burning
            # a CPU core; CARLA camera delivery remains driven by server ticks.
            delta_seconds = clock.tick(60) * 1e-3
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        controller.reset_camera()
                    elif event.key == pygame.K_y:
                        retry_retired_pedestrian_rigs(retired_rigs)
                        try:
                            blocker = respawn_target_blocker(
                                world,
                                walker,
                                spawn_transform,
                            )
                        except RuntimeError as exc:
                            LOG.warning(
                                "Unable to check the pedestrian respawn point: %s",
                                exc,
                            )
                        else:
                            if blocker is not None:
                                LOG.warning(
                                    "Configured pedestrian respawn at "
                                    "(%.2f, %.2f) is blocked by actor id=%d "
                                    "type=%s",
                                    spawn_transform.location.x,
                                    spawn_transform.location.y,
                                    blocker.id,
                                    blocker.type_id,
                                )
                            else:
                                try:
                                    already_home = (
                                        walker.is_alive
                                        and camera.is_alive
                                        and horizontal_distance(
                                            walker.get_location(),
                                            spawn_transform.location,
                                        )
                                        <= RESPAWN_HOME_TOLERANCE_M
                                    )
                                except RuntimeError as exc:
                                    LOG.warning(
                                        "Unable to read the current ego "
                                        "pedestrian location: %s",
                                        exc,
                                    )
                                    already_home = False
                                if already_home:
                                    try:
                                        controller.reset_while_at_spawn(
                                            spawn_transform.rotation.yaw
                                        )
                                    except RuntimeError as exc:
                                        LOG.warning(
                                            "Unable to reset the pedestrian "
                                            "rig at its configured start: %s",
                                            exc,
                                        )
                                    else:
                                        LOG.info(
                                            "Ego pedestrian id=%d is already "
                                            "at configured start (%.2f, %.2f); "
                                            "controls and camera recentered",
                                            walker.id,
                                            spawn_transform.location.x,
                                            spawn_transform.location.y,
                                        )
                                else:
                                    old_walker = walker
                                    old_camera = camera
                                    old_controller = controller
                                    try:
                                        (
                                            new_walker,
                                            new_camera,
                                            new_controller,
                                            new_frame_mailbox,
                                            new_mount_location,
                                        ) = spawn_replacement_pedestrian_rig(
                                            world,
                                            rng,
                                            args,
                                            width,
                                            height,
                                            spawn_transform,
                                            old_walker.type_id,
                                            retired_rigs,
                                        )
                                    except RuntimeError as exc:
                                        LOG.warning(
                                            "Unable to respawn ego pedestrian; "
                                            "the existing rig remains active: %s",
                                            exc,
                                        )
                                    else:
                                        remaining_rig = destroy_pedestrian_rig(
                                            old_controller,
                                            old_camera,
                                            old_walker,
                                            "previous",
                                        )
                                        if any(
                                            actor is not None
                                            for actor in remaining_rig
                                        ):
                                            retired_rigs.append(remaining_rig)
                                        walker = new_walker
                                        camera = new_camera
                                        controller = new_controller
                                        frame_mailbox = new_frame_mailbox
                                        mount_location = new_mount_location
                                        projection_cache.invalidate()
                                        surface = None
                                        frame_id = None
                                        camera_frame_transform = None
                                        box_overlay = None
                                        box_counts = (0, 0)
                                        box_refresh_requested = boxes_enabled
                                        LOG.info(
                                            "Respawned ego pedestrian id=%d at "
                                            "(%.2f, %.2f, %.2f), yaw=%.2f; "
                                            "camera id=%d mount-z=%.2f",
                                            walker.id,
                                            spawn_transform.location.x,
                                            spawn_transform.location.y,
                                            spawn_transform.location.z,
                                            spawn_transform.rotation.yaw,
                                            camera.id,
                                            mount_location.z,
                                        )
                    elif event.key == pygame.K_b:
                        boxes_enabled = not boxes_enabled
                        box_refresh_requested = boxes_enabled
                        if not boxes_enabled:
                            box_overlay = None
                            box_counts = (0, 0)
                        LOG.info(
                            "Ground-truth bounding boxes %s",
                            "enabled" if boxes_enabled else "disabled",
                        )
                    elif event.key == pygame.K_u:
                        topdown_renderer, topdown_enabled = toggle_topdown_map(
                            topdown_renderer,
                            topdown_enabled,
                            world,
                            carla_map,
                            args.topdown_zoom_radius,
                        )

            if not running:
                break
            if not walker.is_alive or not camera.is_alive:
                raise RuntimeError("the pedestrian or camera actor was destroyed")

            keys = pygame.key.get_pressed()
            controller.update(keys, delta_seconds)
            if npc_health_monitor is not None:
                npc_health_monitor.update(snapshot_buffer.latest())
            if topdown_enabled and topdown_renderer is not None:
                topdown_renderer.render(world, walker)
                if not topdown_renderer.ready:
                    topdown_enabled = False
                    LOG.warning(
                        "Top-down map disabled after an OpenCV rendering error"
                    )

            image = frame_mailbox.pop()
            if image is not None:
                surface = image_to_surface(image)
                frame_id = int(image.frame)
                camera_frame_transform = image.transform
                box_refresh_requested = boxes_enabled
                if boxes_enabled:
                    # Never carry geometry from the prior image onto this one.
                    box_overlay = None
                    box_counts = (0, 0)

            if (
                boxes_enabled
                and box_refresh_requested
                and surface is not None
                and frame_id is not None
                and camera_frame_transform is not None
            ):
                snapshot = snapshot_buffer.get(frame_id)
                if snapshot is not None:
                    box_overlay = pygame.Surface(
                        (width, height), pygame.SRCALPHA
                    )
                    try:
                        box_counts = draw_ground_truth_boxes(
                            box_overlay,
                            projection_cache.get(),
                            camera_frame_transform,
                            snapshot,
                            box_calibration,
                            args.bbox_max_distance,
                            excluded_ids=(walker.id,),
                            font=box_font,
                        )
                    except RuntimeError as exc:
                        LOG.debug("Ground-truth overlay skipped: %s", exc)
                        box_overlay = None
                        box_counts = (0, 0)
                    box_refresh_requested = False

            if surface is not None:
                display.blit(surface, (0, 0))
                if boxes_enabled and box_overlay is not None:
                    display.blit(box_overlay, (0, 0))
            else:
                display.fill((0, 0, 0))
                waiting = font.render(
                    "Waiting for the CARLA master clock / first camera frame...",
                    True,
                    (255, 255, 255),
                )
                display.blit(
                    waiting,
                    (
                        (width - waiting.get_width()) // 2,
                        (height - waiting.get_height()) // 2,
                    ),
                )
            draw_hud(
                display,
                font,
                controller,
                frame_id,
                clock.get_fps(),
                boxes_enabled,
                box_counts,
                (len(npc_vehicle_ids), args.npc_vehicles),
                topdown_enabled,
                args.topdown_zoom_radius,
            )
            pygame.display.flip()

    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        if topdown_renderer is not None:
            topdown_renderer.close()
        if world is not None and snapshot_callback_id is not None:
            try:
                world.remove_on_tick(snapshot_callback_id)
            except RuntimeError:
                pass
        if controller is not None:
            try:
                controller.stop()
            except RuntimeError:
                pass
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
            try:
                if camera.destroy():
                    LOG.info("Destroyed RGB camera id=%d", camera.id)
                else:
                    LOG.warning(
                        "CARLA did not confirm destruction of RGB camera id=%d "
                        "(it may already be gone)",
                        camera.id,
                    )
            except RuntimeError:
                pass
        if world is not None and npc_role_name is not None:
            destroy_npc_cars(
                client,
                world,
                npc_vehicle_ids,
                npc_role_name,
            )
        if walker is not None:
            try:
                if walker.destroy():
                    LOG.info("Destroyed pedestrian id=%d", walker.id)
                else:
                    LOG.warning(
                        "CARLA did not confirm destruction of pedestrian id=%d "
                        "(it may already be gone)",
                        walker.id,
                    )
            except RuntimeError:
                pass
        for index, (retired_camera, retired_walker) in enumerate(
            retired_rigs,
            start=1,
        ):
            destroy_pedestrian_rig(
                None,
                retired_camera,
                retired_walker,
                "retired #{}".format(index),
            )
        if pygame_initialized:
            pygame.quit()


if __name__ == "__main__":
    main()

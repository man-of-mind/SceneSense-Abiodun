#!/usr/bin/env python3

"""
Manual pedestrian client with a head-mounted RGB camera.

Version 9 adds a compact six-column Physical AI metrics window to Version 8.
Press U to show it together with the ground-truth boxes, route arrows, and
ego-following top-down map. Measured local proxies are kept separate from
deterministic, explicitly labelled DEMO values.

Version 8 loads coordinate-based ego-pedestrian routes exported by
``physical_ai_scenario_controller_ui_v3.py``. The route start is the exact
startup and Y-respawn transform. Repeated perspective arrows are projected
onto the ground in the RGB stream as passive guidance for the human operator;
the route never takes control of the walker. The arrows and RGB boxes remain
available if either optional OpenCV window cannot be opened.

Pass ``--record-route FILE`` to record the pedestrian path driven with the
keyboard. Recording starts automatically, is checkpointed atomically without
blocking CARLA callbacks, and is finalized when the client exits normally.
Pressing Y successfully respawns the pedestrian and starts a fresh recorded
take so the saved path never contains a teleport back to the start. Re-run the
client with ``--replay-route FILE`` (an alias of
``--pedestrian-route-config``), then press U to render that recorded path as
ground-projected waypoint arrows.

The client connects to the world that is already loaded by a running CARLA
server.  It deliberately does not load a map, change world settings, or call
``world.tick()``.  With ``generate_traffic.py`` acting as the synchronous
master, the camera therefore captures one image per master simulation tick.

The pedestrian defaults to the requested Town10HD world coordinates
``x=70.99, y=60.97``.  Its ground height and initial heading are taken from
the nearest Sidewalk waypoint while the requested x/y values remain exact.
Pass ``--spawn-x`` and ``--spawn-y`` together to replace both coordinates;
pressing Y later returns the ego pedestrian to that same resolved transform
and recenters its camera. When ``--pedestrian-route-config`` is present, its
validated start transform takes precedence over all four spawn arguments for
both startup and Y respawn.
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
    U                   toggle boxes, route arrows, top-down map, and metrics
    Space               jump
    Esc / Q             quit

Example
-------
    # Record a keyboard-driven route; Esc/Q/Ctrl-C finalizes the JSON file.
    python3 pedestrian_head_camera_client_v9.py --record-route recorded_route.json

    # Re-run from the recorded start and press U to show its waypoint arrows.
    python3 pedestrian_head_camera_client_v9.py \\
        --replay-route recorded_route.json \\
        --camera-height-reduction 0.30 --npc-vehicles 30 --tm-port 8000
"""

import argparse
from datetime import datetime, timezone
import logging
import math
import os
import random
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import carla
import numpy as np

from pedestrian_route_config import (
    MAX_ROUTE_CONTROL_POINTS,
    MAX_ROUTE_GUIDE_POINTS,
    MAX_ROUTE_SAMPLING_RESOLUTION_M,
    MAX_SPAWN_HEIGHT_OFFSET_M,
    MIN_ROUTE_SAMPLING_RESOLUTION_M,
    PEDESTRIAN_ROUTE_CONFIG_TYPE,
    PEDESTRIAN_ROUTE_COORDINATE_SYSTEM,
    PEDESTRIAN_ROUTE_SCHEMA_VERSION,
    load_route_config,
    maps_match,
    save_route_config,
)

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

# Perspective ground-route guidance. These pedestrian-scale values keep the
# view legible from a head camera without filling the near field.
ROUTE_GUIDANCE_LOOKAHEAD_M = 35.0
ROUTE_ARROW_SPACING_M = 4.0
ROUTE_ARROW_START_M = 2.0
ROUTE_ARROW_LENGTH_M = 1.4
ROUTE_ARROW_WIDTH_M = 0.75
ROUTE_GROUND_LIFT_M = 0.06
ROUTE_ENDPOINT_TOLERANCE_M = 2.0
ROUTE_CONTROL_TOLERANCE_M = 2.0
ROUTE_PROGRESS_SEARCH_SEGMENTS = 200
ROUTE_LOOKAHEAD_MAX_SEGMENTS = 4096
ROUTE_VERTICAL_TOLERANCE_M = 0.5
ROUTE_COLOR = (32, 205, 238, 165)
ROUTE_OUTLINE_COLOR = (5, 48, 58, 180)
ROUTE_ARROW_COLOR = (42, 224, 255, 205)
ROUTE_ARROW_OUTLINE_COLOR = (4, 55, 68, 225)

# Keyboard-driven route recording. One-meter arc-length samples preserve
# pedestrian-scale turns while remaining below the shared 200k-point schema
# limit for roughly eleven hours of continuous straight 5 m/s running (sharp
# turn pins may reach the cap sooner). Atomic periodic writes happen on a
# dedicated bounded worker, never in CARLA's tick callback.
DEFAULT_ROUTE_RECORD_SPACING_M = 1.0
DEFAULT_ROUTE_RECORD_CHECKPOINT_SECONDS = 10.0
MIN_ROUTE_RECORD_CHECKPOINT_SECONDS = 1.0
ROUTE_RECORD_MIN_DISTANCE_M = 0.05
ROUTE_RECORD_TELEPORT_MIN_DISTANCE_M = 10.0
ROUTE_RECORD_TELEPORT_SPEED_FACTOR = 3.0
ROUTE_RECORD_CONTROL_INTERVAL_M = 25.0
ROUTE_RECORD_CONTROL_TURN_DEGREES = 12.0
ROUTE_RECORD_SNAPSHOT_CAPACITY = 128
ROUTE_RECORD_WRITER_JOIN_SECONDS = 30.0
ROUTE_RECORD_CHECKPOINT_POINTS_PER_SECOND = 200.0
ROUTE_RECORD_MAX_ADAPTIVE_CHECKPOINT_SECONDS = 300.0

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

# Compact Physical AI metrics window. Accuracy and reasoning are explicit
# DEMO-only values because this client has neither a spatial-map estimator nor
# an AI reasoning stage. A dedicated RNG keeps these samples from perturbing
# walker, NPC, or route-recording reproducibility.
LIVE_METRICS_REFRESH_HZ = 5.0
LIVE_METRICS_WINDOW_WIDTH = 1200
LIVE_METRICS_WINDOW_HEIGHT = 184
DEFAULT_METRICS_PLACEHOLDER_SEED = 20260817
METRICS_PLACEHOLDER_REFRESH_SECONDS = 0.5
METRICS_PLACEHOLDER_EWMA_ALPHA = 0.25
METRICS_MEASURED_EWMA_ALPHA = 0.25
METRICS_MAX_SENSOR_AGE_SECONDS = 2.0
METRICS_MAX_MAP_SAMPLE_AGE_SECONDS = 2.0
METRICS_ACCURACY_MIN_CM = 1.0
METRICS_ACCURACY_MAX_CM = 4.0
METRICS_ACCURACY_MEAN_CM = 2.3
METRICS_ACCURACY_SIGMA_CM = 0.45
METRICS_REASONING_MIN_MS = 30.0
METRICS_REASONING_MAX_MS = 60.0
METRICS_REASONING_MEAN_MS = 45.0
METRICS_REASONING_SIGMA_MS = 5.0

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
        self._image_received_at = None
        self._latest_received_at = None
        self._last_control_sampled_at = None

    def push(self, image: carla.Image) -> None:
        # Replacing the reference drops stale frames instead of blocking CARLA's
        # sensor callback thread when rendering falls behind.
        received_at = time.perf_counter()
        with self._lock:
            self._image = image
            self._image_received_at = received_at
            self._latest_received_at = received_at

    def pop(self):
        with self._lock:
            image = self._image
            self._image = None
            self._image_received_at = None
        return image

    def pop_with_timestamp(self):
        """Pop the latest image with its callback-arrival wall timestamp."""
        with self._lock:
            image = self._image
            received_at = self._image_received_at
            self._image = None
            self._image_received_at = None
        return image, received_at

    def latest_received_at(self) -> Optional[float]:
        """Return the newest camera callback arrival without consuming it."""
        with self._lock:
            return self._latest_received_at

    def take_latest_received_at_for_control(self) -> Optional[float]:
        """Return each camera arrival at most once for Sense-to-Act timing."""
        with self._lock:
            received_at = self._latest_received_at
            if (
                received_at is None
                or received_at == self._last_control_sampled_at
            ):
                return None
            self._last_control_sampled_at = received_at
            return received_at


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

    def newer_than(self, frame: Optional[int]) -> List:
        """Return cached snapshots after *frame* in chronological order."""
        with self._lock:
            frames = sorted(
                value
                for value in self._snapshots
                if frame is None or value > int(frame)
            )
            return [self._snapshots[value] for value in frames]


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


def route_location(payload) -> carla.Location:
    """Convert one validated route location into a detached CARLA value."""
    return carla.Location(
        x=float(payload["x"]),
        y=float(payload["y"]),
        z=float(payload["z"]),
    )


def route_transform(payload) -> carla.Transform:
    """Convert one validated route transform without changing its pose."""
    rotation = payload["rotation"]
    return carla.Transform(
        route_location(payload["location"]),
        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"]),
        ),
    )


def copy_location(location: carla.Location) -> carla.Location:
    """Return a detached CARLA location."""
    return carla.Location(x=location.x, y=location.y, z=location.z)


def route_distance(first: carla.Location, second: carla.Location) -> float:
    """Return three-dimensional distance without requiring a live actor."""
    return math.hypot(
        float(first.x) - float(second.x),
        float(first.y) - float(second.y),
        float(first.z) - float(second.z),
    )


def route_horizontal_distance(
    first: carla.Location,
    second: carla.Location,
) -> float:
    """Return XY distance; route starts and actor spawns differ in height."""
    return math.hypot(
        float(first.x) - float(second.x),
        float(first.y) - float(second.y),
    )


def dedupe_route_locations(
    locations: Sequence[carla.Location],
    minimum_distance: float = 0.01,
) -> List[carla.Location]:
    """Copy a route while removing only consecutive near-duplicates."""
    result: List[carla.Location] = []
    for location in locations:
        copied = copy_location(location)
        if result and route_distance(result[-1], copied) < minimum_distance:
            continue
        result.append(copied)
    return result


def route_location_payload(location: carla.Location) -> Dict[str, float]:
    """Serialize one detached CARLA location for the shared route schema."""
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def route_transform_payload(transform: carla.Transform) -> Dict[str, object]:
    """Serialize one CARLA transform without querying a live actor."""
    return {
        "location": route_location_payload(transform.location),
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def walker_ground_location(
    transform: carla.Transform,
    bounding_box,
    fallback_offset: float,
) -> carla.Location:
    """Estimate the navigation-surface point beneath a walker snapshot.

    Bounding-box vertices are transformed locally by the CARLA Python API and
    avoid an RPC in the 20 Hz sampling path. The known spawn-height offset is
    retained as a defensive fallback for test doubles or older bindings.
    """
    resolved_z = float(transform.location.z) - float(fallback_offset)
    try:
        vertices = bounding_box.get_world_vertices(transform)
        vertex_z = [float(vertex.z) for vertex in vertices]
        if vertex_z and all(math.isfinite(value) for value in vertex_z):
            resolved_z = min(vertex_z)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return carla.Location(
        x=float(transform.location.x),
        y=float(transform.location.y),
        z=resolved_z,
    )


def recorded_route_intermediate_waypoints(
    route_path: Sequence[carla.Location],
) -> List[carla.Location]:
    """Select ordered, linear-time fallback controls from a recorded path."""
    if len(route_path) <= 2:
        return []

    selected: List[carla.Location] = []
    distance_since_control = 0.0
    for index in range(1, len(route_path) - 1):
        previous = route_path[index - 1]
        current = route_path[index]
        following = route_path[index + 1]
        incoming_x = float(current.x - previous.x)
        incoming_y = float(current.y - previous.y)
        outgoing_x = float(following.x - current.x)
        outgoing_y = float(following.y - current.y)
        incoming_length = math.hypot(incoming_x, incoming_y)
        outgoing_length = math.hypot(outgoing_x, outgoing_y)
        distance_since_control += incoming_length

        turn_degrees = 0.0
        if incoming_length > 1e-6 and outgoing_length > 1e-6:
            cosine = (
                incoming_x * outgoing_x + incoming_y * outgoing_y
            ) / (incoming_length * outgoing_length)
            cosine = max(-1.0, min(1.0, cosine))
            turn_degrees = math.degrees(math.acos(cosine))

        if (
            turn_degrees >= ROUTE_RECORD_CONTROL_TURN_DEGREES
            or distance_since_control >= ROUTE_RECORD_CONTROL_INTERVAL_M
        ):
            selected.append(copy_location(current))
            distance_since_control = 0.0

    available = max(0, MAX_ROUTE_CONTROL_POINTS - 2)
    if len(selected) <= available:
        return selected
    # The dense planned path remains authoritative. Uniform chronological
    # thinning keeps the fallback controls bounded without reordering loops.
    return [
        selected[min(len(selected) - 1, int(index * len(selected) / available))]
        for index in range(available)
    ]


class AtomicRouteCheckpointWriter:
    """Write only the newest immutable route checkpoint on one worker."""

    def __init__(self, output_path: str, allow_existing: bool = False) -> None:
        self.output_path = os.path.abspath(os.fspath(output_path))
        self.allow_existing = bool(allow_existing)
        self._condition = threading.Condition()
        self._pending = None
        self._closing = False
        self._submitted_generation = 0
        self._successful_generation = 0
        self._failed_generation = 0
        self._last_saved_point_count = 0
        self._has_written = False
        self._last_error = None
        self._unreported_error = None
        self._thread = threading.Thread(
            target=self._run,
            name="pedestrian-route-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, route_source) -> int:
        with self._condition:
            if self._closing:
                raise RuntimeError("route checkpoint writer is already closed")
            self._submitted_generation += 1
            generation = self._submitted_generation
            # One pending slot bounds memory and disk pressure. A later
            # checkpoint supersedes an older one that has not started writing.
            self._pending = (generation, route_source)
            self._condition.notify()
            return generation

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closing:
                    self._condition.wait()
                if self._pending is None and self._closing:
                    return
                generation, route_source = self._pending
                self._pending = None
            try:
                with self._condition:
                    first_write = not self._has_written
                route_data = (
                    route_source()
                    if callable(route_source)
                    else route_source
                )
                if route_data is None:
                    raise RuntimeError(
                        "recorded route became too short before checkpoint"
                    )
                normalized = save_route_config(
                    self.output_path,
                    route_data,
                    overwrite=(self.allow_existing or not first_write),
                )
            except Exception as exc:  # Report on the UI thread; never kill it.
                with self._condition:
                    self._failed_generation = max(
                        self._failed_generation,
                        generation,
                    )
                    self._last_error = (generation, exc)
                    self._unreported_error = (generation, exc)
                    self._condition.notify_all()
            else:
                with self._condition:
                    self._successful_generation = max(
                        self._successful_generation,
                        generation,
                    )
                    self._last_saved_point_count = len(
                        normalized.get("planned_path", [])
                    )
                    self._has_written = True
                    if (
                        self._last_error is not None
                        and self._last_error[0] <= generation
                    ):
                        self._last_error = None
                    if (
                        self._unreported_error is not None
                        and self._unreported_error[0] <= generation
                    ):
                        self._unreported_error = None
                    self._condition.notify_all()

    def wait_for(self, generation: int, timeout: float) -> None:
        """Wait for a requested generation to save or fail."""
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while (
                self._successful_generation < generation
                and self._failed_generation < generation
                and self._thread.is_alive()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "route checkpoint did not finish within {:.0f} seconds"
                        .format(timeout)
                    )
                self._condition.wait(remaining)
            if self._successful_generation >= generation:
                return
            error = self._last_error
            detail = "unknown write failure" if error is None else str(error[1])
            raise RuntimeError(
                "recorded-route checkpoint failed: {}".format(detail)
            )

    def consume_error(self):
        with self._condition:
            error = self._unreported_error
            self._unreported_error = None
            return error

    @property
    def last_saved_point_count(self) -> int:
        with self._condition:
            return self._last_saved_point_count

    def close(self, final_route_data: Optional[Dict[str, object]] = None) -> None:
        final_generation = None
        if final_route_data is not None:
            final_generation = self.submit(final_route_data)
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join(ROUTE_RECORD_WRITER_JOIN_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError(
                "route writer did not finish within {:.0f} seconds".format(
                    ROUTE_RECORD_WRITER_JOIN_SECONDS
                )
            )
        with self._condition:
            required_generation = (
                self._submitted_generation
                if final_generation is None
                else final_generation
            )
            if self._successful_generation < required_generation:
                error = self._last_error
                detail = "unknown write failure" if error is None else str(error[1])
                raise RuntimeError(
                    "final recorded-route save failed: {}".format(detail)
                )


class PedestrianRouteRecorder:
    """Record one continuous keyboard-driven take from world snapshots."""

    def __init__(
        self,
        output_path: str,
        map_name: str,
        route_name: str,
        sample_spacing: float,
        checkpoint_seconds: float,
        maximum_pedestrian_speed: float,
        walker: carla.Walker,
        spawn_transform: carla.Transform,
        baseline_snapshot=None,
        allow_overwrite: bool = False,
        spawn_height_offset_hint: Optional[float] = None,
    ) -> None:
        self.output_path = os.path.abspath(os.fspath(output_path))
        self.map_name = str(map_name)
        self.route_name = str(route_name)
        self.sample_spacing = float(sample_spacing)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.maximum_pedestrian_speed = float(maximum_pedestrian_speed)
        self.spawn_height_offset_hint = (
            None
            if spawn_height_offset_hint is None
            else float(spawn_height_offset_hint)
        )
        self.take_number = 0
        self.reset(walker, spawn_transform, baseline_snapshot)
        self.writer = AtomicRouteCheckpointWriter(
            self.output_path,
            allow_existing=allow_overwrite,
        )

    def reset(
        self,
        walker: carla.Walker,
        spawn_transform: carla.Transform,
        baseline_snapshot=None,
    ) -> None:
        """Discard the draft and begin a continuous take at the Y home pose."""
        self.take_number += 1
        self.actor_id = int(walker.id)
        self.actor_type_id = str(walker.type_id)
        self.bounding_box = walker.bounding_box
        self.start_transform = copy_transform(spawn_transform)
        if self.spawn_height_offset_hint is None:
            start_ground = walker_ground_location(
                self.start_transform,
                self.bounding_box,
                DEFAULT_SPAWN_HEIGHT_OFFSET_M,
            )
            spawn_height_offset = float(
                self.start_transform.location.z - start_ground.z
            )
        else:
            spawn_height_offset = self.spawn_height_offset_hint
            start_ground = copy_location(self.start_transform.location)
            start_ground.z -= spawn_height_offset
        if not (
            math.isfinite(spawn_height_offset)
            and 0.0 <= spawn_height_offset <= MAX_SPAWN_HEIGHT_OFFSET_M
        ):
            spawn_height_offset = DEFAULT_SPAWN_HEIGHT_OFFSET_M
            start_ground.z = (
                self.start_transform.location.z - spawn_height_offset
            )
        self.spawn_height_offset = spawn_height_offset
        self.path = [copy_location(start_ground)]
        self.latest_ground = copy_location(start_ground)
        self.latest_transform = copy_transform(self.start_transform)
        self.last_observed_ground = copy_location(start_ground)
        # Direction of the polyline at the most recently emitted route point.
        # Keeping this reference (rather than only comparing adjacent frames)
        # preserves gradual turns whose total arc is shorter than the regular
        # sampling interval.
        self.sampling_direction = None
        self.distance_since_output = 0.0
        self.recorded_distance = 0.0
        self.maximum_start_displacement = 0.0
        self.raw_snapshot_count = 0
        self.missing_snapshot_count = 0
        self.missed_frame_count = 0
        self.invalid_reason = None
        self.capacity_reached = False
        self.dirty = False
        self.created_utc = datetime.now(timezone.utc).isoformat()
        self.recording_id = uuid.uuid4().hex
        self.last_checkpoint_wall_time = time.monotonic()
        self.last_frame = (
            None if baseline_snapshot is None else int(baseline_snapshot.frame)
        )
        self.start_frame = None
        self.end_frame = None
        self.last_actor_simulation_time = None
        self.start_simulation_time = None
        self.end_simulation_time = None

    def _mark_invalid(self, reason: str) -> None:
        if self.invalid_reason is None:
            self.invalid_reason = reason
            LOG.error("Route recording stopped: %s", reason)

    def observe_snapshot(self, snapshot) -> bool:
        """Consume one previously unseen world snapshot without actor RPCs."""
        frame = int(snapshot.frame)
        if self.last_frame is not None and frame <= self.last_frame:
            return False
        if self.last_frame is not None and frame > self.last_frame + 1:
            self.missed_frame_count += frame - self.last_frame - 1
        self.last_frame = frame
        simulation_time = float(snapshot.timestamp.elapsed_seconds)

        actor_snapshot = snapshot.find(self.actor_id)
        if actor_snapshot is None:
            self.missing_snapshot_count += 1
            return False
        if self.invalid_reason is not None or self.capacity_reached:
            return False

        transform = actor_snapshot.get_transform()
        ground = walker_ground_location(
            transform,
            self.bounding_box,
            self.spawn_height_offset,
        )
        segment_distance = route_horizontal_distance(
            self.last_observed_ground,
            ground,
        )
        if self.last_actor_simulation_time is None:
            if segment_distance > ROUTE_RECORD_TELEPORT_MIN_DISTANCE_M:
                self._mark_invalid(
                    "first recorded frame is {:.2f} m from the configured "
                    "start".format(segment_distance)
                )
                return False
        else:
            delta_time = simulation_time - self.last_actor_simulation_time
            if delta_time < -1e-6:
                self._mark_invalid("CARLA simulation time moved backwards")
                return False
            maximum_continuous_distance = max(
                ROUTE_RECORD_TELEPORT_MIN_DISTANCE_M,
                self.maximum_pedestrian_speed
                * max(delta_time, 0.05)
                * ROUTE_RECORD_TELEPORT_SPEED_FACTOR
                + self.sample_spacing,
            )
            if segment_distance > maximum_continuous_distance:
                self._mark_invalid(
                    "detected a {:.2f} m discontinuity between frames {} and {}"
                    .format(segment_distance, self.end_frame, frame)
                )
                return False

        self.raw_snapshot_count += 1
        if self.start_frame is None:
            self.start_frame = frame
            self.start_simulation_time = simulation_time
        self.end_frame = frame
        self.end_simulation_time = simulation_time
        self.last_actor_simulation_time = simulation_time
        self.latest_transform = copy_transform(transform)
        self.latest_ground = copy_location(ground)

        if segment_distance <= 1e-6:
            self.last_observed_ground = copy_location(ground)
            return False

        self.recorded_distance += segment_distance
        self.maximum_start_displacement = max(
            self.maximum_start_displacement,
            route_horizontal_distance(self.path[0], ground),
        )
        self.dirty = True
        incoming_direction = (
            float(ground.x - self.last_observed_ground.x) / segment_distance,
            float(ground.y - self.last_observed_ground.y) / segment_distance,
        )
        if self.sampling_direction is None:
            self.sampling_direction = incoming_direction
        else:
            cosine = (
                self.sampling_direction[0] * incoming_direction[0]
                + self.sampling_direction[1] * incoming_direction[1]
            )
            cosine = max(-1.0, min(1.0, cosine))
            turn_degrees = math.degrees(math.acos(cosine))
            if turn_degrees >= ROUTE_RECORD_CONTROL_TURN_DEGREES:
                pivot_distance = route_horizontal_distance(
                    self.path[-1],
                    self.last_observed_ground,
                )
                if pivot_distance > 1e-4:
                    if len(self.path) >= MAX_ROUTE_GUIDE_POINTS - 1:
                        self.capacity_reached = True
                        self.latest_ground = copy_location(self.path[-1])
                        LOG.warning(
                            "Route recording reached the %d-point limit and "
                            "was stopped at frame %d",
                            MAX_ROUTE_GUIDE_POINTS,
                            frame,
                        )
                        return True
                    self.path.append(copy_location(self.last_observed_ground))
                self.distance_since_output = 0.0
                self.sampling_direction = incoming_direction

        segment_start = copy_location(self.last_observed_ground)
        remaining = segment_distance
        while self.distance_since_output + remaining >= self.sample_spacing:
            if len(self.path) >= MAX_ROUTE_GUIDE_POINTS - 1:
                self.capacity_reached = True
                self.latest_ground = copy_location(self.path[-1])
                LOG.warning(
                    "Route recording reached the %d-point limit and was "
                    "stopped at frame %d",
                    MAX_ROUTE_GUIDE_POINTS,
                    frame,
                )
                break
            needed = self.sample_spacing - self.distance_since_output
            fraction = needed / remaining
            sample = carla.Location(
                x=segment_start.x + (ground.x - segment_start.x) * fraction,
                y=segment_start.y + (ground.y - segment_start.y) * fraction,
                z=segment_start.z + (ground.z - segment_start.z) * fraction,
            )
            self.path.append(sample)
            self.sampling_direction = incoming_direction
            segment_start = sample
            remaining -= needed
            self.distance_since_output = 0.0
        if not self.capacity_reached:
            self.distance_since_output += remaining
            self.last_observed_ground = copy_location(ground)
        return True

    def _capture_checkpoint(self) -> Dict[str, object]:
        """Capture O(1) immutable metadata around an append-only path prefix."""
        if self.invalid_reason is not None:
            raise RuntimeError(self.invalid_reason)
        # Existing path entries are never mutated; reset() replaces the list.
        # The worker can therefore copy this fixed prefix without racing the
        # 20 Hz UI loop while capture itself remains constant time.
        return {
            "path_reference": self.path,
            "path_length": len(self.path),
            "latest_ground": copy_location(self.latest_ground),
            "latest_transform": copy_transform(self.latest_transform),
            "capacity_reached": self.capacity_reached,
            "recorded_distance": self.recorded_distance,
            "maximum_start_displacement": self.maximum_start_displacement,
            "start_transform": copy_transform(self.start_transform),
            "spawn_height_offset": self.spawn_height_offset,
            "actor_type_id": self.actor_type_id,
            "created_utc": self.created_utc,
            "recording_id": self.recording_id,
            "take_number": self.take_number,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_simulation_time": self.start_simulation_time,
            "end_simulation_time": self.end_simulation_time,
            "raw_snapshot_count": self.raw_snapshot_count,
            "missed_frame_count": self.missed_frame_count,
            "missing_snapshot_count": self.missing_snapshot_count,
        }

    @staticmethod
    def _route_config_from_checkpoint(
        checkpoint: Dict[str, object],
        route_name: str,
        map_name: str,
        sample_spacing: float,
    ) -> Optional[Dict[str, object]]:
        path_reference = checkpoint["path_reference"]
        path_length = int(checkpoint["path_length"])
        route_path = [
            copy_location(path_reference[index])
            for index in range(path_length)
        ]
        latest_ground = checkpoint["latest_ground"]
        if not checkpoint["capacity_reached"]:
            final_distance = route_horizontal_distance(
                route_path[-1], latest_ground
            )
            final_vertical = abs(
                float(route_path[-1].z - latest_ground.z)
            )
            if final_distance > 1e-4 or final_vertical > 1e-4:
                if len(route_path) < MAX_ROUTE_GUIDE_POINTS:
                    route_path.append(copy_location(latest_ground))
                else:
                    route_path[-1] = copy_location(latest_ground)

        if (
            float(checkpoint["recorded_distance"])
            < ROUTE_RECORD_MIN_DISTANCE_M
            or float(checkpoint["maximum_start_displacement"])
            < ROUTE_RECORD_MIN_DISTANCE_M
            or len(route_path) < 2
            or (
                len(route_path) == 2
                and route_horizontal_distance(route_path[0], route_path[-1])
                < ROUTE_RECORD_MIN_DISTANCE_M
            )
        ):
            return None

        intermediates = recorded_route_intermediate_waypoints(route_path)
        end_transform = copy_transform(checkpoint["latest_transform"])
        end_transform.location = copy_location(route_path[-1])
        return {
            "schema_version": PEDESTRIAN_ROUTE_SCHEMA_VERSION,
            "type": PEDESTRIAN_ROUTE_CONFIG_TYPE,
            "name": route_name,
            "map": map_name,
            "coordinate_system": PEDESTRIAN_ROUTE_COORDINATE_SYSTEM,
            "route_sampling_resolution_m": sample_spacing,
            "spawn_height_offset_m": checkpoint["spawn_height_offset"],
            "start": route_transform_payload(checkpoint["start_transform"]),
            "intermediate_waypoints": [
                route_location_payload(location) for location in intermediates
            ],
            "end": route_transform_payload(end_transform),
            "planned_path": [
                route_location_payload(location) for location in route_path
            ],
            "ui_selection": {
                "producer": os.path.basename(__file__),
                "selection_basis": (
                    "recorded_world_snapshot_coordinates_no_catalog_indices"
                ),
            },
            "created_utc": checkpoint["created_utc"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "recording": {
                "recording_id": checkpoint["recording_id"],
                "take_number": checkpoint["take_number"],
                "actor_blueprint": checkpoint["actor_type_id"],
                "sampling_mode": "world_snapshot_horizontal_arclength",
                "sample_spacing_m": sample_spacing,
                "start_frame": checkpoint["start_frame"],
                "end_frame": checkpoint["end_frame"],
                "start_simulation_time_s": checkpoint[
                    "start_simulation_time"
                ],
                "end_simulation_time_s": checkpoint["end_simulation_time"],
                "raw_snapshot_count": checkpoint["raw_snapshot_count"],
                "missed_frame_count": checkpoint["missed_frame_count"],
                "missing_actor_snapshot_count": checkpoint[
                    "missing_snapshot_count"
                ],
                "recorded_distance_m": checkpoint["recorded_distance"],
                "maximum_start_displacement_m": checkpoint[
                    "maximum_start_displacement"
                ],
                "saved_guide_point_count": len(route_path),
                "capacity_reached": checkpoint["capacity_reached"],
            },
        }

    def _checkpoint_is_saveable(self) -> bool:
        if self.recorded_distance < ROUTE_RECORD_MIN_DISTANCE_M:
            return False
        if self.maximum_start_displacement < ROUTE_RECORD_MIN_DISTANCE_M:
            return False
        final_is_distinct = (
            route_horizontal_distance(self.path[-1], self.latest_ground) > 1e-4
            or abs(float(self.path[-1].z - self.latest_ground.z)) > 1e-4
        )
        point_count = len(self.path) + int(
            final_is_distinct and not self.capacity_reached
        )
        if point_count < 2:
            return False
        if point_count == 2:
            final_point = self.latest_ground if final_is_distinct else self.path[-1]
            return (
                route_horizontal_distance(self.path[0], final_point)
                >= ROUTE_RECORD_MIN_DISTANCE_M
            )
        return True

    def build_route_config(self) -> Optional[Dict[str, object]]:
        """Build one schema-valid immutable checkpoint, or None if too short."""
        checkpoint = self._capture_checkpoint()
        return self._route_config_from_checkpoint(
            checkpoint,
            self.route_name,
            self.map_name,
            self.sample_spacing,
        )

    def maybe_checkpoint(self, now: Optional[float] = None) -> bool:
        """Queue a checkpoint, backing off as the serialized route grows."""
        current_time = time.monotonic() if now is None else float(now)
        adaptive_interval = min(
            ROUTE_RECORD_MAX_ADAPTIVE_CHECKPOINT_SECONDS,
            len(self.path) / ROUTE_RECORD_CHECKPOINT_POINTS_PER_SECOND,
        )
        checkpoint_interval = max(
            self.checkpoint_seconds,
            adaptive_interval,
        )
        if (
            not self.dirty
            or current_time - self.last_checkpoint_wall_time
            < checkpoint_interval
        ):
            return False
        return self.checkpoint_now(current_time)

    def checkpoint_now(
        self,
        now: Optional[float] = None,
        wait: bool = False,
    ) -> bool:
        """Queue the current valid take immediately, if it is long enough."""
        current_time = time.monotonic() if now is None else float(now)
        self.last_checkpoint_wall_time = current_time
        if self.invalid_reason is not None:
            raise RuntimeError(self.invalid_reason)
        if not self._checkpoint_is_saveable():
            return False
        checkpoint = self._capture_checkpoint()
        generation = self.writer.submit(
            lambda checkpoint=checkpoint: self._route_config_from_checkpoint(
                checkpoint,
                self.route_name,
                self.map_name,
                self.sample_spacing,
            )
        )
        if wait:
            self.writer.wait_for(
                generation,
                ROUTE_RECORD_WRITER_JOIN_SECONDS,
            )
        self.dirty = False
        return True

    def consume_writer_error(self):
        error = self.writer.consume_error()
        if error is not None:
            # Retry on the next checkpoint interval if additional movement has
            # not already made the take dirty.
            self.dirty = True
        return error

    def status_text(self) -> str:
        if self.invalid_reason is not None:
            state = "INVALID"
        elif self.capacity_reached:
            state = "FULL"
        else:
            state = "REC"
        saved_points = self.writer.last_saved_point_count
        return (
            "{} take {}: {} points / {:.1f} m; checkpoint {} -> {}"
            .format(
                state,
                self.take_number,
                len(self.path),
                self.recorded_distance,
                saved_points,
                self.output_path,
            )
        )

    def finish(self) -> Optional[Dict[str, object]]:
        """Commit the final valid take and wait for the bounded writer."""
        route_data = None
        build_error = None
        try:
            route_data = self.build_route_config()
        except RuntimeError as exc:
            build_error = exc
        try:
            self.writer.close(route_data)
        except RuntimeError as exc:
            if build_error is None:
                build_error = exc
        if build_error is not None:
            raise build_error
        return route_data


def consume_route_recording_snapshots(
    recorder: PedestrianRouteRecorder,
    snapshot_buffer: RecentWorldSnapshots,
) -> int:
    """Feed every cached unseen CARLA frame to the recorder in order."""
    consumed = 0
    for snapshot in snapshot_buffer.newer_than(recorder.last_frame):
        recorder.observe_snapshot(snapshot)
        consumed += 1
    return consumed


def densify_route_controls(
    controls: Sequence[carla.Location],
    resolution: float,
) -> List[carla.Location]:
    """Interpolate the operator-authored pedestrian control polyline."""
    spacing = float(resolution)
    if not (
        math.isfinite(spacing)
        and MIN_ROUTE_SAMPLING_RESOLUTION_M
        <= spacing
        <= MAX_ROUTE_SAMPLING_RESOLUTION_M
    ):
        raise ValueError(
            "pedestrian route sampling resolution must be between {:.2f} "
            "and {:.1f} meters".format(
                MIN_ROUTE_SAMPLING_RESOLUTION_M,
                MAX_ROUTE_SAMPLING_RESOLUTION_M,
            )
        )
    normalized = dedupe_route_locations(controls, minimum_distance=0.05)
    if len(normalized) < 2:
        raise ValueError("pedestrian route needs two distinct control points")
    if len(normalized) > MAX_ROUTE_CONTROL_POINTS:
        raise ValueError(
            "pedestrian route exceeds the {}-control-point limit".format(
                MAX_ROUTE_CONTROL_POINTS
            )
        )
    result = [copy_location(normalized[0])]
    for start, end in zip(normalized, normalized[1:]):
        distance = route_distance(start, end)
        step_count = distance / spacing
        if not math.isfinite(distance) or not math.isfinite(step_count):
            raise ValueError("pedestrian route contains an unbounded segment")
        steps = max(1, int(math.ceil(step_count)))
        if len(result) + steps > MAX_ROUTE_GUIDE_POINTS:
            raise ValueError(
                "pedestrian route would exceed the {}-point visual-guide "
                "limit".format(MAX_ROUTE_GUIDE_POINTS)
            )
        for step in range(1, steps + 1):
            fraction = step / float(steps)
            result.append(
                carla.Location(
                    x=start.x + (end.x - start.x) * fraction,
                    y=start.y + (end.y - start.y) * fraction,
                    z=start.z + (end.z - start.z) * fraction,
                )
            )
    return result


def resolve_route_guidance(route_config):
    """Return the exact spawn transform and a safe dense visual route.

    The saved start transform is actor-space authoritative and is never
    projected or height-adjusted. ``spawn_height_offset_m`` is used only to
    lower the fallback visual marker from actor origin to navigation ground.
    """
    spawn_transform = route_transform(route_config["start"])
    sampling_resolution = float(route_config["route_sampling_resolution_m"])
    visual_start = copy_location(spawn_transform.location)
    if "spawn_height_offset_m" in route_config:
        visual_start.z -= float(route_config["spawn_height_offset_m"])
    controls = [visual_start]
    controls.extend(
        route_location(value)
        for value in route_config["intermediate_waypoints"]
    )
    controls.append(route_location(route_config["end"]["location"]))
    controls = dedupe_route_locations(controls, minimum_distance=0.05)
    if len(controls) < 2:
        raise ValueError("loaded pedestrian route has fewer than two controls")
    if len(controls) > MAX_ROUTE_CONTROL_POINTS:
        raise ValueError(
            "loaded pedestrian route exceeds the {}-control-point limit".format(
                MAX_ROUTE_CONTROL_POINTS
            )
        )

    planned_path = dedupe_route_locations(
        [route_location(value) for value in route_config.get("planned_path", [])],
        minimum_distance=0.01,
    )
    if len(planned_path) > MAX_ROUTE_GUIDE_POINTS:
        raise ValueError(
            "loaded pedestrian planned_path exceeds the {}-point limit".format(
                MAX_ROUTE_GUIDE_POINTS
            )
        )
    planned_path_valid = len(planned_path) >= 2
    if planned_path_valid:
        planned_path_valid = (
            route_distance(planned_path[0], controls[0])
            <= ROUTE_ENDPOINT_TOLERANCE_M
            and route_distance(planned_path[-1], controls[-1])
            <= ROUTE_ENDPOINT_TOLERANCE_M
            and abs(float(planned_path[0].z - controls[0].z))
            <= ROUTE_VERTICAL_TOLERANCE_M
            and abs(float(planned_path[-1].z - controls[-1].z))
            <= ROUTE_VERTICAL_TOLERANCE_M
        )
    if planned_path_valid and len(controls) > 2:
        control_index = 1
        for location in planned_path[1:-1]:
            control = controls[control_index]
            if (
                route_distance(location, control) <= ROUTE_CONTROL_TOLERANCE_M
                and abs(float(location.z - control.z))
                <= ROUTE_VERTICAL_TOLERANCE_M
            ):
                control_index += 1
                if control_index == len(controls) - 1:
                    break
        planned_path_valid = control_index == len(controls) - 1
    if planned_path_valid:
        maximum_gap = max(5.0, sampling_resolution * 3.0)
        planned_path_valid = all(
            route_distance(first, second) <= maximum_gap
            for first, second in zip(planned_path, planned_path[1:])
        )

    if planned_path_valid:
        return spawn_transform, planned_path, "saved planned_path"
    return (
        spawn_transform,
        densify_route_controls(controls, sampling_resolution),
        "densified route controls",
    )


class PedestrianRouteGuidance:
    """Bounded-progress, frame-local route projection for the RGB stream."""

    def __init__(self, route_path: Sequence[carla.Location], fov: float) -> None:
        self.route_path = dedupe_route_locations(route_path)
        if len(self.route_path) < 2:
            raise ValueError("route guidance needs at least two path points")
        if len(self.route_path) > MAX_ROUTE_GUIDE_POINTS:
            raise ValueError(
                "route guidance exceeds the {}-point limit".format(
                    MAX_ROUTE_GUIDE_POINTS
                )
            )
        self.fov = float(fov)
        self.progress_index = 0
        self._progress_acquired = False

    def reset(self) -> None:
        """Restart route acquisition after an exact Y respawn."""
        self.progress_index = 0
        self._progress_acquired = False

    @staticmethod
    def _project_to_segment(
        start: carla.Location,
        end: carla.Location,
        target: carla.Location,
    ) -> carla.Location:
        delta_x = float(end.x - start.x)
        delta_y = float(end.y - start.y)
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared < 1e-9:
            return copy_location(start)
        ratio = (
            (float(target.x - start.x) * delta_x)
            + (float(target.y - start.y) * delta_y)
        ) / length_squared
        ratio = max(0.0, min(1.0, ratio))
        return carla.Location(
            x=start.x + delta_x * ratio,
            y=start.y + delta_y * ratio,
            z=start.z + (end.z - start.z) * ratio,
        )

    def locations_ahead(
        self,
        walker_location: carla.Location,
        max_distance: float = ROUTE_GUIDANCE_LOOKAHEAD_M,
    ) -> List[carla.Location]:
        """Return a bounded lookahead from the segment nearest the walker."""
        route = self.route_path
        if self._progress_acquired:
            # Never reconsider a segment behind committed progress. At a
            # self-crossing, choosing a closer prior branch and then clamping
            # its index would project the guide onto the wrong future segment.
            search_start = self.progress_index
            search_stop = min(
                len(route) - 1,
                self.progress_index + ROUTE_PROGRESS_SEARCH_SEGMENTS,
            )
        else:
            search_start = 0
            search_stop = len(route) - 1

        best_index = search_start
        best_location = copy_location(route[search_start])
        best_score = float("inf")
        tie_break_weight = 0.01 if self._progress_acquired else 1e-6
        for index in range(search_start, search_stop):
            projected = self._project_to_segment(
                route[index], route[index + 1], walker_location
            )
            score = route_horizontal_distance(projected, walker_location)
            score += tie_break_weight * max(0, index - self.progress_index)
            if score < best_score:
                best_index = index
                best_location = projected
                best_score = score

        self.progress_index = best_index
        self._progress_acquired = True

        selected = [best_location]
        travelled = 0.0
        previous = best_location
        lookahead_stop = min(
            len(route),
            best_index + 1 + ROUTE_LOOKAHEAD_MAX_SEGMENTS,
        )
        for index in range(best_index + 1, lookahead_stop):
            location = route[index]
            segment_length = route_distance(previous, location)
            if segment_length < 0.001:
                previous = location
                continue
            if travelled + segment_length >= max_distance:
                remaining = max(0.0, max_distance - travelled)
                ratio = remaining / segment_length
                selected.append(
                    carla.Location(
                        x=previous.x + (location.x - previous.x) * ratio,
                        y=previous.y + (location.y - previous.y) * ratio,
                        z=previous.z + (location.z - previous.z) * ratio,
                    )
                )
                break
            selected.append(copy_location(location))
            travelled += segment_length
            previous = location
        return selected if len(selected) >= 2 else []

    @staticmethod
    def sample_arrow_polygons(
        route_locations: Sequence[carla.Location],
    ) -> List[List[carla.Location]]:
        """Sample pedestrian-scale arrow polygons along a world polyline."""
        if len(route_locations) < 2:
            return []
        arrows = []
        travelled = 0.0
        next_arrow_distance = ROUTE_ARROW_START_M
        half_length = ROUTE_ARROW_LENGTH_M * 0.5
        half_width = ROUTE_ARROW_WIDTH_M * 0.5
        shaft_half_width = half_width * 0.34
        shoulder = ROUTE_ARROW_LENGTH_M * 0.08
        for start, end in zip(route_locations, route_locations[1:]):
            delta_x = float(end.x - start.x)
            delta_y = float(end.y - start.y)
            segment_length = math.hypot(delta_x, delta_y)
            if segment_length < 1e-6:
                continue
            tangent_x = delta_x / segment_length
            tangent_y = delta_y / segment_length
            right_x = -tangent_y
            right_y = tangent_x
            while next_arrow_distance <= travelled + segment_length:
                ratio = (next_arrow_distance - travelled) / segment_length
                center_x = start.x + delta_x * ratio
                center_y = start.y + delta_y * ratio
                center_z = (
                    start.z + (end.z - start.z) * ratio + ROUTE_GROUND_LIFT_M
                )
                local_points = (
                    (-half_length, -shaft_half_width),
                    (shoulder, -shaft_half_width),
                    (shoulder, -half_width),
                    (half_length, 0.0),
                    (shoulder, half_width),
                    (shoulder, shaft_half_width),
                    (-half_length, shaft_half_width),
                )
                arrows.append(
                    [
                        carla.Location(
                            x=center_x + tangent_x * forward + right_x * lateral,
                            y=center_y + tangent_y * forward + right_y * lateral,
                            z=center_z,
                        )
                        for forward, lateral in local_points
                    ]
                )
                next_arrow_distance += ROUTE_ARROW_SPACING_M
            travelled += segment_length
        return arrows

    @staticmethod
    def _camera_points(
        locations: Sequence[carla.Location],
        camera_transform: carla.Transform,
    ) -> np.ndarray:
        world_points = np.asarray(
            [[point.x, point.y, point.z, 1.0] for point in locations],
            dtype=np.float64,
        )
        inverse = np.asarray(
            camera_transform.get_inverse_matrix(), dtype=np.float64
        )
        return (inverse @ world_points.T).T[:, :3]

    @classmethod
    def _project_polygon(
        cls,
        locations: Sequence[carla.Location],
        camera_transform: carla.Transform,
        calibration: np.ndarray,
        width: int,
        height: int,
    ):
        points_camera = cls._camera_points(locations, camera_transform)
        depth = points_camera[:, 0]
        if np.any(depth <= 0.25):
            return None
        horizontal = (
            calibration[0, 2]
            + points_camera[:, 1] / depth * calibration[0, 0]
        )
        vertical = (
            calibration[1, 2]
            - points_camera[:, 2] / depth * calibration[1, 1]
        )
        if not np.all(np.isfinite(horizontal)) or not np.all(np.isfinite(vertical)):
            return None
        margin = 240
        if (
            np.all(horizontal < -margin)
            or np.all(horizontal > width + margin)
            or np.all(vertical < -margin)
            or np.all(vertical > height + margin)
        ):
            return None
        return [
            (int(round(x_coord)), int(round(y_coord)))
            for x_coord, y_coord in zip(horizontal, vertical)
        ]

    @classmethod
    def _project_segments(
        cls,
        locations: Sequence[carla.Location],
        camera_transform: carla.Transform,
        calibration: np.ndarray,
        width: int,
        height: int,
    ) -> List[List[Tuple[int, int]]]:
        lifted = [
            carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + ROUTE_GROUND_LIFT_M,
            )
            for location in locations
        ]
        if not lifted:
            return []
        points_camera = cls._camera_points(lifted, camera_transform)
        depth = points_camera[:, 0]
        safe_depth = np.maximum(depth, 1e-3)
        horizontal = (
            calibration[0, 2]
            + points_camera[:, 1] / safe_depth * calibration[0, 0]
        )
        vertical = (
            calibration[1, 2]
            - points_camera[:, 2] / safe_depth * calibration[1, 1]
        )
        margin = 120
        segments: List[List[Tuple[int, int]]] = []
        segment: List[Tuple[int, int]] = []
        for x_depth, x_coord, y_coord in zip(depth, horizontal, vertical):
            visible = (
                x_depth > 0.25
                and -margin <= x_coord <= width + margin
                and -margin <= y_coord <= height + margin
            )
            if visible:
                segment.append((int(round(x_coord)), int(round(y_coord))))
            elif segment:
                if len(segment) >= 2:
                    segments.append(segment)
                segment = []
        if len(segment) >= 2:
            segments.append(segment)
        return segments

    def render_overlay(
        self,
        width: int,
        height: int,
        camera_transform: carla.Transform,
        walker_location: carla.Location,
    ) -> Optional[pygame.Surface]:
        """Render one alpha overlay from one image/snapshot-aligned pose pair."""
        locations = self.locations_ahead(walker_location)
        if len(locations) < 2:
            return None
        calibration = camera_calibration(width, height, self.fov)
        segments = self._project_segments(
            locations, camera_transform, calibration, width, height
        )
        projected_arrows = []
        for polygon in self.sample_arrow_polygons(locations):
            projected = self._project_polygon(
                polygon, camera_transform, calibration, width, height
            )
            if projected is not None:
                projected_arrows.append(projected)
        if not segments and not projected_arrows:
            return None
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        line_width = max(2, int(width / 480))
        for segment in segments:
            pygame.draw.lines(
                overlay, ROUTE_OUTLINE_COLOR, False, segment, line_width + 3
            )
            pygame.draw.lines(
                overlay, ROUTE_COLOR, False, segment, line_width
            )
        for projected in projected_arrows:
            pygame.draw.polygon(overlay, ROUTE_ARROW_COLOR, projected)
            pygame.draw.polygon(overlay, ROUTE_ARROW_OUTLINE_COLOR, projected, 2)
        return overlay


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


class LiveMetricsModel:
    """Collect measured client proxies and reproducible DEMO-only values."""

    def __init__(self, placeholder_seed: int = DEFAULT_METRICS_PLACEHOLDER_SEED):
        self._rng = random.Random(int(placeholder_seed))
        self._next_placeholder_update_at = 0.0
        self._spatial_accuracy_cm = METRICS_ACCURACY_MEAN_CM
        self._ai_reasoning_ms = METRICS_REASONING_MEAN_MS
        self.reset_live_measurements()

    @staticmethod
    def _ewma(previous, current: float, alpha: float) -> float:
        if previous is None:
            return float(current)
        return (1.0 - float(alpha)) * float(previous) + float(alpha) * float(current)

    def _bounded_gaussian(
        self,
        mean: float,
        sigma: float,
        lower: float,
        upper: float,
    ) -> float:
        return min(float(upper), max(float(lower), self._rng.gauss(mean, sigma)))

    def _update_placeholders(self, now: float) -> None:
        if now + 1.0e-9 < self._next_placeholder_update_at:
            return
        self._next_placeholder_update_at = (
            now + METRICS_PLACEHOLDER_REFRESH_SECONDS
        )
        accuracy_sample = self._bounded_gaussian(
            METRICS_ACCURACY_MEAN_CM,
            METRICS_ACCURACY_SIGMA_CM,
            METRICS_ACCURACY_MIN_CM,
            METRICS_ACCURACY_MAX_CM,
        )
        reasoning_sample = self._bounded_gaussian(
            METRICS_REASONING_MEAN_MS,
            METRICS_REASONING_SIGMA_MS,
            METRICS_REASONING_MIN_MS,
            METRICS_REASONING_MAX_MS,
        )
        self._spatial_accuracy_cm = min(
            METRICS_ACCURACY_MAX_CM,
            max(
                METRICS_ACCURACY_MIN_CM,
                self._ewma(
                    self._spatial_accuracy_cm,
                    accuracy_sample,
                    METRICS_PLACEHOLDER_EWMA_ALPHA,
                ),
            ),
        )
        self._ai_reasoning_ms = min(
            METRICS_REASONING_MAX_MS,
            max(
                METRICS_REASONING_MIN_MS,
                self._ewma(
                    self._ai_reasoning_ms,
                    reasoning_sample,
                    METRICS_PLACEHOLDER_EWMA_ALPHA,
                ),
            ),
        )

    def reset_live_measurements(self) -> None:
        """Reset values tied to the current walker/camera lifetime."""
        self.events_detected = 0
        self._events_detected_sampled_at = None
        self.spatial_map_latency_ms = None
        self._spatial_map_latency_sampled_at = None
        self.sense_to_act_latency_ms = None
        self._sense_to_act_sensed_at = None
        self._overlap_detected = False

    @staticmethod
    def _sample_time(sampled_at) -> Optional[float]:
        try:
            value = time.perf_counter() if sampled_at is None else float(sampled_at)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def note_events_detected(self, count, sampled_at=None) -> None:
        sample_time = self._sample_time(sampled_at)
        if sample_time is None:
            return
        try:
            count = max(0, int(count))
        except (TypeError, ValueError, OverflowError):
            return
        self.events_detected = count
        self._events_detected_sampled_at = sample_time

    def clear_events_detected(self) -> None:
        self.events_detected = 0
        self._events_detected_sampled_at = None

    def note_spatial_map_latency(self, latency_ms, sampled_at=None) -> None:
        sample_time = self._sample_time(sampled_at)
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError, OverflowError):
            return
        if (
            sample_time is None
            or not math.isfinite(latency_ms)
            or latency_ms < 0.0
        ):
            return
        self.spatial_map_latency_ms = self._ewma(
            self.spatial_map_latency_ms,
            latency_ms,
            METRICS_MEASURED_EWMA_ALPHA,
        )
        self._spatial_map_latency_sampled_at = sample_time

    def note_control_submitted(self, sensed_at, submitted_at=None) -> None:
        if sensed_at is None:
            return
        if submitted_at is None:
            submitted_at = time.perf_counter()
        try:
            sensed_at = float(sensed_at)
            latency_seconds = float(submitted_at) - sensed_at
        except (TypeError, ValueError, OverflowError):
            return
        if (
            not math.isfinite(latency_seconds)
            or latency_seconds < 0.0
            or latency_seconds > METRICS_MAX_SENSOR_AGE_SECONDS
        ):
            if (
                math.isfinite(latency_seconds)
                and latency_seconds > METRICS_MAX_SENSOR_AGE_SECONDS
            ):
                self.sense_to_act_latency_ms = None
                self._sense_to_act_sensed_at = None
            return
        self.sense_to_act_latency_ms = self._ewma(
            self.sense_to_act_latency_ms,
            latency_seconds * 1000.0,
            METRICS_MEASURED_EWMA_ALPHA,
        )
        self._sense_to_act_sensed_at = sensed_at

    def note_overlap_detected(self, detected: bool) -> None:
        """Latch the snapshot bounding-box overlap proxy for this walker life."""
        if detected:
            self._overlap_detected = True

    def snapshot(self, now=None) -> Dict[str, object]:
        if now is None:
            now = time.perf_counter()
        now = float(now)
        self._update_placeholders(now)

        events_detected = int(self.events_detected)
        if (
            self._events_detected_sampled_at is None
            or now - self._events_detected_sampled_at
            > METRICS_MAX_SENSOR_AGE_SECONDS
        ):
            events_detected = None

        map_latency = self.spatial_map_latency_ms
        if (
            self._spatial_map_latency_sampled_at is None
            or now - self._spatial_map_latency_sampled_at
            > METRICS_MAX_MAP_SAMPLE_AGE_SECONDS
        ):
            map_latency = None

        sense_to_act = self.sense_to_act_latency_ms
        if (
            self._sense_to_act_sensed_at is None
            or now - self._sense_to_act_sensed_at
            > METRICS_MAX_SENSOR_AGE_SECONDS
        ):
            sense_to_act = None

        return {
            "events_detected": events_detected,
            "spatial_map_latency_ms": map_latency,
            "spatial_map_accuracy_cm": float(self._spatial_accuracy_cm),
            "ai_reasoning_ms": float(self._ai_reasoning_ms),
            "sense_to_act_latency_ms": sense_to_act,
            "outcome": (
                "HAZARD NOT AVOIDED"
                if self._overlap_detected
                else "HAZARD AVOIDED"
            ),
        }


class LiveMetricsRenderer:
    """Render the supplied six-column design in a separate OpenCV window."""

    def __init__(
        self,
        width: int = LIVE_METRICS_WINDOW_WIDTH,
        height: int = LIVE_METRICS_WINDOW_HEIGHT,
        refresh_hz: float = LIVE_METRICS_REFRESH_HZ,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._window_name = "CARLA Pedestrian Live Physical AI Metrics"
        self._window_created = False
        self._last_refresh_at = None
        self._refresh_period = 1.0 / max(0.1, float(refresh_hz))
        self.ready = cv2 is not None

    @staticmethod
    def _format_measurement(value, unit: str) -> str:
        if value is None:
            return "-- {}".format(unit)
        return "{:.1f} {}".format(float(value), unit)

    @staticmethod
    def _draw_centered_text(
        image,
        text: str,
        center_x: float,
        baseline_y: int,
        font_scale: float,
        color,
        thickness: int = 1,
    ) -> None:
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        origin_x = int(round(center_x - size[0] * 0.5))
        cv2.putText(
            image,
            text,
            (origin_x, int(baseline_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def build_frame(self, metrics: Dict[str, object]) -> np.ndarray:
        frame = np.full(
            (self._height, self._width, 3),
            (18, 15, 13),
            dtype=np.uint8,
        )
        header_height = 27
        footer_height = 23
        card_top = header_height
        card_bottom = self._height - footer_height
        card_width = float(self._width) / 6.0
        columns = (
            (
                "SENSE",
                "Events Detected",
                "--"
                if metrics["events_detected"] is None
                else str(int(metrics["events_detected"])),
                (218, 58, 255),
                "LIVE",
            ),
            (
                "SPATIAL MAP",
                "Spatial Map Latency",
                self._format_measurement(metrics["spatial_map_latency_ms"], "ms"),
                (239, 190, 48),
                "LIVE LOCAL",
            ),
            (
                "SPATIAL MAP",
                "Spatial Map Accuracy",
                "{:.1f} cm*".format(metrics["spatial_map_accuracy_cm"]),
                (105, 220, 87),
                "DEMO",
            ),
            (
                "AI REASONING",
                "Reasoning Latency",
                "{:.1f} ms*".format(metrics["ai_reasoning_ms"]),
                (54, 215, 249),
                "DEMO",
            ),
            (
                "ACT",
                "Sense-to-Act Latency",
                self._format_measurement(metrics["sense_to_act_latency_ms"], "ms"),
                (51, 149, 255),
                "LIVE LOCAL",
            ),
            (
                "OUTCOME",
                "BBox Overlap Proxy",
                metrics["outcome"],
                (
                    (88, 225, 120)
                    if metrics["outcome"] == "HAZARD AVOIDED"
                    else (77, 88, 255)
                ),
                "LIVE PROXY",
            ),
        )

        cv2.putText(
            frame,
            "LIVE EGO-PEDESTRIAN PHYSICAL AI METRICS",
            (14, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 224, 230),
            1,
            cv2.LINE_AA,
        )
        for index, (category, label, value, color, source) in enumerate(columns):
            left = int(round(index * card_width))
            right = int(round((index + 1) * card_width))
            center_x = (left + right) * 0.5
            if index > 0:
                cv2.line(
                    frame,
                    (left, card_top + 5),
                    (left, card_bottom - 5),
                    (56, 52, 50),
                    1,
                    cv2.LINE_AA,
                )
            cv2.rectangle(
                frame,
                (left + 9, card_top + 9),
                (left + 14, card_top + 26),
                color,
                -1,
            )
            cv2.putText(
                frame,
                category,
                (left + 22, card_top + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
            self._draw_centered_text(
                frame,
                label,
                center_x,
                card_top + 50,
                0.42,
                (182, 187, 194),
            )
            value_scale = 0.55 if index == 5 else 0.74
            self._draw_centered_text(
                frame,
                value,
                center_x,
                card_top + 91,
                value_scale,
                color,
                2,
            )
            self._draw_centered_text(
                frame,
                source,
                center_x,
                card_bottom - 8,
                0.32,
                (112, 117, 124),
            )

        cv2.putText(
            frame,
            "* Deterministic bounded placeholder; other values are local client proxies",
            (12, self._height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (137, 142, 150),
            1,
            cv2.LINE_AA,
        )
        return frame

    def render(self, metrics: Dict[str, object]) -> bool:
        if not self.ready:
            return False
        now = time.perf_counter()
        if (
            self._last_refresh_at is not None
            and now - self._last_refresh_at < self._refresh_period
        ):
            return True
        self._last_refresh_at = now
        try:
            if self._window_created:
                visible = cv2.getWindowProperty(
                    self._window_name,
                    cv2.WND_PROP_VISIBLE,
                )
                if visible < 1.0:
                    self.close()
                    self.ready = False
                    return False
            else:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name, self._width, self._height)
                self._window_created = True
            cv2.imshow(self._window_name, self.build_frame(metrics))
            cv2.waitKey(1)
        except cv2.error:
            self.close()
            self.ready = False
            return False
        return True

    def close(self) -> None:
        self._last_refresh_at = None
        if cv2 is None or not self._window_created:
            return
        try:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._window_created = False


def actor_footprint_points(
    actor: carla.Actor,
    actor_transform: carla.Transform,
) -> np.ndarray:
    """Return the actor bounding box's four ordered world-XY footprint points."""
    vertices = actor.bounding_box.get_world_vertices(actor_transform)
    points = sorted(
        set(
            (round(float(vertex.x), 5), round(float(vertex.y), 5))
            for vertex in vertices
            if math.isfinite(float(vertex.x)) and math.isfinite(float(vertex.y))
        )
    )
    if len(points) < 3:
        raise ValueError("actor bounding box has no drawable XY footprint")

    def cross(origin, first, second) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("actor bounding box has no convex XY footprint")
    return np.asarray(hull, dtype=np.float32)


def convex_polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Return whether two convex XY polygons overlap using separating axes."""
    first = np.asarray(first, dtype=np.float64).reshape((-1, 2))
    second = np.asarray(second, dtype=np.float64).reshape((-1, 2))
    if len(first) < 3 or len(second) < 3:
        return False
    for polygon in (first, second):
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            axis = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            if float(np.dot(axis, axis)) <= 1.0e-12:
                continue
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                float(np.max(first_projection))
                < float(np.min(second_projection)) - 1.0e-6
                or float(np.max(second_projection))
                < float(np.min(first_projection)) - 1.0e-6
            ):
                return False
    return True


def snapshot_bbox_overlap_proxy(
    ego_actor: carla.Actor,
    actors: Sequence[carla.Actor],
    snapshot,
) -> bool:
    """Detect snapshot-aligned 3D-box overlap for the outcome proxy."""
    try:
        ego_snapshot = snapshot.find(int(ego_actor.id))
        if ego_snapshot is None:
            return False
        ego_transform = ego_snapshot.get_transform()
        ego_vertices = ego_actor.bounding_box.get_world_vertices(ego_transform)
        ego_footprint = actor_footprint_points(ego_actor, ego_transform)
        ego_z_min = min(float(vertex.z) for vertex in ego_vertices)
        ego_z_max = max(float(vertex.z) for vertex in ego_vertices)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False

    for actor in actors:
        try:
            if int(actor.id) == int(ego_actor.id):
                continue
            actor_snapshot = snapshot.find(int(actor.id))
            if actor_snapshot is None:
                continue
            actor_transform = actor_snapshot.get_transform()
            if horizontal_distance(
                ego_transform.location,
                actor_transform.location,
            ) > 15.0:
                continue
            actor_vertices = actor.bounding_box.get_world_vertices(actor_transform)
            actor_z_min = min(float(vertex.z) for vertex in actor_vertices)
            actor_z_max = max(float(vertex.z) for vertex in actor_vertices)
            if actor_z_max < ego_z_min - 0.05 or ego_z_max < actor_z_min - 0.05:
                continue
            actor_footprint = actor_footprint_points(actor, actor_transform)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if convex_polygons_overlap(ego_footprint, actor_footprint):
            return True
    return False


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
    ) -> Optional[float]:
        if not self.ready or hero_actor is None:
            return
        now = time.monotonic()
        if (
            self._last_refresh_time is not None
            and now - self._last_refresh_time < self._refresh_period_seconds
        ):
            return
        self._last_refresh_time = now
        render_started_at = time.perf_counter()

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
        # Measure the local map query/draw/composition work. OpenCV presentation
        # is intentionally excluded; this is not spatial-map-server E2E latency.
        latency_ms = (time.perf_counter() - render_started_at) * 1000.0

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
        return latency_ms

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

    def update(self, keys, delta_seconds: float) -> float:
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
        control_submitted_at = time.perf_counter()
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
        return control_submitted_at

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
    route_available: bool,
    route_enabled: bool,
    route_name: Optional[str],
    route_recording_status: Optional[str],
    metrics_enabled: bool,
) -> None:
    vehicle_boxes, pedestrian_boxes = box_counts
    spawned_npc_cars, requested_npc_cars = npc_car_counts
    lines = [
        "Pedestrian: W/S forward/back   A/D turn   Hold Shift + W/S to run",
        "Camera: Arrow Up/Down pitch   Arrow Left/Right yaw   R recenter",
        "Y: respawn/recenter at configured start   Space: jump   Esc/Q: quit",
        "Visuals: B boxes   U group: route {} / map {} ({:.1f} m) / metrics {}".format(
            "ON" if route_enabled else ("OFF" if route_available else "N/A"),
            "ON" if topdown_enabled else "OFF",
            topdown_radius,
            "ON" if metrics_enabled else "OFF",
        ),
        "Route: {}".format(route_name if route_available else "not loaded"),
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
    if route_recording_status is not None:
        lines.insert(5, "Recording: {}".format(route_recording_status))
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
    parser.add_argument(
        "--metrics-placeholder-seed",
        type=int,
        default=DEFAULT_METRICS_PLACEHOLDER_SEED,
        metavar="SEED",
        help=(
            "dedicated reproducibility seed for the DEMO-only spatial-map "
            "accuracy and AI-reasoning metrics (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--pedestrian-route-config",
        "--route-config",
        "--replay-route",
        dest="pedestrian_route_config",
        default=None,
        metavar="PATH",
        help=(
            "coordinate-based or previously recorded ego-pedestrian route "
            "JSON; its exact start "
            "transform overrides --spawn-x/--spawn-y/--spawn-z/--spawn-yaw "
            "for startup and Y respawn; press U to render its arrows"
        ),
    )
    parser.add_argument(
        "--record-route",
        default=None,
        metavar="PATH",
        help=(
            "record the keyboard-driven pedestrian trajectory to this route "
            "JSON; recording starts automatically and finalizes on exit"
        ),
    )
    parser.add_argument(
        "--record-route-spacing",
        type=float,
        default=DEFAULT_ROUTE_RECORD_SPACING_M,
        metavar="METERS",
        help=(
            "arc-length spacing between recorded guide points "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--record-route-checkpoint-seconds",
        type=float,
        default=DEFAULT_ROUTE_RECORD_CHECKPOINT_SECONDS,
        metavar="SECONDS",
        help=(
            "minimum wall-clock interval for asynchronous atomic route "
            "checkpoints (automatically lengthened for long routes) "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--record-route-name",
        default=None,
        metavar="NAME",
        help="name stored in the recorded route JSON (default: map-based name)",
    )
    parser.add_argument(
        "--record-route-overwrite",
        action="store_true",
        help="allow --record-route to replace an existing file atomically",
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

    if args.record_route is None and args.record_route_overwrite:
        parser.error("--record-route-overwrite requires --record-route PATH")
    if not (
        math.isfinite(args.record_route_spacing)
        and MIN_ROUTE_SAMPLING_RESOLUTION_M
        <= args.record_route_spacing
        <= MAX_ROUTE_SAMPLING_RESOLUTION_M
    ):
        parser.error(
            "--record-route-spacing must be between {:.2f} and {:.1f} meters"
            .format(
                MIN_ROUTE_SAMPLING_RESOLUTION_M,
                MAX_ROUTE_SAMPLING_RESOLUTION_M,
            )
        )
    if not (
        math.isfinite(args.record_route_checkpoint_seconds)
        and args.record_route_checkpoint_seconds
        >= MIN_ROUTE_RECORD_CHECKPOINT_SECONDS
    ):
        parser.error(
            "--record-route-checkpoint-seconds must be at least {:.1f}"
            .format(MIN_ROUTE_RECORD_CHECKPOINT_SECONDS)
        )
    if args.record_route_name is not None:
        args.record_route_name = args.record_route_name.strip()
        if not args.record_route_name:
            parser.error("--record-route-name must not be empty")
    if args.record_route is not None:
        args.record_route = os.path.abspath(os.fspath(args.record_route))
        output_directory = os.path.dirname(args.record_route)
        if not os.path.isdir(output_directory):
            parser.error(
                "--record-route directory does not exist: {!r}".format(
                    output_directory
                )
            )
        if os.path.isdir(args.record_route):
            parser.error("--record-route PATH must name a file, not a directory")
        if os.path.lexists(args.record_route) and not args.record_route_overwrite:
            parser.error(
                "recorded route already exists: {!r}; pass "
                "--record-route-overwrite to replace it".format(
                    args.record_route
                )
            )

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
    metrics_model = LiveMetricsModel(args.metrics_placeholder_seed)
    metrics_renderer = None
    metrics_enabled = False
    u_visuals_enabled = False
    route_config = None
    route_guidance = None
    route_name = None
    route_path_source = None
    route_guidance_enabled = False
    route_overlay = None
    route_refresh_requested = False
    route_recorder = None
    pygame_initialized = False

    try:
        # get_world() attaches to the existing map.  Do not use load_world().
        world = client.get_world()
        carla_map = world.get_map()
        resolved_spawn_transform = None
        if args.pedestrian_route_config is not None:
            # Read and validate after connecting so a map mismatch is rejected
            # before this passive client spawns any actor.
            route_config = load_route_config(args.pedestrian_route_config)
            if not maps_match(route_config["map"], carla_map.name):
                raise ValueError(
                    "pedestrian route map {!r} does not match loaded CARLA "
                    "map {!r}".format(route_config["map"], carla_map.name)
                )
            (
                resolved_spawn_transform,
                route_path,
                route_path_source,
            ) = resolve_route_guidance(route_config)
            route_name = str(route_config["name"])
            route_guidance = PedestrianRouteGuidance(route_path, args.fov)
        snapshot_buffer = RecentWorldSnapshots(
            capacity=(
                ROUTE_RECORD_SNAPSHOT_CAPACITY
                if args.record_route is not None
                else 8
            )
        )
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
            resolved_spawn_transform=resolved_spawn_transform,
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
        pygame.display.set_caption("CARLA Pedestrian Head Camera v9")
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

        if args.record_route is not None:
            default_recording_name = "{} recorded ego pedestrian route".format(
                str(carla_map.name).replace("\\", "/").rsplit("/", 1)[-1]
            )
            recording_spawn_height_offset = None
            if (
                route_config is not None
                and "spawn_height_offset_m" in route_config
            ):
                recording_spawn_height_offset = float(
                    route_config["spawn_height_offset_m"]
                )
            elif route_config is None and args.spawn_z is None:
                # fixed_sidewalk_transform() deliberately places the actor
                # this far above the selected navigation surface.
                recording_spawn_height_offset = DEFAULT_SPAWN_HEIGHT_OFFSET_M
            route_recorder = PedestrianRouteRecorder(
                output_path=args.record_route,
                map_name=carla_map.name,
                route_name=args.record_route_name or default_recording_name,
                sample_spacing=args.record_route_spacing,
                checkpoint_seconds=args.record_route_checkpoint_seconds,
                maximum_pedestrian_speed=max(args.walk_speed, args.run_speed),
                walker=walker,
                spawn_transform=spawn_transform,
                baseline_snapshot=snapshot_buffer.latest(),
                allow_overwrite=args.record_route_overwrite,
                spawn_height_offset_hint=recording_spawn_height_offset,
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
        if route_guidance is not None:
            LOG.info(
                "Loaded ego-pedestrian route %r from %s: %d visual points "
                "(%s); its exact start transform is authoritative for "
                "startup and Y respawn",
                route_name,
                args.pedestrian_route_config,
                len(route_guidance.route_path),
                route_path_source,
            )
        else:
            LOG.info(
                "No ego-pedestrian route loaded; pass "
                "--replay-route PATH to enable RGB guidance"
            )
        if route_recorder is not None:
            LOG.info(
                "Recording keyboard-driven route to %s at %.2f m spacing; "
                "atomic checkpoints start at %.1f-second intervals, back off "
                "for long routes, and finalize on exit",
                route_recorder.output_path,
                route_recorder.sample_spacing,
                route_recorder.checkpoint_seconds,
            )
            LOG.info(
                "A successful Y respawn checkpoints the current take and "
                "starts a fresh take at the configured home transform"
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
            "Press U to toggle boxes, ego-pedestrian route arrows, the "
            "following top-down map, and live metrics together (radius %.1f m)",
            args.topdown_zoom_radius,
        )
        LOG.info(
            "Live metrics: detections/map/control timing are local measured "
            "proxies; Outcome is a snapshot bbox-overlap proxy; DEMO "
            "accuracy=clipped Gaussian mean %.1f sigma %.2f range %.1f..%.1f "
            "cm and reasoning=clipped Gaussian mean %.1f sigma %.1f range "
            "%.1f..%.1f ms (seed=%d, refresh=%.1f s, EWMA alpha=%.2f)",
            METRICS_ACCURACY_MEAN_CM,
            METRICS_ACCURACY_SIGMA_CM,
            METRICS_ACCURACY_MIN_CM,
            METRICS_ACCURACY_MAX_CM,
            METRICS_REASONING_MEAN_MS,
            METRICS_REASONING_SIGMA_MS,
            METRICS_REASONING_MIN_MS,
            METRICS_REASONING_MAX_MS,
            args.metrics_placeholder_seed,
            METRICS_PLACEHOLDER_REFRESH_SECONDS,
            METRICS_PLACEHOLDER_EWMA_ALPHA,
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
        camera_frame_received_at = None
        overlap_refresh_requested = False
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
                                recording_checkpoint_ready = True
                                if route_recorder is not None:
                                    try:
                                        # Prevent the last WalkerControl from
                                        # continuing to move the actor while a
                                        # Y-boundary checkpoint is flushed.
                                        controller.stop()
                                        consume_route_recording_snapshots(
                                            route_recorder,
                                            snapshot_buffer,
                                        )
                                    except RuntimeError as exc:
                                        recording_checkpoint_ready = False
                                        LOG.warning(
                                            "Y reset/respawn canceled because "
                                            "the current recorded take could "
                                            "not be prepared for a checkpoint: %s",
                                            exc,
                                        )
                                    else:
                                        if (
                                            route_recorder.invalid_reason
                                            is not None
                                        ):
                                            LOG.warning(
                                                "Discarding invalid recorded "
                                                "take %d during Y recovery: %s",
                                                route_recorder.take_number,
                                                route_recorder.invalid_reason,
                                            )
                                        else:
                                            try:
                                                route_recorder.checkpoint_now(
                                                    wait=True
                                                )
                                            except RuntimeError as exc:
                                                recording_checkpoint_ready = False
                                                LOG.warning(
                                                    "Y reset/respawn canceled "
                                                    "because the current "
                                                    "recorded take could not "
                                                    "be checkpointed: %s",
                                                    exc,
                                                )
                                if (
                                    recording_checkpoint_ready
                                    and already_home
                                ):
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
                                        metrics_model.reset_live_measurements()
                                        overlap_refresh_requested = False
                                        if route_recorder is not None:
                                            route_recorder.reset(
                                                walker,
                                                spawn_transform,
                                                snapshot_buffer.latest(),
                                            )
                                            LOG.info(
                                                "Route recording restarted as "
                                                "take %d at the configured start",
                                                route_recorder.take_number,
                                            )
                                        if route_guidance is not None:
                                            route_guidance.reset()
                                        route_overlay = None
                                        route_refresh_requested = (
                                            route_guidance_enabled
                                        )
                                        LOG.info(
                                            "Ego pedestrian id=%d is already "
                                            "at configured start (%.2f, %.2f); "
                                            "controls and camera recentered",
                                            walker.id,
                                            spawn_transform.location.x,
                                            spawn_transform.location.y,
                                        )
                                elif recording_checkpoint_ready:
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
                                        camera_frame_received_at = None
                                        box_overlay = None
                                        box_counts = (0, 0)
                                        box_refresh_requested = boxes_enabled
                                        metrics_model.reset_live_measurements()
                                        overlap_refresh_requested = False
                                        if route_guidance is not None:
                                            route_guidance.reset()
                                        route_overlay = None
                                        route_refresh_requested = (
                                            route_guidance_enabled
                                        )
                                        if route_recorder is not None:
                                            route_recorder.reset(
                                                walker,
                                                spawn_transform,
                                                snapshot_buffer.latest(),
                                            )
                                            LOG.info(
                                                "Route recording restarted as "
                                                "take %d after Y respawn",
                                                route_recorder.take_number,
                                            )
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
                            metrics_model.clear_events_detected()
                        LOG.info(
                            "Ground-truth bounding boxes %s",
                            "enabled" if boxes_enabled else "disabled",
                        )
                    elif event.key == pygame.K_u:
                        u_visuals_enabled = not u_visuals_enabled
                        if u_visuals_enabled:
                            boxes_enabled = True
                            box_refresh_requested = True
                            overlap_refresh_requested = True
                            route_guidance_enabled = route_guidance is not None
                            route_refresh_requested = route_guidance_enabled
                            if not topdown_enabled:
                                (
                                    topdown_renderer,
                                    topdown_enabled,
                                ) = toggle_topdown_map(
                                    topdown_renderer,
                                    topdown_enabled,
                                    world,
                                    carla_map,
                                    args.topdown_zoom_radius,
                                )
                            if (
                                metrics_renderer is not None
                                and not metrics_renderer.ready
                            ):
                                metrics_renderer.close()
                                metrics_renderer = None
                            if metrics_renderer is None and cv2 is not None:
                                metrics_renderer = LiveMetricsRenderer()
                            metrics_enabled = bool(
                                metrics_renderer is not None
                                and metrics_renderer.ready
                            )
                            LOG.info(
                                "U visuals enabled: boxes=%s, route arrows=%s, "
                                "top-down map=%s, live metrics=%s",
                                boxes_enabled,
                                route_guidance_enabled,
                                topdown_enabled,
                                metrics_enabled,
                            )
                        else:
                            boxes_enabled = False
                            box_refresh_requested = False
                            box_overlay = None
                            box_counts = (0, 0)
                            metrics_model.clear_events_detected()
                            overlap_refresh_requested = False
                            route_guidance_enabled = False
                            route_refresh_requested = False
                            route_overlay = None
                            if topdown_enabled:
                                (
                                    topdown_renderer,
                                    topdown_enabled,
                                ) = toggle_topdown_map(
                                    topdown_renderer,
                                    topdown_enabled,
                                    world,
                                    carla_map,
                                    args.topdown_zoom_radius,
                                )
                            metrics_enabled = False
                            if metrics_renderer is not None:
                                metrics_renderer.close()
                            LOG.info(
                                "U visuals disabled: boxes, route arrows, "
                                "top-down map, and live metrics off"
                            )

            if not running:
                break
            if not walker.is_alive or not camera.is_alive:
                raise RuntimeError("the pedestrian or camera actor was destroyed")

            keys = pygame.key.get_pressed()
            sensed_at = frame_mailbox.take_latest_received_at_for_control()
            control_submitted_at = controller.update(keys, delta_seconds)
            metrics_model.note_control_submitted(
                sensed_at,
                control_submitted_at,
            )
            if route_recorder is not None:
                consume_route_recording_snapshots(
                    route_recorder,
                    snapshot_buffer,
                )
                try:
                    route_recorder.maybe_checkpoint()
                except RuntimeError as exc:
                    LOG.warning("Recorded-route checkpoint skipped: %s", exc)
                writer_error = route_recorder.consume_writer_error()
                if writer_error is not None:
                    LOG.warning(
                        "Recorded-route checkpoint generation %d failed: %s",
                        writer_error[0],
                        writer_error[1],
                    )
            if npc_health_monitor is not None:
                npc_health_monitor.update(snapshot_buffer.latest())
            if topdown_enabled and topdown_renderer is not None:
                map_latency_ms = topdown_renderer.render(world, walker)
                if map_latency_ms is not None:
                    metrics_model.note_spatial_map_latency(map_latency_ms)
                if not topdown_renderer.ready:
                    topdown_enabled = False
                    LOG.warning(
                        "Top-down map disabled after an OpenCV rendering error"
                    )

            image, image_received_at = frame_mailbox.pop_with_timestamp()
            if image is not None:
                surface = image_to_surface(image)
                frame_id = int(image.frame)
                camera_frame_transform = image.transform
                camera_frame_received_at = image_received_at
                box_refresh_requested = boxes_enabled
                route_refresh_requested = route_guidance_enabled
                route_overlay = None
                if boxes_enabled:
                    # Never carry geometry from the prior image onto this one.
                    box_overlay = None
                    box_counts = (0, 0)
                overlap_refresh_requested = metrics_enabled

            if (
                metrics_enabled
                and overlap_refresh_requested
                and frame_id is not None
            ):
                snapshot = snapshot_buffer.get(frame_id)
                if snapshot is not None:
                    try:
                        overlap_detected = snapshot_bbox_overlap_proxy(
                            walker,
                            projection_cache.get(),
                            snapshot,
                        )
                    except RuntimeError as exc:
                        LOG.debug("Outcome overlap proxy skipped: %s", exc)
                    else:
                        metrics_model.note_overlap_detected(overlap_detected)
                        overlap_refresh_requested = False

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
                        metrics_model.note_events_detected(
                            sum(box_counts),
                            sampled_at=camera_frame_received_at,
                        )
                    except RuntimeError as exc:
                        LOG.debug("Ground-truth overlay skipped: %s", exc)
                        box_overlay = None
                        box_counts = (0, 0)
                    box_refresh_requested = False

            if (
                route_guidance_enabled
                and route_guidance is not None
                and route_refresh_requested
                and surface is not None
                and frame_id is not None
                and camera_frame_transform is not None
            ):
                # Pair the camera pose carried by this image with the walker
                # pose from that exact CARLA frame. This avoids route jitter
                # and direction errors while the head camera is being aimed.
                snapshot = snapshot_buffer.get(frame_id)
                walker_snapshot = (
                    None if snapshot is None else snapshot.find(int(walker.id))
                )
                if walker_snapshot is not None:
                    try:
                        walker_frame_transform = walker_snapshot.get_transform()
                        route_overlay = route_guidance.render_overlay(
                            width,
                            height,
                            camera_frame_transform,
                            walker_frame_transform.location,
                        )
                    except (AttributeError, RuntimeError, ValueError) as exc:
                        LOG.debug("Route overlay skipped: %s", exc)
                        route_overlay = None
                    route_refresh_requested = False

            if surface is not None:
                display.blit(surface, (0, 0))
                if route_guidance_enabled and route_overlay is not None:
                    display.blit(route_overlay, (0, 0))
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
                route_guidance is not None,
                route_guidance_enabled,
                route_name,
                (
                    None
                    if route_recorder is None
                    else route_recorder.status_text()
                ),
                metrics_enabled,
            )
            if metrics_enabled and metrics_renderer is not None:
                if not metrics_renderer.render(metrics_model.snapshot()):
                    metrics_enabled = False
                    LOG.warning(
                        "Live metrics window disabled after an OpenCV rendering error"
                    )
            pygame.display.flip()

    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    finally:
        cleanup_started_with_exception = sys.exc_info()[0] is not None
        route_recording_finalize_error = None
        # Stop motion and detach the callback before a final route write can
        # wait on filesystem I/O.  This keeps the pedestrian stationary during
        # the bounded writer join and freezes the snapshot set being consumed.
        if controller is not None:
            try:
                controller.stop()
            except RuntimeError:
                pass
        if world is not None and snapshot_callback_id is not None:
            try:
                world.remove_on_tick(snapshot_callback_id)
            except RuntimeError:
                pass
        if route_recorder is not None:
            try:
                consume_route_recording_snapshots(
                    route_recorder,
                    snapshot_buffer,
                )
                recorded_route = route_recorder.finish()
            except RuntimeError as exc:
                route_recording_finalize_error = exc
                LOG.error(
                    "Unable to finalize recorded pedestrian route %s: %s",
                    route_recorder.output_path,
                    exc,
                )
            else:
                if recorded_route is None:
                    LOG.warning(
                        "The current recorded take did not move far enough to "
                        "save; any last valid checkpoint at %s was left "
                        "unchanged",
                        route_recorder.output_path,
                    )
                else:
                    LOG.info(
                        "Finalized recorded pedestrian route with %d points "
                        "and %.1f m of travel: %s",
                        len(recorded_route["planned_path"]),
                        route_recorder.recorded_distance,
                        route_recorder.output_path,
                    )
        if topdown_renderer is not None:
            topdown_renderer.close()
        if metrics_renderer is not None:
            metrics_renderer.close()
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
        if (
            route_recording_finalize_error is not None
            and not cleanup_started_with_exception
        ):
            raise route_recording_finalize_error


if __name__ == "__main__":
    main()

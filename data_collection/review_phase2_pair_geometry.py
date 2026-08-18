#!/usr/bin/env python3
"""Visually review the proposed Phase-2 helper/recipient geometry.

This is a geometry-only instrument, not a pilot, corpus collector, detector
evaluation, or C2 result.  It supports the original same-direction route and
the curbside opposite-direction layout already demonstrated by the SceneSense
scenario harness.  Both use the exact production RGB contract.  The script
owns the 10 Hz synchronous clock and requires an otherwise empty dynamic world.

Keys in the combined OpenCV window:

* ``s`` saves the current helper/recipient/combined images;
* ``q`` or Escape stops and deterministically removes every owned actor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import carla
import cv2
import numpy as np

from data_collection.phase2_curbside_scenario import (
    CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE,
    CURBSIDE_HELPER_TRANSFORM,
    CURBSIDE_OCCLUDER_TRANSFORM,
    CURBSIDE_RECIPIENT_TRANSFORM,
    CURBSIDE_WALKER_END,
    CURBSIDE_WALKER_START,
    DirectRouteController,
    ROLE_ORDER,
    legal_opposing_lane_contract,
    load_route_progress,
    opposite_lane_route,
    wrap_degrees,
    world_transform as _world_transform,
)
from data_collection.phase2_signalized_corner_scenario import (
    SIGNALIZED_GEOMETRY_ID,
    SIGNALIZED_HELPER_TRANSFORM,
    SIGNALIZED_OCCLUDER_TRANSFORM,
    SIGNALIZED_RECIPIENT_TRANSFORM,
    SIGNALIZED_WALKER_END,
    SIGNALIZED_WALKER_START,
    controlled_traffic_lights,
    line_of_sight_bearings_deg,
    frozen_routes as signalized_frozen_routes,
    signalized_lane_contract,
)
from data_collection.phase2_midblock_van_scenario import (
    MIDBLOCK_GEOMETRY_ID,
    MIDBLOCK_HELPER_TRANSFORM,
    MIDBLOCK_OCCLUDER_TRANSFORM,
    MIDBLOCK_RECIPIENT_TRANSFORM,
    MIDBLOCK_WALKER_END,
    MIDBLOCK_WALKER_START,
    frozen_routes as midblock_frozen_routes,
    line_of_sight_bearings_deg as midblock_line_of_sight_bearings_deg,
    midblock_lane_contract,
)
from data_collection.phase2_cross_traffic_vehicle_scenario import (
    CROSS_TRAFFIC_GEOMETRY_ID,
    CROSS_TRAFFIC_HELPER_TRANSFORM,
    CROSS_TRAFFIC_OCCLUDER_BLUEPRINT,
    CROSS_TRAFFIC_OCCLUDER_TRANSFORM,
    CROSS_TRAFFIC_RECIPIENT_TRANSFORM,
    CROSS_TRAFFIC_REVIEW_YIELD_TRIGGER_M,
    CROSS_TRAFFIC_TARGET_BLUEPRINT,
    CROSS_TRAFFIC_TARGET_TRANSFORM,
    frozen_routes as cross_traffic_frozen_routes,
    cross_traffic_geometry_contract,
    visibility_state as cross_traffic_visibility_state,
)
from data_collection.phase2_parked_vehicle_pullout_scenario import (
    PULLOUT_GEOMETRY_ID,
    PULLOUT_HELPER_TRANSFORM,
    PULLOUT_OCCLUDER_BLUEPRINT,
    PULLOUT_OCCLUDER_TRANSFORM,
    PULLOUT_RECIPIENT_TRANSFORM,
    PULLOUT_REVIEW_YIELD_TRIGGER_M,
    PULLOUT_TARGET_BLUEPRINT,
    PULLOUT_TARGET_SPEED_MPS,
    PULLOUT_TARGET_START_DELAY_S,
    PULLOUT_TARGET_TRANSFORM,
    frozen_routes as pullout_frozen_routes,
    pullout_geometry_contract,
)
from data_collection.phase2_queue_reveal_vehicle_scenario import (
    QUEUE_REVEAL_GEOMETRY_ID,
    QUEUE_REVEAL_HELPER_TRANSFORM,
    QUEUE_REVEAL_OCCLUDER_BLUEPRINT,
    QUEUE_REVEAL_OCCLUDER_SPEED_MPS,
    QUEUE_REVEAL_OCCLUDER_START_DELAY_S,
    QUEUE_REVEAL_OCCLUDER_TRANSFORM,
    QUEUE_REVEAL_RECIPIENT_TRANSFORM,
    QUEUE_REVEAL_REVIEW_YIELD_TRIGGER_M,
    QUEUE_REVEAL_TARGET_BLUEPRINT,
    QUEUE_REVEAL_TARGET_TRANSFORM,
    frozen_routes as queue_reveal_frozen_routes,
    queue_reveal_geometry_contract,
)


DEFAULT_OUTPUT_ROOT = Path("/tmp")
DEFAULT_ROUTE_PROGRESS = (
    Path(__file__).resolve().parent
    / "routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv"
)
CURBSIDE_ROUTE_PROGRESS = (
    Path(__file__).resolve().parent
    / "routes/town10hd_opt_curbside_recipient_v1.progress.csv"
)
LAYOUTS = (
    "same_direction",
    "curbside_opposite",
    "signalized_corner",
    "midblock_van",
    "cross_traffic_vehicle",
    "parked_vehicle_pullout",
    "queue_reveal_vehicle",
)
SCENARIO_ROLES = ("controlled_positive_occlusion", "matched_benign_negative")
OCCLUDER_SETTLE_MAX_TICKS = 30
OCCLUDER_SETTLE_STABLE_TICKS = 3
OCCLUDER_SETTLE_MAX_XY_DRIFT_M = 0.35
OCCLUDER_SETTLE_MAX_YAW_DRIFT_DEG = 3.0


def offset_transform(
    base: carla.Transform,
    *,
    forward_m: float,
    right_m: float = 0.0,
    z_offset_m: float = 0.15,
) -> carla.Transform:
    """Return an auditable lane-relative offset without mutating ``base``."""

    yaw = math.radians(float(base.rotation.yaw))
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    right_x, right_y = -forward_y, forward_x
    return carla.Transform(
        carla.Location(
            x=float(base.location.x) + forward_x * float(forward_m)
            + right_x * float(right_m),
            y=float(base.location.y) + forward_y * float(forward_m)
            + right_y * float(right_m),
            z=float(base.location.z) + float(z_offset_m),
        ),
        carla.Rotation(
            pitch=float(base.rotation.pitch),
            yaw=float(base.rotation.yaw),
            roll=float(base.rotation.roll),
        ),
    )


class CameraMailbox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Dict[str, Tuple[int, np.ndarray]] = {}

    def push(self, role: str, image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = array.reshape((int(image.height), int(image.width), 4))
        bgr = np.ascontiguousarray(bgra[:, :, :3])
        with self._lock:
            self._latest[str(role)] = (int(image.frame), bgr)

    def aligned(self, minimum_frame: int) -> Optional[Dict[str, Tuple[int, np.ndarray]]]:
        with self._lock:
            if any(role not in self._latest for role in ROLE_ORDER):
                return None
            result = {
                role: (frame, image.copy())
                for role, (frame, image) in self._latest.items()
                if role in ROLE_ORDER
            }
        if any(frame < int(minimum_frame) for frame, _image in result.values()):
            return None
        return result


def _role_blueprint(world: carla.World, role: str, color: str) -> carla.ActorBlueprint:
    blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", f"phase2_geometry_{role}")
    if blueprint.has_attribute("color"):
        blueprint.set_attribute("color", color)
    return blueprint


def _camera_blueprint(world: carla.World) -> carla.ActorBlueprint:
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    for name, value in {
        "image_size_x": "1280",
        "image_size_y": "720",
        "fov": "120.0",
        "sensor_tick": "0.1",
        "gamma": "2.2",
    }.items():
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, value)
    return blueprint


def _dynamic_inventory(world: carla.World) -> Dict[str, int]:
    return {
        pattern: len(world.get_actors().filter(pattern))
        for pattern in ("vehicle.*", "walker.*", "sensor.*", "controller.ai.walker")
    }


def _settle_parked_occluder(
    world: carla.World,
    actor: carla.Actor,
    commanded_transform: carla.Transform,
    *,
    timeout_s: float,
) -> Dict[str, object]:
    """Let a clearance-height vehicle settle before freezing its physics."""

    if timeout_s <= 0.0:
        raise ValueError("occluder settlement timeout must be positive")
    actor.set_simulate_physics(True)
    actor.apply_control(
        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    )
    previous_z: Optional[float] = None
    stable_ticks = 0
    settled_transform: Optional[carla.Transform] = None
    settled_frame: Optional[int] = None
    vertical_speed_mps = float("inf")
    for _unused in range(OCCLUDER_SETTLE_MAX_TICKS):
        settled_frame = int(world.tick(float(timeout_s)))
        settled_transform = actor.get_transform()
        vertical_speed_mps = abs(float(actor.get_velocity().z))
        current_z = float(settled_transform.location.z)
        if (
            previous_z is not None
            and abs(current_z - previous_z) <= 0.002
            and vertical_speed_mps <= 0.02
        ):
            stable_ticks += 1
        else:
            stable_ticks = 0
        previous_z = current_z
        if stable_ticks >= OCCLUDER_SETTLE_STABLE_TICKS:
            break
    if settled_transform is None or stable_ticks < OCCLUDER_SETTLE_STABLE_TICKS:
        raise RuntimeError(
            "parked occluder did not reach a stable grounded pose within "
            f"{OCCLUDER_SETTLE_MAX_TICKS} ticks"
        )

    xy_drift_m = math.hypot(
        float(settled_transform.location.x - commanded_transform.location.x),
        float(settled_transform.location.y - commanded_transform.location.y),
    )
    yaw_drift_deg = abs(
        wrap_degrees(
            float(settled_transform.rotation.yaw)
            - float(commanded_transform.rotation.yaw)
        )
    )
    if xy_drift_m > OCCLUDER_SETTLE_MAX_XY_DRIFT_M:
        raise RuntimeError(
            "parked occluder moved away from its reviewed curb pose while settling: "
            f"xy_drift_m={xy_drift_m:.3f}"
        )
    if yaw_drift_deg > OCCLUDER_SETTLE_MAX_YAW_DRIFT_DEG:
        raise RuntimeError(
            "parked occluder rotated away from its reviewed curb pose while settling: "
            f"yaw_drift_deg={yaw_drift_deg:.3f}"
        )

    actor.set_simulate_physics(False)
    frozen_frame = int(world.tick(float(timeout_s)))
    frozen_transform = actor.get_transform()
    freeze_drift_m = frozen_transform.location.distance(settled_transform.location)
    if freeze_drift_m > 0.01:
        raise RuntimeError(
            "parked occluder pose changed while freezing physics: "
            f"drift_m={freeze_drift_m:.3f}"
        )
    return {
        "pass": True,
        "basis": "gravity_settled_then_physics_frozen",
        "settled_frame": settled_frame,
        "frozen_frame": frozen_frame,
        "stable_ticks": stable_ticks,
        "vertical_speed_mps": vertical_speed_mps,
        "xy_drift_m": xy_drift_m,
        "yaw_drift_deg": yaw_drift_deg,
        "commanded_z_m": float(commanded_transform.location.z),
        "settled_z_m": float(settled_transform.location.z),
        "settled_pitch_deg": float(settled_transform.rotation.pitch),
        "settled_roll_deg": float(settled_transform.rotation.roll),
    }


def _annotate(
    image: np.ndarray, role: str, frame: int, scenario_role: str
) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 54), (20, 20, 20), -1)
    cv2.putText(
        result,
        f"{role.upper()} | {scenario_role} | frame {frame} | 1280x720 FOV 120",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _save_views(
    output_dir: Path,
    views: Dict[str, Tuple[int, np.ndarray]],
    combined: np.ndarray,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for role, (frame, image) in views.items():
        cv2.imwrite(str(output_dir / f"{stamp}_{role}_frame{frame}.png"), image)
    cv2.imwrite(str(output_dir / f"{stamp}_combined.png"), combined)
    print(f"Saved geometry screenshots under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--layout", choices=LAYOUTS, default="same_direction")
    parser.add_argument(
        "--scenario-role",
        choices=SCENARIO_ROLES,
        default="controlled_positive_occlusion",
    )
    parser.add_argument("--recipient-spawn-index", type=int, default=55)
    parser.add_argument("--helper-forward-offset-m", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--recipient-speed-mps", type=float, default=5.0)
    parser.add_argument("--helper-speed-mps", type=float, default=4.5)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="skip the UI but retain cameras and save critical-time views",
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="skip cameras and UI; emit only realized-pose telemetry",
    )
    parser.add_argument(
        "--route-progress-csv", type=Path, default=DEFAULT_ROUTE_PROGRESS
    )
    parser.add_argument(
        "--stationary",
        action="store_true",
        help="hold both vehicles fixed; default follows the accepted v3 route",
    )
    parser.add_argument("--target-x", type=float, default=19.791866)
    parser.add_argument("--target-y", type=float, default=30.7)
    parser.add_argument("--target-z", type=float, default=0.005668)
    parser.add_argument("--target-yaw", type=float, default=-89.580742)
    parser.add_argument("--pedestrian-start-delay-s", type=float, default=3.0)
    parser.add_argument("--pedestrian-speed-mps", type=float, default=1.3)
    parser.add_argument("--target-vehicle-speed-mps", type=float)
    parser.add_argument("--target-vehicle-start-delay-s", type=float)
    parser.add_argument("--queue-occluder-speed-mps", type=float)
    parser.add_argument("--queue-occluder-start-delay-s", type=float)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if bool(args.headless) and bool(args.pose_only):
        parser.error("--headless and --pose-only are mutually exclusive")
    if str(args.scenario_role) == "matched_benign_negative" and str(
        args.layout
    ) not in {
        "curbside_opposite",
        "signalized_corner",
        "midblock_van",
        "cross_traffic_vehicle",
        "parked_vehicle_pullout",
        "queue_reveal_vehicle",
    }:
        parser.error(
            "the matched benign review is defined only for controlled paired layouts"
        )
    hazard_present = str(args.scenario_role) == "controlled_positive_occlusion"
    if not 5.0 <= float(args.helper_forward_offset_m) <= 15.0:
        parser.error("helper forward offset must remain within the reviewed 5-15 m range")
    if float(args.duration_s) <= 0.0:
        parser.error("duration must be positive")
    if not 1.0 <= float(args.recipient_speed_mps) <= 8.0:
        parser.error("recipient speed must remain within 1-8 m/s")
    if not 1.0 <= float(args.helper_speed_mps) <= 8.0:
        parser.error("helper speed must remain within 1-8 m/s")
    if float(args.pedestrian_start_delay_s) < 0.0:
        parser.error("pedestrian start delay must be non-negative")
    if not 1.0 <= float(args.pedestrian_speed_mps) <= 2.0:
        parser.error("pedestrian speed must stay within the realistic 1-2 m/s review range")
    target_vehicle_speed_mps = (
        float(args.target_vehicle_speed_mps)
        if args.target_vehicle_speed_mps is not None
        else (
            PULLOUT_TARGET_SPEED_MPS
            if str(args.layout) == "parked_vehicle_pullout"
            else 3.6
        )
    )
    target_vehicle_start_delay_s = (
        float(args.target_vehicle_start_delay_s)
        if args.target_vehicle_start_delay_s is not None
        else (
            PULLOUT_TARGET_START_DELAY_S
            if str(args.layout) == "parked_vehicle_pullout"
            else 0.0
        )
    )
    if not 2.0 <= target_vehicle_speed_mps <= 6.0:
        parser.error("target vehicle speed must remain within 2-6 m/s")
    if not 0.0 <= target_vehicle_start_delay_s <= 8.0:
        parser.error("target vehicle start delay must remain within 0-8 s")
    queue_occluder_speed_mps = (
        float(args.queue_occluder_speed_mps)
        if args.queue_occluder_speed_mps is not None
        else QUEUE_REVEAL_OCCLUDER_SPEED_MPS
    )
    queue_occluder_start_delay_s = (
        float(args.queue_occluder_start_delay_s)
        if args.queue_occluder_start_delay_s is not None
        else QUEUE_REVEAL_OCCLUDER_START_DELAY_S
    )
    if not 1.5 <= queue_occluder_speed_mps <= 4.0:
        parser.error("queue occluder speed must remain within 1.5-4 m/s")
    if not 2.0 <= queue_occluder_start_delay_s <= 8.0:
        parser.error("queue occluder start delay must remain within 2-8 s")
    route_path: Optional[Path]
    route: Optional[list[carla.Location]]
    if str(args.layout) in {
        "signalized_corner",
        "midblock_van",
        "cross_traffic_vehicle",
        "parked_vehicle_pullout",
        "queue_reveal_vehicle",
    }:
        route_path = None
        route = None
    else:
        route_path = (
            CURBSIDE_ROUTE_PROGRESS
            if str(args.layout) == "curbside_opposite"
            else args.route_progress_csv.resolve()
        )
        route = load_route_progress(route_path)

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.get_world()
    if not str(world.get_map().name).endswith("Town10HD_Opt"):
        raise RuntimeError(f"expected Town10HD_Opt, found {world.get_map().name}")
    inventory = _dynamic_inventory(world)
    if any(inventory.values()):
        raise RuntimeError(f"geometry review requires an empty dynamic world: {inventory}")

    if str(args.layout) == "curbside_opposite":
        geometry_origin = "frozen_20260806_curbside_evidence"
        transforms = {
            "recipient": _world_transform(CURBSIDE_RECIPIENT_TRANSFORM),
            "helper": _world_transform(CURBSIDE_HELPER_TRANSFORM),
        }
        role_routes = {
            "recipient": route,
            "helper": opposite_lane_route(route),
        }
        walker_transform = _world_transform(CURBSIDE_WALKER_START)
        walker_end = carla.Location(
            x=CURBSIDE_WALKER_END[0],
            y=CURBSIDE_WALKER_END[1],
            z=CURBSIDE_WALKER_END[2],
        )
    elif str(args.layout) == "signalized_corner":
        geometry_origin = "user_accepted_positive_and_benign_review_20260817"
        transforms = {
            "recipient": _world_transform(SIGNALIZED_RECIPIENT_TRANSFORM),
            "helper": _world_transform(SIGNALIZED_HELPER_TRANSFORM),
        }
        role_routes = signalized_frozen_routes()
        walker_transform = _world_transform(SIGNALIZED_WALKER_START)
        walker_end = carla.Location(
            x=float(SIGNALIZED_WALKER_END[0]),
            y=float(SIGNALIZED_WALKER_END[1]),
            z=float(SIGNALIZED_WALKER_END[2]),
        )
    elif str(args.layout) == "midblock_van":
        geometry_origin = "user_accepted_positive_and_benign_review_20260818"
        transforms = {
            "recipient": _world_transform(MIDBLOCK_RECIPIENT_TRANSFORM),
            "helper": _world_transform(MIDBLOCK_HELPER_TRANSFORM),
        }
        role_routes = midblock_frozen_routes()
        walker_transform = _world_transform(MIDBLOCK_WALKER_START)
        walker_end = carla.Location(
            x=float(MIDBLOCK_WALKER_END[0]),
            y=float(MIDBLOCK_WALKER_END[1]),
            z=float(MIDBLOCK_WALKER_END[2]),
        )
    elif str(args.layout) == "cross_traffic_vehicle":
        geometry_origin = "user_accepted_positive_and_benign_review_20260817"
        transforms = {
            "recipient": _world_transform(CROSS_TRAFFIC_RECIPIENT_TRANSFORM),
            "helper": _world_transform(CROSS_TRAFFIC_HELPER_TRANSFORM),
        }
        role_routes = cross_traffic_frozen_routes()
        walker_transform = _world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM)
        walker_end = None
    elif str(args.layout) == "parked_vehicle_pullout":
        geometry_origin = "user_accepted_positive_and_benign_review_20260818"
        transforms = {
            "recipient": _world_transform(PULLOUT_RECIPIENT_TRANSFORM),
            "helper": _world_transform(PULLOUT_HELPER_TRANSFORM),
        }
        role_routes = pullout_frozen_routes()
        walker_transform = _world_transform(PULLOUT_TARGET_TRANSFORM)
        walker_end = None
    elif str(args.layout) == "queue_reveal_vehicle":
        geometry_origin = "user_accepted_positive_and_benign_review_20260818"
        transforms = {
            "recipient": _world_transform(QUEUE_REVEAL_RECIPIENT_TRANSFORM),
            "helper": _world_transform(QUEUE_REVEAL_HELPER_TRANSFORM),
        }
        role_routes = queue_reveal_frozen_routes()
        walker_transform = _world_transform(QUEUE_REVEAL_TARGET_TRANSFORM)
        walker_end = None
    else:
        geometry_origin = f"spawn_index_{int(args.recipient_spawn_index)}"
        spawn_points = list(world.get_map().get_spawn_points())
        if not 0 <= int(args.recipient_spawn_index) < len(spawn_points):
            raise ValueError("recipient spawn index is outside the map catalog")
        base = spawn_points[int(args.recipient_spawn_index)]
        transforms = {
            "recipient": offset_transform(base, forward_m=0.0),
            "helper": offset_transform(
                base, forward_m=float(args.helper_forward_offset_m)
            ),
        }
        if route is None:
            raise RuntimeError("same-direction route was not loaded")
        role_routes = {role: route for role in ROLE_ORDER}
        walker_transform = carla.Transform(
            carla.Location(
                x=float(args.target_x),
                y=float(args.target_y),
                z=float(args.target_z) + 0.5,
            ),
            carla.Rotation(yaw=float(args.target_yaw)),
        )
        walker_end = None
    if str(args.layout) == "curbside_opposite":
        lane_contract = legal_opposing_lane_contract(world.get_map(), transforms)
    elif str(args.layout) == "signalized_corner":
        lane_contract = signalized_lane_contract(
            world.get_map(),
            transforms,
            _world_transform(SIGNALIZED_OCCLUDER_TRANSFORM),
        )
    elif str(args.layout) == "midblock_van":
        lane_contract = midblock_lane_contract(
            world.get_map(),
            transforms,
            _world_transform(MIDBLOCK_OCCLUDER_TRANSFORM),
        )
    elif str(args.layout) == "cross_traffic_vehicle":
        lane_contract = cross_traffic_geometry_contract(
            world.get_map(),
            transforms,
            _world_transform(CROSS_TRAFFIC_OCCLUDER_TRANSFORM),
            _world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM),
            role_routes,
        )
    elif str(args.layout) == "parked_vehicle_pullout":
        lane_contract = pullout_geometry_contract(
            world.get_map(),
            transforms,
            _world_transform(PULLOUT_OCCLUDER_TRANSFORM),
            _world_transform(PULLOUT_TARGET_TRANSFORM),
            role_routes,
        )
    elif str(args.layout) == "queue_reveal_vehicle":
        lane_contract = queue_reveal_geometry_contract(
            world.get_map(),
            transforms,
            _world_transform(QUEUE_REVEAL_OCCLUDER_TRANSFORM),
            _world_transform(QUEUE_REVEAL_TARGET_TRANSFORM),
            role_routes,
        )
    else:
        lane_contract = None
    separation = transforms["recipient"].location.distance(
        transforms["helper"].location
    )
    target_location = walker_transform.location
    output_dir = (
        args.output_root.resolve()
        / (
            "phase2_geometry_review_"
            + str(args.layout)
            + "_"
            + ("positive_" if hazard_present else "benign_")
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    original_settings = world.get_settings()
    owned = []
    cameras = []
    collision_sensors = []
    collisions = []
    traffic_light_restore = []
    occluder_settlement: Optional[Dict[str, object]] = None
    target_settlement: Optional[Dict[str, object]] = None
    occluder: Optional[carla.Actor] = None
    target_vehicle: Optional[carla.Actor] = None
    target_vehicle_started = False
    queue_occluder_started = False
    walker_started = False
    walker_completed = False
    review_safety_yield_ever = False
    realized_trace = []
    automatic_capture_times_s = (
        (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
        if str(args.layout) == "cross_traffic_vehicle"
        else
        (1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0)
        if str(args.layout) == "parked_vehicle_pullout"
        else
        (1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0)
        if str(args.layout) == "queue_reveal_vehicle"
        else
        (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
        if str(args.layout) in {"signalized_corner", "midblock_van"}
        else (4.5, 6.0, 7.5, 9.0)
    )
    automatic_captures_written = []
    mailbox = CameraMailbox()
    latest_views: Optional[Dict[str, Tuple[int, np.ndarray]]] = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        world.tick(float(args.timeout_s))

        if str(args.layout) in {"signalized_corner", "cross_traffic_vehicle"}:
            for role, traffic_light in controlled_traffic_lights(world).items():
                original_state = traffic_light.get_state()
                original_frozen = bool(traffic_light.is_frozen())
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
                traffic_light_restore.append(
                    (role, traffic_light, original_state, original_frozen)
                )
            world.tick(float(args.timeout_s))

        vehicles: Dict[str, carla.Actor] = {}
        for role, color in (("recipient", "0,0,255"), ("helper", "0,255,0")):
            actor = world.try_spawn_actor(
                _role_blueprint(world, role, color), transforms[role]
            )
            if actor is None:
                raise RuntimeError(f"exact {role} geometry spawn failed")
            actor.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
            )
            vehicles[role] = actor
            owned.append(actor)

        if str(args.layout) in {
            "curbside_opposite",
            "signalized_corner",
            "midblock_van",
            "cross_traffic_vehicle",
            "parked_vehicle_pullout",
            "queue_reveal_vehicle",
        }:
            occluder_blueprint = world.get_blueprint_library().find(
                {
                    "cross_traffic_vehicle": CROSS_TRAFFIC_OCCLUDER_BLUEPRINT,
                    "parked_vehicle_pullout": PULLOUT_OCCLUDER_BLUEPRINT,
                    "queue_reveal_vehicle": QUEUE_REVEAL_OCCLUDER_BLUEPRINT,
                }.get(str(args.layout), "vehicle.sprinter.mercedes")
            )
            if occluder_blueprint.has_attribute("role_name"):
                occluder_blueprint.set_attribute(
                    "role_name", f"phase2_geometry_{args.layout}_occluder"
                )
            occluder_transform = {
                "curbside_opposite": _world_transform(CURBSIDE_OCCLUDER_TRANSFORM),
                "signalized_corner": _world_transform(
                    SIGNALIZED_OCCLUDER_TRANSFORM
                ),
                "midblock_van": _world_transform(MIDBLOCK_OCCLUDER_TRANSFORM),
                "cross_traffic_vehicle": _world_transform(
                    CROSS_TRAFFIC_OCCLUDER_TRANSFORM
                ),
                "parked_vehicle_pullout": _world_transform(
                    PULLOUT_OCCLUDER_TRANSFORM
                ),
                "queue_reveal_vehicle": _world_transform(
                    QUEUE_REVEAL_OCCLUDER_TRANSFORM
                ),
            }[str(args.layout)]
            occluder = world.try_spawn_actor(
                occluder_blueprint,
                occluder_transform,
            )
            if occluder is None:
                raise RuntimeError(f"exact {args.layout} occluder spawn failed")
            owned.append(occluder)
            if str(args.layout) in {
                "midblock_van",
                "cross_traffic_vehicle",
                "parked_vehicle_pullout",
            }:
                occluder_settlement = _settle_parked_occluder(
                    world,
                    occluder,
                    occluder_transform,
                    timeout_s=float(args.timeout_s),
                )
                if str(args.layout) == "midblock_van":
                    lane_contract = midblock_lane_contract(
                        world.get_map(), transforms, occluder.get_transform()
                    )
                elif str(args.layout) == "cross_traffic_vehicle":
                    lane_contract = cross_traffic_geometry_contract(
                        world.get_map(),
                        transforms,
                        occluder.get_transform(),
                        _world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM),
                        role_routes,
                    )
                else:
                    lane_contract = pullout_geometry_contract(
                        world.get_map(),
                        transforms,
                        occluder.get_transform(),
                        _world_transform(PULLOUT_TARGET_TRANSFORM),
                        role_routes,
                    )
            elif str(args.layout) == "queue_reveal_vehicle":
                occluder.set_simulate_physics(True)
                occluder.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    )
                )
            else:
                occluder.set_simulate_physics(False)

        walker: Optional[carla.Actor] = None
        if hazard_present:
            if str(args.layout) in {
                "cross_traffic_vehicle",
                "parked_vehicle_pullout",
                "queue_reveal_vehicle",
            }:
                target_blueprint = world.get_blueprint_library().find(
                    CROSS_TRAFFIC_TARGET_BLUEPRINT
                    if str(args.layout) == "cross_traffic_vehicle"
                    else (
                        PULLOUT_TARGET_BLUEPRINT
                        if str(args.layout) == "parked_vehicle_pullout"
                        else QUEUE_REVEAL_TARGET_BLUEPRINT
                    )
                )
                if target_blueprint.has_attribute("role_name"):
                    target_blueprint.set_attribute(
                        "role_name", f"phase2_geometry_{args.layout}_target"
                    )
                if target_blueprint.has_attribute("color"):
                    target_blueprint.set_attribute("color", "255,165,0")
                target_vehicle = world.try_spawn_actor(
                    target_blueprint, walker_transform
                )
                if target_vehicle is None:
                    raise RuntimeError("registered cross-traffic target spawn failed")
                target_vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=True
                    )
                )
                owned.append(target_vehicle)
                if str(args.layout) == "queue_reveal_vehicle":
                    target_settlement = _settle_parked_occluder(
                        world,
                        target_vehicle,
                        _world_transform(QUEUE_REVEAL_TARGET_TRANSFORM),
                        timeout_s=float(args.timeout_s),
                    )
                    lane_contract = queue_reveal_geometry_contract(
                        world.get_map(),
                        transforms,
                        occluder.get_transform(),
                        target_vehicle.get_transform(),
                        role_routes,
                    )
            else:
                walker_blueprints = sorted(
                    world.get_blueprint_library().filter("walker.pedestrian.*"),
                    key=lambda item: item.id,
                )
                if not walker_blueprints:
                    raise RuntimeError("no pedestrian blueprint is available")
                walker = world.try_spawn_actor(walker_blueprints[0], walker_transform)
                if walker is None:
                    raise RuntimeError("registered waiting-pedestrian spawn failed")
                owned.append(walker)

        for _unused in range(5):
            world.tick(float(args.timeout_s))
        if walker is not None and str(args.layout) == "same_direction":
            walker.set_simulate_physics(False)

        controllers: Dict[str, DirectRouteController] = {}
        for role, actor in vehicles.items():
            collision_blueprint = world.get_blueprint_library().find(
                "sensor.other.collision"
            )
            collision_sensor = world.spawn_actor(
                collision_blueprint, carla.Transform(), attach_to=actor
            )
            collision_sensor.listen(
                lambda event, name=role: collisions.append(
                    {
                        "role": name,
                        "frame": int(event.frame),
                        "other_actor_id": int(event.other_actor.id),
                        "other_type_id": str(event.other_actor.type_id),
                    }
                )
            )
            collision_sensors.append(collision_sensor)
            owned.append(collision_sensor)

            if bool(args.stationary):
                actor.set_simulate_physics(False)
                continue
            actor.set_simulate_physics(True)
            actor.apply_control(carla.VehicleControl())
            actor.set_autopilot(False, int(args.tm_port))
            controllers[role] = DirectRouteController(
                actor,
                role_routes[role],
                target_speed_mps=(
                    float(args.helper_speed_mps)
                    if role == "helper"
                    else float(args.recipient_speed_mps)
                ),
            )

        if target_vehicle is not None:
            collision_blueprint = world.get_blueprint_library().find(
                "sensor.other.collision"
            )
            collision_sensor = world.spawn_actor(
                collision_blueprint, carla.Transform(), attach_to=target_vehicle
            )
            collision_sensor.listen(
                lambda event: collisions.append(
                    {
                        "role": "target",
                        "frame": int(event.frame),
                        "other_actor_id": int(event.other_actor.id),
                        "other_type_id": str(event.other_actor.type_id),
                    }
                )
            )
            collision_sensors.append(collision_sensor)
            owned.append(collision_sensor)
            if bool(args.stationary) or str(args.layout) == "queue_reveal_vehicle":
                target_vehicle.set_simulate_physics(False)
            else:
                target_vehicle.set_simulate_physics(True)
                target_vehicle.apply_control(carla.VehicleControl())
                target_vehicle.set_autopilot(False, int(args.tm_port))
                controllers["target"] = DirectRouteController(
                    target_vehicle,
                    role_routes["target"],
                    target_speed_mps=target_vehicle_speed_mps,
                    waypoint_reach_m=(
                        0.75
                        if str(args.layout) == "parked_vehicle_pullout"
                        else 3.5
                    ),
                )

        if str(args.layout) == "queue_reveal_vehicle":
            if occluder is None:
                raise RuntimeError("queue-reveal layout lacks its queue member")
            collision_blueprint = world.get_blueprint_library().find(
                "sensor.other.collision"
            )
            collision_sensor = world.spawn_actor(
                collision_blueprint, carla.Transform(), attach_to=occluder
            )
            collision_sensor.listen(
                lambda event: collisions.append(
                    {
                        "role": "occluder",
                        "frame": int(event.frame),
                        "other_actor_id": int(event.other_actor.id),
                        "other_type_id": str(event.other_actor.type_id),
                    }
                )
            )
            collision_sensors.append(collision_sensor)
            owned.append(collision_sensor)
            if bool(args.stationary):
                occluder.set_simulate_physics(False)
            else:
                occluder.set_simulate_physics(True)
                occluder.apply_control(carla.VehicleControl())
                occluder.set_autopilot(False, int(args.tm_port))
                controllers["occluder"] = DirectRouteController(
                    occluder,
                    role_routes["occluder"],
                    target_speed_mps=queue_occluder_speed_mps,
                    waypoint_reach_m=0.75,
                )

        if not bool(args.pose_only):
            camera_transform = carla.Transform(
                carla.Location(x=1.8, y=0.0, z=1.55),
                carla.Rotation(pitch=-4.0, yaw=0.0, roll=0.0),
            )
            for role in ROLE_ORDER:
                camera = world.spawn_actor(
                    _camera_blueprint(world),
                    camera_transform,
                    attach_to=vehicles[role],
                    attachment_type=carla.AttachmentType.Rigid,
                )
                camera.listen(lambda image, name=role: mailbox.push(name, image))
                cameras.append(camera)
                owned.append(camera)

        print(
            "Geometry review ready: "
            f"layout={args.layout}, "
            f"scenario_role={args.scenario_role}, "
            f"hazard_actor_present={hazard_present}, "
            f"lane_contract_pass={lane_contract is not None}, "
            f"geometry_origin={geometry_origin}, "
            f"separation={separation:.2f}m, "
            f"recipient_target_distance={transforms['recipient'].location.distance(target_location):.2f}m, "
            f"helper_target_distance={transforms['helper'].location.distance(target_location):.2f}m, "
            f"motion={'stationary' if args.stationary else 'direct_role_specific_route'}",
            flush=True,
        )
        if str(args.layout) == "signalized_corner":
            print(
                f"Signalized candidate={SIGNALIZED_GEOMETRY_ID}; "
                f"initial_target_bearings_deg={line_of_sight_bearings_deg()}; "
                "recipient/helper approach signals forced green for this review",
                flush=True,
            )
        elif str(args.layout) == "midblock_van":
            print(
                f"Frozen midblock geometry={MIDBLOCK_GEOMETRY_ID}; "
                f"initial_target_bearings_deg={midblock_line_of_sight_bearings_deg()}; "
                f"occluder_settlement={occluder_settlement}; "
                "no junction or traffic-light override",
                flush=True,
            )
        elif str(args.layout) == "cross_traffic_vehicle":
            print(
                f"Frozen cross-traffic geometry={CROSS_TRAFFIC_GEOMETRY_ID}; "
                f"static_visibility={lane_contract['initial_visibility']}; "
                f"registered_conflict={lane_contract['registered_conflict_point']}; "
                "target route is visually accepted and hash-frozen; review-only independent "
                "green signal override is not a legal-phase claim",
                flush=True,
            )
        elif str(args.layout) == "parked_vehicle_pullout":
            print(
                f"Frozen parked-pullout geometry={PULLOUT_GEOMETRY_ID}; "
                f"static_visibility={lane_contract['initial_visibility']}; "
                f"registered_merge={lane_contract['registered_conflict_point']}; "
                f"target_start_delay_s={target_vehicle_start_delay_s:.1f}; "
                "target route is visually accepted and byte-hash frozen; "
                "overall collection remains separately authorized",
                flush=True,
            )
        elif str(args.layout) == "queue_reveal_vehicle":
            print(
                f"Queue-reveal candidate={QUEUE_REVEAL_GEOMETRY_ID}; "
                f"static_visibility={lane_contract['initial_visibility']}; "
                f"stopped_lead={lane_contract['registered_conflict_point']}; "
                f"queue_move_delay_s={queue_occluder_start_delay_s:.1f}; "
                "the Sprinter curb-exit route is explicit but not frozen or "
                "collection-authorized",
                flush=True,
            )
        if bool(args.pose_only):
            print("Pose-only smoke: camera capture disabled.", flush=True)
        elif bool(args.headless):
            print("Headless camera smoke: critical-time views will be saved.", flush=True)
        else:
            print("Press s to save views; q or Escape to stop.", flush=True)

        review_start_frame = int(world.get_snapshot().frame)
        elapsed_sim_s = 0.0
        while elapsed_sim_s < float(args.duration_s):
            tick_started = time.monotonic()
            review_safety_yield_active = False
            if (
                str(args.layout) == "queue_reveal_vehicle"
                and target_vehicle is not None
            ):
                target_vehicle_started = True
            for controller_role, controller in controllers.items():
                if (
                    controller_role == "occluder"
                    and elapsed_sim_s < queue_occluder_start_delay_s
                ):
                    if occluder is None:
                        raise RuntimeError("queue controller lacks its occluder")
                    occluder.apply_control(
                        carla.VehicleControl(
                            throttle=0.0,
                            brake=1.0,
                            hand_brake=True,
                        )
                    )
                    continue
                if controller_role == "occluder":
                    queue_occluder_started = True
                if (
                    controller_role == "target"
                    and elapsed_sim_s < target_vehicle_start_delay_s
                ):
                    if target_vehicle is None:
                        raise RuntimeError("target controller lacks its vehicle")
                    target_vehicle.apply_control(
                        carla.VehicleControl(
                            throttle=0.0,
                            brake=1.0,
                            hand_brake=True,
                        )
                    )
                    continue
                if controller_role == "target":
                    target_vehicle_started = True
                if (
                    controller_role == "recipient"
                    and str(args.layout)
                    in {
                        "cross_traffic_vehicle",
                        "parked_vehicle_pullout",
                        "queue_reveal_vehicle",
                    }
                    and target_vehicle is not None
                    and elapsed_sim_s
                    >= (
                        queue_occluder_start_delay_s
                        if str(args.layout) == "queue_reveal_vehicle"
                        else target_vehicle_start_delay_s
                    )
                ):
                    conflict = lane_contract["registered_conflict_point"]
                    recipient_location = vehicles["recipient"].get_location()
                    target_location_live = target_vehicle.get_location()
                    recipient_conflict_distance = math.hypot(
                        float(recipient_location.x) - float(conflict["x"]),
                        float(recipient_location.y) - float(conflict["y"]),
                    )
                    target_has_cleared = (
                        False
                        if str(args.layout) == "queue_reveal_vehicle"
                        else bool(
                            float(target_location_live.y)
                            >= float(conflict["y"]) + 4.5
                            if str(args.layout) == "cross_traffic_vehicle"
                            else float(target_location_live.x)
                            >= float(conflict["x"]) + 4.5
                        )
                    )
                    yield_trigger_m = (
                        CROSS_TRAFFIC_REVIEW_YIELD_TRIGGER_M
                        if str(args.layout) == "cross_traffic_vehicle"
                        else (
                            PULLOUT_REVIEW_YIELD_TRIGGER_M
                            if str(args.layout) == "parked_vehicle_pullout"
                            else QUEUE_REVEAL_REVIEW_YIELD_TRIGGER_M
                        )
                    )
                    if (
                        recipient_conflict_distance <= yield_trigger_m
                        and not target_has_cleared
                    ):
                        vehicles["recipient"].apply_control(
                            carla.VehicleControl(
                                throttle=0.0,
                                brake=1.0,
                                hand_brake=False,
                            )
                        )
                        review_safety_yield_active = True
                        review_safety_yield_ever = True
                        continue
                controller.tick()
            frame = int(world.tick(float(args.timeout_s)))
            elapsed_sim_s = max(0, frame - review_start_frame) * 0.1
            if (
                walker is not None
                and walker_end is not None
                and not walker_started
                and elapsed_sim_s >= float(args.pedestrian_start_delay_s)
            ):
                location = walker.get_location()
                delta_x = float(walker_end.x - location.x)
                delta_y = float(walker_end.y - location.y)
                norm = math.hypot(delta_x, delta_y)
                if norm <= 1e-6:
                    raise RuntimeError("controlled pedestrian path has zero length")
                walker.apply_control(
                    carla.WalkerControl(
                        direction=carla.Vector3D(
                            x=delta_x / norm,
                            y=delta_y / norm,
                            z=0.0,
                        ),
                        speed=(
                            float(args.pedestrian_speed_mps)
                            / CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE
                        ),
                        jump=False,
                    )
                )
                walker_started = True
                print(
                    f"Controlled pedestrian started at {args.pedestrian_speed_mps:.2f} m/s",
                    flush=True,
                )
            if (
                walker is not None
                and walker_end is not None
                and walker_started
                and not walker_completed
            ):
                walker_location = walker.get_location()
                if walker_location.distance(walker_end) <= 0.25:
                    walker.apply_control(
                        carla.WalkerControl(
                            direction=carla.Vector3D(0.0, 0.0, 0.0),
                            speed=0.0,
                            jump=False,
                        )
                    )
                    walker_completed = True
                    print(
                        "Controlled pedestrian reached and is holding at endpoint",
                        flush=True,
                    )

            trace_row = {
                "frame": frame,
                "elapsed_sim_s": elapsed_sim_s,
                "walker_started": int(walker_started),
                "walker_completed": int(walker_completed),
                "review_safety_yield_active": int(review_safety_yield_active),
                "target_vehicle_started": int(target_vehicle_started),
                "queue_occluder_started": int(queue_occluder_started),
            }
            if walker is None:
                trace_row.update(
                    walker_x="",
                    walker_y="",
                    walker_z="",
                    walker_speed_mps=0.0,
                )
            else:
                walker_location = walker.get_location()
                walker_velocity = walker.get_velocity()
                trace_row.update(
                    walker_x=float(walker_location.x),
                    walker_y=float(walker_location.y),
                    walker_z=float(walker_location.z),
                    walker_speed_mps=math.sqrt(
                        float(walker_velocity.x) ** 2
                        + float(walker_velocity.y) ** 2
                        + float(walker_velocity.z) ** 2
                    ),
                )
            if target_vehicle is None:
                trace_row.update(
                    target_vehicle_x="",
                    target_vehicle_y="",
                    target_vehicle_z="",
                    target_vehicle_yaw="",
                    target_vehicle_speed_mps=0.0,
                    helper_target_range_m="",
                    helper_target_bearing_deg="",
                    helper_target_in_fov=0,
                    helper_target_occluded=0,
                    helper_target_visible=0,
                    recipient_target_range_m="",
                    recipient_target_bearing_deg="",
                    recipient_target_in_fov=0,
                    recipient_target_occluded=0,
                    recipient_target_visible=0,
                )
            else:
                if occluder is None:
                    raise RuntimeError("cross-traffic target lacks its controlled occluder")
                target_transform = target_vehicle.get_transform()
                target_velocity = target_vehicle.get_velocity()
                trace_row.update(
                    target_vehicle_x=float(target_transform.location.x),
                    target_vehicle_y=float(target_transform.location.y),
                    target_vehicle_z=float(target_transform.location.z),
                    target_vehicle_yaw=float(target_transform.rotation.yaw),
                    target_vehicle_speed_mps=math.sqrt(
                        float(target_velocity.x) ** 2
                        + float(target_velocity.y) ** 2
                        + float(target_velocity.z) ** 2
                    ),
                )
                occluder_box = occluder.bounding_box
                for role, observer in vehicles.items():
                    visibility = cross_traffic_visibility_state(
                        observer.get_transform(),
                        target_transform,
                        occluder.get_transform(),
                        occluder_half_length_m=float(occluder_box.extent.x) + 0.10,
                        occluder_half_width_m=float(occluder_box.extent.y) + 0.10,
                        occluder_local_center=occluder_box.location,
                        occluder_local_yaw_deg=float(occluder_box.rotation.yaw),
                    )
                    trace_row.update(
                        {
                            f"{role}_target_range_m": float(visibility["range_m"]),
                            f"{role}_target_bearing_deg": float(
                                visibility["relative_bearing_deg"]
                            ),
                            f"{role}_target_in_fov": int(visibility["in_fov"]),
                            f"{role}_target_occluded": int(
                                visibility["occluded_by_controlled_truck"]
                            ),
                            f"{role}_target_visible": int(
                                visibility["geometrically_visible"]
                            ),
                        }
                    )
            if occluder is None:
                trace_row.update(
                    occluder_x="",
                    occluder_y="",
                    occluder_z="",
                    occluder_yaw="",
                    occluder_speed_mps=0.0,
                    occluder_control_throttle=0.0,
                    occluder_control_steer=0.0,
                    occluder_control_brake=0.0,
                    occluder_control_hand_brake=0,
                )
            else:
                live_occluder_transform = occluder.get_transform()
                live_occluder_velocity = occluder.get_velocity()
                live_occluder_control = occluder.get_control()
                trace_row.update(
                    occluder_x=float(live_occluder_transform.location.x),
                    occluder_y=float(live_occluder_transform.location.y),
                    occluder_z=float(live_occluder_transform.location.z),
                    occluder_yaw=float(live_occluder_transform.rotation.yaw),
                    occluder_speed_mps=math.sqrt(
                        float(live_occluder_velocity.x) ** 2
                        + float(live_occluder_velocity.y) ** 2
                        + float(live_occluder_velocity.z) ** 2
                    ),
                    occluder_control_throttle=float(live_occluder_control.throttle),
                    occluder_control_steer=float(live_occluder_control.steer),
                    occluder_control_brake=float(live_occluder_control.brake),
                    occluder_control_hand_brake=int(
                        bool(live_occluder_control.hand_brake)
                    ),
                )
            queue_yield = (
                controllers["occluder"].last_yield
                if "occluder" in controllers
                else None
            )
            trace_row.update(
                queue_occluder_yield_actor_id=(
                    int(queue_yield["actor_id"]) if queue_yield is not None else ""
                ),
                queue_occluder_yield_actor_type=(
                    str(queue_yield["type_id"]) if queue_yield is not None else ""
                ),
                queue_occluder_yield_forward_m=(
                    float(queue_yield["forward_m"])
                    if queue_yield is not None
                    else ""
                ),
                queue_occluder_yield_lateral_m=(
                    float(queue_yield["lateral_m"])
                    if queue_yield is not None
                    else ""
                ),
                queue_occluder_yield_stopping_m=(
                    float(queue_yield["stopping_m"])
                    if queue_yield is not None
                    else ""
                ),
                queue_occluder_yield_lateral_limit_m=(
                    float(queue_yield["lateral_limit_m"])
                    if queue_yield is not None
                    else ""
                ),
            )
            for role, actor in vehicles.items():
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                trace_row.update(
                    {
                        f"{role}_x": float(transform.location.x),
                        f"{role}_y": float(transform.location.y),
                        f"{role}_z": float(transform.location.z),
                        f"{role}_yaw": float(transform.rotation.yaw),
                        f"{role}_speed_mps": math.sqrt(
                            float(velocity.x) ** 2
                            + float(velocity.y) ** 2
                            + float(velocity.z) ** 2
                        ),
                    }
                )
            realized_trace.append(trace_row)

            if bool(args.pose_only):
                continue
            frame_deadline = time.monotonic() + 1.0
            views = None
            while time.monotonic() < frame_deadline:
                views = mailbox.aligned(frame)
                if views is not None:
                    break
                time.sleep(0.005)
            if views is None:
                continue
            latest_views = views
            annotated = [
                _annotate(
                    views[role][1],
                    role,
                    views[role][0],
                    str(args.scenario_role),
                )
                for role in ROLE_ORDER
            ]
            combined = np.concatenate(annotated, axis=1)
            if bool(args.headless):
                for capture_time_s in automatic_capture_times_s:
                    if (
                        capture_time_s not in automatic_captures_written
                        and elapsed_sim_s >= capture_time_s
                    ):
                        _save_views(output_dir, views, combined)
                        automatic_captures_written.append(capture_time_s)
                continue
            cv2.imshow("Phase-2 paired geometry review", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                _save_views(output_dir, views, combined)
            tick_elapsed = time.monotonic() - tick_started
            if tick_elapsed < 0.1:
                time.sleep(0.1 - tick_elapsed)

        if latest_views is not None:
            annotated = [
                _annotate(
                    latest_views[role][1],
                    role,
                    latest_views[role][0],
                    str(args.scenario_role),
                )
                for role in ROLE_ORDER
            ]
            _save_views(
                output_dir,
                latest_views,
                np.concatenate(annotated, axis=1),
            )
        summary = {
            "schema": "scenesense.phase2_geometry_review.v1",
            "layout": str(args.layout),
            "geometry_id": (
                SIGNALIZED_GEOMETRY_ID
                if str(args.layout) == "signalized_corner"
                else (
                    MIDBLOCK_GEOMETRY_ID
                    if str(args.layout) == "midblock_van"
                    else (
                        CROSS_TRAFFIC_GEOMETRY_ID
                        if str(args.layout) == "cross_traffic_vehicle"
                        else (
                            PULLOUT_GEOMETRY_ID
                            if str(args.layout) == "parked_vehicle_pullout"
                            else (
                                QUEUE_REVEAL_GEOMETRY_ID
                                if str(args.layout) == "queue_reveal_vehicle"
                                else None
                            )
                        )
                    )
                )
            ),
            "scenario_role": str(args.scenario_role),
            "hazard_actor_present": bool(hazard_present),
            "benign_single_difference": (
                None
                if hazard_present
                else (
                    "target_vehicle_absent"
                    if str(args.layout)
                    in {
                        "cross_traffic_vehicle",
                        "parked_vehicle_pullout",
                        "queue_reveal_vehicle",
                    }
                    else "pedestrian_absent"
                )
            ),
            "lane_contract": lane_contract,
            "legal_opposing_lane_contract": (
                lane_contract if str(args.layout) == "curbside_opposite" else None
            ),
            "geometry_origin": geometry_origin,
            "route_progress_csv": (
                str(route_path.resolve()) if route_path is not None else None
            ),
            "route_source": (
                "visually_accepted_frozen_progress_csv"
                if str(args.layout) == "cross_traffic_vehicle"
                else (
                    "visually_accepted_byte_frozen_ego_and_target_routes"
                    if str(args.layout) == "parked_vehicle_pullout"
                    else (
                        "visually_accepted_byte_frozen_ego_and_queue_exit_routes"
                        if str(args.layout) == "queue_reveal_vehicle"
                        else (
                            "visually_accepted_frozen_progress_csv"
                            if str(args.layout)
                            in {"signalized_corner", "midblock_van"}
                            else "frozen_progress_csv"
                        )
                    )
                )
            ),
            "production_camera": {
                "width": 1280,
                "height": 720,
                "fov_deg": 120.0,
            },
            "world_hz": 10.0,
            "target_vehicle_command_speed_mps": (
                target_vehicle_speed_mps
                if target_vehicle is not None
                and str(args.layout) != "queue_reveal_vehicle"
                else 0.0
            ),
            "target_vehicle_start_delay_s": (
                target_vehicle_start_delay_s if target_vehicle is not None else 0.0
            ),
            "pedestrian_speed_mps": (
                float(args.pedestrian_speed_mps)
                if hazard_present and walker_end is not None
                else 0.0
            ),
            "walker_control_command_speed": (
                float(args.pedestrian_speed_mps)
                / CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE
                if hazard_present and walker_end is not None
                else 0.0
            ),
            "walker_control_to_physical_scale": (
                CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE
                if hazard_present and walker_end is not None
                else None
            ),
            "traffic_light_override": (
                {
                    "reason": "deterministic_green_controlled_approaches",
                    "roles": [role for role, *_unused in traffic_light_restore],
                }
                if traffic_light_restore
                else None
            ),
            "occluder_settlement": occluder_settlement,
            "target_settlement": target_settlement,
            "queue_occluder_command_speed_mps": (
                queue_occluder_speed_mps
                if str(args.layout) == "queue_reveal_vehicle"
                else 0.0
            ),
            "queue_occluder_start_delay_s": (
                queue_occluder_start_delay_s
                if str(args.layout) == "queue_reveal_vehicle"
                else 0.0
            ),
            "queue_occluder_started": bool(queue_occluder_started),
            "pedestrian_started": bool(walker_started),
            "pedestrian_completed": bool(walker_completed),
            "review_only_gt_safety_yield_ever": bool(review_safety_yield_ever),
            "collisions": collisions,
            "automatic_captures_written_s": automatic_captures_written,
            "controllers_finished": {
                role: bool(controller.finished)
                for role, controller in controllers.items()
            },
            "owned_actor_transforms": {
                role: {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                    "yaw": float(transform.rotation.yaw),
                }
                for role, transform in transforms.items()
            },
        }
        role_progress_m = {}
        if realized_trace:
            final_row = realized_trace[-1]
            for role in ROLE_ORDER:
                start_transform = transforms[role]
                forward = start_transform.get_forward_vector()
                role_progress_m[role] = (
                    (float(final_row[f"{role}_x"]) - float(start_transform.location.x))
                    * float(forward.x)
                    + (float(final_row[f"{role}_y"]) - float(start_transform.location.y))
                    * float(forward.y)
                )
        summary["role_longitudinal_progress_m"] = role_progress_m
        summary["matched_benign_motion_gate_pass"] = (
            None
            if hazard_present
            else bool(
                not collisions
                and all(role_progress_m.get(role, 0.0) >= 25.0 for role in ROLE_ORDER)
            )
        )
        active_walker_speeds = [
            float(row["walker_speed_mps"])
            for row in realized_trace
            if bool(row["walker_started"])
            and not bool(row["walker_completed"])
            and float(row["walker_speed_mps"]) > 0.05
        ]
        summary["pedestrian_realized_speed_mps_median"] = (
            float(np.median(active_walker_speeds))
            if active_walker_speeds
            else None
        )
        realized_walker_speed = summary["pedestrian_realized_speed_mps_median"]
        summary["pedestrian_physical_speed_gate_pass"] = (
            None
            if len(active_walker_speeds) < 5
            else bool(
                0.85 * float(args.pedestrian_speed_mps)
                <= float(realized_walker_speed)
                <= 1.15 * float(args.pedestrian_speed_mps)
            )
        )
        vehicle_hazard_review_gate: Optional[Dict[str, object]] = None
        if str(args.layout) in {
            "cross_traffic_vehicle",
            "parked_vehicle_pullout",
            "queue_reveal_vehicle",
        }:
            if target_vehicle is None:
                vehicle_hazard_review_gate = {
                    "pass": bool(summary["matched_benign_motion_gate_pass"]),
                    "basis": (
                        "matched_benign_target_absent_zero_collisions_and_ego_motion"
                    ),
                    "target_present": False,
                }
            elif str(args.layout) == "queue_reveal_vehicle":
                helper_visible_times = [
                    float(row["elapsed_sim_s"])
                    for row in realized_trace
                    if bool(row["helper_target_visible"])
                ]
                recipient_visible_times = [
                    float(row["elapsed_sim_s"])
                    for row in realized_trace
                    if bool(row["recipient_target_visible"])
                ]
                first_helper_visible_s = (
                    min(helper_visible_times) if helper_visible_times else None
                )
                first_recipient_visible_s = (
                    min(recipient_visible_times) if recipient_visible_times else None
                )
                visibility_lead_s = (
                    float(first_recipient_visible_s - first_helper_visible_s)
                    if first_helper_visible_s is not None
                    and first_recipient_visible_s is not None
                    else None
                )
                longest_differential_frames = 0
                current_differential_frames = 0
                for row in realized_trace:
                    differential = bool(row["helper_target_visible"]) and not bool(
                        row["recipient_target_visible"]
                    )
                    current_differential_frames = (
                        current_differential_frames + 1 if differential else 0
                    )
                    longest_differential_frames = max(
                        longest_differential_frames, current_differential_frames
                    )
                differential_visibility_s = 0.1 * longest_differential_frames

                moving_queue_rows = [
                    row
                    for row in realized_trace
                    if bool(row["queue_occluder_started"])
                ]
                active_queue_speeds = [
                    float(row["occluder_speed_mps"])
                    for row in moving_queue_rows
                    if float(row["occluder_speed_mps"]) > 0.2
                ]
                queue_speed_median = (
                    float(np.median(active_queue_speeds))
                    if active_queue_speeds
                    else None
                )
                queue_cross_track = [
                    min(
                        math.hypot(
                            float(row["occluder_x"]) - float(point.x),
                            float(row["occluder_y"]) - float(point.y),
                        )
                        for point in role_routes["occluder"]
                    )
                    for row in moving_queue_rows
                ]
                queue_route_max_cross_track_m = (
                    max(queue_cross_track) if queue_cross_track else None
                )
                queue_route_gate = bool(
                    queue_route_max_cross_track_m is not None
                    and queue_route_max_cross_track_m <= 1.0
                    and controllers.get("occluder") is not None
                    and controllers["occluder"].finished
                )
                target_max_speed_mps = max(
                    float(row["target_vehicle_speed_mps"])
                    for row in realized_trace
                )
                conflict = lane_contract["registered_conflict_point"]
                target_conflict_min_distance_m = min(
                    math.hypot(
                        float(row["target_vehicle_x"]) - float(conflict["x"]),
                        float(row["target_vehicle_y"]) - float(conflict["y"]),
                    )
                    for row in realized_trace
                )
                yield_rows = [
                    row
                    for row in realized_trace
                    if bool(row["review_safety_yield_active"])
                ]
                recipient_conflict_distances_during_yield = [
                    math.hypot(
                        float(row["recipient_x"]) - float(conflict["x"]),
                        float(row["recipient_y"]) - float(conflict["y"]),
                    )
                    for row in yield_rows
                ]
                minimum_recipient_conflict_clearance_m = (
                    min(recipient_conflict_distances_during_yield)
                    if recipient_conflict_distances_during_yield
                    else None
                )
                stopped_after_reveal_frames = 0
                longest_stopped_after_reveal_frames = 0
                for row in realized_trace:
                    after_reveal = bool(
                        first_recipient_visible_s is not None
                        and float(row["elapsed_sim_s"])
                        >= float(first_recipient_visible_s)
                    )
                    stopped_near_lead = bool(
                        after_reveal
                        and float(row["recipient_speed_mps"]) <= 0.25
                        and math.hypot(
                            float(row["recipient_x"]) - float(conflict["x"]),
                            float(row["recipient_y"]) - float(conflict["y"]),
                        )
                        <= 15.0
                    )
                    stopped_after_reveal_frames = (
                        stopped_after_reveal_frames + 1
                        if stopped_near_lead
                        else 0
                    )
                    longest_stopped_after_reveal_frames = max(
                        longest_stopped_after_reveal_frames,
                        stopped_after_reveal_frames,
                    )
                stopped_after_reveal_s = 0.1 * longest_stopped_after_reveal_frames
                queue_speed_gate = bool(
                    len(active_queue_speeds) >= 5
                    and queue_speed_median is not None
                    and 0.8 <= queue_speed_median <= 3.5
                )
                safety_stop_gate = bool(
                    review_safety_yield_ever
                    and minimum_recipient_conflict_clearance_m is not None
                    and 4.0 <= minimum_recipient_conflict_clearance_m <= 15.0
                    and stopped_after_reveal_s >= 1.0
                )
                vehicle_hazard_review_gate = {
                    "pass": bool(
                        not collisions
                        and target_max_speed_mps <= 0.1
                        and queue_speed_gate
                        and queue_route_gate
                        and safety_stop_gate
                        and visibility_lead_s is not None
                        and visibility_lead_s >= 0.8
                        and differential_visibility_s >= 0.8
                        and target_conflict_min_distance_m <= 0.75
                    ),
                    "basis": (
                        "zero_collisions_stationary_lead_realistic_queue_exit_"
                        "registered_stop_and_helper_before_recipient_visibility"
                    ),
                    "target_present": True,
                    "target_stationary_max_speed_mps": float(target_max_speed_mps),
                    "queue_occluder_speed_mps_median_while_moving": (
                        queue_speed_median
                    ),
                    "queue_occluder_speed_gate_pass": queue_speed_gate,
                    "queue_occluder_route_gate_pass": queue_route_gate,
                    "queue_occluder_route_max_cross_track_m": (
                        queue_route_max_cross_track_m
                    ),
                    "queue_occluder_route_maximum_allowed_cross_track_m": 1.0,
                    "review_only_gt_safety_stop_gate_pass": safety_stop_gate,
                    "review_only_gt_safety_yield_duration_s": 0.1
                    * len(yield_rows),
                    "recipient_sustained_stop_after_reveal_s": float(
                        stopped_after_reveal_s
                    ),
                    "minimum_recipient_conflict_clearance_during_yield_m": (
                        minimum_recipient_conflict_clearance_m
                    ),
                    "first_helper_visible_s": first_helper_visible_s,
                    "first_recipient_visible_s": first_recipient_visible_s,
                    "helper_visibility_lead_s": visibility_lead_s,
                    "longest_differential_visibility_s": float(
                        differential_visibility_s
                    ),
                    "target_conflict_min_distance_m": float(
                        target_conflict_min_distance_m
                    ),
                    "target_conflict_maximum_allowed_m": 0.75,
                    "collision_count": len(collisions),
                }
            else:
                active_target_speeds = [
                    float(row["target_vehicle_speed_mps"])
                    for row in realized_trace
                    if float(row["elapsed_sim_s"]) >= 2.0
                    and float(row["target_vehicle_speed_mps"]) > 0.2
                ]
                target_speed_median = (
                    float(np.median(active_target_speeds))
                    if active_target_speeds
                    else None
                )
                post_merge_target_speeds = [
                    float(row["target_vehicle_speed_mps"])
                    for row in realized_trace
                    if str(args.layout) == "parked_vehicle_pullout"
                    and float(row["target_vehicle_x"])
                    >= float(lane_contract["registered_conflict_point"]["x"]) + 8.5
                    and float(row["target_vehicle_x"])
                    <= float(role_routes["target"][-1].x) - 2.5
                    and float(row["target_vehicle_speed_mps"]) > 0.2
                ]
                post_merge_target_speed_median = (
                    float(np.median(post_merge_target_speeds))
                    if post_merge_target_speeds
                    else None
                )
                helper_visible_times = [
                    float(row["elapsed_sim_s"])
                    for row in realized_trace
                    if bool(row["helper_target_visible"])
                ]
                recipient_visible_times = [
                    float(row["elapsed_sim_s"])
                    for row in realized_trace
                    if bool(row["recipient_target_visible"])
                ]
                first_helper_visible_s = (
                    min(helper_visible_times) if helper_visible_times else None
                )
                first_recipient_visible_s = (
                    min(recipient_visible_times) if recipient_visible_times else None
                )
                visibility_lead_s = (
                    float(first_recipient_visible_s - first_helper_visible_s)
                    if first_helper_visible_s is not None
                    and first_recipient_visible_s is not None
                    else None
                )
                longest_differential_frames = 0
                current_differential_frames = 0
                for row in realized_trace:
                    differential = bool(row["helper_target_visible"]) and not bool(
                        row["recipient_target_visible"]
                    )
                    current_differential_frames = (
                        current_differential_frames + 1 if differential else 0
                    )
                    longest_differential_frames = max(
                        longest_differential_frames, current_differential_frames
                    )
                differential_visibility_s = 0.1 * longest_differential_frames
                conflict = lane_contract["registered_conflict_point"]
                target_conflict_distances = [
                    math.hypot(
                        float(row["target_vehicle_x"]) - float(conflict["x"]),
                        float(row["target_vehicle_y"]) - float(conflict["y"]),
                    )
                    for row in realized_trace
                ]
                target_conflict_min_distance_m = (
                    min(target_conflict_distances)
                    if target_conflict_distances
                    else None
                )
                moving_target_rows = [
                    row
                    for row in realized_trace
                    if bool(row["target_vehicle_started"])
                ]
                target_route_cross_track_distances_m = [
                    min(
                        math.hypot(
                            float(row["target_vehicle_x"]) - float(point.x),
                            float(row["target_vehicle_y"]) - float(point.y),
                        )
                        for point in role_routes["target"]
                    )
                    for row in moving_target_rows
                ]
                target_route_max_cross_track_m = (
                    max(target_route_cross_track_distances_m)
                    if target_route_cross_track_distances_m
                    else None
                )
                target_route_gate = bool(
                    target_route_max_cross_track_m is not None
                    and target_route_max_cross_track_m
                    <= (
                        1.0
                        if str(args.layout) == "parked_vehicle_pullout"
                        else 2.0
                    )
                )
                yield_rows = [
                    row
                    for row in realized_trace
                    if bool(row["review_safety_yield_active"])
                ]
                recipient_conflict_distances_during_yield = [
                    math.hypot(
                        float(row["recipient_x"]) - float(conflict["x"]),
                        float(row["recipient_y"]) - float(conflict["y"]),
                    )
                    for row in yield_rows
                ]
                minimum_recipient_conflict_clearance_m = (
                    min(recipient_conflict_distances_during_yield)
                    if recipient_conflict_distances_during_yield
                    else None
                )
                safety_yield_gate = bool(
                    review_safety_yield_ever
                    and minimum_recipient_conflict_clearance_m is not None
                    and minimum_recipient_conflict_clearance_m >= 4.0
                )
                if str(args.layout) == "parked_vehicle_pullout":
                    target_speed_gate = bool(
                        target_speed_median is not None
                        and 1.0 <= target_speed_median <= 3.5
                        and len(post_merge_target_speeds) >= 10
                        and post_merge_target_speed_median is not None
                        and 0.75 * target_vehicle_speed_mps
                        <= post_merge_target_speed_median
                        <= 1.25 * target_vehicle_speed_mps
                    )
                    target_speed_gate_basis = (
                        "realistic_turning_median_and_post_merge_command_tracking"
                    )
                else:
                    target_speed_gate = bool(
                        target_speed_median is not None
                        and 0.75 * target_vehicle_speed_mps
                        <= target_speed_median
                        <= 1.25 * target_vehicle_speed_mps
                    )
                    target_speed_gate_basis = "whole_route_command_tracking"
                vehicle_hazard_review_gate = {
                    "pass": bool(
                        not collisions
                        and target_speed_gate
                        and target_route_gate
                        and safety_yield_gate
                        and visibility_lead_s is not None
                        and visibility_lead_s >= 0.8
                        and differential_visibility_s >= 0.8
                        and target_conflict_min_distance_m is not None
                        and target_conflict_min_distance_m
                        <= (
                            0.75
                            if str(args.layout) == "parked_vehicle_pullout"
                            else 1.5
                        )
                    ),
                    "basis": (
                        "zero_collisions_realistic_target_motion_registered_conflict_"
                        "and_helper_before_recipient_geometric_visibility"
                    ),
                    "target_present": True,
                    "target_speed_mps_median_while_moving": target_speed_median,
                    "target_post_merge_speed_mps_median": (
                        post_merge_target_speed_median
                    ),
                    "target_speed_gate_pass": target_speed_gate,
                    "target_speed_gate_basis": target_speed_gate_basis,
                    "target_route_gate_pass": target_route_gate,
                    "target_route_max_cross_track_m": target_route_max_cross_track_m,
                    "target_route_maximum_allowed_cross_track_m": (
                        1.0
                        if str(args.layout) == "parked_vehicle_pullout"
                        else 2.0
                    ),
                    "review_only_gt_safety_yield_gate_pass": safety_yield_gate,
                    "review_only_gt_safety_yield_duration_s": 0.1
                    * len(yield_rows),
                    "minimum_recipient_conflict_clearance_during_yield_m": (
                        minimum_recipient_conflict_clearance_m
                    ),
                    "first_helper_visible_s": first_helper_visible_s,
                    "first_recipient_visible_s": first_recipient_visible_s,
                    "helper_visibility_lead_s": visibility_lead_s,
                    "longest_differential_visibility_s": float(
                        differential_visibility_s
                    ),
                    "target_conflict_min_distance_m": target_conflict_min_distance_m,
                    "target_conflict_maximum_allowed_m": (
                        0.75
                        if str(args.layout) == "parked_vehicle_pullout"
                        else 1.5
                    ),
                    "collision_count": len(collisions),
                }
        summary["vehicle_hazard_review_gate"] = vehicle_hazard_review_gate
        summary["cross_traffic_review_gate"] = (
            vehicle_hazard_review_gate
            if str(args.layout) == "cross_traffic_vehicle"
            else None
        )
        summary["parked_vehicle_pullout_review_gate"] = (
            vehicle_hazard_review_gate
            if str(args.layout) == "parked_vehicle_pullout"
            else None
        )
        summary["queue_reveal_vehicle_review_gate"] = (
            vehicle_hazard_review_gate
            if str(args.layout) == "queue_reveal_vehicle"
            else None
        )
        (output_dir / "geometry_review_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if realized_trace:
            with (output_dir / "realized_pose_trace.csv").open(
                "x", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(realized_trace[0]))
                writer.writeheader()
                writer.writerows(realized_trace)
        if summary["pedestrian_physical_speed_gate_pass"] is False:
            raise RuntimeError(
                "realized pedestrian speed is outside the physical-speed contract: "
                f"requested={float(args.pedestrian_speed_mps):.3f}m/s, "
                f"realized_median={float(realized_walker_speed):.3f}m/s"
            )
        if summary["matched_benign_motion_gate_pass"] is False:
            raise RuntimeError(
                "matched benign motion gate failed: expected zero collisions and "
                f">=25m forward progress per role, observed={role_progress_m}"
            )
        if (
            vehicle_hazard_review_gate is not None
            and vehicle_hazard_review_gate["pass"] is False
        ):
            raise RuntimeError(
                "vehicle-hazard geometry review gate failed: "
                f"{vehicle_hazard_review_gate}"
            )
    finally:
        cv2.destroyAllWindows()
        for camera in reversed(cameras):
            try:
                if camera.is_alive:
                    camera.stop()
            except RuntimeError:
                pass
        for sensor in reversed(collision_sensors):
            try:
                if sensor.is_alive:
                    sensor.stop()
            except RuntimeError:
                pass
        for actor in reversed(owned):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        for _role, traffic_light, original_state, original_frozen in reversed(
            traffic_light_restore
        ):
            try:
                if traffic_light.is_alive:
                    traffic_light.set_state(original_state)
                    traffic_light.freeze(bool(original_frozen))
            except RuntimeError:
                pass
        try:
            world.tick(float(args.timeout_s))
        except RuntimeError:
            pass
        world.apply_settings(original_settings)
        remaining = _dynamic_inventory(world)
        if any(remaining.values()):
            print(f"WARNING: post-review dynamic actors remain: {remaining}", flush=True)

    print(
        f"Geometry review finished; collisions={len(collisions)}; "
        f"artifacts: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

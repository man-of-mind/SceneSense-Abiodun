"""Frozen, road-legal Phase-2 curbside geometry and motion primitives.

This module is the single source shared by the visual-review instrument and
the paired causal pilot.  It deliberately contains no collector, inference,
or evaluation logic.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import carla


ROLE_ORDER = ("helper", "recipient")
CURBSIDE_RECIPIENT_TRANSFORM = (
    57.568340302,
    -67.849853516,
    0.600000,
    179.9765625,
)
# Lane +1 is the nearest legal opposing lane.  The historical 3.6 m offset
# landed on lane -1 and made the nominal helper drive against its legal flow.
CURBSIDE_HELPER_TRANSFORM = (4.505173, -60.912979, 0.400000, 0.535)
CURBSIDE_WALKER_START = (30.498630524, -73.409431458, 1.000000, 89.9765778)
CURBSIDE_WALKER_END = (30.501941681, -65.309432983, 1.000000)
CURBSIDE_OCCLUDER_TRANSFORM = (
    34.999732971,
    -70.711265564,
    0.000000,
    179.9765778,
)
CURBSIDE_OPPOSING_ROUTE_OFFSET_M = 7.0
CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE = 0.05
CURBSIDE_GEOMETRY_ID = "town10hd_opt_curbside_legal_opposing_v1"
CURBSIDE_EXPECTED_LANE_IDS = {"helper": 1, "recipient": -2}


def world_transform(values: Sequence[float]) -> carla.Transform:
    if len(values) != 4:
        raise ValueError("world transform must contain x, y, z, yaw")
    x, y, z, yaw = values
    return carla.Transform(
        carla.Location(x=float(x), y=float(y), z=float(z)),
        carla.Rotation(yaw=float(yaw)),
    )


def load_route_progress(path: Path) -> list[carla.Location]:
    route = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            route.append(
                carla.Location(
                    x=float(row["ego_x"]),
                    y=float(row["ego_y"]),
                    z=float(row.get("ego_z", 0.0)),
                )
            )
    if len(route) < 2:
        raise ValueError("Phase-2 route contains fewer than two points")
    return route


def opposite_lane_route(
    recipient_route: Sequence[carla.Location],
    lateral_offset_m: float = CURBSIDE_OPPOSING_ROUTE_OFFSET_M,
    start_transform: Sequence[float] = CURBSIDE_HELPER_TRANSFORM,
) -> list[carla.Location]:
    """Return the forward-only legal opposing-lane suffix for the helper."""

    if len(recipient_route) < 2:
        raise ValueError("opposite-lane route requires at least two points")
    shifted = [
        carla.Location(
            x=float(point.x),
            y=float(point.y) + float(lateral_offset_m),
            z=float(point.z),
        )
        for point in recipient_route
    ]
    reverse_route = list(reversed(shifted))
    start_x, start_y, _start_z, start_yaw = (float(value) for value in start_transform)
    nearest = min(
        range(len(reverse_route)),
        key=lambda index: math.hypot(
            float(reverse_route[index].x) - start_x,
            float(reverse_route[index].y) - start_y,
        ),
    )
    forward_x = math.cos(math.radians(start_yaw))
    forward_y = math.sin(math.radians(start_yaw))
    while nearest < len(reverse_route) - 1:
        point = reverse_route[nearest]
        ahead_m = (
            (float(point.x) - start_x) * forward_x
            + (float(point.y) - start_y) * forward_y
        )
        if ahead_m > 0.5:
            break
        nearest += 1
    result = reverse_route[nearest:]
    if len(result) < 2:
        raise ValueError("opposite-lane route has no forward suffix from helper spawn")
    return result


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def legal_opposing_lane_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    *,
    expected_lane_ids: Mapping[str, int] | None = None,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Fail unless both poses follow legal, opposing OpenDRIVE lanes."""

    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("lane contract requires exactly helper and recipient transforms")
    if maximum_heading_error_deg <= 0.0:
        raise ValueError("maximum lane-heading error must be positive")
    roles: Dict[str, Dict[str, object]] = {}
    for role in ROLE_ORDER:
        transform = transforms[role]
        waypoint = road_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError(f"{role} start does not project to a driving lane")
        heading_error_deg = abs(
            wrap_degrees(
                float(transform.rotation.yaw)
                - float(waypoint.transform.rotation.yaw)
            )
        )
        roles[role] = {
            "road_id": int(waypoint.road_id),
            "section_id": int(waypoint.section_id),
            "lane_id": int(waypoint.lane_id),
            "lane_heading_deg": float(waypoint.transform.rotation.yaw),
            "commanded_heading_deg": float(transform.rotation.yaw),
            "heading_error_deg": float(heading_error_deg),
        }
        if int(waypoint.lane_id) == 0 or heading_error_deg > maximum_heading_error_deg:
            raise RuntimeError(
                f"{role} violates its legal lane heading: {roles[role]}"
            )
        if expected_lane_ids is not None and int(waypoint.lane_id) != int(
            expected_lane_ids[role]
        ):
            raise RuntimeError(
                f"{role} lane drift: expected {int(expected_lane_ids[role])}, "
                f"observed {int(waypoint.lane_id)}"
            )
    if int(roles["helper"]["lane_id"]) * int(roles["recipient"]["lane_id"]) >= 0:
        raise RuntimeError(
            "helper and recipient are not on opposite OpenDRIVE carriageways: "
            f"{roles}"
        )
    return {
        "pass": True,
        "basis": "opposite_signed_driving_lane_ids_and_native_heading",
        "maximum_heading_error_deg": float(maximum_heading_error_deg),
        "roles": roles,
    }


class DirectRouteController:
    """Non-looping bounded controller used by review and pilot orchestration."""

    def __init__(
        self,
        actor: carla.Actor,
        route: Sequence[carla.Location],
        *,
        target_speed_mps: float,
        waypoint_reach_m: float = 3.5,
    ) -> None:
        if (
            len(route) < 2
            or float(target_speed_mps) <= 0.0
            or float(waypoint_reach_m) <= 0.0
        ):
            raise ValueError("direct route controller configuration is invalid")
        self.actor = actor
        self.route = list(route)
        self.target_speed_mps = float(target_speed_mps)
        self.waypoint_reach_m = float(waypoint_reach_m)
        location = actor.get_location()
        self.index = min(
            range(len(self.route)),
            key=lambda index: self.route[index].distance(location),
        )
        self.finished = False
        self.last_yield: dict[str, object] | None = None

    def _must_yield(self, transform: carla.Transform, speed_mps: float) -> bool:
        self.last_yield = None
        location = transform.location
        forward = transform.get_forward_vector()
        own_box = getattr(self.actor, "bounding_box", None)
        own_half_length_m = float(own_box.extent.x) if own_box is not None else 2.5
        own_half_width_m = float(own_box.extent.y) if own_box is not None else 1.0
        for pattern in ("walker.pedestrian.*", "vehicle.*"):
            for other in self.actor.get_world().get_actors().filter(pattern):
                if int(other.id) == int(self.actor.id):
                    continue
                try:
                    other_transform = other.get_transform()
                except RuntimeError:
                    continue
                other_location = other_transform.location
                dx = float(other_location.x - location.x)
                dy = float(other_location.y - location.y)
                forward_m = dx * float(forward.x) + dy * float(forward.y)
                lateral_m = abs(-dx * float(forward.y) + dy * float(forward.x))
                other_box = getattr(other, "bounding_box", None)
                other_half_length_m = (
                    float(other_box.extent.x) if other_box is not None else 0.4
                )
                other_half_width_m = (
                    float(other_box.extent.y) if other_box is not None else 0.4
                )
                relative_yaw_rad = math.radians(
                    wrap_degrees(
                        float(other_transform.rotation.yaw)
                        - float(transform.rotation.yaw)
                    )
                )
                effective_other_half_length_m = (
                    abs(math.cos(relative_yaw_rad)) * other_half_length_m
                    + abs(math.sin(relative_yaw_rad)) * other_half_width_m
                )
                effective_other_half_width_m = (
                    abs(math.sin(relative_yaw_rad)) * other_half_length_m
                    + abs(math.cos(relative_yaw_rad)) * other_half_width_m
                )
                lateral_limit = (
                    own_half_width_m
                    + effective_other_half_width_m
                    + (0.4 if pattern == "vehicle.*" else 0.6)
                )
                stopping_m = max(
                    12.0 if pattern == "walker.pedestrian.*" else 7.0,
                    own_half_length_m
                    + effective_other_half_length_m
                    + 2.0
                    + speed_mps * speed_mps / 5.0,
                )
                predicted_lateral_m = lateral_m
                prediction_horizon_s = 0.0
                if pattern == "walker.pedestrian.*" and 0.0 < forward_m:
                    try:
                        other_velocity = other.get_velocity()
                    except RuntimeError:
                        other_velocity = carla.Vector3D()
                    walker_speed_mps = math.hypot(
                        float(other_velocity.x), float(other_velocity.y)
                    )
                    if walker_speed_mps >= 0.2:
                        prediction_horizon_s = min(
                            3.0,
                            max(
                                0.0,
                                forward_m / max(float(speed_mps), 0.5),
                            ),
                        )
                        predicted_dx = (
                            dx + float(other_velocity.x) * prediction_horizon_s
                        )
                        predicted_dy = (
                            dy + float(other_velocity.y) * prediction_horizon_s
                        )
                        predicted_lateral_m = abs(
                            -predicted_dx * float(forward.y)
                            + predicted_dy * float(forward.x)
                        )
                if (
                    0.0 < forward_m <= stopping_m
                    and min(lateral_m, predicted_lateral_m) <= lateral_limit
                ):
                    self.last_yield = {
                        "actor_id": int(other.id),
                        "type_id": str(other.type_id),
                        "forward_m": float(forward_m),
                        "lateral_m": float(lateral_m),
                        "predicted_lateral_m": float(predicted_lateral_m),
                        "prediction_horizon_s": float(prediction_horizon_s),
                        "stopping_m": float(stopping_m),
                        "lateral_limit_m": float(lateral_limit),
                    }
                    return True
        return False

    def tick(self) -> None:
        if self.finished:
            self.actor.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
            )
            return
        transform = self.actor.get_transform()
        location = transform.location
        while (
            self.index < len(self.route) - 1
            and self.route[self.index].distance(location) < self.waypoint_reach_m
        ):
            self.index += 1
        final_distance = self.route[-1].distance(location)
        if self.index == len(self.route) - 1 and final_distance < 2.0:
            self.finished = True
            self.actor.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
            )
            return

        target_index = min(self.index + 1, len(self.route) - 1)
        target = self.route[target_index]
        desired_yaw = math.degrees(
            math.atan2(float(target.y - location.y), float(target.x - location.x))
        )
        yaw_error = wrap_degrees(desired_yaw - float(transform.rotation.yaw))
        velocity = self.actor.get_velocity()
        speed_mps = math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
        steer = max(-0.55, min(0.55, yaw_error / 45.0))
        turn_scale = max(0.35, 1.0 - abs(yaw_error) / 90.0)
        commanded_speed = self.target_speed_mps * turn_scale
        speed_error = commanded_speed - speed_mps
        throttle = max(0.0, min(0.65, 0.28 * speed_error))
        brake = max(0.0, min(0.8, -0.35 * speed_error))
        if self._must_yield(transform, speed_mps):
            throttle, brake = 0.0, 1.0
        self.actor.apply_control(
            carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False,
            )
        )


class CurbsideScenarioRuntime:
    """Own the frozen occluder and optional controlled crossing pedestrian."""

    def __init__(
        self,
        world: carla.World,
        *,
        hazard_present: bool,
        pedestrian_role_name: str,
        pedestrian_start_delay_s: float,
        pedestrian_speed_mps: float,
        pedestrian_endpoint_tolerance_m: float,
    ) -> None:
        if float(pedestrian_start_delay_s) < 0.0:
            raise ValueError("pedestrian start delay must be non-negative")
        if not 1.0 <= float(pedestrian_speed_mps) <= 2.0:
            raise ValueError("pedestrian physical speed must remain within 1-2 m/s")
        if float(pedestrian_endpoint_tolerance_m) <= 0.0:
            raise ValueError("pedestrian endpoint tolerance must be positive")
        self.world = world
        self.hazard_present = bool(hazard_present)
        self.pedestrian_role_name = str(pedestrian_role_name)
        self.start_delay_s = float(pedestrian_start_delay_s)
        self.physical_speed_mps = float(pedestrian_speed_mps)
        self.endpoint_tolerance_m = float(pedestrian_endpoint_tolerance_m)
        self.walker_end = carla.Location(
            x=float(CURBSIDE_WALKER_END[0]),
            y=float(CURBSIDE_WALKER_END[1]),
            z=float(CURBSIDE_WALKER_END[2]),
        )
        self.owned: list[carla.Actor] = []
        self.occluder: Optional[carla.Actor] = None
        self.walker: Optional[carla.Actor] = None
        self.walker_started = False
        self.walker_completed = False
        self.trace: list[dict] = []

    def spawn(self) -> None:
        if self.owned:
            raise RuntimeError("curbside scenario actors were already spawned")
        library = self.world.get_blueprint_library()
        occluder_bp = library.find("vehicle.sprinter.mercedes")
        if occluder_bp.has_attribute("role_name"):
            occluder_bp.set_attribute("role_name", "phase2_curbside_occluder")
        self.occluder = self.world.try_spawn_actor(
            occluder_bp, world_transform(CURBSIDE_OCCLUDER_TRANSFORM)
        )
        if self.occluder is None:
            raise RuntimeError("exact curbside occluder spawn failed")
        self.occluder.set_simulate_physics(False)
        self.owned.append(self.occluder)

        if not self.hazard_present:
            return
        walker_blueprints = sorted(
            library.filter("walker.pedestrian.*"), key=lambda item: item.id
        )
        if not walker_blueprints:
            raise RuntimeError("no pedestrian blueprint is available")
        walker_bp = walker_blueprints[0]
        if walker_bp.has_attribute("role_name"):
            walker_bp.set_attribute("role_name", self.pedestrian_role_name)
        self.walker = self.world.try_spawn_actor(
            walker_bp, world_transform(CURBSIDE_WALKER_START)
        )
        if self.walker is None:
            raise RuntimeError("exact curbside pedestrian spawn failed")
        self.owned.append(self.walker)
        actual_role = str(self.walker.attributes.get("role_name", ""))
        if actual_role != self.pedestrian_role_name:
            raise RuntimeError(
                "controlled pedestrian role_name was not realized: "
                f"expected={self.pedestrian_role_name!r}, observed={actual_role!r}"
            )

    def tick(self, elapsed_s: float, frame_id: int) -> None:
        walker = self.walker
        if walker is not None and not self.walker_started and float(elapsed_s) >= self.start_delay_s:
            direction = self.walker_end - walker.get_location()
            norm = math.sqrt(float(direction.x) ** 2 + float(direction.y) ** 2)
            if norm <= 0.0:
                raise RuntimeError("controlled pedestrian has a zero-length path")
            direction.x /= norm
            direction.y /= norm
            direction.z = 0.0
            walker.apply_control(
                carla.WalkerControl(
                    direction=direction,
                    speed=(
                        self.physical_speed_mps
                        / CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE
                    ),
                    jump=False,
                )
            )
            self.walker_started = True

        walker_location = None
        walker_speed_mps = 0.0
        if walker is not None:
            walker_location = walker.get_location()
            velocity = walker.get_velocity()
            walker_speed_mps = math.sqrt(
                float(velocity.x) ** 2
                + float(velocity.y) ** 2
                + float(velocity.z) ** 2
            )
            if (
                self.walker_started
                and not self.walker_completed
                and walker_location.distance(self.walker_end)
                <= self.endpoint_tolerance_m
            ):
                walker.apply_control(
                    carla.WalkerControl(
                        direction=carla.Vector3D(), speed=0.0, jump=False
                    )
                )
                self.walker_completed = True
        self.trace.append(
            {
                "frame_id": int(frame_id),
                "elapsed_s": float(elapsed_s),
                "hazard_present": self.hazard_present,
                "walker_actor_id": int(walker.id) if walker is not None else -1,
                "walker_x": float(walker_location.x) if walker_location is not None else math.nan,
                "walker_y": float(walker_location.y) if walker_location is not None else math.nan,
                "walker_z": float(walker_location.z) if walker_location is not None else math.nan,
                "walker_speed_mps": float(walker_speed_mps),
                "walker_started": bool(self.walker_started),
                "walker_completed": bool(self.walker_completed),
            }
        )

    def summary(self) -> dict:
        active_speeds = [
            float(row["walker_speed_mps"])
            for row in self.trace
            if bool(row["walker_started"])
            and not bool(row["walker_completed"])
            and float(row["walker_speed_mps"]) > 0.05
        ]
        median_speed = (
            float(statistics.median(active_speeds)) if active_speeds else None
        )
        speed_gate = (
            None
            if not self.hazard_present
            else bool(
                len(active_speeds) >= 5
                and 0.85 * self.physical_speed_mps
                <= float(median_speed)
                <= 1.15 * self.physical_speed_mps
            )
        )
        return {
            "schema": "scenesense.phase2_curbside_realization.v1",
            "geometry_id": CURBSIDE_GEOMETRY_ID,
            "hazard_present": self.hazard_present,
            "occluder_actor_id": int(self.occluder.id) if self.occluder else -1,
            "pedestrian_actor_id": int(self.walker.id) if self.walker else -1,
            "pedestrian_role_name": (
                str(self.walker.attributes.get("role_name", ""))
                if self.walker is not None
                else ""
            ),
            "pedestrian_started": bool(self.walker_started),
            "pedestrian_completed": bool(self.walker_completed),
            "pedestrian_realized_speed_mps_median": median_speed,
            "pedestrian_physical_speed_gate_pass": speed_gate,
            "trace_rows": len(self.trace),
        }

    def destroy(self) -> None:
        for actor in reversed(self.owned):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self.owned.clear()

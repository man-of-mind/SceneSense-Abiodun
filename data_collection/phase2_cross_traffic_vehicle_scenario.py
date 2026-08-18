"""Frozen Phase-2 occluded cross-traffic vehicle geometry.

The positive and matched-benign variants passed automatic gates and were
manually accepted on 2026-08-17.  This module still contains geometry and
validation only; overall Suite A/B collection authorization remains separate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import carla

from data_collection.phase2_curbside_scenario import (
    ROLE_ORDER,
    load_route_progress,
    wrap_degrees,
    world_transform,
)
from data_collection.phase2_signalized_corner_scenario import (
    SIGNALIZED_HELPER_TRANSFORM,
    SIGNALIZED_JUNCTION_ID,
    SIGNALIZED_RECIPIENT_TRANSFORM,
    _route_planner_class,
    frozen_routes as signalized_frozen_routes,
)


CROSS_TRAFFIC_GEOMETRY_ID = "town10hd_opt_occluded_cross_traffic_vehicle_v1"
CROSS_TRAFFIC_RECIPIENT_TRANSFORM = SIGNALIZED_RECIPIENT_TRANSFORM
CROSS_TRAFFIC_HELPER_TRANSFORM = SIGNALIZED_HELPER_TRANSFORM

# The target travels north in road 3 lane +1 and continues through junction
# 532 into road 2 lane +1.  A stopped delivery truck occupies the adjacent
# northbound lane +2.  At the initial pose the truck masks the target from the
# west-arm recipient, but not from the south-facing helper.
CROSS_TRAFFIC_TARGET_TRANSFORM = (102.930000, -9.379000, 0.600000, 90.391000)
CROSS_TRAFFIC_TARGET_DESTINATION = (102.570000, 43.970000, 0.000000)
CROSS_TRAFFIC_OCCLUDER_TRANSFORM = (99.409000, -6.310000, 0.800000, 90.391000)
CROSS_TRAFFIC_TARGET_BLUEPRINT = "vehicle.mini.cooper"
CROSS_TRAFFIC_OCCLUDER_BLUEPRINT = "vehicle.carlacola.actors"

CROSS_TRAFFIC_EXPECTED_START_LANES = {
    "recipient": (21, -1),
    "helper": (2, -1),
    "target": (3, 1),
    "occluder": (3, 2),
}
CROSS_TRAFFIC_EXPECTED_TARGET_END_LANE = (2, 1)

# Static blueprint dimensions measured from the packaged CARLA 0.10 asset.
# Live review recomputes visibility with the realized actor bounding box.
CROSS_TRAFFIC_OCCLUDER_HALF_LENGTH_M = 4.10
CROSS_TRAFFIC_OCCLUDER_HALF_WIDTH_M = 1.55
CROSS_TRAFFIC_CAMERA_FORWARD_M = 1.80
CROSS_TRAFFIC_CAMERA_HORIZONTAL_FOV_DEG = 120.0
CROSS_TRAFFIC_MAX_VISIBILITY_RANGE_M = 80.0
# Geometry-review-only GT brake trigger.  It prevents the authoring smoke from
# intentionally crashing; it is not a policy input or controller result.
CROSS_TRAFFIC_REVIEW_YIELD_TRIGGER_M = 14.0
CROSS_TRAFFIC_TARGET_ROUTE_PATH = (
    Path(__file__).resolve().parent
    / "routes/town10hd_opt_cross_traffic_target_v1.progress.csv"
)
CROSS_TRAFFIC_TARGET_ROUTE_SHA256 = (
    "c9f70a5db774bb462b7c7de9debb3d7771e98169aaa6feb3e353980e2bed4cc5"
)


def planned_target_route(road_map: carla.Map) -> list[carla.Location]:
    """Generate the candidate target route for review and map-drift checks."""

    planner = _route_planner_class()(road_map, 0.5)
    traced = planner.trace_route(
        world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM).location,
        carla.Location(*CROSS_TRAFFIC_TARGET_DESTINATION),
    )
    if len(traced) < 50:
        raise RuntimeError("cross-traffic target route is unexpectedly short")
    waypoints = [waypoint for waypoint, _option in traced]
    start = waypoints[0]
    end = waypoints[-1]
    if (int(start.road_id), int(start.lane_id)) != CROSS_TRAFFIC_EXPECTED_START_LANES[
        "target"
    ]:
        raise RuntimeError(
            "cross-traffic target start lane drifted: "
            f"{(int(start.road_id), int(start.lane_id))}"
        )
    if (int(end.road_id), int(end.lane_id)) != CROSS_TRAFFIC_EXPECTED_TARGET_END_LANE:
        raise RuntimeError(
            "cross-traffic target end lane drifted: "
            f"{(int(end.road_id), int(end.lane_id))}"
        )
    junction_ids = {
        int(waypoint.junction_id)
        for waypoint in waypoints
        if bool(waypoint.is_junction)
    }
    if junction_ids != {SIGNALIZED_JUNCTION_ID}:
        raise RuntimeError(
            "cross-traffic target junction sequence drifted: "
            f"expected={{{SIGNALIZED_JUNCTION_ID}}}, observed={junction_ids}"
        )
    return [waypoint.transform.location for waypoint in waypoints]


def frozen_routes() -> Dict[str, list[carla.Location]]:
    """Load all three accepted routes and fail on target-route byte drift."""

    routes = signalized_frozen_routes()
    observed_hash = hashlib.sha256(CROSS_TRAFFIC_TARGET_ROUTE_PATH.read_bytes()).hexdigest()
    if observed_hash != CROSS_TRAFFIC_TARGET_ROUTE_SHA256:
        raise RuntimeError(
            "frozen cross-traffic target route hash drifted: "
            f"expected={CROSS_TRAFFIC_TARGET_ROUTE_SHA256}, observed={observed_hash}"
        )
    target_route = load_route_progress(CROSS_TRAFFIC_TARGET_ROUTE_PATH)
    if len(target_route) != 107:
        raise RuntimeError(
            "frozen cross-traffic target route row count drifted: "
            f"{len(target_route)}"
        )
    routes["target"] = target_route
    return routes


def _camera_location(transform: carla.Transform) -> tuple[float, float]:
    forward = transform.get_forward_vector()
    return (
        float(transform.location.x)
        + CROSS_TRAFFIC_CAMERA_FORWARD_M * float(forward.x),
        float(transform.location.y)
        + CROSS_TRAFFIC_CAMERA_FORWARD_M * float(forward.y),
    )


def _box_center_xy(
    actor_transform: carla.Transform,
    local_center: carla.Location | None,
) -> tuple[float, float]:
    center = local_center or carla.Location()
    yaw = math.radians(float(actor_transform.rotation.yaw))
    return (
        float(actor_transform.location.x)
        + math.cos(yaw) * float(center.x)
        - math.sin(yaw) * float(center.y),
        float(actor_transform.location.y)
        + math.sin(yaw) * float(center.x)
        + math.cos(yaw) * float(center.y),
    )


def _segment_intersects_oriented_box(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    center_xy: tuple[float, float],
    yaw_deg: float,
    half_length_m: float,
    half_width_m: float,
) -> bool:
    """Return whether a finite 2-D segment crosses an oriented rectangle."""

    if half_length_m <= 0.0 or half_width_m <= 0.0:
        raise ValueError("oriented-box extents must be positive")
    angle = math.radians(-float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)

    def local(point: tuple[float, float]) -> tuple[float, float]:
        dx = float(point[0] - center_xy[0])
        dy = float(point[1] - center_xy[1])
        return cosine * dx - sine * dy, sine * dx + cosine * dy

    start_local = local(start_xy)
    end_local = local(end_xy)
    direction = (
        end_local[0] - start_local[0],
        end_local[1] - start_local[1],
    )
    lower, upper = 0.0, 1.0
    for origin, delta, extent in (
        (start_local[0], direction[0], float(half_length_m)),
        (start_local[1], direction[1], float(half_width_m)),
    ):
        if abs(delta) <= 1e-12:
            if abs(origin) > extent:
                return False
            continue
        first = (-extent - origin) / delta
        second = (extent - origin) / delta
        entry, exit_ = min(first, second), max(first, second)
        lower, upper = max(lower, entry), min(upper, exit_)
        if lower > upper:
            return False
    # A hit at the target endpoint is not an intervening occluder.
    return lower < 0.995 and upper > 0.0


def visibility_state(
    observer_transform: carla.Transform,
    target_transform: carla.Transform,
    occluder_transform: carla.Transform,
    *,
    occluder_half_length_m: float = CROSS_TRAFFIC_OCCLUDER_HALF_LENGTH_M,
    occluder_half_width_m: float = CROSS_TRAFFIC_OCCLUDER_HALF_WIDTH_M,
    occluder_local_center: carla.Location | None = None,
    occluder_local_yaw_deg: float = 0.0,
) -> Dict[str, object]:
    """Compute the registered-target geometric FOV and controlled occlusion."""

    observer_xy = _camera_location(observer_transform)
    target_xy = (
        float(target_transform.location.x),
        float(target_transform.location.y),
    )
    delta_x = target_xy[0] - observer_xy[0]
    delta_y = target_xy[1] - observer_xy[1]
    range_m = math.hypot(delta_x, delta_y)
    bearing_world_deg = math.degrees(math.atan2(delta_y, delta_x))
    bearing_deg = wrap_degrees(
        bearing_world_deg - float(observer_transform.rotation.yaw)
    )
    in_fov = bool(
        range_m <= CROSS_TRAFFIC_MAX_VISIBILITY_RANGE_M
        and abs(bearing_deg) <= 0.5 * CROSS_TRAFFIC_CAMERA_HORIZONTAL_FOV_DEG
    )
    box_center = _box_center_xy(occluder_transform, occluder_local_center)
    occluded = bool(
        in_fov
        and _segment_intersects_oriented_box(
            observer_xy,
            target_xy,
            box_center,
            float(occluder_transform.rotation.yaw) + float(occluder_local_yaw_deg),
            float(occluder_half_length_m),
            float(occluder_half_width_m),
        )
    )
    return {
        "range_m": float(range_m),
        "relative_bearing_deg": float(bearing_deg),
        "in_fov": in_fov,
        "occluded_by_controlled_truck": occluded,
        "geometrically_visible": bool(in_fov and not occluded),
    }


def cross_traffic_geometry_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    occluder_transform: carla.Transform,
    target_transform: carla.Transform,
    routes: Mapping[str, Sequence[carla.Location]],
    *,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Fail unless lanes, route conflict, clearance, and initial LOS are sound."""

    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("cross-traffic geometry requires helper and recipient")
    if set(routes) != {"recipient", "helper", "target"}:
        raise ValueError("cross-traffic routes require recipient, helper, and target")
    requested = dict(transforms)
    requested.update(target=target_transform, occluder=occluder_transform)
    roles: Dict[str, Dict[str, object]] = {}
    for role, transform in requested.items():
        waypoint = road_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError(f"{role} does not project to a driving lane")
        observed = (int(waypoint.road_id), int(waypoint.lane_id))
        heading_error = abs(
            wrap_degrees(
                float(transform.rotation.yaw)
                - float(waypoint.transform.rotation.yaw)
            )
        )
        roles[role] = {
            "road_id": observed[0],
            "section_id": int(waypoint.section_id),
            "lane_id": observed[1],
            "is_junction": bool(waypoint.is_junction),
            "heading_error_deg": float(heading_error),
        }
        if (
            observed != CROSS_TRAFFIC_EXPECTED_START_LANES[role]
            or heading_error > maximum_heading_error_deg
        ):
            raise RuntimeError(
                f"{role} cross-traffic lane drift: expected="
                f"{CROSS_TRAFFIC_EXPECTED_START_LANES[role]}, observed={observed}, "
                f"heading_error_deg={heading_error:.3f}"
            )

    def closest_route_pair(
        first: Sequence[carla.Location], second: Sequence[carla.Location]
    ) -> tuple[float, carla.Location, carla.Location]:
        distance, first_location, second_location = min(
            (a.distance(b), a, b) for a in first for b in second
        )
        return float(distance), first_location, second_location

    conflict_distance, recipient_conflict, target_conflict = closest_route_pair(
        routes["recipient"], routes["target"]
    )
    if conflict_distance > 0.75:
        raise RuntimeError(
            "recipient and cross-traffic target paths do not physically conflict: "
            f"minimum_separation_m={conflict_distance:.3f}"
        )
    helper_clearance, _helper_point, _target_point = closest_route_pair(
        routes["helper"], routes["target"]
    )
    if helper_clearance < 3.0:
        raise RuntimeError(
            "helper and target routes lack opposing-lane clearance: "
            f"minimum_separation_m={helper_clearance:.3f}"
        )

    target_to_occluder_center_m = target_transform.location.distance(
        occluder_transform.location
    )
    if target_to_occluder_center_m < 4.25:
        raise RuntimeError(
            "target and adjacent-lane occluder start too close: "
            f"center_distance_m={target_to_occluder_center_m:.3f}"
        )
    initial_visibility = {
        role: visibility_state(
            transforms[role], target_transform, occluder_transform
        )
        for role in ROLE_ORDER
    }
    if not bool(initial_visibility["recipient"]["occluded_by_controlled_truck"]):
        raise RuntimeError("truck does not initially occlude target from recipient")
    if not bool(initial_visibility["helper"]["geometrically_visible"]):
        raise RuntimeError("target is not initially visible to helper")

    return {
        "pass": True,
        "basis": (
            "exact_legal_lanes_route_conflict_opposing_helper_clearance_"
            "and_initial_differential_visibility"
        ),
        "geometry_review_status": "user_accepted_positive_and_benign_20260817",
        "junction_id": SIGNALIZED_JUNCTION_ID,
        "roles": roles,
        "recipient_target_route_min_separation_m": conflict_distance,
        "registered_conflict_point": {
            "x": 0.5 * (float(recipient_conflict.x) + float(target_conflict.x)),
            "y": 0.5 * (float(recipient_conflict.y) + float(target_conflict.y)),
            "z": 0.0,
        },
        "helper_target_route_min_separation_m": helper_clearance,
        "target_occluder_start_center_distance_m": float(
            target_to_occluder_center_m
        ),
        "initial_visibility": initial_visibility,
    }

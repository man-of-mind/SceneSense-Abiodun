"""Frozen Phase-2 parked-van midblock pedestrian geometry.

The positive and matched-benign variants were manually accepted after a
physics-settled occluder review.  Collection remains separately gated by the
Phase-2 suite contract.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Mapping

import carla

from data_collection.phase2_curbside_scenario import (
    ROLE_ORDER,
    load_route_progress,
    wrap_degrees,
    world_transform,
)
from data_collection.phase2_signalized_corner_scenario import _route_planner_class


MIDBLOCK_GEOMETRY_ID = "town10hd_opt_midblock_curbside_van_v1"
MIDBLOCK_ROAD_ID = 12
MIDBLOCK_RECIPIENT_TRANSFORM = (-25.000000, 69.717000, 0.600000, 0.073000)
MIDBLOCK_RECIPIENT_DESTINATION = (25.000000, 69.781000, 0.000000)
MIDBLOCK_HELPER_TRANSFORM = (25.000000, 66.281000, 0.600000, -179.927000)
MIDBLOCK_HELPER_DESTINATION = (-25.000000, 66.217000, 0.000000)

# The van is tangent to the north edge of the recipient lane: close enough to
# hide the pedestrian but outside the travel path.  The pedestrian begins on
# the north sidewalk behind the van from the recipient's perspective.
MIDBLOCK_OCCLUDER_TRANSFORM = (-6.000000, 72.500000, 0.800000, 0.073000)
MIDBLOCK_WALKER_START = (-2.000000, 74.000000, 1.000000, -90.000000)
MIDBLOCK_WALKER_END = (-2.000000, 64.600000, 1.000000)

MIDBLOCK_EXPECTED_LANES = {
    "recipient": (MIDBLOCK_ROAD_ID, 1),
    "helper": (MIDBLOCK_ROAD_ID, -1),
}
MIDBLOCK_OCCLUDER_MIN_CENTER_OFFSET_M = 2.45
MIDBLOCK_OCCLUDER_MAX_CENTER_OFFSET_M = 3.10
MIDBLOCK_ROUTE_PATHS = {
    "recipient": Path(__file__).resolve().parent
    / "routes/town10hd_opt_midblock_van_recipient_v1.progress.csv",
    "helper": Path(__file__).resolve().parent
    / "routes/town10hd_opt_midblock_van_helper_v1.progress.csv",
}
MIDBLOCK_ROUTE_SHA256 = {
    "recipient": "f1d6e525dd1120a064e0414ab777faab31c7049df1554da6c4944f3b39ae3318",
    "helper": "c5fc19f01dc22bf3cdd5cc42bb5dc958d7088bfefac908977aacbe6e79e1cc81",
}


def planned_routes(road_map: carla.Map) -> Dict[str, list[carla.Location]]:
    """Regenerate the routes for an explicit map-drift audit."""

    planner = _route_planner_class()(road_map, 1.5)
    endpoints = {
        "recipient": (
            world_transform(MIDBLOCK_RECIPIENT_TRANSFORM).location,
            carla.Location(*MIDBLOCK_RECIPIENT_DESTINATION),
        ),
        "helper": (
            world_transform(MIDBLOCK_HELPER_TRANSFORM).location,
            carla.Location(*MIDBLOCK_HELPER_DESTINATION),
        ),
    }
    routes: Dict[str, list[carla.Location]] = {}
    for role in ROLE_ORDER:
        start, destination = endpoints[role]
        traced = planner.trace_route(start, destination)
        if len(traced) < 20:
            raise RuntimeError(f"{role} midblock route is unexpectedly short")
        waypoints = [waypoint for waypoint, _option in traced]
        observed_lanes = {
            (int(waypoint.road_id), int(waypoint.lane_id))
            for waypoint in waypoints
        }
        if observed_lanes != {MIDBLOCK_EXPECTED_LANES[role]}:
            raise RuntimeError(
                f"{role} midblock route lane drifted: {observed_lanes}"
            )
        if any(bool(waypoint.is_junction) for waypoint in waypoints):
            raise RuntimeError(f"{role} midblock route unexpectedly enters a junction")
        routes[role] = [waypoint.transform.location for waypoint in waypoints]
    return routes


def frozen_routes() -> Dict[str, list[carla.Location]]:
    """Load the accepted routes and fail on any byte-level drift."""

    routes: Dict[str, list[carla.Location]] = {}
    for role in ROLE_ORDER:
        path = MIDBLOCK_ROUTE_PATHS[role]
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = MIDBLOCK_ROUTE_SHA256[role]
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{role} frozen midblock route hash drifted: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
        route = load_route_progress(path)
        if len(route) != 33:
            raise RuntimeError(
                f"{role} frozen midblock route row count drifted: {len(route)}"
            )
        routes[role] = route
    return routes


def midblock_lane_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    occluder_transform: carla.Transform,
    *,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Validate legal opposite lanes and the curbside occluder offset."""

    if maximum_heading_error_deg <= 0.0:
        raise ValueError("maximum_heading_error_deg must be positive")
    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("midblock lane contract requires helper and recipient")
    roles: Dict[str, Dict[str, object]] = {}
    for role in ROLE_ORDER:
        transform = transforms[role]
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
            "lane_heading_deg": float(waypoint.transform.rotation.yaw),
            "commanded_heading_deg": float(transform.rotation.yaw),
            "heading_error_deg": float(heading_error),
            "is_junction": bool(waypoint.is_junction),
        }
        if (
            observed != MIDBLOCK_EXPECTED_LANES[role]
            or bool(waypoint.is_junction)
            or heading_error > maximum_heading_error_deg
        ):
            raise RuntimeError(f"{role} midblock lane contract failed: {roles[role]}")

    recipient_waypoint = road_map.get_waypoint(
        occluder_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if recipient_waypoint is None:
        raise RuntimeError("midblock occluder has no adjacent driving lane")
    lane_center = recipient_waypoint.transform.location
    center_offset = math.hypot(
        float(occluder_transform.location.x - lane_center.x),
        float(occluder_transform.location.y - lane_center.y),
    )
    if (
        (int(recipient_waypoint.road_id), int(recipient_waypoint.lane_id))
        != MIDBLOCK_EXPECTED_LANES["recipient"]
        or not MIDBLOCK_OCCLUDER_MIN_CENTER_OFFSET_M
        <= center_offset
        <= MIDBLOCK_OCCLUDER_MAX_CENTER_OFFSET_M
    ):
        raise RuntimeError(
            "midblock occluder curb offset drifted: "
            f"road/lane={(recipient_waypoint.road_id, recipient_waypoint.lane_id)}, "
            f"center_offset_m={center_offset:.3f}"
        )
    separation = transforms["recipient"].location.distance(
        transforms["helper"].location
    )
    if separation < 40.0:
        raise RuntimeError("midblock role starts are insufficiently separated")
    return {
        "pass": True,
        "basis": "nonjunction_opposite_signed_lanes_and_bounded_curb_offset",
        "road_id": MIDBLOCK_ROAD_ID,
        "maximum_heading_error_deg": float(maximum_heading_error_deg),
        "start_separation_m": float(separation),
        "occluder_center_offset_from_lane_m": float(center_offset),
        "roles": roles,
    }


def line_of_sight_bearings_deg() -> Dict[str, float]:
    """Return initial target bearings under the production 120-degree FOV."""

    target = world_transform(MIDBLOCK_WALKER_START).location
    result = {}
    for role, values in {
        "recipient": MIDBLOCK_RECIPIENT_TRANSFORM,
        "helper": MIDBLOCK_HELPER_TRANSFORM,
    }.items():
        transform = world_transform(values)
        delta_x = float(target.x - transform.location.x)
        delta_y = float(target.y - transform.location.y)
        bearing = math.degrees(math.atan2(delta_y, delta_x))
        result[role] = wrap_degrees(bearing - float(transform.rotation.yaw))
    return result

"""Candidate Phase-2 signalized-corner geometry primitives.

This module is intentionally limited to geometry authoring and visual review.
The candidate must not enter a corpus until the positive and matched-benign
reviews are accepted and its generated routes are frozen to versioned files.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Mapping

import carla

from data_collection.phase2_curbside_scenario import (
    ROLE_ORDER,
    load_route_progress,
    wrap_degrees,
    world_transform,
)


SIGNALIZED_GEOMETRY_ID = "town10hd_opt_signalized_corner_van_crosswalk_v1"
SIGNALIZED_JUNCTION_ID = 532

# The recipient approaches the west arm in the inner eastbound lane and turns
# into the outer southbound lane.  The helper approaches from the north and
# remains in the inner southbound lane.  Their paths therefore do not merge.
SIGNALIZED_RECIPIENT_TRANSFORM = (60.000000, 24.850000, 0.600000, 0.159198)
SIGNALIZED_RECIPIENT_DESTINATION = (109.945000, -12.016000, 0.000000)
SIGNALIZED_HELPER_TRANSFORM = (106.020000, 50.870000, 0.600000, -89.609268)
SIGNALIZED_HELPER_DESTINATION = (106.445000, -12.005000, 0.000000)

# A stopped delivery van occupies the distinct outer approach lane and ends
# just before the crosswalk.  It is the controlled line-of-sight blocker, not
# ambient NPC
# traffic.  The pedestrian walks north across the recipient's two approach
# lanes and stops on the raised refuge island, matching the physical curb.
SIGNALIZED_OCCLUDER_TRANSFORM = (79.500000, 28.405000, 0.600000, 0.159198)
SIGNALIZED_WALKER_START = (85.700000, 32.000000, 1.000000, -90.000000)
SIGNALIZED_WALKER_END = (85.700000, 22.500000, 1.000000)

SIGNALIZED_EXPECTED_START_LANES = {
    "recipient": (21, -1),
    "helper": (2, -1),
    "occluder": (21, -2),
}
SIGNALIZED_EXPECTED_END_LANES = {
    "recipient": (0, -2),
    "helper": (0, -1),
}

SIGNALIZED_ROUTE_PATHS = {
    "recipient": Path(__file__).resolve().parent
    / "routes/town10hd_opt_signalized_corner_recipient_v1.progress.csv",
    "helper": Path(__file__).resolve().parent
    / "routes/town10hd_opt_signalized_corner_helper_v1.progress.csv",
}
SIGNALIZED_ROUTE_SHA256 = {
    "recipient": "4144eaabf3e6c2bcdbfef2cd5ba639e0d6459adc739c45a3110c196486c73911",
    "helper": "af7352eb95a0e0deffde35d960f365f8954b8be36a6bfc3c937101a658197af6",
}

# Metadata positions identify the two controlled approach signals.  The review
# tool forces only these signals green so its deterministic direct controllers
# do not visibly run a red.  The pedestrian is therefore an unexpected
# crossing hazard, not a claim that the pedestrian signal grants right of way.
SIGNALIZED_CONTROLLED_TRAFFIC_LIGHTS = {
    "recipient": (115.4498291015625, 35.044944763183594, 0.222725972533226),
    "helper": (114.44964599609375, 21.201000213623047, 0.2542538344860077),
}


def _standard_agents_root() -> Path:
    # .../PythonAPI/neu_collab/abiodun/data_collection/file.py -> PythonAPI/carla
    return Path(__file__).resolve().parents[3] / "carla"


def _route_planner_class():
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    except ModuleNotFoundError:
        agents_root = _standard_agents_root()
        if not agents_root.is_dir():
            raise RuntimeError(
                f"packaged CARLA navigation helpers are missing: {agents_root}"
            )
        sys.path.insert(0, str(agents_root))
        from agents.navigation.global_route_planner import GlobalRoutePlanner
    return GlobalRoutePlanner


def planned_routes(road_map: carla.Map) -> Dict[str, list[carla.Location]]:
    """Regenerate the two routes for an explicit map-drift audit."""

    planner = _route_planner_class()(road_map, 2.0)
    endpoints = {
        "recipient": (
            world_transform(SIGNALIZED_RECIPIENT_TRANSFORM).location,
            carla.Location(*SIGNALIZED_RECIPIENT_DESTINATION),
        ),
        "helper": (
            world_transform(SIGNALIZED_HELPER_TRANSFORM).location,
            carla.Location(*SIGNALIZED_HELPER_DESTINATION),
        ),
    }
    routes: Dict[str, list[carla.Location]] = {}
    for role in ROLE_ORDER:
        start, destination = endpoints[role]
        traced = planner.trace_route(start, destination)
        if len(traced) < 10:
            raise RuntimeError(f"{role} signalized route is unexpectedly short")
        waypoints = [waypoint for waypoint, _option in traced]
        junction_ids = {
            int(waypoint.junction_id)
            for waypoint in waypoints
            if bool(waypoint.is_junction)
        }
        if SIGNALIZED_JUNCTION_ID not in junction_ids:
            raise RuntimeError(
                f"{role} route misses junction {SIGNALIZED_JUNCTION_ID}: {junction_ids}"
            )
        end = waypoints[-1]
        expected_end = SIGNALIZED_EXPECTED_END_LANES[role]
        if (int(end.road_id), int(end.lane_id)) != expected_end:
            raise RuntimeError(
                f"{role} route endpoint drifted: expected={expected_end}, "
                f"observed={(int(end.road_id), int(end.lane_id))}"
            )
        routes[role] = [waypoint.transform.location for waypoint in waypoints]
    return routes


def frozen_routes() -> Dict[str, list[carla.Location]]:
    """Load the visually accepted routes and fail on any byte-level drift."""

    routes: Dict[str, list[carla.Location]] = {}
    for role in ROLE_ORDER:
        path = SIGNALIZED_ROUTE_PATHS[role]
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = SIGNALIZED_ROUTE_SHA256[role]
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{role} frozen signalized route hash drifted: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
        route = load_route_progress(path)
        if len(route) < 10:
            raise RuntimeError(f"{role} frozen signalized route is unexpectedly short")
        routes[role] = route
    return routes


def signalized_lane_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    occluder_transform: carla.Transform,
    *,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Fail unless the three actors realize the reviewed, distinct lane set."""

    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("signalized lane contract requires helper and recipient")
    if maximum_heading_error_deg <= 0.0:
        raise ValueError("maximum lane-heading error must be positive")
    requested = dict(transforms)
    requested["occluder"] = occluder_transform
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
        expected = SIGNALIZED_EXPECTED_START_LANES[role]
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
        }
        if observed != expected or heading_error > maximum_heading_error_deg:
            raise RuntimeError(
                f"{role} signalized lane drift: expected={expected}, "
                f"observed={observed}, heading_error={heading_error:.3f}deg"
            )
    if roles["recipient"]["lane_id"] == roles["occluder"]["lane_id"]:
        raise RuntimeError("recipient and occluder unexpectedly share a lane")
    start_separation = transforms["recipient"].location.distance(
        transforms["helper"].location
    )
    if start_separation < 20.0:
        raise RuntimeError("signalized role starts are insufficiently separated")
    return {
        "pass": True,
        "basis": "exact_start_road_lane_heading_and_distinct_exit_lanes",
        "junction_id": SIGNALIZED_JUNCTION_ID,
        "maximum_heading_error_deg": float(maximum_heading_error_deg),
        "start_separation_m": float(start_separation),
        "roles": roles,
    }


def controlled_traffic_lights(world: carla.World) -> Dict[str, carla.TrafficLight]:
    """Resolve the two reviewed approach signals by immutable map position."""

    actors = list(world.get_actors().filter("traffic.traffic_light*"))
    if not actors:
        raise RuntimeError("Town10HD_Opt has no traffic-light actors")
    result: Dict[str, carla.TrafficLight] = {}
    for role, values in SIGNALIZED_CONTROLLED_TRAFFIC_LIGHTS.items():
        location = carla.Location(*values)
        actor = min(actors, key=lambda item: item.get_location().distance(location))
        distance = actor.get_location().distance(location)
        if distance > 0.25:
            raise RuntimeError(
                f"{role} traffic-light metadata drifted by {distance:.3f}m"
            )
        result[role] = actor
    if len({int(actor.id) for actor in result.values()}) != len(result):
        raise RuntimeError("controlled approach traffic lights did not resolve uniquely")
    return result


def line_of_sight_bearings_deg() -> Dict[str, float]:
    """Return initial target bearings used as a cheap geometry sanity check."""

    target = world_transform(SIGNALIZED_WALKER_START).location
    result = {}
    for role, values in {
        "recipient": SIGNALIZED_RECIPIENT_TRANSFORM,
        "helper": SIGNALIZED_HELPER_TRANSFORM,
    }.items():
        transform = world_transform(values)
        delta_x = float(target.x - transform.location.x)
        delta_y = float(target.y - transform.location.y)
        bearing = math.degrees(math.atan2(delta_y, delta_x))
        result[role] = wrap_degrees(bearing - float(transform.rotation.yaw))
    return result

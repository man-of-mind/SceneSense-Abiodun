"""Frozen Phase-2 parked-vehicle pullout geometry.

The positive and matched-benign variants passed automatic gates and were
manually accepted on 2026-08-18.  This module still contains geometry and
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
from data_collection.phase2_cross_traffic_vehicle_scenario import visibility_state
from data_collection.phase2_midblock_van_scenario import (
    MIDBLOCK_HELPER_TRANSFORM,
    MIDBLOCK_RECIPIENT_TRANSFORM,
    MIDBLOCK_ROAD_ID,
    frozen_routes as midblock_frozen_routes,
)


PULLOUT_GEOMETRY_ID = "town10hd_opt_parked_vehicle_pullout_v1"
PULLOUT_RECIPIENT_TRANSFORM = MIDBLOCK_RECIPIENT_TRANSFORM
PULLOUT_HELPER_TRANSFORM = MIDBLOCK_HELPER_TRANSFORM
PULLOUT_OCCLUDER_TRANSFORM = (-6.000000, 72.500000, 0.800000, 0.073000)
PULLOUT_TARGET_TRANSFORM = (0.500000, 72.400000, 0.800000, 0.073000)
PULLOUT_TARGET_BLUEPRINT = "vehicle.mini.cooper"
PULLOUT_OCCLUDER_BLUEPRINT = "vehicle.sprinter.mercedes"
PULLOUT_TARGET_START_DELAY_S = 4.0
PULLOUT_TARGET_SPEED_MPS = 3.0
PULLOUT_REVIEW_YIELD_TRIGGER_M = 14.0
PULLOUT_MERGE_POINT = (5.500000, 69.750000, 0.0)

PULLOUT_EXPECTED_EGO_LANES = {
    "recipient": (MIDBLOCK_ROAD_ID, 1),
    "helper": (MIDBLOCK_ROAD_ID, -1),
}
PULLOUT_OCCLUDER_HALF_LENGTH_M = 3.05
PULLOUT_OCCLUDER_HALF_WIDTH_M = 1.25
PULLOUT_MIN_TARGET_OCCLUDER_CENTER_DISTANCE_M = 5.0
PULLOUT_MIN_HELPER_ROUTE_CLEARANCE_M = 3.0
PULLOUT_TARGET_ROUTE_PATH = (
    Path(__file__).resolve().parent
    / "routes/town10hd_opt_parked_vehicle_pullout_target_v1.progress.csv"
)
PULLOUT_TARGET_ROUTE_SHA256 = (
    "7e101885d4dd52fefb13f8c2b942e0ef33738955bf4651235c13cc5ed2948175"
)

# Accepted explicit path retained here as a readable geometry audit. Runtime
# loads the byte-hashed CSV below so no silent manoeuvre drift is possible.
PULLOUT_TARGET_ROUTE_VALUES = (
    (0.500000, 72.400000, 0.0),
    (1.500000, 72.250000, 0.0),
    (2.500000, 71.800000, 0.0),
    (3.500000, 71.100000, 0.0),
    (4.500000, 70.350000, 0.0),
    (5.500000, 69.750000, 0.0),
    (6.500000, 69.757000, 0.0),
    (7.500000, 69.758000, 0.0),
    (8.500000, 69.759000, 0.0),
    (9.500000, 69.761000, 0.0),
    (10.500000, 69.762000, 0.0),
    (11.500000, 69.763000, 0.0),
    (12.500000, 69.765000, 0.0),
    (13.500000, 69.766000, 0.0),
    (14.500000, 69.767000, 0.0),
    (15.500000, 69.769000, 0.0),
    (16.500000, 69.770000, 0.0),
    (17.500000, 69.771000, 0.0),
    (18.500000, 69.772000, 0.0),
    (19.500000, 69.774000, 0.0),
    (20.500000, 69.775000, 0.0),
    (21.500000, 69.776000, 0.0),
    (22.500000, 69.777000, 0.0),
)


def frozen_routes() -> Dict[str, list[carla.Location]]:
    """Load the three accepted routes and fail on target-route byte drift."""

    routes = midblock_frozen_routes()
    observed_hash = hashlib.sha256(PULLOUT_TARGET_ROUTE_PATH.read_bytes()).hexdigest()
    if observed_hash != PULLOUT_TARGET_ROUTE_SHA256:
        raise RuntimeError(
            "frozen pullout target route hash drifted: "
            f"expected={PULLOUT_TARGET_ROUTE_SHA256}, observed={observed_hash}"
        )
    target_route = load_route_progress(PULLOUT_TARGET_ROUTE_PATH)
    if len(target_route) != len(PULLOUT_TARGET_ROUTE_VALUES):
        raise RuntimeError(
            "frozen pullout target route row count drifted: "
            f"{len(target_route)}"
        )
    for index, (observed, expected) in enumerate(
        zip(target_route, PULLOUT_TARGET_ROUTE_VALUES)
    ):
        drift_m = math.hypot(
            float(observed.x) - float(expected[0]),
            float(observed.y) - float(expected[1]),
        )
        if drift_m > 1e-5:
            raise RuntimeError(
                f"frozen pullout target route point {index} drifted by {drift_m}m"
            )
    routes["target"] = target_route
    return routes


def pullout_geometry_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    occluder_transform: carla.Transform,
    target_transform: carla.Transform,
    routes: Mapping[str, Sequence[carla.Location]],
    *,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Validate legal egos, curb poses, merge conflict, clearance, and LOS."""

    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("pullout geometry requires helper and recipient")
    if set(routes) != {"recipient", "helper", "target"}:
        raise ValueError("pullout routes require helper, recipient, and target")
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
            "lane_id": observed[1],
            "is_junction": bool(waypoint.is_junction),
            "heading_error_deg": float(heading_error),
        }
        if (
            observed != PULLOUT_EXPECTED_EGO_LANES[role]
            or bool(waypoint.is_junction)
            or heading_error > maximum_heading_error_deg
        ):
            raise RuntimeError(f"{role} pullout lane contract failed: {roles[role]}")

    curb_roles = {}
    for role, transform in (
        ("occluder", occluder_transform),
        ("target", target_transform),
    ):
        waypoint = road_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError(f"{role} has no adjacent driving lane")
        center_offset_m = waypoint.transform.location.distance(transform.location)
        observed = (int(waypoint.road_id), int(waypoint.lane_id))
        heading_error = abs(
            wrap_degrees(
                float(transform.rotation.yaw)
                - float(waypoint.transform.rotation.yaw)
            )
        )
        curb_roles[role] = {
            "adjacent_road_id": observed[0],
            "adjacent_lane_id": observed[1],
            "lane_center_offset_m": float(center_offset_m),
            "heading_error_deg": float(heading_error),
        }
        if (
            observed != PULLOUT_EXPECTED_EGO_LANES["recipient"]
            or not 2.4 <= center_offset_m <= 3.1
            or heading_error > maximum_heading_error_deg
        ):
            raise RuntimeError(f"{role} curb pose contract failed: {curb_roles[role]}")

    target_occluder_center_distance_m = target_transform.location.distance(
        occluder_transform.location
    )
    if target_occluder_center_distance_m < PULLOUT_MIN_TARGET_OCCLUDER_CENTER_DISTANCE_M:
        raise RuntimeError("pullout target overlaps or crowds the controlled occluder")
    target_route = routes["target"]
    target_route_start_error_m = math.hypot(
        float(target_route[0].x - target_transform.location.x),
        float(target_route[0].y - target_transform.location.y),
    )
    if target_route_start_error_m > 0.05:
        raise RuntimeError("pullout route does not begin at the target pose")
    merge = carla.Location(*PULLOUT_MERGE_POINT)
    target_merge_distance_m = min(point.distance(merge) for point in target_route)
    recipient_merge_distance_m = min(
        point.distance(merge) for point in routes["recipient"]
    )
    if target_merge_distance_m > 0.05 or recipient_merge_distance_m > 0.75:
        raise RuntimeError(
            "pullout route does not merge into the recipient path: "
            f"target={target_merge_distance_m:.3f}, "
            f"recipient={recipient_merge_distance_m:.3f}"
        )
    helper_target_clearance_m = min(
        helper.distance(target)
        for helper in routes["helper"]
        for target in target_route
    )
    if helper_target_clearance_m < PULLOUT_MIN_HELPER_ROUTE_CLEARANCE_M:
        raise RuntimeError(
            "helper and pullout target routes lack opposing-lane clearance: "
            f"{helper_target_clearance_m:.3f}m"
        )

    initial_visibility = {
        role: visibility_state(
            transforms[role],
            target_transform,
            occluder_transform,
            occluder_half_length_m=PULLOUT_OCCLUDER_HALF_LENGTH_M,
            occluder_half_width_m=PULLOUT_OCCLUDER_HALF_WIDTH_M,
        )
        for role in ROLE_ORDER
    }
    if not bool(initial_visibility["recipient"]["occluded_by_controlled_truck"]):
        raise RuntimeError("pullout target is not initially occluded from recipient")
    if not bool(initial_visibility["helper"]["geometrically_visible"]):
        raise RuntimeError("pullout target is not initially visible to helper")

    return {
        "pass": True,
        "basis": (
            "legal_opposing_egos_bounded_curb_poses_registered_merge_"
            "helper_clearance_and_initial_differential_visibility"
        ),
        "road_id": MIDBLOCK_ROAD_ID,
        "roles": roles,
        "curb_roles": curb_roles,
        "registered_conflict_point": {
            "x": PULLOUT_MERGE_POINT[0],
            "y": PULLOUT_MERGE_POINT[1],
            "z": PULLOUT_MERGE_POINT[2],
        },
        "target_occluder_start_center_distance_m": float(
            target_occluder_center_distance_m
        ),
        "recipient_route_merge_distance_m": float(recipient_merge_distance_m),
        "helper_target_route_min_separation_m": float(helper_target_clearance_m),
        "initial_visibility": initial_visibility,
    }

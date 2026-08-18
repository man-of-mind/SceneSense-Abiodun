"""Frozen Phase-2 queue-reveal stopped-lead vehicle geometry.

The recipient initially queues behind a Sprinter in its legal lane. The
opposing helper can see a stopped lead vehicle beyond the Sprinter. After a
bounded delay, the Sprinter pulls into the curb bay and reveals the stopped
lead to the recipient. The positive and target-absent benign variants passed
automatic and human review on 2026-08-18. Overall corpus authorization remains
separate.
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
)
from data_collection.phase2_cross_traffic_vehicle_scenario import visibility_state
from data_collection.phase2_midblock_van_scenario import (
    MIDBLOCK_HELPER_TRANSFORM,
    MIDBLOCK_RECIPIENT_TRANSFORM,
    MIDBLOCK_ROAD_ID,
    frozen_routes as midblock_frozen_routes,
)


QUEUE_REVEAL_GEOMETRY_ID = "town10hd_opt_queue_reveal_lead_vehicle_v1"
QUEUE_REVEAL_RECIPIENT_TRANSFORM = MIDBLOCK_RECIPIENT_TRANSFORM
QUEUE_REVEAL_HELPER_TRANSFORM = MIDBLOCK_HELPER_TRANSFORM
QUEUE_REVEAL_OCCLUDER_TRANSFORM = (-10.000000, 69.730000, 0.600000, 0.073000)
QUEUE_REVEAL_TARGET_TRANSFORM = (14.500000, 69.768000, 0.600000, 0.073000)
QUEUE_REVEAL_OCCLUDER_BLUEPRINT = "vehicle.sprinter.mercedes"
QUEUE_REVEAL_TARGET_BLUEPRINT = "vehicle.lincoln.mkz"
QUEUE_REVEAL_OCCLUDER_START_DELAY_S = 5.0
QUEUE_REVEAL_OCCLUDER_SPEED_MPS = 2.5
QUEUE_REVEAL_REVIEW_YIELD_TRIGGER_M = 14.0
QUEUE_REVEAL_CONFLICT_POINT = (14.500000, 69.768000, 0.0)

QUEUE_REVEAL_EXPECTED_EGO_LANES = {
    "recipient": (MIDBLOCK_ROAD_ID, 1),
    "helper": (MIDBLOCK_ROAD_ID, -1),
}
QUEUE_REVEAL_EXPECTED_QUEUE_LANE = (MIDBLOCK_ROAD_ID, 1)
QUEUE_REVEAL_OCCLUDER_HALF_LENGTH_M = 3.05
QUEUE_REVEAL_OCCLUDER_HALF_WIDTH_M = 1.25
QUEUE_REVEAL_MIN_INITIAL_CENTER_GAP_M = 10.0
QUEUE_REVEAL_MIN_FINAL_TARGET_CLEARANCE_M = 5.0
QUEUE_REVEAL_MIN_HELPER_ROUTE_CLEARANCE_M = 3.0
QUEUE_REVEAL_OCCLUDER_ROUTE_PATH = (
    Path(__file__).resolve().parent
    / "routes/town10hd_opt_queue_reveal_occluder_v1.progress.csv"
)
QUEUE_REVEAL_OCCLUDER_ROUTE_SHA256 = (
    "57371f1b7004b7bb5a709b44705167a327f19779e0684a70bf30380aae2ec870"
)

# The queue member moves forward and right into the curb bay. Its route is
# explicit so the review cannot silently choose an unsafe lane or manoeuvre.
QUEUE_REVEAL_OCCLUDER_ROUTE_VALUES = (
    (-10.000000, 69.730000, 0.0),
    (-9.000000, 70.150000, 0.0),
    (-8.000000, 70.700000, 0.0),
    (-7.000000, 71.300000, 0.0),
    (-6.000000, 71.900000, 0.0),
    (-5.000000, 72.300000, 0.0),
    (-4.000000, 72.500000, 0.0),
    (-3.000000, 72.500000, 0.0),
    (-2.000000, 72.500000, 0.0),
    (-1.000000, 72.500000, 0.0),
    (0.000000, 72.500000, 0.0),
    (1.000000, 72.500000, 0.0),
    (2.000000, 72.500000, 0.0),
)


def frozen_routes() -> Dict[str, list[carla.Location]]:
    """Load accepted routes and fail on queue-member route byte drift."""

    routes = midblock_frozen_routes()
    observed_hash = hashlib.sha256(
        QUEUE_REVEAL_OCCLUDER_ROUTE_PATH.read_bytes()
    ).hexdigest()
    if observed_hash != QUEUE_REVEAL_OCCLUDER_ROUTE_SHA256:
        raise RuntimeError(
            "frozen queue-reveal occluder route hash drifted: "
            f"expected={QUEUE_REVEAL_OCCLUDER_ROUTE_SHA256}, "
            f"observed={observed_hash}"
        )
    occluder_route = load_route_progress(QUEUE_REVEAL_OCCLUDER_ROUTE_PATH)
    if len(occluder_route) != len(QUEUE_REVEAL_OCCLUDER_ROUTE_VALUES):
        raise RuntimeError(
            "frozen queue-reveal occluder route row count drifted: "
            f"{len(occluder_route)}"
        )
    for index, (observed, expected) in enumerate(
        zip(occluder_route, QUEUE_REVEAL_OCCLUDER_ROUTE_VALUES)
    ):
        drift_m = math.hypot(
            float(observed.x) - float(expected[0]),
            float(observed.y) - float(expected[1]),
        )
        if drift_m > 1e-5:
            raise RuntimeError(
                "frozen queue-reveal occluder route point "
                f"{index} drifted by {drift_m}m"
            )
    routes["occluder"] = occluder_route
    return routes


def queue_reveal_geometry_contract(
    road_map: carla.Map,
    transforms: Mapping[str, carla.Transform],
    occluder_transform: carla.Transform,
    target_transform: carla.Transform,
    routes: Mapping[str, Sequence[carla.Location]],
    *,
    maximum_heading_error_deg: float = 5.0,
) -> Dict[str, object]:
    """Validate lane legality, route clearance, conflict, and initial LOS."""

    if set(transforms) != set(ROLE_ORDER):
        raise ValueError("queue-reveal geometry requires helper and recipient")
    if set(routes) != {"recipient", "helper", "occluder"}:
        raise ValueError("queue-reveal routes require recipient, helper, and occluder")

    roles: Dict[str, Dict[str, object]] = {}
    requested = dict(transforms)
    requested.update(occluder=occluder_transform, target=target_transform)
    expected = dict(QUEUE_REVEAL_EXPECTED_EGO_LANES)
    expected.update(
        occluder=QUEUE_REVEAL_EXPECTED_QUEUE_LANE,
        target=QUEUE_REVEAL_EXPECTED_QUEUE_LANE,
    )
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
            "lane_id": observed[1],
            "is_junction": bool(waypoint.is_junction),
            "heading_error_deg": float(heading_error),
        }
        if (
            observed != expected[role]
            or bool(waypoint.is_junction)
            or heading_error > maximum_heading_error_deg
        ):
            raise RuntimeError(f"{role} queue-reveal lane contract failed: {roles[role]}")

    initial_gap_m = occluder_transform.location.distance(target_transform.location)
    if initial_gap_m < QUEUE_REVEAL_MIN_INITIAL_CENTER_GAP_M:
        raise RuntimeError("queue member is too close to the stopped lead")
    occluder_route = routes["occluder"]
    route_start_error_m = math.hypot(
        float(occluder_route[0].x - occluder_transform.location.x),
        float(occluder_route[0].y - occluder_transform.location.y),
    )
    if route_start_error_m > 0.05:
        raise RuntimeError("queue-member route does not begin at its spawn pose")
    final_target_clearance_m = math.hypot(
        float(occluder_route[-1].x - target_transform.location.x),
        float(occluder_route[-1].y - target_transform.location.y),
    )
    if final_target_clearance_m < QUEUE_REVEAL_MIN_FINAL_TARGET_CLEARANCE_M:
        raise RuntimeError("queue-member curb endpoint crowds the stopped lead")

    conflict = carla.Location(*QUEUE_REVEAL_CONFLICT_POINT)
    recipient_conflict_distance_m = min(
        math.hypot(
            float(point.x - conflict.x),
            float(point.y - conflict.y),
        )
        for point in routes["recipient"]
    )
    if recipient_conflict_distance_m > 0.75:
        raise RuntimeError(
            "stopped lead is not on the recipient path: "
            f"{recipient_conflict_distance_m:.3f}m"
        )
    helper_target_clearance_m = min(
        math.hypot(
            float(point.x - target_transform.location.x),
            float(point.y - target_transform.location.y),
        )
        for point in routes["helper"]
    )
    if helper_target_clearance_m < QUEUE_REVEAL_MIN_HELPER_ROUTE_CLEARANCE_M:
        raise RuntimeError(
            "helper lacks opposing-lane clearance from the stopped lead: "
            f"{helper_target_clearance_m:.3f}m"
        )

    initial_visibility = {
        role: visibility_state(
            transforms[role],
            target_transform,
            occluder_transform,
            occluder_half_length_m=QUEUE_REVEAL_OCCLUDER_HALF_LENGTH_M,
            occluder_half_width_m=QUEUE_REVEAL_OCCLUDER_HALF_WIDTH_M,
        )
        for role in ROLE_ORDER
    }
    if not bool(initial_visibility["recipient"]["occluded_by_controlled_truck"]):
        raise RuntimeError("stopped lead is not initially queue-occluded from recipient")
    if not bool(initial_visibility["helper"]["geometrically_visible"]):
        raise RuntimeError("stopped lead is not initially visible to helper")

    curb_endpoint = occluder_route[-1]
    curb_waypoint = road_map.get_waypoint(
        curb_endpoint,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if curb_waypoint is None:
        raise RuntimeError("queue-member curb endpoint has no adjacent driving lane")
    curb_offset_m = curb_waypoint.transform.location.distance(curb_endpoint)
    if not 2.4 <= curb_offset_m <= 3.1:
        raise RuntimeError(
            f"queue-member endpoint is not in the bounded curb bay: {curb_offset_m:.3f}m"
        )

    return {
        "pass": True,
        "basis": (
            "legal_opposing_egos_same_lane_queue_pair_bounded_curb_exit_"
            "registered_stopped_lead_and_initial_differential_visibility"
        ),
        "road_id": MIDBLOCK_ROAD_ID,
        "roles": roles,
        "registered_conflict_point": {
            "x": QUEUE_REVEAL_CONFLICT_POINT[0],
            "y": QUEUE_REVEAL_CONFLICT_POINT[1],
            "z": QUEUE_REVEAL_CONFLICT_POINT[2],
        },
        "initial_queue_center_gap_m": float(initial_gap_m),
        "queue_route_start_error_m": float(route_start_error_m),
        "queue_curb_endpoint_offset_m": float(curb_offset_m),
        "queue_endpoint_target_clearance_m": float(final_target_clearance_m),
        "recipient_route_conflict_distance_m": float(recipient_conflict_distance_m),
        "helper_target_route_min_separation_m": float(helper_target_clearance_m),
        "initial_visibility": initial_visibility,
    }

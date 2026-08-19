"""Frozen paired-route contract for Phase-2 naturalistic Suite B.

The two accepted advisor loops share a long prefix.  Starting every short
trajectory at row zero would therefore create false route diversity.  This
module freezes six geometry-only start strata on each source loop and places
the helper ahead on the same native lane.  It contains no perception outcome,
collector, policy, or collection-authority logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import carla

from data_collection.phase2_curbside_scenario import load_route_progress, wrap_degrees


NATURALISTIC_PAIR_CONTRACT_ID = "town10hd_opt_same_lane_helper_ahead_v1"
MIN_INITIAL_CENTER_SEPARATION_M = 10.0
MAX_INITIAL_CENTER_SEPARATION_M = 20.0
MAX_NATIVE_HEADING_ERROR_DEG = 3.0
MAX_SOURCE_ROUTE_STEP_M = 6.5
MAX_SOURCE_ROUTE_SEAM_M = 8.0


@dataclass(frozen=True)
class AnchorSpec:
    anchor_id: str
    recipient_start_index: int
    helper_start_index: int


@dataclass(frozen=True)
class RouteFamilySpec:
    family_id: str
    source_path: Path
    source_sha256: str
    source_row_count: int
    anchors: tuple[AnchorSpec, ...]


_ROUTE_ROOT = Path(__file__).resolve().parent / "routes"
ROUTE_FAMILIES: Mapping[str, RouteFamilySpec] = {
    "signalized_demo_region": RouteFamilySpec(
        family_id="signalized_demo_region",
        source_path=_ROUTE_ROOT
        / "town10hd_opt_advisor_demo_loop_v2.progress.csv",
        source_sha256=(
            "3315d27a1c9c7500e7df426c70beb0a1622b838cc98debfb0f2e3de54c024f46"
        ),
        source_row_count=130,
        anchors=(
            AnchorSpec("a0", 0, 4),
            AnchorSpec("a1", 15, 19),
            AnchorSpec("a2", 38, 42),
            AnchorSpec("a3", 59, 63),
            AnchorSpec("a4", 82, 86),
            AnchorSpec("a5", 103, 107),
        ),
    ),
    "safe_perimeter": RouteFamilySpec(
        family_id="safe_perimeter",
        source_path=_ROUTE_ROOT
        / "town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv",
        source_sha256=(
            "f3dc2f4d8c59905801fdfad2df7a19f2b427459d4039ed3a8cdec3535e818ce1"
        ),
        source_row_count=85,
        anchors=(
            AnchorSpec("a0", 0, 4),
            AnchorSpec("a1", 14, 18),
            AnchorSpec("a2", 27, 30),
            AnchorSpec("a3", 38, 42),
            AnchorSpec("a4", 56, 60),
            AnchorSpec("a5", 78, 82),
        ),
    ),
}


def anchor_spec(family_id: str, anchor_id: str) -> AnchorSpec:
    family = ROUTE_FAMILIES.get(str(family_id))
    if family is None:
        raise ValueError(f"unknown naturalistic route family: {family_id}")
    matches = [item for item in family.anchors if item.anchor_id == str(anchor_id)]
    if len(matches) != 1:
        raise ValueError(
            f"unknown naturalistic anchor {anchor_id!r} for {family_id!r}"
        )
    return matches[0]


def load_source_route(family_id: str) -> list[carla.Location]:
    """Load one source loop and fail on byte, row, step, or seam drift."""

    family = ROUTE_FAMILIES.get(str(family_id))
    if family is None:
        raise ValueError(f"unknown naturalistic route family: {family_id}")
    observed_hash = hashlib.sha256(family.source_path.read_bytes()).hexdigest()
    if observed_hash != family.source_sha256:
        raise RuntimeError(
            f"naturalistic source route hash drifted for {family_id}: "
            f"expected={family.source_sha256}, observed={observed_hash}"
        )
    route = load_route_progress(family.source_path)
    if len(route) != family.source_row_count:
        raise RuntimeError(
            f"naturalistic source route row count drifted for {family_id}: "
            f"{len(route)}"
        )
    steps = [left.distance(right) for left, right in zip(route, route[1:])]
    if max(steps) > MAX_SOURCE_ROUTE_STEP_M:
        raise RuntimeError(
            f"naturalistic source route has an unsafe step for {family_id}: "
            f"{max(steps):.3f}m"
        )
    seam_m = route[-1].distance(route[0])
    if seam_m > MAX_SOURCE_ROUTE_SEAM_M:
        raise RuntimeError(
            f"naturalistic source route is not a bounded loop for {family_id}: "
            f"{seam_m:.3f}m"
        )
    return route


def _rotated_route(
    route: Sequence[carla.Location], start_index: int
) -> list[carla.Location]:
    if not 0 <= int(start_index) < len(route):
        raise ValueError(f"route start index is outside 0..{len(route) - 1}")
    return [
        carla.Location(x=float(point.x), y=float(point.y), z=float(point.z))
        for point in list(route[int(start_index) :]) + list(route[: int(start_index) + 1])
    ]


def point_to_polyline_distance_m(
    x_m: float,
    y_m: float,
    route: Sequence[carla.Location],
) -> float:
    """Return 2-D distance to the continuous piecewise-linear route."""

    if not route:
        raise ValueError("cannot measure cross-track distance to an empty route")
    if len(route) == 1:
        return math.hypot(
            float(x_m) - float(route[0].x),
            float(y_m) - float(route[0].y),
        )
    distances = []
    for start, end in zip(route, route[1:]):
        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        denominator = dx * dx + dy * dy
        fraction = (
            0.0
            if denominator <= 1e-12
            else max(
                0.0,
                min(
                    1.0,
                    (
                        (float(x_m) - float(start.x)) * dx
                        + (float(y_m) - float(start.y)) * dy
                    )
                    / denominator,
                ),
            )
        )
        nearest_x = float(start.x) + fraction * dx
        nearest_y = float(start.y) + fraction * dy
        distances.append(
            math.hypot(float(x_m) - nearest_x, float(y_m) - nearest_y)
        )
    return float(min(distances))


def _spawn_transform(
    road_map: carla.Map, location: carla.Location
) -> tuple[carla.Transform, carla.Waypoint]:
    waypoint = road_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError("naturalistic start does not project to a driving lane")
    native = waypoint.transform
    return (
        carla.Transform(
            carla.Location(
                x=float(native.location.x),
                y=float(native.location.y),
                z=float(native.location.z) + 0.60,
            ),
            carla.Rotation(
                pitch=float(native.rotation.pitch),
                yaw=float(native.rotation.yaw),
                roll=float(native.rotation.roll),
            ),
        ),
        waypoint,
    )


def resolve_pair(
    road_map: carla.Map,
    family_id: str,
    anchor_id: str,
) -> tuple[Dict[str, carla.Transform], Dict[str, list[carla.Location]], dict]:
    """Resolve a deterministic same-lane pair and its fail-closed contract."""

    family = ROUTE_FAMILIES.get(str(family_id))
    if family is None:
        raise ValueError(f"unknown naturalistic route family: {family_id}")
    anchor = anchor_spec(family_id, anchor_id)
    route = load_source_route(family_id)
    if not (
        0 <= anchor.recipient_start_index < anchor.helper_start_index < len(route)
    ):
        raise RuntimeError(f"invalid naturalistic anchor ordering: {anchor}")

    recipient_transform, recipient_waypoint = _spawn_transform(
        road_map, route[anchor.recipient_start_index]
    )
    helper_transform, helper_waypoint = _spawn_transform(
        road_map, route[anchor.helper_start_index]
    )
    roles = {
        "recipient": (recipient_transform, recipient_waypoint),
        "helper": (helper_transform, helper_waypoint),
    }
    role_contract = {}
    for role, (transform, waypoint) in roles.items():
        heading_error = abs(
            wrap_degrees(
                float(transform.rotation.yaw)
                - float(waypoint.transform.rotation.yaw)
            )
        )
        role_contract[role] = {
            "road_id": int(waypoint.road_id),
            "section_id": int(waypoint.section_id),
            "lane_id": int(waypoint.lane_id),
            "is_junction": bool(waypoint.is_junction),
            "native_heading_deg": float(waypoint.transform.rotation.yaw),
            "heading_error_deg": float(heading_error),
        }
        if bool(waypoint.is_junction) or heading_error > MAX_NATIVE_HEADING_ERROR_DEG:
            raise RuntimeError(
                f"naturalistic {role} start violates native-lane contract: "
                f"{role_contract[role]}"
            )
    recipient_lane = role_contract["recipient"]
    helper_lane = role_contract["helper"]
    if (
        recipient_lane["road_id"],
        recipient_lane["section_id"],
        recipient_lane["lane_id"],
    ) != (
        helper_lane["road_id"],
        helper_lane["section_id"],
        helper_lane["lane_id"],
    ):
        raise RuntimeError("naturalistic helper is not ahead on the recipient lane")

    along_route_separation_m = sum(
        route[index].distance(route[index + 1])
        for index in range(
            anchor.recipient_start_index, anchor.helper_start_index
        )
    )
    center_separation_m = recipient_transform.location.distance(
        helper_transform.location
    )
    if not (
        MIN_INITIAL_CENTER_SEPARATION_M
        <= along_route_separation_m
        <= MAX_INITIAL_CENTER_SEPARATION_M
    ):
        raise RuntimeError(
            "naturalistic helper separation is outside the reviewed envelope: "
            f"{along_route_separation_m:.3f}m"
        )
    heading_difference_deg = abs(
        wrap_degrees(
            float(helper_transform.rotation.yaw)
            - float(recipient_transform.rotation.yaw)
        )
    )
    if heading_difference_deg > MAX_NATIVE_HEADING_ERROR_DEG:
        raise RuntimeError(
            f"naturalistic pair headings diverge by {heading_difference_deg:.3f}deg"
        )

    cumulative_m = sum(
        left.distance(right) for left, right in zip(route, route[1:])
    )
    anchor_progress_m = sum(
        route[index].distance(route[index + 1])
        for index in range(anchor.recipient_start_index)
    )
    contract = {
        "pass": True,
        "basis": "byte_frozen_loop_nonjunction_same_native_lane_helper_ahead",
        "collection_authorized": False,
        "visual_review_status": "accepted_both_route_families_20260818",
        "pair_contract_id": NATURALISTIC_PAIR_CONTRACT_ID,
        "family_id": family.family_id,
        "anchor_id": anchor.anchor_id,
        "source_route_path": str(family.source_path),
        "source_route_sha256": family.source_sha256,
        "source_route_row_count": family.source_row_count,
        "recipient_start_index": anchor.recipient_start_index,
        "helper_start_index": anchor.helper_start_index,
        "anchor_progress_m": float(anchor_progress_m),
        "anchor_progress_fraction": float(anchor_progress_m / cumulative_m),
        "along_route_separation_m": float(along_route_separation_m),
        "center_separation_m": float(center_separation_m),
        "heading_difference_deg": float(heading_difference_deg),
        "roles": role_contract,
    }
    transforms = {
        role: transform for role, (transform, _waypoint) in roles.items()
    }
    routes = {
        "recipient": _rotated_route(route, anchor.recipient_start_index),
        "helper": _rotated_route(route, anchor.helper_start_index),
    }
    return transforms, routes, contract

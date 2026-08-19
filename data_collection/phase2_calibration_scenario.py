"""Runtime adapters for the frozen Phase-2 Suite-A/B geometry contracts.

This module is deliberately headless.  It resolves the exact transforms and
byte-frozen routes that were accepted in ``review_phase2_pair_geometry.py``,
places the two collector-owned egos, and owns only controlled scenario actors.
It neither launches CARLA nor selects corpus rows.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import carla

from data_collection.phase2_curbside_scenario import (
    CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE,
    CURBSIDE_HELPER_TRANSFORM,
    CURBSIDE_OCCLUDER_TRANSFORM,
    CURBSIDE_RECIPIENT_TRANSFORM,
    CURBSIDE_WALKER_END,
    CURBSIDE_WALKER_START,
    DirectRouteController,
    legal_opposing_lane_contract,
    load_route_progress,
    world_transform,
    wrap_degrees,
)
from data_collection.phase2_signalized_corner_scenario import (
    SIGNALIZED_HELPER_TRANSFORM,
    SIGNALIZED_OCCLUDER_TRANSFORM,
    SIGNALIZED_RECIPIENT_TRANSFORM,
    SIGNALIZED_WALKER_END,
    SIGNALIZED_WALKER_START,
    controlled_traffic_lights,
    frozen_routes as signalized_frozen_routes,
    signalized_lane_contract,
)
from data_collection.phase2_midblock_van_scenario import (
    MIDBLOCK_HELPER_TRANSFORM,
    MIDBLOCK_OCCLUDER_TRANSFORM,
    MIDBLOCK_RECIPIENT_TRANSFORM,
    MIDBLOCK_WALKER_END,
    MIDBLOCK_WALKER_START,
    frozen_routes as midblock_frozen_routes,
    midblock_lane_contract,
)
from data_collection.phase2_cross_traffic_vehicle_scenario import (
    CROSS_TRAFFIC_HELPER_TRANSFORM,
    CROSS_TRAFFIC_OCCLUDER_BLUEPRINT,
    CROSS_TRAFFIC_OCCLUDER_TRANSFORM,
    CROSS_TRAFFIC_RECIPIENT_TRANSFORM,
    CROSS_TRAFFIC_REVIEW_YIELD_TRIGGER_M,
    CROSS_TRAFFIC_TARGET_BLUEPRINT,
    CROSS_TRAFFIC_TARGET_TRANSFORM,
    cross_traffic_geometry_contract,
    frozen_routes as cross_traffic_frozen_routes,
)
from data_collection.phase2_parked_vehicle_pullout_scenario import (
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
from data_collection.phase2_naturalistic_pair_scenario import resolve_pair


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = ("helper", "recipient")
DESIGNED_GEOMETRIES = {
    "curbside_bus_occluded_pedestrian",
    "signalized_corner_occluded_pedestrian",
    "parked_van_midblock_occluded_pedestrian",
    "occluded_cross_traffic_vehicle",
    "parked_vehicle_pullout",
    "queue_reveal_lead_vehicle",
}
NATURALISTIC_ROUTES = {
    "town10hd_opt_curbside_corridor",
    "town10hd_opt_signalized_demo_region",
    "town10hd_opt_safe_perimeter",
}
OCCLUDER_SETTLE_MAX_TICKS = 30
OCCLUDER_SETTLE_STABLE_TICKS = 3
OCCLUDER_SETTLE_MAX_XY_DRIFT_M = 0.35
OCCLUDER_SETTLE_MAX_YAW_DRIFT_DEG = 3.0
EGO_PLACEMENT_MAX_TICKS = 5
CURBSIDE_ROUTE_PATHS = {
    "helper": REPO_ROOT
    / "data_collection/routes/town10hd_opt_curbside_helper_v1.progress.csv",
    "recipient": REPO_ROOT
    / "data_collection/routes/town10hd_opt_curbside_recipient_v1.progress.csv",
}
MIDBLOCK_AMBIENT_MOTION_ROUTE_PATHS = {
    "helper": REPO_ROOT
    / "data_collection/routes/town10hd_opt_midblock_van_helper_ambient_v1.progress.csv",
    "recipient": REPO_ROOT
    / "data_collection/routes/town10hd_opt_midblock_van_recipient_ambient_v1.progress.csv",
}
MIDBLOCK_AMBIENT_MOTION_ROUTE_SHA256 = {
    "helper": "41507e1048b800e08f1579389c4ad092b8142387588f9d6b7a69f2130928f82f",
    "recipient": "4b6517eb498d8c3c5775bd86ce88194a79486f3c1006ec87cbc134ae3bd7cf77",
}


DIRECT_ROUTE_YIELD_FIELDS = (
    ("actor_id", int),
    ("type_id", str),
    ("forward_m", float),
    ("lateral_m", float),
    ("predicted_lateral_m", float),
    ("prediction_horizon_s", float),
    ("stopping_m", float),
    ("lateral_limit_m", float),
)


def direct_route_yield_trace_fields(
    role: str,
    event: Optional[Mapping[str, object]],
) -> dict[str, object]:
    """Return a stable, CSV-safe snapshot of one controller yield decision.

    ``DirectRouteController.last_yield`` is deliberately diagnostic state, not
    scenario truth.  Keeping that distinction in the field prefix prevents a
    controller-side stop from being confused with the separate GT safety
    override used by some vehicle-hazard geometry reviews.
    """

    prefix = f"{role}_direct_route_yield"
    fields: dict[str, object] = {f"{prefix}_active": int(event is not None)}
    for source_name, converter in DIRECT_ROUTE_YIELD_FIELDS:
        output_name = "actor_type" if source_name == "type_id" else source_name
        value = None if event is None else event.get(source_name)
        fields[f"{prefix}_{output_name}"] = (
            "" if value is None else converter(value)
        )
    return fields


@dataclass(frozen=True)
class ResolvedScenario:
    geometry_or_route_id: str
    layout: str
    scenario_role: str
    hazard_present: bool
    transforms: Mapping[str, carla.Transform]
    routes: Mapping[str, Sequence[carla.Location]]
    lane_contract: Mapping[str, object]
    naturalistic_family: Optional[str] = None
    naturalistic_anchor_id: Optional[str] = None

    @property
    def ambient_route_paths(self) -> tuple[Path, ...]:
        if self.layout in {"curbside_opposite", "curbside_natural"}:
            return tuple(CURBSIDE_ROUTE_PATHS[role] for role in ROLE_NAMES)
        if self.layout in {"signalized_corner", "cross_traffic_vehicle"}:
            return (
                REPO_ROOT
                / "data_collection/routes/town10hd_opt_signalized_corner_helper_v1.progress.csv",
                REPO_ROOT
                / "data_collection/routes/town10hd_opt_signalized_corner_recipient_v1.progress.csv",
            )
        if self.layout in {
            "midblock_van",
            "parked_vehicle_pullout",
            "queue_reveal_vehicle",
        }:
            return (
                REPO_ROOT
                / "data_collection/routes/town10hd_opt_midblock_van_helper_v1.progress.csv",
                REPO_ROOT
                / "data_collection/routes/town10hd_opt_midblock_van_recipient_v1.progress.csv",
            )
        family_to_path = {
            "signalized_demo_region": REPO_ROOT
            / "data_collection/routes/town10hd_opt_advisor_demo_loop_v2.progress.csv",
            "safe_perimeter": REPO_ROOT
            / "data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv",
        }
        return (family_to_path[str(self.naturalistic_family)],)

    @property
    def ambient_motion_route_paths(self) -> tuple[Path, ...]:
        """Long legal routes for motion, distinct from local spawn support.

        The reviewed midblock ego paths are open 48 m segments. They remain
        the local spawn corridor, while ambient vehicles follow byte-frozen
        CARLA-lane continuations long enough that no 12 s audit reaches an
        artificial endpoint wrap/U-turn.
        """

        if self.layout not in {
            "midblock_van",
            "parked_vehicle_pullout",
            "queue_reveal_vehicle",
        }:
            return self.ambient_route_paths
        for role, path in MIDBLOCK_AMBIENT_MOTION_ROUTE_PATHS.items():
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != MIDBLOCK_AMBIENT_MOTION_ROUTE_SHA256[role]:
                raise RuntimeError(
                    f"{role} ambient motion route hash drifted: {observed}"
                )
        return tuple(MIDBLOCK_AMBIENT_MOTION_ROUTE_PATHS[role] for role in ROLE_NAMES)

    @property
    def protected_locations(self) -> tuple[carla.Location, ...]:
        """Controlled occupancy excluded from ambient spawning in both arms.

        The registered pedestrian crossing is a segment, not only its initial
        actor origin.  Likewise, vehicle-hazard designs have a registered
        conflict point that must not be pre-populated by an ambient vehicle.
        Protecting those causal geometry elements prevents a technically legal
        route-derived spawn from corrupting the designed event before motion
        begins.
        """

        values = {
            "curbside_opposite": (
                CURBSIDE_OCCLUDER_TRANSFORM,
                CURBSIDE_WALKER_START,
                CURBSIDE_WALKER_END,
            ),
            "signalized_corner": (
                SIGNALIZED_OCCLUDER_TRANSFORM,
                SIGNALIZED_WALKER_START,
                SIGNALIZED_WALKER_END,
            ),
            "midblock_van": (
                MIDBLOCK_OCCLUDER_TRANSFORM,
                MIDBLOCK_WALKER_START,
                MIDBLOCK_WALKER_END,
            ),
            "cross_traffic_vehicle": (
                CROSS_TRAFFIC_OCCLUDER_TRANSFORM,
                CROSS_TRAFFIC_TARGET_TRANSFORM,
            ),
            "parked_vehicle_pullout": (
                PULLOUT_OCCLUDER_TRANSFORM,
                PULLOUT_TARGET_TRANSFORM,
            ),
            "queue_reveal_vehicle": (
                QUEUE_REVEAL_OCCLUDER_TRANSFORM,
                QUEUE_REVEAL_TARGET_TRANSFORM,
            ),
        }.get(self.layout, ())
        controlled = [transform.location for transform in self.transforms.values()]
        for item in values:
            if len(item) == 3:
                controlled.append(
                    carla.Location(
                        x=float(item[0]), y=float(item[1]), z=float(item[2])
                    )
                )
            else:
                controlled.append(world_transform(item).location)
        pedestrian_segment = {
            "curbside_opposite": (CURBSIDE_WALKER_START, CURBSIDE_WALKER_END),
            "signalized_corner": (SIGNALIZED_WALKER_START, SIGNALIZED_WALKER_END),
            "midblock_van": (MIDBLOCK_WALKER_START, MIDBLOCK_WALKER_END),
        }.get(self.layout)
        if pedestrian_segment is not None:
            start, end = pedestrian_segment
            # Protect the crossing itself, not merely its sidewalk endpoints.
            # The intermediate samples cover both travel lanes and prevent an
            # ambient queue from being initialized across the walker path.
            for fraction in (0.25, 0.5, 0.75):
                controlled.append(
                    carla.Location(
                        x=float(start[0]) + fraction * (float(end[0]) - float(start[0])),
                        y=float(start[1]) + fraction * (float(end[1]) - float(start[1])),
                        z=float(start[2]) + fraction * (float(end[2]) - float(start[2])),
                    )
                )
        if self.layout == "queue_reveal_vehicle":
            # The queue member is intentionally released after a five-second
            # hold and then traverses this frozen curb-exit path.  Protecting
            # only its spawn pose allowed an ambient vehicle to be initialized
            # inside the future swept corridor, creating a deterministic crash
            # when the reviewed motion began.
            controlled.extend(self.routes["occluder"])
        conflict = self.lane_contract.get("registered_conflict_point")
        if conflict is not None:
            controlled.append(
                carla.Location(
                    x=float(conflict["x"]),
                    y=float(conflict["y"]),
                    z=float(conflict["z"]),
                )
            )
        return tuple(controlled)


def _curbside_routes() -> Dict[str, list[carla.Location]]:
    return {
        role: load_route_progress(path) for role, path in CURBSIDE_ROUTE_PATHS.items()
    }


def resolve_scenario(
    road_map: carla.Map,
    *,
    geometry_or_route_id: str,
    scenario_role: str,
    route_start_anchor_id: Optional[str] = None,
) -> ResolvedScenario:
    """Resolve a frozen manifest row to its reviewed geometry and routes."""

    identity = str(geometry_or_route_id)
    role = str(scenario_role)
    if identity not in DESIGNED_GEOMETRIES | NATURALISTIC_ROUTES:
        raise ValueError(f"unsupported Phase-2 geometry/route: {identity}")
    naturalistic = identity in NATURALISTIC_ROUTES
    if naturalistic != (role == "naturalistic_operation"):
        raise ValueError("naturalistic routes and scenario_role disagree")
    if not naturalistic and role not in {
        "controlled_positive_occlusion",
        "matched_benign_negative",
    }:
        raise ValueError("designed geometry has an invalid scenario role")
    hazard_present = role == "controlled_positive_occlusion"

    if identity in {
        "curbside_bus_occluded_pedestrian",
        "town10hd_opt_curbside_corridor",
    }:
        transforms = {
            "helper": world_transform(CURBSIDE_HELPER_TRANSFORM),
            "recipient": world_transform(CURBSIDE_RECIPIENT_TRANSFORM),
        }
        routes = _curbside_routes()
        lane_contract = legal_opposing_lane_contract(road_map, transforms)
        layout = (
            "curbside_opposite"
            if identity == "curbside_bus_occluded_pedestrian"
            else "curbside_natural"
        )
        return ResolvedScenario(
            identity, layout, role, hazard_present, transforms, routes, lane_contract
        )

    if identity == "signalized_corner_occluded_pedestrian":
        transforms = {
            "helper": world_transform(SIGNALIZED_HELPER_TRANSFORM),
            "recipient": world_transform(SIGNALIZED_RECIPIENT_TRANSFORM),
        }
        routes = signalized_frozen_routes()
        lane_contract = signalized_lane_contract(
            road_map, transforms, world_transform(SIGNALIZED_OCCLUDER_TRANSFORM)
        )
        return ResolvedScenario(
            identity, "signalized_corner", role, hazard_present, transforms, routes, lane_contract
        )

    if identity == "parked_van_midblock_occluded_pedestrian":
        transforms = {
            "helper": world_transform(MIDBLOCK_HELPER_TRANSFORM),
            "recipient": world_transform(MIDBLOCK_RECIPIENT_TRANSFORM),
        }
        routes = midblock_frozen_routes()
        lane_contract = midblock_lane_contract(
            road_map, transforms, world_transform(MIDBLOCK_OCCLUDER_TRANSFORM)
        )
        return ResolvedScenario(
            identity, "midblock_van", role, hazard_present, transforms, routes, lane_contract
        )

    if identity == "occluded_cross_traffic_vehicle":
        transforms = {
            "helper": world_transform(CROSS_TRAFFIC_HELPER_TRANSFORM),
            "recipient": world_transform(CROSS_TRAFFIC_RECIPIENT_TRANSFORM),
        }
        routes = cross_traffic_frozen_routes()
        lane_contract = cross_traffic_geometry_contract(
            road_map,
            transforms,
            world_transform(CROSS_TRAFFIC_OCCLUDER_TRANSFORM),
            world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM),
            routes,
        )
        return ResolvedScenario(
            identity, "cross_traffic_vehicle", role, hazard_present, transforms, routes, lane_contract
        )

    if identity == "parked_vehicle_pullout":
        transforms = {
            "helper": world_transform(PULLOUT_HELPER_TRANSFORM),
            "recipient": world_transform(PULLOUT_RECIPIENT_TRANSFORM),
        }
        routes = pullout_frozen_routes()
        lane_contract = pullout_geometry_contract(
            road_map,
            transforms,
            world_transform(PULLOUT_OCCLUDER_TRANSFORM),
            world_transform(PULLOUT_TARGET_TRANSFORM),
            routes,
        )
        return ResolvedScenario(
            identity, "parked_vehicle_pullout", role, hazard_present, transforms, routes, lane_contract
        )

    if identity == "queue_reveal_lead_vehicle":
        transforms = {
            "helper": world_transform(QUEUE_REVEAL_HELPER_TRANSFORM),
            "recipient": world_transform(QUEUE_REVEAL_RECIPIENT_TRANSFORM),
        }
        routes = queue_reveal_frozen_routes()
        lane_contract = queue_reveal_geometry_contract(
            road_map,
            transforms,
            world_transform(QUEUE_REVEAL_OCCLUDER_TRANSFORM),
            world_transform(QUEUE_REVEAL_TARGET_TRANSFORM),
            routes,
        )
        return ResolvedScenario(
            identity, "queue_reveal_vehicle", role, hazard_present, transforms, routes, lane_contract
        )

    family = {
        "town10hd_opt_signalized_demo_region": "signalized_demo_region",
        "town10hd_opt_safe_perimeter": "safe_perimeter",
    }.get(identity)
    if family is None:
        raise AssertionError("curbside naturalistic route should have returned above")
    anchor_id = str(route_start_anchor_id or "")
    transforms, routes, lane_contract = resolve_pair(road_map, family, anchor_id)
    return ResolvedScenario(
        identity,
        "naturalistic_pair",
        role,
        False,
        transforms,
        routes,
        lane_contract,
        naturalistic_family=family,
        naturalistic_anchor_id=anchor_id,
    )


def _speed(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def _settle_parked_actor(
    world: carla.World, actor: carla.Actor, commanded: carla.Transform
) -> dict:
    """Apply the same gravity-settle/freeze contract used by visual review."""

    actor.set_simulate_physics(True)
    actor.apply_control(
        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
    )
    previous_z: Optional[float] = None
    stable_ticks = 0
    settled: Optional[carla.Transform] = None
    vertical_speed_mps = float("inf")
    for _unused in range(OCCLUDER_SETTLE_MAX_TICKS):
        world.tick(2.0)
        settled = actor.get_transform()
        vertical_speed_mps = abs(float(actor.get_velocity().z))
        current_z = float(settled.location.z)
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
    if settled is None or stable_ticks < OCCLUDER_SETTLE_STABLE_TICKS:
        raise RuntimeError("parked controlled actor did not settle within 30 ticks")
    xy_drift_m = math.hypot(
        float(settled.location.x - commanded.location.x),
        float(settled.location.y - commanded.location.y),
    )
    yaw_drift_deg = abs(
        wrap_degrees(float(settled.rotation.yaw) - float(commanded.rotation.yaw))
    )
    if xy_drift_m > OCCLUDER_SETTLE_MAX_XY_DRIFT_M:
        raise RuntimeError(f"parked controlled actor XY drifted {xy_drift_m:.3f} m")
    if yaw_drift_deg > OCCLUDER_SETTLE_MAX_YAW_DRIFT_DEG:
        raise RuntimeError(f"parked controlled actor yaw drifted {yaw_drift_deg:.3f} deg")
    actor.set_simulate_physics(False)
    world.tick(2.0)
    if actor.get_transform().location.distance(settled.location) > 0.01:
        raise RuntimeError("parked controlled actor moved while freezing physics")
    return {
        "pass": True,
        "stable_ticks": stable_ticks,
        "vertical_speed_mps": vertical_speed_mps,
        "xy_drift_m": xy_drift_m,
        "yaw_drift_deg": yaw_drift_deg,
        "settled_z_m": float(settled.location.z),
    }


class CalibrationScenarioRuntime:
    """Own controlled actors and exact route motion for one audit trajectory."""

    def __init__(
        self,
        world: carla.World,
        scenario: ResolvedScenario,
        egos: Mapping[str, carla.Actor],
        *,
        tm_port: int,
        helper_speed_mps: float = 4.5,
        recipient_speed_mps: float = 5.0,
        pedestrian_speed_mps: float = 1.3,
        pedestrian_start_delay_s: float = 3.0,
    ) -> None:
        if set(egos) != set(ROLE_NAMES):
            raise ValueError("scenario runtime requires helper and recipient egos")
        if not 1.0 <= float(pedestrian_speed_mps) <= 2.0:
            raise ValueError("controlled pedestrian speed must remain within 1-2 m/s")
        self.world = world
        self.scenario = scenario
        self.egos = dict(egos)
        self.tm_port = int(tm_port)
        self.helper_speed_mps = float(helper_speed_mps)
        self.recipient_speed_mps = float(recipient_speed_mps)
        self.pedestrian_speed_mps = float(pedestrian_speed_mps)
        self.pedestrian_start_delay_s = float(pedestrian_start_delay_s)
        self.owned: list[carla.Actor] = []
        self.controllers: Dict[str, DirectRouteController] = {}
        self.occluder: Optional[carla.Actor] = None
        self.target_vehicle: Optional[carla.Actor] = None
        self.walker: Optional[carla.Actor] = None
        self.walker_end: Optional[carla.Location] = None
        self.walker_started = False
        self.walker_completed = False
        self.target_started = False
        self.queue_occluder_started = False
        self.review_only_yield_ever = False
        self.trace: list[dict] = []
        self.first_direct_route_yield: Dict[str, Optional[dict[str, object]]] = {
            role: None for role in ROLE_NAMES
        }
        self._last_controller_tick_roles: set[str] = set()
        self.settlement: Dict[str, object] = {}
        self.controlled_spawn_barrier_frame_id: Optional[int] = None
        self._traffic_light_restore: list[tuple[carla.TrafficLight, object, bool]] = []
        self.realized_lane_contract: Mapping[str, object] = scenario.lane_contract

    def place_egos(self, maximum_pose_error_m: float = 0.05) -> dict:
        """Relocate both staged collector egos behind one synchronous barrier.

        CARLA queues ``set_transform`` until the next server tick.  Verifying
        immediately happens to work when the requested geometry equals the
        staging spawn (the curbside pair), but can read an old staging pose for
        another geometry. Command both roles first so neither receives a
        relative motion frame, then use a bounded shared orchestrator-owned
        tick barrier until both frozen actors realize their transforms.
        """

        for role in ROLE_NAMES:
            actor = self.egos[role]
            expected = self.scenario.transforms[role]
            actor.set_autopilot(False, self.tm_port)
            actor.set_simulate_physics(False)
            actor.set_transform(expected)
            actor.set_target_velocity(carla.Vector3D())
            actor.set_target_angular_velocity(carla.Vector3D())

        realized = {}
        for placement_tick_count in range(1, EGO_PLACEMENT_MAX_TICKS + 1):
            placement_barrier_frame_id = int(self.world.tick(2.0))
            realized = {}
            placement_complete = True
            for role in ROLE_NAMES:
                actor = self.egos[role]
                expected = self.scenario.transforms[role]
                observed = actor.get_transform()
                pose_error_m = observed.location.distance(expected.location)
                yaw_error_deg = abs(
                    wrap_degrees(
                        float(observed.rotation.yaw)
                        - float(expected.rotation.yaw)
                    )
                )
                realized[role] = {
                    "actor_id": int(actor.id),
                    "pose_error_m": float(pose_error_m),
                    "yaw_error_deg": float(yaw_error_deg),
                    "x": float(observed.location.x),
                    "y": float(observed.location.y),
                    "z": float(observed.location.z),
                    "yaw_deg": float(observed.rotation.yaw),
                    "placement_barrier_frame_id": placement_barrier_frame_id,
                    "placement_tick_count": placement_tick_count,
                }
                placement_complete &= (
                    pose_error_m <= float(maximum_pose_error_m)
                    and yaw_error_deg <= 0.25
                )
            if placement_complete:
                return realized
        raise RuntimeError(
            "exact scenario placement did not realize behind the bounded shared "
            f"barrier: {realized}"
        )

    def _spawn_vehicle(
        self, blueprint_id: str, transform: carla.Transform, role_name: str
    ) -> carla.Actor:
        blueprint = self.world.get_blueprint_library().find(str(blueprint_id))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(role_name))
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"controlled actor spawn failed: {role_name}")
        actor.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )
        self.owned.append(actor)
        return actor

    def spawn_controlled_actors(self) -> dict:
        layout = self.scenario.layout
        if layout in {"signalized_corner", "cross_traffic_vehicle"}:
            for _role, traffic_light in controlled_traffic_lights(self.world).items():
                state = traffic_light.get_state()
                frozen = bool(traffic_light.is_frozen())
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
                self._traffic_light_restore.append((traffic_light, state, frozen))

        occluder_spec = {
            "curbside_opposite": ("vehicle.sprinter.mercedes", CURBSIDE_OCCLUDER_TRANSFORM),
            "signalized_corner": ("vehicle.sprinter.mercedes", SIGNALIZED_OCCLUDER_TRANSFORM),
            "midblock_van": ("vehicle.sprinter.mercedes", MIDBLOCK_OCCLUDER_TRANSFORM),
            "cross_traffic_vehicle": (CROSS_TRAFFIC_OCCLUDER_BLUEPRINT, CROSS_TRAFFIC_OCCLUDER_TRANSFORM),
            "parked_vehicle_pullout": (PULLOUT_OCCLUDER_BLUEPRINT, PULLOUT_OCCLUDER_TRANSFORM),
            "queue_reveal_vehicle": (QUEUE_REVEAL_OCCLUDER_BLUEPRINT, QUEUE_REVEAL_OCCLUDER_TRANSFORM),
        }.get(layout)
        if occluder_spec is not None:
            self.occluder = self._spawn_vehicle(
                occluder_spec[0],
                world_transform(occluder_spec[1]),
                f"phase2_audit_{layout}_occluder",
            )
            if layout == "queue_reveal_vehicle":
                self.occluder.set_simulate_physics(True)
            elif layout in {"midblock_van", "cross_traffic_vehicle", "parked_vehicle_pullout"}:
                # Settled after all controlled actors cross the shared spawn
                # realization barrier below.
                self.occluder.set_simulate_physics(True)
            else:
                self.occluder.set_simulate_physics(False)

        if self.scenario.hazard_present and layout in {
            "curbside_opposite",
            "signalized_corner",
            "midblock_van",
        }:
            start, end = {
                "curbside_opposite": (CURBSIDE_WALKER_START, CURBSIDE_WALKER_END),
                "signalized_corner": (SIGNALIZED_WALKER_START, SIGNALIZED_WALKER_END),
                "midblock_van": (MIDBLOCK_WALKER_START, MIDBLOCK_WALKER_END),
            }[layout]
            blueprints = sorted(
                self.world.get_blueprint_library().filter("walker.pedestrian.*"),
                key=lambda item: item.id,
            )
            if not blueprints:
                raise RuntimeError("no walker blueprint is available")
            blueprint = blueprints[0]
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute(
                    "role_name", f"phase2_registered_target_{layout}_pedestrian"
                )
            self.walker = self.world.try_spawn_actor(blueprint, world_transform(start))
            if self.walker is None:
                raise RuntimeError("registered pedestrian spawn failed")
            observed_role = str(self.walker.attributes.get("role_name", ""))
            if observed_role != f"phase2_registered_target_{layout}_pedestrian":
                raise RuntimeError(
                    "registered pedestrian role_name was not realized: "
                    f"{observed_role!r}"
                )
            self.owned.append(self.walker)
            self.walker_end = carla.Location(
                x=float(end[0]), y=float(end[1]), z=float(end[2])
            )

        if self.scenario.hazard_present and layout in {
            "cross_traffic_vehicle",
            "parked_vehicle_pullout",
            "queue_reveal_vehicle",
        }:
            blueprint_id, transform_values = {
                "cross_traffic_vehicle": (
                    CROSS_TRAFFIC_TARGET_BLUEPRINT,
                    CROSS_TRAFFIC_TARGET_TRANSFORM,
                ),
                "parked_vehicle_pullout": (
                    PULLOUT_TARGET_BLUEPRINT,
                    PULLOUT_TARGET_TRANSFORM,
                ),
                "queue_reveal_vehicle": (
                    QUEUE_REVEAL_TARGET_BLUEPRINT,
                    QUEUE_REVEAL_TARGET_TRANSFORM,
                ),
            }[layout]
            self.target_vehicle = self._spawn_vehicle(
                blueprint_id,
                world_transform(transform_values),
                f"phase2_registered_target_{layout}_vehicle",
            )
            observed_role = str(self.target_vehicle.attributes.get("role_name", ""))
            if observed_role != f"phase2_registered_target_{layout}_vehicle":
                raise RuntimeError(
                    "registered target vehicle role_name was not realized: "
                    f"{observed_role!r}"
                )
        if self.owned:
            # CARLA can return a new actor proxy before its requested transform
            # is visible to get_transform()/map projection.  One shared tick
            # realizes every controlled spawn before settlement or lane checks.
            self.controlled_spawn_barrier_frame_id = int(self.world.tick(2.0))

        if self.occluder is not None and layout in {
            "midblock_van",
            "cross_traffic_vehicle",
            "parked_vehicle_pullout",
        }:
            assert occluder_spec is not None
            self.settlement["occluder"] = _settle_parked_actor(
                self.world,
                self.occluder,
                world_transform(occluder_spec[1]),
            )
        if self.target_vehicle is not None and layout == "queue_reveal_vehicle":
            self.settlement["target"] = _settle_parked_actor(
                self.world,
                self.target_vehicle,
                world_transform(QUEUE_REVEAL_TARGET_TRANSFORM),
            )

        if self.occluder is not None and layout == "midblock_van":
            self.realized_lane_contract = midblock_lane_contract(
                self.world.get_map(),
                self.scenario.transforms,
                self.occluder.get_transform(),
            )
        elif self.occluder is not None and layout == "cross_traffic_vehicle":
            self.realized_lane_contract = cross_traffic_geometry_contract(
                self.world.get_map(),
                self.scenario.transforms,
                self.occluder.get_transform(),
                (
                    self.target_vehicle.get_transform()
                    if self.target_vehicle is not None
                    else world_transform(CROSS_TRAFFIC_TARGET_TRANSFORM)
                ),
                self.scenario.routes,
            )
        elif self.occluder is not None and layout == "parked_vehicle_pullout":
            self.realized_lane_contract = pullout_geometry_contract(
                self.world.get_map(),
                self.scenario.transforms,
                self.occluder.get_transform(),
                (
                    self.target_vehicle.get_transform()
                    if self.target_vehicle is not None
                    else world_transform(PULLOUT_TARGET_TRANSFORM)
                ),
                self.scenario.routes,
            )
        elif self.occluder is not None and layout == "queue_reveal_vehicle":
            self.realized_lane_contract = queue_reveal_geometry_contract(
                self.world.get_map(),
                self.scenario.transforms,
                self.occluder.get_transform(),
                (
                    self.target_vehicle.get_transform()
                    if self.target_vehicle is not None
                    else world_transform(QUEUE_REVEAL_TARGET_TRANSFORM)
                ),
                self.scenario.routes,
            )

        return {
            "layout": layout,
            "hazard_present": self.scenario.hazard_present,
            "owned_actor_ids": [int(actor.id) for actor in self.owned],
            "controlled_spawn_barrier_frame_id": self.controlled_spawn_barrier_frame_id,
            "traffic_light_override_count": len(self._traffic_light_restore),
            "settlement": dict(self.settlement),
        }

    def activate_motion(self) -> None:
        for role in ROLE_NAMES:
            actor = self.egos[role]
            actor.set_autopilot(False, self.tm_port)
            actor.set_simulate_physics(True)
            actor.apply_control(carla.VehicleControl())
            self.controllers[role] = DirectRouteController(
                actor,
                self.scenario.routes[role],
                target_speed_mps=(
                    self.helper_speed_mps if role == "helper" else self.recipient_speed_mps
                ),
            )
        if self.target_vehicle is not None and self.scenario.layout != "queue_reveal_vehicle":
            self.target_vehicle.set_simulate_physics(True)
            self.target_vehicle.apply_control(carla.VehicleControl())
            self.controllers["target"] = DirectRouteController(
                self.target_vehicle,
                self.scenario.routes["target"],
                target_speed_mps=(
                    PULLOUT_TARGET_SPEED_MPS
                    if self.scenario.layout == "parked_vehicle_pullout"
                    else 3.6
                ),
                waypoint_reach_m=(
                    0.75 if self.scenario.layout == "parked_vehicle_pullout" else 3.5
                ),
            )
        if self.scenario.layout == "queue_reveal_vehicle" and self.occluder is not None:
            self.occluder.set_simulate_physics(True)
            self.occluder.apply_control(carla.VehicleControl())
            self.controllers["occluder"] = DirectRouteController(
                self.occluder,
                self.scenario.routes["occluder"],
                target_speed_mps=QUEUE_REVEAL_OCCLUDER_SPEED_MPS,
                waypoint_reach_m=0.75,
            )

    def before_tick(self, elapsed_s: float) -> None:
        layout = self.scenario.layout
        self._last_controller_tick_roles.clear()
        for controller_role, controller in self.controllers.items():
            if (
                controller_role == "occluder"
                and float(elapsed_s) < QUEUE_REVEAL_OCCLUDER_START_DELAY_S
            ):
                assert self.occluder is not None
                self.occluder.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                )
                continue
            if controller_role == "occluder":
                self.queue_occluder_started = True
            target_delay_s = (
                PULLOUT_TARGET_START_DELAY_S
                if layout == "parked_vehicle_pullout"
                else 0.0
            )
            if controller_role == "target" and float(elapsed_s) < target_delay_s:
                assert self.target_vehicle is not None
                self.target_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                )
                continue
            if controller_role == "target":
                self.target_started = True
            if (
                controller_role == "recipient"
                and self.target_vehicle is not None
                and layout in {
                    "cross_traffic_vehicle",
                    "parked_vehicle_pullout",
                    "queue_reveal_vehicle",
                }
                and float(elapsed_s)
                >= (
                    QUEUE_REVEAL_OCCLUDER_START_DELAY_S
                    if layout == "queue_reveal_vehicle"
                    else target_delay_s
                )
            ):
                conflict = self.realized_lane_contract["registered_conflict_point"]
                recipient_location = self.egos["recipient"].get_location()
                target_location = self.target_vehicle.get_location()
                recipient_distance = math.hypot(
                    float(recipient_location.x) - float(conflict["x"]),
                    float(recipient_location.y) - float(conflict["y"]),
                )
                target_cleared = (
                    False
                    if layout == "queue_reveal_vehicle"
                    else (
                        float(target_location.y) >= float(conflict["y"]) + 4.5
                        if layout == "cross_traffic_vehicle"
                        else float(target_location.x) >= float(conflict["x"]) + 4.5
                    )
                )
                trigger = {
                    "cross_traffic_vehicle": CROSS_TRAFFIC_REVIEW_YIELD_TRIGGER_M,
                    "parked_vehicle_pullout": PULLOUT_REVIEW_YIELD_TRIGGER_M,
                    "queue_reveal_vehicle": QUEUE_REVEAL_REVIEW_YIELD_TRIGGER_M,
                }[layout]
                if recipient_distance <= trigger and not target_cleared:
                    self.egos["recipient"].apply_control(
                        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False)
                    )
                    self.review_only_yield_ever = True
                    continue
            controller.tick()
            self._last_controller_tick_roles.add(controller_role)

    def after_tick(self, frame_id: int, elapsed_s: float) -> None:
        if (
            self.walker is not None
            and self.walker_end is not None
            and not self.walker_started
            and float(elapsed_s) >= self.pedestrian_start_delay_s
        ):
            location = self.walker.get_location()
            dx = float(self.walker_end.x - location.x)
            dy = float(self.walker_end.y - location.y)
            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                raise RuntimeError("controlled pedestrian path has zero length")
            self.walker.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(x=dx / norm, y=dy / norm, z=0.0),
                    speed=(
                        self.pedestrian_speed_mps
                        / CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE
                    ),
                    jump=False,
                )
            )
            self.walker_started = True
        if (
            self.walker is not None
            and self.walker_end is not None
            and self.walker_started
            and not self.walker_completed
            and self.walker.get_location().distance(self.walker_end) <= 0.25
        ):
            self.walker.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(), speed=0.0, jump=False)
            )
            self.walker_completed = True
        row = {
            "frame_id": int(frame_id),
            "elapsed_s": float(elapsed_s),
            "walker_started": int(self.walker_started),
            "walker_completed": int(self.walker_completed),
            "target_started": int(self.target_started),
            "queue_occluder_started": int(self.queue_occluder_started),
            "review_only_gt_safety_yield_ever": int(self.review_only_yield_ever),
        }
        for role, actor in self.egos.items():
            transform = actor.get_transform()
            row.update(
                {
                    f"{role}_x": float(transform.location.x),
                    f"{role}_y": float(transform.location.y),
                    f"{role}_z": float(transform.location.z),
                    f"{role}_yaw_deg": float(transform.rotation.yaw),
                    f"{role}_speed_mps": _speed(actor),
                }
            )
            controller = self.controllers.get(role)
            yield_event = (
                controller.last_yield
                if controller is not None
                and role in self._last_controller_tick_roles
                and not bool(getattr(controller, "finished", False))
                else None
            )
            row.update(direct_route_yield_trace_fields(role, yield_event))
            if yield_event is not None and self.first_direct_route_yield[role] is None:
                self.first_direct_route_yield[role] = {
                    "frame_id": int(frame_id),
                    "elapsed_s": float(elapsed_s),
                    "actor_id": int(yield_event["actor_id"]),
                    "actor_type": str(yield_event["type_id"]),
                }
        row["walker_speed_mps"] = _speed(self.walker) if self.walker is not None else 0.0
        row["target_vehicle_speed_mps"] = (
            _speed(self.target_vehicle) if self.target_vehicle is not None else 0.0
        )
        self.trace.append(row)

    def summary(self) -> dict:
        return {
            "schema": "scenesense.phase2_calibration_scenario_runtime.v2",
            "geometry_or_route_id": self.scenario.geometry_or_route_id,
            "layout": self.scenario.layout,
            "scenario_role": self.scenario.scenario_role,
            "hazard_present": self.scenario.hazard_present,
            "trace_frame_count": len(self.trace),
            "walker_started": self.walker_started,
            "walker_completed": self.walker_completed,
            "target_started": self.target_started,
            "queue_occluder_started": self.queue_occluder_started,
            "review_only_gt_safety_yield_ever": self.review_only_yield_ever,
            "direct_route_yield_ever_by_role": {
                role: self.first_direct_route_yield[role] is not None
                for role in ROLE_NAMES
            },
            "first_direct_route_yield_by_role": {
                role: self.first_direct_route_yield[role] for role in ROLE_NAMES
            },
            "owned_actor_ids": [int(actor.id) for actor in self.owned],
            "controlled_spawn_barrier_frame_id": self.controlled_spawn_barrier_frame_id,
            "lane_contract": dict(self.realized_lane_contract),
            "settlement": dict(self.settlement),
        }

    def destroy(self) -> None:
        for traffic_light, state, frozen in reversed(self._traffic_light_restore):
            try:
                traffic_light.set_state(state)
                traffic_light.freeze(bool(frozen))
            except RuntimeError:
                pass
        self._traffic_light_restore.clear()
        for actor in reversed(self.owned):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self.owned.clear()

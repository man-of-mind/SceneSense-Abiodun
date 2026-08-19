#!/usr/bin/env python3
"""Derived safe-spawn launcher for the read-only advisor traffic script.

The advisor script remains the population implementation. This wrapper only
filters its already-shuffled CARLA spawn-point list to the registered ego-route
corridor and raises the pairwise seed clearance before delegating to ``main``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_agent.advisor_helper_scripts.codes import generate_traffic_v1 as advisor


READY_SCHEMA = "scenesense.held_ambient_population_ready.v1"
RELEASE_SCHEMA = "scenesense.held_ambient_population_release.v1"
RELEASED_SCHEMA = "scenesense.held_ambient_population_released.v1"
FAILED_SCHEMA = "scenesense.held_ambient_population_failed.v1"
VEHICLE_SETTLE_MAX_TICKS = 30
VEHICLE_SETTLE_STABLE_TICKS = 3
VEHICLE_SETTLE_MAX_HORIZONTAL_DRIFT_M = 0.25
VEHICLE_SETTLE_MAX_YAW_DRIFT_DEG = 3.0
EXTERNAL_TICK_TRANSIENT_RETRIES = 3
HELD_VEHICLE_STAGNATION_LIMIT = 5


def _wait_for_external_tick_with_retry(
    world: object, *args: object, **kwargs: object
) -> object:
    """Wait as a sync follower, retrying CARLA's transient bare exception.

    CARLA 0.10 can occasionally surface a bare ``std::exception`` while a
    newly spawned walker controller is being registered.  The server and sync
    owner remain healthy, so one subsequent tick completes registration.  Do
    not retry timeouts or descriptive RuntimeErrors: those indicate a real
    clock/server failure and must remain fail-fast.
    """

    for attempt in range(1, EXTERNAL_TICK_TRANSIENT_RETRIES + 1):
        try:
            return world.wait_for_tick(*args, **kwargs)
        except RuntimeError as exc:
            if str(exc).strip() != "std::exception" or attempt == EXTERNAL_TICK_TRANSIENT_RETRIES:
                raise
            advisor.logging.warning(
                "Transient CARLA std::exception while waiting for external "
                "sync tick; retrying (%d/%d)",
                attempt,
                EXTERNAL_TICK_TRANSIENT_RETRIES,
            )


class _RetryingWorldProxy:
    """Delegate a CARLA world while hardening every follower tick wait."""

    def __init__(self, world: object) -> None:
        self._world = world

    def wait_for_tick(self, *args: object, **kwargs: object) -> object:
        return _wait_for_external_tick_with_retry(self._world, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._world, name)


class _RetryingClientProxy:
    """Return the retrying world to the unmodified advisor main loop."""

    def __init__(self, client: object) -> None:
        self._client = client

    def get_world(self) -> _RetryingWorldProxy:
        return _RetryingWorldProxy(self._client.get_world())

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)


def _write_json_create(path: Path, payload: object) -> None:
    """Publish a handshake record atomically and without overwriting evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"population handshake path already exists: {path}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _zero_and_hold(actor: object, *, tm_port: int, vehicle: bool) -> None:
    """Make a newly spawned ambient actor inert before another world tick."""

    if vehicle:
        actor.set_autopilot(False, int(tm_port))
    actor.set_target_velocity(advisor.carla.Vector3D())
    actor.set_target_angular_velocity(advisor.carla.Vector3D())
    actor.set_simulate_physics(False)


def _held_population_rows(population: object) -> tuple[list[dict], list[dict]]:
    """Return ID-free comparison rows and ID-bearing diagnostics for held actors."""

    vehicle_ids = list(population.vehicle_ids)
    walker_ids = [record.get("id") for record in population.walkers]
    walker_records = {
        record.get("id"): record for record in population.walkers
    }
    live = population._live_actor_map([*vehicle_ids, *walker_ids])
    signature = []
    diagnostics = []
    for actor_id in [*vehicle_ids, *walker_ids]:
        actor = live.get(actor_id)
        if actor is None:
            continue
        transform = actor.get_transform()
        row = {
            "type_id": str(actor.type_id),
            "role_name": str(actor.attributes.get("role_name", "")),
            "x": round(float(transform.location.x), 4),
            "y": round(float(transform.location.y), 4),
            "yaw_deg": round(float(transform.rotation.yaw), 3),
        }
        walker_record = walker_records.get(actor_id)
        if walker_record is None:
            row.update(
                {
                    "motion_mode": str(
                        getattr(
                            population,
                            "_scenesense_released_vehicle_motion_mode",
                            "runner_owned_unspecified",
                        )
                    ),
                    "motion_speed_mps": round(
                        float(
                            getattr(
                                population,
                                "_scenesense_released_vehicle_speed_mps",
                                0.0,
                            )
                        ),
                        4,
                    ),
                    "motion_target_x": None,
                    "motion_target_y": None,
                    "motion_target_z": None,
                }
            )
        else:
            walker_mode = str(
                getattr(
                    population,
                    "_scenesense_released_walker_motion_mode",
                    "walker_ai_destination",
                )
            )
            destination = walker_record.get("_scenesense_release_destination")
            if walker_mode == "runner_owned_stationary":
                row.update(
                    {
                        "motion_mode": walker_mode,
                        "motion_speed_mps": 0.0,
                        "motion_target_x": None,
                        "motion_target_y": None,
                        "motion_target_z": None,
                    }
                )
            else:
                if destination is None:
                    raise RuntimeError(
                        "held walker has no immutable release destination"
                    )
                row.update(
                    {
                        "motion_mode": "walker_ai_destination",
                        "motion_speed_mps": round(float(walker_record["speed"]), 4),
                        "motion_target_x": round(float(destination.x), 4),
                        "motion_target_y": round(float(destination.y), 4),
                        "motion_target_z": round(float(destination.z), 4),
                    }
                )
        signature.append(row)
        diagnostics.append(
            {
                **row,
                "actor_id": int(actor.id),
                "z": round(float(transform.location.z), 4),
            }
        )
    sort_key = lambda item: (
        item["type_id"],
        item["role_name"],
        item["x"],
        item["y"],
        item["yaw_deg"],
    )
    return sorted(signature, key=sort_key), sorted(diagnostics, key=sort_key)


def _nearest_unique_spawn_assignments(
    actors: Sequence[object], spawn_points: Sequence[object]
) -> list[tuple[object, object]]:
    """Recover the commanded spawn transform for each realized vehicle."""

    remaining = list(spawn_points)
    assignments = []
    for actor in sorted(actors, key=lambda value: int(value.id)):
        if not remaining:
            raise RuntimeError("vehicle spawn assignment ran out of candidate transforms")
        location = actor.get_location()
        transform = min(
            remaining,
            key=lambda candidate: location.distance(candidate.location),
        )
        remaining.remove(transform)
        assignments.append((actor, transform))
    return assignments


def _settled_spawn_transform(commanded: object, settled: object) -> object:
    """Keep registered XY/heading while retaining the physical road height."""

    return advisor.carla.Transform(
        advisor.carla.Location(
            x=float(commanded.location.x),
            y=float(commanded.location.y),
            z=float(settled.location.z),
        ),
        advisor.carla.Rotation(
            pitch=float(commanded.rotation.pitch),
            yaw=float(commanded.rotation.yaw),
            roll=float(commanded.rotation.roll),
        ),
    )


def _registered_spawn_pose_errors(
    commanded: object, realized: object
) -> tuple[float, float]:
    """Return horizontal and wrapped-yaw errors from a registered pose."""

    horizontal_error = math.hypot(
        float(realized.location.x) - float(commanded.location.x),
        float(realized.location.y) - float(commanded.location.y),
    )
    yaw_error = abs(
        (
            float(realized.rotation.yaw)
            - float(commanded.rotation.yaw)
            + 180.0
        )
        % 360.0
        - 180.0
    )
    return horizontal_error, yaw_error


def _require_registered_spawn_pose(
    *,
    actor_id: int,
    commanded: object,
    realized: object,
    maximum_horizontal_error_m: float,
    maximum_yaw_error_deg: float,
) -> tuple[float, float]:
    """Enforce the configured registered-pose contract and return its errors."""

    horizontal_error, yaw_error = _registered_spawn_pose_errors(
        commanded, realized
    )
    if (
        horizontal_error > float(maximum_horizontal_error_m)
        or yaw_error > float(maximum_yaw_error_deg)
    ):
        raise RuntimeError(
            "settled vehicle did not retain its registered spawn pose: "
            f"actor_id={actor_id} xy_error={horizontal_error:.6f} "
            f"yaw_error={yaw_error:.6f} "
            f"xy_limit={float(maximum_horizontal_error_m):.6f} "
            f"yaw_limit={float(maximum_yaw_error_deg):.6f}"
        )
    return horizontal_error, yaw_error


def _settle_and_hold_spawned_vehicles(
    population: object,
    assignments: Sequence[tuple[object, object]],
    *,
    maximum_horizontal_error_m: float,
    maximum_yaw_error_deg: float,
) -> None:
    """Resolve suspension before READY, then freeze the stable road pose.

    Releasing an actor directly from the route transform's +0.6 m spawn height
    made CARLA behavior nondeterministic: some vehicles landed and drove while
    others remained physically asleep despite a non-zero throttle command.
    Settle under a stationary brake while the external owner ticks, then retain
    the settled Z while restoring exact registered XY/yaw for pair matching.
    """

    for actor, commanded in assignments:
        actor.set_autopilot(False, population.tm_port)
        actor.set_simulate_physics(False)
        actor.set_transform(commanded)
    population._wait_for_actor_update()
    live = population._live_actor_map([int(actor.id) for actor, _ in assignments])
    for actor, _commanded in assignments:
        realized = live.get(int(actor.id))
        if realized is None:
            raise RuntimeError("spawned vehicle disappeared before settlement")
        realized.set_simulate_physics(True)
        realized.apply_control(
            advisor.carla.VehicleControl(
                throttle=0.0, brake=1.0, hand_brake=True
            )
        )

    previous_z: dict[int, float] = {}
    stable_ticks = 0
    settled_live = live
    for _tick in range(VEHICLE_SETTLE_MAX_TICKS):
        population._wait_for_actor_update()
        settled_live = population._live_actor_map(
            [int(actor.id) for actor, _ in assignments]
        )
        all_stable = len(settled_live) == len(assignments)
        for actor, commanded in assignments:
            realized = settled_live.get(int(actor.id))
            if realized is None:
                all_stable = False
                continue
            transform = realized.get_transform()
            velocity = realized.get_velocity()
            horizontal_drift = math.hypot(
                float(transform.location.x) - float(commanded.location.x),
                float(transform.location.y) - float(commanded.location.y),
            )
            yaw_drift = abs(
                (
                    float(transform.rotation.yaw)
                    - float(commanded.rotation.yaw)
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            prior = previous_z.get(int(actor.id))
            z_step = (
                abs(float(transform.location.z) - prior)
                if prior is not None
                else float("inf")
            )
            previous_z[int(actor.id)] = float(transform.location.z)
            if (
                horizontal_drift > VEHICLE_SETTLE_MAX_HORIZONTAL_DRIFT_M
                or yaw_drift > VEHICLE_SETTLE_MAX_YAW_DRIFT_DEG
                or abs(float(velocity.z)) > 0.05
                or z_step > 0.003
            ):
                all_stable = False
        stable_ticks = stable_ticks + 1 if all_stable else 0
        if stable_ticks >= VEHICLE_SETTLE_STABLE_TICKS:
            break
    else:
        raise RuntimeError("spawned vehicles did not settle before READY")

    for actor, commanded in assignments:
        realized = settled_live[int(actor.id)]
        frozen = _settled_spawn_transform(commanded, realized.get_transform())
        realized.set_simulate_physics(False)
        realized.set_target_velocity(advisor.carla.Vector3D())
        realized.set_target_angular_velocity(advisor.carla.Vector3D())
        realized.set_transform(frozen)
    population._wait_for_actor_update()
    frozen_live = population._live_actor_map(
        [int(actor.id) for actor, _ in assignments]
    )
    for actor, commanded in assignments:
        realized = frozen_live.get(int(actor.id))
        if realized is None:
            raise RuntimeError("settled vehicle disappeared before READY")
        transform = realized.get_transform()
        _require_registered_spawn_pose(
            actor_id=int(actor.id),
            commanded=commanded,
            realized=transform,
            maximum_horizontal_error_m=maximum_horizontal_error_m,
            maximum_yaw_error_deg=maximum_yaw_error_deg,
        )


def _route_xy(path: Path) -> List[Tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        points = [
            (float(row["ego_x"]), float(row["ego_y"]))
            for row in csv.DictReader(stream)
        ]
    if len(points) < 2:
        raise ValueError(f"route progress CSV must contain at least two points: {path}")
    return points


def _route_distance_and_heading_error_deg(
    transform: object, points: Sequence[Tuple[float, float]]
) -> Tuple[float, float]:
    x = float(transform.location.x)
    y = float(transform.location.y)
    distances = [math.hypot(x - route_x, y - route_y) for route_x, route_y in points]
    index = min(range(len(points)), key=distances.__getitem__)
    # These are spawn-support polylines, not implicit loops. At the terminal
    # point use the preceding segment rather than inventing a reversed
    # end-to-start tangent that can reject a correctly headed spawn.
    if index == len(points) - 1:
        route_x, route_y = points[index - 1]
        next_x, next_y = points[index]
    else:
        route_x, route_y = points[index]
        next_x, next_y = points[index + 1]
    route_yaw = math.degrees(math.atan2(next_y - route_y, next_x - route_x))
    heading_error = (
        float(transform.rotation.yaw) - route_yaw + 180.0
    ) % 360.0 - 180.0
    return float(distances[index]), float(abs(heading_error))


def _route_derived_spawn_transforms(
    world: object,
    routes: Sequence[Sequence[Tuple[float, float]]],
    protected: Sequence[Tuple[float, float]],
    *,
    protected_clearance_m: float,
    pairwise_clearance_m: float,
    cross_route_clearance_m: float | None = None,
) -> list[object]:
    """Derive deterministic legal vehicle spawns from reviewed route samples.

    Town10HD's built-in spawn catalog is sparse and is not a faithful capacity
    description of a reviewed lane. Projecting the frozen route samples back to
    driving waypoints preserves the reviewed lane/direction contract while the
    greedy clearance pass prevents unsafe initial packing.  Junction waypoints
    are never valid initial placements: Traffic Manager can deadlock when two
    newly registered actors already occupy conflicting junction reservations.
    One legal-lane continuation point beyond each open route is admitted so
    rejecting junction interiors does not artificially remove downstream
    capacity.
    """

    by_route: list[list[object]] = []
    for points in routes:
        candidates = []
        for index, (x, y) in enumerate(points):
            if index == len(points) - 1:
                previous_index = index - 1
                while previous_index >= 0 and math.hypot(
                    x - points[previous_index][0],
                    y - points[previous_index][1],
                ) < 1e-6:
                    previous_index -= 1
                if previous_index < 0:
                    continue
                route_x, route_y = points[previous_index]
                next_x, next_y = points[index]
            else:
                route_x, route_y = points[index]
                next_index = index + 1
                while next_index < len(points) and math.hypot(
                    points[next_index][0] - x,
                    points[next_index][1] - y,
                ) < 1e-6:
                    next_index += 1
                if next_index >= len(points):
                    continue
                next_x, next_y = points[next_index]
            waypoint = world.get_map().get_waypoint(
                advisor.carla.Location(x=float(x), y=float(y), z=0.0),
                project_to_road=True,
                lane_type=advisor.carla.LaneType.Driving,
            )
            if waypoint is None or bool(getattr(waypoint, "is_junction", False)):
                continue
            transform = waypoint.transform
            if math.hypot(
                float(transform.location.x) - float(x),
                float(transform.location.y) - float(y),
            ) > 2.0:
                continue
            route_yaw = math.degrees(
                math.atan2(next_y - route_y, next_x - route_x)
            )
            heading_error = (
                float(transform.rotation.yaw) - route_yaw + 180.0
            ) % 360.0 - 180.0
            if abs(heading_error) > 45.0:
                continue
            if any(
                math.hypot(
                    float(transform.location.x) - protected_x,
                    float(transform.location.y) - protected_y,
                )
                < float(protected_clearance_m)
                for protected_x, protected_y in protected
            ):
                continue
            spawn = advisor.carla.Transform(
                advisor.carla.Location(
                    x=float(transform.location.x),
                    y=float(transform.location.y),
                    z=float(transform.location.z) + 0.6,
                ),
                advisor.carla.Rotation(
                    pitch=float(transform.rotation.pitch),
                    yaw=float(transform.rotation.yaw),
                    roll=float(transform.rotation.roll),
                ),
            )
            candidates.append(spawn)

        terminal_waypoint = world.get_map().get_waypoint(
            advisor.carla.Location(
                x=float(points[-1][0]), y=float(points[-1][1]), z=0.0
            ),
            project_to_road=True,
            lane_type=advisor.carla.LaneType.Driving,
        )
        next_waypoints = (
            list(terminal_waypoint.next(float(pairwise_clearance_m)))
            if terminal_waypoint is not None
            and callable(getattr(terminal_waypoint, "next", None))
            else []
        )
        terminal_previous_index = len(points) - 2
        while terminal_previous_index >= 0 and math.hypot(
            points[-1][0] - points[terminal_previous_index][0],
            points[-1][1] - points[terminal_previous_index][1],
        ) < 1e-6:
            terminal_previous_index -= 1
        if next_waypoints and terminal_previous_index >= 0:
            terminal_yaw = math.degrees(
                math.atan2(
                    points[-1][1] - points[terminal_previous_index][1],
                    points[-1][0] - points[terminal_previous_index][0],
                )
            )
            continuation = min(
                next_waypoints,
                key=lambda candidate: abs(
                    (
                        float(candidate.transform.rotation.yaw)
                        - terminal_yaw
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                ),
            )
            transform = continuation.transform
            heading_error = abs(
                (float(transform.rotation.yaw) - terminal_yaw + 180.0)
                % 360.0
                - 180.0
            )
            protected_tail = any(
                math.hypot(
                    float(transform.location.x) - protected_x,
                    float(transform.location.y) - protected_y,
                )
                < float(protected_clearance_m)
                for protected_x, protected_y in protected
            )
            if (
                not bool(getattr(continuation, "is_junction", False))
                and heading_error <= 45.0
                and not protected_tail
            ):
                candidates.append(
                    advisor.carla.Transform(
                        advisor.carla.Location(
                            x=float(transform.location.x),
                            y=float(transform.location.y),
                            z=float(transform.location.z) + 0.6,
                        ),
                        advisor.carla.Rotation(
                            pitch=float(transform.rotation.pitch),
                            yaw=float(transform.rotation.yaw),
                            roll=float(transform.rotation.roll),
                        ),
                    )
                )
        by_route.append(candidates)

    cross_clearance = (
        float(pairwise_clearance_m)
        if cross_route_clearance_m is None
        else float(cross_route_clearance_m)
    )

    # Search route phases, rather than using one arbitrary endpoint alignment.
    # Opposing reviewed paths often begin at opposite corridor ends. A single
    # greedy pass can then return a nominally large pool whose first N entries
    # contain four vehicles on one lane and only two on the other, producing a
    # deterministic queue/gridlock. The phase search maximizes the minimum
    # per-route capacity and the final round-robin ordering guarantees that a
    # requested prefix remains balanced.
    if len(by_route) <= 1:
        phase_choices = [range(1)]
    elif len(by_route) == 2:
        phase_choices = [range(len(values)) for values in by_route]
    else:
        raise ValueError("route-derived fallback supports at most two reviewed routes")
    best_selected: list[tuple[int, object]] = []
    best_score: tuple[int, int, int] | None = None
    for offsets in itertools.product(*phase_choices):
        phased = [
            [*values[offset:], *values[:offset]] if values else []
            for values, offset in zip(by_route, offsets)
        ]
        interleaved = []
        for ordinal in range(max((len(values) for values in phased), default=0)):
            for route_index, values in enumerate(phased):
                if ordinal < len(values):
                    interleaved.append((route_index, values[ordinal]))
        selected: list[tuple[int, object]] = []
        for route_index, candidate in interleaved:
            if all(
                candidate.location.distance(existing.location)
                >= (
                    float(pairwise_clearance_m)
                    if route_index == existing_route_index
                    else cross_clearance
                )
                for existing_route_index, existing in selected
            ):
                selected.append((route_index, candidate))
        counts = [
            sum(route_index == expected for route_index, _candidate in selected)
            for expected in range(len(by_route))
        ]
        score = (
            min(counts, default=0),
            len(selected),
            -abs(max(counts, default=0) - min(counts, default=0)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_selected = selected

    selected_by_route = [
        [candidate for observed, candidate in best_selected if observed == expected]
        for expected in range(len(by_route))
    ]
    balanced = []
    for ordinal in range(
        max((len(values) for values in selected_by_route), default=0)
    ):
        for values in selected_by_route:
            if ordinal < len(values):
                balanced.append(values[ordinal])
    return balanced


def _pairwise_spaced_spawn_transforms(
    transforms: Sequence[object], *, clearance_m: float
) -> list[object]:
    """Mirror the advisor's initial vehicle-clearance admission pass.

    A corridor-filtered catalog count is not a realizable vehicle capacity:
    CARLA's native catalog can contain neighboring lane points closer than the
    advisor's configured vehicle clearance.  ``_spawn_vehicles`` walks the
    points in order for the initial population and rejects such neighbors, so
    use the same greedy rule before deciding whether route-derived fallback is
    required.
    """

    selected = []
    for candidate in transforms:
        if all(
            candidate.location.distance(existing.location) >= float(clearance_m)
            for existing in selected
        ):
            selected.append(candidate)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vehicle-spawn-clearance-m", type=float, required=True)
    parser.add_argument(
        "--route-progress-csv", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--protected-location",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="exclude vehicle spawn points close to an owned scenario actor",
    )
    parser.add_argument("--protected-clearance-m", type=float, default=12.0)
    parser.add_argument("--minimum-route-offset-m", type=float, default=0.0)
    parser.add_argument("--maximum-route-offset-m", type=float, required=True)
    parser.add_argument("--maximum-route-heading-error-deg", type=float, required=True)
    parser.add_argument("--minimum-filtered-spawn-points", type=int, required=True)
    parser.add_argument("--traffic-leading-distance-m", type=float, required=True)
    parser.add_argument("--traffic-speed-difference-pct", type=float, required=True)
    parser.add_argument("--traffic-desired-speed-mps", type=float, required=True)
    parser.add_argument(
        "--registered-spawn-maximum-horizontal-error-m", type=float, default=0.10
    )
    parser.add_argument(
        "--registered-spawn-maximum-yaw-error-deg", type=float, default=0.10
    )
    parser.add_argument(
        "--released-vehicle-motion-mode",
        choices=(
            "runner_owned_direct_route",
            "runner_owned_tm_autonomous",
            "runner_owned_deterministic_trace_replay",
            "runner_owned_stationary_context",
        ),
        default="runner_owned_direct_route",
    )
    parser.add_argument(
        "--released-walker-motion-mode",
        choices=("walker_ai_destination", "runner_owned_stationary"),
        default="walker_ai_destination",
    )
    parser.add_argument("--defer-vehicle-control-to-runner", action="store_true")
    parser.add_argument("--route-derived-spawn-fallback", action="store_true")
    parser.add_argument("--route-derived-spawn-spacing-m", type=float)
    parser.add_argument("--route-derived-cross-route-clearance-m", type=float)
    parser.add_argument("--population-ready-manifest", type=Path)
    parser.add_argument("--population-release-sentinel", type=Path)
    parser.add_argument("--population-released-manifest", type=Path)
    derived, remaining = parser.parse_known_args(sys.argv[1:])
    if derived.vehicle_spawn_clearance_m < 4.0:
        raise ValueError("derived vehicle spawn clearance must be at least 4 m")
    if derived.maximum_route_offset_m <= 0.0:
        raise ValueError("maximum route offset must be positive")
    if not 0.0 <= derived.minimum_route_offset_m < derived.maximum_route_offset_m:
        raise ValueError(
            "minimum route offset must be non-negative and below the maximum"
        )
    if not 0.0 < derived.maximum_route_heading_error_deg < 90.0:
        raise ValueError("maximum route heading error must be within (0, 90) degrees")
    if derived.minimum_filtered_spawn_points < 0:
        raise ValueError("minimum filtered spawn points must be non-negative")
    if derived.protected_clearance_m < 4.0:
        raise ValueError("protected-location clearance must be at least 4 m")
    if derived.traffic_leading_distance_m < 2.5:
        raise ValueError("traffic leading distance must be at least 2.5 m")
    if not 0.0 <= derived.traffic_speed_difference_pct <= 80.0:
        raise ValueError("traffic speed difference must be within 0-80 percent")
    if not 3.0 <= derived.traffic_desired_speed_mps <= 12.0:
        raise ValueError("traffic desired speed must be within 3-12 m/s")
    if not 0.0 <= derived.registered_spawn_maximum_horizontal_error_m <= 0.25:
        raise ValueError(
            "registered-spawn horizontal tolerance must be within [0, 0.25] m"
        )
    if not 0.0 <= derived.registered_spawn_maximum_yaw_error_deg <= 0.50:
        raise ValueError(
            "registered-spawn yaw tolerance must be within [0, 0.50] degrees"
        )
    if derived.route_derived_spawn_fallback:
        if (
            derived.route_derived_spawn_spacing_m is None
            or derived.route_derived_cross_route_clearance_m is None
        ):
            raise ValueError(
                "route-derived fallback requires same- and cross-route clearances"
            )
        if derived.route_derived_spawn_spacing_m < 4.0:
            raise ValueError("route-derived spawn spacing must be at least 4 m")
        if derived.route_derived_cross_route_clearance_m < 4.0:
            raise ValueError("route-derived cross-route clearance must be at least 4 m")
        if (
            derived.route_derived_cross_route_clearance_m
            > derived.route_derived_spawn_spacing_m
        ):
            raise ValueError(
                "cross-route clearance cannot exceed same-route spacing"
            )
    elif (
        derived.route_derived_spawn_spacing_m is not None
        or derived.route_derived_cross_route_clearance_m is not None
    ):
        raise ValueError("route-derived clearances require the fallback flag")
    handshake_values = (
        derived.population_ready_manifest,
        derived.population_release_sentinel,
        derived.population_released_manifest,
    )
    if any(value is not None for value in handshake_values) and not all(
        value is not None for value in handshake_values
    ):
        raise ValueError("all three population handshake paths must be supplied together")
    handshake_enabled = all(value is not None for value in handshake_values)
    if handshake_enabled and not derived.defer_vehicle_control_to_runner:
        raise ValueError("held population handshake requires deferred vehicle control")
    ready_path = (
        derived.population_ready_manifest.expanduser().resolve()
        if derived.population_ready_manifest is not None
        else None
    )
    release_path = (
        derived.population_release_sentinel.expanduser().resolve()
        if derived.population_release_sentinel is not None
        else None
    )
    released_path = (
        derived.population_released_manifest.expanduser().resolve()
        if derived.population_released_manifest is not None
        else None
    )

    route_paths = [path.expanduser().resolve() for path in derived.route_progress_csv]
    routes = [_route_xy(path) for path in route_paths]
    protected = [tuple(float(value) for value in pair) for pair in derived.protected_location]
    original_init = advisor.TrafficPopulationManager.__init__
    original_spawn_vehicles = advisor.TrafficPopulationManager._spawn_vehicles
    original_spawn_walker_bodies_once = (
        advisor.TrafficPopulationManager._spawn_walker_bodies_once
    )
    original_initialize_walker_controllers = (
        advisor.TrafficPopulationManager._initialize_walker_controllers
    )
    original_spawn_initial_population = (
        advisor.TrafficPopulationManager.spawn_initial_population
    )
    original_reconcile = advisor.TrafficPopulationManager.reconcile

    def filtered_init(
        population: object,
        client: object,
        world: object,
        traffic_manager: object,
        args: object,
        vehicle_blueprints: object,
        walker_blueprints: object,
        vehicle_spawn_points: object,
        synchronous_master: object,
    ) -> None:
        all_points = list(vehicle_spawn_points)
        eligible = []
        for transform in all_points:
            waypoint = world.get_map().get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=advisor.carla.LaneType.Driving,
            )
            if waypoint is None or bool(getattr(waypoint, "is_junction", False)):
                continue
            distance_m, heading_error_deg = min(
                (
                    _route_distance_and_heading_error_deg(transform, route_points)
                    for route_points in routes
                ),
                key=lambda result: (result[0], result[1]),
            )
            if (
                float(derived.minimum_route_offset_m)
                <= distance_m
                <= float(derived.maximum_route_offset_m)
                and heading_error_deg
                <= float(derived.maximum_route_heading_error_deg)
                and all(
                    math.hypot(
                        float(transform.location.x) - protected_x,
                        float(transform.location.y) - protected_y,
                    )
                    >= float(derived.protected_clearance_m)
                    for protected_x, protected_y in protected
                )
            ):
                eligible.append(transform)
        native_corridor_count = len(eligible)
        eligible = _pairwise_spaced_spawn_transforms(
            eligible, clearance_m=float(derived.vehicle_spawn_clearance_m)
        )
        spawn_source = "native_carla_catalog"
        native_eligible_count = len(eligible)
        if (
            len(eligible) < int(derived.minimum_filtered_spawn_points)
            and derived.route_derived_spawn_fallback
        ):
            eligible = _route_derived_spawn_transforms(
                world,
                routes,
                protected,
                protected_clearance_m=float(derived.protected_clearance_m),
                pairwise_clearance_m=float(
                    derived.route_derived_spawn_spacing_m
                ),
                cross_route_clearance_m=float(
                    derived.route_derived_cross_route_clearance_m
                ),
            )
            spawn_source = "reviewed_route_derived_fallback"
        if len(eligible) < int(derived.minimum_filtered_spawn_points):
            raise RuntimeError(
                "advisor route-corridor spawn filter has insufficient capacity: "
                f"eligible={len(eligible)}, "
                f"required={int(derived.minimum_filtered_spawn_points)}, "
                f"native_eligible={native_eligible_count}, source={spawn_source}"
            )
        advisor.logging.info(
            "Derived route-corridor spawn filter: source=%s eligible=%d "
            "native_eligible=%d native_corridor=%d catalog=%d offset=%.1f-%.1fm "
            "heading_error<=%.1fdeg actor_clearance>=%.1fm "
            "same_route_spacing>=%.1fm cross_route_clearance>=%.1fm "
            "protected_clearance>=%.1fm routes=%s",
            spawn_source,
            len(eligible),
            native_eligible_count,
            native_corridor_count,
            len(all_points),
            float(derived.minimum_route_offset_m),
            float(derived.maximum_route_offset_m),
            float(derived.maximum_route_heading_error_deg),
            float(derived.vehicle_spawn_clearance_m),
            float(
                derived.route_derived_spawn_spacing_m
                if derived.route_derived_spawn_spacing_m is not None
                else derived.vehicle_spawn_clearance_m
            ),
            float(
                derived.route_derived_cross_route_clearance_m
                if derived.route_derived_cross_route_clearance_m is not None
                else derived.vehicle_spawn_clearance_m
            ),
            float(derived.protected_clearance_m),
            [str(path) for path in route_paths],
        )
        original_init(
            population,
            client,
            world,
            traffic_manager,
            args,
            vehicle_blueprints,
            walker_blueprints,
            eligible,
            synchronous_master,
        )
        population._scenesense_released_vehicle_motion_mode = str(
            derived.released_vehicle_motion_mode
        )
        population._scenesense_released_vehicle_speed_mps = float(
            0.0
            if derived.released_vehicle_motion_mode
            == "runner_owned_stationary_context"
            else derived.traffic_desired_speed_mps
        )
        population._scenesense_released_walker_motion_mode = str(
            derived.released_walker_motion_mode
        )
        if handshake_enabled:
            # This process is a follower of the audit orchestrator's single
            # synchronous tick owner.  Harden only CARLA's known bare transient
            # registration failure; all descriptive clock failures still abort.
            population._wait_for_actor_update = population.world.wait_for_tick
        population._scenesense_population_released = not handshake_enabled
        population._scenesense_spawn_source = spawn_source
        population._scenesense_native_corridor_count = native_corridor_count
        population._scenesense_native_eligible_count = native_eligible_count
        population._scenesense_vehicle_candidate_count = len(eligible)
        population._scenesense_vehicle_spawn_clearance_m = float(
            derived.vehicle_spawn_clearance_m
        )

    def safe_spawn_vehicles(
        population: object, count: int, shuffle_spawn_points: bool = True
    ) -> List[int]:
        actor_ids = original_spawn_vehicles(population, count, shuffle_spawn_points)
        if handshake_enabled and actor_ids:
            # The advisor command chains SetAutopilot(True) to SpawnActor. Undo it
            # in the command domain before the first externally owned sync tick.
            population._apply_batch_sync(
                [
                    advisor.carla.command.SetAutopilot(
                        actor_id, False, population.tm_port
                    )
                    for actor_id in actor_ids
                ]
            )
            # Actor transforms may not be visible until the next externally
            # owned tick. Resolve them, then restore every vehicle to its exact
            # commanded route transform while physics is held. This prevents
            # suspension/contact resolution from becoming a matched-arm state
            # difference before READY.
            population._wait_for_actor_update()
        live = population._live_actor_map(actor_ids)
        for actor in live.values():
            population.traffic_manager.distance_to_leading_vehicle(
                actor, float(derived.traffic_leading_distance_m)
            )
            population.traffic_manager.vehicle_percentage_speed_difference(
                actor, float(derived.traffic_speed_difference_pct)
            )
            population.traffic_manager.set_desired_speed(
                actor, float(derived.traffic_desired_speed_mps) * 3.6
            )
            population.traffic_manager.auto_lane_change(actor, False)
            if derived.defer_vehicle_control_to_runner:
                actor.set_autopilot(False, population.tm_port)
            if handshake_enabled and not population._scenesense_population_released:
                _zero_and_hold(actor, tm_port=population.tm_port, vehicle=True)
        if handshake_enabled and actor_ids:
            if len(live) != len(actor_ids):
                raise RuntimeError("not every spawned vehicle became live at stabilization")
            assignments = _nearest_unique_spawn_assignments(
                list(live.values()), population.vehicle_spawn_points
            )
            _settle_and_hold_spawned_vehicles(
                population,
                assignments,
                maximum_horizontal_error_m=float(
                    derived.registered_spawn_maximum_horizontal_error_m
                ),
                maximum_yaw_error_deg=float(
                    derived.registered_spawn_maximum_yaw_error_deg
                ),
            )
        return actor_ids

    def held_spawn_walker_bodies_once(population: object, count: int) -> list[dict]:
        records = original_spawn_walker_bodies_once(population, count)
        if not handshake_enabled or not records:
            return records
        # Followers cannot request a tick in apply_batch_sync. Wait for the
        # orchestrator's next tick so every body is addressable, then freeze it
        # before a walker controller is ever started.
        population._wait_for_actor_update()
        live = population._live_actor_map([record.get("id") for record in records])
        for actor in live.values():
            _zero_and_hold(actor, tm_port=population.tm_port, vehicle=False)
        return records

    def prepare_held_walker_controllers(
        population: object, walker_records: Sequence[dict]
    ) -> int:
        if not handshake_enabled or population._scenesense_population_released:
            return original_initialize_walker_controllers(population, walker_records)
        candidates = [
            record
            for record in walker_records
            if record.get("con") is not None
            and not record.get("controller_ready", False)
        ]
        if not candidates:
            return 0
        actor_ids = []
        for record in candidates:
            actor_ids.extend((record["id"], record["con"]))
        live = population._live_actor_map(actor_ids)
        prepared = 0
        for record in candidates:
            body = live.get(record["id"])
            controller = live.get(record["con"])
            if not population._has_type(body, "walker.pedestrian."):
                continue
            if not population._has_type(controller, "controller.ai.walker"):
                population.orphan_controller_ids.add(record["con"])
                record["con"] = None
                record["controller_ready"] = False
                continue
            try:
                parent = controller.parent
            except (AttributeError, RuntimeError):
                parent = None
            if parent is not None and parent.id != record["id"]:
                population.orphan_controller_ids.add(record["con"])
                record["con"] = None
                record["controller_ready"] = False
                continue
            destination = population._random_navigation_location()
            if destination is None:
                continue
            _zero_and_hold(body, tm_port=population.tm_port, vehicle=False)
            record["_scenesense_release_destination"] = destination
            record["controller_ready"] = True
            prepared += 1
        return prepared

    def hold_all_owned(population: object) -> None:
        vehicles = population._live_actor_map(population.vehicle_ids)
        walkers = population._live_actor_map(
            [record.get("id") for record in population.walkers]
        )
        for actor in vehicles.values():
            _zero_and_hold(actor, tm_port=population.tm_port, vehicle=True)
        for actor in walkers.values():
            _zero_and_hold(actor, tm_port=population.tm_port, vehicle=False)

    def population_is_ready(population: object) -> bool:
        if len(population.vehicle_ids) != int(population.target_vehicle_count):
            return False
        if len(population.walkers) != int(population.target_walker_count):
            return False
        if any(
            record.get("id") is None
            or record.get("con") is None
            or not record.get("controller_ready", False)
            for record in population.walkers
        ):
            return False
        actor_ids = [*population.vehicle_ids]
        for record in population.walkers:
            actor_ids.extend((record["id"], record["con"]))
        return len(population._live_actor_map(actor_ids)) == len(actor_ids)

    def release_population(population: object) -> dict:
        vehicles = population._live_actor_map(population.vehicle_ids)
        walker_ids = [record.get("id") for record in population.walkers]
        controller_ids = [record.get("con") for record in population.walkers]
        live = population._live_actor_map([*walker_ids, *controller_ids])
        vehicle_motion_mode = str(
            population._scenesense_released_vehicle_motion_mode
        )
        runner_held_vehicles = vehicle_motion_mode in {
            "runner_owned_deterministic_trace_replay",
            "runner_owned_stationary_context",
        }
        stationary_walkers = (
            str(population._scenesense_released_walker_motion_mode)
            == "runner_owned_stationary"
        )
        for actor in vehicles.values():
            actor.set_simulate_physics(not runner_held_vehicles)
            actor.set_autopilot(False, population.tm_port)
            if runner_held_vehicles:
                actor.set_target_velocity(advisor.carla.Vector3D())
                actor.set_target_angular_velocity(advisor.carla.Vector3D())
            else:
                actor.apply_control(
                    advisor.carla.VehicleControl(
                        throttle=0.0, brake=0.0, hand_brake=False
                    )
                )
        started = 0
        for record in population.walkers:
            body = live.get(record.get("id"))
            controller = live.get(record.get("con"))
            destination = record.get("_scenesense_release_destination")
            if not population._has_type(body, "walker.pedestrian."):
                raise RuntimeError("held walker body disappeared before release")
            if not population._has_type(controller, "controller.ai.walker"):
                raise RuntimeError("held walker controller disappeared before release")
            if destination is None and not stationary_walkers:
                raise RuntimeError("held walker has no release destination")
            if stationary_walkers:
                _zero_and_hold(body, tm_port=population.tm_port, vehicle=False)
            else:
                body.set_simulate_physics(True)
                controller.start()
                controller.go_to_location(destination)
                controller.set_max_speed(float(record["speed"]))
                started += 1
        population._scenesense_population_released = True
        return {
            "schema": RELEASED_SCHEMA,
            "status": "released",
            "vehicle_count": len(vehicles),
            "walker_count": len(walker_ids),
            "walker_controller_count": len(controller_ids),
            "walker_controller_started_count": started,
            "vehicle_motion_mode": str(
                population._scenesense_released_vehicle_motion_mode
            ),
            "walker_motion_mode": str(
                population._scenesense_released_walker_motion_mode
            ),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }

    def held_spawn_initial_population(population: object) -> None:
        if not handshake_enabled:
            original_spawn_initial_population(population)
            return
        original_spawn_initial_population(population)
        stagnant_vehicle_reconciles = 0
        while not population_is_ready(population):
            vehicles_before = len(population.vehicle_ids)
            hold_all_owned(population)
            population.world.wait_for_tick()
            original_reconcile(population)
            ready_controllers = sum(
                bool(record.get("controller_ready", False))
                for record in population.walkers
            )
            vehicle_deficit = (
                len(population.vehicle_ids) < int(population.target_vehicle_count)
            )
            non_vehicle_population_ready = (
                len(population.walkers) == int(population.target_walker_count)
                and ready_controllers == int(population.target_walker_count)
            )
            if (
                vehicle_deficit
                and non_vehicle_population_ready
                and len(population.vehicle_ids) <= vehicles_before
            ):
                stagnant_vehicle_reconciles += 1
            else:
                stagnant_vehicle_reconciles = 0
            if stagnant_vehicle_reconciles >= HELD_VEHICLE_STAGNATION_LIMIT:
                raise RuntimeError(
                    "held ambient vehicle population made no progress: "
                    f"vehicles={len(population.vehicle_ids)}/"
                    f"{int(population.target_vehicle_count)} "
                    f"candidates={getattr(population, '_scenesense_vehicle_candidate_count', 'unknown')} "
                    f"source={getattr(population, '_scenesense_spawn_source', 'unknown')} "
                    f"native_corridor={getattr(population, '_scenesense_native_corridor_count', 'unknown')} "
                    f"native_eligible={getattr(population, '_scenesense_native_eligible_count', 'unknown')} "
                    f"clearance_m={getattr(population, '_scenesense_vehicle_spawn_clearance_m', 'unknown')} "
                    f"stagnant_reconciles={stagnant_vehicle_reconciles}"
                )
        hold_all_owned(population)
        held_vehicles = population._live_actor_map(population.vehicle_ids)
        junction_actor_ids = []
        for actor_id, actor in held_vehicles.items():
            waypoint = population.world.get_map().get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=advisor.carla.LaneType.Driving,
            )
            if waypoint is None or bool(getattr(waypoint, "is_junction", False)):
                junction_actor_ids.append(int(actor_id))
        if junction_actor_ids:
            raise RuntimeError(
                "held ambient vehicle spawn contract violated: actors in or "
                f"unresolved against a junction={sorted(junction_actor_ids)}"
            )
        signature, diagnostics = _held_population_rows(population)
        expected_actor_count = (
            int(population.target_vehicle_count) + int(population.target_walker_count)
        )
        if len(signature) != expected_actor_count:
            raise RuntimeError(
                "held population signature is incomplete: "
                f"observed={len(signature)} expected={expected_actor_count}"
            )
        _write_json_create(
            ready_path,
            {
                "schema": READY_SCHEMA,
                "status": "held_ready",
                "vehicle_count": int(population.target_vehicle_count),
                "walker_count": int(population.target_walker_count),
                "walker_controller_count": int(population.target_walker_count),
                "vehicle_spawn_contract": {
                    "all_outside_junctions": True,
                    "verified_vehicle_count": len(held_vehicles),
                },
                "spawn_signature_basis": (
                    "id_free_held_type_role_pose_and_motion_before_any_ambient_motion"
                ),
                "spawn_signature": signature,
                "held_pose_diagnostics": diagnostics,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        advisor.logging.info(
            "Held ambient population ready; waiting for orchestrator release: %s",
            release_path,
        )
        while not release_path.is_file():
            # The external orchestrator deliberately stops ticking after it sees
            # READY. Poll wall time here so RELEASE can be acknowledged without
            # advancing a variable number of simulation frames.
            time.sleep(0.01)
        with release_path.open("r", encoding="utf-8") as stream:
            release = json.load(stream)
        if release.get("schema") != RELEASE_SCHEMA:
            raise RuntimeError("population release sentinel schema mismatch")
        _write_json_create(released_path, release_population(population))
        advisor.logging.info("Ambient population released by orchestrator")

    advisor.VEHICLE_SPAWN_CLEARANCE_M = float(derived.vehicle_spawn_clearance_m)
    advisor.TrafficPopulationManager.__init__ = filtered_init
    advisor.TrafficPopulationManager._spawn_vehicles = safe_spawn_vehicles
    advisor.TrafficPopulationManager._spawn_walker_bodies_once = (
        held_spawn_walker_bodies_once
    )
    advisor.TrafficPopulationManager._initialize_walker_controllers = (
        prepare_held_walker_controllers
    )
    advisor.TrafficPopulationManager.spawn_initial_population = (
        held_spawn_initial_population
    )
    original_argv = list(sys.argv)
    original_client_factory = advisor.carla.Client
    sys.argv = [sys.argv[0], *remaining]
    if handshake_enabled:
        # The advisor reference calls ``world.wait_for_tick`` directly in its
        # maintenance loop, outside TrafficPopulationManager.  Return a world
        # proxy from its client so that setup and the entire released lifetime
        # share the same bounded transient handling without editing the
        # advisor-owned source.
        advisor.carla.Client = lambda *args, **kwargs: _RetryingClientProxy(
            original_client_factory(*args, **kwargs)
        )
    try:
        try:
            advisor.main()
        except KeyboardInterrupt:
            # This wrapper is normally stopped by its owning orchestrator. The
            # advisor main already ran ownership-scoped cleanup in ``finally``;
            # convert the expected SIGINT into a clean child-process exit.
            pass
        except BaseException as exc:
            if handshake_enabled and not ready_path.exists():
                failure_path = ready_path.with_name("population.failed.json")
                _write_json_create(
                    failure_path,
                    {
                        "schema": FAILED_SCHEMA,
                        "status": "failed_before_ready",
                        "error": f"{type(exc).__name__}: {exc}",
                        "written_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise
    finally:
        advisor.carla.Client = original_client_factory
        sys.argv = original_argv


if __name__ == "__main__":
    main()

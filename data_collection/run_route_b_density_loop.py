#!/usr/bin/env python3
"""Drive the accepted Route B loop under low / medium / dense NPC traffic.

Stage two of the Route B work. The route geometry, the ego, the completion
detector, the chase spectator, and the per-loop metrics all come unchanged from
``run_route_b_ego_loop``; the NPC vehicles, pedestrians, Traffic Manager setup,
population maintenance, ownership, and cleanup all come unchanged from the
advisor helper ``generate_traffic_v1``. This file only wires the two together
and adds the density profiles.

Order matters: the ego is spawned first so its Route B start pose is reserved,
then the NPC population is spawned around it.

The density profiles are environmental conditions only. Nothing here is a
UE-agent action, and no sensors, perception, OAI, or network profiles are
involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import sys
import time
import types
from pathlib import Path
import math
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ADVISOR_CODES = REPO_ROOT / "rl_agent" / "advisor_helper_scripts" / "codes"
CARLA_AGENTS_ROOT = REPO_ROOT.parents[1] / "carla"

for _path in (str(REPO_ROOT), str(CARLA_AGENTS_ROOT), str(ADVISOR_CODES)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import carla  # noqa: E402

import ego_route_config  # noqa: E402
import generate_traffic_v1 as traffic  # noqa: E402

from data_collection.run_route_b_ego_loop import (  # noqa: E402
    CollisionMailbox,
    RouteBError,
    TickPacer,
    chase_spectator,
    destroy_actor,
    location_of,
    planned_length_m,
    region_labels,
    transform_of,
    wrap_degrees,
)

DEFAULT_ROUTE = (
    REPO_ROOT
    / "data_collection"
    / "routes"
    / "town10hd_opt_route_b_full_map_loop_v1.json"
)
REQUIRED_MAP = "Town10HD_Opt"
CLEANUP_RPC_TIMEOUT_S = 2.0

DENSITY_PROFILES = {
    "low": {"vehicles": 5, "pedestrians": 5},
    "medium": {"vehicles": 10, "pedestrians": 10},
    "dense": {"vehicles": 20, "pedestrians": 20},
}


class DensityCollisionMailbox(CollisionMailbox):
    """Retain actor identity and CARLA simulation time for each callback."""

    def callback(self, event: Any) -> None:
        super().callback(event)
        row = self._rows[-1]
        try:
            row["other_actor_id"] = int(event.other_actor.id)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            row["simulation_time_s"] = round(float(event.timestamp), 3)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


def traffic_args(args: argparse.Namespace, vehicles: int, pedestrians: int) -> types.SimpleNamespace:
    """Build the argument object TrafficPopulationManager expects."""

    return types.SimpleNamespace(
        number_of_vehicles=int(vehicles),
        number_of_walkers=int(pedestrians),
        car_lights_on=False,
        hero=False,
        asynch=False,
        replenish_interval=float(args.replenish_interval_s),
        population_log_interval=float(args.population_log_interval_s),
    )


def spawn_ego(world: Any, route: dict, args: argparse.Namespace) -> tuple[Any, Any, CollisionMailbox]:
    """Spawn the ego and its collision sensor at the Route B start pose."""

    blueprint_library = world.get_blueprint_library()
    matches = [
        item for item in blueprint_library.filter(args.ego_blueprint)
        if str(item.id) == args.ego_blueprint
    ]
    if len(matches) != 1:
        raise RouteBError(
            f"expected exactly one blueprint {args.ego_blueprint!r}, found {len(matches)}"
        )
    ego_blueprint = matches[0]
    ego_blueprint.set_attribute("role_name", args.ego_role_name)

    start_transform = transform_of(route["start"])
    spawn_transform = carla.Transform(
        carla.Location(
            x=start_transform.location.x,
            y=start_transform.location.y,
            z=start_transform.location.z + float(args.spawn_z_offset_m),
        ),
        start_transform.rotation,
    )
    vehicle = world.try_spawn_actor(ego_blueprint, spawn_transform)
    if vehicle is None:
        raise RouteBError(
            f"could not spawn the ego at the Route B start {spawn_transform.location}"
        )

    collision_sensor = world.spawn_actor(
        blueprint_library.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=vehicle,
    )
    collisions = DensityCollisionMailbox()
    collision_sensor.listen(collisions.callback)
    return vehicle, collision_sensor, collisions


class RoadblockJanitor:
    """Remove NPC vehicles that have become permanent roadblocks.

    CARLA's Traffic Manager does not recover a vehicle that has been shunted
    out of its lane by a collision: it simply stops forever and everything
    behind it queues up, including the ego. Left alone that is a hard deadlock
    and it silently ruins any multi-loop collection. Anything stationary for
    longer than the timeout, while not held at a red light, is destroyed; the
    population manager's own reconcile() then replenishes back to target.
    """

    def __init__(
        self, population: Any, world: Any, timeout_s: float, clear_ahead_m: float,
        ego_id: int, allow_interventions: bool,
    ) -> None:
        self.population = population
        self.world = world
        self.timeout_s = float(timeout_s)
        self.clear_ahead_m = float(clear_ahead_m)
        # A collision can spin an NPC round so it faces the ego. Counting the
        # ego as "traffic ahead" would then mark that NPC as merely queuing and
        # it would never be cleared - the exact deadlock this class exists for.
        self.ego_id = int(ego_id)
        self.allow_interventions = bool(allow_interventions)
        self.relocation_points: list = []
        self._stopped_since: dict[int, float] = {}
        self._observed_ids: set[int] = set()
        self.removed_total = 0
        self.removed_ids: list[int] = []
        self.obstruction_events: list[dict[str, Any]] = []
        self.intervention_events: list[dict[str, Any]] = []

    def sweep(self, sim_now_s: float) -> list[int]:
        """Destroy long-stationary NPC vehicles; return the removed ids."""

        try:
            actors = self.world.get_actors(list(self.population.vehicle_ids))
        except RuntimeError:
            return []

        stuck: list[Any] = []
        live_ids = set()
        for actor in actors:
            try:
                if not actor.is_alive:
                    continue
                velocity = actor.get_velocity()
                speed = math.sqrt(
                    velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                )
            except RuntimeError:
                continue
            live_ids.add(int(actor.id))
            # Queued traffic is not a roadblock. Only the head of a jam has
            # is_at_traffic_light() True, so also treat "something close ahead"
            # as a legitimate reason to be stopped.
            if (
                speed > 0.3
                or held_at_red_light(actor)
                or vehicle_ahead(
                    self.world, actor, self.clear_ahead_m, {self.ego_id}
                ) is not None
            ):
                self._stopped_since.pop(int(actor.id), None)
                self._observed_ids.discard(int(actor.id))
                continue
            first = self._stopped_since.setdefault(int(actor.id), sim_now_s)
            if sim_now_s - first >= self.timeout_s:
                stuck.append(actor)

        for actor_id in list(self._stopped_since):
            if actor_id not in live_ids:
                self._stopped_since.pop(actor_id, None)
                self._observed_ids.discard(actor_id)

        removed: list[int] = []
        for actor in stuck:
            actor_id = int(actor.id)
            try:
                location = actor.get_location()
                where = f"({location.x:.1f}, {location.y:.1f})"
                location_row = {
                    "x": round(float(location.x), 2),
                    "y": round(float(location.y), 2),
                }
            except RuntimeError:
                where = "(unknown)"
                location_row = {}
            type_id = str(getattr(actor, "type_id", "?"))
            event = {
                "sim_s": round(float(sim_now_s), 2),
                "actor_id": actor_id,
                "actor_type": type_id,
                **location_row,
            }
            if not self.allow_interventions:
                if actor_id not in self._observed_ids:
                    self._observed_ids.add(actor_id)
                    self.obstruction_events.append(event)
                    logging.warning(
                        "roadblock observed: stuck NPC %d %s at %s "
                        "(stationary >= %.0f s); intervention disabled",
                        actor_id, type_id, where, self.timeout_s,
                    )
                continue
            outcome = clear_blocker(
                actor, self.population, self.relocation_points, self.removed_total
            )
            if outcome == "UNCLEARED":
                logging.warning(
                    "roadblock NOT cleared: NPC %d %s at %s resisted removal",
                    actor_id, type_id, where,
                )
                continue
            self._stopped_since.pop(actor_id, None)
            removed.append(actor_id)
            self.removed_ids.append(actor_id)
            self.intervention_events.append({"action": outcome, **event})
            logging.warning(
                "roadblock cleared (%s): stuck NPC %d %s at %s (stationary >= %.0f s)",
                outcome, actor_id, type_id, where, self.timeout_s,
            )
        self.removed_total += len(removed)
        return removed


def clear_blocker(
    actor: Any, population: Any, relocation_points: list, index: int
) -> str:
    """Get a wrecked NPC out of the lane, robustly.

    actor.destroy() is not reliable on these vehicles (CARLA raises a bare
    std::exception on some of them), and dropping ownership after a failed
    destroy leaves the wreck physically present but unowned, which is what
    turned a recoverable jam into a permanent deadlock. So: try relocation
    first, verify it actually moved, and only release ownership once the actor
    is confirmed gone.
    """
    actor_id = int(actor.id)
    if relocation_points:
        target = relocation_points[index % len(relocation_points)]
        try:
            actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            actor.set_transform(target)
            moved = actor.get_location().distance(target.location) < 5.0
            if moved:
                return "RELOCATED"
        except RuntimeError:
            pass
    try:
        actor.destroy()
    except RuntimeError:
        pass
    try:
        gone = not actor.is_alive
    except (AttributeError, RuntimeError):
        gone = True
    if gone:
        if actor_id in population.vehicle_ids:
            population.vehicle_ids.remove(actor_id)
        return "DESTROYED"
    # Still there: keep ownership so we retry rather than orphaning it.
    return "UNCLEARED"


def walker_ahead(
    agent: Any, world: Any, ego: Any, reach_m: float
) -> bool:
    """True when a pedestrian is in the ego's path within *reach_m*."""

    try:
        origin = ego.get_location()
        walkers = [
            w for w in world.get_actors().filter("*walker.pedestrian*")
            if w.get_location().distance(origin) < reach_m
        ]
        if not walkers:
            return False
        detected, _actor, _distance = agent._vehicle_obstacle_detected(
            walkers, reach_m
        )
        return bool(detected)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def collision_incidents(
    events: list[dict[str, Any]], gap_s: float = 1.0
) -> list[dict[str, Any]]:
    """Collapse sustained same-actor callbacks into distinct contact incidents."""

    incidents: list[dict[str, Any]] = []
    latest_by_actor: dict[tuple[Any, str], dict[str, Any]] = {}
    ordered = sorted(
        events,
        key=lambda row: (
            float(row.get("episode_simulation_time_s", float("inf"))),
            int(row.get("frame_id", 0)),
        ),
    )
    for event in ordered:
        actor_key = (
            event.get("other_actor_id"), str(event.get("other_actor_type", "unknown"))
        )
        event_time = float(event.get("episode_simulation_time_s", 0.0))
        current = latest_by_actor.get(actor_key)
        if current is None or event_time - float(current["last_episode_simulation_time_s"]) > gap_s:
            current = {
                "other_actor_id": event.get("other_actor_id"),
                "other_actor_type": str(event.get("other_actor_type", "unknown")),
                "first_frame_id": int(event.get("frame_id", 0)),
                "last_frame_id": int(event.get("frame_id", 0)),
                "first_simulation_time_s": event.get("simulation_time_s"),
                "last_simulation_time_s": event.get("simulation_time_s"),
                "first_episode_simulation_time_s": round(event_time, 3),
                "last_episode_simulation_time_s": round(event_time, 3),
                "callback_count": 1,
                "x": event.get("x"),
                "y": event.get("y"),
                "ego_speed_mps": event.get("ego_speed_mps"),
                "walker_braking_active": bool(event.get("walker_braking_active", False)),
            }
            incidents.append(current)
            latest_by_actor[actor_key] = current
            continue
        current["last_frame_id"] = int(event.get("frame_id", 0))
        current["last_simulation_time_s"] = event.get("simulation_time_s")
        current["last_episode_simulation_time_s"] = round(event_time, 3)
        current["callback_count"] = int(current["callback_count"]) + 1
        current["walker_braking_active"] = bool(
            current["walker_braking_active"]
            or event.get("walker_braking_active", False)
        )
    return incidents


def agent_blocker(agent: Any, world: Any, max_distance_m: float) -> Any:
    """Ask BasicAgent what it is actually braking for.

    A straight-line corridor probe misses the vehicle the agent can see around
    a bend or across a junction, which is exactly where the ego gets stuck. The
    agent's own bounding-box detector is what produces the emergency brake, so
    it is the correct source of truth for "what is blocking me".
    """
    try:
        vehicles = world.get_actors().filter("*vehicle*")
        detected, actor, _distance = agent._vehicle_obstacle_detected(
            vehicles, float(max_distance_m)
        )
        return actor if detected else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def describe_nearby(world: Any, ego: Any, radius_m: float) -> str:
    """List vehicles and walkers near the ego, for deadlock diagnostics."""

    try:
        origin = ego.get_location()
    except RuntimeError:
        return "unavailable"
    items: list[str] = []
    for pattern in ("vehicle.*", "walker.pedestrian.*"):
        for actor in world.get_actors().filter(pattern):
            if int(actor.id) == int(ego.id):
                continue
            try:
                location = actor.get_location()
            except RuntimeError:
                continue
            distance = location.distance(origin)
            if distance <= radius_m:
                items.append(
                    f"{actor.type_id}#{actor.id}@{distance:.1f}m"
                )
    return ", ".join(sorted(items)[:8]) or "nothing within %.0f m" % radius_m


def held_at_red_light(actor: Any) -> bool:
    """True while the actor is stopped at a red light (a legitimate wait)."""

    try:
        if not actor.is_at_traffic_light():
            return False
        light = actor.get_traffic_light()
        return light is not None and light.get_state() == carla.TrafficLightState.Red
    except (AttributeError, RuntimeError):
        return False


def vehicle_ahead(
    world: Any, ego: Any, reach_m: float, ignore_ids: set[int] | None = None
) -> Any:
    """Return the closest vehicle roughly ahead of *ego* within *reach_m*."""

    try:
        ego_transform = ego.get_transform()
    except RuntimeError:
        return None
    forward = ego_transform.get_forward_vector()
    origin = ego_transform.location
    best = None
    best_distance = reach_m
    for actor in world.get_actors().filter("vehicle.*"):
        if int(actor.id) == int(ego.id):
            continue
        if ignore_ids and int(actor.id) in ignore_ids:
            continue
        try:
            location = actor.get_location()
        except RuntimeError:
            continue
        dx = float(location.x) - float(origin.x)
        dy = float(location.y) - float(origin.y)
        longitudinal = dx * float(forward.x) + dy * float(forward.y)
        lateral = abs(dx * -float(forward.y) + dy * float(forward.x))
        if longitudinal <= 0.0 or lateral > 2.5:
            continue
        distance = math.hypot(dx, dy)
        if distance < best_distance:
            best, best_distance = actor, distance
    return best


def overtake_direction(world_map: Any, ego: Any) -> str | None:
    """Pick an adjacent same-direction driving lane to step into, if one exists."""

    try:
        waypoint = world_map.get_waypoint(
            ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except RuntimeError:
        return None
    if waypoint is None:
        return None
    for name, neighbour in (
        ("left", waypoint.get_left_lane()),
        ("right", waypoint.get_right_lane()),
    ):
        if neighbour is None:
            continue
        try:
            if neighbour.lane_type != carla.LaneType.Driving:
                continue
            # Only a same-direction neighbour is a legal overtake lane.
            if (neighbour.lane_id < 0) != (waypoint.lane_id < 0):
                continue
        except (AttributeError, RuntimeError):
            continue
        return name
    return None


def drive_one_loop_with_traffic(
    world: Any,
    vehicle: Any,
    agent: Any,
    route: dict,
    collisions: CollisionMailbox,
    args: argparse.Namespace,
    loop_index: int,
    maintain: Any,
    janitor: RoadblockJanitor,
) -> dict[str, Any]:
    """Route B loop hardened against NPC roadblocks and ego deadlock."""

    start_transform = transform_of(route["start"])
    vias = [location_of(item) for item in route["intermediate_waypoints"]]
    labels = region_labels(route, len(vias))
    legs = list(vias) + [start_transform.location]
    world_map = world.get_map()

    fixed_delta_s = float(args.fixed_delta_seconds)
    pacer = TickPacer(float(args.real_time_tick_period_s))

    collisions_at_start = collisions.count()
    collision_rows_at_start = len(collisions.rows)
    roadblocks_at_start = janitor.removed_total
    janitor_obstructions_at_start = len(janitor.obstruction_events)
    janitor_interventions_at_start = len(janitor.intervention_events)
    wall_start_s = time.monotonic()
    sim_elapsed_s = 0.0
    driven_m = 0.0
    ticks = 0
    reached: list[dict[str, Any]] = []

    blocked_since_s: float | None = None
    last_progress_s = 0.0
    last_progress_m = 0.0
    overtakes = 0
    replans = 0
    walker_brake_ticks = 0
    overtake_attempts: list[dict[str, Any]] = []
    janitor_last_sweep_s = 0.0
    tick_telemetry: dict[int, dict[str, Any]] = {}
    abort_reason = ""
    watchdog_aborted = False

    previous = vehicle.get_transform().location
    leg_index = 0
    agent.set_destination(legs[0])

    maximum_sim_s = float(args.maximum_loop_sim_s)
    while leg_index < len(legs):
        if sim_elapsed_s > maximum_sim_s:
            abort_reason = (
                f"loop {loop_index} exceeded the {maximum_sim_s:.0f} s simulated budget "
                f"at leg {leg_index + 1}/{len(legs)}"
            )
            break

        control = agent.run_step()
        control.manual_gear_shift = False
        walker_braking_active = bool(args.brake_for_walkers and walker_ahead(
            agent, world, vehicle, float(args.walker_brake_distance_m)
        ))
        if walker_braking_active:
            # BasicAgent never looks at pedestrians - "walker" does not appear
            # anywhere in it - so without this the ego drives straight through
            # them. This is the same detector BehaviorAgent's
            # pedestrian_avoid_manager uses, applied to the walker list.
            control = agent.add_emergency_stop(control)
            walker_brake_ticks += 1
        vehicle.apply_control(control)
        frame_id = int(world.tick())
        ticks += 1
        sim_elapsed_s += fixed_delta_s

        transform = vehicle.get_transform()
        current = transform.location
        driven_m += math.hypot(
            float(current.x) - float(previous.x),
            float(current.y) - float(previous.y),
        )
        previous = current

        maintain()
        if sim_elapsed_s - janitor_last_sweep_s >= float(args.janitor_interval_s):
            janitor_last_sweep_s = sim_elapsed_s
            janitor.sweep(sim_elapsed_s)

        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        tick_telemetry[frame_id] = {
            "episode_simulation_time_s": round(sim_elapsed_s, 3),
            "ego_speed_mps": round(speed_mps, 3),
            "walker_braking_active": walker_braking_active,
        }

        # Progress is measured by distance covered, not by brake state: a queue
        # behind a dead NPC brakes legitimately and forever.
        if driven_m - last_progress_m > 1.0 or held_at_red_light(vehicle):
            last_progress_m = driven_m
            last_progress_s = sim_elapsed_s

        # Waiting at a red light is a legitimate stop, not a roadblock, and
        # must never trigger an overtake.
        if speed_mps < 0.3 and not held_at_red_light(vehicle):
            if blocked_since_s is None:
                blocked_since_s = sim_elapsed_s
        else:
            blocked_since_s = None

        blocked_for_s = (
            0.0 if blocked_since_s is None else sim_elapsed_s - blocked_since_s
        )
        blocker = (
            (
                agent_blocker(agent, world, float(args.blocker_reach_m))
                or vehicle_ahead(world, vehicle, float(args.blocker_reach_m))
            )
            if blocked_for_s >= float(args.ego_block_timeout_s)
            else None
        )
        # Stopped with a clear road ahead usually means a stale plan; a fresh
        # one from the current pose is cheap and often unwedges it.
        if (
            blocked_for_s >= float(args.ego_block_timeout_s)
            and blocker is None
            and replans < int(args.maximum_replans)
        ):
            replans += 1
            agent.set_destination(legs[leg_index])
            blocked_since_s = sim_elapsed_s
            logging.warning(
                "ego stopped %.1f s with clear road at leg %d -> REPLAN",
                blocked_for_s, leg_index,
            )

        # Only act on something we can actually see blocking the lane.
        if blocker is not None:
            blocker_id = int(blocker.id)
            blocker_type = str(getattr(blocker, "type_id", ""))
            record = {
                "sim_s": round(sim_elapsed_s, 2),
                "blocked_for_s": round(blocked_for_s, 2),
                "blocker_type": blocker_type,
                "blocker_id": blocker_id,
                "direction": "",
                "leg_index": leg_index,
            }
            try:
                blocker_location = blocker.get_location()
                record["x"] = round(float(blocker_location.x), 2)
                record["y"] = round(float(blocker_location.y), 2)
            except RuntimeError:
                pass
            try:
                blocker_velocity = blocker.get_velocity()
                blocker_speed = math.sqrt(
                    blocker_velocity.x ** 2 + blocker_velocity.y ** 2
                    + blocker_velocity.z ** 2
                )
            except RuntimeError:
                blocker_speed = 0.0
            direction = overtake_direction(world_map, vehicle)
            record["direction"] = direction or ""
            if not args.allow_scenario_interventions:
                record["action"] = "OBSERVED_NO_INTERVENTION"
            elif blocker_speed < 0.3:
                # Deterministic unblock. Ownership deliberately does not gate
                # this: a wreck that resisted an earlier destroy is exactly the
                # thing most likely to be sitting here.
                outcome = clear_blocker(
                    blocker, janitor.population, janitor.relocation_points,
                    janitor.removed_total,
                )
                record["action"] = f"BLOCKER_{outcome}"
                if outcome != "UNCLEARED":
                    janitor.removed_total += 1
                    janitor.intervention_events.append({
                        "action": outcome,
                        "sim_s": round(sim_elapsed_s, 2),
                        "actor_id": blocker_id,
                        "actor_type": blocker_type,
                        **{key: record[key] for key in ("x", "y") if key in record},
                    })
            elif direction is not None and overtakes < int(args.maximum_overtakes):
                try:
                    agent.lane_change(
                        direction,
                        same_lane_time=0.0,
                        other_lane_time=float(args.overtake_other_lane_s),
                        lane_change_time=2.0,
                    )
                    record["action"] = "LANE_CHANGE"
                    overtakes += 1
                    janitor.intervention_events.append({
                        "action": "FORCED_OVERTAKE",
                        "sim_s": round(sim_elapsed_s, 2),
                        "actor_id": blocker_id,
                        "actor_type": blocker_type,
                        **{key: record[key] for key in ("x", "y") if key in record},
                    })
                except (RuntimeError, ValueError, IndexError) as exc:
                    record["action"] = f"LANE_CHANGE_FAILED: {exc}"
            else:
                record["action"] = "NO_ADJACENT_LANE"
            overtake_attempts.append(record)
            logging.warning(
                "ego blocked %.1f s at leg %d by %s -> %s",
                blocked_for_s, leg_index, record["blocker_type"] or "unknown",
                record["action"],
            )
            blocked_since_s = sim_elapsed_s  # re-arm before judging again

        if sim_elapsed_s - last_progress_s > float(args.no_progress_timeout_s):
            blocker = vehicle_ahead(world, vehicle, float(args.blocker_reach_m))
            nearby = describe_nearby(world, vehicle, 20.0)
            abort_reason = (
                f"loop {loop_index} made no route progress for "
                f"{args.no_progress_timeout_s:.0f} s at leg {leg_index + 1}/{len(legs)} "
                f"near ({current.x:.1f}, {current.y:.1f}); "
                f"blocker={getattr(blocker, 'type_id', 'none')} "
                f"speed={speed_mps:.2f} throttle={control.throttle:.2f} "
                f"brake={control.brake:.2f} at_light={vehicle.is_at_traffic_light()} "
                f"overtakes_attempted={len(overtake_attempts)} "
                f"npc_roadblocks_cleared={janitor.removed_total}; nearby=[{nearby}]"
            )
            watchdog_aborted = True
            break

        if args.spectator:
            chase_spectator(
                world, vehicle, args.spectator_behind_m,
                args.spectator_above_m, args.spectator_pitch_deg,
            )

        if agent.done():
            target = legs[leg_index]
            arrival_error_m = math.hypot(
                float(current.x) - float(target.x),
                float(current.y) - float(target.y),
            )
            if arrival_error_m > float(args.leg_arrival_radius_m):
                # A lane-change manoeuvre replaces the agent's plan with a short
                # path; when that finishes the agent reports done() nowhere near
                # the via. Re-issue the leg instead of banking a false arrival.
                replans += 1
                agent.set_destination(target)
                pacer.wait()
                continue
            reached.append({
                "leg_index": leg_index,
                "region": labels[leg_index] if leg_index < len(labels) else "start",
                "arrival_error_m": arrival_error_m,
                "sim_s": sim_elapsed_s,
            })
            leg_index += 1
            if leg_index < len(legs):
                agent.set_destination(legs[leg_index])

        pacer.wait()

    for _ in range(int(round(1.0 / fixed_delta_s))):
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        )
        frame_id = int(world.tick())
        ticks += 1
        sim_elapsed_s += fixed_delta_s
        try:
            velocity = vehicle.get_velocity()
            speed_mps = math.sqrt(
                velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
            )
        except RuntimeError:
            speed_mps = 0.0
        tick_telemetry[frame_id] = {
            "episode_simulation_time_s": round(sim_elapsed_s, 3),
            "ego_speed_mps": round(speed_mps, 3),
            "walker_braking_active": False,
        }
        if args.spectator:
            chase_spectator(
                world, vehicle, args.spectator_behind_m,
                args.spectator_above_m, args.spectator_pitch_deg,
            )
        pacer.wait()

    final = vehicle.get_transform()
    return_position_error_m = math.hypot(
        float(final.location.x) - float(start_transform.location.x),
        float(final.location.y) - float(start_transform.location.y),
    )
    return_heading_error_deg = abs(
        wrap_degrees(
            float(final.rotation.yaw) - float(start_transform.rotation.yaw)
        )
    )
    all_reached = len(reached) == len(legs)
    completed = bool(
        all_reached
        and not abort_reason
        and return_position_error_m <= float(args.completion_radius_m)
        and return_heading_error_deg <= float(args.completion_heading_tolerance_deg)
    )
    raw_collision_events = collisions.rows[collision_rows_at_start:]
    for event in raw_collision_events:
        telemetry = tick_telemetry.get(int(event.get("frame_id", -1)))
        if telemetry is not None:
            event.update(telemetry)
    incidents = collision_incidents(raw_collision_events)
    intervention_events = janitor.intervention_events[janitor_interventions_at_start:]
    return {
        "loop_index": loop_index,
        "completed": completed,
        "all_ordered_waypoints_reached": all_reached,
        "waypoints_reached": len(reached),
        "waypoints_expected": len(legs),
        "regions_covered": ",".join(sorted(
            {row["region"] for row in reached
             if row["region"] not in ("start", "unlabelled")}
        )),
        "driven_distance_m": round(driven_m, 2),
        "simulation_duration_s": round(sim_elapsed_s, 2),
        "wall_clock_duration_s": round(time.monotonic() - wall_start_s, 2),
        "ticks": ticks,
        "return_position_error_m": round(return_position_error_m, 3),
        "return_heading_error_deg": round(return_heading_error_deg, 3),
        "collision_count": len(raw_collision_events),
        "collision_events": raw_collision_events,
        "collision_incident_count": len(incidents),
        "collision_incidents": incidents,
        "ego_overtakes": overtakes,
        "ego_block_events": len(overtake_attempts),
        "overtake_attempts": overtake_attempts,
        "npc_roadblocks_cleared": janitor.removed_total - roadblocks_at_start,
        "roadblock_relocations": sum(
            1 for event in intervention_events if event.get("action") == "RELOCATED"
        ),
        "roadblock_destructions": sum(
            1 for event in intervention_events if event.get("action") == "DESTROYED"
        ),
        "intervention_count": len(intervention_events),
        "intervention_events": intervention_events,
        "roadblock_observations": janitor.obstruction_events[
            janitor_obstructions_at_start:
        ],
        "ego_replans": replans,
        "walker_brake_ticks": walker_brake_ticks,
        "watchdog_aborted": watchdog_aborted,
        "abort_reason": abort_reason,
    }


def qualification_status(rows: list[dict[str, Any]], cleanup_ok: bool) -> str:
    """Return the bounded pedestrian-braking qualification terminal."""

    if len(rows) != 1 or not cleanup_ok:
        return "FAIL"
    row = rows[0]
    if int(row.get("intervention_count", 0)) > 0:
        return "INTERVENED"
    if (
        not bool(row.get("completed"))
        or bool(row.get("watchdog_aborted"))
        or int(row.get("collision_incident_count", 0)) > 0
    ):
        return "FAIL"
    if int(row.get("walker_brake_ticks", 0)) == 0:
        return "PEDESTRIAN_BRAKING_NOT_EXERCISED"
    return "PASS"


def run(args: argparse.Namespace) -> int:
    profile = DENSITY_PROFILES[args.density]
    vehicles = profile["vehicles"] if args.vehicles is None else int(args.vehicles)
    pedestrians = (
        profile["pedestrians"] if args.pedestrians is None else int(args.pedestrians)
    )

    route_path = args.route_config.resolve()
    if not route_path.is_file():
        raise RouteBError(f"Route B config not found: {route_path}")
    route = ego_route_config.load_route_config(route_path)
    if not ego_route_config.maps_match(route["map"], REQUIRED_MAP):
        raise RouteBError(f"route targets {route['map']!r}, expected {REQUIRED_MAP}")

    output_paths = (args.out_csv.resolve(), args.summary_json.resolve())
    if output_paths[0] == output_paths[1]:
        raise RouteBError("--out-csv and --summary-json must be different paths")
    for output_path in output_paths:
        if output_path.exists():
            raise RouteBError(
                f"refusing to append to or overwrite existing validation output: "
                f"{output_path}"
            )

    random.seed(int(args.seed))

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.load_world(REQUIRED_MAP, True)
    observed_map = str(world.get_map().name)
    if not ego_route_config.maps_match(observed_map, REQUIRED_MAP):
        raise RouteBError(
            f"fresh-world reload resolved to {observed_map!r}, expected {REQUIRED_MAP}"
        )
    print(
        f"fresh-world reload complete: requested={REQUIRED_MAP} observed={observed_map}",
        flush=True,
    )

    original_settings = world.get_settings()
    traffic_manager = None
    population = None
    vehicle = None
    collision_sensor = None
    rows: list[dict[str, Any]] = []
    status = "FAIL"
    cleanup_ok = False

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta_seconds)
        world.apply_settings(settings)

        # Ego first: this reserves the Route B start pose before any NPC
        # vehicle can take the spawn point next to it.
        vehicle, collision_sensor, collisions = spawn_ego(world, route, args)
        for _ in range(int(args.warmup_ticks)):
            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            )
            world.tick()

        traffic_manager = client.get_trafficmanager(int(args.tm_port))
        # A larger gap and a slower fleet are the two settings that most reduce
        # NPC-on-NPC shunts, which are what leave permanent roadblocks behind.
        traffic_manager.set_global_distance_to_leading_vehicle(
            float(args.tm_leading_distance_m)
        )
        traffic_manager.set_random_device_seed(int(args.seed))
        traffic_manager.set_synchronous_mode(True)
        if args.hybrid_physics:
            # Stock generate_traffic's --hybrid. NPCs beyond the radius of the
            # hero (our ego) run simplified physics and cannot collide, so the
            # far-field pile-ups that later become roadblocks never happen.
            # Vehicles near the ego keep full physics, so what the route
            # actually drives through is unchanged.
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(
                float(args.hybrid_physics_radius_m)
            )
        if args.respawn_dormant:
            traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.global_percentage_speed_difference(
            float(args.tm_speed_difference_pct)
        )

        world.set_pedestrians_seed(int(args.seed))
        world.set_pedestrians_cross_factor(traffic.PERCENTAGE_PEDESTRIANS_CROSSING)

        vehicle_blueprints = traffic.get_actor_blueprints(world, "vehicle.*", "All")
        walker_blueprints = traffic.get_actor_blueprints(
            world, "walker.pedestrian.*", "All"
        )
        if not vehicle_blueprints or not walker_blueprints:
            raise RouteBError("no vehicle or pedestrian blueprints available")
        if args.safe_vehicles:
            # Same rule as generate_traffic_v1's --safe. Trucks, vans, buses and
            # two-wheelers are the ones that get shunted across lanes and become
            # permanent roadblocks, so keep the NPC fleet to ordinary cars.
            cars = [
                bp for bp in vehicle_blueprints
                if bp.has_attribute("base_type")
                and bp.get_attribute("base_type").as_str() == "car"
            ]
            if cars:
                excluded = len(vehicle_blueprints) - len(cars)
                vehicle_blueprints = cars
                print(
                    f"safe-vehicle filter: {len(cars)} car blueprints kept, "
                    f"{excluded} non-car blueprints excluded",
                    flush=True,
                )
        vehicle_blueprints = sorted(vehicle_blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        population_args = traffic_args(args, vehicles, pedestrians)
        # This client owns the world tick, so the population manager is the
        # synchronous master for its own spawn batches.
        population = traffic.TrafficPopulationManager(
            client,
            world,
            traffic_manager,
            population_args,
            vehicle_blueprints,
            walker_blueprints,
            spawn_points,
            True,
        )
        def harden_npcs() -> None:
            """Apply per-vehicle TM settings to every owned NPC.

            TM's automatic lane changing is the single biggest source of
            NPC-on-NPC side-swipes here, and a shunted vehicle never recovers,
            so it is disabled. Re-applied after replenishment because new
            vehicles come back with TM defaults.
            """
            try:
                actors = world.get_actors(list(population.vehicle_ids))
            except RuntimeError:
                return
            for actor in actors:
                try:
                    traffic_manager.auto_lane_change(actor, False)
                    traffic_manager.random_left_lanechange_percentage(actor, 0)
                    traffic_manager.random_right_lanechange_percentage(actor, 0)
                    traffic_manager.distance_to_leading_vehicle(
                        actor, float(args.tm_leading_distance_m)
                    )
                    traffic_manager.ignore_lights_percentage(actor, 0.0)
                    traffic_manager.ignore_signs_percentage(actor, 0.0)
                    traffic_manager.ignore_walkers_percentage(actor, 0.0)
                except (AttributeError, RuntimeError):
                    continue

        population.spawn_initial_population()
        if args.harden_npcs:
            harden_npcs()

        print(
            f"density={args.density} requested vehicles={vehicles} pedestrians={pedestrians} "
            f"spawned vehicles={len(population.vehicle_ids)} walkers={len(population.walkers)} "
            f"seed={args.seed} lane_offset_m={args.lane_offset_m}",
            flush=True,
        )

        next_reconcile_at = time.monotonic() + float(args.replenish_interval_s)

        def maintain_population() -> None:
            nonlocal next_reconcile_at
            now = time.monotonic()
            if now < next_reconcile_at:
                return
            next_reconcile_at = now + float(args.replenish_interval_s)
            try:
                before = set(population.vehicle_ids)
                population.reconcile()
                if args.harden_npcs and set(population.vehicle_ids) != before:
                    harden_npcs()
            except RuntimeError as error:
                logging.error("population maintenance failed: %s", error)

        from agents.navigation.basic_agent import BasicAgent  # noqa: E402

        def new_agent() -> Any:
            agent = BasicAgent(
                vehicle,
                target_speed=float(args.target_speed_kph),
                opt_dict={
                    "sampling_resolution": float(route["route_sampling_resolution_m"]),
                    "base_tlight_threshold": 5.0,
                    "base_vehicle_threshold": float(args.vehicle_threshold_m),
                    # BasicAgent's default point check misses large NPC bodies
                    # (a carlacola truck produced a 29 s sustained contact at
                    # medium density); bounding-box detection handles them.
                    "use_bbs_detection": bool(args.use_bbs_detection),
                    "max_brake": 1.0,
                    "offset": float(args.lane_offset_m),
                },
            )
            agent.follow_speed_limits(False)
            agent.set_target_speed(float(args.target_speed_kph))
            return agent

        length_m = planned_length_m(route)
        print(
            f"route_id={route['name']!r} map={observed_map} "
            f"planned_route_length_m={length_m:.1f} loops={args.loops}",
            flush=True,
        )

        janitor = RoadblockJanitor(
            population, world, float(args.npc_stuck_timeout_s),
            float(args.npc_clear_ahead_m), int(vehicle.id),
            bool(args.allow_scenario_interventions),
        )
        janitor.relocation_points = list(spawn_points)

        for loop_index in range(1, int(args.loops) + 1):
            result = drive_one_loop_with_traffic(
                world, vehicle, new_agent(), route, collisions, args, loop_index,
                maintain_population, janitor,
            )
            result["route_id"] = str(route["name"])
            result["planned_route_length_m"] = round(length_m, 2)
            result["density"] = args.density
            result["npc_vehicles_requested"] = vehicles
            result["npc_pedestrians_requested"] = pedestrians
            result["npc_vehicles_live"] = len(population.vehicle_ids)
            result["npc_pedestrians_live"] = len(population.walkers)
            result["seed"] = int(args.seed)
            result["lane_offset_m"] = float(args.lane_offset_m)
            rows.append(result)
            print(
                f"loop {loop_index}/{args.loops} [{args.density}]: "
                f"completed={result['completed']} "
                f"driven_m={result['driven_distance_m']:.1f} "
                f"sim_s={result['simulation_duration_s']:.1f} "
                f"wall_s={result['wall_clock_duration_s']:.1f} "
                f"return_pos_err_m={result['return_position_error_m']:.2f} "
                f"return_yaw_err_deg={result['return_heading_error_deg']:.2f} "
                f"collisions={result['collision_count']} "
                f"npc_v={result['npc_vehicles_live']} npc_p={result['npc_pedestrians_live']} "
                f"blocks={result['ego_block_events']} overtakes={result['ego_overtakes']} "
                f"roadblocks_cleared={result['npc_roadblocks_cleared']} "
                f"walker_brake_ticks={result['walker_brake_ticks']} "
                f"collision_incidents={result['collision_incident_count']} "
                f"watchdog_aborted={result['watchdog_aborted']}",
                flush=True,
            )
    finally:
        cleanup_rpc_available = True
        try:
            client.set_timeout(CLEANUP_RPC_TIMEOUT_S)
            client.get_server_version()
            client.set_timeout(args.timeout)
        except RuntimeError as error:
            cleanup_rpc_available = False
            logging.warning(
                "CARLA server unavailable during cleanup; skipping remaining RPCs: %s",
                error,
            )
        if cleanup_rpc_available:
            if population is not None:
                try:
                    population.destroy()
                except Exception as error:  # noqa: BLE001 - cleanup must not mask
                    logging.warning("NPC cleanup error: %s", error)
            for actor in (collision_sensor, vehicle):
                destroy_actor(actor)
            try:
                world.tick()
            except RuntimeError:
                pass
            cleanup_ok = verify_cleanup(world, vehicle, collision_sensor, population)
            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                    traffic_manager.set_hybrid_physics_mode(False)
                    if args.respawn_dormant:
                        traffic_manager.set_respawn_dormant_vehicles(False)
                except RuntimeError as error:
                    logging.warning("could not restore Traffic Manager modes: %s", error)
                    cleanup_ok = False
            try:
                world.apply_settings(original_settings)
            except RuntimeError:
                cleanup_ok = False
        else:
            cleanup_ok = False
        status = qualification_status(rows, cleanup_ok)
        write_outputs(args, rows, status, cleanup_ok)
    return 0 if status == "PASS" else 1


def verify_cleanup(
    world: Any, vehicle: Any, collision_sensor: Any, population: Any
) -> bool:
    """Confirm every actor this runner owned is gone."""

    owned: list[int] = []
    for actor in (vehicle, collision_sensor):
        if actor is not None:
            owned.append(int(actor.id))
    if population is not None:
        owned.extend(int(value) for value in population.vehicle_ids)
        for record in population.walkers:
            for key in ("id", "con"):
                if record.get(key) is not None:
                    owned.append(int(record[key]))
        owned.extend(int(value) for value in population.orphan_controller_ids)

    remaining: list[int] = []
    for actor_id in dict.fromkeys(owned):
        try:
            actor = world.get_actor(actor_id)
        except RuntimeError:
            continue
        if actor is None:
            continue
        try:
            if bool(actor.is_alive):
                remaining.append(actor_id)
        except (AttributeError, RuntimeError):
            remaining.append(actor_id)
    if remaining:
        print(
            f"warning: {len(remaining)} owned actors still alive after cleanup: "
            f"{remaining[:10]}",
            file=sys.stderr,
            flush=True,
        )
    return not remaining


DENSITY_FIELDS = [
    "route_id", "density", "loop_index", "status", "cleanup_succeeded", "completed",
    "all_ordered_waypoints_reached", "waypoints_reached", "waypoints_expected",
    "regions_covered", "npc_vehicles_requested", "npc_pedestrians_requested",
    "npc_vehicles_live", "npc_pedestrians_live", "seed", "lane_offset_m",
    "planned_route_length_m", "driven_distance_m", "simulation_duration_s",
    "wall_clock_duration_s", "ticks", "return_position_error_m",
    "return_heading_error_deg", "collision_count", "collision_incident_count",
    "ego_block_events", "ego_overtakes", "ego_replans",
    "npc_roadblocks_cleared", "roadblock_relocations", "roadblock_destructions",
    "intervention_count", "walker_brake_ticks", "watchdog_aborted", "abort_reason",
]


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    status: str,
    cleanup_ok: bool,
) -> None:
    """One CSV row per loop plus a small summary, including the NPC columns."""

    if not rows:
        print(
            f"no completed loops recorded; status={status} cleanup_succeeded={cleanup_ok}",
            file=sys.stderr, flush=True,
        )
        return

    for row in rows:
        row["status"] = status
        row["cleanup_succeeded"] = bool(cleanup_ok)

    csv_path = args.out_csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DENSITY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DENSITY_FIELDS})

    durations = [float(row["simulation_duration_s"]) for row in rows]
    summary = {
        "route_id": str(rows[0]["route_id"]),
        "density": args.density,
        "npc_vehicles_requested": rows[0]["npc_vehicles_requested"],
        "npc_pedestrians_requested": rows[0]["npc_pedestrians_requested"],
        "seed": int(args.seed),
        "lane_offset_m": float(args.lane_offset_m),
        "loops": len(rows),
        "loops_completed": sum(1 for row in rows if row["completed"]),
        "status": status,
        "cleanup_succeeded": bool(cleanup_ok),
        "planned_route_length_m": float(rows[0]["planned_route_length_m"]),
        "driven_distance_m_median": round(
            statistics.median(float(row["driven_distance_m"]) for row in rows), 2
        ),
        "simulation_duration_s_median": round(statistics.median(durations), 2),
        "simulation_duration_s_min": round(min(durations), 2),
        "simulation_duration_s_max": round(max(durations), 2),
        "wall_clock_duration_s_total": round(
            sum(float(row["wall_clock_duration_s"]) for row in rows), 2
        ),
        "collision_count_total": sum(int(row["collision_count"]) for row in rows),
        "collision_incident_count_total": sum(
            int(row["collision_incident_count"]) for row in rows
        ),
        "return_position_error_m_max": max(
            float(row["return_position_error_m"]) for row in rows
        ),
        "return_heading_error_deg_max": max(
            float(row["return_heading_error_deg"]) for row in rows
        ),
        "per_loop": [
            {key: row[key] for key in DENSITY_FIELDS if key in row} for row in rows
        ],
        "ego_block_events_total": sum(int(row["ego_block_events"]) for row in rows),
        "ego_overtakes_total": sum(int(row["ego_overtakes"]) for row in rows),
        "npc_roadblocks_cleared_total": sum(
            int(row["npc_roadblocks_cleared"]) for row in rows
        ),
        "roadblock_relocations_total": sum(
            int(row["roadblock_relocations"]) for row in rows
        ),
        "roadblock_destructions_total": sum(
            int(row["roadblock_destructions"]) for row in rows
        ),
        "intervention_count_total": sum(
            int(row["intervention_count"]) for row in rows
        ),
        "walker_brake_ticks_total": sum(
            int(row["walker_brake_ticks"]) for row in rows
        ),
        "watchdog_aborted_any": any(bool(row["watchdog_aborted"]) for row in rows),
        "overtake_attempts": [
            {"loop_index": row["loop_index"], **item}
            for row in rows for item in row.get("overtake_attempts", [])
        ],
        "collision_events": [
            {"loop_index": row["loop_index"], **event}
            for row in rows for event in row.get("collision_events", [])
        ],
        "collision_incidents": [
            {"loop_index": row["loop_index"], **incident}
            for row in rows for incident in row.get("collision_incidents", [])
        ],
        "intervention_events": [
            {"loop_index": row["loop_index"], **event}
            for row in rows for event in row.get("intervention_events", [])
        ],
        "roadblock_observations": [
            {"loop_index": row["loop_index"], **event}
            for row in rows for event in row.get("roadblock_observations", [])
        ],
    }
    if len(durations) > 1:
        summary["simulation_duration_s_stdev"] = round(statistics.stdev(durations), 2)

    summary_path = args.summary_json.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {csv_path} and {summary_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density", choices=sorted(DENSITY_PROFILES), default="low",
        help="low=5/5, medium=10/10, dense=20/20 (vehicles/pedestrians)",
    )
    parser.add_argument("--vehicles", type=int, default=None,
                        help="override the profile NPC vehicle count")
    parser.add_argument("--pedestrians", type=int, default=None,
                        help="override the profile pedestrian count")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--tm-speed-difference-pct", type=float, default=35.0)
    parser.add_argument("--tm-leading-distance-m", type=float, default=4.0)
    parser.add_argument("--no-npc-hardening", dest="harden_npcs",
                        action="store_false",
                        help="leave NPCs on stock TM behaviour (auto lane change on)")
    parser.set_defaults(harden_npcs=True)
    parser.add_argument("--replenish-interval-s", type=float, default=5.0)
    parser.add_argument("--population-log-interval-s", type=float, default=60.0)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz")
    # 'hero' is what Traffic Manager looks for to centre hybrid physics.
    parser.add_argument("--ego-role-name", default="hero")
    parser.add_argument("--no-hybrid-physics", dest="hybrid_physics",
                        action="store_false",
                        help="keep full physics for every NPC, however distant")
    parser.set_defaults(hybrid_physics=True)
    parser.add_argument("--hybrid-physics-radius-m", type=float, default=70.0)
    parser.add_argument("--respawn-dormant", action="store_true", default=False,
                        help="let TM respawn dormant vehicles (stock --respawn)")
    parser.add_argument("--target-speed-kph", type=float, default=25.0)
    # Matches the accepted ego-only default; see that runner for the
    # StaticMeshActor_153 kerb intrusion this clears.
    parser.add_argument("--lane-offset-m", type=float, default=-0.5)
    parser.add_argument("--vehicle-threshold-m", type=float, default=6.0,
                        help="BasicAgent base vehicle detection distance")
    parser.add_argument("--no-safe-vehicles", dest="safe_vehicles",
                        action="store_false",
                        help="allow trucks/vans/buses/bikes in the NPC fleet")
    parser.set_defaults(safe_vehicles=True)
    parser.add_argument("--npc-clear-ahead-m", type=float, default=10.0,
                        help="a stopped NPC with a vehicle this close ahead is queuing, not stuck")
    parser.add_argument("--npc-stuck-timeout-s", type=float, default=60.0,
                        help="destroy an NPC stationary this long off a red light")
    parser.add_argument("--janitor-interval-s", type=float, default=2.0,
                        help="simulated seconds between roadblock sweeps")
    parser.add_argument("--ego-block-timeout-s", type=float, default=12.0,
                        help="ego stationary this long triggers an overtake attempt")
    parser.add_argument("--maximum-overtakes", type=int, default=6,
                        help="cap on ego lane-change attempts per loop")
    parser.add_argument(
        "--allow-scenario-interventions", action="store_true", default=False,
        help="allow roadblock removal/relocation and forced ego lane changes; any use is INTERVENED",
    )
    parser.add_argument("--overtake-other-lane-s", type=float, default=4.0)
    parser.add_argument("--blocker-reach-m", type=float, default=15.0)
    parser.add_argument("--no-brake-for-walkers", dest="brake_for_walkers",
                        action="store_false",
                        help="leave BasicAgent's stock pedestrian blindness in place")
    parser.set_defaults(brake_for_walkers=True)
    parser.add_argument("--walker-brake-distance-m", type=float, default=10.0)
    parser.add_argument("--leg-arrival-radius-m", type=float, default=12.0,
                        help="how close counts as reaching an ordered waypoint")
    parser.add_argument("--maximum-replans", type=int, default=8)
    parser.add_argument("--no-progress-timeout-s", type=float, default=180.0,
                        help="abort the loop cleanly after this long with no progress")
    parser.add_argument("--no-bbs-detection", dest="use_bbs_detection",
                        action="store_false",
                        help="disable bounding-box NPC detection")
    parser.set_defaults(use_bbs_detection=True)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--real-time-tick-period-s", type=float, default=0.05)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    parser.add_argument("--spawn-z-offset-m", type=float, default=0.3)
    parser.add_argument("--completion-radius-m", type=float, default=6.0)
    parser.add_argument("--completion-heading-tolerance-deg", type=float, default=25.0)
    # Congestion and red lights make a dense loop far slower than the ego-only
    # loop, so the bound is 15 minutes of simulated time per loop.
    parser.add_argument("--maximum-loop-sim-s", type=float, default=900.0)
    parser.add_argument(
        "--out-csv", type=Path, required=True,
        help="new CSV path; existing files are never appended to or overwritten",
    )
    parser.add_argument(
        "--summary-json", type=Path, required=True,
        help="new JSON path; existing files are never overwritten",
    )
    parser.add_argument("--spectator", dest="spectator", action="store_true")
    parser.add_argument("--no-spectator", dest="spectator", action="store_false")
    parser.set_defaults(spectator=True)
    parser.add_argument("--spectator-behind-m", type=float, default=10.0)
    parser.add_argument("--spectator-above-m", type=float, default=5.0)
    parser.add_argument("--spectator-pitch-deg", type=float, default=-15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    args = build_parser().parse_args(argv)
    if args.loops != 1:
        raise SystemExit("--loops must be exactly 1; each episode requires a fresh command/world")
    for name in ("vehicles", "pedestrians"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name} must not be negative")
    try:
        return run(args)
    except (RouteBError, ego_route_config.RouteConfigError, RuntimeError) as exc:
        print(f"route-b density runner failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("interrupted; NPCs destroyed and settings restored", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

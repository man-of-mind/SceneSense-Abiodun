#!/usr/bin/env python3
"""Drive the ego-only Route B full-map loop in Town10HD_Opt for visual review.

This runner is deliberately small.  It reuses the shared advisor route format
(``ego_route_config``) and the stock CARLA ``BasicAgent`` for lane following,
junction handling, and traffic lights, rather than introducing another
geometric controller.  Route A files are read-only here and are never written.

Scope of this first version: one ego vehicle, no NPC vehicles, no pedestrians,
no cameras/radar, no perception, no OAI, no spatial-map process.

The route is driven as an ordered sequence of legs.  Each leg targets the next
intermediate waypoint; the final leg targets the start location.  A loop is
only reported complete after every ordered waypoint has been reached and the
ego is back inside the start gate on the expected heading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
ADVISOR_CODES = REPO_ROOT / "rl_agent" / "advisor_helper_scripts" / "codes"
CARLA_AGENTS_ROOT = REPO_ROOT.parents[1] / "carla"

for _path in (str(REPO_ROOT), str(CARLA_AGENTS_ROOT), str(ADVISOR_CODES)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import carla  # noqa: E402

import ego_route_config  # noqa: E402

DEFAULT_ROUTE = (
    REPO_ROOT
    / "data_collection"
    / "routes"
    / "town10hd_opt_route_b_full_map_loop_v1.json"
)
REQUIRED_MAP = "Town10HD_Opt"


class RouteBError(RuntimeError):
    """Fail-closed runner error."""


# --------------------------------------------------------------------------- utils


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def location_of(payload: Mapping[str, Any]) -> carla.Location:
    return carla.Location(
        x=float(payload["x"]), y=float(payload["y"]), z=float(payload["z"])
    )


def transform_of(payload: Mapping[str, Any]) -> carla.Transform:
    location = payload["location"]
    rotation = payload["rotation"]
    return carla.Transform(
        carla.Location(
            x=float(location["x"]),
            y=float(location["y"]),
            z=float(location["z"]),
        ),
        carla.Rotation(
            pitch=float(rotation["pitch"]),
            yaw=float(rotation["yaw"]),
            roll=float(rotation["roll"]),
        ),
    )


def planned_length_m(route: Mapping[str, Any]) -> float:
    """Closed length of the exported dense path, including the closing seam."""

    path = route.get("planned_path") or []
    if len(path) < 2:
        return 0.0
    total = 0.0
    for previous, current in zip(path, path[1:]):
        total += math.hypot(
            float(current["x"]) - float(previous["x"]),
            float(current["y"]) - float(previous["y"]),
        )
    total += math.hypot(
        float(path[0]["x"]) - float(path[-1]["x"]),
        float(path[0]["y"]) - float(path[-1]["y"]),
    )
    return total


def region_labels(route: Mapping[str, Any], via_count: int) -> list[str]:
    """Per-waypoint B1/B2/B3 labels when the route file supplies them."""

    labels = route.get("route_b_regions")
    if isinstance(labels, list) and len(labels) == via_count:
        return [str(item) for item in labels]
    return ["unlabelled"] * via_count


# ----------------------------------------------------------------- carla plumbing


def destroy_actor(actor: Any) -> None:
    if actor is None:
        return
    try:
        if str(getattr(actor, "type_id", "")).startswith("sensor."):
            actor.stop()
    except (AttributeError, RuntimeError):
        pass
    try:
        actor.destroy()
    except RuntimeError:
        pass


def chase_spectator(world: Any, actor: Any, behind_m: float, above_m: float,
                    pitch_deg: float) -> None:
    followed = actor.get_transform()
    forward = followed.get_forward_vector()
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(
                x=float(followed.location.x) - behind_m * float(forward.x),
                y=float(followed.location.y) - behind_m * float(forward.y),
                z=float(followed.location.z) + above_m,
            ),
            carla.Rotation(
                pitch=pitch_deg, yaw=float(followed.rotation.yaw), roll=0.0
            ),
        )
    )


class CollisionMailbox:
    """Collect collision events without blocking the tick loop."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def callback(self, event: Any) -> None:
        impulse = event.normal_impulse
        row = {
            "frame_id": int(event.frame),
            "other_actor_type": str(event.other_actor.type_id),
            "impulse_norm": float(
                math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            ),
        }
        # The sensor rides the ego, so its measurement transform locates the
        # contact well enough to find the offending spot on the route.
        try:
            location = event.transform.location
            row["x"] = round(float(location.x), 2)
            row["y"] = round(float(location.y), 2)
        except (AttributeError, RuntimeError):
            pass
        self._rows.append(row)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def count(self) -> int:
        return len(self._rows)


class TickPacer:
    """Hold synchronous ticks to a wall-clock period, catching up when late."""

    def __init__(self, period_s: float) -> None:
        if period_s <= 0.0:
            raise ValueError("pacing period must be positive")
        self.period_s = float(period_s)
        self._next_deadline_s = time.monotonic() + self.period_s

    def wait(self) -> None:
        sleep_s = self._next_deadline_s - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        self._next_deadline_s += self.period_s


# ------------------------------------------------------------------- driving loop


def drive_one_loop(
    world: Any,
    vehicle: Any,
    agent: Any,
    route: Mapping[str, Any],
    collisions: CollisionMailbox,
    args: argparse.Namespace,
    loop_index: int,
) -> dict[str, Any]:
    """Drive the ordered via sequence and return to the start gate."""

    start_transform = transform_of(route["start"])
    vias = [location_of(item) for item in route["intermediate_waypoints"]]
    labels = region_labels(route, len(vias))
    legs = list(vias) + [start_transform.location]

    fixed_delta_s = float(args.fixed_delta_seconds)
    pacer = TickPacer(float(args.real_time_tick_period_s))

    collisions_at_start = collisions.count()
    collision_rows_at_start = len(collisions.rows)
    wall_start_s = time.monotonic()
    sim_elapsed_s = 0.0
    driven_m = 0.0
    ticks = 0
    reached: list[dict[str, Any]] = []
    stalled_ticks = 0

    previous = vehicle.get_transform().location
    leg_index = 0
    agent.set_destination(legs[0])

    maximum_sim_s = float(args.maximum_loop_sim_s)
    while leg_index < len(legs):
        if sim_elapsed_s > maximum_sim_s:
            raise RouteBError(
                f"loop {loop_index} exceeded the {maximum_sim_s:.0f} s simulated budget "
                f"at leg {leg_index + 1}/{len(legs)}"
            )

        control = agent.run_step()
        control.manual_gear_shift = False
        vehicle.apply_control(control)
        world.tick()
        ticks += 1
        sim_elapsed_s += fixed_delta_s

        transform = vehicle.get_transform()
        current = transform.location
        driven_m += math.hypot(
            float(current.x) - float(previous.x),
            float(current.y) - float(previous.y),
        )
        previous = current

        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )
        # A red light is a legitimate stop, so only flag a stall when the agent
        # is not reporting a hazard-induced brake.
        if speed_mps < 0.1 and control.brake < 0.5:
            stalled_ticks += 1
        else:
            stalled_ticks = 0
        if stalled_ticks * fixed_delta_s > float(args.stall_timeout_s):
            raise RouteBError(
                f"loop {loop_index} stalled for more than {args.stall_timeout_s:.0f} s "
                f"at leg {leg_index + 1}/{len(legs)}"
            )

        if args.spectator:
            chase_spectator(
                world, vehicle, args.spectator_behind_m,
                args.spectator_above_m, args.spectator_pitch_deg,
            )
        if agent.done():
            target = legs[leg_index]
            reached.append(
                {
                    "leg_index": leg_index,
                    "region": labels[leg_index] if leg_index < len(labels) else "start",
                    "target_x": float(target.x),
                    "target_y": float(target.y),
                    "arrival_x": float(current.x),
                    "arrival_y": float(current.y),
                    "arrival_error_m": math.hypot(
                        float(current.x) - float(target.x),
                        float(current.y) - float(target.y),
                    ),
                    "sim_s": sim_elapsed_s,
                }
            )
            leg_index += 1
            if leg_index < len(legs):
                agent.set_destination(legs[leg_index])

        pacer.wait()

    # Bring the ego to rest so the return pose is measured stationary.
    for _ in range(int(round(1.0 / fixed_delta_s))):
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        )
        world.tick()
        ticks += 1
        sim_elapsed_s += fixed_delta_s
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
    ordered_regions_covered = sorted(
        {row["region"] for row in reached if row["region"] not in ("start", "unlabelled")}
    )
    all_vias_reached = len(reached) == len(legs)
    completed = bool(
        all_vias_reached
        and return_position_error_m <= float(args.completion_radius_m)
        and return_heading_error_deg <= float(args.completion_heading_tolerance_deg)
    )
    return {
        "loop_index": loop_index,
        "completed": completed,
        "all_ordered_waypoints_reached": all_vias_reached,
        "waypoints_reached": len(reached),
        "waypoints_expected": len(legs),
        "regions_covered": ",".join(ordered_regions_covered),
        "driven_distance_m": round(driven_m, 2),
        "simulation_duration_s": round(sim_elapsed_s, 2),
        "wall_clock_duration_s": round(time.monotonic() - wall_start_s, 2),
        "ticks": ticks,
        "return_position_error_m": round(return_position_error_m, 3),
        "return_heading_error_deg": round(return_heading_error_deg, 3),
        "collision_count": collisions.count() - collisions_at_start,
        "collision_events": collisions.rows[collision_rows_at_start:],
        "leg_arrivals": reached,
    }


# ------------------------------------------------------------------------ runner


def run(args: argparse.Namespace) -> int:
    route_path = args.route_config.resolve()
    if not route_path.is_file():
        raise RouteBError(
            f"Route B config not found: {route_path}. Author it first with "
            f"data_collection/author_advisor_demo_route.py (see --help)."
        )
    route = ego_route_config.load_route_config(route_path)
    if not ego_route_config.maps_match(route["map"], REQUIRED_MAP):
        raise RouteBError(
            f"route targets {route['map']!r}, expected {REQUIRED_MAP}"
        )
    if not route["intermediate_waypoints"]:
        raise RouteBError("Route B must define ordered intermediate waypoints")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    observed_map = str(world.get_map().name)
    if not ego_route_config.maps_match(observed_map, REQUIRED_MAP):
        raise RouteBError(
            f"CARLA is running {observed_map!r}; load {REQUIRED_MAP} first"
        )

    original_settings = world.get_settings()
    vehicle = None
    collision_sensor = None
    owned_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    status = "INCOMPLETE"
    cleanup_ok = False

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(args.fixed_delta_seconds)
        world.apply_settings(settings)

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
        owned_ids.append(int(vehicle.id))

        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=vehicle
        )
        owned_ids.append(int(collision_sensor.id))
        collisions = CollisionMailbox()
        collision_sensor.listen(collisions.callback)

        # Settle the spawn before the agent takes over.
        for _ in range(int(args.warmup_ticks)):
            vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            )
            world.tick()

        from agents.navigation.basic_agent import BasicAgent  # noqa: E402

        def new_agent() -> Any:
            """Build a fresh agent per loop, seeded at the ego's current pose.

            A reused agent re-plans the next loop from the stale
            ``target_waypoint`` left at the end of the previous one. Starting
            each loop from a clean local planner keeps per-loop timings
            independent, which is the point of the repeatability run.
            """
            agent = BasicAgent(
                vehicle,
                target_speed=float(args.target_speed_kph),
                opt_dict={
                    "sampling_resolution": float(route["route_sampling_resolution_m"]),
                    "base_tlight_threshold": 5.0,
                    "max_brake": 0.5,
                    "offset": float(args.lane_offset_m),
                },
            )
            agent.follow_speed_limits(False)
            agent.set_target_speed(float(args.target_speed_kph))
            return agent

        length_m = planned_length_m(route)
        print(
            f"route_id={route['name']!r} map={observed_map} "
            f"ordered_waypoints={len(route['intermediate_waypoints'])} "
            f"planned_route_length_m={length_m:.1f} loops={args.loops}",
            flush=True,
        )

        for loop_index in range(1, int(args.loops) + 1):
            result = drive_one_loop(
                world, vehicle, new_agent(), route, collisions, args, loop_index
            )
            result["route_id"] = str(route["name"])
            result["planned_route_length_m"] = round(length_m, 2)
            rows.append(result)
            print(
                f"loop {loop_index}/{args.loops}: completed={result['completed']} "
                f"driven_m={result['driven_distance_m']:.1f} "
                f"sim_s={result['simulation_duration_s']:.1f} "
                f"wall_s={result['wall_clock_duration_s']:.1f} "
                f"return_pos_err_m={result['return_position_error_m']:.2f} "
                f"return_yaw_err_deg={result['return_heading_error_deg']:.2f} "
                f"collisions={result['collision_count']} "
                f"regions={result['regions_covered']}",
                flush=True,
            )

        status = "COMPLETE" if all(row["completed"] for row in rows) else "INCOMPLETE"
        return 0 if status == "COMPLETE" else 1
    finally:
        for actor in (collision_sensor, vehicle):
            destroy_actor(actor)

        def remaining_live() -> list[int]:
            # get_actor still returns a handle for a destroyed actor, so the
            # is_alive flag is what actually proves the teardown landed.
            live: list[int] = []
            for actor_id in owned_ids:
                try:
                    actor = world.get_actor(int(actor_id))
                except RuntimeError:
                    actor = None
                if actor is None:
                    continue
                try:
                    if bool(actor.is_alive):
                        live.append(int(actor_id))
                except (AttributeError, RuntimeError):
                    live.append(int(actor_id))
            return live

        remaining = remaining_live()
        for _ in range(20):
            if not remaining:
                break
            try:
                world.tick()
            except RuntimeError:
                break
            remaining = remaining_live()
        cleanup_ok = not remaining
        if remaining:
            print(
                f"warning: owned actors still alive after cleanup: {remaining}",
                file=sys.stderr,
                flush=True,
            )
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            cleanup_ok = False
        write_outputs(args, rows, status, cleanup_ok)


def write_outputs(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    status: str,
    cleanup_ok: bool,
) -> None:
    if not rows:
        print(
            f"no completed loops recorded; status={status} cleanup_succeeded={cleanup_ok}",
            file=sys.stderr,
            flush=True,
        )
        return

    fields = [
        "route_id", "loop_index", "completed", "all_ordered_waypoints_reached",
        "waypoints_reached", "waypoints_expected", "regions_covered",
        "planned_route_length_m", "driven_distance_m", "simulation_duration_s",
        "wall_clock_duration_s", "ticks", "return_position_error_m",
        "return_heading_error_deg", "collision_count",
    ]
    csv_path = args.out_csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    sim_durations = [float(row["simulation_duration_s"]) for row in rows]
    driven = [float(row["driven_distance_m"]) for row in rows]
    summary = {
        "route_id": str(rows[0]["route_id"]),
        "loops": len(rows),
        "status": status,
        "cleanup_succeeded": bool(cleanup_ok),
        "planned_route_length_m": float(rows[0]["planned_route_length_m"]),
        "driven_distance_m_median": round(statistics.median(driven), 2),
        "simulation_duration_s_median": round(statistics.median(sim_durations), 2),
        "simulation_duration_s_min": round(min(sim_durations), 2),
        "simulation_duration_s_max": round(max(sim_durations), 2),
        "wall_clock_duration_s_total": round(
            sum(float(row["wall_clock_duration_s"]) for row in rows), 2
        ),
        "collision_count_total": sum(int(row["collision_count"]) for row in rows),
        "return_position_error_m_max": max(
            float(row["return_position_error_m"]) for row in rows
        ),
        "return_heading_error_deg_max": max(
            float(row["return_heading_error_deg"]) for row in rows
        ),
        "loops_completed": sum(1 for row in rows if row["completed"]),
        "per_loop": [
            {key: row[key] for key in fields if key in row} for row in rows
        ],
        "collision_events": [
            {"loop_index": row["loop_index"], **event}
            for row in rows
            for event in row.get("collision_events", [])
        ],
    }
    if len(sim_durations) > 1:
        summary["simulation_duration_s_stdev"] = round(
            statistics.stdev(sim_durations), 2
        )
    summary_path = args.summary_json.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {csv_path} and {summary_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz")
    parser.add_argument("--ego-role-name", default="route_b_ego")
    parser.add_argument("--target-speed-kph", type=float, default=25.0)
    # Town10HD_Opt has a parked-car-sized static mesh (StaticMeshActor_153,
    # centre ~(13.9, 31.3), extent 5.1x2.0x2.1) whose edge intrudes into the
    # eastbound y~28 lane just after the Route B start, leaving only ~0.1 m
    # clearance for a lane-centred ego. Measured: at offset 0.0 loops 2+ scrape
    # it (1-20 contacts); at -0.5 three consecutive loops are contact-free.
    # Negative shifts left of travel, away from that kerb. Pass 0.0 to observe
    # the unmitigated lane-centre behaviour.
    parser.add_argument("--lane-offset-m", type=float, default=-0.5)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--real-time-tick-period-s", type=float, default=0.05)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    parser.add_argument("--spawn-z-offset-m", type=float, default=0.3)
    parser.add_argument("--completion-radius-m", type=float, default=6.0)
    parser.add_argument(
        "--completion-heading-tolerance-deg", type=float, default=25.0
    )
    parser.add_argument("--maximum-loop-sim-s", type=float, default=900.0)
    parser.add_argument("--stall-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "data_collection" / "routes" / "route_b_loops.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "data_collection" / "routes" / "route_b_summary.json",
    )
    parser.add_argument("--spectator", dest="spectator", action="store_true")
    parser.add_argument("--no-spectator", dest="spectator", action="store_false")
    parser.set_defaults(spectator=True)
    parser.add_argument("--spectator-behind-m", type=float, default=10.0)
    parser.add_argument("--spectator-above-m", type=float, default=5.0)
    parser.add_argument("--spectator-pitch-deg", type=float, default=-15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.loops < 1:
        raise SystemExit("--loops must be at least 1")
    try:
        return run(args)
    except (RouteBError, ego_route_config.RouteConfigError) as exc:
        print(f"route-b runner failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("interrupted; world settings restored", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

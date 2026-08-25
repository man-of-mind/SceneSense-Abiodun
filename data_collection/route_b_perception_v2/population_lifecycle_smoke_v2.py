#!/usr/bin/env python3
"""Short disposable lifecycle smoke for the simulated-time reconcile schedule (v2).

v2 differences from v1:

* Reconciliation is driven by the real ``SimTimeReconcileSchedule`` object the
  Route B density runner now uses, so the realised interval is asserted to be
  five *simulated* seconds rather than five wall-clock seconds.
* The collector is given the population manager, so the new per-saved-frame
  controller-health telemetry is exercised and asserted here too.

Everything else is unchanged from v1 and the phased replacement mechanism itself
is untouched: the real sensing rig (three 1280x720 cameras plus the 200,000 PPS
radar through ``PerceptionCollectorV2`` with the fast rasterizer), a small
managed population, deliberate walker losses, and the requirement that each
replacement progress

    body spawn -> observed route tick -> controller attach -> observed route tick
    -> controller start

while the route's tick owner (``SamplingWorld``) sees a strictly contiguous CARLA
frame sequence and the aggregator sees no dropped, duplicate or out-of-order
callback.

This is an integration smoke, not a unit-test suite, and not a Route B loop: the
ego stays parked and everything written is deleted at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_COLLECTION = HERE.parent
AB = DATA_COLLECTION.parent
ADVISOR = AB / "rl_agent" / "advisor_helper_scripts" / "codes"
for path in (str(AB), str(DATA_COLLECTION), str(ADVISOR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import carla  # noqa: E402

# The same module the Route B density runner imports - not the stale
# data_collection/generate_traffic_v11.py snapshot.
import generate_traffic_v1 as traffic  # noqa: E402
import data_collection.run_route_b_perception_collection_v2 as collection  # noqa: E402
import data_collection.run_route_b_density_loop as density  # noqa: E402
import carla_collect_parked_ego_fusion_training_data as parked  # noqa: E402
from data_collection.render_provenance_v1 import (  # noqa: E402
    assert_epic_rendering,
    inspect_launch,
)

WORLD_DELTA_S = 0.05


class SmokeCollisions:
    """Minimal stand-in for the route runner's collision mailbox."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def count(self) -> int:
        return 0


def phase_history(events: list[dict[str, Any]], body_id: int) -> list[dict[str, Any]]:
    return [
        event for event in events
        if event.get("body_id") == body_id
        and event.get("phase") in (
            "body_spawned", "controller_spawned", "controller_started",
            "controller_attach_deferred", "controller_start_deferred",
            "controller_rejected", "controller_repair_queued",
            "body_retired_controller_loss", "body_retired_confirmed",
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--ticks", type=int, default=640)
    parser.add_argument("--replenish-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--kill-ticks", type=int, nargs="+", default=(120, 400),
        help="route ticks at which one managed walker is attacked",
    )
    parser.add_argument(
        "--kill-mode", choices=("body", "controller"), default="body",
        help="body: destroy the walker body (v1/v2 behaviour). controller: "
             "destroy only the AI controller and leave the body alive, which "
             "reproduces the pathological controller-loss condition",
    )
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path("/tmp/route_b_sim_time_lifecycle_smoke"),
        help="disposable dataset directory; deleted on exit",
    )
    parser.add_argument(
        "--report-json", type=Path,
        default=HERE / "population_lifecycle_smoke_v2.json",
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one linear smoke
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": "route_b_perception_v2.population_lifecycle_smoke.v2",
        "parameters": {
            "walkers": int(args.walkers),
            "vehicles": int(args.vehicles),
            "ticks": int(args.ticks),
            "kill_ticks": list(args.kill_ticks),
            "kill_mode": str(args.kill_mode),
            "replenish_interval_s": float(args.replenish_interval_s),
            "seed": int(args.seed),
        },
    }

    launch = inspect_launch(int(args.port))
    assert_epic_rendering({"launch": launch, "no_rendering_mode": False})
    report["launch"] = launch

    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world("Town10HD_Opt", True)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = WORLD_DELTA_S
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    report["map"] = str(world.get_map().name)
    report["no_rendering_mode"] = bool(world.get_settings().no_rendering_mode)

    random.seed(int(args.seed))
    traffic_manager = client.get_trafficmanager(int(args.tm_port))
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(int(args.seed))
    world.set_pedestrians_seed(int(args.seed))
    world.set_pedestrians_cross_factor(traffic.PERCENTAGE_PEDESTRIANS_CROSSING)

    walker_blueprints = traffic.get_actor_blueprints(world, "walker.pedestrian.*", "All")
    vehicle_blueprints = traffic.get_actor_blueprints(world, "vehicle.*", "All")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    population: Any = None
    collector: Any = None
    ego: Any = None
    checks: dict[str, Any] = {}
    lifecycle_rows: list[dict[str, Any]] = []
    tick_error = ""
    try:
        ego_bp = world.get_blueprint_library().filter("vehicle.lincoln.mkz*")[0]
        ego_bp.set_attribute("role_name", "hero")
        ego = world.spawn_actor(ego_bp, spawn_points[0])
        ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        for _ in range(10):
            world.tick()

        population_args = types.SimpleNamespace(
            number_of_vehicles=int(args.vehicles),
            number_of_walkers=int(args.walkers),
            car_lights_on=False, hero=False, asynch=False,
            replenish_interval=float(args.replenish_interval_s),
            population_log_interval=1e9,
        )
        population = traffic.TrafficPopulationManager(
            client, world, traffic_manager, population_args,
            vehicle_blueprints, walker_blueprints, spawn_points, True,
        )
        population.spawn_initial_population()
        target = int(args.walkers)
        report["initial_walkers"] = len(population.walkers)
        report["initial_vehicles"] = len(population.vehicle_ids)

        collector = collection.PerceptionCollectorV2(
            parked=parked, world=world, client=client, ego=ego,
            collisions=SmokeCollisions(), rpc_port=int(args.port), split="smoke",
            seed_bundle=None, rasterizer="fast", output_dir=work_dir,
            density="traffic_30_30", vehicles=int(args.vehicles),
            pedestrians=int(args.walkers), scenario_seed=int(args.seed),
            tm_seed=int(args.seed), target_speed_kph=0.0, hybrid_physics=False,
            route_path=collection.DEFAULT_ROUTE,
            progress_path=collection.DEFAULT_PROGRESS,
            population=population,
        )
        report["radar_attributes"] = dict(collector.radar_attributes)

        sampling = collection.SamplingWorld(world, collector, population)
        population.begin_route_mode()
        kill_schedule = {int(tick): None for tick in args.kill_ticks}
        killed: list[dict[str, Any]] = []
        # The real object the density runner uses, not a copy of its rule.
        schedule = density.SimTimeReconcileSchedule(float(args.replenish_interval_s))
        sim_now_s = 0.0

        for route_tick in range(1, int(args.ticks) + 1):
            ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            # The single authoritative tick. Any hidden frame raises here.
            frame_id = sampling.tick()
            sim_now_s += WORLD_DELTA_S
            population.note_route_tick(frame_id)

            if route_tick in kill_schedule and kill_schedule[route_tick] is None:
                victims = [
                    record for record in population.walkers
                    if record.get("controller_ready", False)
                    and record.get("id") is not None
                    and int(record["id"]) not in {row["body_id"] for row in killed}
                ]
                if victims:
                    victim = victims[0]
                    body_id = int(victim["id"])
                    controller_id = victim.get("con")
                    # Destroy behind the manager's back; verified 0 frame delta.
                    if args.kill_mode == "controller":
                        # Body survives: only its AI controller disappears.
                        target_id = None if controller_id is None else int(controller_id)
                    else:
                        target_id = body_id
                    actor = None if target_id is None else world.get_actor(target_id)
                    destroyed = bool(actor.destroy()) if actor is not None else False
                    killed.append({
                        "route_tick": route_tick, "frame_id": frame_id,
                        "kill_mode": str(args.kill_mode),
                        "destroyed_actor_id": target_id,
                        "body_id": body_id,
                        "controller_id": controller_id,
                        "destroy_result": destroyed,
                    })
                    kill_schedule[route_tick] = body_id

            if schedule.due(sim_now_s):
                population.reconcile()

        population.end_route_mode()
        report["reconcile_schedule"] = {
            "interval_s": schedule.interval_s,
            "fired_sim_s": [round(value, 4) for value in schedule.fired_sim_s],
            "realised_intervals_s": schedule.realised_intervals_s(),
            "simulated_seconds_elapsed": round(sim_now_s, 4),
        }
        report["killed"] = killed
        report["route_ticks"] = int(args.ticks)
        report["observed_ticks"] = population.observed_ticks

        events = population.lifecycle_events
        report["lifecycle_event_count"] = len(events)
        replacement_ids = [
            event["body_id"] for event in events
            if event.get("phase") == "body_spawned"
            and event.get("route_mode") is True
            and event.get("body_id") is not None
        ]
        report["replacement_body_ids"] = replacement_ids

        for body_id in replacement_ids:
            history = phase_history(events, body_id)
            spawned = next(
                (e for e in history if e["phase"] == "body_spawned"), None)
            attached = next(
                (e for e in history if e["phase"] == "controller_spawned"), None)
            started = next(
                (e for e in history if e["phase"] == "controller_started"), None)
            record = next(
                (r for r in population.walkers
                 if r.get("id") is not None and int(r["id"]) == int(body_id)), None)
            controller_id = None if record is None else record.get("con")
            parent_ok = False
            controller_alive = False
            if controller_id is not None:
                try:
                    controller = world.get_actor(int(controller_id))
                except RuntimeError:
                    controller = None
                if controller is not None and getattr(controller, "is_alive", False):
                    controller_alive = str(controller.type_id).startswith(
                        "controller.ai.walker")
                    try:
                        parent = controller.parent
                    except (AttributeError, RuntimeError):
                        parent = None
                    parent_ok = parent is not None and int(parent.id) == int(body_id)
            lifecycle_rows.append({
                "body_id": int(body_id),
                "controller_id": None if controller_id is None else int(controller_id),
                "body_spawn_tick": None if spawned is None else spawned["observed_tick"],
                "controller_attach_tick": None if attached is None else attached["observed_tick"],
                "controller_start_tick": None if started is None else started["observed_tick"],
                "attach_after_observed_tick": bool(
                    spawned is not None and attached is not None
                    and attached["observed_tick"] > spawned["observed_tick"]),
                "start_after_observed_tick": bool(
                    attached is not None and started is not None
                    and started["observed_tick"] > attached["observed_tick"]),
                "controller_alive": controller_alive,
                "controller_parent_matches_body": parent_ok,
                "controller_ready": bool(
                    record is not None and record.get("controller_ready", False)),
                "phase": None if record is None else record.get("phase"),
                "phases_seen": [e["phase"] for e in history],
            })
        report["replacement_lifecycle"] = lifecycle_rows

        live_walkers = {
            int(a.id) for a in world.get_actors().filter("walker.pedestrian.*")
        }
        owned_walkers = [
            int(r["id"]) for r in population.walkers if r.get("id") is not None
        ]
        report["final_population"] = {
            "owned_walker_records": len(owned_walkers),
            "owned_present_in_world": len([i for i in owned_walkers if i in live_walkers]),
            "world_walkers": len(live_walkers),
            "owned_vehicle_records": len(population.vehicle_ids),
            "controllers_ready": sum(
                1 for r in population.walkers if r.get("controller_ready", False)),
            "pending_phases": sum(
                1 for r in population.walkers
                if r.get("phase") in (traffic.PHASE_BODY_PENDING,
                                      traffic.PHASE_CONTROLLER_PENDING)),
            "orphan_controllers": len(population.orphan_controller_ids),
        }
        # --- forced controller-loss audit ---------------------------------
        # One body must never collect a second controller: that cycle is what
        # the retirement repair removes.
        killed_bodies = {int(row["body_id"]) for row in killed}
        killed_controllers = {
            int(row["controller_id"]) for row in killed
            if row.get("controller_id") is not None
        }
        owned_body_ids = {int(i) for i in owned_walkers}
        forced_rows: list[dict[str, Any]] = []
        for row in killed:
            old_body = int(row["body_id"])
            retired = [
                e for e in events
                if e.get("phase") == "body_retired_controller_loss"
                and e.get("body_id") == old_body
            ]
            confirmed = [
                e for e in events
                if e.get("phase") == "body_retired_confirmed"
                and e.get("body_id") == old_body
            ]
            attached = [
                e for e in events
                if e.get("phase") == "controller_spawned"
                and e.get("body_id") == old_body
            ]
            reattached_after_kill = [
                e for e in attached
                if int(e.get("observed_tick", 0)) >= int(row["route_tick"])
            ]
            forced_rows.append({
                "kill_route_tick": int(row["route_tick"]),
                "kill_mode": row.get("kill_mode"),
                "old_body_id": old_body,
                "old_controller_id": row.get("controller_id"),
                "retired": bool(retired),
                "retire_reason": retired[0].get("detail") if retired else None,
                "retire_observed_tick": (
                    retired[0].get("observed_tick") if retired else None),
                "destruction_confirmed": bool(confirmed),
                "controllers_ever_attached_to_old_body": [
                    e.get("controller_id") for e in attached],
                "controllers_attached_after_kill": len(reattached_after_kill),
                "old_body_still_owned": old_body in owned_body_ids,
                "old_body_alive_in_world": old_body in live_walkers,
                "phases_seen": [
                    e["phase"] for e in phase_history(events, old_body)],
            })
        report["forced_controller_loss"] = forced_rows

        # Global form of the same property, over every body of the run rather
        # than only the two forced victims.
        controllers_per_body: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        for event in events:
            phase = str(event.get("phase"))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if phase == "controller_spawned" and event.get("body_id") is not None:
                key = str(int(event["body_id"]))
                controllers_per_body[key] = controllers_per_body.get(key, 0) + 1
        report["lifecycle_phase_counts"] = phase_counts
        report["controllers_per_body"] = controllers_per_body
        report["max_controllers_on_one_body"] = (
            max(controllers_per_body.values()) if controllers_per_body else 0)

        health_rows = collector.controller_health_samples
        report["controller_health_samples"] = len(health_rows)
        report["controller_health_last"] = health_rows[-1] if health_rows else None
        report["controller_health_ready_min"] = (
            min(int(row["controllers_marked_ready"]) for row in health_rows)
            if health_rows else None)
        report["controller_health_bodies_min"] = (
            min(int(row["managed_walker_bodies_alive"]) for row in health_rows)
            if health_rows else None)
        report["controller_health_attached_min"] = (
            min(int(row["live_attached_walker_controllers"]) for row in health_rows)
            if health_rows else None)
        report["controller_health_orphans_max"] = (
            max(int(row["orphan_controllers"]) for row in health_rows)
            if health_rows else None)
        # Verified live + attached + ready, against the same floor the episode
        # gate controllers_ready_95pct_every_saved_frame uses.
        report["controller_health_live_ready_min"] = (
            min(int(row["live_ready_walker_controllers"]) for row in health_rows)
            if health_rows else None)
        below_floor = [
            row for row in health_rows
            if int(row["managed_walker_bodies_alive"]) > 0
            and int(row["live_ready_walker_controllers"]) < math.ceil(
                collection.CONTROLLER_READY_FRACTION_GATE
                * float(row["managed_walker_bodies_alive"]))
        ]
        report["controller_health_frames_below_ready_floor"] = len(below_floor)
        report["controller_health_first_frame_below_floor"] = (
            below_floor[0]["frame_id"] if below_floor else None)
        report["cadence"] = {
            "raw_callbacks": collector.aggregator.raw_callbacks,
            "dropped_callback_frames": collector.aggregator.dropped_callback_frames,
            "duplicate_callbacks": collector.aggregator.duplicate_callbacks,
            "out_of_order_callbacks": collector.aggregator.out_of_order_callbacks,
            "timestamp_reversals": collector.aggregator.timestamp_reversals,
            "prepared_inputs": len(collector.prepared_records),
            "saved_samples": collector.saved,
            "expected_window_callbacks": collector.aggregator.expected_window_callbacks,
            "window_callbacks_exact": bool(collector.prepared_records) and all(
                int(row["window_callbacks"]) == collector.aggregator.expected_window_callbacks
                for row in collector.prepared_records),
        }
        # Hidden ticks would show up as raw_callbacks exceeding warmup + route ticks.
        expected_callbacks = collection.CADENCE_WARMUP_TICKS + int(args.ticks)
        report["cadence"]["expected_raw_callbacks"] = expected_callbacks

        final = report["final_population"]
        cadence = report["cadence"]
        intervals = report["reconcile_schedule"]["realised_intervals_s"]
        checks = {
            # Compared on the world-tick grid, not in floats: the schedule can
            # only fire on a tick boundary, so a 2.0 s interval legitimately
            # realises as 1.95/2.00/2.05 s. abs(1.95 - 2.0) is 0.05000000000000018
            # in float64 and failed a <= 0.05 test by 7e-17.
            "reconciles_fired_on_simulated_time": len(intervals) >= 3 and all(
                abs(round(value / WORLD_DELTA_S)
                    - round(float(args.replenish_interval_s) / WORLD_DELTA_S)) <= 1
                for value in intervals),
            "at_least_one_walker_destroyed": len(killed) >= 1 and all(
                row["destroy_result"] for row in killed),
            "every_loss_replaced": len(lifecycle_rows) >= len(killed),
            "controller_health_recorded": bool(health_rows) and all(
                int(row["managed_walker_bodies_alive"]) >= 0 for row in health_rows),
            "controller_health_ready_tracks_bodies": bool(health_rows) and all(
                int(row["controllers_marked_ready"])
                >= int(row["managed_walker_bodies_alive"]) - 1
                for row in health_rows),
            "window_callbacks_exact": report["cadence"]["window_callbacks_exact"],
            "every_replacement_phased_body_then_controller": bool(lifecycle_rows) and all(
                row["attach_after_observed_tick"] for row in lifecycle_rows),
            "every_replacement_phased_controller_then_start": bool(lifecycle_rows) and all(
                row["start_after_observed_tick"] for row in lifecycle_rows),
            "every_replacement_controller_ready": bool(lifecycle_rows) and all(
                row["controller_ready"] for row in lifecycle_rows),
            "every_replacement_controller_parent_correct": bool(lifecycle_rows) and all(
                row["controller_parent_matches_body"] and row["controller_alive"]
                for row in lifecycle_rows),
            "population_back_to_target": final["owned_walker_records"] == target
            and final["owned_present_in_world"] == target,
            "vehicles_back_to_target":
                final["owned_vehicle_records"] == int(args.vehicles),
            # The deployed floor, not final-frame equality: a controller that
            # vanishes inside the last reconcile window is legitimately still
            # counted as attached-but-unreconciled on the final saved frame.
            # 29 of 30 bodies is 96.7%, above the 0.95 gate.
            "verified_ready_controllers_hold_deployed_floor":
                report["controller_health_frames_below_ready_floor"] == 0,
            "registry_fully_recovered_at_end":
                final["controllers_ready"] == target
                and final["pending_phases"] == 0
                and final["orphan_controllers"] == 0,
            "every_forced_loss_retired_old_body": bool(forced_rows) and all(
                row["retired"] and row["destruction_confirmed"]
                for row in forced_rows),
            "no_controller_reattached_to_retired_body": bool(forced_rows) and all(
                row["controllers_attached_after_kill"] == 0
                for row in forced_rows),
            "at_most_one_controller_per_body_lifetime": bool(forced_rows) and all(
                len(row["controllers_ever_attached_to_old_body"]) <= 1
                for row in forced_rows),
            "no_body_anywhere_received_two_controllers": bool(controllers_per_body)
            and int(report["max_controllers_on_one_body"]) <= 1,
            "retired_body_destroyed_and_dropped": bool(forced_rows) and all(
                not row["old_body_still_owned"]
                and not row["old_body_alive_in_world"]
                for row in forced_rows),
            "replacements_use_new_body_and_controller_ids": bool(lifecycle_rows) and all(
                row["body_id"] not in killed_bodies
                and (row["controller_id"] is None
                     or row["controller_id"] not in killed_controllers)
                for row in lifecycle_rows),
            "all_controllers_ready_at_end": final["controllers_ready"] == target
            and final["pending_phases"] == 0,
            "zero_unobserved_world_frame_gaps": not tick_error,
            "raw_callbacks_match_observed_ticks":
                cadence["raw_callbacks"] == expected_callbacks,
            "zero_dropped_callbacks": cadence["dropped_callback_frames"] == 0,
            "zero_duplicate_callbacks": cadence["duplicate_callbacks"] == 0,
            "zero_out_of_order_callbacks": cadence["out_of_order_callbacks"] == 0,
            "zero_timestamp_reversals": cadence["timestamp_reversals"] == 0,
        }
    except collection.TickOwnershipError as exc:
        tick_error = str(exc)
        report["tick_ownership_error"] = tick_error
        checks["zero_unobserved_world_frame_gaps"] = False
    except Exception as exc:  # noqa: BLE001 - the smoke reports, it does not mask
        report["error"] = f"{type(exc).__name__}: {exc}"
        checks["smoke_completed"] = False
    finally:
        if population is not None:
            try:
                population.end_route_mode()
            except Exception:  # noqa: BLE001
                pass
        sensor_cleanup_ok = False
        if collector is not None:
            try:
                sensor_cleanup_ok = bool(collector.stop_sensors())
            except Exception as exc:  # noqa: BLE001
                report["sensor_cleanup_error"] = str(exc)
            report["sensor_cleanup"] = collector.cleanup_records
        checks["sensor_cleanup_succeeded"] = sensor_cleanup_ok

        tracked: list[int] = []
        if population is not None:
            tracked += [int(r["id"]) for r in population.walkers if r.get("id") is not None]
            tracked += [int(r["con"]) for r in population.walkers if r.get("con") is not None]
            tracked += [int(i) for i in population.orphan_controller_ids]
            tracked += [int(i) for i in population.vehicle_ids]
            try:
                population.destroy()
            except Exception as exc:  # noqa: BLE001
                report["population_cleanup_error"] = str(exc)
        try:
            world.tick()
        except RuntimeError as exc:
            report["cleanup_tick_error"] = str(exc)
        if ego is not None:
            try:
                ego.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()
        except RuntimeError:
            pass
        surviving = []
        for actor_id in dict.fromkeys(tracked):
            try:
                actor = world.get_actor(actor_id)
            except RuntimeError:
                continue
            if actor is not None and getattr(actor, "is_alive", False):
                surviving.append(actor_id)
        leftover_walkers = sum(1 for _ in world.get_actors().filter("walker.pedestrian.*"))
        leftover_controllers = sum(1 for _ in world.get_actors().filter("controller.ai.walker"))
        leftover_sensors = sum(1 for _ in world.get_actors().filter("sensor.*"))
        report["cleanup"] = {
            "tracked_actors": len(dict.fromkeys(tracked)),
            "surviving_tracked": surviving,
            "walkers_left_in_world": leftover_walkers,
            "controllers_left_in_world": leftover_controllers,
            "sensors_left_in_world": leftover_sensors,
        }
        checks["cleanup_leaves_no_actors"] = (
            not surviving and leftover_walkers == 0
            and leftover_controllers == 0 and leftover_sensors == 0
        )
        try:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(original_settings)
            report["settings_restored"] = True
        except RuntimeError as exc:
            report["settings_restored"] = False
            report["settings_restore_error"] = str(exc)
        if work_dir.exists() and not args.keep_work_dir:
            report["work_dir_bytes"] = sum(
                p.stat().st_size for p in work_dir.rglob("*") if p.is_file())
            shutil.rmtree(work_dir, ignore_errors=True)

    report["checks"] = checks
    passed = bool(checks) and all(bool(value) for value in checks.values())
    report["status"] = (
        "SIM_TIME_LIFECYCLE_SMOKE_PASSED" if passed
        else "SIM_TIME_LIFECYCLE_SMOKE_FAILED"
    )
    Path(args.report_json).write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

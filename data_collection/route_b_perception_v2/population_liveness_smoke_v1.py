#!/usr/bin/env python3
"""Bounded disposable integration smoke for the walker-liveness fix.

Spawns a small managed walker population through the unmodified
``TrafficPopulationManager``, lets it stabilize, destroys exactly one managed
walker behind the manager's back, then calls normal reconciliation and checks
that the manager notices and repairs it.

This is an integration smoke, not a unit-test suite: one population, one
deliberate loss, one reconcile, one set of assertions. Nothing is persisted into
the dataset and every actor is destroyed at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path
from typing import Any

AB = Path(__file__).resolve().parents[2]
ADVISOR = AB / "rl_agent" / "advisor_helper_scripts" / "codes"
for path in (str(AB), str(ADVISOR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import carla  # noqa: E402
import generate_traffic_v1 as traffic  # noqa: E402

from data_collection.render_provenance_v1 import (  # noqa: E402
    assert_epic_rendering,
    inspect_launch,
)

WORLD_DELTA_S = 0.05


def counts(world: Any, owned_ids: list[int]) -> dict[str, Any]:
    """Three independent counts of the same owned walker bodies."""
    present = {int(a.id) for a in world.get_actors().filter("walker.pedestrian.*")}
    by_id = 0
    for actor_id in owned_ids:
        try:
            actor = world.get_actor(int(actor_id))
        except RuntimeError:
            actor = None
        if actor is not None and getattr(actor, "is_alive", False):
            by_id += 1
    return {
        "world_snapshot_walkers": len(present),
        "manager_owned_records": len(owned_ids),
        "owned_present_in_snapshot": len([i for i in owned_ids if int(i) in present]),
        "per_id_lookup_alive": by_id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--stabilize-ticks", type=int, default=120)
    parser.add_argument("--settle-ticks", type=int, default=40)
    parser.add_argument("--report-json", type=Path,
                        default=Path(__file__).resolve().parent / "population_liveness_smoke_v1.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {"schema": "route_b_perception_v2.population_liveness_smoke.v1"}

    launch = inspect_launch(int(args.port))
    assert_epic_rendering({"launch": launch, "no_rendering_mode": False})
    report["launch"] = launch

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world("Town10HD_Opt", True)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = WORLD_DELTA_S
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    report["no_rendering_mode"] = bool(world.get_settings().no_rendering_mode)
    report["map"] = str(world.get_map().name)

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

    population = None
    checks: dict[str, Any] = {}
    try:
        population_args = types.SimpleNamespace(
            number_of_vehicles=0, number_of_walkers=int(args.walkers),
            car_lights_on=False, hero=False, asynch=False,
            replenish_interval=5.0, population_log_interval=1e9,
        )
        population = traffic.TrafficPopulationManager(
            client, world, traffic_manager, population_args,
            vehicle_blueprints, walker_blueprints, spawn_points, True,
        )
        population.spawn_initial_population()
        target = int(args.walkers)
        report["target_walkers"] = target
        report["spawned"] = len(population.walkers)

        for _ in range(int(args.stabilize_ticks)):
            world.tick()
        owned = [int(r["id"]) for r in population.walkers if r.get("id") is not None]
        report["after_stabilize"] = counts(world, owned)

        # Destroy exactly one managed walker behind the manager's back.
        victim_record = next(r for r in population.walkers if r.get("id") is not None)
        victim_id = int(victim_record["id"])
        victim_controller = victim_record.get("con")
        world.get_actor(victim_id).destroy()
        world.tick()
        report["destroyed_walker_id"] = victim_id
        report["destroyed_walker_controller_id"] = (
            int(victim_controller) if victim_controller is not None else None
        )
        report["after_kill"] = counts(world, owned)

        # Normal reconciliation - nothing special, the same call maintain_population makes.
        before_ids = {int(r["id"]) for r in population.walkers if r.get("id") is not None}
        population.reconcile()
        for _ in range(int(args.settle_ticks)):
            world.tick()
        after_ids = {int(r["id"]) for r in population.walkers if r.get("id") is not None}
        lost = sorted(before_ids - after_ids)
        replacements = sorted(after_ids - before_ids)
        owned_after = sorted(after_ids)
        report["after_reconcile"] = counts(world, owned_after)
        report["lost_ids"] = lost
        report["replacement_ids"] = replacements

        replacement_records = [
            r for r in population.walkers
            if r.get("id") is not None and int(r["id"]) in set(replacements)
        ]
        controllers_ready = [bool(r.get("controller_ready", False)) for r in replacement_records]
        controller_ids = [r.get("con") for r in replacement_records]
        controller_alive = []
        for controller_id in controller_ids:
            if controller_id is None:
                controller_alive.append(False)
                continue
            try:
                actor = world.get_actor(int(controller_id))
            except RuntimeError:
                actor = None
            controller_alive.append(
                actor is not None and getattr(actor, "is_alive", False)
                and str(actor.type_id).startswith("controller.ai.walker")
            )
        report["replacement_controller_ready"] = controllers_ready
        report["replacement_controller_alive"] = controller_alive

        snapshot = report["after_reconcile"]
        checks = {
            "stabilized_at_target": report["after_stabilize"]["owned_present_in_snapshot"] == target,
            "kill_visible_in_world": report["after_kill"]["owned_present_in_snapshot"] == target - 1,
            "exactly_one_loss_recorded": lost == [victim_id],
            "exactly_one_replacement_created": len(replacements) == 1,
            "population_back_to_target": snapshot["manager_owned_records"] == target,
            "world_snapshot_back_to_target": snapshot["owned_present_in_snapshot"] == target,
            "per_id_agrees_with_snapshot":
                snapshot["per_id_lookup_alive"] == snapshot["owned_present_in_snapshot"],
            "replacement_controller_working":
                bool(replacement_records) and all(controllers_ready) and all(controller_alive),
        }
    finally:
        surviving: list[int] = []
        if population is not None:
            tracked = [int(r["id"]) for r in population.walkers if r.get("id") is not None]
            tracked += [int(r["con"]) for r in population.walkers if r.get("con") is not None]
            tracked += [int(i) for i in population.orphan_controller_ids]
            try:
                population.destroy()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
                report["cleanup_error"] = str(exc)
            try:
                world.tick()
            except RuntimeError:
                pass
            for actor_id in dict.fromkeys(tracked):
                try:
                    actor = world.get_actor(actor_id)
                except RuntimeError:
                    continue
                if actor is not None and getattr(actor, "is_alive", False):
                    surviving.append(actor_id)
            leftover = sum(1 for _ in world.get_actors().filter("walker.pedestrian.*"))
            report["cleanup"] = {
                "tracked_actors": len(dict.fromkeys(tracked)),
                "surviving_tracked": surviving,
                "walkers_left_in_world": leftover,
            }
            checks["cleanup_leaves_no_actors"] = not surviving and leftover == 0
        try:
            traffic_manager.set_synchronous_mode(False)
            world.apply_settings(original_settings)
            report["settings_restored"] = True
        except RuntimeError as exc:
            report["settings_restored"] = False
            report["settings_restore_error"] = str(exc)
            checks["settings_restored"] = False

    report["checks"] = checks
    report["status"] = "POPULATION_FIX_PASSED" if all(checks.values()) else "POPULATION_FIX_FAILED"
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "POPULATION_FIX_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

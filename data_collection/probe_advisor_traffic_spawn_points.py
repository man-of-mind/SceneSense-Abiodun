#!/usr/bin/env python3
"""Probe Town10HD_Opt traffic spawn points for physical route obstructions.

Each candidate is exercised alone under the production Traffic Manager safety
profile.  The probe owns the synchronous clock, records every collision and
trajectory sample, and restores the world before returning.  Its output is a
diagnostic input to the richer-corpus safe-spawn allowlist; it is not corpus
data and cannot satisfy a smoke gate by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import carla
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data_collection" / "experiments" / "traffic_spawn_probe"


def _role(actor: object) -> str:
    try:
        return str(actor.attributes.get("role_name", ""))
    except (AttributeError, RuntimeError):
        return ""


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--spawn-index", action="append", type=int, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--world-tick-hz", type=float, default=20.0)
    parser.add_argument("--leading-distance-m", type=float, default=12.0)
    parser.add_argument("--speed-difference-pct", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7190)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.world_tick_hz <= 0.0:
        parser.error("duration and world tick rate must be positive")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    if not str(world.get_map().name).endswith("Town10HD_Opt"):
        raise RuntimeError(f"expected Town10HD_Opt, found {world.get_map().name}")
    occupied = [
        actor
        for pattern in ("vehicle.*", "walker.*", "sensor.*", "controller.ai.walker")
        for actor in world.get_actors().filter(pattern)
    ]
    if occupied:
        raise RuntimeError(
            "spawn probe requires an empty dynamic world: "
            + ", ".join(f"{actor.id}:{actor.type_id}" for actor in occupied[:20])
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root.resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_rows: List[Dict[str, object]] = []
    collision_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    original_settings = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    spawn_points = list(world.get_map().get_spawn_points())
    ticks = int(math.ceil(args.duration_s * args.world_tick_hz))

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.world_tick_hz
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(args.leading_distance_m)
        world.tick(2.0)

        for spawn_index in args.spawn_index:
            if not 0 <= spawn_index < len(spawn_points):
                raise ValueError(f"invalid spawn index {spawn_index}")
            blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
            blueprint.set_attribute("role_name", f"traffic_spawn_probe_{spawn_index}")
            vehicle = world.try_spawn_actor(blueprint, spawn_points[spawn_index])
            if vehicle is None:
                summaries.append({"spawn_index": spawn_index, "pass": False, "failure": "spawn_failed"})
                continue
            sensor = None
            local_collisions: List[Dict[str, object]] = []
            local_trajectory: List[Dict[str, object]] = []
            try:
                collision_bp = world.get_blueprint_library().find("sensor.other.collision")
                sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

                def on_collision(event: object, index: int = spawn_index) -> None:
                    other = event.other_actor
                    impulse = event.normal_impulse
                    row = {
                        "spawn_index": index,
                        "frame_id": int(event.frame),
                        "other_actor_id": int(other.id),
                        "other_type_id": str(other.type_id),
                        "other_role_name": _role(other),
                        "normal_impulse_magnitude": float(
                            math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
                        ),
                    }
                    local_collisions.append(row)
                    collision_rows.append(row)

                sensor.listen(on_collision)
                tm.set_random_device_seed(args.seed + spawn_index)
                vehicle.set_autopilot(True, args.tm_port)
                tm.distance_to_leading_vehicle(vehicle, args.leading_distance_m)
                tm.vehicle_percentage_speed_difference(vehicle, args.speed_difference_pct)
                tm.auto_lane_change(vehicle, False)
                for _ in range(ticks):
                    frame_id = world.tick(2.0)
                    transform = vehicle.get_transform()
                    velocity = vehicle.get_velocity()
                    row = {
                        "spawn_index": spawn_index,
                        "frame_id": int(frame_id),
                        "world_x": float(transform.location.x),
                        "world_y": float(transform.location.y),
                        "speed_mps": float(
                            math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                        ),
                    }
                    local_trajectory.append(row)
                    trajectory_rows.append(row)
                trajectory = pd.DataFrame(local_trajectory)
                displacement = math.hypot(
                    float(trajectory.iloc[-1]["world_x"] - trajectory.iloc[0]["world_x"]),
                    float(trajectory.iloc[-1]["world_y"] - trajectory.iloc[0]["world_y"]),
                )
                summaries.append(
                    {
                        "spawn_index": spawn_index,
                        "pass": not local_collisions,
                        "failure": "" if not local_collisions else "collision",
                        "collision_callback_rows": len(local_collisions),
                        "first_collision_frame": (
                            int(local_collisions[0]["frame_id"]) if local_collisions else None
                        ),
                        "maximum_collision_impulse": (
                            max(row["normal_impulse_magnitude"] for row in local_collisions)
                            if local_collisions
                            else 0.0
                        ),
                        "displacement_m": float(displacement),
                        "median_speed_mps": float(trajectory["speed_mps"].median()),
                    }
                )
            finally:
                if sensor is not None:
                    try:
                        sensor.stop()
                    except RuntimeError:
                        pass
                    try:
                        sensor.destroy()
                    except RuntimeError:
                        pass
                try:
                    vehicle.destroy()
                except RuntimeError:
                    pass
                world.tick(2.0)
    finally:
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)

    _write_csv(
        output_dir / "trajectories.csv",
        trajectory_rows,
        ["spawn_index", "frame_id", "world_x", "world_y", "speed_mps"],
    )
    _write_csv(
        output_dir / "collisions.csv",
        collision_rows,
        [
            "spawn_index", "frame_id", "other_actor_id", "other_type_id",
            "other_role_name", "normal_impulse_magnitude",
        ],
    )
    payload = {
        "map": str(world.get_map().name),
        "duration_s_per_candidate": float(args.duration_s),
        "world_tick_hz": float(args.world_tick_hz),
        "tm_port": int(args.tm_port),
        "seed": int(args.seed),
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

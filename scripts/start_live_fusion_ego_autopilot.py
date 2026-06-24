#!/usr/bin/env python3
"""Attach Traffic Manager autopilot to a live fusion ego vehicle.

Run this after `carla_split_inference_udp_fusion_object_ego_client.py` has
spawned the ego vehicle with `--no-ego-freeze`. The fusion client keeps owning
the camera/radar/model/overlay loop; this helper only tells CARLA Traffic
Manager to drive that existing vehicle along a fixed route.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

PYTHONAPI_ROOT = Path(__file__).resolve().parents[2]
CARLA_AGENTS_ROOT = PYTHONAPI_ROOT / "carla"
if CARLA_AGENTS_ROOT.exists() and str(CARLA_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CARLA_AGENTS_ROOT))

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception:  # pragma: no cover - depends on CARLA install layout.
    GlobalRoutePlanner = None

import carla  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive an already-spawned live fusion ego vehicle."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--role-name", default="scenesense_fusion_ego_live")
    parser.add_argument("--wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--speed-difference-pct", type=float, default=60.0)
    parser.add_argument("--follow-distance-m", type=float, default=28.0)
    parser.add_argument("--ignore-lights-pct", type=float, default=0.0)
    parser.add_argument("--disable-lane-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-spawn-indices", default="80,85,91,94,99,80")
    parser.add_argument("--route-repeat", type=int, default=2)
    parser.add_argument("--route-point-spacing-m", type=float, default=3.0)
    return parser.parse_args()


def parse_indices(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def append_spaced(route: List["carla.Location"], loc: "carla.Location", spacing_m: float) -> None:
    point = carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z))
    if not route or route[-1].distance(point) >= float(spacing_m):
        route.append(point)


def build_route(world: "carla.World", indices: List[int], repeat: int, spacing_m: float) -> List["carla.Location"]:
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No CARLA spawn points are available.")
    invalid = [idx for idx in indices if idx < 0 or idx >= len(spawn_points)]
    if invalid:
        raise ValueError(f"Invalid spawn indices {invalid}; available range is 0..{len(spawn_points) - 1}.")

    base_indices = list(indices)
    if repeat > 1 and len(base_indices) >= 2:
        loop_body = base_indices[1:]
        for _ in range(max(0, int(repeat) - 1)):
            base_indices.extend(loop_body)

    key_points = [spawn_points[idx].location for idx in base_indices]
    route: List["carla.Location"] = []
    spacing_m = max(1.0, float(spacing_m))
    if len(key_points) < 2:
        return [carla.Location(x=float(p.x), y=float(p.y), z=float(p.z)) for p in key_points]

    if GlobalRoutePlanner is not None:
        planner = GlobalRoutePlanner(world.get_map(), spacing_m)
        for start, end in zip(key_points[:-1], key_points[1:]):
            trace = planner.trace_route(start, end)
            if not trace:
                append_spaced(route, start, spacing_m)
                append_spaced(route, end, spacing_m)
                continue
            for waypoint, _road_option in trace:
                append_spaced(route, waypoint.transform.location, spacing_m)
            append_spaced(route, end, spacing_m)
    else:
        for point in key_points:
            append_spaced(route, point, spacing_m)
    return route


def find_ego(world: "carla.World", role_name: str, timeout_s: float) -> "carla.Vehicle":
    deadline = time.time() + max(0.0, float(timeout_s))
    while True:
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") == role_name:
                return actor
        if time.time() >= deadline:
            raise RuntimeError(f"No vehicle with role_name={role_name!r} appeared within {timeout_s:.1f}s.")
        time.sleep(0.5)


def main() -> int:
    args = parse_args()
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(int(args.tm_port))

    ego = find_ego(world, str(args.role_name), float(args.wait_timeout_s))
    route = build_route(
        world,
        parse_indices(str(args.route_spawn_indices)),
        int(args.route_repeat),
        float(args.route_point_spacing_m),
    )

    ego.set_simulate_physics(True)
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
    ego.set_autopilot(True, int(args.tm_port))
    traffic_manager.ignore_lights_percentage(ego, max(0.0, min(100.0, float(args.ignore_lights_pct))))
    traffic_manager.vehicle_percentage_speed_difference(ego, float(args.speed_difference_pct))
    traffic_manager.distance_to_leading_vehicle(ego, float(args.follow_distance_m))
    if bool(args.disable_lane_change):
        try:
            traffic_manager.auto_lane_change(ego, False)
        except Exception:
            pass
    if route:
        traffic_manager.set_path(ego, route)

    print(
        "Live fusion ego autopilot enabled: "
        f"actor_id={ego.id}, role_name={args.role_name}, route_points={len(route)}, "
        f"speed_diff={float(args.speed_difference_pct):.1f}%, "
        f"follow_distance={float(args.follow_distance_m):.1f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

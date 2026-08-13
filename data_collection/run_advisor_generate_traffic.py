#!/usr/bin/env python3
"""Derived safe-spawn launcher for the read-only advisor traffic script.

The advisor script remains the population implementation. This wrapper only
filters its already-shuffled CARLA spawn-point list to the registered ego-route
corridor and raises the pairwise seed clearance before delegating to ``main``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_agent.advisor_helper_scripts.codes import generate_traffic_v1 as advisor


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
    next_x, next_y = points[(index + 1) % len(points)]
    route_x, route_y = points[index]
    route_yaw = math.degrees(math.atan2(next_y - route_y, next_x - route_x))
    heading_error = (
        float(transform.rotation.yaw) - route_yaw + 180.0
    ) % 360.0 - 180.0
    return float(distances[index]), float(abs(heading_error))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vehicle-spawn-clearance-m", type=float, required=True)
    parser.add_argument("--route-progress-csv", type=Path, required=True)
    parser.add_argument("--maximum-route-offset-m", type=float, required=True)
    parser.add_argument("--maximum-route-heading-error-deg", type=float, required=True)
    parser.add_argument("--minimum-filtered-spawn-points", type=int, required=True)
    parser.add_argument("--traffic-leading-distance-m", type=float, required=True)
    parser.add_argument("--traffic-speed-difference-pct", type=float, required=True)
    parser.add_argument("--traffic-desired-speed-mps", type=float, required=True)
    parser.add_argument("--defer-vehicle-control-to-runner", action="store_true")
    derived, remaining = parser.parse_known_args(sys.argv[1:])
    if derived.vehicle_spawn_clearance_m < 4.0:
        raise ValueError("derived vehicle spawn clearance must be at least 4 m")
    if derived.maximum_route_offset_m <= 0.0:
        raise ValueError("maximum route offset must be positive")
    if not 0.0 < derived.maximum_route_heading_error_deg < 90.0:
        raise ValueError("maximum route heading error must be within (0, 90) degrees")
    if derived.minimum_filtered_spawn_points <= 0:
        raise ValueError("minimum filtered spawn points must be positive")
    if derived.traffic_leading_distance_m < 2.5:
        raise ValueError("traffic leading distance must be at least 2.5 m")
    if not 0.0 <= derived.traffic_speed_difference_pct <= 80.0:
        raise ValueError("traffic speed difference must be within 0-80 percent")
    if not 3.0 <= derived.traffic_desired_speed_mps <= 12.0:
        raise ValueError("traffic desired speed must be within 3-12 m/s")

    route_path = derived.route_progress_csv.expanduser().resolve()
    points = _route_xy(route_path)
    original_init = advisor.TrafficPopulationManager.__init__
    original_spawn_vehicles = advisor.TrafficPopulationManager._spawn_vehicles

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
            distance_m, heading_error_deg = _route_distance_and_heading_error_deg(
                transform, points
            )
            if (
                distance_m <= float(derived.maximum_route_offset_m)
                and heading_error_deg
                <= float(derived.maximum_route_heading_error_deg)
            ):
                eligible.append(transform)
        if len(eligible) < int(derived.minimum_filtered_spawn_points):
            raise RuntimeError(
                "advisor route-corridor spawn filter has insufficient capacity: "
                f"eligible={len(eligible)}, "
                f"required={int(derived.minimum_filtered_spawn_points)}"
            )
        advisor.logging.info(
            "Derived route-corridor spawn filter: eligible=%d/%d offset<=%.1fm "
            "heading_error<=%.1fdeg pairwise_clearance>=%.1fm route=%s",
            len(eligible),
            len(all_points),
            float(derived.maximum_route_offset_m),
            float(derived.maximum_route_heading_error_deg),
            float(derived.vehicle_spawn_clearance_m),
            route_path,
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

    def safe_spawn_vehicles(
        population: object, count: int, shuffle_spawn_points: bool = True
    ) -> List[int]:
        actor_ids = original_spawn_vehicles(population, count, shuffle_spawn_points)
        live = population._live_actor_map(actor_ids)
        for actor in live.values():
            population.traffic_manager.distance_to_leading_vehicle(
                actor, float(derived.traffic_leading_distance_m)
            )
            population.traffic_manager.vehicle_percentage_speed_difference(
                actor, float(derived.traffic_speed_difference_pct)
            )
            population.traffic_manager.set_desired_speed(
                actor, float(derived.traffic_desired_speed_mps)
            )
            population.traffic_manager.auto_lane_change(actor, False)
            if derived.defer_vehicle_control_to_runner:
                actor.set_autopilot(False, population.tm_port)
        return actor_ids

    advisor.VEHICLE_SPAWN_CLEARANCE_M = float(derived.vehicle_spawn_clearance_m)
    advisor.TrafficPopulationManager.__init__ = filtered_init
    advisor.TrafficPopulationManager._spawn_vehicles = safe_spawn_vehicles
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *remaining]
    try:
        advisor.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

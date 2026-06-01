#!/usr/bin/env python3
"""Rank spawn points for curbside hidden-pedestrian SceneSense demos."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ABIODUN_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scenesense_scenario_harness as harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank CARLA spawn points for non-intersection curbside occlusion demos."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="")
    parser.add_argument("--load-town", action="store_true")
    parser.add_argument(
        "--no-fallback-current-world",
        action="store_true",
        help="Exit instead of falling back to the current world if --load-town fails.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--route-distance-m", type=float, default=80.0)
    parser.add_argument(
        "--output-root",
        default=str(ABIODUN_DIR / "metrics_logs" / "scenesense_scenarios"),
    )
    return parser.parse_args()


def route_yaw_delta(route) -> float:
    if len(route) < 2:
        return 999.0
    start_yaw = float(route[0].transform.rotation.yaw)
    return max(
        harness.angular_difference_deg(start_yaw, float(waypoint.transform.rotation.yaw))
        for waypoint in route
    )


def nearest_traffic_light_gap(world, location) -> float:
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    if not lights:
        return 999.0
    return min(float(location.distance(actor.get_location())) for actor in lights)


def score_row(row: Dict[str, object]) -> float:
    score = 0.0
    score += float(row["route_yaw_delta_deg"]) * 3.0
    score += max(0.0, 50.0 - float(row["nearest_traffic_light_m"])) * 2.0
    score += max(0.0, 45.0 - float(row["branch_distance_m"])) * 5.0
    score += abs(float(row["route_length_m"]) - 80.0) * 0.5
    return round(score, 3)


def scenario_command(town: str, spawn_index: int, include_load_town: bool) -> str:
    lines = [
        "python3 scenesense_scenarios/scenesense_scenario_harness.py \\",
        "  --scenario curbside_parked_vehicle_pedestrian_occlusion \\",
    ]
    if include_load_town:
        lines.append(f"  --load-town --town {town} \\")
    lines.extend(
        [
            "  --anchor-source spawn_point \\",
            f"  --anchor-spawn-index {spawn_index} \\",
            f"  --ego-spawn-index {spawn_index} \\",
            "  --seed 7 \\",
            "  --duration-s 60 \\",
            "  --ego-sensors \\",
            "  --ego-camera-preview \\",
            "  --scripted-ego-drive \\",
            "  --ego-drive-mode waypoint \\",
            "  --ego-route-choice straight \\",
            "  --ego-target-speed 4.2 \\",
            "  --target-crossing \\",
            "  --target-crossing-delay-s 1.0 \\",
            "  --target-crossing-speed 1.8 \\",
            "  --target-crossing-trigger-distance-m 24.0 \\",
            "  --stop-on-target-collision \\",
            "  --spectator-focus conflict",
        ]
    )
    return "\n".join(lines)


def server_available_maps(client) -> List[str]:
    try:
        return [str(item) for item in client.get_available_maps()]
    except Exception:
        return []


def load_or_get_world(client, args):
    requested_town = str(args.town or "").strip()
    if args.load_town:
        if not requested_town:
            raise RuntimeError("--load-town requires --town.")
        try:
            return client.load_world(requested_town), requested_town, True
        except RuntimeError as exc:
            maps = server_available_maps(client)
            message = [
                f"Unable to load town '{requested_town}': {exc}",
                "This usually means the packaged .umap for that town is not installed.",
            ]
            if maps:
                message.append("CARLA server available maps:")
                message.extend(f"  {name}" for name in maps)
            if args.no_fallback_current_world:
                raise RuntimeError("\n".join(message)) from exc
            print("\n".join(message))
            print("Falling back to the current CARLA world for scouting.")
    world = client.get_world()
    return world, str(world.get_map().name).split("/")[-1], False


def main() -> int:
    args = parse_args()
    carla = harness._bootstrap_carla()
    harness.carla = carla
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(15.0)
    world, town_label, include_load_town = load_or_get_world(client, args)
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    rows: List[Dict[str, object]] = []
    for index, spawn_point in enumerate(spawn_points):
        try:
            waypoint = world_map.get_waypoint(
                spawn_point.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            waypoint = None
        if waypoint is None:
            continue
        route = harness.generate_route_waypoints(
            waypoint,
            "straight",
            total_distance_m=float(args.route_distance_m),
        )
        if len(route) < 14:
            continue
        route_length = 0.0
        previous = route[0]
        for current in route[1:]:
            route_length += previous.transform.location.distance(current.transform.location)
            previous = current
        branch_distance = harness.first_route_branch_distance(waypoint, "right")
        branch_distance = 999.0 if branch_distance is None else float(branch_distance)
        row: Dict[str, object] = {
            "score": 0.0,
            "town": str(town_label),
            "spawn_index": int(index),
            "spawn_x": round(float(spawn_point.location.x), 3),
            "spawn_y": round(float(spawn_point.location.y), 3),
            "spawn_yaw": round(float(spawn_point.rotation.yaw), 3),
            "route_length_m": round(float(route_length), 3),
            "route_yaw_delta_deg": round(float(route_yaw_delta(route)), 3),
            "nearest_traffic_light_m": round(float(nearest_traffic_light_gap(world, spawn_point.location)), 3),
            "branch_distance_m": round(float(branch_distance), 3),
        }
        row["score"] = score_row(row)
        rows.append(row)

    rows = sorted(rows, key=lambda row: float(row["score"]))
    if not rows:
        print("No curbside spawn candidates found.")
        return 1

    out_dir = Path(args.output_root).expanduser().resolve() / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_curbside_spawn_scout_{harness.sanitize_token(town_label)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "curbside_spawn_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# SceneSense Curbside Spawn Scout",
        "",
        "Lower score is better. Prefer long straight routes, far from traffic lights, with no early branch.",
        "",
        "| rank | score | spawn_index | yaw_delta | nearest_tl_m | branch_m | spawn |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        md_lines.append(
            "| {rank} | {score} | {spawn_index} | {route_yaw_delta_deg} | "
            "{nearest_traffic_light_m} | {branch_distance_m} | "
            "({spawn_x}, {spawn_y}, yaw={spawn_yaw}) |".format(rank=rank, **row)
        )
    md_lines.extend(["", "## Trial Commands", ""])
    for rank, row in enumerate(rows[: min(5, int(args.top))], start=1):
        md_lines.extend(
            [
                f"### Candidate {rank}: spawn {row['spawn_index']}",
                "",
                "```bash",
                scenario_command(str(town_label), int(row["spawn_index"]), bool(include_load_town)),
                "```",
                "",
            ]
        )
    (out_dir / "curbside_spawn_candidates.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote scout results to {out_dir}")
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        print(
            f"{rank:02d}. score={row['score']} spawn={row['spawn_index']} "
            f"yaw_delta={row['route_yaw_delta_deg']} nearest_tl={row['nearest_traffic_light_m']} "
            f"branch={row['branch_distance_m']} loc=({row['spawn_x']}, {row['spawn_y']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

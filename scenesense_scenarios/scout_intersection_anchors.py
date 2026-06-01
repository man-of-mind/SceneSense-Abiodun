#!/usr/bin/env python3
"""Scout CARLA anchors for clean SceneSense intersection occlusion scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ABIODUN_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scenesense_scenario_harness as harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank traffic-light anchors/spawn points for intersection occlusion scenarios."
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--traffic-lights", default=str(harness.TRAFFIC_LIGHTS_JSON))
    parser.add_argument("--radius-m", type=float, default=95.0, help="Spawn search radius around each anchor.")
    parser.add_argument("--min-distance-m", type=float, default=20.0, help="Minimum ego spawn distance from anchor.")
    parser.add_argument("--max-distance-m", type=float, default=90.0, help="Maximum ego spawn distance from anchor.")
    parser.add_argument("--route-distance-m", type=float, default=90.0, help="Distance to roll out route waypoints.")
    parser.add_argument("--top", type=int, default=20, help="Number of candidates to print.")
    parser.add_argument(
        "--route-choice",
        choices=("left", "right", "both"),
        default="both",
        help="Route branch type to scout.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ABIODUN_DIR / "metrics_logs" / "scenesense_scenarios"),
        help="Root for scout outputs.",
    )
    return parser.parse_args()


def load_traffic_lights(path: Path) -> List[Dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if "id" in row and "location" in row]


def carla_location(carla_module, row: Dict[str, object]):
    loc = row["location"]
    return carla_module.Location(
        x=float(loc["x"]),
        y=float(loc["y"]),
        z=float(loc.get("z", 0.0)),
    )


def route_heading_delta(route) -> float:
    if len(route) < 2:
        return 0.0
    first = float(route[0].transform.rotation.yaw)
    deltas = [
        abs(harness.signed_angular_difference_deg(float(wp.transform.rotation.yaw), first))
        for wp in route[1:]
    ]
    return max(deltas) if deltas else 0.0


def min_route_distance_to_anchor(route, anchor) -> float:
    if not route:
        return float("inf")
    return min(float(wp.transform.location.distance(anchor)) for wp in route)


def route_length(route) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    previous = route[0]
    for waypoint in route[1:]:
        total += previous.transform.location.distance(waypoint.transform.location)
        previous = waypoint
    return float(total)


def score_candidate(
    route_choice: str,
    spawn_distance: float,
    branch_distance: Optional[float],
    min_anchor_distance: float,
    heading_delta: float,
    length_m: float,
) -> float:
    score = 0.0
    if branch_distance is None:
        score += 500.0
    else:
        score += abs(branch_distance - 22.0) * 2.0
    score += abs(spawn_distance - 45.0) * 0.4
    score += min_anchor_distance * 1.5
    score += max(0.0, 45.0 - heading_delta) * 4.0
    score += max(0.0, 55.0 - length_m) * 3.0
    if route_choice == "left" and heading_delta < 35.0:
        score += 100.0
    if route_choice == "right" and heading_delta < 35.0:
        score += 100.0
    return float(score)


def build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    carla = harness._bootstrap_carla()
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    world_map = world.get_map()
    traffic_lights = load_traffic_lights(Path(args.traffic_lights).expanduser().resolve())
    spawn_points = world_map.get_spawn_points()
    route_choices = ("left", "right") if args.route_choice == "both" else (args.route_choice,)
    rows: List[Dict[str, object]] = []

    for light in traffic_lights:
        anchor = carla_location(carla, light)
        candidates = [
            sp
            for sp in spawn_points
            if args.min_distance_m <= sp.location.distance(anchor) <= args.radius_m
            and sp.location.distance(anchor) <= args.max_distance_m
        ]
        for sp_index, spawn_point in enumerate(candidates):
            spawn_distance = float(spawn_point.location.distance(anchor))
            bearing_to_anchor = harness.vector_bearing_deg(spawn_point.location, anchor)
            heading_to_anchor_error = harness.angular_difference_deg(
                float(spawn_point.rotation.yaw),
                bearing_to_anchor,
            )
            if heading_to_anchor_error > 125.0:
                continue
            try:
                start_wp = world_map.get_waypoint(
                    spawn_point.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except RuntimeError:
                continue
            if start_wp is None:
                continue

            for route_choice in route_choices:
                branch_distance = harness.first_route_branch_distance(start_wp, route_choice)
                route = harness.generate_route_waypoints(
                    start_wp,
                    route_choice,
                    total_distance_m=float(args.route_distance_m),
                )
                heading_delta = route_heading_delta(route)
                min_anchor_distance = min_route_distance_to_anchor(route, anchor)
                length_m = route_length(route)
                score = score_candidate(
                    route_choice,
                    spawn_distance,
                    branch_distance,
                    min_anchor_distance,
                    heading_delta,
                    length_m,
                )
                rows.append(
                    {
                        "score": round(score, 3),
                        "traffic_light_id": str(light["id"]),
                        "route_choice": route_choice,
                        "spawn_index": sp_index,
                        "spawn_x": round(float(spawn_point.location.x), 3),
                        "spawn_y": round(float(spawn_point.location.y), 3),
                        "spawn_z": round(float(spawn_point.location.z), 3),
                        "spawn_yaw": round(float(spawn_point.rotation.yaw), 3),
                        "anchor_x": round(float(anchor.x), 3),
                        "anchor_y": round(float(anchor.y), 3),
                        "spawn_distance_m": round(spawn_distance, 3),
                        "heading_to_anchor_error_deg": round(float(heading_to_anchor_error), 3),
                        "branch_distance_m": ""
                        if branch_distance is None
                        else round(float(branch_distance), 3),
                        "route_heading_delta_deg": round(float(heading_delta), 3),
                        "route_min_anchor_distance_m": round(float(min_anchor_distance), 3),
                        "route_length_m": round(float(length_m), 3),
                    }
                )
    return sorted(rows, key=lambda row: float(row["score"]))


def write_outputs(rows: List[Dict[str, object]], output_root: Path, top: int) -> Path:
    out_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_intersection_anchor_scout"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "intersection_anchor_candidates.csv"
    fieldnames = list(rows[0].keys()) if rows else ["score"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# SceneSense Intersection Anchor Scout",
        "",
        "Lower score is better. Prefer rows with non-empty `branch_distance_m`, high route heading change, and small route-to-anchor distance.",
        "",
        "| rank | score | tl_id | route | branch_m | heading_delta | min_anchor_m | spawn |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:top], start=1):
        md_lines.append(
            "| {rank} | {score} | {traffic_light_id} | {route_choice} | {branch_distance_m} | "
            "{route_heading_delta_deg} | {route_min_anchor_distance_m} | ({spawn_x}, {spawn_y}, yaw={spawn_yaw}) |".format(
                rank=rank,
                **row,
            )
        )
    (out_dir / "intersection_anchor_candidates.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    if not rows:
        print("No intersection candidates found.")
        return 1
    out_dir = write_outputs(rows, Path(args.output_root).expanduser().resolve(), int(args.top))
    print(f"Wrote scout results to {out_dir}")
    print("Top candidates:")
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        print(
            f"{rank:02d}. score={row['score']} tl={row['traffic_light_id']} "
            f"route={row['route_choice']} branch_m={row['branch_distance_m']} "
            f"heading_delta={row['route_heading_delta_deg']} min_anchor_m={row['route_min_anchor_distance_m']} "
            f"spawn=({row['spawn_x']}, {row['spawn_y']}, yaw={row['spawn_yaw']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

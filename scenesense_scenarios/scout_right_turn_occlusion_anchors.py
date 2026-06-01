#!/usr/bin/env python3
"""Rank CARLA anchors for right-turn hidden-pedestrian SceneSense demos."""

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
        description=(
            "Rank traffic-light anchors for a right-turn hidden-pedestrian scenario. "
            "The score favors nearby crosswalk geometry and a straight approach for the truck/queue."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--traffic-lights", default=str(harness.TRAFFIC_LIGHTS_JSON))
    parser.add_argument("--anchor-radius-m", type=float, default=90.0)
    parser.add_argument("--ego-distance-m", type=float, default=42.0)
    parser.add_argument("--route-distance-m", type=float, default=85.0)
    parser.add_argument("--top", type=int, default=20)
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


def nearest_driving_lane_gap(world_map, location, carla_module) -> Optional[float]:
    try:
        waypoint = world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
    except RuntimeError:
        return None
    if waypoint is None:
        return None
    return float(location.distance(waypoint.transform.location))


def score_row(row: Dict[str, object]) -> float:
    score = 0.0
    if row["branch_distance_m"] == "":
        score += 500.0
    score += float(row["crosswalk_gap_m"] if row["crosswalk_gap_m"] != "" else 40.0) * 9.0
    score += float(row["occluder_yaw_delta_deg"]) * 2.5
    score += max(0.0, float(row["occluder_lane_gap_m"] if row["occluder_lane_gap_m"] != "" else 10.0) - 2.5) * 12.0
    score += abs(float(row["spawn_distance_m"]) - 42.0) * 0.2
    score += abs(float(row["conflict_distance_m"]) - float(row["desired_conflict_distance_m"])) * 1.0
    score += max(0.0, 8.0 - float(row["occluder_to_conflict_distance_m"])) * 10.0
    return float(score)


def build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    carla = harness._bootstrap_carla()
    harness.carla = carla
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    world_map = world.get_map()
    traffic_lights = load_traffic_lights(Path(args.traffic_lights).expanduser().resolve())
    rows: List[Dict[str, object]] = []

    for light in traffic_lights:
        anchor = carla_location(carla, light)
        candidates = harness.spawn_points_near(
            world,
            anchor,
            radius_m=float(args.anchor_radius_m),
            min_distance_m=6.0,
        )
        if not candidates:
            continue
        try:
            ego_sp = harness.choose_ego_spawn_toward_anchor(
                world,
                candidates,
                anchor,
                target_distance_m=float(args.ego_distance_m),
                route_choice="right",
            )
            start_wp = world_map.get_waypoint(
                ego_sp.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            continue
        if start_wp is None:
            continue

        branch_distance = harness.first_route_branch_distance(start_wp, "right")
        route = harness.generate_route_waypoints(
            start_wp,
            "right",
            total_distance_m=float(args.route_distance_m),
        )
        if len(route) < 2:
            continue
        turn_entry_distance = branch_distance or 20.0
        desired_conflict_distance = turn_entry_distance + 4.0
        crosswalk_candidate = harness.route_crosswalk_candidate(
            world_map,
            route,
            min_distance_m=max(10.0, turn_entry_distance - 3.0),
            max_distance_m=min(42.0, turn_entry_distance + 14.0),
            desired_distance_m=desired_conflict_distance,
        )
        if crosswalk_candidate is not None:
            conflict_distance, conflict_tf, crosswalk_gap_m = crosswalk_candidate
        else:
            conflict_distance = harness.clamp(desired_conflict_distance, 20.0, 34.0)
            conflict_tf = harness.route_transform_at_distance(route, conflict_distance)
            crosswalk_gap_m = None

        occluder_distance = max(10.0, min(conflict_distance - 5.5, turn_entry_distance + 2.0))
        occluder_tf = harness.route_transform_at_distance(route, occluder_distance)
        occluder_lateral_offset_m = 4.6
        target_start_lateral_offset_m = 6.4
        occluder_location = harness.offset_location(
            occluder_tf.location,
            float(occluder_tf.rotation.yaw),
            forward_m=0.0,
            right_m=occluder_lateral_offset_m,
            z_offset_m=0.4,
        )
        target_start_location = harness.offset_location(
            conflict_tf.location,
            float(conflict_tf.rotation.yaw),
            forward_m=0.0,
            right_m=target_start_lateral_offset_m,
            z_offset_m=1.0,
        )
        occluder_lane_gap = nearest_driving_lane_gap(world_map, occluder_location, carla)
        occluder_yaw_delta = harness.angular_difference_deg(
            float(ego_sp.rotation.yaw),
            float(occluder_tf.rotation.yaw),
        )
        row: Dict[str, object] = {
            "score": 0.0,
            "traffic_light_id": str(light["id"]),
            "route_choice": "right",
            "spawn_x": round(float(ego_sp.location.x), 3),
            "spawn_y": round(float(ego_sp.location.y), 3),
            "spawn_yaw": round(float(ego_sp.rotation.yaw), 3),
            "spawn_distance_m": round(float(ego_sp.location.distance(anchor)), 3),
            "branch_distance_m": "" if branch_distance is None else round(float(branch_distance), 3),
            "desired_conflict_distance_m": round(float(desired_conflict_distance), 3),
            "conflict_distance_m": round(float(conflict_distance), 3),
            "conflict_x": round(float(conflict_tf.location.x), 3),
            "conflict_y": round(float(conflict_tf.location.y), 3),
            "crosswalk_gap_m": "" if crosswalk_gap_m is None else round(float(crosswalk_gap_m), 3),
            "occluder_distance_m": round(float(occluder_distance), 3),
            "occluder_to_conflict_distance_m": round(float(conflict_distance - occluder_distance), 3),
            "occluder_x": round(float(occluder_location.x), 3),
            "occluder_y": round(float(occluder_location.y), 3),
            "occluder_yaw": round(float(occluder_tf.rotation.yaw), 3),
            "occluder_yaw_delta_deg": round(float(occluder_yaw_delta), 3),
            "occluder_lane_gap_m": "" if occluder_lane_gap is None else round(float(occluder_lane_gap), 3),
            "target_start_x": round(float(target_start_location.x), 3),
            "target_start_y": round(float(target_start_location.y), 3),
        }
        row["score"] = round(score_row(row), 3)
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["score"]))


def scenario_command(traffic_light_id: str) -> str:
    return "\n".join(
        [
            "python3 scenesense_scenarios/scenesense_scenario_harness.py \\",
            "  --scenario right_turn_truck_pedestrian_occlusion \\",
            f"  --traffic-light-id {traffic_light_id} \\",
            "  --seed 7 \\",
            "  --duration-s 60 \\",
            "  --ego-sensors \\",
            "  --ego-camera-preview \\",
            "  --scripted-ego-drive \\",
            "  --ego-drive-mode waypoint \\",
            "  --ego-route-choice right \\",
            "  --ego-target-speed 4.5 \\",
            "  --target-crossing \\",
            "  --target-crossing-delay-s 1.0 \\",
            "  --target-crossing-speed 2.2 \\",
            "  --target-crossing-trigger-distance-m 12.0 \\",
            "  --stop-on-target-collision \\",
            "  --spectator-focus conflict",
        ]
    )


def write_outputs(rows: List[Dict[str, object]], output_root: Path, top: int) -> Path:
    out_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_right_turn_occlusion_scout"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "right_turn_occlusion_candidates.csv"
    fieldnames = list(rows[0].keys()) if rows else ["score"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# SceneSense Right-Turn Occlusion Anchor Scout",
        "",
        "Lower score is better. Prefer low `crosswalk_gap_m`, low `occluder_yaw_delta_deg`, and low `occluder_lane_gap_m`.",
        "",
        "| rank | score | tl_id | branch_m | crosswalk_gap | yaw_delta | lane_gap | conflict_m | spawn |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:top], start=1):
        md_lines.append(
            "| {rank} | {score} | {traffic_light_id} | {branch_distance_m} | {crosswalk_gap_m} | "
            "{occluder_yaw_delta_deg} | {occluder_lane_gap_m} | {conflict_distance_m} | "
            "({spawn_x}, {spawn_y}, yaw={spawn_yaw}) |".format(rank=rank, **row)
        )
    if rows:
        md_lines.extend(["", "## Trial Commands", ""])
        for rank, row in enumerate(rows[: min(5, top)], start=1):
            md_lines.extend(
                [
                    f"### Candidate {rank}: traffic light {row['traffic_light_id']}",
                    "",
                    "```bash",
                    scenario_command(str(row["traffic_light_id"])),
                    "```",
                    "",
                ]
            )
    (out_dir / "right_turn_occlusion_candidates.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    if not rows:
        print("No right-turn occlusion candidates found.")
        return 1
    out_dir = write_outputs(rows, Path(args.output_root).expanduser().resolve(), int(args.top))
    print(f"Wrote scout results to {out_dir}")
    print("Top candidates:")
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        print(
            f"{rank:02d}. score={row['score']} tl={row['traffic_light_id']} "
            f"branch_m={row['branch_distance_m']} crosswalk_gap={row['crosswalk_gap_m']} "
            f"yaw_delta={row['occluder_yaw_delta_deg']} lane_gap={row['occluder_lane_gap_m']} "
            f"conflict_m={row['conflict_distance_m']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

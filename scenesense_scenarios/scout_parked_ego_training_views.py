#!/usr/bin/env python3
"""Scout parked-ego views for RGB+radar training data collection.

The existing intersection scouts focus on moving-ego scenario routes. This
script ranks fixed ego-camera poses instead: a parked vehicle should face a
dense intersection, see traffic from several directions, and include crosswalk
geometry for pedestrian samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ABIODUN_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scenesense_scenario_harness as harness


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank CARLA spawn points for parked-ego RGB+radar training views. "
            "Higher quality_score is better."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--traffic-lights", default=str(harness.TRAFFIC_LIGHTS_JSON))
    parser.add_argument("--top", type=int, default=12, help="Number of rows to print in the summary.")
    parser.add_argument("--min-distance-m", type=float, default=12.0)
    parser.add_argument("--max-distance-m", type=float, default=45.0)
    parser.add_argument(
        "--target-distance-m",
        type=float,
        default=24.0,
        help="Preferred parked ego distance from the traffic-light/intersection anchor.",
    )
    parser.add_argument(
        "--coverage-range-m",
        type=float,
        default=95.0,
        help="Range used when scoring visible road spawn/crosswalk coverage.",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=120.0,
        help="Candidate camera FoV. 110-120 is usually cleaner than a single 180-degree camera.",
    )
    parser.add_argument(
        "--max-yaw-offset-deg",
        type=float,
        default=95.0,
        help="Reject candidates that must rotate more than this to face the intersection anchor.",
    )
    parser.add_argument(
        "--forward-offsets-m",
        type=parse_float_list,
        default=[0.0],
        help="Comma-separated ego spawn forward offsets to test.",
    )
    parser.add_argument(
        "--right-offsets-m",
        type=parse_float_list,
        default=[0.0, 3.0, -3.0],
        help="Comma-separated ego spawn right offsets to test.",
    )
    parser.add_argument(
        "--preferred-lateral-side",
        choices=("any", "right", "left"),
        default="any",
        help=(
            "Bias the scout toward a parked ego shifted to this side of the "
            "map spawn pose. 'right' means positive --ego-spawn-right-offset-m."
        ),
    )
    parser.add_argument(
        "--require-preferred-lateral-side",
        action="store_true",
        help="If set, discard candidates that are not on the preferred lateral side.",
    )
    parser.add_argument(
        "--lateral-side-bonus",
        type=float,
        default=12.0,
        help="Quality-score bonus for candidates on the preferred lateral side.",
    )
    parser.add_argument("--npc-vehicles", type=int, default=35)
    parser.add_argument("--npc-pedestrians", type=int, default=45)
    parser.add_argument("--spawn-radius", type=float, default=95.0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument(
        "--output-root",
        default=str(ABIODUN_DIR / "metrics_logs" / "parked_ego_view_scout"),
        help="Root directory for CSV/Markdown scout outputs.",
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


def offset_location(carla_module, origin, yaw_deg: float, forward_m: float, right_m: float):
    yaw_rad = math.radians(float(yaw_deg))
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)
    right_x = math.cos(yaw_rad + math.pi / 2.0)
    right_y = math.sin(yaw_rad + math.pi / 2.0)
    return carla_module.Location(
        x=float(origin.x) + forward_x * float(forward_m) + right_x * float(right_m),
        y=float(origin.y) + forward_y * float(forward_m) + right_y * float(right_m),
        z=float(origin.z),
    )


def locations_in_camera_view(
    *,
    ego_location,
    camera_yaw_deg: float,
    fov_deg: float,
    max_range_m: float,
    locations: Iterable[object],
    min_range_m: float = 4.0,
) -> Tuple[int, float, Tuple[int, int, int]]:
    angles: List[float] = []
    half_fov = float(fov_deg) / 2.0
    for loc in locations:
        distance = float(ego_location.distance(loc))
        if distance < float(min_range_m) or distance > float(max_range_m):
            continue
        bearing = harness.vector_bearing_deg(ego_location, loc)
        rel = harness.signed_angular_difference_deg(bearing, float(camera_yaw_deg))
        if abs(rel) <= half_fov:
            angles.append(float(rel))

    if not angles:
        return 0, 0.0, (0, 0, 0)

    left = sum(1 for angle in angles if angle < -half_fov / 3.0)
    center = sum(1 for angle in angles if -half_fov / 3.0 <= angle <= half_fov / 3.0)
    right = sum(1 for angle in angles if angle > half_fov / 3.0)
    spread = max(angles) - min(angles)
    return len(angles), float(spread), (left, center, right)


def score_candidate(
    *,
    distance_m: float,
    target_distance_m: float,
    yaw_offset_deg: float,
    road_points: int,
    crosswalk_points: int,
    road_spread_deg: float,
    road_sectors: Sequence[int],
    lateral_side_bonus: float = 0.0,
) -> float:
    active_sectors = sum(1 for count in road_sectors if int(count) > 0)
    sector_balance = max(road_sectors) - min(road_sectors) if road_sectors else 0
    quality = 0.0
    quality += min(int(road_points), 45) * 2.4
    quality += min(int(crosswalk_points), 100) * 0.35
    quality += min(float(road_spread_deg), 120.0) * 0.55
    quality += active_sectors * 24.0
    quality -= abs(float(distance_m) - float(target_distance_m)) * 1.8
    quality -= abs(float(yaw_offset_deg)) * 0.12
    quality -= float(sector_balance) * 0.25
    quality += float(lateral_side_bonus)
    return float(quality)


def lateral_side(right_offset_m: float) -> str:
    if float(right_offset_m) > 0.25:
        return "right"
    if float(right_offset_m) < -0.25:
        return "left"
    return "center"


def numeric_token(prefix: str, value: float) -> str:
    value = float(value)
    sign = "p" if value > 0 else "m" if value < 0 else ""
    magnitude = str(round(abs(value), 2)).replace(".", "p")
    return f"{prefix}{sign}{magnitude}"


def build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    carla = harness._bootstrap_carla()
    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    world_map = world.get_map()
    spawn_points = list(world_map.get_spawn_points())
    traffic_lights = load_traffic_lights(Path(args.traffic_lights).expanduser().resolve())

    try:
        crosswalk_locations = list(world_map.get_crosswalks())
    except RuntimeError:
        crosswalk_locations = []

    rows: List[Dict[str, object]] = []
    road_locations = [point.location for point in spawn_points]

    for light in traffic_lights:
        anchor = carla_location(carla, light)
        for spawn_index, spawn_point in enumerate(spawn_points):
            base_distance = float(spawn_point.location.distance(anchor))
            offset_bound = math.hypot(
                max(abs(v) for v in args.forward_offsets_m),
                max(abs(v) for v in args.right_offsets_m),
            )
            if base_distance > float(args.max_distance_m) + offset_bound + 5.0:
                continue
            for forward_offset_m in args.forward_offsets_m:
                for right_offset_m in args.right_offsets_m:
                    side = lateral_side(float(right_offset_m))
                    preferred = str(args.preferred_lateral_side)
                    if bool(args.require_preferred_lateral_side) and preferred != "any" and side != preferred:
                        continue
                    ego_location = offset_location(
                        carla,
                        spawn_point.location,
                        float(spawn_point.rotation.yaw),
                        float(forward_offset_m),
                        float(right_offset_m),
                    )
                    distance_m = float(ego_location.distance(anchor))
                    if not (float(args.min_distance_m) <= distance_m <= float(args.max_distance_m)):
                        continue

                    bearing_to_anchor = harness.vector_bearing_deg(ego_location, anchor)
                    yaw_offset_deg = harness.signed_angular_difference_deg(
                        bearing_to_anchor,
                        float(spawn_point.rotation.yaw),
                    )
                    if abs(yaw_offset_deg) > float(args.max_yaw_offset_deg):
                        continue
                    camera_yaw_world = float(spawn_point.rotation.yaw) + float(yaw_offset_deg)

                    road_count, road_spread, road_sectors = locations_in_camera_view(
                        ego_location=ego_location,
                        camera_yaw_deg=camera_yaw_world,
                        fov_deg=float(args.camera_fov),
                        max_range_m=float(args.coverage_range_m),
                        locations=road_locations,
                    )
                    crosswalk_count, crosswalk_spread, _ = locations_in_camera_view(
                        ego_location=ego_location,
                        camera_yaw_deg=camera_yaw_world,
                        fov_deg=float(args.camera_fov),
                        max_range_m=float(args.coverage_range_m),
                        locations=crosswalk_locations,
                        min_range_m=1.0,
                    )
                    side_bonus = (
                        float(args.lateral_side_bonus)
                        if preferred != "any" and side == preferred
                        else 0.0
                    )
                    quality = score_candidate(
                        distance_m=distance_m,
                        target_distance_m=float(args.target_distance_m),
                        yaw_offset_deg=yaw_offset_deg,
                        road_points=road_count,
                        crosswalk_points=crosswalk_count,
                        road_spread_deg=road_spread,
                        road_sectors=road_sectors,
                        lateral_side_bonus=side_bonus,
                    )
                    rows.append(
                        {
                            "quality_score": round(quality, 3),
                            "parking_side": side,
                            "traffic_light_id": str(light["id"]),
                            "spawn_index": int(spawn_index),
                            "spawn_x": round(float(spawn_point.location.x), 3),
                            "spawn_y": round(float(spawn_point.location.y), 3),
                            "spawn_z": round(float(spawn_point.location.z), 3),
                            "spawn_yaw": round(float(spawn_point.rotation.yaw), 3),
                            "ego_x": round(float(ego_location.x), 3),
                            "ego_y": round(float(ego_location.y), 3),
                            "anchor_x": round(float(anchor.x), 3),
                            "anchor_y": round(float(anchor.y), 3),
                            "distance_to_anchor_m": round(distance_m, 3),
                            "ego_spawn_forward_offset_m": round(float(forward_offset_m), 3),
                            "ego_spawn_right_offset_m": round(float(right_offset_m), 3),
                            "ego_spawn_yaw_offset_deg": round(float(yaw_offset_deg), 3),
                            "camera_yaw_world_deg": round(float(camera_yaw_world), 3),
                            "camera_fov_deg": round(float(args.camera_fov), 3),
                            "road_spawn_points_in_view": int(road_count),
                            "road_angle_spread_deg": round(float(road_spread), 3),
                            "crosswalk_points_in_view": int(crosswalk_count),
                            "crosswalk_angle_spread_deg": round(float(crosswalk_spread), 3),
                            "left_road_points": int(road_sectors[0]),
                            "center_road_points": int(road_sectors[1]),
                            "right_road_points": int(road_sectors[2]),
                        }
                    )

    return sorted(rows, key=lambda row: float(row["quality_score"]), reverse=True)


def command_block(row: Dict[str, object], args: argparse.Namespace, max_samples: int) -> str:
    fwd_token = numeric_token("fwd", float(row["ego_spawn_forward_offset_m"]))
    right_token = numeric_token("right", float(row["ego_spawn_right_offset_m"]))
    experiment_id = (
        f"parked_ego_training_tl{row['traffic_light_id']}_"
        f"spawn{row['spawn_index']}_{right_token}_{fwd_token}_{max_samples}samp"
    )
    return "\n".join(
        [
            "```bash",
            "python3 carla_collect_parked_ego_fusion_training_data.py \\",
            f"  --experiment-id {experiment_id} \\",
            f"  --max-samples {int(max_samples)} \\",
            "  --sample-stride 1 \\",
            "  --fps 10 \\",
            f"  --camera-width {int(args.camera_width)} \\",
            f"  --camera-height {int(args.camera_height)} \\",
            f"  --camera-fov {float(args.camera_fov):.1f} \\",
            "  --model-input-width 768 \\",
            "  --model-input-height 432 \\",
            f"  --ego-spawn-index {int(row['spawn_index'])} \\",
            f"  --ego-spawn-forward-offset-m {float(row['ego_spawn_forward_offset_m']):.2f} \\",
            f"  --ego-spawn-right-offset-m {float(row['ego_spawn_right_offset_m']):.2f} \\",
            f"  --ego-spawn-yaw-offset-deg {float(row['ego_spawn_yaw_offset_deg']):.2f} \\",
            "  --ego-camera-x 1.8 \\",
            "  --ego-camera-y 0.0 \\",
            "  --ego-camera-z 1.55 \\",
            "  --ego-camera-pitch -4.0 \\",
            "  --ego-camera-yaw 0.0 \\",
            f"  --radar-hfov {float(args.camera_fov):.1f} \\",
            "  --radar-vfov 30 \\",
            "  --radar-range 120 \\",
            f"  --npc-vehicles {int(args.npc_vehicles)} \\",
            f"  --npc-pedestrians {int(args.npc_pedestrians)} \\",
            f"  --spawn-radius {float(args.spawn_radius):.1f} \\",
            "  --include-pedestrians",
            "```",
        ]
    )


def write_outputs(rows: List[Dict[str, object]], output_root: Path, args: argparse.Namespace) -> Path:
    out_dir = output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_parked_ego_view_scout"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "parked_ego_training_view_candidates.csv"
    fieldnames = list(rows[0].keys()) if rows else ["quality_score"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Parked-Ego RGB+Radar Training View Scout",
        "",
        "Higher `quality_score` is better. The score is a geometry proxy: it rewards road spawn points, crosswalk points, angular spread, and left/center/right coverage inside the candidate camera FoV.",
        "",
        "A single 180-degree RGB camera is usually distorted for training. Start with one 110-120 degree front view; if we need true 180-degree awareness later, use two/three camera streams or multi-yaw captures.",
        "",
        "| rank | quality | side | tl_id | spawn | offsets fwd/right | dist_m | yaw_offset | road_pts | crosswalk_pts | L/C/R road pts | ego_xy |",
        "|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        md_lines.append(
            "| {rank} | {quality_score} | {parking_side} | {traffic_light_id} | {spawn_index} | "
            "{ego_spawn_forward_offset_m}/{ego_spawn_right_offset_m} | "
            "{distance_to_anchor_m} | {ego_spawn_yaw_offset_deg} | "
            "{road_spawn_points_in_view} | {crosswalk_points_in_view} | "
            "{left_road_points}/{center_road_points}/{right_road_points} | "
            "({ego_x}, {ego_y}) |".format(rank=rank, **row)
        )
    if rows:
        md_lines.extend(
            [
                "",
                "## Top Candidate Smoke Collection",
                "",
                command_block(rows[0], args, max_samples=60),
                "",
                "## Top Candidate Pilot Collection",
                "",
                command_block(rows[0], args, max_samples=600),
                "",
                "## Top Candidate First Full Collection",
                "",
                command_block(rows[0], args, max_samples=12000),
            ]
        )
    (out_dir / "parked_ego_training_view_candidates.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> int:
    args = parse_args()
    try:
        rows = build_rows(args)
    except RuntimeError as exc:
        print(
            "Unable to read the CARLA world. Start CARLA/Town10HD_Opt first, "
            f"then rerun the scout. Details: {exc}"
        )
        return 2
    if not rows:
        print("No parked-ego training-view candidates found.")
        return 1

    out_dir = write_outputs(rows, Path(args.output_root).expanduser().resolve(), args)
    print(f"Wrote scout results to {out_dir}")
    print("Top parked-ego training-view candidates:")
    for rank, row in enumerate(rows[: int(args.top)], start=1):
        print(
            f"{rank:02d}. quality={row['quality_score']} tl={row['traffic_light_id']} "
            f"spawn={row['spawn_index']} side={row['parking_side']} "
            f"offsets(fwd/right)={row['ego_spawn_forward_offset_m']}/{row['ego_spawn_right_offset_m']} "
            f"dist={row['distance_to_anchor_m']}m "
            f"yaw_offset={row['ego_spawn_yaw_offset_deg']}deg "
            f"road_pts={row['road_spawn_points_in_view']} "
            f"crosswalk_pts={row['crosswalk_points_in_view']} "
            f"L/C/R={row['left_road_points']}/{row['center_road_points']}/{row['right_road_points']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

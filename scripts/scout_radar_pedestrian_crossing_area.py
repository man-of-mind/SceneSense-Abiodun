#!/usr/bin/env python3
"""Scout clean CARLA areas for radar pedestrian crossing diagnostics.

The radar diagnostic needs a pedestrian to walk laterally across the radar FOV.
Some spawn points put that crossing line over a raised median or slab, which
causes CARLA's walker controller to stall. This scout scores spawn-point /
distance / amplitude combinations by sampling the proposed crossing line and
checking how close each point is to a flat drivable road surface.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from carla_radar_pedestrian_distance_pps_diagnostic import (  # noqa: E402
    carla,
    transform_forward_right,
    transform_matrix,
)


def parse_float_list(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--load-town", action="store_true")
    parser.add_argument("--spawn-indices", default="", help="Optional comma-separated spawn indices to inspect.")
    parser.add_argument("--distance-list-m", default="15,20,30,40")
    parser.add_argument("--amplitude-list-m", default="6,8")
    parser.add_argument("--samples-per-crossing", type=int, default=17)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--max-road-projection-m", type=float, default=1.4)
    parser.add_argument("--max-z-range-m", type=float, default=0.35)
    parser.add_argument("--radar-x", type=float, default=1.8)
    parser.add_argument("--radar-y", type=float, default=0.0)
    parser.add_argument("--radar-z", type=float, default=1.55)
    parser.add_argument("--radar-pitch", type=float, default=-4.0)
    parser.add_argument("--radar-yaw", type=float, default=0.0)
    parser.add_argument("--radar-roll", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/radar_pedestrian_crossing_scout",
        help="Directory for candidate CSV and command snippets.",
    )
    return parser.parse_args()


def parse_indices(raw: str, count: int) -> List[int]:
    if not raw.strip():
        return list(range(count))
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    invalid = [idx for idx in indices if idx < 0 or idx >= count]
    if invalid:
        raise ValueError(f"Invalid spawn indices {invalid}; valid range is 0..{count - 1}")
    return indices


def sensor_world_transform(ego_tf: "carla.Transform", args: argparse.Namespace) -> "carla.Transform":
    sensor_tf = carla.Transform(
        carla.Location(x=float(args.radar_x), y=float(args.radar_y), z=float(args.radar_z)),
        carla.Rotation(
            pitch=float(args.radar_pitch),
            yaw=float(args.radar_yaw),
            roll=float(args.radar_roll),
        ),
    )
    world_mat = transform_matrix(ego_tf) @ transform_matrix(sensor_tf)
    return carla.Transform(
        carla.Location(x=float(world_mat[0, 3]), y=float(world_mat[1, 3]), z=float(world_mat[2, 3])),
        carla.Rotation(
            pitch=float(args.radar_pitch),
            yaw=float(ego_tf.rotation.yaw + args.radar_yaw),
            roll=float(args.radar_roll),
        ),
    )


def horizontal_distance(a: "carla.Location", b: "carla.Location") -> float:
    return float(math.hypot(float(a.x - b.x), float(a.y - b.y)))


def crossing_points(
    radar_tf: "carla.Transform",
    *,
    distance_m: float,
    amplitude_m: float,
    samples: int,
) -> List["carla.Location"]:
    forward, right = transform_forward_right(radar_tf)
    origin = carla.Location(
        x=float(radar_tf.location.x),
        y=float(radar_tf.location.y),
        z=float(radar_tf.location.z),
    )
    count = max(3, int(samples))
    offsets = [(-float(amplitude_m) + 2.0 * float(amplitude_m) * i / float(count - 1)) for i in range(count)]
    points: List["carla.Location"] = []
    for offset in offsets:
        points.append(
            carla.Location(
                x=float(origin.x + forward[0] * float(distance_m) + right[0] * offset),
                y=float(origin.y + forward[1] * float(distance_m) + right[1] * offset),
                z=float(origin.z),
            )
        )
    return points


def score_candidate(
    world_map: "carla.Map",
    spawn_index: int,
    spawn_tf: "carla.Transform",
    distance_m: float,
    amplitude_m: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    radar_tf = sensor_world_transform(spawn_tf, args)
    points = crossing_points(
        radar_tf,
        distance_m=float(distance_m),
        amplitude_m=float(amplitude_m),
        samples=int(args.samples_per_crossing),
    )
    road_distances: List[float] = []
    road_zs: List[float] = []
    intersections = 0
    same_road_ids: List[int] = []
    for point in points:
        try:
            waypoint = world_map.get_waypoint(
                carla.Location(x=float(point.x), y=float(point.y), z=float(point.z)),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception:
            waypoint = None
        if waypoint is None:
            road_distances.append(float("inf"))
            continue
        road_distances.append(horizontal_distance(point, waypoint.transform.location))
        road_zs.append(float(waypoint.transform.location.z))
        intersections += int(bool(getattr(waypoint, "is_intersection", False)))
        same_road_ids.append(int(getattr(waypoint, "road_id", -1)))

    finite_distances = [value for value in road_distances if math.isfinite(value)]
    road_ok_ratio = float(
        sum(1 for value in finite_distances if value <= float(args.max_road_projection_m)) / max(1, len(points))
    )
    max_projection = max(finite_distances) if finite_distances else float("inf")
    mean_projection = sum(finite_distances) / len(finite_distances) if finite_distances else float("inf")
    z_range = max(road_zs) - min(road_zs) if road_zs else float("inf")
    intersection_fraction = intersections / max(1, len(points))
    dominant_road_fraction = 0.0
    if same_road_ids:
        dominant_road_fraction = max(same_road_ids.count(road_id) for road_id in set(same_road_ids)) / len(same_road_ids)

    # Lower score is better. Penalize geometry that looks like medians/slabs:
    # large projection to nearest lane, changing elevation, or crossing many
    # road IDs/intersections.
    score = (
        max_projection * 30.0
        + mean_projection * 10.0
        + z_range * 80.0
        + (1.0 - road_ok_ratio) * 150.0
        + intersection_fraction * 25.0
        + (1.0 - dominant_road_fraction) * 20.0
    )
    if max_projection > float(args.max_road_projection_m):
        score += 80.0
    if z_range > float(args.max_z_range_m):
        score += 80.0

    return {
        "score": round(float(score), 3),
        "spawn_index": int(spawn_index),
        "distance_m": float(distance_m),
        "amplitude_m": float(amplitude_m),
        "road_ok_ratio": round(road_ok_ratio, 3),
        "max_projection_m": round(float(max_projection), 3) if math.isfinite(max_projection) else "inf",
        "mean_projection_m": round(float(mean_projection), 3) if math.isfinite(mean_projection) else "inf",
        "z_range_m": round(float(z_range), 3) if math.isfinite(z_range) else "inf",
        "intersection_fraction": round(float(intersection_fraction), 3),
        "dominant_road_fraction": round(float(dominant_road_fraction), 3),
        "spawn_x": round(float(spawn_tf.location.x), 3),
        "spawn_y": round(float(spawn_tf.location.y), 3),
        "spawn_z": round(float(spawn_tf.location.z), 3),
        "spawn_yaw": round(float(spawn_tf.rotation.yaw), 3),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def command_for(row: Dict[str, object]) -> str:
    return "\n".join(
        [
            "python3 carla_radar_pedestrian_distance_pps_diagnostic.py \\",
            f"  --experiment-id radar_ped_cross_scout_spawn{row['spawn_index']}_dist{int(float(row['distance_m']))}_amp{int(float(row['amplitude_m']))}_visual \\",
            "  --walker-motion-mode cross \\",
            "  --walker-motion-control walker_control_nudge \\",
            f"  --walker-motion-amplitude-m {float(row['amplitude_m']):.1f} \\",
            "  --walker-motion-speed-mps 3.0 \\",
            "  --walker-nudge-after-frames 8 \\",
            "  --pps-list 48000 \\",
            f"  --distance-list-m {float(row['distance_m']):.1f} \\",
            "  --frames-per-condition 1200 \\",
            "  --warmup-frames 20 \\",
            f"  --ego-spawn-index {int(row['spawn_index'])} \\",
            "  --walker-physics-mode default \\",
            "  --plot-min-distance-m 10 \\",
            "  --preview \\",
            "  --preview-width 1280 \\",
            "  --preview-height 720",
        ]
    )


def main() -> int:
    args = parse_args()
    if carla is None:
        raise SystemExit("Could not import carla. Run inside the CARLA PythonAPI environment.")
    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.load_world(str(args.town)) if bool(args.load_town) else client.get_world()
    world_map = world.get_map()
    spawn_points = list(world_map.get_spawn_points())
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points.")

    distances = parse_float_list(str(args.distance_list_m))
    amplitudes = parse_float_list(str(args.amplitude_list_m))
    indices = parse_indices(str(args.spawn_indices), len(spawn_points))

    rows: List[Dict[str, object]] = []
    for index in indices:
        spawn_tf = spawn_points[index]
        for distance in distances:
            for amplitude in amplitudes:
                rows.append(score_candidate(world_map, index, spawn_tf, distance, amplitude, args))
    rows.sort(key=lambda row: float(row["score"]))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "radar_pedestrian_crossing_candidates.csv", rows)
    command_lines: List[str] = []
    for rank, row in enumerate(rows[: max(1, int(args.top))], start=1):
        print(
            "{rank:02d}. score={score} spawn={spawn} dist={dist:g} amp={amp:g} "
            "ok={ok} max_proj={maxp} z_range={zr} intersection={inter}".format(
                rank=rank,
                score=row["score"],
                spawn=row["spawn_index"],
                dist=float(row["distance_m"]),
                amp=float(row["amplitude_m"]),
                ok=row["road_ok_ratio"],
                maxp=row["max_projection_m"],
                zr=row["z_range_m"],
                inter=row["intersection_fraction"],
            )
        )
        command_lines.append(f"# Candidate {rank:02d}: score={row['score']}\n{command_for(row)}\n")
    (output_dir / "visual_probe_commands.sh").write_text("\n".join(command_lines), encoding="utf-8")
    print(f"\nWrote {output_dir / 'radar_pedestrian_crossing_candidates.csv'}")
    print(f"Wrote {output_dir / 'visual_probe_commands.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

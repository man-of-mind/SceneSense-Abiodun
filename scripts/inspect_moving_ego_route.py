#!/usr/bin/env python3
"""Inspect CARLA spawn-index routes before moving-ego data collection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from carla_collect_moving_ego_fusion_training_data import (  # noqa: E402
    GlobalRoutePlanner,
    carla,
    copy_location,
    parse_spawn_index_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to CARLA, build GlobalRoutePlanner paths for spawn-index "
            "routes, and report the route distance."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help=(
            "Comma-separated spawn indices, e.g. 80,85,91,94,99,80. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument("--spacing-m", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/moving_route_inspection",
        help="Directory for route CSV/JSON summaries.",
    )
    parser.add_argument(
        "--list-spawns",
        action="store_true",
        help="Also write all spawn point transforms to spawn_points.csv.",
    )
    return parser.parse_args()


def distance_3d(a: "carla.Location", b: "carla.Location") -> float:
    return math.sqrt(
        float(a.x - b.x) ** 2 + float(a.y - b.y) ** 2 + float(a.z - b.z) ** 2
    )


def route_distance(points: Sequence["carla.Location"]) -> float:
    return sum(distance_3d(a, b) for a, b in zip(points[:-1], points[1:]))


def append_spaced(route: List["carla.Location"], location: "carla.Location", spacing_m: float) -> None:
    candidate = copy_location(location)
    if not route or distance_3d(route[-1], candidate) >= float(spacing_m):
        route.append(candidate)


def build_route(
    world: "carla.World",
    spawn_points: Sequence["carla.Transform"],
    indices: Sequence[int],
    spacing_m: float,
) -> List["carla.Location"]:
    if GlobalRoutePlanner is None:
        raise RuntimeError("GlobalRoutePlanner is unavailable in this CARLA PythonAPI environment.")
    planner = GlobalRoutePlanner(world.get_map(), max(1.0, float(spacing_m)))
    route: List["carla.Location"] = []
    key_points = [spawn_points[idx].location for idx in indices]
    for start, end in zip(key_points[:-1], key_points[1:]):
        trace = planner.trace_route(start, end)
        if not trace:
            append_spaced(route, start, spacing_m)
            append_spaced(route, end, spacing_m)
            continue
        for waypoint, _road_option in trace:
            append_spaced(route, waypoint.transform.location, spacing_m)
        append_spaced(route, end, spacing_m)
    return route


def spawn_row(index: int, transform: "carla.Transform") -> Dict[str, float]:
    return {
        "spawn_index": int(index),
        "x": round(float(transform.location.x), 4),
        "y": round(float(transform.location.y), 4),
        "z": round(float(transform.location.z), 4),
        "yaw": round(float(transform.rotation.yaw), 4),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(10.0)
    world = client.get_world()
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points.")

    if bool(args.list_spawns):
        write_csv(
            output_dir / "spawn_points.csv",
            [spawn_row(i, transform) for i, transform in enumerate(spawn_points)],
        )
        print(f"Wrote {output_dir / 'spawn_points.csv'}")

    if not args.route:
        print("No --route supplied. Use --list-spawns or pass one or more --route values.")
        return 0

    summary_rows: List[Dict[str, object]] = []
    for route_text in args.route:
        indices = parse_spawn_index_list(route_text)
        invalid = [idx for idx in indices if idx < 0 or idx >= len(spawn_points)]
        if invalid:
            raise ValueError(
                f"Invalid route indices {invalid}; valid range is 0..{len(spawn_points) - 1}."
            )
        if len(indices) < 2:
            raise ValueError(f"Route needs at least two spawn indices: {route_text}")

        route = build_route(world, spawn_points, indices, float(args.spacing_m))
        route_len_m = route_distance(route)
        key_len_m = route_distance([spawn_points[idx].location for idx in indices])
        label = "_".join(str(idx) for idx in indices)
        route_csv = output_dir / f"route_{label}.csv"
        write_csv(
            route_csv,
            [
                {
                    "route_point": i,
                    "x": round(float(point.x), 4),
                    "y": round(float(point.y), 4),
                    "z": round(float(point.z), 4),
                }
                for i, point in enumerate(route)
            ],
        )
        row = {
            "route": ",".join(str(idx) for idx in indices),
            "route_points": len(route),
            "route_length_m": round(route_len_m, 3),
            "keypoint_polyline_m": round(key_len_m, 3),
            "spacing_m": float(args.spacing_m),
            "route_csv": str(route_csv),
        }
        summary_rows.append(row)
        print(
            f"{row['route']}: route_length={row['route_length_m']}m, "
            f"points={row['route_points']}, csv={route_csv}"
        )

    write_csv(output_dir / "route_summary.csv", summary_rows)
    (output_dir / "route_summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'route_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

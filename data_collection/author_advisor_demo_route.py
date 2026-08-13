#!/usr/bin/env python3
"""Author an advisor-demo Town10HD_Opt ego route with the v2 UI planner.

The advisor UI files are imported as a read-only planning library.  This
utility performs the same spawn/road-catalog selection and calls
``ScenarioControllerV2.plan_vehicle_route_for_export`` before serializing via
``ego_route_config.save_route_config``.  The JSON remains loadable and editable
in the UI; the companion CSV is the collector's Traffic Manager path input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import carla


REPO_ROOT = Path(__file__).resolve().parents[1]
ADVISOR_CODES = REPO_ROOT / "rl_agent" / "advisor_helper_scripts" / "codes"
CARLA_AGENTS_ROOT = REPO_ROOT.parents[1] / "carla"
DEFAULT_SCENARIO_CONFIG = ADVISOR_CODES / "physical_ai_scenario_config_v2.yaml"
DEFAULT_TRAFFIC_LIGHT_DATA = REPO_ROOT / "traffic_lights_data.json"
DEFAULT_ROUTE = (
    REPO_ROOT / "data_collection" / "routes" / "town10hd_opt_advisor_demo_loop_v1.json"
)
DEFAULT_PROGRESS = DEFAULT_ROUTE.with_suffix(".progress.csv")
DEFAULT_VIA_XY = (
    (19.791866302490234, 32.016666412353516),
    (47.52729797363281, 40.43889617919922),
    (56.35101318359375, 62.8189811706543),
    (8.827261924743652, 62.21647644042969),
    (-13.132160186767578, 28.438270568847656),
)

for _path in (str(REPO_ROOT), str(CARLA_AGENTS_ROOT), str(ADVISOR_CODES)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ego_route_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _location_payload(location: carla.Location) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _transform_payload(transform: carla.Transform) -> dict[str, object]:
    return {
        "location": _location_payload(transform.location),
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def _nearest_index(
    location: carla.Location,
    candidates: Sequence[carla.Location],
) -> int:
    if not candidates:
        raise ValueError("cannot resolve a route control against an empty catalog")
    return min(
        range(len(candidates)),
        key=lambda index: float(candidates[index].distance(location)),
    )


def _downsample(
    locations: Iterable[carla.Location],
    minimum_spacing_m: float,
) -> list[carla.Location]:
    result: list[carla.Location] = []
    for location in locations:
        candidate = carla.Location(
            x=float(location.x), y=float(location.y), z=float(location.z)
        )
        if not result or result[-1].distance(candidate) >= minimum_spacing_m:
            result.append(candidate)
    return result


def _trim_initial_progress_behind_spawn(
    locations: Sequence[carla.Location],
    start_transform: carla.Transform,
    minimum_forward_m: float,
) -> list[carla.Location]:
    """Start the TM path ahead of the spawned ego, not at a snapped point behind it."""

    yaw = math.radians(float(start_transform.rotation.yaw))
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    origin = start_transform.location
    for index, location in enumerate(locations):
        longitudinal_m = (
            (float(location.x) - float(origin.x)) * forward_x
            + (float(location.y) - float(origin.y)) * forward_y
        )
        if longitudinal_m >= minimum_forward_m:
            return list(locations[index:])
    raise RuntimeError(
        "route planner produced no collector progress point sufficiently ahead of the ego spawn"
    )


def _write_progress_csv(
    path: Path,
    locations: Sequence[carla.Location],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ego_x", "ego_y", "ego_z"))
        writer.writeheader()
        for location in locations:
            writer.writerow(
                {
                    "ego_x": f"{float(location.x):.6f}",
                    "ego_y": f"{float(location.y):.6f}",
                    "ego_z": f"{float(location.z):.6f}",
                }
            )


def author_route(args: argparse.Namespace) -> tuple[Path, Path]:
    scenario_config = args.scenario_config.resolve()
    traffic_light_data = args.traffic_light_data.resolve()
    for path in (scenario_config, traffic_light_data):
        if not path.is_file():
            raise FileNotFoundError(path)

    client = None
    world = None
    last_error: Exception | None = None
    for _attempt in range(10):
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(args.timeout)
            world = client.get_world()
            break
        except RuntimeError as exc:
            last_error = exc
            time.sleep(1.0)
    if client is None or world is None:
        raise RuntimeError(f"CARLA get_world failed after 10 attempts: {last_error}")
    map_name = str(world.get_map().name)
    if not map_name.endswith("Town10HD_Opt"):
        raise RuntimeError(f"expected Town10HD_Opt, found {map_name}")

    # On this CARLA 0.10 Linux build, importing the full v2 UI before the first
    # successful RPC can make that first Client.get_world() abort.  The UI's
    # own planning code is read-only, so establish the connection first and
    # then import it.  A dedicated regression test covers this order.
    import physical_ai_scenario_controller_ui_v2 as advisor_ui_v2

    config = advisor_ui_v2.v1.load_yaml_config(scenario_config)
    controller = advisor_ui_v2.ScenarioControllerV2(
        client,
        world,
        traffic_light_data,
        float(config["ui"].get("map_waypoint_spacing_m", 2.0)),
        master_clock=False,
        fixed_delta_seconds=float(config.get("world", {}).get("fixed_delta_seconds", 0.05)),
        traffic_manager_port=int(args.tm_port),
        restore_world_settings=False,
    )
    try:
        spawn_points = list(controller.vehicle_spawn_preview)
        for index in (args.start_spawn_index, args.end_spawn_index):
            if not 0 <= index < len(spawn_points):
                raise ValueError(
                    f"spawn index {index} is outside 0..{len(spawn_points) - 1}"
                )
        start_transform = advisor_ui_v2.v1.copy_transform(
            spawn_points[args.start_spawn_index]
        )
        end_transform = advisor_ui_v2.v1.copy_transform(
            spawn_points[args.end_spawn_index]
        )
        desired_vias = [
            carla.Location(x=float(x), y=float(y), z=float(start_transform.location.z))
            for x, y in (args.via or DEFAULT_VIA_XY)
        ]
        selected_via_indices = [
            _nearest_index(location, controller.road_preview)
            for location in desired_vias
        ]
        selected_vias = [
            advisor_ui_v2.v1.copy_location(controller.road_preview[index])
            for index in selected_via_indices
        ]
        planned = controller.plan_vehicle_route_for_export(
            start_transform,
            selected_vias,
            end_transform,
            float(args.sampling_resolution_m),
        )
        resolved_via_indices = [
            _nearest_index(location, controller.road_preview)
            for location in planned["intermediate_waypoints"]
        ]
        planned_path = list(planned["planned_path"])
        route_data = {
            "schema_version": ego_route_config.ROUTE_SCHEMA_VERSION,
            "type": ego_route_config.ROUTE_CONFIG_TYPE,
            "name": str(args.route_name),
            "map": map_name,
            "coordinate_system": ego_route_config.ROUTE_COORDINATE_SYSTEM,
            "route_sampling_resolution_m": float(args.sampling_resolution_m),
            "start": _transform_payload(planned["start_transform"]),
            "intermediate_waypoints": [
                _location_payload(location)
                for location in planned["intermediate_waypoints"]
            ],
            "end": _transform_payload(planned["end_transform"]),
            "planned_path": [
                _location_payload(location) for location in planned_path
            ],
            "loop": True,
            "ui_selection": {
                "producer": Path(__file__).name,
                "planner": "physical_ai_scenario_controller_ui_v2.py",
                "selection_basis": "advisor_blocker_coordinates_to_ui_catalog",
                "vehicle_start_spawn_index": int(args.start_spawn_index),
                "vehicle_end_spawn_index": int(args.end_spawn_index),
                "vehicle_waypoint_indices": resolved_via_indices,
                "vehicle_spawn_catalog_size": len(spawn_points),
                "road_waypoint_catalog_size": len(controller.road_preview),
            },
            "provenance": {
                "scenario_config": str(scenario_config.relative_to(REPO_ROOT)),
                "scenario_config_sha256": _sha256(scenario_config),
                "ui_planner": str(Path(advisor_ui_v2.__file__).resolve().relative_to(REPO_ROOT)),
                "ui_planner_sha256": _sha256(Path(advisor_ui_v2.__file__).resolve()),
                "traffic_manager_port": int(args.tm_port),
                "collector_progress_initial_forward_m": float(
                    args.progress_initial_forward_m
                ),
                "advisor_blocker_control_xy": [
                    {"x": float(x), "y": float(y)}
                    for x, y in (args.via or DEFAULT_VIA_XY)
                ],
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        route_path = args.route_config.resolve()
        progress_path = args.progress_csv.resolve()
        route_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = ego_route_config.save_route_config(route_path, route_data)
        progress_source = _trim_initial_progress_behind_spawn(
            planned_path,
            planned["start_transform"],
            float(args.progress_initial_forward_m),
        )
        progress = _downsample(progress_source, float(args.progress_min_spacing_m))
        if len(progress) < 2:
            raise RuntimeError("route planner produced fewer than two progress points")
        _write_progress_csv(progress_path, progress)
        print(
            f"saved UI-compatible route {route_path} with "
            f"{len(normalized['planned_path'])} planned points; "
            f"collector path {progress_path} has {len(progress)} points"
        )
        return route_path, progress_path
    finally:
        result = controller.shutdown()
        if result.get("errors"):
            raise RuntimeError("route-planner cleanup failed: " + "; ".join(result["errors"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument("--traffic-light-data", type=Path, default=DEFAULT_TRAFFIC_LIGHT_DATA)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--progress-csv", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--route-name", default="Town10HD_Opt advisor demo loop v1")
    parser.add_argument("--start-spawn-index", type=int, default=55)
    parser.add_argument("--end-spawn-index", type=int, default=53)
    parser.add_argument(
        "--via",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="repeatable UI road-control coordinate; defaults to advisor blockers",
    )
    parser.add_argument("--sampling-resolution-m", type=float, default=2.0)
    parser.add_argument("--progress-min-spacing-m", type=float, default=3.0)
    parser.add_argument("--progress-initial-forward-m", type=float, default=6.0)
    args = parser.parse_args()
    if (
        args.sampling_resolution_m <= 0.0
        or args.progress_min_spacing_m <= 0.0
        or args.progress_initial_forward_m <= 0.0
    ):
        parser.error("route sampling and progress spacing must be positive")
    author_route(args)


if __name__ == "__main__":
    import subprocess

    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-c",
                "from data_collection.author_advisor_demo_route import main; main()",
                *sys.argv[1:],
            ],
            cwd=REPO_ROOT,
        )
    )

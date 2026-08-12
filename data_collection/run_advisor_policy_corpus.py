#!/usr/bin/env python3
"""Collect the advisor-rich policy corpus with one CARLA sync ticker.

The advisor scripts remain read-only population clients.  This runner owns the
episode lifecycle: it makes the empty world synchronous, ticks while the
populators start, yields sole tick ownership to the fusion collector, ticks
their shutdown, verifies cleanup, and restores asynchronous mode.  The
collector is always launched in observe-existing mode by the resolved config.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import carla
import numpy as np
import pandas as pd
import yaml

from data_collection import run_policy_corpus as base_runner
from rl_agent.policy.replay import _greedy_prediction_matches, _normalize_class


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "policy_corpus_advisor_rich_v3.yaml"
)
DYNAMIC_PATTERNS = (
    "vehicle.*",
    "walker.pedestrian.*",
    "sensor.*",
    "controller.ai.walker",
)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _connect(config: Mapping[str, object]) -> Tuple[carla.Client, carla.World]:
    connection = config["carla"]
    last_error = ""
    for _attempt in range(10):
        try:
            client = carla.Client(str(connection["host"]), int(connection["port"]))
            client.set_timeout(float(connection.get("timeout_s", 10.0)))
            # This CARLA 0.10 Linux package intermittently aborts get_world()
            # when it is the first RPC on a fresh client. The lightweight
            # version request reliably establishes the session and is needed
            # for the required version check anyway.
            server_version = str(client.get_server_version())
            world = client.get_world()
            if not str(world.get_map().name).endswith(str(connection["expected_town"])):
                raise RuntimeError(
                    f"expected {connection['expected_town']}, found {world.get_map().name}"
                )
            if server_version != str(connection["expected_server_version"]):
                raise RuntimeError(
                    f"expected CARLA {connection['expected_server_version']}, "
                    f"found {server_version}"
                )
            return client, world
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"CARLA connection failed after 10 attempts: {last_error}")


def _actor_inventory(world: carla.World) -> Dict[str, int]:
    actors = world.get_actors()
    return {pattern: int(len(actors.filter(pattern))) for pattern in DYNAMIC_PATTERNS}


def _role_inventory(world: carla.World) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for actor in world.get_actors():
        try:
            role = str(actor.attributes.get("role_name", "")).strip()
        except (AttributeError, RuntimeError):
            continue
        if role:
            counts[role] = counts.get(role, 0) + 1
    return counts


def _matching_role_count(roles: Mapping[str, int], prefix: str) -> int:
    return sum(count for role, count in roles.items() if role.startswith(prefix))


def _require_empty_async(world: carla.World) -> Dict[str, object]:
    inventory = _actor_inventory(world)
    occupied = {name: value for name, value in inventory.items() if value}
    settings = world.get_settings()
    if occupied:
        raise RuntimeError(f"advisor corpus requires an empty dynamic world: {occupied}")
    if bool(settings.synchronous_mode):
        raise RuntimeError(
            "advisor corpus requires asynchronous startup; a stale synchronous world has no known owner"
        )
    return {
        "dynamic_actor_counts": inventory,
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
    }


def _set_sync_master(
    client: carla.Client,
    world: carla.World,
    tm_port: int,
    fixed_delta_seconds: float,
) -> object:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(fixed_delta_seconds)
    world.apply_settings(settings)
    client.get_trafficmanager(int(tm_port)).set_synchronous_mode(True)
    world.tick(2.0)
    return original


def _restore_async(
    client: carla.Client,
    world: carla.World,
    tm_port: int,
    original_settings: object,
) -> None:
    client.get_trafficmanager(int(tm_port)).set_synchronous_mode(False)
    world.apply_settings(original_settings)


def _spawn_ego_reservations(
    world: carla.World,
    spawn_indices: Sequence[int],
) -> List[carla.Actor]:
    spawn_points = list(world.get_map().get_spawn_points())
    actors: List[carla.Actor] = []
    try:
        for spawn_index in spawn_indices:
            if not 0 <= int(spawn_index) < len(spawn_points):
                raise ValueError(f"ego reservation spawn index {spawn_index} is invalid")
            blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute(
                    "role_name", f"advisor_ego_spawn_reservation_{int(spawn_index)}"
                )
            actor = world.try_spawn_actor(blueprint, spawn_points[int(spawn_index)])
            if actor is None:
                raise RuntimeError(
                    f"unable to reserve advisor route-corridor spawn {spawn_index}"
                )
            actor.set_simulate_physics(False)
            actors.append(actor)
        return actors
    except Exception:
        _destroy_ego_reservations(world, actors)
        raise


def _destroy_ego_reservations(
    world: carla.World, actors: Sequence[carla.Actor]
) -> None:
    if not actors:
        return
    for actor in reversed(list(actors)):
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass
    world.tick(2.0)


def _population_commands(
    config: Mapping[str, object], run_spec: Mapping[str, object]
) -> Tuple[List[str], List[str]]:
    integration = config["advisor_integration"]
    family = str(run_spec["scenario_family"])
    family_spec = integration["families"][family]
    host = str(config["carla"]["host"])
    port = str(config["carla"]["port"])
    tm_port = str(integration["tm_port"])
    pedestrian_location_args = [
        str(value)
        for location in integration["pedestrian_locations"]
        for value in ("--pedestrian-location", *location)
    ]
    blocker = [
        sys.executable,
        "-u",
        str(base_runner._resolve_repo_path(str(integration["spawn_blocker_entrypoint"]))),
        "--host",
        host,
        "--port",
        port,
        "--ego-role-name",
        str(integration["ego_role_name"]),
        "--pedestrian-speed",
        str(integration["pedestrian_speed_mps"]),
        "--min-pedestrian-speed",
        str(integration["minimum_pedestrian_speed_mps"]),
        "--update-hz",
        str(integration["update_hz"]),
        "--tick-timeout",
        str(integration["tick_timeout_s"]),
        "--no-intercept-debug",
        *pedestrian_location_args,
        *(str(value) for value in integration.get("common_blocker_args", [])),
        *(str(value) for value in family_spec.get("blocker_args", [])),
    ]
    traffic = [
        sys.executable,
        "-u",
        str(base_runner._resolve_repo_path(str(integration["generate_traffic_script"]))),
        "--host",
        host,
        "--port",
        port,
        "--tm-port",
        tm_port,
        "--number-of-vehicles",
        str(family_spec["number_of_vehicles"]),
        "--number-of-walkers",
        str(family_spec["number_of_walkers"]),
        "--seed",
        str(run_spec["seed"]),
        "--seedw",
        str(int(run_spec["seed"]) + int(integration["walker_seed_offset"])),
        "--replenish-interval",
        str(integration["replenish_interval_s"]),
        "--population-log-interval",
        str(integration["population_log_interval_s"]),
        *(str(value) for value in integration.get("common_traffic_args", [])),
        *(str(value) for value in family_spec.get("traffic_args", [])),
    ]
    return blocker, traffic


def _start_process(command: Sequence[str], log_path: Path) -> Tuple[subprocess.Popen, object]:
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def _tick_until(
    world: carla.World,
    processes: Sequence[subprocess.Popen],
    predicate,
    timeout_s: float,
    label: str,
) -> Dict[str, object]:
    deadline = time.monotonic() + float(timeout_s)
    last_inventory: Dict[str, int] = {}
    last_roles: Dict[str, int] = {}
    while time.monotonic() < deadline:
        failures = [process.returncode for process in processes if process.poll() is not None]
        if failures:
            raise RuntimeError(f"{label} process exited early: returncodes={failures}")
        world.tick(2.0)
        last_inventory = _actor_inventory(world)
        last_roles = _role_inventory(world)
        if predicate(last_inventory, last_roles):
            return {"actor_counts": last_inventory, "role_counts": last_roles}
        time.sleep(0.01)
    raise RuntimeError(
        f"timed out waiting for {label}; actor_counts={last_inventory}, roles={last_roles}"
    )


def _blocker_ready(
    inventory: Mapping[str, int],
    roles: Mapping[str, int],
    family_spec: Mapping[str, object],
) -> bool:
    del inventory
    required = family_spec.get("minimum_blocker_role_prefix_counts", {})
    return all(
        _matching_role_count(roles, str(prefix)) >= int(count)
        for prefix, count in required.items()
    )


def _population_ready(
    inventory: Mapping[str, int],
    roles: Mapping[str, int],
    family_spec: Mapping[str, object],
) -> bool:
    required = family_spec.get("minimum_ready_actor_counts", {})
    if not all(int(inventory.get(str(pattern), 0)) >= int(count) for pattern, count in required.items()):
        return False
    minimum_autopilot = int(family_spec.get("minimum_autopilot_vehicles", 0))
    return _matching_role_count(roles, "autopilot") >= minimum_autopilot


def _stop_processes(
    world: carla.World,
    processes: Sequence[Tuple[str, subprocess.Popen, object]],
    timeout_s: float,
) -> List[Dict[str, object]]:
    for _name, process, _stream in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline and any(
        process.poll() is None for _name, process, _stream in processes
    ):
        try:
            world.tick(2.0)
        except RuntimeError:
            pass
        time.sleep(0.02)
    for _name, process, _stream in reversed(processes):
        if process.poll() is None:
            process.terminate()
    terminate_deadline = time.monotonic() + 3.0
    while time.monotonic() < terminate_deadline and any(
        process.poll() is None for _name, process, _stream in processes
    ):
        try:
            world.tick(2.0)
        except RuntimeError:
            pass
        time.sleep(0.02)
    results: List[Dict[str, object]] = []
    for name, process, stream in processes:
        if process.poll() is None:
            process.kill()
        returncode = process.wait(timeout=3.0)
        stream.close()
        results.append({"name": name, "returncode": int(returncode)})
    return results


def _tick_until_empty(world: carla.World, timeout_s: float) -> Dict[str, int]:
    deadline = time.monotonic() + float(timeout_s)
    inventory = _actor_inventory(world)
    while time.monotonic() < deadline:
        if not any(inventory.values()):
            return inventory
        world.tick(2.0)
        time.sleep(0.02)
        inventory = _actor_inventory(world)
    raise RuntimeError(f"dynamic actors leaked after advisor episode: {inventory}")


def _static_preflight(config: Mapping[str, object]) -> Dict[str, object]:
    preflight = base_runner._static_preflight(config)
    integration = config["advisor_integration"]
    files = {
        "generate_traffic": base_runner._resolve_repo_path(
            str(integration["generate_traffic_script"])
        ),
        "spawn_blocker": base_runner._resolve_repo_path(
            str(integration["spawn_blocker_script"])
        ),
        "spawn_blocker_entrypoint": base_runner._resolve_repo_path(
            str(integration["spawn_blocker_entrypoint"])
        ),
        "route_config": base_runner._resolve_repo_path(str(integration["route_config"])),
        "route_progress_csv": base_runner._resolve_repo_path(
            str(integration["route_progress_csv"])
        ),
        "scenario_ui_v2": base_runner._resolve_repo_path(
            str(integration["scenario_ui_v2"])
        ),
        "scenario_config_v2": base_runner._resolve_repo_path(
            str(integration["scenario_config_v2"])
        ),
        "ego_route_config": base_runner._resolve_repo_path(
            str(integration["ego_route_config_module"])
        ),
        "pole_camera_client": base_runner._resolve_repo_path(
            str(integration["pole_camera_client"])
        ),
        "traffic_light_data": base_runner._resolve_repo_path(
            str(integration["traffic_light_data"])
        ),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing advisor integration prerequisites: " + ", ".join(missing))
    route = json.loads(files["route_config"].read_text(encoding="utf-8"))
    if not str(route.get("map", "")).endswith(str(config["carla"]["expected_town"])):
        raise ValueError("advisor route map does not match collection town")
    if route.get("loop") is not True or len(route.get("planned_path", [])) < 2:
        raise ValueError("advisor route must be a non-empty loop")
    progress = pd.read_csv(files["route_progress_csv"])
    if not {"ego_x", "ego_y", "ego_z"}.issubset(progress.columns) or len(progress) < 2:
        raise ValueError("advisor route progress CSV is invalid")
    pedestrian_speed = float(integration["pedestrian_speed_mps"])
    minimum_speed = float(integration["minimum_pedestrian_speed_mps"])
    if not 1.0 <= pedestrian_speed <= 2.0:
        raise ValueError("reactive pedestrian speed must remain in the pinned 1-2 m/s walking band")
    if not 0.0 <= minimum_speed <= pedestrian_speed:
        raise ValueError("minimum pedestrian speed is invalid")
    if int(integration["tm_port"]) != 8010:
        raise ValueError("advisor integration Traffic Manager port must be 8010")
    preflight["advisor_integration"] = {
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": base_runner._sha256(path),
            }
            for name, path in files.items()
        },
        "route_points": int(len(route["planned_path"])),
        "progress_points": int(len(progress)),
        "route_loop": True,
        "tm_port": 8010,
        "reactive_pedestrian_speed_mps": pedestrian_speed,
    }
    return preflight


def _longest_true_dwell(mask: Sequence[bool], timestamps: Sequence[float]) -> float:
    values = np.asarray(mask, dtype=bool)
    times = np.asarray(timestamps, dtype=float)
    if not len(values):
        return 0.0
    deltas = np.diff(times)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    nominal = float(np.median(deltas)) if len(deltas) else 0.0
    longest = 0.0
    start = None
    for index, enabled in enumerate(values):
        if enabled and start is None:
            start = index
        if not enabled and start is not None:
            longest = max(longest, times[index - 1] - times[start] + nominal)
            start = None
    if start is not None:
        longest = max(longest, times[-1] - times[start] + nominal)
    return float(longest)


def _run_smoke_gate(batch_dir: Path, config: Mapping[str, object]) -> Dict[str, object]:
    gate = config["advisor_integration"]["smoke_gate"]
    gt_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    metric_frames: List[pd.DataFrame] = []
    overlay_count = 0
    for run_spec in config["smoke_runs"]:
        run_dir = batch_dir / "runs" / str(run_spec["episode_id"])
        gt = pd.read_csv(base_runner._single_csv(run_dir, "_object_ground_truth.csv"))
        pred = pd.read_csv(base_runner._single_csv(run_dir, "_object_predictions.csv"))
        metrics = pd.read_csv(base_runner._single_csv(run_dir, "_metrics.csv"))
        gt["episode_id"] = str(run_spec["episode_id"])
        pred["episode_id"] = str(run_spec["episode_id"])
        metrics["episode_id"] = str(run_spec["episode_id"])
        metrics["scenario_family"] = str(run_spec["scenario_family"])
        gt_frames.append(gt)
        prediction_frames.append(pred)
        metric_frames.append(metrics)
        overlay_count += len(list((run_dir / "overlays").glob("*.png")))
    gt_all = pd.concat(gt_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    gt_all["class_name"] = gt_all["class_name"].map(_normalize_class)
    predictions_all["class_name"] = predictions_all["class_name"].map(_normalize_class)
    failures: List[str] = []
    classes = set(gt_all["class_name"].dropna().astype(str))
    for class_name in ("vehicle", "pedestrian"):
        if class_name not in classes:
            failures.append(f"missing_{class_name}_ground_truth")

    in_scope = _truthy(gt_all["in_camera_frustum"]) & (
        pd.to_numeric(gt_all["distance_m"], errors="coerce")
        <= float(gate["headline_range_m"])
    )
    role_name = gt_all.get("role_name", pd.Series("", index=gt_all.index)).astype(str)
    controlled = gt_all[
        in_scope
        & (gt_all["class_name"] == "pedestrian")
        & role_name.str.startswith(str(gate["pedestrian_role_prefix"]))
    ].copy()
    controlled["world_x"] = pd.to_numeric(controlled["origin_x"], errors="coerce")
    controlled["world_y"] = pd.to_numeric(controlled["origin_y"], errors="coerce")
    scores = pd.to_numeric(
        predictions_all.get("score", pd.Series(1.0, index=predictions_all.index)),
        errors="coerce",
    )
    predictions_all = predictions_all[scores >= float(gate["prediction_score_min"])].copy()
    matches = []
    for episode_id, episode_gt in controlled.groupby("episode_id"):
        episode_pred = predictions_all[predictions_all["episode_id"] == episode_id]
        match = _greedy_prediction_matches(
            episode_gt,
            episode_pred,
            float(gate["association_gate_m"]),
        )
        if not match.empty:
            matches.append(match)
    matched_rows = int(sum(len(frame) for frame in matches))
    pedestrian_coverage = (
        100.0 * matched_rows / len(controlled) if len(controlled) else 0.0
    )
    if len(controlled) == 0:
        failures.append("no_close_controlled_pedestrian_gt")
    if pedestrian_coverage < float(gate["minimum_controlled_pedestrian_coverage_pct"]):
        failures.append("controlled_pedestrian_coverage_below_gate")

    controlled_speed_parts = []
    for (_episode_id, _actor_id), group in controlled.groupby(
        ["episode_id", "actor_id"]
    ):
        group = group.sort_values("carla_timestamp")
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        controlled_speed_parts.append(pd.Series(np.hypot(dx, dy) / dt, index=group.index))
    controlled_speed = (
        pd.concat(controlled_speed_parts).sort_index().replace([np.inf, -np.inf], np.nan).dropna()
        if controlled_speed_parts
        else pd.Series(dtype=float)
    )
    controlled_active_rows = int(
        (controlled_speed >= float(gate["minimum_controlled_pedestrian_active_speed_mps"])).sum()
    )
    if controlled_active_rows < int(gate["minimum_controlled_pedestrian_active_rows"]):
        failures.append("controlled_pedestrian_did_not_realize_crossing_motion")
    if len(controlled_speed) and controlled_speed.max() > float(gate["pedestrian_speed_max_mps"]):
        failures.append("controlled_pedestrian_speed_above_realistic_maximum")

    route_metrics = metrics_all[
        metrics_all["scenario_family"].isin(["mixed_urban", "ped_crossing"])
    ]
    route_ego_speed = pd.to_numeric(route_metrics["ego_speed_mps"], errors="coerce").dropna()
    route_ego_speed_p95 = float(route_ego_speed.quantile(0.95)) if len(route_ego_speed) else 0.0
    if route_ego_speed_p95 < float(gate["route_ego_speed_p95_min_mps"]):
        failures.append("ego_route_motion_below_gate")

    exact = gt_all[
        role_name == str(gate["exact_fast_role_name"])
    ].sort_values(["episode_id", "carla_timestamp"]).copy()
    speed_parts = []
    for _actor_id, group in exact.groupby(["episode_id", "actor_id"]):
        dt = pd.to_numeric(group["carla_timestamp"], errors="coerce").diff()
        dx = pd.to_numeric(group["origin_x"], errors="coerce").diff()
        dy = pd.to_numeric(group["origin_y"], errors="coerce").diff()
        speed_parts.append(pd.Series(np.hypot(dx, dy) / dt, index=group.index))
    exact["derived_speed_mps"] = (
        pd.concat(speed_parts).sort_index() if speed_parts else pd.Series(dtype=float)
    )
    exact_mask = (
        _truthy(exact["in_camera_frustum"])
        & (pd.to_numeric(exact["distance_m"], errors="coerce") <= float(gate["fast_range_max_m"]))
        & (exact["derived_speed_mps"] >= float(gate["fast_speed_min_mps"]))
    )
    fast_dwell = 0.0
    for _episode_id, group in exact.groupby("episode_id"):
        group_mask = exact_mask.loc[group.index]
        fast_dwell = max(
            fast_dwell,
            _longest_true_dwell(
                group_mask.tolist(),
                pd.to_numeric(group["carla_timestamp"], errors="coerce").tolist(),
            ),
        )
    if exact.empty:
        failures.append("missing_exact_fast_target_gt")
    elif fast_dwell < float(gate["fast_dwell_min_s"]):
        failures.append("exact_fast_target_dwell_below_gate")
    if overlay_count < int(gate["minimum_overlay_images"]):
        failures.append("insufficient_visual_overlays")
    summary = {
        "pass": not failures,
        "failures": failures,
        "gt_classes": sorted(classes),
        "controlled_pedestrian_eligible_rows": int(len(controlled)),
        "controlled_pedestrian_matched_rows": matched_rows,
        "controlled_pedestrian_coverage_pct": pedestrian_coverage,
        "controlled_pedestrian_active_rows": controlled_active_rows,
        "controlled_pedestrian_speed_p50_mps": (
            float(controlled_speed.quantile(0.50)) if len(controlled_speed) else None
        ),
        "controlled_pedestrian_speed_p95_mps": (
            float(controlled_speed.quantile(0.95)) if len(controlled_speed) else None
        ),
        "controlled_pedestrian_speed_max_mps": (
            float(controlled_speed.max()) if len(controlled_speed) else None
        ),
        "route_ego_speed_p95_mps": route_ego_speed_p95,
        "legacy_pedestrian_coverage_pct": float(gate["legacy_pedestrian_coverage_pct"]),
        "exact_fast_target_rows": int(len(exact)),
        "exact_fast_dwell_s": fast_dwell,
        "overlay_images": overlay_count,
    }
    (batch_dir / "smoke_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _validate_advisor_contract(config: Mapping[str, object]) -> None:
    base_runner._validate_collection_contract(config)
    integration = config.get("advisor_integration")
    if not isinstance(integration, Mapping):
        raise ValueError("advisor_integration mapping is required")
    pedestrian_speed = float(integration["pedestrian_speed_mps"])
    minimum_speed = float(integration["minimum_pedestrian_speed_mps"])
    if not 1.0 <= pedestrian_speed <= 2.0:
        raise ValueError("reactive pedestrian speed must remain in the pinned 1-2 m/s walking band")
    if not 0.0 <= minimum_speed <= pedestrian_speed:
        raise ValueError("minimum pedestrian speed is invalid")
    if int(integration["tm_port"]) != 8010:
        raise ValueError("advisor integration Traffic Manager port must be 8010")
    if not math.isclose(float(integration["fixed_delta_seconds"]), 0.05, abs_tol=1e-12):
        raise ValueError("advisor integration fixed delta must be 0.05 s")
    pedestrian_locations = integration.get("pedestrian_locations", [])
    if len(pedestrian_locations) != 1 or any(
        len(location) != 4 for location in pedestrian_locations
    ):
        raise ValueError("exactly one explicit close-crossing pedestrian XYZYAW is required")
    route_path = base_runner._resolve_repo_path(str(integration["route_progress_csv"]))
    route = pd.read_csv(route_path)
    maximum_offset = float(integration["maximum_pedestrian_route_offset_m"])
    for location in pedestrian_locations:
        minimum_offset = np.hypot(
            route["ego_x"].astype(float) - float(location[0]),
            route["ego_y"].astype(float) - float(location[1]),
        ).min()
        if float(minimum_offset) > maximum_offset:
            raise ValueError(
                "pedestrian location is not close enough to the frozen UI route: "
                f"location={location}, offset={float(minimum_offset):.3f} m"
            )
    family_names = set(config.get("family_args", {}))
    if set(integration.get("families", {})) != family_names:
        raise ValueError("advisor population families must exactly match collector families")
    for run_spec in [*config.get("smoke_runs", []), *config.get("runs", [])]:
        options = base_runner._effective_options(base_runner._resolved_run_args(config, run_spec))
        for option in ("--npc-vehicles", "--npc-pedestrians"):
            if options.get(option) != "0":
                raise ValueError(f"{run_spec['episode_id']} must use observe-existing {option}=0")
        if options.get("--tm-port") != str(integration["tm_port"]):
            raise ValueError(f"{run_spec['episode_id']} collector TM port is not aligned")
        if options.get("--ego-fixed-path-progress-csv") != str(integration["route_progress_csv"]):
            raise ValueError(f"{run_spec['episode_id']} does not use the frozen advisor route")
        if options.get("--ego-spawn-index") != str(integration["ego_spawn_index"]):
            raise ValueError(f"{run_spec['episode_id']} collector spawn differs from reservation")
        walker_ignore = float(options.get("--ego-ignore-walkers-pct", "0"))
        route_control = str(options.get("--ego-route-control", "traffic_manager"))
        family = str(run_spec["scenario_family"])
        expected_route_control = (
            "traffic_manager" if family == "exact_fast_convoy" else "direct"
        )
        if route_control != expected_route_control:
            raise ValueError(
                f"{run_spec['episode_id']} must use {expected_route_control} ego route control"
            )
        if family == "ped_crossing":
            if not math.isclose(walker_ignore, 100.0, abs_tol=1e-12):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use the pinned "
                    "100% ego walker-ignore exception"
                )
            if options.get("--ego-direct-yield-to-controlled-pedestrian") != "true":
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use direct ego yield"
                )
            if not math.isclose(
                float(options.get("--ego-spawn-forward-offset-m", "0")),
                14.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must start at the pinned "
                    "close-route offset"
                )
            if not math.isclose(
                float(options.get("--ego-direct-route-speed-mps", "0")),
                3.5,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian crossing must use the pinned "
                    "3.5 m/s urban ego speed"
                )
            if "--braking-margin" not in integration["families"][family].get(
                "blocker_args", []
            ):
                raise ValueError(
                    f"{run_spec['episode_id']} pedestrian blocker must use the pinned "
                    "conservative arming margin"
                )
        elif not math.isclose(walker_ignore, 0.0, abs_tol=1e-12):
            raise ValueError(
                f"{run_spec['episode_id']} walker-ignore exception leaked outside ped_crossing"
            )
        elif options.get("--ego-direct-yield-to-controlled-pedestrian") == "true":
            raise ValueError(
                f"{run_spec['episode_id']} pedestrian-yield control leaked outside ped_crossing"
            )
    reservation_indices = [int(value) for value in integration["ego_reservation_spawn_indices"]]
    if int(integration["ego_spawn_index"]) not in reservation_indices:
        raise ValueError("ego reservation corridor must include the collector spawn")
    if len(reservation_indices) != len(set(reservation_indices)):
        raise ValueError("ego reservation spawn indices must be unique")


def _load_config(path: Path) -> Dict[str, object]:
    config = base_runner._load_config(path)
    _validate_advisor_contract(config)
    return config


def run_batch(
    config_path: Path,
    mode: str,
    batch_dir: Path | None,
    dry_run: bool,
    only_episode_ids: Sequence[str] = (),
) -> Path:
    config = _load_config(config_path)
    selected_runs: Iterable[Mapping[str, object]] = (
        config["smoke_runs"] if mode == "smoke" else config["runs"]
    )
    selected_runs = list(selected_runs)
    if only_episode_ids:
        wanted = set(only_episode_ids)
        selected_runs = [
            item for item in selected_runs if str(item["episode_id"]) in wanted
        ]
        found = {str(item["episode_id"]) for item in selected_runs}
        if found != wanted:
            raise ValueError("unknown --only-episode values: " + ", ".join(sorted(wanted - found)))
    preflight = _static_preflight(config)
    client = world = None
    if not dry_run:
        client, world = _connect(config)
        preflight["live_carla"] = _require_empty_async(world)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if batch_dir is None:
        batch_dir = base_runner._resolve_repo_path(str(config["output_root"])) / f"{timestamp}_{mode}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    resolved_config_path = batch_dir / "resolved_collection_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest: MutableMapping[str, object] = {
        "schema": "policy_corpus_batch.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "dry_run" if dry_run else "running",
        "config_path": str(config_path),
        "config_sha256": base_runner._sha256(config_path),
        "batch_dir": str(batch_dir),
        "preflight": preflight,
        "runs": [],
    }
    manifest_path = batch_dir / "batch_manifest.json"
    base_runner._write_manifest(manifest_path, manifest)

    integration = config["advisor_integration"]
    for run_spec in selected_runs:
        episode_id = str(run_spec["episode_id"])
        family = str(run_spec["scenario_family"])
        family_spec = integration["families"][family]
        run_dir = batch_dir / "runs" / episode_id
        collector_command = base_runner._run_command(config, run_spec, run_dir)
        blocker_command, traffic_command = _population_commands(config, run_spec)
        record: MutableMapping[str, object] = {
            **dict(run_spec),
            "command": collector_command,
            "blocker_command": blocker_command,
            "traffic_command": traffic_command,
            "run_dir": str(run_dir),
            "status": "planned" if dry_run else "running",
        }
        manifest["runs"].append(record)
        base_runner._write_manifest(manifest_path, manifest)
        if dry_run:
            continue
        assert client is not None and world is not None
        print(f"[{episode_id}] configuring single 20 Hz sync master", flush=True)
        run_dir.mkdir(parents=True, exist_ok=False)
        processes: List[Tuple[str, subprocess.Popen, object]] = []
        ego_reservations: List[carla.Actor] = []
        original_settings = None
        collector_result = None
        try:
            _require_empty_async(world)
            original_settings = _set_sync_master(
                client,
                world,
                int(integration["tm_port"]),
                float(integration["fixed_delta_seconds"]),
            )
            ego_reservations = _spawn_ego_reservations(
                world,
                [int(value) for value in integration["ego_reservation_spawn_indices"]],
            )
            record["ego_spawn_reservation"] = {
                "actor_ids": [int(actor.id) for actor in ego_reservations],
                "spawn_indices": [
                    int(value) for value in integration["ego_reservation_spawn_indices"]
                ],
            }
            blocker, blocker_stream = _start_process(
                blocker_command, run_dir / "spawn_blocker.log"
            )
            processes.append(("spawn_blocker_v4", blocker, blocker_stream))
            record["blocker_ready"] = _tick_until(
                world,
                [blocker],
                lambda inventory, roles: _blocker_ready(inventory, roles, family_spec),
                float(integration["population_start_timeout_s"]),
                "spawn_blocker_v4 readiness",
            )
            traffic, traffic_stream = _start_process(
                traffic_command, run_dir / "generate_traffic.log"
            )
            processes.append(("generate_traffic_v1", traffic, traffic_stream))
            record["population_ready"] = _tick_until(
                world,
                [blocker, traffic],
                lambda inventory, roles: _population_ready(inventory, roles, family_spec),
                float(integration["population_start_timeout_s"]),
                "advisor traffic population",
            )
            _destroy_ego_reservations(world, ego_reservations)
            ego_reservations = []
            record["ego_spawn_reservation"]["released_before_collector"] = True
            print(
                f"[{episode_id}] population ready; yielding sole tick ownership to collector",
                flush=True,
            )
            log_path = run_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log_stream:
                collector_result = subprocess.run(
                    collector_command,
                    cwd=REPO_ROOT,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=float(integration["collector_timeout_s"]),
                )
            record["returncode"] = int(collector_result.returncode)
        except Exception as exc:
            record["orchestration_error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "orchestration_failed"
            manifest["status"] = "failed"
            raise
        finally:
            if original_settings is not None:
                try:
                    _destroy_ego_reservations(world, ego_reservations)
                    ego_reservations = []
                    client.get_trafficmanager(int(integration["tm_port"])).set_synchronous_mode(True)
                    record["populator_shutdown"] = _stop_processes(
                        world,
                        processes,
                        float(integration["population_shutdown_timeout_s"]),
                    )
                    record["postflight_dynamic_actor_counts"] = _tick_until_empty(
                        world, float(integration["population_shutdown_timeout_s"])
                    )
                finally:
                    _restore_async(
                        client,
                        world,
                        int(integration["tm_port"]),
                        original_settings,
                    )
                    record["restored_world"] = _require_empty_async(world)
            base_runner._write_manifest(manifest_path, manifest)

        assert collector_result is not None
        log_path = run_dir / "run.log"
        record["basic_gate"] = base_runner._basic_run_gate(run_dir, run_spec, config)
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        known_teardown_abort = (
            collector_result.returncode == -6
            and "libc++abi" in log_tail
            and "std::exception" in log_tail
        )
        record["known_carla_teardown_abort"] = known_teardown_abort
        accepted = collector_result.returncode == 0 or (
            known_teardown_abort and record["basic_gate"]["pass"]
        )
        if record["basic_gate"]["pass"] and accepted:
            record["status"] = (
                "complete" if collector_result.returncode == 0 else "complete_with_teardown_warning"
            )
        else:
            record["status"] = "gate_failed" if not record["basic_gate"]["pass"] else "collector_failed"
            manifest["status"] = "failed"
            base_runner._write_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"{episode_id} failed: returncode={collector_result.returncode}, "
                f"basic_gate={record['basic_gate']}"
            )
        base_runner._write_manifest(manifest_path, manifest)
        print(f"[{episode_id}] complete and actor-clean", flush=True)

    if dry_run:
        base_runner._write_manifest(manifest_path, manifest)
        return batch_dir
    if mode == "smoke" and not only_episode_ids:
        smoke_gate = _run_smoke_gate(batch_dir, config)
        manifest["smoke_gate"] = smoke_gate
        manifest["status"] = "smoke_pass" if smoke_gate["pass"] else "smoke_gate_failed"
        base_runner._write_manifest(manifest_path, manifest)
        if not smoke_gate["pass"]:
            raise RuntimeError("advisor-rich smoke gate failed: " + ", ".join(smoke_gate["failures"]))
    else:
        manifest["status"] = "collection_complete_pending_verification"
        base_runner._write_manifest(manifest_path, manifest)
    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--only-episode", action="append", default=[])
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.validate_config:
        config = _load_config(config_path)
        print(
            json.dumps(
                {
                    "experiment_name": config["experiment_name"],
                    "full_runs": len(config.get("runs", [])),
                    "smoke_runs": len(config.get("smoke_runs", [])),
                    "status": "VALID",
                },
                sort_keys=True,
            )
        )
        return
    print(
        run_batch(
            config_path,
            args.mode,
            args.batch_dir,
            args.dry_run,
            args.only_episode,
        )
    )


if __name__ == "__main__":
    # This shipping libcarla aborts its first RPC when a CARLA-owning module is
    # loaded directly by runpy.  A clean ``python -c`` dispatcher is reliable
    # on L10319 and keeps the implementation importable for tests.
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-c",
                "from data_collection.run_advisor_policy_corpus import main; main()",
                *sys.argv[1:],
            ],
            cwd=REPO_ROOT,
        )
    )

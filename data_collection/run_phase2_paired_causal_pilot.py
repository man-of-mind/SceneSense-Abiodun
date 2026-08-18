#!/usr/bin/env python3
"""Validate, plan, and eventually run the two-trajectory paired Phase-2 pilot.

The checked-in v1 integration config is deliberately unauthorized even though
its road-legal geometry is reviewed. Thus validation and dry-run command review
fail closed before CARLA. Authorization must be a separate reviewed config;
the same runner then owns the one synchronous ticker and advances the frozen
helper/recipient collectors in lockstep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import yaml

from data_collection import run_advisor_policy_corpus as advisor
from data_collection import run_policy_corpus as base_runner
from data_collection.phase2_curbside_scenario import (
    CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE,
    CURBSIDE_EXPECTED_LANE_IDS,
    CURBSIDE_GEOMETRY_ID,
    CURBSIDE_HELPER_TRANSFORM,
    CURBSIDE_OCCLUDER_TRANSFORM,
    CURBSIDE_RECIPIENT_TRANSFORM,
    CURBSIDE_WALKER_END,
    CURBSIDE_WALKER_START,
    CurbsideScenarioRuntime,
    DirectRouteController,
    legal_opposing_lane_contract,
    load_route_progress,
    wrap_degrees,
    world_transform,
)
from data_collection.phase2_paired_causal_collector import _require_inherited_contract
from phase2_map_sharing.pilot_contract import load_and_validate_pilot_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "phase2_paired_causal_pilot_integration_v1.yaml"
)
REQUIRED_SCENARIO_ROLES = {
    "controlled_positive_occlusion",
    "matched_benign_negative",
}
ROLE_NAMES = ("helper", "recipient")


def _numeric_sequence_close(
    observed: Sequence[object], expected: Sequence[object], *, tolerance: float = 1e-6
) -> bool:
    return len(observed) == len(expected) and all(
        math.isclose(float(left), float(right), abs_tol=float(tolerance))
        for left, right in zip(observed, expected)
    )


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _drop_options(arguments: Sequence[object], option_names: set[str]) -> list[str]:
    tokens = [str(value) for value in arguments]
    retained: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in option_names:
            retained.append(token)
            index += 1
            continue
        index += 1
        if index < len(tokens) and not tokens[index].startswith("--"):
            index += 1
    return retained


def _load_config(path: Path) -> tuple[dict, dict, dict]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Phase-2 integration config root must be a mapping")
    if config.get("schema_version") != "scenesense.phase2_paired_causal_integration.v1":
        raise ValueError("unexpected Phase-2 integration schema")
    implementation_status = str(config.get("implementation_status", ""))
    if implementation_status not in {"offline_dry_run_only", "reviewed_pilot_only"}:
        raise ValueError("integration status must be offline_dry_run_only or reviewed_pilot_only")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "carla_launch", "oai_launch", "full_collection"
    }:
        raise ValueError("integration authorization mapping is required")
    geometry = config.get("scenario_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("scenario geometry mapping is required")
    if implementation_status == "offline_dry_run_only":
        if any(bool(value) for value in authorization.values()):
            raise ValueError("offline integration must not authorize a live run")
    else:
        if not bool(authorization.get("carla_launch")) or bool(
            authorization.get("oai_launch")
        ) or bool(authorization.get("full_collection")):
            raise ValueError("reviewed pilot may authorize only the two-trajectory CARLA launch")
        if geometry.get("status") != "reviewed_positive_and_benign_routes":
            raise ValueError("live pilot requires reviewed positive and benign route geometry")

    if geometry.get("status") != "reviewed_positive_and_benign_routes":
        raise ValueError("paired pilot must use the human-reviewed positive/benign geometry")
    if geometry.get("layout_id") != CURBSIDE_GEOMETRY_ID:
        raise ValueError("paired pilot geometry ID has drifted")
    if geometry.get("population_mode") != "frozen_curbside_pilot_no_ambient":
        raise ValueError("pilot must exclude ambient traffic until causal capture is validated")
    if not bool(geometry.get("reload_world_before_trajectory")):
        raise ValueError("matched trajectories must reload Town10HD_Opt before each run")
    if geometry.get("town") != "Town10HD_Opt":
        raise ValueError("paired pilot is frozen to Town10HD_Opt")
    if geometry.get("expected_lane_id_by_role") != CURBSIDE_EXPECTED_LANE_IDS:
        raise ValueError("legal opposing-lane IDs have drifted")
    if not _numeric_sequence_close(
        geometry["occluder"]["transform"], CURBSIDE_OCCLUDER_TRANSFORM
    ):
        raise ValueError("curbside occluder transform has drifted")
    pedestrian = geometry["pedestrian"]
    if not _numeric_sequence_close(
        pedestrian["start_transform"], CURBSIDE_WALKER_START
    ) or not _numeric_sequence_close(
        pedestrian["end_location"], CURBSIDE_WALKER_END
    ):
        raise ValueError("controlled pedestrian geometry has drifted")
    if not math.isclose(
        float(pedestrian["carla_control_to_physical_scale"]),
        CARLA_WALKER_CONTROL_TO_PHYSICAL_SCALE,
        abs_tol=1e-12,
    ):
        raise ValueError("controlled pedestrian CARLA speed conversion has drifted")

    contract_path = _repo_path(config["contract_config"])
    contract_summary = load_and_validate_pilot_config(contract_path)
    with contract_path.open("r", encoding="utf-8") as stream:
        contract_config = yaml.safe_load(stream)
    if implementation_status == "reviewed_pilot_only" and not bool(
        contract_summary.get("live_run_authorized")
    ):
        raise ValueError("reviewed integration requires a separately reviewed pilot contract")
    source_path = _repo_path(config["source_collection_config"])
    source = advisor._load_config(source_path)
    expected_collector = (
        REPO_ROOT / "data_collection" / "phase2_paired_causal_collector.py"
    ).resolve()
    if _repo_path(config["collector"]) != expected_collector:
        raise ValueError("paired pilot collector entrypoint has drifted")

    trajectories = config.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 2:
        raise ValueError("paired pilot integration requires exactly two trajectories")
    if {str(item["scenario_role"]) for item in trajectories} != REQUIRED_SCENARIO_ROLES:
        raise ValueError("paired pilot scenario roles are incomplete")
    if len({str(item["trajectory_id"]) for item in trajectories}) != 2:
        raise ValueError("paired pilot trajectory IDs must be unique")
    positive = next(
        item for item in trajectories if item["scenario_role"] == "controlled_positive_occlusion"
    )
    benign = next(
        item for item in trajectories if item["scenario_role"] == "matched_benign_negative"
    )
    for field in ("seed", "population_family", "matched_pair_id"):
        if positive[field] != benign[field]:
            raise ValueError(f"positive/benign pilot trajectories must match {field}")
    if not bool(positive["controlled_pedestrian"]) or bool(
        benign["controlled_pedestrian"]
    ):
        raise ValueError("the only registered hazard difference must be positive pedestrian on")
    if not str(positive["target_truth_role_prefix"]) or str(
        benign["target_truth_role_prefix"]
    ):
        raise ValueError("only the positive trajectory may register a target truth role")

    clock = config["clock"]
    capture = config["capture"]
    if clock["owner"] != "paired_orchestrator" or int(clock["tm_port"]) != 8010:
        raise ValueError("paired orchestrator must own TM 8010")
    if not math.isclose(float(clock["world_hz"]), 10.0, abs_tol=1e-12):
        raise ValueError("paired pilot world clock must be 10 Hz")
    if not math.isclose(float(clock["fixed_delta_seconds"]), 0.1, abs_tol=1e-12):
        raise ValueError("paired pilot fixed delta must be 0.1 s")
    expected_frames = round(
        float(clock["world_hz"]) * float(capture["controlled_window_seconds"])
    )
    if int(capture["frames_per_trajectory"]) != expected_frames:
        raise ValueError("capture frame count does not match duration and world clock")
    if bool(capture["warnings_actuated"]):
        raise ValueError("C2 pilot warnings must not actuate")
    if set(capture["required_roles"]) != {"helper", "recipient"}:
        raise ValueError("capture roles must be helper and recipient")
    compute = capture.get("compute_assignment")
    if not isinstance(compute, Mapping):
        raise ValueError("compute assignment mapping is required")
    if compute.get("status") != "correctness_only_shared_gpu":
        raise ValueError("pilot compute must be labeled correctness_only_shared_gpu")
    if compute.get("purpose") != "causal_capture_correctness_not_inference_benchmarking":
        raise ValueError("pilot compute purpose label is missing")
    if int(compute.get("host_gpu_count", -1)) != 1:
        raise ValueError("L10319 pilot compute contract is one shared GPU")
    if bool(compute.get("shared_gpu_timing_is_citable")):
        raise ValueError("shared-GPU inference timing cannot be treated as citable")
    for field in ("front_device_by_role", "back_device_by_role"):
        if not isinstance(compute.get(field), Mapping) or set(compute[field]) != {
            "helper", "recipient"
        }:
            raise ValueError(f"compute assignment {field} must name helper and recipient")
    if implementation_status == "reviewed_pilot_only":
        snapshot = compute.get("reviewed_host_snapshot")
        if not isinstance(snapshot, Mapping) or snapshot.get("verdict") != (
            "accepted_for_correctness_only_pilot"
        ):
            raise ValueError("reviewed integration requires the accepted host-GPU snapshot")
        contract_gpu = contract_config["review_evidence"]["host_gpu_capacity"]
        for field in (
            "gpu_name", "memory_total_mib", "memory_used_mib", "memory_free_mib",
            "utilization_percent_with_carla", "carla_gpu_memory_mib", "verdict",
        ):
            if snapshot.get(field) != contract_gpu.get(field):
                raise ValueError(f"integration/contract GPU evidence differs for {field}")

    roles = config["roles"]
    if set(roles) != {"helper", "recipient"}:
        raise ValueError("role configuration must be helper and recipient")
    if len({int(item["ego_spawn_index"]) for item in roles.values()}) != 2:
        raise ValueError("helper and recipient must use distinct ego spawns")
    if int(roles["helper"]["ego_spawn_index"]) != 61 or int(
        roles["recipient"]["ego_spawn_index"]
    ) != 152:
        raise ValueError("reviewed curbside spawn indices have drifted")
    if not _numeric_sequence_close(
        roles["helper"]["expected_transform"], CURBSIDE_HELPER_TRANSFORM
    ) or not _numeric_sequence_close(
        roles["recipient"]["expected_transform"], CURBSIDE_RECIPIENT_TRANSFORM
    ):
        raise ValueError("reviewed paired ego transforms have drifted")
    all_ports = [
        int(port)
        for role_ports in capture["ports"].values()
        for port in role_ports.values()
    ]
    if len(all_ports) != 8 or len(set(all_ports)) != 8:
        raise ValueError("all helper/recipient loopback UDP ports must be unique")
    if int(config["verification"]["required_gate_count"]) != 9:
        raise ValueError("paired pilot verifier must retain all nine hard gates")
    if bool(config["verification"]["performance_gain_is_a_gate"]):
        raise ValueError("pilot performance gain must not be an acceptance gate")

    route_paths = {
        role: _repo_path(geometry["route_progress_csv_by_role"][role])
        for role in ROLE_NAMES
    }
    for path in (
        contract_path,
        source_path,
        _repo_path(config["collector"]),
        *route_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"paired-pilot prerequisite missing: {path}")
    routes = {role: load_route_progress(path) for role, path in route_paths.items()}
    if len(routes["helper"]) != 18 or len(routes["recipient"]) != 28:
        raise ValueError("reviewed curbside route point counts have drifted")
    helper_start = world_transform(CURBSIDE_HELPER_TRANSFORM).location
    recipient_start = world_transform(CURBSIDE_RECIPIENT_TRANSFORM).location
    if math.hypot(
        float(routes["helper"][0].x - helper_start.x),
        float(routes["helper"][0].y - helper_start.y),
    ) > 3.0:
        raise ValueError("helper route no longer begins near the reviewed spawn")
    if math.hypot(
        float(routes["recipient"][0].x - recipient_start.x),
        float(routes["recipient"][0].y - recipient_start.y),
    ) > 0.25:
        raise ValueError("recipient route no longer begins at the reviewed spawn")
    return config, source, contract_summary


def _collector_command(
    config: Mapping[str, object],
    source: Mapping[str, object],
    trajectory: Mapping[str, object],
    role: str,
    trajectory_dir: Path,
) -> list[str]:
    capture = config["capture"]
    role_config = config["roles"][role]
    ports = capture["ports"][role]
    role_dir = trajectory_dir / role
    coordinator_dir = trajectory_dir / "coordination"
    dropped = {
        "--sync-world", "--async-world", "--external-sync-ticker",
        "--sensor-platform",
        "--ego-spawn-index", "--ego-role-name", "--max-frames",
        "--ego-spawn-forward-offset-m", "--ego-spawn-right-offset-m",
        "--ego-spawn-z-offset-m", "--ego-spawn-yaw-offset-deg",
        "--ego-freeze", "--no-ego-freeze",
        "--ego-fixed-path-spawn-indices", "--ego-fixed-path-progress-csv",
        "--ego-fixed-path-loop", "--no-ego-fixed-path-loop",
        "--transport-label", "--fps", "--world-tick-hz",
        "--camera-width", "--camera-height", "--camera-fov",
        "--radar-points-per-second", "--radar-raster-radius-px",
        "--radar-temporal-window-frames", "--npc-vehicles", "--npc-pedestrians",
        "--camera-source-port", "--remote-port", "--remote-source-port",
        "--camera-result-port", "--run-id", "--run-group", "--metrics-run-dir",
        "--enable-run-logging", "--disable-run-logging", "--ego-route-control",
        "--front-device", "--back-device",
    }
    inherited = _drop_options(source["common_args"], dropped)
    inherited.extend(
        [
            "--async-world",
            "--external-sync-ticker",
            "--sensor-platform", "ego_vehicle",
            "--ego-spawn-index", str(role_config["ego_spawn_index"]),
            "--ego-spawn-require-exact",
            "--ego-spawn-forward-offset-m", str(role_config["ego_spawn_forward_offset_m"]),
            "--ego-spawn-right-offset-m", str(role_config["ego_spawn_right_offset_m"]),
            "--ego-spawn-z-offset-m", str(role_config["ego_spawn_z_offset_m"]),
            "--ego-spawn-yaw-offset-deg", str(role_config["ego_spawn_yaw_offset_deg"]),
            "--ego-freeze",
            "--ego-role-name", str(role_config["ego_role_name"]),
            "--ego-route-control", "traffic_manager",
            "--front-device", str(capture["compute_assignment"]["front_device_by_role"][role]),
            "--back-device", str(capture["compute_assignment"]["back_device_by_role"][role]),
            "--npc-vehicles", "0",
            "--npc-pedestrians", "0",
            "--fps", "10.0",
            "--world-tick-hz", "10.0",
            "--camera-width", "1280",
            "--camera-height", "720",
            "--camera-fov", "120.0",
            "--radar-points-per-second", "200000",
            "--radar-raster-radius-px", "4",
            "--radar-temporal-window-frames", "2",
            "--max-frames", str(capture["frames_per_trajectory"]),
            "--transport-label", f"phase2_paired_{role}_loopback",
            "--camera-source-port", str(ports["camera_source"]),
            "--remote-port", str(ports["remote"]),
            "--remote-source-port", str(ports["remote_source"]),
            "--camera-result-port", str(ports["camera_result"]),
            "--run-id", f"{trajectory['trajectory_id']}_{role}",
            "--run-group", str(trajectory["matched_pair_id"]),
            "--metrics-run-dir", str(role_dir),
            "--enable-run-logging",
            "--headless",
            "--phase2-role", role,
            "--phase2-trajectory-id", str(trajectory["trajectory_id"]),
            "--phase2-scenario-role", str(trajectory["scenario_role"]),
            "--phase2-contract-config", str(_repo_path(config["contract_config"])),
            "--phase2-geometry-id", str(config["scenario_geometry"]["layout_id"]),
            "--phase2-motion-owner", "external_orchestrator",
            "--phase2-ready-sentinel", str(coordinator_dir / f"{role}.ready.json"),
            "--phase2-capture-start-sentinel", str(coordinator_dir / "capture.start.json"),
            "--phase2-tick-ready", str(coordinator_dir / f"{role}.tick_ready.json"),
            "--phase2-heartbeat", str(coordinator_dir / f"{role}.heartbeat.json"),
            "--phase2-start-timeout-s", str(config["clock"]["startup_timeout_s"]),
        ]
    )
    # Remove duplicate boolean flags inherited from the v5 chain, preserving
    # last-option semantics for valued options.
    deduped: list[str] = []
    for token in inherited:
        if token in {"--headless", "--enable-run-logging", "--sensor-every-tick"}:
            if token in deduped:
                continue
        deduped.append(token)
    _require_inherited_contract(deduped)
    return [
        sys.executable,
        "-m",
        "data_collection.phase2_paired_causal_collector",
        *deduped,
    ]


def build_plan(
    config: Mapping[str, object], source: Mapping[str, object], output_dir: Path
) -> dict:
    trajectories = []
    for trajectory in config["trajectories"]:
        trajectory_dir = output_dir / str(trajectory["trajectory_id"])
        trajectories.append(
            {
                **dict(trajectory),
                "trajectory_dir": str(trajectory_dir),
                "population_mode": config["scenario_geometry"]["population_mode"],
                "population_commands": [],
                "collector_commands": {
                    role: _collector_command(
                        config, source, trajectory, role, trajectory_dir
                    )
                    for role in ("helper", "recipient")
                },
            }
        )
    return {
        "schema": "scenesense.phase2_paired_causal_plan.v1",
        "implementation_status": config["implementation_status"],
        "live_authorized": bool(config["authorization"]["carla_launch"]),
        "scenario_geometry_status": config["scenario_geometry"]["status"],
        "scenario_geometry_id": config["scenario_geometry"]["layout_id"],
        "population_mode": config["scenario_geometry"]["population_mode"],
        "compute_purpose": config["capture"]["compute_assignment"]["purpose"],
        "inference_timing_citable": False,
        "single_sync_ticker": True,
        "trajectories": trajectories,
    }


def _read_heartbeat(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _require_udp_ports_available(config: Mapping[str, object]) -> None:
    """Fail before CARLA mutation if any paired loopback port is already bound."""

    ports = sorted(
        int(port)
        for role_ports in config["capture"]["ports"].values()
        for port in role_ports.values()
    )
    sockets = []
    try:
        for port in ports:
            candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                candidate.bind(("127.0.0.1", int(port)))
            except OSError as exc:
                candidate.close()
                raise RuntimeError(
                    f"paired pilot UDP port {port} is unavailable: {exc}"
                ) from exc
            sockets.append(candidate)
    finally:
        for candidate in sockets:
            candidate.close()


def _wait_for_ready(
    world: object,
    processes: Mapping[str, subprocess.Popen],
    paths: Mapping[str, Path],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        failures = {
            role: process.returncode
            for role, process in processes.items()
            if process.poll() is not None
        }
        if failures:
            raise RuntimeError(f"paired collector exited before readiness: {failures}")
        world.tick(2.0)
        if all(path.is_file() for path in paths.values()):
            return
        time.sleep(0.01)
    raise RuntimeError("timed out waiting for both paired collectors to become ready")


def _wait_for_frame(
    processes: Mapping[str, subprocess.Popen],
    heartbeat_paths: Mapping[str, Path],
    tick_ready_paths: Mapping[str, Path],
    target_frame: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        failures = {
            role: process.returncode
            for role, process in processes.items()
            if process.poll() is not None
        }
        if failures:
            raise RuntimeError(f"paired collector exited before frame {target_frame}: {failures}")
        heartbeats = {
            role: _read_heartbeat(path) for role, path in heartbeat_paths.items()
        }
        advanced_heartbeats = {
            role: heartbeat
            for role, heartbeat in heartbeats.items()
            if heartbeat is not None
            and int(heartbeat.get("frame_id", -1)) > int(target_frame)
        }
        if advanced_heartbeats:
            raise RuntimeError(
                "paired completion barrier advanced unexpectedly: "
                f"expected frame={int(target_frame)}, observed={advanced_heartbeats}"
            )
        if all(
            heartbeat is not None
            and heartbeat.get("status") == "frame_complete"
            and heartbeat.get("source_role") == role
            and int(heartbeat.get("frame_id", -1)) == int(target_frame)
            for role, heartbeat in heartbeats.items()
        ):
            return
        # If a collector has armed the following tick without publishing this
        # frame's completion heartbeat, it missed the sensor frame. Fail fast
        # instead of waiting for the full timeout or silently skipping data.
        tick_ready = {
            role: _read_heartbeat(path) for role, path in tick_ready_paths.items()
        }
        skipped = {
            role: payload
            for role, payload in tick_ready.items()
            if payload is not None
            and int(payload.get("after_frame_id", -1)) >= int(target_frame)
            and not (
                heartbeats.get(role) is not None
                and heartbeats[role].get("status") == "frame_complete"
                and heartbeats[role].get("source_role") == role
                and int(heartbeats[role].get("frame_id", -1)) == int(target_frame)
            )
        }
        if skipped:
            raise RuntimeError(
                f"paired collector skipped CARLA frame {target_frame}: {skipped}"
            )
        time.sleep(0.01)
    observed = {
        role: _read_heartbeat(path) for role, path in heartbeat_paths.items()
    }
    raise RuntimeError(
        f"paired collectors did not both complete CARLA frame {target_frame}; "
        f"observed_heartbeats={observed}"
    )


def _wait_for_tick_ready(
    processes: Mapping[str, subprocess.Popen],
    tick_ready_paths: Mapping[str, Path],
    after_frame: int,
    timeout_s: float,
) -> None:
    """Wait for both pre-action decisions before allowing exactly one tick."""

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        failures = {
            role: process.returncode
            for role, process in processes.items()
            if process.poll() is not None
        }
        if failures:
            raise RuntimeError(
                f"paired collector exited before arming frame {int(after_frame) + 1}: "
                f"{failures}"
            )
        payloads = {
            role: _read_heartbeat(path) for role, path in tick_ready_paths.items()
        }
        advanced = {
            role: payload
            for role, payload in payloads.items()
            if payload is not None
            and int(payload.get("after_frame_id", -1)) > int(after_frame)
        }
        if advanced:
            raise RuntimeError(
                "paired tick-ready barrier advanced unexpectedly: "
                f"expected after_frame={int(after_frame)}, observed={advanced}"
            )
        if all(
            payload is not None
            and payload.get("status") == "armed_for_next_frame"
            and payload.get("source_role") == role
            and int(payload.get("after_frame_id", -1)) == int(after_frame)
            and int(payload.get("minimum_capture_frame", -1)) == int(after_frame) + 1
            for role, payload in payloads.items()
        ):
            return
        time.sleep(0.01)
    observed = {
        role: _read_heartbeat(path) for role, path in tick_ready_paths.items()
    }
    raise RuntimeError(
        f"paired collectors did not both arm CARLA frame {int(after_frame) + 1}; "
        f"observed_tick_ready={observed}"
    )


def _write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_progress(path: Path, event: str, **fields: object) -> None:
    payload = {
        "schema": "scenesense.phase2_pilot_progress.v1",
        "event": str(event),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _find_role_actor(world: object, role_name: str) -> object:
    matches = [
        actor
        for actor in world.get_actors().filter("vehicle.*")
        if str(actor.attributes.get("role_name", "")) == str(role_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one live vehicle role={role_name!r}, found {len(matches)}")
    return matches[0]


def _prepare_reviewed_ego_motion(
    world: object,
    config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, DirectRouteController], dict]:
    geometry = config["scenario_geometry"]
    transforms = {}
    actors = {}
    controllers = {}
    realized = {}
    for role in ROLE_NAMES:
        role_config = config["roles"][role]
        actor = _find_role_actor(world, str(role_config["ego_role_name"]))
        transform = actor.get_transform()
        expected = world_transform(role_config["expected_transform"])
        pose_error_m = float(transform.location.distance(expected.location))
        yaw_error_deg = abs(
            wrap_degrees(float(transform.rotation.yaw) - float(expected.rotation.yaw))
        )
        realized[role] = {
            "actor_id": int(actor.id),
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw_deg": float(transform.rotation.yaw),
            "pose_error_m": pose_error_m,
            "yaw_error_deg": yaw_error_deg,
        }
        if pose_error_m > float(geometry["maximum_pose_error_m"]):
            raise RuntimeError(f"{role} realized pose drifted: {realized[role]}")
        if yaw_error_deg > float(geometry["maximum_yaw_error_deg"]):
            raise RuntimeError(f"{role} realized yaw drifted: {realized[role]}")
        actors[role] = actor
        transforms[role] = transform

    lane_contract = legal_opposing_lane_contract(
        world.get_map(),
        transforms,
        expected_lane_ids=geometry["expected_lane_id_by_role"],
        maximum_heading_error_deg=float(geometry["maximum_lane_heading_error_deg"]),
    )
    for role in ROLE_NAMES:
        actor = actors[role]
        actor.set_autopilot(False, int(config["clock"]["tm_port"]))
        actor.set_simulate_physics(True)
        actor.apply_control(
            advisor.carla.VehicleControl(
                throttle=0.0, brake=0.0, hand_brake=False
            )
        )
        route = load_route_progress(
            _repo_path(geometry["route_progress_csv_by_role"][role])
        )
        controllers[role] = DirectRouteController(
            actor,
            route,
            target_speed_mps=float(config["roles"][role]["target_speed_mps"]),
        )
    return actors, controllers, {
        "schema": "scenesense.phase2_realized_geometry.v1",
        "geometry_id": geometry["layout_id"],
        "roles": realized,
        "lane_contract": lane_contract,
        "motion_owner": "paired_orchestrator",
    }


def _write_scenario_artifacts(
    trajectory_dir: Path, runtime: CurbsideScenarioRuntime
) -> dict:
    scenario_dir = trajectory_dir / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    summary = runtime.summary()
    (scenario_dir / "realization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if runtime.trace:
        with (scenario_dir / "realized_trace.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(runtime.trace[0]))
            writer.writeheader()
            writer.writerows(runtime.trace)
    return summary


def _add_ego_collision_sensors(
    monitor: advisor.TrafficSanityMonitor,
    world: object,
    role_config: Mapping[str, object],
) -> list[int]:
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    ids = []
    for role in ROLE_NAMES:
        role_name = str(role_config[role]["ego_role_name"])
        actor = _find_role_actor(world, role_name)
        actor_id = int(actor.id)
        ids.append(actor_id)
        monitor.actor_ids.append(actor_id)
        monitor.actor_metadata[actor_id] = {
            "role_name": role_name,
            "type_id": str(actor.type_id),
        }
        sensor = world.spawn_actor(collision_bp, advisor.carla.Transform(), attach_to=actor)
        sensor.listen(
            lambda event, ego_id=actor_id: monitor._on_collision(ego_id, event)
        )
        monitor.collision_sensors.append(sensor)
    return ids


def _wait_collectors_exit(
    world: object,
    processes: Mapping[str, subprocess.Popen],
    timeout_s: float,
) -> dict[str, int]:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes.values()
    ):
        time.sleep(0.02)
    # A final bounded tick can release deferred sensor destruction, but only
    # after every registered frame heartbeat has already completed.
    for _unused in range(3):
        if not any(process.poll() is None for process in processes.values()):
            break
        world.tick(2.0)
        time.sleep(0.02)
    alive = [role for role, process in processes.items() if process.poll() is None]
    if alive:
        raise RuntimeError(f"paired collectors did not exit after capture: {alive}")
    return {role: int(process.returncode) for role, process in processes.items()}


def run_live(
    config: Mapping[str, object],
    source: Mapping[str, object],
    plan: Mapping[str, object],
    output_dir: Path,
) -> None:
    """Run only the reviewed two-trajectory causal-capture pilot.

    There is deliberately no advisor traffic population here.  This pilot
    validates causal capture on the frozen curbside pair; environmental
    variation belongs to the post-pilot corpus design.
    """

    if not bool(plan.get("live_authorized")):
        raise RuntimeError(
            "live paired pilot is HOLD: reviewed launch authorization is false"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    _require_udp_ports_available(config)
    _append_progress(
        progress_path,
        "launch_validation_passed",
        trajectory_count=2,
        geometry_id=plan["scenario_geometry_id"],
    )
    _write_json_create(output_dir / "resolved_plan.json", plan)
    with (output_dir / "resolved_integration_config.yaml").open(
        "x", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=False)
    batch_manifest: MutableMapping[str, object] = {
        "schema": "scenesense.phase2_paired_pilot_batch.v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warnings_actuated": False,
        "population_mode": config["scenario_geometry"]["population_mode"],
        "geometry_id": config["scenario_geometry"]["layout_id"],
        "inference_timing_citable": False,
        "trajectories": [],
    }
    _write_json_create(output_dir / "batch_manifest.json", batch_manifest)
    manifest_path = output_dir / "batch_manifest.json"

    client, world = advisor._connect(source)
    advisor._require_empty_async(world)
    try:
        for trajectory_plan in plan["trajectories"]:
            trajectory_id = str(trajectory_plan["trajectory_id"])
            trajectory_dir = output_dir / trajectory_id
            trajectory_dir.mkdir(parents=True, exist_ok=False)
            (trajectory_dir / "coordination").mkdir()
            record: MutableMapping[str, object] = {
                "trajectory_id": trajectory_id,
                "scenario_role": trajectory_plan["scenario_role"],
                "matched_pair_id": trajectory_plan["matched_pair_id"],
                "seed": int(trajectory_plan["seed"]),
                "status": "running",
            }
            batch_manifest["trajectories"].append(record)
            manifest_path.write_text(
                json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _append_progress(
                progress_path,
                "trajectory_started",
                trajectory_id=trajectory_id,
                scenario_role=trajectory_plan["scenario_role"],
            )
            original_settings = None
            collector_processes: Dict[str, subprocess.Popen] = {}
            collector_streams = []
            traffic_monitor = None
            scenario_runtime: Optional[CurbsideScenarioRuntime] = None
            try:
                if not bool(
                    config["scenario_geometry"]["reload_world_before_trajectory"]
                ):
                    raise RuntimeError("world reload contract was disabled after validation")
                world = client.load_world(
                    str(config["scenario_geometry"]["town"]), True
                )
                record["world_reload"] = {
                    "requested_map": str(config["scenario_geometry"]["town"]),
                    "resolved_map": str(world.get_map().name),
                }
                advisor._require_empty_async(world)
                original_settings = advisor._set_sync_master(
                    client,
                    world,
                    int(config["clock"]["tm_port"]),
                    float(config["clock"]["fixed_delta_seconds"]),
                )

                pedestrian = config["scenario_geometry"]["pedestrian"]
                scenario_runtime = CurbsideScenarioRuntime(
                    world,
                    hazard_present=bool(trajectory_plan["controlled_pedestrian"]),
                    pedestrian_role_name=str(pedestrian["role_name"]),
                    pedestrian_start_delay_s=float(pedestrian["start_delay_s"]),
                    pedestrian_speed_mps=float(pedestrian["physical_speed_mps"]),
                    pedestrian_endpoint_tolerance_m=float(
                        pedestrian["endpoint_tolerance_m"]
                    ),
                )
                scenario_runtime.spawn()
                for _unused in range(5):
                    world.tick(2.0)

                for role in ROLE_NAMES:
                    log_stream = (trajectory_dir / f"{role}.collector.log").open(
                        "x", encoding="utf-8"
                    )
                    process = subprocess.Popen(
                        trajectory_plan["collector_commands"][role],
                        cwd=REPO_ROOT,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    collector_processes[role] = process
                    collector_streams.append(log_stream)
                ready_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.ready.json"
                    for role in ROLE_NAMES
                }
                _wait_for_ready(
                    world,
                    collector_processes,
                    ready_paths,
                    float(config["clock"]["startup_timeout_s"]),
                )
                _append_progress(
                    progress_path,
                    "collectors_ready",
                    trajectory_id=trajectory_id,
                    roles=list(ROLE_NAMES),
                )

                actors, controllers, geometry_summary = _prepare_reviewed_ego_motion(
                    world, config
                )
                record["realized_geometry"] = geometry_summary
                (trajectory_dir / "realized_geometry.json").write_text(
                    json.dumps(
                        geometry_summary, indent=2, sort_keys=True, allow_nan=False
                    )
                    + "\n",
                    encoding="utf-8",
                )
                traffic_monitor = advisor.TrafficSanityMonitor(
                    world=world,
                    traffic_manager=client.get_trafficmanager(
                        int(config["clock"]["tm_port"])
                    ),
                    output_dir=trajectory_dir / "traffic_sanity",
                    integration=source["advisor_integration"],
                )
                traffic_monitor.start()
                record["ego_actor_ids"] = _add_ego_collision_sensors(
                    traffic_monitor, world, config["roles"]
                )

                start_frame = int(world.get_snapshot().frame)
                start_simulation_s = float(
                    world.get_snapshot().timestamp.elapsed_seconds
                )
                _write_json_create(
                    trajectory_dir / "coordination/capture.start.json",
                    {
                        "schema": "scenesense.phase2_capture_barrier.v1",
                        "trajectory_id": trajectory_id,
                        "after_frame_id": start_frame,
                        "next_frame_is_first_capture": True,
                        "motion_owner": "paired_orchestrator",
                    },
                )
                _append_progress(
                    progress_path,
                    "capture_started",
                    trajectory_id=trajectory_id,
                    after_frame_id=start_frame,
                    requested_frames=int(config["capture"]["frames_per_trajectory"]),
                )
                heartbeat_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.heartbeat.json"
                    for role in ROLE_NAMES
                }
                tick_ready_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.tick_ready.json"
                    for role in ROLE_NAMES
                }
                captured_frames = []
                ego_motion_rows = []
                previous_frame = start_frame
                for _unused in range(int(config["capture"]["frames_per_trajectory"])):
                    _wait_for_tick_ready(
                        collector_processes,
                        tick_ready_paths,
                        previous_frame,
                        float(config["clock"]["per_frame_timeout_s"]),
                    )
                    for controller in controllers.values():
                        controller.tick()
                    target_frame = int(world.tick(2.0))
                    snapshot = world.get_snapshot()
                    elapsed_s = float(snapshot.timestamp.elapsed_seconds) - start_simulation_s
                    scenario_runtime.tick(elapsed_s, target_frame)
                    motion_row = {
                        "frame_id": target_frame,
                        "elapsed_s": elapsed_s,
                    }
                    for role, actor in actors.items():
                        transform = actor.get_transform()
                        velocity = actor.get_velocity()
                        motion_row.update(
                            {
                                f"{role}_x": float(transform.location.x),
                                f"{role}_y": float(transform.location.y),
                                f"{role}_z": float(transform.location.z),
                                f"{role}_yaw_deg": float(transform.rotation.yaw),
                                f"{role}_speed_mps": math.sqrt(
                                    float(velocity.x) ** 2
                                    + float(velocity.y) ** 2
                                    + float(velocity.z) ** 2
                                ),
                            }
                        )
                    ego_motion_rows.append(motion_row)
                    _wait_for_frame(
                        collector_processes,
                        heartbeat_paths,
                        tick_ready_paths,
                        target_frame,
                        float(config["clock"]["per_frame_timeout_s"]),
                    )
                    captured_frames.append(target_frame)
                    previous_frame = target_frame
                    if len(captured_frames) % 10 == 0:
                        _append_progress(
                            progress_path,
                            "capture_progress",
                            trajectory_id=trajectory_id,
                            completed_frames=len(captured_frames),
                            requested_frames=int(
                                config["capture"]["frames_per_trajectory"]
                            ),
                            latest_carla_frame=target_frame,
                        )

                with (trajectory_dir / "ego_motion_trace.csv").open(
                    "x", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(ego_motion_rows[0]))
                    writer.writeheader()
                    writer.writerows(ego_motion_rows)
                record["collector_returncodes"] = _wait_collectors_exit(
                    world,
                    collector_processes,
                    float(config["clock"]["shutdown_timeout_s"]),
                )
                if any(code != 0 for code in record["collector_returncodes"].values()):
                    raise RuntimeError(
                        f"paired collector returncodes: {record['collector_returncodes']}"
                    )
                record["captured_frame_count"] = len(captured_frames)
                record["first_frame_id"] = captured_frames[0]
                record["last_frame_id"] = captured_frames[-1]
                scenario_summary = _write_scenario_artifacts(
                    trajectory_dir, scenario_runtime
                )
                record["scenario_realization"] = scenario_summary
                if bool(trajectory_plan["controlled_pedestrian"]):
                    if not bool(scenario_summary["pedestrian_completed"]) or not bool(
                        scenario_summary["pedestrian_physical_speed_gate_pass"]
                    ):
                        raise RuntimeError(
                            f"controlled pedestrian realization failed: {scenario_summary}"
                        )
                elif int(scenario_summary["pedestrian_actor_id"]) != -1:
                    raise RuntimeError("matched benign trajectory unexpectedly spawned a pedestrian")

                traffic_summary = traffic_monitor.stop()
                traffic_monitor = None
                record["traffic_sanity"] = traffic_summary
                record["integrity"] = {
                    "unintended_collision_count": int(
                        traffic_summary.get("collision_events", -1)
                    ),
                    "persistent_gridlock": bool(
                        "persistent_gridlock_above_gate"
                        in set(traffic_summary.get("failures", []))
                    ),
                    "actor_cleanup_complete": False,
                    "dropped_required_stream": False,
                }
                if not bool(traffic_summary.get("pass")):
                    raise RuntimeError(
                        f"traffic sanity failed: {traffic_summary['failures']}"
                    )
                record["status"] = "capture_complete_pending_cleanup"
                _append_progress(
                    progress_path,
                    "capture_complete_pending_cleanup",
                    trajectory_id=trajectory_id,
                    captured_frames=len(captured_frames),
                )
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                batch_manifest["status"] = "failed"
                _append_progress(
                    progress_path,
                    "trajectory_failed",
                    trajectory_id=trajectory_id,
                    error=record["error"],
                )
                raise
            finally:
                for process in collector_processes.values():
                    if process.poll() is None:
                        process.send_signal(advisor.signal.SIGINT)
                deadline = time.monotonic() + min(
                    10.0, float(config["clock"]["shutdown_timeout_s"])
                )
                while time.monotonic() < deadline and any(
                    process.poll() is None for process in collector_processes.values()
                ):
                    if original_settings is not None:
                        try:
                            world.tick(2.0)
                        except RuntimeError:
                            pass
                    time.sleep(0.02)
                for process in collector_processes.values():
                    if process.poll() is None:
                        process.terminate()
                for process in collector_processes.values():
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3.0)
                if traffic_monitor is not None:
                    record["traffic_sanity"] = traffic_monitor.stop()
                    traffic_monitor = None
                if scenario_runtime is not None:
                    scenario_runtime.destroy()
                if original_settings is not None:
                    advisor._restore_async(
                        client,
                        world,
                        int(config["clock"]["tm_port"]),
                        original_settings,
                    )
                    cleanup = advisor._tick_until_empty(
                        world, float(config["clock"]["shutdown_timeout_s"])
                    )
                    record["postflight_dynamic_actor_counts"] = cleanup
                    if "integrity" in record:
                        record["integrity"]["actor_cleanup_complete"] = not any(
                            cleanup.values()
                        )
                for stream in collector_streams:
                    stream.close()
                manifest_path.write_text(
                    json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if not bool(record["integrity"]["actor_cleanup_complete"]):
                raise RuntimeError(
                    f"actor cleanup failed: {record['postflight_dynamic_actor_counts']}"
                )
            record["status"] = "complete"
            _append_progress(
                progress_path,
                "trajectory_complete",
                trajectory_id=trajectory_id,
                postflight_dynamic_actor_counts=record[
                    "postflight_dynamic_actor_counts"
                ],
            )
            manifest_path.write_text(
                json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        batch_manifest["status"] = "complete"
        batch_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _append_progress(
            progress_path,
            "batch_capture_complete",
            trajectory_count=len(batch_manifest["trajectories"]),
            next_gate="offline_replay_and_nine_gate_verification",
        )
        manifest_path.write_text(
            json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        try:
            advisor._require_empty_async(world)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-config", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config, source, contract_summary = _load_config(config_path)
    if args.output_dir is None:
        if args.launch:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = _repo_path(config["output_root"]) / f"{stamp}_pilot"
        else:
            output_dir = Path("/tmp/phase2_paired_plan").resolve()
    else:
        output_dir = args.output_dir.resolve()
    plan = build_plan(config, source, output_dir)
    result = {
        "verdict": "PASS",
        "config": str(config_path),
        "contract": contract_summary,
        "plan": plan,
        "note": "validation/dry-run only; no CARLA or OAI process was started",
    }
    if args.launch:
        try:
            run_live(config, source, plan, output_dir)
        except BaseException as exc:
            if output_dir.is_dir() and not (output_dir / "FAILED.json").exists():
                _write_json_create(
                    output_dir / "FAILED.json",
                    {
                        "schema": "scenesense.phase2_run_sentinel.v1",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "written_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise
        summary = {
            "schema": "scenesense.phase2_run_sentinel.v1",
            "status": "collection_complete_pending_replay_and_verification",
            "batch_root": str(output_dir),
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_create(output_dir / "RESULTS_SUMMARY.json", summary)
        _write_json_create(output_dir / "COMPLETED.json", summary)
        result["note"] = (
            "paired pilot capture completed; stop for replay, nine-gate verification, and human review"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

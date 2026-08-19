#!/usr/bin/env python3
"""Run the Phase-2 ambient-motion contract without sensors or model capture.

This bounded preflight exists to catch population, lane-following, collision,
gridlock, liveness, and cleanup failures before the expensive calibration
audit. It uses the frozen audit config and scenarios but does not produce
perception data and cannot authorize downstream evaluation by itself.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import carla
import pandas as pd
import yaml

from data_collection import run_advisor_policy_corpus as advisor
from data_collection.phase2_calibration_scenario import (
    CalibrationScenarioRuntime,
    resolve_scenario,
)
from data_collection.run_phase2_calibration_audit import (
    DEFAULT_CONFIG,
    POPULATION_RELEASE_SCHEMA,
    _add_collision_sensors,
    _compare_ambient_initial_signatures,
    _compare_ambient_trajectories,
    _load_config,
    _load_world_with_retry,
    _population_command,
    _ambient_counts,
    _require_population_process_alive,
    _scenario_owned_nontreatment_signature,
    _traffic_monitor_integration,
    _validate_population_ready_manifest,
    _wait_for_population_release_ack,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "data_collection/experiments/phase2_traffic_preflight_v1"
PROGRESS_SCHEMA = "scenesense.phase2_traffic_preflight_progress.v1"
SUMMARY_SCHEMA = "scenesense.phase2_traffic_preflight_summary.v1"
MINIMUM_NATIVE_DRIVING_FRACTION = 1.0
MAXIMUM_LANE_CENTER_OFFSET_M = 1.5


def _write_json_create(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_progress(path: Path, event: str, **fields: object) -> None:
    payload = {
        "schema": PROGRESS_SCHEMA,
        "event": event,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _dynamic_counts(world: object) -> dict[str, int]:
    actors = world.get_actors()
    return {
        pattern: int(len(actors.filter(pattern)))
        for pattern in (
            "vehicle.*",
            "walker.pedestrian.*",
            "controller.ai.walker",
            "sensor.*",
        )
    }


def _wait_ready(
    world: object, process: subprocess.Popen, path: Path, timeout_s: float
) -> dict:
    deadline = time.monotonic() + float(timeout_s)
    failure_path = path.with_name("population.failed.json")
    while time.monotonic() < deadline:
        if failure_path.is_file():
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                "population failed before READY: "
                f"{failure.get('error', 'unspecified child failure')}"
            )
        if process.poll() is not None:
            raise RuntimeError(
                "population exited before READY without a failure manifest: "
                f"returncode={process.returncode}"
            )
        world.tick(2.0)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise RuntimeError(f"population READY timed out: {path}")


def _stop_population(process: subprocess.Popen, world: object) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 25.0
    while process.poll() is None and time.monotonic() < deadline:
        world.tick(2.0)
        time.sleep(0.02)
    if process.poll() is None:
        process.terminate()
    return int(process.wait(timeout=5.0))


def _spawn_staged_egos(world: object, config: Mapping[str, object]) -> dict:
    spawn_points = list(world.get_map().get_spawn_points())
    egos = {}
    try:
        for role in ("helper", "recipient"):
            role_config = config["staging_roles"][role]
            index = int(role_config["ego_spawn_index"])
            blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
            blueprint.set_attribute("role_name", str(role_config["ego_role_name"]))
            actor = world.try_spawn_actor(blueprint, spawn_points[index])
            if actor is None:
                raise RuntimeError(f"could not stage {role} at spawn index {index}")
            actor.set_autopilot(False, int(config["clock"]["tm_port"]))
            actor.set_simulate_physics(False)
            egos[role] = actor
        return egos
    except BaseException:
        for actor in reversed(list(egos.values())):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        raise


def _anchor_id(row: Mapping[str, object]) -> str | None:
    value = row.get("route_start_anchor_id")
    return None if value is None or pd.isna(value) else str(value)


def _lane_motion_audit(world: object, rows: Sequence[Mapping[str, object]]) -> dict:
    """Gate every recorded NPC centre on a native driving lane."""

    world_map = world.get_map()
    by_actor: dict[int, dict[str, object]] = {}
    for row in rows:
        actor_id = int(row["actor_id"])
        location = carla.Location(
            x=float(row["world_x"]),
            y=float(row["world_y"]),
            z=float(row["world_z"]),
        )
        native = world_map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        projected = world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        record = by_actor.setdefault(
            actor_id,
            {
                "frame_count": 0,
                "native_driving_frame_count": 0,
                "maximum_lane_center_offset_m": 0.0,
                "road_lane_sequence": [],
            },
        )
        record["frame_count"] = int(record["frame_count"]) + 1
        if native is not None:
            record["native_driving_frame_count"] = int(
                record["native_driving_frame_count"]
            ) + 1
            road_lane = [int(native.road_id), int(native.lane_id)]
            if not record["road_lane_sequence"] or record["road_lane_sequence"][-1] != road_lane:
                record["road_lane_sequence"].append(road_lane)
        if projected is None:
            offset_m = float("inf")
        else:
            offset_m = float(projected.transform.location.distance(location))
        record["maximum_lane_center_offset_m"] = max(
            float(record["maximum_lane_center_offset_m"]), offset_m
        )

    failures = []
    for actor_id, record in sorted(by_actor.items()):
        frame_count = int(record["frame_count"])
        fraction = (
            int(record["native_driving_frame_count"]) / frame_count
            if frame_count
            else 0.0
        )
        record["native_driving_fraction"] = float(fraction)
        if fraction < MINIMUM_NATIVE_DRIVING_FRACTION:
            failures.append(f"actor_{actor_id}_left_native_driving_lane")
        if float(record["maximum_lane_center_offset_m"]) > MAXIMUM_LANE_CENTER_OFFSET_M:
            failures.append(f"actor_{actor_id}_lane_center_offset_above_gate")
    return {
        "pass": not failures,
        "failures": failures,
        "minimum_native_driving_fraction": MINIMUM_NATIVE_DRIVING_FRACTION,
        "maximum_lane_center_offset_gate_m": MAXIMUM_LANE_CENTER_OFFSET_M,
        "actors": {str(key): value for key, value in sorted(by_actor.items())},
    }


def _run_cell(
    client: object,
    config: Mapping[str, object],
    row: Mapping[str, object],
    run_dir: Path,
    *,
    visual_spectator_role: str | None = None,
) -> dict:
    world = _load_world_with_retry(
        client, str(config["carla"]["expected_town"]), True
    )
    advisor._require_empty_async(world)
    original_settings = advisor._set_sync_master(
        client,
        world,
        int(config["clock"]["tm_port"]),
        float(config["clock"]["fixed_delta_seconds"]),
    )
    coordination = run_dir / "coordination"
    coordination.mkdir(parents=True)
    egos = {}
    runtime = None
    monitor = None
    population = None
    population_stream = None
    result = {
        "trajectory_id": str(row["trajectory_id"]),
        "group_id": str(row["group_id"]),
        "scenario_role": str(row["scenario_role"]),
        "geometry_or_route_id": str(row["geometry_or_route_id"]),
        "traffic_density": str(row["traffic_density"]),
        "visual_spectator_role": visual_spectator_role,
    }
    try:
        egos = _spawn_staged_egos(world, config)
        scenario = resolve_scenario(
            world.get_map(),
            geometry_or_route_id=str(row["geometry_or_route_id"]),
            scenario_role=str(row["scenario_role"]),
            route_start_anchor_id=_anchor_id(row),
        )
        runtime = CalibrationScenarioRuntime(
            world,
            scenario,
            egos,
            tm_port=int(config["clock"]["tm_port"]),
            helper_speed_mps=float(config["staging_roles"]["helper"]["target_speed_mps"]),
            recipient_speed_mps=float(config["staging_roles"]["recipient"]["target_speed_mps"]),
            pedestrian_speed_mps=float(config["controlled_motion"]["pedestrian_speed_mps"]),
            pedestrian_start_delay_s=float(config["controlled_motion"]["pedestrian_start_delay_s"]),
        )
        result["ego_placement"] = runtime.place_egos()
        result["controlled_setup"] = runtime.spawn_controlled_actors()
        layer_id, layer, counts = _ambient_counts(config, row)
        result["ambient_evidence_layer"] = layer_id
        result["ambient_evidence_role"] = str(layer["evidence_role"])
        result["ambient_counts"] = dict(counts)
        result["ambient_motion_contract"] = {
            "vehicle": str(layer["released_vehicle_motion_mode"]),
            "walker": str(layer["released_walker_motion_mode"]),
        }
        population_required = bool(layer["population_process_required"])
        expected_population_mode = (
            "naturalistic_tm" if population_required else "scenario_owned_only"
        )
        if str(row["ambient_population_mode"]) != expected_population_mode:
            raise RuntimeError(
                "manifest/runtime ambient-population mode mismatch: "
                f"manifest={row['ambient_population_mode']} "
                f"runtime={expected_population_mode}"
            )
        result["ambient_population_mode"] = expected_population_mode
        result["scenario_owned_nontreatment_signature"] = (
            _scenario_owned_nontreatment_signature(world, config)
        )
        if population_required:
            command = _population_command(config, row, scenario, coordination)
            result["population_command"] = command
            population_stream = (run_dir / "population.log").open(
                "x", encoding="utf-8"
            )
            population = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=population_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ready = _wait_ready(
                world,
                population,
                coordination / "population.ready.json",
                float(config["ambient_traffic"]["population_start_timeout_s"]),
            )
            result["spawn_signature"] = _validate_population_ready_manifest(
                ready,
                vehicles=int(counts["vehicles"]),
                walkers=int(counts["walkers"]),
            )
        else:
            if int(counts["vehicles"]) or int(counts["walkers"]):
                raise RuntimeError(
                    "scenario-owned-only layer requested generic ambient actors"
                )
            result["population_command"] = None
            result["population_ready"] = {
                "applicable": False,
                "basis": "scenario_owned_only_no_population_process",
            }
            result["spawn_signature"] = []

        monitor = advisor.TrafficSanityMonitor(
            world=world,
            traffic_manager=client.get_trafficmanager(int(config["clock"]["tm_port"])),
            output_dir=run_dir / "traffic_sanity",
            integration=_traffic_monitor_integration(config, scenario),
        )
        monitor.start()
        _add_collision_sensors(monitor, world, [*egos.values(), *runtime.owned])
        if population_required:
            assert population is not None
            _write_json_create(
                coordination / "population.release.json",
                {
                    "schema": POPULATION_RELEASE_SCHEMA,
                    "trajectory_id": str(row["trajectory_id"]),
                    "release_basis": "traffic_preflight_monitor_owned",
                },
            )
            result["population_release"] = _wait_for_population_release_ack(
                population,
                coordination / "population.released.json",
                float(config["ambient_traffic"]["population_start_timeout_s"]),
            )
        else:
            result["population_release"] = {
                "applicable": False,
                "basis": "scenario_owned_only_no_population_release",
            }
        monitor.activate_vehicle_motion(client)
        setup_barrier_frame_id = int(world.tick(2.0))
        result[
            "release_barrier_frame_id"
            if population_required
            else "scenario_setup_barrier_frame_id"
        ] = setup_barrier_frame_id
        runtime.activate_motion()
        start_s = float(world.get_snapshot().timestamp.elapsed_seconds)
        for frame_index in range(int(config["clock"]["frames_per_trajectory"])):
            if population is not None:
                _require_population_process_alive(
                    population,
                    phase=f"traffic preflight frame {frame_index} pre-tick",
                )
            runtime.before_tick(
                frame_index * float(config["clock"]["fixed_delta_seconds"])
            )
            monitor.before_world_tick()
            monitor.raise_if_failed()
            if visual_spectator_role is not None:
                followed = egos[visual_spectator_role].get_transform()
                forward = followed.get_forward_vector()
                world.get_spectator().set_transform(
                    carla.Transform(
                        carla.Location(
                            x=float(followed.location.x) - 10.0 * float(forward.x),
                            y=float(followed.location.y) - 10.0 * float(forward.y),
                            z=float(followed.location.z) + 5.0,
                        ),
                        carla.Rotation(
                            pitch=-15.0,
                            yaw=float(followed.rotation.yaw),
                            roll=0.0,
                        ),
                    )
                )
            frame_id = int(world.tick(2.0))
            snapshot = world.get_snapshot()
            elapsed_s = float(snapshot.timestamp.elapsed_seconds) - start_s
            runtime.after_tick(frame_id, elapsed_s)
            monitor.observe_snapshot(snapshot)
            if population is not None:
                _require_population_process_alive(
                    population,
                    phase=f"traffic preflight frame {frame_index} post-tick",
                )
            monitor.raise_if_failed()

        result["scenario_summary"] = runtime.summary()
        result["lane_motion_audit"] = _lane_motion_audit(
            world, list(monitor.trajectory_rows)
        )
        result["traffic_sanity"] = monitor.stop()
        monitor = None
        if population is not None:
            result["population_exit_code"] = _stop_population(population, world)
            population = None
            if result["population_exit_code"] != 0:
                raise RuntimeError(
                    f"population cleanup returned {result['population_exit_code']}"
                )
        else:
            result["population_exit_code"] = None
        if not bool(result["traffic_sanity"].get("pass")):
            raise RuntimeError(
                "traffic sanity failed: "
                f"{result['traffic_sanity'].get('failures')}"
            )
        if not bool(result["lane_motion_audit"].get("pass")):
            raise RuntimeError(
                "lane motion audit failed: "
                f"{result['lane_motion_audit'].get('failures')}"
            )
        result["pass"] = True
        return result
    finally:
        if monitor is not None:
            result["traffic_sanity"] = monitor.stop()
        if population is not None:
            result["population_exit_code"] = _stop_population(population, world)
        if population_stream is not None:
            population_stream.close()
        if runtime is not None:
            runtime.destroy()
        for actor in reversed(list(egos.values())):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        world.tick(2.0)
        advisor._restore_async(
            client,
            world,
            int(config["clock"]["tm_port"]),
            original_settings,
        )
        deadline = time.monotonic() + 10.0
        dynamic_counts = _dynamic_counts(world)
        while any(dynamic_counts.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
            dynamic_counts = _dynamic_counts(world)
        result["postflight_dynamic_counts"] = dynamic_counts
        if any(dynamic_counts.values()):
            raise RuntimeError(f"dynamic actor leak: {dynamic_counts}")


def _select_rows(
    selected: pd.DataFrame, trajectory_ids: Sequence[str], all_cells: bool
) -> list[dict]:
    if all_cells:
        return selected.to_dict("records")
    requested = list(dict.fromkeys(str(value) for value in trajectory_ids))
    available = set(selected["trajectory_id"].astype(str))
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"unknown audit trajectory IDs: {missing}")
    by_id = selected.set_index(selected["trajectory_id"].astype(str), drop=False)
    return [by_id.loc[trajectory_id].to_dict() for trajectory_id in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--trajectory-id", action="append", default=[])
    selection.add_argument("--all-audit-cells", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--visual-spectator-role",
        choices=("helper", "recipient"),
        help=(
            "move only CARLA's spectator camera behind this ego for manual "
            "inspection; it does not alter captured sensors"
        ),
    )
    parser.add_argument("--launch", action="store_true", required=True)
    args = parser.parse_args()

    config, source, selected = _load_config(args.config.resolve())
    rows = _select_rows(selected, args.trajectory_id, args.all_audit_cells)
    if not rows:
        raise ValueError("traffic preflight selection is empty")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_preflight")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    with (output_dir / "resolved_config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=False)
    _write_json_create(
        output_dir / "plan.json",
        {
            "schema": "scenesense.phase2_traffic_preflight_plan.v1",
            "trajectory_ids": [str(row["trajectory_id"]) for row in rows],
            "trajectory_count": len(rows),
            "sensors_started": False,
            "model_started": False,
            "oai_started": False,
        },
    )

    client = None
    results = []
    pair_checks = []
    verdict = "PASS"
    error = None
    try:
        client, _world = advisor._connect(source)
        for ordinal, row in enumerate(rows, start=1):
            trajectory_id = str(row["trajectory_id"])
            _append_progress(
                progress_path,
                "trajectory_started",
                trajectory_id=trajectory_id,
                ordinal=ordinal,
                total=len(rows),
            )
            run_dir = output_dir / trajectory_id
            run_dir.mkdir()
            result = _run_cell(
                client,
                config,
                row,
                run_dir,
                visual_spectator_role=args.visual_spectator_role,
            )
            results.append(result)
            _write_json_create(run_dir / "result.json", result)
            _append_progress(
                progress_path,
                "trajectory_complete",
                trajectory_id=trajectory_id,
                ordinal=ordinal,
                total=len(rows),
            )

        by_group = {}
        for result in results:
            by_group.setdefault(result["group_id"], {})[
                result["scenario_role"]
            ] = result
        for group_id, arms in sorted(by_group.items()):
            positive = arms.get("controlled_positive_occlusion")
            benign = arms.get("matched_benign_negative")
            if positive is None or benign is None:
                continue
            scenario_owned_only = {
                str(positive.get("ambient_population_mode")),
                str(benign.get("ambient_population_mode")),
            } == {"scenario_owned_only"}
            check = _compare_ambient_initial_signatures(
                positive["spawn_signature"],
                benign["spawn_signature"],
                config["verification"]["matched_pair_initial_realization_gate"],
            )
            check["group_id"] = group_id
            owned_check = _compare_ambient_initial_signatures(
                positive.get("scenario_owned_nontreatment_signature", []),
                benign.get("scenario_owned_nontreatment_signature", []),
                config["verification"]["matched_pair_initial_realization_gate"],
            )
            if not positive.get("scenario_owned_nontreatment_signature") or not benign.get(
                "scenario_owned_nontreatment_signature"
            ):
                owned_check["pass"] = False
                owned_check.setdefault("failures", []).append(
                    "missing_scenario_owned_nontreatment_signature"
                )
            trajectory_check = _compare_ambient_trajectories(
                Path(
                    str(
                        positive["traffic_sanity"][
                            "ambient_actor_trajectory_csv"
                        ]
                    )
                ),
                Path(
                    str(
                        benign["traffic_sanity"][
                            "ambient_actor_trajectory_csv"
                        ]
                    )
                ),
                config["verification"]["matched_pair_trajectory_gate"],
                allow_declared_both_empty=scenario_owned_only,
            )
            pair_check = {
                "group_id": group_id,
                "initial_realization": check,
                "owned_nontreatment_realization": owned_check,
                "full_trajectory": trajectory_check,
                "pass": bool(
                    check["pass"]
                    and owned_check["pass"]
                    and trajectory_check["pass"]
                ),
            }
            pair_checks.append(pair_check)
            if not bool(check["pass"]):
                raise RuntimeError(
                    f"traffic preflight matched-pair drift for {group_id}: "
                    f"{check['failures']}"
                )
            if not bool(owned_check["pass"]):
                raise RuntimeError(
                    f"traffic preflight scenario-owned pair drift for {group_id}: "
                    f"{owned_check['failures']}"
                )
            if not bool(trajectory_check["pass"]):
                raise RuntimeError(
                    f"traffic preflight full-trajectory drift for {group_id}: "
                    f"{trajectory_check['failures']}"
                )
    except BaseException as exc:
        verdict = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        _append_progress(progress_path, "preflight_failed", error=error)
        raise
    finally:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "verdict": verdict,
            "error": error,
            "trajectory_count_requested": len(rows),
            "trajectory_count_completed": len(results),
            "results": results,
            "matched_pair_checks": pair_checks,
            "scientific_scope": (
                "traffic_motion_only_no_perception_or_controller_authorization"
            ),
        }
        _write_json_create(output_dir / "summary.json", summary)
        sentinel = "COMPLETED.json" if verdict == "PASS" else "FAILED.json"
        _write_json_create(
            output_dir / sentinel,
            {
                "schema": "scenesense.phase2_traffic_preflight_completion.v1",
                "verdict": verdict,
                "error": error,
                "summary": str(output_dir / "summary.json"),
            },
        )
        print(json.dumps({"verdict": verdict, "error": error, "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()

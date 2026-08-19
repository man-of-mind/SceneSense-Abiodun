#!/usr/bin/env python3
"""Validate or detach only the three-trajectory decision-opportunity pilot.

The module is a fail-closed overlay on the existing Phase-2 calibration-audit
runner.  Validation and dry-run modes never contact CARLA.  Detached launch is
permitted only after a human visual-acceptance record is present and bound to
the exact pilot config hash.  No OAI or downstream stage is ever chained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from data_collection.run_phase2_calibration_audit import (
    REPO_ROOT,
    _load_config as _load_audit_config,
    _repo_path,
    _select_trajectory_ids,
    _sha256,
    build_plan,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs/phase2_decision_opportunity_pilot_v1.yaml"
)
PILOT_SCHEMA = "scenesense.phase2_decision_opportunity_pilot.v1"
ACCEPTANCE_SCHEMA = "scenesense.phase2_decision_opportunity_visual_acceptance.v1"
LAUNCH_SCHEMA = "scenesense.phase2_decision_opportunity_detached_launch.v1"
STARTUP_SCHEMA = "scenesense.phase2_decision_opportunity_startup.v1"
EXPECTED_TRAJECTORY_IDS = (
    "sa_curbside_bus_occluded_pedestrian_low_short_r00_pos",
    "sa_curbside_bus_occluded_pedestrian_low_short_r00_ben",
    "sb_town10hd_opt_signalized_demo_region_r00_natural",
)
EXPECTED_ROLES = (
    "controlled_positive_occlusion",
    "matched_benign_negative",
    "naturalistic_operation",
)
EXPECTED_TREATMENT = {
    "pedestrian_start_delay_s": 2.0,
    "pedestrian_speed_mps": 1.3,
    "curbside_retention_start_offset_s": 3.0,
    "world_hz": 10.0,
    "duration_s": 12.0,
    "frames_per_trajectory": 120,
    "helper_speed_mps": 4.5,
    "recipient_speed_mps": 5.0,
}
EXPECTED_AUTHORIZATION = {
    "carla_launch_after_visual_acceptance": True,
    "oai_launch": False,
    "full_corpus_collection": False,
    "downstream_replay": False,
    "controller_evaluation": False,
    "rl_training": False,
}
EXPECTED_STATIC_TRUTH = {
    "enabled": True,
    "semantic_labels": ["Car", "Truck", "Bus"],
    "required_semantic_classes": ["Car"],
    "selection_contract": (
        "town10hd_opt_static_vehicle_like_car_truck_bus_all_enabled_after_"
        "fresh_reload_no_environment_toggles.v1"
    ),
    "enabled_state_basis": (
        "explicit_all_enabled_registry_valid_only_after_fresh_world_reload_"
        "and_before_any_environment_toggle_or_dynamic_actor_spawn"
    ),
}
EXPECTED_OVERLAY_KEYS = {
    "schema_version",
    "stage_id",
    "implementation_status",
    "base_audit_config",
    "base_audit_config_sha256",
    "decision_contract",
    "decision_contract_sha256",
    "output_root",
    "manual_detached_launch_only",
    "authorization",
    "trajectory_ids",
    "treatment",
    "static_environment_truth",
    "visual_acceptance",
}
EXPECTED_ACCEPTANCE_KEYS = {
    "schema",
    "status",
    "pilot_config_sha256",
    "accepted_utc",
    "operator",
    "geometry_review_summary",
    "geometry_review_summary_sha256",
    "checks",
}
TIMESTAMP_PATTERN = re.compile(r"^[0-9]{8}_[0-9]{6}$")
PILOT_TRAJECTORY_COUNT = 3
PILOT_ROLE_COUNT = 2
PILOT_RETAINED_FRAMES_PER_ROLE = 40


def _require_exact_keys(
    mapping: Mapping[str, object], expected: set[str], label: str
) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_yaml(config: Mapping[str, object]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=False)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_yaml_mapping(path: Path, label: str) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return value


def _validate_acceptance(
    pilot: Mapping[str, object],
    pilot_config_path: Path,
    *,
    allow_missing: bool,
) -> dict:
    contract = pilot["visual_acceptance"]
    if not isinstance(contract, Mapping):
        raise ValueError("visual_acceptance must be a mapping")
    _require_exact_keys(
        contract,
        {"required_before_launch", "record", "record_schema", "required_checks"},
        "visual_acceptance",
    )
    if contract["required_before_launch"] is not True:
        raise ValueError("visual acceptance must be mandatory before launch")
    if str(contract["record_schema"]) != ACCEPTANCE_SCHEMA:
        raise ValueError("visual acceptance schema declaration drifted")
    required_checks = tuple(str(value) for value in contract["required_checks"])
    if not required_checks or len(required_checks) != len(set(required_checks)):
        raise ValueError("visual acceptance checks must be non-empty and unique")
    record_path = _repo_path(contract["record"])
    if not record_path.is_file():
        if not allow_missing:
            raise RuntimeError(
                "pilot launch is blocked pending human visual acceptance: "
                f"{record_path}"
            )
        return {
            "status": "blocked_pending_visual_acceptance",
            "record": str(record_path),
            "record_sha256": None,
            "required_checks": list(required_checks),
        }
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid visual acceptance record: {record_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("visual acceptance record must contain an object")
    _require_exact_keys(payload, EXPECTED_ACCEPTANCE_KEYS, "visual acceptance record")
    if payload["schema"] != ACCEPTANCE_SCHEMA or payload["status"] != "accepted":
        raise ValueError("visual acceptance record is not an accepted v1 record")
    expected_config_hash = _sha256(pilot_config_path)
    if str(payload["pilot_config_sha256"]) != expected_config_hash:
        raise ValueError("visual acceptance record is bound to a different pilot config")
    operator = str(payload["operator"]).strip()
    if not operator:
        raise ValueError("visual acceptance operator must be non-empty")
    accepted_utc = str(payload["accepted_utc"])
    try:
        parsed = datetime.fromisoformat(accepted_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("visual acceptance accepted_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("visual acceptance accepted_utc must include a timezone")
    summary_path = _repo_path(payload["geometry_review_summary"])
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"visual acceptance geometry summary is missing: {summary_path}"
        )
    summary_hash = _sha256(summary_path)
    if str(payload["geometry_review_summary_sha256"]) != summary_hash:
        raise ValueError("visual acceptance geometry-summary hash drifted")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid geometry review summary: {summary_path}") from exc
    if not isinstance(summary, Mapping):
        raise ValueError("geometry review summary must contain an object")
    expected_summary_scalars = {
        "schema": "scenesense.phase2_geometry_review.v1",
        "layout": "curbside_opposite",
        "scenario_role": "controlled_positive_occlusion",
        "hazard_actor_present": True,
        "pedestrian_physical_speed_gate_pass": True,
    }
    for name, expected in expected_summary_scalars.items():
        if summary.get(name) != expected:
            raise ValueError(
                f"geometry review summary {name} differs: "
                f"expected={expected!r}, observed={summary.get(name)!r}"
            )
    expected_summary_numbers = {
        "world_hz": 10.0,
        "helper_command_speed_mps": 4.5,
        "recipient_command_speed_mps": 5.0,
        "pedestrian_start_delay_s": 2.0,
        "pedestrian_speed_mps": 1.3,
    }
    for name, expected in expected_summary_numbers.items():
        try:
            observed = float(summary[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"geometry review summary {name} is invalid") from exc
        if observed != expected:
            raise ValueError(
                f"geometry review summary {name} differs: "
                f"expected={expected}, observed={observed}"
            )
    try:
        first_motion_s = float(summary["pedestrian_first_physical_motion_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "geometry review pedestrian_first_physical_motion_s is invalid"
        ) from exc
    if not 2.0 <= first_motion_s <= 2.2:
        raise ValueError(
            "geometry review pedestrian first physical motion is outside "
            f"[2.0, 2.2] s: observed={first_motion_s}"
        )
    collisions = summary.get("collisions")
    if not isinstance(collisions, list) or collisions:
        raise ValueError("geometry review summary must contain zero collisions")
    lane_contract = summary.get("legal_opposing_lane_contract")
    if not isinstance(lane_contract, Mapping) or lane_contract.get("pass") is not True:
        raise ValueError("geometry review legal-opposing lane contract did not pass")
    lane_roles = lane_contract.get("roles")
    if not isinstance(lane_roles, Mapping):
        raise ValueError("geometry review lane roles are missing")
    expected_lanes = {"helper": (17, 1), "recipient": (10, -2)}
    for role, (road_id, lane_id) in expected_lanes.items():
        role_contract = lane_roles.get(role)
        if not isinstance(role_contract, Mapping) or (
            int(role_contract.get("road_id", 0)),
            int(role_contract.get("lane_id", 0)),
        ) != (road_id, lane_id):
            raise ValueError(
                f"geometry review {role} road/lane differs from {road_id}/{lane_id}"
            )
    checks = payload["checks"]
    if not isinstance(checks, Mapping):
        raise ValueError("visual acceptance checks must be a mapping")
    _require_exact_keys(checks, set(required_checks), "visual acceptance checks")
    if any(checks[name] is not True for name in required_checks):
        raise ValueError("every required visual acceptance check must be true")
    return {
        "status": "accepted",
        "record": str(record_path),
        "record_sha256": _sha256(record_path),
        "operator": operator,
        "accepted_utc": accepted_utc,
        "geometry_review_summary": str(summary_path),
        "geometry_review_summary_sha256": summary_hash,
        "required_checks": list(required_checks),
    }


def _load_and_resolve(
    pilot_config_path: Path,
    *,
    require_visual_acceptance: bool,
) -> tuple[dict, dict, dict, object, object]:
    path = Path(pilot_config_path).resolve()
    pilot = _load_yaml_mapping(path, "pilot config")
    _require_exact_keys(pilot, EXPECTED_OVERLAY_KEYS, "pilot config")
    if pilot["schema_version"] != PILOT_SCHEMA:
        raise ValueError("unexpected decision-opportunity pilot schema")
    if pilot["stage_id"] != "phase2_decision_opportunity_pilot_v1":
        raise ValueError("pilot stage ID drifted")
    if pilot["implementation_status"] != "designed_pending_visual_acceptance":
        raise ValueError("pilot implementation status drifted")
    if pilot["manual_detached_launch_only"] is not True:
        raise ValueError("pilot must require a manual detached launch")
    if pilot["authorization"] != EXPECTED_AUTHORIZATION:
        raise ValueError("pilot authorization drifted or broadened")
    if tuple(str(value) for value in pilot["trajectory_ids"]) != EXPECTED_TRAJECTORY_IDS:
        raise ValueError("pilot must contain the exact preregistered trajectories")
    treatment = pilot["treatment"]
    if not isinstance(treatment, Mapping):
        raise ValueError("pilot treatment must be a mapping")
    _require_exact_keys(treatment, set(EXPECTED_TREATMENT), "pilot treatment")
    if {name: float(treatment[name]) for name in treatment} != EXPECTED_TREATMENT:
        raise ValueError("pilot treatment drifted")
    if pilot["static_environment_truth"] != EXPECTED_STATIC_TRUTH:
        raise ValueError("pilot static-environment truth contract drifted")

    base_path = _repo_path(pilot["base_audit_config"])
    if not base_path.is_file():
        raise FileNotFoundError(f"base audit config is missing: {base_path}")
    if _sha256(base_path) != str(pilot["base_audit_config_sha256"]):
        raise ValueError("base audit config hash drifted")
    decision_path = _repo_path(pilot["decision_contract"])
    if not decision_path.is_file():
        raise FileNotFoundError(f"pilot decision contract is missing: {decision_path}")
    if _sha256(decision_path) != str(pilot["decision_contract_sha256"]):
        raise ValueError("pilot decision-contract hash drifted")

    base, source, selected = _load_audit_config(base_path)
    selected = _select_trajectory_ids(selected, EXPECTED_TRAJECTORY_IDS)
    if tuple(selected["scenario_role"].astype(str)) != EXPECTED_ROLES:
        raise ValueError("pilot trajectory roles drifted")
    if tuple(selected["geometry_or_route_id"].astype(str)) != (
        "curbside_bus_occluded_pedestrian",
        "curbside_bus_occluded_pedestrian",
        "town10hd_opt_signalized_demo_region",
    ):
        raise ValueError("pilot trajectory geometry/route selection drifted")

    acceptance = _validate_acceptance(
        pilot,
        path,
        allow_missing=not require_visual_acceptance,
    )
    resolved = copy.deepcopy(base)
    resolved["stage_id"] = str(pilot["stage_id"])
    resolved["output_root"] = str(pilot["output_root"])
    resolved["controlled_motion"]["pedestrian_start_delay_s"] = float(
        treatment["pedestrian_start_delay_s"]
    )
    resolved["capture"]["raw_window_start_offset_s_by_geometry_or_route"][
        "curbside_bus_occluded_pedestrian"
    ] = float(treatment["curbside_retention_start_offset_s"])
    resolved["static_environment_truth"] = copy.deepcopy(
        pilot["static_environment_truth"]
    )
    input_bytes = int(resolved["storage"]["measured_role_input_bytes_per_frame"])
    logits_bytes = int(resolved["storage"]["measured_role_logits_bytes_per_frame"])
    estimated_heavy_bytes = (
        PILOT_TRAJECTORY_COUNT
        * PILOT_ROLE_COUNT
        * PILOT_RETAINED_FRAMES_PER_ROLE
        * (input_bytes + logits_bytes)
    )
    resolved["pilot_provenance"] = {
        "schema": PILOT_SCHEMA,
        "pilot_config": str(path),
        "pilot_config_sha256": _sha256(path),
        "base_audit_config": str(base_path),
        "base_audit_config_sha256": _sha256(base_path),
        "decision_contract": str(decision_path),
        "decision_contract_sha256": _sha256(decision_path),
        "visual_acceptance": acceptance,
        "trajectory_ids": list(EXPECTED_TRAJECTORY_IDS),
        "pilot_subset_estimated_heavy_bytes": estimated_heavy_bytes,
        "pilot_subset_estimate_basis": {
            "trajectory_count": PILOT_TRAJECTORY_COUNT,
            "role_count": PILOT_ROLE_COUNT,
            "retained_frames_per_role": PILOT_RETAINED_FRAMES_PER_ROLE,
            "measured_role_input_bytes_per_frame": input_bytes,
            "measured_role_logits_bytes_per_frame": logits_bytes,
        },
        "base_audit_storage_reservation_is_conservative": True,
        "no_oai_or_downstream_chaining": True,
    }

    assertions = {
        "world_hz": float(resolved["clock"]["world_hz"]),
        "duration_s": float(resolved["clock"]["duration_s"]),
        "frames_per_trajectory": int(resolved["clock"]["frames_per_trajectory"]),
        "pedestrian_speed_mps": float(resolved["controlled_motion"]["pedestrian_speed_mps"]),
        "pedestrian_start_delay_s": float(resolved["controlled_motion"]["pedestrian_start_delay_s"]),
        "helper_speed_mps": float(resolved["staging_roles"]["helper"]["target_speed_mps"]),
        "recipient_speed_mps": float(resolved["staging_roles"]["recipient"]["target_speed_mps"]),
        "curbside_retention_start_offset_s": float(
            resolved["capture"]["raw_window_start_offset_s_by_geometry_or_route"]
            ["curbside_bus_occluded_pedestrian"]
        ),
    }
    if assertions != EXPECTED_TREATMENT:
        raise ValueError(f"resolved pilot treatment differs: {assertions}")
    if resolved["authorization"] != {
        "carla_launch": True,
        "oai_launch": False,
        "remaining_calibration": False,
        "validation_collection": False,
        "test_collection": False,
        "controller_evaluation": False,
        "rl_training": False,
    }:
        raise ValueError("resolved base authorization drifted or broadened")
    if resolved["manual_detached_launch_only"] is not True:
        raise ValueError("resolved base runner must remain manual/detached only")
    return pilot, resolved, acceptance, source, selected


def build_launch_spec(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
    operator_quality: str = "Epic",
    require_visual_acceptance: bool = False,
    include_plan: bool = False,
) -> dict:
    path = Path(config_path).resolve()
    pilot, resolved, acceptance, source, selected = _load_and_resolve(
        path,
        require_visual_acceptance=require_visual_acceptance,
    )
    if str(operator_quality) != "Epic" or str(
        resolved["carla"]["renderer_quality_level"]
    ) != "Epic":
        raise ValueError("pilot requires an operator-declared Epic CARLA server")
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if TIMESTAMP_PATTERN.fullmatch(stamp) is None:
        raise ValueError("launch timestamp must use YYYYMMDD_HHMMSS")
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else _repo_path(pilot["output_root"])
    )
    batch_root = root / f"{stamp}_pilot"
    resolved_path = root / f"{stamp}_pilot.resolved.yaml"
    plan = build_plan(resolved, source, selected, batch_root)
    if int(plan["trajectory_count"]) != 3 or int(plan["group_count"]) != 2:
        raise ValueError("pilot plan is not the exact three-trajectory/two-group stage")
    if [item["trajectory_id"] for item in plan["trajectories"]] != list(
        EXPECTED_TRAJECTORY_IDS
    ):
        raise ValueError("pilot plan trajectory order drifted")
    free_bytes = shutil.disk_usage(root.parent).free
    required = int(resolved["storage"]["preflight_required_free_bytes"])
    if free_bytes < required:
        raise RuntimeError(
            f"pilot storage preflight failed: free={free_bytes}, required={required}"
        )
    resolved_text = _resolved_yaml(resolved)
    run_log = root / f"{stamp}_pilot.run.log"
    manifest = root / f"{stamp}_pilot.launch.json"
    startup_ack = root / f"{stamp}_pilot.STARTUP_ACK.json"
    startup_failed = root / f"{stamp}_pilot.STARTUP_FAILED.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.run_phase2_calibration_audit",
        "--config",
        str(resolved_path),
        "--output-dir",
        str(batch_root),
        "--operator-quality",
        "Epic",
        "--launch",
    ]
    for trajectory_id in EXPECTED_TRAJECTORY_IDS:
        command.extend(("--trajectory-id", trajectory_id))
    spec = {
        "schema": LAUNCH_SCHEMA,
        "status": (
            "validated_ready_not_started"
            if acceptance["status"] == "accepted"
            else "validated_blocked_pending_visual_acceptance"
        ),
        "pilot_config": str(path),
        "pilot_config_sha256": _sha256(path),
        "base_audit_config": str(_repo_path(pilot["base_audit_config"])),
        "base_audit_config_sha256": str(pilot["base_audit_config_sha256"]),
        "decision_contract": str(_repo_path(pilot["decision_contract"])),
        "decision_contract_sha256": str(pilot["decision_contract_sha256"]),
        "visual_acceptance": acceptance,
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": _text_sha256(resolved_text),
        "plan_sha256": _canonical_json_sha256(plan),
        "batch_root": str(batch_root),
        "run_log": str(run_log),
        "launch_manifest": str(manifest),
        "startup_ack": str(startup_ack),
        "startup_failed": str(startup_failed),
        "command": command,
        "operator_quality": "Epic",
        "required_server_launch_flag": "-quality-level=Epic",
        "trajectory_ids": list(EXPECTED_TRAJECTORY_IDS),
        "trajectory_count": 3,
        "group_count": 2,
        "treatment": dict(EXPECTED_TREATMENT),
        "static_environment_truth": copy.deepcopy(EXPECTED_STATIC_TRUTH),
        "estimated_minutes": plan["estimated_minutes"],
        "estimated_heavy_bytes": int(
            resolved["pilot_provenance"]["pilot_subset_estimated_heavy_bytes"]
        ),
        "base_audit_reserved_heavy_bytes": plan["estimated_heavy_bytes"],
        "estimate_basis": {
            "trajectory_count": PILOT_TRAJECTORY_COUNT,
            "role_count": PILOT_ROLE_COUNT,
            "retained_frames_per_role": PILOT_RETAINED_FRAMES_PER_ROLE,
            "measured_role_input_bytes_per_frame": int(
                resolved["storage"]["measured_role_input_bytes_per_frame"]
            ),
            "measured_role_logits_bytes_per_frame": int(
                resolved["storage"]["measured_role_logits_bytes_per_frame"]
            ),
            "minutes_basis": "three_trajectories_times_2p9_minutes_conservative",
        },
        "storage_preflight": {
            "free_bytes": int(free_bytes),
            "required_free_bytes": required,
            "stage_hard_cap_bytes": int(resolved["storage"]["stage_hard_cap_bytes"]),
            "required_free_floor_bytes": int(resolved["storage"]["required_free_floor_bytes"]),
        },
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "progress_log": str(batch_root / "progress.jsonl"),
        "manual_detached_only": True,
        "oai_launched": False,
        "next_stage_chained": False,
    }
    if include_plan:
        spec["plan"] = plan
    return spec


def _write_json_x(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def launch_detached(spec: Mapping[str, object]) -> dict:
    if spec.get("schema") != LAUNCH_SCHEMA:
        raise ValueError("unexpected pilot launch spec schema")
    if spec.get("status") != "validated_ready_not_started":
        raise RuntimeError("pilot launch remains blocked pending visual acceptance")
    current = build_launch_spec(
        Path(str(spec["pilot_config"])),
        output_root=Path(str(spec["batch_root"])).parent,
        timestamp=Path(str(spec["batch_root"])).name.removesuffix("_pilot"),
        operator_quality=str(spec["operator_quality"]),
        require_visual_acceptance=True,
    )
    immutable = {
        "pilot_config_sha256",
        "base_audit_config_sha256",
        "decision_contract_sha256",
        "visual_acceptance",
        "resolved_config_sha256",
        "plan_sha256",
        "batch_root",
        "run_log",
        "launch_manifest",
        "startup_ack",
        "startup_failed",
        "command",
    }
    if any(current[name] != spec[name] for name in immutable):
        raise RuntimeError("pilot launch spec became stale before launch")
    _pilot, resolved, _acceptance, _source, _selected = _load_and_resolve(
        Path(str(spec["pilot_config"])),
        require_visual_acceptance=True,
    )
    root = Path(str(spec["batch_root"])).parent
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(str(spec[name]))
        for name in (
            "batch_root",
            "run_log",
            "launch_manifest",
            "resolved_config",
            "startup_ack",
            "startup_failed",
        )
    ]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to reuse pilot launch artifact: {path}")
    resolved_path = Path(str(spec["resolved_config"]))
    with resolved_path.open("x", encoding="utf-8") as stream:
        stream.write(_resolved_yaml(resolved))
    if _sha256(resolved_path) != str(spec["resolved_config_sha256"]):
        raise RuntimeError("materialized resolved pilot config hash mismatch")
    # Exercise the exact child-side loader before consuming the CARLA run slot.
    _load_audit_config(resolved_path)

    log_path = Path(str(spec["run_log"]))
    log_stream = log_path.open("x", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(value) for value in spec["command"]],
            cwd=REPO_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        raise
    log_stream.close()
    launched = {
        **dict(spec),
        "status": "launch_requested_pending_startup_ack",
        "pid": int(process.pid),
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    launched.pop("plan", None)
    _write_json_x(Path(str(spec["launch_manifest"])), launched)

    batch_root = Path(str(spec["batch_root"]))
    deadline = time.monotonic() + 15.0
    evidence: Optional[Path] = None
    while time.monotonic() < deadline:
        for candidate in (
            batch_root / "progress.jsonl",
            batch_root / "plan.json",
            batch_root / "FAILED.json",
        ):
            if candidate.is_file():
                evidence = candidate
                break
        if evidence is not None:
            break
        return_code = process.poll()
        if return_code is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            failed = {
                "schema": STARTUP_SCHEMA,
                "status": "startup_failed",
                "returncode": int(return_code),
                "run_log_tail": tail,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_x(Path(str(spec["startup_failed"])), failed)
            raise RuntimeError(
                "detached pilot exited before startup acknowledgement: "
                f"returncode={return_code}; log_tail={tail!r}"
            )
        time.sleep(0.05)
    if evidence is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        failed = {
            "schema": STARTUP_SCHEMA,
            "status": "startup_ack_timeout_terminated",
            "run_log_tail": tail,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_x(Path(str(spec["startup_failed"])), failed)
        raise RuntimeError(
            "detached pilot produced no startup artifact within 15 s; child "
            f"was terminated; log_tail={tail!r}"
        )
    acknowledged = {
        "schema": STARTUP_SCHEMA,
        "status": "launched_detached_startup_acknowledged",
        "pid": int(process.pid),
        "startup_evidence": str(evidence),
        "launch_manifest": str(spec["launch_manifest"]),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_x(Path(str(spec["startup_ack"])), acknowledged)
    return {**dict(spec), **acknowledged}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--operator-quality", choices=("Epic",), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-launch", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch-detached", action="store_true")
    args = parser.parse_args()
    spec = build_launch_spec(
        args.config,
        output_root=args.output_root,
        operator_quality=args.operator_quality,
        require_visual_acceptance=args.launch_detached,
        include_plan=args.dry_run,
    )
    result = launch_detached(spec) if args.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

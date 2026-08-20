#!/usr/bin/env python3
"""Review-gate and detach only the exact Phase-2 16-row factor tranche.

Offline modes do not contact CARLA.  Launch is fail-closed on the exact design
hashes, verified runtime adapters, and a human acceptance record covering all
eight positive physical-factor corners.  The detached stage treats the generic
audit runner's completion as raw capture only; scientific completion requires
an atomic factor-smoke result validation pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import yaml

from data_collection.run_phase2_calibration_audit import (
    FACTOR_RETENTION_PRE_ONSET_S,
    REPO_ROOT,
    _load_config as _load_audit_config,
    _sha256,
)
from data_collection.validate_phase2_factor_realization_smoke import (
    build_plan as build_factor_plan,
    load_config as load_factor_config,
    validate_results,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs/phase2_factor_realization_detached_v1.yaml"
)
OVERLAY_SCHEMA = "scenesense.phase2_factor_realization_detached_launch.v1"
REVIEW_PLAN_SCHEMA = "scenesense.phase2_factor_realization_corner_review_plan.v1"
ACCEPTANCE_SCHEMA = "scenesense.phase2_factor_realization_corner_acceptance.v1"
LAUNCH_SCHEMA = "scenesense.phase2_factor_realization_detached_spec.v1"
STAGE_SCHEMA = "scenesense.phase2_factor_realization_stage.v1"
TIMESTAMP_RE = re.compile(r"^[0-9]{8}_[0-9]{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POSITIVE_ROLE = "controlled_positive_occlusion"
EXPECTED_AUTHORIZATION = {
    "carla_launch_after_corner_acceptance": True,
    "oai_launch": False,
    "old_15_trajectory_audit_chain": False,
    "additional_calibration": False,
    "validation_collection": False,
    "test_collection": False,
    "controller_evaluation": False,
    "rl_training": False,
}
EXPECTED_STATIC_ENVIRONMENT_TRUTH = {
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


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _yaml_text(value: Mapping[str, object]) -> str:
    return yaml.safe_dump(dict(value), sort_keys=False)


def _json_text(value: Mapping[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json_x(path: Path, payload: Mapping[str, object]) -> None:
    """Durably publish complete JSON at a create-only destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _unlink_uncommitted_json(path: Path) -> None:
    """Roll back this invocation's non-terminal JSON before failure commit."""

    path.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_progress(path: Path, event: str, **fields: object) -> None:
    payload = {
        "schema": "scenesense.phase2_factor_realization_progress.v1",
        "event": str(event),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _require_exact_keys(
    mapping: Mapping[str, object], expected: set[str], label: str
) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return value


def _load_overlay(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    overlay = _load_yaml(path, "factor detached-launch config")
    expected_keys = {
        "schema_version",
        "stage_id",
        "implementation_status",
        "base_audit_config",
        "base_audit_config_sha256",
        "factor_smoke_config",
        "factor_smoke_config_sha256",
        "factor_smoke_contract",
        "factor_smoke_contract_sha256",
        "expected_factor_plan_sha256",
        "base_runner",
        "base_runner_sha256",
        "factor_validator",
        "factor_validator_sha256",
        "factor_postflight",
        "factor_postflight_sha256",
        "geometry_reviewer",
        "geometry_reviewer_sha256",
        "source_tree_fingerprint",
        "output_root",
        "manual_detached_launch_only",
        "authorization",
        "exact_batch",
        "manual_corner_review",
        "runtime_completion",
    }
    _require_exact_keys(overlay, expected_keys, "factor detached-launch config")
    if overlay["schema_version"] != OVERLAY_SCHEMA:
        raise ValueError("unexpected factor detached-launch schema")
    if overlay["stage_id"] != "phase2_factor_realization_smoke_v1":
        raise ValueError("factor detached-launch stage ID drifted")
    if overlay["manual_detached_launch_only"] is not True:
        raise ValueError("factor tranche must remain manual/detached only")
    if overlay["authorization"] != EXPECTED_AUTHORIZATION:
        raise ValueError("factor detached-launch authorization drifted or broadened")
    exact = overlay["exact_batch"]
    if not isinstance(exact, Mapping):
        raise ValueError("factor detached-launch exact-batch contract is missing")
    _require_exact_keys(
        exact,
        {
            "trajectory_count",
            "group_count",
            "positive_corner_count",
            "atomic_all_or_none",
            "partial_admission",
            "expected_world_minutes",
            "raw_capture_subdirectory",
            "result_bundle_name",
            "raw_validation_name",
            "atomic_validation_name",
        },
        "factor detached-launch exact-batch contract",
    )
    if (
        int(exact.get("trajectory_count", 0)),
        int(exact.get("group_count", 0)),
        int(exact.get("positive_corner_count", 0)),
        exact.get("atomic_all_or_none"),
        exact.get("partial_admission"),
    ) != (16, 8, 8, True, False):
        raise ValueError("factor detached-launch exact-batch contract drifted")
    if (
        exact["raw_capture_subdirectory"],
        exact["result_bundle_name"],
        exact["raw_validation_name"],
        exact["atomic_validation_name"],
    ) != (
        "raw_capture",
        "factor_smoke_results.json",
        "factor_smoke_validation.json",
        "factor_smoke_validation.json",
    ):
        raise ValueError("factor exact-batch artifact names drifted")
    completion = overlay["runtime_completion"]
    if not isinstance(completion, Mapping):
        raise ValueError("factor runtime-completion contract is missing")
    _require_exact_keys(
        completion,
        {
            "generic_audit_completed_is_raw_capture_only",
            "stage_completed_requires_atomic_result_validation_pass",
            "required_atomic_validator_verdict",
            "stage_failure_retains_excluded_fixture",
            "progress_jsonl_required",
            "results_summary_required_on_success_and_failure",
            "completion_sentinel",
            "failure_sentinel",
            "no_downstream_chaining",
        },
        "factor runtime-completion contract",
    )
    if any(
        completion.get(name) is not expected
        for name, expected in {
            "generic_audit_completed_is_raw_capture_only": True,
            "stage_completed_requires_atomic_result_validation_pass": True,
            "stage_failure_retains_excluded_fixture": True,
            "progress_jsonl_required": True,
            "results_summary_required_on_success_and_failure": True,
            "no_downstream_chaining": True,
        }.items()
    ):
        raise ValueError("factor runtime-completion contract drifted")
    if (
        completion["completion_sentinel"],
        completion["failure_sentinel"],
    ) != ("COMPLETED.json", "FAILED.json"):
        raise ValueError("factor completion sentinel names drifted")
    if (
        completion.get("required_atomic_validator_verdict")
        != "PASS_ATOMIC_EXACT_16_ADMITTED"
    ):
        raise ValueError("factor atomic validator verdict contract drifted")
    for field in (
        "base_audit_config",
        "factor_smoke_config",
        "factor_smoke_contract",
        "base_runner",
        "factor_validator",
        "factor_postflight",
        "geometry_reviewer",
    ):
        candidate = _repo_path(overlay[field])
        if not candidate.is_file():
            raise FileNotFoundError(f"factor launch prerequisite missing: {candidate}")
        hash_field = f"{field}_sha256"
        if hash_field in overlay and _sha256(candidate) != str(overlay[hash_field]):
            raise ValueError(f"factor launch {field} hash drifted")
    if _sha256(_repo_path(overlay["geometry_reviewer"])) != str(
        overlay["geometry_reviewer_sha256"]
    ):
        raise ValueError("factor geometry reviewer hash drifted")
    return overlay


def _relevant_source_tree_fingerprint(
    overlay: Mapping[str, object],
) -> dict[str, Any]:
    contract = overlay.get("source_tree_fingerprint")
    if not isinstance(contract, Mapping):
        raise ValueError("factor relevant-source fingerprint contract is missing")
    _require_exact_keys(
        contract,
        {"schema", "roots", "include_suffixes", "excluded_directory_names"},
        "factor relevant-source fingerprint contract",
    )
    if contract["schema"] != "scenesense.phase2_factor_relevant_source_tree.v1":
        raise ValueError("unexpected factor relevant-source fingerprint schema")
    roots = [str(value) for value in contract["roots"]]
    suffixes = {str(value) for value in contract["include_suffixes"]}
    excluded = {str(value) for value in contract["excluded_directory_names"]}
    if roots != ["data_collection", "phase2_map_sharing"]:
        raise ValueError("factor relevant-source roots drifted")
    if suffixes != {".py", ".yaml", ".yml"}:
        raise ValueError("factor relevant-source suffixes drifted")
    if excluded != {"__pycache__", "experiments", "geometry_reviews"}:
        raise ValueError("factor relevant-source exclusions drifted")
    repo = REPO_ROOT.resolve()
    files: list[Path] = []
    for value in roots:
        root = (repo / value).resolve()
        if not root.is_dir() or not root.is_relative_to(repo):
            raise ValueError(f"factor relevant-source root is invalid: {root}")
        files.extend(
            candidate.resolve()
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix in suffixes
            and not (set(candidate.relative_to(root).parts) & excluded)
        )
    unique = sorted(set(files), key=lambda candidate: candidate.relative_to(repo).as_posix())
    if not unique:
        raise ValueError("factor relevant-source fingerprint is empty")
    entries = [
        {
            "path": candidate.relative_to(repo).as_posix(),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for candidate in unique
    ]
    return {
        "schema": contract["schema"],
        "file_count": len(entries),
        "entries": entries,
        "manifest_sha256": _canonical_sha256(entries),
    }


def _load_factor_plan(
    overlay: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    smoke_path = _repo_path(overlay["factor_smoke_config"])
    smoke = load_factor_config(smoke_path)
    plan = build_factor_plan(smoke)
    if plan["plan_sha256"] != str(overlay["expected_factor_plan_sha256"]):
        raise ValueError("factor plan hash drifted")
    if (
        int(plan["trajectory_count"]),
        int(plan["group_count"]),
        int(plan["positive_trajectory_count"]),
        int(plan["benign_trajectory_count"]),
    ) != (16, 8, 8, 8):
        raise ValueError("factor plan is not the exact eight-pair tranche")
    return smoke, plan


def _positive_rows(plan: Mapping[str, object]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in plan["rows"]
        if row["scenario_role"] == POSITIVE_ROLE
    ]
    if len(rows) != 8:
        raise ValueError("corner review must contain exactly eight positives")
    return rows


def _corner_retention_contract(
    overlay: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, Any]:
    """Resolve the exact 10 Hz sample window used by the later raw capture."""

    base = _load_yaml(_repo_path(overlay["base_audit_config"]), "base audit config")
    clock = base["clock"]
    capture = base["capture"]
    review = overlay["manual_corner_review"]["retention_window_preflight"]
    dt = float(clock["fixed_delta_seconds"])
    duration = float(clock["duration_s"])
    window = float(capture["raw_window_duration_s"])
    retained = int(capture["retained_frames_per_role"])
    configured_pre = float(review["configured_pre_authored_onset_s"])
    postflight_minimum = float(
        review["postflight_minimum_post_realized_onset_s"]
    )
    prelaunch_minimum = float(
        review["prelaunch_minimum_post_realized_onset_s"]
    )
    margin = float(review["prelaunch_margin_over_postflight_s"])
    if not (
        retained == int(review["retained_frame_count"]) == 40
        and _float_equal(dt, review["sample_period_s"])
        and _float_equal(dt, 0.1)
        and _float_equal(window, retained * dt)
        and duration >= window
        and _float_equal(configured_pre, FACTOR_RETENTION_PRE_ONSET_S)
        and _float_equal(prelaunch_minimum - postflight_minimum, margin)
        and _float_equal(margin, dt)
    ):
        raise ValueError("corner-retention preflight differs from exact runtime contract")
    onset = float(row["requested_factor_contract"]["requested_hazard_onset_s"])
    start_offset = max(0.0, min(duration - window, onset - configured_pre))
    # Capture time zero is the pre-frame barrier; the first sensor sample is
    # one tick later.  Retention begins at the first sample on/after offset.
    first_sample_index = max(1, math.ceil((start_offset - 1e-9) / dt))
    first_sample_s = first_sample_index * dt
    last_sample_s = first_sample_s + (retained - 1) * dt
    return {
        "schema": "scenesense.phase2_factor_corner_retention_preflight.v1",
        "configured_start_offset_s": start_offset,
        "expected_first_retained_sample_s": first_sample_s,
        "expected_last_retained_sample_s": last_sample_s,
        "retained_frame_count": retained,
        "sample_period_s": dt,
        "prelaunch_minimum_post_realized_onset_s": prelaunch_minimum,
        "postflight_minimum_post_realized_onset_s": postflight_minimum,
        "prelaunch_margin_over_postflight_s": margin,
        "first_sample_basis": review["first_sample_basis"],
        "authored_onset_policy_visibility": review[
            "authored_onset_policy_visibility"
        ],
    }


def build_corner_review_plan(
    config_path: Path = DEFAULT_CONFIG,
    *,
    review_root: Optional[Path] = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    overlay = _load_overlay(path)
    _smoke, plan = _load_factor_plan(overlay)
    review = overlay["manual_corner_review"]
    root = (
        Path(review_root).resolve()
        if review_root is not None
        else Path(str(review["output_root"])).resolve()
    )
    layout_by_geometry = {
        str(key): str(value)
        for key, value in review["layout_by_geometry"].items()
    }
    commands = []
    for row in _positive_rows(plan):
        requested = row["requested_factor_contract"]
        geometry = str(row["geometry_or_route_id"])
        command = [
            sys.executable,
            "-m",
            str(review["reviewer_module"]),
            "--layout",
            layout_by_geometry[geometry],
            "--scenario-role",
            POSITIVE_ROLE,
            "--duration-s",
            str(float(review["duration_s"])),
            "--helper-speed-mps",
            str(float(requested["requested_helper_speed_mps"])),
            "--recipient-speed-mps",
            str(float(requested["requested_recipient_speed_mps"])),
            "--output-root",
            str(root),
            "--factor-smoke-config",
            str(_repo_path(overlay["factor_smoke_config"])),
            "--factor-trajectory-id",
            str(row["trajectory_id"]),
        ]
        if str(row["hazard_class"]) == "pedestrian":
            command.extend(
                (
                    "--pedestrian-speed-mps",
                    str(float(requested["requested_hazard_actor_speed_mps"])),
                    "--pedestrian-start-delay-s",
                    str(float(requested["requested_hazard_onset_s"])),
                )
            )
        elif str(row["hazard_class"]) == "vehicle":
            command.extend(
                (
                    "--target-vehicle-speed-mps",
                    str(float(requested["requested_hazard_actor_speed_mps"])),
                    "--target-vehicle-start-delay-s",
                    str(float(requested["requested_hazard_onset_s"])),
                )
            )
        else:
            raise ValueError(f"unsupported corner hazard class: {row['hazard_class']}")
        commands.append(
            {
                "trajectory_id": row["trajectory_id"],
                "trajectory_row_sha256": row["trajectory_row_sha256"],
                "geometry_or_route_id": geometry,
                "hazard_class": row["hazard_class"],
                "closing_speed_band": row["closing_speed_band"],
                "time_to_hazard_band": row["time_to_hazard_band"],
                "requested_factor_contract": requested,
                "maximum_surface_clearance_m": float(
                    _smoke["factor_contract"]
                    ["positive_hazard_surface_clearance_max_m_by_class"]
                    [row["hazard_class"]]
                ),
                "retention_window_preflight": _corner_retention_contract(
                    overlay, row
                ),
                "command": command,
                "command_shell": shlex.join(command),
            }
        )
    result = {
        "schema": REVIEW_PLAN_SCHEMA,
        "stage_id": overlay["stage_id"],
        "launch_config": str(path),
        "launch_config_sha256": _sha256(path),
        "factor_plan_sha256": plan["plan_sha256"],
        "geometry_reviewer_sha256": overlay["geometry_reviewer_sha256"],
        "review_root": str(root),
        "positive_corner_count": len(commands),
        "required_human_checks": list(review["required_human_checks"]),
        "commands": commands,
        "carla_or_oai_started_by_this_plan": False,
        "next_action": "operator_runs_each_command_and_records_hash_bound_acceptance",
    }
    result["review_plan_sha256"] = _canonical_sha256(result)
    return result


def _float_equal(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _artifact_manifest_sha256(directory: Path) -> str:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"corner review artifact directory is empty: {directory}")
    manifest = [
        {
            "path": str(path.relative_to(directory)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    return _canonical_sha256(manifest)


def _match_and_validate_summary(
    summary_path: Path,
    row: Mapping[str, object],
    review_plan: Mapping[str, object],
) -> bool:
    summary = _load_json(summary_path, "corner geometry summary")
    requested = row["requested_factor_contract"]
    expected_layout = {
        "curbside_bus_occluded_pedestrian": "curbside_opposite",
        "occluded_cross_traffic_vehicle": "cross_traffic_vehicle",
    }[str(row["geometry_or_route_id"])]
    if (
        summary.get("schema") != "scenesense.phase2_geometry_review.v1"
        or summary.get("layout") != expected_layout
        or summary.get("scenario_role") != POSITIVE_ROLE
        or summary.get("hazard_actor_present") is not True
        or not _float_equal(summary.get("world_hz"), 10.0)
        or not _float_equal(
            summary.get("helper_command_speed_mps"),
            requested["requested_helper_speed_mps"],
        )
        or not _float_equal(
            summary.get("recipient_command_speed_mps"),
            requested["requested_recipient_speed_mps"],
        )
    ):
        return False
    collisions = summary.get("collisions")
    if not isinstance(collisions, list) or collisions:
        raise ValueError(f"corner review contains collisions: {summary_path}")
    lane_contract = summary.get("lane_contract")
    if not isinstance(lane_contract, Mapping) or lane_contract.get("pass") is not True:
        raise ValueError(f"corner review lane contract failed: {summary_path}")
    if str(row["hazard_class"]) == "pedestrian":
        if not (
            _float_equal(
                summary.get("pedestrian_speed_mps"),
                requested["requested_hazard_actor_speed_mps"],
            )
            and _float_equal(
                summary.get("pedestrian_start_delay_s"),
                requested["requested_hazard_onset_s"],
            )
            and summary.get("pedestrian_physical_speed_gate_pass") is True
        ):
            return False
        first_motion = float(summary.get("pedestrian_first_physical_motion_s"))
        onset = float(requested["requested_hazard_onset_s"])
        if not onset <= first_motion <= onset + 0.3:
            raise ValueError(
                f"corner pedestrian onset drifted: {summary_path}: {first_motion}"
            )
    else:
        if not (
            _float_equal(
                summary.get("target_vehicle_command_speed_mps"),
                requested["requested_hazard_actor_speed_mps"],
            )
            and _float_equal(
                summary.get("target_vehicle_start_delay_s"),
                requested["requested_hazard_onset_s"],
            )
        ):
            return False
        gate = summary.get("vehicle_hazard_review_gate")
        if not isinstance(gate, Mapping) or gate.get("pass") is not True:
            raise ValueError(f"corner vehicle hazard gate failed: {summary_path}")
    factor_runtime = summary.get("factor_runtime_contract")
    realized = summary.get("realized_factors")
    factor_gate = summary.get("factor_realization_gate")
    if not isinstance(factor_runtime, Mapping) or not isinstance(realized, Mapping):
        raise ValueError(f"corner review lacks exact factor-runtime output: {summary_path}")
    if not isinstance(factor_gate, Mapping) or factor_gate.get("pass") is not True:
        raise ValueError(f"corner review exact physical-factor gate failed: {summary_path}")
    if (
        factor_runtime.get("trajectory_id") != row["trajectory_id"]
        or factor_runtime.get("trajectory_row_sha256") != row["trajectory_row_sha256"]
        or factor_runtime.get("requested_factors") != requested
    ):
        raise ValueError(f"corner factor-runtime contract differs from its row: {summary_path}")
    closing = float(realized["pre_intervention_radial_closing_speed_mps"])
    horizon = float(realized["pre_intervention_hazard_proximity_horizon_s"])
    clearance = float(realized["pre_intervention_minimum_surface_clearance_m"])
    if not (
        float(requested["requested_closing_speed_band_min_mps"])
        <= closing
        <= float(requested["requested_closing_speed_band_max_mps"])
        and float(requested["requested_proximity_horizon_band_min_s"])
        <= horizon
        <= float(requested["requested_proximity_horizon_band_max_s"])
        and 0.0 <= clearance <= float(row["maximum_surface_clearance_m"])
    ):
        raise ValueError(f"corner physical factors are outside their exact cell: {summary_path}")
    retention = row["retention_window_preflight"]
    realized_onset = float(realized["realized_hazard_onset_s"])
    expected_last_sample = float(retention["expected_last_retained_sample_s"])
    post_realized_onset = expected_last_sample - realized_onset
    if post_realized_onset + 1e-9 < float(
        retention["prelaunch_minimum_post_realized_onset_s"]
    ):
        raise ValueError(
            "corner realized onset leaves insufficient exact retained-sample "
            f"post-window margin: {summary_path}: {post_realized_onset}"
        )
    for basis in (
        "geometry_measurement_basis",
        "closing_speed_measurement_basis",
        "proximity_horizon_measurement_basis",
    ):
        if realized.get(basis) != requested[basis]:
            raise ValueError(f"corner factor basis {basis} drifted: {summary_path}")
    png_files = list(summary_path.parent.glob("*.png"))
    if not png_files:
        raise ValueError(f"corner review has no saved visual evidence: {summary_path}")
    return True


def record_corner_acceptance(
    config_path: Path,
    *,
    review_root: Path,
    operator: str,
    confirmed_checks: Sequence[str],
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    overlay = _load_overlay(path)
    review_plan = build_corner_review_plan(path, review_root=review_root)
    required = list(review_plan["required_human_checks"])
    if set(confirmed_checks) != set(required) or len(confirmed_checks) != len(required):
        raise ValueError("every required human corner-review check must be confirmed exactly once")
    operator = str(operator).strip()
    if not operator:
        raise ValueError("corner-review operator must be non-empty")
    root = Path(review_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else _repo_path(overlay["manual_corner_review"]["acceptance_record"])
    )
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite factor corner acceptance: {destination}"
        )
    archive_root = _repo_path(overlay["manual_corner_review"]["archive_root"])
    if archive_root.exists():
        raise FileExistsError(
            f"refusing to overwrite factor corner archive: {archive_root}"
        )
    summaries = sorted(root.rglob("geometry_review_summary.json"))
    if len(summaries) != 8:
        raise ValueError(
            f"corner-review root must contain exactly eight summaries, found {len(summaries)}"
        )
    unmatched = list(summaries)
    source_artifacts = []
    for row in review_plan["commands"]:
        matches = [
            candidate
            for candidate in unmatched
            if _match_and_validate_summary(candidate, row, review_plan)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"corner {row['trajectory_id']} matched {len(matches)} summaries; expected one"
            )
        summary_path = matches[0]
        unmatched.remove(summary_path)
        source_artifacts.append(
            {
                "trajectory_id": row["trajectory_id"],
                "trajectory_row_sha256": row["trajectory_row_sha256"],
                "source_artifact_directory": str(summary_path.parent),
                "artifact_manifest_sha256": _artifact_manifest_sha256(
                    summary_path.parent
                ),
                "geometry_review_summary": str(summary_path),
                "geometry_review_summary_sha256": _sha256(summary_path),
            }
        )
    if unmatched:
        raise ValueError(f"unassigned corner-review summaries remain: {unmatched}")
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = archive_root.with_name(
        archive_root.name + ".staging_" + review_plan["review_plan_sha256"][:12]
    )
    if staging_root.exists():
        raise FileExistsError(
            f"refusing to reuse factor corner archive staging path: {staging_root}"
        )
    staging_root.mkdir()
    try:
        artifacts = []
        for source in source_artifacts:
            trajectory_id = str(source["trajectory_id"])
            source_dir = Path(str(source["source_artifact_directory"]))
            staged_dir = staging_root / trajectory_id
            shutil.copytree(source_dir, staged_dir)
            staged_summary = staged_dir / "geometry_review_summary.json"
            if _sha256(staged_summary) != source["geometry_review_summary_sha256"]:
                raise RuntimeError("archived factor corner summary hash changed during copy")
            if _artifact_manifest_sha256(staged_dir) != source[
                "artifact_manifest_sha256"
            ]:
                raise RuntimeError("archived factor corner manifest changed during copy")
            artifacts.append(
                {
                    "trajectory_id": trajectory_id,
                    "trajectory_row_sha256": source["trajectory_row_sha256"],
                    "artifact_directory": str(archive_root / trajectory_id),
                    "artifact_manifest_sha256": source["artifact_manifest_sha256"],
                    "geometry_review_summary": str(
                        archive_root / trajectory_id / "geometry_review_summary.json"
                    ),
                    "geometry_review_summary_sha256": source[
                        "geometry_review_summary_sha256"
                    ],
                }
            )
        staging_root.rename(archive_root)
    except BaseException:
        # Preserve an incomplete staging tree for forensic inspection. The
        # create-only contract refuses to reuse it, so it cannot be mistaken
        # for accepted evidence.
        raise
    acceptance = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "source_review_root": str(root),
        "launch_config": str(path),
        "launch_config_sha256": _sha256(path),
        "factor_plan_sha256": review_plan["factor_plan_sha256"],
        "review_plan_sha256": review_plan["review_plan_sha256"],
        "geometry_reviewer_sha256": review_plan["geometry_reviewer_sha256"],
        "accepted_utc": datetime.now(timezone.utc).isoformat(),
        "operator": operator,
        "checks": {name: True for name in required},
        "review_artifacts": artifacts,
    }
    try:
        _write_json_x(destination, acceptance)
    except BaseException:
        # The durable archive is intentionally not removed on an acceptance
        # write failure: deleting reviewed evidence would be destructive. The
        # operator must inspect and explicitly resolve the orphaned archive.
        raise
    return {**acceptance, "acceptance_record": str(destination)}


def _validate_corner_acceptance(
    config_path: Path,
    overlay: Mapping[str, object],
    *,
    allow_missing: bool,
) -> dict[str, Any]:
    record_path = _repo_path(overlay["manual_corner_review"]["acceptance_record"])
    if not record_path.is_file():
        if not allow_missing:
            raise RuntimeError(
                f"factor launch blocked pending manual corner acceptance: {record_path}"
            )
        return {
            "status": "blocked_pending_manual_corner_acceptance",
            "record": str(record_path),
            "record_sha256": None,
        }
    acceptance = _load_json(record_path, "factor corner acceptance")
    expected_keys = {
        "schema",
        "status",
        "source_review_root",
        "launch_config",
        "launch_config_sha256",
        "factor_plan_sha256",
        "review_plan_sha256",
        "geometry_reviewer_sha256",
        "accepted_utc",
        "operator",
        "checks",
        "review_artifacts",
    }
    _require_exact_keys(acceptance, expected_keys, "factor corner acceptance")
    if acceptance["schema"] != ACCEPTANCE_SCHEMA or acceptance["status"] != "accepted":
        raise ValueError("factor corner acceptance status/schema is invalid")
    source_review_root = Path(str(acceptance["source_review_root"])).resolve()
    if str(source_review_root) != acceptance["source_review_root"]:
        raise ValueError("factor corner acceptance source review root is not canonical")
    review_plan = build_corner_review_plan(
        config_path, review_root=source_review_root
    )
    expected_scalars = {
        "launch_config": str(config_path.resolve()),
        "launch_config_sha256": _sha256(config_path.resolve()),
        "factor_plan_sha256": review_plan["factor_plan_sha256"],
        "review_plan_sha256": review_plan["review_plan_sha256"],
        "geometry_reviewer_sha256": overlay["geometry_reviewer_sha256"],
    }
    for name, expected in expected_scalars.items():
        if acceptance[name] != expected:
            raise ValueError(f"factor corner acceptance {name} drifted")
    if not str(acceptance["operator"]).strip():
        raise ValueError("factor corner acceptance operator is empty")
    try:
        accepted_at = datetime.fromisoformat(
            str(acceptance["accepted_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("factor corner acceptance timestamp is invalid") from exc
    if accepted_at.tzinfo is None:
        raise ValueError("factor corner acceptance timestamp lacks a timezone")
    checks = acceptance["checks"]
    required = set(review_plan["required_human_checks"])
    if not isinstance(checks, Mapping) or set(checks) != required or any(
        checks[name] is not True for name in required
    ):
        raise ValueError("factor corner acceptance checks are incomplete")
    artifacts = acceptance["review_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        raise ValueError("factor corner acceptance must bind eight artifacts")
    expected_rows = {
        row["trajectory_id"]: row for row in review_plan["commands"]
    }
    if {item.get("trajectory_id") for item in artifacts} != set(expected_rows):
        raise ValueError("factor corner acceptance trajectory IDs drifted")
    for item in artifacts:
        row = expected_rows[item["trajectory_id"]]
        if item.get("trajectory_row_sha256") != row["trajectory_row_sha256"]:
            raise ValueError("factor corner acceptance row hash drifted")
        summary_path = Path(str(item["geometry_review_summary"])).resolve()
        directory = Path(str(item["artifact_directory"])).resolve()
        if summary_path.parent != directory or not summary_path.is_file():
            raise FileNotFoundError("accepted factor corner summary is missing")
        if _sha256(summary_path) != item.get("geometry_review_summary_sha256"):
            raise ValueError("accepted factor corner summary hash drifted")
        if _artifact_manifest_sha256(directory) != item.get(
            "artifact_manifest_sha256"
        ):
            raise ValueError("accepted factor corner artifact manifest drifted")
        if not _match_and_validate_summary(summary_path, row, review_plan):
            raise ValueError("accepted factor corner summary no longer matches its row")
    return {
        "status": "accepted",
        "record": str(record_path),
        "record_sha256": _sha256(record_path),
        "operator": acceptance["operator"],
        "accepted_utc": acceptance["accepted_utc"],
        "artifact_count": len(artifacts),
    }


def _selected_v2_rows(
    smoke: Mapping[str, object], plan: Mapping[str, object]
) -> pd.DataFrame:
    manifest_path = _repo_path(smoke["source_design"]["manifest"])
    manifest = pd.read_csv(manifest_path)
    order = {
        row["trajectory_id"]: index for index, row in enumerate(plan["rows"])
    }
    selected = manifest[
        manifest["trajectory_id"].astype(str).isin(order)
    ].copy()
    selected["_factor_order"] = selected["trajectory_id"].astype(str).map(order)
    selected = selected.sort_values("_factor_order").drop(columns="_factor_order")
    if list(selected["trajectory_id"].astype(str)) != list(order):
        raise ValueError("resolved audit selection differs from exact factor plan")
    return selected


def _resolve_audit_config(
    config_path: Path,
    overlay: Mapping[str, object],
    smoke: Mapping[str, object],
    plan: Mapping[str, object],
    factor_plan_path: Path,
    factor_plan_file_sha256: str,
    source_tree_fingerprint: Mapping[str, object],
) -> tuple[dict[str, Any], Mapping[str, object], pd.DataFrame]:
    base_path = _repo_path(overlay["base_audit_config"])
    base, source, _base_selected = _load_audit_config(base_path)
    resolved = copy.deepcopy(base)
    resolved["stage_id"] = str(overlay["stage_id"])
    resolved["output_root"] = str(overlay["output_root"])
    source_design = smoke["source_design"]
    resolved["design"] = {
        "config": str(source_design["design_config"]),
        "config_sha256": str(source_design["design_config_sha256"]),
        "trajectory_manifest": str(source_design["manifest"]),
        "trajectory_manifest_sha256": str(source_design["manifest_sha256"]),
        "selector": {
            "split": "calibration",
            "expected_group_count": 8,
            "expected_trajectory_count": 16,
            "exact_trajectory_ids": [row["trajectory_id"] for row in plan["rows"]],
        },
    }
    # The endpoint postflight adjudicates detections against static occluder
    # truth as well as dynamic actors.  Capture this registry before any
    # dynamic actor is spawned; omitting it would make a completed exact-16
    # raw capture scientifically unusable and fail only after the long run.
    resolved["static_environment_truth"] = copy.deepcopy(
        EXPECTED_STATIC_ENVIRONMENT_TRUTH
    )
    input_bytes = int(resolved["storage"]["measured_role_input_bytes_per_frame"])
    logits_bytes = int(resolved["storage"]["measured_role_logits_bytes_per_frame"])
    retained = int(resolved["capture"]["retained_frames_per_role"])
    selected = _selected_v2_rows(smoke, plan)
    estimated = sum(
        2
        * retained
        * (
            input_bytes
            + (
                logits_bytes
                if str(tier) == "inputs_plus_logits_window"
                else 0
            )
        )
        for tier in selected["raw_retention_tier"]
    )
    resolved["storage"]["estimated_heavy_bytes"] = estimated
    resolved["factor_realization_runtime"] = {
        "schema": "scenesense.phase2_factor_realization_runtime_config.v1",
        "enabled": True,
        "factor_smoke_config": str(_repo_path(overlay["factor_smoke_config"])),
        "factor_smoke_config_sha256": str(overlay["factor_smoke_config_sha256"]),
        "factor_smoke_plan": str(factor_plan_path),
        "factor_smoke_plan_sha256": str(factor_plan_file_sha256),
        "exact_trajectory_count": 16,
        "atomic_batch": True,
    }
    resolved["factor_launch_provenance"] = {
        "schema": OVERLAY_SCHEMA,
        "launch_config": str(config_path),
        "launch_config_sha256": _sha256(config_path),
        "factor_plan_sha256": plan["plan_sha256"],
        "base_runner_sha256": str(overlay["base_runner_sha256"]),
        "factor_validator_sha256": str(overlay["factor_validator_sha256"]),
        "factor_postflight_sha256": str(overlay["factor_postflight_sha256"]),
        "relevant_source_tree_fingerprint": dict(source_tree_fingerprint),
        "atomic_all_or_none": True,
        "no_downstream_chaining": True,
    }
    return resolved, source, selected


def build_launch_spec(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_root: Optional[Path] = None,
    timestamp: Optional[str] = None,
    operator_quality: str = "Epic",
    require_corner_acceptance: bool = False,
    include_plan: bool = False,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    overlay = _load_overlay(path)
    smoke, factor_plan = _load_factor_plan(overlay)
    acceptance = _validate_corner_acceptance(
        path,
        overlay,
        allow_missing=not require_corner_acceptance,
    )
    if operator_quality != "Epic":
        raise ValueError("factor tranche requires operator-declared Epic quality")
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if TIMESTAMP_RE.fullmatch(stamp) is None:
        raise ValueError("factor launch timestamp must use YYYYMMDD_HHMMSS")
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else _repo_path(overlay["output_root"])
    )
    batch_root = root / f"{stamp}_factor16"
    resolved_path = root / f"{stamp}_factor16.resolved.yaml"
    factor_plan_path = root / f"{stamp}_factor16.plan.json"
    result_bundle_path = (
        batch_root
        / str(overlay["exact_batch"]["raw_capture_subdirectory"])
        / str(overlay["exact_batch"]["result_bundle_name"])
    )
    factor_plan_file_sha256 = _text_sha256(_json_text(factor_plan))
    source_tree_fingerprint = _relevant_source_tree_fingerprint(overlay)
    resolved, _source, selected = _resolve_audit_config(
        path,
        overlay,
        smoke,
        factor_plan,
        factor_plan_path,
        factor_plan_file_sha256,
        source_tree_fingerprint,
    )
    raw_capture_root = batch_root / str(
        overlay["exact_batch"]["raw_capture_subdirectory"]
    )
    audit_plan = {
        "schema": "scenesense.phase2_factor_raw_capture_plan.v1",
        "stage_id": overlay["stage_id"],
        "trajectory_count": len(selected),
        "group_count": int(selected["group_id"].nunique()),
        "trajectory_ids": list(selected["trajectory_id"].astype(str)),
        "raw_capture_root": str(raw_capture_root),
        "raw_capture_progress_log": str(raw_capture_root / "progress.jsonl"),
        "raw_capture_run_log": str(batch_root / "raw_capture.run.log"),
        "generic_audit_completion_is_scientific_pass": False,
        "oai_launched": False,
        "next_stage_chained": False,
    }
    if (
        int(audit_plan["trajectory_count"]),
        int(audit_plan["group_count"]),
        audit_plan["trajectory_ids"],
    ) != (
        16,
        8,
        [row["trajectory_id"] for row in factor_plan["rows"]],
    ):
        raise ValueError("resolved raw-capture plan is not the exact factor tranche")
    runtime_ready = bool(factor_plan["runtime_ready"])
    status = (
        "validated_blocked_runtime_adapters"
        if not runtime_ready
        else (
            "validated_ready_not_started"
            if acceptance["status"] == "accepted"
            else "validated_blocked_pending_manual_corner_acceptance"
        )
    )
    resolved_text = _yaml_text(resolved)
    run_log = root / f"{stamp}_factor16.run.log"
    launch_manifest = root / f"{stamp}_factor16.launch.json"
    startup_ack = root / f"{stamp}_factor16.STARTUP_ACK.json"
    startup_failed = root / f"{stamp}_factor16.STARTUP_FAILED.json"
    command = [
        sys.executable,
        "-m",
        "data_collection.launch_phase2_factor_realization_smoke",
        "--config",
        str(path),
        "--resolved-config",
        str(resolved_path),
        "--factor-plan",
        str(factor_plan_path),
        "--batch-root",
        str(batch_root),
        "--run-stage",
    ]
    free_bytes = shutil.disk_usage(root.parent).free
    required_bytes = int(resolved["storage"]["preflight_required_free_bytes"])
    if runtime_ready and free_bytes < required_bytes:
        raise RuntimeError(
            f"factor storage preflight failed: free={free_bytes}, required={required_bytes}"
        )
    spec = {
        "schema": LAUNCH_SCHEMA,
        "status": status,
        "launch_config": str(path),
        "launch_config_sha256": _sha256(path),
        "factor_smoke_config_sha256": overlay["factor_smoke_config_sha256"],
        "factor_smoke_contract_sha256": overlay["factor_smoke_contract_sha256"],
        "base_runner_sha256": overlay["base_runner_sha256"],
        "factor_validator_sha256": overlay["factor_validator_sha256"],
        "factor_postflight_sha256": overlay["factor_postflight_sha256"],
        "relevant_source_tree_fingerprint": source_tree_fingerprint,
        "factor_plan_sha256": factor_plan["plan_sha256"],
        "runtime_ready": runtime_ready,
        "runtime_blockers": list(factor_plan["runtime_blockers"]),
        "manual_corner_acceptance": acceptance,
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": _text_sha256(resolved_text),
        "factor_plan": str(factor_plan_path),
        "factor_plan_file_sha256": factor_plan_file_sha256,
        "batch_root": str(batch_root),
        "raw_capture_root": str(raw_capture_root),
        "result_bundle": str(result_bundle_path),
        "run_log": str(run_log),
        "launch_manifest": str(launch_manifest),
        "startup_ack": str(startup_ack),
        "startup_failed": str(startup_failed),
        "command": command,
        "operator_quality": "Epic",
        "required_server_launch_flag": "-quality-level=Epic",
        "trajectory_count": 16,
        "group_count": 8,
        "atomic_all_or_none": True,
        "partial_admission_authorized": False,
        "estimated_minutes": float(overlay["exact_batch"]["expected_world_minutes"]),
        "estimated_heavy_bytes": int(resolved["storage"]["estimated_heavy_bytes"]),
        "storage_preflight": {
            "free_bytes": int(free_bytes),
            "required_free_bytes": required_bytes,
            "stage_hard_cap_bytes": int(resolved["storage"]["stage_hard_cap_bytes"]),
            "required_free_floor_bytes": int(
                resolved["storage"]["required_free_floor_bytes"]
            ),
        },
        "progress_log": str(batch_root / "progress.jsonl"),
        "completion_sentinel": str(batch_root / "COMPLETED.json"),
        "failure_sentinel": str(batch_root / "FAILED.json"),
        "results_summary": str(batch_root / "RESULTS_SUMMARY.json"),
        "atomic_validation": str(
            batch_root / str(overlay["exact_batch"]["atomic_validation_name"])
        ),
        "oai_launched": False,
        "old_audit_chained": False,
        "next_stage_chained": False,
    }
    if include_plan:
        spec["factor_plan_payload"] = factor_plan
        spec["raw_capture_plan"] = audit_plan
        spec["resolved_config_payload"] = resolved
    return spec


def launch_detached(spec: Mapping[str, object]) -> dict[str, Any]:
    if spec.get("schema") != LAUNCH_SCHEMA:
        raise ValueError("unexpected factor launch spec schema")
    if spec.get("status") != "validated_ready_not_started":
        raise RuntimeError(f"factor launch remains blocked: {spec.get('status')}")
    current = build_launch_spec(
        Path(str(spec["launch_config"])),
        output_root=Path(str(spec["batch_root"])).parent,
        timestamp=Path(str(spec["batch_root"])).name.removesuffix("_factor16"),
        operator_quality=str(spec["operator_quality"]),
        require_corner_acceptance=True,
        include_plan=True,
    )
    immutable = {
        "launch_config_sha256",
        "factor_smoke_config_sha256",
        "factor_smoke_contract_sha256",
        "base_runner_sha256",
        "factor_validator_sha256",
        "factor_postflight_sha256",
        "relevant_source_tree_fingerprint",
        "factor_plan_sha256",
        "factor_plan_file_sha256",
        "runtime_ready",
        "runtime_blockers",
        "manual_corner_acceptance",
        "resolved_config_sha256",
        "batch_root",
        "raw_capture_root",
        "result_bundle",
        "run_log",
        "launch_manifest",
        "startup_ack",
        "startup_failed",
        "command",
    }
    if any(current[name] != spec[name] for name in immutable):
        raise RuntimeError("factor launch spec became stale before launch")
    root = Path(str(spec["batch_root"])).parent
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(str(spec[name]))
        for name in (
            "batch_root",
            "run_log",
            "launch_manifest",
            "resolved_config",
            "factor_plan",
            "startup_ack",
            "startup_failed",
        )
    ]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to reuse factor launch artifact: {path}")
    resolved_path = Path(str(spec["resolved_config"]))
    with resolved_path.open("x", encoding="utf-8") as stream:
        stream.write(_yaml_text(current["resolved_config_payload"]))
    if _sha256(resolved_path) != spec["resolved_config_sha256"]:
        raise RuntimeError("materialized factor resolved-config hash mismatch")
    factor_plan_path = Path(str(spec["factor_plan"]))
    _write_json_x(factor_plan_path, current["factor_plan_payload"])
    if _sha256(factor_plan_path) != spec["factor_plan_file_sha256"]:
        raise RuntimeError("materialized factor plan file hash mismatch")
    if current["factor_plan_payload"]["plan_sha256"] != spec["factor_plan_sha256"]:
        raise RuntimeError("materialized factor plan hash mismatch")
    # The exact child loader is exercised only after all immutable artifacts
    # exist. Failure here consumes no CARLA run slot.
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
        **{key: value for key, value in spec.items() if not key.endswith("_payload")},
        "status": "launch_requested_pending_startup_ack",
        "pid": int(process.pid),
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_x(Path(str(spec["launch_manifest"])), launched)
    batch_root = Path(str(spec["batch_root"]))
    stage_failed = batch_root / "FAILED.json"
    stage_progress = batch_root / "progress.jsonl"
    deadline = time.monotonic() + 15.0
    evidence: Optional[Path] = None
    while time.monotonic() < deadline:
        # A fast fail can write progress and FAILED before this process gets a
        # scheduling slice.  Failure therefore has priority over progress and
        # must never be reported as a successful startup acknowledgement.
        for candidate in (stage_failed, stage_progress):
            if candidate.is_file():
                evidence = candidate
                break
        if evidence is not None:
            break
        return_code = process.poll()
        if return_code is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            failure = {
                "schema": STAGE_SCHEMA,
                "status": "startup_failed",
                "returncode": int(return_code),
                "run_log_tail": tail,
                "written_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_x(Path(str(spec["startup_failed"])), failure)
            raise RuntimeError(
                "detached factor stage exited before startup acknowledgement: "
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
        failure = {
            "schema": STAGE_SCHEMA,
            "status": "startup_ack_timeout_terminated",
            "run_log_tail": tail,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_x(Path(str(spec["startup_failed"])), failure)
        raise RuntimeError(
            "detached factor stage produced no startup artifact within 15 s"
        )
    if evidence == stage_failed:
        try:
            stage_failure = _load_json(stage_failed, "factor startup failure")
        except ValueError:
            stage_failure = {
                "unparsed_failure_path": str(stage_failed),
                "run_log_tail": log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:],
            }
        startup_failure = {
            "schema": STAGE_SCHEMA,
            "status": "stage_failed_during_startup_ack",
            "stage_failure": stage_failure,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_x(Path(str(spec["startup_failed"])), startup_failure)
        raise RuntimeError(
            "detached factor stage wrote FAILED before startup acknowledgement: "
            f"{stage_failure}"
        )
    acknowledgement = {
        "schema": STAGE_SCHEMA,
        "status": "launched_detached_startup_acknowledged",
        "pid": int(process.pid),
        "startup_evidence": str(evidence),
        "launch_manifest": str(spec["launch_manifest"]),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_x(Path(str(spec["startup_ack"])), acknowledgement)
    return {**dict(spec), **acknowledgement}


def _run_stage(
    config_path: Path,
    resolved_config: Path,
    factor_plan_path: Path,
    batch_root: Path,
) -> int:
    overlay = _load_overlay(config_path)
    smoke, plan = _load_factor_plan(overlay)
    if not resolved_config.is_file() or not factor_plan_path.is_file():
        raise ValueError("factor stage immutable launch inputs are missing")
    resolved_payload = _load_yaml(resolved_config, "factor resolved audit config")
    provenance = resolved_payload.get("factor_launch_provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("launch_config") != str(config_path.resolve())
        or provenance.get("launch_config_sha256") != _sha256(config_path.resolve())
        or provenance.get("factor_plan_sha256") != plan["plan_sha256"]
        or provenance.get("atomic_all_or_none") is not True
        or provenance.get("no_downstream_chaining") is not True
    ):
        raise ValueError("factor resolved-config launch provenance drifted")
    runtime = resolved_payload.get("factor_realization_runtime")
    if not isinstance(runtime, Mapping) or (
        Path(str(runtime.get("factor_smoke_plan", ""))).resolve()
        != factor_plan_path.resolve()
    ):
        raise ValueError("factor resolved-config plan path drifted")
    source_before = _relevant_source_tree_fingerprint(overlay)
    if provenance.get("relevant_source_tree_fingerprint") != source_before:
        raise ValueError("factor relevant-source tree drifted before raw capture")
    for field in (
        "base_runner_sha256",
        "factor_validator_sha256",
        "factor_postflight_sha256",
    ):
        if provenance.get(field) != overlay[field]:
            raise ValueError(f"factor resolved-config {field} drifted")
    _load_audit_config(resolved_config)
    materialized_plan = _load_json(factor_plan_path, "materialized factor plan")
    if materialized_plan != plan:
        raise ValueError("materialized factor plan differs from current exact plan")
    batch_root.mkdir(parents=True, exist_ok=False)
    progress = batch_root / "progress.jsonl"
    summary_path = batch_root / "RESULTS_SUMMARY.json"
    failed_path = batch_root / "FAILED.json"
    completed_path = batch_root / "COMPLETED.json"
    raw_root = batch_root / str(overlay["exact_batch"]["raw_capture_subdirectory"])
    result_path = raw_root / str(overlay["exact_batch"]["result_bundle_name"])
    raw_validation_path = raw_root / str(
        overlay["exact_batch"]["raw_validation_name"]
    )
    validation_path = batch_root / str(
        overlay["exact_batch"]["atomic_validation_name"]
    )
    try:
        _append_progress(
            progress,
            "stage_started",
            trajectory_count=16,
            group_count=8,
            factor_plan_sha256=plan["plan_sha256"],
            atomic_all_or_none=True,
            oai_executed=False,
        )
        audit_log = batch_root / "raw_capture.run.log"
        command = [
            sys.executable,
            "-m",
            "data_collection.run_phase2_calibration_audit",
            "--config",
            str(resolved_config),
            "--output-dir",
            str(raw_root),
            "--operator-quality",
            "Epic",
            "--launch",
        ]
        _append_progress(progress, "raw_capture_started", raw_capture_root=str(raw_root))
        with audit_log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = audit_log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"raw capture failed: returncode={completed.returncode}; tail={tail!r}"
            )
        if not (raw_root / "COMPLETED.json").is_file():
            raise RuntimeError("raw capture returned zero without its completion sentinel")
        _append_progress(
            progress,
            "raw_capture_complete_pending_atomic_postflight",
            generic_audit_completion_is_scientific_pass=False,
        )
        if not result_path.is_file():
            raise RuntimeError(
                "factor runtime did not produce its required result bundle: "
                f"{result_path}"
            )
        if not raw_validation_path.is_file():
            raise RuntimeError(
                "factor runtime did not produce its required raw validation: "
                f"{raw_validation_path}"
            )
        result = _load_json(result_path, "factor result bundle")
        validation = validate_results(result, smoke, plan)
        raw_validation = _load_json(
            raw_validation_path, "factor raw atomic validation"
        )
        if raw_validation != validation:
            raise RuntimeError(
                "factor raw validation differs from independent outer validation"
            )
        required_verdict = str(
            overlay["runtime_completion"]["required_atomic_validator_verdict"]
        )
        if validation.get("verdict") != required_verdict:
            raise RuntimeError(
                "factor validator did not return the registered atomic PASS: "
                f"expected={required_verdict!r}, "
                f"observed={validation.get('verdict')!r}"
            )
        source_after = _relevant_source_tree_fingerprint(overlay)
        if source_after != source_before:
            raise RuntimeError(
                "factor result-defining Python/YAML source tree changed during the stage"
            )
        _write_json_x(validation_path, validation)
        summary = {
            "schema": STAGE_SCHEMA,
            "status": "complete_atomic_exact_16_admitted",
            "verdict": validation["verdict"],
            "batch_root": str(batch_root),
            "raw_capture_root": str(raw_root),
            "factor_plan_sha256": plan["plan_sha256"],
            "factor_result_sha256": _sha256(result_path),
            "factor_validation_sha256": _sha256(validation_path),
            "relevant_source_tree_manifest_sha256": source_after[
                "manifest_sha256"
            ],
            "trajectory_count": 16,
            "group_count": 8,
            "atomic_all_or_none": True,
            "partial_admission": False,
            "oai_executed": False,
            "downstream_stage_chained": False,
            "near_zero_or_negative_margin_is_valid_and_not_retuned": True,
            "next_action": "human_review_before_any_additional_calibration",
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_progress(progress, "stage_complete", verdict=validation["verdict"])
        _write_json_x(summary_path, summary)
        # COMPLETED is the final fallible filesystem commit.  Nothing after it
        # may enter the failure handler and create a contradictory FAILED.
        _write_json_x(completed_path, summary)
        try:
            print(json.dumps(summary, indent=2, sort_keys=True))
        except (BrokenPipeError, OSError):
            pass
        return 0
    except BaseException as exc:
        # A publish can report an fsync error after its create-only hard link
        # succeeded.  A complete, valid terminal is authoritative; never add a
        # contradictory FAILED.  A malformed/partial injected terminal is an
        # uncommitted artifact and is durably rolled back before failure.
        if completed_path.exists():
            try:
                committed = _load_json(completed_path, "factor completion sentinel")
            except ValueError:
                _unlink_uncommitted_json(completed_path)
            else:
                if (
                    committed.get("status") == "complete_atomic_exact_16_admitted"
                    and committed.get("verdict")
                    == "PASS_ATOMIC_EXACT_16_ADMITTED"
                ):
                    try:
                        print(json.dumps(committed, indent=2, sort_keys=True))
                    except (BrokenPipeError, OSError):
                        pass
                    return 0
                _unlink_uncommitted_json(completed_path)
        failure = {
            "schema": STAGE_SCHEMA,
            "status": "failed_excluded_atomic_fixture",
            "verdict": "FAIL_HOLD_EXCLUDE_ALL_16",
            "error": f"{type(exc).__name__}: {exc}",
            "batch_root": str(batch_root),
            "atomic_all_or_none": True,
            "partial_admission": False,
            "oai_executed": False,
            "downstream_stage_chained": False,
            "next_action": "human_review_root_cause_no_scaling",
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        if summary_path.exists():
            try:
                existing_summary = _load_json(
                    summary_path, "factor results summary"
                )
            except ValueError:
                _unlink_uncommitted_json(summary_path)
            else:
                if existing_summary.get("status") != "failed_excluded_atomic_fixture":
                    _unlink_uncommitted_json(summary_path)
        if not summary_path.exists():
            _write_json_x(summary_path, failure)
        if not failed_path.exists():
            try:
                _write_json_x(failed_path, failure)
            except BaseException:
                # If directory fsync failed after the hard-link commit, retain
                # the valid failure terminal and do not obscure the stage code.
                try:
                    published_failure = _load_json(
                        failed_path, "factor failure sentinel"
                    )
                except ValueError:
                    raise
                if published_failure != failure:
                    raise
        try:
            _append_progress(progress, "stage_failed", error=failure["error"])
        except (OSError, ValueError):
            pass
        try:
            print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        except (BrokenPipeError, OSError):
            pass
        return 1


def _write_child_preflight_failure_create_only(
    batch_root: Path,
    exc: BaseException,
) -> Optional[dict[str, Any]]:
    """Materialize the advertised failure bundle before `_run_stage` owns it.

    The detached parent proves that the batch path is absent before spawning
    this child.  If it is nevertheless present here, do not reuse or pollute
    it: an existing root may belong to another process or a partially running
    stage.  Failures after `_run_stage` creates the root are handled there.
    """

    root = Path(batch_root).resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None
    failure = {
        "schema": STAGE_SCHEMA,
        "status": "failed_excluded_atomic_fixture",
        "phase": "child_preflight_before_raw_capture",
        "verdict": "FAIL_HOLD_EXCLUDE_ALL_16",
        "error": f"{type(exc).__name__}: {exc}",
        "batch_root": str(root),
        "atomic_all_or_none": True,
        "partial_admission": False,
        "oai_executed": False,
        "downstream_stage_chained": False,
        "next_action": "human_review_root_cause_no_scaling",
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_x(root / "RESULTS_SUMMARY.json", failure)
    _write_json_x(root / "FAILED.json", failure)
    _append_progress(
        root / "progress.jsonl",
        "stage_preflight_failed",
        error=failure["error"],
        atomic_all_or_none=True,
    )
    return failure


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--operator-quality", choices=("Epic",), default="Epic")
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--operator")
    parser.add_argument("--confirmed-check", action="append", default=[])
    parser.add_argument("--confirm-all-listed-checks", action="store_true")
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument("--review-plan-output", type=Path)
    # Internal child-only inputs. They are rejected unless --run-stage is used.
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--factor-plan", type=Path)
    parser.add_argument("--batch-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-corner-review-plan", action="store_true")
    mode.add_argument("--write-corner-review-plan", action="store_true")
    mode.add_argument("--record-corner-acceptance", action="store_true")
    mode.add_argument("--validate-launch", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch-detached", action="store_true")
    mode.add_argument("--run-stage", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.run_stage:
        if not all((args.resolved_config, args.factor_plan, args.batch_root)):
            raise ValueError("internal factor stage inputs are incomplete")
        try:
            return _run_stage(
                args.config.resolve(),
                args.resolved_config.resolve(),
                args.factor_plan.resolve(),
                args.batch_root.resolve(),
            )
        except BaseException as exc:
            failure = _write_child_preflight_failure_create_only(
                args.batch_root.resolve(), exc
            )
            diagnostic = failure or {
                "schema": STAGE_SCHEMA,
                "status": "child_preflight_failed_existing_batch_not_modified",
                "error": f"{type(exc).__name__}: {exc}",
                "batch_root": str(args.batch_root.resolve()),
                "written_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(json.dumps(diagnostic, indent=2, sort_keys=True), file=sys.stderr)
            return 1
    if any((args.resolved_config, args.factor_plan, args.batch_root)):
        raise ValueError("internal factor stage inputs require --run-stage")
    if args.print_corner_review_plan or args.write_corner_review_plan:
        if args.review_root is None:
            raise ValueError(
                "corner-review plan requires a fresh explicit --review-root"
            )
        plan = build_corner_review_plan(args.config, review_root=args.review_root)
        if args.write_corner_review_plan:
            if args.review_plan_output is None:
                raise ValueError("--write-corner-review-plan requires --review-plan-output")
            _write_json_x(args.review_plan_output.resolve(), plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.record_corner_acceptance:
        if args.review_root is None or args.operator is None:
            raise ValueError(
                "recording corner acceptance requires --review-root and --operator"
            )
        result = record_corner_acceptance(
            args.config,
            review_root=args.review_root,
            operator=args.operator,
            confirmed_checks=(
                build_corner_review_plan(
                    args.config, review_root=args.review_root
                )["required_human_checks"]
                if args.confirm_all_listed_checks
                else args.confirmed_check
            ),
            output_path=args.acceptance_output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    spec = build_launch_spec(
        args.config,
        output_root=args.output_root,
        operator_quality=args.operator_quality,
        require_corner_acceptance=args.launch_detached,
        include_plan=args.dry_run,
    )
    result = launch_detached(spec) if args.launch_detached else spec
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

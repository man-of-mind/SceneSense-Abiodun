#!/usr/bin/env python3
"""Plan or run a frozen, bounded Phase-2 calibration tranche.

The historical configuration remains bounded to its 15 audit trajectories.
An explicitly enabled v2 factor-smoke overlay may instead pin the exact 16
replicate-0 rows.  Both modes stop at the first failure and never chain into
remaining calibration, validation/test, OAI, controller evaluation, or RL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from data_collection import run_advisor_policy_corpus as advisor
from data_collection import run_policy_corpus as base_runner
from data_collection.phase2_calibration_scenario import (
    CalibrationScenarioRuntime,
    ROLE_NAMES,
    resolve_scenario,
)
from data_collection.phase2_factor_realization_runtime import (
    FactorRuntimeContract,
    canonical_sha256,
    nontreatment_plan_record,
)
from data_collection.phase2_paired_causal_collector import _require_inherited_contract
from data_collection.phase2_static_environment_truth_v1 import (
    MANIFEST_JSON_NAME as STATIC_ENVIRONMENT_MANIFEST_JSON_NAME,
    OBJECTS_CSV_NAME as STATIC_ENVIRONMENT_OBJECTS_CSV_NAME,
    OBJECT_FIELDS as STATIC_ENVIRONMENT_OBJECT_FIELDS,
    capture_static_environment_truth_v1,
)
from data_collection.run_advisor_generate_traffic import (
    READY_SCHEMA as POPULATION_READY_SCHEMA,
    RELEASE_SCHEMA as POPULATION_RELEASE_SCHEMA,
    RELEASED_SCHEMA as POPULATION_RELEASED_SCHEMA,
)
from data_collection.run_phase2_paired_causal_pilot import (
    _drop_options,
    _find_role_actor,
    _require_udp_ports_available,
    _wait_collectors_exit,
    _wait_for_frame,
    _wait_for_ready,
    _wait_for_tick_ready,
)
from phase2_map_sharing.causal_contract import (
    CAUSAL_AUDIT_SCHEMA,
    CausalDecisionAudit,
    CausalField,
    DecisionRecord,
)
from data_collection.validate_phase2_factor_realization_smoke import (
    build_plan as build_factor_smoke_plan,
    load_config as load_factor_smoke_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs/phase2_calibration_audit_v1.yaml"
)
EXPECTED_MANIFEST_COLUMNS = {
    "split",
    "group_id",
    "geometry_or_route_id",
    "traffic_density",
    "traffic_density_status",
    "ambient_population_mode",
    "ambient_population_process_required",
    "carla_seed",
    "traffic_seed",
    "sensor_seed",
    "raw_retention_tier",
    "raw_window_duration_s",
    "trajectory_id",
    "scenario_role",
    "controlled_hazard_present",
    "route_start_anchor_id",
    "weather",
}


def _factor_runtime_bundle(
    config: Mapping[str, object],
) -> Optional[tuple[dict, dict]]:
    """Load and byte-verify the optional exact-16 factor runtime contract."""

    runtime = config.get("factor_realization_runtime")
    if runtime is None:
        return None
    if not isinstance(runtime, Mapping):
        raise ValueError("factor_realization_runtime must be a mapping")
    _require_exact_keys(
        runtime,
        {
            "schema",
            "enabled",
            "factor_smoke_config",
            "factor_smoke_config_sha256",
            "factor_smoke_plan",
            "factor_smoke_plan_sha256",
            "exact_trajectory_count",
            "atomic_batch",
        },
        "factor_realization_runtime",
    )
    if (
        runtime["schema"]
        != "scenesense.phase2_factor_realization_runtime_config.v1"
    ):
        raise ValueError("unsupported factor-realization runtime schema")
    if runtime["enabled"] is not True:
        raise ValueError("present factor_realization_runtime must be enabled")
    if runtime["atomic_batch"] is not True:
        raise ValueError("factor-smoke runtime must be atomic")
    if int(runtime["exact_trajectory_count"]) != 16:
        raise ValueError("factor-smoke runtime must contain exactly 16 trajectories")
    smoke_config_path = _repo_path(runtime["factor_smoke_config"])
    smoke_plan_path = _repo_path(runtime["factor_smoke_plan"])
    for label, path, expected in (
        (
            "config",
            smoke_config_path,
            str(runtime["factor_smoke_config_sha256"]),
        ),
        ("plan", smoke_plan_path, str(runtime["factor_smoke_plan_sha256"])),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"factor-smoke {label} is missing: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"factor-smoke {label} hash drift: expected={expected} observed={observed}"
            )
    smoke_config = load_factor_smoke_config(smoke_config_path)
    expected_plan = build_factor_smoke_plan(smoke_config)
    smoke_plan = json.loads(smoke_plan_path.read_text(encoding="utf-8"))
    if smoke_plan != expected_plan:
        raise ValueError("factor-smoke plan differs from the config-derived plan")
    if int(smoke_plan["trajectory_count"]) != 16:
        raise ValueError("factor-smoke plan trajectory count drifted")
    return smoke_config, smoke_plan


def _factor_contracts(
    config: Mapping[str, object], selected: pd.DataFrame
) -> dict[str, FactorRuntimeContract]:
    bundle = _factor_runtime_bundle(config)
    if bundle is None:
        return {}
    smoke_config, plan = bundle
    maximum_by_class = smoke_config["factor_contract"][
        "positive_hazard_surface_clearance_max_m_by_class"
    ]
    contracts = {
        str(row["trajectory_id"]): FactorRuntimeContract.from_plan_row(
            row,
            maximum_surface_clearance_m=float(maximum_by_class[row["hazard_class"]]),
        )
        for row in plan["rows"]
    }
    selected_ids = selected["trajectory_id"].astype(str).tolist()
    if selected_ids != [str(row["trajectory_id"]) for row in plan["rows"]]:
        raise ValueError("factor-smoke selected order differs from the immutable plan")
    if set(contracts) != set(selected_ids):
        raise ValueError("factor-smoke selected trajectories differ from the plan")
    return contracts
SCENARIO_ROLES = {
    "controlled_positive_occlusion",
    "matched_benign_negative",
    "naturalistic_operation",
}
FROZEN_WARNING_EMISSION_CONFIDENCE_FLOORS = (0.05, 0.10, 0.15, 0.20)
FROZEN_MAP_ASSOCIATION_BASE_GATES_M = (2.0, 3.0, 4.0)
FROZEN_MAP_TRACK_TTLS_S = (0.5, 1.0)
FROZEN_WARNING_UNCERTAINTY_MULTIPLIERS = (0.0, 1.0, 2.0)
FROZEN_REPLAY_COMBINATIONS = 72
STATIC_ENVIRONMENT_SEMANTIC_LABELS = ("Car", "Truck", "Bus")
STATIC_ENVIRONMENT_REQUIRED_SEMANTIC_CLASSES = ("Car",)
STATIC_ENVIRONMENT_SELECTION_CONTRACT = (
    "town10hd_opt_static_vehicle_like_car_truck_bus_all_enabled_after_"
    "fresh_reload_no_environment_toggles.v1"
)
STATIC_ENVIRONMENT_ENABLED_STATE_BASIS = (
    "explicit_all_enabled_registry_valid_only_after_fresh_world_reload_"
    "and_before_any_environment_toggle_or_dynamic_actor_spawn"
)
STATIC_ENVIRONMENT_SEMANTIC_HASH_EXCLUDED_FIELDS = frozenset(
    {"capture_clock_id", "capture_frame_id", "capture_timestamp_s"}
)
STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS = (
    "sha256_canonical_json_sorted_static_ids_classes_enabled_transforms_"
    "oriented_bboxes_and_map_excluding_capture_clock_fields"
)
FACTOR_RETENTION_PRE_ONSET_S = 0.9
FACTOR_RETENTION_MINIMUM_POST_ONSET_S = 3.0


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _ambient_layer(
    config: Mapping[str, object], scenario_role: object
) -> tuple[str, Mapping[str, object]]:
    """Resolve the explicit evidence-layer contract for one trajectory."""

    traffic = config["ambient_traffic"]
    role = str(scenario_role)
    try:
        layer_id = str(traffic["layer_by_scenario_role"][role])
        layer = traffic["layers"][layer_id]
    except KeyError as exc:
        raise ValueError(
            f"ambient traffic layer is not defined for scenario role {role!r}"
        ) from exc
    return layer_id, layer


def _ambient_counts(
    config: Mapping[str, object], row: Mapping[str, object]
) -> tuple[str, Mapping[str, object], Mapping[str, object]]:
    layer_id, layer = _ambient_layer(config, row["scenario_role"])
    density = str(row["traffic_density"])
    try:
        counts = layer["counts_by_density"][density]
    except KeyError as exc:
        raise ValueError(
            f"ambient layer {layer_id!r} has no counts for density {density!r}"
        ) from exc
    return layer_id, layer, counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retention_window_for_row(
    config: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, object]:
    """Resolve a historical geometry window or a v2 onset-aligned window.

    Authored onset remains evaluation/orchestration metadata.  The derived
    retention offset is passed only to the artifact logger, never into either
    policy feature projection.
    """

    duration = float(config["clock"]["duration_s"])
    window = float(config["capture"]["raw_window_duration_s"])
    if config.get("factor_realization_runtime") is None:
        offset = float(
            config["capture"]["raw_window_start_offset_s_by_geometry_or_route"][
                str(row["geometry_or_route_id"])
            ]
        )
        return {
            "start_offset_s": offset,
            "end_offset_s": offset + window,
            "basis": "historical_geometry_static_offset",
            "authored_onset_policy_visibility": "not_applicable",
        }
    onset = float(row["requested_hazard_onset_s"])
    offset = max(
        0.0,
        min(duration - window, onset - FACTOR_RETENTION_PRE_ONSET_S),
    )
    end = offset + window
    if not offset - 1e-12 <= onset <= end + 1e-12:
        raise ValueError("factor retention window does not contain authored onset")
    if end + 1e-12 < min(
        duration, onset + FACTOR_RETENTION_MINIMUM_POST_ONSET_S
    ):
        raise ValueError("factor retention window lacks the registered post-onset span")
    return {
        "start_offset_s": offset,
        "end_offset_s": end,
        "authored_onset_s": onset,
        "pre_onset_s": onset - offset,
        "post_onset_s": end - onset,
        "basis": "authored_onset_minus_0p9s_bounded_to_trajectory_evaluation_metadata_only",
        "authored_onset_policy_visibility": "forbidden",
    }


def _expected_retention_bytes(
    *,
    storage: Mapping[str, object],
    retained_frames_per_role: int,
    tiers: Sequence[object],
) -> tuple[list[int], int]:
    """Return exact two-role heavy-byte estimates for mixed retention tiers."""

    input_bytes = int(storage["measured_role_input_bytes_per_frame"])
    logits_bytes = int(storage["measured_role_logits_bytes_per_frame"])
    retained_frames = int(retained_frames_per_role)
    if input_bytes <= 0 or logits_bytes <= 0 or retained_frames <= 0:
        raise ValueError("retention byte inputs and frame count must be positive")
    estimates = []
    for tier in tiers:
        value = str(tier)
        if value not in {"inputs_only_window", "inputs_plus_logits_window"}:
            raise ValueError(f"unsupported manifest retention tier: {value}")
        estimates.append(
            len(ROLE_NAMES)
            * retained_frames
            * (
                input_bytes
                + (logits_bytes if value == "inputs_plus_logits_window" else 0)
            )
        )
    if not estimates:
        raise ValueError("retention estimate requires at least one trajectory")
    return estimates, sum(estimates)


def _static_environment_truth_config(
    config: Mapping[str, object],
) -> Optional[Mapping[str, object]]:
    """Validate the opt-in contract used only by future bounded captures.

    Absence preserves the semantics of the already accepted v1 calibration
    audit.  Presence is deliberately strict: a caller cannot silently broaden
    the catalog to CARLA's very large ``Other`` class or weaken the fresh-world
    basis behind the explicit all-enabled registry.
    """

    value = config.get("static_environment_truth")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("static_environment_truth must be a mapping")
    _require_exact_keys(
        value,
        {
            "enabled",
            "semantic_labels",
            "required_semantic_classes",
            "selection_contract",
            "enabled_state_basis",
        },
        "static_environment_truth",
    )
    if value["enabled"] is not True:
        raise ValueError(
            "static_environment_truth must be absent or explicitly enabled"
        )
    if tuple(str(item) for item in value["semantic_labels"]) != (
        STATIC_ENVIRONMENT_SEMANTIC_LABELS
    ):
        raise ValueError(
            "static environment labels must be exactly Car, Truck, Bus"
        )
    if tuple(str(item) for item in value["required_semantic_classes"]) != (
        STATIC_ENVIRONMENT_REQUIRED_SEMANTIC_CLASSES
    ):
        raise ValueError("static environment required class must be exactly Car")
    if str(value["selection_contract"]) != STATIC_ENVIRONMENT_SELECTION_CONTRACT:
        raise ValueError("static environment selection contract drifted")
    if str(value["enabled_state_basis"]) != STATIC_ENVIRONMENT_ENABLED_STATE_BASIS:
        raise ValueError("static environment enabled-state basis drifted")
    carla_config = config.get("carla")
    if not isinstance(carla_config, Mapping):
        raise ValueError("CARLA config is required for static environment truth")
    if str(carla_config.get("expected_town")) != "Town10HD_Opt":
        raise ValueError("static environment truth is frozen to Town10HD_Opt")
    if carla_config.get("reload_world_before_trajectory") is not True:
        raise ValueError(
            "static environment all-enabled basis requires a fresh world reload"
        )
    return value


def _static_environment_semantic_sha256(static_dir: Path) -> str:
    """Hash static semantics/geometry independently of snapshot time."""

    csv_path = Path(static_dir) / STATIC_ENVIRONMENT_OBJECTS_CSV_NAME
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != STATIC_ENVIRONMENT_OBJECT_FIELDS:
            raise ValueError("static environment CSV columns differ from schema")
        rows = list(reader)
    if not rows:
        raise ValueError("static environment semantic hash requires objects")
    included_fields = tuple(
        field
        for field in STATIC_ENVIRONMENT_OBJECT_FIELDS
        if field not in STATIC_ENVIRONMENT_SEMANTIC_HASH_EXCLUDED_FIELDS
    )
    canonical_rows = [
        {field: row[field] for field in included_fields}
        for row in sorted(
            rows, key=lambda item: int(item["carla_environment_object_id"])
        )
    ]
    canonical = json.dumps(
        {
            "schema": "scenesense.phase2_static_environment_semantic_hash.v1",
            "basis": STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS,
            "objects": canonical_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _capture_static_environment_truth_before_dynamic_actors(
    world: object,
    trajectory_dir: Path,
    config: Mapping[str, object],
) -> Optional[dict]:
    """Capture and seal the per-trajectory static map-object catalog.

    The CARLA API does not expose an enabled-state getter.  Therefore this
    runner may construct an explicit all-enabled registry only at the narrow
    point enforced here: after its fresh map reload, before any dynamic actor
    exists, and before this owner has issued any environment-object toggle.
    """

    contract = _static_environment_truth_config(config)
    if contract is None:
        return None
    dynamic_inventory = advisor._actor_inventory(world)
    occupied = {
        name: int(count) for name, count in dynamic_inventory.items() if int(count)
    }
    if occupied:
        raise RuntimeError(
            "static environment truth must precede every dynamic actor spawn: "
            f"{occupied}"
        )

    labels = [getattr(advisor.carla.CityObjectLabel, name) for name in (
        STATIC_ENVIRONMENT_SEMANTIC_LABELS
    )]
    enabled_state_by_id: dict[int, bool] = {}
    queried_counts: dict[str, int] = {}
    for name, label in zip(STATIC_ENVIRONMENT_SEMANTIC_LABELS, labels):
        objects = world.get_environment_objects(label)
        if objects is None:
            raise RuntimeError(
                f"CARLA returned no static environment result for label {name}"
            )
        objects = list(objects)
        queried_counts[name] = len(objects)
        for environment_object in objects:
            native_id = int(environment_object.id)
            if native_id in enabled_state_by_id:
                raise RuntimeError(
                    "duplicate static environment object across semantic labels: "
                    f"{native_id}"
                )
            enabled_state_by_id[native_id] = True

    static_dir = trajectory_dir / "static_environment_truth"
    result = capture_static_environment_truth_v1(
        world,
        static_dir,
        semantic_labels=labels,
        required_semantic_classes=STATIC_ENVIRONMENT_REQUIRED_SEMANTIC_CLASSES,
        enabled_state_by_id=enabled_state_by_id,
        selection_contract=STATIC_ENVIRONMENT_SELECTION_CONTRACT,
    )
    manifest_path = static_dir / STATIC_ENVIRONMENT_MANIFEST_JSON_NAME
    return {
        "schema": "scenesense.phase2_static_environment_runner_record.v1",
        "status": "complete",
        "path": str(static_dir.relative_to(trajectory_dir)),
        "artifact_manifest_sha256": _sha256(manifest_path),
        "static_geometry_semantic_sha256": (
            _static_environment_semantic_sha256(static_dir)
        ),
        "static_geometry_semantic_hash_basis": (
            STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS
        ),
        "semantic_labels": list(STATIC_ENVIRONMENT_SEMANTIC_LABELS),
        "required_semantic_classes": list(
            STATIC_ENVIRONMENT_REQUIRED_SEMANTIC_CLASSES
        ),
        "selection_contract": STATIC_ENVIRONMENT_SELECTION_CONTRACT,
        "enabled_state_basis": STATIC_ENVIRONMENT_ENABLED_STATE_BASIS,
        "enabled_state_is_owner_assertion_not_rpc_observation": True,
        "fresh_world_reload_performed_by_runner": True,
        "environment_object_toggle_calls_before_snapshot": 0,
        "dynamic_actor_inventory_before_snapshot": dict(dynamic_inventory),
        "queried_object_counts": queried_counts,
        "capture_result": result,
    }


def _write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _persist_factor_forensic_then_finalize(
    *,
    scenario_dir: Path,
    trajectory_id: str,
    contract: FactorRuntimeContract,
    nontreatment_plan_sha256: str,
    scenario_summary: Mapping[str, object],
    scenario_runtime: CalibrationScenarioRuntime,
) -> dict[str, object]:
    """Persist the exact diagnostic before converting it to a hard gate.

    A failed physical realization is still useful for repairing provisional
    controls.  Create-only persistence must therefore finish before
    ``factor_result()`` is allowed to raise and stop the atomic batch.
    """

    factor_diagnostic = {
        key: scenario_summary[key]
        for key in (
            "realized_factors",
            "factor_realization_gate",
            "registered_target_absent",
            "realized_factors_status",
            "factor_reference_trajectory_id",
        )
        if key in scenario_summary
    }
    artifact_path = scenario_dir / "factor_realization.json"
    _write_json_create(
        artifact_path,
        {
            "schema": "scenesense.phase2_factor_realization_runtime.v1",
            "trajectory_id": str(trajectory_id),
            "trajectory_row_sha256": contract.trajectory_row_sha256,
            "scenario_role": contract.scenario_role,
            "requested_factors": dict(contract.requested),
            "nontreatment_plan_sha256": str(nontreatment_plan_sha256),
            **factor_diagnostic,
        },
    )
    if not artifact_path.is_file():  # pragma: no cover - defensive filesystem gate
        raise RuntimeError("factor-realization forensic artifact was not persisted")
    return scenario_runtime.factor_result()


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _append_progress(path: Path, event: str, **fields: object) -> None:
    payload = {
        "schema": "scenesense.phase2_calibration_audit_progress.v1",
        "event": str(event),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected)}"
        )


def _validate_replay_grid(replay: Mapping[str, object]) -> int:
    """Validate the frozen map-engine grid and its fixed source contract."""

    _require_exact_keys(
        replay,
        {"fixed_source_contract", "map_engine_axes", "expected_combinations"},
        "verification.replay_grid",
    )
    source = replay["fixed_source_contract"]
    if not isinstance(source, Mapping):
        raise ValueError("replay fixed_source_contract must be a mapping")
    _require_exact_keys(
        source,
        {"detector_confidence_floor", "source_local_tracker"},
        "verification.replay_grid.fixed_source_contract",
    )
    if float(source["detector_confidence_floor"]) != 0.05:
        raise ValueError("source detector confidence floor must remain fixed at 0.05")
    tracker = source["source_local_tracker"]
    if not isinstance(tracker, Mapping):
        raise ValueError("source_local_tracker contract must be a mapping")
    _require_exact_keys(
        tracker,
        {"status", "association_gate_m", "maximum_missed_frames"},
        "verification.replay_grid.fixed_source_contract.source_local_tracker",
    )
    if tracker["status"] != "fixed_capture_contract_not_replayed_or_tuned":
        raise ValueError("source-local tracker must remain fixed during map replay")
    if float(tracker["association_gate_m"]) != 5.0:
        raise ValueError("fixed source-local tracker association gate must be 5 m")
    if float(tracker["maximum_missed_frames"]) != 3.0:
        raise ValueError("fixed source-local tracker missed-frame limit must be 3")

    axes = replay["map_engine_axes"]
    if not isinstance(axes, Mapping):
        raise ValueError("replay map_engine_axes must be a mapping")
    _require_exact_keys(
        axes,
        {
            "warning_emission_confidence_floors",
            "association_base_gates_m",
            "track_ttls_s",
            "warning_uncertainty_multipliers",
        },
        "verification.replay_grid.map_engine_axes",
    )
    expected_axes = {
        "warning_emission_confidence_floors": (
            FROZEN_WARNING_EMISSION_CONFIDENCE_FLOORS
        ),
        "association_base_gates_m": FROZEN_MAP_ASSOCIATION_BASE_GATES_M,
        "track_ttls_s": FROZEN_MAP_TRACK_TTLS_S,
        "warning_uncertainty_multipliers": (
            FROZEN_WARNING_UNCERTAINTY_MULTIPLIERS
        ),
    }
    for name, expected in expected_axes.items():
        try:
            observed = tuple(float(value) for value in axes[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"map-engine replay axis {name} must be numeric") from exc
        if observed != expected:
            raise ValueError(
                f"map-engine replay axis {name} drifted: "
                f"expected={list(expected)}, observed={list(observed)}"
            )
    combinations = math.prod(len(values) for values in expected_axes.values())
    if combinations != FROZEN_REPLAY_COMBINATIONS:
        raise AssertionError("internal frozen replay-grid cardinality is inconsistent")
    if float(replay["expected_combinations"]) != float(combinations):
        raise ValueError(
            f"registered map-engine replay grid must contain {combinations} combinations"
        )
    return combinations


def _load_world_with_retry(
    client: object, town: str, reset_settings: bool, *, attempts: int = 3
) -> object:
    """Reset the requested map, avoiding CARLA's same-map load failure."""

    if attempts <= 0:
        raise ValueError("world-load attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            current = client.get_world()
            current_name = str(current.get_map().name)
            if current_name.endswith(str(town)):
                return client.reload_world(bool(reset_settings))
            return client.load_world(str(town), bool(reset_settings))
        except RuntimeError as exc:
            if str(exc).strip() not in {"Operation aborted.", "std::exception"}:
                raise
            if attempt == attempts:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable world-load retry state")


def _load_config(path: Path) -> tuple[dict, dict, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != (
        "scenesense.phase2_calibration_audit.v1"
    ):
        raise ValueError("unexpected calibration-audit config schema")
    if config.get("implementation_status") != "reviewed_audit_stage_only":
        raise ValueError("calibration audit must be reviewed_audit_stage_only")
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("audit authorization mapping is required")
    _require_exact_keys(
        authorization,
        {
            "carla_launch",
            "oai_launch",
            "remaining_calibration",
            "validation_collection",
            "test_collection",
            "controller_evaluation",
            "rl_training",
        },
        "authorization",
    )
    if not bool(authorization["carla_launch"]) or any(
        bool(authorization[field])
        for field in authorization
        if field != "carla_launch"
    ):
        raise ValueError("only the bounded CARLA audit may be authorized")
    if not bool(config.get("manual_detached_launch_only")):
        raise ValueError("audit must require detached/manual launch")
    _static_environment_truth_config(config)
    factor_bundle = _factor_runtime_bundle(config)

    design = config["design"]
    for field in ("config", "trajectory_manifest"):
        candidate = _repo_path(design[field])
        if not candidate.is_file():
            raise FileNotFoundError(f"audit design prerequisite missing: {candidate}")
        expected_hash = str(design[f"{field}_sha256"])
        observed_hash = _sha256(candidate)
        if observed_hash != expected_hash:
            raise ValueError(
                f"audit design {field} hash drift: expected={expected_hash}, "
                f"observed={observed_hash}"
            )
    source_path = _repo_path(config["source_collection_config"])
    if _sha256(source_path) != str(config["source_collection_config_sha256"]):
        raise ValueError("source collection config hash drifted")
    source = advisor._load_config(source_path)
    for field in ("causal_contract_config", "retention_config", "collector"):
        if not _repo_path(config[field]).is_file():
            raise FileNotFoundError(f"audit prerequisite missing: {_repo_path(config[field])}")

    manifest = pd.read_csv(_repo_path(design["trajectory_manifest"]))
    missing = EXPECTED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"trajectory manifest is missing columns: {sorted(missing)}")
    selector = design["selector"]
    split_rows = manifest[
        manifest["split"].astype(str).eq(str(selector["split"]))
    ].copy()
    selected = (
        split_rows
        if factor_bundle is not None
        else split_rows[
            split_rows["raw_retention_tier"].astype(str).eq(
                str(selector["raw_retention_tier"])
            )
        ].copy()
    )
    exact_ids = selector.get("exact_trajectory_ids")
    if factor_bundle is not None:
        if not isinstance(exact_ids, list) or len(exact_ids) != 16:
            raise ValueError("factor-smoke selector must pin an ordered exact-16 ID list")
        if len(set(str(value) for value in exact_ids)) != 16:
            raise ValueError("factor-smoke selector contains duplicate trajectory IDs")
        available = set(selected["trajectory_id"].astype(str))
        missing_ids = sorted(set(str(value) for value in exact_ids) - available)
        if missing_ids:
            raise ValueError(f"factor-smoke selector IDs are absent: {missing_ids}")
        order = {str(value): index for index, value in enumerate(exact_ids)}
        selected = selected[
            selected["trajectory_id"].astype(str).isin(order)
        ].copy()
        selected["_factor_selection_order"] = selected["trajectory_id"].astype(str).map(order)
        selected = selected.sort_values("_factor_selection_order").drop(
            columns="_factor_selection_order"
        )
    elif exact_ids is not None:
        raise ValueError("exact_trajectory_ids require factor_realization_runtime")
    expected_trajectory_count = int(selector["expected_trajectory_count"])
    expected_group_count = int(selector["expected_group_count"])
    if len(selected) != expected_trajectory_count:
        raise ValueError(
            "audit selector trajectory count drifted: "
            f"observed={len(selected)} expected={expected_trajectory_count}"
        )
    if selected["group_id"].nunique() != expected_group_count:
        raise ValueError(
            "audit selector group count drifted: "
            f"observed={selected['group_id'].nunique()} expected={expected_group_count}"
        )
    if selected["trajectory_id"].duplicated().any():
        raise ValueError("audit trajectory IDs are not unique")
    if factor_bundle is not None:
        allowed_tiers = {"inputs_only_window", "inputs_plus_logits_window"}
        observed_tiers = set(selected["raw_retention_tier"].astype(str))
        if not observed_tiers <= allowed_tiers:
            raise ValueError(
                f"factor-smoke manifest has unsupported retention tiers: {observed_tiers}"
            )
        if not all(
            math.isclose(float(value), 4.0, abs_tol=1e-12)
            for value in selected["raw_window_duration_s"]
        ):
            raise ValueError("factor-smoke rows must retain exact four-second windows")
    designed = selected[selected["scenario_role"].ne("naturalistic_operation")]
    expected_hazard_marker = {
        "controlled_positive_occlusion": "1",
        "matched_benign_negative": "0",
        "naturalistic_operation": "unforced",
    }
    for row in selected.to_dict("records"):
        if str(row["controlled_hazard_present"]) != expected_hazard_marker[
            str(row["scenario_role"])
        ]:
            raise ValueError(
                "scenario role and controlled-hazard marker disagree for "
                f"{row['trajectory_id']}"
            )
        designed_role = str(row["scenario_role"]) != "naturalistic_operation"
        if designed_role:
            expected_population = (
                "not_applicable",
                "not_applicable",
                "scenario_owned_only",
                0,
            )
        else:
            if str(row["traffic_density"]) not in {"sparse", "typical", "dense"}:
                raise ValueError(
                    "naturalistic manifest row has invalid traffic density: "
                    f"{row['trajectory_id']}"
                )
            expected_population = (
                str(row["traffic_density"]),
                "realized_nuisance_factor",
                "naturalistic_tm",
                1,
            )
        observed_population = (
            str(row["traffic_density"]),
            str(row["traffic_density_status"]),
            str(row["ambient_population_mode"]),
            int(row["ambient_population_process_required"]),
        )
        if observed_population != expected_population:
            raise ValueError(
                "manifest ambient-population contract disagrees with scenario role: "
                f"{row['trajectory_id']} observed={observed_population} "
                f"expected={expected_population}"
            )
    pair_sizes = designed.groupby("group_id")["scenario_role"].agg(set)
    expected_pair = {"controlled_positive_occlusion", "matched_benign_negative"}
    if any(value != expected_pair for value in pair_sizes):
        raise ValueError("each designed audit group must be an exact positive/benign pair")
    if factor_bundle is None:
        if len(designed) != 12 or len(selected) - len(designed) != 3:
            raise ValueError(
                "historical audit must contain 12 designed and three naturalistic trajectories"
            )
    elif len(designed) != 16 or len(selected) != len(designed):
        raise ValueError("factor-smoke tranche must contain exactly 16 designed trajectories")
    for _group_id, rows in designed.groupby("group_id"):
        for field in ("carla_seed", "traffic_seed", "sensor_seed", "traffic_density"):
            if rows[field].nunique(dropna=False) != 1:
                raise ValueError(f"matched audit pair differs in {field}")

    clock = config["clock"]
    if not math.isclose(float(clock["world_hz"]), 10.0, abs_tol=1e-12):
        raise ValueError("audit world clock must be 10 Hz")
    if not math.isclose(float(clock["fixed_delta_seconds"]), 0.1, abs_tol=1e-12):
        raise ValueError("audit fixed delta must be 0.1 s")
    if int(clock["frames_per_trajectory"]) != round(
        float(clock["duration_s"]) * float(clock["world_hz"])
    ):
        raise ValueError("audit duration and frame count disagree")
    capture = config["capture"]
    if bool(capture["warnings_actuated"]):
        raise ValueError("calibration warnings must remain record-only")
    if int(capture["retained_frames_per_role"]) != 40 or not math.isclose(
        float(capture["raw_window_duration_s"]), 4.0, abs_tol=1e-12
    ):
        raise ValueError("audit retention must be exactly 40 frames at 10 Hz")
    identities = set(selected["geometry_or_route_id"].astype(str))
    offset_identities = set(capture["raw_window_start_offset_s_by_geometry_or_route"])
    if factor_bundle is None and offset_identities != identities:
        raise ValueError("raw-window offset table does not cover exact audit identities")
    if factor_bundle is not None and not identities <= offset_identities:
        raise ValueError("raw-window offset table omits a factor-smoke identity")
    for identity, offset in capture[
        "raw_window_start_offset_s_by_geometry_or_route"
    ].items():
        if not 0.0 <= float(offset) <= float(clock["duration_s"]) - 4.0:
            raise ValueError(f"raw-window offset is outside the trajectory: {identity}")
    if set(config["staging_roles"]) != set(ROLE_NAMES):
        raise ValueError("staging roles must be helper and recipient")
    ports = [
        int(value)
        for role_ports in capture["ports"].values()
        for value in role_ports.values()
    ]
    if len(ports) != 8 or len(set(ports)) != 8:
        raise ValueError("all audit loopback UDP ports must be unique")
    traffic = config["ambient_traffic"]
    if bool(traffic["use_spawn_blocker"]):
        raise ValueError("controlled hazards must not be duplicated by spawn_blocker")
    if not bool(traffic["safe_blueprints"]):
        raise ValueError("audit ambient traffic must use the --safe filter")
    if set(traffic["layer_by_scenario_role"]) != SCENARIO_ROLES:
        raise ValueError("ambient layer map must cover every scenario role exactly")
    layers = traffic["layers"]
    if set(layers) != {"designed_frozen", "naturalistic_tm"}:
        raise ValueError("audit requires exactly the designed and naturalistic layers")
    if traffic["layer_by_scenario_role"] != {
        "controlled_positive_occlusion": "designed_frozen",
        "matched_benign_negative": "designed_frozen",
        "naturalistic_operation": "naturalistic_tm",
    }:
        raise ValueError("scenario roles are assigned to the wrong evidence layer")
    expected_layer_motion = {
        "designed_frozen": (
            "stationary_context",
            "runner_owned_stationary_context",
            "runner_owned_stationary",
        ),
        "naturalistic_tm": (
            "tm_autonomous",
            "runner_owned_tm_autonomous",
            "walker_ai_destination",
        ),
    }
    naturalistic_density_names = {"sparse", "typical", "dense"}
    for layer_id, layer in layers.items():
        expected_density_names = (
            {"not_applicable"}
            if layer_id == "designed_frozen"
            else naturalistic_density_names
        )
        if set(layer["counts_by_density"]) != expected_density_names:
            raise ValueError(f"ambient layer {layer_id} has an incomplete density table")
        observed_motion = (
            str(layer["npc_route_mode"]),
            str(layer["released_vehicle_motion_mode"]),
            str(layer["released_walker_motion_mode"]),
        )
        if observed_motion != expected_layer_motion[layer_id]:
            raise ValueError(f"ambient layer {layer_id} has the wrong motion contract")
        expected_population_process = layer_id == "naturalistic_tm"
        if bool(layer["population_process_required"]) != expected_population_process:
            raise ValueError(
                f"ambient layer {layer_id} has the wrong population-process contract"
            )
        minimum_offset = float(layer["minimum_route_offset_m"])
        maximum_offset = float(layer["maximum_route_offset_m"])
        if not 0.0 <= minimum_offset < maximum_offset <= 40.0:
            raise ValueError(f"ambient layer {layer_id} route offset is invalid")
        if not 0.0 < float(layer["maximum_route_heading_error_deg"]) < 90.0:
            raise ValueError(f"ambient layer {layer_id} heading gate is invalid")
        for density, counts in layer["counts_by_density"].items():
            vehicles = int(counts["vehicles"])
            walkers = int(counts["walkers"])
            minimum_walkers = int(counts["minimum_walkers_ready"])
            if vehicles < 0 or walkers < 0:
                raise ValueError(f"ambient layer {layer_id}/{density} has invalid counts")
            if layer_id == "naturalistic_tm" and vehicles <= 0:
                raise ValueError(
                    f"naturalistic layer {density} requires moving vehicle traffic"
                )
            if minimum_walkers != walkers:
                raise ValueError(
                    f"ambient layer {layer_id}/{density} must require every walker"
                )
    if bool(layers["designed_frozen"]["route_derived_spawn_fallback"]):
        raise ValueError("designed static context must never fall back onto ego routes")
    if not bool(layers["naturalistic_tm"]["route_derived_spawn_fallback"]):
        raise ValueError("naturalistic traffic must retain the reviewed-route fallback")
    designed_counts = layers["designed_frozen"]["counts_by_density"]
    naturalistic_counts = layers["naturalistic_tm"]["counts_by_density"]
    if int(designed_counts["not_applicable"]["vehicles"]) != 0:
        raise ValueError("designed layer must not depend on generic ambient vehicles")
    if int(designed_counts["not_applicable"]["walkers"]) != 0:
        raise ValueError("designed layer must not depend on generic ambient walkers")
    if [int(naturalistic_counts[name]["vehicles"]) for name in ("sparse", "typical", "dense")] != [6, 10, 15]:
        raise ValueError("naturalistic vehicle counts drifted")
    if [int(naturalistic_counts[name]["walkers"]) for name in ("sparse", "typical", "dense")] != [4, 8, 12]:
        raise ValueError("naturalistic walker counts drifted")
    for field in (
        "vehicle_spawn_clearance_m_by_density",
        "protected_location_clearance_m_by_density",
        "route_derived_same_route_spacing_m_by_density",
        "route_derived_cross_route_clearance_m_by_density",
    ):
        if set(traffic[field]) != naturalistic_density_names:
            raise ValueError(f"{field} does not cover every traffic density")
        if any(float(value) < 4.0 for value in traffic[field].values()):
            raise ValueError(f"{field} values must be at least 4 m")
    for density in naturalistic_density_names:
        same_route = float(
            traffic["route_derived_same_route_spacing_m_by_density"][density]
        )
        cross_route = float(
            traffic["route_derived_cross_route_clearance_m_by_density"][density]
        )
        vehicle_clearance = float(
            traffic["vehicle_spawn_clearance_m_by_density"][density]
        )
        if cross_route > same_route:
            raise ValueError(
                "route-derived cross-route clearance cannot exceed same-route spacing"
            )
        if vehicle_clearance > cross_route:
            raise ValueError(
                "advisor actor clearance cannot exceed route-derived cross-route clearance"
            )
    # The shared monitor can still read historical trace-replay outputs.  The
    # current audit deliberately selects frozen context or ordinary TM instead.
    replay_speed = float(traffic["npc_trace_replay_speed_mps"])
    replay_delta = float(traffic["npc_trace_replay_fixed_delta_seconds"])
    replay_step = float(traffic["npc_trace_replay_step_m"])
    replay_horizon = float(traffic["npc_trace_replay_horizon_m"])
    replay_clearance = float(traffic["npc_trace_replay_minimum_clearance_m"])
    if not 2.0 <= replay_speed <= 10.0:
        raise ValueError("deterministic replay speed must be within 2-10 m/s")
    if not math.isclose(
        replay_delta, float(clock["fixed_delta_seconds"]), abs_tol=1e-12
    ):
        raise ValueError("deterministic replay and CARLA clock deltas differ")
    if not 0.25 <= replay_step <= 2.0:
        raise ValueError("deterministic replay waypoint step must be within 0.25-2 m")
    required_replay_distance = replay_speed * float(clock["duration_s"])
    if replay_horizon < required_replay_distance + 10.0:
        raise ValueError("deterministic replay horizon lacks the 10 m reserve")
    if not 3.0 <= replay_clearance <= 8.0:
        raise ValueError("deterministic replay clearance must be within 3-8 m")
    if not bool(traffic["require_all_walker_controllers_ready"]):
        raise ValueError("audit must wait for every requested ambient walker controller")
    minimum_per_frame_observation = float(
        traffic["traffic_sanity_gate"][
            "minimum_per_actor_frame_observation_fraction"
        ]
    )
    if not 0.0 < minimum_per_frame_observation <= 1.0:
        raise ValueError("traffic per-actor frame observation fraction must be within (0, 1]")
    stationary_path_limit = float(
        traffic["traffic_sanity_gate"][
            "maximum_stationary_context_path_distance_m"
        ]
    )
    if not 0.0 < stationary_path_limit <= 0.25:
        raise ValueError("stationary-context path-distance gate must be within (0, 0.25]")
    static_collision_gate = float(
        traffic["traffic_sanity_gate"][
            "minimum_static_collision_horizontal_impulse"
        ]
    )
    if not 0.0 < static_collision_gate <= 1000.0:
        raise ValueError(
            "static collision horizontal-impulse gate must be within (0, 1000]"
        )
    if not 1.0 <= float(config["controlled_motion"]["pedestrian_speed_mps"]) <= 2.0:
        raise ValueError("controlled pedestrian speed is not realistic")
    storage = config["storage"]
    if bool(storage["allow_automatic_dataset_deletion"]):
        raise ValueError("audit must never delete prior datasets")
    retained_frames = int(capture["retained_frames_per_role"])
    expected_by_trajectory, expected_stage = _expected_retention_bytes(
        storage=storage,
        retained_frames_per_role=retained_frames,
        tiers=selected["raw_retention_tier"].tolist(),
    )
    if int(storage["estimated_heavy_bytes"]) != expected_stage:
        raise ValueError("audit heavy-data estimate does not match the frozen window count")
    if int(storage["per_trajectory_hard_cap_bytes"]) < max(expected_by_trajectory):
        raise ValueError("per-trajectory raw cap is below the measured planning estimate")
    if int(storage["estimated_heavy_bytes"]) > int(storage["stage_hard_cap_bytes"]):
        raise ValueError("estimated audit data exceed the hard cap")
    if int(storage["preflight_required_free_bytes"]) != (
        int(storage["required_free_floor_bytes"]) + int(storage["stage_hard_cap_bytes"])
    ):
        raise ValueError("storage reservation arithmetic is inconsistent")
    oai = config["verification"]["oai_fields"]
    if oai.get("status") != "not_measured_in_carla_only_audit_remains_blocking":
        raise ValueError("CARLA audit must not claim to measure OAI delivery")
    _validate_replay_grid(config["verification"]["replay_grid"])
    verification = config["verification"]
    if int(verification["required_retained_frame_pairs_per_role"]) != int(
        capture["retained_frames_per_role"]
    ):
        raise ValueError("capture and verifier retained-frame counts disagree")
    if int(verification["required_causal_decisions_per_role"]) != 2 * int(
        clock["frames_per_trajectory"]
    ):
        raise ValueError("audit requires placement and publication decisions per frame")
    pair_gate = verification["matched_pair_initial_realization_gate"]
    _require_exact_keys(
        pair_gate,
        {
            "maximum_horizontal_error_m",
            "maximum_yaw_error_deg",
        },
        "matched_pair_initial_realization_gate",
    )
    if not 0.0 <= float(pair_gate["maximum_horizontal_error_m"]) <= 0.25:
        raise ValueError("matched-pair horizontal tolerance must be within [0, 0.25] m")
    if not 0.0 <= float(pair_gate["maximum_yaw_error_deg"]) <= 0.50:
        raise ValueError("matched-pair yaw tolerance must be within [0, 0.50] degrees")
    trajectory_gate = verification["matched_pair_trajectory_gate"]
    _require_exact_keys(
        trajectory_gate,
        {
            "required_frames_per_actor",
            "maximum_horizontal_error_m",
            "maximum_vertical_error_m",
            "maximum_speed_error_mps",
            "require_identical_replay_plan_sha256",
        },
        "matched_pair_trajectory_gate",
    )
    if int(trajectory_gate["required_frames_per_actor"]) != int(
        clock["frames_per_trajectory"]
    ):
        raise ValueError("matched trajectory gate must cover the full capture")
    for field in ("maximum_horizontal_error_m", "maximum_speed_error_mps"):
        if not 0.0 <= float(trajectory_gate[field]) <= 0.05:
            raise ValueError(f"matched trajectory {field} tolerance is too loose")
    if not 0.0 <= float(trajectory_gate["maximum_vertical_error_m"]) <= 0.25:
        raise ValueError("matched trajectory vertical tolerance exceeds settle bound")
    if trajectory_gate["require_identical_replay_plan_sha256"] is not True:
        raise ValueError("matched trajectories must use the identical replay plan")
    radar_gate = verification["radar_density_gate"]
    if float(radar_gate["reference_projected_points_median"]) <= 0.0:
        raise ValueError("radar-density reference must be positive")
    if not 0.0 < float(radar_gate["relative_tolerance"]) < 1.0:
        raise ValueError("radar-density tolerance must be within (0, 1)")
    if not 1 <= int(radar_gate["minimum_metric_frames"]) <= int(
        clock["frames_per_trajectory"]
    ):
        raise ValueError("radar-density minimum frame count is invalid")
    selected = selected.reset_index(drop=True)
    _factor_contracts(config, selected)
    return config, source, selected


def _collector_command(
    config: Mapping[str, object],
    source: Mapping[str, object],
    row: Mapping[str, object],
    role: str,
    trajectory_dir: Path,
) -> list[str]:
    role_config = config["staging_roles"][role]
    capture = config["capture"]
    ports = capture["ports"][role]
    dropped = {
        "--sync-world", "--async-world", "--external-sync-ticker",
        "--sensor-platform", "--ego-spawn-index", "--ego-role-name", "--max-frames",
        "--ego-spawn-forward-offset-m", "--ego-spawn-right-offset-m",
        "--ego-spawn-z-offset-m", "--ego-spawn-yaw-offset-deg",
        "--ego-freeze", "--no-ego-freeze", "--ego-fixed-path-spawn-indices",
        "--ego-fixed-path-progress-csv", "--ego-fixed-path-loop",
        "--no-ego-fixed-path-loop", "--transport-label", "--fps", "--world-tick-hz",
        "--camera-width", "--camera-height", "--camera-fov",
        "--radar-points-per-second", "--radar-raster-radius-px",
        "--radar-temporal-window-frames", "--npc-vehicles", "--npc-pedestrians",
        "--camera-source-port", "--remote-port", "--remote-source-port",
        "--camera-result-port", "--run-id", "--run-group", "--metrics-run-dir",
        "--enable-run-logging", "--disable-run-logging", "--ego-route-control",
        "--front-device", "--back-device",
        "--phase2-tracker-association-gate-m",
        "--phase2-tracker-maximum-missed-frames",
        "--phase2-retention-tier",
        "--seed",
    }
    inherited = _drop_options(source["common_args"], dropped)
    geometry_id = str(row["geometry_or_route_id"])
    retention_window = _retention_window_for_row(config, row)
    offset_s = float(retention_window["start_offset_s"])
    coordination = trajectory_dir / "coordination"
    fixed_tracker = config["verification"]["replay_grid"][
        "fixed_source_contract"
    ]["source_local_tracker"]
    inherited.extend(
        [
            "--async-world", "--external-sync-ticker", "--sensor-platform", "ego_vehicle",
            "--ego-spawn-index", str(role_config["ego_spawn_index"]),
            "--ego-spawn-require-exact",
            "--ego-spawn-forward-offset-m", str(role_config["ego_spawn_forward_offset_m"]),
            "--ego-spawn-right-offset-m", str(role_config["ego_spawn_right_offset_m"]),
            "--ego-spawn-z-offset-m", str(role_config["ego_spawn_z_offset_m"]),
            "--ego-spawn-yaw-offset-deg", str(role_config["ego_spawn_yaw_offset_deg"]),
            "--ego-freeze", "--ego-role-name", str(role_config["ego_role_name"]),
            "--ego-route-control", "traffic_manager",
            "--front-device", str(capture["compute_assignment"]["front_device_by_role"][role]),
            "--back-device", str(capture["compute_assignment"]["back_device_by_role"][role]),
            "--npc-vehicles", "0", "--npc-pedestrians", "0",
            "--fps", "10.0", "--world-tick-hz", "10.0",
            "--camera-width", "1280", "--camera-height", "720", "--camera-fov", "120.0",
            "--radar-points-per-second", "200000", "--radar-raster-radius-px", "4",
            "--radar-temporal-window-frames", "2",
            "--max-frames", str(config["clock"]["frames_per_trajectory"]),
            "--seed", str(int(row["sensor_seed"])),
            "--transport-label", f"phase2_audit_{role}_loopback",
            "--camera-source-port", str(ports["camera_source"]),
            "--remote-port", str(ports["remote"]),
            "--remote-source-port", str(ports["remote_source"]),
            "--camera-result-port", str(ports["camera_result"]),
            "--run-id", f"{row['trajectory_id']}_{role}",
            "--run-group", str(row["group_id"]),
            "--metrics-run-dir", str(trajectory_dir / role),
            "--enable-run-logging", "--headless",
            "--phase2-role", role,
            "--phase2-trajectory-id", str(row["trajectory_id"]),
            "--phase2-scenario-role", str(row["scenario_role"]),
            "--phase2-contract-config", str(_repo_path(config["causal_contract_config"])),
            "--phase2-retention-config", str(_repo_path(config["retention_config"])),
            "--phase2-retention-start-offset-s", str(offset_s),
            "--phase2-retention-frame-count", str(capture["retained_frames_per_role"]),
            "--phase2-retention-tier", str(row["raw_retention_tier"]),
            "--phase2-tracker-association-gate-m",
            str(fixed_tracker["association_gate_m"]),
            "--phase2-tracker-maximum-missed-frames",
            str(fixed_tracker["maximum_missed_frames"]),
            "--phase2-geometry-id", geometry_id,
            "--phase2-motion-owner", "external_orchestrator",
            "--phase2-ready-sentinel", str(coordination / f"{role}.ready.json"),
            "--phase2-capture-start-sentinel", str(coordination / "capture.start.json"),
            "--phase2-tick-ready", str(coordination / f"{role}.tick_ready.json"),
            "--phase2-heartbeat", str(coordination / f"{role}.heartbeat.json"),
            "--phase2-start-timeout-s", str(config["clock"]["startup_timeout_s"]),
        ]
    )
    deduped: list[str] = []
    for token in inherited:
        if token in {"--headless", "--enable-run-logging", "--sensor-every-tick"}:
            if token in deduped:
                continue
        deduped.append(token)
    _require_inherited_contract(deduped)
    return [sys.executable, "-m", "data_collection.phase2_paired_causal_collector", *deduped]


def _population_command(
    config: Mapping[str, object],
    row: Mapping[str, object],
    scenario: object,
    coordination_dir: Path,
) -> list[str]:
    traffic = config["ambient_traffic"]
    _layer_id, layer, counts = _ambient_counts(config, row)
    if not bool(layer["population_process_required"]):
        raise ValueError(
            "generic population command is forbidden for scenario-owned-only rows"
        )
    density = str(row["traffic_density"])
    vehicle_clearance = float(
        traffic["vehicle_spawn_clearance_m_by_density"][density]
    )
    protected_clearance = float(
        traffic["protected_location_clearance_m_by_density"][density]
    )
    same_route_spacing = float(
        traffic["route_derived_same_route_spacing_m_by_density"][density]
    )
    cross_route_clearance = float(
        traffic["route_derived_cross_route_clearance_m_by_density"][density]
    )
    registered_pose_gate = config["verification"][
        "matched_pair_initial_realization_gate"
    ]
    protected = []
    for location in scenario.protected_locations:
        protected.extend(
            ["--protected-location", str(location.x), str(location.y)]
        )
    route_args = [
        token
        for path in scenario.ambient_route_paths
        for token in ("--route-progress-csv", str(path))
    ]
    fallback_args = (
        [
            "--route-derived-spawn-fallback",
            "--route-derived-spawn-spacing-m",
            str(same_route_spacing),
            "--route-derived-cross-route-clearance-m",
            str(cross_route_clearance),
        ]
        if bool(layer["route_derived_spawn_fallback"])
        else []
    )
    return [
        sys.executable,
        "-u",
        str(_repo_path(traffic["entrypoint"])),
        "--host", str(config["carla"]["host"]),
        "--port", str(config["carla"]["port"]),
        "--tm-port", str(config["clock"]["tm_port"]),
        "--number-of-vehicles", str(counts["vehicles"]),
        "--number-of-walkers", str(counts["walkers"]),
        "--seed", str(int(row["traffic_seed"])),
        "--seedw", str(int(row["traffic_seed"]) + int(traffic["walker_seed_offset"])),
        "--replenish-interval", str(traffic["replenish_interval_s"]),
        "--population-log-interval", str(traffic["population_log_interval_s"]),
        "--safe",
        "--vehicle-spawn-clearance-m", str(vehicle_clearance),
        *route_args,
        *protected,
        "--protected-clearance-m", str(protected_clearance),
        "--minimum-route-offset-m", str(layer["minimum_route_offset_m"]),
        "--maximum-route-offset-m", str(layer["maximum_route_offset_m"]),
        "--maximum-route-heading-error-deg", str(layer["maximum_route_heading_error_deg"]),
        "--minimum-filtered-spawn-points", str(int(counts["vehicles"])),
        *fallback_args,
        "--traffic-leading-distance-m", str(traffic["tm_distance_to_leading_vehicle_m"]),
        "--traffic-speed-difference-pct", str(traffic["tm_speed_difference_pct"]),
        "--traffic-desired-speed-mps", str(traffic["tm_desired_speed_mps"]),
        "--registered-spawn-maximum-horizontal-error-m",
        str(registered_pose_gate["maximum_horizontal_error_m"]),
        "--registered-spawn-maximum-yaw-error-deg",
        str(registered_pose_gate["maximum_yaw_error_deg"]),
        "--released-vehicle-motion-mode",
        str(layer["released_vehicle_motion_mode"]),
        "--released-walker-motion-mode",
        str(layer["released_walker_motion_mode"]),
        "--defer-vehicle-control-to-runner",
        "--population-ready-manifest", str(coordination_dir / "population.ready.json"),
        "--population-release-sentinel", str(coordination_dir / "population.release.json"),
        "--population-released-manifest", str(coordination_dir / "population.released.json"),
    ]


def build_plan(
    config: Mapping[str, object],
    source: Mapping[str, object],
    selected: pd.DataFrame,
    output_dir: Path,
) -> dict:
    trajectories = []
    factor_contracts = _factor_contracts(config, selected)
    for row in selected.to_dict("records"):
        trajectory_dir = output_dir / str(row["trajectory_id"])
        layer_id, layer, counts = _ambient_counts(config, row)
        planned = {
                **{key: (None if pd.isna(value) else value) for key, value in row.items()},
                "trajectory_dir": str(trajectory_dir),
                "ambient_evidence_layer": layer_id,
                "ambient_evidence_role": str(layer["evidence_role"]),
                "ambient_counts": dict(counts),
                "ambient_vehicle_motion_mode": str(
                    layer["released_vehicle_motion_mode"]
                ),
                "ambient_walker_motion_mode": str(
                    layer["released_walker_motion_mode"]
                ),
                "retention_window": _retention_window_for_row(config, row),
                "retention_start_offset_s": float(
                    _retention_window_for_row(config, row)["start_offset_s"]
                ),
                "collector_commands": {
                    role: _collector_command(config, source, row, role, trajectory_dir)
                    for role in ROLE_NAMES
                },
                "population_command_status": (
                    "resolved_after_frozen_geometry_is_loaded"
                    if bool(layer["population_process_required"])
                    else "not_launched_scenario_owned_only"
                ),
            }
        factor_contract = factor_contracts.get(str(row["trajectory_id"]))
        if factor_contract is not None:
            planned["factor_runtime_contract"] = {
                "trajectory_row_sha256": factor_contract.trajectory_row_sha256,
                "requested_factors": dict(factor_contract.requested),
                "authored_onset_policy_visibility": (
                    "forbidden_evaluation_metadata_only"
                ),
            }
        trajectories.append(planned)
    return {
        "schema": "scenesense.phase2_calibration_audit_plan.v1",
        "stage_id": config["stage_id"],
        "trajectory_count": len(trajectories),
        "group_count": len({item["group_id"] for item in trajectories}),
        "single_sync_ticker": True,
        "oai_launched": False,
        "next_stage_chained": False,
        "factor_realization_runtime_enabled": bool(factor_contracts),
        "estimated_minutes": round(
            len(trajectories) * 2.9, 1
        ),
        "estimated_heavy_bytes": int(config["storage"]["estimated_heavy_bytes"]),
        "trajectories": trajectories,
    }


def _select_trajectory_ids(
    selected: pd.DataFrame, trajectory_ids: Sequence[str]
) -> pd.DataFrame:
    """Return an ordered, fail-closed subset for bounded live regression."""

    requested = list(dict.fromkeys(str(value) for value in trajectory_ids))
    if not requested:
        return selected.copy()
    available = set(selected["trajectory_id"].astype(str))
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"unknown calibration trajectory IDs: {missing}")
    order = {trajectory_id: index for index, trajectory_id in enumerate(requested)}
    subset = selected[
        selected["trajectory_id"].astype(str).isin(requested)
    ].copy()
    subset["_selection_order"] = subset["trajectory_id"].astype(str).map(order)
    return subset.sort_values("_selection_order").drop(columns="_selection_order")


def _audit_record(payload: Mapping[str, object]) -> CausalDecisionAudit:
    if payload.get("schema") != CAUSAL_AUDIT_SCHEMA:
        raise ValueError("causal audit schema mismatch")
    decision_payload = payload["decision"]
    audit = CausalDecisionAudit(
        decision=DecisionRecord(**decision_payload),
        fields=tuple(CausalField(**item) for item in payload["fields"]),
    )
    canonical_payload = audit.to_dict()
    canonical = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if str(payload.get("record_sha256")) != digest:
        raise ValueError("causal audit record hash mismatch")
    return audit


def _verify_artifact_manifest(role_dir: Path) -> int:
    path = role_dir / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for item in payload["files"]:
        candidate = role_dir / str(item["path"])
        if not candidate.is_file():
            raise FileNotFoundError(f"manifest artifact missing: {candidate}")
        if candidate.stat().st_size != int(item["bytes"]) or _sha256(candidate) != str(
            item["sha256"]
        ):
            raise ValueError(f"manifest artifact drifted: {candidate}")
        count += 1
    return count


def _single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {path}, found {len(matches)}")
    return matches[0]


def _role_metrics_csv(role_dir: Path) -> Path:
    """Resolve the production logger's fixed ``role/streams`` layout."""

    return _single(role_dir / "streams", "*_metrics.csv")


def verify_trajectory(
    config: Mapping[str, object], row: Mapping[str, object], trajectory_dir: Path
) -> dict:
    expected_frames = int(config["verification"]["required_retained_frame_pairs_per_role"])
    expected_decisions = int(config["verification"]["required_causal_decisions_per_role"])
    local_fields = set(config["verification"]["local_loopback_fields"])
    retention_tier = str(row["raw_retention_tier"])
    if retention_tier not in {"inputs_only_window", "inputs_plus_logits_window"}:
        raise ValueError(f"unsupported manifest retention tier: {retention_tier}")
    expected_logits = (
        expected_frames if retention_tier == "inputs_plus_logits_window" else 0
    )
    by_role = {}
    for role in ROLE_NAMES:
        role_dir = trajectory_dir / role
        runtime = json.loads(
            (role_dir / "phase2_runtime_summary.json").read_text(encoding="utf-8")
        )
        if runtime.get("status") != "complete" or runtime.get("quota_stop_reason") is not None:
            raise ValueError(f"{role} runtime did not complete cleanly: {runtime}")
        if str(runtime.get("retention_tier")) != retention_tier:
            raise ValueError(f"{role} runtime retention tier differs from manifest")
        if int(runtime["raw_input_files_written"]) != expected_frames:
            raise ValueError(f"{role} did not retain exactly {expected_frames} inputs")
        if int(runtime["logits_files_written"]) != expected_logits:
            raise ValueError(
                f"{role} retained {runtime['logits_files_written']} logits, "
                f"expected {expected_logits} for {retention_tier}"
            )
        raw_dir = role_dir / "retained_inputs"
        inputs = {
            path.name.removesuffix("_inputs.npz")
            for path in raw_dir.glob("frame_*_inputs.npz")
        }
        logits = {
            path.name.removesuffix("_logits.npz")
            for path in raw_dir.glob("frame_*_logits.npz")
        }
        if len(inputs) != expected_frames:
            raise ValueError(f"{role} retained input frame set is incomplete")
        if retention_tier == "inputs_plus_logits_window" and inputs != logits:
            raise ValueError(f"{role} retained input/logit frame sets differ")
        if retention_tier == "inputs_only_window" and logits:
            raise ValueError(f"{role} inputs-only tier contains retained logits")
        audits = []
        for line in (role_dir / "runtime/causal_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line:
                audits.append(_audit_record(json.loads(line)))
        if len(audits) != expected_decisions:
            raise ValueError(
                f"{role} causal decision count is {len(audits)}, expected {expected_decisions}"
            )
        metrics = pd.read_csv(_role_metrics_csv(role_dir))
        missing_local = local_fields - set(metrics.columns)
        if missing_local:
            raise ValueError(f"{role} metrics lack local-loopback fields: {sorted(missing_local)}")
        if len(metrics) != int(config["clock"]["frames_per_trajectory"]):
            raise ValueError(f"{role} metrics do not cover all capture frames")
        if metrics[list(local_fields)].isna().any().any():
            raise ValueError(f"{role} local-loopback timing/byte fields contain missing values")
        radar_gate = config["verification"]["radar_density_gate"]
        radar = pd.to_numeric(
            metrics["radar_projected_points"], errors="coerce"
        ).dropna()
        if len(radar) < int(radar_gate["minimum_metric_frames"]):
            raise ValueError(f"{role} lacks enough radar-density frames")
        radar_median = float(radar.median())
        radar_reference = float(radar_gate["reference_projected_points_median"])
        radar_relative_error = abs(radar_median - radar_reference) / radar_reference
        if radar_relative_error > float(radar_gate["relative_tolerance"]):
            raise ValueError(
                f"{role} radar density is off contract: median={radar_median:.1f}, "
                f"reference={radar_reference:.1f}"
            )
        gt = pd.read_csv(_single(role_dir / "evaluation_truth", "*_object_ground_truth.csv"))
        if not set(int(value.split("_")[1]) for value in inputs).issubset(
            set(pd.to_numeric(gt["frame_id"]).astype(int))
        ):
            raise ValueError(f"{role} retained frames are not recoverable in truth stream")
        by_role[role] = {
            "retained_frame_pairs": (
                len(inputs)
                if retention_tier == "inputs_plus_logits_window"
                else 0
            ),
            "retained_input_frames": len(inputs),
            "retention_tier": retention_tier,
            "retained_logit_frames": len(logits),
            "causal_decisions": len(audits),
            "metric_frames": len(metrics),
            "radar_projected_points_median": radar_median,
            "radar_density_relative_error": radar_relative_error,
            "truth_rows": len(gt),
            "artifact_manifest_files": _verify_artifact_manifest(role_dir),
            "heavy_bytes": sum(path.stat().st_size for path in raw_dir.glob("*.npz")),
        }
    combinations = _validate_replay_grid(config["verification"]["replay_grid"])
    return {
        "pass": True,
        "trajectory_id": str(row["trajectory_id"]),
        "roles": by_role,
        "replay_combinations_supported": combinations,
        "oai_field_status": config["verification"]["oai_fields"]["status"],
    }


def _traffic_monitor_integration(
    config: Mapping[str, object], scenario: object
) -> dict:
    traffic = config["ambient_traffic"]
    scenario_role = (
        str(scenario.scenario_role)
        if hasattr(scenario, "scenario_role")
        else "controlled_positive_occlusion"
    )
    layer_id, layer = _ambient_layer(config, scenario_role)
    return {
        "traffic_sanity_gate": dict(traffic["traffic_sanity_gate"]),
        "ambient_evidence_layer": layer_id,
        "ambient_evidence_role": str(layer["evidence_role"]),
        "expected_stationary_context": (
            str(layer["npc_route_mode"]) == "stationary_context"
        ),
        "external_sync_tick_owner": True,
        "traffic_expected_frame_count": int(config["clock"]["frames_per_trajectory"]),
        "tm_port": int(config["clock"]["tm_port"]),
        "tm_distance_to_leading_vehicle_m": float(traffic["tm_distance_to_leading_vehicle_m"]),
        "tm_speed_difference_pct": float(traffic["tm_speed_difference_pct"]),
        "tm_desired_speed_mps": float(traffic["tm_desired_speed_mps"]),
        "npc_route_mode": str(layer["npc_route_mode"]),
        "npc_direct_route_speed_mps": float(
            traffic["npc_direct_route_speed_mps"]
        ),
        "npc_trace_replay_speed_mps": float(
            traffic["npc_trace_replay_speed_mps"]
        ),
        "npc_trace_replay_fixed_delta_seconds": float(
            traffic["npc_trace_replay_fixed_delta_seconds"]
        ),
        "npc_trace_replay_step_m": float(traffic["npc_trace_replay_step_m"]),
        "npc_trace_replay_horizon_m": float(
            traffic["npc_trace_replay_horizon_m"]
        ),
        "npc_trace_replay_minimum_clearance_m": float(
            traffic["npc_trace_replay_minimum_clearance_m"]
        ),
        "ambient_walker_motion_mode": str(layer["released_walker_motion_mode"]),
        "npc_loop_route_progress_csvs": [
            str(path) for path in getattr(
                scenario, "ambient_motion_route_paths", scenario.ambient_route_paths
            )
        ],
        "npc_loop_repetitions": int(traffic["npc_loop_repetitions"]),
    }


def _add_collision_sensors(
    monitor: advisor.TrafficSanityMonitor,
    world: object,
    actors: Sequence[object],
) -> None:
    blueprint = world.get_blueprint_library().find("sensor.other.collision")
    existing = set(monitor.actor_ids)
    for actor in actors:
        actor_id = int(actor.id)
        if actor_id in existing:
            continue
        monitor.actor_metadata[actor_id] = {
            "role_name": str(actor.attributes.get("role_name", "")),
            "type_id": str(actor.type_id),
            "monitoring_scope": "owned_scenario_actor",
        }
        sensor = world.spawn_actor(blueprint, advisor.carla.Transform(), attach_to=actor)
        sensor.listen(
            lambda event, owned_id=actor_id: monitor._on_collision(owned_id, event)
        )
        monitor.collision_sensors.append(sensor)


def _ambient_initial_signature(world: object) -> list[dict]:
    """Capture an ID-free live-pose diagnostic while the population is held."""

    rows = []
    for pattern in ("vehicle.*", "walker.pedestrian.*"):
        for actor in world.get_actors().filter(pattern):
            role_name = str(actor.attributes.get("role_name", ""))
            if role_name.startswith("phase2_") or role_name.startswith("scenesense_"):
                continue
            if pattern == "vehicle.*" and not role_name.startswith("autopilot"):
                continue
            transform = actor.get_transform()
            rows.append(
                {
                    "type_id": str(actor.type_id),
                    "role_name": role_name,
                    "x": round(float(transform.location.x), 4),
                    "y": round(float(transform.location.y), 4),
                    "z": round(float(transform.location.z), 4),
                    "yaw_deg": round(float(transform.rotation.yaw), 3),
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item["type_id"], item["role_name"], item["x"], item["y"], item["z"]
        ),
    )


def _scenario_owned_nontreatment_signature(
    world: object, config: Mapping[str, object]
) -> list[dict]:
    """ID-free initial-state evidence shared by a designed causal pair.

    Registered hazard actors are the treatment and are intentionally omitted.
    The helper, recipient, and scenario occluder must still realize the same
    initial geometry across positive and benign twins.
    """

    ego_roles = {
        str(value["ego_role_name"])
        for value in config["staging_roles"].values()
    }
    rows = []
    for actor in world.get_actors().filter("vehicle.*"):
        role_name = str(actor.attributes.get("role_name", ""))
        if role_name not in ego_roles and not (
            role_name.startswith("phase2_") and role_name.endswith("_occluder")
        ):
            continue
        transform = actor.get_transform()
        rows.append(
            {
                "type_id": str(actor.type_id),
                "role_name": role_name,
                "x": round(float(transform.location.x), 4),
                "y": round(float(transform.location.y), 4),
                "yaw_deg": round(float(transform.rotation.yaw), 3),
                "motion_mode": "frozen_scenario_contract",
                "motion_speed_mps": None,
                "motion_target_x": None,
                "motion_target_y": None,
                "motion_target_z": None,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["type_id"], item["role_name"], item["x"], item["y"]
        ),
    )


def _validate_population_ready_manifest(
    payload: Mapping[str, object], *, vehicles: int, walkers: int
) -> list[dict]:
    if payload.get("schema") != POPULATION_READY_SCHEMA:
        raise RuntimeError("ambient population READY manifest schema mismatch")
    if payload.get("status") != "held_ready":
        raise RuntimeError("ambient population was not held at READY")
    expected = {
        "vehicle_count": int(vehicles),
        "walker_count": int(walkers),
        "walker_controller_count": int(walkers),
    }
    for field, value in expected.items():
        if int(payload.get(field, -1)) != value:
            raise RuntimeError(
                f"ambient population READY {field} mismatch: "
                f"observed={payload.get(field)} expected={value}"
            )
    spawn_contract = payload.get("vehicle_spawn_contract")
    if not isinstance(spawn_contract, Mapping):
        raise RuntimeError("ambient vehicle spawn contract is missing")
    if spawn_contract.get("all_outside_junctions") is not True:
        raise RuntimeError("ambient vehicle spawn contract admits a junction spawn")
    if int(spawn_contract.get("verified_vehicle_count", -1)) != int(vehicles):
        raise RuntimeError("ambient vehicle spawn contract count mismatch")
    if payload.get("spawn_signature_basis") != (
        "id_free_held_type_role_pose_and_motion_before_any_ambient_motion"
    ):
        raise RuntimeError("ambient population spawn-signature basis mismatch")
    signature = payload.get("spawn_signature")
    if not isinstance(signature, list) or len(signature) != vehicles + walkers:
        raise RuntimeError("ambient population spawn signature is incomplete")
    required = {
        "type_id",
        "role_name",
        "x",
        "y",
        "yaw_deg",
        "motion_mode",
        "motion_speed_mps",
        "motion_target_x",
        "motion_target_y",
        "motion_target_z",
    }
    if any(not isinstance(row, Mapping) or set(row) != required for row in signature):
        raise RuntimeError("ambient population spawn-signature row schema mismatch")
    return [dict(row) for row in signature]


def _wait_for_population_release_ack(
    process: subprocess.Popen, path: Path, timeout_s: float
) -> dict:
    """Wait in wall time only; advancing CARLA here would de-pair the arms."""

    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "ambient population process exited before RELEASE acknowledgement: "
                f"returncode={return_code}"
            )
        if path.is_file():
            payload = _read_json(path)
            if payload.get("schema") != POPULATION_RELEASED_SCHEMA:
                raise RuntimeError("ambient population RELEASED manifest schema mismatch")
            if payload.get("status") != "released":
                raise RuntimeError("ambient population RELEASE was not acknowledged")
            return payload
        time.sleep(0.01)
    raise RuntimeError("timed out waiting for ambient population RELEASE acknowledgement")


def _require_population_process_alive(
    process: subprocess.Popen, *, phase: str
) -> None:
    """Fail before a departed population can masquerade as stopped traffic."""

    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            "ambient population process exited unexpectedly "
            f"during {phase}: returncode={return_code}"
        )


def _wrapped_angle_error_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _compare_ambient_initial_signatures(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict:
    """Compare immutable held spawn provenance for a designed matched pair.

    Vertical physics settling is deliberately absent from this contract. The
    handshake freezes actors before ambient motion and compares the causal spawn
    choices: type/role identity, horizontal pose, and yaw.
    """

    horizontal_limit = float(gate["maximum_horizontal_error_m"])
    yaw_limit = float(gate["maximum_yaw_error_deg"])

    def grouped(
        rows: Sequence[Mapping[str, object]],
    ) -> Dict[tuple[str, str], list[Mapping[str, object]]]:
        result: Dict[tuple[str, str], list[Mapping[str, object]]] = {}
        for row in rows:
            key = (str(row["type_id"]), str(row["role_name"]))
            result.setdefault(key, []).append(row)
        for values in result.values():
            values.sort(
                key=lambda item: (
                    float(item["x"]),
                    float(item["y"]),
                    float(item["yaw_deg"]),
                )
            )
        return result

    left_groups = grouped(left)
    right_groups = grouped(right)
    identity_counts_match = {
        key: len(values) for key, values in left_groups.items()
    } == {key: len(values) for key, values in right_groups.items()}
    failures = []
    comparisons = []
    if not identity_counts_match:
        failures.append("type_role_multiset_mismatch")
    else:
        for key in sorted(left_groups):
            for ordinal, (left_row, right_row) in enumerate(
                zip(left_groups[key], right_groups[key])
            ):
                dx = float(left_row["x"]) - float(right_row["x"])
                dy = float(left_row["y"]) - float(right_row["y"])
                horizontal_error = math.hypot(dx, dy)
                yaw_error = _wrapped_angle_error_deg(
                    float(left_row["yaw_deg"]), float(right_row["yaw_deg"])
                )
                motion_fields = (
                    "motion_mode",
                    "motion_speed_mps",
                    "motion_target_x",
                    "motion_target_y",
                    "motion_target_z",
                )
                motion_contract_match = all(
                    left_row.get(field) == right_row.get(field)
                    for field in motion_fields
                )
                comparison = {
                    "type_id": key[0],
                    "role_name": key[1],
                    "ordinal": ordinal,
                    "horizontal_error_m": horizontal_error,
                    "yaw_error_deg": yaw_error,
                    "motion_contract_match": motion_contract_match,
                }
                comparisons.append(comparison)
                if horizontal_error > horizontal_limit:
                    failures.append("horizontal_pose_drift")
                if yaw_error > yaw_limit:
                    failures.append("yaw_drift")
                if not motion_contract_match:
                    failures.append("motion_contract_drift")

    return {
        "schema": "scenesense.phase2_matched_spawn_provenance_gate.v1",
        "pass": not failures,
        "basis": (
            "same_id_free_held_type_role_multiset_bounded_spawn_xy_yaw_and_"
            "exact_future_motion_contract_before_any_ambient_motion"
        ),
        "left_actor_count": len(left),
        "right_actor_count": len(right),
        "identity_counts_match": identity_counts_match,
        "maximum_observed_horizontal_error_m": max(
            (item["horizontal_error_m"] for item in comparisons), default=None
        ),
        "maximum_observed_yaw_error_deg": max(
            (item["yaw_error_deg"] for item in comparisons), default=None
        ),
        "limits": {
            "maximum_horizontal_error_m": horizontal_limit,
            "maximum_yaw_error_deg": yaw_limit,
        },
        "failures": sorted(set(failures)),
        "comparisons": comparisons,
    }


def _compare_static_environment_records(
    left: object,
    right: object,
) -> Optional[dict]:
    """Compare only capture-time-independent static semantics for a pair."""

    records = [left, right]
    if all(value is None for value in records):
        return None
    failures = []
    if any(not isinstance(value, Mapping) for value in records):
        failures.append("missing_static_environment_record")
    else:
        semantic_hashes = {
            str(value.get("static_geometry_semantic_sha256", ""))
            for value in records
        }
        if any(len(value) != 64 for value in semantic_hashes):
            failures.append("invalid_static_geometry_semantic_hash")
        if len(semantic_hashes) != 1:
            failures.append("static_geometry_semantic_drift")
        if any(
            value.get("status") != "complete"
            or value.get("selection_contract")
            != STATIC_ENVIRONMENT_SELECTION_CONTRACT
            or value.get("static_geometry_semantic_hash_basis")
            != STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS
            for value in records
        ):
            failures.append("static_environment_contract_drift")
    return {
        "schema": "scenesense.phase2_matched_static_environment_gate.v1",
        "pass": not failures,
        "basis": STATIC_ENVIRONMENT_SEMANTIC_HASH_BASIS,
        "left_static_geometry_semantic_sha256": (
            left.get("static_geometry_semantic_sha256")
            if isinstance(left, Mapping)
            else None
        ),
        "right_static_geometry_semantic_sha256": (
            right.get("static_geometry_semantic_sha256")
            if isinstance(right, Mapping)
            else None
        ),
        "failures": sorted(set(failures)),
    }


def _require_completed_pair_match(
    batch: Mapping[str, object],
    group_id: str,
    initial_gate: Mapping[str, object],
    trajectory_gate: Mapping[str, object],
) -> Optional[dict]:
    rows = [
        row
        for row in batch["trajectories"]
        if str(row["group_id"]) == str(group_id)
        and str(row["scenario_role"]) != "naturalistic_operation"
    ]
    if len(rows) < 2:
        return None
    if len(rows) != 2 or any(row.get("status") != "complete" for row in rows):
        return None
    population_modes = {
        str(row.get("ambient_population_mode", "external_population"))
        for row in rows
    }
    scenario_owned_only = population_modes == {"scenario_owned_only"}
    if "scenario_owned_only" in population_modes and not scenario_owned_only:
        raise RuntimeError(
            f"matched pair disagrees on ambient population mode for {group_id}"
        )
    static_result = _compare_static_environment_records(
        rows[0].get("static_environment_truth"),
        rows[1].get("static_environment_truth"),
    )
    if static_result is not None:
        for row in rows:
            row["matched_pair_static_environment_gate"] = static_result
        if not static_result["pass"]:
            raise RuntimeError(
                "matched positive/benign static environment drifted for "
                f"{group_id}: {static_result['failures']}"
            )
    result = _compare_ambient_initial_signatures(
        rows[0].get("ambient_spawn_signature", []),
        rows[1].get("ambient_spawn_signature", []),
        initial_gate,
    )
    for row in rows:
        row["matched_pair_initial_realization_gate"] = result
    if not result["pass"]:
        raise RuntimeError(
            "matched positive/benign ambient realization drifted for "
            f"{group_id}: {result['failures']}"
        )
    if scenario_owned_only and any(
        row.get("ambient_spawn_signature", []) for row in rows
    ):
        raise RuntimeError(
            f"scenario-owned-only pair contains generic ambient actors for {group_id}"
        )
    owned_result = _compare_ambient_initial_signatures(
        rows[0].get("scenario_owned_nontreatment_signature", []),
        rows[1].get("scenario_owned_nontreatment_signature", []),
        initial_gate,
    )
    if not rows[0].get("scenario_owned_nontreatment_signature") or not rows[1].get(
        "scenario_owned_nontreatment_signature"
    ):
        owned_result["pass"] = False
        owned_result.setdefault("failures", []).append(
            "missing_scenario_owned_nontreatment_signature"
        )
    for row in rows:
        row["matched_pair_owned_nontreatment_gate"] = owned_result
    if not owned_result["pass"]:
        raise RuntimeError(
            "matched positive/benign scenario-owned geometry drifted for "
            f"{group_id}: {owned_result['failures']}"
        )
    factor_hashes = [row.get("nontreatment_plan_sha256") for row in rows]
    if any(value is not None for value in factor_hashes):
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in factor_hashes
        ):
            raise RuntimeError(
                f"factor pair lacks a complete non-treatment plan hash for {group_id}"
            )
        if len(set(factor_hashes)) != 1:
            raise RuntimeError(
                f"factor pair non-treatment plan differs for {group_id}"
            )
    trajectory_result = _compare_ambient_trajectories(
        Path(str(rows[0]["traffic_sanity"]["ambient_actor_trajectory_csv"])),
        Path(str(rows[1]["traffic_sanity"]["ambient_actor_trajectory_csv"])),
        trajectory_gate,
        allow_declared_both_empty=scenario_owned_only,
    )
    for row in rows:
        row["matched_pair_full_trajectory_gate"] = trajectory_result
    if not trajectory_result["pass"]:
        raise RuntimeError(
            "matched positive/benign ambient trajectories drifted for "
            f"{group_id}: {trajectory_result['failures']}"
        )
    result_record = {
        "initial_realization": result,
        "owned_nontreatment_realization": owned_result,
        "full_trajectory": trajectory_result,
    }
    if all(isinstance(value, str) for value in factor_hashes):
        result_record["nontreatment_plan_sha256"] = factor_hashes[0]
    if static_result is not None:
        result_record["static_environment"] = static_result
    return result_record


def _persist_completed_factor_pair_postflights(
    *,
    batch: MutableMapping[str, object],
    group_id: str,
    output_dir: Path,
    factor_smoke_config: Mapping[str, object],
    factor_smoke_plan: Mapping[str, object],
) -> int:
    """Fail-fast replay both rows only after their matched-pair gates exist."""

    records = [
        row
        for row in batch["trajectories"]
        if str(row.get("group_id")) == str(group_id)
        and str(row.get("scenario_role")) != "naturalistic_operation"
    ]
    if len(records) < 2:
        return 0
    if len(records) != 2 or any(row.get("status") != "complete" for row in records):
        return 0
    gate_names = (
        "matched_pair_initial_realization_gate",
        "matched_pair_owned_nontreatment_gate",
        "matched_pair_static_environment_gate",
        "matched_pair_full_trajectory_gate",
    )
    for row in records:
        for name in gate_names:
            gate = row.get(name)
            if not isinstance(gate, Mapping) or gate.get("pass") is not True:
                raise RuntimeError(
                    f"factor pair lacks passed {name}: {row.get('trajectory_id')}"
                )
    plan_rows = {
        str(row["trajectory_id"]): row for row in factor_smoke_plan["rows"]
    }
    from phase2_map_sharing.factor_smoke_postflight import (
        analyze_and_persist_trajectory_artifacts,
    )

    written = 0
    for record in records:
        trajectory_id = str(record["trajectory_id"])
        plan_row = plan_rows.get(trajectory_id)
        if plan_row is None:
            raise RuntimeError(f"factor plan row disappeared: {trajectory_id}")
        postflight_path = (
            Path(output_dir) / trajectory_id / "scenario/factor_smoke_postflight.json"
        )
        prior = record.get("factor_postflight_artifact")
        if prior is not None:
            if (
                not isinstance(prior, Mapping)
                or Path(str(prior.get("path", ""))).resolve()
                != postflight_path.resolve()
                or not postflight_path.is_file()
                or prior.get("sha256") != _sha256(postflight_path)
            ):
                raise RuntimeError(f"factor postflight record drifted: {trajectory_id}")
            continue
        if postflight_path.exists():
            raise RuntimeError(
                f"unregistered factor postflight already exists: {trajectory_id}"
            )
        trajectory_postflight = analyze_and_persist_trajectory_artifacts(
            trajectory_dir=Path(output_dir) / trajectory_id,
            trajectory_row=plan_row,
            smoke_config=factor_smoke_config,
        )
        record["factor_postflight_artifact"] = {
            "path": str(postflight_path),
            "sha256": _sha256(postflight_path),
            "postflight_sha256": trajectory_postflight["postflight_sha256"],
            "status": "complete_excluded_until_atomic_exact_16_pass",
        }
        written += 1
    return written


def _compare_ambient_trajectories(
    left_path: Path,
    right_path: Path,
    gate: Mapping[str, object],
    *,
    allow_declared_both_empty: bool = False,
) -> dict:
    """Compare the complete ID-free ambient future of a matched pair."""

    required_columns = {
        "replay_identity",
        "replay_plan_sha256",
        "replay_tick_index",
        "world_x",
        "world_y",
        "world_z",
        "speed_mps",
    }
    tables = []
    for path in (Path(left_path), Path(right_path)):
        table = pd.read_csv(path)
        missing = sorted(required_columns - set(table.columns))
        if missing:
            raise RuntimeError(
                f"ambient trajectory {path} lacks deterministic fields: {missing}"
            )
        if table[list(required_columns)].isna().any().any():
            raise RuntimeError(
                f"ambient trajectory {path} contains incomplete replay rows"
            )
        if table.duplicated(["replay_identity", "replay_tick_index"]).any():
            raise RuntimeError(
                f"ambient trajectory {path} has duplicate replay coordinates"
            )
        tables.append(table)
    left, right = tables
    required_frames = int(gate["required_frames_per_actor"])
    failures = []
    if left.empty or right.empty:
        both_empty = bool(left.empty and right.empty)
        if not (both_empty and allow_declared_both_empty):
            failures.append(
                "unexpected_empty_ambient_trajectory"
                if both_empty
                else "one_sided_empty_ambient_trajectory"
            )
        return {
            "schema": "scenesense.phase2_matched_ambient_trajectory_gate.v1",
            "pass": not failures,
            "basis": (
                "declared_scenario_owned_only_no_generic_ambient_actors"
                if both_empty and allow_declared_both_empty
                else "ambient_trajectory_presence_contract"
            ),
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_actor_count": (
                0 if left.empty else int(left["replay_identity"].nunique())
            ),
            "right_actor_count": (
                0 if right.empty else int(right["replay_identity"].nunique())
            ),
            "identity_match": both_empty,
            "paired_rows": 0,
            "maximum_horizontal_error_m": None,
            "maximum_vertical_error_m": None,
            "maximum_speed_error_mps": None,
            "limits": dict(gate),
            "failures": failures,
        }
    left_counts = left.groupby("replay_identity")["replay_tick_index"].nunique()
    right_counts = right.groupby("replay_identity")["replay_tick_index"].nunique()
    identity_match = set(left_counts.index) == set(right_counts.index)
    if not identity_match:
        failures.append("replay_identity_set_mismatch")
    if any(int(value) != required_frames for value in left_counts.values) or any(
        int(value) != required_frames for value in right_counts.values
    ):
        failures.append("incomplete_actor_trajectory")
    merged = left.merge(
        right,
        on=["replay_identity", "replay_tick_index"],
        suffixes=("_left", "_right"),
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        failures.append("replay_frame_key_mismatch")
    paired = merged[merged["_merge"] == "both"].copy()
    if len(paired):
        horizontal = np.hypot(
            pd.to_numeric(paired["world_x_left"])
            - pd.to_numeric(paired["world_x_right"]),
            pd.to_numeric(paired["world_y_left"])
            - pd.to_numeric(paired["world_y_right"]),
        )
        vertical = (
            pd.to_numeric(paired["world_z_left"])
            - pd.to_numeric(paired["world_z_right"])
        ).abs()
        speed = (
            pd.to_numeric(paired["speed_mps_left"])
            - pd.to_numeric(paired["speed_mps_right"])
        ).abs()
        plan_match = (
            paired["replay_plan_sha256_left"].astype(str)
            == paired["replay_plan_sha256_right"].astype(str)
        )
        maximum_horizontal = float(horizontal.max())
        maximum_vertical = float(vertical.max())
        maximum_speed = float(speed.max())
        if maximum_horizontal > float(gate["maximum_horizontal_error_m"]):
            failures.append("horizontal_trajectory_drift")
        if maximum_vertical > float(gate["maximum_vertical_error_m"]):
            failures.append("vertical_trajectory_drift")
        if maximum_speed > float(gate["maximum_speed_error_mps"]):
            failures.append("speed_trajectory_drift")
        if bool(gate["require_identical_replay_plan_sha256"]) and not bool(
            plan_match.all()
        ):
            failures.append("replay_plan_hash_mismatch")
    else:
        maximum_horizontal = None
        maximum_vertical = None
        maximum_speed = None
        failures.append("no_paired_trajectory_rows")
    return {
        "schema": "scenesense.phase2_matched_ambient_trajectory_gate.v1",
        "pass": not failures,
        "basis": (
            "same_id_free_replay_identity_plan_hash_and_full_frame_xy_z_speed"
        ),
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_actor_count": int(len(left_counts)),
        "right_actor_count": int(len(right_counts)),
        "identity_match": bool(identity_match),
        "paired_rows": int(len(paired)),
        "maximum_horizontal_error_m": maximum_horizontal,
        "maximum_vertical_error_m": maximum_vertical,
        "maximum_speed_error_mps": maximum_speed,
        "limits": dict(gate),
        "failures": sorted(set(failures)),
    }


def _stage_heavy_bytes(output_dir: Path) -> int:
    # The current runtime co-locates input and logit NPZ files under
    # ``retained_inputs``.  Count every NPZ there, while also supporting the
    # versioned/separate ``retained_logits`` layout without weakening the cap.
    return int(
        sum(
            path.stat().st_size
            for pattern in ("retained_inputs/*.npz", "retained_logits/*.npz")
            for path in output_dir.rglob(pattern)
        )
    )


def run_live(
    config: Mapping[str, object],
    source: Mapping[str, object],
    selected: pd.DataFrame,
    plan: Mapping[str, object],
    output_dir: Path,
) -> None:
    storage = config["storage"]
    storage_probe = _repo_path(config["output_root"]).parent
    free_bytes = shutil.disk_usage(storage_probe).free
    if free_bytes < int(storage["preflight_required_free_bytes"]):
        raise RuntimeError(
            "insufficient free space for audit reservation: "
            f"free={free_bytes}, required={storage['preflight_required_free_bytes']}"
        )
    _require_udp_ports_available({"capture": config["capture"]})
    factor_contracts = _factor_contracts(config, selected)
    factor_smoke_bundle = _factor_runtime_bundle(config) if factor_contracts else None
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.jsonl"
    _write_json_create(output_dir / "plan.json", plan)
    with (output_dir / "resolved_config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=False)
    batch: MutableMapping[str, object] = {
        "schema": "scenesense.phase2_calibration_audit_batch.v1",
        "stage_id": config["stage_id"],
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer_quality_operator_declared": config["carla"]["renderer_quality_level"],
        "oai_launched": False,
        "next_stage_chained": False,
        "storage_preflight": {
            "free_bytes": int(free_bytes),
            "stage_hard_cap_bytes": int(storage["stage_hard_cap_bytes"]),
            "required_free_floor_bytes": int(storage["required_free_floor_bytes"]),
        },
        "trajectories": [],
    }
    batch_path = output_dir / "batch_manifest.json"
    _write_json_create(batch_path, batch)
    _append_progress(progress_path, "stage_started", trajectory_count=len(selected))
    client, world = advisor._connect(source)
    advisor._require_empty_async(world)
    try:
        for index, (row, trajectory_plan) in enumerate(
            zip(selected.to_dict("records"), plan["trajectories"]), start=1
        ):
            trajectory_id = str(row["trajectory_id"])
            trajectory_dir = output_dir / trajectory_id
            trajectory_dir.mkdir(parents=True, exist_ok=False)
            (trajectory_dir / "coordination").mkdir()
            record: MutableMapping[str, object] = {
                "trajectory_id": trajectory_id,
                "group_id": str(row["group_id"]),
                "scenario_role": str(row["scenario_role"]),
                "geometry_or_route_id": str(row["geometry_or_route_id"]),
                "traffic_density": str(row["traffic_density"]),
                "status": "running",
            }
            factor_contract = factor_contracts.get(trajectory_id)
            if factor_contract is not None:
                record["trajectory_row_sha256"] = (
                    factor_contract.trajectory_row_sha256
                )
                record["requested_factors"] = dict(factor_contract.requested)
                record["authored_onset_policy_visibility"] = (
                    "forbidden_evaluation_metadata_only"
                )
            batch["trajectories"].append(record)
            _replace_json(batch_path, batch)
            _append_progress(
                progress_path,
                "trajectory_started",
                trajectory_id=trajectory_id,
                ordinal=index,
                total=len(selected),
            )
            original_settings = None
            collectors: Dict[str, subprocess.Popen] = {}
            collector_streams = []
            population_process: Optional[subprocess.Popen] = None
            population_stream = None
            scenario_runtime: Optional[CalibrationScenarioRuntime] = None
            traffic_monitor: Optional[advisor.TrafficSanityMonitor] = None
            try:
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
                weather_name = str(row["weather"])
                try:
                    weather = getattr(advisor.carla.WeatherParameters, weather_name)
                except AttributeError as exc:
                    raise ValueError(f"unknown frozen CARLA weather: {weather_name}") from exc
                world.set_weather(weather)
                record["seeds"] = {
                    "carla_seed": int(row["carla_seed"]),
                    "traffic_seed": int(row["traffic_seed"]),
                    "sensor_seed": int(row["sensor_seed"]),
                }
                record["weather"] = weather_name
                static_environment_truth = (
                    _capture_static_environment_truth_before_dynamic_actors(
                        world,
                        trajectory_dir,
                        config,
                    )
                )
                if static_environment_truth is not None:
                    record["static_environment_truth"] = static_environment_truth
                    # Persist the sealed artifact provenance before starting a
                    # collector process that can spawn either ego vehicle.
                    _replace_json(batch_path, batch)
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
                    collectors[role] = process
                    collector_streams.append(log_stream)
                ready_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.ready.json"
                    for role in ROLE_NAMES
                }
                _wait_for_ready(
                    world,
                    collectors,
                    ready_paths,
                    float(config["clock"]["startup_timeout_s"]),
                )
                egos = {
                    role: _find_role_actor(
                        world, str(config["staging_roles"][role]["ego_role_name"])
                    )
                    for role in ROLE_NAMES
                }
                anchor_value = row.get("route_start_anchor_id")
                anchor_id = None if pd.isna(anchor_value) else str(anchor_value)
                scenario = resolve_scenario(
                    world.get_map(),
                    geometry_or_route_id=str(row["geometry_or_route_id"]),
                    scenario_role=str(row["scenario_role"]),
                    route_start_anchor_id=anchor_id,
                )
                scenario_runtime = CalibrationScenarioRuntime(
                    world,
                    scenario,
                    egos,
                    tm_port=int(config["clock"]["tm_port"]),
                    helper_speed_mps=float(config["staging_roles"]["helper"]["target_speed_mps"]),
                    recipient_speed_mps=float(config["staging_roles"]["recipient"]["target_speed_mps"]),
                    pedestrian_speed_mps=float(config["controlled_motion"]["pedestrian_speed_mps"]),
                    pedestrian_start_delay_s=float(config["controlled_motion"]["pedestrian_start_delay_s"]),
                    factor_contract=factor_contract,
                    cadence_s=float(config["clock"]["fixed_delta_seconds"]),
                )
                record["realized_ego_placement"] = scenario_runtime.place_egos()
                record["controlled_actor_setup"] = scenario_runtime.spawn_controlled_actors()
                layer_id, layer, counts = _ambient_counts(config, row)
                record["ambient_evidence_layer"] = layer_id
                record["ambient_evidence_role"] = str(layer["evidence_role"])
                record["ambient_counts"] = dict(counts)
                record["ambient_motion_contract"] = {
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
                record["ambient_population_mode"] = expected_population_mode
                record["scenario_owned_nontreatment_signature"] = (
                    _scenario_owned_nontreatment_signature(world, config)
                )
                if factor_contract is not None:
                    nontreatment = nontreatment_plan_record(
                        row,
                        scenario_owned_signature=record[
                            "scenario_owned_nontreatment_signature"
                        ],
                    )
                    record["nontreatment_plan"] = nontreatment
                    record["nontreatment_plan_sha256"] = canonical_sha256(
                        nontreatment
                    )

                coordination_dir = trajectory_dir / "coordination"
                population_ready_path = coordination_dir / "population.ready.json"
                population_release_path = coordination_dir / "population.release.json"
                population_released_path = coordination_dir / "population.released.json"
                if population_required:
                    population_command = _population_command(
                        config, row, scenario, coordination_dir
                    )
                    record["population_command"] = population_command
                    population_stream = (trajectory_dir / "generate_traffic.log").open(
                        "x", encoding="utf-8"
                    )
                    population_process = subprocess.Popen(
                        population_command,
                        cwd=REPO_ROOT,
                        stdout=population_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    record["population_ready"] = advisor._tick_until(
                        world,
                        [population_process],
                        lambda inventory, roles: (
                            advisor._matching_role_count(roles, "autopilot")
                            >= int(counts["vehicles"])
                            and int(inventory.get("walker.pedestrian.*", 0))
                            >= int(counts["walkers"])
                            and int(inventory.get("controller.ai.walker", 0))
                            >= int(counts["walkers"])
                            and population_ready_path.is_file()
                        ),
                        float(config["ambient_traffic"]["population_start_timeout_s"]),
                        "Phase-2 audit ambient population",
                    )
                    population_ready_manifest = _read_json(population_ready_path)
                    record["population_ready_manifest"] = population_ready_manifest
                    record["ambient_spawn_signature"] = (
                        _validate_population_ready_manifest(
                            population_ready_manifest,
                            vehicles=int(counts["vehicles"]),
                            walkers=int(counts["walkers"]),
                        )
                    )
                    record["ambient_held_live_pose_diagnostic"] = (
                        _ambient_initial_signature(world)
                    )
                else:
                    if int(counts["vehicles"]) or int(counts["walkers"]):
                        raise RuntimeError(
                            "scenario-owned-only layer requested generic ambient actors"
                        )
                    record["population_command"] = None
                    record["population_ready"] = {
                        "applicable": False,
                        "basis": "scenario_owned_only_no_population_process",
                    }
                    record["population_ready_manifest"] = None
                    record["ambient_spawn_signature"] = []
                    record["ambient_held_live_pose_diagnostic"] = []
                traffic_monitor = advisor.TrafficSanityMonitor(
                    world=world,
                    traffic_manager=client.get_trafficmanager(int(config["clock"]["tm_port"])),
                    output_dir=trajectory_dir / "traffic_sanity",
                    integration=_traffic_monitor_integration(config, scenario),
                )
                traffic_monitor.start()
                _add_collision_sensors(
                    traffic_monitor,
                    world,
                    [*egos.values(), *scenario_runtime.owned],
                )
                if population_required:
                    assert population_process is not None
                    _write_json_create(
                        population_release_path,
                        {
                            "schema": POPULATION_RELEASE_SCHEMA,
                            "trajectory_id": trajectory_id,
                            "release_basis": (
                                "monitor_owned_and_spawn_provenance_captured_no_intervening_tick"
                            ),
                            "written_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    record["population_released_manifest"] = (
                        _wait_for_population_release_ack(
                            population_process,
                            population_released_path,
                            float(
                                config["ambient_traffic"][
                                    "population_start_timeout_s"
                                ]
                            ),
                        )
                    )
                    for field, expected in (
                        ("vehicle_count", int(counts["vehicles"])),
                        ("walker_count", int(counts["walkers"])),
                        ("walker_controller_count", int(counts["walkers"])),
                    ):
                        if int(
                            record["population_released_manifest"].get(field, -1)
                        ) != expected:
                            raise RuntimeError(
                                f"ambient population RELEASED {field} mismatch"
                            )
                else:
                    record["population_released_manifest"] = {
                        "applicable": False,
                        "basis": "scenario_owned_only_no_population_release",
                    }
                traffic_monitor.activate_vehicle_motion(client)
                # The follower has issued the physics/controller commands, but
                # CARLA realizes them only on a synchronous world tick. Keep
                # this fixed barrier outside scenario motion and retained
                # capture so no vehicle remains in the held sleeping state.
                setup_barrier_frame_id = int(world.tick(2.0))
                record[
                    "population_release_barrier_frame_id"
                    if population_required
                    else "scenario_setup_barrier_frame_id"
                ] = setup_barrier_frame_id
                scenario_runtime.activate_motion()
                start_snapshot = world.get_snapshot()
                start_frame = int(start_snapshot.frame)
                start_s = float(start_snapshot.timestamp.elapsed_seconds)
                _write_json_create(
                    trajectory_dir / "coordination/capture.start.json",
                    {
                        "schema": "scenesense.phase2_capture_barrier.v1",
                        "trajectory_id": trajectory_id,
                        "after_frame_id": start_frame,
                        "next_frame_is_first_capture": True,
                        "motion_owner": "calibration_audit_orchestrator",
                    },
                )
                heartbeat_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.heartbeat.json"
                    for role in ROLE_NAMES
                }
                tick_ready_paths = {
                    role: trajectory_dir / "coordination" / f"{role}.tick_ready.json"
                    for role in ROLE_NAMES
                }
                previous_frame = start_frame
                captured = []
                for frame_index in range(int(config["clock"]["frames_per_trajectory"])):
                    if population_process is not None:
                        _require_population_process_alive(
                            population_process,
                            phase=f"capture frame {frame_index} pre-tick",
                        )
                    _wait_for_tick_ready(
                        collectors,
                        tick_ready_paths,
                        previous_frame,
                        float(config["clock"]["per_frame_timeout_s"]),
                    )
                    elapsed_before = frame_index * float(config["clock"]["fixed_delta_seconds"])
                    scenario_runtime.before_tick(elapsed_before)
                    traffic_monitor.before_world_tick()
                    traffic_monitor.raise_if_failed()
                    target_frame = int(world.tick(2.0))
                    snapshot = world.get_snapshot()
                    elapsed_after = float(snapshot.timestamp.elapsed_seconds) - start_s
                    scenario_runtime.after_tick(target_frame, elapsed_after)
                    traffic_monitor.observe_snapshot(snapshot)
                    traffic_monitor.raise_if_failed()
                    _wait_for_frame(
                        collectors,
                        heartbeat_paths,
                        tick_ready_paths,
                        target_frame,
                        float(config["clock"]["per_frame_timeout_s"]),
                    )
                    if population_process is not None:
                        _require_population_process_alive(
                            population_process,
                            phase=f"capture frame {frame_index} post-tick",
                        )
                    traffic_monitor.raise_if_failed()
                    captured.append(target_frame)
                    previous_frame = target_frame
                    if len(captured) % 10 == 0:
                        _append_progress(
                            progress_path,
                            "capture_progress",
                            trajectory_id=trajectory_id,
                            completed_frames=len(captured),
                            requested_frames=int(config["clock"]["frames_per_trajectory"]),
                        )
                record["collector_returncodes"] = _wait_collectors_exit(
                    world, collectors, float(config["clock"]["shutdown_timeout_s"])
                )
                if any(code != 0 for code in record["collector_returncodes"].values()):
                    raise RuntimeError(
                        f"paired collector returncodes: {record['collector_returncodes']}"
                    )
                scenario_dir = trajectory_dir / "scenario"
                scenario_dir.mkdir()
                scenario_summary = scenario_runtime.summary()
                _write_json_create(scenario_dir / "realization_summary.json", scenario_summary)
                if factor_contract is not None:
                    factor_result = _persist_factor_forensic_then_finalize(
                        scenario_dir=scenario_dir,
                        trajectory_id=trajectory_id,
                        contract=factor_contract,
                        nontreatment_plan_sha256=str(
                            record["nontreatment_plan_sha256"]
                        ),
                        scenario_summary=scenario_summary,
                        scenario_runtime=scenario_runtime,
                    )
                    record.update(factor_result)
                if scenario_runtime.trace:
                    with (scenario_dir / "realized_trace.csv").open(
                        "x", encoding="utf-8", newline=""
                    ) as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=list(scenario_runtime.trace[0])
                        )
                        writer.writeheader()
                        writer.writerows(scenario_runtime.trace)
                if (
                    str(row["scenario_role"]) == "controlled_positive_occlusion"
                    and str(row["geometry_or_route_id"])
                    in {
                        "curbside_bus_occluded_pedestrian",
                        "signalized_corner_occluded_pedestrian",
                        "parked_van_midblock_occluded_pedestrian",
                    }
                    and not scenario_runtime.walker_completed
                ):
                    raise RuntimeError("controlled pedestrian did not complete its route")
                traffic_summary = traffic_monitor.stop()
                traffic_monitor = None
                record["traffic_sanity"] = traffic_summary
                if not bool(traffic_summary.get("pass")):
                    raise RuntimeError(
                        f"traffic sanity failed: {traffic_summary.get('failures')}"
                    )
                record["trajectory_verification"] = verify_trajectory(
                    config, row, trajectory_dir
                )
                record["captured_frame_count"] = len(captured)
                record["status"] = "capture_verified_pending_cleanup"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                batch["status"] = "failed_hold"
                _append_progress(
                    progress_path,
                    "trajectory_failed",
                    trajectory_id=trajectory_id,
                    error=record["error"],
                )
                raise
            finally:
                for process in collectors.values():
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                for process in collectors.values():
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                if traffic_monitor is not None:
                    record["traffic_sanity"] = traffic_monitor.stop()
                    traffic_monitor = None
                if population_process is not None:
                    record["population_shutdown"] = advisor._stop_processes(
                        world,
                        [("generate_traffic_v1", population_process, population_stream)],
                        float(config["ambient_traffic"]["population_shutdown_timeout_s"]),
                    )
                    population_process = None
                    population_stream = None
                if scenario_runtime is not None:
                    scenario_runtime.destroy()
                for stream in collector_streams:
                    stream.close()
                if original_settings is not None:
                    advisor._restore_async(
                        client, world, int(config["clock"]["tm_port"]), original_settings
                    )
                    cleanup = advisor._tick_until_empty(
                        world, float(config["clock"]["shutdown_timeout_s"])
                    )
                    record["postflight_dynamic_actor_counts"] = cleanup
                _replace_json(batch_path, batch)
            try:
                if any(record.get("postflight_dynamic_actor_counts", {}).values()):
                    raise RuntimeError(
                        "actor cleanup failed: "
                        f"{record['postflight_dynamic_actor_counts']}"
                    )
                heavy_bytes = _stage_heavy_bytes(output_dir)
                free_after = shutil.disk_usage(output_dir).free
                record["stage_heavy_bytes_after_trajectory"] = heavy_bytes
                record["free_bytes_after_trajectory"] = int(free_after)
                if heavy_bytes > int(storage["stage_hard_cap_bytes"]):
                    raise RuntimeError("calibration-audit stage raw cap exceeded")
                if free_after < int(storage["required_free_floor_bytes"]):
                    raise RuntimeError("calibration-audit free-space floor crossed")
                record["status"] = "complete"
                pair_match = _require_completed_pair_match(
                    batch,
                    str(row["group_id"]),
                    config["verification"]["matched_pair_initial_realization_gate"],
                    config["verification"]["matched_pair_trajectory_gate"],
                )
                if factor_contract is not None and pair_match is not None:
                    if factor_smoke_bundle is None:
                        raise RuntimeError("factor runtime bundle disappeared")
                    factor_smoke_config, factor_smoke_plan = factor_smoke_bundle
                    postflight_count = _persist_completed_factor_pair_postflights(
                        batch=batch,
                        group_id=str(row["group_id"]),
                        output_dir=output_dir,
                        factor_smoke_config=factor_smoke_config,
                        factor_smoke_plan=factor_smoke_plan,
                    )
                    if postflight_count:
                        _append_progress(
                            progress_path,
                            "factor_pair_postflight_complete",
                            group_id=str(row["group_id"]),
                            trajectory_count=postflight_count,
                            atomic_admission=False,
                        )
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                batch["status"] = "failed_hold"
                _append_progress(
                    progress_path,
                    "trajectory_postflight_failed",
                    trajectory_id=trajectory_id,
                    error=record["error"],
                )
                _replace_json(batch_path, batch)
                raise
            _append_progress(
                progress_path,
                "trajectory_complete",
                trajectory_id=trajectory_id,
                ordinal=index,
                total=len(selected),
                stage_heavy_bytes=heavy_bytes,
            )
            _replace_json(batch_path, batch)
        if factor_contracts:
            _append_progress(
                progress_path,
                "raw_capture_complete_pending_factor_postflight",
                trajectory_count=len(selected),
                generic_audit_completion_is_scientific_pass=False,
            )
            smoke_bundle = _factor_runtime_bundle(config)
            if smoke_bundle is None:
                raise RuntimeError("factor runtime disappeared before postflight")
            factor_smoke_config, factor_smoke_plan = smoke_bundle
            # Lazy import keeps the historical audit path independent of the
            # offline tracker/model-analysis environment.
            from phase2_map_sharing.factor_smoke_postflight import (
                analyze_batch_artifacts,
            )

            _factor_result, factor_validation = analyze_batch_artifacts(
                batch_root=output_dir,
                smoke_config=factor_smoke_config,
                factor_plan=factor_smoke_plan,
                write_outputs=True,
            )
            if factor_validation.get("verdict") != "PASS_ATOMIC_EXACT_16_ADMITTED":
                raise RuntimeError("factor postflight lacked the registered atomic PASS")
            batch["factor_postflight"] = {
                "result_bundle": str(output_dir / "factor_smoke_results.json"),
                "atomic_validation": str(
                    output_dir / "factor_smoke_validation.json"
                ),
                "verdict": factor_validation["verdict"],
            }
            batch["status"] = "factor_postflight_complete_atomic_exact_16_validated"
            _append_progress(
                progress_path,
                "factor_postflight_complete",
                trajectory_count=len(selected),
                verdict=factor_validation["verdict"],
                next_action="outer_atomic_stage_validation_then_human_review",
            )
        else:
            batch["status"] = "audit_capture_and_per_trajectory_verification_complete"
            _append_progress(
                progress_path,
                "stage_complete",
                trajectory_count=len(selected),
                next_action="human_review_replay_sufficiency_and_oai_field_gate",
            )
        batch["completed_utc"] = datetime.now(timezone.utc).isoformat()
        batch["stage_heavy_bytes"] = _stage_heavy_bytes(output_dir)
        batch["oai_field_gate"] = config["verification"]["oai_fields"]["status"]
        _replace_json(batch_path, batch)
    finally:
        try:
            advisor._require_empty_async(world)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--operator-quality", choices=("Epic",), default="Epic")
    parser.add_argument(
        "--trajectory-id",
        action="append",
        default=[],
        help="bounded regression subset; repeat to preserve a matched pair",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-config", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, source, selected = _load_config(config_path)
    selected = _select_trajectory_ids(selected, args.trajectory_id)
    if args.operator_quality != config["carla"]["renderer_quality_level"]:
        raise ValueError("operator quality declaration differs from frozen audit quality")
    if args.output_dir is None:
        if args.launch:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = _repo_path(config["output_root"]) / f"{stamp}_audit"
        else:
            output_dir = Path("/tmp/phase2_calibration_audit_plan").resolve()
    else:
        output_dir = args.output_dir.resolve()
    plan = build_plan(config, source, selected, output_dir)
    result = {
        "verdict": "PASS",
        "config": str(config_path),
        "selected_groups": int(selected["group_id"].nunique()),
        "selected_trajectories": len(selected),
        "estimated_minutes": plan["estimated_minutes"],
        "estimated_heavy_bytes": plan["estimated_heavy_bytes"],
        "output_dir": str(output_dir),
        "note": "validation/dry-run only; no CARLA or OAI process was started",
    }
    if args.dry_run:
        result["plan"] = plan
    if args.launch:
        try:
            run_live(config, source, selected, plan, output_dir)
        except BaseException as exc:
            if output_dir.is_dir() and not (output_dir / "FAILED.json").exists():
                _write_json_create(
                    output_dir / "FAILED.json",
                    {
                        "schema": "scenesense.phase2_calibration_audit_sentinel.v1",
                        "status": "failed_hold",
                        "error": f"{type(exc).__name__}: {exc}",
                        "written_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise
        factor_runtime_enabled = config.get("factor_realization_runtime") is not None
        factor_validation_path = output_dir / "factor_smoke_validation.json"
        factor_validation = None
        if factor_runtime_enabled:
            if not factor_validation_path.is_file():
                raise RuntimeError(
                    "factor runtime completed without atomic validation artifact"
                )
            factor_validation = json.loads(
                factor_validation_path.read_text(encoding="utf-8")
            )
            if factor_validation.get("verdict") != (
                "PASS_ATOMIC_EXACT_16_ADMITTED"
            ):
                raise RuntimeError(
                    "factor runtime completed without registered atomic PASS"
                )
        summary = {
            "schema": "scenesense.phase2_calibration_audit_sentinel.v1",
            "status": (
                "factor_smoke_atomic_exact_16_admitted_stop_for_human_gate"
                if factor_runtime_enabled
                else "audit_complete_stop_for_human_gate"
            ),
            "batch_root": str(output_dir),
            "oai_field_gate": config["verification"]["oai_fields"]["status"],
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }
        if factor_runtime_enabled:
            summary["factor_smoke_validation"] = {
                "path": str(factor_validation_path),
                "sha256": _sha256(factor_validation_path),
                "verdict": factor_validation["verdict"],
            }
        _write_json_create(output_dir / "RESULTS_SUMMARY.json", summary)
        _write_json_create(output_dir / "COMPLETED.json", summary)
        result["note"] = (
            "factor smoke admitted atomically; no downstream stage was chained"
            if factor_runtime_enabled
            else "audit complete; no downstream stage was chained"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

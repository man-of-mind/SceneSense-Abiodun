#!/usr/bin/env python3
"""Offline contract and result validator for the Phase-2 factor smoke.

This module deliberately has no CARLA imports and no process-launching path.
It selects sixteen immutable replicate-0 calibration rows, attaches requested
physical factors, and validates a later result bundle before those rows may be
counted in calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from phase2_map_sharing.factor_smoke_runtime_contract import (
    FEATURE_SOURCE_STAGE,
    POLICY_AUDIT_SCHEMA,
    aggregate_guardrail_reports,
    analyze_installed_track_guardrails,
    build_recipient_available_endpoint,
    canonical_sha256 as _runtime_canonical_sha256,
    summarize_policy_audits,
    validate_availability_record,
    validate_recipient_map_target_match,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "data_collection/configs/phase2_factor_realization_smoke_v1.yaml"
RESULT_SCHEMA = "scenesense.phase2_factor_realization_smoke_results.v1"
SUMMARY_SCHEMA = "scenesense.phase2_factor_realization_smoke_validation.v1"
ROW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

REQUESTED_FACTOR_STRING_FIELDS = (
    "factor_realization_status",
    "time_to_hazard_label_status",
    "hazard_actor_role",
    "onset_driver_role",
    "geometry_measurement_basis",
    "closing_speed_measurement_basis",
    "proximity_horizon_measurement_basis",
)
REQUESTED_FACTOR_FLOAT_FIELDS = (
    "requested_helper_speed_mps",
    "requested_recipient_speed_mps",
    "requested_hazard_actor_speed_mps",
    "requested_onset_driver_speed_mps",
    "requested_hazard_onset_s",
    "requested_closing_speed_target_mps",
    "requested_closing_speed_band_min_mps",
    "requested_closing_speed_band_max_mps",
    "requested_proximity_horizon_target_s",
    "requested_proximity_horizon_band_min_s",
    "requested_proximity_horizon_band_max_s",
    "minimum_onset_driver_speed_mps",
)


class ContractError(ValueError):
    """Raised when a design or result violates the frozen smoke contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "config must be a mapping")
    _require(
        config.get("schema_version")
        == "scenesense.phase2_factor_realization_smoke_config.v1",
        "unsupported factor-smoke config schema",
    )
    return config


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _requested_factor_contract(row: Mapping[str, str]) -> dict[str, Any]:
    """Return the v2 row's complete, typed factor-control contract."""

    contract: dict[str, Any] = {
        "closing_speed_band": str(row["closing_speed_band"]),
        # This is retained as the design's historical stratification name.  The
        # measured quantity is the typed proximity horizon below, not collision TTC.
        "time_to_hazard_band": str(row["time_to_hazard_band"]),
    }
    for field in REQUESTED_FACTOR_STRING_FIELDS:
        value = str(row[field]).strip()
        _require(value != "", f"manifest factor field {field} is empty")
        contract[field] = value
    for field in REQUESTED_FACTOR_FLOAT_FIELDS:
        value = _finite(f"manifest.{field}", row[field])
        _require(value >= 0.0, f"manifest factor field {field} must be nonnegative")
        contract[field] = value

    closing_min = contract["requested_closing_speed_band_min_mps"]
    closing_target = contract["requested_closing_speed_target_mps"]
    closing_max = contract["requested_closing_speed_band_max_mps"]
    _require(
        closing_min <= closing_target <= closing_max,
        "manifest closing-speed target is outside its declared band",
    )
    horizon_min = contract["requested_proximity_horizon_band_min_s"]
    horizon_target = contract["requested_proximity_horizon_target_s"]
    horizon_max = contract["requested_proximity_horizon_band_max_s"]
    _require(
        horizon_min <= horizon_target <= horizon_max,
        "manifest proximity-horizon target is outside its declared band",
    )
    _require(
        contract["requested_onset_driver_speed_mps"]
        >= contract["minimum_onset_driver_speed_mps"],
        "manifest onset-driver request is below its measurement floor",
    )
    return contract


def _validate_feature_contract(config: Mapping[str, Any]) -> str:
    feature = config["policy_feature_contract"]
    _require(bool(feature["projection_must_be_exact"]), "feature projection must be exact")
    forbidden = tuple(str(item).lower() for item in feature["forbidden_feature_tokens"])
    all_names: list[str] = []
    for stage in ("placement_features", "publication_features"):
        names = [str(item) for item in feature[stage]]
        _require(names, f"{stage} must not be empty")
        _require(len(names) == len(set(names)), f"{stage} contains duplicate names")
        for name in names:
            lowered = name.lower()
            hits = [token for token in forbidden if token in lowered]
            _require(not hits, f"forbidden policy feature {name!r} matches {hits}")
        all_names.extend(f"{stage}:{name}" for name in names)
    return _canonical_sha256(
        {
            "placement_features": list(feature["placement_features"]),
            "publication_features": list(feature["publication_features"]),
            "forbidden_feature_tokens": list(feature["forbidden_feature_tokens"]),
            "projection_must_be_exact": True,
        }
    )


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the static design and return the exact sixteen-row plan."""

    authorization = config["authorization"]
    _require(bool(authorization["offline_plan_validation"]), "offline validation disabled")
    for forbidden in (
        "carla_launch",
        "oai_launch",
        "warning_parameter_selection",
        "old_15_trajectory_audit_chain",
        "remaining_calibration",
        "validation_collection",
        "test_collection",
        "controller_evaluation",
        "rl_training",
    ):
        _require(not bool(authorization[forbidden]), f"{forbidden} must remain false")

    source = config["source_design"]
    manifest_path = _resolve_repo_path(source["manifest"])
    design_config_path = _resolve_repo_path(source["design_config"])
    _require(manifest_path.is_file(), f"manifest is missing: {manifest_path}")
    _require(design_config_path.is_file(), f"design config is missing: {design_config_path}")
    actual_manifest_sha = _sha256_file(manifest_path)
    actual_design_sha = _sha256_file(design_config_path)
    _require(actual_manifest_sha == source["manifest_sha256"], "source manifest hash drifted")
    _require(actual_design_sha == source["design_config_sha256"], "source design config hash drifted")
    design_document = yaml.safe_load(design_config_path.read_text(encoding="utf-8"))
    _require(isinstance(design_document, Mapping), "source design config must be a mapping")
    _require(
        design_document.get("schema_version") == source["design_config_schema"],
        "source design config schema drifted",
    )
    _require(
        design_document.get("design_id") == source["design_id"],
        "source design ID drifted",
    )
    design_authorization = design_document.get("authorization", {})
    _require(isinstance(design_authorization, Mapping), "source authorization is missing")
    for forbidden in (
        "carla_launch",
        "oai_launch",
        "full_collection",
        "controller_evaluation",
        "rl_training",
    ):
        _require(
            design_authorization.get(forbidden) is False,
            f"source design unexpectedly authorizes {forbidden}",
        )

    selector = source["selector"]
    geometries = {str(item) for item in selector["geometries"]}
    speed_bands = {str(item) for item in selector["closing_speed_bands"]}
    tth_bands = {str(item) for item in selector["time_to_hazard_bands"]}
    scenario_roles = {str(item) for item in selector["scenario_roles"]}
    manifest_rows = _read_manifest(manifest_path)
    _require(manifest_rows, "source manifest is empty")
    required_columns = {
        "schema",
        "design_id",
        "suite_id",
        "split",
        "group_id",
        "matched_pair_id",
        "geometry_or_route_id",
        "hazard_class",
        "closing_speed_band",
        "time_to_hazard_band",
        "trajectory_id",
        "scenario_role",
        "controlled_hazard_present",
        *REQUESTED_FACTOR_STRING_FIELDS,
        *REQUESTED_FACTOR_FLOAT_FIELDS,
    }
    _require(
        required_columns <= set(manifest_rows[0]),
        f"source manifest lacks factor columns: {sorted(required_columns - set(manifest_rows[0]))}",
    )
    rows = [
        row
        for row in manifest_rows
        if row["suite_id"] == str(selector["suite_id"])
        and row["split"] == str(selector["split"])
        and row["geometry_or_route_id"] in geometries
        and row["closing_speed_band"] in speed_bands
        and row["time_to_hazard_band"] in tth_bands
        and row["scenario_role"] in scenario_roles
        and row["group_id"].endswith("_" + str(selector["replicate_suffix"]))
    ]
    rows.sort(key=lambda row: row["trajectory_id"])

    _require(len(rows) == int(selector["expected_trajectories"]), "trajectory count drifted")
    _require(len({row["trajectory_id"] for row in rows}) == len(rows), "duplicate trajectory ID")
    groups = {row["group_id"] for row in rows}
    _require(len(groups) == int(selector["expected_groups"]), "group count drifted")
    positives = [row for row in rows if row["scenario_role"] == "controlled_positive_occlusion"]
    benign = [row for row in rows if row["scenario_role"] == "matched_benign_negative"]
    _require(
        len(positives) == int(selector["expected_positive_trajectories"]),
        "positive trajectory count drifted",
    )
    _require(
        len(benign) == int(selector["expected_benign_trajectories"]),
        "benign trajectory count drifted",
    )

    expected_classes = {
        str(key): str(value)
        for key, value in selector["expected_hazard_class_by_geometry"].items()
    }
    feature_contract_sha = _validate_feature_contract(config)
    pinned = {str(key): str(value) for key, value in config["pinned_trajectory_row_sha256"].items()}
    _require(set(pinned) == {row["trajectory_id"] for row in rows}, "pinned row IDs drifted")

    plan_rows: list[dict[str, Any]] = []
    cell_counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        trajectory_id = row["trajectory_id"]
        row_sha = _canonical_sha256(row)
        _require(ROW_HASH_RE.fullmatch(pinned[trajectory_id]) is not None, "invalid pinned row hash")
        _require(row_sha == pinned[trajectory_id], f"row hash drifted: {trajectory_id}")
        _require(row["schema"] == source["manifest_schema"], f"manifest schema drifted: {trajectory_id}")
        _require(row["design_id"] == source["design_id"], f"design ID drifted: {trajectory_id}")
        geometry = row["geometry_or_route_id"]
        _require(row["hazard_class"] == expected_classes[geometry], f"hazard class drifted: {trajectory_id}")
        positive = row["scenario_role"] == "controlled_positive_occlusion"
        _require(row["controlled_hazard_present"] == ("1" if positive else "0"), f"treatment drift: {trajectory_id}")
        _require(row["matched_pair_id"] == row["group_id"], f"pair ID drift: {trajectory_id}")
        _require(
            row["factor_realization_status"]
            == str(selector["expected_factor_realization_status"]),
            f"factor realization status drifted: {trajectory_id}",
        )
        _require(
            row["time_to_hazard_label_status"]
            == str(selector["expected_time_to_hazard_label_status"]),
            f"time-to-hazard label status drifted: {trajectory_id}",
        )
        key = (geometry, row["closing_speed_band"], row["time_to_hazard_band"])
        cell_counts[key] = cell_counts.get(key, 0) + 1
        requested_factor_contract = _requested_factor_contract(row)
        plan_rows.append(
            {
                "trajectory_id": trajectory_id,
                "trajectory_row_sha256": row_sha,
                "group_id": row["group_id"],
                "matched_pair_id": row["matched_pair_id"],
                "geometry_or_route_id": geometry,
                "hazard_class": row["hazard_class"],
                "scenario_role": row["scenario_role"],
                "controlled_hazard_present": positive,
                "closing_speed_band": row["closing_speed_band"],
                "time_to_hazard_band": row["time_to_hazard_band"],
                "requested_factor_contract": requested_factor_contract,
                "policy_feature_contract_sha256": feature_contract_sha,
            }
        )

    expected_cells = {
        (geometry, speed, tth)
        for geometry in geometries
        for speed in speed_bands
        for tth in tth_bands
    }
    _require(set(cell_counts) == expected_cells, "factor-cell coverage drifted")
    _require(all(count == 2 for count in cell_counts.values()), "each cell must contain one pair")
    for group_id in groups:
        pair = [row for row in plan_rows if row["group_id"] == group_id]
        _require(len(pair) == 2, f"group {group_id} is not a two-trajectory pair")
        _require(
            {row["scenario_role"] for row in pair}
            == {"controlled_positive_occlusion", "matched_benign_negative"},
            f"group {group_id} does not contain positive and benign twins",
        )
        _require(
            len(
                {
                    _canonical_sha256(row["requested_factor_contract"])
                    for row in pair
                }
            )
            == 1,
            f"group {group_id} positive/benign factor controls differ",
        )

    readiness = config["runtime_readiness"]
    readiness_keys = (
        "factor_adapter_status",
        "recipient_install_event_status",
        "policy_feature_projection_status",
        "launch_wrapper_status",
    )
    blockers = [
        f"{key}={readiness[key]}"
        for key in readiness_keys
        if str(readiness[key]) != "ready_verified"
    ]
    plan = {
        "schema": "scenesense.phase2_factor_realization_smoke_plan.v1",
        "stage_id": config["stage_id"],
        "source_manifest": str(source["manifest"]),
        "source_manifest_sha256": actual_manifest_sha,
        "source_manifest_schema": source["manifest_schema"],
        "source_design_config_sha256": actual_design_sha,
        "source_design_config_schema": source["design_config_schema"],
        "source_design_id": source["design_id"],
        "policy_feature_contract_sha256": feature_contract_sha,
        "trajectory_count": len(plan_rows),
        "group_count": len(groups),
        "positive_trajectory_count": len(positives),
        "benign_trajectory_count": len(benign),
        "world_time_estimate_minutes": 46.4,
        "reuse_if_atomic_pass": "replicate_0_calibration_tranche",
        "reuse_if_fail": "excluded_factor_smoke_fixture_only",
        "runtime_ready": not blockers,
        "runtime_blockers": blockers,
        "collection_authorized_by_this_tool": False,
        "oai_authorized": False,
        "warning_selection_authorized": False,
        "rows": plan_rows,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _validate_exact_feature_projection(
    projection: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    expected = config["policy_feature_contract"]
    _require(bool(projection.get("consumer_enforces_exact_projection")), "feature consumer is not exact")
    _require(
        ROW_HASH_RE.fullmatch(str(projection.get("consumer_code_sha256", ""))) is not None,
        "feature consumer code hash is invalid",
    )
    _require(
        int(projection.get("placement_decision_count", 0)) > 0,
        "no realized placement decisions passed through the projection",
    )
    _require(
        int(projection.get("publication_decision_count", 0)) > 0,
        "no realized publication decisions passed through the projection",
    )
    _require(
        list(projection.get("placement_features", [])) == list(expected["placement_features"]),
        "realized placement feature projection drifted",
    )
    _require(
        list(projection.get("publication_features", [])) == list(expected["publication_features"]),
        "realized publication feature projection drifted",
    )
    forbidden = tuple(str(item).lower() for item in expected["forbidden_feature_tokens"])
    for stage in ("placement_features", "publication_features"):
        for name in projection[stage]:
            hits = [token for token in forbidden if token in str(name).lower()]
            _require(not hits, f"realized policy feature {name!r} matches forbidden tokens {hits}")
    exercise = projection.get("projection_exercise")
    expected_exercise = config["policy_projection_exercise"]
    _require(isinstance(exercise, Mapping), "policy projection exercise is missing")
    for field in (
        "role",
        "policy_action_selected_from_features",
        "policy_performance_evaluated",
        "observed_policy_state_complete",
        "fixed_actions",
    ):
        _require(
            exercise.get(field) == expected_exercise[field],
            f"policy projection exercise {field} drifted",
        )
    _require(
        exercise.get("policy_action_selected_from_features") is False
        and exercise.get("policy_performance_evaluated") is False
        and exercise.get("observed_policy_state_complete") is False,
        "fixed-action projection audit is being misrepresented as policy evidence",
    )
    _require(
        list(exercise.get("fixture_backed_fields", []))
        == sorted(expected_exercise["fixture_backed_fields"]),
        "fixture-backed field declaration drifted",
    )
    abstracted = expected_exercise["local_loopback_transport_abstracted_fields"]
    _require(
        list(exercise.get("local_loopback_transport_abstracted_fields", []))
        == sorted(abstracted["fields"]),
        "transport-abstracted field declaration drifted",
    )
    _require(
        projection.get("zero_track_policy_state_handling_status")
        == "future_controller_blocker_missingness_contract_not_frozen"
        and projection.get("environment_readiness_claimed") is False,
        "zero-track future-controller blocker is not explicit",
    )


def _event_time(event: Mapping[str, Any], name: str) -> float | None:
    status = str(event.get("status", ""))
    _require(status in {"event", "censored", "miss"}, f"{name}.status is invalid")
    if status == "event":
        return _finite(f"{name}.at_s", event.get("at_s"))
    _finite(f"{name}.censor_at_s", event.get("censor_at_s"))
    return None


def _validate_policy_audit(
    audit: Mapping[str, Any],
    *,
    trajectory_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = config["causal_policy_audit_contract"]
    feature_contract = config["policy_feature_contract"]
    _require(audit.get("schema") == contract["schema"] == POLICY_AUDIT_SCHEMA, "policy audit schema drifted")
    _require(audit.get("trajectory_id") == trajectory_id, "policy audit trajectory ID drifted")
    _require(str(audit.get("clock_id", "")).strip(), "policy audit clock_id is required")
    _require(str(audit.get("arm_id", "")).strip(), "policy audit arm_id is required")
    locus = str(audit.get("decision_locus", ""))
    _require(locus in {"helper", "edge", "recipient"}, "policy audit decision locus is invalid")
    body = {key: value for key, value in audit.items() if key != "audit_sha256"}
    _require(
        str(audit.get("audit_sha256", "")) == _runtime_canonical_sha256(body),
        "policy audit hash is inconsistent",
    )
    _require(bool(audit.get("consumer_enforces_exact_projection")), "policy loader is not exact")
    _require(
        ROW_HASH_RE.fullmatch(str(audit.get("consumer_code_sha256", ""))) is not None,
        "policy consumer code hash is invalid",
    )
    expected_projection = config["policy_projection_exercise"]
    projection = audit.get("projection_exercise")
    _require(isinstance(projection, Mapping), "per-row policy projection exercise is missing")
    expected_projection_record = {
        "role": expected_projection["role"],
        "policy_action_selected_from_features": expected_projection[
            "policy_action_selected_from_features"
        ],
        "policy_performance_evaluated": expected_projection[
            "policy_performance_evaluated"
        ],
        "observed_policy_state_complete": expected_projection[
            "observed_policy_state_complete"
        ],
        "fixed_actions": dict(expected_projection["fixed_actions"]),
        "fixture_backed_fields": sorted(expected_projection["fixture_backed_fields"]),
        "local_loopback_transport_abstracted_fields": sorted(
            expected_projection["local_loopback_transport_abstracted_fields"]["fields"]
        ),
    }
    _require(
        dict(projection) == expected_projection_record,
        "per-row policy projection exercise drifted",
    )
    for stage in contract["required_stages"]:
        _require(
            list(audit.get(f"{stage}_features", []))
            == list(feature_contract[f"{stage}_features"]),
            f"{stage} audit feature contract drifted",
        )

    decisions = audit.get("decisions")
    _require(isinstance(decisions, list), "policy audit decisions must be a list")
    counts = {stage: 0 for stage in contract["required_stages"]}
    forbidden_tokens = tuple(
        str(item).lower() for item in feature_contract["forbidden_feature_tokens"]
    )
    for decision in decisions:
        _require(isinstance(decision, Mapping), "policy decision audit must be a mapping")
        stage = str(decision.get("decision_stage", ""))
        _require(stage in counts, "policy decision stage is invalid")
        _require(
            decision.get("action")
            == expected_projection["fixed_actions"][stage],
            f"policy audit {stage} action differs from fixed exercise",
        )
        _require(decision.get("clock_id") == audit["clock_id"], "policy decision clock drifted")
        _require(decision.get("arm_id") == audit["arm_id"], "policy decision arm drifted")
        decision_at = _finite("decision_at_s", decision.get("decision_at_s"))
        record_body = {key: value for key, value in decision.items() if key != "record_sha256"}
        _require(
            str(decision.get("record_sha256", ""))
            == _runtime_canonical_sha256(record_body),
            "policy decision record hash is inconsistent",
        )
        fields = decision.get("fields")
        _require(isinstance(fields, list), "policy decision fields must be a list")
        expected_names = list(feature_contract[f"{stage}_features"])
        observed_names = [str(field.get("feature_name", "")) for field in fields]
        _require(observed_names == expected_names, "policy decision did not consume the exact ordered feature set")
        _require(len(observed_names) == len(set(observed_names)), "policy decision has duplicate fields")
        for field in fields:
            name = str(field["feature_name"])
            hits = [token for token in forbidden_tokens if token in name.lower()]
            _require(not hits, f"policy audit field {name!r} contains forbidden tokens {hits}")
            _require(
                field.get("source_stage") == FEATURE_SOURCE_STAGE[name],
                f"policy field {name} source provenance drifted",
            )
            fixture_fields = set(expected_projection["fixture_backed_fields"])
            abstracted_contract = expected_projection[
                "local_loopback_transport_abstracted_fields"
            ]
            abstracted_fields = set(abstracted_contract["fields"])
            evidence_kind = str(field.get("evidence_kind", ""))
            expected_kind = (
                "preregistered_fixture"
                if name in fixture_fields
                else "local_loopback_transport_abstraction"
                if name in abstracted_fields
                else "observed"
            )
            _require(
                evidence_kind == expected_kind,
                f"policy field {name} evidence kind drifted",
            )
            if name in abstracted_fields:
                _require(
                    abstracted_contract["fields"][name] == field.get("source_stage"),
                    f"policy field {name} abstraction source drifted",
                )
            observed_at = _finite(f"{name}.observed_at_s", field.get("observed_at_s"))
            available_at = _finite(f"{name}.available_at_s", field.get("available_at_s"))
            _require(observed_at <= available_at <= decision_at + 1e-12, f"policy field {name} violates causal availability")
            _require(
                ROW_HASH_RE.fullmatch(str(field.get("value_sha256", ""))) is not None,
                f"policy field {name} value hash is invalid",
            )
            components = field.get("component_provenance", [])
            _require(isinstance(components, list), "component provenance must be a list")
            if FEATURE_SOURCE_STAGE[name] == "derived_relative_kinematics":
                expected_recipient = (
                    "recipient_localization" if locus == "recipient" else "recipient_state_transport"
                )
                expected_sources = {"helper_localization", expected_recipient}
                _require(
                    {str(item.get("source_stage")) for item in components}
                    == expected_sources,
                    f"relative field {name} lacks both causal role components",
                )
                component_observed = [
                    _finite("component.observed_at_s", item.get("observed_at_s"))
                    for item in components
                ]
                component_available = [
                    _finite("component.available_at_s", item.get("available_at_s"))
                    for item in components
                ]
                _require(
                    all(a <= b <= decision_at + 1e-12 for a, b in zip(component_observed, component_available)),
                    f"relative field {name} component violates causal availability",
                )
                _require(abs(observed_at - max(component_observed)) <= 1e-12, f"relative field {name} observed time is not component-derived")
                _require(abs(available_at - max(component_available)) <= 1e-12, f"relative field {name} availability is not component-derived")
            else:
                _require(not components, f"non-derived field {name} has component provenance")
        counts[stage] += 1
    minimum = int(contract["minimum_decisions_per_stage_per_trajectory"])
    _require(all(value >= minimum for value in counts.values()), "policy audit lacks a required stage decision")
    _require(dict(audit.get("decision_counts", {})) == counts, "policy audit decision counts are not recomputable")
    canary = audit.get("forbidden_field_canary")
    _require(isinstance(canary, Mapping), "forbidden-field canary evidence is missing")
    expected_canary = contract["forbidden_field_canary"]
    _require(str(canary.get("canary_field_name", "")).lower() == "ground_truth_id", "forbidden canary name drifted")
    _require(int(canary.get("attempt_count", -1)) == 1, "forbidden canary must run once per trajectory")
    _require(
        int(canary.get("rejection_count", -1))
        == int(expected_canary["required_rejections_per_trajectory"]),
        "forbidden canary rejection count drifted",
    )
    _require(
        int(canary.get("acceptance_count", -1))
        <= int(expected_canary["maximum_acceptances"]),
        "forbidden canary reached a policy",
    )
    exposure = audit.get("policy_state_exposure")
    _require(isinstance(exposure, Mapping), "policy-state exposure diagnostic is missing")
    samples = exposure.get("samples")
    _require(isinstance(samples, list) and samples, "policy-state exposure samples are missing")
    zero_source = 0
    zero_map = 0
    for sample in samples:
        _require(isinstance(sample, Mapping), "policy-state exposure sample is invalid")
        sample_body = {key: value for key, value in sample.items() if key != "record_sha256"}
        _require(
            sample.get("record_sha256") == _runtime_canonical_sha256(sample_body),
            "policy-state exposure hash is inconsistent",
        )
        _finite("policy-state sample_at_s", sample.get("sample_at_s"))
        source_count = int(sample.get("source_track_count", -1))
        map_count = int(sample.get("installed_map_track_count", -1))
        _require(source_count >= 0 and map_count >= 0, "policy-state count is negative")
        zero_source += int(source_count == 0)
        zero_map += int(map_count == 0)
    _require(int(exposure.get("sample_count", -1)) == len(samples), "policy-state sample count drifted")
    _require(int(exposure.get("zero_source_track_sample_count", -1)) == zero_source, "zero-source count drifted")
    _require(int(exposure.get("zero_installed_map_track_sample_count", -1)) == zero_map, "zero-map count drifted")
    _require(bool(exposure.get("zero_object_state_seen")) == bool(zero_source or zero_map), "zero-object state flag drifted")
    _require(
        exposure.get("zero_track_policy_state_handling_status")
        == "future_controller_blocker_missingness_contract_not_frozen"
        and exposure.get("environment_readiness_claimed") is False,
        "policy-state diagnostic overclaims future-controller readiness",
    )
    return {
        "consumer_code_sha256": audit["consumer_code_sha256"],
        "counts": counts,
        "audit_sha256": audit["audit_sha256"],
        "zero_object_state_seen": bool(exposure["zero_object_state_seen"]),
    }


def _validate_endpoint(
    endpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    map_target_matches: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    contract = config["installed_track_endpoint"]
    status = str(endpoint.get("endpoint_status", ""))
    _require(status in set(contract["allowed_statuses"]), "endpoint status is not typed")
    _require(str(endpoint.get("clock_id", "")).strip(), "endpoint clock_id is required")
    horizon = _finite("endpoint.evaluation_horizon_s", endpoint.get("evaluation_horizon_s"))
    _require(horizon > 0.0, "endpoint evaluation horizon must be positive")
    chain_sha = str(endpoint.get("evidence_chain_sha256", ""))
    _require(ROW_HASH_RE.fullmatch(chain_sha) is not None, "endpoint evidence chain hash is invalid")

    helper = endpoint.get("helper_source_confirmation")
    install = endpoint.get("helper_track_recipient_install")
    recipient_confirmation = endpoint.get("recipient_self_source_confirmation")
    recipient_install = endpoint.get("recipient_self_track_recipient_install")
    _require(isinstance(helper, Mapping), "helper source event is required")
    _require(isinstance(install, Mapping), "recipient install event is required")
    _require(isinstance(recipient_confirmation, Mapping), "recipient-self confirmation event is required")
    _require(isinstance(recipient_install, Mapping), "recipient-self install event is required")
    helper_at = _event_time(helper, "helper_source_confirmation")
    helper_available_at = _event_time(install, "helper_track_recipient_install")
    recipient_confirmation_at = _event_time(
        recipient_confirmation, "recipient_self_source_confirmation"
    )
    recipient_available_at = _event_time(
        recipient_install, "recipient_self_track_recipient_install"
    )

    if helper_available_at is not None:
        for field in contract["required_install_provenance"]:
            _require(field in install, f"recipient install is missing {field}")
        for field in ("contribution_id", "source_track_id", "recipient_map_track_id"):
            _require(str(install[field]).strip(), f"recipient install {field} is empty")
        published = _finite("published_at_s", install["published_at_s"])
        installed = _finite("installed_at_s", install["installed_at_s"])
        available = _finite("available_at_s", install["available_at_s"])
        _require(abs(available - helper_available_at) <= 1e-9, "install event at_s differs from available_at_s")
        _require(published <= installed <= available, "publish/install/available order is invalid")
        if helper_at is not None:
            _require(helper_at <= published, "helper confirmation occurs after publication")

    if recipient_available_at is not None:
        for field in (
            "local_install_id",
            "source_track_id",
            "recipient_map_track_id",
            "confirmed_at_s",
            "installed_at_s",
            "available_at_s",
            "clock_id",
            "consumer_boundary",
        ):
            _require(field in recipient_install, f"recipient-self install is missing {field}")
        local_confirmed = _finite("recipient_self.confirmed_at_s", recipient_install["confirmed_at_s"])
        local_installed = _finite("recipient_self.installed_at_s", recipient_install["installed_at_s"])
        local_available = _finite("recipient_self.available_at_s", recipient_install["available_at_s"])
        _require(
            abs(local_available - recipient_available_at) <= 1e-9,
            "recipient-self event at_s differs from available_at_s",
        )
        _require(
            local_confirmed <= local_installed <= local_available,
            "recipient-self confirmation/install/available order is invalid",
        )
        if recipient_confirmation_at is not None:
            _require(
                abs(local_confirmed - recipient_confirmation_at) <= 1e-9,
                "recipient-self source confirmation differs from install provenance",
            )

    if status == "numeric":
        _require(
            helper_at is not None
            and helper_available_at is not None
            and recipient_confirmation_at is not None
            and recipient_available_at is not None,
            "numeric endpoint lacks events",
        )
        expected_margin = recipient_available_at - helper_available_at
        observed_margin = _finite("recipient_available_confirmed_track_margin_s", endpoint.get("recipient_available_confirmed_track_margin_s"))
        _require(abs(expected_margin - observed_margin) <= 1e-9, "numeric endpoint margin is inconsistent")
    elif status == "ego_right_censored":
        _require(
            helper_available_at is not None and recipient_available_at is None,
            "ego censoring has inconsistent events",
        )
        lower_bound = _finite("recipient_available_confirmed_track_margin_lower_bound_s", endpoint.get("recipient_available_confirmed_track_margin_lower_bound_s"))
        _require(abs(lower_bound - (horizon - helper_available_at)) <= 1e-9, "censored lower bound is inconsistent")
    elif status == "cooperative_miss":
        _require(
            helper_available_at is None and recipient_available_at is not None,
            "cooperative miss has inconsistent events",
        )
    elif status == "both_miss":
        _require(
            helper_available_at is None and recipient_available_at is None,
            "both_miss contains an event",
        )

    association = endpoint.get("evaluation_association")
    _require(isinstance(association, Mapping), "endpoint evaluation association is missing")
    for match in map_target_matches:
        try:
            validate_recipient_map_target_match(match)
        except (TypeError, ValueError, KeyError) as exc:
            raise ContractError(f"recipient-map target match is invalid: {exc}") from exc
    recomputed = build_recipient_available_endpoint(
        provenance,
        helper_source_track_id=association.get("helper_source_track_id"),
        recipient_source_track_id=association.get("recipient_source_track_id"),
        recipient_map_track_id=association.get("recipient_map_track_id"),
        evaluation_horizon_s=horizon,
        evaluation_recipient_map_target_matches=map_target_matches,
    )
    _require(dict(endpoint) == recomputed, "endpoint is not recomputable from immutable availability provenance")


def _validate_capture_model_identity(
    identity: object, trajectory_id: str
) -> str:
    _require(isinstance(identity, Mapping), f"capture model identity missing: {trajectory_id}")
    _require(set(identity) == {"helper", "recipient"}, f"capture model roles drifted: {trajectory_id}")
    for role in ("helper", "recipient"):
        record = identity[role]
        _require(isinstance(record, Mapping), f"{role} model identity is invalid: {trajectory_id}")
        for field in (
            "model_sha256",
            "config_sha256",
            "checkpoint_sha256_at_capture",
            "checkpoint_sha256_recomputed",
            "manifest_sha256",
        ):
            _require(
                ROW_HASH_RE.fullmatch(str(record.get(field, ""))) is not None,
                f"{role} {field} is invalid: {trajectory_id}",
            )
        captured = str(record["checkpoint_sha256_at_capture"])
        recomputed = str(record["checkpoint_sha256_recomputed"])
        _require(
            record.get("checkpoint_identity_basis") == "capture_time_file_bytes"
            and record.get("checkpoint_hash_status")
            == "capture_time_sha256_recomputed_equal"
            and record.get("checkpoint_sha256_equal") is True
            and captured == recomputed == str(record["model_sha256"]),
            f"{role} capture-time checkpoint identity differs: {trajectory_id}",
        )
        checkpoint = Path(str(record.get("checkpoint_path_at_capture", ""))).resolve()
        _require(checkpoint.is_file(), f"{role} checkpoint is unavailable: {trajectory_id}")
        _require(
            _sha256_file(checkpoint) == captured,
            f"{role} checkpoint changed after postflight: {trajectory_id}",
        )
    model_shas = {str(identity[role]["model_sha256"]) for role in ("helper", "recipient")}
    _require(
        len(model_shas) == 1,
        f"helper and recipient checkpoint bytes differ: {trajectory_id}",
    )
    return next(iter(model_shas))


def _validate_dependency_fingerprints(
    observed: object, config: Mapping[str, Any]
) -> None:
    pinned = config["recipient_endpoint_runtime"]["dependency_sha256"]
    _require(isinstance(observed, Mapping), "postflight dependency fingerprints are missing")
    _require(set(observed) == set(pinned), "postflight dependency fingerprint keys drifted")
    for name, contract in pinned.items():
        item = observed[name]
        _require(isinstance(item, Mapping), f"dependency {name} evidence is invalid")
        path = _resolve_repo_path(contract["path"]).resolve()
        _require(Path(str(item.get("path", ""))).resolve() == path, f"dependency {name} path drifted")
        expected_sha = str(contract["sha256"])
        _require(
            item.get("sha256") == expected_sha
            and ROW_HASH_RE.fullmatch(expected_sha) is not None
            and _sha256_file(path) == expected_sha,
            f"dependency {name} bytes drifted",
        )


def _validate_retention_window_evidence(
    evidence: object,
    *,
    trajectory_id: str,
    realized_onset_s: float | None,
) -> None:
    _require(isinstance(evidence, Mapping), f"retention evidence missing: {trajectory_id}")
    body = {key: value for key, value in evidence.items() if key != "retention_evidence_sha256"}
    _require(
        evidence.get("retention_evidence_sha256") == _canonical_sha256(body),
        f"retention evidence hash drifted: {trajectory_id}",
    )
    _require(evidence.get("trajectory_id") == trajectory_id, f"retention trajectory drifted: {trajectory_id}")
    _require(evidence.get("exact_aligned_40_input_frames_at_10_hz") is True, f"retention alignment failed: {trajectory_id}")
    _require(
        evidence.get("elapsed_time_basis")
        == "scenario_realized_trace_frame_id_join",
        f"retention elapsed-time basis drifted: {trajectory_id}",
    )
    trace_path = Path(str(evidence.get("realized_trace_path", ""))).resolve()
    _require(
        trace_path.is_file()
        and _sha256_file(trace_path) == evidence.get("realized_trace_sha256"),
        f"retention realized trace drifted: {trajectory_id}",
    )
    trace = pd.read_csv(trace_path)
    _require(
        {"frame_id", "elapsed_s"}.issubset(trace.columns)
        and not trace["frame_id"].astype(int).duplicated().any(),
        f"retention realized trace is invalid: {trajectory_id}",
    )
    trace_by_frame = trace.set_index(trace["frame_id"].astype(int), drop=False)
    roles = evidence.get("roles")
    _require(isinstance(roles, Mapping) and set(roles) == {"helper", "recipient"}, f"retention roles drifted: {trajectory_id}")
    frame_sets = []
    timestamps_by_role: dict[str, list[float]] = {}
    elapsed_by_role: dict[str, list[float]] = {}
    for role in ("helper", "recipient"):
        item = roles[role]
        frames = [int(value) for value in item.get("retained_input_frame_ids", [])]
        _require(
            int(item.get("retained_input_frame_count", -1)) == 40
            and len(frames) == 40
            and len(set(frames)) == 40
            and all(right - left == 1 for left, right in zip(frames, frames[1:])),
            f"{role} retention is not an exact 40-frame window: {trajectory_id}",
        )
        _require(
            abs(_finite("retention span", item.get("measured_window_span_s")) - 3.9)
            <= 1e-6,
            f"{role} retained window is not 3.9 seconds: {trajectory_id}",
        )
        metrics_path = Path(str(item.get("metrics_path", ""))).resolve()
        _require(
            metrics_path.is_file()
            and _sha256_file(metrics_path) == item.get("metrics_sha256"),
            f"{role} retention metrics drifted: {trajectory_id}",
        )
        role_dir = metrics_path.parent.parent
        disk_frames = sorted(
            int(path.name.split("_")[1])
            for path in (role_dir / "retained_inputs").glob("frame_*_inputs.npz")
        )
        _require(
            disk_frames == frames,
            f"{role} retained-input membership drifted: {trajectory_id}",
        )
        metrics = pd.read_csv(metrics_path)
        metric_frame_ids = metrics["frame_id"].astype(int)
        _require(
            not metric_frame_ids.duplicated().any(),
            f"{role} metrics contain duplicate frame IDs: {trajectory_id}",
        )
        metrics_by_frame = metrics.set_index(metric_frame_ids, drop=False)
        _require(
            all(frame in metrics_by_frame.index for frame in frames)
            and all(frame in trace_by_frame.index for frame in frames),
            f"{role} retained frame lacks causal time evidence: {trajectory_id}",
        )
        timestamps = [
            float(metrics_by_frame.loc[frame]["carla_timestamp"])
            for frame in frames
        ]
        elapsed = [float(trace_by_frame.loc[frame]["elapsed_s"]) for frame in frames]
        _require(
            all(
                abs((right - left) - 0.1) <= 1e-6
                for left, right in zip(timestamps, timestamps[1:])
            )
            and all(
                abs((right - left) - 0.1) <= 1e-6
                for left, right in zip(elapsed, elapsed[1:])
            )
            and abs(float(item.get("first_episode_relative_s")) - elapsed[0])
            <= 1e-9
            and abs(float(item.get("last_episode_relative_s")) - elapsed[-1])
            <= 1e-9,
            f"{role} retained timing is not recomputable: {trajectory_id}",
        )
        timestamps_by_role[role] = timestamps
        elapsed_by_role[role] = elapsed
        frame_sets.append(frames)
    _require(frame_sets[0] == frame_sets[1], f"retained role frame IDs differ: {trajectory_id}")
    _require(
        _finite("maximum_pair_timestamp_error_s", evidence.get("maximum_pair_timestamp_error_s"))
        <= 1e-9,
        f"retained role timestamps differ: {trajectory_id}",
    )
    helper_timestamps = timestamps_by_role["helper"]
    recipient_timestamps = timestamps_by_role["recipient"]
    helper_elapsed = elapsed_by_role["helper"]
    recipient_elapsed = elapsed_by_role["recipient"]
    _require(
        max(
            abs(left - right)
            for left, right in zip(helper_timestamps, recipient_timestamps)
        )
        <= 1e-9
        and helper_elapsed == recipient_elapsed,
        f"retained role clocks differ: {trajectory_id}",
    )
    if realized_onset_s is None:
        _require(
            evidence.get("realized_onset_status")
            == "not_applicable_matched_benign"
            and "realized_hazard_onset_s" not in evidence,
            f"benign retention fabricates an onset: {trajectory_id}",
        )
    else:
        _require(
            evidence.get("realized_onset_status") == "measured_positive"
            and abs(
                _finite("retention realized onset", evidence.get("realized_hazard_onset_s"))
                - realized_onset_s
            )
            <= 1e-9
            and _finite(
                "post-realized-onset span",
                evidence.get("measured_post_realized_onset_span_s"),
            )
            >= 2.8 - 1e-9,
            f"positive retention does not cover realized onset: {trajectory_id}",
        )
        _require(
            abs(
                float(evidence["measured_pre_realized_onset_span_s"])
                - (realized_onset_s - recipient_elapsed[0])
            )
            <= 1e-9
            and abs(
                float(evidence["measured_post_realized_onset_span_s"])
                - (recipient_elapsed[-1] - realized_onset_s)
            )
            <= 1e-9,
            f"positive retention spans are not trace-recomputable: {trajectory_id}",
        )


def _validate_batch_input_evidence(
    evidence: object,
    *,
    factor_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(isinstance(evidence, Mapping), "batch input evidence is missing")
    snapshot = evidence.get("batch_manifest_prepostflight")
    _require(isinstance(snapshot, Mapping), "pre-postflight batch snapshot is missing")
    _require(
        evidence.get("batch_manifest_prepostflight_sha256")
        == _canonical_sha256(snapshot),
        "pre-postflight batch snapshot hash drifted",
    )
    rows = snapshot.get("trajectories")
    _require(
        isinstance(rows, list)
        and len(rows) == 16
        and all(item.get("status") == "complete" for item in rows),
        "pre-postflight batch snapshot is not exact-16 complete",
    )
    for prefix in ("raw_plan", "resolved_config"):
        path = Path(str(evidence.get(f"{prefix}_path", ""))).resolve()
        _require(
            path.is_file() and _sha256_file(path) == evidence.get(f"{prefix}_sha256"),
            f"{prefix} evidence drifted",
        )
    resolved_path = Path(str(evidence["resolved_config_path"])).resolve()
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    expected_collision_contract = {
        "minimum_static_collision_horizontal_impulse": float(
            resolved["ambient_traffic"]["traffic_sanity_gate"]
            ["minimum_static_collision_horizontal_impulse"]
        ),
        "same_pair_incident_separation_frames_strictly_greater_than": 10,
        "actor_to_actor_contact_rule": "other_actor_id_gt_zero",
        "static_contact_rule": "static_type_and_minimum_horizontal_impulse",
    }
    _require(
        evidence.get("collision_relevance_contract")
        == expected_collision_contract,
        "collision relevance contract differs from resolved audit config",
    )
    _require(
        evidence.get("factor_plan_sha256") == _canonical_sha256(factor_plan),
        "materialized factor-plan evidence drifted",
    )
    return snapshot


def _validate_structural_capture_evidence(
    evidence: object,
    *,
    trajectory_id: str,
    batch_record: Mapping[str, Any],
    collision_relevance_contract: Mapping[str, Any],
) -> None:
    _require(
        isinstance(evidence, Mapping),
        f"structural capture evidence missing: {trajectory_id}",
    )
    body = {
        key: value
        for key, value in evidence.items()
        if key != "structural_capture_sha256"
    }
    _require(
        evidence.get("structural_capture_sha256") == _canonical_sha256(body),
        f"structural capture evidence hash drifted: {trajectory_id}",
    )
    traffic = evidence.get("traffic_sanity")
    verification = evidence.get("trajectory_verification")
    _require(
        evidence.get("trajectory_id") == trajectory_id
        and evidence.get("structural_capture_pass") is True
        and isinstance(traffic, Mapping)
        and traffic == batch_record.get("traffic_sanity")
        and traffic.get("pass") is True
        and int(traffic.get("collision_events", -1)) == 0,
        f"traffic/collision structural gate failed: {trajectory_id}",
    )
    _require(
        isinstance(verification, Mapping)
        and verification == batch_record.get("trajectory_verification")
        and verification.get("pass") is True,
        f"trajectory verification structural gate failed: {trajectory_id}",
    )
    gates = evidence.get("matched_pair_gates")
    _require(
        evidence.get("collision_relevance_contract")
        == collision_relevance_contract,
        f"collision relevance contract drifted: {trajectory_id}",
    )
    gate_names = {
        "matched_pair_initial_realization_gate",
        "matched_pair_owned_nontreatment_gate",
        "matched_pair_static_environment_gate",
        "matched_pair_full_trajectory_gate",
    }
    _require(
        isinstance(gates, Mapping) and set(gates) == gate_names,
        f"matched-pair structural gates are incomplete: {trajectory_id}",
    )
    for name in sorted(gate_names):
        _require(
            isinstance(gates[name], Mapping)
            and gates[name] == batch_record.get(name)
            and gates[name].get("pass") is True,
            f"matched-pair structural gate failed: {name}: {trajectory_id}",
        )
    artifacts = evidence.get("traffic_sanity_artifacts")
    expected_names = {
        "traffic_sanity_summary.json",
        "npc_collision_events.csv",
        "npc_trajectories.csv",
        "ambient_actor_trajectories.csv",
    }
    _require(
        isinstance(artifacts, Mapping) and set(artifacts) == expected_names,
        f"traffic-sanity artifact evidence is incomplete: {trajectory_id}",
    )
    parent: Path | None = None
    for name in sorted(expected_names):
        item = artifacts[name]
        _require(isinstance(item, Mapping), f"traffic artifact evidence invalid: {name}")
        path = Path(str(item.get("path", ""))).resolve()
        if parent is None:
            parent = path.parent
        _require(
            path.parent == parent
            and path.name == name
            and path.is_file()
            and path.stat().st_size == int(item.get("bytes", -1))
            and _sha256_file(path) == item.get("sha256"),
            f"traffic-sanity artifact drifted: {trajectory_id}: {name}",
        )
    assert parent is not None
    summary = json.loads((parent / "traffic_sanity_summary.json").read_text(encoding="utf-8"))
    _require(
        summary == dict(traffic),
        f"traffic-sanity summary is not batch-recomputable: {trajectory_id}",
    )
    with (parent / "npc_collision_events.csv").open("r", encoding="utf-8", newline="") as stream:
        collision_rows = list(csv.DictReader(stream))
    relevant_rows = []
    ignored_rows = 0
    for row in collision_rows:
        try:
            frame_id = int(float(row.get("frame_id", -1)))
            first = int(float(row.get("npc_actor_id", -1)))
            second = int(float(row.get("other_actor_id", -1)))
            impulse_x = float(row.get("normal_impulse_x", 0.0) or 0.0)
            impulse_y = float(row.get("normal_impulse_y", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"collision-event artifact has invalid numeric fields: {trajectory_id}"
            ) from exc
        other_type = str(row.get("other_type_id", "") or "")
        static_impulse_gate = float(
            collision_relevance_contract[
                "minimum_static_collision_horizontal_impulse"
            ]
        )
        relevant = second > 0 or (
            other_type.startswith("static.")
            and math.hypot(impulse_x, impulse_y) >= static_impulse_gate
        )
        if not relevant:
            ignored_rows += 1
            continue
        relevant_rows.append(
            {
                "frame_id": frame_id,
                "pair_low": min(first, second),
                "pair_high": max(first, second),
                "other_type_id": other_type,
                "owner_scope": (
                    "ambient_npc"
                    if "contact_owner_scope" not in row
                    else "unknown"
                    if row.get("contact_owner_scope") is None
                    else str(row["contact_owner_scope"])
                ),
            }
        )
    relevant_rows.sort(
        key=lambda row: (
            row["owner_scope"], row["pair_low"], row["pair_high"],
            row["other_type_id"], row["frame_id"],
        )
    )
    last_frame_by_key: dict[tuple[object, ...], int] = {}
    incident_counts: dict[str, int] = {}
    incident_count = 0
    for row in relevant_rows:
        key = (
            row["owner_scope"], row["pair_low"], row["pair_high"],
            row["other_type_id"],
        )
        previous = last_frame_by_key.get(key)
        if previous is None or int(row["frame_id"]) - previous > int(
            collision_relevance_contract[
                "same_pair_incident_separation_frames_strictly_greater_than"
            ]
        ):
            incident_count += 1
            scope = str(row["owner_scope"])
            incident_counts[scope] = incident_counts.get(scope, 0) + 1
        last_frame_by_key[key] = int(row["frame_id"])
    _require(
        len(collision_rows) == int(traffic.get("collision_callback_rows", -1))
        and ignored_rows == int(traffic.get("ignored_static_contact_rows", -1))
        and incident_count == int(traffic.get("collision_events", -1)) == 0
        and incident_counts == dict(traffic.get("collision_events_by_owner_scope", {})),
        f"collision-event artifact contradicts the registered relevance/dedup gate: {trajectory_id}",
    )


def _validate_artifact_evidence(
    record: Mapping[str, Any],
    *,
    batch_trajectory_record: Mapping[str, Any],
) -> None:
    trajectory_id = str(record["trajectory_id"])
    evidence = record.get("artifact_evidence")
    _require(isinstance(evidence, Mapping), f"artifact evidence missing: {trajectory_id}")
    _require(
        record.get("artifact_manifest_sha256") == _canonical_sha256(evidence),
        f"artifact evidence digest drifted: {trajectory_id}",
    )
    factor_path = Path(str(evidence.get("factor_realization_path", ""))).resolve()
    _require(
        factor_path.is_file()
        and _sha256_file(factor_path) == evidence.get("factor_realization_sha256"),
        f"factor artifact bytes drifted: {trajectory_id}",
    )
    postflight_path = Path(
        str(evidence.get("postflight_artifact_path", ""))
    ).resolve()
    _require(
        postflight_path.is_file()
        and _sha256_file(postflight_path)
        == evidence.get("postflight_artifact_sha256"),
        f"per-trajectory postflight artifact bytes drifted: {trajectory_id}",
    )
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    _require(
        postflight.get("postflight_sha256") == evidence.get("postflight_sha256")
        and postflight.get("postflight_sha256")
        == _canonical_sha256(
            {
                key: value
                for key, value in postflight.items()
                if key != "postflight_sha256"
            }
        ),
        f"per-trajectory postflight self hash drifted: {trajectory_id}",
    )
    _require(
        evidence.get("batch_trajectory_record_sha256")
        == _canonical_sha256(batch_trajectory_record),
        f"batch trajectory record evidence drifted: {trajectory_id}",
    )
    fingerprints = evidence.get("input_fingerprints")
    _require(isinstance(fingerprints, Mapping) and fingerprints, f"input fingerprints missing: {trajectory_id}")
    for raw_path, expected_sha in fingerprints.items():
        path = Path(str(raw_path)).resolve()
        _require(
            path.is_file() and _sha256_file(path) == expected_sha,
            f"postflight input bytes drifted: {trajectory_id}: {path}",
        )
    manifests = evidence.get("collector_artifact_manifests")
    _require(
        isinstance(manifests, Mapping) and set(manifests) == {"helper", "recipient"},
        f"collector artifact manifests missing: {trajectory_id}",
    )
    for role in ("helper", "recipient"):
        sealed = manifests[role]
        _require(
            isinstance(sealed, Mapping)
            and sealed.get("source_role") == role
            and sealed.get("all_listed_files_rehashed_equal") is True,
            f"{role} artifact-manifest verification is invalid: {trajectory_id}",
        )
        manifest_path = Path(str(sealed.get("artifact_manifest_path", ""))).resolve()
        _require(
            manifest_path.is_file()
            and _sha256_file(manifest_path)
            == sealed.get("artifact_manifest_sha256"),
            f"{role} artifact manifest bytes drifted: {trajectory_id}",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files")
        _require(isinstance(entries, list) and entries, f"{role} artifact manifest is empty: {trajectory_id}")
        verified = []
        role_dir = manifest_path.parent
        for item in entries:
            relative = Path(str(item.get("path", "")))
            _require(
                not relative.is_absolute() and ".." not in relative.parts,
                f"{role} artifact path escapes role directory: {trajectory_id}",
            )
            path = (role_dir / relative).resolve()
            try:
                path.relative_to(role_dir.resolve())
            except ValueError as exc:
                raise ContractError(
                    f"{role} artifact path escapes role directory: {trajectory_id}"
                ) from exc
            _require(path.is_file(), f"{role} artifact is missing: {trajectory_id}: {path}")
            size = path.stat().st_size
            digest = _sha256_file(path)
            _require(
                size == int(item.get("bytes", -1)) and digest == item.get("sha256"),
                f"{role} artifact bytes drifted: {trajectory_id}: {path}",
            )
            verified.append({"path": str(relative), "bytes": size, "sha256": digest})
        _require(
            int(sealed.get("listed_file_count", -1)) == len(verified)
            and int(sealed.get("listed_total_bytes", -1))
            == sum(item["bytes"] for item in verified)
            and sealed.get("listed_entries_sha256") == _canonical_sha256(verified),
            f"{role} collector-tree manifest summary drifted: {trajectory_id}",
        )
    _require(
        evidence.get("retention_evidence_sha256")
        == record["retention_window_evidence"]["retention_evidence_sha256"],
        f"retention evidence is not bound into artifact evidence: {trajectory_id}",
    )
    _require(
        evidence.get("structural_capture_sha256")
        == record["structural_capture_gates"]["structural_capture_sha256"],
        f"structural capture evidence is not bound into artifact evidence: {trajectory_id}",
    )


def validate_results(
    result: Mapping[str, Any], config: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an immutable post-capture bundle for atomic calibration admission."""

    _require(result.get("schema") == RESULT_SCHEMA, "unsupported result-bundle schema")
    result_body = {
        key: value for key, value in result.items() if key != "result_bundle_sha256"
    }
    _require(
        result.get("result_bundle_sha256") == _canonical_sha256(result_body),
        "result-bundle self hash is inconsistent",
    )
    _require(result.get("stage_id") == config["stage_id"], "result stage ID drifted")
    _require(
        result.get("source_manifest_sha256") == plan["source_manifest_sha256"],
        "result source-manifest hash drifted",
    )
    _require(result.get("plan_sha256") == plan["plan_sha256"], "result plan hash drifted")
    _require(
        result.get("policy_feature_contract_sha256")
        == plan["policy_feature_contract_sha256"],
        "result policy-feature contract hash drifted",
    )
    _require(result.get("warnings_actuated") is False, "warnings_actuated must be explicit false")
    _require(result.get("oai_executed") is False, "oai_executed must be explicit false")
    _require(
        result.get("downstream_stage_chained") is False,
        "downstream_stage_chained must be explicit false",
    )
    _require(
        result.get("atomic_exact_trajectory_count") == 16
        and result.get("partial_admission") is False,
        "result does not declare exact atomic admission",
    )
    _require(
        result.get("policy_action_selected_from_features") is False
        and result.get("policy_performance_evaluated") is False
        and result.get("observed_policy_state_complete") is False,
        "result overclaims policy selection, performance, or observed state",
    )
    _validate_exact_feature_projection(result.get("policy_feature_projection", {}), config)
    _validate_dependency_fingerprints(result.get("dependency_fingerprints"), config)
    batch_snapshot = _validate_batch_input_evidence(
        result.get("batch_input_evidence"), factor_plan=plan
    )
    batch_snapshot_rows = {
        str(item.get("trajectory_id")): item
        for item in batch_snapshot["trajectories"]
    }

    records = result.get("trajectories")
    _require(isinstance(records, list), "result trajectories must be a list")
    _require(len(records) == int(plan["trajectory_count"]), "result trajectory count drifted")
    record_by_id = {str(record.get("trajectory_id")): record for record in records}
    _require(len(record_by_id) == len(records), "duplicate result trajectory ID")
    plan_by_id = {row["trajectory_id"]: row for row in plan["rows"]}
    _require(set(record_by_id) == set(plan_by_id), "result trajectory IDs drifted")

    contexts: dict[str, dict[str, str]] = {}
    endpoint_status_counts: dict[str, int] = {}
    policy_audit_evidence: list[dict[str, Any]] = []
    policy_audit_records: list[Mapping[str, Any]] = []
    guardrail_reports: list[dict[str, Any]] = []
    capture_model_shas: set[str] = set()
    for trajectory_id, row in plan_by_id.items():
        record = record_by_id[trajectory_id]
        _require(record.get("trajectory_row_sha256") == row["trajectory_row_sha256"], f"row hash mismatch: {trajectory_id}")
        artifact_sha = str(record.get("artifact_manifest_sha256", ""))
        _require(ROW_HASH_RE.fullmatch(artifact_sha) is not None, f"artifact hash invalid: {trajectory_id}")
        _require(
            trajectory_id in batch_snapshot_rows,
            f"trajectory absent from pre-postflight batch snapshot: {trajectory_id}",
        )
        _validate_artifact_evidence(
            record,
            batch_trajectory_record=batch_snapshot_rows[trajectory_id],
        )
        _validate_structural_capture_evidence(
            record.get("structural_capture_gates"),
            trajectory_id=trajectory_id,
            batch_record=batch_snapshot_rows[trajectory_id],
            collision_relevance_contract=result["batch_input_evidence"]
            ["collision_relevance_contract"],
        )
        _require(
            record["artifact_evidence"].get("dependency_fingerprints")
            == result.get("dependency_fingerprints"),
            f"row dependency fingerprints drifted: {trajectory_id}",
        )
        _require(record.get("group_id") == row["group_id"], f"group mismatch: {trajectory_id}")
        _require(record.get("scenario_role") == row["scenario_role"], f"role mismatch: {trajectory_id}")
        context_sha = str(record.get("nontreatment_plan_sha256", ""))
        _require(ROW_HASH_RE.fullmatch(context_sha) is not None, f"context hash invalid: {trajectory_id}")
        contexts.setdefault(row["group_id"], {})[row["scenario_role"]] = context_sha

        requested = record.get("requested_factors")
        _require(isinstance(requested, Mapping), f"requested factors missing: {trajectory_id}")
        expected_requested = row["requested_factor_contract"]
        _require(dict(requested) == expected_requested, f"requested factors drifted: {trajectory_id}")
        capture_model_shas.add(
            _validate_capture_model_identity(
                record.get("capture_model_identity"), trajectory_id
            )
        )

        policy_audit = record.get("causal_policy_audit")
        _require(isinstance(policy_audit, Mapping), f"causal policy audit missing: {trajectory_id}")
        policy_audit_evidence.append(
            _validate_policy_audit(
                policy_audit,
                trajectory_id=trajectory_id,
                config=config,
            )
        )
        policy_audit_records.append(policy_audit)
        availability = record.get("recipient_availability_provenance")
        _require(
            isinstance(availability, Mapping),
            f"recipient availability provenance missing: {trajectory_id}",
        )
        try:
            validate_availability_record(availability)
        except (TypeError, ValueError, KeyError) as exc:
            raise ContractError(
                f"recipient availability provenance invalid: {trajectory_id}: {exc}"
            ) from exc
        _require(
            availability.get("trajectory_id") == trajectory_id,
            f"availability trajectory ID drifted: {trajectory_id}",
        )
        _require(
            availability.get("transport_mode") == "local_loopback"
            and availability.get("oai_executed") is False,
            f"availability transport scope drifted: {trajectory_id}",
        )
        truth_matches = record.get("evaluation_truth_match_by_attempt_id", {})
        _require(
            isinstance(truth_matches, Mapping),
            f"truth-match diagnostic is not a mapping: {trajectory_id}",
        )
        recomputed_guardrail = analyze_installed_track_guardrails(
            availability,
            evaluation_truth_match_by_attempt_id=truth_matches,
        )
        observed_guardrail = record.get("installed_track_guardrails")
        _require(
            isinstance(observed_guardrail, Mapping)
            and dict(observed_guardrail) == recomputed_guardrail,
            f"installed-track guardrails are not recomputable: {trajectory_id}",
        )
        _require(
            bool(recomputed_guardrail["structural_pass"]),
            f"installed-track structural integrity failed: {trajectory_id}",
        )
        guardrail_reports.append(recomputed_guardrail)

        if row["controlled_hazard_present"]:
            realized = record.get("realized_factors")
            _require(isinstance(realized, Mapping), f"realized factors missing: {trajectory_id}")
            for field in config["factor_contract"]["realized_metrics_required"]:
                _require(field in realized, f"realized {field} missing: {trajectory_id}")
            onset_at = _finite("realized_hazard_onset_s", realized["realized_hazard_onset_s"])
            _validate_retention_window_evidence(
                record.get("retention_window_evidence"),
                trajectory_id=trajectory_id,
                realized_onset_s=onset_at,
            )
            helper_speed = _finite(
                "realized_helper_speed_mps", realized["realized_helper_speed_mps"]
            )
            recipient_speed = _finite(
                "realized_recipient_speed_mps", realized["realized_recipient_speed_mps"]
            )
            hazard_speed = _finite(
                "realized_hazard_actor_speed_mps",
                realized["realized_hazard_actor_speed_mps"],
            )
            onset_driver_speed = _finite(
                "realized_onset_driver_speed_mps",
                realized["realized_onset_driver_speed_mps"],
            )
            _require(onset_at >= 0.0, f"realized onset is negative: {trajectory_id}")
            _require(helper_speed >= 0.0, f"helper speed is negative: {trajectory_id}")
            _require(recipient_speed >= 0.0, f"recipient speed is negative: {trajectory_id}")
            _require(hazard_speed >= 0.0, f"hazard speed is negative: {trajectory_id}")
            _require(
                onset_driver_speed >= expected_requested["minimum_onset_driver_speed_mps"],
                f"onset driver did not reach the measurement floor: {trajectory_id}",
            )
            if row["hazard_class"] == "pedestrian":
                pedestrian_range = config["factor_contract"][
                    "controlled_pedestrian_speed_range_mps"
                ]
                _require(
                    float(pedestrian_range[0]) <= hazard_speed <= float(pedestrian_range[1]),
                    f"pedestrian speed is outside the physical contract: {trajectory_id}",
                )
            closing = _finite(
                "pre_intervention_radial_closing_speed_mps",
                realized["pre_intervention_radial_closing_speed_mps"],
            )
            tth = _finite(
                "pre_intervention_hazard_proximity_horizon_s",
                realized["pre_intervention_hazard_proximity_horizon_s"],
            )
            clearance = _finite(
                "pre_intervention_minimum_surface_clearance_m",
                realized["pre_intervention_minimum_surface_clearance_m"],
            )
            _require(
                expected_requested["requested_closing_speed_band_min_mps"]
                <= closing
                <= expected_requested["requested_closing_speed_band_max_mps"],
                f"realized closing speed is outside its band: {trajectory_id}",
            )
            _require(
                expected_requested["requested_proximity_horizon_band_min_s"]
                <= tth
                <= expected_requested["requested_proximity_horizon_band_max_s"],
                f"realized proximity horizon is outside its band: {trajectory_id}",
            )
            maximum_clearance = float(
                config["factor_contract"]["positive_hazard_surface_clearance_max_m_by_class"]
                [row["hazard_class"]]
            )
            _require(
                0.0 <= clearance <= maximum_clearance,
                "predicted conflict-surface clearance is outside registered "
                f"proximity gate: {trajectory_id}",
            )
            for basis_field in (
                "geometry_measurement_basis",
                "closing_speed_measurement_basis",
                "proximity_horizon_measurement_basis",
            ):
                _require(
                    realized[basis_field] == expected_requested[basis_field],
                    f"realized {basis_field} differs from its design row: {trajectory_id}",
                )
            endpoint = record.get("installed_track_endpoint")
            _require(isinstance(endpoint, Mapping), f"installed-track endpoint missing: {trajectory_id}")
            map_target_matches = record.get(
                "evaluation_recipient_map_target_matches"
            )
            _require(
                isinstance(map_target_matches, list),
                f"recipient-map target-match evidence missing: {trajectory_id}",
            )
            _validate_endpoint(endpoint, availability, map_target_matches, config)
            endpoint_status = str(endpoint["endpoint_status"])
            endpoint_status_counts[endpoint_status] = endpoint_status_counts.get(endpoint_status, 0) + 1
        else:
            _validate_retention_window_evidence(
                record.get("retention_window_evidence"),
                trajectory_id=trajectory_id,
                realized_onset_s=None,
            )
            _require(bool(record.get("registered_target_absent")), f"benign target is not absent: {trajectory_id}")
            _require(
                record.get("realized_factors_status")
                == "not_applicable_matched_benign_registered_target_absent",
                f"benign realized-factor status is not typed: {trajectory_id}",
            )
            _require(
                "realized_factors" not in record,
                f"benign row fabricates realized hazard factors: {trajectory_id}",
            )
            _require(
                record.get("factor_reference_trajectory_id")
                == trajectory_id.removesuffix("_ben") + "_pos",
                f"benign factor reference drifted: {trajectory_id}",
            )
            _require(
                "installed_track_endpoint" not in record,
                f"benign row fabricates a registered-target endpoint: {trajectory_id}",
            )

    for group_id, pair_contexts in contexts.items():
        _require(
            set(pair_contexts)
            == {"controlled_positive_occlusion", "matched_benign_negative"},
            f"pair context roles missing: {group_id}",
        )
        _require(len(set(pair_contexts.values())) == 1, f"nontreatment plan differs in pair: {group_id}")

    _require(
        len(capture_model_shas) == 1,
        "checkpoint bytes differ across the exact-16 frozen M-prime tranche",
    )

    code_hashes = {item["consumer_code_sha256"] for item in policy_audit_evidence}
    _require(len(code_hashes) == 1, "policy consumer code changed within the exact tranche")
    projection = result["policy_feature_projection"]
    _require(
        dict(projection) == summarize_policy_audits(policy_audit_records),
        "batch policy projection is not exactly recomputable from per-row audits",
    )
    _require(
        int(projection["trajectory_audit_count"]) == len(policy_audit_evidence),
        "policy projection trajectory audit count drifted",
    )
    _require(
        projection["consumer_code_sha256"] == next(iter(code_hashes)),
        "policy projection code hash differs from per-row audits",
    )
    for stage in config["causal_policy_audit_contract"]["required_stages"]:
        expected_count = sum(item["counts"][stage] for item in policy_audit_evidence)
        _require(
            int(projection[f"{stage}_decision_count"]) == expected_count,
            f"policy projection {stage} count is not recomputable",
        )
    _require(
        result.get("installed_track_quality_guardrails")
        == aggregate_guardrail_reports(guardrail_reports),
        "batch installed-track guardrail aggregate is not recomputable",
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "stage_id": config["stage_id"],
        "verdict": "PASS_ATOMIC_EXACT_16_ADMITTED",
        "trajectory_count": len(records),
        "group_count": len(contexts),
        "endpoint_status_counts": endpoint_status_counts,
        "admission_scope": "these_exact_16_replicate_0_rows_only",
        "next_action": "human_review_before_any_additional_calibration_tranche",
        "oai_status": "not_run_not_claimed",
        "warning_selection_status": "not_run_not_authorized",
    }
    summary["validated_result_sha256"] = _canonical_sha256(result)
    return summary


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--validate-results", type=Path)
    parser.add_argument("--write-summary", type=Path)
    parser.add_argument(
        "--require-runtime-ready",
        action="store_true",
        help="fail unless a later version has verified every runtime adapter",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        plan = build_plan(config)
        if args.validate_results is not None:
            result = json.loads(args.validate_results.read_text(encoding="utf-8"))
            summary = validate_results(result, config, plan)
            if args.write_summary is not None:
                _write_create_only(args.write_summary, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.write_plan is not None:
            _write_create_only(args.write_plan, plan)
        summary = {
            "schema": SUMMARY_SCHEMA,
            "stage_id": config["stage_id"],
            "verdict": (
                "PASS_OFFLINE_DESIGN_COLLECTION_BLOCKED"
                if not plan["runtime_ready"]
                else "PASS_OFFLINE_DESIGN_RUNTIME_READY_REQUIRES_SEPARATE_LAUNCH_AUTHORITY"
            ),
            "trajectory_count": plan["trajectory_count"],
            "group_count": plan["group_count"],
            "plan_sha256": plan["plan_sha256"],
            "runtime_ready": plan["runtime_ready"],
            "runtime_blockers": plan["runtime_blockers"],
            "collection_authorized_by_this_tool": False,
            "next_action": (
                "run_hash_bound_manual_eight_corner_review_no_exact16_launch_yet"
                if plan["runtime_ready"]
                else "implement_and_review_runtime_adapters_without_launching_carla"
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.require_runtime_ready and not plan["runtime_ready"]:
            return 2
        return 0
    except (
        ContractError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        failure = {
            "schema": SUMMARY_SCHEMA,
            "stage_id": "phase2_factor_realization_smoke_v1",
            "verdict": "FAIL_HOLD_NO_COLLECTION",
            "error": f"{type(exc).__name__}: {exc}",
            "next_action": "repair_contract_before_any_carla_launch",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

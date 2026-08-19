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

import yaml


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


def _event_time(event: Mapping[str, Any], name: str) -> float | None:
    status = str(event.get("status", ""))
    _require(status in {"event", "censored", "miss"}, f"{name}.status is invalid")
    if status == "event":
        return _finite(f"{name}.at_s", event.get("at_s"))
    _finite(f"{name}.censor_at_s", event.get("censor_at_s"))
    return None


def _validate_endpoint(endpoint: Mapping[str, Any], config: Mapping[str, Any]) -> None:
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
    recipient = endpoint.get("recipient_own_confirmation")
    _require(isinstance(helper, Mapping), "helper source event is required")
    _require(isinstance(install, Mapping), "recipient install event is required")
    _require(isinstance(recipient, Mapping), "recipient own event is required")
    helper_at = _event_time(helper, "helper_source_confirmation")
    available_at = _event_time(install, "helper_track_recipient_install")
    recipient_at = _event_time(recipient, "recipient_own_confirmation")

    if available_at is not None:
        for field in contract["required_install_provenance"]:
            _require(field in install, f"recipient install is missing {field}")
        for field in ("contribution_id", "source_track_id", "recipient_map_track_id"):
            _require(str(install[field]).strip(), f"recipient install {field} is empty")
        published = _finite("published_at_s", install["published_at_s"])
        installed = _finite("installed_at_s", install["installed_at_s"])
        available = _finite("available_at_s", install["available_at_s"])
        _require(abs(available - available_at) <= 1e-9, "install event at_s differs from available_at_s")
        _require(published <= installed <= available, "publish/install/available order is invalid")
        if helper_at is not None:
            _require(helper_at <= published, "helper confirmation occurs after publication")

    if status == "numeric":
        _require(helper_at is not None and available_at is not None and recipient_at is not None, "numeric endpoint lacks events")
        expected_margin = recipient_at - available_at
        observed_margin = _finite("recipient_available_confirmed_track_margin_s", endpoint.get("recipient_available_confirmed_track_margin_s"))
        _require(abs(expected_margin - observed_margin) <= 1e-9, "numeric endpoint margin is inconsistent")
    elif status == "ego_right_censored":
        _require(available_at is not None and recipient_at is None, "ego censoring has inconsistent events")
        lower_bound = _finite("recipient_available_confirmed_track_margin_lower_bound_s", endpoint.get("recipient_available_confirmed_track_margin_lower_bound_s"))
        _require(abs(lower_bound - (horizon - available_at)) <= 1e-9, "censored lower bound is inconsistent")
    elif status == "cooperative_miss":
        _require(available_at is None, "cooperative miss contains an install event")
    elif status == "both_miss":
        _require(available_at is None and recipient_at is None, "both_miss contains an event")


def validate_results(
    result: Mapping[str, Any], config: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an immutable post-capture bundle for atomic calibration admission."""

    _require(result.get("schema") == RESULT_SCHEMA, "unsupported result-bundle schema")
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
    _validate_exact_feature_projection(result.get("policy_feature_projection", {}), config)

    records = result.get("trajectories")
    _require(isinstance(records, list), "result trajectories must be a list")
    _require(len(records) == int(plan["trajectory_count"]), "result trajectory count drifted")
    record_by_id = {str(record.get("trajectory_id")): record for record in records}
    _require(len(record_by_id) == len(records), "duplicate result trajectory ID")
    plan_by_id = {row["trajectory_id"]: row for row in plan["rows"]}
    _require(set(record_by_id) == set(plan_by_id), "result trajectory IDs drifted")

    contexts: dict[str, dict[str, str]] = {}
    endpoint_status_counts: dict[str, int] = {}
    for trajectory_id, row in plan_by_id.items():
        record = record_by_id[trajectory_id]
        _require(record.get("trajectory_row_sha256") == row["trajectory_row_sha256"], f"row hash mismatch: {trajectory_id}")
        artifact_sha = str(record.get("artifact_manifest_sha256", ""))
        _require(ROW_HASH_RE.fullmatch(artifact_sha) is not None, f"artifact hash invalid: {trajectory_id}")
        _require(record.get("group_id") == row["group_id"], f"group mismatch: {trajectory_id}")
        _require(record.get("scenario_role") == row["scenario_role"], f"role mismatch: {trajectory_id}")
        context_sha = str(record.get("nontreatment_plan_sha256", ""))
        _require(ROW_HASH_RE.fullmatch(context_sha) is not None, f"context hash invalid: {trajectory_id}")
        contexts.setdefault(row["group_id"], {})[row["scenario_role"]] = context_sha

        requested = record.get("requested_factors")
        _require(isinstance(requested, Mapping), f"requested factors missing: {trajectory_id}")
        expected_requested = row["requested_factor_contract"]
        _require(dict(requested) == expected_requested, f"requested factors drifted: {trajectory_id}")

        if row["controlled_hazard_present"]:
            realized = record.get("realized_factors")
            _require(isinstance(realized, Mapping), f"realized factors missing: {trajectory_id}")
            for field in config["factor_contract"]["realized_metrics_required"]:
                _require(field in realized, f"realized {field} missing: {trajectory_id}")
            onset_at = _finite("realized_hazard_onset_s", realized["realized_hazard_onset_s"])
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
                f"positive is not a non-colliding physical hazard: {trajectory_id}",
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
            _validate_endpoint(endpoint, config)
            endpoint_status = str(endpoint["endpoint_status"])
            endpoint_status_counts[endpoint_status] = endpoint_status_counts.get(endpoint_status, 0) + 1
        else:
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

    for group_id, pair_contexts in contexts.items():
        _require(
            set(pair_contexts)
            == {"controlled_positive_occlusion", "matched_benign_negative"},
            f"pair context roles missing: {group_id}",
        )
        _require(len(set(pair_contexts.values())) == 1, f"nontreatment plan differs in pair: {group_id}")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "stage_id": config["stage_id"],
        "verdict": "PASS_ADMIT_EXACT_BATCH_AS_CALIBRATION_TRANCHE",
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
            "next_action": "implement_and_review_runtime_adapters_without_launching_carla",
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.require_runtime_ready and not plan["runtime_ready"]:
            return 2
        return 0
    except (ContractError, KeyError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
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

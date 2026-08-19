"""Build and validate the deterministic Phase-2 Suite A/B design manifest.

This is an offline design tool. It never imports CARLA, launches a process, or
authorizes collection. The output enumerates independent scenario groups and
their frozen calibration/validation/test assignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml
from scipy.stats import nct, t


CONFIG_SCHEMA_V1 = "scenesense.phase2_suite_design.v1"
CONFIG_SCHEMA_V2 = "scenesense.phase2_suite_design.v2"
MANIFEST_SCHEMA_V1 = "scenesense.phase2_suite_design_manifest.v1"
MANIFEST_SCHEMA_V2 = "scenesense.phase2_suite_design_manifest.v2"
SPLITS = ("calibration", "validation", "test")

FACTOR_REALIZATION_COLUMNS = {
    "factor_realization_status",
    "time_to_hazard_label_status",
    "hazard_actor_role",
    "onset_driver_role",
    "geometry_measurement_basis",
    "closing_speed_measurement_basis",
    "proximity_horizon_measurement_basis",
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
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(master_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) + 1


def _deterministic_multiset(
    counts: Mapping[str, int], *, master_seed: int, namespace: str
) -> list[str]:
    tagged = []
    for value, count in counts.items():
        for occurrence in range(int(count)):
            key = hashlib.sha256(
                f"{master_seed}:{namespace}:{value}:{occurrence}".encode("utf-8")
            ).digest()
            tagged.append((key, str(value)))
    return [value for _key, value in sorted(tagged)]


def paired_t_power(
    *, sample_count: int, effect_s: float, paired_sd_s: float, alpha: float
) -> float:
    if sample_count < 2 or effect_s <= 0 or paired_sd_s <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid paired-power inputs")
    degrees = sample_count - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees))
    noncentrality = float(effect_s / paired_sd_s * math.sqrt(sample_count))
    return float(
        nct.cdf(-critical, degrees, noncentrality)
        + 1.0
        - nct.cdf(critical, degrees, noncentrality)
    )


def _retention_tier(split: str, *, audit: bool, config: Mapping[str, object]) -> str:
    retention = config["retention"]
    if audit:
        return str(retention["audit_tier"])
    return str(retention[f"{split}_tier"])


def _factor_realization_fields(
    config: Mapping[str, object],
    geometry: Mapping[str, object],
    *,
    closing_band: str,
    tth_band: str,
) -> dict[str, object]:
    """Resolve deterministic Suite-A controls for a v2 manifest row.

    The historical ``time_to_hazard_band`` label never had executable
    semantics.  V2 deliberately keeps it as a provisional factor label and
    records a measurable constant-velocity proximity-horizon diagnostic.  It
    is not silently renamed to collision TTC.
    """

    factor = config["suite_a"]["factor_realization"]
    if factor.get("schema") != "scenesense.phase2_factor_realization.v1":
        raise ValueError("unsupported Suite-A factor-realization schema")
    closing_ranges = config["suite_a"]["closing_speed_bands"]
    horizon_ranges = config["suite_a"]["time_to_hazard_bands"]
    if set(factor["requested_recipient_speed_mps_by_closing_speed_band"]) != set(
        closing_ranges
    ):
        raise ValueError("requested recipient speeds do not cover closing-speed bands")
    if set(factor["requested_hazard_onset_s_by_factor_cell"]) != set(
        closing_ranges
    ):
        raise ValueError("requested onset table does not cover closing-speed bands")
    for name, values in factor["requested_hazard_onset_s_by_factor_cell"].items():
        if set(values) != set(horizon_ranges):
            raise ValueError(f"requested onset table is incomplete for {name}")
    geometry_contracts = factor["geometry_contracts"]
    configured_geometry_ids = {
        str(item["geometry_id"]) for item in config["suite_a"]["geometries"]
    }
    if set(geometry_contracts) != configured_geometry_ids:
        raise ValueError("factor realization geometry coverage differs from Suite A")
    identity = str(geometry["geometry_id"])
    if identity not in geometry_contracts:
        raise ValueError(f"factor realization lacks geometry {identity}")
    contract = geometry_contracts[identity]
    expected_contract_keys = {
        "hazard_actor_role",
        "onset_driver_role",
        "geometry_measurement_basis",
        "requested_hazard_actor_speed_mps",
        "requested_onset_driver_speed_mps",
    }
    if set(contract) != expected_contract_keys:
        raise ValueError(
            f"factor geometry contract keys differ for {identity}: "
            f"missing={sorted(expected_contract_keys - set(contract))}, "
            f"extra={sorted(set(contract) - expected_contract_keys)}"
        )
    closing_min, closing_max = (
        float(value) for value in closing_ranges[closing_band]
    )
    horizon_min, horizon_max = (
        float(value) for value in horizon_ranges[tth_band]
    )
    recipient_speed = float(
        factor["requested_recipient_speed_mps_by_closing_speed_band"][closing_band]
    )
    if recipient_speed <= 0.0 or not math.isfinite(recipient_speed):
        raise ValueError("requested recipient speed must be finite and positive")
    hazard_onset_s = float(
        factor["requested_hazard_onset_s_by_factor_cell"][closing_band][tth_band]
    )
    if hazard_onset_s < 0.0:
        raise ValueError("requested hazard onset cannot be negative")
    gate = factor["realized_gate"]
    return {
        "factor_realization_status": str(factor["status"]),
        "time_to_hazard_label_status": str(factor["time_to_hazard_label_status"]),
        "hazard_actor_role": str(contract["hazard_actor_role"]),
        "onset_driver_role": str(contract["onset_driver_role"]),
        "geometry_measurement_basis": str(contract["geometry_measurement_basis"]),
        "closing_speed_measurement_basis": str(
            gate["closing_speed_measurement_basis"]
        ),
        "proximity_horizon_measurement_basis": str(
            gate["proximity_horizon_measurement_basis"]
        ),
        "requested_helper_speed_mps": float(factor["requested_helper_speed_mps"]),
        "requested_recipient_speed_mps": recipient_speed,
        "requested_hazard_actor_speed_mps": float(
            contract["requested_hazard_actor_speed_mps"]
        ),
        "requested_onset_driver_speed_mps": float(
            contract["requested_onset_driver_speed_mps"]
        ),
        "requested_hazard_onset_s": hazard_onset_s,
        "requested_closing_speed_target_mps": 0.5
        * (closing_min + closing_max),
        "requested_closing_speed_band_min_mps": closing_min,
        "requested_closing_speed_band_max_mps": closing_max,
        "requested_proximity_horizon_target_s": 0.5
        * (horizon_min + horizon_max),
        "requested_proximity_horizon_band_min_s": horizon_min,
        "requested_proximity_horizon_band_max_s": horizon_max,
        "minimum_onset_driver_speed_mps": float(
            gate["minimum_onset_driver_speed_mps"]
        ),
    }


def _naturalistic_factor_fields(config: Mapping[str, object]) -> dict[str, object]:
    factor = config["suite_a"]["factor_realization"]
    controls = factor["naturalistic_ego_controls"]
    return {
        "factor_realization_status": "not_applicable_unforced_naturalistic",
        "time_to_hazard_label_status": "not_applicable_unforced_naturalistic",
        "hazard_actor_role": "not_applicable",
        "onset_driver_role": "not_applicable",
        "geometry_measurement_basis": "not_applicable_unforced_naturalistic",
        "closing_speed_measurement_basis": "not_applicable_unforced_naturalistic",
        "proximity_horizon_measurement_basis": "not_applicable_unforced_naturalistic",
        "requested_helper_speed_mps": float(controls["helper_speed_mps"]),
        "requested_recipient_speed_mps": float(controls["recipient_speed_mps"]),
        "requested_hazard_actor_speed_mps": math.nan,
        "requested_onset_driver_speed_mps": math.nan,
        "requested_hazard_onset_s": math.nan,
        "requested_closing_speed_target_mps": math.nan,
        "requested_closing_speed_band_min_mps": math.nan,
        "requested_closing_speed_band_max_mps": math.nan,
        "requested_proximity_horizon_target_s": math.nan,
        "requested_proximity_horizon_band_min_s": math.nan,
        "requested_proximity_horizon_band_max_s": math.nan,
        "minimum_onset_driver_speed_mps": math.nan,
    }


def _power_reference(config: Mapping[str, object]) -> Mapping[str, object]:
    """Return the arithmetic reference without granting endpoint authority.

    V1's warning-lead calculation is preserved for historical reproducibility.
    V2 names the recipient-available endpoint but deliberately registers no
    effect size until its runtime event chain and calibration yields exist.
    """

    power = config["power"]
    if str(config.get("schema_version")) == CONFIG_SCHEMA_V1:
        return power
    if power.get("status") != (
        "no_v2_power_authorization_historical_warning_sensitivity_only"
    ):
        raise ValueError("v2 power status must deny endpoint authorization")
    if power.get("primary_endpoint") != "recipient_available_confirmed_track_margin_s":
        raise ValueError("v2 primary endpoint name drifted")
    if power.get("primary_endpoint_status") != (
        "pending_recipient_install_runtime_factor_smoke_and_calibration_"
        "no_effect_size_registered"
    ):
        raise ValueError("v2 primary endpoint status drifted")
    if power.get("registered_effect_size_s") is not None:
        raise ValueError("v2 must not fabricate a registered effect size")
    historical = power.get("historical_warning_reference")
    if not isinstance(historical, Mapping) or historical.get("status") != (
        "non_authoritative_reference_only"
    ):
        raise ValueError("v2 historical warning reference is not bounded")
    if historical.get("endpoint") != "paired_registered_target_warning_lead_s":
        raise ValueError("v2 historical warning endpoint drifted")
    return historical


def build_manifest(config: Mapping[str, object]) -> pd.DataFrame:
    config_schema = str(config.get("schema_version"))
    if config_schema not in {CONFIG_SCHEMA_V1, CONFIG_SCHEMA_V2}:
        raise ValueError("unsupported suite-design config schema")
    manifest_schema = (
        MANIFEST_SCHEMA_V2 if config_schema == CONFIG_SCHEMA_V2 else MANIFEST_SCHEMA_V1
    )
    if any(bool(value) for value in config["authorization"].values()):
        raise ValueError("suite-design config must not authorize runtime work")

    power = _power_reference(config)
    effect_s = float(power["smallest_effect_s"])
    effect = power["smallest_effect_interpretation"]
    world_hz = float(config["common"]["world_hz"])
    policy_hz = float(config["common"]["policy_control_hz_surrogate"])
    if effect["role"] != "cross_cell_research_floor_not_a_braking_safety_threshold":
        raise ValueError("smallest-effect claim boundary drifted")
    if int(effect["sensor_frame_count"]) != round(effect_s * world_hz):
        raise ValueError("smallest effect no longer equals the declared sensor frames")
    if int(effect["policy_decision_count"]) != round(effect_s * policy_hz):
        raise ValueError("smallest effect no longer equals the declared policy decisions")
    closing_bands = config["suite_a"]["closing_speed_bands"]
    expected_distances = {
        str(name): [effect_s * float(value) for value in values]
        for name, values in closing_bands.items()
    }
    observed_distances = {
        str(name): [float(value) for value in values]
        for name, values in effect["distance_equivalent_m_by_closing_speed_band"].items()
    }
    if observed_distances != expected_distances:
        raise ValueError("smallest-effect closing-distance arithmetic drifted")

    nuisance = config["warning_nuisance_gate"]
    expected_nuisance = {
        "population": "suite_a_matched_benign_negative",
        "timing_endpoint_basis": "registered_target_warning_only",
        "aggregation": {
            "false_warning_active_frame_rate": (
                "sum_false_warning_active_frames_over_sum_eligible_benign_frames"
            ),
            "false_warning_episode_rate_per_minute": (
                "sum_false_warning_episodes_over_sum_eligible_benign_exposure_minutes"
            ),
            "uncertainty_unit": "paired_trajectory_cluster",
            "threshold_basis": (
                "pooled_point_estimate_with_cluster_interval_reported"
            ),
        },
        "adjudicated_false_warning_active_frame_rate_max": 0.10,
        "false_warning_episodes_per_minute_max": 1.0,
        "cooperative_vs_ego_noninferiority_margin_pp": 2.0,
        "apply_to": "every_candidate_retained_for_validation",
        "failure_action": "stop_before_validation",
        "note": "research_usability_gate_not_a_certified_automotive_requirement",
    }
    if config_schema == CONFIG_SCHEMA_V2:
        expected_nuisance.update(
            {
                "status": "historical_failed_secondary_not_blocking_C2",
                "scope": "binding_only_for_any_future_warning_claim",
                "c2_installed_track_endpoint_blocking": False,
                "apply_to": "future_warning_claim_candidates_only",
                "failure_action": (
                    "block_future_warning_claim_not_C2_installed_track_evidence"
                ),
            }
        )
    if nuisance != expected_nuisance:
        raise ValueError("absolute warning-nuisance gate drifted")
    if config_schema == CONFIG_SCHEMA_V2:
        expected_track_guardrails = {
            "status": "two_stage_calibration_contract_no_collection_authority",
            "endpoint": "recipient_available_confirmed_track_margin_s",
            "pre_16_calibration_contract": {
                "status": (
                    "definitions_denominators_and_structural_integrity_gates_"
                    "must_be_frozen"
                ),
                "metric_definitions": {
                    "false_recipient_install_rate": {
                        "numerator": (
                            "recipient_installs_without_registered_source_target_"
                            "correspondence"
                        ),
                        "denominator": "all_recipient_install_attempts",
                    },
                    "duplicate_recipient_install_rate": {
                        "numerator": (
                            "repeated_recipient_installs_for_same_contribution_and_"
                            "source_track"
                        ),
                        "denominator": "all_recipient_installs",
                    },
                    "source_to_recipient_track_fragmentation_rate": {
                        "numerator": (
                            "extra_recipient_track_ids_beyond_one_per_source_track_"
                            "within_episode"
                        ),
                        "denominator": (
                            "source_tracks_with_at_least_one_recipient_install"
                        ),
                    },
                    "recipient_map_pollution_rate": {
                        "numerator": (
                            "recipient_map_tracks_without_valid_local_or_installed_"
                            "source_provenance"
                        ),
                        "denominator": "all_recipient_map_tracks",
                    },
                },
                "structural_integrity_gates": [
                    "every_install_has_unique_contribution_source_and_recipient_track_ids",
                    "publish_install_available_timestamps_are_monotone_on_one_clock",
                    "every_metric_event_is_recomputable_from_immutable_provenance",
                    "zero_missing_denominators_or_untyped_zero_exposure_cases",
                ],
                "failure_action": "block_exact_16_calibration_tranche",
            },
            "numeric_threshold_contract": {
                "status": (
                    "estimate_on_exact_16_then_register_before_additional_collection"
                ),
                "estimation_source": (
                    "exact_16_factor_realization_calibration_tranche"
                ),
                "same_16_research_usability_claim_authorized": False,
                "registration_deadline": (
                    "before_any_additional_calibration_or_validation"
                ),
                "failure_action": "block_additional_calibration_and_validation",
            },
        }
        if config.get("installed_track_quality_guardrails") != expected_track_guardrails:
            raise ValueError("v2 installed-track quality guardrails drifted")

    master_seed = int(config["master_seed"])
    common = config["common"]
    renderer = common.get("renderer_quality")
    expected_renderer = {
        "primary_level": "Epic",
        "required_server_launch_flag": "-quality-level=Epic",
        "provenance_source": "operator_declared_server_launch_flag",
        "rpc_introspection_available": False,
        "existing_stress_level": "Low",
        "future_low_collection_authorized": False,
    }
    if renderer != expected_renderer:
        raise ValueError(
            "Suite A/B renderer contract must lock explicit Epic primary and "
            "keep Low as an existing, non-collecting stress condition"
        )
    raw_window_s = float(config["retention"]["raw_window_duration_s"])
    rows: list[dict] = []

    suite_a = config["suite_a"]
    suite_a_population = suite_a["ambient_population_contract"]
    expected_suite_a_population = {
        "mode": "scenario_owned_only",
        "population_process_required": False,
        "generic_vehicles": 0,
        "generic_walkers": 0,
        "traffic_density_status": "not_applicable",
    }
    if suite_a_population != expected_suite_a_population:
        raise ValueError("Suite A ambient-population contract drifted")
    replicates = int(suite_a["replicates_per_factor_cell"])
    split_by_replicate = {
        split: {int(value) for value in suite_a["split_by_replicate"][split]}
        for split in SPLITS
    }
    for geometry in suite_a["geometries"]:
        for closing_band in suite_a["closing_speed_bands"]:
            for tth_band in suite_a["time_to_hazard_bands"]:
                for replicate in range(replicates):
                    matching = [
                        split
                        for split, indices in split_by_replicate.items()
                        if replicate in indices
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            f"Suite A replicate {replicate} maps to {matching}"
                        )
                    split = matching[0]
                    group_id = (
                        f"sa_{geometry['geometry_id']}_{closing_band}_"
                        f"{tth_band}_r{replicate:02d}"
                    )
                    if config_schema == CONFIG_SCHEMA_V2:
                        audit_cell = geometry["audit_factor_cell"]
                        audit = (
                            split == "calibration"
                            and closing_band == str(audit_cell["closing_speed_band"])
                            and tth_band == str(audit_cell["time_to_hazard_band"])
                        )
                        factor_fields = _factor_realization_fields(
                            config,
                            geometry,
                            closing_band=str(closing_band),
                            tth_band=str(tth_band),
                        )
                    else:
                        audit = (
                            split == "calibration"
                            and closing_band
                            == next(iter(suite_a["closing_speed_bands"]))
                            and tth_band
                            == next(iter(suite_a["time_to_hazard_bands"]))
                        )
                        factor_fields = {}
                    shared = {
                        "schema": manifest_schema,
                        "design_id": config["design_id"],
                        "suite_id": "A",
                        "suite_label": suite_a["label"],
                        "split": split,
                        "group_id": group_id,
                        "matched_pair_id": group_id,
                        "geometry_or_route_id": geometry["geometry_id"],
                        "geometry_or_route_status": geometry["implementation_status"],
                        "hazard_class": geometry["hazard_class"],
                        "closing_speed_band": closing_band,
                        "time_to_hazard_band": tth_band,
                        "traffic_density": "not_applicable",
                        "traffic_density_status": suite_a_population[
                            "traffic_density_status"
                        ],
                        "ambient_population_mode": suite_a_population["mode"],
                        "ambient_population_process_required": int(
                            bool(suite_a_population["population_process_required"])
                        ),
                        "weather": suite_a["weather"],
                        "renderer_quality_level": renderer["primary_level"],
                        "renderer_server_launch_flag": renderer[
                            "required_server_launch_flag"
                        ],
                        "renderer_contract_role": "primary",
                        "carla_seed": _stable_seed(master_seed, f"{group_id}:carla"),
                        "traffic_seed": _stable_seed(master_seed, f"{group_id}:traffic"),
                        "sensor_seed": _stable_seed(master_seed, f"{group_id}:sensor"),
                        "raw_retention_tier": _retention_tier(
                            split, audit=audit, config=config
                        ),
                        "raw_window_duration_s": raw_window_s if split != "test" else 0.0,
                        "raw_window_anchor": (
                            config["retention"]["raw_window_anchor"]
                            if split != "test"
                            else "none"
                        ),
                        "confirmatory_locked": int(split == "test"),
                        "pair_contract_id": "",
                        "route_start_anchor_id": "",
                        "recipient_start_index": "",
                        "helper_start_index": "",
                        "recipient_route_sha256": "",
                        "helper_route_sha256": "",
                        **factor_fields,
                    }
                    for role, present in (
                        ("controlled_positive_occlusion", 1),
                        ("matched_benign_negative", 0),
                    ):
                        rows.append(
                            {
                                **shared,
                                "trajectory_id": f"{group_id}_{'pos' if present else 'ben'}",
                                "scenario_role": role,
                                "controlled_hazard_present": present,
                            }
                        )
    suite_b = config["suite_b"]
    naturalistic_factor_fields = (
        _naturalistic_factor_fields(config)
        if config_schema == CONFIG_SCHEMA_V2
        else {}
    )
    suite_b_population = suite_b["ambient_population_contract"]
    expected_suite_b_population = {
        "mode": "naturalistic_tm",
        "population_process_required": True,
        "traffic_density_status": "realized_nuisance_factor",
    }
    if suite_b_population != expected_suite_b_population:
        raise ValueError("Suite B ambient-population contract drifted")
    repository_root = Path(__file__).resolve().parents[1]
    for route in suite_b["routes"]:
        route_id = str(route["route_id"])
        anchors = list(route["start_anchor_schedule"])
        if not anchors:
            raise ValueError(f"Suite B route has no start anchors: {route_id}")
        anchor_ids = [str(item["anchor_id"]) for item in anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError(f"Suite B route reuses an anchor ID: {route_id}")
        if str(route["implementation_status"]) == "reviewed_visual_route":
            acceptance_path = repository_root / str(
                route["visual_acceptance_record"]
            )
            if not acceptance_path.is_file():
                raise FileNotFoundError(acceptance_path)
            observed_acceptance_hash = _sha256(acceptance_path)
            expected_acceptance_hash = str(
                route["visual_acceptance_record_sha256"]
            )
            if observed_acceptance_hash != expected_acceptance_hash:
                raise ValueError(
                    f"Suite B visual acceptance hash drifted for {route_id}: "
                    f"expected={expected_acceptance_hash}, "
                    f"observed={observed_acceptance_hash}"
                )
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            if (
                str(acceptance.get("route_id")) != route_id
                or bool(acceptance.get("collection_authorized"))
                or str(acceptance.get("pair_contract_id"))
                != str(route["pair_contract_id"])
                or str(acceptance.get("shared_contract_status"))
                != "final_after_both_naturalistic_route_families_accepted"
                or str(acceptance.get("route", {}).get("sha256"))
                != str(route["recipient_route_sha256"])
                or set(acceptance.get("visual_runs", {})) != set(anchor_ids)
            ):
                raise ValueError(
                    f"Suite B visual acceptance contract is invalid for {route_id}"
                )
        for role in ("recipient", "helper"):
            route_path = repository_root / str(route[f"{role}_route"])
            if not route_path.is_file():
                raise FileNotFoundError(route_path)
            observed_hash = _sha256(route_path)
            expected_hash = str(route[f"{role}_route_sha256"])
            if observed_hash != expected_hash:
                raise ValueError(
                    f"Suite B {role} route hash drifted for {route_id}: "
                    f"expected={expected_hash}, observed={observed_hash}"
                )
        route_offset = 0
        for split in SPLITS:
            group_count = int(suite_b["split_counts_per_route"][split])
            if group_count % len(anchors) != 0:
                raise ValueError(
                    f"Suite B anchors are not balanced in {route_id}/{split}"
                )
            density = _deterministic_multiset(
                suite_b["density_counts_per_route_split"][split],
                master_seed=master_seed,
                namespace=f"{route_id}:{split}:density",
            )
            weather = _deterministic_multiset(
                suite_b["weather_counts_per_route_split"][split],
                master_seed=master_seed,
                namespace=f"{route_id}:{split}:weather",
            )
            if len(density) != group_count or len(weather) != group_count:
                raise ValueError(f"Suite B quota mismatch for {route_id}/{split}")
            for local_index in range(group_count):
                anchor = anchors[local_index % len(anchors)]
                ordinal = route_offset + local_index
                group_id = f"sb_{route_id}_r{ordinal:02d}"
                audit = split == "calibration" and local_index == 0
                rows.append(
                    {
                        "schema": manifest_schema,
                        "design_id": config["design_id"],
                        "suite_id": "B",
                        "suite_label": suite_b["label"],
                        "split": split,
                        "group_id": group_id,
                        "matched_pair_id": "",
                        "trajectory_id": f"{group_id}_natural",
                        "scenario_role": "naturalistic_operation",
                        "controlled_hazard_present": "unforced",
                        "geometry_or_route_id": route_id,
                        "geometry_or_route_status": route["implementation_status"],
                        "hazard_class": "natural_prevalence",
                        "closing_speed_band": "natural",
                        "time_to_hazard_band": "natural",
                        "traffic_density": density[local_index],
                        "traffic_density_status": suite_b_population[
                            "traffic_density_status"
                        ],
                        "ambient_population_mode": suite_b_population["mode"],
                        "ambient_population_process_required": int(
                            bool(suite_b_population["population_process_required"])
                        ),
                        "weather": weather[local_index],
                        "renderer_quality_level": renderer["primary_level"],
                        "renderer_server_launch_flag": renderer[
                            "required_server_launch_flag"
                        ],
                        "renderer_contract_role": "primary",
                        "carla_seed": _stable_seed(master_seed, f"{group_id}:carla"),
                        "traffic_seed": _stable_seed(master_seed, f"{group_id}:traffic"),
                        "sensor_seed": _stable_seed(master_seed, f"{group_id}:sensor"),
                        "raw_retention_tier": _retention_tier(
                            split, audit=audit, config=config
                        ),
                        "raw_window_duration_s": raw_window_s if split != "test" else 0.0,
                        "raw_window_anchor": (
                            config["retention"]["raw_window_anchor"]
                            if split != "test"
                            else "none"
                        ),
                        "confirmatory_locked": int(split == "test"),
                        "pair_contract_id": str(route["pair_contract_id"]),
                        "route_start_anchor_id": str(anchor["anchor_id"]),
                        "recipient_start_index": int(
                            anchor["recipient_start_index"]
                        ),
                        "helper_start_index": int(anchor["helper_start_index"]),
                        "recipient_route_sha256": str(
                            route["recipient_route_sha256"]
                        ),
                        "helper_route_sha256": str(route["helper_route_sha256"]),
                        **naturalistic_factor_fields,
                    }
                )
            route_offset += group_count

    manifest = pd.DataFrame(rows)
    validate_manifest(manifest, config)
    return manifest


def validate_manifest(manifest: pd.DataFrame, config: Mapping[str, object]) -> None:
    if manifest.empty or manifest["trajectory_id"].duplicated().any():
        raise ValueError("manifest is empty or contains duplicate trajectory IDs")
    if set(manifest["suite_id"]) != {"A", "B"}:
        raise ValueError("manifest must contain Suite A and Suite B")
    config_schema = str(config.get("schema_version"))
    expected_manifest_schema = (
        MANIFEST_SCHEMA_V2
        if config_schema == CONFIG_SCHEMA_V2
        else MANIFEST_SCHEMA_V1
    )
    if set(manifest["schema"].astype(str)) != {expected_manifest_schema}:
        raise ValueError("manifest schema does not match suite-design config")
    if set(manifest["renderer_quality_level"].astype(str)) != {"Epic"}:
        raise ValueError("every primary Suite A/B row must declare Epic rendering")
    if set(manifest["renderer_server_launch_flag"].astype(str)) != {
        "-quality-level=Epic"
    }:
        raise ValueError("every primary Suite A/B row must declare the exact Epic flag")
    if set(manifest["renderer_contract_role"].astype(str)) != {"primary"}:
        raise ValueError("Suite A/B design rows must remain primary-renderer rows")
    labels = dict(manifest.groupby("suite_id")["suite_label"].first())
    if labels != {"A": "designed_decision_opportunities", "B": "naturalistic_operation"}:
        raise ValueError(f"Suite A/B labels drifted: {labels}")
    pilot_ids = set(str(value) for value in config["pilot_exclusion_group_ids"])
    if pilot_ids & set(manifest["group_id"].astype(str)):
        raise ValueError("excluded pilot group entered the scientific manifest")

    group_splits = manifest.groupby("group_id")["split"].nunique()
    if int(group_splits.max()) != 1:
        raise ValueError("a trajectory group crosses data splits")
    group_seeds = manifest.groupby("group_id")["carla_seed"].nunique()
    if int(group_seeds.max()) != 1:
        raise ValueError("matched group members do not share CARLA seed")
    distinct_group_seeds = manifest.groupby("group_id")["carla_seed"].first()
    if distinct_group_seeds.duplicated().any():
        raise ValueError("CARLA seed is reused across independent groups")

    suite_a = manifest[manifest["suite_id"] == "A"]
    if set(suite_a["traffic_density"].astype(str)) != {"not_applicable"}:
        raise ValueError("Suite A must not claim a realized traffic-density factor")
    if set(suite_a["traffic_density_status"].astype(str)) != {"not_applicable"}:
        raise ValueError("Suite A traffic-density status drifted")
    if set(suite_a["ambient_population_mode"].astype(str)) != {
        "scenario_owned_only"
    }:
        raise ValueError("Suite A ambient-population mode drifted")
    if set(suite_a["ambient_population_process_required"].astype(int)) != {0}:
        raise ValueError("Suite A must not launch a generic population process")
    pair_sizes = suite_a.groupby("group_id").size()
    if set(pair_sizes) != {2}:
        raise ValueError("every Suite A group must have positive and benign members")
    pair_roles = suite_a.groupby("group_id")["scenario_role"].agg(set)
    expected_roles = {"controlled_positive_occlusion", "matched_benign_negative"}
    if any(value != expected_roles for value in pair_roles):
        raise ValueError("Suite A positive/benign role pairing drifted")

    if config_schema == CONFIG_SCHEMA_V2:
        missing_factor_columns = FACTOR_REALIZATION_COLUMNS - set(manifest.columns)
        if missing_factor_columns:
            raise ValueError(
                "v2 manifest lacks factor-realization columns: "
                f"{sorted(missing_factor_columns)}"
            )
        if set(suite_a["factor_realization_status"].astype(str)) != {
            "provisional_controls_pending_bounded_factor_smoke"
        }:
            raise ValueError("Suite A factor-realization status drifted")
        if set(suite_a["time_to_hazard_label_status"].astype(str)) != {
            "not_scientifically_realized_until_bounded_factor_smoke"
        }:
            raise ValueError("Suite A time-to-hazard claim boundary drifted")
        numeric_fields = {
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
        }
        converted = suite_a[list(sorted(numeric_fields))].apply(
            pd.to_numeric, errors="coerce"
        )
        if converted.isna().any().any() or not np_isfinite_frame(converted):
            raise ValueError("Suite A factor-realization controls must be finite")
        if not (
            converted["requested_closing_speed_band_min_mps"]
            <= converted["requested_closing_speed_target_mps"]
        ).all() or not (
            converted["requested_closing_speed_target_mps"]
            <= converted["requested_closing_speed_band_max_mps"]
        ).all():
            raise ValueError("requested closing-speed targets lie outside their bands")
        if not (
            converted["requested_proximity_horizon_band_min_s"]
            <= converted["requested_proximity_horizon_target_s"]
        ).all() or not (
            converted["requested_proximity_horizon_target_s"]
            <= converted["requested_proximity_horizon_band_max_s"]
        ).all():
            raise ValueError("requested proximity-horizon targets lie outside their bands")
        pair_factor_fields = sorted(
            FACTOR_REALIZATION_COLUMNS - {"factor_realization_status"}
        )
        pair_cardinality = suite_a.groupby("group_id")[pair_factor_fields].nunique(
            dropna=False
        )
        if int(pair_cardinality.to_numpy().max()) != 1:
            raise ValueError("Suite A matched pairs do not share requested controls")

    group_counts = (
        manifest.drop_duplicates("group_id").groupby(["suite_id", "split"]).size()
    )
    expected = {
        ("A", "calibration"): 24,
        ("A", "validation"): 24,
        ("A", "test"): 72,
        ("B", "calibration"): 18,
        ("B", "validation"): 18,
        ("B", "test"): 54,
    }
    if group_counts.to_dict() != expected:
        raise ValueError(f"group-count contract drifted: {group_counts.to_dict()}")

    suite_a_groups = suite_a.drop_duplicates("group_id")
    cell_columns = [
        "geometry_or_route_id",
        "closing_speed_band",
        "time_to_hazard_band",
    ]
    cell_counts = suite_a_groups.groupby(cell_columns + ["split"]).size().unstack(
        fill_value=0
    )
    if not all(
        tuple(int(row[split]) for split in SPLITS) == (1, 1, 3)
        for _, row in cell_counts.iterrows()
    ):
        raise ValueError("Suite A cells are not split 1/1/3")

    suite_b = manifest[manifest["suite_id"] == "B"]
    if config_schema == CONFIG_SCHEMA_V2:
        if set(suite_b["factor_realization_status"].astype(str)) != {
            "not_applicable_unforced_naturalistic"
        }:
            raise ValueError("Suite B factor-realization status drifted")
        if not suite_b["requested_hazard_actor_speed_mps"].isna().all():
            raise ValueError("Suite B must not fabricate requested hazard controls")
        if not suite_b["requested_hazard_onset_s"].isna().all():
            raise ValueError("Suite B must not fabricate a hazard onset")
    if set(suite_b["traffic_density_status"].astype(str)) != {
        "realized_nuisance_factor"
    }:
        raise ValueError("Suite B traffic-density status drifted")
    if set(suite_b["ambient_population_mode"].astype(str)) != {"naturalistic_tm"}:
        raise ValueError("Suite B ambient-population mode drifted")
    if set(suite_b["ambient_population_process_required"].astype(int)) != {1}:
        raise ValueError("Suite B must launch its naturalistic population process")
    if suite_b["pair_contract_id"].astype(str).eq("").any():
        raise ValueError("Suite B row lacks a paired-route contract")
    if suite_b["route_start_anchor_id"].astype(str).eq("").any():
        raise ValueError("Suite B row lacks a pre-registered start anchor")
    if suite_b["recipient_route_sha256"].astype(str).str.len().ne(64).any():
        raise ValueError("Suite B recipient route hash is invalid")
    if suite_b["helper_route_sha256"].astype(str).str.len().ne(64).any():
        raise ValueError("Suite B helper route hash is invalid")
    for route_id in (
        "town10hd_opt_signalized_demo_region",
        "town10hd_opt_safe_perimeter",
    ):
        rows = suite_b[suite_b["geometry_or_route_id"] == route_id]
        anchor_counts = rows.groupby(["route_start_anchor_id", "split"]).size()
        for anchor_id in (f"a{index}" for index in range(6)):
            observed = tuple(
                int(anchor_counts.get((anchor_id, split), 0)) for split in SPLITS
            )
            if observed != (1, 1, 3):
                raise ValueError(
                    f"Suite B anchor {route_id}/{anchor_id} is not split 1/1/3: "
                    f"{observed}"
                )

    if (manifest.loc[manifest["split"] == "test", "raw_window_duration_s"] != 0).any():
        raise ValueError("confirmatory test rows must not retain heavy raw windows")


def np_isfinite_frame(frame: pd.DataFrame) -> bool:
    """Avoid an eager NumPy dependency in the offline design generator."""

    return all(math.isfinite(float(value)) for value in frame.to_numpy().ravel())


def build_power_sensitivity(config: Mapping[str, object], manifest: pd.DataFrame) -> pd.DataFrame:
    power = _power_reference(config)
    historical_only = str(config.get("schema_version")) == CONFIG_SCHEMA_V2
    planned_test = int(
        manifest[
            (manifest["suite_id"] == "A")
            & (manifest["split"] == "test")
            & (manifest["scenario_role"] == "controlled_positive_occlusion")
        ]["group_id"].nunique()
    )
    censor_rates = sorted(
        {0.0, float(power["planned_censor_fraction"]), 0.20}
    )
    rows = []
    for censor_fraction in censor_rates:
        effective = max(2, math.floor(planned_test * (1.0 - censor_fraction)))
        for paired_sd_s in power["paired_sd_sensitivity_s"]:
            rows.append(
                {
                    "planned_test_groups": planned_test,
                    "censor_fraction": censor_fraction,
                    "effective_numeric_pairs": effective,
                    "smallest_effect_s": float(power["smallest_effect_s"]),
                    "paired_sd_s": float(paired_sd_s),
                    "two_sided_alpha": float(power["two_sided_alpha"]),
                    "approximate_paired_t_power": paired_t_power(
                        sample_count=effective,
                        effect_s=float(power["smallest_effect_s"]),
                        paired_sd_s=float(paired_sd_s),
                        alpha=float(power["two_sided_alpha"]),
                    ),
                    "status": (
                        "historical_warning_endpoint_sensitivity_only_not_v2_power"
                        if historical_only
                        else "sensitivity_only_not_pilot_estimated"
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(
    config: Mapping[str, object], manifest: pd.DataFrame, power_table: pd.DataFrame
) -> dict:
    retention = config["retention"]
    frames = int(
        round(
            float(retention["raw_window_duration_s"])
            * float(config["common"]["world_hz"])
        )
    )
    roles = len(config["common"]["roles"])
    estimated_bytes = 0
    for row in manifest.itertuples(index=False):
        estimated_bytes += int(retention["estimated_lightweight_bytes_per_trajectory"])
        if row.raw_retention_tier in {"inputs_only_window", "inputs_plus_logits_window"}:
            estimated_bytes += (
                frames
                * roles
                * int(retention["pilot_measured_role_input_bytes_per_frame"])
            )
        if row.raw_retention_tier == "inputs_plus_logits_window":
            estimated_bytes += (
                frames
                * roles
                * int(retention["pilot_measured_role_logits_bytes_per_frame"])
            )

    group_frame = manifest.drop_duplicates("group_id")
    trajectory_count = len(manifest)
    power_reference = _power_reference(config)
    is_v2 = str(config.get("schema_version")) == CONFIG_SCHEMA_V2
    planned_censor = float(power_reference["planned_censor_fraction"])
    planned_sd = 1.25
    planned_row = power_table[
        (power_table["censor_fraction"] == planned_censor)
        & (power_table["paired_sd_s"] == planned_sd)
    ].iloc[0]
    pending_statuses = sorted(
        set(
            manifest.loc[
                ~manifest["geometry_or_route_status"].astype(str).str.startswith(
                    "reviewed"
                ),
                "geometry_or_route_status",
            ].astype(str)
        )
    )
    runtime_minutes = trajectory_count * float(
        config["runtime_estimate"]["pilot_measured_minutes_per_trajectory"]
    )
    return {
        "schema": (
            "scenesense.phase2_suite_design_summary.v2"
            if str(config.get("schema_version")) == CONFIG_SCHEMA_V2
            else "scenesense.phase2_suite_design_summary.v1"
        ),
        "design_id": config["design_id"],
        "collection_authorized": False,
        "suite_labels": {
            "A": "designed_decision_opportunities",
            "B": "naturalistic_operation",
        },
        "ambient_population_contracts": {
            "A": dict(config["suite_a"]["ambient_population_contract"]),
            "B": dict(config["suite_b"]["ambient_population_contract"]),
        },
        "renderer_contract": dict(config["common"]["renderer_quality"]),
        "primary_endpoint": str(config["power"]["primary_endpoint"]),
        "primary_endpoint_status": (
            str(config["power"]["primary_endpoint_status"])
            if is_v2
            else "registered_v1_warning_endpoint"
        ),
        "registered_effect_size_s": (
            None if is_v2 else float(power_reference["smallest_effect_s"])
        ),
        "historical_warning_reference": (
            {
                "status": "non_authoritative_reference_only",
                "endpoint": str(power_reference["endpoint"]),
                "smallest_effect_s": float(power_reference["smallest_effect_s"]),
                "smallest_effect_interpretation": dict(
                    power_reference["smallest_effect_interpretation"]
                ),
                "planned_censor_fraction": planned_censor,
                "minimum_power": float(power_reference["minimum_power"]),
                "sensitivity_power_at_sd_1_25_s": float(
                    planned_row["approximate_paired_t_power"]
                ),
            }
            if is_v2
            else None
        ),
        "smallest_effect_interpretation": (
            None
            if is_v2
            else dict(power_reference["smallest_effect_interpretation"])
        ),
        "warning_nuisance_gate": dict(config["warning_nuisance_gate"]),
        "installed_track_quality_guardrails": (
            dict(config["installed_track_quality_guardrails"])
            if is_v2
            else None
        ),
        "independent_group_count": int(group_frame["group_id"].nunique()),
        "trajectory_count": trajectory_count,
        "group_counts": {
            f"{suite}_{split}": int(count)
            for (suite, split), count in group_frame.groupby(
                ["suite_id", "split"]
            ).size().items()
        },
        "trajectory_counts": {
            split: int(count)
            for split, count in manifest.groupby("split").size().items()
        },
        "suite_a_test_positive_group_count": int(
            manifest[
                (manifest["suite_id"] == "A")
                & (manifest["split"] == "test")
                & (manifest["scenario_role"] == "controlled_positive_occlusion")
            ]["group_id"].nunique()
        ),
        "power_status": (
            "not_authorized_pending_recipient_endpoint_runtime_and_calibration"
            if is_v2
            else "conditional_on_calibration_simulation_gate"
        ),
        "planned_censor_fraction": None if is_v2 else planned_censor,
        "sensitivity_power_at_sd_1_25_s": (
            None
            if is_v2
            else float(planned_row["approximate_paired_t_power"])
        ),
        "minimum_required_calibration_power": (
            None if is_v2 else float(power_reference["minimum_power"])
        ),
        "estimated_storage_bytes": int(estimated_bytes),
        "design_raw_cap_bytes": int(retention["design_raw_cap_bytes"]),
        "storage_estimate_within_cap": estimated_bytes
        <= int(retention["design_raw_cap_bytes"]),
        "estimated_capture_hours": runtime_minutes / 60.0,
        "pending_manual_scenario_statuses": pending_statuses,
        "blocking_gates": (
            (["author_and_visually_review_all_pending_geometry_and_route_families"]
             if pending_statuses else [])
            + (
                [
                    "per_row_factor_runtime_adapter_and_positive_realization_gate",
                    "bounded_factor_smoke_before_factor_freeze",
                ]
                if str(config.get("schema_version")) == CONFIG_SCHEMA_V2
                else []
            )
            + ["calibration_replay_sufficiency_capture"]
            + (
                [
                    "register_recipient_endpoint_effect_size_and_power_after_calibration",
                    "freeze_installed_track_metric_definitions_denominators_and_structural_gates_before_exact_16_calibration",
                    "estimate_then_register_numeric_track_quality_thresholds_before_additional_collection",
                ]
                if is_v2
                else [
                    "calibration_simulation_power_at_least_0_80_for_all_registered_endpoints",
                    "calibration_absolute_warning_nuisance_gate",
                ]
            )
            + ["review_exact_local_and_oai_timestamp_byte_fields"]
        ),
    }


def write_design(
    config_path: Path, output_dir: Path, *, overwrite: bool = False
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config root must be a mapping")
    manifest = build_manifest(config)
    power_table = build_power_sensitivity(config, manifest)
    summary = summarize(config, manifest, power_table)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"design output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    manifest.to_csv(
        output_dir / "trajectory_group_manifest.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    power_table.to_csv(output_dir / "power_sensitivity.csv", index=False)
    (output_dir / "design_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    module_path = Path(__file__).resolve()
    provenance = {
        "schema": "scenesense.phase2_suite_design_provenance.v1",
        "design_id": config["design_id"],
        "runtime_authorized": False,
        "config_sha256": _sha256(config_path),
        "config_semantic_sha256": _semantic_sha256(config),
        "module_sha256": _sha256(module_path),
        "deterministic_master_seed": int(config["master_seed"]),
    }
    (output_dir / "design_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / "artifact_manifest.json"
    artifact_manifest = {
        "schema": "scenesense.phase2_suite_design_artifact_manifest.v1",
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path != manifest_path
        ],
    }
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root
        / "phase2_map_sharing/configs/phase2_suite_ab_design_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root
        / "phase2_map_sharing/design/phase2_suite_ab_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = write_design(
        args.config.resolve(), args.output_dir.resolve(), overwrite=args.overwrite
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

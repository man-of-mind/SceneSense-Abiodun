"""Offline validation of the paired-causal pilot configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from .causal_contract import FIELD_SOURCE_ALLOWLIST
from .retention import RetentionLimits
from .schemas_v2 import FORBIDDEN_RUNTIME_KEYS, PLACEMENT_ACTIONS, PUBLICATION_ACTIONS


EXPECTED_SENSOR_CONTRACT = {
    "world_hz": 10.0,
    "fixed_delta_seconds": 0.1,
    "rgb_width": 1280,
    "rgb_height": 720,
    "rgb_fov_degrees": 120.0,
    "radar_points_per_second": 200000,
    "radar_raster_radius": 4,
    "temporal_window": 2,
    "truth_position_convention": "actor_origin",
}

EXPECTED_STATIC_ANCHORS = {
    "clear": 50.3,
    "mild": 19.5,
    "mid": 15.6,
    "strong": 8.2,
}

REQUIRED_LOCAL_METRICS = {
    "compact_payload_bytes_by_object_count",
    "segmentation_accuracy",
    "pedestrian_recall",
    "vehicle_recall",
    "localization_error",
    "inference_latency_p50_p95",
    "sustainable_fps",
    "cpu_gpu_utilization",
    "memory_occupancy",
    "oai_delivery_latency_prb_queue_by_anchor",
}


def validate_pilot_config(config: Mapping[str, object]) -> dict:
    if config.get("contract_version") != "scenesense.phase2_paired_causal_pilot.v1":
        raise ValueError("unexpected paired-causal pilot contract version")
    authorization = config["authorization"]
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "carla", "oai", "full_collection", "controller_evaluation", "rl_training"
    }:
        raise ValueError("pilot authorization mapping is incomplete")
    implementation_status = str(config.get("implementation_status", ""))
    if implementation_status == "offline_contract_only":
        if any(bool(value) for value in authorization.values()):
            raise ValueError("offline readiness config must not authorize a live or scientific run")
        live_run_authorized = False
    elif implementation_status == "reviewed_pilot_only":
        if not bool(authorization["carla"]) or any(
            bool(authorization[field])
            for field in (
                "oai", "full_collection", "controller_evaluation", "rl_training"
            )
        ):
            raise ValueError("reviewed pilot may authorize CARLA capture only")
        review = config.get("review_evidence")
        if not isinstance(review, Mapping):
            raise ValueError("reviewed pilot requires review_evidence")
        if review.get("geometry_id") != "town10hd_opt_curbside_legal_opposing_v1":
            raise ValueError("reviewed pilot geometry evidence has drifted")
        if review.get("geometry_verdict") != "manual_positive_and_benign_pass":
            raise ValueError("reviewed pilot geometry verdict is missing")
        gpu = review.get("host_gpu_capacity")
        if not isinstance(gpu, Mapping) or gpu.get("verdict") != (
            "accepted_for_correctness_only_pilot"
        ):
            raise ValueError("reviewed pilot host-GPU verdict is missing")
        if int(gpu.get("memory_total_mib", 0)) <= 0 or int(
            gpu.get("memory_free_mib", 0)
        ) <= 0:
            raise ValueError("reviewed pilot host-GPU inventory is invalid")
        if bool(gpu.get("inference_timing_citable")):
            raise ValueError("shared-GPU pilot inference timing cannot be citable")
        live_run_authorized = True
    else:
        raise ValueError(
            "pilot status must be offline_contract_only or reviewed_pilot_only"
        )

    pilot = config["pilot"]
    if not isinstance(pilot, Mapping) or int(pilot["trajectory_count"]) != 2:
        raise ValueError("pilot must contain exactly two trajectories")
    if set(pilot["required_roles"]) != {
        "controlled_positive_occlusion",
        "matched_benign_negative",
    }:
        raise ValueError("pilot roles must be one controlled positive and one matched benign")
    warnings = pilot["warnings"]
    if warnings.get("mode") != "record_only" or bool(
        warnings.get("actuate_brake_or_steer")
    ):
        raise ValueError("C2 pilot warnings must be record-only and non-actuating")
    if set(pilot["arms"]) != {"ego_only", "send_everything", "hazard_only"}:
        raise ValueError("pilot arm set is incomplete")
    if not bool(pilot.get("arm_state_isolation_required")):
        raise ValueError("counterfactual arm-state isolation must be required")

    sensor = config["sensor_contract"]
    for key, expected in EXPECTED_SENSOR_CONTRACT.items():
        if sensor.get(key) != expected:
            raise ValueError(f"sensor contract mismatch for {key}")
    expected_returns = int(
        float(sensor["radar_points_per_second"]) / float(sensor["world_hz"])
    )
    if int(sensor.get("expected_radar_returns_per_frame", -1)) != expected_returns:
        raise ValueError("expected radar returns/frame is inconsistent with pps and world_hz")

    retention = config["raw_retention"]
    RetentionLimits.from_mapping(retention)
    if retention.get("mode") != "controlled_windows_only":
        raise ValueError("continuous raw retention is forbidden")
    if retention.get("quota_action") != "stop_raw_keep_lightweight_logs":
        raise ValueError("quota action must preserve lightweight logs")
    if bool(retention.get("allow_automatic_dataset_deletion")):
        raise ValueError("pilot overflow must never delete existing datasets")

    causal = config["causal_contract"]
    required_metadata = {
        "source_stage",
        "observed_at_s",
        "available_at_s",
        "consuming_decision_id",
        "consuming_decision_stage",
        "clock_id",
        "arm_id",
    }
    if set(causal["required_field_metadata"]) != required_metadata:
        raise ValueError("causal field metadata contract is incomplete")
    if causal.get("required_inequality") != "available_at_s <= decision_at_s":
        raise ValueError("causal availability inequality is not pinned")
    if set(causal["placement_actions"]) != set(PLACEMENT_ACTIONS):
        raise ValueError("placement action contract is incomplete")
    if set(causal["publication_actions"]) != set(PUBLICATION_ACTIONS):
        raise ValueError("publication action contract is incomplete")
    if not {"evaluation_truth", "shadow_inference"}.issubset(
        set(causal["forbidden_runtime_sources"])
    ):
        raise ValueError("forbidden runtime source contract is incomplete")
    if not set(FORBIDDEN_RUNTIME_KEYS).issubset(
        set(causal["forbidden_runtime_fields"])
    ):
        raise ValueError("forbidden runtime field contract is incomplete")
    configured_sources = {
        str(field): frozenset(str(source) for source in sources)
        for field, sources in causal["field_source_allowlist"].items()
    }
    if configured_sources != FIELD_SOURCE_ALLOWLIST:
        raise ValueError("causal field/producer allowlist has drifted")

    network = config["network_design"]
    anchors = {
        str(item["id"]): float(item["snr_db"])
        for item in network["measured_static_anchors"]
    }
    if anchors != EXPECTED_STATIC_ANCHORS:
        raise ValueError("measured static channel anchors have drifted")
    core = set(network["decision_core_requires"])
    required_core = {
        "stable_good",
        "stable_bad",
        "degradation_and_recovery",
        "burst_and_queue_recovery",
        "heldout_intermediate_near_split_knee",
    }
    if not required_core.issubset(core):
        raise ValueError("minimal sequential network decision core is incomplete")
    if bool(network.get("policy_observes_anchor_id")):
        raise ValueError("policy must observe causal telemetry, not channel-rung labels")

    post_pilot = config["post_pilot_freeze"]
    for field in (
        "smallest_effect_of_interest",
        "false_warning_ceiling",
        "positive_hazard_events_per_cell",
        "matched_benign_events_per_cell",
        "expected_censoring_fraction",
        "confirmatory_counts",
    ):
        if post_pilot.get(field) is not None:
            raise ValueError(f"{field} must be frozen only after the pilot")

    local = config["local_calibration_before_dynamic_ladder"]
    if not bool(local.get("required")):
        raise ValueError("LOCAL calibration must precede the dynamic ladder")
    if not REQUIRED_LOCAL_METRICS.issubset(set(local["metrics"])):
        raise ValueError("LOCAL calibration metrics are incomplete")

    return {
        "verdict": "PASS",
        "live_run_authorized": live_run_authorized,
        "implementation_status": implementation_status,
        "trajectory_count": 2,
        "sensor_contract": "exact_training_contract",
        "warnings": "record_only",
        "raw_retention": "hard_quota_no_deletion",
        "minimal_transition_core_required_before_rl_go_no_go": True,
    }


def load_and_validate_pilot_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("pilot config root must be a mapping")
    return validate_pilot_config(config)

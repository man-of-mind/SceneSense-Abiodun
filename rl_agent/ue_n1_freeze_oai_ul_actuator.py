"""Assemble and validate the UE-N1 OAI UL actuator interface freeze.

This module is intentionally offline.  It validates pinned declarations and
historical evidence, then emits a create-only interface bundle.  It contains
no Telnet client and cannot launch OAI or CARLA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n1_oai_ul_actuator_interface_v1.json"

CONFIG_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_config.v1"
MANIFEST_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_manifest.v1"
TERMINAL_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_decision.v1"
INTERFACE_ID = "ue_n1_oai_ul_actuator_interface_v1"
FROZEN_STATUS = "FROZEN_INTERFACE_ONLY"
NEXT_ITEM = "UE-N2"
# Refreshed after the frozen config is final.  Alternate/self-consistent config
# retargets are not authority for this create-only interface.
FROZEN_CONFIG_SHA256 = "9b39a9580a737753c54358b08509079cdb7db60935c2fc58cbf8fc04e2f54b14"

EXPECTED_AUTHORITY = {
    "interface_freeze_only": True,
    "numeric_calibration_authorized": False,
    "numeric_bounds_authorized": False,
    "launcher_or_runtime_edit_authorized": False,
    "socket_execution_authorized": False,
    "oai_run_authorized": False,
    "carla_run_authorized": False,
    "policy_training_authorized": False,
}

EXPECTED_TELEMETRY: dict[str, tuple[str, tuple[str, ...], str]] = {
    "GNB_MAC_PUSCH_POWER_CONTROL": (
        "REQUIRED_WHEN_PUSCH_RECEIVED",
        (
            "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
            "txpower_calc", "rbSize", "mcs", "rssi",
        ),
        "MAC_POWER_CONTROL_NORMALIZED_INSTANTANEOUS_PUSCH_SNR_NOT_RAW_PHY_SNR",
    ),
    "GNB_MAC_UL_MCS_DECISION": (
        "REQUIRED_WHEN_NEW_UL_SCHEDULED",
        (
            "rnti", "frame", "slot", "sched_frame", "sched_slot",
            "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
            "pre_phr_mcs", "post_phr_mcs", "final_mcs",
            "estimated_ul_buffer", "sched_ul_bytes", "B", "min_rb",
            "available_rb_before", "available_rb_after", "ph", "pcmax",
            "rb_size_final", "tbs_final", "force_ul_mcs",
        ),
        "SCHEDULER_EMA_AND_SELECTED_TO_FINAL_MCS_PATH",
    ),
    "GNB_MAC_UL": (
        "REQUIRED_WHEN_UL_SCHEDULED",
        ("rnti", "frame", "slot", "mcs", "tbs"),
        "FINAL_UL_GRANT_CROSS_CHECK",
    ),
    "NRUE_MAC_DCI_GRANT": (
        "REQUIRED_FOR_UE_GRANT_AND_HARQ_VIEW",
        (
            "direction", "dci_format", "rnti_type", "rnti", "dci_frame",
            "dci_slot", "sched_frame", "sched_slot", "mcs", "mcs_table",
            "rb_start", "rb_size", "start_symbol", "nr_symbols", "tbs",
            "harq_pid", "ndi", "rv", "round", "qam_mod_order",
            "target_code_rate", "tpc", "n_cce", "N_cce",
        ),
        "UE_OBSERVED_GRANT_AND_RETRANSMISSION_PROXY_ONLY_NOT_DIRECT_CRC_OR_BLER",
    ),
    "NRUE_MAC_RLC_BUFFER_STATUS": (
        "REQUIRED_FOR_CAUSAL_UE_BACKLOG",
        (
            "rnti", "ue_id", "frame", "slot", "lcid", "lcgid",
            "bytes_in_buffer", "bj", "pbr", "priority",
        ),
        "RLC_BUFFER_OCCUPANCY",
    ),
    "NRUE_MAC_BSR_STATUS": (
        "REQUIRED_FOR_CAUSAL_UE_BSR",
        (
            "rnti", "ue_id", "frame", "slot", "bsr_type", "trigger_mask",
            "bsr_sent", "padding_len", "num_sdus", "sdu_bytes", "lcg0_bytes",
            "lcg1_bytes", "lcg2_bytes", "lcg3_bytes", "lcg4_bytes",
            "lcg5_bytes", "lcg6_bytes", "lcg7_bytes", "bsr_lcg_id",
            "bsr_index", "bsr_long0_index", "bsr_long1_index",
            "bsr_long2_index", "bsr_long3_index", "bsr_long4_index",
            "bsr_long5_index", "bsr_long6_index", "bsr_long7_index",
        ),
        "BSR_AND_LCG_BACKLOG",
    ),
    "GNB_MAC_BLER_MCS_DECISION": (
        "UNAVAILABLE_AS_DIRECT_UL_BLER_UNDER_CURRENT_SINR_POLICY",
        (
            "direction", "rnti", "frame", "diff", "old_mcs", "new_mcs",
            "max_mcs_input", "max_mcs_applied", "min_mcs", "opt_max_mcs",
            "num_sched", "num_retx", "bler_window_ppm", "bler_before_ppm",
            "bler_after_ppm", "lower_ppm", "upper_ppm", "branch", "updated",
        ),
        "NO_DIRECT_UL_BLER_CLAIM",
    ),
}

EXPECTED_SOURCE_KEYS = {
    "channelmod_documentation",
    "channelmod_parser",
    "rfsim_channel_binding",
    "rfsim_channel_application",
    "channel_descriptor",
    "phy_pusch_snr",
    "scheduler_mapping_and_ema",
    "scheduler_ul_path",
    "telemetry_schema",
    "telemetry_event_clock",
    "telemetry_csv_formatter",
    "gnb_radio_config",
    "rfsim_channel_config",
    "existing_control_helper",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "interface_id", "repository_root", "authority", "predecessor",
    "scope", "actuator", "attach_lifecycle", "control_transport", "schedule",
    "command_timing", "signal_contract", "telemetry", "scheduler",
    "calibration", "oai_revision", "contract", "sources", "runtime_artifacts",
    "mechanism_evidence", "output",
}
CANONICAL_COMMAND_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")


class InterfaceFreezeError(RuntimeError):
    """Raised when the UE-N1 declaration or evidence fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def validate_channel_command_literal(value: Any) -> Decimal:
    """Validate an RFsim command scalar before OAI's permissive ``atof``.

    This enforces lexical safety only.  It deliberately defines no numeric
    calibration or operating bound.
    """

    if not isinstance(value, str) or value == "-0" or not CANONICAL_COMMAND_RE.fullmatch(value):
        raise InterfaceFreezeError("channel command is not a canonical base-10 decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise InterfaceFreezeError("channel command is not a finite decimal") from exc
    if not parsed.is_finite():
        raise InterfaceFreezeError("channel command is not finite")
    return parsed


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterfaceFreezeError(f"{label} must be a mapping")
    return value


def _repo_path(relative: str) -> Path:
    candidate = (ROOT / str(relative)).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise InterfaceFreezeError(f"path escapes repository: {relative}") from exc
    return candidate


def _pinned(path: Path, expected_sha256: str, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise InterfaceFreezeError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise InterfaceFreezeError(
            f"{label} hash drift: expected={expected_sha256} actual={actual}"
        )
    return path


def _input_record(path: Path, kind: str, label: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    record: dict[str, Any] = {
        "kind": kind,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if label is not None:
        record["label"] = label
    return record


def _require_exact(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(actual) != dict(expected):
        raise InterfaceFreezeError(f"{label} contract mismatch")


def _validate_telemetry_declaration(config: Mapping[str, Any]) -> None:
    telemetry = _mapping(config.get("telemetry"), "telemetry")
    if set(telemetry) != {
        "availability_timestamp_required", "radio_frame_slot_is_availability_time",
        "source_event_timestamp_clock", "csv_time_semantics",
        "live_ingest_availability_fields", "join_contract", "events",
        "ul_outcome_contract", "observation_semantics",
    }:
        raise InterfaceFreezeError("telemetry top-level key set differs")
    if telemetry.get("availability_timestamp_required") is not True:
        raise InterfaceFreezeError("radio observation availability timestamp is required")
    if telemetry.get("radio_frame_slot_is_availability_time") is not False:
        raise InterfaceFreezeError("radio frame/slot cannot be availability time")
    if telemetry.get("source_event_timestamp_clock") != "CLOCK_REALTIME" or telemetry.get(
        "csv_time_semantics"
    ) != "PRODUCER_EVENT_EMISSION_TIME_NOT_RECORDER_AVAILABILITY":
        raise InterfaceFreezeError("T-tracer event/CSV time semantics differ")
    if telemetry.get("live_ingest_availability_fields") != [
        "ingest_available_wall_time_ns", "ingest_available_monotonic_ns"
    ]:
        raise InterfaceFreezeError("live-ingest availability fields differ")
    _require_exact(
        _mapping(telemetry.get("join_contract"), "telemetry.join_contract"),
        {
            "frame_slot_alone_authorized": False,
            "frame_slot_wrap_period_s": 10.24,
            "required_identity_and_time": [
                "control_session_id", "rnti",
                "unwrapped_radio_cycle_or_source_event_time",
                "ingest_availability_time",
            ],
            "decision_and_scheduled_frame_slot_distinct": True,
        },
        "telemetry join",
    )
    _require_exact(
        _mapping(telemetry.get("ul_outcome_contract"), "ul_outcome_contract"),
        {
            "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED_CURRENT_SINR_TRACE",
            "missing_direct_bler_is_zero": False,
            "ue_grant_round_semantics": "RETRANSMISSION_PROXY_ONLY",
            "lower_bound_acceptance": (
                "REQUIRE_GENUINE_UL_CRC_HARQ_OUTCOME_OR_EXPLICIT_MISSING_EVIDENCE_GATE"
            ),
        },
        "UL outcome availability",
    )
    _require_exact(
        _mapping(telemetry.get("observation_semantics"), "observation_semantics"),
        {
            "missing_pusch_snr": "UNAVAILABLE_OR_DTX_NEVER_ZERO",
            "ema_settling_axis": (
                "ACCEPTED_PUSCH_OBSERVATION_COUNT_NOT_FIXED_WALL_TIME"
            ),
            "required_ema_context_fields": [
                "accepted_pusch_observations_since_command", "age_since_command_ns"
            ],
            "grant_is_confirmed_delivery": False,
            "missing_rlc_or_bsr_forward_fill_zero_authorized": False,
        },
        "radio observation semantics",
    )
    records = telemetry.get("events")
    if not isinstance(records, list) or len(records) != len(EXPECTED_TELEMETRY):
        raise InterfaceFreezeError("telemetry event set is incomplete")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in records:
        record = _mapping(value, "telemetry event")
        event_id = str(record.get("id", ""))
        if event_id in by_id:
            raise InterfaceFreezeError(f"duplicate telemetry event: {event_id}")
        by_id[event_id] = record
    if set(by_id) != set(EXPECTED_TELEMETRY):
        raise InterfaceFreezeError("telemetry event IDs differ from the frozen set")
    for event_id, (requirement, fields, primary_use) in EXPECTED_TELEMETRY.items():
        expected = {
            "id": event_id,
            "requirement": requirement,
            "fields": list(fields),
            "primary_use": primary_use,
        }
        if dict(by_id[event_id]) != expected:
            raise InterfaceFreezeError(f"telemetry schema drift: {event_id}")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterfaceFreezeError(f"cannot load UE-N1 config: {path}") from exc
    if config.get("schema") != CONFIG_SCHEMA or config.get("interface_id") != INTERFACE_ID:
        raise InterfaceFreezeError("invalid UE-N1 config schema/interface ID")
    if set(config) != EXPECTED_TOP_LEVEL_KEYS:
        raise InterfaceFreezeError("UE-N1 config top-level key set differs")
    if sha256_file(path) != FROZEN_CONFIG_SHA256:
        raise InterfaceFreezeError("UE-N1 config does not match the frozen config seal")
    repository_root = (path.parent / str(config.get("repository_root", ""))).resolve()
    if repository_root != ROOT:
        raise InterfaceFreezeError(f"repository root mismatch: {repository_root} != {ROOT}")
    _require_exact(_mapping(config.get("authority"), "authority"), EXPECTED_AUTHORITY, "authority")

    predecessor = _mapping(config.get("predecessor"), "predecessor")
    if set(predecessor) != {
        "checklist_item", "bundle_dir", "terminal_json", "terminal_sha256",
        "manifest_json", "manifest_sha256", "registry_csv", "registry_sha256",
        "required_status", "required_next", "required_profiles",
    }:
        raise InterfaceFreezeError("A4 predecessor key set differs")
    expected_predecessor = {
        "checklist_item": "UE-A4",
        "required_status": "FROZEN",
        "required_next": "UE-N1",
        "required_profiles": 72,
    }
    for key, expected in expected_predecessor.items():
        if predecessor.get(key) != expected:
            raise InterfaceFreezeError(f"A4 predecessor mismatch: {key}")

    _require_exact(
        _mapping(config.get("scope"), "scope"),
        {
            "ue_count": 1,
            "direction": "UPLINK_ONLY",
            "rf_simulator_server_role": "GNB",
            "downlink_actuation_authorized": False,
            "multi_ue_actuation_authorized": False,
        },
        "single-UE scope",
    )

    actuator = _mapping(config.get("actuator"), "actuator")
    expected_actuator = {
        "subsystem": "OAI_RFSIM_CHANNELMOD",
        "channel_model_name": "rfsimu_channel_ue0",
        "channel_model_owner": "rfsimulator",
        "channel_model_type": "AWGN",
        "model_index_binding": "RESOLVE_EXACT_NAME_EACH_GNB_SESSION",
        "hardcoded_model_index_authorized": False,
        "resolution_command": "channelmod show current",
        "modify_command_template": (
            "channelmod modify {resolved_model_index} noise_power_dB "
            "{channel_command_db}"
        ),
        "mutable_parameter": "noise_power_dB",
        "command_value_field": "channel_command_db",
        "command_value_semantics": (
            "RFSIM_CHANNEL_NOISE_CONTROL_NOT_TARGET_OR_ACHIEVED_SNR"
        ),
        "command_value_lexical_contract": (
            "FINITE_CANONICAL_BASE10_DECIMAL_STRING_NO_EXPONENT_WHITESPACE_OR_CONTROL_CHARACTERS"
        ),
        "oai_atof_is_input_validation": False,
        "fixed_path_loss_parameter": "ploss",
        "fixed_path_loss_db": 0,
        "global_noise_parameter": "noise_power_dBFS",
        "global_noise_requirement": "UNSET",
        "prohibited_channel_model_names": [
            "rfsimu_channel_enB0", "rfsimu_channel_enB1", "rfsimu_channel_ue1"
        ],
    }
    _require_exact(actuator, expected_actuator, "physical actuator")

    attach = _mapping(config.get("attach_lifecycle"), "attach_lifecycle")
    _require_exact(
        attach,
        {
            "initial_and_restore_channel_command_db": -50,
            "runtime_command_gate": [
                "UE_ATTACHED",
                "OAITUN_UE1_PRESENT",
                "REACHABILITY_PASS",
                "UPLINK_TRAFFIC_ACTIVE",
                "TELEMETRY_RECORDER_READY",
                "EXACT_UL_CHANNEL_OBJECT_VALIDATED",
                "FRESH_CURRENT_SESSION_PUSCH_OBSERVATION_PRESENT",
            ],
            "source_template_initial_channel_command_db": -10,
            "source_template_is_effective_runtime_config": False,
            "n2_effective_runtime_config_requirement": (
                "CREATE_HASH_AND_RECORD_EFFECTIVE_CONFIG_OR_EXACT_OVERRIDE_AND_ARGV"
            ),
            "pre_attach_show_current_required_channel_command_db": -50,
            "restore_on_normal_shutdown": True,
            "restore_on_failure": True,
            "tunnel_presence_alone_is_sufficient": False,
        },
        "clean-attach lifecycle",
    )

    control = _mapping(config.get("control_transport"), "control_transport")
    _require_exact(
        control,
        {
            "protocol": "TELNET",
            "host": "127.0.0.1",
            "port": 9090,
            "server_process": "GNB",
            "n2_connection_lifecycle": "ONE_PERSISTENT_CONNECTION_PER_TRACE",
            "reconnect_per_command_authorized": False,
            "response_requirement": "FULL_RESPONSE_AND_PROMPT_WITH_MATCHING_ECHO",
            "modify_response_echoes_model_name_or_index": False,
            "resolution_binding_lifetime": "CONTROL_SESSION_ID",
            "connection_loss_trace_result": "FAILED",
            "post_loss_reconnect_scope": "BEST_EFFORT_RESTORE_CLEAN_ONLY",
            "cleanup_success_can_change_failed_to_pass": False,
            "fail_closed_conditions": [
                "CONTROL_ERROR",
                "MISSING_OR_DUPLICATE_MODEL_NAME",
                "UNEXPECTED_OWNER_OR_TYPE",
                "NONZERO_PATH_LOSS",
                "GLOBAL_NOISE_CONFIGURED",
                "RESPONSE_OR_ECHO_MISMATCH",
                "CONNECTION_LOST",
            ],
        },
        "persistent Telnet transport",
    )

    schedule = _mapping(config.get("schedule"), "schedule")
    _require_exact(
        schedule,
        {
            "clock": "time.monotonic_ns",
            "period_ms": 100,
            "formula": "anchor_monotonic_ns + trace_index * 100000000",
            "trace_index_contract": "UNIQUE_CONTIGUOUS_ZERO_BASED",
            "strictly_monotonic": True,
            "relative_to_actual_completion": False,
            "catch_up_policy": "NEVER_BURST_OBSOLETE_COMMANDS",
            "numeric_lateness_or_jitter_acceptance": "NOT_DEFINED_UE_N1",
        },
        "100-ms monotonic schedule",
    )

    timing = _mapping(config.get("command_timing"), "command_timing")
    expected_timing_fields = [
        "trace_id", "trace_index", "desired_achieved_pusch_snr_db",
        "channel_command_db",
        "scheduled_monotonic_ns", "send_monotonic_ns", "send_wall_time_ns",
        "response_received_monotonic_ns", "response_received_wall_time_ns",
        "control_session_id", "resolved_model_index", "resolved_model_name",
        "resolved_model_owner", "resolved_model_type",
        "show_current_response_sha256", "response_text_sha256", "echoed_owner",
        "echoed_noise_power_db", "echoed_path_loss_db", "status",
    ]
    _require_exact(
        timing,
        {
            "required_fields": expected_timing_fields,
            "send_time_semantics": "LOWER_BRACKET_FOR_HANDLER_COMPLETION",
            "response_received_semantics": (
                "ACK_UPPER_BOUND_FOR_HANDLER_COMPLETION_NOT_RF_SAMPLE_APPLICATION_TIME"
            ),
            "prohibited_fields": ["command_applied_at", "command_applied_at_ns"],
            "post_command_observation_semantics": (
                "FIRST_CAUSALLY_LATER_PUSCH_IS_A_CANDIDATE_NOT_PROOF_OF_NEW_CHANNEL_APPLICATION"
            ),
            "effect_lag_semantics": (
                "ESTIMATE_FROM_MEASURED_STEP_RESPONSE_IN_UE_N2_NO_INVENTED_APPLY_TIMESTAMP"
            ),
        },
        "command timing",
    )

    _require_exact(
        _mapping(config.get("signal_contract"), "signal_contract"),
        {
            "desired_achieved_pusch_snr_db": (
                "SAVED_EXPERIMENT_TRACE_VALUE_NOT_POLICY_OBSERVATION_AND_NOT_RADIO_TRUTH"
            ),
            "channel_command_db": "RFSIM_NOISE_POWER_DB_COMMAND_VALUE",
            "instantaneous_mac_normalized_pusch_snr_db": (
                "GNB_MAC_PUSCH_POWER_CONTROL.snrx10_DIV_10_NOT_RAW_PHY_SNR"
            ),
            "cqi_domain_pusch_snr_db": (
                "GNB_MAC_PUSCH_POWER_CONTROL.(snrx10_PLUS_10_TIMES_txpower_calc)_DIV_10_"
                "QUANTIZED_0P5DB_AND_CENSORED_AT_UL_CQI_SATURATION"
            ),
            "scheduler_ema_snr_db": "GNB_MAC_UL_MCS_DECISION.avg_snr_x10_DIV_10",
            "selected_mcs": "GNB_MAC_UL_MCS_DECISION.selected_mcs",
            "final_mcs": "GNB_MAC_UL_MCS_DECISION.final_mcs",
        },
        "target/command/achieved signal distinction",
    )
    _validate_telemetry_declaration(config)

    _require_exact(
        _mapping(config.get("scheduler"), "scheduler"),
        {
            "policy_environment": "SCENESENSE_MCS_POLICY",
            "policy_value": "sinr",
            "force_mcs_environment": "SCENESENSE_FORCE_UL_MCS",
            "force_mcs_requirement": "UNSET",
            "power_control_target_parameter": "pusch_TargetSNRx10",
            "power_control_target_value_x10": 150,
            "power_control_target_semantics": (
                "FIXED_OAI_UE_POWER_CONTROL_TARGET_NOT_DESIRED_ACHIEVED_PUSCH_SNR_TRACE"
            ),
            "mcs_table": 0,
            "ul_layers": 1,
            "resource_blocks": 106,
            "numerology": 1,
            "band": 78,
            "scheduler_ema_constant": 0.975,
            "scheduler_ema_is_instantaneous": False,
            "selected_mcs_may_differ_from_final_mcs": True,
        },
        "scheduler",
    )
    _require_exact(
        _mapping(config.get("calibration"), "calibration"),
        {
            "status": "NOT_PERFORMED_INTERFACE_ONLY",
            "desired_achieved_pusch_snr_to_command_mapping": "NOT_DEFINED_UE_N1",
            "command_operating_bounds_db": "NOT_DEFINED_UE_N1",
            "attach_safe_achieved_snr_bounds_db": "NOT_DEFINED_UE_N1",
            "command_latency_or_jitter_bounds": "NOT_DEFINED_UE_N1",
            "desired_to_measured_achieved_error_or_lag": "NOT_DEFINED_UE_N1",
        },
        "no-calibration boundary",
    )

    revision = _mapping(config.get("oai_revision"), "oai_revision")
    _require_exact(
        revision,
        {
            "git_head": "7473cdb52e1cf3c40e1e1f189f03b2785bf15610",
            "branch": "scenesense-nrue-grant-trace",
            "dirty_tree_at_freeze": True,
            "revision_authority": "EXACT_FILE_HASHES_BELOW_COMMIT_ALONE_IS_INSUFFICIENT",
        },
        "OAI dirty-revision disclosure",
    )

    contract = _mapping(config.get("contract"), "contract")
    if set(contract) != {"path", "sha256"} or contract.get("path") != (
        "rl_agent/UE_N1_OAI_UL_ACTUATOR_INTERFACE_CONTRACT_V1.md"
    ) or not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("sha256", ""))):
        raise InterfaceFreezeError("UE-N1 contract seal declaration differs")

    sources = _mapping(config.get("sources"), "sources")
    if set(sources) != EXPECTED_SOURCE_KEYS:
        raise InterfaceFreezeError("exact OAI/source seal set differs")
    for label, spec_value in sources.items():
        spec = _mapping(spec_value, f"sources.{label}")
        if set(spec) != {"path", "sha256"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(spec.get("sha256", ""))
        ):
            raise InterfaceFreezeError(f"invalid source seal: {label}")

    runtime_artifacts = _mapping(config.get("runtime_artifacts"), "runtime_artifacts")
    if set(runtime_artifacts) != {"authority", "execution_claimed", "files"} or (
        runtime_artifacts.get("authority")
        != "CURRENT_ARTIFACT_SEALS_ONLY_REVERIFY_AT_UE_N2_PREFLIGHT"
        or runtime_artifacts.get("execution_claimed") is not False
    ):
        raise InterfaceFreezeError("runtime-artifact authority differs")
    runtime_files = _mapping(runtime_artifacts.get("files"), "runtime artifact files")
    expected_runtime_paths = {
        "gnb_softmodem": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem",
        "ue_softmodem": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem",
        "telnet_server_library": (
            "OAI/openairinterface5g/cmake_targets/ran_build/build/libtelnetsrv.so"
        ),
    }
    if set(runtime_files) != set(expected_runtime_paths):
        raise InterfaceFreezeError("runtime-artifact seal set differs")
    for label, expected_path in expected_runtime_paths.items():
        spec = _mapping(runtime_files[label], f"runtime_artifacts.{label}")
        if (
            set(spec) != {"path", "sha256"}
            or spec.get("path") != expected_path
            or not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", "")))
        ):
            raise InterfaceFreezeError(f"runtime-artifact declaration differs: {label}")

    mechanism = _mapping(config.get("mechanism_evidence"), "mechanism_evidence")
    if set(mechanism) != {
        "claim", "overall_experiment_status", "single_ue_calibration_claimed",
        "cadence_claimed", "numeric_bound_claimed", "files",
    }:
        raise InterfaceFreezeError("mechanism-evidence key set differs")
    required_mechanism = {
        "claim": "TWO_UE_CHANNELMOD_READ_MODIFY_READ_MECHANISM_ONLY",
        "overall_experiment_status": "FAILED_HOLD",
        "single_ue_calibration_claimed": False,
        "cadence_claimed": False,
        "numeric_bound_claimed": False,
    }
    for key, expected in required_mechanism.items():
        if mechanism.get(key) != expected:
            raise InterfaceFreezeError(f"mechanism-evidence boundary mismatch: {key}")
    evidence_files = _mapping(mechanism.get("files"), "mechanism evidence files")
    if set(evidence_files) != {"runtime_switch", "failed_terminal", "results_summary"}:
        raise InterfaceFreezeError("mechanism evidence file set differs")
    for label, spec_value in evidence_files.items():
        spec = _mapping(spec_value, f"mechanism_evidence.files.{label}")
        if set(spec) != {"path", "sha256"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(spec.get("sha256", ""))
        ):
            raise InterfaceFreezeError(f"mechanism evidence seal differs: {label}")

    output = _mapping(config.get("output"), "output")
    _require_exact(
        output,
        {
            "root": "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v1",
            "resolved_config_json": "resolved_config.json",
            "report_md": "REPORT.md",
            "manifest_json": "manifest.json",
            "terminal_json": "UE_N1_INTERFACE_FROZEN.json",
            "terminal_status": FROZEN_STATUS,
            "next_checklist_item": NEXT_ITEM,
        },
        "UE-N1 output/next-item",
    )
    return config


def _parse_t_event_fields(text: str, event_id: str) -> tuple[str, ...]:
    lines = text.splitlines()
    marker = f"ID = {event_id}"
    try:
        start = lines.index(marker)
    except ValueError as exc:
        raise InterfaceFreezeError(f"telemetry event absent from pinned source: {event_id}") from exc
    for line in lines[start + 1 :]:
        if line.startswith("ID = "):
            break
        stripped = line.strip()
        if stripped.startswith("FORMAT = "):
            fields = []
            for declaration in stripped.removeprefix("FORMAT = ").split(" : "):
                if "," not in declaration:
                    raise InterfaceFreezeError(f"cannot parse telemetry format: {event_id}")
                fields.append(declaration.rsplit(",", 1)[1])
            return tuple(fields)
    raise InterfaceFreezeError(f"telemetry format absent: {event_id}")


def _verify_predecessor(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = _mapping(config["predecessor"], "predecessor")
    bundle = _repo_path(str(spec["bundle_dir"]))
    terminal_path = _pinned(
        bundle / str(spec["terminal_json"]), str(spec["terminal_sha256"]), "A4 terminal"
    )
    manifest_path = _pinned(
        bundle / str(spec["manifest_json"]), str(spec["manifest_sha256"]), "A4 manifest"
    )
    registry_path = _pinned(
        bundle / str(spec["registry_csv"]), str(spec["registry_sha256"]), "A4 registry"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        terminal.get("status") != spec["required_status"]
        or terminal.get("next_checklist_item") != spec["required_next"]
        or terminal.get("profile_count") != spec["required_profiles"]
        or terminal.get("manifest_sha256") != spec["manifest_sha256"]
    ):
        raise InterfaceFreezeError("A4 terminal does not authorize UE-N1")
    if (
        manifest.get("status") != "FROZEN"
        or manifest.get("counts", {}).get("profiles") != 72
        or manifest.get("counts", {}).get("technically_valid") != 72
        or manifest.get("counts", {}).get("quality_masked") != 0
    ):
        raise InterfaceFreezeError("A4 manifest semantic gate failed")
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (
        len(rows) != 72
        or {int(row["action_index"]) for row in rows} != set(range(72))
        or len({row["profile_id"] for row in rows}) != 72
        or {row["technical_validity_status"] for row in rows} != {"TECHNICALLY_VALID"}
        or {row["quality_mask_applied"] for row in rows} != {"False"}
    ):
        raise InterfaceFreezeError("A4 registry semantic gate failed")
    return [
        _input_record(terminal_path, "predecessor", "UE-A4 terminal"),
        _input_record(manifest_path, "predecessor", "UE-A4 manifest"),
        _input_record(registry_path, "predecessor", "UE-A4 technical registry"),
    ]


def _verify_mechanism_evidence(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    mechanism = _mapping(config["mechanism_evidence"], "mechanism_evidence")
    paths: dict[str, Path] = {}
    for label, spec_value in _mapping(mechanism["files"], "mechanism files").items():
        spec = _mapping(spec_value, f"mechanism.{label}")
        paths[label] = _pinned(_repo_path(str(spec["path"])), str(spec["sha256"]), label)

    switch = json.loads(paths["runtime_switch"].read_text(encoding="utf-8"))
    names = {"rfsimu_channel_ue0", "rfsimu_channel_ue1"}
    if (
        switch.get("block") != "RUNTIME_SWITCH_R1"
        or switch.get("both_uplinks_modified") is not True
        or set(switch.get("before", {})) != names
        or set(switch.get("after", {})) != names
        or set(switch.get("active_uplink_objects", {}).get("active", {})) != names
        or not all(switch["active_uplink_objects"]["active"].values())
        or switch.get("active_uplink_objects", {}).get("fallback_lines") != []
    ):
        raise InterfaceFreezeError("two-UE runtime-switch mechanism evidence failed")
    for name in names:
        before = _mapping(switch["before"][name], f"before.{name}")
        after = _mapping(switch["after"][name], f"after.{name}")
        if (
            before.get("model_name") != name
            or after.get("model_name") != name
            or before.get("model_type") != "AWGN"
            or after.get("model_type") != "AWGN"
            or before.get("path_loss_db") != 0.0
            or after.get("path_loss_db") != 0.0
            or before.get("noise_power_db") != -50.0
            or after.get("noise_power_db") != -4.0
        ):
            raise InterfaceFreezeError(f"two-UE transition content failed: {name}")
        response = _mapping(switch["modify_responses"][name], f"response.{name}")
        expected_command = (
            f"channelmod modify {after['model_index']} noise_power_dB -4"
        )
        if response.get("command") != expected_command or "noise: -4.000000" not in str(
            response.get("response", "")
        ):
            raise InterfaceFreezeError(f"two-UE echoed modification failed: {name}")

    failed = json.loads(paths["failed_terminal"].read_text(encoding="utf-8"))
    summary = json.loads(paths["results_summary"].read_text(encoding="utf-8"))
    if failed != summary:
        raise InterfaceFreezeError("historical FAILED and results summary differ")
    error = str(failed.get("error", ""))
    if (
        failed.get("status") != "FAILED_HOLD"
        or failed.get("decision") != "HOLD_REPAIR"
        or failed.get("next_stage_launched") is not False
        or "off the registered strong SNR/MCS rung" not in error
        or "median_pusch_snr_db': 6.0" not in error
        or "median_ul_mcs': 8.0" not in error
    ):
        raise InterfaceFreezeError("historical failure boundary is not represented honestly")
    return [
        _input_record(paths["runtime_switch"], "mechanism_evidence", "two-UE runtime switch"),
        _input_record(paths["failed_terminal"], "mechanism_evidence", "failed terminal"),
        _input_record(paths["results_summary"], "mechanism_evidence", "failed results summary"),
    ]


def _verify_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for label, spec_value in _mapping(config["sources"], "sources").items():
        spec = _mapping(spec_value, f"sources.{label}")
        path = _pinned(_repo_path(str(spec["path"])), str(spec["sha256"]), label)
        paths[label] = path
        records.append(_input_record(path, "source", label))

    source_markers: dict[str, tuple[str, ...]] = {
        "channelmod_parser": (
            'strcmp(param,"ploss") == 0',
            'strcmp(param,"noise_power_dB") == 0',
            "defined_channels[cd_id]->noise_power_dB=dbl",
        ),
        "rfsim_channel_binding": (
            '"rfsimu_channel_%s%d"',
            '(bridge->role == SIMU_ROLE_SERVER) ? "ue" : "enB"',
            "Random channel %s in rfsimulator activated",
        ),
        "rfsim_channel_application": (
            "path_loss_dB / 20.0",
            "noise_power_dB / 10.0",
        ),
        "phy_pusch_snr": (
            "int SNRtimes10 = dB_fixed_x10(pusch->ulsch_power_tot)",
            "cqi=(640+SNRtimes10)/5",
        ),
        "scheduler_mapping_and_ema": (
            "#define PC_AVG_CNST 0.975f",
            "static const int SINRx10_MCS_mapping[29]",
            "get_mcs_from_SINRx10",
        ),
        "scheduler_ul_path": (
            'getenv("SCENESENSE_MCS_POLICY")',
            "get_mcs_from_SINRx10(current_BWP->mcs_table",
            "T_GNB_MAC_UL_MCS_DECISION",
        ),
        "rfsim_channel_config": (
            'model_name     = "rfsimu_channel_ue0"',
            'type           = "AWGN"',
            "ploss_dB       = 0",
        ),
        "telemetry_event_clock": (
            "clock_gettime(CLOCK_REALTIME, &T_HEADER_time)",
        ),
        "telemetry_csv_formatter": (
            "e.sending_time.tv_sec",
            "e.sending_time.tv_nsec / 1000",
        ),
        "gnb_radio_config": (
            "pusch_TargetSNRx10          = 150",
        ),
    }
    for label, markers in source_markers.items():
        text = paths[label].read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise InterfaceFreezeError(f"pinned {label} lost semantic marker: {marker}")

    telemetry_text = paths["telemetry_schema"].read_text(encoding="utf-8")
    for event_id, (_, expected_fields, _) in EXPECTED_TELEMETRY.items():
        actual_fields = _parse_t_event_fields(telemetry_text, event_id)
        if actual_fields != expected_fields:
            raise InterfaceFreezeError(
                f"pinned telemetry fields differ for {event_id}: {actual_fields}"
            )
    return records


def _verify_runtime_artifacts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = _mapping(config["runtime_artifacts"], "runtime_artifacts")
    for label, spec_value in _mapping(runtime["files"], "runtime files").items():
        spec = _mapping(spec_value, f"runtime_artifacts.{label}")
        path = _pinned(_repo_path(str(spec["path"])), str(spec["sha256"]), label)
        records.append(_input_record(path, "runtime_artifact", label))
    return records


def _verify_all_inputs(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    contract_spec = _mapping(config["contract"], "contract")
    contract_path = _pinned(
        _repo_path(str(contract_spec["path"])), str(contract_spec["sha256"]), "UE-N1 contract"
    )
    records = [_input_record(Path(config_path), "config", "UE-N1 frozen config")]
    records.append(_input_record(Path(__file__), "assembler", "UE-N1 assembler/validator"))
    records.append(_input_record(contract_path, "contract", "UE-N1 interface contract"))
    records.extend(_verify_predecessor(config))
    records.extend(_verify_sources(config))
    records.extend(_verify_runtime_artifacts(config))
    records.extend(_verify_mechanism_evidence(config))
    return records


def _report(config: Mapping[str, Any]) -> str:
    return """# UE-N1 OAI uplink actuator interface v1

**Status:** `FROZEN_INTERFACE_ONLY`

UE-N1 freezes the single-UE, gNB-side RFsim uplink actuator as the exact
`rfsimu_channel_ue0` AWGN object and the exact mutable parameter
`noise_power_dB`. Its session-local integer index must be resolved dynamically
from `channelmod show current`; `ploss` stays zero and global
`noise_power_dBFS` stays unset.

The UE attaches at the clean `-50` command before any runtime change. UE-N2
must use one persistent Telnet connection, a monotonic 100-ms schedule, and no
catch-up burst. The returned response/prompt is an ACK upper bound for command
handler completion, not a physical application timestamp.

Desired achieved PUSCH SNR, RFsim command, MAC-normalized instantaneous PUSCH
SNR, scheduler EMA SNR, and fixed OAI `pusch_TargetSNRx10` remain distinct. No
numeric mapping, command range, achieved-SNR bound, latency bound, or attach-
safe lower limit is claimed here.

Current SINR-policy traces do not provide direct UL BLER. BLER remains
`UNAVAILABLE_UNRESOLVED`; UE grant rounds are only a retransmission proxy, and
missing radio/backlog observations are never zero-filled.

The pinned two-UE runtime switch is mechanism evidence only. Its enclosing
experiment ended `FAILED_HOLD`; it is not a passing single-UE calibration or
cadence result.

No launcher/runtime source was edited, and no OAI, CARLA, Telnet, or other
socket execution was performed. The next checklist item is **UE-N2**.
"""


def _terminal_payload(created_at: str) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "interface_id": INTERFACE_ID,
        "status": FROZEN_STATUS,
        "created_at": created_at,
        "claim_scope": "INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION",
        "predecessor": "UE-A4",
        "next_checklist_item": NEXT_ITEM,
        "actuator": "GNB_RFSIM_RFSIMU_CHANNEL_UE0_NOISE_POWER_DB",
        "model_index_resolution": "DYNAMIC_BY_EXACT_NAME",
        "persistent_telnet_required_next": True,
        "schedule_period_ms": 100,
        "numeric_calibration_status": "NOT_PERFORMED",
        "numeric_bounds_status": "NOT_DEFINED",
        "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED",
        "runtime_executed": False,
        "socket_executed": False,
        "oai_run": False,
        "carla_run": False,
    }


def assemble(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    now: str | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    target = (
        _repo_path(str(config["output"]["root"]))
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if target.exists():
        raise InterfaceFreezeError(f"refusing to overwrite create-only UE-N1 bundle: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    input_records = _verify_all_inputs(config, config_path)
    created_at = now or utc_now()
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        resolved_name = str(config["output"]["resolved_config_json"])
        report_name = str(config["output"]["report_md"])
        manifest_name = str(config["output"]["manifest_json"])
        terminal_name = str(config["output"]["terminal_json"])
        _write_json(temp / resolved_name, config)
        (temp / report_name).write_text(_report(config), encoding="utf-8")
        outputs = []
        for name in (resolved_name, report_name):
            path = temp / name
            outputs.append(
                {"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )
        terminal_payload = _terminal_payload(created_at)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "interface_id": INTERFACE_ID,
            "status": FROZEN_STATUS,
            "created_at": created_at,
            "claim_scope": "INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION",
            "authority": config["authority"],
            "interface": {
                "ue_count": 1,
                "direction": "UPLINK_ONLY",
                "channel_model_name": "rfsimu_channel_ue0",
                "model_index_resolution": "DYNAMIC_BY_EXACT_NAME",
                "mutable_parameter": "noise_power_dB",
                "fixed_path_loss_db": 0,
                "global_noise_requirement": "UNSET",
                "clean_attach_and_restore_command_db": -50,
                "persistent_telnet_required_next": True,
                "schedule_period_ms": 100,
                "catch_up_policy": "NEVER_BURST_OBSOLETE_COMMANDS",
                "response_ack_semantics": (
                    "UPPER_BOUND_FOR_HANDLER_COMPLETION_NOT_APPLICATION_TIMESTAMP"
                ),
            },
            "gates": {
                "a4_predecessor": "PASS_PINNED_72_UNFILTERED_TECHNICAL_ACTIONS",
                "single_ue_ul_scope": "PASS",
                "dynamic_model_index": "PASS_DECLARED",
                "awgn_noise_power_db": "PASS_DECLARED",
                "ploss_zero_global_noise_unset": "PASS_DECLARED",
                "clean_attach_then_runtime": "PASS_DECLARED",
                "persistent_telnet": "REQUIRED_UE_N2_NOT_EXECUTED",
                "monotonic_100ms_no_catch_up": "PASS_DECLARED",
                "target_command_achieved_distinct": "PASS_DECLARED",
                "ack_not_apply_timestamp": "PASS_DECLARED",
                "telemetry_exact_source_schema": "PASS_PINNED",
                "runtime_artifact_seals": "PASS_CURRENT_REVERIFY_UE_N2",
                "direct_ul_bler": "UNAVAILABLE_UNRESOLVED_NOT_ZERO",
                "two_ue_mechanism_evidence": "PASS_MECHANISM_ONLY_OVERALL_RUN_FAILED_HOLD",
                "numeric_calibration": "NOT_AUTHORIZED_NOT_PERFORMED",
                "numeric_bounds": "NOT_AUTHORIZED_NOT_DEFINED",
                "runtime_or_socket": "NOT_AUTHORIZED_NOT_EXECUTED",
            },
            "deferred": [
                "PERSISTENT_TELNET_IMPLEMENTATION_AND_100MS_REPLAY",
                "DESIRED_ACHIEVED_PUSCH_SNR_TO_COMMAND_CALIBRATION",
                "ATTACH_SAFE_ACHIEVED_SNR_BOUNDS",
                "COMMAND_JITTER_LAG_AND_RADIO_RESPONSE",
            ],
            "inputs": input_records,
            "outputs": outputs,
            "terminal_decision_path": terminal_name,
            "terminal_decision_payload": terminal_payload,
            "terminal_decision_payload_sha256": hashlib.sha256(
                canonical_json_bytes(terminal_payload)
            ).hexdigest(),
        }
        _write_json(temp / manifest_name, manifest)
        terminal = {
            **terminal_payload,
            "manifest_sha256": sha256_file(temp / manifest_name),
        }
        _write_json(temp / terminal_name, terminal)
        validate_bundle(temp, config_path=config_path)
        if target.exists():
            raise InterfaceFreezeError(
                f"refusing to overwrite create-only UE-N1 bundle: {target}"
            )
        os.rename(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def validate_bundle(
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    manifest_name = str(config["output"]["manifest_json"])
    terminal_name = str(config["output"]["terminal_json"])
    manifest_path = output_dir / manifest_name
    terminal_path = output_dir / terminal_name
    if not manifest_path.is_file() or not terminal_path.is_file():
        raise InterfaceFreezeError("UE-N1 manifest or terminal is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if set(manifest) != {
        "schema", "interface_id", "status", "created_at", "claim_scope",
        "authority", "interface", "gates", "deferred", "inputs", "outputs",
        "terminal_decision_path", "terminal_decision_payload",
        "terminal_decision_payload_sha256",
    }:
        raise InterfaceFreezeError("UE-N1 manifest key set differs")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("interface_id") != INTERFACE_ID
        or manifest.get("status") != FROZEN_STATUS
        or manifest.get("claim_scope") != "INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION"
        or manifest.get("authority") != EXPECTED_AUTHORITY
    ):
        raise InterfaceFreezeError("UE-N1 manifest schema/status/authority mismatch")
    expected_gates = {
        "a4_predecessor": "PASS_PINNED_72_UNFILTERED_TECHNICAL_ACTIONS",
        "single_ue_ul_scope": "PASS",
        "dynamic_model_index": "PASS_DECLARED",
        "awgn_noise_power_db": "PASS_DECLARED",
        "ploss_zero_global_noise_unset": "PASS_DECLARED",
        "clean_attach_then_runtime": "PASS_DECLARED",
        "persistent_telnet": "REQUIRED_UE_N2_NOT_EXECUTED",
        "monotonic_100ms_no_catch_up": "PASS_DECLARED",
        "target_command_achieved_distinct": "PASS_DECLARED",
        "ack_not_apply_timestamp": "PASS_DECLARED",
        "telemetry_exact_source_schema": "PASS_PINNED",
        "runtime_artifact_seals": "PASS_CURRENT_REVERIFY_UE_N2",
        "direct_ul_bler": "UNAVAILABLE_UNRESOLVED_NOT_ZERO",
        "two_ue_mechanism_evidence": "PASS_MECHANISM_ONLY_OVERALL_RUN_FAILED_HOLD",
        "numeric_calibration": "NOT_AUTHORIZED_NOT_PERFORMED",
        "numeric_bounds": "NOT_AUTHORIZED_NOT_DEFINED",
        "runtime_or_socket": "NOT_AUTHORIZED_NOT_EXECUTED",
    }
    if manifest.get("gates") != expected_gates:
        raise InterfaceFreezeError("UE-N1 manifest gate set differs")
    if manifest.get("deferred") != [
        "PERSISTENT_TELNET_IMPLEMENTATION_AND_100MS_REPLAY",
        "DESIRED_ACHIEVED_PUSCH_SNR_TO_COMMAND_CALIBRATION",
        "ATTACH_SAFE_ACHIEVED_SNR_BOUNDS",
        "COMMAND_JITTER_LAG_AND_RADIO_RESPONSE",
    ]:
        raise InterfaceFreezeError("UE-N1 deferred-work declaration differs")
    expected_interface = {
        "ue_count": 1,
        "direction": "UPLINK_ONLY",
        "channel_model_name": "rfsimu_channel_ue0",
        "model_index_resolution": "DYNAMIC_BY_EXACT_NAME",
        "mutable_parameter": "noise_power_dB",
        "fixed_path_loss_db": 0,
        "global_noise_requirement": "UNSET",
        "clean_attach_and_restore_command_db": -50,
        "persistent_telnet_required_next": True,
        "schedule_period_ms": 100,
        "catch_up_policy": "NEVER_BURST_OBSOLETE_COMMANDS",
        "response_ack_semantics": (
            "UPPER_BOUND_FOR_HANDLER_COMPLETION_NOT_APPLICATION_TIMESTAMP"
        ),
    }
    if manifest.get("interface") != expected_interface:
        raise InterfaceFreezeError("UE-N1 manifest interface snapshot differs")

    resolved_name = str(config["output"]["resolved_config_json"])
    report_name = str(config["output"]["report_md"])
    output_records = manifest.get("outputs")
    if not isinstance(output_records, list) or len(output_records) != 2 or {
        record.get("path") for record in output_records if isinstance(record, Mapping)
    } != {resolved_name, report_name}:
        raise InterfaceFreezeError("UE-N1 output seal set differs")
    for value in output_records:
        record = _mapping(value, "output seal")
        if set(record) != {"path", "sha256", "bytes"}:
            raise InterfaceFreezeError("UE-N1 output seal fields differ")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise InterfaceFreezeError("UE-N1 output path must be flat")
        path = (output_dir / relative).resolve()
        if (
            path.parent != output_dir
            or not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != int(record.get("bytes", -1))
        ):
            raise InterfaceFreezeError(f"UE-N1 output seal mismatch: {relative}")
    resolved = json.loads((output_dir / resolved_name).read_text(encoding="utf-8"))
    if resolved != config:
        raise InterfaceFreezeError("UE-N1 resolved config differs from frozen config")
    if (output_dir / report_name).read_text(encoding="utf-8") != _report(config):
        raise InterfaceFreezeError("UE-N1 deterministic report content differs")

    expected_inputs = _verify_all_inputs(config, config_path)
    if manifest.get("inputs") != expected_inputs:
        raise InterfaceFreezeError("UE-N1 input seals differ from reconstructed evidence")

    payload = dict(terminal)
    manifest_sha = payload.pop("manifest_sha256", None)
    if manifest_sha != sha256_file(manifest_path):
        raise InterfaceFreezeError("UE-N1 terminal does not seal the manifest")
    if payload != _terminal_payload(str(manifest.get("created_at"))):
        raise InterfaceFreezeError("UE-N1 terminal payload differs")
    if manifest.get("terminal_decision_path") != terminal_name:
        raise InterfaceFreezeError("UE-N1 terminal path differs")
    if manifest.get("terminal_decision_payload") != payload or manifest.get(
        "terminal_decision_payload_sha256"
    ) != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise InterfaceFreezeError("UE-N1 terminal payload seal differs")

    expected_names = {manifest_name, terminal_name, resolved_name, report_name}
    actual_names = {path.name for path in output_dir.iterdir()}
    if actual_names != expected_names:
        raise InterfaceFreezeError("UE-N1 bundle contains unexpected or missing files")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate", type=Path, help="validate an existing bundle only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            manifest = validate_bundle(args.validate, config_path=args.config)
            print(f"Validated UE-N1 bundle: {Path(args.validate).resolve()}")
            print(f"Status: {manifest['status']} (next {NEXT_ITEM})")
            return 0
        output = assemble(args.config, args.output_dir)
        print(f"UE-N1 interface bundle: {output}")
        print(f"Status: {FROZEN_STATUS} (next {NEXT_ITEM})")
        return 0
    except (InterfaceFreezeError, OSError, ValueError) as exc:
        print(f"UE-N1 freeze failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

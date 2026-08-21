"""Create and validate the final UE-N1 v2 actuator interface freeze.

This is an offline evidence assembler. It has no Telnet client, launcher, or
runtime path and cannot execute OAI or CARLA. The immutable v1 bundle is pinned
as superseded pre-final evidence and is never modified.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n1_freeze_oai_ul_actuator as v1  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n1_oai_ul_actuator_interface_v2.json"
CONFIG_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_config.v2"
MANIFEST_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_manifest.v2"
TERMINAL_SCHEMA = "scenesense.ue_n1_oai_ul_actuator_interface_decision.v2"
INTERFACE_ID = "ue_n1_oai_ul_actuator_interface_v2"
STATUS = "FROZEN_INTERFACE_ONLY"
NEXT_ITEM = "UE-N2"
FROZEN_CONFIG_SHA256 = "78193f8aaa9c6a3facbb86d22b107d4a142a54b280c69c90cd0372149d47ae82"

UTC_CREATED_AT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
COMMAND_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_TOP_LEVEL = {
    "schema", "interface_id", "repository_root", "authority", "supersedes",
    "predecessor", "scope", "actuator", "attach_lifecycle",
    "control_transport", "schedule", "command_timing", "causal_classification",
    "policy_availability", "raw_event_envelope", "signal_contract", "telemetry",
    "scheduler", "calibration", "oai_revision", "contract", "sources",
    "runtime_artifacts", "mechanism_evidence", "output",
}
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
CORE_SOURCE_KEYS = set(v1.EXPECTED_SOURCE_KEYS)
V2_SOURCE_KEYS = CORE_SOURCE_KEYS | {"v1_offline_helper_dependency"}
RUNTIME_PATHS = {
    "gnb_softmodem": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem",
    "ue_softmodem": "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem",
    "telnet_server_library": (
        "OAI/openairinterface5g/cmake_targets/ran_build/build/libtelnetsrv.so"
    ),
    "rfsimulator_library": (
        "OAI/openairinterface5g/cmake_targets/ran_build/build/librfsimulator.so"
    ),
}
RAW_EVENT_FIELDS = [
    "ran_epoch_id", "control_session_id", "source_event_id", "source_event_index",
    "source_event_realtime_sec", "source_event_realtime_nsec",
    "source_event_timestamp_ns", "rnti", "frame", "slot",
    "unwrapped_absolute_slot", "collector_ingest_wall_time_ns",
    "collector_ingest_monotonic_ns", "raw_event_sha256", "missing_reason_code",
]
COMMAND_FIELDS = [
    "trace_id", "trace_index", "desired_achieved_pusch_snr_db",
    "commanded_noise_power_db", "scheduled_monotonic_ns", "send_monotonic_ns",
    "send_wall_time_ns", "response_received_monotonic_ns",
    "response_received_wall_time_ns", "control_session_id", "resolved_model_index",
    "resolved_model_name", "resolved_model_owner", "resolved_model_type",
    "show_current_response_sha256", "response_text_sha256", "echoed_owner",
    "echoed_noise_power_db", "echoed_path_loss_db", "status",
]


class InterfaceV2Error(RuntimeError):
    """Raised when v2 declarations, pins, or bundle bytes fail closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_created_at(value: Any) -> str:
    if not isinstance(value, str) or not UTC_CREATED_AT_RE.fullmatch(value):
        raise InterfaceV2Error("created_at must be canonical UTC ISO with six fractional digits and Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise InterfaceV2Error("created_at is not a valid UTC calendar timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise InterfaceV2Error("created_at is not UTC")
    if parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") != value:
        raise InterfaceV2Error("created_at is not canonical UTC ISO")
    return value


def validate_commanded_noise_power_literal(value: Any) -> Decimal:
    if not isinstance(value, str) or value == "-0" or not COMMAND_RE.fullmatch(value):
        raise InterfaceV2Error("commanded_noise_power_db is not a canonical base-10 decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise InterfaceV2Error("commanded_noise_power_db is not finite") from exc
    if not parsed.is_finite():
        raise InterfaceV2Error("commanded_noise_power_db is not finite")
    # OAI consumes this exact token with C ``atof`` into a double. Decimal
    # finiteness alone is insufficient: a sufficiently long finite decimal
    # overflows binary64 to infinity. This is an input-representability guard,
    # not a calibrated operating bound.
    try:
        oai_binary64 = float(value)
    except (OverflowError, ValueError) as exc:
        raise InterfaceV2Error(
            "commanded_noise_power_db is not finite in OAI binary64"
        ) from exc
    if not math.isfinite(oai_binary64):
        raise InterfaceV2Error(
            "commanded_noise_power_db is not finite in OAI binary64"
        )
    oai_binary32 = ctypes.c_float(oai_binary64).value
    if not math.isfinite(oai_binary32):
        raise InterfaceV2Error(
            "commanded_noise_power_db is not finite in OAI binary32 storage"
        )
    return parsed


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterfaceV2Error(f"{label} must be a mapping")
    return value


def _exact(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(actual) != dict(expected):
        raise InterfaceV2Error(f"{label} contract mismatch")


def _repo_path(relative: str) -> Path:
    path = (ROOT / str(relative)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise InterfaceV2Error(f"path escapes repository: {relative}") from exc
    return path


def _pinned(path: Path, expected: str, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise InterfaceV2Error(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise InterfaceV2Error(f"{label} hash drift: expected={expected} actual={actual}")
    return path


def _record(path: Path, kind: str, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "kind": kind,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any target entry.

    Linux ``renameat2(RENAME_NOREPLACE)`` closes the exists-check/rename race.
    There is deliberately no unsafe fallback: lack of kernel/libc support
    fails the create-only publication.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise InterfaceV2Error("atomic create-only renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(Path(source)),
        at_fdcwd,
        os.fsencode(Path(target)),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise InterfaceV2Error(f"refusing to overwrite create-only UE-N1 v2 bundle: {target}")
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise InterfaceV2Error("atomic create-only renameat2 is unsupported")
    raise OSError(error_number, os.strerror(error_number), str(target))


def _validate_telemetry(config: Mapping[str, Any]) -> None:
    telemetry = _mapping(config["telemetry"], "telemetry")
    if set(telemetry) != {
        "source_event_timestamp_clock", "csv_time_semantics",
        "direct_ul_bler_status", "missing_direct_bler_is_zero",
        "ue_grant_round_semantics", "lower_bound_acceptance", "events",
        "ema_context_fields",
    }:
        raise InterfaceV2Error("telemetry key set differs")
    expected_scalars = {
        "source_event_timestamp_clock": "CLOCK_REALTIME",
        "csv_time_semantics": (
            "PRODUCER_EVENT_EMISSION_TIME_NOT_COLLECTOR_INGEST_OR_UE_POLICY_AVAILABILITY"
        ),
        "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED_CURRENT_SINR_TRACE",
        "missing_direct_bler_is_zero": False,
        "ue_grant_round_semantics": "RETRANSMISSION_PROXY_ONLY",
        "lower_bound_acceptance": (
            "REQUIRE_GENUINE_UL_CRC_HARQ_OUTCOME_OR_EXPLICIT_MISSING_EVIDENCE_GATE"
        ),
        "ema_context_fields": [
            "accepted_pusch_observations_since_command", "age_since_command_ns"
        ],
    }
    for key, expected in expected_scalars.items():
        if telemetry.get(key) != expected:
            raise InterfaceV2Error(f"telemetry semantics differ: {key}")
    records = telemetry.get("events")
    if not isinstance(records, list) or len(records) != len(v1.EXPECTED_TELEMETRY):
        raise InterfaceV2Error("telemetry event count differs")
    by_id = {str(_mapping(row, "telemetry event").get("id")): row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(v1.EXPECTED_TELEMETRY):
        raise InterfaceV2Error("telemetry event ID set differs")
    expected_semantics = {
        "GNB_MAC_PUSCH_POWER_CONTROL": (
            "MAC_NORMALIZED_PUSCH_SNR_POST_ACTION_COLLECTOR_EVIDENCE"
        ),
        "GNB_MAC_UL_MCS_DECISION": (
            "SCHEDULER_EMA_AND_SELECTED_TO_FINAL_MCS_POST_ACTION_COLLECTOR_EVIDENCE"
        ),
        "GNB_MAC_UL": "SCHEDULED_GRANT_NOT_CONFIRMED_DELIVERY",
        "NRUE_MAC_DCI_GRANT": "UE_GRANT_RETRANSMISSION_PROXY_NOT_DIRECT_CRC_OR_BLER",
        "NRUE_MAC_RLC_BUFFER_STATUS": (
            "RLC_BUFFER_POST_ACTION_COLLECTOR_EVIDENCE_NO_ZERO_FILL"
        ),
        "NRUE_MAC_BSR_STATUS": "BSR_POST_ACTION_COLLECTOR_EVIDENCE_NO_ZERO_FILL",
        "GNB_MAC_BLER_MCS_DECISION": (
            "NOT_DIRECT_UL_BLER_EVIDENCE_UNDER_CURRENT_SINR_PATH"
        ),
    }
    for event_id, (_, fields, _) in v1.EXPECTED_TELEMETRY.items():
        expected = {
            "id": event_id,
            "fields": list(fields),
            "semantics": expected_semantics[event_id],
        }
        if dict(_mapping(by_id[event_id], event_id)) != expected:
            raise InterfaceV2Error(f"telemetry declaration drift: {event_id}")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterfaceV2Error(f"cannot load UE-N1 v2 config: {path}") from exc
    if config.get("schema") != CONFIG_SCHEMA or config.get("interface_id") != INTERFACE_ID:
        raise InterfaceV2Error("invalid UE-N1 v2 schema/interface ID")
    if set(config) != EXPECTED_TOP_LEVEL:
        raise InterfaceV2Error("UE-N1 v2 top-level key set differs")
    if sha256_file(path) != FROZEN_CONFIG_SHA256:
        raise InterfaceV2Error("UE-N1 v2 config does not match its canonical seal")
    if (path.parent / str(config["repository_root"])).resolve() != ROOT:
        raise InterfaceV2Error("UE-N1 v2 repository root differs")
    _exact(_mapping(config["authority"], "authority"), EXPECTED_AUTHORITY, "authority")

    supersedes = _mapping(config["supersedes"], "supersedes")
    if (
        supersedes.get("interface_id") != "ue_n1_oai_ul_actuator_interface_v1"
        or supersedes.get("authority_status") != "SUPERSEDED_PRE_FINAL_OBSERVATION_AUDIT"
        or supersedes.get("v1_bytes_mutable") is not False
        or supersedes.get("v1_remains_execution_authority") is not False
        or set(supersedes) != {
            "interface_id", "authority_status", "bundle_dir", "manifest_json",
            "manifest_sha256", "terminal_json", "terminal_sha256",
            "resolved_config_json", "resolved_config_sha256", "report_md",
            "report_sha256", "v1_bytes_mutable", "v1_remains_execution_authority",
        }
    ):
        raise InterfaceV2Error("v1 supersession declaration differs")

    predecessor = _mapping(config["predecessor"], "predecessor")
    if set(predecessor) != {
        "checklist_item", "bundle_dir", "terminal_json", "terminal_sha256",
        "manifest_json", "manifest_sha256", "registry_csv", "registry_sha256",
        "required_status", "required_next", "required_profiles",
    } or any(
        predecessor.get(key) != value
        for key, value in {
            "checklist_item": "UE-A4", "required_status": "FROZEN",
            "required_next": "UE-N1", "required_profiles": 72,
        }.items()
    ):
        raise InterfaceV2Error("A4 predecessor declaration differs")

    scope = _mapping(config["scope"], "scope")
    _exact(scope, {
        "ue_count": 1, "direction": "UPLINK_ONLY", "rf_simulator_server_role": "GNB",
        "downlink_actuation_authorized": False, "multi_ue_actuation_authorized": False,
    }, "scope")
    actuator = _mapping(config["actuator"], "actuator")
    if (
        set(actuator) != {
            "subsystem", "channel_model_name", "channel_model_owner",
            "channel_model_type", "model_index_binding",
            "hardcoded_model_index_authorized", "resolution_command",
            "modify_command_template", "oai_mutable_parameter",
            "canonical_command_field", "command_value_semantics",
            "command_value_lexical_contract", "oai_atof_is_input_validation",
            "fixed_path_loss_parameter", "fixed_path_loss_db",
            "global_noise_parameter", "global_noise_requirement",
            "prohibited_channel_model_names",
        }
        or actuator.get("channel_model_name") != "rfsimu_channel_ue0"
        or actuator.get("model_index_binding")
        != "RESOLVE_EXACT_NAME_EACH_GNB_CONTROL_SESSION"
        or actuator.get("hardcoded_model_index_authorized") is not False
        or actuator.get("oai_mutable_parameter") != "noise_power_dB"
        or actuator.get("canonical_command_field") != "commanded_noise_power_db"
        or actuator.get("command_value_lexical_contract")
        != (
            "FINITE_CANONICAL_BASE10_DECIMAL_STRING_NO_EXPONENT_WHITESPACE_"
            "OR_CONTROL_CHARACTERS_AND_FINITE_OAI_BINARY32_STORAGE"
        )
        or actuator.get("fixed_path_loss_db") != 0
        or actuator.get("global_noise_requirement") != "UNSET"
        or actuator.get("oai_atof_is_input_validation") is not False
    ):
        raise InterfaceV2Error("actuator distinction differs")

    attach = _mapping(config["attach_lifecycle"], "attach_lifecycle")
    if (
        set(attach) != {
            "initial_and_restore_commanded_noise_power_db",
            "source_template_initial_noise_power_db",
            "source_template_is_effective_runtime_config",
            "n2_effective_runtime_config_requirement",
            "pre_attach_show_current_required_noise_power_db", "runtime_command_gate",
            "restore_on_normal_shutdown", "restore_on_failure",
            "tunnel_presence_alone_is_sufficient",
        }
        or attach.get("initial_and_restore_commanded_noise_power_db") != "-50"
        or attach.get("source_template_initial_noise_power_db") != -10
        or attach.get("source_template_is_effective_runtime_config") is not False
        or attach.get("pre_attach_show_current_required_noise_power_db") != -50
        or attach.get("runtime_command_gate") != [
            "UE_ATTACHED_CURRENT_SESSION", "OAITUN_UE1_PRESENT", "REACHABILITY_PASS",
            "UPLINK_TRAFFIC_ACTIVE", "TELEMETRY_RECORDER_READY",
            "EXACT_ACTIVE_UL_CHANNEL_OBJECT_VALIDATED",
            "FRESH_CURRENT_SESSION_PUSCH_OBSERVATION_PRESENT",
        ]
    ):
        raise InterfaceV2Error("attach lifecycle differs")
    validate_commanded_noise_power_literal(
        attach["initial_and_restore_commanded_noise_power_db"]
    )
    control = _mapping(config["control_transport"], "control_transport")
    if (
        set(control) != {
            "protocol", "host", "port", "server_process", "n2_connection_lifecycle",
            "reconnect_per_command_authorized", "response_requirement",
            "modify_response_echoes_model_name_or_index",
            "resolution_binding_lifetime", "connection_loss_trace_result",
            "post_loss_reconnect_scope", "cleanup_success_can_change_failed_to_pass",
            "fail_closed_conditions",
        }
        or control.get("n2_connection_lifecycle") != "ONE_PERSISTENT_CONNECTION_PER_TRACE"
        or control.get("reconnect_per_command_authorized") is not False
        or control.get("connection_loss_trace_result") != "FAILED"
        or control.get("cleanup_success_can_change_failed_to_pass") is not False
    ):
        raise InterfaceV2Error("persistent control lifecycle differs")
    schedule = _mapping(config["schedule"], "schedule")
    if schedule != {
        "clock": "time.monotonic_ns", "period_ms": 100,
        "formula": "anchor_monotonic_ns + trace_index * 100000000",
        "trace_index_contract": "UNIQUE_CONTIGUOUS_ZERO_BASED",
        "strictly_monotonic": True, "relative_to_actual_completion": False,
        "catch_up_policy": "NEVER_BURST_OBSOLETE_COMMANDS",
        "numeric_lateness_or_jitter_acceptance": "NOT_DEFINED_UE_N1",
    }:
        raise InterfaceV2Error("100-ms schedule differs")

    timing = _mapping(config["command_timing"], "command_timing")
    if (
        set(timing) != {
            "required_fields", "send_time_semantics", "response_received_semantics",
            "prohibited_fields", "post_command_observation_semantics",
            "n2_first_effect_fields", "first_effect_semantics",
        }
        or timing.get("required_fields") != COMMAND_FIELDS
        or "ACK_UPPER_BOUND" not in str(timing.get("response_received_semantics"))
        or timing.get("prohibited_fields")
        != ["command_applied_at", "command_applied_at_ns", "application_timestamp_ns"]
        or timing.get("first_effect_semantics")
        != "MEASURED_STEP_RESPONSE_ESTIMATE_NOT_COMMAND_APPLICATION_TIMESTAMP"
    ):
        raise InterfaceV2Error("command timing/effect distinction differs")

    causal = _mapping(config["causal_classification"], "causal_classification")
    if (
        set(causal) != {
            "control_and_evaluation_only_fields",
            "control_fields_in_policy_state_authorized",
            "post_action_collector_evidence_fields",
            "collector_evidence_is_ue_policy_observation",
            "future_trace_values_in_policy_state_authorized",
            "ground_truth_in_policy_state_authorized",
        }
        or causal.get("control_and_evaluation_only_fields")
        != ["desired_achieved_pusch_snr_db", "commanded_noise_power_db"]
        or causal.get("control_fields_in_policy_state_authorized") is not False
        or causal.get("collector_evidence_is_ue_policy_observation") is not False
        or causal.get("future_trace_values_in_policy_state_authorized") is not False
        or causal.get("ground_truth_in_policy_state_authorized") is not False
    ):
        raise InterfaceV2Error("causal field classification differs")
    policy = _mapping(config["policy_availability"], "policy_availability")
    if (
        set(policy) != {
            "status", "collector_ingest_is_policy_availability",
            "required_future_fields", "admission_predicate",
            "availability_must_be_measured_non_null",
            "feedback_path_must_be_ue_visible_and_measured",
        }
        or policy.get("status") != "UNBOUND_UNTIL_MEASURED_UE_VISIBLE_FEEDBACK_PATH"
        or policy.get("collector_ingest_is_policy_availability") is not False
        or policy.get("admission_predicate")
        != (
            "policy_observation_available_monotonic_ns <= decision_cutoff_monotonic_ns "
            "AND observation.ran_epoch_id == decision.ran_epoch_id AND "
            "observation.control_session_id == decision.control_session_id"
        )
        or policy.get("availability_must_be_measured_non_null") is not True
        or policy.get("feedback_path_must_be_ue_visible_and_measured") is not True
    ):
        raise InterfaceV2Error("future policy availability gate differs")

    envelope = _mapping(config["raw_event_envelope"], "raw_event_envelope")
    if (
        set(envelope) != {
            "required_fields", "source_event_index_scope", "source_timestamp_clock",
            "collector_ingest_clock", "frame_slot_alone_authorized",
            "frame_slot_wrap_period_s", "missing_numeric_value",
            "missing_reason_required_when_value_absent", "missing_pusch_semantics",
            "missing_reason_codes", "zero_fill_authorized",
            "forward_fill_authorized",
        }
        or envelope.get("required_fields") != RAW_EVENT_FIELDS
        or envelope.get("source_event_index_scope")
        != "STRICTLY_MONOTONIC_WITHIN_RAN_EPOCH"
        or envelope.get("frame_slot_alone_authorized") is not False
        or envelope.get("missing_numeric_value", "sentinel") is not None
        or envelope.get("missing_reason_required_when_value_absent") is not True
        or envelope.get("missing_pusch_semantics")
        != "MISSING_UNRESOLVED_NOT_DTX_WITHOUT_DTX_EVIDENCE"
        or envelope.get("zero_fill_authorized") is not False
        or envelope.get("forward_fill_authorized") is not False
    ):
        raise InterfaceV2Error("raw event envelope/missingness contract differs")

    signals = _mapping(config["signal_contract"], "signal_contract")
    if set(signals) != {
        "desired_achieved_pusch_snr_db", "commanded_noise_power_db",
        "oai_pusch_TargetSNRx10", "gnb_mac_normalized_pusch_snr_db",
        "cqi_domain_pusch_snr_db", "scheduler_ema_snr_db", "selected_mcs",
        "final_mcs",
    } or "POLICY_OBSERVATION" not in str(signals["commanded_noise_power_db"]):
        raise InterfaceV2Error("signal distinction differs")
    _validate_telemetry(config)

    scheduler = _mapping(config["scheduler"], "scheduler")
    if (
        set(scheduler) != {
            "policy_environment", "policy_value", "force_mcs_environment",
            "force_mcs_requirement", "power_control_target_parameter",
            "power_control_target_value_x10", "power_control_target_semantics",
            "mcs_table", "ul_layers", "resource_blocks", "numerology", "band",
            "scheduler_ema_constant", "scheduler_ema_is_instantaneous",
            "selected_mcs_may_differ_from_final_mcs",
        }
        or scheduler.get("policy_value") != "sinr"
        or scheduler.get("force_mcs_requirement") != "UNSET"
        or scheduler.get("power_control_target_parameter") != "pusch_TargetSNRx10"
        or scheduler.get("power_control_target_value_x10") != 150
        or scheduler.get("mcs_table") != 0
        or scheduler.get("ul_layers") != 1
        or scheduler.get("resource_blocks") != 106
        or scheduler.get("numerology") != 1
        or scheduler.get("band") != 78
    ):
        raise InterfaceV2Error("scheduler freeze differs")
    calibration = _mapping(config["calibration"], "calibration")
    if set(calibration) != {
        "status", "desired_achieved_pusch_snr_to_commanded_noise_mapping",
        "commanded_noise_operating_bounds_db", "attach_safe_achieved_snr_bounds_db",
        "command_latency_or_jitter_bounds", "desired_to_measured_achieved_error_or_lag",
    } or calibration.get("status") != "NOT_PERFORMED_INTERFACE_ONLY" or any(
        value != "NOT_DEFINED_UE_N1" for key, value in calibration.items() if key != "status"
    ):
        raise InterfaceV2Error("numeric calibration/bounds were introduced")

    revision = _mapping(config["oai_revision"], "oai_revision")
    _exact(revision, {
        "git_head": "7473cdb52e1cf3c40e1e1f189f03b2785bf15610",
        "branch": "scenesense-nrue-grant-trace", "dirty_tree_at_freeze": True,
        "revision_authority": (
            "EXACT_FILE_AND_RUNTIME_ARTIFACT_HASHES_COMMIT_ALONE_IS_INSUFFICIENT"
        ),
    }, "OAI revision disclosure")
    contract = _mapping(config["contract"], "contract")
    if (
        set(contract) != {"path", "sha256"}
        or contract.get("path") != "rl_agent/UE_N1_OAI_UL_ACTUATOR_INTERFACE_CONTRACT_V2.md"
        or not SHA256_RE.fullmatch(str(contract.get("sha256", "")))
    ):
        raise InterfaceV2Error("v2 contract seal declaration differs")

    if "channel_command_db" in json.dumps(config, sort_keys=True):
        raise InterfaceV2Error("superseded channel_command_db field leaked into v2")
    if "target_snr_db" in json.dumps(config, sort_keys=True):
        raise InterfaceV2Error("generic target_snr_db field leaked into v2")
    sources = _mapping(config["sources"], "sources")
    if set(sources) != V2_SOURCE_KEYS:
        raise InterfaceV2Error("source seal set differs")
    for label, value in sources.items():
        spec = _mapping(value, f"sources.{label}")
        if set(spec) != {"path", "sha256"} or not SHA256_RE.fullmatch(str(spec.get("sha256", ""))):
            raise InterfaceV2Error(f"invalid source seal: {label}")
    runtime = _mapping(config["runtime_artifacts"], "runtime_artifacts")
    if (
        set(runtime) != {"authority", "execution_claimed", "files"}
        or runtime.get("authority") != "CURRENT_ARTIFACT_SEALS_ONLY_REVERIFY_AT_UE_N2_PREFLIGHT"
        or runtime.get("execution_claimed") is not False
        or set(_mapping(runtime.get("files"), "runtime files")) != set(RUNTIME_PATHS)
    ):
        raise InterfaceV2Error("runtime artifact authority/set differs")
    for label, expected_path in RUNTIME_PATHS.items():
        spec = _mapping(runtime["files"][label], f"runtime.{label}")
        if (
            set(spec) != {"path", "sha256"}
            or spec.get("path") != expected_path
            or not SHA256_RE.fullmatch(str(spec.get("sha256", "")))
        ):
            raise InterfaceV2Error(f"runtime artifact declaration differs: {label}")
    mechanism = _mapping(config["mechanism_evidence"], "mechanism_evidence")
    if (
        set(mechanism) != {
            "claim", "overall_experiment_status", "single_ue_calibration_claimed",
            "cadence_claimed", "numeric_bound_claimed", "files",
        }
        or mechanism.get("claim")
        != "TWO_UE_CHANNELMOD_READ_MODIFY_READ_MECHANISM_ONLY"
        or mechanism.get("overall_experiment_status") != "FAILED_HOLD"
        or mechanism.get("single_ue_calibration_claimed") is not False
        or mechanism.get("cadence_claimed") is not False
        or mechanism.get("numeric_bound_claimed") is not False
        or set(_mapping(mechanism.get("files"), "mechanism files"))
        != {"runtime_switch", "failed_terminal", "results_summary"}
    ):
        raise InterfaceV2Error("mechanism evidence boundary differs")
    output = _mapping(config["output"], "output")
    _exact(output, {
        "root": "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v2",
        "resolved_config_json": "resolved_config.json", "report_md": "REPORT.md",
        "manifest_json": "manifest.json",
        "terminal_json": "UE_N1_INTERFACE_V2_FROZEN.json",
        "terminal_status": STATUS, "next_checklist_item": NEXT_ITEM,
    }, "output")
    return config


def _verify_superseded_v1(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = _mapping(config["supersedes"], "supersedes")
    bundle = _repo_path(str(spec["bundle_dir"]))
    entries = (
        ("manifest_json", "manifest_sha256", "v1 manifest"),
        ("terminal_json", "terminal_sha256", "v1 terminal"),
        ("resolved_config_json", "resolved_config_sha256", "v1 resolved config"),
        ("report_md", "report_sha256", "v1 report"),
    )
    expected_names = {str(spec[path_key]) for path_key, _, _ in entries}
    if not bundle.is_dir() or {path.name for path in bundle.iterdir()} != expected_names:
        raise InterfaceV2Error("superseded v1 bundle entry set differs")
    paths = {
        label: _pinned(bundle / str(spec[path_key]), str(spec[hash_key]), label)
        for path_key, hash_key, label in entries
    }
    manifest = json.loads(paths["v1 manifest"].read_text(encoding="utf-8"))
    terminal = json.loads(paths["v1 terminal"].read_text(encoding="utf-8"))
    if (
        manifest.get("interface_id") != "ue_n1_oai_ul_actuator_interface_v1"
        or manifest.get("status") != "FROZEN_INTERFACE_ONLY"
        or terminal.get("interface_id") != "ue_n1_oai_ul_actuator_interface_v1"
        or terminal.get("status") != "FROZEN_INTERFACE_ONLY"
        or terminal.get("manifest_sha256") != spec["manifest_sha256"]
    ):
        raise InterfaceV2Error("superseded v1 semantic seal failed")
    return [_record(path, "superseded_pre_final", label) for label, path in paths.items()]


def _verify_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    core_config = dict(config)
    core_config["sources"] = {
        key: value for key, value in config["sources"].items() if key in CORE_SOURCE_KEYS
    }
    try:
        core_records = v1._verify_sources(core_config)
    except v1.InterfaceFreezeError as exc:
        raise InterfaceV2Error(str(exc)) from exc
    records = [dict(record) for record in core_records]
    helper = _mapping(config["sources"]["v1_offline_helper_dependency"], "v1 helper")
    path = _pinned(_repo_path(str(helper["path"])), str(helper["sha256"]), "v1 offline helper")
    records.append(_record(path, "source", "v1_offline_helper_dependency"))
    return records


def _verify_runtime_artifacts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for label, value in config["runtime_artifacts"]["files"].items():
        spec = _mapping(value, label)
        path = _pinned(_repo_path(str(spec["path"])), str(spec["sha256"]), label)
        records.append(_record(path, "runtime_artifact", label))
    return records


def _verify_inputs(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    contract = _mapping(config["contract"], "contract")
    contract_path = _pinned(
        _repo_path(str(contract["path"])), str(contract["sha256"]), "v2 contract"
    )
    records = [
        _record(config_path, "config", "UE-N1 v2 canonical config"),
        _record(Path(__file__), "assembler", "UE-N1 v2 assembler/validator"),
        _record(contract_path, "contract", "UE-N1 v2 interface contract"),
    ]
    records.extend(_verify_superseded_v1(config))
    try:
        records.extend(v1._verify_predecessor(config))
    except v1.InterfaceFreezeError as exc:
        raise InterfaceV2Error(str(exc)) from exc
    records.extend(_verify_sources(config))
    records.extend(_verify_runtime_artifacts(config))
    try:
        records.extend(v1._verify_mechanism_evidence(config))
    except v1.InterfaceFreezeError as exc:
        raise InterfaceV2Error(str(exc)) from exc
    return records


def _report(_: Mapping[str, Any]) -> str:
    return """# UE-N1 OAI uplink actuator interface v2

**Status:** `FROZEN_INTERFACE_ONLY` — final v2 authority; next UE-N2

This create-only v2 bundle supersedes immutable v1 as pre-final observation-
audit evidence. It freezes the single-UE gNB-side RFsim uplink actuator:
`rfsimu_channel_ue0`, dynamically resolved per control session, with OAI
parameter `noise_power_dB`, canonical experiment field
`commanded_noise_power_db`, `ploss=0`, global noise unset, and clean `-50`
attachment/restoration.

`desired_achieved_pusch_snr_db` and `commanded_noise_power_db` are experiment
control/evaluation only and are excluded from policy state. gNB radio and
scheduler values remain post-action collector evidence until a measured UE-
visible feedback path supplies a policy availability timestamp no later than
the decision cutoff. Collector ingest time is not policy availability.

UE-N2 must use persistent Telnet and a monotonic 100-ms/no-catch-up schedule.
Send and response receipt bracket handler completion; the response ACK is not
an application timestamp. First-effect lag is estimated from the measured step
response. Direct UL BLER remains `UNAVAILABLE_UNRESOLVED` until genuine UL
CRC/HARQ evidence is bound.

The raw-event envelope preserves RAN epoch/session, source index and full
timestamp, unwrapped slot, collector ingest times, raw hash, and explicit
missing reason. Missing PUSCH is unresolved and is not called DTX without DTX
evidence.

Current `nr-softmodem`, `nr-uesoftmodem`, `libtelnetsrv.so`, and
`librfsimulator.so` hashes are sealed and require UE-N2 preflight recheck. No
numeric calibration/bounds, runtime edit, OAI/CARLA run, or socket execution
occurred.
"""


def _terminal(created_at: str) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "interface_id": INTERFACE_ID,
        "status": STATUS,
        "created_at": created_at,
        "claim_scope": "FINAL_V2_INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION",
        "supersedes_interface_id": "ue_n1_oai_ul_actuator_interface_v1",
        "supersedes_authority_status": "SUPERSEDED_PRE_FINAL_OBSERVATION_AUDIT",
        "predecessor": "UE-A4",
        "next_checklist_item": NEXT_ITEM,
        "canonical_command_field": "commanded_noise_power_db",
        "oai_mutable_parameter": "noise_power_dB",
        "policy_observation_binding": "UNBOUND_UNTIL_MEASURED_UE_VISIBLE_FEEDBACK_PATH",
        "direct_ul_bler_status": "UNAVAILABLE_UNRESOLVED",
        "numeric_calibration_status": "NOT_PERFORMED",
        "numeric_bounds_status": "NOT_DEFINED",
        "runtime_executed": False,
        "socket_executed": False,
        "oai_run": False,
        "carla_run": False,
    }


def _manifest_interface() -> dict[str, Any]:
    return {
        "ue_count": 1,
        "direction": "UPLINK_ONLY",
        "channel_model_name": "rfsimu_channel_ue0",
        "model_index_resolution": "DYNAMIC_EXACT_NAME_PER_CONTROL_SESSION",
        "oai_mutable_parameter": "noise_power_dB",
        "canonical_command_field": "commanded_noise_power_db",
        "fixed_path_loss_db": 0,
        "global_noise_requirement": "UNSET",
        "clean_attach_and_restore_commanded_noise_power_db": "-50",
        "persistent_telnet_required_next": True,
        "schedule_period_ms": 100,
        "catch_up_policy": "NEVER_BURST_OBSOLETE_COMMANDS",
        "response_ack_semantics": "HANDLER_COMPLETION_UPPER_BOUND_NOT_APPLICATION_TIMESTAMP",
    }


def _manifest_causality() -> dict[str, Any]:
    return {
        "control_evaluation_only": [
            "desired_achieved_pusch_snr_db", "commanded_noise_power_db"
        ],
        "control_fields_in_policy_state": False,
        "collector_ingest_is_policy_availability": False,
        "post_action_collector_evidence_is_policy_observation": False,
        "future_policy_admission": (
            "policy_observation_available_monotonic_ns <= decision_cutoff_monotonic_ns "
            "AND observation.ran_epoch_id == decision.ran_epoch_id AND "
            "observation.control_session_id == decision.control_session_id"
        ),
        "missing_pusch": "MISSING_UNRESOLVED_NOT_DTX_WITHOUT_DTX_EVIDENCE",
    }


GATES = {
    "v1_supersession": "PASS_PINNED_IMMUTABLE_PRE_FINAL",
    "a4_predecessor": "PASS_PINNED_72_UNFILTERED_TECHNICAL_ACTIONS",
    "single_ue_ul_scope": "PASS",
    "dynamic_model_index": "PASS_DECLARED",
    "awgn_noise_power_db": "PASS_DECLARED",
    "canonical_commanded_noise_power_db": "PASS_EXCLUDED_FROM_POLICY_STATE",
    "clean_attach_then_runtime": "PASS_DECLARED",
    "persistent_telnet": "REQUIRED_UE_N2_NOT_EXECUTED",
    "monotonic_100ms_no_catch_up": "PASS_DECLARED",
    "ack_not_apply_timestamp": "PASS_DECLARED",
    "first_effect_lag": "DEFINED_FOR_UE_N2_MEASUREMENT_NOT_EXECUTED",
    "raw_event_envelope": "PASS_DECLARED",
    "collector_vs_policy_availability": "PASS_DISTINCT_POLICY_PATH_UNBOUND",
    "direct_ul_bler": "UNAVAILABLE_UNRESOLVED_NOT_ZERO",
    "runtime_artifact_seals_including_rfsimulator": "PASS_CURRENT_REVERIFY_UE_N2",
    "two_ue_mechanism_evidence": "PASS_MECHANISM_ONLY_OVERALL_RUN_FAILED_HOLD",
    "numeric_calibration": "NOT_AUTHORIZED_NOT_PERFORMED",
    "numeric_bounds": "NOT_AUTHORIZED_NOT_DEFINED",
    "runtime_or_socket": "NOT_AUTHORIZED_NOT_EXECUTED",
}
DEFERRED = [
    "PERSISTENT_TELNET_IMPLEMENTATION_AND_100MS_REPLAY",
    "COMMAND_ACK_BRACKET_AND_FIRST_EFFECT_LAG_MEASUREMENT",
    "DESIRED_ACHIEVED_PUSCH_SNR_TO_COMMANDED_NOISE_CALIBRATION",
    "ATTACH_SAFE_ACHIEVED_SNR_BOUNDS",
    "GENUINE_UL_CRC_HARQ_OUTCOME_BINDING",
    "MEASURED_UE_VISIBLE_POLICY_FEEDBACK_AVAILABILITY",
]


def assemble(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    now: str | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    created_at = validate_created_at(now if now is not None else utc_now())
    target = _repo_path(str(config["output"]["root"])) if output_dir is None else Path(
        output_dir
    ).expanduser().resolve()
    if target.exists():
        raise InterfaceV2Error(f"refusing to overwrite create-only UE-N1 v2 bundle: {target}")
    inputs = _verify_inputs(config, config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        resolved_name = config["output"]["resolved_config_json"]
        report_name = config["output"]["report_md"]
        manifest_name = config["output"]["manifest_json"]
        terminal_name = config["output"]["terminal_json"]
        _write_json(temp / resolved_name, config)
        (temp / report_name).write_text(_report(config), encoding="utf-8")
        outputs = [
            {
                "path": name,
                "sha256": sha256_file(temp / name),
                "bytes": (temp / name).stat().st_size,
            }
            for name in (resolved_name, report_name)
        ]
        terminal_payload = _terminal(created_at)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "interface_id": INTERFACE_ID,
            "status": STATUS,
            "created_at": created_at,
            "claim_scope": "FINAL_V2_INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION",
            "authority": config["authority"],
            "supersession": {
                "supersedes_interface_id": "ue_n1_oai_ul_actuator_interface_v1",
                "supersedes_manifest_sha256": config["supersedes"]["manifest_sha256"],
                "v1_authority_status": "SUPERSEDED_PRE_FINAL_OBSERVATION_AUDIT",
                "v1_bytes_mutated": False,
            },
            "interface": _manifest_interface(),
            "causality": _manifest_causality(),
            "gates": GATES,
            "deferred": DEFERRED,
            "inputs": inputs,
            "outputs": outputs,
            "terminal_decision_path": terminal_name,
            "terminal_decision_payload": terminal_payload,
            "terminal_decision_payload_sha256": hashlib.sha256(
                canonical_json_bytes(terminal_payload)
            ).hexdigest(),
        }
        _write_json(temp / manifest_name, manifest)
        _write_json(
            temp / terminal_name,
            {**terminal_payload, "manifest_sha256": sha256_file(temp / manifest_name)},
        )
        validate_bundle(temp, config_path=config_path)
        _rename_noreplace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def validate_bundle(output_dir: Path, *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    manifest_path = output_dir / config["output"]["manifest_json"]
    terminal_path = output_dir / config["output"]["terminal_json"]
    if not manifest_path.is_file() or not terminal_path.is_file():
        raise InterfaceV2Error("v2 manifest or terminal is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "schema", "interface_id", "status", "created_at", "claim_scope",
        "authority", "supersession", "interface", "causality", "gates",
        "deferred", "inputs", "outputs", "terminal_decision_path",
        "terminal_decision_payload", "terminal_decision_payload_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise InterfaceV2Error("v2 manifest key set differs")
    created_at = validate_created_at(manifest.get("created_at"))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("interface_id") != INTERFACE_ID
        or manifest.get("status") != STATUS
        or manifest.get("claim_scope") != "FINAL_V2_INTERFACE_ONLY_NO_RUNTIME_OR_CALIBRATION"
        or manifest.get("authority") != EXPECTED_AUTHORITY
        or manifest.get("interface") != _manifest_interface()
        or manifest.get("causality") != _manifest_causality()
        or manifest.get("gates") != GATES
        or manifest.get("deferred") != DEFERRED
    ):
        raise InterfaceV2Error("v2 manifest authority/interface/gates differ")
    expected_supersession = {
        "supersedes_interface_id": "ue_n1_oai_ul_actuator_interface_v1",
        "supersedes_manifest_sha256": config["supersedes"]["manifest_sha256"],
        "v1_authority_status": "SUPERSEDED_PRE_FINAL_OBSERVATION_AUDIT",
        "v1_bytes_mutated": False,
    }
    if manifest.get("supersession") != expected_supersession:
        raise InterfaceV2Error("v2 supersession snapshot differs")

    resolved_name = config["output"]["resolved_config_json"]
    report_name = config["output"]["report_md"]
    output_records = manifest.get("outputs")
    if (
        not isinstance(output_records, list)
        or len(output_records) != 2
        or {row.get("path") for row in output_records if isinstance(row, Mapping)}
        != {resolved_name, report_name}
    ):
        raise InterfaceV2Error("v2 output seal set differs")
    for value in output_records:
        record = _mapping(value, "output seal")
        if set(record) != {"path", "sha256", "bytes"}:
            raise InterfaceV2Error("v2 output seal fields differ")
        relative = Path(str(record["path"]))
        path = (output_dir / relative).resolve()
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or path.parent != output_dir
            or not path.is_file()
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise InterfaceV2Error(f"v2 output seal mismatch: {relative}")
    resolved = json.loads((output_dir / resolved_name).read_text(encoding="utf-8"))
    if resolved != config:
        raise InterfaceV2Error("v2 resolved config differs")
    if (output_dir / report_name).read_text(encoding="utf-8") != _report(config):
        raise InterfaceV2Error("v2 deterministic report differs")
    if manifest.get("inputs") != _verify_inputs(config, config_path):
        raise InterfaceV2Error("v2 reconstructed input seals differ")

    payload = dict(terminal)
    terminal_manifest_sha = payload.pop("manifest_sha256", None)
    validate_created_at(payload.get("created_at"))
    if payload != _terminal(created_at):
        raise InterfaceV2Error("v2 terminal payload differs")
    if terminal_manifest_sha != sha256_file(manifest_path):
        raise InterfaceV2Error("v2 terminal does not seal manifest")
    if (
        manifest.get("terminal_decision_path") != config["output"]["terminal_json"]
        or manifest.get("terminal_decision_payload") != payload
        or manifest.get("terminal_decision_payload_sha256")
        != hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    ):
        raise InterfaceV2Error("v2 terminal payload seal differs")
    expected_files = {
        config["output"]["manifest_json"], config["output"]["terminal_json"],
        resolved_name, report_name,
    }
    if {path.name for path in output_dir.iterdir()} != expected_files:
        raise InterfaceV2Error("v2 bundle contains unexpected or missing entries")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            manifest = validate_bundle(args.validate, config_path=args.config)
            print(f"Validated UE-N1 v2 bundle: {Path(args.validate).resolve()}")
            print(f"Status: {manifest['status']} (next {NEXT_ITEM})")
            return 0
        output = assemble(args.config, args.output_dir)
        print(f"UE-N1 v2 interface bundle: {output}")
        print(f"Status: {STATUS} (next {NEXT_ITEM})")
        return 0
    except (InterfaceV2Error, OSError, ValueError) as exc:
        print(f"UE-N1 v2 freeze failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

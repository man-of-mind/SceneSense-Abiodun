#!/usr/bin/env python3
"""Prepare or execute bounded UE-N3B cold-attach confirmation.

PREPARE_ONLY is the default and remains usable while the create-only N3A
adjudication seal is pending.  EXECUTE_LIVE fails closed until that predecessor
and explicit live authority are frozen.  Each live repetition starts a fresh
single-UE RAN from run-local configs: the two DL RFsim channels remain at -50 dB
and only the gNB-owned UL channel starts at -2.5 dB.  No candidate Telnet
modify is permitted.  After the exact 600-frame/60-second service window, one
Telnet modify restores the UL channel to -50 dB and a bounded recovery is
checked before teardown.

This stage records cold-attach evidence for review.  It cannot promote a target
mapping, connectivity bound, usable-service bound, or operational SNR bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3b_oai_ul_cold_attach_confirmation_v1.json"
PREPARE_ONLY = "PREPARE_ONLY"
EXECUTE_LIVE = "EXECUTE_LIVE"
SCHEMA = "scenesense.ue_n3b_oai_ul_cold_attach_confirmation_config.v1"
PLAN_BLOCKED = "UE_N3B_COLD_ATTACH_PLAN_FROZEN_PREREQUISITES_PENDING"
PLAN_READY = "UE_N3B_COLD_ATTACH_PLAN_FROZEN_READY_FOR_EXPLICIT_EXECUTE_LIVE"
REP_PASSED = "UE_N3B_COLD_ATTACH_SERVICE_REPETITION_PASSED"
REP_ATTACH_FAILED = "UE_N3B_COLD_ATTACH_REPETITION_VALID_ATTACH_FAILURE"
REP_SERVICE_FAILED = "UE_N3B_COLD_ATTACH_REPETITION_VALID_SERVICE_FAILURE"
REP_ACHIEVED_SNR_MISMATCH = "UE_N3B_COLD_ATTACH_REPETITION_VALID_ACHIEVED_SNR_MISMATCH"
REP_UNCONFIRMED = "UE_N3B_COLD_ATTACH_REPETITION_EVIDENCE_UNCONFIRMED"
CAMPAIGN_PASSED = "UE_N3B_COLD_ATTACH_CONFIRMATION_3_OF_3_PASSED_REVIEW_REQUIRED"
CAMPAIGN_NOT_3_OF_3 = "UE_N3B_COLD_ATTACH_CONFIRMATION_NOT_3_OF_3_REVIEW_REQUIRED"
RESTORE_FAILED = "UE_N3B_FAILED_RESTORE"
PENDING = "PENDING_CREATE_ONLY_N3A_ADJUDICATION"
LIVE_AUTHORITY_BASIS = "USER_REQUEST_2026-08-21_EXECUTE_N3B_AFTER_SEALED_N3A_ADJUDICATION"

EXPECTED_N3A_LIVE = {
    "directory": (
        "rl_agent/experiments/ue_n3a_oai_ul_sustain_replication_v1/"
        "20260821_live_02"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": "62639405273bd77aba5ff345bba2e2f99d2f15dfe962aa54083d5142c1b1b6ce",
    "terminal": "UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED.json",
    "terminal_sha256": "b05f8537cf671984545d426d58b2efea923dc348b052e20f0eddaaa18625b798",
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": "44343c9212f7c005887b252f27027ca044080c19033b4d2e33822a09acc8c4ed",
    "required_status": "UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED",
    "source_config_sha256": "f1afb806478d0fc17d381b2dd2f119173d4745696abff90be945bce1bdd71358",
    "source_runner_sha256": "b72f98d1e492840fde074772e9e90a1831c4b0f8d777795b29b27a9e1a482487",
}


class ColdAttachFailure(calibration.CalibrationFailure):
    """Fail-closed authority, evidence, infrastructure, or lifecycle failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ColdAttachFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ColdAttachFailure(f"path escapes repository root: {relative}") from exc
    return path


def pending_adjudication(block: Mapping[str, Any]) -> bool:
    required = (
        "directory", "manifest_sha256", "terminal_sha256",
        "resolved_config_sha256", "source_config_sha256", "source_runner_sha256",
    )
    return any(str(block.get(key, "")) == PENDING for key in required)


def _runtime_seal(config: Mapping[str, Any], relative: str) -> str:
    matches = [
        str(row.get("sha256", "")) for row in config["runtime_seals"]
        if row.get("path") == relative
    ]
    require(len(matches) == 1, f"runtime seal is not unique: {relative}")
    return matches[0]


def validate_config(
    config: Mapping[str, Any], *, verify_hashes: bool = True,
    require_live: bool = False,
) -> None:
    require(config.get("schema") == SCHEMA, "unexpected N3B config schema")
    require(
        config.get("claim_boundary")
        == "COLD_ATTACH_CONFIRMATION_EVIDENCE_ONLY_REVIEW_REQUIRED_NO_BOUND_PROMOTION",
        "N3B claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_plan_authorized") is True,
            "offline plan authority is absent")
    require(
        authority.get("live_oai_run_authorized")
        == authority.get("live_socket_execution_authorized"),
        "live OAI and socket authority must change together",
    )
    live_enabled = authority.get("live_oai_run_authorized") is True
    expected_basis = LIVE_AUTHORITY_BASIS if live_enabled else "NOT_AUTHORIZED_PREPARE_ONLY"
    require(authority.get("live_authority_basis") == expected_basis,
            "live authority basis drift")
    for key in (
        "carla_run_authorized", "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized", "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized", "operational_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    if require_live:
        require(live_enabled, "EXECUTE_LIVE authority is absent")

    predecessors = config["predecessors"]
    require(predecessors.get("n3a_live_evidence") == EXPECTED_N3A_LIVE,
            "sealed N3A live evidence drift")
    adjudication = predecessors["n3a_adjudication"]
    if pending_adjudication(adjudication):
        for key in (
            "directory", "manifest_sha256", "terminal_sha256",
            "resolved_config_sha256", "source_config_sha256", "source_runner_sha256",
        ):
            require(str(adjudication.get(key, "")) == PENDING,
                    "adjudication predecessor is only partially pinned")
        require(not require_live, "N3A adjudication predecessor is still pending")
    else:
        for key in (
            "manifest_sha256", "terminal_sha256", "resolved_config_sha256",
            "source_config_sha256", "source_runner_sha256",
        ):
            require(re.fullmatch(r"[0-9a-f]{64}", str(adjudication.get(key, ""))) is not None,
                    f"adjudication {key} is malformed")
        require(
            adjudication.get("required_status")
            == "UE_N3_BOUND_BRACKETED_REVIEW_REQUIRED",
            "adjudication status contract drift",
        )
        require(
            adjudication.get("n3b_eligibility_status")
            == "UE_N3A_USABLE_SERVICE_BRACKET_ACCEPTED_FOR_N3B",
            "adjudication eligibility contract drift",
        )
        require(math.isclose(float(adjudication.get("n3b_selected_command_db", math.nan)), -2.5),
                "adjudication did not select command -2.5")
    require(
        predecessors.get("ue_n1_bundle")
        == "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v2",
        "UE-N1 bundle drift",
    )

    campaign = config["campaign"]
    require(int(campaign["repetitions"]) == 3, "N3B requires exactly three repetitions")
    require(campaign["one_fresh_ran_per_repetition"] is True,
            "every repetition must use a fresh RAN")
    require(campaign["run_local_configs_only"] is True,
            "N3B must use run-local configs")
    require(int(campaign["candidate_application_count"]) == 0,
            "candidate Telnet application is forbidden")
    require([float(value) for value in campaign["commanded_noise_power_db"]] == [-2.5],
            "N3B candidate command drift")
    require(campaign["continue_after_valid_attach_or_service_failure"] is True,
            "valid candidate failures must be retained")
    require(campaign["stop_on_invalid_or_unclean_evidence"] is True,
            "invalid evidence must stop the campaign")

    channel = config["startup_channel"]
    require(channel == {
        "rfsimu_channel_enB0": -50.0,
        "rfsimu_channel_enB1": -50.0,
        "rfsimu_channel_ue0": -2.5,
    }, "startup channel values drift")
    rung, traffic = config["rung"], config["traffic"]
    require(math.isclose(float(rung["candidate_lead_s"]), 5.0),
            "candidate RNTI lead must remain 5 seconds")
    require(math.isclose(float(rung["clean_lead_s"]), 5.0),
            "inherited lead alias must match the 5-second candidate lead")
    require(math.isclose(float(rung["measured_service_s"]), 60.0),
            "service window must remain exactly 60 seconds")
    require(math.isclose(float(rung["clean_recovery_s"]), 5.0),
            "clean recovery must remain 5 seconds")
    duration = sum(float(rung[key]) for key in (
        "candidate_lead_s", "measured_service_s", "clean_recovery_s"
    ))
    require(math.isclose(duration, 70.0)
            and math.isclose(float(rung["service_duration_s"]), 70.0),
            "bounded traffic duration must remain 70 seconds")
    require(math.isclose(float(traffic["fps"]), 10.0), "probe must use 10 Hz")
    require(int(rung["sender_frames"]) == 700, "probe must schedule exactly 700 frames")
    require(int(rung["expected_service_frames"]) == 600,
            "authoritative service window must contain 600 frames")
    require(float(rung["receiver_capture_duration_s"]) > 70.0,
            "receiver requires a bounded completion margin")
    require(math.isclose(float(config["radio"]["attach_timeout_s"]), 180.0),
            "cold attach/PDU/ext-DN gate must remain 180 seconds")
    require(int(traffic["frame_bytes"]) == 12_500
            and int(traffic["chunk_bytes"]) == 12_500,
            "matched 1 Mbps probe shape drift")

    gates = config["transport_gates"]
    require(math.isclose(float(gates["primary_complete_frame_ratio"]), 0.99),
            "service delivery gate must remain 99 percent")
    require(int(gates["maximum_interarrival_gaps_gte_1s"]) == 0,
            "service window permits no one-second outage")
    require(gates["expected_source_ip"] == "192.168.70.134",
            "expected UPF-SNAT source drift")
    require(gates["required_stop_reason"] == "DURATION_COMPLETE",
            "receiver completion contract drift")
    require(config["preflight"]["fail_if_carla_active"] is True,
            "CARLA fail-closed gate disabled")
    require(math.isclose(float(config["analysis"]["expected_achieved_pusch_snr_db"]), 6.0)
            and math.isclose(float(config["analysis"]["achieved_snr_tolerance_db"]), 0.5),
            "N3B achieved-SNR confirmation gate drift")

    required_paths = {
        "rl_agent/ue_n3b_oai_ul_cold_attach_confirmation_v1.py",
        "rl_agent/ue_n3_oai_ul_command_calibration_v1.py",
        "rl_agent/ue_n2_oai_ul_calibration_smoke.py",
        "rl_agent/ue_n3_structured_udp_receiver.py",
        "oai_layer_latency/carla_shaped_udp_burst_sender.py",
        "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf",
        "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf",
        "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/channelmod_rfsimu.conf",
    }
    sealed = [str(row.get("path", "")) for row in config["runtime_seals"]]
    require(len(sealed) == len(set(sealed)), "runtime seal paths repeat")
    require(required_paths.issubset(sealed), "required N3B runtime seals are absent")
    if verify_hashes:
        for seal in config["runtime_seals"]:
            path = resolve_repo_path(str(seal["path"]))
            require(path.is_file(), f"sealed runtime file missing: {path}")
            require(n2.sha256(path) == seal["sha256"],
                    f"runtime seal drift: {seal['path']}")


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    ready = (
        not pending_adjudication(config["predecessors"]["n3a_adjudication"])
        and config["authority"]["live_oai_run_authorized"] is True
        and config["authority"]["live_socket_execution_authorized"] is True
    )
    return [
        {
            "sequence_index": index - 1,
            "repetition_index": index,
            "commanded_noise_power_db": -2.5,
            "fresh_ran_epoch_required": True,
            "candidate_baked_before_ue_launch": True,
            "candidate_application_count": 0,
            "attach_pdu_ext_dn_timeout_s": 180.0,
            "measured_service_s": 60.0,
            "expected_service_frames": 600,
            "restore_commanded_noise_power_db": -50.0,
            "status": (
                "READY_FOR_EXPLICIT_EXECUTE_LIVE"
                if ready else "BLOCKED_PENDING_PREREQUISITES"
            ),
        }
        for index in range(1, int(config["campaign"]["repetitions"]) + 1)
    ]


def _verify_manifest_inventory(directory: Path, manifest: Mapping[str, Any]) -> set[str]:
    rows = list(manifest.get("outputs", []))
    require(rows, f"predecessor manifest has no outputs: {directory}")
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in seen,
                f"blank or duplicate predecessor output: {relative!r}")
        seen.add(relative)
        artifact = (directory / relative).resolve()
        try:
            artifact.relative_to(directory.resolve())
        except ValueError as exc:
            raise ColdAttachFailure(f"predecessor output escapes directory: {relative}") from exc
        require(artifact.is_file(), f"predecessor output missing: {relative}")
        require(artifact.stat().st_size == int(row.get("bytes", -1)),
                f"predecessor output size drift: {relative}")
        require(n2.sha256(artifact) == row.get("sha256"),
                f"predecessor output hash drift: {relative}")
    return seen


def verify_predecessor(
    config: Mapping[str, Any], name: str, *, proof_path: Path | None = None,
) -> dict[str, Any]:
    expected = config["predecessors"][name]
    require(not (name == "n3a_adjudication" and pending_adjudication(expected)),
            "N3A adjudication predecessor is pending")
    directory = resolve_repo_path(str(expected["directory"]))
    paths = {
        "manifest": directory / str(expected["manifest"]),
        "terminal": directory / str(expected["terminal"]),
        "resolved_config": directory / str(expected["resolved_config"]),
    }
    for key, path in paths.items():
        require(path.is_file(), f"{name} predecessor {key} missing: {path}")
        require(n2.sha256(path) == expected[f"{key}_sha256"],
                f"{name} predecessor {key} seal drift")
    manifest, terminal = load_json(paths["manifest"]), load_json(paths["terminal"])
    require(manifest.get("status") == expected["required_status"]
            and terminal.get("status") == expected["required_status"],
            f"{name} predecessor status mismatch")
    require(terminal.get("manifest_sha256") == expected["manifest_sha256"],
            f"{name} terminal does not bind its manifest")
    seen = _verify_manifest_inventory(directory, manifest)
    require(str(expected["resolved_config"]) in seen,
            f"{name} resolved config is absent from manifest")
    if name == "n3a_live_evidence":
        require(expected == EXPECTED_N3A_LIVE, "N3A evidence pin drift")
        require(manifest.get("config_sha256") == expected["source_config_sha256"]
                and manifest.get("runner_sha256") == expected["source_runner_sha256"],
                "N3A source identity mismatch")
        require(terminal.get("sustain_candidate_3_of_3_pass") is True,
                "N3A predecessor lacks 3/3 sustained candidate passes")
        require(terminal.get("cold_attach_bound_evaluated") is False,
                "N3A predecessor unexpectedly evaluated cold attach")
    else:
        require(manifest.get("config_sha256") == expected["source_config_sha256"]
                and manifest.get("runner_sha256") == expected["source_runner_sha256"],
                "adjudication source identity mismatch")
        require(terminal.get("n3b_eligibility_status")
                == "UE_N3A_USABLE_SERVICE_BRACKET_ACCEPTED_FOR_N3B",
                "adjudication terminal does not authorize N3B eligibility")
        require(math.isclose(float(terminal.get("n3b_selected_command_db", math.nan)), -2.5),
                "adjudication terminal did not select -2.5")
        for key in (
            "target_mapping_promoted", "numeric_bound_promoted",
            "connectivity_bound_promoted", "usable_service_bound_promoted",
            "operational_bound_promoted",
        ):
            require(terminal.get(key) is False,
                    f"adjudication unexpectedly promoted {key}")
        require(terminal.get("source_verification", {}).get("campaign_manifest_sha256")
                == EXPECTED_N3A_LIVE["manifest_sha256"],
                "adjudication is not bound to sealed N3A live evidence")
    proof = {
        "status": "VERIFIED_READ_ONLY_PREDECESSOR",
        "predecessor": name,
        "directory": str(directory),
        "manifest_sha256": n2.sha256(paths["manifest"]),
        "terminal_sha256": n2.sha256(paths["terminal"]),
        "resolved_config_sha256": n2.sha256(paths["resolved_config"]),
        "verified_output_count": len(seen),
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def rewrite_channel_noise(text: str, values: Mapping[str, float]) -> str:
    """Rewrite one noise value per named RFsim model, failing on ambiguity."""
    rewritten = text
    for name, value in values.items():
        pattern = re.compile(
            rf'(model_name\s*=\s*"{re.escape(name)}"(?:(?!model_name|\}}).)*?'
            rf'noise_power_dB\s*=\s*)[-+0-9.eE]+(\s*;)',
            re.DOTALL,
        )
        rewritten, count = pattern.subn(
            lambda match: f"{match.group(1)}{float(value):.1f}{match.group(2)}",
            rewritten,
        )
        require(count == 1, f"expected one channel noise field for {name}, found {count}")
    return rewritten


def configured_channel_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("rfsimu_channel_enB0", "rfsimu_channel_enB1", "rfsimu_channel_ue0"):
        match = re.search(
            rf'model_name\s*=\s*"{re.escape(name)}"(?:(?!model_name|\}}).)*?'
            rf'noise_power_dB\s*=\s*([-+0-9.eE]+)\s*;',
            text,
            re.DOTALL,
        )
        require(match is not None, f"missing configured channel model: {name}")
        values[name] = float(match.group(1))
    return values


def validate_runtime_channel_state(
    payload: str, expected: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Validate an unambiguous three-model Telnet runtime snapshot."""
    try:
        models = n2.parse_channel_models(payload)
    except n2.SmokeFailure as exc:
        raise ColdAttachFailure(f"ambiguous runtime channel state: {exc}") from exc
    require(set(models) == set(expected),
            f"runtime channel model set mismatch: {sorted(models)}")
    for name, target in expected.items():
        row = models[name]
        require(row.get("model_type") == "AWGN",
                f"runtime model type mismatch for {name}: {row}")
        require(math.isclose(float(row.get("path_loss_db", math.nan)), 0.0),
                f"runtime path loss mismatch for {name}: {row}")
        require(math.isclose(float(row.get("noise_power_db", math.nan)), float(target)),
                f"runtime noise mismatch for {name}: {row}")
    require(models["rfsimu_channel_ue0"].get("owner") == "rfsimulator",
            "runtime UL model owner mismatch")
    return models


def classify_repetition(summary: Mapping[str, Any]) -> dict[str, Any]:
    attach = dict(summary.get("attach_gate") or {})
    transport = dict(summary.get("transport") or {})
    tail = dict(summary.get("service_tail") or {})
    service = dict(summary.get("service_window") or {})
    recovery = dict(summary.get("clean_recovery") or {})
    base_valid = (
        summary.get("candidate_baked_config_verified") is True
        and summary.get("startup_channel_runtime_verified") is True
        and int(summary.get("candidate_application_count", -1)) == 0
        and int(summary.get("restore_application_count", -1)) == 1
        and summary.get("clean_restore_verified") is True
        and summary.get("source_oai_configs_unchanged") is True
        and summary.get("cleanup_clean") is True
    )
    attach_passed = attach.get("passed") is True
    attach_failed_cleanly = (
        attach.get("status") == "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED"
        and attach.get("ran_processes_alive_at_terminal") is True
        and attach.get("core_ready_at_terminal") is True
    )
    exact_window = (
        tail.get("status") == "TAIL_ACCEPTED"
        and service.get("full_nominal_window_observed") is True
        and service.get("exact_frozen_frame_set_pass") is True
        and int(service.get("required_expected_frames", -1)) == 600
        and int(service.get("expected_frames", -1)) == 600
        and service.get("integrity_gate") is True
        and transport.get("integrity_gate") is True
    )
    service_pass = (
        exact_window
        and service.get("primary_99_pass") is True
        and service.get("no_one_second_outage_pass") is True
        and recovery.get("passed") is True
    )
    achieved_p05 = tail.get("achieved_pusch_snr_db_p05")
    achieved_p50 = tail.get("achieved_pusch_snr_db_median")
    achieved_p95 = tail.get("achieved_pusch_snr_db_p95")
    achieved_snr_gate = (
        achieved_p50 is not None
        and math.isfinite(float(achieved_p50))
        and abs(float(achieved_p50) - 6.0) <= 0.5
    )
    recognized_loss = summary.get("hard_loss_reason") in {
        "CURRENT_RNTI_PUSCH_SILENCE", "RNTI_CHANGED", "UE_TUNNEL_IDENTITY_LOST",
    }
    corroborated_loss = (
        summary.get("hard_loss_reason") != "CURRENT_RNTI_PUSCH_SILENCE"
        or summary.get("receiver_service_outage_detected") is True
    )
    if base_valid and attach_failed_cleanly:
        outcome, valid, passed = "COLD_ATTACH_FAILED", True, False
    elif base_valid and attach_passed and exact_window:
        if service_pass and achieved_snr_gate:
            outcome, valid, passed = "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED", True, True
        elif service_pass:
            outcome, valid, passed = "ACHIEVED_SNR_OUTSIDE_FROZEN_CANDIDATE_BAND", True, False
        else:
            outcome, valid, passed = "SERVICE_GATE_FAILED", True, False
    elif (base_valid and attach_passed and transport.get("integrity_gate") is True
          and recognized_loss and corroborated_loss and recovery.get("passed") is True):
        outcome, valid, passed = "HARD_SERVICE_LOSS_AFTER_COLD_ATTACH", True, False
    else:
        outcome, valid, passed = "EVIDENCE_UNCONFIRMED", False, False
    return {
        "evidence_valid_for_aggregation": valid,
        "joint_candidate_confirmation_pass": passed,
        "classified_outcome": outcome,
        "cold_attach_gate_pass": attach_passed,
        "clean_attach_failure_evidence": attach_failed_cleanly,
        "exact_600_frame_service_evidence": exact_window,
        "authoritative_service_gate_pass": service_pass,
        "achieved_snr_gate_pass": achieved_snr_gate,
        "achieved_pusch_snr_db_p05": achieved_p05,
        "achieved_pusch_snr_db_p50": achieved_p50,
        "achieved_pusch_snr_db_p95": achieved_p95,
        "expected_achieved_pusch_snr_db": 6.0,
        "achieved_snr_tolerance_db": 0.5,
        "recognized_service_loss": recognized_loss and corroborated_loss,
        "candidate_application_count_zero": int(summary.get("candidate_application_count", -1)) == 0,
        "single_restore_application": int(summary.get("restore_application_count", -1)) == 1,
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(results) == 3, "N3B requires three completed repetition results")
    require(all(row.get("evidence_valid_for_aggregation") is True for row in results),
            "N3B contains invalid repetition evidence")
    attach_passes = sum(bool(row.get("cold_attach_gate_pass")) for row in results)
    service_passes = sum(bool(row.get("authoritative_service_gate_pass")) for row in results)
    snr_passes = sum(bool(row.get("achieved_snr_gate_pass")) for row in results)
    joint_passes = sum(bool(row.get("joint_candidate_confirmation_pass")) for row in results)
    return {
        "status": CAMPAIGN_PASSED if joint_passes == 3 else CAMPAIGN_NOT_3_OF_3,
        "cold_attach_passes": attach_passes,
        "cold_attach_3_of_3_pass": attach_passes == 3,
        "authoritative_service_gate_passes": service_passes,
        "authoritative_service_gate_3_of_3_pass": service_passes == 3,
        "achieved_snr_band_passes": snr_passes,
        "achieved_snr_band_3_of_3_pass": snr_passes == 3,
        "joint_candidate_confirmation_passes": joint_passes,
        "joint_candidate_confirmation_3_of_3_pass": joint_passes == 3,
        "valid_nonconfirming_outcomes_retained": 3 - joint_passes,
        "cold_attach_bound_evaluated": True,
        "operational_bound_promoted": False,
    }


class ColdAttachRepetitionRunner(calibration.RungRunner):
    """One fresh-RAN N3B cold-attach repetition."""

    def __init__(
        self, config_path: Path, output_dir: Path, *, repetition_index: int,
        n3a_proof: Mapping[str, Any], adjudication_proof: Mapping[str, Any],
    ) -> None:
        super().__init__(
            config_path, output_dir, rung_index=0, command_db=-2.5,
            clean_control_proof=None,
        )
        self.repetition_index = int(repetition_index)
        self.n3a_proof = dict(n3a_proof)
        self.adjudication_proof = dict(adjudication_proof)
        self.application_count = 0
        self.restore_application_count = 0
        self.candidate_baked_config_verified = False
        self.startup_channel_runtime_verified = False
        self.source_hashes_before: dict[str, str] = {}
        self.ue_launch_monotonic_ns: int | None = None
        self.attach_gate: dict[str, Any] | None = None

    def verify_dependencies(self) -> None:
        validate_config(self.config, verify_hashes=True, require_live=True)
        for name, frozen in (
            ("n3a_live_evidence", self.n3a_proof),
            ("n3a_adjudication", self.adjudication_proof),
        ):
            observed = verify_predecessor(
                self.config, name,
                proof_path=self.output_dir / f"{name}_predecessor.json",
            )
            require(observed["manifest_sha256"] == frozen.get("manifest_sha256")
                    and observed["terminal_sha256"] == frozen.get("terminal_sha256"),
                    f"{name} proof changed between repetitions")
        n2.atomic_json(self.output_dir / "runtime_seals.json", {
            "status": "MATCHED",
            "observed_at": n2.utc_now(),
            "files": [
                {"path": row["path"], "expected_sha256": row["sha256"],
                 "observed_sha256": n2.sha256(self.path(row["path"]))}
                for row in self.config["runtime_seals"]
            ],
        })

    def assert_carla_absent(self) -> None:
        require(self.config["preflight"]["fail_if_carla_active"] is True,
                "CARLA fail-closed gate disabled")
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,comm=,args="], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ColdAttachFailure(f"CARLA detector failed closed: {exc}") from exc
        require(result.returncode == 0, "CARLA process detector failed closed")
        markers = [
            str(value).lower()
            for value in self.config["preflight"]["carla_process_markers"]
        ]
        matches = [
            line.strip() for line in result.stdout.splitlines()
            if any(marker in line.lower() for marker in markers)
        ]
        busy = [
            int(port) for port in self.config["preflight"]["carla_ports"]
            if not self.strict_port_free(int(port), socket.SOCK_STREAM)
            or not self.strict_port_free(int(port), socket.SOCK_DGRAM)
        ]
        evidence = {
            "status": "PASSED" if not matches and not busy else "FAILED",
            "process_matches": matches, "busy_ports": busy,
            "detector_scope": "PROCESS_COMM_AND_ARGS_PLUS_TCP_AND_UDP_PORTS",
            "checked_at": n2.utc_now(),
        }
        n2.atomic_json(self.output_dir / "carla_absent_gate.json", evidence)
        require(not matches and not busy, f"CARLA_ACTIVE_FAIL_CLOSED: {evidence}")
        self.last_carla_check_monotonic_ns = time.monotonic_ns()

    def materialize_configs(self) -> tuple[Path, Path]:
        paths = self.config["paths"]
        conf_root = self.path(paths["oai_ran_conf"])
        sources = {
            "gnb_base": conf_root / paths["gnb_base_config"],
            "ue_base": conf_root / paths["ue_base_config"],
            "channel": conf_root / paths["channel_config"],
        }
        self.source_hashes_before = {key: n2.sha256(path) for key, path in sources.items()}
        gnb_base = sources["gnb_base"].read_text(encoding="utf-8")
        ue_base = sources["ue_base"].read_text(encoding="utf-8")
        channel = rewrite_channel_noise(
            sources["channel"].read_text(encoding="utf-8"),
            self.config["startup_channel"],
        )
        require(configured_channel_values(channel) == self.config["startup_channel"],
                "run-local startup channel verification failed")
        require("noise_power_dBFS" not in channel,
                "global noise_power_dBFS must remain unset")
        expected_imsi = str(self.config["radio"]["expected_imsi"])
        require(len(re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue_base)) == 1,
                "effective UE config is not single-UE")
        require(re.findall(r'(?m)^\s*imsi\s*=\s*"([0-9]+)"\s*;', ue_base)
                == [expected_imsi], "effective UE IMSI mismatch")
        marker = '@include "channelmod_rfsimu_LEO_satellite.conf"'
        require(marker in ue_base, "UE base config lacks expected channel include")
        runtime = self.output_dir / "runtime"
        runtime.mkdir()
        channel_path = runtime / "effective_channel_cold_attach_minus2p5.conf"
        gnb_path = runtime / "effective_gnb_cold_attach_minus2p5.conf"
        ue_path = runtime / "effective_ue_cold_attach_minus2p5.conf"
        n2.atomic_text(channel_path, channel)
        n2.atomic_text(gnb_path, gnb_base + "\n\n" + channel + "\n")
        n2.atomic_text(ue_path, ue_base.replace(marker, channel))
        self.candidate_baked_config_verified = True
        n2.atomic_json(runtime / "config_hashes.json", {
            "source_before": self.source_hashes_before,
            "run_local": {
                "channel_sha256": n2.sha256(channel_path),
                "gnb_sha256": n2.sha256(gnb_path),
                "ue_sha256": n2.sha256(ue_path),
            },
            "startup_channel": configured_channel_values(channel),
            "candidate_baked_before_ue_launch": True,
            "source_paths_written": False,
        })
        return gnb_path, ue_path

    def start_ran(self, gnb_config: Path, ue_config: Path) -> None:
        radio = self.config["radio"]
        build = self.path(self.config["paths"]["oai_ran_build"])
        gnb = [
            "sudo", "-n", "env", "-u", "SCENESENSE_FORCE_UL_MCS",
            f"SCENESENSE_MCS_POLICY={radio['mcs_policy']}", "./nr-softmodem",
            "-O", str(gnb_config), "--gNBs.[0].min_rxtxtime", "6", "--rfsim",
            "--rfsimulator.[0].options", "chanmod", "--telnetsrv",
            "--telnetsrv.listenaddr", self.config["actuator"]["telnet_host"],
            "--telnetsrv.listenport", str(self.config["actuator"]["telnet_port"]),
            "--T_stdout", "2", "--T_nowait", "--T_port",
            str(self.config["telemetry"]["gnb_port"]),
        ]
        self.spawn("gnb", gnb, "logs/gnb.log", cwd=build, root_owned=True)
        time.sleep(float(radio["gnb_start_lead_s"]))
        ue = [
            "sudo", "-n", "./nr-uesoftmodem", "--rfsim",
            "--rfsimulator.[0].serveraddr", "127.0.0.1",
            "--rfsimulator.[0].options", "chanmod", "-r", str(radio["prb"]),
            "--numerology", str(radio["numerology"]), "--band", str(radio["band"]),
            "-C", str(radio["downlink_frequency_hz"]), "-O", str(ue_config),
            "--T_stdout", "2", "--T_nowait", "--T_port",
            str(self.config["telemetry"]["ue_port"]),
        ]
        self.ue_launch_monotonic_ns = time.monotonic_ns()
        self.spawn("ue", ue, "logs/ue.log", cwd=build, root_owned=True)
        n2.atomic_json(self.output_dir / "cold_start_identity.json", {
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "repetition_index": self.repetition_index,
            "candidate_baked_before_ue_launch": self.candidate_baked_config_verified,
            "ue_launch_monotonic_ns": self.ue_launch_monotonic_ns,
        })

    def _core_ready(self) -> bool:
        try:
            for container in self.config["radio"]["core_containers"]:
                state = n2.run_checked([
                    "sudo", "-n", "docker", "inspect", "-f",
                    "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                    container,
                ], timeout=10).stdout.strip()
                if not state.startswith("true") or "unhealthy" in state:
                    return False
            return True
        except Exception:
            return False

    def wait_cold_attach(self) -> bool:
        require(self.ue_launch_monotonic_ns is not None, "UE launch anchor missing")
        radio = self.config["radio"]
        deadline = time.monotonic() + float(radio["attach_timeout_s"])
        interface = str(radio["ue_interface"])
        last_ips: list[str] = []
        while time.monotonic() < deadline:
            self.assert_carla_absent()
            require(all(proc.process.poll() is None for proc in self.processes[:2]),
                    "gNB or UE exited during cold-attach gate")
            result = subprocess.run(
                ["ip", "-j", "-4", "addr", "show", "dev", interface],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                try:
                    last_ips = [
                        str(info["local"]) for row in json.loads(result.stdout)
                        for info in row.get("addr_info", [])
                        if info.get("family") == "inet" and info.get("local")
                    ]
                except (json.JSONDecodeError, KeyError, TypeError):
                    last_ips = []
            if len(last_ips) == 1:
                ping = subprocess.run(
                    ["ping", "-I", interface, "-c", "3", "-W", "2", radio["ext_dn_ip"]],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
                if ping.returncode == 0:
                    self.ue_ip = last_ips[0]
                    n2.atomic_text(self.output_dir / "logs/attach_ping.log", ping.stdout)
                    self.attach_gate = {
                        "status": "COLD_ATTACH_PDU_EXT_DN_GATE_PASSED",
                        "passed": True,
                        "timeout_s": float(radio["attach_timeout_s"]),
                        "duration_s": (time.monotonic_ns() - self.ue_launch_monotonic_ns) / 1e9,
                        "interface": interface,
                        "discovered_ipv4": self.ue_ip,
                        "pdu_session_evidence": "SINGLE_OAI_TUNNEL_IPV4",
                        "ext_dn_ping_pass": True,
                    }
                    n2.atomic_json(self.output_dir / "attach_gate.json", self.attach_gate)
                    n2.atomic_json(self.output_dir / "ue_network_identity.json", {
                        "ue_count": 1, "imsi": radio["expected_imsi"],
                        "interface": interface, "discovered_ipv4": self.ue_ip,
                        "ext_dn_ip": radio["ext_dn_ip"], "ping_pass": True,
                    })
                    return True
            time.sleep(1.0)
        alive = all(proc.process.poll() is None for proc in self.processes[:2])
        self.attach_gate = {
            "status": "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED",
            "passed": False,
            "timeout_s": float(radio["attach_timeout_s"]),
            "duration_s": (time.monotonic_ns() - self.ue_launch_monotonic_ns) / 1e9,
            "interface": interface,
            "observed_ipv4": last_ips,
            "ran_processes_alive_at_terminal": alive,
            "core_ready_at_terminal": self._core_ready(),
        }
        n2.atomic_json(self.output_dir / "attach_gate.json", self.attach_gate)
        return False

    def open_and_validate_candidate_telnet(self) -> int:
        actuator = self.config["actuator"]
        self.telnet = n2.TelnetSession(
            actuator["telnet_host"], int(actuator["telnet_port"]),
            float(actuator["response_timeout_s"]), int(actuator["max_response_bytes"]),
        )
        response = self.telnet.command("channelmod show current")[-1]
        n2.atomic_text(self.output_dir / "channel_state_startup_candidate.txt", response)
        models = validate_runtime_channel_state(response, self.config["startup_channel"])
        row = models[actuator["channel_model_name"]]
        require(self.application_count == 0, "candidate application count changed")
        self.startup_channel_runtime_verified = True
        n2.atomic_json(self.output_dir / "startup_channel_runtime_gate.json", {
            "status": "PASSED",
            "startup_channel_runtime_verified": True,
            "models": models,
            "candidate_application_count": self.application_count,
        })
        self.control_validated = True
        return int(row["model_index"])

    def restore_clean_once(self, model_index: int) -> None:
        require(self.telnet is not None, "restore control session unavailable")
        require(self.restore_application_count == 0,
                "clean restore may be applied only once")
        clean = str(self.config["actuator"]["clean_and_restore_commanded_noise_power_db"])
        self.restore_application_count = 1
        attempted = {
            "restore_application_index": 0,
            "target_noise_power_db": clean,
            "attempted_at_monotonic_ns": time.monotonic_ns(),
            "status": "SEND_ATTEMPT_STARTED_ACK_UNCONFIRMED",
        }
        self.command_rows.append(attempted)
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(
            f"channelmod modify {model_index} noise_power_dB {clean}"
        )
        attempted.update({
            "send_monotonic_ns": sent_mono, "send_wall_time_ns": sent_wall,
            "response_received_monotonic_ns": ack_mono,
            "response_received_wall_time_ns": ack_wall,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        })
        self.validate_modify_response(response, clean)
        state = self.telnet.command("channelmod show current")[-1]
        row = n2.parse_channel_models(state).get(
            self.config["actuator"]["channel_model_name"], {}
        )
        require(math.isclose(float(row.get("noise_power_db", math.nan)), -50.0),
                f"clean restore verification failed: {row}")
        n2.atomic_text(self.output_dir / "channel_state_restored.txt", state)
        attempted["status"] = "ACK_AND_POST_STATE_VALIDATED_ONCE"
        self.restored = True

    def best_effort_restore(self) -> None:
        if self.restored or self.restore_application_count != 0 or self.telnet is None:
            return
        try:
            response = self.telnet.command("channelmod show current")[-1]
            row = n2.parse_channel_models(response).get(
                self.config["actuator"]["channel_model_name"], {}
            )
            if row:
                self.restore_clean_once(int(row["model_index"]))
        except Exception:
            return

    def source_integrity(self) -> dict[str, Any]:
        paths = self.config["paths"]
        conf_root = self.path(paths["oai_ran_conf"])
        current = {
            "gnb_base": n2.sha256(conf_root / paths["gnb_base_config"]),
            "ue_base": n2.sha256(conf_root / paths["ue_base_config"]),
            "channel": n2.sha256(conf_root / paths["channel_config"]),
        }
        unchanged = bool(self.source_hashes_before) and current == self.source_hashes_before
        result = {
            "status": "UNCHANGED" if unchanged else "DRIFT_DETECTED",
            "unchanged": unchanged,
            "before": self.source_hashes_before,
            "after": current,
            "source_paths_written": False,
        }
        n2.atomic_json(self.output_dir / "source_oai_config_integrity.json", result)
        return result

    def write_manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        classification = classify_repetition(summary)
        if classification["evidence_valid_for_aggregation"]:
            if classification["joint_candidate_confirmation_pass"]:
                final_status = REP_PASSED
            elif classification["classified_outcome"] == "COLD_ATTACH_FAILED":
                final_status = REP_ATTACH_FAILED
            elif classification["classified_outcome"] == "ACHIEVED_SNR_OUTSIDE_FROZEN_CANDIDATE_BAND":
                final_status = REP_ACHIEVED_SNR_MISMATCH
            else:
                final_status = REP_SERVICE_FAILED
        elif status == RESTORE_FAILED:
            final_status = RESTORE_FAILED
        elif status == "FAILED":
            final_status = "FAILED"
        else:
            final_status = REP_UNCONFIRMED
        augmented = {
            **dict(summary), "status": final_status,
            "engine_status": status,
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": -2.5,
            "outcome_classification": classification,
            **classification,
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "restore_application_count": self.restore_application_count,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
        }
        n2.atomic_json(self.output_dir / self.config["output"]["repetition_summary"], augmented)
        excluded = {"manifest.json", "FAILED.json"}
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name in excluded or path.name.startswith("UE_N3B_"):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size, "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / self.config["output"]["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3b_cold_attach_repetition_manifest.v1",
            "status": final_status,
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": -2.5,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "restore_application_count": self.restore_application_count,
            "n3a_live_manifest_sha256": self.n3a_proof["manifest_sha256"],
            "n3a_adjudication_manifest_sha256": self.adjudication_proof["manifest_sha256"],
            "evidence_valid_for_aggregation": classification["evidence_valid_for_aggregation"],
            "joint_candidate_confirmation_pass": classification["joint_candidate_confirmation_pass"],
            "review_required": True,
            "operational_bound_promoted": False,
            "outputs": files,
        })
        terminal = {**augmented, "manifest_sha256": n2.sha256(manifest_path)}
        name = self.config["output"]["failure"] if final_status == "FAILED" else f"{final_status}.json"
        n2.atomic_json(self.output_dir / name, terminal)

    def run(self) -> int:
        n2.atomic_json(self.output_dir / "resolved_config.json", {
            **self.config,
            "resolved_repetition": {
                "repetition_index": self.repetition_index,
                "commanded_noise_power_db": -2.5,
            },
        })
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise ColdAttachFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught] = signal.getsignal(caught)
            signal.signal(caught, terminate)
        try:
            self.preflight()
            gnb_config, ue_config = self.materialize_configs()
            self.start_ran(gnb_config, ue_config)
            attached = self.wait_cold_attach()
            n2.wait_tcp(int(self.config["actuator"]["telnet_port"]), 15)
            model_index = self.open_and_validate_candidate_telnet()
            if not attached:
                self.restore_clean_once(model_index)
                self.write_command_log()
                self.cleanup(strict=True)
                source = self.source_integrity()
                cleanup = load_json(self.output_dir / "cleanup_report.json")
                summary = {
                    "status": REP_ATTACH_FAILED,
                    "attach_gate": self.attach_gate,
                    "service_tail": None, "service_window": None,
                    "transport": None,
                    "clean_recovery": {"status": "NOT_APPLICABLE_NO_PDU_SESSION", "passed": None},
                    "hard_loss_reason": None,
                    "candidate_baked_config_verified": self.candidate_baked_config_verified,
                    "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                    "candidate_application_count": self.application_count,
                    "restore_application_count": self.restore_application_count,
                    "clean_restore_verified": self.restored,
                    "source_oai_configs_unchanged": source["unchanged"],
                    "cleanup_clean": cleanup.get("clean") is True,
                }
                self.write_manifest_terminal(REP_ATTACH_FAILED, summary)
                return 0 if classify_repetition(summary)["evidence_valid_for_aggregation"] else 1

            self.start_telemetry()
            time.sleep(0.75)
            self.start_probe()
            # The inherited lead establishes one current RNTI under the already
            # baked candidate. It is diagnostic and outside the 600-frame unit.
            self.establish_clean_lead()
            tail_start = time.monotonic_ns()
            tail_start_wall = time.time_ns()
            frozen_duration_ns = 60_000_000_000
            frozen_end = tail_start + frozen_duration_ns
            frozen_end_wall = tail_start_wall + frozen_duration_ns
            tail: dict[str, Any] | None = None
            service_loss_status: str | None = None
            try:
                self.wait_for(60.0, enforce_silence=True)
                tail = calibration.summarize_tail(
                    self.live_csv.snapshot() if self.live_csv else [],
                    self.live_mcs.snapshot() if self.live_mcs else [],
                    start_ns=tail_start, end_ns=frozen_end,
                    expected_rnti=int(self.current_rnti),
                    minimum_pusch=int(self.config["rung"]["minimum_service_pusch_samples"]),
                    minimum_mcs=int(self.config["rung"]["minimum_service_mcs_samples"]),
                    required_mcs_table=int(self.config["analysis"]["scheduler_required_mcs_table"]),
                    required_force_mcs=int(self.config["analysis"]["scheduler_required_force_ul_mcs"]),
                )
                tail.update({
                    "start_wall_time_ns": tail_start_wall,
                    "end_wall_time_ns": frozen_end_wall,
                    "frozen_duration_ns": frozen_duration_ns,
                })
            except calibration.HardServiceLoss as exc:
                self.hard_loss_reason = str(exc)
                service_loss_status = calibration.classify_service_loss_reason(str(exc))

            self.restore_clean_once(model_index)
            pusch_baseline = self.live_csv.count() if self.live_csv else 0
            if self.event_tail is not None:
                self.event_tail.poll()
            receiver_baseline = self.event_tail.accepted_count if self.event_tail else 0
            recovery = self.verify_recovery(pusch_baseline, receiver_baseline, required=False)
            transport = self.finish_probe(allow_partial_sender=service_loss_status is not None)
            service = None
            if tail is not None:
                service = calibration.evaluate_tail_service(
                    self.output_dir / "traffic/sender.csv",
                    self.output_dir / "traffic/receiver_events.jsonl",
                    start_wall_ns=tail_start_wall, end_wall_ns=frozen_end_wall,
                    fps=10.0, expected_tail_frames=600,
                    expected_source_ip=self.config["transport_gates"]["expected_source_ip"],
                    structural_integrity=bool(transport["integrity_gate"]),
                    gates=self.config["transport_gates"],
                )
                n2.atomic_json(self.output_dir / "service_window_summary.json", service)
            self.write_command_log()
            self.cleanup(strict=True)
            source = self.source_integrity()
            self.extract_ttracer()
            self.write_raw_limit_record()
            cleanup = load_json(self.output_dir / "cleanup_report.json")
            summary = {
                "status": service_loss_status or "COLD_ATTACH_SERVICE_WINDOW_CAPTURED",
                "attach_gate": self.attach_gate,
                "service_tail": tail,
                "service_window": service,
                "transport": transport,
                "clean_recovery": recovery,
                "hard_loss_reason": self.hard_loss_reason,
                "receiver_service_outage_detected": self.receiver_service_outage_detected,
                "candidate_baked_config_verified": self.candidate_baked_config_verified,
                "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                "candidate_application_count": self.application_count,
                "restore_application_count": self.restore_application_count,
                "clean_restore_verified": self.restored,
                "source_oai_configs_unchanged": source["unchanged"],
                "cleanup_clean": cleanup.get("clean") is True,
            }
            self.write_manifest_terminal(str(summary["status"]), summary)
            return 0 if classify_repetition(summary)["evidence_valid_for_aggregation"] else 1
        except (Exception, KeyboardInterrupt) as exc:
            self.best_effort_restore()
            try:
                self.write_command_log()
            except Exception:
                pass
            cleanup_errors = self.cleanup(strict=False)
            source = self.source_integrity() if self.source_hashes_before else {"unchanged": False}
            status = (
                RESTORE_FAILED
                if self.startup_channel_runtime_verified and not self.restored
                else "FAILED"
            )
            failure = {
                "status": status,
                "error_type": type(exc).__name__, "error": str(exc),
                "attach_gate": self.attach_gate,
                "candidate_baked_config_verified": self.candidate_baked_config_verified,
                "startup_channel_runtime_verified": self.startup_channel_runtime_verified,
                "candidate_application_count": self.application_count,
                "restore_application_count": self.restore_application_count,
                "clean_restore_verified": self.restored,
                "source_oai_configs_unchanged": source.get("unchanged") is True,
                "cleanup_clean": not cleanup_errors,
                "cleanup_errors": cleanup_errors,
            }
            self.write_manifest_terminal(status, failure)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


def verify_repetition_evidence(
    directory: Path, *, repetition_index: int, config_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    summary_path = directory / "repetition_summary.json"
    require(manifest_path.is_file() and summary_path.is_file(),
            f"repetition evidence is incomplete: {directory}")
    manifest, summary = load_json(manifest_path), load_json(summary_path)
    status = str(summary.get("status", ""))
    require(status in {
        REP_PASSED, REP_ATTACH_FAILED, REP_SERVICE_FAILED, REP_ACHIEVED_SNR_MISMATCH,
    },
            f"repetition status is not valid evidence: {status}")
    terminal_path = directory / f"{status}.json"
    require(terminal_path.is_file(), "repetition terminal is absent")
    terminal = load_json(terminal_path)
    require(manifest.get("status") == terminal.get("status") == status,
            "repetition status binding mismatch")
    require(terminal.get("manifest_sha256") == n2.sha256(manifest_path),
            "repetition terminal/manifest hash mismatch")
    require(manifest.get("config_sha256") == config_sha256
            and manifest.get("runner_sha256") == runner_sha256,
            "repetition source seal mismatch")
    require(int(manifest.get("repetition_index", -1)) == repetition_index,
            "repetition identity mismatch")
    require(manifest.get("candidate_application_count") == 0
            and manifest.get("restore_application_count") == 1,
            "repetition command-count contract failed")
    for payload, label in ((manifest, "manifest"), (summary, "summary"), (terminal, "terminal")):
        require(math.isclose(float(payload.get("commanded_noise_power_db", math.nan)), -2.5),
                f"repetition {label} command identity mismatch")
    require(terminal.get("evidence_valid_for_aggregation") is True,
            "repetition evidence is not valid for aggregation")
    _verify_manifest_inventory(directory, manifest)
    cleanup = load_json(directory / "cleanup_report.json")
    require(cleanup.get("clean") is True and not cleanup.get("errors"),
            "repetition cleanup is not clean")
    ran_epoch = str(manifest.get("ran_epoch_id", ""))
    control_session = str(manifest.get("control_session_id", ""))
    require(ran_epoch and control_session and ran_epoch != control_session,
            "repetition lacks distinct RAN/control identities")
    return {
        "status": "VERIFIED_N3B_REPETITION_EVIDENCE",
        "directory": str(directory),
        "repetition_index": repetition_index,
        "manifest_sha256": n2.sha256(manifest_path),
        "terminal_sha256": n2.sha256(terminal_path),
        "ran_epoch_id": ran_epoch,
        "control_session_id": control_session,
        "cold_attach_gate_pass": terminal["cold_attach_gate_pass"],
        "authoritative_service_gate_pass": terminal["authoritative_service_gate_pass"],
        "achieved_snr_gate_pass": terminal["achieved_snr_gate_pass"],
        "joint_candidate_confirmation_pass": terminal["joint_candidate_confirmation_pass"],
    }


class CampaignRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise ColdAttachFailure(f"create-only output already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)

    def write_plan(self, *, require_live: bool = False) -> list[dict[str, Any]]:
        validate_config(self.config, verify_hashes=True, require_live=require_live)
        n2.atomic_json(self.output_dir / self.config["output"]["resolved_config"], self.config)
        rows = campaign_plan_rows(self.config)
        calibration.write_csv(self.output_dir / self.config["output"]["campaign_plan"], rows)
        return rows

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(self.output_dir / self.config["output"]["campaign_summary"], summary)
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path(self.config["output"]["manifest"]):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3B_") or path.name == "FAILED.json"
            ):
                continue
            files.append({"path": str(relative), "bytes": path.stat().st_size,
                          "sha256": n2.sha256(path)})
        manifest_path = self.output_dir / self.config["output"]["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3b_cold_attach_campaign_manifest.v1",
            "status": status,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "outputs": files,
        })
        terminal = {
            **dict(summary), "status": status,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "manifest_sha256": n2.sha256(manifest_path),
        }
        terminal_name = self.config["output"]["failure"] if status == "FAILED" else f"{status}.json"
        n2.atomic_json(self.output_dir / terminal_name, terminal)

    def prepare(self) -> int:
        try:
            rows = self.write_plan(require_live=False)
            pending = pending_adjudication(self.config["predecessors"]["n3a_adjudication"])
            n3a_proof = verify_predecessor(
                self.config, "n3a_live_evidence",
                proof_path=self.output_dir / "n3a_live_evidence_predecessor.json",
            )
            authority_ready = (
                self.config["authority"]["live_oai_run_authorized"] is True
                and self.config["authority"]["live_socket_execution_authorized"] is True
            )
            adjudication_proof = None
            if not pending:
                adjudication_proof = verify_predecessor(
                    self.config, "n3a_adjudication",
                    proof_path=self.output_dir / "n3a_adjudication_predecessor.json",
                )
            ready = not pending and authority_ready and adjudication_proof is not None
            status = PLAN_READY if ready else PLAN_BLOCKED
            summary = {
                "status": status,
                "runtime_executed": False, "socket_executed": False,
                "repetitions_planned": len(rows), "plan": rows,
                "adjudication_predecessor_pending": pending,
                "n3a_live_predecessor": n3a_proof,
                "n3a_adjudication_predecessor": adjudication_proof,
                "live_authority_ready": authority_ready,
                "live_execution_blocked": not ready,
                "cold_attach_bound_evaluated": False,
                "operational_bound_promoted": False,
                "next": (
                    "EXPLICIT_EXECUTE_LIVE"
                    if ready
                    else "FINALIZE_ADJUDICATION_SEALS_AND_EXPLICIT_LIVE_AUTHORITY"
                ),
            }
            self.manifest_terminal(status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                "runtime_executed": False, "socket_executed": False,
                "cold_attach_bound_evaluated": False,
            }
            self.manifest_terminal("FAILED", failure)
            return 1

    def execute(self) -> int:
        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        proofs: list[dict[str, Any]] = []
        n3a_proof: dict[str, Any] | None = None
        adjudication_proof: dict[str, Any] | None = None
        ran_epochs: set[str] = set()
        control_sessions: set[str] = set()
        try:
            rows = self.write_plan(require_live=True)
            n3a_proof = verify_predecessor(
                self.config, "n3a_live_evidence",
                proof_path=self.output_dir / "n3a_live_evidence_predecessor.json",
            )
            adjudication_proof = verify_predecessor(
                self.config, "n3a_adjudication",
                proof_path=self.output_dir / "n3a_adjudication_predecessor.json",
            )
            for row in rows:
                index = int(row["repetition_index"])
                directory = self.output_dir / "repetitions" / f"rep_{index:02d}_minus2p5_cold"
                runner = ColdAttachRepetitionRunner(
                    self.config_path, directory, repetition_index=index,
                    n3a_proof=n3a_proof, adjudication_proof=adjudication_proof,
                )
                rc = runner.run()
                summary_path = directory / self.config["output"]["repetition_summary"]
                require(summary_path.is_file(), f"repetition summary absent: {directory}")
                result = load_json(summary_path)
                results.append(result)
                require(rc == 0, f"invalid repetition evidence at repetition {index}")
                proof = verify_repetition_evidence(
                    directory, repetition_index=index,
                    config_sha256=n2.sha256(self.config_path),
                    runner_sha256=n2.sha256(Path(__file__).resolve()),
                )
                require(proof["ran_epoch_id"] not in ran_epochs,
                        "fresh RAN epoch identity was reused")
                require(proof["control_session_id"] not in control_sessions,
                        "control-session identity was reused")
                ran_epochs.add(proof["ran_epoch_id"])
                control_sessions.add(proof["control_session_id"])
                proofs.append(proof)
            aggregation = aggregate_results(results)
            status = str(aggregation["status"])
            summary = {
                "status": status, "runtime_executed": True, "socket_executed": True,
                "repetitions_planned": 3, "repetitions_executed": len(results),
                "fresh_ran_epoch_count": len(ran_epochs),
                "unique_control_session_count": len(control_sessions),
                **aggregation, "results": results, "repetition_evidence": proofs,
                "n3a_live_predecessor": n3a_proof,
                "n3a_adjudication_predecessor": adjudication_proof,
                "next": "REVIEW_N3B_EVIDENCE_NO_AUTOMATIC_BOUND_PROMOTION",
            }
            self.manifest_terminal(status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                "repetitions_planned": len(rows), "repetitions_executed": len(results),
                "results": results, "repetition_evidence": proofs,
                "n3a_live_predecessor": n3a_proof,
                "n3a_adjudication_predecessor": adjudication_proof,
                "cold_attach_bound_evaluated": False,
                "operational_bound_promoted": False,
            }
            self.manifest_terminal("FAILED", failure)
            return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=(PREPARE_ONLY, EXECUTE_LIVE), default=PREPARE_ONLY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = CampaignRunner(Path(args.config), Path(args.output_dir))
    return runner.prepare() if args.mode == PREPARE_ONLY else runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())

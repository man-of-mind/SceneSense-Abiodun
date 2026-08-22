#!/usr/bin/env python3
"""Freeze the bounded UE-N3C adjacent cold-attach refinement plan.

This version is intentionally PREPARE-only.  It verifies the sealed N3B
outcome and the sealed command-ladder evidence, then emits a three-repetition
plan for the single adjacent stronger RFsim command, -3.0 dB.  EXECUTE_LIVE
fails closed until a create-only N3B adjudication is sealed and a distinct
explicit live configuration and live engine are reviewed.

The command-ladder observation of 6.5 dB median achieved PUSCH SNR is used only
as a pre-registered consistency expectation (plus or minus 0.5 dB).  This
stage cannot promote an actuator mapping or any numeric, connectivity,
usable-service, or operational bound.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as calibration  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3c_oai_ul_cold_attach_refinement_v1.json"
PREPARE_ONLY = "PREPARE_ONLY"
EXECUTE_LIVE = "EXECUTE_LIVE"
SCHEMA = "scenesense.ue_n3c_oai_ul_cold_attach_refinement_config.v1"
PLAN_BLOCKED = "UE_N3C_COLD_ATTACH_REFINEMENT_PLAN_FROZEN_PREREQUISITES_PENDING"
PLAN_READY = "UE_N3C_COLD_ATTACH_REFINEMENT_PLAN_FROZEN_READY_FOR_EXPLICIT_EXECUTE_LIVE"
FAILED = "FAILED"
PENDING = "PENDING_CREATE_ONLY_N3B_ADJUDICATION"
LIVE_AUTHORITY_BASIS = (
    "EXPLICIT_FUTURE_USER_AUTHORITY_AFTER_SEALED_N3B_ADJUDICATION"
)
CANDIDATE_COMMAND_DB = -3.0
EXPECTED_ACHIEVED_PUSCH_SNR_DB = 6.5
ACHIEVED_SNR_TOLERANCE_DB = 0.5

EXPECTED_N3B_LIVE = {
    "directory": (
        "rl_agent/experiments/ue_n3b_oai_ul_cold_attach_confirmation_v1/"
        "20260821_live_01"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": "ac76763ea9651212f0003c35eb19092d85dbf28b104572e9a9ffc107cb298f3a",
    "terminal": "UE_N3B_COLD_ATTACH_CONFIRMATION_NOT_3_OF_3_REVIEW_REQUIRED.json",
    "terminal_sha256": "50066ffcba4137f585a55ad8d9ebcc728a7ca0e8cceec9728c0b8cc7f644c183",
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": "fdc46f717392b0b71fff9fc6f66d0afc9da831e48ce41cc566af909bb61d7072",
    "required_status": "UE_N3B_COLD_ATTACH_CONFIRMATION_NOT_3_OF_3_REVIEW_REQUIRED",
    "source_config_sha256": "72583ccbf56347d65f8a3c937505b932cc08670064097e4f7388bccd347a33e4",
    "source_runner_sha256": "3cb0b1e975ae58f9fffc510ec20622bded9237848498f5d3c13bfd0b0f3b57f7",
}

EXPECTED_COMMAND_LADDER = {
    "directory": (
        "rl_agent/experiments/ue_n3_oai_ul_command_calibration_v1/"
        "20260821_command_search_02"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": "e8235586c07c5996aab17219280da35947bc8e54b05fc43ee326ee5373618f82",
    "required_status": "UE_N3_TARGET_CALIBRATION_UNRESOLVED",
    "source_config_sha256": "04609685a6993b0ef6bbf55c6d694736ea5f8cb4715abe0ebb2b2b9a793943da",
    "source_runner_sha256": "30cf1615f51c7cd0ebe4087f7b6ca66f37a563f2d5c452e474ae838a23c8878b",
    "rung_directory": "rungs/rung_02_minus3p0",
    "rung_manifest": "manifest.json",
    "rung_manifest_sha256": "0913985a99d3765a9c88d824ddb2ef38079085826dfded9dce583e621db7f302",
    "rung_summary": "rung_summary.json",
}


class ColdAttachRefinementFailure(calibration.CalibrationFailure):
    """Fail-closed plan, evidence, authority, or infrastructure failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ColdAttachRefinementFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ColdAttachRefinementFailure(
            f"path escapes repository root: {relative}"
        ) from exc
    return path


def pending_adjudication(block: Mapping[str, Any]) -> bool:
    required = (
        "directory",
        "manifest_sha256",
        "terminal_sha256",
        "resolved_config_sha256",
        "source_config_sha256",
        "source_runner_sha256",
    )
    return any(str(block.get(key, "")) == PENDING for key in required)


def _verify_runtime_seals(config: Mapping[str, Any]) -> None:
    required_paths = {
        "rl_agent/ue_n3c_oai_ul_cold_attach_refinement_v1.py",
        "rl_agent/ue_n3b_oai_ul_cold_attach_confirmation_v1.py",
        "rl_agent/ue_n3_oai_ul_command_calibration_v1.py",
        "rl_agent/ue_n2_oai_ul_calibration_smoke.py",
        "rl_agent/ue_n3_structured_udp_receiver.py",
        "oai_layer_latency/carla_shaped_udp_burst_sender.py",
        (
            "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/"
            "gnb.sa.band78.fr1.106PRB.usrpb210.conf"
        ),
        "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/ue.conf",
        (
            "OAI/openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/"
            "channelmod_rfsimu.conf"
        ),
    }
    seals = list(config["runtime_seals"])
    paths = [str(row.get("path", "")) for row in seals]
    require(len(paths) == len(set(paths)), "runtime seal paths repeat")
    require(required_paths.issubset(paths), "required N3C runtime seals are absent")
    for seal in seals:
        path = resolve_repo_path(str(seal["path"]))
        require(path.is_file(), f"sealed runtime file missing: {path}")
        require(
            re.fullmatch(r"[0-9a-f]{64}", str(seal.get("sha256", ""))) is not None,
            f"malformed runtime seal: {seal.get('path')}",
        )
        require(
            n2.sha256(path) == seal["sha256"],
            f"runtime seal drift: {seal['path']}",
        )


def validate_config(
    config: Mapping[str, Any], *, verify_hashes: bool = True,
    require_live: bool = False,
) -> None:
    require(config.get("schema") == SCHEMA, "unexpected N3C config schema")
    require(
        config.get("experiment_id") == "ue_n3c_oai_ul_cold_attach_refinement_v1",
        "N3C experiment identity drift",
    )
    require(
        config.get("claim_boundary")
        == "ADJACENT_COLD_ATTACH_REFINEMENT_EVIDENCE_ONLY_NO_BOUND_PROMOTION",
        "N3C claim boundary drift",
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
    require(
        authority.get("live_authority_basis")
        == (LIVE_AUTHORITY_BASIS if live_enabled else "NOT_AUTHORIZED_PREPARE_ONLY"),
        "live authority basis drift",
    )
    for key in (
        "carla_run_authorized",
        "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized",
        "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized",
        "operational_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    if require_live:
        require(live_enabled, "EXECUTE_LIVE authority is absent")

    predecessors = config["predecessors"]
    require(predecessors.get("n3b_live_evidence") == EXPECTED_N3B_LIVE,
            "sealed N3B live evidence drift")
    require(predecessors.get("command_ladder_expectation") == EXPECTED_COMMAND_LADDER,
            "sealed -3.0 command-ladder expectation drift")
    require(
        predecessors.get("ue_n1_bundle")
        == "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v2",
        "UE-N1 bundle drift",
    )
    adjudication = predecessors["n3b_adjudication"]
    if pending_adjudication(adjudication):
        for key in (
            "directory",
            "manifest_sha256",
            "terminal_sha256",
            "resolved_config_sha256",
            "source_config_sha256",
            "source_runner_sha256",
        ):
            require(str(adjudication.get(key, "")) == PENDING,
                    "N3B adjudication predecessor is only partially pinned")
        require(not require_live, "N3B adjudication predecessor is still pending")
    else:
        for key in (
            "manifest_sha256",
            "terminal_sha256",
            "resolved_config_sha256",
            "source_config_sha256",
            "source_runner_sha256",
        ):
            require(
                re.fullmatch(r"[0-9a-f]{64}", str(adjudication.get(key, "")))
                is not None,
                f"N3B adjudication {key} is malformed",
            )
        require(
            adjudication.get("required_status")
            == "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED",
            "N3B adjudication status contract drift",
        )
        require(
            adjudication.get("n3c_eligibility_status")
            == "UE_N3B_VALID_COLD_ATTACH_FAILURE_ACCEPTED_FOR_N3C",
            "N3B adjudication eligibility drift",
        )
        require(
            math.isclose(
                float(adjudication.get("n3c_selected_command_db", math.nan)),
                CANDIDATE_COMMAND_DB,
            ),
            "N3B adjudication did not select command -3.0",
        )

    campaign = config["campaign"]
    require(int(campaign["repetitions"]) == 3,
            "N3C requires exactly three repetitions")
    require(campaign["one_fresh_ran_per_repetition"] is True,
            "every repetition must use a fresh RAN")
    require(campaign["run_local_configs_only"] is True,
            "N3C must use run-local configs")
    require(campaign["candidate_baked_before_ue_launch"] is True,
            "candidate must be baked before UE launch")
    require(int(campaign["candidate_application_count"]) == 0,
            "candidate Telnet application is forbidden")
    require(
        [float(value) for value in campaign["commanded_noise_power_db"]]
        == [CANDIDATE_COMMAND_DB],
        "N3C candidate command drift",
    )
    require(campaign["continue_after_valid_attach_or_service_failure"] is True,
            "valid candidate failures must be retained")
    require(campaign["stop_on_invalid_or_unclean_evidence"] is True,
            "invalid evidence must stop the campaign")
    require(campaign["post_restore_recovery_fail_closed"] is True,
            "post-restore recovery must be infrastructure fail-closed")

    require(
        config["startup_channel"] == {
            "rfsimu_channel_enB0": -50.0,
            "rfsimu_channel_enB1": -50.0,
            "rfsimu_channel_ue0": CANDIDATE_COMMAND_DB,
        },
        "startup channel values drift",
    )
    rung, traffic = config["rung"], config["traffic"]
    require(math.isclose(float(rung["candidate_lead_s"]), 5.0),
            "candidate lead must remain 5 seconds")
    require(math.isclose(float(rung["clean_lead_s"]), 5.0),
            "lead alias must remain 5 seconds")
    require(math.isclose(float(rung["measured_service_s"]), 60.0),
            "service window must remain exactly 60 seconds")
    require(math.isclose(float(rung["clean_recovery_s"]), 5.0),
            "clean recovery must remain 5 seconds")
    require(math.isclose(float(rung["service_duration_s"]), 70.0),
            "bounded traffic duration must remain 70 seconds")
    require(int(rung["sender_frames"]) == 700,
            "probe must schedule exactly 700 frames")
    require(int(rung["expected_service_frames"]) == 600,
            "authoritative service window must contain 600 frames")
    require(math.isclose(float(traffic["fps"]), 10.0),
            "probe must use 10 Hz")
    require(int(traffic["frame_bytes"]) == 12_500
            and int(traffic["chunk_bytes"]) == 12_500,
            "matched 1 Mbps probe shape drift")
    require(math.isclose(float(config["radio"]["attach_timeout_s"]), 180.0),
            "cold attach/PDU/ext-DN gate must remain 180 seconds")

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

    analysis = config["analysis"]
    require(
        math.isclose(
            float(analysis["expected_achieved_pusch_snr_db"]),
            EXPECTED_ACHIEVED_PUSCH_SNR_DB,
        )
        and math.isclose(
            float(analysis["achieved_snr_tolerance_db"]),
            ACHIEVED_SNR_TOLERANCE_DB,
        ),
        "N3C achieved-SNR consistency band drift",
    )
    require(
        analysis["achieved_snr_expectation_role"]
        == "CONSISTENCY_ONLY_FROM_SEALED_WARM_ATTACHED_COMMAND_LADDER_NOT_MAPPING",
        "achieved-SNR expectation role drift",
    )
    require(analysis["post_restore_recovery_required"] is True,
            "post-restore recovery must be required")
    if verify_hashes:
        _verify_runtime_seals(config)


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    ready = (
        not pending_adjudication(config["predecessors"]["n3b_adjudication"])
        and config["authority"]["live_oai_run_authorized"] is True
        and config["authority"]["live_socket_execution_authorized"] is True
    )
    status = (
        "READY_FOR_EXPLICIT_EXECUTE_LIVE"
        if ready else "BLOCKED_PENDING_PREREQUISITES"
    )
    return [
        {
            "sequence_index": index - 1,
            "repetition_index": index,
            "commanded_noise_power_db": CANDIDATE_COMMAND_DB,
            "expected_achieved_pusch_snr_db": EXPECTED_ACHIEVED_PUSCH_SNR_DB,
            "achieved_snr_tolerance_db": ACHIEVED_SNR_TOLERANCE_DB,
            "achieved_snr_expectation_role": "CONSISTENCY_ONLY_NOT_PROMOTED",
            "fresh_ran_epoch_required": True,
            "candidate_baked_before_ue_launch": True,
            "candidate_application_count": 0,
            "attach_pdu_ext_dn_timeout_s": 180.0,
            "measured_service_s": 60.0,
            "expected_service_frames": 600,
            "restore_commanded_noise_power_db": -50.0,
            "post_restore_recovery_required": True,
            "status": status,
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
            raise ColdAttachRefinementFailure(
                f"predecessor output escapes directory: {relative}"
            ) from exc
        require(artifact.is_file(), f"predecessor output missing: {relative}")
        require(artifact.stat().st_size == int(row.get("bytes", -1)),
                f"predecessor output size drift: {relative}")
        require(n2.sha256(artifact) == row.get("sha256"),
                f"predecessor output hash drift: {relative}")
    return seen


def verify_n3b_live(config: Mapping[str, Any], *, proof_path: Path | None = None) -> dict[str, Any]:
    expected = config["predecessors"]["n3b_live_evidence"]
    require(expected == EXPECTED_N3B_LIVE, "N3B live evidence pin drift")
    directory = resolve_repo_path(str(expected["directory"]))
    paths = {
        "manifest": directory / str(expected["manifest"]),
        "terminal": directory / str(expected["terminal"]),
        "resolved_config": directory / str(expected["resolved_config"]),
    }
    for key, path in paths.items():
        require(path.is_file(), f"N3B predecessor {key} missing: {path}")
        require(n2.sha256(path) == expected[f"{key}_sha256"],
                f"N3B predecessor {key} seal drift")
    manifest, terminal = load_json(paths["manifest"]), load_json(paths["terminal"])
    require(manifest.get("status") == terminal.get("status")
            == expected["required_status"], "N3B predecessor status mismatch")
    require(terminal.get("manifest_sha256") == expected["manifest_sha256"],
            "N3B terminal does not bind its manifest")
    require(manifest.get("config_sha256") == expected["source_config_sha256"]
            and manifest.get("runner_sha256") == expected["source_runner_sha256"],
            "N3B source identity mismatch")
    seen = _verify_manifest_inventory(directory, manifest)
    require(str(expected["resolved_config"]) in seen,
            "N3B resolved config is absent from manifest")
    require(int(terminal.get("repetitions_executed", -1)) == 3,
            "N3B did not execute three repetitions")
    require(int(terminal.get("cold_attach_passes", -1)) == 0,
            "N3B outcome is not 0/3 cold attach")
    require(int(terminal.get("valid_nonconfirming_outcomes_retained", -1)) == 3,
            "N3B lacks three valid nonconfirming outcomes")
    require(int(terminal.get("fresh_ran_epoch_count", -1)) == 3
            and int(terminal.get("unique_control_session_count", -1)) == 3,
            "N3B fresh-start identities are incomplete")
    for key in (
        "target_mapping_promoted",
        "numeric_bound_promoted",
        "connectivity_bound_promoted",
        "usable_service_bound_promoted",
        "operational_bound_promoted",
    ):
        require(terminal.get(key) is False,
                f"N3B predecessor unexpectedly promoted {key}")
    proof = {
        "status": "VERIFIED_READ_ONLY_N3B_PREDECESSOR",
        "directory": str(directory),
        "manifest_sha256": n2.sha256(paths["manifest"]),
        "terminal_sha256": n2.sha256(paths["terminal"]),
        "resolved_config_sha256": n2.sha256(paths["resolved_config"]),
        "verified_output_count": len(seen),
        "cold_attach_passes": 0,
        "cold_attach_trials": 3,
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def verify_command_ladder_expectation(
    config: Mapping[str, Any], *, proof_path: Path | None = None,
) -> dict[str, Any]:
    expected = config["predecessors"]["command_ladder_expectation"]
    require(expected == EXPECTED_COMMAND_LADDER,
            "command-ladder expectation pin drift")
    directory = resolve_repo_path(str(expected["directory"]))
    manifest_path = directory / str(expected["manifest"])
    require(manifest_path.is_file(), "command-ladder campaign manifest missing")
    require(n2.sha256(manifest_path) == expected["manifest_sha256"],
            "command-ladder campaign manifest seal drift")
    manifest = load_json(manifest_path)
    require(manifest.get("status") == expected["required_status"],
            "command-ladder status mismatch")
    require(manifest.get("config_sha256") == expected["source_config_sha256"]
            and manifest.get("runner_sha256") == expected["source_runner_sha256"],
            "command-ladder source identity mismatch")
    _verify_manifest_inventory(directory, manifest)
    rung = directory / str(expected["rung_directory"])
    rung_manifest_path = rung / str(expected["rung_manifest"])
    summary_path = rung / str(expected["rung_summary"])
    require(n2.sha256(rung_manifest_path) == expected["rung_manifest_sha256"],
            "-3.0 rung manifest seal drift")
    rung_manifest = load_json(rung_manifest_path)
    _verify_manifest_inventory(rung, rung_manifest)
    summary = load_json(summary_path)
    tail, service = summary["tail"], summary["tail_service"]
    require(math.isclose(float(summary["commanded_noise_power_db"]), -3.0),
            "command-ladder rung is not -3.0")
    require(math.isclose(float(tail["achieved_pusch_snr_db_p05"]), 6.0)
            and math.isclose(float(tail["achieved_pusch_snr_db_median"]), 6.5)
            and math.isclose(float(tail["achieved_pusch_snr_db_p95"]), 7.0),
            "sealed -3.0 achieved-SNR evidence drift")
    require(int(tail["pusch_samples"]) == 665 and int(tail["mcs_samples"]) == 665,
            "sealed -3.0 telemetry sample count drift")
    require(int(service["received_frames"]) == 50
            and int(service["expected_frames"]) == 50
            and service["primary_99_pass"] is True
            and service["no_one_second_outage_pass"] is True,
            "sealed -3.0 short service evidence drift")
    proof = {
        "status": "VERIFIED_CONSISTENCY_EXPECTATION_ONLY_NOT_MAPPING",
        "directory": str(rung),
        "campaign_manifest_sha256": n2.sha256(manifest_path),
        "rung_manifest_sha256": n2.sha256(rung_manifest_path),
        "commanded_noise_power_db": -3.0,
        "achieved_pusch_snr_db_p05": 6.0,
        "achieved_pusch_snr_db_p50": 6.5,
        "achieved_pusch_snr_db_p95": 7.0,
        "mapping_promoted": False,
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def verify_n3b_adjudication(
    config: Mapping[str, Any], *, proof_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the future create-only N3B review before plan readiness."""
    expected = config["predecessors"]["n3b_adjudication"]
    require(not pending_adjudication(expected),
            "N3B adjudication predecessor is still pending")
    directory = resolve_repo_path(str(expected["directory"]))
    paths = {
        "manifest": directory / str(expected["manifest"]),
        "terminal": directory / str(expected["terminal"]),
        "resolved_config": directory / str(expected["resolved_config"]),
    }
    for key, path in paths.items():
        require(path.is_file(), f"N3B adjudication {key} missing: {path}")
        require(n2.sha256(path) == expected[f"{key}_sha256"],
                f"N3B adjudication {key} seal drift")
    manifest, terminal = load_json(paths["manifest"]), load_json(paths["terminal"])
    require(manifest.get("status") == terminal.get("status")
            == expected["required_status"], "N3B adjudication status mismatch")
    require(terminal.get("manifest_sha256") == expected["manifest_sha256"],
            "N3B adjudication terminal does not bind its manifest")
    require(manifest.get("config_sha256") == expected["source_config_sha256"]
            and manifest.get("runner_sha256") == expected["source_runner_sha256"],
            "N3B adjudication source identity mismatch")
    seen = _verify_manifest_inventory(directory, manifest)
    require(str(expected["resolved_config"]) in seen,
            "N3B adjudication resolved config is absent from manifest")
    require(
        manifest.get("source_n3b_manifest_sha256")
        == EXPECTED_N3B_LIVE["manifest_sha256"],
        "N3B adjudication is not bound to the sealed N3B live campaign",
    )
    require(terminal.get("n3b_outcome_accepted") is True
            and int(terminal.get("cold_attach_passes", -1)) == 0
            and int(terminal.get("cold_attach_failures", -1)) == 3,
            "N3B adjudication did not accept the 0/3 cold-attach outcome")
    require(terminal.get("cold_achieved_pusch_snr_db_p05") is None
            and terminal.get("cold_achieved_pusch_snr_db_p50") is None
            and terminal.get("cold_achieved_pusch_snr_db_p95") is None,
            "N3B adjudication unexpectedly assigns cold achieved SNR")
    require(
        terminal.get("n3c_eligibility_status")
        == expected["n3c_eligibility_status"],
        "N3B adjudication N3C eligibility mismatch",
    )
    require(
        math.isclose(
            float(terminal.get("n3c_selected_command_db", math.nan)),
            CANDIDATE_COMMAND_DB,
        ),
        "N3B adjudication selected command mismatch",
    )
    require(terminal.get("n3c_execution_authorized") is False
            and terminal.get("n3c_executed") is False,
            "offline N3B adjudication unexpectedly authorizes or executes N3C")
    for key in (
        "target_mapping_promoted",
        "numeric_bound_promoted",
        "operational_bound_promoted",
        "connectivity_bound_promoted",
        "usable_service_bound_promoted",
    ):
        require(terminal.get(key) is False,
                f"N3B adjudication unexpectedly promoted {key}")
    proof = {
        "status": "VERIFIED_READ_ONLY_N3B_ADJUDICATION",
        "directory": str(directory),
        "manifest_sha256": n2.sha256(paths["manifest"]),
        "terminal_sha256": n2.sha256(paths["terminal"]),
        "resolved_config_sha256": n2.sha256(paths["resolved_config"]),
        "verified_output_count": len(seen),
        "n3c_selected_command_db": CANDIDATE_COMMAND_DB,
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


class RecoveryCapable(Protocol):
    def verify_recovery(
        self, baseline_count: int, receiver_baseline_count: int, *, required: bool = True,
    ) -> Mapping[str, Any]: ...


def verify_post_restore_recovery_fail_closed(
    runner: RecoveryCapable, baseline_count: int, receiver_baseline_count: int,
) -> Mapping[str, Any]:
    """Invoke the inherited recovery gate as required infrastructure evidence."""
    result = runner.verify_recovery(
        baseline_count, receiver_baseline_count, required=True,
    )
    require(result.get("passed") is True,
            "clean -50 recovery is infrastructure-invalid evidence")
    return result


def classify_repetition(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a future N3C repetition without permitting bound promotion."""
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
    delivery_pass = (
        exact_window
        and service.get("primary_99_pass") is True
        and service.get("no_one_second_outage_pass") is True
    )
    achieved_p50 = tail.get("achieved_pusch_snr_db_median")
    achieved_snr_gate = (
        achieved_p50 is not None
        and math.isfinite(float(achieved_p50))
        and abs(float(achieved_p50) - EXPECTED_ACHIEVED_PUSCH_SNR_DB)
        <= ACHIEVED_SNR_TOLERANCE_DB
    )
    recovery_required = attach_passed
    recovery_passed = recovery.get("passed") is True
    infrastructure_invalid = bool(
        base_valid and attach_passed and not recovery_passed
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
    elif infrastructure_invalid:
        outcome, valid, passed = (
            "CLEAN_RECOVERY_INFRASTRUCTURE_INVALID", False, False
        )
    elif base_valid and attach_passed and exact_window and recovery_passed:
        if delivery_pass and achieved_snr_gate:
            outcome, valid, passed = (
                "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED", True, True
            )
        elif delivery_pass:
            outcome, valid, passed = (
                "ACHIEVED_SNR_OUTSIDE_CONSISTENCY_BAND", True, False
            )
        else:
            outcome, valid, passed = "SERVICE_GATE_FAILED", True, False
    elif (
        base_valid and attach_passed and transport.get("integrity_gate") is True
        and recognized_loss and corroborated_loss and recovery_passed
    ):
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
        "authoritative_service_gate_pass": delivery_pass and recovery_passed,
        "achieved_snr_gate_pass": achieved_snr_gate,
        "expected_achieved_pusch_snr_db": EXPECTED_ACHIEVED_PUSCH_SNR_DB,
        "achieved_snr_tolerance_db": ACHIEVED_SNR_TOLERANCE_DB,
        "achieved_snr_expectation_role": "CONSISTENCY_ONLY_NOT_PROMOTED",
        "post_restore_recovery_required": recovery_required,
        "post_restore_recovery_passed": recovery_passed,
        "infrastructure_invalid": infrastructure_invalid,
        "recognized_service_loss": recognized_loss and corroborated_loss,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "operational_bound_promoted": False,
    }


class CampaignRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise ColdAttachRefinementFailure(
                f"create-only output already exists: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True)

    def write_plan(self, *, require_live: bool = False) -> list[dict[str, Any]]:
        validate_config(self.config, verify_hashes=True, require_live=require_live)
        n2.atomic_json(
            self.output_dir / self.config["output"]["resolved_config"], self.config,
        )
        rows = campaign_plan_rows(self.config)
        calibration.write_csv(
            self.output_dir / self.config["output"]["campaign_plan"], rows,
        )
        return rows

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(
            self.output_dir / self.config["output"]["campaign_summary"], summary,
        )
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path(self.config["output"]["manifest"]):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3C_") or path.name == "FAILED.json"
            ):
                continue
            files.append({
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / self.config["output"]["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3c_cold_attach_refinement_plan_manifest.v1",
            "status": status,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "runtime_executed": False,
            "socket_executed": False,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "outputs": files,
        })
        terminal = {
            **dict(summary),
            "status": status,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "operational_bound_promoted": False,
            "manifest_sha256": n2.sha256(manifest_path),
        }
        name = self.config["output"]["failure"] if status == FAILED else f"{status}.json"
        n2.atomic_json(self.output_dir / name, terminal)

    def prepare(self) -> int:
        try:
            rows = self.write_plan(require_live=False)
            n3b_proof = verify_n3b_live(
                self.config,
                proof_path=self.output_dir / "n3b_live_evidence_predecessor.json",
            )
            expectation_proof = verify_command_ladder_expectation(
                self.config,
                proof_path=self.output_dir / "command_ladder_expectation_predecessor.json",
            )
            pending = pending_adjudication(
                self.config["predecessors"]["n3b_adjudication"]
            )
            adjudication_proof = None
            if not pending:
                adjudication_proof = verify_n3b_adjudication(
                    self.config,
                    proof_path=self.output_dir / "n3b_adjudication_predecessor.json",
                )
            authority_ready = (
                self.config["authority"]["live_oai_run_authorized"] is True
                and self.config["authority"]["live_socket_execution_authorized"] is True
            )
            ready = not pending and authority_ready and adjudication_proof is not None
            status = PLAN_READY if ready else PLAN_BLOCKED
            summary = {
                "status": status,
                "runtime_executed": False,
                "socket_executed": False,
                "repetitions_planned": len(rows),
                "plan": rows,
                "n3b_live_predecessor": n3b_proof,
                "command_ladder_expectation_predecessor": expectation_proof,
                "n3b_adjudication_predecessor_pending": pending,
                "n3b_adjudication_predecessor": adjudication_proof,
                "live_authority_ready": authority_ready,
                "live_execution_blocked": not ready,
                "live_engine_status": "NOT_IMPLEMENTED_IN_PREPARE_ONLY_VERSION",
                "cold_attach_bound_evaluated": False,
                "expected_achieved_pusch_snr_db": EXPECTED_ACHIEVED_PUSCH_SNR_DB,
                "achieved_snr_expectation_role": "CONSISTENCY_ONLY_NOT_PROMOTED",
                "next": (
                    "IMPLEMENT_AND_REVIEW_DISTINCT_LIVE_VERSION"
                    if ready
                    else "SEAL_N3B_ADJUDICATION_THEN_CREATE_EXPLICIT_LIVE_VERSION"
                ),
            }
            self.manifest_terminal(status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest_terminal(FAILED, {
                "status": FAILED,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_executed": False,
                "socket_executed": False,
                "cold_attach_bound_evaluated": False,
            })
            return 1

    def execute(self) -> int:
        try:
            self.write_plan(require_live=True)
            raise ColdAttachRefinementFailure(
                "N3C live engine is not present in this PREPARE-only version"
            )
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest_terminal(FAILED, {
                "status": FAILED,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_executed": False,
                "socket_executed": False,
                "cold_attach_bound_evaluated": False,
            })
            return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode", choices=(PREPARE_ONLY, EXECUTE_LIVE), default=PREPARE_ONLY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = CampaignRunner(Path(args.config), Path(args.output_dir))
    return runner.prepare() if args.mode == PREPARE_ONLY else runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())

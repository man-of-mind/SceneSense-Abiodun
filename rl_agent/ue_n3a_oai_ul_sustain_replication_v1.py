#!/usr/bin/env python3
"""Prepare or execute the bounded UE-N3A sustained-service replication.

``PREPARE_ONLY`` is the default.  Explicit live execution alternates three
fresh-RAN repetitions at commanded -2.5 dB with three fresh-RAN adjacent
checks at -2.0 dB.  The former must sustain an exact 600-frame/60-second
candidate window; the latter must produce a cleanly observed hard service
loss.  Every repetition starts and restores at -50 dB.

This stage records replication evidence for review.  It cannot promote a
command mapping, a connectivity bound, or a usable-service bound.
"""

from __future__ import annotations

import argparse
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


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3a_oai_ul_sustain_replication_v1.json"
PREPARE_ONLY = "PREPARE_ONLY"
EXECUTE_LIVE = "EXECUTE_LIVE"
PLAN_FROZEN = "UE_N3A_SUSTAIN_REPLICATION_PLAN_FROZEN_REVIEW_REQUIRED"
CAPTURED_REVIEW_REQUIRED = "UE_N3A_SUSTAIN_REPLICATION_CAPTURED_REVIEW_REQUIRED"
UNSTABLE_REVIEW_REQUIRED = "UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED"
SUSTAIN_REP_PASSED = "UE_N3A_SUSTAINED_CANDIDATE_REPLICATION_PASSED"
HARD_LOSS_REP_CAPTURED = "UE_N3A_ADJACENT_HARD_LOSS_REPLICATION_CAPTURED"
VALID_SURPRISE_CAPTURED = "UE_N3A_VALID_SURPRISE_OUTCOME_CAPTURED"
EVIDENCE_UNCONFIRMED = "UE_N3A_REPLICATION_EVIDENCE_UNCONFIRMED"
RESTORE_FAILED = "UE_N3A_FAILED_RESTORE"
SCHEMA = "scenesense.ue_n3a_oai_ul_sustain_replication_config.v1"
AUTHORITY_BASIS = (
    "USER_REQUEST_2026-08-21_CONTINUE_LOWER_OAI_SNR_SEARCH_"
    "AFTER_CLEAN_CONTROL_PASS"
)

EXPECTED_PREDECESSORS = {
    "clean_control": {
        "directory": (
            "rl_agent/experiments/ue_n3_oai_ul_live_stage_v1/"
            "20260821_clean_control_03"
        ),
        "manifest": "manifest.json",
        "manifest_sha256": (
            "ed28aac841e34d5e0da555286e06c3ecd2545d58776f46928bbd0a8f40a4a185"
        ),
        "terminal": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED.json",
        "terminal_sha256": (
            "cab56b66541743a74d845e7f7289c512ba37d6e8526fd5fa22fa82ca9990baba"
        ),
        "resolved_config": "resolved_config.json",
        "resolved_config_sha256": (
            "a51e296b1582f15b2ed38bff2091bb17b91cce1148277506cc72374d894f11b8"
        ),
        "required_status": "UE_N3_CLEAN_RECEIVER_CONTROL_PASSED",
        "source_config_sha256": (
            "fb11c681a8c272c97e1386c26988d94f1a108ec74c1c7053cd71ca68c4a835e4"
        ),
        "source_runner_sha256": (
            "c258972b2a68d0598341eb8e422a1414c582cab1774a78b8692237ef65fb0ab3"
        ),
    },
    "command_search": {
        "directory": (
            "rl_agent/experiments/ue_n3_oai_ul_command_calibration_v1/"
            "20260821_command_search_02"
        ),
        "manifest": "manifest.json",
        "manifest_sha256": (
            "e8235586c07c5996aab17219280da35947bc8e54b05fc43ee326ee5373618f82"
        ),
        "terminal": "UE_N3_TARGET_CALIBRATION_UNRESOLVED.json",
        "terminal_sha256": (
            "cc773f849ecd5b90dc108278b3d367c746a428010ce82a65233f5cd69cce7890"
        ),
        "resolved_config": "resolved_config.json",
        "resolved_config_sha256": (
            "8d500b33156596bd78ddea2c146d98d854ceaab74ca01b95cd11f9ac661388cf"
        ),
        "required_status": "UE_N3_TARGET_CALIBRATION_UNRESOLVED",
        "source_config_sha256": (
            "04609685a6993b0ef6bbf55c6d694736ea5f8cb4715abe0ebb2b2b9a793943da"
        ),
        "source_runner_sha256": (
            "30cf1615f51c7cd0ebe4087f7b6ca66f37a563f2d5c452e474ae838a23c8878b"
        ),
    },
}


class SustainReplicationFailure(calibration.CalibrationFailure):
    """Fail-closed N3A plan, evidence, infrastructure, or outcome failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SustainReplicationFailure(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise SustainReplicationFailure(
            f"path escapes repository root: {relative}"
        ) from exc
    return path


def _runtime_seal(config: Mapping[str, Any], relative: str) -> str:
    matches = [
        str(row.get("sha256", ""))
        for row in config["runtime_seals"]
        if row.get("path") == relative
    ]
    require(matches and len(matches) == 1, f"runtime seal is not unique: {relative}")
    return matches[0]


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    require(config.get("schema") == SCHEMA, "unexpected N3A config schema")
    require(
        config.get("claim_boundary")
        == "SUSTAINED_REPLICATION_EVIDENCE_ONLY_REVIEW_REQUIRED_NO_BOUND_PROMOTION",
        "N3A claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_plan_authorized") is True,
            "offline plan authority is absent")
    require(authority.get("live_oai_run_authorized") is True,
            "frozen live OAI authority is absent")
    require(authority.get("live_socket_execution_authorized") is True,
            "frozen socket authority is absent")
    require(authority.get("live_authority_basis") == AUTHORITY_BASIS,
            "live authority basis drift")
    for key in (
        "carla_run_authorized", "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized", "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized", "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")

    predecessors = config["predecessors"]
    for name, expected in EXPECTED_PREDECESSORS.items():
        require(predecessors.get(name) == expected,
                f"pinned {name} predecessor drift")
    require(
        predecessors.get("ue_n1_bundle")
        == "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v2",
        "UE-N1 bundle drift",
    )

    campaign = config["campaign"]
    require(int(campaign["repetitions_per_condition"]) == 3,
            "N3A requires exactly three repetitions per condition")
    require(campaign["one_fresh_ran_per_repetition"] is True,
            "every repetition must use a fresh RAN")
    require(campaign["one_candidate_application_per_repetition"] is True,
            "every repetition must apply exactly one candidate")
    require(campaign["execution_order"] == "ALTERNATING_PAIRED_BY_REPETITION",
            "N3A execution order must alternate conditions")
    require(campaign["continue_after_valid_scientific_surprise"] is True,
            "valid scientific surprises must be retained across all repetitions")
    require(campaign["stop_on_invalid_or_unclean_evidence"] is True,
            "invalid or unclean evidence must stop the campaign")
    require([float(value) for value in campaign["commanded_noise_power_db"]]
            == [-2.5, -2.0], "N3A command pair drift")
    conditions = campaign["conditions"]
    require(len(conditions) == 2, "N3A requires exactly two conditions")
    sustain, loss = conditions
    require(
        sustain == {
            "condition_id": "SUSTAIN_CANDIDATE_MINUS2P5",
            "commanded_noise_power_db": -2.5,
            "expected_outcome": "SUSTAINED_SERVICE",
            "expected_achieved_pusch_snr_db": 6.0,
            "achieved_tolerance_db": 0.5,
        },
        "sustained-candidate condition drift",
    )
    require(
        loss == {
            "condition_id": "ADJACENT_HARD_LOSS_MINUS2P0",
            "commanded_noise_power_db": -2.0,
            "expected_outcome": "HARD_SERVICE_LOSS",
            "accepted_hard_loss_reasons": [
                "CURRENT_RNTI_PUSCH_SILENCE",
                "RNTI_CHANGED",
                "UE_TUNNEL_IDENTITY_LOST",
            ],
        },
        "adjacent hard-loss condition drift",
    )

    rung, traffic = config["rung"], config["traffic"]
    require(math.isclose(float(rung["clean_commanded_noise_power_db"]), -50.0),
            "each repetition must start and restore at -50 dB")
    require(math.isclose(float(rung["clean_lead_s"]), 5.0),
            "clean lead must remain 5 seconds")
    require(math.isclose(float(rung["settle_s"]), 10.0),
            "candidate settle must remain 10 seconds")
    require(math.isclose(float(rung["measured_tail_s"]), 60.0),
            "candidate measured tail must be exactly 60 seconds")
    require(math.isclose(float(rung["clean_recovery_s"]), 5.0),
            "clean recovery must remain 5 seconds")
    duration = sum(float(rung[key]) for key in (
        "clean_lead_s", "settle_s", "measured_tail_s", "clean_recovery_s"
    ))
    require(math.isclose(float(rung["service_duration_s"]), duration)
            and math.isclose(duration, 80.0), "repetition duration must be 80 seconds")
    require(math.isclose(float(traffic["fps"]), 10.0), "probe must use 10 Hz")
    require(int(rung["sender_frames"]) == round(duration * float(traffic["fps"])) == 800,
            "repetition must schedule exactly 800 frames")
    require(
        int(rung["expected_tail_frames"])
        == round(float(rung["measured_tail_s"]) * float(traffic["fps"]))
        == 600,
        "candidate tail must contain exactly 600 scheduled frames",
    )
    require(float(rung["receiver_capture_duration_s"]) > duration,
            "receiver capture must include a terminal margin")
    require(int(rung["minimum_tail_pusch_samples"]) == 1440,
            "PUSCH evidence floor drift")
    require(int(rung["minimum_tail_mcs_samples"]) == 360,
            "MCS evidence floor drift")
    require(int(rung["minimum_recovery_pusch_samples"]) >= 5,
            "clean recovery PUSCH floor is too small")
    require(int(rung["minimum_recovery_receiver_frames"]) >= 5,
            "clean recovery receiver floor is too small")
    require(math.isclose(float(rung["hard_loss_silence_s"]), 2.0),
            "hard-loss silence definition drift")
    require(int(traffic["frame_bytes"]) == 12_500
            and int(traffic["chunk_bytes"]) == 12_500,
            "matched probe must use one 12500-byte datagram")
    require(int(traffic["expected_chunks_per_frame"]) == 1,
            "probe must use one chunk per frame")

    gates = config["transport_gates"]
    require(math.isclose(float(gates["primary_complete_frame_ratio"]), 0.99),
            "candidate service gate must be 99 percent")
    require([float(value) for value in gates["sensitivity_complete_frame_ratios"]]
            == [0.95, 0.90], "sensitivity thresholds drift")
    require(int(gates["maximum_interarrival_gaps_gte_1s"]) == 0,
            "candidate window permits no one-second outage")
    require(gates["expected_source_ip"] == "192.168.70.134",
            "expected UPF-SNAT source drift")
    require(gates["required_stop_reason"] == "DURATION_COMPLETE",
            "receiver must complete its bounded capture")
    require(config["preflight"]["fail_if_carla_active"] is True,
            "CARLA gate must fail closed")
    require(config["analysis"]["direct_ul_bler_zero_fill_authorized"] is False,
            "direct UL BLER zero-fill is forbidden")

    required_runtime_paths = {
        "rl_agent/ue_n3a_oai_ul_sustain_replication_v1.py",
        "rl_agent/ue_n3_oai_ul_command_calibration_v1.py",
        "rl_agent/ue_n2_oai_ul_calibration_smoke.py",
        "rl_agent/ue_n3_structured_udp_receiver.py",
        "oai_layer_latency/carla_shaped_udp_burst_sender.py",
    }
    sealed_paths = [str(row.get("path", "")) for row in config["runtime_seals"]]
    require(len(sealed_paths) == len(set(sealed_paths)), "runtime seal paths repeat")
    require(required_runtime_paths.issubset(sealed_paths),
            "required N3A runtime seals are absent")
    require(
        _runtime_seal(config, "rl_agent/ue_n3_oai_ul_command_calibration_v1.py")
        == EXPECTED_PREDECESSORS["command_search"]["source_runner_sha256"],
        "command-calibration engine seal drift",
    )
    if verify_hashes:
        for seal in config["runtime_seals"]:
            path = resolve_repo_path(str(seal["path"]))
            require(path.is_file(), f"sealed runtime file missing: {path}")
            require(n2.sha256(path) == seal["sha256"],
                    f"runtime seal drift: {seal['path']}")


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = list(config["campaign"]["conditions"])
    for repetition_index in range(1, int(config["campaign"]["repetitions_per_condition"]) + 1):
        for condition_index, condition in enumerate(conditions):
            rows.append({
                "sequence_index": len(rows),
                "repetition_index": repetition_index,
                "condition_index": condition_index,
                "condition_id": condition["condition_id"],
                "commanded_noise_power_db": float(condition["commanded_noise_power_db"]),
                "expected_outcome": condition["expected_outcome"],
                "fresh_ran_epoch_required": True,
                "candidate_application_count": 1,
                "clean_start_and_restore_command_db": -50.0,
                "measured_tail_s": float(config["rung"]["measured_tail_s"]),
                "expected_tail_frames": int(config["rung"]["expected_tail_frames"]),
                "status": "BLOCKED_PENDING_EXPLICIT_EXECUTE_LIVE",
            })
    return rows


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
            raise SustainReplicationFailure(
                f"predecessor output escapes directory: {relative}"
            ) from exc
        require(artifact.is_file(), f"predecessor output missing: {relative}")
        require(artifact.stat().st_size == int(row.get("bytes", -1)),
                f"predecessor output size drift: {relative}")
        require(n2.sha256(artifact) == row.get("sha256"),
                f"predecessor output hash drift: {relative}")
    return seen


def verify_predecessor(
    config: Mapping[str, Any], name: str, *, proof_path: Path | None = None,
) -> dict[str, Any]:
    expected = EXPECTED_PREDECESSORS[name]
    require(config["predecessors"][name] == expected,
            f"{name} predecessor is not the frozen source")
    directory = resolve_repo_path(str(expected["directory"]))
    manifest_path = directory / str(expected["manifest"])
    terminal_path = directory / str(expected["terminal"])
    resolved_path = directory / str(expected["resolved_config"])
    for path in (manifest_path, terminal_path, resolved_path):
        require(path.is_file(), f"{name} predecessor file missing: {path}")
    require(n2.sha256(manifest_path) == expected["manifest_sha256"],
            f"{name} manifest seal drift")
    require(n2.sha256(terminal_path) == expected["terminal_sha256"],
            f"{name} terminal seal drift")
    require(n2.sha256(resolved_path) == expected["resolved_config_sha256"],
            f"{name} resolved-config seal drift")
    manifest, terminal = load_json(manifest_path), load_json(terminal_path)
    require(manifest.get("status") == expected["required_status"],
            f"{name} manifest status mismatch")
    require(terminal.get("status") == expected["required_status"],
            f"{name} terminal status mismatch")
    require(terminal.get("manifest_sha256") == expected["manifest_sha256"],
            f"{name} terminal does not bind its manifest")
    require(manifest.get("config_sha256") == expected["source_config_sha256"],
            f"{name} source config mismatch")
    require(manifest.get("runner_sha256") == expected["source_runner_sha256"],
            f"{name} source runner mismatch")
    seen = _verify_manifest_inventory(directory, manifest)
    require(str(expected["resolved_config"]) in seen,
            f"{name} resolved config is absent from manifest")

    if name == "clean_control":
        require(manifest.get("mode") == "CLEAN_RECEIVER_CONTROL",
                "clean predecessor mode mismatch")
        require(terminal.get("primary_usable_service_pass") is True,
                "clean predecessor lacks primary service pass")
        require(terminal.get("clean_restore_verified") is True,
                "clean predecessor lacks -50 restore")
        summary = load_json(directory / "summary.json")
        require(summary.get("receiver_gate", {}).get("primary_usable_service_pass") is True,
                "clean predecessor summary lacks receiver pass")
        require(summary.get("restored_to_clean_minus50") is True
                and summary.get("cleanup_clean") is True,
                "clean predecessor lacks clean restore/cleanup")
    else:
        require(terminal.get("target_mapping_promoted") is False
                and terminal.get("numeric_bound_promoted") is False,
                "command search unexpectedly promoted a result")
        require(terminal.get("cold_attach_bound_evaluated") is False,
                "command search unexpectedly evaluated cold attach")
        summary_path = directory / "campaign_summary.json"
        require("campaign_summary.json" in seen,
                "command-search campaign summary is absent from manifest")
        summary = load_json(summary_path)
        require(summary.get("status") == expected["required_status"],
                "command-search summary status mismatch")
        by_command = {
            float(row["commanded_noise_power_db"]): row
            for row in summary.get("rung_results", [])
        }
        require(set([-2.5, -2.0]).issubset(by_command),
                "command search lacks both N3A source rungs")
        candidate = by_command[-2.5]
        require(candidate.get("status") == calibration.RUNG_CAPTURED,
                "-2.5 source rung was not captured")
        require(math.isclose(float(candidate.get("achieved_pusch_snr_db_median")), 6.0),
                "-2.5 source median is not 6.0 dB")
        require(candidate.get("tail", {}).get("status") == "TAIL_ACCEPTED",
                "-2.5 source radio tail was not accepted")
        require(candidate.get("tail_service", {}).get("primary_99_pass") is True
                and candidate.get("tail_service", {}).get("no_one_second_outage_pass") is True,
                "-2.5 source exact tail lacked usable service")
        require(candidate.get("clean_restore_verified") is True,
                "-2.5 source did not restore to -50")
        adjacent = by_command[-2.0]
        require(adjacent.get("status_before_recovery_gate") == calibration.RUNG_HARD_LOSS,
                "-2.0 source did not capture hard service loss")
        require(adjacent.get("hard_loss_reason") == "CURRENT_RNTI_PUSCH_SILENCE"
                and adjacent.get("receiver_service_outage_detected") is True,
                "-2.0 source loss is not joint receiver/PUSCH silence")
        require(adjacent.get("clean_restore_verified") is True,
                "-2.0 source did not restore to -50")

    proof = {
        "status": "VERIFIED_READ_ONLY_PREDECESSOR",
        "predecessor": name,
        "directory": str(directory),
        "manifest": manifest_path.name,
        "manifest_sha256": n2.sha256(manifest_path),
        "terminal": terminal_path.name,
        "terminal_sha256": n2.sha256(terminal_path),
        "resolved_config": resolved_path.name,
        "resolved_config_sha256": n2.sha256(resolved_path),
        "source_config_sha256": manifest.get("config_sha256"),
        "source_runner_sha256": manifest.get("runner_sha256"),
        "verified_output_count": len(seen),
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def classify_repetition_summary(
    summary: Mapping[str, Any], condition: Mapping[str, Any],
) -> dict[str, Any]:
    engine_status = str(summary.get("status", ""))
    recovery = dict(summary.get("clean_recovery") or {})
    transport = dict(summary.get("transport") or {})
    common_evidence_valid = (
        summary.get("clean_restore_verified") is True
        and recovery.get("passed") is True
        and transport.get("integrity_gate") is True
        and int(summary.get("candidate_application_count", -1)) == 1
    )
    validation_errors: list[str] = []
    if summary.get("clean_restore_verified") is not True:
        validation_errors.append("CLEAN_MINUS50_RESTORE_NOT_VERIFIED")
    if recovery.get("passed") is not True:
        validation_errors.append("CLEAN_RECOVERY_NOT_VERIFIED")
    if transport.get("integrity_gate") is not True:
        validation_errors.append("WHOLE_CAPTURE_STRUCTURAL_INTEGRITY_FAILED")
    if int(summary.get("candidate_application_count", -1)) != 1:
        validation_errors.append("CANDIDATE_APPLICATION_COUNT_NOT_ONE")

    tail = dict(summary.get("tail") or {})
    service = dict(summary.get("tail_service") or {})
    exact_tail = (
        tail.get("status") == "TAIL_ACCEPTED"
        and service.get("full_nominal_window_observed") is True
        and service.get("exact_frozen_frame_set_pass") is True
        and int(service.get("required_expected_frames", -1)) == 600
        and int(service.get("expected_frames", -1)) == 600
        and service.get("integrity_gate") is True
    )
    tail_evidence_valid = (
        common_evidence_valid
        and engine_status == calibration.RUNG_CAPTURED
        and exact_tail
    )
    tail_service_pass = (
        tail_evidence_valid
        and service.get("primary_99_pass") is True
        and service.get("no_one_second_outage_pass") is True
    )

    recognized_loss_pairs = {
        calibration.RUNG_HARD_LOSS: "CURRENT_RNTI_PUSCH_SILENCE",
        calibration.RUNG_DETACHED: "UE_TUNNEL_IDENTITY_LOST",
        calibration.RUNG_IDENTITY_DISCONTINUITY: "RNTI_CHANGED",
    }
    loss_reason = summary.get("hard_loss_reason")
    loss_pair_valid = recognized_loss_pairs.get(engine_status) == loss_reason
    loss_corroborated = (
        summary.get("receiver_service_outage_detected") is True
        if loss_reason == "CURRENT_RNTI_PUSCH_SILENCE"
        else loss_reason in {"UE_TUNNEL_IDENTITY_LOST", "RNTI_CHANGED"}
    )
    hard_loss_evidence_valid = (
        common_evidence_valid
        and loss_pair_valid
        and loss_corroborated
        and summary.get("tail") is None
        and summary.get("tail_service") is None
    )

    if tail_evidence_valid:
        classified_outcome = (
            "SUSTAINED_SERVICE" if tail_service_pass else "SERVICE_GATE_FAILED"
        )
        evidence_valid = True
    elif hard_loss_evidence_valid:
        classified_outcome = "HARD_SERVICE_LOSS"
        evidence_valid = True
    else:
        classified_outcome = "EVIDENCE_UNCONFIRMED"
        evidence_valid = False
        if common_evidence_valid:
            validation_errors.append(
                "NEITHER_EXACT_TAIL_NOR_RECOGNIZED_HARD_LOSS_WAS_CONFIRMED"
            )

    achieved = tail.get("achieved_pusch_snr_db_median")
    target = condition.get("expected_achieved_pusch_snr_db")
    tolerance = condition.get("achieved_tolerance_db")
    target_pass = None
    if target is not None and tolerance is not None:
        target_pass = (
            achieved is not None
            and math.isfinite(float(achieved))
            and abs(float(achieved) - float(target)) <= float(tolerance)
        )
    if condition["expected_outcome"] == "SUSTAINED_SERVICE":
        matches_expected = (
            classified_outcome == "SUSTAINED_SERVICE" and target_pass is True
        )
    else:
        matches_expected = classified_outcome == "HARD_SERVICE_LOSS"

    mismatch_reasons: list[str] = []
    if evidence_valid and not matches_expected:
        mismatch_reasons.append(
            f"EXPECTED_{condition['expected_outcome']}_OBSERVED_{classified_outcome}"
        )
        if target_pass is False:
            mismatch_reasons.append("ACHIEVED_SNR_OUTSIDE_EXPECTED_TOLERANCE")
    return {
        "evidence_valid": evidence_valid,
        "matches_expected_outcome": matches_expected,
        # Kept as a compatibility alias for evidence aggregation.  Scientific
        # expectation matching is deliberately a separate field.
        "accepted": evidence_valid,
        "classified_outcome": classified_outcome,
        "expected_outcome": condition["expected_outcome"],
        "expected_achieved_pusch_snr_db": target,
        "achieved_tolerance_db": tolerance,
        "achieved_target_pass": target_pass,
        "exact_600_frame_tail_evidence_valid": tail_evidence_valid,
        "tail_primary_99_and_no_gap_pass": tail_service_pass,
        "whole_capture_primary_delivery_gate_applied": False,
        "recognized_hard_loss_status_reason_pair": loss_pair_valid,
        "observed_hard_loss_reason": loss_reason,
        "hard_loss_direct_or_outage_corroboration_pass": loss_corroborated,
        "joint_receiver_and_pusch_silence_pass": (
            loss_reason == "CURRENT_RNTI_PUSCH_SILENCE"
            and summary.get("receiver_service_outage_detected") is True
        ),
        "validation_errors": validation_errors,
        "expected_outcome_mismatch_reasons": mismatch_reasons,
    }


def classify_campaign_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sustain = [
        row for row in results
        if row.get("condition_id") == "SUSTAIN_CANDIDATE_MINUS2P5"
    ]
    adjacent = [
        row for row in results
        if row.get("condition_id") == "ADJACENT_HARD_LOSS_MINUS2P0"
    ]
    require(len(sustain) == len(adjacent) == 3,
            "N3A did not complete both 3/3 replication sets")
    require(all(row.get("evidence_valid_for_aggregation") for row in results),
            "N3A contains invalid repetition evidence")
    sustain_matches = sum(bool(row.get("matches_expected_outcome")) for row in sustain)
    adjacent_matches = sum(bool(row.get("matches_expected_outcome")) for row in adjacent)
    all_expected = sustain_matches == adjacent_matches == 3
    return {
        "status": CAPTURED_REVIEW_REQUIRED if all_expected else UNSTABLE_REVIEW_REQUIRED,
        "sustain_candidate_expected_matches": sustain_matches,
        "adjacent_hard_loss_expected_matches": adjacent_matches,
        "sustain_candidate_3_of_3_pass": sustain_matches == 3,
        "adjacent_hard_loss_3_of_3_captured": adjacent_matches == 3,
        "valid_mixed_outcomes_retained": not all_expected,
        "sustain_achieved_pusch_snr_db_medians": [
            row.get("tail", {}).get("achieved_pusch_snr_db_median")
            for row in sustain
            if row.get("tail") is not None
        ],
    }


class ReplicationRunner(calibration.RungRunner):
    """One N3A condition repetition in one independently created RAN epoch."""

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        *,
        condition_index: int,
        repetition_index: int,
        clean_control_proof: Mapping[str, Any],
        command_search_proof: Mapping[str, Any],
    ) -> None:
        config = load_json(config_path.resolve())
        condition = config["campaign"]["conditions"][condition_index]
        super().__init__(
            config_path,
            output_dir,
            rung_index=condition_index,
            command_db=float(condition["commanded_noise_power_db"]),
            clean_control_proof=clean_control_proof,
        )
        self.condition_index = int(condition_index)
        self.condition = dict(condition)
        self.repetition_index = int(repetition_index)
        self.command_search_proof = dict(command_search_proof)

    def verify_dependencies(self) -> None:
        validate_config(self.config, verify_hashes=True)
        observed_clean = verify_predecessor(
            self.config, "clean_control",
            proof_path=self.output_dir / "clean_control_predecessor.json",
        )
        observed_search = verify_predecessor(
            self.config, "command_search",
            proof_path=self.output_dir / "command_search_predecessor.json",
        )
        for observed, frozen, label in (
            (observed_clean, self.clean_control_proof, "clean control"),
            (observed_search, self.command_search_proof, "command search"),
        ):
            require(
                observed["manifest_sha256"] == frozen.get("manifest_sha256")
                and observed["terminal_sha256"] == frozen.get("terminal_sha256")
                and observed["resolved_config_sha256"]
                == frozen.get("resolved_config_sha256"),
                f"{label} proof changed between repetitions",
            )
        n2.atomic_json(self.output_dir / "runtime_seals.json", {
            "status": "MATCHED",
            "observed_at": n2.utc_now(),
            "files": [
                {
                    "path": seal["path"],
                    "expected_sha256": seal["sha256"],
                    "observed_sha256": n2.sha256(self.path(seal["path"])),
                }
                for seal in self.config["runtime_seals"]
            ],
        })

    def assert_carla_absent(self) -> None:
        require(self.config["preflight"]["fail_if_carla_active"] is True,
                "CARLA fail-closed gate disabled")
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,comm=,args="],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SustainReplicationFailure(
                f"CARLA detector failed closed: {exc}"
            ) from exc
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
            "process_matches": matches,
            "busy_ports": busy,
            "detector_scope": "PROCESS_COMM_AND_ARGS_PLUS_TCP_AND_UDP_PORTS",
            "checked_at": n2.utc_now(),
        }
        n2.atomic_json(self.output_dir / "carla_absent_gate.json", evidence)
        require(not matches and not busy, f"CARLA_ACTIVE_FAIL_CLOSED: {evidence}")
        self.last_carla_check_monotonic_ns = time.monotonic_ns()

    def write_manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        classification = classify_repetition_summary(summary, self.condition)
        evidence_valid = bool(classification["evidence_valid"])
        matches_expected = bool(classification["matches_expected_outcome"])
        if evidence_valid and matches_expected:
            final_status = (
                SUSTAIN_REP_PASSED
                if self.condition["expected_outcome"] == "SUSTAINED_SERVICE"
                else HARD_LOSS_REP_CAPTURED
            )
        elif evidence_valid:
            final_status = VALID_SURPRISE_CAPTURED
        elif status == calibration.RESTORE_FAILED:
            final_status = RESTORE_FAILED
        elif status == "FAILED":
            final_status = "FAILED"
        else:
            final_status = EVIDENCE_UNCONFIRMED
        augmented = {
            **dict(summary),
            "status": final_status,
            "engine_status": status,
            "condition_index": self.condition_index,
            "condition_id": self.condition["condition_id"],
            "expected_outcome": self.condition["expected_outcome"],
            "repetition_index": self.repetition_index,
            "outcome_classification": classification,
            "evidence_valid_for_aggregation": evidence_valid,
            "matches_expected_outcome": matches_expected,
            "accepted_for_replication_contract": matches_expected,
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "clean_restore_verified": self.restored,
            "candidate_application_count": self.application_count,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
        }
        summary_path = self.output_dir / self.config["output"]["repetition_summary"]
        n2.atomic_json(summary_path, augmented)
        excluded = {"manifest.json", "FAILED.json"}
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.name in excluded or path.name.startswith("UE_N3A_"):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest = {
            "schema": "scenesense.ue_n3a_sustain_repetition_manifest.v1",
            "status": final_status,
            "condition_index": self.condition_index,
            "condition_id": self.condition["condition_id"],
            "expected_outcome": self.condition["expected_outcome"],
            "repetition_index": self.repetition_index,
            "commanded_noise_power_db": self.command_db,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "ran_epoch_id": self.ran_epoch_id,
            "control_session_id": self.control_session_id,
            "candidate_application_count": self.application_count,
            "clean_control_manifest_sha256": self.clean_control_proof["manifest_sha256"],
            "command_search_manifest_sha256": self.command_search_proof["manifest_sha256"],
            "evidence_valid_for_aggregation": evidence_valid,
            "matches_expected_outcome": matches_expected,
            "accepted_for_replication_contract": matches_expected,
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
            "outputs": files,
        }
        manifest_path = self.output_dir / self.config["output"]["manifest"]
        n2.atomic_json(manifest_path, manifest)
        terminal = {
            **augmented,
            "manifest_sha256": n2.sha256(manifest_path),
        }
        terminal_name = (
            self.config["output"]["failure"]
            if final_status == "FAILED"
            else f"{final_status}.json"
        )
        n2.atomic_json(self.output_dir / terminal_name, terminal)

    def run(self) -> int:
        # The inherited engine owns the tested attach, actuator, exact frozen
        # tail, service-loss, restore/recovery, receiver, and cleanup sequence.
        super().run()
        summary_path = self.output_dir / self.config["output"]["repetition_summary"]
        if not summary_path.is_file():
            return 1
        return 0 if load_json(summary_path).get("evidence_valid_for_aggregation") else 1


def verify_repetition_evidence(
    directory: Path,
    *,
    plan_row: Mapping[str, Any],
    config_sha256: str,
    runner_sha256: str,
    engine_runner_sha256: str,
) -> dict[str, Any]:
    matching_status = (
        SUSTAIN_REP_PASSED
        if plan_row["expected_outcome"] == "SUSTAINED_SERVICE"
        else HARD_LOSS_REP_CAPTURED
    )
    manifest_path = directory / "manifest.json"
    summary_path = directory / "repetition_summary.json"
    require(manifest_path.is_file() and summary_path.is_file(),
            f"accepted repetition evidence is incomplete: {directory}")
    manifest, summary = load_json(manifest_path), load_json(summary_path)
    observed_status = str(summary.get("status", ""))
    require(observed_status in {matching_status, VALID_SURPRISE_CAPTURED},
            f"repetition status is not valid scientific evidence: {observed_status}")
    terminal_path = directory / f"{observed_status}.json"
    require(terminal_path.is_file(), "repetition terminal is absent")
    terminal = load_json(terminal_path)
    manifest_hash = n2.sha256(manifest_path)
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"), (summary, "summary")):
        require(payload.get("status") == observed_status,
                f"repetition {label} status mismatch")
        require(payload.get("condition_id") == plan_row["condition_id"],
                f"repetition {label} condition mismatch")
        require(int(payload.get("repetition_index", -1)) == int(plan_row["repetition_index"]),
                f"repetition {label} index mismatch")
        require(math.isclose(float(payload.get("commanded_noise_power_db", math.nan)),
                             float(plan_row["commanded_noise_power_db"]), abs_tol=1e-9),
                f"repetition {label} command mismatch")
        require(payload.get("target_mapping_promoted") is False
                and payload.get("numeric_bound_promoted") is False
                and payload.get("connectivity_bound_promoted") is False
                and payload.get("usable_service_bound_promoted") is False,
                f"repetition {label} unexpectedly promoted a result")
    require(terminal.get("manifest_sha256") == manifest_hash,
            "repetition terminal/manifest hash mismatch")
    require(manifest.get("config_sha256") == config_sha256,
            "repetition config seal mismatch")
    require(manifest.get("runner_sha256") == runner_sha256,
            "repetition runner seal mismatch")
    require(manifest.get("engine_runner_sha256") == engine_runner_sha256,
            "repetition engine seal mismatch")
    require(manifest.get("candidate_application_count") == 1
            and terminal.get("candidate_application_count") == 1,
            "repetition did not apply exactly one candidate")
    require(terminal.get("clean_restore_verified") is True
            and terminal.get("evidence_valid_for_aggregation") is True,
            "repetition lacks valid restore/outcome evidence")
    matches_expected = observed_status == matching_status
    require(terminal.get("matches_expected_outcome") is matches_expected,
            "repetition expectation classification mismatch")
    _verify_manifest_inventory(directory, manifest)
    cleanup = load_json(directory / "cleanup_report.json")
    require(cleanup.get("clean") is True and not cleanup.get("errors"),
            "repetition cleanup is not clean")
    ran_epoch = str(manifest.get("ran_epoch_id", ""))
    control_session = str(manifest.get("control_session_id", ""))
    require(ran_epoch and control_session and ran_epoch != control_session,
            "repetition lacks distinct RAN/control identities")
    return {
        "status": "VERIFIED_N3A_REPETITION_EVIDENCE",
        "directory": str(directory),
        "condition_id": plan_row["condition_id"],
        "repetition_index": int(plan_row["repetition_index"]),
        "commanded_noise_power_db": float(plan_row["commanded_noise_power_db"]),
        "ran_epoch_id": ran_epoch,
        "control_session_id": control_session,
        "manifest_sha256": manifest_hash,
        "terminal_sha256": n2.sha256(terminal_path),
        "summary_sha256": n2.sha256(summary_path),
        "matches_expected_outcome": matches_expected,
        "observed_status": observed_status,
    }


class CampaignRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists():
            raise SustainReplicationFailure(
                f"create-only output already exists: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True)

    def write_plan(self) -> list[dict[str, Any]]:
        validate_config(self.config, verify_hashes=True)
        n2.atomic_json(self.output_dir / self.config["output"]["resolved_config"], self.config)
        rows = campaign_plan_rows(self.config)
        calibration.write_csv(
            self.output_dir / self.config["output"]["campaign_plan"], rows
        )
        return rows

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(
            self.output_dir / self.config["output"]["campaign_summary"], summary
        )
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path(self.config["output"]["manifest"]):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3A_") or path.name == "FAILED.json"
            ):
                continue
            files.append({
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / self.config["output"]["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3a_sustain_replication_campaign_manifest.v1",
            "status": status,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "engine_runner_sha256": n2.sha256(Path(calibration.__file__).resolve()),
            "review_required": True,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "cold_attach_bound_evaluated": False,
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
            "cold_attach_bound_evaluated": False,
            "manifest_sha256": n2.sha256(manifest_path),
        }
        terminal_name = (
            self.config["output"]["failure"]
            if status == "FAILED"
            else f"{status}.json"
        )
        n2.atomic_json(self.output_dir / terminal_name, terminal)

    def prepare(self) -> int:
        try:
            rows = self.write_plan()
            summary = {
                "status": PLAN_FROZEN,
                "runtime_executed": False,
                "socket_executed": False,
                "repetitions_planned": len(rows),
                "execution_order": self.config["campaign"]["execution_order"],
                "plan": rows,
                "review_required": True,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
                "next": "REVIEW_THEN_EXPLICIT_EXECUTE_LIVE",
            }
            self.manifest_terminal(PLAN_FROZEN, summary)
            print(json.dumps({"output_dir": str(self.output_dir), "status": PLAN_FROZEN},
                             sort_keys=True))
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_executed": False,
                "socket_executed": False,
                "review_required": True,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
            }
            self.manifest_terminal("FAILED", failure)
            print(json.dumps({"output_dir": str(self.output_dir), **failure}, sort_keys=True),
                  file=sys.stderr)
            return 1

    def execute(self) -> int:
        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        proofs: list[dict[str, Any]] = []
        clean_proof: dict[str, Any] | None = None
        search_proof: dict[str, Any] | None = None
        ran_epochs: set[str] = set()
        control_sessions: set[str] = set()
        previous_handlers: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            raise SustainReplicationFailure(
                f"received termination signal {signal.Signals(signum).name}"
            )

        for caught in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[caught] = signal.getsignal(caught)
            signal.signal(caught, terminate)
        try:
            rows = self.write_plan()
            authority = self.config["authority"]
            require(authority["live_oai_run_authorized"] is True
                    and authority["live_socket_execution_authorized"] is True
                    and authority["live_authority_basis"] == AUTHORITY_BASIS,
                    "exact N3A live authority is absent")
            clean_proof = verify_predecessor(
                self.config, "clean_control",
                proof_path=self.output_dir / "clean_control_predecessor.json",
            )
            search_proof = verify_predecessor(
                self.config, "command_search",
                proof_path=self.output_dir / "command_search_predecessor.json",
            )
            for row in rows:
                label = str(row["commanded_noise_power_db"]).replace(
                    "-", "minus"
                ).replace(".", "p")
                directory = (
                    self.output_dir / "repetitions"
                    / f"sequence_{int(row['sequence_index']):02d}_rep_"
                    f"{int(row['repetition_index']):02d}_{label}"
                )
                runner = ReplicationRunner(
                    self.config_path,
                    directory,
                    condition_index=int(row["condition_index"]),
                    repetition_index=int(row["repetition_index"]),
                    clean_control_proof=clean_proof,
                    command_search_proof=search_proof,
                )
                rc = runner.run()
                summary_path = directory / self.config["output"]["repetition_summary"]
                require(summary_path.is_file(),
                        f"repetition summary is absent: {directory}")
                result = load_json(summary_path)
                results.append(result)
                if rc != 0:
                    raise SustainReplicationFailure(
                        f"unexpected repetition outcome at sequence {row['sequence_index']}: "
                        f"{result.get('status')}"
                    )
                proof = verify_repetition_evidence(
                    directory,
                    plan_row=row,
                    config_sha256=n2.sha256(self.config_path),
                    runner_sha256=n2.sha256(Path(__file__).resolve()),
                    engine_runner_sha256=n2.sha256(Path(calibration.__file__).resolve()),
                )
                require(proof["ran_epoch_id"] not in ran_epochs,
                        "fresh RAN epoch identity was reused")
                require(proof["control_session_id"] not in control_sessions,
                        "control-session identity was reused")
                ran_epochs.add(proof["ran_epoch_id"])
                control_sessions.add(proof["control_session_id"])
                proofs.append(proof)

            aggregation = classify_campaign_results(results)
            final_status = str(aggregation["status"])
            summary = {
                "status": final_status,
                "runtime_executed": True,
                "socket_executed": True,
                "repetitions_planned": len(rows),
                "repetitions_executed": len(results),
                "fresh_ran_epoch_count": len(ran_epochs),
                "unique_control_session_count": len(control_sessions),
                "execution_order": self.config["campaign"]["execution_order"],
                **aggregation,
                "results": results,
                "repetition_evidence": proofs,
                "clean_control_predecessor": clean_proof,
                "command_search_predecessor": search_proof,
                "review_required": True,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
                "next": "REVIEW_N3A_EVIDENCE_BEFORE_ANY_BOUND_OR_COLD_ATTACH_CLAIM",
            }
            self.manifest_terminal(final_status, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repetitions_planned": len(rows),
                "repetitions_executed": len(results),
                "results": results,
                "repetition_evidence": proofs,
                "clean_control_predecessor": clean_proof,
                "command_search_predecessor": search_proof,
                "review_required": True,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
                "cold_attach_bound_evaluated": False,
            }
            self.manifest_terminal("FAILED", failure)
            return 1
        finally:
            for caught, previous in previous_handlers.items():
                signal.signal(caught, previous)


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

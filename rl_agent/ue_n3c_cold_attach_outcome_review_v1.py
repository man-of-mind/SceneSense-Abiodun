#!/usr/bin/env python3
"""Create-only offline adjudication of the sealed UE-N3C live campaign.

The tool verifies the complete campaign and every nested repetition inventory,
preserves the mixed 2/3 confirmation result, and may make only the adjacent
stronger RFsim command (-3.5 dB) eligible for a separately authorized N3D
experiment.  It never opens sockets, starts OAI/CARLA, or promotes a mapping or
network bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3c_cold_attach_outcome_review_v1.json"
OUTPUT_ROOT = ROOT / "rl_agent/experiments/ue_n3c_cold_attach_outcome_review_v1"
SCHEMA = "scenesense.ue_n3c_cold_attach_outcome_review_config.v1"
SUCCESS = "UE_N3C_OUTCOME_ADJUDICATED_N3D_ELIGIBLE_REVIEW_REQUIRED"
UNRESOLVED = "UE_N3C_OUTCOME_ADJUDICATION_UNRESOLVED_REVIEW_REQUIRED"

PASS_STATUS = "UE_N3C_COLD_ATTACH_SERVICE_REPETITION_PASSED"
NONCONFIRM_STATUS = (
    "UE_N3C_COLD_ATTACH_REPETITION_VALID_OPERATIONAL_SCREEN_NONCONFIRMATION"
)
CAMPAIGN_STATUS = "UE_N3C_COLD_ATTACH_REFINEMENT_NOT_3_OF_3_REVIEW_REQUIRED"
NONCONFIRM_ROLE = (
    "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION_NOT_CANDIDATE_CAUSAL_OR_"
    "PHYSICAL_BOUNDARY_PROOF"
)

OUTPUT = {
    "resolved_config": "resolved_config.json",
    "source_verification": "source_evidence_verification.json",
    "adjudications_json": "per_repetition_adjudication.json",
    "adjudications_csv": "per_repetition_adjudication.csv",
    "summary": "n3c_outcome_review.json",
    "report": "REPORT.md",
    "manifest": "manifest.json",
    "failure_context": "failure_context.json",
    "failure": "FAILED.json",
}

SOURCE = {
    "directory": (
        "rl_agent/experiments/ue_n3c_oai_ul_cold_attach_refinement_live_v1/"
        "20260821_live_01"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "7aaa76319d7fdfbe82082a89b514da2b7a3071d0f3b1b52f5e13ed32cb0c6f3b"
    ),
    "terminal": f"{CAMPAIGN_STATUS}.json",
    "terminal_sha256": (
        "afa36b3f486c816924114de441364a8aaf06f39e76d9273c3eb3646e3e3877a2"
    ),
    "campaign_summary": "campaign_summary.json",
    "campaign_summary_sha256": (
        "41d50ef8c2701a9dcb29ccf1f8a152ab300b5cc337210e3d13e3734f55e677c6"
    ),
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": (
        "23ec2a7231edecf97a160162b6927e8f7b1ba1f573064f6f1e8a00ed07da4d77"
    ),
    "required_status": CAMPAIGN_STATUS,
    "source_config_sha256": (
        "13ea9536bd5492a98d6f42b61ed5371d6e02cd3bd47381f69fd877797209fc4d"
    ),
    "source_runner_sha256": (
        "d3f62a4b116786f7e684090a02be41c3a93d6b9013fca2da9bb8e28636b45f7d"
    ),
    "source_prepare_runner_sha256": (
        "5eaf61f41114fe51ab8219520c22b859636083c24fc9da94c831b42707216d43"
    ),
    "source_n3b_runner_sha256": (
        "3cb0b1e975ae58f9fffc510ec20622bded9237848498f5d3c13bfd0b0f3b57f7"
    ),
    "source_engine_runner_sha256": (
        "30cf1615f51c7cd0ebe4087f7b6ca66f37a563f2d5c452e474ae838a23c8878b"
    ),
    "expected_root_output_count": 251,
}

COMMAND_LADDER = {
    "directory": (
        "rl_agent/experiments/ue_n3_oai_ul_command_calibration_v1/"
        "20260821_command_search_02"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "e8235586c07c5996aab17219280da35947bc8e54b05fc43ee326ee5373618f82"
    ),
    "required_status": "UE_N3_TARGET_CALIBRATION_UNRESOLVED",
    "expected_root_output_count": 529,
    "rung_directory": "rungs/rung_01_minus3p5",
    "rung_manifest": "manifest.json",
    "rung_manifest_sha256": (
        "45d9bebea2e681f99c2a22e1a8092ef26290849806b7dcf306be131dc5d36c31"
    ),
    "rung_terminal": "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY.json",
    "rung_terminal_sha256": (
        "e25435257b4d616e6da41d960eb50789b1ae21ab816f78d8bfecab0f36927a11"
    ),
    "rung_summary": "rung_summary.json",
    "rung_summary_sha256": (
        "e7dcca3082a7bea7860d641b5496544fd10d3b61690cb8e08e7e11403011758b"
    ),
    "required_rung_status": "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY",
    "expected_rung_output_count": 103,
    "commanded_noise_power_db": -3.5,
    "observed_hot_pusch_snr_db_p05": 7.0,
    "observed_hot_pusch_snr_db_p50": 7.5,
    "observed_hot_pusch_snr_db_p95": 7.5,
    "provenance_role": (
        "EXPECTATION_ONLY_ALREADY_ATTACHED_HOT_RUNG_NOT_COLD_ATTAINED"
    ),
}

EXPECTED_REPETITIONS = (
    (
        "rep_01_minus3p0_cold", 1, PASS_STATUS, 109,
        "8a210b13eec1113722368b980255ca95858747fecbfc0e1eb10b932793e9a762",
        "e1851a3977c5091435fdf08614a6fe6134e978f5af5c67341c1fbeb7681bc105",
        "5104733789024de8ff9e3ae3f601815658b8e52876f96c4f838ab36b8272ec8d",
    ),
    (
        "rep_02_minus3p0_cold", 2, NONCONFIRM_STATUS, 21,
        "d9d66341f4f3ee813778e001781467a8a5b80a48cf9143a629d21060848bbe95",
        "b43bb259f26f64d0cdd0aff0b72c4b02eaebb99af53ae9a52020a45efb615a10",
        "581253b46d5afda8a00ddedbfe441717107de64921e5de68e01676fb31749c08",
    ),
    (
        "rep_03_minus3p0_cold", 3, PASS_STATUS, 109,
        "00c1c83bbb55d83f1f4c1072202f29d5d1329ee2ef6814ea40a4770bd2d3c9eb",
        "35cead22ad4c08d3059b1d367d4363ca1403d76b47c89d65cdadb69f61d4d7bd",
        "48a58f5edd1d6a354c2cb83faa6f4c371b46b97a0dba02217570eb38c1b7892a",
    ),
)

PROMOTION_KEYS = (
    "target_mapping_promoted",
    "numeric_bound_promoted",
    "operational_bound_promoted",
    "connectivity_bound_promoted",
    "usable_service_bound_promoted",
)


class ReviewFailure(RuntimeError):
    """Raised when a seal, contract, or evidence condition is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_text(path: Path, text: str) -> None:
    """Write one immutable leaf; refuse to replace any existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_new_json(path: Path, value: Any) -> None:
    write_new_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ReviewFailure(f"path escapes repository root: {relative}") from exc
    return path


def _all_promotions_false(payload: Mapping[str, Any], label: str) -> None:
    for key in PROMOTION_KEYS:
        require(payload.get(key, False) is False,
                f"{label} unexpectedly promotes {key}")


def verify_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    allowed_uninventoried: Sequence[str] = (),
) -> dict[str, str]:
    """Hash every manifest row and require exact directory completeness."""
    rows = list(manifest.get("outputs", []))
    require(rows, f"manifest has no output inventory: {directory}")
    root = directory.resolve()
    observed: dict[str, str] = {}
    for row in rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in observed,
                f"blank or duplicate output path: {relative!r}")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ReviewFailure(f"manifest path escapes source: {relative}") from exc
        require(artifact.is_file(), f"manifest output missing: {relative}")
        require(artifact.stat().st_size == int(row.get("bytes", -1)),
                f"manifest output size drift: {relative}")
        digest = sha256(artifact)
        require(digest == row.get("sha256"),
                f"manifest output hash drift: {relative}")
        observed[relative] = digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*") if path.is_file()
    }
    expected = set(observed) | set(allowed_uninventoried)
    require(
        actual == expected,
        "manifest inventory is not directory-complete: "
        f"unexpected={sorted(actual - expected)} missing={sorted(expected - actual)}",
    )
    return observed


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    require(config.get("schema") == SCHEMA, "unexpected review config schema")
    require(config.get("experiment_id") == "ue_n3c_cold_attach_outcome_review_v1",
            "unexpected experiment identity")
    require(
        config.get("claim_boundary")
        == "OFFLINE_N3C_MIXED_OUTCOME_REVIEW_ONLY_N3D_ELIGIBILITY_NO_EXECUTION_NO_PROMOTION",
        "review claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_review_authorized") is True,
            "offline review authority is absent")
    forbidden = (
        "oai_run_authorized", "socket_execution_authorized",
        "carla_run_authorized", "n3d_execution_authorized",
        "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized",
        "operational_bound_promotion_authorized",
        "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized",
        "policy_training_authorized",
    )
    for key in forbidden:
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    require(config.get("source") == SOURCE, "sealed N3C live source drift")
    require(config.get("command_ladder") == COMMAND_LADDER,
            "command-ladder provenance drift")
    contract = config["contract"]
    require(int(contract["repetitions"]) == 3, "three repetitions are required")
    require(math.isclose(float(contract["tested_commanded_noise_power_db"]), -3.0),
            "tested N3C command drift")
    require(int(contract["expected_joint_confirmation_passes"]) == 2,
            "expected N3C joint-pass count drift")
    require(int(contract["expected_operational_nonconfirmations"]) == 1,
            "expected N3C nonconfirmation count drift")
    require(int(contract["required_floor_confirmation_passes"]) == 3,
            "3/3 floor rule drift")
    require(contract["failed_repetition_cold_snr_must_be_null"] is True,
            "failure cold-SNR null rule removed")
    require(contract["operational_nonconfirmation_role"] == NONCONFIRM_ROLE,
            "operational-screen evidence role drift")
    require(contract["mixed_result_does_not_confirm_floor"] is True,
            "mixed-result claim boundary removed")
    require(math.isclose(float(contract["n3d_selected_command_db"]), -3.5),
            "N3D adjacent candidate drift")
    require(contract["n3d_selection_is_eligibility_only"] is True,
            "N3D eligibility-only scope removed")
    require(contract["separate_live_authority_required"] is True,
            "separate N3D live-authority requirement removed")
    require(
        contract["n3d_command_relation"]
        == "ADJACENT_STRONGER_RFSIM_COMMAND_MINUS0P5_DB",
        "N3D command relation drift",
    )
    require(math.isclose(
        float(contract["n3d_selected_command_db"]),
        float(contract["tested_commanded_noise_power_db"]) - 0.5,
        abs_tol=1e-9,
    ), "N3D candidate is not exactly the adjacent stronger command")
    rows = config["expected_repetitions"]
    observed = tuple(
        (
            row["directory"], int(row["repetition_index"]), row["status"],
            int(row["manifest_output_count"]), row["manifest_sha256"],
            row["terminal_sha256"], row["summary_sha256"],
        )
        for row in rows
    )
    require(observed == EXPECTED_REPETITIONS, "expected repetition seals drift")
    require(config.get("output") == OUTPUT, "output contract drift")
    seals = config["runtime_seals"]
    require(
        len(seals) == 1
        and seals[0]["path"] == "rl_agent/ue_n3c_cold_attach_outcome_review_v1.py",
        "review runtime seal set drift",
    )
    if verify_hashes:
        runner = resolve_repo_path(str(seals[0]["path"]))
        require(runner.is_file() and sha256(runner) == seals[0]["sha256"],
                "review runtime seal drift")


def _startup_integrity_ok(
    summary: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    startup: Mapping[str, Any],
    integrity: Mapping[str, Any],
    *,
    repetition_index: int,
    command_db: float,
) -> bool:
    models = dict(startup.get("models") or {})
    ue = dict(models.get("rfsimu_channel_ue0") or {})
    enb0 = dict(models.get("rfsimu_channel_enB0") or {})
    enb1 = dict(models.get("rfsimu_channel_enB1") or {})
    return all((
        int(summary.get("repetition_index", -1)) == repetition_index,
        math.isclose(float(summary.get("commanded_noise_power_db", math.nan)),
                     command_db, abs_tol=1e-9),
        summary.get("evidence_valid_for_aggregation") is True,
        summary.get("candidate_baked_config_verified") is True,
        summary.get("startup_channel_runtime_verified") is True,
        int(summary.get("candidate_application_count", -1)) == 0,
        int(summary.get("restore_application_count", -1)) == 1,
        summary.get("clean_restore_verified") is True,
        summary.get("cleanup_clean") is True,
        summary.get("source_oai_configs_unchanged") is True,
        summary.get("live_execution_attempted") is True,
        summary.get("runtime_executed") is True,
        summary.get("socket_executed") is True,
        summary.get("infrastructure_invalid") is False,
        all(summary.get(key, False) is False for key in PROMOTION_KEYS),
        startup.get("status") == "PASSED",
        startup.get("startup_channel_runtime_verified") is True,
        int(startup.get("candidate_application_count", -1)) == 0,
        math.isclose(float(ue.get("noise_power_db", math.nan)), command_db,
                     abs_tol=1e-9),
        math.isclose(float(enb0.get("noise_power_db", math.nan)), -50.0,
                     abs_tol=1e-9),
        math.isclose(float(enb1.get("noise_power_db", math.nan)), -50.0,
                     abs_tol=1e-9),
        cleanup.get("clean") is True,
        not cleanup.get("errors"),
        integrity.get("status") == "UNCHANGED",
        integrity.get("unchanged") is True,
        integrity.get("source_paths_written") is False,
        integrity.get("before") == integrity.get("after"),
    ))


def _valid_joint_pass(
    summary: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    startup: Mapping[str, Any],
    integrity: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    common = _startup_integrity_ok(
        summary, cleanup, startup, integrity,
        repetition_index=int(summary.get("repetition_index", -1)),
        command_db=float(contract["tested_commanded_noise_power_db"]),
    )
    attach = dict(summary.get("attach_gate") or {})
    tail = dict(summary.get("service_tail") or {})
    window = dict(summary.get("service_window") or {})
    transport = dict(summary.get("transport") or {})
    sender = dict(transport.get("sender_completion") or {})
    recovery = dict(summary.get("clean_recovery") or {})
    outcome = dict(summary.get("outcome_classification") or {})
    return common and all((
        summary.get("status") == PASS_STATUS,
        summary.get("engine_status") == "COLD_ATTACH_SERVICE_WINDOW_CAPTURED",
        summary.get("classified_outcome")
        == "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED",
        summary.get("clean_attach_failure_evidence") is False,
        summary.get("cold_attach_gate_pass") is True,
        summary.get("authoritative_service_gate_pass") is True,
        summary.get("joint_candidate_confirmation_pass") is True,
        summary.get("exact_600_frame_service_evidence") is True,
        summary.get("achieved_snr_gate_pass") is True,
        summary.get("achieved_snr_expectation_role")
        == "CONSISTENCY_ONLY_NOT_PROMOTED",
        summary.get("attach_failure_evidence_role") is None,
        summary.get("candidate_causal_attach_failure_confirmed") is False,
        summary.get("post_restore_recovery_required") is True,
        summary.get("post_restore_recovery_passed") is True,
        summary.get("review_before_next_action_required") is False,
        attach.get("passed") is True,
        attach.get("status") == "COLD_ATTACH_PDU_EXT_DN_GATE_PASSED",
        attach.get("ext_dn_ping_pass") is True,
        attach.get("pdu_session_evidence") == "SINGLE_OAI_TUNNEL_IPV4",
        int(attach.get("timeout_s", -1)) == 180,
        bool(attach.get("discovered_ipv4")),
        tail.get("status") == "TAIL_ACCEPTED",
        tail.get("mcs_seals_ok") is True,
        int(tail.get("pusch_samples", -1)) >= 1440,
        int(tail.get("mcs_samples", -1)) >= 360,
        tail.get("observed_rntis") == [tail.get("expected_rnti")],
        math.isclose(float(tail.get("achieved_pusch_snr_db_p05", math.nan)),
                     6.0, abs_tol=1e-9),
        math.isclose(float(tail.get("achieved_pusch_snr_db_median", math.nan)),
                     6.5, abs_tol=1e-9),
        math.isclose(float(tail.get("achieved_pusch_snr_db_p95", math.nan)),
                     7.0, abs_tol=1e-9),
        window.get("integrity_gate") is True,
        int(window.get("expected_frames", -1)) == 600,
        int(window.get("received_frames", -1)) >= 594,
        window.get("primary_99_pass") is True,
        window.get("no_one_second_outage_pass") is True,
        window.get("full_nominal_window_observed") is True,
        window.get("exact_frozen_frame_set_pass") is True,
        transport.get("integrity_gate") is True,
        transport.get("primary_99_pass") is True,
        transport.get("no_one_second_outage_pass") is True,
        transport.get("source_isolated") is True,
        sender.get("complete") is True,
        sender.get("bounded_wait_timed_out") is False,
        int(sender.get("process_returncode", -1)) == 0,
        int(sender.get("rows", -1)) == 700,
        int(sender.get("unique_frames", -1)) == 700,
        sender.get("missing_frames") == [],
        sender.get("outside_expected_frames") == [],
        recovery.get("required") is True,
        recovery.get("passed") is True,
        recovery.get("application_delivery_passed") is True,
        recovery.get("radio_recovery_passed") is True,
        recovery.get("tunnel_recovered") is True,
        outcome.get("classified_outcome")
        == "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED",
        outcome.get("joint_candidate_confirmation_pass") is True,
        outcome.get("evidence_valid_for_aggregation") is True,
        outcome.get("review_before_next_action_required") is False,
        all(outcome.get(key, False) is False for key in PROMOTION_KEYS),
    ))


def _valid_operational_nonconfirmation(
    summary: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    startup: Mapping[str, Any],
    integrity: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    common = _startup_integrity_ok(
        summary, cleanup, startup, integrity,
        repetition_index=int(summary.get("repetition_index", -1)),
        command_db=float(contract["tested_commanded_noise_power_db"]),
    )
    attach = dict(summary.get("attach_gate") or {})
    recovery = dict(summary.get("clean_recovery") or {})
    outcome = dict(summary.get("outcome_classification") or {})
    cold_values = (
        summary.get("observed_cold_achieved_pusch_snr_db_p05"),
        summary.get("observed_cold_achieved_pusch_snr_db_p50"),
        summary.get("observed_cold_achieved_pusch_snr_db_p95"),
    )
    return common and all((
        summary.get("status") == NONCONFIRM_STATUS,
        summary.get("engine_status") == NONCONFIRM_STATUS,
        summary.get("classified_outcome") == "COLD_ATTACH_FAILED",
        summary.get("clean_attach_failure_evidence") is True,
        summary.get("cold_attach_gate_pass") is False,
        summary.get("authoritative_service_gate_pass") is False,
        summary.get("joint_candidate_confirmation_pass") is False,
        summary.get("exact_600_frame_service_evidence") is False,
        summary.get("achieved_snr_gate_pass") is False,
        summary.get("service_tail") is None,
        summary.get("service_window") is None,
        summary.get("transport") is None,
        all(value is None for value in cold_values),
        summary.get("attach_failure_evidence_role") == NONCONFIRM_ROLE,
        summary.get("candidate_causal_attach_failure_confirmed") is False,
        summary.get("post_restore_clean_reattach_evaluated") is False,
        summary.get("post_restore_recovery_required") is False,
        summary.get("post_restore_recovery_passed") is False,
        summary.get("review_before_next_action_required") is True,
        attach.get("passed") is False,
        attach.get("status") == "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED",
        int(attach.get("timeout_s", -1)) == 180,
        float(attach.get("duration_s", 0.0)) >= 180.0,
        attach.get("observed_ipv4") == [],
        attach.get("core_ready_at_terminal") is True,
        attach.get("ran_processes_alive_at_terminal") is True,
        recovery.get("status") == "NOT_APPLICABLE_NO_PDU_SESSION",
        recovery.get("passed") is None,
        outcome.get("classified_outcome") == "COLD_ATTACH_FAILED",
        outcome.get("clean_attach_failure_evidence") is True,
        outcome.get("evidence_valid_for_aggregation") is True,
        outcome.get("attach_failure_evidence_role") == NONCONFIRM_ROLE,
        outcome.get("candidate_causal_attach_failure_confirmed") is False,
        outcome.get("review_before_next_action_required") is True,
        all(outcome.get(key, False) is False for key in PROMOTION_KEYS),
    ))


def _verify_command_ladder(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["command_ladder"]
    directory = resolve_repo_path(str(source["directory"]))
    manifest_path = directory / source["manifest"]
    require(manifest_path.is_file() and sha256(manifest_path) == source["manifest_sha256"],
            "command-ladder campaign manifest seal drift")
    manifest = load_json(manifest_path)
    require(manifest.get("schema")
            == "scenesense.ue_n3_command_calibration_campaign_manifest.v1",
            "command-ladder manifest schema mismatch")
    require(manifest.get("status") == source["required_status"],
            "command-ladder status mismatch")
    _all_promotions_false(manifest, "command-ladder campaign")
    root_inventory = verify_inventory(
        directory, manifest,
        allowed_uninventoried=(source["manifest"], f"{source['required_status']}.json"),
    )
    require(len(root_inventory) == int(source["expected_root_output_count"]),
            "command-ladder root output count drift")
    rung_dir = directory / source["rung_directory"]
    rung_manifest_path = rung_dir / source["rung_manifest"]
    rung_terminal_path = rung_dir / source["rung_terminal"]
    rung_summary_path = rung_dir / source["rung_summary"]
    for path, expected, label in (
        (rung_manifest_path, source["rung_manifest_sha256"], "manifest"),
        (rung_terminal_path, source["rung_terminal_sha256"], "terminal"),
        (rung_summary_path, source["rung_summary_sha256"], "summary"),
    ):
        require(path.is_file() and sha256(path) == expected,
                f"-3.5 rung {label} seal drift")
    rung_manifest = load_json(rung_manifest_path)
    require(rung_manifest.get("schema")
            == "scenesense.ue_n3_command_calibration_rung_manifest.v1",
            "-3.5 rung manifest schema mismatch")
    require(rung_manifest.get("status") == source["required_rung_status"],
            "-3.5 rung status mismatch")
    _all_promotions_false(rung_manifest, "-3.5 rung")
    rung_inventory = verify_inventory(
        rung_dir, rung_manifest,
        allowed_uninventoried=(source["rung_manifest"], source["rung_terminal"]),
    )
    require(len(rung_inventory) == int(source["expected_rung_output_count"]),
            "-3.5 rung output count drift")
    for path in rung_dir.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(directory))
            require(root_inventory.get(relative) == sha256(path),
                    f"command-ladder root does not bind {relative}")
    summary = load_json(rung_summary_path)
    require(summary.get("status") == source["required_rung_status"],
            "-3.5 rung summary status mismatch")
    require(math.isclose(float(summary.get("commanded_noise_power_db", math.nan)),
                         float(source["commanded_noise_power_db"]), abs_tol=1e-9),
            "-3.5 rung command mismatch")
    tail = dict(summary.get("tail") or {})
    require(tail.get("status") == "TAIL_ACCEPTED"
            and tail.get("mcs_seals_ok") is True,
            "-3.5 hot rung tail is not accepted")
    for key, expected in (
        ("achieved_pusch_snr_db_p05", source["observed_hot_pusch_snr_db_p05"]),
        ("achieved_pusch_snr_db_median", source["observed_hot_pusch_snr_db_p50"]),
        ("achieved_pusch_snr_db_p95", source["observed_hot_pusch_snr_db_p95"]),
    ):
        require(math.isclose(float(tail.get(key, math.nan)), float(expected),
                             abs_tol=1e-9), f"-3.5 rung {key} drift")
    require(dict(summary.get("command") or {}).get("candidate_applied_once") is True,
            "-3.5 provenance is not a hot-applied rung")
    require(summary.get("cold_attach_bound_evaluated") in (None, False),
            "-3.5 hot rung unexpectedly evaluated cold attach")
    _all_promotions_false(summary, "-3.5 rung summary")
    return {
        "status": "SEALED_MINUS3P5_EXPECTATION_PROVENANCE_VERIFIED",
        "campaign_manifest_sha256": source["manifest_sha256"],
        "campaign_manifest_output_count": len(root_inventory),
        "rung_manifest_sha256": source["rung_manifest_sha256"],
        "rung_terminal_sha256": source["rung_terminal_sha256"],
        "rung_summary_sha256": source["rung_summary_sha256"],
        "rung_manifest_output_count": len(rung_inventory),
        "commanded_noise_power_db": float(source["commanded_noise_power_db"]),
        "hot_observed_pusch_snr_db_p05": float(source["observed_hot_pusch_snr_db_p05"]),
        "hot_observed_pusch_snr_db_p50": float(source["observed_hot_pusch_snr_db_p50"]),
        "hot_observed_pusch_snr_db_p95": float(source["observed_hot_pusch_snr_db_p95"]),
        "provenance_role": source["provenance_role"],
        "cold_attach_evidence": False,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
    }


def verify_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    contract = config["contract"]
    directory = resolve_repo_path(str(source["directory"]))
    manifest_path = directory / source["manifest"]
    terminal_path = directory / source["terminal"]
    summary_path = directory / source["campaign_summary"]
    resolved_path = directory / source["resolved_config"]
    for path, expected, label in (
        (manifest_path, source["manifest_sha256"], "manifest"),
        (terminal_path, source["terminal_sha256"], "terminal"),
        (summary_path, source["campaign_summary_sha256"], "campaign summary"),
        (resolved_path, source["resolved_config_sha256"], "resolved config"),
    ):
        require(path.is_file() and sha256(path) == expected,
                f"N3C {label} seal drift")
    manifest = load_json(manifest_path)
    terminal = load_json(terminal_path)
    campaign = load_json(summary_path)
    resolved = load_json(resolved_path)
    require(manifest.get("schema")
            == "scenesense.ue_n3c_cold_attach_live_campaign_manifest.v1",
            "N3C campaign manifest schema mismatch")
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"),
                           (campaign, "summary")):
        require(payload.get("status") == source["required_status"],
                f"N3C {label} status mismatch")
        _all_promotions_false(payload, f"N3C {label}")
    require(terminal.get("manifest_sha256") == source["manifest_sha256"],
            "N3C terminal does not bind campaign manifest")
    require(manifest.get("config_sha256") == source["source_config_sha256"]
            and manifest.get("runner_sha256") == source["source_runner_sha256"]
            and manifest.get("engine_runner_sha256")
            == source["source_engine_runner_sha256"],
            "N3C campaign runtime provenance mismatch")
    root_inventory = verify_inventory(
        directory, manifest,
        allowed_uninventoried=(source["manifest"], source["terminal"]),
    )
    require(len(root_inventory) == int(source["expected_root_output_count"]),
            "N3C root output count drift")
    require(root_inventory.get(source["campaign_summary"])
            == source["campaign_summary_sha256"],
            "root manifest does not bind campaign summary")
    require(root_inventory.get(source["resolved_config"])
            == source["resolved_config_sha256"],
            "root manifest does not bind resolved config")
    rcampaign = dict(resolved.get("campaign") or {})
    radio = dict(resolved.get("radio") or {})
    startup = dict(resolved.get("startup_channel") or {})
    require(rcampaign.get("repetitions") == 3
            and rcampaign.get("candidate_application_count") == 0
            and rcampaign.get("one_fresh_ran_per_repetition") is True,
            "resolved N3C campaign contract drift")
    require(startup.get("rfsimu_channel_ue0")
            == float(contract["tested_commanded_noise_power_db"]),
            "resolved N3C startup command drift")
    require(int(radio.get("attach_timeout_s", -1)) == 180,
            "resolved N3C attach timeout drift")
    runtime_seals = {
        row["path"]: row["sha256"] for row in resolved.get("runtime_seals", [])
    }
    required_runtime = {
        "rl_agent/ue_n3c_oai_ul_cold_attach_refinement_live_v1.py":
            source["source_runner_sha256"],
        "rl_agent/ue_n3c_oai_ul_cold_attach_refinement_v1.py":
            source["source_prepare_runner_sha256"],
        "rl_agent/ue_n3b_oai_ul_cold_attach_confirmation_v1.py":
            source["source_n3b_runner_sha256"],
        "rl_agent/ue_n3_oai_ul_command_calibration_v1.py":
            source["source_engine_runner_sha256"],
    }
    require(all(runtime_seals.get(path) == digest
                for path, digest in required_runtime.items()),
            "resolved N3C runtime-seal provenance drift")
    require(campaign.get("live_execution_attempted") is True
            and campaign.get("runtime_executed") is True
            and campaign.get("socket_executed") is True,
            "N3C source was not a live campaign")
    require(int(campaign.get("repetitions_planned", -1)) == 3
            and int(campaign.get("repetitions_executed", -1)) == 3,
            "N3C source is not a complete three-repetition campaign")
    require(int(campaign.get("joint_candidate_confirmation_passes", -1)) == 2
            and campaign.get("joint_candidate_confirmation_3_of_3_pass") is False,
            "N3C mixed joint-confirmation result drift")
    require(int(campaign.get("cold_attach_passes", -1)) == 2
            and int(campaign.get("authoritative_service_gate_passes", -1)) == 2,
            "N3C attach/service pass counts drift")
    require(int(campaign.get("operational_screen_attach_nonconfirmations", -1)) == 1
            and int(campaign.get("valid_nonconfirming_outcomes_retained", -1)) == 1,
            "N3C operational-screen nonconfirmation count drift")
    require(campaign.get("candidate_causal_attach_failure_confirmed") is False
            and campaign.get("review_before_next_action_required") is True,
            "N3C campaign mixed-outcome claim boundary drift")
    require(int(campaign.get("fresh_ran_epoch_count", -1)) == 3
            and int(campaign.get("unique_control_session_count", -1)) == 3,
            "N3C source lacks three fresh experiment identities")

    repetition_root = directory / "repetitions"
    expected_names = {row[0] for row in EXPECTED_REPETITIONS}
    actual_names = {path.name for path in repetition_root.iterdir() if path.is_dir()}
    require(actual_names == expected_names, "N3C repetition directory set drift")
    proofs = {
        Path(str(row["directory"])).name: row
        for row in campaign.get("repetition_evidence", [])
    }
    require(set(proofs) == expected_names, "campaign repetition-proof set drift")
    epochs: set[str] = set()
    sessions: set[str] = set()
    verified: list[dict[str, Any]] = []
    for spec in config["expected_repetitions"]:
        name = spec["directory"]
        status = spec["status"]
        rep_dir = repetition_root / name
        rep_manifest_path = rep_dir / "manifest.json"
        rep_terminal_path = rep_dir / f"{status}.json"
        rep_summary_path = rep_dir / "repetition_summary.json"
        for path, expected, label in (
            (rep_manifest_path, spec["manifest_sha256"], "manifest"),
            (rep_terminal_path, spec["terminal_sha256"], "terminal"),
            (rep_summary_path, spec["summary_sha256"], "summary"),
        ):
            require(path.is_file() and sha256(path) == expected,
                    f"{name} {label} seal drift")
        rep_manifest = load_json(rep_manifest_path)
        rep_terminal = load_json(rep_terminal_path)
        rep_summary = load_json(rep_summary_path)
        require(rep_manifest.get("schema")
                == "scenesense.ue_n3c_cold_attach_repetition_manifest.v1",
                f"{name} manifest schema mismatch")
        for payload, label in ((rep_manifest, "manifest"),
                               (rep_terminal, "terminal"),
                               (rep_summary, "summary")):
            require(payload.get("status") == status,
                    f"{name} {label} status mismatch")
            require(int(payload.get("repetition_index", -1))
                    == int(spec["repetition_index"]),
                    f"{name} {label} repetition index mismatch")
            require(math.isclose(
                float(payload.get("commanded_noise_power_db", math.nan)),
                float(contract["tested_commanded_noise_power_db"]), abs_tol=1e-9,
            ), f"{name} {label} command mismatch")
            _all_promotions_false(payload, f"{name} {label}")
        require(rep_terminal.get("manifest_sha256") == spec["manifest_sha256"],
                f"{name} terminal does not bind nested manifest")
        require(rep_manifest.get("config_sha256") == source["source_config_sha256"]
                and rep_manifest.get("runner_sha256") == source["source_runner_sha256"]
                and rep_manifest.get("engine_runner_sha256")
                == source["source_engine_runner_sha256"],
                f"{name} runtime provenance mismatch")
        require(int(rep_manifest.get("candidate_application_count", -1)) == 0
                and int(rep_manifest.get("restore_application_count", -1)) == 1,
                f"{name} command/restore count drift")
        rep_inventory = verify_inventory(
            rep_dir, rep_manifest,
            allowed_uninventoried=("manifest.json", f"{status}.json"),
        )
        require(len(rep_inventory) == int(spec["manifest_output_count"]),
                f"{name} nested output count drift")
        required = {
            "repetition_summary.json", "attach_gate.json", "cleanup_report.json",
            "startup_channel_runtime_gate.json", "source_oai_config_integrity.json",
            "cold_start_identity.json", "runtime_seals.json",
        }
        require(required.issubset(rep_inventory),
                f"{name} nested manifest lacks required evidence")
        # The root campaign must bind every nested byte, including the nested
        # manifest and terminal that are intentionally outside the nested list.
        for artifact in rep_dir.rglob("*"):
            if artifact.is_file():
                relative = str(artifact.relative_to(directory))
                require(root_inventory.get(relative) == sha256(artifact),
                        f"root campaign does not bind {relative}")
        proof = proofs[name]
        require(proof.get("manifest_sha256") == spec["manifest_sha256"]
                and proof.get("terminal_sha256") == spec["terminal_sha256"],
                f"campaign proof mismatch for {name}")
        identity = load_json(rep_dir / "cold_start_identity.json")
        epoch = str(rep_manifest.get("ran_epoch_id", ""))
        session = str(rep_manifest.get("control_session_id", ""))
        require(epoch and session and epoch != session,
                f"{name} lacks distinct RAN/control identities")
        require(identity.get("ran_epoch_id") == epoch
                and identity.get("control_session_id") == session
                and identity.get("candidate_baked_before_ue_launch") is True,
                f"{name} cold-start identity mismatch")
        require(epoch not in epochs and session not in sessions,
                f"{name} reuses a campaign identity")
        epochs.add(epoch)
        sessions.add(session)
        cleanup = load_json(rep_dir / "cleanup_report.json")
        startup_gate = load_json(rep_dir / "startup_channel_runtime_gate.json")
        integrity = load_json(rep_dir / "source_oai_config_integrity.json")
        valid = (
            _valid_joint_pass(rep_summary, cleanup, startup_gate, integrity, contract)
            if status == PASS_STATUS else
            _valid_operational_nonconfirmation(
                rep_summary, cleanup, startup_gate, integrity, contract
            )
        )
        require(valid, f"{name} evidence semantics are invalid")
        verified.append({
            "directory": name,
            "repetition_index": int(spec["repetition_index"]),
            "status": status,
            "manifest_sha256": spec["manifest_sha256"],
            "terminal_sha256": spec["terminal_sha256"],
            "summary_sha256": spec["summary_sha256"],
            "manifest_output_count": len(rep_inventory),
            "ran_epoch_id": epoch,
            "control_session_id": session,
            "evidence_semantics_verified": True,
        })
    ladder = _verify_command_ladder(config)
    return {
        "status": "SEALED_N3C_LIVE_01_AND_MINUS3P5_EXPECTATION_PROVENANCE_VERIFIED",
        "source_directory": str(directory),
        "campaign_manifest_sha256": source["manifest_sha256"],
        "campaign_terminal_sha256": source["terminal_sha256"],
        "campaign_summary_sha256": source["campaign_summary_sha256"],
        "resolved_config_sha256": source["resolved_config_sha256"],
        "campaign_manifest_output_count": len(root_inventory),
        "verified_repetition_count": len(verified),
        "unique_ran_epoch_count": len(epochs),
        "unique_control_session_count": len(sessions),
        "repetitions": verified,
        "command_ladder_expectation_provenance": ladder,
    }


def adjudicate_repetition(
    summary: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    startup: Mapping[str, Any],
    integrity: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    is_pass = summary.get("status") == PASS_STATUS
    is_nonconfirm = summary.get("status") == NONCONFIRM_STATUS
    valid_pass = is_pass and _valid_joint_pass(
        summary, cleanup, startup, integrity, contract
    )
    valid_nonconfirm = is_nonconfirm and _valid_operational_nonconfirmation(
        summary, cleanup, startup, integrity, contract
    )
    valid = valid_pass or valid_nonconfirm
    attach = dict(summary.get("attach_gate") or {})
    tail = dict(summary.get("service_tail") or {})
    failure_cold_values = (
        summary.get("observed_cold_achieved_pusch_snr_db_p05"),
        summary.get("observed_cold_achieved_pusch_snr_db_p50"),
        summary.get("observed_cold_achieved_pusch_snr_db_p95"),
    )
    outcome = (
        "JOINT_COLD_ATTACH_AND_SERVICE_CONFIRMATION"
        if valid_pass else
        "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION"
        if valid_nonconfirm else
        "INVALID_EVIDENCE"
    )
    return {
        "repetition_index": int(summary.get("repetition_index", -1)),
        "commanded_noise_power_db": summary.get("commanded_noise_power_db"),
        "adjudicated_outcome": outcome,
        "evidence_valid": valid,
        "cold_attach_pass": True if valid_pass else False if valid_nonconfirm else None,
        "authoritative_service_pass": (
            True if valid_pass else False if valid_nonconfirm else None
        ),
        "joint_candidate_confirmation_pass": (
            True if valid_pass else False if valid_nonconfirm else None
        ),
        "attach_gate_duration_s": attach.get("duration_s"),
        "attach_timeout_s": attach.get("timeout_s"),
        "core_ready_at_terminal": attach.get("core_ready_at_terminal"),
        "ran_processes_alive_at_terminal": attach.get("ran_processes_alive_at_terminal"),
        "observed_ipv4": (
            [attach.get("discovered_ipv4")] if valid_pass else attach.get("observed_ipv4")
        ),
        "candidate_application_count": summary.get("candidate_application_count"),
        "restore_application_count": summary.get("restore_application_count"),
        "cleanup_verified": cleanup.get("clean") is True and not cleanup.get("errors"),
        "source_oai_configs_unchanged": integrity.get("unchanged") is True,
        "service_achieved_pusch_snr_db_p05": (
            tail.get("achieved_pusch_snr_db_p05") if valid_pass else None
        ),
        "service_achieved_pusch_snr_db_p50": (
            tail.get("achieved_pusch_snr_db_median") if valid_pass else None
        ),
        "service_achieved_pusch_snr_db_p95": (
            tail.get("achieved_pusch_snr_db_p95") if valid_pass else None
        ),
        "failed_cold_achieved_pusch_snr_db_p05": (
            failure_cold_values[0] if valid_nonconfirm else None
        ),
        "failed_cold_achieved_pusch_snr_db_p50": (
            failure_cold_values[1] if valid_nonconfirm else None
        ),
        "failed_cold_achieved_pusch_snr_db_p95": (
            failure_cold_values[2] if valid_nonconfirm else None
        ),
        "failed_cold_achieved_snr_is_null": (
            all(value is None for value in failure_cold_values)
            if valid_nonconfirm else None
        ),
        "failed_cold_achieved_snr_interpretation": (
            "UNOBSERVED_NO_SERVING_RNTI_PUSCH_WINDOW"
            if valid_nonconfirm and all(value is None for value in failure_cold_values)
            else None
        ),
        "attach_failure_evidence_role": (
            summary.get("attach_failure_evidence_role") if valid_nonconfirm else None
        ),
        "candidate_causal_attach_failure_confirmed": False,
        "physical_rf_cutoff_established": False,
        "mapping_or_bound_promoted": False,
    }


def aggregate_adjudications(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    ladder: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(contract["repetitions"])
    require(len(rows) == required, "adjudication repetition count drift")
    valid = [row for row in rows if row["evidence_valid"]]
    joint_passes = sum(row["joint_candidate_confirmation_pass"] is True for row in valid)
    nonconfirmations = sum(
        row["adjudicated_outcome"] == "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION"
        for row in valid
    )
    failure_null = all(
        row["failed_cold_achieved_snr_is_null"] is True
        for row in valid
        if row["adjudicated_outcome"] == "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION"
    )
    accepted = (
        len(valid) == required
        and joint_passes == int(contract["expected_joint_confirmation_passes"])
        and nonconfirmations == int(contract["expected_operational_nonconfirmations"])
        and failure_null
    )
    n3d = float(contract["n3d_selected_command_db"]) if accepted else None
    return {
        "status": SUCCESS if accepted else UNRESOLVED,
        "n3c_mixed_outcome_accepted_for_n3d_eligibility": accepted,
        "tested_commanded_noise_power_db": float(
            contract["tested_commanded_noise_power_db"]
        ),
        "repetitions": required,
        "valid_repetitions": len(valid),
        "joint_confirmation_passes": joint_passes,
        "operational_screen_nonconfirmations": nonconfirmations,
        "joint_confirmation_result": "2_OF_3_PASS" if accepted else "UNRESOLVED",
        "required_floor_confirmation_result": "3_OF_3_REQUIRED_NOT_MET",
        "n3c_floor_confirmed": False,
        "failed_repetition_cold_achieved_pusch_snr_db_p05": None,
        "failed_repetition_cold_achieved_pusch_snr_db_p50": None,
        "failed_repetition_cold_achieved_pusch_snr_db_p95": None,
        "failed_repetition_cold_achieved_snr_status": (
            "UNOBSERVED_NO_SERVING_RNTI_PUSCH_WINDOW" if accepted else "UNRESOLVED"
        ),
        "operational_nonconfirmation_role": NONCONFIRM_ROLE,
        "candidate_causal_attach_failure_confirmed": False,
        "physical_rf_cutoff_status": "NOT_ESTABLISHED",
        "hard_loss_boundary_status": "NOT_ESTABLISHED",
        "target_mapping_status": "NOT_ESTABLISHED_NOT_PROMOTED",
        "l_attach_status": "UNRESOLVED_MIXED_2_OF_3_AT_MINUS3_COMMAND",
        "l_operational_status": "UNRESOLVED_MIXED_2_OF_3_AT_MINUS3_COMMAND",
        "n3d_selected_command_db": n3d,
        "n3d_eligibility_status": (
            "UE_N3C_VALID_MIXED_OUTCOME_ACCEPTED_FOR_ADJACENT_N3D_CANDIDATE"
            if accepted else "N3D_ELIGIBILITY_UNRESOLVED"
        ),
        "n3d_selection_scope": "ELIGIBILITY_ONLY",
        "n3d_selection_basis": (
            "ADJACENT_STRONGER_RFSIM_COMMAND_AFTER_VALID_2_OF_3_MIXED_OUTCOME"
        ),
        "n3d_command_relation": contract["n3d_command_relation"],
        "n3d_expectation_provenance": dict(ladder),
        "n3d_expected_achieved_snr_role": (
            "EXPECTATION_ONLY_NOT_COLD_ATTAINED_NOT_A_PROMOTED_MAPPING"
        ),
        "separate_live_authority_required": True,
        "n3d_execution_authorized": False,
        "n3d_executed": False,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "operational_bound_promoted": False,
        "connectivity_bound_promoted": False,
        "usable_service_bound_promoted": False,
    }


class ReviewRunner:
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        *,
        output_root: Path | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        allowed_root = (output_root or OUTPUT_ROOT).resolve()
        protected_sources = (
            resolve_repo_path(SOURCE["directory"]),
            resolve_repo_path(COMMAND_LADDER["directory"]),
        )
        for protected in protected_sources:
            try:
                allowed_root.relative_to(protected)
            except ValueError:
                pass
            else:
                raise ReviewFailure(
                    f"output root is inside sealed source evidence: {protected}"
                )
        allowed_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir.resolve()
        require(self.output_dir.parent == allowed_root,
                f"output must be one immutable leaf under {allowed_root}")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.output_dir.name)
                is not None, "unsafe output leaf name")
        for protected in protected_sources:
            try:
                self.output_dir.relative_to(protected)
            except ValueError:
                pass
            else:
                raise ReviewFailure(
                    f"output directory is inside sealed source evidence: {protected}"
                )
        try:
            self.output_dir.mkdir()
        except FileExistsError as exc:
            raise ReviewFailure(
                f"create-only output already exists: {self.output_dir}"
            ) from exc

    def _seal(self, status: str, terminal: Mapping[str, Any]) -> None:
        excluded = {OUTPUT["manifest"], OUTPUT["failure"]}
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if (not path.is_file() or path.name in excluded
                    or path.name.startswith("UE_N3C_OUTCOME_")):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        manifest_path = self.output_dir / OUTPUT["manifest"]
        write_new_json(manifest_path, {
            "schema": "scenesense.ue_n3c_cold_attach_outcome_review_manifest.v1",
            "status": status,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_n3c_manifest_sha256": SOURCE["manifest_sha256"],
            "command_ladder_manifest_sha256": COMMAND_LADDER["manifest_sha256"],
            "command_ladder_minus3p5_rung_manifest_sha256": (
                COMMAND_LADDER["rung_manifest_sha256"]
            ),
            "offline_only": True,
            "oai_run_authorized": False,
            "socket_execution_authorized": False,
            "carla_run_authorized": False,
            "n3d_execution_authorized": False,
            "n3d_executed": False,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "operational_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "outputs": files,
        })
        payload = {
            **dict(terminal),
            "status": status,
            "offline_only": True,
            "oai_run_authorized": False,
            "socket_execution_authorized": False,
            "carla_run_authorized": False,
            "n3d_execution_authorized": False,
            "n3d_executed": False,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "operational_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "manifest_sha256": sha256(manifest_path),
        }
        terminal_name = OUTPUT["failure"] if status == "FAILED" else f"{status}.json"
        write_new_json(self.output_dir / terminal_name, payload)

    def run(self) -> int:
        try:
            validate_config(self.config, verify_hashes=True)
            write_new_json(self.output_dir / OUTPUT["resolved_config"], self.config)
            verification = verify_source(self.config)
            write_new_json(self.output_dir / OUTPUT["source_verification"], verification)
            source_dir = resolve_repo_path(self.config["source"]["directory"])
            rows = []
            for spec in self.config["expected_repetitions"]:
                rep_dir = source_dir / "repetitions" / spec["directory"]
                row = adjudicate_repetition(
                    load_json(rep_dir / "repetition_summary.json"),
                    load_json(rep_dir / "cleanup_report.json"),
                    load_json(rep_dir / "startup_channel_runtime_gate.json"),
                    load_json(rep_dir / "source_oai_config_integrity.json"),
                    self.config["contract"],
                )
                row["source_directory"] = spec["directory"]
                row["source_manifest_sha256"] = spec["manifest_sha256"]
                rows.append(row)
            write_new_json(
                self.output_dir / OUTPUT["adjudications_json"],
                {
                    "schema": "scenesense.ue_n3c_cold_attach_adjudications.v1",
                    "rows": rows,
                },
            )
            write_new_csv(self.output_dir / OUTPUT["adjudications_csv"], rows)
            aggregate = aggregate_adjudications(
                rows,
                self.config["contract"],
                verification["command_ladder_expectation_provenance"],
            )
            status = str(aggregate["status"])
            selection = (
                "- N3D eligibility: RFsim command -3.5 dB, adjacent and stronger "
                "than -3.0 dB; execution requires separate live authority.\n"
                if aggregate["n3c_mixed_outcome_accepted_for_n3d_eligibility"] else
                "- N3D eligibility is unresolved.\n"
            )
            report = (
                "# UE-N3C cold-attach/service outcome review\n\n"
                f"- Contract status: `{status}`\n"
                "- RFsim command -3.0 dB joint cold-attach + 60 s service passes: "
                f"{aggregate['joint_confirmation_passes']}/3.\n"
                "- One repetition is retained as a valid operational-screen "
                "nonconfirmation; it is not candidate-causal or physical-boundary proof.\n"
                "- Cold achieved PUSCH SNR on the failed repetition is unobserved "
                "(null); -3.0 dB is the RFsim command, not that missing measurement.\n"
                "- The 3/3 confirmation rule was not met. L_attach and L_operational "
                "remain unresolved, and -3.0 dB is not promoted as a floor.\n"
                f"{selection}"
                "- The prior -3.5 dB hot rung (PUSCH p05/p50/p95 "
                "7.0/7.5/7.5 dB) is expectation-only provenance, not cold evidence "
                "or a promoted command-to-SNR mapping.\n"
                "- No target mapping or numeric, operational, connectivity, or "
                "usable-service bound is promoted. No OAI, socket, or CARLA action "
                "was authorized or executed by this review.\n"
            )
            write_new_text(self.output_dir / OUTPUT["report"], report)
            summary = {
                **aggregate,
                "source_verification": verification,
                "adjudications": rows,
                "review_scope": (
                    "SEALED_N3C_MIXED_OUTCOME_AND_N3D_ELIGIBILITY_ONLY"
                ),
                "review_required": True,
            }
            write_new_json(self.output_dir / OUTPUT["summary"], summary)
            self._seal(status, summary)
            print(json.dumps({"output_dir": str(self.output_dir), "status": status},
                             sort_keys=True))
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            failure = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "offline_only": True,
                "review_required": True,
                "n3d_execution_authorized": False,
                "n3d_executed": False,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "operational_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
            }
            try:
                write_new_json(self.output_dir / OUTPUT["failure_context"], failure)
                self._seal("FAILED", failure)
            except Exception as sealing_exc:  # pragma: no cover - last-resort stderr
                failure["sealing_error"] = str(sealing_exc)
            print(json.dumps({"output_dir": str(self.output_dir), **failure}, sort_keys=True),
                  file=sys.stderr)
            return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return ReviewRunner(Path(args.config), Path(args.output_dir)).run()


if __name__ == "__main__":
    raise SystemExit(main())

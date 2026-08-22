#!/usr/bin/env python3
"""Freeze the PREPARE-only UE-N3D -3.5 dB cold-attach refinement plan.

This stage is deliberately unable to run OAI, open the RFsim control socket,
or start CARLA.  It verifies the sealed mixed 2/3 N3C outcome and the sealed
already-attached -3.5 dB command-ladder rung, then writes a create-only plan
for three future fresh-RAN cold starts.  A distinct N3C adjudication and a
separately reviewed live implementation are prerequisites for execution.

The warm-rung 7.5 +/- 0.5 dB achieved-PUSCH band is consistency-only.  It is
not an actuator mapping, a cold-attach observation, or a promoted bound.
"""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3d_oai_ul_cold_attach_refinement_v1.json"
PREPARE_ONLY = "PREPARE_ONLY"
SCHEMA = "scenesense.ue_n3d_oai_ul_cold_attach_refinement_config.v1"
PLAN_BLOCKED = "UE_N3D_COLD_ATTACH_REFINEMENT_PLAN_FROZEN_PREREQUISITES_PENDING"
FAILED = "FAILED"
PENDING = "PENDING_CREATE_ONLY_N3C_ADJUDICATION"
CANDIDATE_COMMAND_DB = -3.5
EXPECTED_ACHIEVED_PUSCH_SNR_DB = 7.5
ACHIEVED_SNR_TOLERANCE_DB = 0.5
SCIENTIFIC_CONTRACT_SHA256 = (
    "e0d49ea252a721be150aa7d5eaf43b5bd3b03940f225fd3eb4776004021cdc13"
)

EXPECTED_OUTPUT = {
    "resolved_config": "resolved_config.json",
    "campaign_plan": "campaign_plan.csv",
    "campaign_summary": "campaign_summary.json",
    "manifest": "manifest.json",
    "failure": "FAILED.json",
}

SUPERSEDED_PRE_FINAL_PLAN = {
    "directory": (
        "rl_agent/experiments/ue_n3d_oai_ul_cold_attach_refinement_v1/"
        "20260821_plan_01"
    ),
    "manifest_sha256": "b80c62d97f46674cae2892a6bbbc5c8c352500bdcafb103b6ca13c02278a6594",
    "disposition": "SUPERSEDED_PRE_FINAL_NON_AUTHORITATIVE_DO_NOT_PIN",
}

EXPECTED_N3C_LIVE = {
    "directory": (
        "rl_agent/experiments/ue_n3c_oai_ul_cold_attach_refinement_live_v1/"
        "20260821_live_01"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": "7aaa76319d7fdfbe82082a89b514da2b7a3071d0f3b1b52f5e13ed32cb0c6f3b",
    "terminal": "UE_N3C_COLD_ATTACH_REFINEMENT_NOT_3_OF_3_REVIEW_REQUIRED.json",
    "terminal_sha256": "afa36b3f486c816924114de441364a8aaf06f39e76d9273c3eb3646e3e3877a2",
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": "23ec2a7231edecf97a160162b6927e8f7b1ba1f573064f6f1e8a00ed07da4d77",
    "campaign_summary": "campaign_summary.json",
    "campaign_summary_sha256": "41d50ef8c2701a9dcb29ccf1f8a152ab300b5cc337210e3d13e3734f55e677c6",
    "required_status": "UE_N3C_COLD_ATTACH_REFINEMENT_NOT_3_OF_3_REVIEW_REQUIRED",
    "source_config_sha256": "13ea9536bd5492a98d6f42b61ed5371d6e02cd3bd47381f69fd877797209fc4d",
    "source_runner_sha256": "d3f62a4b116786f7e684090a02be41c3a93d6b9013fca2da9bb8e28636b45f7d",
    "manifest_output_count": 251,
}

EXPECTED_N3C_REPETITIONS = (
    {
        "directory": "repetitions/rep_01_minus3p0_cold",
        "status": "UE_N3C_COLD_ATTACH_SERVICE_REPETITION_PASSED",
        "manifest_sha256": "8a210b13eec1113722368b980255ca95858747fecbfc0e1eb10b932793e9a762",
        "terminal_sha256": "e1851a3977c5091435fdf08614a6fe6134e978f5af5c67341c1fbeb7681bc105",
        "manifest_output_count": 109,
        "joint_pass": True,
    },
    {
        "directory": "repetitions/rep_02_minus3p0_cold",
        "status": "UE_N3C_COLD_ATTACH_REPETITION_VALID_OPERATIONAL_SCREEN_NONCONFIRMATION",
        "manifest_sha256": "d9d66341f4f3ee813778e001781467a8a5b80a48cf9143a629d21060848bbe95",
        "terminal_sha256": "b43bb259f26f64d0cdd0aff0b72c4b02eaebb99af53ae9a52020a45efb615a10",
        "manifest_output_count": 21,
        "joint_pass": False,
    },
    {
        "directory": "repetitions/rep_03_minus3p0_cold",
        "status": "UE_N3C_COLD_ATTACH_SERVICE_REPETITION_PASSED",
        "manifest_sha256": "00c1c83bbb55d83f1f4c1072202f29d5d1329ee2ef6814ea40a4770bd2d3c9eb",
        "terminal_sha256": "35cead22ad4c08d3059b1d367d4363ca1403d76b47c89d65cdadb69f61d4d7bd",
        "manifest_output_count": 109,
        "joint_pass": True,
    },
)

EXPECTED_N3C_ADJUDICATION_PENDING = {
    "directory": PENDING,
    "manifest": "manifest.json",
    "manifest_sha256": PENDING,
    "terminal": "UE_N3C_OUTCOME_ADJUDICATED_N3D_ELIGIBLE_REVIEW_REQUIRED.json",
    "terminal_sha256": PENDING,
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": PENDING,
    "source_config_sha256": PENDING,
    "source_runner_sha256": PENDING,
    "required_status": "UE_N3C_OUTCOME_ADJUDICATED_N3D_ELIGIBLE_REVIEW_REQUIRED",
    "n3d_eligibility_status": (
        "UE_N3C_VALID_MIXED_OUTCOME_ACCEPTED_FOR_ADJACENT_N3D_CANDIDATE"
    ),
    "n3d_selected_command_db": CANDIDATE_COMMAND_DB,
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
    "manifest_output_count": 529,
    "rung_directory": "rungs/rung_01_minus3p5",
    "rung_manifest": "manifest.json",
    "rung_manifest_sha256": "45d9bebea2e681f99c2a22e1a8092ef26290849806b7dcf306be131dc5d36c31",
    "rung_manifest_output_count": 103,
    "rung_summary": "rung_summary.json",
    "rung_summary_sha256": "e7dcca3082a7bea7860d641b5496544fd10d3b61690cb8e08e7e11403011758b",
    "rung_terminal": "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY.json",
    "rung_terminal_sha256": "e25435257b4d616e6da41d960eb50789b1ae21ab816f78d8bfecab0f36927a11",
    "rung_resolved_config": "resolved_config.json",
    "rung_resolved_config_sha256": "fe562882a508de68bbe45bbc01c495491412bffdd4cf491e5f5df953fc8cd6ef",
    "tail_service_summary": "tail_service_summary.json",
    "tail_service_summary_sha256": "a60a58d9e5bb2af764a98b7033353b6afa4101d2e45ece1cee0871d4a7764900",
}

PROTECTED_SOURCE_DIRECTORIES = (
    EXPECTED_N3C_LIVE["directory"],
    EXPECTED_COMMAND_LADDER["directory"],
    "rl_agent/experiments/ue_n3c_oai_ul_cold_attach_refinement_v1",
    "rl_agent/experiments/ue_n3c_cold_attach_outcome_review_v1",
)


class ColdAttachRefinementFailure(RuntimeError):
    """Fail-closed plan, evidence, authority, or containment failure."""


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


def validate_output_contract(config: Mapping[str, Any]) -> None:
    require(config.get("output") == EXPECTED_OUTPUT, "N3D output leaf contract drift")


def _scientific_contract_sha256(config: Mapping[str, Any]) -> str:
    keys = (
        "paths",
        "campaign",
        "startup_channel",
        "rung",
        "traffic",
        "transport_gates",
        "preflight",
        "radio",
        "actuator",
        "telemetry",
        "analysis",
        "output",
    )
    projection = {key: config[key] for key in keys}
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _terminal_name(status: str) -> str:
    return "FAILED.json" if status == FAILED else f"{status}.json"


def _verify_manifest_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    expected_output_count: int | None = None,
    strict_complete: bool = False,
) -> set[str]:
    rows = list(manifest.get("outputs", []))
    require(rows, f"manifest has no outputs: {directory}")
    if expected_output_count is not None:
        require(
            len(rows) == expected_output_count,
            f"manifest output-count mismatch: {directory}",
        )
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in seen, f"blank/duplicate output: {relative!r}")
        seen.add(relative)
        artifact = (directory / relative).resolve()
        try:
            artifact.relative_to(directory.resolve())
        except ValueError as exc:
            raise ColdAttachRefinementFailure(
                f"manifest output escapes directory: {relative}"
            ) from exc
        require(artifact.is_file(), f"manifest output missing: {relative}")
        require(
            artifact.stat().st_size == int(row.get("bytes", -1)),
            f"manifest output size drift: {relative}",
        )
        require(n2.sha256(artifact) == row.get("sha256"), f"output hash drift: {relative}")
    if strict_complete:
        expected = seen | {"manifest.json", _terminal_name(str(manifest.get("status", "")))}
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        }
        require(
            actual == expected,
            f"strict manifest completeness mismatch at {directory}: "
            f"unexpected={sorted(actual - expected)} missing={sorted(expected - actual)}",
        )
    return seen


def _verify_runtime_seals(config: Mapping[str, Any]) -> None:
    seals = list(config["runtime_seals"])
    paths = [str(row.get("path", "")) for row in seals]
    require(len(paths) == len(set(paths)), "runtime seal paths repeat")
    required = {
        "rl_agent/ue_n3d_oai_ul_cold_attach_refinement_v1.py",
        "rl_agent/ue_n3c_oai_ul_cold_attach_refinement_live_v1.py",
        "rl_agent/ue_n3_oai_ul_command_calibration_v1.py",
        "rl_agent/ue_n2_oai_ul_calibration_smoke.py",
    }
    require(required.issubset(paths), "required N3D runtime seals are absent")
    for row in seals:
        path = resolve_repo_path(str(row.get("path", "")))
        expected = str(row.get("sha256", ""))
        require(path.is_file(), f"sealed runtime file missing: {path}")
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"malformed runtime seal: {row.get('path')}")
        require(n2.sha256(path) == expected, f"runtime seal drift: {row.get('path')}")


def pending_adjudication(block: Mapping[str, Any]) -> bool:
    return block == EXPECTED_N3C_ADJUDICATION_PENDING


def validate_config(
    config: Mapping[str, Any], *, verify_hashes: bool = True,
    require_live: bool = False,
) -> None:
    require(config.get("schema") == SCHEMA, "unexpected N3D config schema")
    require(
        config.get("experiment_id") == "ue_n3d_oai_ul_cold_attach_refinement_v1",
        "N3D experiment identity drift",
    )
    require(
        config.get("claim_boundary")
        == "ADJACENT_STRONGER_COLD_ATTACH_SCREEN_EVIDENCE_ONLY_NO_BOUND_PROMOTION",
        "N3D claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_plan_authorized") is True,
            "offline plan authority is absent")
    for key in (
        "live_oai_run_authorized",
        "live_socket_execution_authorized",
        "carla_run_authorized",
        "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized",
        "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized",
        "operational_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden N3D authority enabled: {key}")
    require(authority.get("live_authority_basis") == "NOT_AUTHORIZED_PREPARE_ONLY",
            "N3D live authority basis drift")
    require(not require_live, "N3D live execution is absent in PREPARE-only version")

    predecessors = config["predecessors"]
    require(predecessors.get("n3c_live_evidence") == EXPECTED_N3C_LIVE,
            "sealed N3C live evidence pin drift")
    require(predecessors.get("n3c_adjudication") == EXPECTED_N3C_ADJUDICATION_PENDING,
            "N3C adjudication must remain explicitly pending")
    require(predecessors.get("command_ladder_expectation") == EXPECTED_COMMAND_LADDER,
            "sealed -3.5 command-ladder pin drift")
    require(
        predecessors.get("ue_n1_bundle")
        == "rl_agent/registries/ue_n1_oai_ul_actuator_interface_v2",
        "UE-N1 bundle drift",
    )

    require(config["paths"]["output_root"]
            == "rl_agent/experiments/ue_n3d_oai_ul_cold_attach_refinement_v1",
            "N3D output root drift")
    require("n3d_live_runner" not in config["paths"],
            "PREPARE-only config unexpectedly names a live runner")
    validate_output_contract(config)
    require(
        _scientific_contract_sha256(config) == SCIENTIFIC_CONTRACT_SHA256,
        "N3D scientific contract digest drift",
    )

    campaign = config["campaign"]
    require(int(campaign["repetitions"]) == 3, "N3D requires three repetitions")
    require(campaign["one_fresh_ran_per_repetition"] is True,
            "every N3D repetition must use a fresh RAN")
    require(campaign["run_local_configs_only"] is True,
            "N3D must use run-local configurations")
    require(campaign["candidate_baked_before_ue_launch"] is True,
            "candidate must be baked before UE launch")
    require(int(campaign["candidate_application_count"]) == 0,
            "candidate Telnet application is forbidden")
    require([float(value) for value in campaign["commanded_noise_power_db"]]
            == [CANDIDATE_COMMAND_DB], "N3D candidate command drift")
    require(campaign["continue_after_valid_attach_or_service_failure"] is True,
            "valid candidate failures must be retained")
    require(campaign["stop_on_invalid_or_unclean_evidence"] is True,
            "invalid evidence must stop the future campaign")
    require(campaign["post_restore_recovery_fail_closed"] is True,
            "post-restore recovery must remain fail-closed")
    require(campaign["automatic_next_candidate_authorized"] is False,
            "automatic N3E progression is forbidden")

    require(
        config["startup_channel"] == {
            "rfsimu_channel_enB0": -50.0,
            "rfsimu_channel_enB1": -50.0,
            "rfsimu_channel_ue0": CANDIDATE_COMMAND_DB,
        },
        "N3D startup channel drift",
    )
    rung, traffic = config["rung"], config["traffic"]
    require(math.isclose(float(rung["candidate_lead_s"]), 5.0),
            "candidate lead must remain 5 seconds")
    require(math.isclose(float(rung["measured_service_s"]), 60.0),
            "service window must remain exactly 60 seconds")
    require(math.isclose(float(rung["clean_recovery_s"]), 5.0),
            "clean recovery must remain 5 seconds")
    require(int(rung["sender_frames"]) == 700,
            "future sender must schedule 700 frames")
    require(int(rung["expected_service_frames"]) == 600,
            "authoritative service unit must contain 600 frames")
    require(math.isclose(float(traffic["fps"]), 10.0), "probe must use 10 Hz")
    require(int(traffic["frame_bytes"]) == 12_500
            and int(traffic["chunk_bytes"]) == 12_500,
            "matched 1 Mbps probe shape drift")
    require(math.isclose(float(config["radio"]["attach_timeout_s"]), 180.0),
            "cold attach/PDU/ext-DN gate must remain 180 seconds")
    require(config["preflight"]["fail_if_carla_active"] is True,
            "CARLA fail-closed gate disabled")
    gates = config["transport_gates"]
    require(math.isclose(float(gates["primary_complete_frame_ratio"]), 0.99),
            "delivery gate must remain 99 percent")
    require(int(gates["maximum_interarrival_gaps_gte_1s"]) == 0,
            "service window permits no one-second outage")

    analysis = config["analysis"]
    require(math.isclose(float(analysis["expected_achieved_pusch_snr_db"]),
                         EXPECTED_ACHIEVED_PUSCH_SNR_DB),
            "N3D achieved-SNR center drift")
    require(math.isclose(float(analysis["achieved_snr_tolerance_db"]),
                         ACHIEVED_SNR_TOLERANCE_DB),
            "N3D achieved-SNR tolerance drift")
    require(
        analysis["achieved_snr_expectation_role"]
        == "CONSISTENCY_ONLY_FROM_SEALED_WARM_ATTACHED_COMMAND_LADDER_NOT_MAPPING",
        "achieved-SNR evidence role drift",
    )
    require(analysis["post_restore_recovery_required"] is True,
            "post-restore recovery must be required for attached trials")
    require(analysis["attach_failure_evidence_role"]
            == "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION_NOT_CANDIDATE_CAUSAL_OR_PHYSICAL_BOUNDARY_PROOF",
            "attach-failure claim boundary drift")
    if verify_hashes:
        _verify_runtime_seals(config)


def verify_n3c_live(
    config: Mapping[str, Any], *, proof_path: Path | None = None,
) -> dict[str, Any]:
    expected = config["predecessors"]["n3c_live_evidence"]
    require(expected == EXPECTED_N3C_LIVE, "N3C live evidence pin drift")
    directory = resolve_repo_path(str(expected["directory"]))
    paths = {
        "manifest": directory / str(expected["manifest"]),
        "terminal": directory / str(expected["terminal"]),
        "resolved_config": directory / str(expected["resolved_config"]),
        "campaign_summary": directory / str(expected["campaign_summary"]),
    }
    for key, path in paths.items():
        require(path.is_file(), f"N3C predecessor {key} missing: {path}")
        require(n2.sha256(path) == expected[f"{key}_sha256"],
                f"N3C predecessor {key} seal drift")
    manifest, terminal = load_json(paths["manifest"]), load_json(paths["terminal"])
    require(manifest.get("status") == terminal.get("status")
            == expected["required_status"], "N3C predecessor status mismatch")
    require(terminal.get("manifest_sha256") == expected["manifest_sha256"],
            "N3C terminal does not bind its manifest")
    require(manifest.get("config_sha256") == expected["source_config_sha256"]
            and manifest.get("runner_sha256") == expected["source_runner_sha256"],
            "N3C source identity mismatch")
    seen = _verify_manifest_inventory(
        directory,
        manifest,
        expected_output_count=int(expected["manifest_output_count"]),
        strict_complete=True,
    )
    require(str(expected["resolved_config"]) in seen
            and str(expected["campaign_summary"]) in seen,
            "N3C required outputs are absent from manifest")
    require(int(terminal.get("repetitions_executed", -1)) == 3
            and int(terminal.get("cold_attach_passes", -1)) == 2
            and int(terminal.get("joint_candidate_confirmation_passes", -1)) == 2,
            "N3C predecessor is not the sealed mixed 2/3 outcome")
    require(int(terminal.get("valid_nonconfirming_outcomes_retained", -1)) == 1,
            "N3C lacks the retained valid nonconfirming outcome")
    require(int(terminal.get("fresh_ran_epoch_count", -1)) == 3
            and int(terminal.get("unique_control_session_count", -1)) == 3,
            "N3C fresh-start identities are incomplete")
    require(terminal.get("review_before_next_action_required") is True,
            "N3C predecessor does not require review before N3D")
    for key in (
        "target_mapping_promoted",
        "numeric_bound_promoted",
        "connectivity_bound_promoted",
        "usable_service_bound_promoted",
        "operational_bound_promoted",
    ):
        require(terminal.get(key) is False, f"N3C unexpectedly promoted {key}")

    repetitions = []
    for row in EXPECTED_N3C_REPETITIONS:
        rep_dir = directory / str(row["directory"])
        rep_manifest_path = rep_dir / "manifest.json"
        terminal_path = rep_dir / f"{row['status']}.json"
        require(n2.sha256(rep_manifest_path) == row["manifest_sha256"],
                f"N3C repetition manifest drift: {rep_dir}")
        require(n2.sha256(terminal_path) == row["terminal_sha256"],
                f"N3C repetition terminal drift: {rep_dir}")
        rep_manifest, rep_terminal = load_json(rep_manifest_path), load_json(terminal_path)
        require(rep_manifest.get("status") == rep_terminal.get("status") == row["status"],
                f"N3C repetition status drift: {rep_dir}")
        require(rep_terminal.get("manifest_sha256") == row["manifest_sha256"],
                f"N3C repetition terminal binding drift: {rep_dir}")
        require(math.isclose(float(rep_terminal.get("commanded_noise_power_db", math.nan)), -3.0),
                f"N3C repetition command drift: {rep_dir}")
        require(rep_terminal.get("candidate_application_count") == 0
                and rep_terminal.get("restore_application_count") == 1,
                f"N3C repetition command-count drift: {rep_dir}")
        require(rep_terminal.get("evidence_valid_for_aggregation") is True,
                f"N3C repetition is not valid evidence: {rep_dir}")
        require(rep_terminal.get("joint_candidate_confirmation_pass") is row["joint_pass"],
                f"N3C repetition joint outcome drift: {rep_dir}")
        _verify_manifest_inventory(
            rep_dir,
            rep_manifest,
            expected_output_count=int(row["manifest_output_count"]),
            strict_complete=True,
        )
        repetitions.append({
            "directory": str(rep_dir),
            "status": row["status"],
            "manifest_sha256": row["manifest_sha256"],
            "terminal_sha256": row["terminal_sha256"],
            "joint_candidate_confirmation_pass": row["joint_pass"],
        })
    proof = {
        "status": "VERIFIED_READ_ONLY_N3C_MIXED_2_OF_3_PREDECESSOR",
        "directory": str(directory),
        "manifest_sha256": n2.sha256(paths["manifest"]),
        "terminal_sha256": n2.sha256(paths["terminal"]),
        "resolved_config_sha256": n2.sha256(paths["resolved_config"]),
        "campaign_summary_sha256": n2.sha256(paths["campaign_summary"]),
        "verified_output_count": len(seen),
        "cold_attach_passes": 2,
        "joint_candidate_confirmation_passes": 2,
        "trials": 3,
        "review_before_next_action_required": True,
        "repetitions": repetitions,
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def verify_command_ladder_expectation(
    config: Mapping[str, Any], *, proof_path: Path | None = None,
) -> dict[str, Any]:
    expected = config["predecessors"]["command_ladder_expectation"]
    require(expected == EXPECTED_COMMAND_LADDER, "command-ladder pin drift")
    directory = resolve_repo_path(str(expected["directory"]))
    manifest_path = directory / str(expected["manifest"])
    require(n2.sha256(manifest_path) == expected["manifest_sha256"],
            "command-ladder campaign manifest drift")
    manifest = load_json(manifest_path)
    require(manifest.get("status") == expected["required_status"],
            "command-ladder campaign status drift")
    require(manifest.get("config_sha256") == expected["source_config_sha256"]
            and manifest.get("runner_sha256") == expected["source_runner_sha256"],
            "command-ladder source identity drift")
    _verify_manifest_inventory(
        directory,
        manifest,
        expected_output_count=int(expected["manifest_output_count"]),
        strict_complete=True,
    )
    rung = directory / str(expected["rung_directory"])
    rung_paths = {
        "manifest": rung / str(expected["rung_manifest"]),
        "summary": rung / str(expected["rung_summary"]),
        "terminal": rung / str(expected["rung_terminal"]),
        "resolved_config": rung / str(expected["rung_resolved_config"]),
        "tail_service_summary": rung / str(expected["tail_service_summary"]),
    }
    hash_keys = {
        "manifest": "rung_manifest_sha256",
        "summary": "rung_summary_sha256",
        "terminal": "rung_terminal_sha256",
        "resolved_config": "rung_resolved_config_sha256",
        "tail_service_summary": "tail_service_summary_sha256",
    }
    for key, path in rung_paths.items():
        require(n2.sha256(path) == expected[hash_keys[key]],
                f"sealed -3.5 rung {key} drift")
    rung_manifest = load_json(rung_paths["manifest"])
    summary, terminal = load_json(rung_paths["summary"]), load_json(rung_paths["terminal"])
    require(rung_manifest.get("status") == terminal.get("status")
            == "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY",
            "sealed -3.5 rung status drift")
    require(terminal.get("manifest_sha256") == expected["rung_manifest_sha256"],
            "sealed -3.5 terminal does not bind its manifest")
    _verify_manifest_inventory(
        rung,
        rung_manifest,
        expected_output_count=int(expected["rung_manifest_output_count"]),
        strict_complete=True,
    )
    tail, service, recovery = summary["tail"], summary["tail_service"], summary["clean_recovery"]
    require(math.isclose(float(summary["commanded_noise_power_db"]), CANDIDATE_COMMAND_DB),
            "command-ladder rung is not -3.5")
    require(math.isclose(float(tail["achieved_pusch_snr_db_p05"]), 7.0)
            and math.isclose(float(tail["achieved_pusch_snr_db_median"]), 7.5)
            and math.isclose(float(tail["achieved_pusch_snr_db_p95"]), 7.5),
            "sealed -3.5 achieved-SNR evidence drift")
    require(int(tail["pusch_samples"]) == 699 and int(tail["mcs_samples"]) == 699,
            "sealed -3.5 telemetry sample-count drift")
    require(int(tail["mcs_table"]) == 0 and int(tail["force_ul_mcs"]) == -1,
            "sealed -3.5 scheduler seal drift")
    require(int(service["received_frames"]) == 50
            and int(service["expected_frames"]) == 50
            and service["primary_99_pass"] is True
            and service["no_one_second_outage_pass"] is True,
            "sealed -3.5 short-service evidence drift")
    require(recovery.get("passed") is True
            and summary.get("clean_restore_verified") is True,
            "sealed -3.5 recovery evidence drift")
    require(summary.get("candidate_application_count") == 1,
            "warm-rung provenance unexpectedly lacks its one Telnet application")
    require(summary.get("target_mapping_promoted") is False
            and summary.get("numeric_bound_promoted") is False,
            "sealed -3.5 rung unexpectedly promoted a claim")
    proof = {
        "status": "VERIFIED_CONSISTENCY_EXPECTATION_ONLY_NOT_MAPPING",
        "directory": str(rung),
        "campaign_manifest_sha256": n2.sha256(manifest_path),
        "rung_manifest_sha256": n2.sha256(rung_paths["manifest"]),
        "rung_summary_sha256": n2.sha256(rung_paths["summary"]),
        "rung_terminal_sha256": n2.sha256(rung_paths["terminal"]),
        "commanded_noise_power_db": CANDIDATE_COMMAND_DB,
        "achieved_pusch_snr_db_p05": 7.0,
        "achieved_pusch_snr_db_p50": 7.5,
        "achieved_pusch_snr_db_p95": 7.5,
        "pusch_samples": 699,
        "mcs_samples": 699,
        "service_frames_received": 50,
        "service_frames_expected": 50,
        "warm_already_attached_session": True,
        "cold_attach_evidence": False,
        "mapping_promoted": False,
        "numeric_bound_promoted": False,
        "verified_at": n2.utc_now(),
    }
    if proof_path is not None:
        n2.atomic_json(proof_path, proof)
    return proof


def campaign_plan_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(pending_adjudication(config["predecessors"]["n3c_adjudication"]),
            "N3C adjudication must remain pending in PREPARE v1")
    return [
        {
            "sequence_index": index - 1,
            "repetition_index": index,
            "commanded_noise_power_db": CANDIDATE_COMMAND_DB,
            "candidate_direction": "STRONGER_RFSIM_CONDITION_THAN_MINUS3P0",
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
            "post_restore_recovery_required_for_attached_trial": True,
            "automatic_next_candidate_authorized": False,
            "status": "BLOCKED_PENDING_N3C_ADJUDICATION_AND_DISTINCT_LIVE_REVIEW",
        }
        for index in range(1, int(config["campaign"]["repetitions"]) + 1)
    ]


class RecoveryCapable(Protocol):
    def verify_recovery(
        self, baseline_count: int, receiver_baseline_count: int, *, required: bool = True,
    ) -> Mapping[str, Any]: ...


def verify_post_restore_recovery_fail_closed(
    runner: RecoveryCapable, baseline_count: int, receiver_baseline_count: int,
) -> Mapping[str, Any]:
    result = runner.verify_recovery(
        baseline_count, receiver_baseline_count, required=True,
    )
    require(result.get("passed") is True,
            "clean -50 recovery is infrastructure-invalid evidence")
    return result


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
    recovery_passed = recovery.get("passed") is True
    infrastructure_invalid = bool(base_valid and attach_passed and not recovery_passed)
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
        outcome, valid, passed = "CLEAN_RECOVERY_INFRASTRUCTURE_INVALID", False, False
    elif base_valid and attach_passed and exact_window and recovery_passed:
        if delivery_pass and achieved_snr_gate:
            outcome, valid, passed = "COLD_ATTACH_AND_CANDIDATE_SERVICE_CONFIRMED", True, True
        elif delivery_pass:
            outcome, valid, passed = "ACHIEVED_SNR_OUTSIDE_CONSISTENCY_BAND", True, False
        else:
            outcome, valid, passed = "SERVICE_GATE_FAILED", True, False
    elif (
        base_valid and attach_passed and transport.get("integrity_gate") is True
        and recognized_loss and corroborated_loss and recovery_passed
    ):
        outcome, valid, passed = "HARD_SERVICE_LOSS_AFTER_COLD_ATTACH", True, False
    else:
        outcome, valid, passed = "EVIDENCE_UNCONFIRMED", False, False
    attach_nonconfirmation = outcome == "COLD_ATTACH_FAILED"
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
        "post_restore_recovery_required": attach_passed,
        "post_restore_recovery_passed": recovery_passed,
        "post_restore_clean_reattach_evaluated": False if attach_nonconfirmation else None,
        "candidate_causal_attach_failure_confirmed": False,
        "attach_failure_evidence_role": (
            "VALID_OPERATIONAL_SCREEN_NONCONFIRMATION_NOT_CANDIDATE_CAUSAL_OR_"
            "PHYSICAL_BOUNDARY_PROOF" if attach_nonconfirmation else None
        ),
        "infrastructure_invalid": infrastructure_invalid,
        "recognized_service_loss": recognized_loss and corroborated_loss,
        "review_before_next_action_required": not passed,
        "review_before_promotion_required": True,
        "automatic_next_candidate_authorized": False,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "connectivity_bound_promoted": False,
        "usable_service_bound_promoted": False,
        "operational_bound_promoted": False,
    }


class CampaignRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        validate_output_contract(self.config)
        validate_config(self.config, verify_hashes=True, require_live=False)
        self.output_dir = output_dir.resolve()
        for relative in PROTECTED_SOURCE_DIRECTORIES:
            source = resolve_repo_path(relative)
            try:
                self.output_dir.relative_to(source)
            except ValueError:
                pass
            else:
                raise ColdAttachRefinementFailure(
                    f"output directory is inside sealed source evidence: {source}"
                )
        if self.output_dir.exists():
            raise ColdAttachRefinementFailure(
                f"create-only output already exists: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True)

    def manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        n2.atomic_json(self.output_dir / EXPECTED_OUTPUT["campaign_summary"], summary)
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.output_dir)
            if relative == Path(EXPECTED_OUTPUT["manifest"]):
                continue
            if path.parent == self.output_dir and (
                path.name.startswith("UE_N3D_") or path.name == "FAILED.json"
            ):
                continue
            files.append({
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": n2.sha256(path),
            })
        manifest_path = self.output_dir / EXPECTED_OUTPUT["manifest"]
        n2.atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3d_cold_attach_refinement_plan_manifest.v1",
            "status": status,
            "config_sha256": n2.sha256(self.config_path),
            "runner_sha256": n2.sha256(Path(__file__).resolve()),
            "runtime_executed": False,
            "socket_executed": False,
            "carla_executed": False,
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
        terminal_name = EXPECTED_OUTPUT["failure"] if status == FAILED else f"{status}.json"
        n2.atomic_json(self.output_dir / terminal_name, terminal)

    def prepare(self) -> int:
        try:
            n2.atomic_json(self.output_dir / EXPECTED_OUTPUT["resolved_config"], self.config)
            rows = campaign_plan_rows(self.config)
            calibration.write_csv(self.output_dir / EXPECTED_OUTPUT["campaign_plan"], rows)
            n3c_proof = verify_n3c_live(
                self.config,
                proof_path=self.output_dir / "n3c_live_evidence_predecessor.json",
            )
            expectation = verify_command_ladder_expectation(
                self.config,
                proof_path=self.output_dir / "command_ladder_expectation_predecessor.json",
            )
            require(pending_adjudication(self.config["predecessors"]["n3c_adjudication"]),
                    "N3C adjudication is not explicitly pending")
            summary = {
                "status": PLAN_BLOCKED,
                "runtime_executed": False,
                "socket_executed": False,
                "carla_executed": False,
                "n3d_executed": False,
                "repetitions_planned": len(rows),
                "plan": rows,
                "n3c_live_predecessor": n3c_proof,
                "n3c_adjudication_predecessor_pending": True,
                "n3c_adjudication_predecessor": None,
                "command_ladder_expectation_predecessor": expectation,
                "live_authority_ready": False,
                "live_execution_blocked": True,
                "live_engine_status": "NOT_IMPLEMENTED_IN_PREPARE_ONLY_VERSION",
                "candidate_commanded_noise_power_db": CANDIDATE_COMMAND_DB,
                "candidate_direction": "STRONGER_RFSIM_CONDITION_THAN_MINUS3P0",
                "cold_attach_bound_evaluated": False,
                "expected_achieved_pusch_snr_db": EXPECTED_ACHIEVED_PUSCH_SNR_DB,
                "achieved_snr_tolerance_db": ACHIEVED_SNR_TOLERANCE_DB,
                "achieved_snr_expectation_role": "CONSISTENCY_ONLY_NOT_PROMOTED",
                "automatic_next_candidate_authorized": False,
                "supersedes_pre_final_plan": SUPERSEDED_PRE_FINAL_PLAN,
                "authoritative_prepare_source_for_live_pinning": True,
                "next": "SEAL_N3C_ADJUDICATION_THEN_CREATE_AND_REVIEW_DISTINCT_LIVE_VERSION",
            }
            self.manifest_terminal(PLAN_BLOCKED, summary)
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest_terminal(FAILED, {
                "status": FAILED,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime_executed": False,
                "socket_executed": False,
                "carla_executed": False,
                "n3d_executed": False,
                "cold_attach_bound_evaluated": False,
            })
            return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=(PREPARE_ONLY,), default=PREPARE_ONLY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return CampaignRunner(Path(args.config), Path(args.output_dir)).prepare()


if __name__ == "__main__":
    raise SystemExit(main())

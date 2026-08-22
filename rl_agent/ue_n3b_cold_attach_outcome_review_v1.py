#!/usr/bin/env python3
"""Create-only review of the sealed UE-N3B cold-attach outcome.

This offline tool verifies the complete N3B live_01 evidence tree, keeps
commanded RFsim noise power separate from achieved PUSCH SNR, and determines
whether the adjacent stronger command is eligible for a future N3C plan.  It
does not open sockets, run CARLA/OAI, execute N3C, or promote any bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3b_cold_attach_outcome_review_v1.json"
SCHEMA = "scenesense.ue_n3b_cold_attach_outcome_review_config.v1"
SUCCESS = "UE_N3B_OUTCOME_ADJUDICATED_N3C_ELIGIBLE_REVIEW_REQUIRED"
UNRESOLVED = "UE_N3B_OUTCOME_ADJUDICATION_UNRESOLVED_REVIEW_REQUIRED"

OUTPUT = {
    "resolved_config": "resolved_config.json",
    "source_verification": "source_evidence_verification.json",
    "adjudications_json": "per_repetition_adjudication.json",
    "adjudications_csv": "per_repetition_adjudication.csv",
    "summary": "n3b_outcome_review.json",
    "report": "REPORT.md",
    "manifest": "manifest.json",
    "failure": "FAILED.json",
}

SOURCE = {
    "directory": (
        "rl_agent/experiments/ue_n3b_oai_ul_cold_attach_confirmation_v1/"
        "20260821_live_01"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "ac76763ea9651212f0003c35eb19092d85dbf28b104572e9a9ffc107cb298f3a"
    ),
    "terminal": "UE_N3B_COLD_ATTACH_CONFIRMATION_NOT_3_OF_3_REVIEW_REQUIRED.json",
    "terminal_sha256": (
        "50066ffcba4137f585a55ad8d9ebcc728a7ca0e8cceec9728c0b8cc7f644c183"
    ),
    "campaign_summary": "campaign_summary.json",
    "campaign_summary_sha256": (
        "4115431fa6ea396a6ad5e72cb99f2bc2bfaabdfefd4d76286fb7c7f70f440563"
    ),
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": (
        "fdc46f717392b0b71fff9fc6f66d0afc9da831e48ce41cc566af909bb61d7072"
    ),
    "required_status": "UE_N3B_COLD_ATTACH_CONFIRMATION_NOT_3_OF_3_REVIEW_REQUIRED",
    "source_config_sha256": (
        "72583ccbf56347d65f8a3c937505b932cc08670064097e4f7388bccd347a33e4"
    ),
    "source_runner_sha256": (
        "3cb0b1e975ae58f9fffc510ec20622bded9237848498f5d3c13bfd0b0f3b57f7"
    ),
    "source_engine_runner_sha256": (
        "30cf1615f51c7cd0ebe4087f7b6ca66f37a563f2d5c452e474ae838a23c8878b"
    ),
    "expected_root_output_count": 71,
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
    "rung_directory": "rungs/rung_02_minus3p0",
    "rung_manifest": "manifest.json",
    "rung_manifest_sha256": (
        "0913985a99d3765a9c88d824ddb2ef38079085826dfded9dce583e621db7f302"
    ),
    "rung_terminal": "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY.json",
    "rung_summary": "rung_summary.json",
    "required_rung_status": "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY",
    "expected_rung_output_count": 103,
    "commanded_noise_power_db": -3.0,
    "observed_hot_pusch_snr_db_p05": 6.0,
    "observed_hot_pusch_snr_db_p50": 6.5,
    "observed_hot_pusch_snr_db_p95": 7.0,
    "provenance_role": "EXPECTATION_ONLY_ALREADY_ATTACHED_HOT_RUNG_NOT_COLD_ATTAINED",
}

EXPECTED_REPETITIONS = (
    (
        "rep_01_minus2p5_cold", 1,
        "515ffd44044b17f84605877e5b1b4fe514bcca2dc254270647fc90ab2476ad76",
        "d5b6598b5f1ce1b272c72aa0f70541fce0bd3f88da08fe10c4453495fe05d3b4",
        "5107dbce85e347479f3ba039a3e35f36d4e0f956d74769d678646c78732b5712",
    ),
    (
        "rep_02_minus2p5_cold", 2,
        "2240bc109df6c7cfe224dc56b565d1893431a9fa545e19131c8a6a9069ea271b",
        "ad30cf50b831986aaf877b882074d66a2ee3cdd358f2e4f9dc0f0473116bd584",
        "988bfa548a9f509e4e2de2d1fa2dddf1ea9db5e4430d364029f37618aafa042c",
    ),
    (
        "rep_03_minus2p5_cold", 3,
        "761bdddba720a5a1e3558664900f3df57cd5d7f1c51fabcd1453b4728a372376",
        "ba2afe9a0306bc4ddaf42ae26d11083cb72bf7fffc5891fc4bca1d9df7e38eea",
        "039e92aaa577e293a1a576e2982e9aa35cc332565f6b447b58f7f030e45421aa",
    ),
)

REP_STATUS = "UE_N3B_COLD_ATTACH_REPETITION_VALID_ATTACH_FAILURE"


class ReviewFailure(RuntimeError):
    """Raised for source-integrity, config, or evidence-validity failures."""


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ReviewFailure(f"path escapes repository root: {relative}") from exc
    return path


def _all_promotions_false(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "target_mapping_promoted", "numeric_bound_promoted",
        "operational_bound_promoted", "connectivity_bound_promoted",
        "usable_service_bound_promoted",
    ):
        require(payload.get(key, False) is False,
                f"{label} unexpectedly promotes {key}")


def verify_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    allowed_uninventoried: Sequence[str] = (),
) -> dict[str, str]:
    rows = list(manifest.get("outputs", []))
    require(rows, f"manifest has no output inventory: {directory}")
    observed: dict[str, str] = {}
    root = directory.resolve()
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
        observed_hash = sha256(artifact)
        require(observed_hash == row.get("sha256"),
                f"manifest output hash drift: {relative}")
        observed[relative] = observed_hash
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*") if path.is_file()
    }
    expected = set(observed) | set(allowed_uninventoried)
    require(actual == expected,
            "manifest inventory is not directory-complete: "
            f"unexpected={sorted(actual - expected)} missing={sorted(expected - actual)}")
    return observed


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    require(config.get("schema") == SCHEMA, "unexpected review config schema")
    require(config.get("experiment_id") == "ue_n3b_cold_attach_outcome_review_v1",
            "unexpected review experiment identity")
    require(
        config.get("claim_boundary")
        == "OFFLINE_N3B_OUTCOME_REVIEW_ONLY_N3C_ELIGIBILITY_NO_EXECUTION_NO_PROMOTION",
        "review claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_review_authorized") is True,
            "offline review authority is absent")
    for key in (
        "oai_run_authorized", "socket_execution_authorized", "carla_run_authorized",
        "n3c_execution_authorized", "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized", "operational_bound_promotion_authorized",
        "connectivity_bound_promotion_authorized",
        "usable_service_bound_promotion_authorized", "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    require(config.get("source") == SOURCE, "sealed N3B live_01 source drift")
    require(config.get("command_ladder") == COMMAND_LADDER,
            "command-ladder expectation provenance drift")
    contract = config["contract"]
    require(int(contract["repetitions"]) == 3, "three repetitions are required")
    require(math.isclose(float(contract["tested_commanded_noise_power_db"]), -2.5),
            "tested N3B command drift")
    require(int(contract["attach_timeout_s"]) == 180, "attach timeout drift")
    require(contract["expected_cold_attach_passes"] == 0,
            "sealed cold-attach outcome expectation drift")
    require(contract["cold_achieved_snr_must_be_null"] is True,
            "cold achieved-SNR null contract removed")
    require(math.isclose(float(contract["n3c_selected_command_db"]), -3.0),
            "adjacent stronger N3C candidate drift")
    require(
        contract["hot_cold_endpoint_separation"]
        == "ALREADY_ATTACHED_SERVICE_DOES_NOT_ESTABLISH_COLD_ATTACH",
        "hot/cold endpoint separation drift",
    )
    require(contract["n3c_selection_is_eligibility_only"] is True,
            "N3C selection scope drift")
    rows = config["expected_repetitions"]
    require(len(rows) == 3, "expected repetition inventory must have three rows")
    observed = tuple(
        (
            row["directory"], int(row["repetition_index"]),
            row["manifest_sha256"], row["terminal_sha256"], row["summary_sha256"],
        )
        for row in rows
    )
    require(observed == EXPECTED_REPETITIONS, "expected repetition seals drift")
    require(config.get("output") == OUTPUT, "create-only output contract drift")
    seals = config["runtime_seals"]
    require(
        len(seals) == 1
        and seals[0]["path"] == "rl_agent/ue_n3b_cold_attach_outcome_review_v1.py",
        "review runtime seal set drift",
    )
    if verify_hashes:
        runner = resolve_repo_path(str(seals[0]["path"]))
        require(runner.is_file() and sha256(runner) == seals[0]["sha256"],
                "review tool runtime seal drift")


def _verify_command_ladder(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["command_ladder"]
    directory = resolve_repo_path(str(source["directory"]))
    manifest_path = directory / source["manifest"]
    require(manifest_path.is_file(), "command-ladder manifest is missing")
    require(sha256(manifest_path) == source["manifest_sha256"],
            "command-ladder campaign manifest seal drift")
    manifest = load_json(manifest_path)
    require(manifest.get("schema")
            == "scenesense.ue_n3_command_calibration_campaign_manifest.v1",
            "command-ladder manifest schema mismatch")
    require(manifest.get("status") == source["required_status"],
            "command-ladder status mismatch")
    _all_promotions_false(manifest, "command-ladder campaign")
    root_inventory = verify_inventory(
        directory,
        manifest,
        allowed_uninventoried=(source["manifest"], f"{source['required_status']}.json"),
    )
    require(len(root_inventory) == int(source["expected_root_output_count"]),
            "command-ladder root output count drift")
    rung_dir = directory / source["rung_directory"]
    rung_manifest_path = rung_dir / source["rung_manifest"]
    require(rung_manifest_path.is_file(), "-3.0 rung manifest is missing")
    require(sha256(rung_manifest_path) == source["rung_manifest_sha256"],
            "-3.0 rung manifest seal drift")
    relative_rung_manifest = str(rung_manifest_path.relative_to(directory))
    require(root_inventory.get(relative_rung_manifest) == source["rung_manifest_sha256"],
            "command-ladder root does not bind the -3.0 rung manifest")
    rung_manifest = load_json(rung_manifest_path)
    require(rung_manifest.get("schema")
            == "scenesense.ue_n3_command_calibration_rung_manifest.v1",
            "-3.0 rung manifest schema mismatch")
    require(rung_manifest.get("status") == source["required_rung_status"],
            "-3.0 rung status mismatch")
    _all_promotions_false(rung_manifest, "-3.0 rung")
    rung_inventory = verify_inventory(
        rung_dir,
        rung_manifest,
        allowed_uninventoried=(source["rung_manifest"], source["rung_terminal"]),
    )
    require(len(rung_inventory) == int(source["expected_rung_output_count"]),
            "-3.0 rung output count drift")
    summary_path = rung_dir / source["rung_summary"]
    require(rung_inventory.get(source["rung_summary"]) == sha256(summary_path),
            "-3.0 rung summary is not manifest-bound")
    summary = load_json(summary_path)
    require(summary.get("status") == source["required_rung_status"],
            "-3.0 rung summary status mismatch")
    require(math.isclose(float(summary.get("commanded_noise_power_db", math.nan)),
                         float(source["commanded_noise_power_db"]), abs_tol=1e-9),
            "-3.0 rung command mismatch")
    tail = dict(summary.get("tail") or {})
    require(tail.get("status") == "TAIL_ACCEPTED" and tail.get("mcs_seals_ok") is True,
            "-3.0 rung tail is not accepted")
    for key, expected in (
        ("achieved_pusch_snr_db_p05", source["observed_hot_pusch_snr_db_p05"]),
        ("achieved_pusch_snr_db_median", source["observed_hot_pusch_snr_db_p50"]),
        ("achieved_pusch_snr_db_p95", source["observed_hot_pusch_snr_db_p95"]),
    ):
        require(math.isclose(float(tail.get(key, math.nan)), float(expected), abs_tol=1e-9),
                f"-3.0 rung {key} drift")
    require(dict(summary.get("command") or {}).get("candidate_applied_once") is True,
            "-3.0 source is not a hot-applied command rung")
    require(summary.get("cold_attach_bound_evaluated", False) is False,
            "-3.0 expectation source unexpectedly evaluated cold attach")
    _all_promotions_false(summary, "-3.0 rung summary")
    return {
        "status": "SEALED_MINUS3_EXPECTATION_PROVENANCE_VERIFIED",
        "campaign_manifest_sha256": source["manifest_sha256"],
        "campaign_manifest_output_count": len(root_inventory),
        "rung_manifest_sha256": source["rung_manifest_sha256"],
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


def _validate_repetition_summary(
    summary: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    startup: Mapping[str, Any],
    integrity: Mapping[str, Any],
    *,
    repetition_index: int,
    command_db: float,
    attach_timeout_s: int,
) -> bool:
    attach = dict(summary.get("attach_gate") or {})
    outcome = dict(summary.get("outcome_classification") or {})
    models = dict(startup.get("models") or {})
    ue_model = dict(models.get("rfsimu_channel_ue0") or {})
    enb0 = dict(models.get("rfsimu_channel_enB0") or {})
    enb1 = dict(models.get("rfsimu_channel_enB1") or {})
    return all((
        summary.get("status") == REP_STATUS,
        int(summary.get("repetition_index", -1)) == repetition_index,
        math.isclose(float(summary.get("commanded_noise_power_db", math.nan)),
                     command_db, abs_tol=1e-9),
        summary.get("engine_status") == REP_STATUS,
        summary.get("classified_outcome") == "COLD_ATTACH_FAILED",
        summary.get("evidence_valid_for_aggregation") is True,
        summary.get("clean_attach_failure_evidence") is True,
        summary.get("cold_attach_gate_pass") is False,
        summary.get("authoritative_service_gate_pass") is False,
        summary.get("joint_candidate_confirmation_pass") is False,
        summary.get("candidate_baked_config_verified") is True,
        summary.get("startup_channel_runtime_verified") is True,
        int(summary.get("candidate_application_count", -1)) == 0,
        summary.get("candidate_application_count_zero") is True,
        int(summary.get("restore_application_count", -1)) == 1,
        summary.get("single_restore_application") is True,
        summary.get("clean_restore_verified") is True,
        summary.get("cleanup_clean") is True,
        summary.get("source_oai_configs_unchanged") is True,
        summary.get("exact_600_frame_service_evidence") is False,
        summary.get("service_tail") is None,
        summary.get("service_window") is None,
        summary.get("transport") is None,
        summary.get("achieved_pusch_snr_db_p05") is None,
        summary.get("achieved_pusch_snr_db_p50") is None,
        summary.get("achieved_pusch_snr_db_p95") is None,
        attach.get("passed") is False,
        attach.get("status") == "COLD_ATTACH_OR_PDU_EXT_DN_GATE_FAILED",
        int(attach.get("timeout_s", -1)) == attach_timeout_s,
        float(attach.get("duration_s", 0.0)) >= float(attach_timeout_s),
        attach.get("observed_ipv4") == [],
        attach.get("core_ready_at_terminal") is True,
        attach.get("ran_processes_alive_at_terminal") is True,
        outcome.get("classified_outcome") == "COLD_ATTACH_FAILED",
        outcome.get("evidence_valid_for_aggregation") is True,
        outcome.get("clean_attach_failure_evidence") is True,
        outcome.get("achieved_pusch_snr_db_p05") is None,
        outcome.get("achieved_pusch_snr_db_p50") is None,
        outcome.get("achieved_pusch_snr_db_p95") is None,
        startup.get("status") == "PASSED",
        startup.get("startup_channel_runtime_verified") is True,
        int(startup.get("candidate_application_count", -1)) == 0,
        math.isclose(float(ue_model.get("noise_power_db", math.nan)),
                     command_db, abs_tol=1e-9),
        math.isclose(float(enb0.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-9),
        math.isclose(float(enb1.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-9),
        cleanup.get("clean") is True,
        not cleanup.get("errors"),
        integrity.get("status") == "UNCHANGED",
        integrity.get("unchanged") is True,
        integrity.get("source_paths_written") is False,
        integrity.get("before") == integrity.get("after"),
    ))


def verify_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    contract = config["contract"]
    directory = resolve_repo_path(str(source["directory"]))
    manifest_path = directory / source["manifest"]
    terminal_path = directory / source["terminal"]
    summary_path = directory / source["campaign_summary"]
    resolved_path = directory / source["resolved_config"]
    for path, expected, label in (
        (manifest_path, source["manifest_sha256"], "N3B manifest"),
        (terminal_path, source["terminal_sha256"], "N3B terminal"),
        (summary_path, source["campaign_summary_sha256"], "N3B campaign summary"),
        (resolved_path, source["resolved_config_sha256"], "N3B resolved config"),
    ):
        require(path.is_file() and sha256(path) == expected, f"{label} seal drift")
    manifest, terminal, campaign, resolved = (
        load_json(manifest_path), load_json(terminal_path),
        load_json(summary_path), load_json(resolved_path),
    )
    require(manifest.get("schema") == "scenesense.ue_n3b_cold_attach_campaign_manifest.v1",
            "N3B campaign manifest schema mismatch")
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"),
                           (campaign, "summary")):
        require(payload.get("status") == source["required_status"],
                f"N3B {label} status mismatch")
        _all_promotions_false(payload, f"N3B {label}")
    require(terminal.get("manifest_sha256") == source["manifest_sha256"],
            "N3B terminal does not bind campaign manifest")
    require(manifest.get("config_sha256") == source["source_config_sha256"]
            and manifest.get("runner_sha256") == source["source_runner_sha256"]
            and manifest.get("engine_runner_sha256")
            == source["source_engine_runner_sha256"],
            "N3B campaign runtime provenance mismatch")
    root_inventory = verify_inventory(
        directory,
        manifest,
        allowed_uninventoried=(source["manifest"], source["terminal"]),
    )
    require(len(root_inventory) == int(source["expected_root_output_count"]),
            "N3B root output count drift")
    require(root_inventory.get(source["campaign_summary"])
            == source["campaign_summary_sha256"],
            "N3B root manifest does not bind campaign summary")
    require(root_inventory.get(source["resolved_config"])
            == source["resolved_config_sha256"],
            "N3B root manifest does not bind resolved config")
    require(resolved["campaign"]["repetitions"] == 3
            and resolved["campaign"]["candidate_application_count"] == 0,
            "N3B resolved campaign contract drift")
    require(resolved["startup_channel"]["rfsimu_channel_ue0"]
            == float(contract["tested_commanded_noise_power_db"]),
            "N3B startup candidate differs from review contract")
    require(resolved["radio"]["attach_timeout_s"] == int(contract["attach_timeout_s"]),
            "N3B attach timeout differs from review contract")
    require(campaign.get("runtime_executed") is True
            and campaign.get("socket_executed") is True,
            "N3B source was not a live campaign")
    require(int(campaign.get("repetitions_planned", -1)) == 3
            and int(campaign.get("repetitions_executed", -1)) == 3,
            "N3B source is not the complete three-repetition campaign")
    require(int(campaign.get("cold_attach_passes", -1)) == 0
            and campaign.get("cold_attach_3_of_3_pass") is False,
            "N3B campaign cold-attach endpoint drift")
    require(int(campaign.get("fresh_ran_epoch_count", -1)) == 3
            and int(campaign.get("unique_control_session_count", -1)) == 3,
            "N3B source lacks three fresh experiment identities")

    expected_names = {row[0] for row in EXPECTED_REPETITIONS}
    repetition_root = directory / "repetitions"
    actual_names = {path.name for path in repetition_root.iterdir() if path.is_dir()}
    require(actual_names == expected_names, "N3B repetition directory set drift")
    proofs = {
        Path(str(row["directory"])).name: row
        for row in campaign.get("repetition_evidence", [])
    }
    require(set(proofs) == expected_names, "N3B campaign repetition-proof set drift")
    epochs: set[str] = set()
    sessions: set[str] = set()
    verified: list[dict[str, Any]] = []
    for row in config["expected_repetitions"]:
        name = row["directory"]
        rep_dir = repetition_root / name
        rep_manifest_path = rep_dir / "manifest.json"
        rep_terminal_path = rep_dir / f"{REP_STATUS}.json"
        rep_summary_path = rep_dir / "repetition_summary.json"
        for path, expected, label in (
            (rep_manifest_path, row["manifest_sha256"], "manifest"),
            (rep_terminal_path, row["terminal_sha256"], "terminal"),
            (rep_summary_path, row["summary_sha256"], "summary"),
        ):
            require(path.is_file() and sha256(path) == expected,
                    f"{name} {label} seal drift")
        rep_manifest, rep_terminal, rep_summary = (
            load_json(rep_manifest_path), load_json(rep_terminal_path),
            load_json(rep_summary_path),
        )
        require(rep_manifest.get("schema")
                == "scenesense.ue_n3b_cold_attach_repetition_manifest.v1",
                f"{name} manifest schema mismatch")
        for payload, label in ((rep_manifest, "manifest"),
                               (rep_terminal, "terminal"),
                               (rep_summary, "summary")):
            require(payload.get("status") == REP_STATUS,
                    f"{name} {label} status mismatch")
            require(int(payload.get("repetition_index", -1))
                    == int(row["repetition_index"]),
                    f"{name} {label} repetition index mismatch")
            require(math.isclose(
                float(payload.get("commanded_noise_power_db", math.nan)),
                float(contract["tested_commanded_noise_power_db"]), abs_tol=1e-9,
            ), f"{name} {label} command mismatch")
            _all_promotions_false(payload, f"{name} {label}")
        require(rep_terminal.get("manifest_sha256") == row["manifest_sha256"],
                f"{name} terminal does not bind nested manifest")
        require(rep_manifest.get("config_sha256") == source["source_config_sha256"]
                and rep_manifest.get("runner_sha256") == source["source_runner_sha256"]
                and rep_manifest.get("engine_runner_sha256")
                == source["source_engine_runner_sha256"],
                f"{name} runtime provenance mismatch")
        require(int(rep_manifest.get("candidate_application_count", -1)) == 0,
                f"{name} candidate was not solely baked at startup")
        rep_inventory = verify_inventory(
            rep_dir,
            rep_manifest,
            allowed_uninventoried=("manifest.json", f"{REP_STATUS}.json"),
        )
        require(len(rep_inventory) == 20, f"{name} nested output count drift")
        required = {
            "repetition_summary.json", "attach_gate.json", "cleanup_report.json",
            "startup_channel_runtime_gate.json", "source_oai_config_integrity.json",
            "cold_start_identity.json", "runtime_seals.json",
        }
        require(required.issubset(rep_inventory),
                f"{name} nested manifest lacks required evidence")
        for relative, expected_hash in (
            (f"repetitions/{name}/manifest.json", row["manifest_sha256"]),
            (f"repetitions/{name}/{REP_STATUS}.json", row["terminal_sha256"]),
            (f"repetitions/{name}/repetition_summary.json", row["summary_sha256"]),
        ):
            require(root_inventory.get(relative) == expected_hash,
                    f"N3B root manifest does not bind {relative}")
        proof = proofs[name]
        require(proof.get("manifest_sha256") == row["manifest_sha256"]
                and proof.get("terminal_sha256") == row["terminal_sha256"],
                f"N3B campaign proof mismatch for {name}")
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
        startup = load_json(rep_dir / "startup_channel_runtime_gate.json")
        integrity = load_json(rep_dir / "source_oai_config_integrity.json")
        evidence_valid = _validate_repetition_summary(
            rep_summary, cleanup, startup, integrity,
            repetition_index=int(row["repetition_index"]),
            command_db=float(contract["tested_commanded_noise_power_db"]),
            attach_timeout_s=int(contract["attach_timeout_s"]),
        )
        require(evidence_valid, f"{name} is not valid cold-attach failure evidence")
        verified.append({
            "directory": name,
            "repetition_index": int(row["repetition_index"]),
            "status": REP_STATUS,
            "manifest_sha256": row["manifest_sha256"],
            "terminal_sha256": row["terminal_sha256"],
            "summary_sha256": row["summary_sha256"],
            "manifest_output_count": len(rep_inventory),
            "ran_epoch_id": epoch,
            "control_session_id": session,
            "valid_cold_attach_failure_evidence": True,
        })
    ladder = _verify_command_ladder(config)
    return {
        "status": "SEALED_N3B_LIVE_01_AND_MINUS3_EXPECTATION_PROVENANCE_VERIFIED",
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
    valid = _validate_repetition_summary(
        summary, cleanup, startup, integrity,
        repetition_index=int(summary.get("repetition_index", -1)),
        command_db=float(contract["tested_commanded_noise_power_db"]),
        attach_timeout_s=int(contract["attach_timeout_s"]),
    )
    attach = dict(summary.get("attach_gate") or {})
    achieved_values = (
        summary.get("achieved_pusch_snr_db_p05"),
        summary.get("achieved_pusch_snr_db_p50"),
        summary.get("achieved_pusch_snr_db_p95"),
    )
    cold_snr_null = all(value is None for value in achieved_values)
    outcome = "VALID_COLD_ATTACH_FAILURE" if valid else "INVALID_EVIDENCE"
    return {
        "repetition_index": int(summary.get("repetition_index", -1)),
        "commanded_noise_power_db": float(summary.get("commanded_noise_power_db", math.nan)),
        "adjudicated_outcome": outcome,
        "evidence_valid": valid,
        "cold_attach_pass": False if valid else None,
        "attach_pdu_ext_dn_timeout_s": attach.get("timeout_s"),
        "attach_gate_duration_s": attach.get("duration_s"),
        "core_ready_at_terminal": attach.get("core_ready_at_terminal"),
        "ran_processes_alive_at_terminal": attach.get("ran_processes_alive_at_terminal"),
        "observed_ipv4": attach.get("observed_ipv4"),
        "candidate_baked_before_ue_launch": summary.get("candidate_baked_config_verified"),
        "candidate_application_count": summary.get("candidate_application_count"),
        "single_restore_application": summary.get("single_restore_application"),
        "clean_restore_verified": summary.get("clean_restore_verified"),
        "cleanup_verified": cleanup.get("clean") is True and not cleanup.get("errors"),
        "source_oai_configs_unchanged": integrity.get("unchanged") is True,
        "cold_achieved_pusch_snr_db_p05": achieved_values[0],
        "cold_achieved_pusch_snr_db_p50": achieved_values[1],
        "cold_achieved_pusch_snr_db_p95": achieved_values[2],
        "cold_achieved_snr_is_null": cold_snr_null,
        "cold_achieved_snr_interpretation": (
            "UNOBSERVED_NO_SERVING_RNTI_PUSCH_WINDOW"
            if cold_snr_null else "UNEXPECTED_COLD_SNR_VALUE_REVIEW_REQUIRED"
        ),
        "physical_rf_cutoff_established": False,
        "initial_access_or_pdu_gate_failed": valid,
    }


def aggregate_adjudications(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    ladder: Mapping[str, Any],
) -> dict[str, Any]:
    required = int(contract["repetitions"])
    require(len(rows) == required, "adjudication repetition count drift")
    valid = [row for row in rows if row["evidence_valid"]]
    passes = sum(row["cold_attach_pass"] is True for row in valid)
    failures = sum(row["cold_attach_pass"] is False for row in valid)
    all_cold_snr_null = all(row["cold_achieved_snr_is_null"] for row in valid)
    accepted = (
        len(valid) == required
        and passes == int(contract["expected_cold_attach_passes"])
        and failures == required
        and all_cold_snr_null
    )
    return {
        "status": SUCCESS if accepted else UNRESOLVED,
        "n3b_outcome_accepted": accepted,
        "tested_commanded_noise_power_db": float(contract["tested_commanded_noise_power_db"]),
        "cold_attach_repetitions": required,
        "valid_repetitions": len(valid),
        "cold_attach_passes": passes,
        "cold_attach_failures": failures,
        "cold_attach_result": "0_OF_3_PASS" if accepted else "UNRESOLVED",
        "cold_achieved_pusch_snr_db_p05": None,
        "cold_achieved_pusch_snr_db_p50": None,
        "cold_achieved_pusch_snr_db_p95": None,
        "cold_achieved_snr_status": (
            "UNOBSERVED_NO_SERVING_RNTI_PUSCH_WINDOW"
            if accepted else "UNRESOLVED"
        ),
        "commanded_vs_achieved_separation": (
            "MINUS2P5_IS_RFSIM_COMMAND_NOT_COLD_ACHIEVED_PUSCH_SNR"
        ),
        "hot_cold_endpoint_separation": contract["hot_cold_endpoint_separation"],
        "hot_service_endpoint_status": (
            "SEPARATE_ALREADY_ATTACHED_PREDECESSOR_NOT_RE_ADJUDICATED_OR_PROMOTED_HERE"
        ),
        "cold_attach_endpoint_status": (
            "VALID_FAILURE_AT_MINUS2P5_COMMAND_COLD_ACHIEVED_SNR_UNOBSERVED"
            if accepted else "UNRESOLVED"
        ),
        "physical_rf_cutoff_status": "NOT_ESTABLISHED",
        "hard_loss_boundary_status": "NOT_ESTABLISHED",
        "l_attach_status": "UNRESOLVED_COLD_ATTACH_FAILED_AT_MINUS2P5_COMMAND",
        "l_operational_status": (
            "UNRESOLVED_REQUIRES_COLD_ATTACH_AND_SUSTAINED_SERVICE_CONFIRMATION"
        ),
        "n3c_selected_command_db": (
            float(contract["n3c_selected_command_db"]) if accepted else None
        ),
        "n3c_eligibility_status": (
            "UE_N3B_VALID_COLD_ATTACH_FAILURE_ACCEPTED_FOR_N3C"
            if accepted else "N3C_ELIGIBILITY_UNRESOLVED"
        ),
        "n3c_selection_scope": "ELIGIBILITY_ONLY",
        "n3c_selection_basis": "ADJACENT_STRONGER_COMMAND",
        "n3c_expectation_provenance": dict(ladder),
        "n3c_expected_achieved_snr_role": (
            "EXPECTATION_ONLY_NOT_COLD_ATTAINED_NOT_A_PROMOTED_MAPPING"
        ),
        "n3c_execution_authorized": False,
        "n3c_executed": False,
        "target_mapping_promoted": False,
        "numeric_bound_promoted": False,
        "operational_bound_promoted": False,
        "connectivity_bound_promoted": False,
        "usable_service_bound_promoted": False,
    }


class ReviewRunner:
    def __init__(self, config_path: Path, output_dir: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        self.output_dir = output_dir.resolve()
        for protected in (
            resolve_repo_path(SOURCE["directory"]),
            resolve_repo_path(COMMAND_LADDER["directory"]),
        ):
            try:
                self.output_dir.relative_to(protected)
            except ValueError:
                pass
            else:
                raise ReviewFailure(
                    f"output directory is inside sealed source evidence: {protected}"
                )
        if self.output_dir.exists():
            raise ReviewFailure(f"create-only output already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True)

    def write_manifest_terminal(self, status: str, summary: Mapping[str, Any]) -> None:
        atomic_json(self.output_dir / OUTPUT["summary"], summary)
        excluded = {OUTPUT["manifest"], OUTPUT["failure"]}
        files = []
        for path in sorted(self.output_dir.rglob("*")):
            if (not path.is_file() or path.name in excluded
                    or path.name.startswith("UE_N3B_")):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        manifest_path = self.output_dir / OUTPUT["manifest"]
        atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3b_cold_attach_outcome_review_manifest.v1",
            "status": status,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_n3b_manifest_sha256": SOURCE["manifest_sha256"],
            "command_ladder_manifest_sha256": COMMAND_LADDER["manifest_sha256"],
            "command_ladder_minus3_rung_manifest_sha256": (
                COMMAND_LADDER["rung_manifest_sha256"]
            ),
            "offline_only": True,
            "n3c_execution_authorized": False,
            "n3c_executed": False,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "operational_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "outputs": files,
        })
        terminal = {
            **dict(summary),
            "status": status,
            "offline_only": True,
            "n3c_execution_authorized": False,
            "n3c_executed": False,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "operational_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "manifest_sha256": sha256(manifest_path),
        }
        terminal_name = OUTPUT["failure"] if status == "FAILED" else f"{status}.json"
        atomic_json(self.output_dir / terminal_name, terminal)

    def run(self) -> int:
        try:
            validate_config(self.config, verify_hashes=True)
            atomic_json(self.output_dir / OUTPUT["resolved_config"], self.config)
            verification = verify_source(self.config)
            atomic_json(self.output_dir / OUTPUT["source_verification"], verification)
            source_dir = resolve_repo_path(self.config["source"]["directory"])
            adjudications = []
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
                adjudications.append(row)
            atomic_json(
                self.output_dir / OUTPUT["adjudications_json"],
                {"schema": "scenesense.ue_n3b_cold_attach_adjudications.v1",
                 "rows": adjudications},
            )
            write_csv(self.output_dir / OUTPUT["adjudications_csv"], adjudications)
            ladder = verification["command_ladder_expectation_provenance"]
            aggregate = aggregate_adjudications(
                adjudications, self.config["contract"], ladder
            )
            status = str(aggregate["status"])
            selection = (
                "- N3C eligibility: command -3.0 dB, as the adjacent stronger command; "
                "N3C execution is not authorized.\n"
                if aggregate["n3b_outcome_accepted"] else
                "- N3C eligibility is unresolved.\n"
            )
            report = (
                "# UE-N3B cold-attach outcome review\n\n"
                f"- Contract status: `{status}`\n"
                "- RFsim command -2.5 dB cold-attach/PDU passes: "
                f"{aggregate['cold_attach_passes']}/3; failures: "
                f"{aggregate['cold_attach_failures']}/3.\n"
                "- Cold achieved PUSCH SNR: unobserved (null); -2.5 dB is the "
                "RFsim command, not a cold achieved-SNR measurement.\n"
                "- The already-attached hot-service endpoint and cold-attach endpoint "
                "remain separate.\n"
                "- L_attach and L_operational remain unresolved; no physical RF cutoff "
                "or hard-loss boundary is claimed.\n"
                f"{selection}"
                "- The prior -3.0 dB hot rung (PUSCH p05/p50/p95 6.0/6.5/7.0 dB) "
                "is expectation-only provenance, not cold evidence or a promoted mapping.\n"
                "- No target mapping or numeric, operational, connectivity, or usable-service "
                "bound is promoted.\n"
            )
            atomic_text(self.output_dir / OUTPUT["report"], report)
            summary = {
                **aggregate,
                "source_verification": verification,
                "adjudications": adjudications,
                "review_scope": "SEALED_N3B_COLD_ATTACH_OUTCOME_AND_N3C_ELIGIBILITY_ONLY",
                "review_required": True,
            }
            self.write_manifest_terminal(status, summary)
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
                "n3c_execution_authorized": False,
                "n3c_executed": False,
                "target_mapping_promoted": False,
                "numeric_bound_promoted": False,
                "operational_bound_promoted": False,
                "connectivity_bound_promoted": False,
                "usable_service_bound_promoted": False,
            }
            self.write_manifest_terminal("FAILED", failure)
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

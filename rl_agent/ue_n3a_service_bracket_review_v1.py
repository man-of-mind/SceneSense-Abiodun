#!/usr/bin/env python3
"""Offline, create-only review of the sealed UE-N3A live_02 campaign.

The review adjudicates the frozen usable-service endpoint independently from
the pre-registered expectation about *how* service would fail.  It never opens
sockets, starts CARLA/OAI, promotes a bound, or authorizes N3B execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3a_service_bracket_review_v1.json"
SCHEMA = "scenesense.ue_n3a_service_bracket_review_config.v1"
SUCCESS = "UE_N3_BOUND_BRACKETED_REVIEW_REQUIRED"
UNRESOLVED = "UE_N3_SERVICE_BRACKET_UNRESOLVED_REVIEW_REQUIRED"

OUTPUT = {
    "resolved_config": "resolved_config.json",
    "source_verification": "source_evidence_verification.json",
    "adjudications_json": "per_repetition_adjudication.json",
    "adjudications_csv": "per_repetition_adjudication.csv",
    "summary": "service_bracket_review.json",
    "report": "REPORT.md",
    "manifest": "manifest.json",
    "failure": "FAILED.json",
}

SOURCE = {
    "directory": (
        "rl_agent/experiments/ue_n3a_oai_ul_sustain_replication_v1/"
        "20260821_live_02"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "62639405273bd77aba5ff345bba2e2f99d2f15dfe962aa54083d5142c1b1b6ce"
    ),
    "terminal": "UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED.json",
    "terminal_sha256": (
        "b05f8537cf671984545d426d58b2efea923dc348b052e20f0eddaaa18625b798"
    ),
    "campaign_summary": "campaign_summary.json",
    "campaign_summary_sha256": (
        "0f5ecb7ee1b2dba6cd9c94a787a74e8fbca1183754b573229212a88e8929cb91"
    ),
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": (
        "44343c9212f7c005887b252f27027ca044080c19033b4d2e33822a09acc8c4ed"
    ),
    "required_status": "UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED",
    "source_config_sha256": (
        "f1afb806478d0fc17d381b2dd2f119173d4745696abff90be945bce1bdd71358"
    ),
    "source_runner_sha256": (
        "b72f98d1e492840fde074772e9e90a1831c4b0f8d777795b29b27a9e1a482487"
    ),
    "source_engine_runner_sha256": (
        "30cf1615f51c7cd0ebe4087f7b6ca66f37a563f2d5c452e474ae838a23c8878b"
    ),
}

FROZEN_PLAN = {
    "directory": (
        "rl_agent/experiments/ue_n3a_oai_ul_sustain_replication_v1/"
        "20260821_plan_02"
    ),
    "manifest": "manifest.json",
    "manifest_sha256": (
        "86497a46d87c369bc804f8bce0d034ecef26a5cd7fa141887289e736893ab802"
    ),
    "terminal": "UE_N3A_SUSTAIN_REPLICATION_PLAN_FROZEN_REVIEW_REQUIRED.json",
    "terminal_sha256": (
        "c1eee8c262308b47bb9ce5d3f9d395167e539c561e2106960f96854d92b1206e"
    ),
    "resolved_config": "resolved_config.json",
    "resolved_config_sha256": (
        "44343c9212f7c005887b252f27027ca044080c19033b4d2e33822a09acc8c4ed"
    ),
    "required_status": "UE_N3A_SUSTAIN_REPLICATION_PLAN_FROZEN_REVIEW_REQUIRED",
}

EXPECTED_REPETITIONS = (
    ("sequence_00_rep_01_minus2p5", 1, -2.5, "SUSTAIN_CANDIDATE_MINUS2P5",
     "UE_N3A_SUSTAINED_CANDIDATE_REPLICATION_PASSED",
     "a97c6544c2ca46931945cd22bdfbedee2e40032c00d550cbfb480f84068a816e",
     "46c2e7874eed6ea61b18d6eb786995375b241e19114795a593a1ea6a9f2248a4",
     "00f7c56508c0b931124c3eb9d3002ea8a9ae3475806f9a19f1f2668cfbaea490"),
    ("sequence_01_rep_01_minus2p0", 1, -2.0, "ADJACENT_HARD_LOSS_MINUS2P0",
     "UE_N3A_VALID_SURPRISE_OUTCOME_CAPTURED",
     "90a60fa77a6871b045d00a6de14b6bc0d283f7e46565ad3baa553c9c73859a13",
     "34dab362efc964df6b709be6c50c2fc3697606cac2370ec93cd9fe1d95b998d4",
     "405532dd0e8c0cba3fe85149b01e10011952b3b4f0a9d5dfd4fae1f8cd44e412"),
    ("sequence_02_rep_02_minus2p5", 2, -2.5, "SUSTAIN_CANDIDATE_MINUS2P5",
     "UE_N3A_SUSTAINED_CANDIDATE_REPLICATION_PASSED",
     "85b38bcfca058fc966026f70eeef4b1514ff06ef3f7decd4a324194d78b76d43",
     "c2c229c997c3e654ca01cbd43798bfed096297973716e190369bb67f4b62199b",
     "6308b31177b2ccee5bd2902456050e81f205e81848f57dedcd569485bb2da7a6"),
    ("sequence_03_rep_02_minus2p0", 2, -2.0, "ADJACENT_HARD_LOSS_MINUS2P0",
     "UE_N3A_VALID_SURPRISE_OUTCOME_CAPTURED",
     "5702d030349131e137d8a42ac7b8eb5c08bdc8fec84d5c0389d1b12d5226819b",
     "7331bd20382ac55dc416566833b9517c80df28aab4ecdbd54d898655952c0f01",
     "1e3b86a435e9ce24e2ec851b01a498cc4ed860fe1ffc75fb83af6158b2016f2f"),
    ("sequence_04_rep_03_minus2p5", 3, -2.5, "SUSTAIN_CANDIDATE_MINUS2P5",
     "UE_N3A_SUSTAINED_CANDIDATE_REPLICATION_PASSED",
     "f991dfc52f5ba1c6be3719ccee0f530059b646f699d0537289fc2636fabe7f6e",
     "4ec149fbd05858da05ffe12140f52a9b4008f3afcd7141b483d01b3a76867ae1",
     "64b37d8873f65a12b427963d6cf7eba0281b9217f5694dccd307541e64af36d9"),
    ("sequence_05_rep_03_minus2p0", 3, -2.0, "ADJACENT_HARD_LOSS_MINUS2P0",
     "UE_N3A_VALID_SURPRISE_OUTCOME_CAPTURED",
     "4e54cbaed8eb1f53fb1d797b33258b9e3bf32e7e76804776d5b41981964f958c",
     "a910491a21735d9c8ae1e7e2aad6cd96194433531ce4c0ea8ed36e4d2c2256b2",
     "5f4e2903b5cf542f1dfb72775bd62b719a6a0dd80ec034ef8011f7d22c0dcac2"),
)

RECOGNIZED_HARD_LOSS = {
    "HARD_SERVICE_LOSS_BEFORE_TARGET_CONFIRMATION": "CURRENT_RNTI_PUSCH_SILENCE",
    "DETACHED_BEFORE_TARGET_CONFIRMATION": "UE_TUNNEL_IDENTITY_LOST",
    "RNTI_IDENTITY_DISCONTINUITY_BEFORE_TARGET_CONFIRMATION": "RNTI_CHANGED",
}


class ReviewFailure(RuntimeError):
    """Source-integrity, config, or evidence-validity failure."""


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


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    require(config.get("schema") == SCHEMA, "unexpected review config schema")
    require(config.get("experiment_id") == "ue_n3a_service_bracket_review_v1",
            "unexpected review experiment identity")
    require(
        config.get("claim_boundary")
        == "OFFLINE_SERVICE_BRACKET_REVIEW_ONLY_NO_BOUND_PROMOTION_NO_N3B_AUTHORITY",
        "review claim boundary drift",
    )
    authority = config["authority"]
    require(authority.get("offline_review_authorized") is True,
            "offline review authority is absent")
    for key in (
        "oai_run_authorized", "socket_execution_authorized", "carla_run_authorized",
        "n3b_execution_authorized", "target_mapping_promotion_authorized",
        "numeric_bound_promotion_authorized", "operational_bound_promotion_authorized",
        "connectivity_bound_promotion_authorized", "usable_service_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        require(authority.get(key) is False, f"forbidden authority enabled: {key}")
    require(config.get("source") == SOURCE, "sealed live_02 source drift")
    require(config.get("frozen_plan") == FROZEN_PLAN,
            "sealed plan_02 source drift")

    contract = config["contract"]
    require(int(contract["repetitions_per_endpoint"]) == 3,
            "review requires three repetitions per endpoint")
    require([float(value) for value in contract["commands_db"]] == [-2.5, -2.0],
            "review command pair drift")
    require(math.isclose(float(contract["pass_command_db"]), -2.5),
            "pass command drift")
    require(math.isclose(float(contract["fail_command_db"]), -2.0),
            "fail command drift")
    require(math.isclose(float(contract["pass_expected_achieved_snr_db"]), 6.0),
            "passing achieved-SNR target drift")
    require(math.isclose(float(contract["pass_achieved_tolerance_db"]), 0.5),
            "passing achieved-SNR tolerance drift")
    require(
        contract.get("observed_fail_snr_role")
        == "POST_LIVE_SOURCE_SEALED_ALL_THREE_MEDIANS_EQUAL_NOT_PREREGISTERED_TARGET",
        "observed failing achieved-SNR role drift",
    )
    require(math.isclose(float(contract["primary_complete_frame_ratio"]), 0.99),
            "primary endpoint drift")
    require(int(contract["maximum_interarrival_gaps_gte_1s"]) == 0,
            "one-second gap endpoint drift")
    require(math.isclose(float(contract["measured_tail_s"]), 60.0)
            and int(contract["expected_tail_frames"]) == 600,
            "exact 60-second/600-frame endpoint drift")
    require(contract["mechanism_expectation_is_not_service_endpoint"] is True,
            "mechanism/service separation is absent")
    require(contract["hard_loss_is_acceptable_fail_evidence"] is True,
            "recognized hard loss must remain acceptable fail evidence")
    require(contract["n3b_selected_command_db_if_bracketed"] == -2.5,
            "N3B-only selection drift")
    require(contract["expected_observed_fail_snr_db"] == 5.0
            and contract["expected_observed_pass_snr_db"] == 6.0
            and contract["expected_observed_bracket_width_db"] == 1.0,
            "observed bracket reporting contract drift")

    rows = config["expected_repetitions"]
    require(len(rows) == 6, "expected repetition inventory must contain six rows")
    observed = tuple(
        (
            row["directory"], int(row["repetition_index"]),
            float(row["commanded_noise_power_db"]), row["condition_id"], row["status"],
            row["manifest_sha256"], row["terminal_sha256"], row["summary_sha256"],
        )
        for row in rows
    )
    require(observed == EXPECTED_REPETITIONS, "expected repetition seals drift")
    require(config.get("output") == OUTPUT, "create-only output contract drift")
    runtime = config["runtime_seals"]
    require(len(runtime) == 1
            and runtime[0]["path"] == "rl_agent/ue_n3a_service_bracket_review_v1.py",
            "review runtime seal set drift")
    if verify_hashes:
        path = resolve_repo_path(str(runtime[0]["path"]))
        require(path.is_file() and sha256(path) == runtime[0]["sha256"],
                "review tool runtime seal drift")


def verify_inventory(
    directory: Path, manifest: Mapping[str, Any],
    *, allowed_uninventoried: Sequence[str] = (),
) -> dict[str, str]:
    rows = list(manifest.get("outputs", []))
    require(rows, f"manifest has no output inventory: {directory}")
    observed: dict[str, str] = {}
    for row in rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in observed,
                f"blank or duplicate output path: {relative!r}")
        artifact = (directory / relative).resolve()
        try:
            artifact.relative_to(directory.resolve())
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
        str(path.relative_to(directory))
        for path in directory.rglob("*") if path.is_file()
    }
    expected_actual = set(observed) | set(allowed_uninventoried)
    require(actual == expected_actual,
            "manifest inventory is not directory-complete: "
            f"unexpected={sorted(actual - expected_actual)} "
            f"missing={sorted(expected_actual - actual)}")
    return observed


def _all_promotions_false(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "target_mapping_promoted", "numeric_bound_promoted",
        "connectivity_bound_promoted", "usable_service_bound_promoted",
    ):
        require(payload.get(key) is False, f"{label} unexpectedly sets {key}")
    # The sealed N3A predecessor schema predates this explicit field.  Absence
    # is accepted only for that hash-pinned evidence; a true value is not.
    require(payload.get("operational_bound_promoted", False) is False,
            f"{label} unexpectedly promotes an operational bound")
    require(payload.get("cold_attach_bound_evaluated") is False,
            f"{label} unexpectedly evaluated cold attach")


def verify_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    plan = config["frozen_plan"]
    directory = resolve_repo_path(str(source["directory"]))
    plan_directory = resolve_repo_path(str(plan["directory"]))
    manifest_path = directory / source["manifest"]
    terminal_path = directory / source["terminal"]
    summary_path = directory / source["campaign_summary"]
    resolved_path = directory / source["resolved_config"]
    plan_manifest_path = plan_directory / plan["manifest"]
    plan_terminal_path = plan_directory / plan["terminal"]
    plan_resolved_path = plan_directory / plan["resolved_config"]
    for path in (manifest_path, terminal_path, summary_path, resolved_path):
        require(path.is_file(), f"sealed campaign source file missing: {path}")
    for path in (plan_manifest_path, plan_terminal_path, plan_resolved_path):
        require(path.is_file(), f"sealed plan source file missing: {path}")
    for path, expected, label in (
        (manifest_path, source["manifest_sha256"], "campaign manifest"),
        (terminal_path, source["terminal_sha256"], "campaign terminal"),
        (summary_path, source["campaign_summary_sha256"], "campaign summary"),
        (resolved_path, source["resolved_config_sha256"], "resolved source config"),
    ):
        require(sha256(path) == expected, f"{label} seal drift")
    for path, expected, label in (
        (plan_manifest_path, plan["manifest_sha256"], "plan manifest"),
        (plan_terminal_path, plan["terminal_sha256"], "plan terminal"),
        (plan_resolved_path, plan["resolved_config_sha256"], "plan resolved config"),
    ):
        require(sha256(path) == expected, f"{label} seal drift")

    plan_manifest, plan_terminal = (
        load_json(plan_manifest_path), load_json(plan_terminal_path)
    )
    require(plan_manifest.get("status") == plan["required_status"]
            and plan_terminal.get("status") == plan["required_status"],
            "frozen plan status mismatch")
    require(plan_terminal.get("manifest_sha256") == plan["manifest_sha256"],
            "frozen plan terminal does not bind its manifest")
    require(plan_manifest.get("config_sha256") == source["source_config_sha256"]
            and plan_manifest.get("runner_sha256") == source["source_runner_sha256"],
            "frozen plan runtime provenance mismatch")
    _all_promotions_false(plan_manifest, "frozen plan manifest")
    _all_promotions_false(plan_terminal, "frozen plan terminal")
    require(plan_terminal.get("runtime_executed") is False
            and plan_terminal.get("socket_executed") is False,
            "frozen plan unexpectedly executed runtime or sockets")
    plan_inventory = verify_inventory(
        plan_directory, plan_manifest,
        allowed_uninventoried=(plan["manifest"], plan["terminal"]),
    )
    require(plan_inventory.get(plan["resolved_config"])
            == plan["resolved_config_sha256"],
            "frozen plan resolved config is not manifest-bound")
    require(plan_resolved_path.read_bytes() == resolved_path.read_bytes(),
            "live_02 resolved config is not byte-identical to frozen plan_02")
    resolved = load_json(resolved_path)
    contract = config["contract"]
    require(float(resolved["rung"]["measured_tail_s"])
            == float(contract["measured_tail_s"])
            and int(resolved["rung"]["expected_tail_frames"])
            == int(contract["expected_tail_frames"]),
            "frozen plan/live tail statistical unit differs from review contract")
    require(float(resolved["transport_gates"]["primary_complete_frame_ratio"])
            == float(contract["primary_complete_frame_ratio"])
            and int(resolved["transport_gates"]["maximum_interarrival_gaps_gte_1s"])
            == int(contract["maximum_interarrival_gaps_gte_1s"]),
            "frozen plan/live usable-service endpoint differs from review contract")
    source_conditions = resolved["campaign"]["conditions"]
    require(len(source_conditions) == 2, "frozen condition inventory drift")
    passing_condition, failing_condition = source_conditions
    require(
        passing_condition.get("condition_id") == "SUSTAIN_CANDIDATE_MINUS2P5"
        and math.isclose(float(passing_condition["commanded_noise_power_db"]), -2.5)
        and passing_condition.get("expected_outcome") == "SUSTAINED_SERVICE",
        "frozen passing condition identity drift",
    )
    require(float(passing_condition["expected_achieved_pusch_snr_db"])
            == float(contract["pass_expected_achieved_snr_db"])
            and float(passing_condition["achieved_tolerance_db"])
            == float(contract["pass_achieved_tolerance_db"]),
            "frozen passing achieved-SNR gate differs from review contract")
    require(
        failing_condition.get("condition_id") == "ADJACENT_HARD_LOSS_MINUS2P0"
        and math.isclose(float(failing_condition["commanded_noise_power_db"]), -2.0)
        and failing_condition.get("expected_outcome") == "HARD_SERVICE_LOSS"
        and set(failing_condition.get("accepted_hard_loss_reasons", []))
        == {"CURRENT_RNTI_PUSCH_SILENCE", "RNTI_CHANGED", "UE_TUNNEL_IDENTITY_LOST"}
        and "expected_achieved_pusch_snr_db" not in failing_condition
        and "achieved_tolerance_db" not in failing_condition,
        "frozen failing condition drift or post-live SNR gate misrepresented as preregistered",
    )

    manifest, terminal, campaign = (
        load_json(manifest_path), load_json(terminal_path), load_json(summary_path)
    )
    require(manifest.get("schema")
            == "scenesense.ue_n3a_sustain_replication_campaign_manifest.v1",
            "source campaign manifest schema mismatch")
    for payload, label in ((manifest, "manifest"), (terminal, "terminal"),
                           (campaign, "summary")):
        require(payload.get("status") == source["required_status"],
                f"source campaign {label} status mismatch")
        _all_promotions_false(payload, f"source campaign {label}")
    require(terminal.get("manifest_sha256") == source["manifest_sha256"],
            "source terminal does not bind campaign manifest")
    require(manifest.get("config_sha256") == source["source_config_sha256"],
            "source config seal mismatch")
    require(manifest.get("runner_sha256") == source["source_runner_sha256"],
            "source runner seal mismatch")
    require(manifest.get("engine_runner_sha256")
            == source["source_engine_runner_sha256"],
            "source engine seal mismatch")
    root_inventory = verify_inventory(
        directory, manifest,
        allowed_uninventoried=(source["manifest"], source["terminal"]),
    )
    require(root_inventory.get(source["campaign_summary"])
            == source["campaign_summary_sha256"],
            "campaign summary is not bound by root inventory")
    require(root_inventory.get(source["resolved_config"])
            == source["resolved_config_sha256"],
            "resolved source config is not bound by root inventory")
    require(int(campaign.get("repetitions_planned", -1)) == 6
            and int(campaign.get("repetitions_executed", -1)) == 6,
            "source campaign is not the complete six-repetition run")
    require(int(campaign.get("fresh_ran_epoch_count", -1)) == 6
            and int(campaign.get("unique_control_session_count", -1)) == 6,
            "source campaign lacks six independent identities")

    expected_dirs = {str(row[0]) for row in EXPECTED_REPETITIONS}
    repetition_root = directory / "repetitions"
    actual_dirs = {
        path.name for path in repetition_root.iterdir() if path.is_dir()
    }
    require(actual_dirs == expected_dirs,
            f"source repetition directory set drift: {sorted(actual_dirs)}")
    campaign_proofs = {
        Path(str(row["directory"])).name: row
        for row in campaign.get("repetition_evidence", [])
    }
    require(set(campaign_proofs) == expected_dirs,
            "campaign summary repetition-proof set drift")

    verified: list[dict[str, Any]] = []
    epochs: set[str] = set()
    sessions: set[str] = set()
    for row in config["expected_repetitions"]:
        name = row["directory"]
        rep_dir = repetition_root / name
        rep_manifest_path = rep_dir / "manifest.json"
        rep_terminal_path = rep_dir / f"{row['status']}.json"
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
                == "scenesense.ue_n3a_sustain_repetition_manifest.v1",
                f"{name} manifest schema mismatch")
        for payload, label in ((rep_manifest, "manifest"),
                               (rep_terminal, "terminal"),
                               (rep_summary, "summary")):
            require(payload.get("status") == row["status"],
                    f"{name} {label} status mismatch")
            require(payload.get("condition_id") == row["condition_id"],
                    f"{name} {label} condition mismatch")
            require(int(payload.get("repetition_index", -1))
                    == int(row["repetition_index"]),
                    f"{name} {label} repetition mismatch")
            require(math.isclose(float(payload.get("commanded_noise_power_db", math.nan)),
                                 float(row["commanded_noise_power_db"]), abs_tol=1e-9),
                    f"{name} {label} command mismatch")
            _all_promotions_false(payload, f"{name} {label}")
        require(rep_terminal.get("manifest_sha256") == row["manifest_sha256"],
                f"{name} terminal/manifest link mismatch")
        require(rep_manifest.get("config_sha256") == source["source_config_sha256"]
                and rep_manifest.get("runner_sha256") == source["source_runner_sha256"]
                and rep_manifest.get("engine_runner_sha256")
                == source["source_engine_runner_sha256"],
                f"{name} runtime provenance mismatch")
        require(rep_manifest.get("candidate_application_count") == 1,
                f"{name} candidate application count mismatch")
        rep_inventory = verify_inventory(
            rep_dir, rep_manifest,
            allowed_uninventoried=("manifest.json", f"{row['status']}.json"),
        )
        required = {"repetition_summary.json", "cleanup_report.json",
                    "transport_summary.json", "runtime_seals.json"}
        require(required.issubset(rep_inventory),
                f"{name} manifest lacks required evidence: {sorted(required - rep_inventory.keys())}")
        for relative, expected_hash in (
            (f"repetitions/{name}/manifest.json", row["manifest_sha256"]),
            (f"repetitions/{name}/{row['status']}.json", row["terminal_sha256"]),
            (f"repetitions/{name}/repetition_summary.json", row["summary_sha256"]),
        ):
            require(root_inventory.get(relative) == expected_hash,
                    f"root campaign manifest does not bind {relative}")
        proof = campaign_proofs[name]
        require(proof.get("manifest_sha256") == row["manifest_sha256"]
                and proof.get("terminal_sha256") == row["terminal_sha256"]
                and proof.get("summary_sha256") == row["summary_sha256"],
                f"campaign summary proof mismatch for {name}")
        epoch = str(rep_manifest.get("ran_epoch_id", ""))
        session = str(rep_manifest.get("control_session_id", ""))
        require(epoch and session and epoch != session,
                f"{name} lacks distinct RAN/control identities")
        require(epoch not in epochs and session not in sessions,
                f"{name} reuses an experiment identity")
        epochs.add(epoch)
        sessions.add(session)
        verified.append({
            "directory": name,
            "repetition_index": int(row["repetition_index"]),
            "commanded_noise_power_db": float(row["commanded_noise_power_db"]),
            "condition_id": row["condition_id"],
            "status": row["status"],
            "manifest_sha256": row["manifest_sha256"],
            "terminal_sha256": row["terminal_sha256"],
            "summary_sha256": row["summary_sha256"],
            "manifest_output_count": len(rep_inventory),
            "ran_epoch_id": epoch,
            "control_session_id": session,
        })
    return {
        "status": "SEALED_LIVE_02_VERIFIED",
        "source_directory": str(directory),
        "frozen_plan_directory": str(plan_directory),
        "frozen_plan_manifest_sha256": plan["manifest_sha256"],
        "frozen_plan_terminal_sha256": plan["terminal_sha256"],
        "plan_live_resolved_configs_byte_identical": True,
        "frozen_fail_condition_had_preregistered_snr_target": False,
        "frozen_plan_manifest_output_count": len(plan_inventory),
        "campaign_manifest_sha256": source["manifest_sha256"],
        "campaign_terminal_sha256": source["terminal_sha256"],
        "campaign_summary_sha256": source["campaign_summary_sha256"],
        "resolved_config_sha256": source["resolved_config_sha256"],
        "campaign_manifest_output_count": len(root_inventory),
        "verified_repetition_count": len(verified),
        "unique_ran_epoch_count": len(epochs),
        "unique_control_session_count": len(sessions),
        "repetitions": verified,
    }


def _exact_tail_evidence(summary: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    tail = dict(summary.get("tail") or {})
    service = dict(summary.get("tail_service") or {})
    expected_frames = int(contract["expected_tail_frames"])
    expected_indices = service.get("expected_frame_indices")
    exact_indices = (
        isinstance(expected_indices, list)
        and len(expected_indices) == expected_frames
        and len(set(expected_indices)) == expected_frames
        and all(isinstance(index, int) and not isinstance(index, bool)
                for index in expected_indices)
        and expected_indices
        == list(range(expected_indices[0], expected_indices[0] + expected_frames))
    )
    duration_ns = int(round(float(contract["measured_tail_s"]) * 1_000_000_000))
    tail_start = tail.get("start_wall_time_ns")
    tail_end = tail.get("end_wall_time_ns")
    service_start = service.get("start_wall_time_ns")
    service_end = service.get("classification_end_wall_time_ns")
    exact_duration = (
        all(isinstance(value, int) and not isinstance(value, bool)
            for value in (tail_start, tail_end, service_start, service_end))
        and tail_start == service_start
        and tail_end == service_end
        and tail_end - tail_start == duration_ns
    )
    return (
        summary.get("engine_status") == "UE_N3_COMMAND_RUNG_CAPTURED_PROPOSAL_ONLY"
        and tail.get("status") == "TAIL_ACCEPTED"
        and tail.get("mcs_seals_ok") is True
        and int(tail.get("pusch_samples", 0)) >= int(tail.get("minimum_pusch_samples", 1))
        and int(tail.get("mcs_samples", 0)) >= int(tail.get("minimum_mcs_samples", 1))
        and service.get("integrity_gate") is True
        and service.get("full_nominal_window_observed") is True
        and service.get("exact_frozen_frame_set_pass") is True
        and service.get("window_role") == "COMMAND_CONDITION_MEASURED_TAIL"
        and exact_indices
        and exact_duration
        and int(service.get("required_expected_frames", -1))
        == expected_frames
        and int(service.get("expected_frames", -1))
        == expected_frames
    )


def _common_valid(summary: Mapping[str, Any], cleanup: Mapping[str, Any]) -> bool:
    return (
        summary.get("clean_restore_verified") is True
        and dict(summary.get("clean_recovery") or {}).get("passed") is True
        and dict(summary.get("transport") or {}).get("integrity_gate") is True
        and int(summary.get("candidate_application_count", -1)) == 1
        and cleanup.get("clean") is True
        and not cleanup.get("errors")
    )


def adjudicate_repetition(
    summary: Mapping[str, Any], cleanup: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    command = float(summary.get("commanded_noise_power_db", math.nan))
    common = _common_valid(summary, cleanup)
    exact_tail = _exact_tail_evidence(summary, contract)
    tail = dict(summary.get("tail") or {})
    service = dict(summary.get("tail_service") or {})
    median = tail.get("achieved_pusch_snr_db_median")
    ratio = service.get("complete_frame_ratio")
    maximum_gap = service.get("maximum_interarrival_or_boundary_gap_s")
    primary_numeric_pass = (
        ratio is not None and math.isfinite(float(ratio))
        and float(ratio) >= float(contract["primary_complete_frame_ratio"])
    )
    no_gap_numeric_pass = (
        maximum_gap is not None and math.isfinite(float(maximum_gap))
        and float(maximum_gap) < 1.0
    )
    endpoint_pass = (
        exact_tail
        and service.get("primary_99_pass") is True
        and service.get("no_one_second_outage_pass") is True
        and primary_numeric_pass
        and no_gap_numeric_pass
    )
    engine_status = str(summary.get("engine_status", ""))
    hard_reason = summary.get("hard_loss_reason")
    hard_pair = RECOGNIZED_HARD_LOSS.get(engine_status) == hard_reason
    hard_corroborated = (
        summary.get("receiver_service_outage_detected") is True
        if hard_reason == "CURRENT_RNTI_PUSCH_SILENCE"
        else hard_reason in {"UE_TUNNEL_IDENTITY_LOST", "RNTI_CHANGED"}
    )
    valid_hard_loss = (
        common and hard_pair and hard_corroborated
        and summary.get("tail") is None and summary.get("tail_service") is None
    )
    tail_evidence_valid = common and exact_tail
    evidence_valid = tail_evidence_valid or valid_hard_loss
    usable_endpoint = (
        "PASS" if tail_evidence_valid and endpoint_pass
        else "FAIL" if (tail_evidence_valid or valid_hard_loss)
        else "INVALID"
    )

    if math.isclose(command, float(contract["pass_command_db"]), abs_tol=1e-9):
        target = float(contract["pass_expected_achieved_snr_db"])
        tolerance = float(contract["pass_achieved_tolerance_db"])
        preregistered_pass_snr_target_pass = (
            median is not None and math.isfinite(float(median))
            and abs(float(median) - target) <= tolerance
        )
        observed_fail_snr_match = None
        snr_bracket_eligible = preregistered_pass_snr_target_pass
        mechanism_expected = "SUSTAINED_SERVICE"
    elif math.isclose(command, float(contract["fail_command_db"]), abs_tol=1e-9):
        preregistered_pass_snr_target_pass = None
        observed_fail_snr_match = (
            median is not None and math.isfinite(float(median))
            and math.isclose(
                float(median), float(contract["expected_observed_fail_snr_db"]),
                abs_tol=1e-9,
            )
        ) if exact_tail else None
        snr_bracket_eligible = observed_fail_snr_match
        mechanism_expected = "HARD_SERVICE_LOSS"
    else:
        raise ReviewFailure(f"unexpected command in sealed repetition: {command}")
    mechanism_observed = (
        "HARD_SERVICE_LOSS" if valid_hard_loss
        else "SUSTAINED_SERVICE" if tail_evidence_valid and endpoint_pass
        else "EXACT_TAIL_SERVICE_GATE_FAILED" if tail_evidence_valid
        else "UNCONFIRMED"
    )
    mechanism_match = mechanism_observed == mechanism_expected
    expected_service_role = (
        usable_endpoint == "PASS"
        if math.isclose(command, float(contract["pass_command_db"]), abs_tol=1e-9)
        else usable_endpoint == "FAIL"
    )
    numeric_bracket_eligible = (
        evidence_valid and expected_service_role and snr_bracket_eligible is True
    )

    return {
        "repetition_index": int(summary.get("repetition_index", -1)),
        "condition_id": summary.get("condition_id"),
        "commanded_noise_power_db": command,
        "evidence_valid": evidence_valid,
        "usable_service_endpoint": usable_endpoint,
        "exact_60s_600_frame_tail_valid": exact_tail,
        "tail_complete_frame_ratio": service.get("complete_frame_ratio"),
        "tail_primary_threshold_recomputed_pass": primary_numeric_pass,
        "tail_primary_99_pass": service.get("primary_99_pass"),
        "tail_no_one_second_gap_pass": service.get("no_one_second_outage_pass"),
        "tail_maximum_gap_s": service.get("maximum_interarrival_or_boundary_gap_s"),
        "tail_no_one_second_gap_recomputed_pass": no_gap_numeric_pass,
        "achieved_pusch_snr_db_median": median,
        "preregistered_pass_snr_target_pass": preregistered_pass_snr_target_pass,
        "source_sealed_observed_fail_snr_match": observed_fail_snr_match,
        "expected_service_role_match": expected_service_role,
        "numeric_bracket_eligible": numeric_bracket_eligible,
        "recognized_hard_loss_evidence": valid_hard_loss,
        "mechanism_expected": mechanism_expected,
        "mechanism_observed": mechanism_observed,
        "mechanism_expectation_match": mechanism_match,
        "mechanism_mismatch_does_not_override_service_endpoint": True,
        "clean_restore_verified": summary.get("clean_restore_verified"),
        "clean_recovery_verified": dict(summary.get("clean_recovery") or {}).get("passed"),
        "transport_integrity_verified": dict(summary.get("transport") or {}).get("integrity_gate"),
        "cleanup_verified": cleanup.get("clean") is True and not cleanup.get("errors"),
    }


def aggregate_adjudications(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any],
) -> dict[str, Any]:
    passing = [row for row in rows if math.isclose(
        float(row["commanded_noise_power_db"]), float(contract["pass_command_db"]),
        abs_tol=1e-9,
    )]
    failing = [row for row in rows if math.isclose(
        float(row["commanded_noise_power_db"]), float(contract["fail_command_db"]),
        abs_tol=1e-9,
    )]
    require(len(passing) == len(failing) == int(contract["repetitions_per_endpoint"]),
            "adjudication does not contain both 3-repetition endpoints")
    pass_count = sum(row["usable_service_endpoint"] == "PASS" for row in passing)
    fail_count = sum(row["usable_service_endpoint"] == "FAIL" for row in failing)
    pass_eligible_count = sum(row["numeric_bracket_eligible"] for row in passing)
    fail_eligible_count = sum(row["numeric_bracket_eligible"] for row in failing)
    all_valid = all(row["evidence_valid"] for row in rows)
    pass_values = [
        float(row["achieved_pusch_snr_db_median"])
        for row in passing if row["achieved_pusch_snr_db_median"] is not None
    ]
    pass_snr = statistics.median(pass_values) if len(pass_values) == len(passing) else None
    fail_values = [
        float(row["achieved_pusch_snr_db_median"])
        for row in failing if row["achieved_pusch_snr_db_median"] is not None
    ]
    fail_snr = statistics.median(fail_values) if len(fail_values) == len(failing) else None
    bracketed = (
        all_valid
        and pass_count == fail_count == 3
        and pass_eligible_count == fail_eligible_count == 3
        and pass_snr is not None
        and fail_snr is not None
    )
    width = pass_snr - fail_snr if bracketed else None
    if bracketed:
        require(math.isclose(fail_snr, float(contract["expected_observed_fail_snr_db"]),
                             abs_tol=1e-9), "observed failing achieved endpoint drift")
        require(math.isclose(pass_snr, float(contract["expected_observed_pass_snr_db"]),
                             abs_tol=1e-9), "observed passing achieved endpoint drift")
        require(math.isclose(float(width),
                             float(contract["expected_observed_bracket_width_db"]),
                             abs_tol=1e-9), "observed achieved bracket width drift")
    return {
        "status": SUCCESS if bracketed else UNRESOLVED,
        "contract_bracketed": bracketed,
        "pass_command_db": float(contract["pass_command_db"]),
        "pass_repetitions": pass_count,
        "pass_numeric_bracket_eligible_repetitions": pass_eligible_count,
        "fail_command_db": float(contract["fail_command_db"]),
        "fail_repetitions": fail_count,
        "fail_numeric_bracket_eligible_repetitions": fail_eligible_count,
        "required_repetitions_per_endpoint": int(contract["repetitions_per_endpoint"]),
        "observed_achieved_fail_endpoint_snr_db": fail_snr,
        "observed_achieved_pass_endpoint_snr_db": pass_snr,
        "observed_achieved_bracket_width_db": width,
        "observed_fail_snr_role": contract["observed_fail_snr_role"],
        "mechanism_mismatch_count": sum(not row["mechanism_expectation_match"] for row in rows),
        "mechanism_mismatch_is_endpoint_failure": False,
        "n3b_selected_command_db": (
            float(contract["n3b_selected_command_db_if_bracketed"])
            if bracketed else None
        ),
        "n3b_eligibility_status": (
            "UE_N3A_USABLE_SERVICE_BRACKET_ACCEPTED_FOR_N3B"
            if bracketed else "N3B_ELIGIBILITY_UNRESOLVED"
        ),
        "selection_scope": "N3B_ELIGIBILITY_ONLY",
        "n3b_selection_role": "CANDIDATE_FOR_N3B_ONLY_NOT_A_PROMOTED_BOUND",
        "n3b_execution_authorized": False,
        "n3b_executed": False,
        "hard_loss_boundary_status": "UNRESOLVED_MECHANISM_NOT_CONFIRMED_AT_MINUS2P0",
        "l_attach_status": "PENDING_N3B_COLD_ATTACH",
        "l_operational_status": "PENDING_N3B_COLD_ATTACH",
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
        for protected in (resolve_repo_path(SOURCE["directory"]),
                          resolve_repo_path(FROZEN_PLAN["directory"])):
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
            if not path.is_file() or path.name in excluded or path.name.startswith("UE_N3_"):
                continue
            files.append({
                "path": str(path.relative_to(self.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        manifest_path = self.output_dir / OUTPUT["manifest"]
        atomic_json(manifest_path, {
            "schema": "scenesense.ue_n3a_service_bracket_review_manifest.v1",
            "status": status,
            "config_sha256": sha256(self.config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_campaign_manifest_sha256": SOURCE["manifest_sha256"],
            "offline_only": True,
            "n3b_execution_authorized": False,
            "n3b_executed": False,
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
            "n3b_execution_authorized": False,
            "n3b_executed": False,
            "target_mapping_promoted": False,
            "numeric_bound_promoted": False,
            "operational_bound_promoted": False,
            "connectivity_bound_promoted": False,
            "usable_service_bound_promoted": False,
            "manifest_sha256": sha256(manifest_path),
        }
        terminal_name = OUTPUT["failure"] if status == "FAILED" \
            else f"{status}.json"
        atomic_json(self.output_dir / terminal_name, terminal)

    def run(self) -> int:
        try:
            validate_config(self.config, verify_hashes=True)
            atomic_json(
                self.output_dir / OUTPUT["resolved_config"], self.config
            )
            verification = verify_source(self.config)
            atomic_json(
                self.output_dir / OUTPUT["source_verification"],
                verification,
            )
            source_dir = resolve_repo_path(self.config["source"]["directory"])
            adjudications = []
            for row in self.config["expected_repetitions"]:
                rep_dir = source_dir / "repetitions" / row["directory"]
                summary = load_json(rep_dir / "repetition_summary.json")
                cleanup = load_json(rep_dir / "cleanup_report.json")
                adjudication = adjudicate_repetition(
                    summary, cleanup, self.config["contract"]
                )
                adjudication["source_directory"] = row["directory"]
                adjudication["source_manifest_sha256"] = row["manifest_sha256"]
                adjudications.append(adjudication)
            atomic_json(
                self.output_dir / OUTPUT["adjudications_json"],
                {"schema": "scenesense.ue_n3a_service_endpoint_adjudications.v1",
                 "rows": adjudications},
            )
            write_csv(
                self.output_dir / OUTPUT["adjudications_csv"],
                adjudications,
            )
            aggregate = aggregate_adjudications(adjudications, self.config["contract"])
            status = str(aggregate["status"])
            selection_line = (
                "- Selected command for a future N3B review: -2.5 dB. N3B is not "
                "authorized or executed by this review.\n"
                if aggregate["contract_bracketed"] else
                "- No N3B candidate is selected because the service bracket is unresolved.\n"
            )
            report = (
                "# UE-N3A service-bracket review\n\n"
                f"- Contract status: `{status}`\n"
                f"- Command -2.5 dB service passes: {aggregate['pass_repetitions']}/3\n"
                f"- Command -2.0 dB service failures: {aggregate['fail_repetitions']}/3\n"
                f"- Observed achieved-SNR bracket: {aggregate['observed_achieved_fail_endpoint_snr_db']} dB "
                f"(fail) to {aggregate['observed_achieved_pass_endpoint_snr_db']} dB (pass); "
                f"width {aggregate['observed_achieved_bracket_width_db']} dB.\n"
                f"- Pre-registered mechanism mismatches: {aggregate['mechanism_mismatch_count']}; "
                "these do not override the frozen usable-service endpoint.\n"
                f"{selection_line}"
                "- No mapping or numeric, operational, connectivity, or usable-service bound "
                "is promoted.\n"
            )
            atomic_text(self.output_dir / OUTPUT["report"], report)
            summary = {
                **aggregate,
                "source_verification": verification,
                "adjudications": adjudications,
                "review_scope": (
                    "FROZEN_USABLE_SERVICE_ENDPOINT_WITH_MECHANISM_REPORTED_SEPARATELY"
                ),
                "review_required": True,
                "n3b_authorized_or_run": False,
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
                "n3b_authorized_or_run": False,
                "n3b_execution_authorized": False,
                "n3b_executed": False,
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

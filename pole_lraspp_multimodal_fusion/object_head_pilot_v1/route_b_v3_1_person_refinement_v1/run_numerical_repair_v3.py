#!/usr/bin/env python3
"""Complete the registered person refinement under the authorized FP32 policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
POLICY_SOURCE = PACKAGE_ROOT / "configs/full_fp32_person_policy_v3.json"
SOURCE_EXPERIMENT = ROOT / "experiments/route_b_v3_1_person_refinement_continuation_v2/20260828_134000"
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_person_refinement_numerical_repair_v3"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_PERSON_REFINEMENT_NUMERICAL_REPAIR_V3_REPORT.md"
BASE_CHECKPOINT = ROOT / "experiments/route_b_v3_1_person_refinement_v1/20260828_163100/checkpoints/route_b_v3_1_person_refinement_v1/epoch_040.pt"
HISTORICAL_BASE_EXPERIMENT = ROOT / "experiments/route_b_v3_1_person_refinement_v1/20260828_163100"
TERMINALS = {
    "LRASPP_PERSON_REFINEMENT_SERVICE_READY",
    "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
    "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    "LRASPP_PERSON_REFINEMENT_NUMERICAL_ROOT_CAUSE_UNRESOLVED",
    "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID",
    "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE",
}
PROGRESS_FIELDS = ("created_utc", "attempt", "phase", "epoch", "optimizer_steps", "detail")

for path in (str(PACKAGE_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_accepted_continuation_v2 as accepted  # noqa: E402
import run_pipeline_v1 as core  # noqa: E402
import policy_v1 as selection_policy  # noqa: E402


class ContractInvalid(RuntimeError):
    pass


class NumericalUnresolved(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def run_logged(command: list[str], log: Path, marker: str) -> int:
    with log.open("a" if log.exists() else "x", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] {marker}\ncommand={json.dumps(command)}\n")
        stream.flush()
        return subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True,
        ).wait()


def source_hashes() -> dict[str, str]:
    paths = [
        PACKAGE_ROOT / "diagnose_numerics_v2.py",
        PACKAGE_ROOT / "person_losses_v1.py",
        PACKAGE_ROOT / "person_model_v1.py",
        PACKAGE_ROOT / "train_v1.py",
        PACKAGE_ROOT / "run_numerical_repair_v3.py",
        POLICY_SOURCE,
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def git_state(policy: dict[str, Any]) -> tuple[str, str, list[str]]:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if branch != "master" or head != policy["required_starting_head"]:
        raise ContractInvalid(
            f"required local master/{policy['required_starting_head']}, got {branch}/{head}"
        )
    return branch, head, status


def setup(timestamp: str | None, phase1_source: Path,
          phase2_source: Path) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], str]:
    policy = json.loads(POLICY_SOURCE.read_text(encoding="utf-8"))
    branch, head, status = git_state(policy)
    phase1_source = phase1_source.resolve(strict=True)
    phase2_source = phase2_source.resolve(strict=True)
    if sha256(phase1_source) != policy["phase1_reproduction_sha256"]:
        raise ContractInvalid("Phase-1 reproduction artifact SHA mismatch")
    phase1 = json.loads(phase1_source.read_text(encoding="utf-8"))
    phase2 = json.loads(phase2_source.read_text(encoding="utf-8"))
    if (
        not phase1.get("deterministic_batch_134_failure_reproduced")
        or phase1.get("exact_fp16_nonfinite_loss_batch") != 134
        or phase1.get("exact_first_fp16_nonfinite_operation", {}).get("operation")
        != "person_trunk.conv1"
    ):
        raise ContractInvalid("Phase-1 deterministic root-cause evidence is invalid")
    if not phase2.get("full_fp32_policy_authorized"):
        raise NumericalUnresolved("full-FP32 person path did not pass batches 133--135")
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = EXPERIMENT_PARENT / stamp
    experiment.mkdir(parents=True, exist_ok=False)
    for name in (
        "logs", "resolved_configs", "provenance", "numerical", "predictions",
        "decisions", "checkpoints", "candidate_recovery_checkpoints", "person_metrics",
    ):
        (experiment / name).mkdir()
    for name in (
        "ACCEPTED_BASE_DECISION.json", "ACCEPTED_BASE_PREFLIGHT.json",
        "BASE_DIAGNOSTIC.json", "QUALIFICATION.json", "REGISTRATION.json",
    ):
        shutil.copyfile(SOURCE_EXPERIMENT / name, experiment / name)
    shutil.copyfile(
        SOURCE_EXPERIMENT / "decisions/epoch_040_decode.json",
        experiment / "decisions/epoch_040_decode.json",
    )
    config_path = experiment / "resolved_configs/person_refinement_v1.json"
    shutil.copyfile(SOURCE_EXPERIMENT / "resolved_configs/person_refinement_v1.json", config_path)
    policy_path = experiment / "resolved_configs/full_fp32_person_policy_v3.json"
    shutil.copyfile(POLICY_SOURCE, policy_path)
    phase1_path = experiment / "numerical/PHASE1_NUMERICAL_REPRODUCTION.json"
    phase2_path = experiment / "numerical/PHASE2_FULL_FP32_QUALIFICATION.json"
    shutil.copyfile(phase1_source, phase1_path)
    shutil.copyfile(phase2_source, phase2_path)
    os.symlink(str((SOURCE_EXPERIMENT / "dataset").resolve()), experiment / "dataset")
    os.symlink(str((SOURCE_EXPERIMENT / "contracts").resolve()), experiment / "contracts")
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    write_json_x(experiment / "STATUS.json", {
        "schema": "route_b_v3_1_person_refinement_numerical_repair_status_v3",
        "created_utc": utc_now(), "updated_utc": utc_now(), "phase": "setup",
        "detail": "", "terminal": None, "supervisor_pid": os.getpid(),
        "experiment": str(experiment), "interpreter": sys.executable,
        "required_starting_head": head,
    })
    with (experiment / "PROGRESS.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writeheader()
    core.progress(experiment, "supervisor_started", str(experiment))
    write_json_x(experiment / "provenance/PROVENANCE.json", {
        "schema": "route_b_v3_1_person_refinement_numerical_repair_provenance_v3",
        "created_utc": utc_now(), "branch": branch, "execution_head": head,
        "initial_git_status": status, "source_hashes": source_hashes(),
        "source_failed_experiment": str(SOURCE_EXPERIMENT),
        "source_failed_terminal_sha256": sha256(SOURCE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
        "historical_base_terminal_sha256": sha256(HISTORICAL_BASE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
        "base_recovery_runs_launched": 0, "base_inference_runs_launched": 0,
        "base_diagnostic_runs_launched": 0, "registration_runs_launched": 0,
        "candidate_training_start_epoch": 1, "locked_test_paths_read": 0,
        "carla_commands_launched": 0, "oai_commands_launched": 0,
        "q_or_ae_commands_launched": 0, "measurement_files_modified": 0,
    })
    write_json_x(experiment / "PIPELINE_STARTED.json", {
        "schema": "route_b_v3_1_person_refinement_numerical_repair_started_v3",
        "created_utc": utc_now(), "experiment": str(experiment),
        "execution_head": head, "pid": os.getpid(),
    })
    return experiment, config_path, policy_path, policy, phase2, head


def preflight(experiment: Path, config_path: Path, policy_path: Path,
              policy: dict[str, Any], phase2: dict[str, Any]) -> dict[str, Any]:
    source_registration = SOURCE_EXPERIMENT / "REGISTRATION.json"
    source_terminal = SOURCE_EXPERIMENT / "TERMINAL_VERDICT.txt"
    base_terminal = HISTORICAL_BASE_EXPERIMENT / "TERMINAL_VERDICT.txt"
    checks = {
        "base_checkpoint_sha": sha256(BASE_CHECKPOINT) == policy["base_checkpoint_sha256"],
        "source_registration_sha": sha256(source_registration) == policy["source_registration_sha256"],
        "copied_registration_sha": sha256(experiment / "REGISTRATION.json") == policy["source_registration_sha256"],
        "resolved_config_sha": sha256(config_path)
        == json.loads(source_registration.read_text())["resolved_config_sha256"],
        "source_runtime_terminal_preserved": source_terminal.read_text().strip()
        == policy["source_failed_terminal"]
        and sha256(source_terminal) == policy["source_failed_terminal_sha256"],
        "historical_base_terminal_preserved": base_terminal.read_text().strip()
        == "LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED"
        and sha256(base_terminal) == "a34671b16116c8bca0d607b31a53e2a2c43aa69d5c09a317fb754863426404a2",
        "phase1_sha": sha256(experiment / "numerical/PHASE1_NUMERICAL_REPRODUCTION.json")
        == policy["phase1_reproduction_sha256"],
        "phase2_exact_copy": json.loads(
            (experiment / "numerical/PHASE2_FULL_FP32_QUALIFICATION.json").read_text()
        ) == phase2,
        "phase2_fp32_authorized": bool(phase2.get("full_fp32_policy_authorized")),
        "phase2_p2_proof": bool(phase2.get("full_fp32_p2_inherited_person_proof", {}).get("all_pass")),
        "phase2_vehicle_transport_invariants": bool(
            phase2.get("transport_and_vehicle_outputs_exactly_unchanged")
        ),
        "phase2_split_parity": bool(
            phase2.get("monolithic_split_parity", {}).get("outputs_bit_identical")
        ),
        "no_candidate_checkpoint_preexists": not any(experiment.rglob("epoch_*.pt")),
        "dataset_and_contract_links_reused": (experiment / "dataset").is_symlink()
        and (experiment / "contracts").is_symlink(),
        "policy_copy_sha": sha256(policy_path) == sha256(POLICY_SOURCE),
    }
    result = {
        "schema": "route_b_v3_1_person_refinement_numerical_repair_preflight_v3",
        "created_utc": utc_now(), "all_pass": all(checks.values()), "checks": checks,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "source_registration": str(source_registration),
        "source_registration_sha256": sha256(source_registration),
        "base_recovery_or_diagnostic_repeated": False,
    }
    write_json_x(experiment / "PREFLIGHT.json", result)
    if not result["all_pass"]:
        raise ContractInvalid(f"numerical repair preflight failed: {checks}")
    write_text_x(experiment / "PREFLIGHT_COMPLETE", "PASS\n")
    return result


def register_policy(experiment: Path, policy: dict[str, Any], phase2: dict[str, Any]) -> Path:
    phase1_path = experiment / "numerical/PHASE1_NUMERICAL_REPRODUCTION.json"
    phase2_path = experiment / "numerical/PHASE2_FULL_FP32_QUALIFICATION.json"
    registration_path = experiment / "REGISTRATION.json"
    policy_registration = experiment / "NUMERICAL_POLICY_REGISTRATION.json"
    actual_deltas = {
        batch: payload["actual_full_fp32_implementation_max_abs_deltas"]
        for batch, payload in phase2["policies"]["person_tail_fp32"]["batches"].items()
    }
    all_actual_deltas_zero = all(
        all(float(value) == 0.0 for value in deltas.values())
        for deltas in actual_deltas.values()
    )
    authorized = (
        phase2["full_fp32_policy_authorized"]
        and phase2["full_fp32_batches_133_135_pass"]
        and phase2["full_fp32_p2_inherited_person_proof"]["all_pass"]
        and phase2["transport_and_vehicle_outputs_exactly_unchanged"]
        and phase2["monolithic_split_parity"]["outputs_bit_identical"]
        and all_actual_deltas_zero
        and all(value is False for value in policy["scientific_design_changes"].values())
    )
    value = {
        "schema": "route_b_v3_1_person_refinement_numerical_policy_registration_v3",
        "created_utc": utc_now(), "authorized": bool(authorized),
        "decision": "FULL_FP32_PERSON_PATH_AUTHORIZED_AFTER_DETERMINISTIC_BATCH134_ROOT_CAUSE",
        "policy": policy["policy"],
        "scientific_design_changes": policy["scientific_design_changes"],
        "maximum_further_dtype_patches": 0,
        "phase1_reproduction": str(phase1_path), "phase1_reproduction_sha256": sha256(phase1_path),
        "phase2_qualification": str(phase2_path), "phase2_qualification_sha256": sha256(phase2_path),
        "source_registration": str(registration_path),
        "source_registration_sha256": sha256(registration_path),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "actual_implementation_max_abs_deltas": actual_deltas,
        "source_hashes": source_hashes(),
        "candidate_restart_epoch": 1,
    }
    write_json_x(policy_registration, value)
    if not authorized:
        raise NumericalUnresolved("coherent full-FP32 policy qualification did not pass")
    write_text_x(experiment / "NUMERICAL_POLICY_REGISTERED", "AUTHORIZED\n")
    return policy_registration


def train_candidate(experiment: Path, config_path: Path, policy_registration: Path,
                    config: dict[str, Any], retry: dict[str, int]) -> None:
    command = [
        sys.executable, str(PACKAGE_ROOT / "train_v1.py"),
        "--experiment", str(experiment), "--config", str(config_path),
        "--base-checkpoint", str(BASE_CHECKPOINT),
        "--base-sha256", sha256(BASE_CHECKPOINT),
        "--numerical-policy-registration", str(policy_registration),
        "--attempt", "1",
    ]
    core.progress(experiment, "person_training_full_fp32", attempt=1, epoch=1)
    rc = run_logged(command, experiment / "logs/person_training.log", "fresh full-FP32 person training")
    if rc == 0:
        return
    latest_path = experiment / "PERSON_LATEST_SAFE.json"
    if not latest_path.is_file():
        raise RuntimeError("full-FP32 candidate training failed before an epoch checkpoint")
    if retry["used"] >= int(config["maximum_runtime_retries"]):
        raise RuntimeError(f"full-FP32 candidate retry already consumed rc={rc}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = Path(latest["path"])
    if sha256(checkpoint) != latest["sha256"]:
        raise RuntimeError("latest full-FP32 candidate checkpoint SHA mismatch")
    retry["used"] += 1
    core.progress(
        experiment, "person_training_transient_retry", latest["path"],
        attempt=2, epoch=latest["epoch"],
    )
    retry_command = command[:-2] + [
        "--attempt", "2", "--resume-checkpoint", latest["path"],
        "--resume-sha256", latest["sha256"],
    ]
    if run_logged(
        retry_command, experiment / "logs/person_training.log",
        "sole transient full-FP32 retry",
    ) != 0:
        raise RuntimeError("sole transient full-FP32 candidate retry failed")


def service_gaps(metrics: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    return accepted.service_gaps(metrics, targets)


def build_selection_failure_audit(experiment: Path, config: dict[str, Any]) -> dict[str, Any]:
    output = experiment / "FINAL_SELECTION_FAILURE_AUDIT.json"
    if output.exists():
        raise FileExistsError(f"create-only selection failure audit exists: {output}")
    base = json.loads((experiment / "decisions/epoch_040_decode.json").read_text())
    base_inference_path = Path(base["prediction_root"]) / "inference_manifest.json"
    base_inference = json.loads(base_inference_path.read_text())
    base.update({
        "label": "epoch_040_base", "selection_order": 0,
        "vehicle_detection_rows": selection_policy.vehicle_rows(base_inference_path, base_inference),
    })
    records = [base]
    for epoch in (6, 12, 18):
        record = json.loads(
            (experiment / f"decisions/person_epoch_{epoch:03d}_primary.json").read_text()
        )
        inference_path = Path(record["prediction_root"]) / "inference_manifest.json"
        inference = json.loads(inference_path.read_text())
        record.update({
            "label": f"person_epoch_{epoch:03d}", "selection_order": epoch,
            "vehicle_detection_rows": selection_policy.vehicle_rows(inference_path, inference),
        })
        records.append(record)
    for record in records:
        gates = selection_policy.eligibility(
            record, base, config["final_eligibility"], int(base["vehicle_detection_rows"]),
        )
        record["eligibility_gates"] = gates
        record["eligible"] = all(gates.values())
        record["material_gain"] = selection_policy.material(
            record["metrics"], base["metrics"], config["material_gain"],
        )
        record["normalized_person_deficit"] = selection_policy.deficit(record["metrics"])
        record["service_targets"] = selection_policy.service_targets(
            record["metrics"], config["service_targets"],
        )
        record["service_gaps"] = service_gaps(record["metrics"], config["service_targets"])
    candidates = records[1:]
    eligible = [record for record in candidates if record["eligible"]]
    nondominated = [
        record["label"] for record in candidates
        if not any(selection_policy.dominates(other, record)
                   for other in candidates if other is not record)
    ]
    diagnostic_ranking = sorted(candidates, key=lambda record: (
        record["normalized_person_deficit"], -record["metrics"]["person_f1"],
        -record["metrics"]["person_recall"], record["metrics"]["person_xy_mae_m"],
        record["selection_order"],
    ))
    result = {
        "schema": "route_b_v3_1_person_refinement_no_eligible_selection_audit_v3",
        "created_utc": utc_now(), "terminal": "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID",
        "reason": "no eligible person-refinement checkpoint under unchanged registered gates",
        "records": records, "eligible_labels": [record["label"] for record in eligible],
        "nondominated_labels": nondominated,
        "diagnostic_continuous_deficit_ranking_not_a_selection": [
            {"label": record["label"],
             "normalized_person_deficit": record["normalized_person_deficit"]}
            for record in diagnostic_ranking
        ],
        "selected": None, "selected_checkpoint": None, "selected_checkpoint_sha256": None,
        "selected_only_v025_sensitivity_run": False,
        "selected_only_v025_not_run_reason": "no legally eligible selected checkpoint",
        "selection_or_evaluation_rules_changed": False,
    }
    if eligible:
        raise ContractInvalid("selection failure audit unexpectedly found an eligible candidate")
    write_json_x(output, result)
    return result


def retention_without_selection(experiment: Path, config: dict[str, Any],
                                audit: dict[str, Any]) -> dict[str, Any]:
    output = experiment / "RETENTION_NO_SELECTION.json"
    if output.exists():
        raise FileExistsError(f"create-only no-selection retention audit exists: {output}")
    paths = {
        "epoch_010": (ROOT / config["resume_checkpoint"]).resolve(strict=True),
        **{
            f"recovered_epoch_{epoch:03d}": HISTORICAL_BASE_EXPERIMENT
            / f"checkpoints/{config['name']}/epoch_{epoch:03d}.pt"
            for epoch in (20, 30, 40)
        },
        **{
            f"person_epoch_{epoch:03d}": experiment
            / f"checkpoints/{config['name']}/epoch_{epoch:03d}.pt"
            for epoch in (6, 12, 18)
        },
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    result = {
        "schema": "route_b_v3_1_person_refinement_no_selection_retention_v3",
        "created_utc": utc_now(), "all_pass": not missing,
        "missing": missing, "checkpoint_deletions_performed": 0,
        "retained_paths": {name: str(path) for name, path in paths.items()},
        "retained_sha256": {name: sha256(path) for name, path in paths.items() if path.is_file()},
        "nondominated_labels": audit["nondominated_labels"],
        "all_nondominated_present": all(
            paths[label].is_file() for label in audit["nondominated_labels"]
        ),
        "selected_checkpoint": None,
        "selected_checkpoint_absent_reason": "no eligible candidate under unchanged gates",
    }
    result["all_pass"] = result["all_pass"] and result["all_nondominated_present"]
    write_json_x(output, result)
    if not result["all_pass"]:
        raise ContractInvalid(f"no-selection retention audit failed: {result}")
    return result


def make_report(experiment: Path, terminal: str, config: dict[str, Any],
                policy: dict[str, Any], selection: dict[str, Any] | None,
                retention: dict[str, Any] | None, notification: dict[str, Any],
                retry_used: int, error: str | None, wall_seconds: float) -> str:
    phase1 = json.loads((experiment / "numerical/PHASE1_NUMERICAL_REPRODUCTION.json").read_text())
    phase2 = json.loads((experiment / "numerical/PHASE2_FULL_FP32_QUALIFICATION.json").read_text())
    first = phase1["exact_first_fp16_nonfinite_operation"]
    comparison = {
        name: {
            batch: {
                "loss": payload["loss_value"], "finite": payload["loss_finite"],
                "first_nonfinite": payload["first_nonfinite_operation"],
                "gradients_finite": payload["gradients"]["all_trainable_gradients_finite"],
            }
            for batch, payload in values["batches"].items()
        }
        for name, values in phase1["policies"].items()
    }
    lines = [
        "# Route B v3.1 LR-ASPP person-refinement numerical repair v3", "",
        f"Terminal: `{terminal}`", "", f"Experiment: `{experiment}`", "",
        "## Deterministic numerical root cause", "",
        "The registered sampler, seed, batch size, worker configuration, augmentation, and optimizer updates were replayed through batch 132 before cloning the exact state for batches 133--135.",
        f"The first FP16 failure is epoch 1 batch 134 operation `{first['operation']}`: dtype `{first['dtype']}`, shape `{first['shape']}`, NaN/+inf/-inf `{first['nan']}`/`{first['positive_infinity']}`/`{first['negative_infinity']}`, maximum finite absolute activation `{first['maximum_absolute_activation']}`.",
        f"FP16/BF16/FP32 comparison: `{comparison}`.",
        "Inputs, targets, parameters, optimizer state, transported low/high tensors, and the native feature were independently finite before the failing convolution.", "",
        "## Authorized numerical policy", "",
        f"Final policy: `{policy['policy']}`.",
        f"P2 inherited-person proof: loss/dtype `{phase2['full_fp32_p2_inherited_person_proof']['loss']}` / `{phase2['full_fp32_p2_inherited_person_proof']['loss_dtype']}`; inherited gradient `{phase2['full_fp32_p2_inherited_person_proof']['gradients']['modules']['inherited_person_heatmap']}`; all-pass `{phase2['full_fp32_p2_inherited_person_proof']['all_pass']}`.",
        f"Actual implementation deltas on batches 133--135: `{ {batch: payload['actual_full_fp32_implementation_max_abs_deltas'] for batch, payload in phase2['policies']['person_tail_fp32']['batches'].items()} }`.",
        "No model equation, initialization, LR, scheduler, loss, weight, sampler, target, decoder, batch-size, architecture, or evaluation-rule change was made. No further dtype patch is authorized.", "",
        "## Execution", "",
        "Base recovery, epochs 11--40, base inference, PR diagnostics, and scientific registration were not repeated. Candidate training restarted from epoch 1 in a fresh create-only experiment.",
        f"Transient retries used: `{retry_used}` of one. Error: `{error or 'none'}`.",
    ]
    if selection:
        lines.extend(["", "## Complete primary v0.10 metrics", ""])
        for record in selection["records"]:
            lines.append(
                f"- `{record['label']}`: metrics `{record['metrics']}`; eligibility `{record['eligibility_gates']}`; material `{record['material_gain']}`; normalized person deficit `{record['normalized_person_deficit']}`."
            )
        selected = selection["selected"]
        gaps = service_gaps(selected["metrics"], config["service_targets"])
        training = json.loads((experiment / "PERSON_TRAINING_COMPLETE.json").read_text())
        epoch_metrics = {
            str(epoch): json.loads((experiment / f"person_metrics/epoch_{epoch:03d}.json").read_text())
            for epoch in range(1, 19)
        }
        resources = {
            record["label"]: {
                key: json.loads((Path(record["prediction_root"]) / "inference_manifest.json").read_text())[key]
                for key in ("wall_seconds", "peak_allocated_mib", "peak_reserved_mib")
            }
            for record in selection["records"] if record["label"] != "epoch_040_base"
        }
        lines.extend(["", "## Selection, sensitivity, runtime, and retention", "",
            f"Selected checkpoint: `{selected['checkpoint']}` with SHA-256 `{selected['checkpoint_sha256']}` (`{selected['label']}`).",
            f"Ranking: `{selection['ranking']}`. Non-dominated checkpoints: `{selection['nondominated_labels']}`.",
            f"Continuous service gaps: `{gaps}`. Service gates: `{selected['service_targets']}`.",
            f"Selected-only v0.25 sensitivity: `{selection['selected_v025_sensitivity']}`.",
            f"All 18 epoch training metrics: `{epoch_metrics}`.",
            f"Training optimizer steps/wall/VRAM allocated/reserved: `{training['optimizer_steps']}` / `{sum(float(value['epoch_seconds']) for value in epoch_metrics.values())}` s / `{training['peak_allocated_mib']}` / `{training['peak_reserved_mib']}` MiB.",
            f"Evaluation inference wall/VRAM: `{resources}`.",
            f"Retention audit: `{retention}`. No designated candidate checkpoint was deleted.",
        ])
    elif (experiment / "FINAL_SELECTION_FAILURE_AUDIT.json").is_file():
        audit = json.loads((experiment / "FINAL_SELECTION_FAILURE_AUDIT.json").read_text())
        retained = json.loads((experiment / "RETENTION_NO_SELECTION.json").read_text())
        lines.extend(["", "## Complete primary v0.10 metrics and failed eligibility", ""])
        for record in audit["records"]:
            lines.append(
                f"- `{record['label']}`: checkpoint `{record.get('checkpoint')}` SHA `{record.get('checkpoint_sha256')}`; metrics `{record['metrics']}`; eligibility `{record['eligibility_gates']}`; material `{record['material_gain']}`; normalized person deficit `{record['normalized_person_deficit']}`; service gaps `{record['service_gaps']}`."
            )
        training = json.loads((experiment / "PERSON_TRAINING_COMPLETE.json").read_text())
        epoch_metrics = {
            str(epoch): json.loads((experiment / f"person_metrics/epoch_{epoch:03d}.json").read_text())
            for epoch in range(1, 19)
        }
        resources = {
            record["label"]: {
                key: json.loads((Path(record["prediction_root"]) / "inference_manifest.json").read_text())[key]
                for key in ("wall_seconds", "peak_allocated_mib", "peak_reserved_mib")
            }
            for record in audit["records"] if record["label"] != "epoch_040_base"
        }
        lines.extend(["", "## No legal selection, sensitivity, runtime, and retention", "",
            "No candidate passed all unchanged eligibility gates; therefore there is no selected checkpoint or SHA, and selected-only v0.25 sensitivity was not legally runnable.",
            f"Diagnostic continuous-deficit ranking (not a selection): `{audit['diagnostic_continuous_deficit_ranking_not_a_selection']}`. Non-dominated candidates: `{audit['nondominated_labels']}`.",
            f"All 18 epoch training metrics: `{epoch_metrics}`.",
            f"Training optimizer steps/VRAM allocated/reserved: `{training['optimizer_steps']}` / `{training['peak_allocated_mib']}` / `{training['peak_reserved_mib']}` MiB.",
            f"Evaluation inference wall/VRAM: `{resources}`.",
            f"Retention audit: `{retained}`. All designated candidates, including the non-dominated epoch 18 checkpoint, remain retained.",
        ])
    lines.extend(["", "## Operational boundary", "",
        "The locked test remained absent and unopened. No q/AE, tracking, CARLA, OAI, live-runtime, or 288-measurement work was performed. Checkpoints, datasets, predictions, and experiment artifacts are not committed.",
        f"Notification result: `{notification}`.",
        f"Supervisor wall time: `{wall_seconds:.1f}` seconds.", "",
    ])
    return "\n".join(lines)


def notify(terminal: str, experiment: Path) -> dict[str, Any]:
    command = [
        "notify-send", "LR-ASPP FP32 person refinement complete",
        f"{terminal}\n{experiment}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command, "returncode": result.returncode,
            "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:],
            "delivered": result.returncode == 0,
        }
    except Exception as exc:
        return {"command": command, "delivered": False,
                "error": f"{type(exc).__name__}: {exc}"}


def commit_report(message: str = "Complete FP32 LR-ASPP person refinement") -> dict[str, Any]:
    explicit = [
        PACKAGE_ROOT / "diagnose_numerics_v2.py",
        PACKAGE_ROOT / "person_losses_v1.py",
        PACKAGE_ROOT / "person_model_v1.py",
        PACKAGE_ROOT / "train_v1.py",
        PACKAGE_ROOT / "run_numerical_repair_v3.py",
        POLICY_SOURCE, TRACKED_REPORT,
    ]
    subprocess.run(["git", "add", "--", *map(str, explicit)], cwd=ROOT, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    forbidden = [name for name in staged if name.startswith("experiments/")
                 or name.endswith(".pt") or "/predictions/" in name or "/dataset/" in name]
    if forbidden:
        raise RuntimeError(f"forbidden artifacts staged: {forbidden}")
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"local source/config/report commit failed: {result.stderr[-2000:]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {"head": head, "staged_files": staged, "stdout": result.stdout[-2000:]}


def amend_existing(experiment_arg: Path) -> int:
    experiment = experiment_arg.resolve(strict=True)
    if experiment.parent != EXPERIMENT_PARENT.resolve(strict=True):
        raise ContractInvalid(f"unexpected experiment parent: {experiment}")
    terminal = (experiment / "TERMINAL_VERDICT.txt").read_text().strip()
    if terminal != "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID":
        raise ContractInvalid(f"report amendment requires contract-invalid terminal, got {terminal}")
    if (experiment / "FINAL_SELECTION.json").exists():
        raise ContractInvalid("refusing no-selection amendment when final selection exists")
    config = json.loads((experiment / "resolved_configs/person_refinement_v1.json").read_text())
    policy = json.loads((experiment / "resolved_configs/full_fp32_person_policy_v3.json").read_text())
    audit = build_selection_failure_audit(experiment, config)
    retained = retention_without_selection(experiment, config, audit)
    notification = json.loads((experiment / "NOTIFICATION.json").read_text())
    pipeline = json.loads((experiment / "PIPELINE_COMPLETE.json").read_text())
    report = make_report(
        experiment, terminal, config, policy, None, retained, notification,
        int(pipeline["retry_used"]), pipeline["error"], float(pipeline["wall_seconds"]),
    )
    (experiment / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    TRACKED_REPORT.write_text(report, encoding="utf-8")
    amendment_path = experiment / "REPORT_AMENDMENT.json"
    write_json_x(amendment_path, {
        "schema": "route_b_v3_1_person_refinement_contract_invalid_report_amendment_v3",
        "created_utc": utc_now(), "terminal_changed": False, "terminal": terminal,
        "selection_rules_changed": False, "evaluation_rules_changed": False,
        "selection_failure_audit": str(experiment / "FINAL_SELECTION_FAILURE_AUDIT.json"),
        "retention_audit": str(experiment / "RETENTION_NO_SELECTION.json"),
        "reason": "add complete candidate metrics, eligibility gates, service gaps, and no-selection retention evidence",
        "source_commit": None,
    })
    commit = commit_report("Report no-eligible FP32 refinement outcome")
    amendment = json.loads(amendment_path.read_text())
    amendment["source_commit"] = commit
    write_json_atomic(amendment_path, amendment)
    print(json.dumps({
        "experiment": str(experiment), "terminal": terminal,
        "source_commit": commit["head"], "eligible_labels": audit["eligible_labels"],
        "nondominated_labels": audit["nondominated_labels"],
        "selected": None, "v025_run": False,
    }, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    parser.add_argument("--phase1-reproduction", type=Path)
    parser.add_argument("--phase2-qualification", type=Path)
    parser.add_argument("--amend-existing", type=Path)
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    if args.amend_existing is not None:
        if args.timestamp or args.phase1_reproduction or args.phase2_qualification:
            raise ValueError("--amend-existing is mutually exclusive with run creation arguments")
        return amend_existing(args.amend_existing)
    if args.phase1_reproduction is None or args.phase2_qualification is None:
        parser.error("fresh runs require --phase1-reproduction and --phase2-qualification")
    started = time.monotonic()
    experiment, config_path, policy_path, policy, phase2, execution_head = setup(
        args.timestamp, args.phase1_reproduction, args.phase2_qualification,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    retry = {"used": 0}
    terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
    selection: dict[str, Any] | None = None
    retention: dict[str, Any] | None = None
    error: str | None = None
    try:
        core.progress(experiment, "numerical_repair_preflight")
        preflight(experiment, config_path, policy_path, policy, phase2)
        core.progress(experiment, "full_fp32_policy_registration")
        numerical_registration = register_policy(experiment, policy, phase2)
        train_candidate(experiment, config_path, numerical_registration, config, retry)
        base_record = json.loads((experiment / "decisions/epoch_040_decode.json").read_text())
        base_record["checkpoint"] = str(BASE_CHECKPOINT)
        selection = core.evaluate_and_select(experiment, config_path, base_record, config)
        terminal = selection["terminal"]
        acceptance = json.loads((experiment / "ACCEPTED_BASE_DECISION.json").read_text())
        retention = accepted.retention_audit(experiment, selection, config, acceptance)
    except NumericalUnresolved as exc:
        terminal, error = (
            "LRASPP_PERSON_REFINEMENT_NUMERICAL_ROOT_CAUSE_UNRESOLVED",
            f"{type(exc).__name__}: {exc}",
        )
    except (ContractInvalid, core.ContractInvalid, accepted.ContractInvalid) as exc:
        terminal, error = (
            "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID",
            f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE", f"{type(exc).__name__}: {exc}"
    if terminal not in TERMINALS:
        terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
        error = (error + "; " if error else "") + "unauthorized terminal produced"
    notification = notify(terminal, experiment)
    report = make_report(
        experiment, terminal, config, policy, selection, retention, notification,
        retry["used"], error, time.monotonic() - started,
    )
    write_text_x(experiment / "FINAL_REPORT.md", report)
    TRACKED_REPORT.write_text(report, encoding="utf-8")
    write_json_x(experiment / "NOTIFICATION.json", notification)
    try:
        commit_result = commit_report()
    except Exception as exc:
        terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
        error = (error + "; " if error else "") + f"{type(exc).__name__}: {exc}"
        commit_result = {"error": str(exc)}
    preserved_terminals = {
        "base_recovery": {
            "path": str(HISTORICAL_BASE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
            "value": (HISTORICAL_BASE_EXPERIMENT / "TERMINAL_VERDICT.txt").read_text().strip(),
            "sha256": sha256(HISTORICAL_BASE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
        },
        "batch134_runtime": {
            "path": str(SOURCE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
            "value": (SOURCE_EXPERIMENT / "TERMINAL_VERDICT.txt").read_text().strip(),
            "sha256": sha256(SOURCE_EXPERIMENT / "TERMINAL_VERDICT.txt"),
        },
    }
    terminals_unchanged = (
        preserved_terminals["base_recovery"]["sha256"]
        == "a34671b16116c8bca0d607b31a53e2a2c43aa69d5c09a317fb754863426404a2"
        and preserved_terminals["batch134_runtime"]["sha256"]
        == policy["source_failed_terminal_sha256"]
    )
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_person_refinement_numerical_repair_complete_v3",
        "created_utc": utc_now(), "terminal": terminal, "error": error,
        "retry_used": retry["used"], "notification": notification,
        "source_commit": commit_result, "required_starting_head": execution_head,
        "preserved_historical_terminals": preserved_terminals,
        "historical_terminals_unchanged": terminals_unchanged,
        "base_recovery_repeated": False, "epochs_11_through_40_repeated": False,
        "base_inference_repeated": False, "base_diagnostic_repeated": False,
        "registration_repeated": False, "candidate_restart_epoch": 1,
        "wall_seconds": time.monotonic() - started,
    })
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
    status = json.loads((experiment / "STATUS.json").read_text())
    status.update({
        "phase": "complete", "terminal": terminal, "detail": error or "",
        "updated_utc": utc_now(), "source_commit": commit_result.get("head"),
    })
    write_json_atomic(experiment / "STATUS.json", status)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment), "error": error,
        "selected": selection["selected"]["label"] if selection else None,
        "selected_checkpoint": selection["selected"]["checkpoint"] if selection else None,
        "selected_sha256": selection["selected"]["checkpoint_sha256"] if selection else None,
        "source_commit": commit_result.get("head"),
        "historical_terminals_unchanged": terminals_unchanged,
    }, indent=2), flush=True)
    return 0 if terminal in {
        "LRASPP_PERSON_REFINEMENT_SERVICE_READY",
        "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
        "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

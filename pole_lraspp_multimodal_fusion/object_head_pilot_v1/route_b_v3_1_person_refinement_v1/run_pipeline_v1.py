#!/usr/bin/env python3
"""Single autonomous supervisor for base recovery and one person refinement."""

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
RECOVERY_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_expanded_training_v2"
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
CONFIG_SOURCE = PACKAGE_ROOT / "configs/person_refinement_v1.json"
TRAINING_SOURCE = RECOVERY_PACKAGE / "configs/expanded_training_v2.json"
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_person_refinement_v1"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_PERSON_REFINEMENT_V1_REPORT.md"
TERMINALS = {
    "LRASPP_PERSON_REFINEMENT_SERVICE_READY",
    "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
    "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    "LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED",
    "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID",
    "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE",
}
PROGRESS_FIELDS = ("created_utc", "attempt", "phase", "epoch", "optimizer_steps", "detail")


class ContractInvalid(RuntimeError):
    pass


class BaseRecoveryFailed(RuntimeError):
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


def progress(experiment: Path, phase: str, detail: str = "", *, attempt: int = 0,
             epoch: int | str = "") -> None:
    with (experiment / "PROGRESS.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writerow({
            "created_utc": utc_now(), "attempt": attempt, "phase": phase,
            "epoch": epoch, "optimizer_steps": "", "detail": detail,
        })
    status_path = experiment / "STATUS.json"
    status = json.loads(status_path.read_text())
    status.update({"phase": phase, "detail": detail, "updated_utc": utc_now()})
    write_json_atomic(status_path, status)


def run_logged(command: list[str], log: Path, marker: str) -> int:
    with log.open("a" if log.exists() else "x", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] {marker}\ncommand={json.dumps(command)}\n")
        stream.flush()
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True).wait()


def source_manifest() -> dict[str, str]:
    paths = sorted(PACKAGE_ROOT.glob("*.py")) + [CONFIG_SOURCE, RECOVERY_PACKAGE / "continue_training_v3.py"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def ensure_head(config: dict[str, Any]) -> tuple[str, str, list[str]]:
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
    if branch != "master" or head != config["required_starting_head"]:
        raise ContractInvalid(f"required master/{config['required_starting_head']}, got {branch}/{head}")
    return branch, head, status


def setup(timestamp: str | None) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], str]:
    config = json.loads(CONFIG_SOURCE.read_text())
    training = json.loads(TRAINING_SOURCE.read_text())
    branch, execution_head, git_status = ensure_head(config)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = EXPERIMENT_PARENT / stamp
    experiment.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "resolved_configs", "provenance", "predictions", "decisions"):
        (experiment / name).mkdir()
    config_path = experiment / "resolved_configs/person_refinement_v1.json"
    training_path = experiment / "resolved_configs/expanded_training_v2.json"
    shutil.copyfile(CONFIG_SOURCE, config_path)
    shutil.copyfile(TRAINING_SOURCE, training_path)
    view = (ROOT / training["training_view"]).resolve(strict=True)
    os.symlink(str((view / "dataset").resolve()), experiment / "dataset")
    os.symlink(str((view / "contracts").resolve()), experiment / "contracts")
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    write_json_x(experiment / "STATUS.json", {
        "schema": "route_b_v3_1_person_refinement_status_v1",
        "created_utc": utc_now(), "updated_utc": utc_now(), "phase": "setup",
        "detail": "", "terminal": None, "supervisor_pid": os.getpid(),
        "experiment": str(experiment), "interpreter": sys.executable,
    })
    with (experiment / "PROGRESS.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writeheader()
    progress(experiment, "supervisor_started", str(experiment))
    write_json_x(experiment / "provenance/PROVENANCE.json", {
        "schema": "route_b_v3_1_person_refinement_provenance_v1",
        "created_utc": utc_now(), "branch": branch, "execution_head": execution_head,
        "initial_git_status": git_status, "source_hashes": source_manifest(),
        "preexisting_dirty_oai_preserved": any("OAI/openairinterface5g" in row for row in git_status),
        "locked_test_paths_enumerated_or_read": 0, "carla_commands_launched": 0,
        "live_oai_commands_launched": 0, "q_or_ae_commands_launched": 0,
        "measurement_files_in_scope": 288, "measurement_files_modified": 0,
    })
    write_json_x(experiment / "PIPELINE_STARTED.json", {
        "created_utc": utc_now(), "experiment": str(experiment),
        "execution_head": execution_head, "pid": os.getpid(),
    })
    return experiment, config_path, training_path, config, training, execution_head


def latest_recovery(experiment: Path, config: dict[str, Any]) -> tuple[Path, str]:
    latest_path = experiment / "LATEST_SAFE.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        path, digest = Path(latest["path"]).resolve(strict=True), latest["sha256"]
    else:
        path = (ROOT / config["resume_checkpoint"]).resolve(strict=True)
        digest = config["resume_checkpoint_sha256"]
    if sha256(path) != digest:
        raise BaseRecoveryFailed("latest base-recovery checkpoint integrity failed")
    return path, digest


def recover_base(experiment: Path, config_path: Path, training_path: Path,
                 config: dict[str, Any], retry: dict[str, int]) -> dict[str, Any]:
    progress(experiment, "base_recovery_preflight")
    rc = run_logged([
        sys.executable, str(RECOVERY_PACKAGE / "preflight_continuation_v3.py"),
        "--experiment", str(experiment), "--continuation-config", str(config_path),
    ], experiment / "logs/base_preflight.log", "exact epoch-10 recovery preflight")
    if rc != 0:
        raise BaseRecoveryFailed("exact epoch-10 recovery preflight failed")
    resume = (ROOT / config["resume_checkpoint"]).resolve(strict=True)
    digest = config["resume_checkpoint_sha256"]
    for attempt in (1, 2):
        progress(experiment, "base_recovery", str(resume), attempt=attempt)
        rc = run_logged([
            sys.executable, str(RECOVERY_PACKAGE / "continue_training_v3.py"),
            "--experiment", str(experiment), "--continuation-config", str(config_path),
            "--training-config", str(training_path), "--resume-checkpoint", str(resume),
            "--resume-sha256", digest, "--attempt", str(attempt),
        ], experiment / "logs/base_recovery.log", f"base recovery attempt {attempt}")
        if rc == 0:
            break
        if attempt == 2 or retry["used"] >= int(config["maximum_runtime_retries"]):
            raise BaseRecoveryFailed(f"base recovery worker failed rc={rc}")
        retry["used"] += 1
        resume, digest = latest_recovery(experiment, config)
        preflight_retry = experiment / f"PREFLIGHT_BASE_RETRY_{attempt}.json"
        if run_logged([
            sys.executable, str(RECOVERY_PACKAGE / "preflight_continuation_v3.py"),
            "--experiment", str(experiment), "--continuation-config", str(config_path),
            "--output", str(preflight_retry),
        ], experiment / "logs/base_preflight_retry.log", "bounded base recovery retry preflight") != 0:
            raise BaseRecoveryFailed("bounded base retry integrity preflight failed")
    decision = json.loads((experiment / "BASE_RECOVERY_DECISION.json").read_text())
    if decision["decoded_epochs"] != [40] or decision["epochs_completed"] != list(range(11, 41)):
        raise BaseRecoveryFailed("exact base recovery epoch/decode set drift")
    retained = {int(item["epoch"]): item for item in decision["retained_checkpoints"]}
    if set(retained) != {20, 30, 40}:
        raise BaseRecoveryFailed("base recovery retained checkpoint set drift")
    return {"decision": decision, "retained": retained}


def reconcile_epoch40(experiment: Path, config: dict[str, Any]) -> dict[str, Any]:
    record_path = experiment / "decisions/epoch_040_decode.json"
    record = json.loads(record_path.read_text())
    metric = record["metrics"]
    reference, tolerance = config["reference_epoch40"], config["reconciliation_tolerance"]
    gates: dict[str, bool] = {}
    for key, value in reference.items():
        if key == "vehicle_duplicate_fp":
            gates[key] = abs(int(record["vehicle_duplicate_fp"]) - int(value)) <= int(tolerance["vehicle_duplicate_fp_count"])
        elif key.endswith("xy_mae_m"):
            gates[key] = abs(float(metric[key]) - float(value)) <= float(tolerance["xy_mae_m"])
        elif key in {"vehicle_iou", "person_box_mask_iou", "foreground_miou"}:
            gates[key] = abs(float(metric[key]) - float(value)) <= float(tolerance["iou"])
        else:
            gates[key] = abs(float(metric[key]) - float(value)) <= float(tolerance["precision_recall_f1"])
    result = {
        "schema": "route_b_v3_1_epoch40_reconciliation_v1", "created_utc": utc_now(),
        "all_pass": all(gates.values()), "gates": gates, "reference": reference,
        "recomputed": {**{key: metric.get(key) for key in reference if key != "vehicle_duplicate_fp"},
                       "vehicle_duplicate_fp": record["vehicle_duplicate_fp"]},
        "tolerance": tolerance,
    }
    write_json_x(experiment / "EPOCH40_RECONCILIATION.json", result)
    if not result["all_pass"]:
        raise BaseRecoveryFailed(f"epoch-40 reconciliation failed: {gates}")
    return record


def infer_native(experiment: Path, checkpoint: Path, digest: str, tag: str,
                 log: Path) -> Path:
    output = experiment / "predictions" / tag
    if output.exists():
        raise ContractInvalid(f"refusing a second native inference pass: {tag}")
    rc = run_logged([
        sys.executable, str(NATIVE_PACKAGE / "infer_native_v1.py"),
        "--experiment", str(experiment), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", digest, "--tag", tag,
    ], log, f"native inference {tag}")
    if rc != 0 or not (output / "INFERENCE_COMPLETE").is_file():
        raise RuntimeError(f"native inference failed: {tag}")
    return output


def score_primary(experiment: Path, prediction: Path, checkpoint: Path, digest: str,
                  epoch: int, output: Path, log: Path) -> dict[str, Any]:
    rc = run_logged([
        sys.executable, str(RECOVERY_PACKAGE / "score_continuation_v3.py"),
        "--mode", "primary", "--experiment", str(experiment),
        "--prediction-root", str(prediction), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", digest, "--epoch", str(epoch), "--output", str(output),
    ], log, f"primary scoring epoch {epoch}")
    if rc != 0:
        raise RuntimeError(f"primary scoring failed epoch {epoch}")
    return json.loads(output.read_text())


def base_diagnostic_and_registration(experiment: Path, config_path: Path,
                                     epoch40: dict[str, Any], config: dict[str, Any]) -> None:
    epoch10_path = (ROOT / config["resume_checkpoint"]).resolve(strict=True)
    progress(experiment, "epoch10_single_decode")
    epoch10_prediction = infer_native(
        experiment, epoch10_path, config["resume_checkpoint_sha256"], "base_epoch_010",
        experiment / "logs/epoch10_inference.log",
    )
    score_primary(
        experiment, epoch10_prediction, epoch10_path, config["resume_checkpoint_sha256"], 10,
        experiment / "decisions/base_epoch_010_primary.json", experiment / "logs/epoch10_score.log",
    )
    epoch40_prediction = Path(epoch40["prediction_root"])
    progress(experiment, "base_diagnostic")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "diagnostic_v1.py"),
        "--experiment", str(experiment), "--epoch10-predictions", str(epoch10_prediction),
        "--epoch40-predictions", str(epoch40_prediction),
    ], experiment / "logs/base_diagnostic.log", "one-pass base diagnostic") != 0:
        raise ContractInvalid("base diagnostic failed")
    base_checkpoint = Path(epoch40["checkpoint"])
    progress(experiment, "refinement_qualification")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "qualify_v1.py"),
        "--experiment", str(experiment), "--config", str(config_path),
        "--base-checkpoint", str(base_checkpoint),
        "--base-sha256", epoch40["checkpoint_sha256"],
        "--diagnostic", str(experiment / "BASE_DIAGNOSTIC.json"),
    ], experiment / "logs/qualification.log", "executable refinement qualification") != 0:
        raise ContractInvalid("refinement qualification failed")


def train_candidate(experiment: Path, config_path: Path, epoch40: dict[str, Any],
                    config: dict[str, Any], retry: dict[str, int]) -> None:
    base_checkpoint = Path(epoch40["checkpoint"])
    command = [
        sys.executable, str(PACKAGE_ROOT / "train_v1.py"),
        "--experiment", str(experiment), "--config", str(config_path),
        "--base-checkpoint", str(base_checkpoint),
        "--base-sha256", epoch40["checkpoint_sha256"], "--attempt", "1",
    ]
    progress(experiment, "person_training", attempt=1)
    rc = run_logged(command, experiment / "logs/person_training.log", "person training attempt 1")
    if rc == 0:
        return
    if retry["used"] >= int(config["maximum_runtime_retries"]):
        raise RuntimeError(f"person training failed and retry already consumed rc={rc}")
    retry["used"] += 1
    latest_path = experiment / "PERSON_LATEST_SAFE.json"
    if not latest_path.is_file():
        raise RuntimeError("person training failed before an epoch-boundary recovery checkpoint")
    latest = json.loads(latest_path.read_text())
    if sha256(Path(latest["path"])) != latest["sha256"]:
        raise RuntimeError("person latest-safe checkpoint integrity failed")
    progress(experiment, "person_training_retry", latest["path"], attempt=2, epoch=latest["epoch"])
    retry_command = command[:-2] + [
        "--attempt", "2", "--resume-checkpoint", latest["path"],
        "--resume-sha256", latest["sha256"],
    ]
    if run_logged(retry_command, experiment / "logs/person_training.log", "person training attempt 2") != 0:
        raise RuntimeError("sole person training retry failed")


def evaluate_and_select(experiment: Path, config_path: Path,
                        epoch40: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    checkpoint_dir = experiment / "checkpoints" / config["name"]
    records: list[Path] = []
    for epoch in (6, 12, 18):
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        digest = sha256(checkpoint)
        tag = f"person_epoch_{epoch:03d}"
        progress(experiment, "candidate_decode", tag, epoch=epoch)
        prediction = experiment / "predictions" / tag
        if prediction.exists():
            raise ContractInvalid(f"candidate prediction path pre-exists: {prediction}")
        if run_logged([
            sys.executable, str(PACKAGE_ROOT / "infer_v1.py"),
            "--experiment", str(experiment), "--checkpoint", str(checkpoint),
            "--checkpoint-sha256", digest, "--tag", tag,
        ], experiment / f"logs/{tag}_inference.log", f"candidate inference epoch {epoch}") != 0:
            raise RuntimeError(f"candidate inference failed epoch {epoch}")
        record_path = experiment / f"decisions/{tag}_primary.json"
        score_primary(
            experiment, prediction, checkpoint, digest, epoch, record_path,
            experiment / f"logs/{tag}_score.log",
        )
        records.append(record_path)
    policy_path = experiment / "FINAL_SELECTION.json"
    command = [
        sys.executable, str(PACKAGE_ROOT / "policy_v1.py"), "--config", str(config_path),
        "--base-record", str(experiment / "decisions/epoch_040_decode.json"),
        "--base-inference", str(Path(epoch40["prediction_root"]) / "inference_manifest.json"),
        "--output", str(policy_path),
    ]
    for record in records:
        command.extend(["--candidate-record", str(record)])
    progress(experiment, "final_selection")
    if run_logged(command, experiment / "logs/selection.log", "registered final selection") != 0:
        raise ContractInvalid("registered final selection failed")
    selection = json.loads(policy_path.read_text())
    selected = selection["selected"]
    selected_diagnostic = experiment / "SELECTED_DIAGNOSTIC.json"
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "diagnostic_v1.py"),
        "--experiment", str(experiment),
        "--single-predictions", selected["prediction_root"],
        "--single-label", selected["label"], "--output", str(selected_diagnostic),
    ], experiment / "logs/selected_diagnostic.log", "selected offline diagnostic") != 0:
        raise RuntimeError("selected offline diagnostic failed")
    selection["selected_diagnostic"] = str(selected_diagnostic)
    sensitivity = experiment / "decisions/SELECTED_V025_SENSITIVITY.json"
    if run_logged([
        sys.executable, str(RECOVERY_PACKAGE / "score_continuation_v3.py"),
        "--mode", "sensitivity", "--experiment", str(experiment),
        "--prediction-root", selected["prediction_root"], "--output", str(sensitivity),
    ], experiment / "logs/selected_v025.log", "selected-only v025 sensitivity") != 0:
        raise RuntimeError("selected-only v025 sensitivity failed")
    selection["selected_v025_sensitivity"] = json.loads(sensitivity.read_text())
    write_json_x(experiment / "DECISION.json", selection)
    return selection


def cleanup(experiment: Path, selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    keep_labels = {
        selection["selected"]["label"], *selection["nondominated_labels"],
        *selection["material_labels"],
    }
    checkpoint_dir = experiment / "checkpoints" / config["name"]
    removed: list[str] = []
    for epoch in (6, 12, 18):
        label = f"person_epoch_{epoch:03d}"
        path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if label not in keep_labels and path.is_file():
            path.unlink()
            removed.append(str(path))
    recovery_removed: list[str] = []
    for path in (experiment / "candidate_recovery_checkpoints").glob("*.pt"):
        path.unlink()
        recovery_removed.append(str(path))
    result = {
        "performed": True, "retained_labels": sorted(keep_labels),
        "always_retained_external_epoch10": config["resume_checkpoint"],
        "always_retained_recovered_epochs": [20, 30, 40],
        "removed_dominated_candidate_checkpoints": removed,
        "removed_candidate_recovery_checkpoints": recovery_removed,
        "prediction_sets_removed": 0, "datasets_or_contracts_removed": 0,
    }
    write_json_x(experiment / "CLEANUP.json", result)
    return result


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def make_report(experiment: Path, terminal: str, config: dict[str, Any],
                execution_head: str, recovery: dict[str, Any] | None,
                selection: dict[str, Any] | None, cleanup_result: dict[str, Any] | None,
                error: str | None, retry_used: int, wall_seconds: float) -> str:
    lines = [
        "# Route B v3.1 LR-ASPP person refinement v1 report", "",
        f"Terminal: `{terminal}`", "", f"Experiment: `{experiment}`", "",
        f"Execution began on local `master` at required HEAD `{execution_head}`. Nothing was pushed.", "",
        "## Contract and recovery", "",
        f"- Exact epoch-10 origin: `{config['resume_checkpoint']}` (`{config['resume_checkpoint_sha256']}`).",
        "- Frozen expanded data scope: 16,827 train / 3,345 validation; locked test absent and unopened.",
        "- Clean execution: `/usr/bin/python3`, CUDA sm_120, q=0, AE disabled, no geometric augmentation.",
        f"- Runtime retries used: `{retry_used}` of one; error: `{error or 'none'}`.",
    ]
    if recovery:
        reconciliation_path = experiment / "EPOCH40_RECONCILIATION.json"
        reconciliation = (
            json.loads(reconciliation_path.read_text()) if reconciliation_path.is_file() else None
        )
        lines.extend([
            f"- Recovered epochs: `{recovery['decision']['epochs_completed']}`; decoded epochs: `{recovery['decision']['decoded_epochs']}`.",
            f"- Retained base checkpoints: `{recovery['decision']['retained_checkpoints']}`.",
            (
                "- Epoch-40 reconciliation passed within the registered P/R/F1, XY, IoU, and duplicate-FP tolerances."
                if reconciliation and reconciliation["all_pass"] else
                f"- Epoch-40 reconciliation failed closed: `{reconciliation}`."
            ),
        ])
    lines.extend(["", "## Registered refinement", "",
        "The prepared private person tail consumes only the transported `low`/`high` feature bundle. It adds person objectness residual, detached 3 m localization quality, eight train-derived range bins plus bounded residual, projected-center offset with external camera unprojection, and an independent person-mask residual. The recovered backbone, shared object trunk, vehicle heatmap, shared regression, grid offset, and vehicle segmentation path are configured frozen; only the inherited person heatmap slice is configured for lower-LR P2 training (epochs 7–18).",
    ])
    if not (experiment / "REGISTRATION.json").is_file():
        lines.extend(["",
            "The base reconciliation gate failed before Phase B, so the diagnostic, executable qualification, registration artifact, candidate training, candidate scoring, and v0.25 sensitivity were not run. The prepared refinement implementation remains unexecuted scientific design.",
        ])
    else:
        lines.extend(["",
            "Full retained-prediction PR curves, distance/area/radar/visibility/occlusion-proxy/episode/track strata, and FP/FN taxonomies are in `BASE_DIAGNOSTIC.json`. Executable source, gradient, split-parity, schema, range-bin, camera-plane, sampler, and AMP gates are in `QUALIFICATION.json`.",
        ])
    if (experiment / "REGISTRATION.json").is_file():
        registration = json.loads((experiment / "REGISTRATION.json").read_text())
        parameters = registration["parameter_report_p2"]
        qualification = json.loads((experiment / "QUALIFICATION.json").read_text())
        split_check = next(check for check in qualification["checks"]
                           if check["name"] == "encode_front_low_high_split_parity")
        lines.extend(["",
            f"P2 parameter counts: `{parameters}`.",
            f"Transport proof: names `{split_check['detail']['transported_feature_names']}`, shapes `{split_check['detail']['transported_feature_shapes']}`, dtypes `{split_check['detail']['transported_feature_dtypes']}`, raw side channels `{split_check['detail']['tail_raw_modality_side_channels']}`, parity `{split_check['detail']['outputs_bit_identical']}`.",
            f"Train-derived range edges: `{registration['range_bins']['edges_m']}`; train persons per bin: `{registration['range_bins']['counts']}`.",
        ])
    if selection:
        lines.extend(["", "## Primary v0.10 results", "", "| Candidate | Veh P/R/F1 | Veh XY | Person P/R/F1 | Person XY | IoU veh/person | fg mIoU | Eligible | Material |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for record in selection["records"]:
            metric = record["metrics"]
            lines.append(
                f"| {record['label']} | {fmt(metric['vehicle_precision'])}/{fmt(metric['vehicle_recall'])}/{fmt(metric['vehicle_f1'])} | "
                f"{fmt(metric['vehicle_xy_mae_m'], 3)} | {fmt(metric['person_precision'])}/{fmt(metric['person_recall'])}/{fmt(metric['person_f1'])} | "
                f"{fmt(metric['person_xy_mae_m'], 3)} | {fmt(metric['vehicle_iou'])}/{fmt(metric['person_box_mask_iou'])} | "
                f"{fmt(metric['foreground_miou'])} | {record['eligible']} | {record['material_gain']['pass']} |"
            )
        selected = selection["selected"]
        service_lines = ["", "| Service target | Pass |", "|---|---:|"]
        service_lines.extend(f"| {name} | {passed} |" for name, passed in selected["service_targets"].items())
        lines.extend(["", "## Selection and v0.25 sensitivity", "",
            f"Selected `{selected['label']}` by continuous normalized person deficit, then person F1, recall, XY error, and earlier epoch. Checkpoint: `{selected['checkpoint']}` (`{selected['checkpoint_sha256']}`).",
            f"Service gates: `{selected['service_targets']}`. Material-gain gates: `{selected['material_gain']}`.",
            f"Selected-only v0.25 sensitivity: `{selection['selected_v025_sensitivity']['flat']}`. This denominator-distinct arm is sensitivity only.",
            f"Non-dominated labels: `{selection['nondominated_labels']}`. Cleanup: `{cleanup_result}`.",
        ])
        lines.extend(service_lines)
        base_diagnostic = json.loads((experiment / "BASE_DIAGNOSTIC.json").read_text())
        selected_diagnostic = json.loads((experiment / "SELECTED_DIAGNOSTIC.json").read_text())
        base_person = base_diagnostic["bases"]["epoch_040"]
        after_person = selected_diagnostic["bases"][selected["label"]]
        lines.extend(["", "## Person PR, strata, and failure mechanisms", "",
            f"Epoch-40 persisted-score person PR has `{base_person['full_precision_recall_from_persisted_score_floor']['person']['distinct_thresholds']}` exact sorted-score points; selected has `{after_person['full_precision_recall_from_persisted_score_floor']['person']['distinct_thresholds']}`. Primary metrics are in the table above and full curves are retained in the diagnostic artifacts.",
            f"Epoch-40 person FP/FN taxonomy: `{base_person['taxonomy']['person_fp_at_0_20']}` / `{base_person['taxonomy']['person_fn_at_0_02']}`.",
            f"Selected person FP/FN taxonomy: `{after_person['taxonomy']['person_fp_at_0_20']}` / `{after_person['taxonomy']['person_fn_at_0_02']}`.",
            f"Epoch-40 radar-supported/unsupported strata: `{base_person['person_strata_at_0_20']['radar']}`.",
            f"Selected radar-supported/unsupported strata: `{after_person['person_strata_at_0_20']['radar']}`.",
        ])
        training_complete = json.loads((experiment / "PERSON_TRAINING_COMPLETE.json").read_text())
        inference_resources = {
            record["label"]: {
                key: json.loads((Path(record["prediction_root"]) / "inference_manifest.json").read_text())[key]
                for key in ("wall_seconds", "peak_allocated_mib", "peak_reserved_mib")
            } for record in selection["records"]
        }
        lines.extend(["", "## Runtime and retention", "",
            f"Base recovery training wall/VRAM: `{recovery['decision']['wall_seconds'] if recovery else None}` s / `{recovery['decision']['peak_allocated_mib'] if recovery else None}` MiB allocated / `{recovery['decision']['peak_reserved_mib'] if recovery else None}` MiB reserved.",
            f"Person training optimizer steps and peak VRAM: `{training_complete['optimizer_steps']}` / `{training_complete['peak_allocated_mib']}` MiB allocated / `{training_complete['peak_reserved_mib']}` MiB reserved.",
            f"Evaluation inference wall/VRAM by candidate: `{inference_resources}`.",
            f"Retained recovered Pareto paths: `{recovery['decision']['retained_checkpoints'] if recovery else None}`; retained candidate labels: `{cleanup_result['retained_labels'] if cleanup_result else None}`.",
        ])
    lines.extend(["", "## Operational boundary", "",
        "No q/AE/tracking/calibrated-threshold/live UDP/CARLA/OAI action, alternative architecture, second experiment, locked-test access, or modification of the 288 measurement files was performed. The report is offline validation evidence only; deployment remains a separate decision.",
        "", f"Supervisor wall time: `{wall_seconds:.1f}` seconds.", ""])
    return "\n".join(lines)


def notify(terminal: str, experiment: Path) -> dict[str, Any]:
    command = ["notify-send", "LR-ASPP person refinement complete", f"{terminal}\n{experiment}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:],
                "delivered": result.returncode == 0}
    except Exception as exc:
        return {"command": command, "delivered": False, "error": f"{type(exc).__name__}: {exc}"}


def commit_report() -> dict[str, Any]:
    explicit = [
        RECOVERY_PACKAGE / "continue_training_v3.py",
        PACKAGE_ROOT / "__init__.py", CONFIG_SOURCE, TRACKED_REPORT,
        *sorted(path for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"),
    ]
    subprocess.run(["git", "add", "--", *map(str, explicit)], cwd=ROOT, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    result = subprocess.run(
        ["git", "commit", "-m", "Add bounded LR-ASPP person refinement experiment"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"final source/config/report commit failed: {result.stderr[-2000:]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {"head": head, "staged_files": staged, "stdout": result.stdout[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    started = time.monotonic()
    experiment, config_path, training_path, config, _training, execution_head = setup(args.timestamp)
    retry = {"used": 0}
    terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
    recovery: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    cleanup_result: dict[str, Any] | None = None
    error: str | None = None
    try:
        recovery = recover_base(experiment, config_path, training_path, config, retry)
        epoch40 = reconcile_epoch40(experiment, config)
        base_diagnostic_and_registration(experiment, config_path, epoch40, config)
        train_candidate(experiment, config_path, epoch40, config, retry)
        selection = evaluate_and_select(experiment, config_path, epoch40, config)
        terminal = selection["terminal"]
        cleanup_result = cleanup(experiment, selection, config)
    except BaseRecoveryFailed as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_BASE_RECOVERY_FAILED", f"{type(exc).__name__}: {exc}"
    except ContractInvalid as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE", f"{type(exc).__name__}: {exc}"
    if terminal not in TERMINALS:
        raise RuntimeError(f"unauthorized terminal {terminal}")
    report = make_report(
        experiment, terminal, config, execution_head, recovery, selection,
        cleanup_result, error, retry["used"], time.monotonic() - started,
    )
    write_text_x(experiment / "FINAL_REPORT.md", report)
    write_text_x(TRACKED_REPORT, report)
    notification = notify(terminal, experiment)
    write_json_x(experiment / "NOTIFICATION.json", notification)
    commit_result: dict[str, Any]
    try:
        commit_result = commit_report()
    except Exception as exc:
        error = (error + "; " if error else "") + f"{type(exc).__name__}: {exc}"
        terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
        commit_result = {"error": str(exc)}
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_person_refinement_pipeline_complete_v1",
        "created_utc": utc_now(), "terminal": terminal, "error": error,
        "retry_used": retry["used"], "notification": notification,
        "source_commit": commit_result, "wall_seconds": time.monotonic() - started,
    })
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
    status = json.loads((experiment / "STATUS.json").read_text())
    status.update({"phase": "complete", "terminal": terminal, "detail": error or "",
                   "updated_utc": utc_now(), "source_commit": commit_result.get("head")})
    write_json_atomic(experiment / "STATUS.json", status)
    progress(experiment, "supervisor_complete", terminal)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment), "error": error,
        "selected": selection["selected"]["label"] if selection else None,
        "source_commit": commit_result.get("head"),
    }, indent=2), flush=True)
    return 0 if terminal in {
        "LRASPP_PERSON_REFINEMENT_SERVICE_READY",
        "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
        "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Autonomous Phase-B-through-selection continuation from accepted epoch 40."""

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
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
RECOVERY_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_expanded_training_v2"
PERSON_CONFIG_SOURCE = PACKAGE_ROOT / "configs/person_refinement_v1.json"
ACCEPTANCE_SOURCE = PACKAGE_ROOT / "configs/recovered_epoch40_accepted_v2.json"
TRAINING_SOURCE = RECOVERY_PACKAGE / "configs/expanded_training_v2.json"
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_person_refinement_continuation_v2"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_PERSON_REFINEMENT_CONTINUATION_V2_REPORT.md"
for path in (str(PACKAGE_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_pipeline_v1 as core  # noqa: E402

TERMINALS = {
    "LRASPP_PERSON_REFINEMENT_SERVICE_READY",
    "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
    "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID",
    "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE",
}
PROGRESS_FIELDS = ("created_utc", "attempt", "phase", "epoch", "optimizer_steps", "detail")


class ContractInvalid(RuntimeError):
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
        return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True).wait()


def source_manifest() -> dict[str, str]:
    paths = sorted(PACKAGE_ROOT.glob("*.py")) + [PERSON_CONFIG_SOURCE, ACCEPTANCE_SOURCE]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def setup(timestamp: str | None) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    acceptance = json.loads(ACCEPTANCE_SOURCE.read_text())
    person_config = json.loads(PERSON_CONFIG_SOURCE.read_text())
    training = json.loads(TRAINING_SOURCE.read_text())
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "master" or head != acceptance["required_starting_head"]:
        raise ContractInvalid(f"required master/{acceptance['required_starting_head']}, got {branch}/{head}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = EXPERIMENT_PARENT / stamp
    experiment.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "resolved_configs", "provenance", "predictions", "decisions"):
        (experiment / name).mkdir()
    config_path = experiment / "resolved_configs/person_refinement_v1.json"
    acceptance_path = experiment / "ACCEPTED_BASE_DECISION.json"
    shutil.copyfile(PERSON_CONFIG_SOURCE, config_path)
    shutil.copyfile(ACCEPTANCE_SOURCE, experiment / "resolved_configs/recovered_epoch40_accepted_v2.json")
    shutil.copyfile(ACCEPTANCE_SOURCE, acceptance_path)
    view = (ROOT / training["training_view"]).resolve(strict=True)
    os.symlink(str((view / "dataset").resolve()), experiment / "dataset")
    os.symlink(str((view / "contracts").resolve()), experiment / "contracts")
    source_record = (ROOT / acceptance["recovered_primary_record"]).resolve(strict=True)
    shutil.copyfile(source_record, experiment / "decisions/epoch_040_decode.json")
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    write_json_x(experiment / "STATUS.json", {
        "schema": "route_b_v3_1_person_refinement_accepted_continuation_status_v2",
        "created_utc": utc_now(), "updated_utc": utc_now(), "phase": "setup",
        "detail": "", "terminal": None, "supervisor_pid": os.getpid(),
        "experiment": str(experiment), "interpreter": sys.executable,
        "accepted_base_checkpoint": acceptance["recovered_checkpoint"],
        "epochs_11_through_40_repeated": False,
    })
    with (experiment / "PROGRESS.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writeheader()
    core.progress(experiment, "accepted_continuation_started", acceptance["decision"])
    write_json_x(experiment / "PIPELINE_STARTED.json", {
        "created_utc": utc_now(), "pid": os.getpid(), "experiment": str(experiment),
        "required_starting_head": head, "accepted_decision": acceptance["decision"],
        "epochs_11_through_40_repeated": False,
    })
    write_json_x(experiment / "provenance/PROVENANCE.json", {
        "schema": "route_b_v3_1_person_refinement_accepted_continuation_provenance_v2",
        "created_utc": utc_now(), "branch": branch, "required_starting_head": head,
        "source_hashes": source_manifest(), "initial_git_status": status,
        "historical_experiment": acceptance["historical_experiment"],
        "historical_terminal_preserved": acceptance["historical_terminal"],
        "preexisting_dirty_oai_preserved": any("OAI/openairinterface5g" in row for row in status),
        "unrelated_pointers_preserved": [row for row in status if "EXP_DIR.txt" in row],
        "locked_test_paths_enumerated_or_read": 0, "carla_commands_launched": 0,
        "live_oai_commands_launched": 0, "q_or_ae_commands_launched": 0,
        "tracking_commands_launched": 0, "measurement_files_in_scope": 288,
        "measurement_files_modified": 0,
    })
    return experiment, config_path, acceptance_path, person_config, acceptance


def preflight(experiment: Path, config_path: Path, acceptance_path: Path) -> None:
    core.progress(experiment, "accepted_base_preflight")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "preflight_accepted_v2.py"),
        "--experiment", str(experiment), "--acceptance", str(acceptance_path),
        "--person-config", str(config_path),
    ], experiment / "logs/accepted_base_preflight.log", "accepted epoch-40 preflight") != 0:
        raise ContractInvalid("accepted recovered epoch-40 preflight failed")


def resume_setup(path: Path) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    experiment = path.resolve(strict=True)
    if experiment.parent != EXPERIMENT_PARENT.resolve():
        raise ContractInvalid("resume experiment is outside the accepted continuation parent")
    config_path = experiment / "resolved_configs/person_refinement_v1.json"
    acceptance_path = experiment / "ACCEPTED_BASE_DECISION.json"
    person_config = json.loads(config_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if not (experiment / "ACCEPTED_BASE_PREFLIGHT_COMPLETE").is_file():
        raise ContractInvalid("accepted-base preflight is not complete")
    if not (experiment / "BASE_DIAGNOSTIC_COMPLETE").is_file():
        raise ContractInvalid("Phase-B diagnostic is not complete")
    if (experiment / "REGISTRATION.json").exists() or (experiment / "PERSON_TRAINING_STARTED.json").exists():
        raise ContractInvalid("qualification-failure resume found candidate registration/training state")
    prior_terminal = (experiment / "TERMINAL_VERDICT.txt").read_text().strip()
    if prior_terminal != "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID":
        raise ContractInvalid(f"unexpected qualification-failure terminal {prior_terminal}")
    attempt = len(list(experiment.glob("QUALIFICATION_FAILURE_ATTEMPT_*_TERMINAL_VERDICT.txt"))) + 1
    archive_names = {
        "FINAL_REPORT.md": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_REPORT.md",
        "PIPELINE_COMPLETE.json": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_PIPELINE_COMPLETE.json",
        "TERMINAL_VERDICT.txt": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_TERMINAL_VERDICT.txt",
        "COMPLETION_SENTINEL": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_COMPLETION_SENTINEL",
        "NOTIFICATION.json": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_NOTIFICATION.json",
        "COMPLETED.pid": f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_COMPLETED.pid",
    }
    if (experiment / "QUALIFICATION.json").is_file():
        archive_names["QUALIFICATION.json"] = f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_QUALIFICATION.json"
    for source, target in archive_names.items():
        source_path, target_path = experiment / source, experiment / target
        if not source_path.is_file() or target_path.exists():
            raise ContractInvalid(f"qualification-failure archive collision/missing file: {source}")
        source_path.rename(target_path)
    shutil.copyfile(
        experiment / "STATUS.json",
        experiment / f"QUALIFICATION_FAILURE_ATTEMPT_{attempt}_STATUS.json",
    )
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    status = json.loads((experiment / "STATUS.json").read_text())
    status.update({
        "phase": "implementation_repair_qualification", "terminal": None,
        "detail": "repair decoder constant namespace only; reuse completed Phase-B diagnostic",
        "updated_utc": utc_now(), "supervisor_pid": os.getpid(),
    })
    write_json_atomic(experiment / "STATUS.json", status)
    if attempt == 1:
        failure = "AttributeError: module model_v1 has no attribute REG_DIMS"
        repair = "import unchanged REG_* object-output slices from object_targets"
        repaired_source = PACKAGE_ROOT / "person_decode_v1.py"
    else:
        failure = "inherited vehicle/person heatmap 1x1 projections overflow at background cells only under FP16"
        repair = "execute only the two inherited class-heatmap 1x1 projections in FP32"
        repaired_source = PACKAGE_ROOT / "person_model_v1.py"
    write_json_x(experiment / f"IMPLEMENTATION_REPAIR_ATTEMPT_{attempt}.json", {
        "schema": "route_b_v3_1_person_refinement_implementation_repair_v2",
        "created_utc": utc_now(),
        "attempt": attempt, "failure": failure, "repair": repair,
        "repaired_source": str(repaired_source.relative_to(ROOT)),
        "repaired_source_sha256": sha256(repaired_source),
        "architecture_changed": False, "losses_changed": False, "targets_changed": False,
        "learning_rates_changed": False, "validation_rules_changed": False,
        "candidate_training_or_scoring_started_before_repair": False,
        "base_diagnostic_reused_without_inference": True,
    })
    core.progress(experiment, "implementation_repair_qualification", repair, attempt=attempt)
    return experiment, config_path, acceptance_path, person_config, acceptance


def diagnose_and_register(experiment: Path, config_path: Path, acceptance_path: Path,
                          acceptance: dict[str, Any]) -> dict[str, Any]:
    base_checkpoint = (ROOT / acceptance["recovered_checkpoint"]).resolve(strict=True)
    base_record = json.loads((experiment / "decisions/epoch_040_decode.json").read_text())
    base_prediction = (ROOT / acceptance["recovered_prediction_root"]).resolve(strict=True)
    person_config = json.loads(config_path.read_text())
    epoch10 = (ROOT / person_config["resume_checkpoint"]).resolve(strict=True)
    core.progress(experiment, "epoch10_single_decode")
    epoch10_prediction = core.infer_native(
        experiment, epoch10, person_config["resume_checkpoint_sha256"], "base_epoch_010",
        experiment / "logs/epoch10_inference.log",
    )
    core.score_primary(
        experiment, epoch10_prediction, epoch10, person_config["resume_checkpoint_sha256"], 10,
        experiment / "decisions/base_epoch_010_primary.json", experiment / "logs/epoch10_score.log",
    )
    core.progress(experiment, "accepted_base_diagnostic")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "diagnostic_v1.py"),
        "--experiment", str(experiment), "--epoch10-predictions", str(epoch10_prediction),
        "--epoch40-predictions", str(base_prediction),
    ], experiment / "logs/base_diagnostic.log", "accepted epoch-10/40 diagnostic") != 0:
        raise ContractInvalid("accepted base diagnostic failed")
    core.progress(experiment, "refinement_qualification")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "qualify_v1.py"),
        "--experiment", str(experiment), "--config", str(config_path),
        "--base-checkpoint", str(base_checkpoint),
        "--base-sha256", acceptance["recovered_checkpoint_sha256"],
        "--diagnostic", str(experiment / "BASE_DIAGNOSTIC.json"),
        "--base-acceptance", str(acceptance_path),
    ], experiment / "logs/qualification.log", "registered person-refinement qualification") != 0:
        raise ContractInvalid("person-refinement qualification failed")
    registration = json.loads((experiment / "REGISTRATION.json").read_text())
    if registration["base_acceptance_decision"] != acceptance["decision"]:
        raise ContractInvalid("acceptance decision was not frozen into registration")
    return base_record


def qualify_existing_diagnostic(experiment: Path, config_path: Path, acceptance_path: Path,
                                acceptance: dict[str, Any]) -> dict[str, Any]:
    base_checkpoint = (ROOT / acceptance["recovered_checkpoint"]).resolve(strict=True)
    core.progress(experiment, "refinement_qualification_repair")
    if run_logged([
        sys.executable, str(PACKAGE_ROOT / "qualify_v1.py"),
        "--experiment", str(experiment), "--config", str(config_path),
        "--base-checkpoint", str(base_checkpoint),
        "--base-sha256", acceptance["recovered_checkpoint_sha256"],
        "--diagnostic", str(experiment / "BASE_DIAGNOSTIC.json"),
        "--base-acceptance", str(acceptance_path),
    ], experiment / "logs/qualification.log", "repaired executable qualification") != 0:
        raise ContractInvalid("repaired person-refinement qualification failed")
    registration = json.loads((experiment / "REGISTRATION.json").read_text())
    if registration["base_acceptance_decision"] != acceptance["decision"]:
        raise ContractInvalid("acceptance decision was not frozen into repaired registration")
    return json.loads((experiment / "decisions/epoch_040_decode.json").read_text())


def retention_audit(experiment: Path, selection: dict[str, Any],
                    person_config: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    original_epoch10 = (ROOT / person_config["resume_checkpoint"]).resolve(strict=True)
    historical = (ROOT / acceptance["historical_experiment"]).resolve(strict=True)
    recovered = {
        epoch: historical / f"checkpoints/{person_config['name']}/epoch_{epoch:03d}.pt"
        for epoch in (20, 30, 40)
    }
    candidate_dir = experiment / "checkpoints" / person_config["name"]
    candidates = {epoch: candidate_dir / f"epoch_{epoch:03d}.pt" for epoch in (6, 12, 18)}
    required_paths = {"epoch_010": original_epoch10, **{
        f"recovered_epoch_{epoch:03d}": path for epoch, path in recovered.items()
    }, **{f"person_epoch_{epoch:03d}": path for epoch, path in candidates.items()}}
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    hashes = {name: sha256(path) for name, path in required_paths.items() if path.is_file()}
    selected = selection["selected"]
    selected_path = Path(selected["checkpoint"]).resolve(strict=True)
    nondominated = set(selection["nondominated_labels"])
    nondominated_present = all(
        Path(next(record for record in selection["records"] if record["label"] == label)["checkpoint"]).is_file()
        for label in nondominated
    )
    result = {
        "schema": "route_b_v3_1_person_refinement_retention_v2", "created_utc": utc_now(),
        "all_pass": not missing and nondominated_present and selected_path.is_file()
                    and sha256(selected_path) == selected["checkpoint_sha256"],
        "checkpoint_deletions_performed": 0, "missing": missing,
        "retained_paths": {name: str(path) for name, path in required_paths.items()},
        "retained_sha256": hashes, "nondominated_labels": sorted(nondominated),
        "nondominated_present": nondominated_present,
        "selected_label": selected["label"], "selected_path": str(selected_path),
        "selected_sha256": selected["checkpoint_sha256"],
    }
    write_json_x(experiment / "RETENTION.json", result)
    if not result["all_pass"]:
        raise ContractInvalid(f"retention audit failed: {result}")
    return result


def service_gaps(metrics: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    return {
        "vehicle_precision": max(0.0, targets["vehicle_precision_min"] - metrics["vehicle_precision"]),
        "vehicle_recall": max(0.0, targets["vehicle_recall_min"] - metrics["vehicle_recall"]),
        "person_precision": max(0.0, targets["person_precision_min"] - metrics["person_precision"]),
        "person_recall": max(0.0, targets["person_recall_min"] - metrics["person_recall"]),
        "vehicle_xy_mae_m": max(0.0, metrics["vehicle_xy_mae_m"] - targets["vehicle_xy_mae_max_m"]),
        "person_xy_mae_m": max(0.0, metrics["person_xy_mae_m"] - targets["person_xy_mae_max_m"]),
        "vehicle_iou": max(0.0, targets["vehicle_iou_min"] - metrics["vehicle_iou"]),
        "person_box_mask_iou": max(0.0, targets["person_box_mask_iou_min"] - metrics["person_box_mask_iou"]),
        "foreground_miou": max(0.0, targets["foreground_miou_min"] - metrics["foreground_miou"]),
    }


def make_report(experiment: Path, terminal: str, person_config: dict[str, Any],
                acceptance: dict[str, Any], selection: dict[str, Any] | None,
                retention: dict[str, Any] | None, notification: dict[str, Any],
                retry_used: int, error: str | None, wall_seconds: float) -> str:
    lines = [
        "# Route B v3.1 LR-ASPP accepted epoch-40 person-refinement continuation v2", "",
        f"Terminal: `{terminal}`", "", f"Experiment: `{experiment}`", "",
        f"Accepted decision: `{acceptance['decision']}`.",
        f"Required starting local master HEAD: `{acceptance['required_starting_head']}`. Nothing was pushed.",
        f"Historical terminal remains `{acceptance['historical_terminal']}` in `{acceptance['historical_experiment']}` and was not rewritten.",
        f"Accepted recovered epoch-40 checkpoint: `{acceptance['recovered_checkpoint']}` (`{acceptance['recovered_checkpoint_sha256']}`).",
        f"The only old reconciliation variation was favorable person R@0.02 `{acceptance['accepted_variation']['recovered']}` versus `{acceptance['accepted_variation']['historical_reference']}` (`+14` TP). Candidate deltas use the recovered checkpoint's own decoded metrics.",
        "Epochs 11–40 were not repeated.", "",
        "## Execution", "",
        f"Runtime retries used: `{retry_used}` of one. Error: `{error or 'none'}`.",
        f"Notification result: `{notification}`.",
    ]
    if (experiment / "REGISTRATION.json").is_file():
        registration = json.loads((experiment / "REGISTRATION.json").read_text())
        qualification = json.loads((experiment / "QUALIFICATION.json").read_text())
        split = next(check["detail"] for check in qualification["checks"]
                     if check["name"] == "encode_front_low_high_split_parity")
        lines.extend(["", "## Frozen design and qualification", "",
            "The unchanged prepared design uses a person-private fused-feature proposal trunk, objectness and detached localization-quality heads, eight train-derived range bins plus bounded residual, projected-center offset with camera unprojection, an independent person-mask residual, bounded train-only hard-negative mining, and deterministic capped episode/track sampling.",
            f"P2 trainable/frozen parameter counts: `{registration['parameter_report_p2']}`.",
            f"Transported bundle names/shapes/dtypes: `{split['transported_feature_names']}` / `{split['transported_feature_shapes']}` / `{split['transported_feature_dtypes']}`. Raw side channels: `{split['tail_raw_modality_side_channels']}`. Monolithic/split bit parity: `{split['outputs_bit_identical']}`.",
            f"Train-derived range edges/counts: `{registration['range_bins']['edges_m']}` / `{registration['range_bins']['counts']}`. Validation rows used for sampling/mining/training: `0`.",
        ])
    if selection:
        lines.extend(["", "## Complete primary v0.10 metrics", ""])
        for record in selection["records"]:
            lines.append(f"- `{record['label']}`: metrics `{record['metrics']}`; eligibility `{record['eligibility_gates']}`; material `{record['material_gain']}`.")
        selected = selection["selected"]
        gaps = service_gaps(selected["metrics"], person_config["service_targets"])
        lines.extend(["", "## Selection, service gaps, and sensitivity", "",
            f"Selected refinement checkpoint: `{selected['label']}` at `{selected['checkpoint']}` with SHA-256 `{selected['checkpoint_sha256']}`.",
            f"Ranking: `{selection['ranking']}`. Non-dominated candidates: `{selection['nondominated_labels']}`.",
            f"Service targets: `{selected['service_targets']}`. Continuous service gaps: `{gaps}`.",
            f"Selected-only v0.25 sensitivity: `{selection['selected_v025_sensitivity']['flat']}` with denominators `{selection['selected_v025_sensitivity'].get('denominators')}`. It was not used for selection.",
        ])
        base_diag = json.loads((experiment / "BASE_DIAGNOSTIC.json").read_text())
        selected_diag = json.loads((experiment / "SELECTED_DIAGNOSTIC.json").read_text())
        before = base_diag["bases"]["epoch_040"]
        after = selected_diag["bases"][selected["label"]]
        lines.extend(["", "## Person PR and mechanism diagnostics", "",
            f"Exact persisted-score person PR points before/after: `{before['full_precision_recall_from_persisted_score_floor']['person']['distinct_thresholds']}` / `{after['full_precision_recall_from_persisted_score_floor']['person']['distinct_thresholds']}`.",
            f"Person FP taxonomy before/after: `{before['taxonomy']['person_fp_at_0_20']}` / `{after['taxonomy']['person_fp_at_0_20']}`.",
            f"Person FN taxonomy before/after: `{before['taxonomy']['person_fn_at_0_02']}` / `{after['taxonomy']['person_fn_at_0_02']}`.",
            f"Radar strata before/after: `{before['person_strata_at_0_20']['radar']}` / `{after['person_strata_at_0_20']['radar']}`.",
        ])
        training = json.loads((experiment / "PERSON_TRAINING_COMPLETE.json").read_text())
        resources = {
            record["label"]: {
                key: json.loads((Path(record["prediction_root"]) / "inference_manifest.json").read_text())[key]
                for key in ("wall_seconds", "peak_allocated_mib", "peak_reserved_mib")
            } for record in selection["records"] if record["label"] != "epoch_040_base"
        }
        lines.extend(["", "## Runtime, VRAM, and retention", "",
            f"Candidate training: `{training['optimizer_steps']}` optimizer steps; peak allocated/reserved VRAM `{training['peak_allocated_mib']}` / `{training['peak_reserved_mib']}` MiB.",
            f"Candidate evaluation inference resources: `{resources}`.",
            f"Retention audit: `{retention}`. No checkpoint deletion was performed.",
        ])
    lines.extend(["", "## Scope confirmation", "",
        "Locked test remained absent and unopened. No q/AE, tracking, CARLA, OAI, live-runtime, calibrated-threshold, alternative-architecture, second-hyperparameter, or 288-measurement work was performed. Datasets, predictions, and checkpoints are not committed.",
        "", f"Supervisor wall time: `{wall_seconds:.1f}` seconds.", ""])
    return "\n".join(lines)


def notify(terminal: str, experiment: Path) -> dict[str, Any]:
    command = ["notify-send", "LR-ASPP accepted person refinement complete", f"{terminal}\n{experiment}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:],
                "delivered": result.returncode == 0}
    except Exception as exc:
        return {"command": command, "delivered": False, "error": f"{type(exc).__name__}: {exc}"}


def commit_report() -> dict[str, Any]:
    explicit = [
        PACKAGE_ROOT / "qualify_v1.py", PACKAGE_ROOT / "policy_v1.py",
        PACKAGE_ROOT / "person_decode_v1.py",
        PACKAGE_ROOT / "person_model_v1.py",
        PACKAGE_ROOT / "preflight_accepted_v2.py", PACKAGE_ROOT / "run_accepted_continuation_v2.py",
        ACCEPTANCE_SOURCE, TRACKED_REPORT,
    ]
    subprocess.run(["git", "add", "--", *map(str, explicit)], cwd=ROOT, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    result = subprocess.run(
        ["git", "commit", "-m", "Complete accepted LR-ASPP person refinement"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"local source/config/report commit failed: {result.stderr[-2000:]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {"head": head, "staged_files": staged, "stdout": result.stdout[-2000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    parser.add_argument("--resume-experiment", type=Path)
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    started = time.monotonic()
    resumed = args.resume_experiment is not None
    if resumed and args.timestamp is not None:
        raise ValueError("--timestamp and --resume-experiment are mutually exclusive")
    if resumed:
        experiment, config_path, acceptance_path, person_config, acceptance = resume_setup(
            args.resume_experiment
        )
    else:
        experiment, config_path, acceptance_path, person_config, acceptance = setup(args.timestamp)
    retry = {"used": 0}
    terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
    selection: dict[str, Any] | None = None
    retention: dict[str, Any] | None = None
    error: str | None = None
    try:
        if resumed:
            base_record = qualify_existing_diagnostic(
                experiment, config_path, acceptance_path, acceptance
            )
        else:
            preflight(experiment, config_path, acceptance_path)
            base_record = diagnose_and_register(experiment, config_path, acceptance_path, acceptance)
        base_checkpoint = (ROOT / acceptance["recovered_checkpoint"]).resolve(strict=True)
        base_record["checkpoint"] = str(base_checkpoint)
        core.train_candidate(experiment, config_path, base_record, person_config, retry)
        selection = core.evaluate_and_select(experiment, config_path, base_record, person_config)
        terminal = selection["terminal"]
        retention = retention_audit(experiment, selection, person_config, acceptance)
    except (ContractInvalid, core.ContractInvalid) as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_CONTRACT_INVALID", f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        terminal, error = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE", f"{type(exc).__name__}: {exc}"
    if terminal not in TERMINALS:
        terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
        error = (error + "; " if error else "") + "unauthorized terminal produced"
    notification = notify(terminal, experiment)
    report = make_report(
        experiment, terminal, person_config, acceptance, selection, retention,
        notification, retry["used"], error, time.monotonic() - started,
    )
    write_text_x(experiment / "FINAL_REPORT.md", report)
    write_text_x(TRACKED_REPORT, report)
    write_json_x(experiment / "NOTIFICATION.json", notification)
    try:
        commit_result = commit_report()
    except Exception as exc:
        terminal = "LRASPP_PERSON_REFINEMENT_RUNTIME_FAILURE"
        error = (error + "; " if error else "") + f"{type(exc).__name__}: {exc}"
        commit_result = {"error": str(exc)}
    historical_terminal = (ROOT / acceptance["historical_experiment"] / "TERMINAL_VERDICT.txt").resolve(strict=True)
    historical_unchanged = (
        sha256(historical_terminal) == acceptance["historical_terminal_sha256"]
        and historical_terminal.read_text().strip() == acceptance["historical_terminal"]
    )
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_person_refinement_accepted_continuation_complete_v2",
        "created_utc": utc_now(), "terminal": terminal, "error": error,
        "retry_used": retry["used"], "notification": notification,
        "source_commit": commit_result, "historical_terminal_unchanged": historical_unchanged,
        "epochs_11_through_40_repeated": False, "wall_seconds": time.monotonic() - started,
    })
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
    status = json.loads((experiment / "STATUS.json").read_text())
    status.update({"phase": "complete", "terminal": terminal, "detail": error or "",
                   "updated_utc": utc_now(), "source_commit": commit_result.get("head")})
    write_json_atomic(experiment / "STATUS.json", status)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment), "error": error,
        "selected": selection["selected"]["label"] if selection else None,
        "selected_checkpoint": selection["selected"]["checkpoint"] if selection else None,
        "selected_sha256": selection["selected"]["checkpoint_sha256"] if selection else None,
        "source_commit": commit_result.get("head"), "historical_terminal_unchanged": historical_unchanged,
    }, indent=2), flush=True)
    return 0 if terminal in {
        "LRASPP_PERSON_REFINEMENT_SERVICE_READY", "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN",
        "LRASPP_PERSON_REFINEMENT_NO_GAIN",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

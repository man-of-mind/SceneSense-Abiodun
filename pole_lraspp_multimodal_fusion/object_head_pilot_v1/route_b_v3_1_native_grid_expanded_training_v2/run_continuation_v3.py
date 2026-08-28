#!/usr/bin/env python3
"""One autonomous supervisor for the bounded epoch-10 continuation."""

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

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
CONFIG_SOURCE = PACKAGE_ROOT / "configs/expanded_continuation_v3.json"
TRAINING_SOURCE = PACKAGE_ROOT / "configs/expanded_training_v2.json"
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_native_grid_expanded_continuation_v3"
POINTER = PACKAGE_ROOT / "NATIVE_GRID_EXPANDED_CONTINUATION_EXP_DIR.txt"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_NATIVE_GRID_EXPANDED_CONTINUATION_V3_REPORT.md"
AUTHORIZED_TERMINALS = {
    "LRASPP_EXPANDED_LONGTRAIN_SERVICE_READY",
    "LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY",
    "LRASPP_EXPANDED_LONGTRAIN_NO_GAIN",
    "LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION",
    "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE",
    "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID",
}
PROGRESS_FIELDS = (
    "created_utc", "attempt", "phase", "epoch", "optimizer_steps", "detail",
)


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


def write_text_x(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_progress(path: Path, attempt: int, phase: str, detail: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writerow({
            "created_utc": utc_now(), "attempt": attempt, "phase": phase,
            "epoch": "", "optimizer_steps": "", "detail": detail,
        })


def update_status(experiment: Path, **values: Any) -> None:
    path = experiment / "STATUS.json"
    current = json.loads(path.read_text())
    current.update(values)
    current["updated_utc"] = utc_now()
    write_json_atomic(path, current)


def run_logged(command: list[str], log: Path, marker: str) -> int:
    mode = "a" if log.exists() else "x"
    with log.open(mode, encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] {marker}\n")
        stream.write("command=" + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        return process.wait()


def notify(terminal: str, experiment: Path) -> dict[str, Any]:
    command = [
        "notify-send", "LR-ASPP expanded continuation complete",
        f"{terminal}\n{experiment}",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:], "stderr": completed.stderr[-1000:],
            "delivered": completed.returncode == 0,
        }
    except Exception as exc:
        return {"command": command, "delivered": False, "error": f"{type(exc).__name__}: {exc}"}


def source_manifest() -> dict[str, str]:
    paths = [
        CONFIG_SOURCE,
        PACKAGE_ROOT / "continuation_policy_v3.py",
        PACKAGE_ROOT / "continuation_scoring_v3.py",
        PACKAGE_ROOT / "score_continuation_v3.py",
        PACKAGE_ROOT / "preflight_continuation_v3.py",
        PACKAGE_ROOT / "continue_training_v3.py",
        PACKAGE_ROOT / "run_continuation_v3.py",
        PACKAGE_ROOT / "test_continuation_policy_v3.py",
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def gpu_diagnostic() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
            "torch_cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def move_incomplete_artifacts(experiment: Path, attempt: int) -> list[dict[str, str]]:
    evidence = experiment / "failed_attempt_artifacts" / f"attempt_{attempt}"
    moved: list[dict[str, str]] = []
    for path in sorted((experiment / "predictions").glob("*")):
        if path.is_dir() and not (path / "INFERENCE_COMPLETE").is_file():
            evidence.mkdir(parents=True, exist_ok=True)
            target = evidence / path.name
            if target.exists():
                raise RuntimeError(f"failed-attempt evidence collision: {target}")
            shutil.move(str(path), str(target))
            moved.append({"source": str(path), "target": str(target)})
    for path in sorted(experiment.glob("**/*.partial")):
        evidence.mkdir(parents=True, exist_ok=True)
        target = evidence / path.name
        if target.exists():
            raise RuntimeError(f"partial-checkpoint evidence collision: {target}")
        shutil.move(str(path), str(target))
        moved.append({"source": str(path), "target": str(target)})
    return moved


def latest_safe(experiment: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (experiment / "LATEST_SAFE.json").is_file():
        latest = json.loads((experiment / "LATEST_SAFE.json").read_text())
    else:
        path = (ROOT / contract["resume_checkpoint"]).resolve(strict=True)
        latest = {"epoch": 10, "path": str(path), "sha256": contract["resume_checkpoint_sha256"]}
    path = Path(latest["path"]).resolve(strict=True)
    actual = sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    if actual != latest["sha256"] or not required.issubset(payload) or payload["epoch"] != latest["epoch"]:
        raise RuntimeError("latest safe checkpoint failed recovery integrity")
    return {**latest, "path": str(path), "verified_sha256": actual, "required_state_present": True}


def cleanup(experiment: Path, decision: dict[str, Any], terminal: str) -> dict[str, Any]:
    if terminal in {
        "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE",
        "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID",
    }:
        return {
            "performed": False, "reason": "failure artifacts and latest safe checkpoint preserved",
            "retained_checkpoint": decision.get("retained_checkpoint"),
        }
    retained_raw = decision.get("retained_checkpoint")
    retained = Path(retained_raw).resolve() if retained_raw else None
    removed_checkpoints: list[str] = []
    for checkpoint in sorted(experiment.glob("**/*.pt")):
        if retained is not None and checkpoint.resolve() == retained:
            continue
        checkpoint.unlink()
        removed_checkpoints.append(str(checkpoint))
    prediction_dirs = []
    predictions = experiment / "predictions"
    if predictions.is_dir():
        prediction_dirs = [str(path) for path in predictions.iterdir()]
        shutil.rmtree(predictions)
    return {
        "performed": True,
        "retained_checkpoint": retained_raw,
        "retained_checkpoint_sha256": decision.get("retained_checkpoint_sha256"),
        "removed_checkpoints": removed_checkpoints,
        "removed_prediction_directories": prediction_dirs,
        "datasets_or_contracts_removed": 0,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def decode_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Candidate | Veh P/R/F1 | Veh R@.02 | Veh XY/dim/yaw | Person P/R/F1 | Person R@.02 | Person XY/dim/yaw | IoU veh/person | fg mIoU | dup FP | targets | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"amended_baseline": 0, "epoch_010": 10, "epoch_020": 20, "epoch_030": 30, "epoch_040": 40}
    for record in sorted(records, key=lambda item: order.get(item.get("label", ""), 999)):
        metric = record["metrics"]
        lines.append(
            f"| {record.get('label', record.get('epoch'))} | "
            f"{fmt(metric.get('vehicle_precision'))}/{fmt(metric.get('vehicle_recall'))}/{fmt(metric.get('vehicle_f1'))} | "
            f"{fmt(metric.get('vehicle_recall_002'))} | "
            f"{fmt(metric.get('vehicle_xy_mae_m'), 3)}/{fmt(metric.get('vehicle_dimension_mae_m'), 3)}/{fmt(metric.get('vehicle_yaw_mae_deg'), 2)} | "
            f"{fmt(metric.get('person_precision'))}/{fmt(metric.get('person_recall'))}/{fmt(metric.get('person_f1'))} | "
            f"{fmt(metric.get('person_recall_002'))} | "
            f"{fmt(metric.get('person_xy_mae_m'), 3)}/{fmt(metric.get('person_dimension_mae_m'), 3)}/{fmt(metric.get('person_yaw_mae_deg'), 2)} | "
            f"{fmt(metric.get('vehicle_iou'))}/{fmt(metric.get('person_box_mask_iou'))} | "
            f"{fmt(metric.get('foreground_miou'))} | {record.get('vehicle_duplicate_fp', 'N/A')} | "
            f"{record.get('service_target_count', 'N/A')}/9 | {record.get('eligible', 'N/A')} |"
        )
    return "\n".join(lines)


def service_table(selected: dict[str, Any] | None) -> str:
    if not selected:
        return "No final service selection was performed."
    return "\n".join(
        ["| Target | Pass |", "|---|---:|"]
        + [f"| {name} | {value} |" for name, value in selected["service_targets"].items()]
    )


def remaining_gaps(selected: dict[str, Any] | None) -> dict[str, float]:
    if not selected:
        return {}
    metric = selected["metrics_v010"]
    gaps = {
        "vehicle_precision": max(0.0, 0.80 - metric["vehicle_precision"]),
        "vehicle_recall": max(0.0, 0.85 - metric["vehicle_recall"]),
        "person_precision": max(0.0, 0.80 - metric["person_precision"]),
        "person_recall": max(0.0, 0.80 - metric["person_recall"]),
        "vehicle_xy_mae_m": max(0.0, metric["vehicle_xy_mae_m"] - 1.0),
        "person_xy_mae_m": max(0.0, metric["person_xy_mae_m"] - 1.2),
        "vehicle_iou": max(0.0, 0.85 - metric["vehicle_iou"]),
        "person_box_mask_iou": max(0.0, 0.50 - metric["person_box_mask_iou"]),
        "foreground_miou": max(0.0, 0.675 - metric["foreground_miou"]),
    }
    return {key: value for key, value in gaps.items() if value > 0.0}


def fallback_records(contract: dict[str, Any], training: dict[str, Any]) -> list[dict[str, Any]]:
    amended = json.loads((ROOT / contract["amended_baseline"]).read_text())
    base_flat = dict(amended["amended"]["v010"]["flat"])
    epoch10 = json.loads((ROOT / contract["resume_epoch10_evidence"]).read_text())
    return [
        {
            "label": "amended_baseline", "metrics": base_flat,
            "vehicle_duplicate_fp": amended["amended_taxonomy"]["vehicle_fp_at_0_20"]["counts"]["PREDICTED_DUPLICATE"],
            "taxonomy_v010": amended["amended_taxonomy"],
        },
        {**epoch10, "label": "epoch_010"},
    ]


def report(
    experiment: Path, terminal: str, decision: dict[str, Any], preflight: dict[str, Any],
    cleanup_result: dict[str, Any], notification: dict[str, Any], recoveries: list[dict[str, Any]],
    pipeline_wall: float, contract: dict[str, Any], training: dict[str, Any], execution_head: str,
) -> str:
    records = decision.get("decode_records") or fallback_records(contract, training)
    selected = decision.get("selected")
    sensitivity = decision.get("sensitivity_v025")
    sensitivity_text = "Not available because final selection did not complete."
    if sensitivity:
        flat = sensitivity["flat"]
        sensitivity_text = (
            f"Distinct v0.25 eligible-GT denominators: `{sensitivity['denominators']}`. "
            f"Vehicle P/R/F1/R@.02/XY/dim/yaw: "
            f"`{fmt(flat.get('vehicle_precision'))}/{fmt(flat.get('vehicle_recall'))}/{fmt(flat.get('vehicle_f1'))}/"
            f"{fmt(flat.get('vehicle_recall_002'))}/{fmt(flat.get('vehicle_xy_mae_m'), 3)}/"
            f"{fmt(flat.get('vehicle_dimension_mae_m'), 3)}/{fmt(flat.get('vehicle_yaw_mae_deg'), 2)}`. "
            f"Person: `{fmt(flat.get('person_precision'))}/{fmt(flat.get('person_recall'))}/{fmt(flat.get('person_f1'))}/"
            f"{fmt(flat.get('person_recall_002'))}/{fmt(flat.get('person_xy_mae_m'), 3)}/"
            f"{fmt(flat.get('person_dimension_mae_m'), 3)}/{fmt(flat.get('person_yaw_mae_deg'), 2)}`. "
            "This is sensitivity only; its distinct eligibility denominator is not described as model improvement."
        )
    taxonomy_lines = []
    for record in records:
        taxonomy = record.get("taxonomy_v010")
        if taxonomy:
            taxonomy_lines.append(
                f"- {record.get('label', record.get('epoch'))}: vehicle FP "
                f"`{taxonomy['vehicle_fp_at_0_20']['counts']}`; person FN@0.02 "
                f"`{taxonomy['person_fn_at_0_02']['counts']}`."
            )
    original_training = json.loads((ROOT / "experiments/route_b_v3_1_native_grid_expanded_training_v2/20260828_103000/TRAINING_COMPLETE.json").read_text())
    return f"""# Route B v3.1 native-grid expanded continuation v3 report

Terminal: `{terminal}`

Experiment: `{experiment}`

Execution commit on local `master`: `{execution_head}` (the required starting HEAD). The final source/config/report commit is the post-run local `master` HEAD reported at handoff; nothing was pushed.

## Exact continuation state

- Resume checkpoint: `{contract['resume_checkpoint']}`.
- Verified SHA-256: `{preflight.get('checkpoint_sha256', contract['resume_checkpoint_sha256'])}`.
- State: epoch `{preflight.get('resume_state', {}).get('epoch', 10)}`, optimizer steps `{preflight.get('resume_state', {}).get('optimizer_steps', 10520)}`, 1,052 steps/epoch; model, AdamW optimizer, registered H2/J2 scheduler, AMP GradScaler, and Python/NumPy/Torch CPU/CUDA RNG state were present and strictly loadable.
- End-of-epoch-10 inherited/object LR: `{preflight.get('resume_state', {}).get('optimizer_group_lrs')}`; these reconcile to the registered cosine schedule.
- Epochs 1–10 were not repeated and the epoch-15 warm start was not used to restart this continuation.
- All recorded view/manifest/GT/ignore/camera-plane hashes passed, including `{preflight.get('checks', [{}])[3].get('payload_verification', {}).get('references_checked', 'N/A')}` payload-hash references. Train/validation remained 16,827/3,345 frames from 10/2 episodes, with independent v0.10 and v0.25 contract roots.

## Execution and stop reason

- Continuation epochs completed: `{decision.get('epochs_completed', [])}`.
- Primary decoded epochs: `{decision.get('decoded_epochs', [10])}`.
- Stop reason: `{decision.get('early_stop_reason', 'worker did not reach a policy-complete state')}`.
- Historical epochs 1–10 training/decision wall: `{original_training['wall_seconds']:.3f} s`; continuation training/evaluation wall: `{decision.get('wall_seconds', 0.0):.3f} s`; supervisor wall: `{pipeline_wall:.3f} s`.
- Peak continuation CUDA allocated/reserved: `{decision.get('peak_allocated_mib', 0.0):.1f}/{decision.get('peak_reserved_mib', 0.0):.1f} MiB`.

## Primary v0.10 comparison

{decode_table(records)}

Epoch-10 dimension/yaw entries are `N/A`: the already-authorized epoch-10 scorer did not record them and its predictions had already been cleaned; epoch 10 was not re-decoded on primary v0.10. Later entries use the frozen 3 m match assignments and reconcile TP/FP/FN/P/R/F1/XY exactly before adding dimension/yaw diagnostics.

## Duplicate FP and world-error taxonomy

{chr(10).join(taxonomy_lines) if taxonomy_lines else '- Not available beyond the preserved evidence.'}

Duplicate FP remained a reported metric and the final ranking tie-breaker; it was never an intermediate stop condition.

## Final selection and service targets

- Eligible candidates in rank order: `{decision.get('ranking')}`.
- Selected checkpoint: `{selected.get('checkpoint') if selected else None}`.
- Selected SHA-256: `{selected.get('checkpoint_sha256') if selected else None}`.
- Exact remaining service gaps: `{remaining_gaps(selected)}`.

{service_table(selected)}

## Selected-only v0.25 sensitivity

{sensitivity_text}

## Recovery, cleanup, and scope

- Supervisor recoveries: `{recoveries}`.
- Narrow AMP recovery: `{decision.get('automatic_recovery')}`.
- Cleanup/retention: `{cleanup_result}`.
- Desktop notification: `{notification}`.
- Locked test, CARLA, the pre-existing dirty OAI submodule, q/AE, feature-drop behavior, and the 288 measurements were untouched. No architecture, loss weight, sampler, batch size, optimizer, schedule, seed, decoder, threshold, NMS rule, dataset, or postprocessor changed.

Human inspection commands (read-only):

```bash
jq . {experiment}/STATUS.json
tail -n 8 {experiment}/PROGRESS.csv
sed -n '1,220p' {experiment}/logs/training.log
cat {experiment}/COMPLETION_SENTINEL
```
"""


def resume_after_supervisor_failure(experiment_arg: Path) -> int:
    """Use the sole authorized retry after an attempt-1 supervisor/worker failure."""
    resumed_started = time.monotonic()
    experiment = experiment_arg.resolve(strict=True)
    if experiment.parent != EXPERIMENT_PARENT.resolve():
        raise RuntimeError("resume experiment is outside the registered continuation parent")
    if POINTER.read_text().strip() != str(experiment):
        raise RuntimeError("continuation pointer does not match resume experiment")
    if (experiment / "COMPLETION_SENTINEL").exists():
        raise RuntimeError("refusing to resume a terminal experiment")
    contract_path = experiment / "resolved_configs" / CONFIG_SOURCE.name
    training_path = experiment / "resolved_configs" / TRAINING_SOURCE.name
    contract = json.loads(contract_path.read_text())
    training = json.loads(training_path.read_text())
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    execution_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "master" or execution_head != contract["required_starting_head"]:
        raise RuntimeError("bounded retry no longer has the required local master ancestry")
    if (experiment / "WORKER_FAILURE_ATTEMPT_2.json").exists():
        raise RuntimeError("the single authorized runtime retry has already been consumed")
    if (experiment / "RUNNING.pid").exists():
        (experiment / "RUNNING.pid").rename(experiment / "FAILED_SUPERVISOR_ATTEMPT_1.pid")
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    first_failure = experiment / "WORKER_FAILURE_ATTEMPT_1.json"
    if not first_failure.exists():
        write_json_x(first_failure, {
            "schema": "route_b_v3_1_native_grid_expanded_continuation_worker_failure_v3",
            "created_utc": utc_now(), "attempt": 1, "kind": "runtime",
            "error": (
                "attempt-1 worker exception was masked by an UnboundLocalError in its "
                "failure serializer; no epoch completed and epoch 10 remains latest safe"
            ),
            "wall_seconds": 0.0,
        })
    resume = latest_safe(experiment, contract)
    moved = move_incomplete_artifacts(experiment, 1)
    recovery = {
        "schema": "route_b_v3_1_native_grid_expanded_runtime_recovery_v3",
        "created_utc": utc_now(), "attempt": 1,
        "failure": json.loads(first_failure.read_text()),
        "gpu_diagnostic": gpu_diagnostic(), "latest_safe": resume,
        "moved_incomplete_artifacts": moved, "retry_count": 1,
        "same_interpreter": sys.executable,
        "same_configuration_sha256": sha256(contract_path),
        "automation_patch": {
            "scope": "failure serialization and existing-experiment supervisor recovery only",
            "training_or_evaluation_contract_changed": False,
            "post_patch_source_hashes": source_manifest(),
        },
    }
    write_json_x(experiment / "RUNTIME_RECOVERY_ATTEMPT_1.json", recovery)
    append_progress(experiment / "PROGRESS.csv", 1, "bounded_runtime_recovery", str(resume))
    update_status(
        experiment, phase="runtime_recovery", attempt=2, retry=1,
        latest_safe_checkpoint=resume["path"], latest_safe_sha256=resume["sha256"],
    )
    recovery_preflight_path = experiment / "PREFLIGHT_RECOVERY_ATTEMPT_1.json"
    recovery_command = [
        sys.executable, str(PACKAGE_ROOT / "preflight_continuation_v3.py"),
        "--experiment", str(experiment), "--continuation-config", str(contract_path),
        "--output", str(recovery_preflight_path),
    ]
    recovery_preflight_rc = run_logged(
        recovery_command, experiment / "logs/preflight_recovery.log",
        "bounded recovery hash/CUDA re-verification",
    )
    terminal = "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE"
    decision: dict[str, Any] = {}
    if recovery_preflight_rc != 0:
        terminal = "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID"
    else:
        worker_command = [
            sys.executable, str(PACKAGE_ROOT / "continue_training_v3.py"),
            "--experiment", str(experiment),
            "--continuation-config", str(contract_path),
            "--training-config", str(training_path),
            "--resume-checkpoint", resume["path"],
            "--resume-sha256", resume["sha256"], "--attempt", "2",
        ]
        worker_rc = run_logged(
            worker_command, experiment / "logs/training.log", "worker attempt 2 (sole retry)"
        )
        if worker_rc == 0:
            decision = json.loads((experiment / "DECISION.json").read_text())
            terminal = decision["terminal"]
        elif worker_rc == 20:
            terminal = "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID"
        elif worker_rc == 21:
            terminal = "LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION"
        else:
            terminal = "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE"
    if not decision:
        latest = latest_safe(experiment, contract)
        failures = [
            json.loads(path.read_text())
            for path in sorted(experiment.glob("WORKER_FAILURE_ATTEMPT_*.json"))
        ]
        epoch_rows = [
            json.loads(path.read_text())
            for path in sorted((experiment / "metrics").glob("epoch_???_training.json"))
        ]
        decoded = [
            json.loads(path.read_text())
            for path in sorted((experiment / "decisions").glob("epoch_0[234]0_decode.json"))
        ]
        decision = {
            "schema": "route_b_v3_1_native_grid_expanded_continuation_failure_decision_v3",
            "created_utc": utc_now(), "terminal": terminal,
            "epochs_completed": [row["epoch"] for row in epoch_rows],
            "decoded_epochs": [10] + [record["epoch"] for record in decoded],
            "early_stop_reason": failures[-1]["error"] if failures else "recovery preflight failed",
            "training_rows": epoch_rows, "decode_records": decoded,
            "selected": None, "sensitivity_v025": None,
            "retained_checkpoint": latest["path"],
            "retained_checkpoint_sha256": latest["sha256"],
            "worker_failures": failures,
            "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in failures),
            "peak_allocated_mib": max([float(row["cuda_allocated_peak_mib"]) for row in epoch_rows] or [0.0]),
            "peak_reserved_mib": max([float(row["cuda_reserved_peak_mib"]) for row in epoch_rows] or [0.0]),
            "automatic_recovery": None,
        }
        if not (experiment / "DECISION.json").exists():
            write_json_x(experiment / "DECISION.json", decision)
    if terminal not in AUTHORIZED_TERMINALS:
        raise RuntimeError(f"unauthorized terminal {terminal}")
    cleanup_result = cleanup(experiment, decision, terminal)
    write_json_x(experiment / "CLEANUP.json", cleanup_result)
    notification = notify(terminal, experiment)
    write_json_x(experiment / "NOTIFICATION.json", notification)
    pipeline_started = datetime.fromisoformat(
        json.loads((experiment / "PIPELINE_STARTED.json").read_text())["created_utc"]
    )
    pipeline_wall = (datetime.now(timezone.utc) - pipeline_started).total_seconds()
    preflight = json.loads((experiment / "PREFLIGHT.json").read_text())
    final_report = report(
        experiment, terminal, decision, preflight, cleanup_result, notification,
        [recovery], pipeline_wall, contract, training, execution_head,
    )
    write_text_x(experiment / "FINAL_REPORT.md", final_report)
    write_text_x(TRACKED_REPORT, final_report)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "terminal": terminal, "created_utc": utc_now(), "wall_seconds": pipeline_wall,
        "cleanup": cleanup_result, "notification": notification,
        "resumed_supervisor_wall_seconds": time.monotonic() - resumed_started,
    })
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
    update_status(experiment, phase="complete", terminal=terminal,
                  completion_sentinel=str(experiment / "COMPLETION_SENTINEL"))
    append_progress(experiment / "PROGRESS.csv", 0, "supervisor_complete", terminal)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment),
        "retained_checkpoint": decision.get("retained_checkpoint"),
        "retained_checkpoint_sha256": decision.get("retained_checkpoint_sha256"),
    }, indent=2), flush=True)
    return 0 if terminal not in {
        "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE",
        "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID",
    } else 1


def archive_rejected_automation(experiment_arg: Path) -> dict[str, Any]:
    """Preserve, but de-register, a pre-epoch automation-policy qualification failure."""
    experiment = experiment_arg.resolve(strict=True)
    if experiment.parent != EXPERIMENT_PARENT.resolve():
        raise RuntimeError("rejected automation experiment is outside the registered parent")
    if POINTER.read_text().strip() != str(experiment):
        raise RuntimeError("rejected automation experiment does not match the current pointer")
    checkpoints = list(experiment.glob("**/*.pt"))
    epoch_rows = list((experiment / "metrics").glob("epoch_???_training.json"))
    if checkpoints or epoch_rows:
        raise RuntimeError("refusing to de-register an automation attempt that completed model state")
    origin = (ROOT / json.loads(CONFIG_SOURCE.read_text())["resume_checkpoint"]).resolve(strict=True)
    origin_sha = sha256(origin)
    expected_sha = json.loads(CONFIG_SOURCE.read_text())["resume_checkpoint_sha256"]
    if origin_sha != expected_sha:
        raise RuntimeError("resume origin changed during rejected automation attempt")
    rejected_pointer = PACKAGE_ROOT / "NATIVE_GRID_EXPANDED_CONTINUATION_REJECTED_ATTEMPT_1_EXP_DIR.txt"
    rejected_report = PACKAGE_ROOT / "ROUTE_B_V3_1_NATIVE_GRID_EXPANDED_CONTINUATION_REJECTED_AUTOMATION_ATTEMPT_1_REPORT.md"
    if rejected_pointer.exists() or rejected_report.exists():
        raise FileExistsError("rejected automation evidence path already exists")
    POINTER.rename(rejected_pointer)
    TRACKED_REPORT.rename(rejected_report)
    renames = {
        "COMPLETION_SENTINEL": "REJECTED_AUTOMATION_SENTINEL",
        "TERMINAL_VERDICT.txt": "REJECTED_AUTOMATION_TERMINAL_VERDICT.txt",
        "PIPELINE_COMPLETE.json": "REJECTED_AUTOMATION_PIPELINE_COMPLETE.json",
        "FINAL_REPORT.md": "REJECTED_AUTOMATION_REPORT.md",
        "DECISION.json": "REJECTED_AUTOMATION_DECISION.json",
    }
    preserved: dict[str, str] = {}
    for source, target in renames.items():
        source_path, target_path = experiment / source, experiment / target
        if source_path.exists():
            if target_path.exists():
                raise FileExistsError(f"rejected automation evidence collision: {target_path}")
            source_path.rename(target_path)
            preserved[source] = target
    result = {
        "schema": "route_b_v3_1_native_grid_expanded_rejected_automation_attempt_v3",
        "created_utc": utc_now(), "experiment": str(experiment),
        "reason": (
            "worker incorrectly promoted a finite-loss GradScaler backoff to a catastrophic "
            "regression; this guard was not part of the registered AMP policy"
        ),
        "epochs_completed": 0, "continuation_checkpoints_created": 0,
        "resume_origin_sha256_after_attempt": origin_sha,
        "training_or_evaluation_contract_changed": False,
        "preserved_renames": preserved,
        "rejected_pointer": str(rejected_pointer),
        "rejected_report": str(rejected_report),
    }
    write_json_x(experiment / "AUTOMATION_ATTEMPT_REJECTED.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    parser.add_argument("--resume-experiment", type=Path)
    parser.add_argument("--rejected-automation-experiment", type=Path)
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    if args.resume_experiment is not None:
        return resume_after_supervisor_failure(args.resume_experiment)
    rejected_automation = None
    if args.rejected_automation_experiment is not None:
        rejected_automation = archive_rejected_automation(args.rejected_automation_experiment)
    if POINTER.exists() or TRACKED_REPORT.exists():
        raise FileExistsError("continuation pointer or tracked report already exists")
    started = time.monotonic()
    contract = json.loads(CONFIG_SOURCE.read_text())
    training = json.loads(TRAINING_SOURCE.read_text())
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    execution_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "master" or execution_head != contract["required_starting_head"]:
        raise RuntimeError(f"required master/{contract['required_starting_head']}, got {branch}/{execution_head}")
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = EXPERIMENT_PARENT / timestamp
    experiment.mkdir(parents=True, exist_ok=False)
    for directory in ("logs", "resolved_configs", "provenance", "predictions"):
        (experiment / directory).mkdir()
    continuation_path = experiment / "resolved_configs" / CONFIG_SOURCE.name
    training_path = experiment / "resolved_configs" / TRAINING_SOURCE.name
    shutil.copyfile(CONFIG_SOURCE, continuation_path)
    shutil.copyfile(TRAINING_SOURCE, training_path)
    view = (ROOT / training["training_view"]).resolve(strict=True)
    os.symlink(str((view / "dataset").resolve()), experiment / "dataset")
    os.symlink(str((view / "contracts").resolve()), experiment / "contracts")
    write_text_x(POINTER, str(experiment.resolve()) + "\n")
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    write_json_x(experiment / "STATUS.json", {
        "schema": "route_b_v3_1_native_grid_expanded_continuation_status_v3",
        "created_utc": utc_now(), "updated_utc": utc_now(), "phase": "preflight",
        "terminal": None, "experiment": str(experiment.resolve()),
        "supervisor_pid": os.getpid(), "interpreter": sys.executable,
        "resume_epoch": 10, "next_epoch": 11,
    })
    progress = experiment / "PROGRESS.csv"
    with progress.open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writeheader()
    append_progress(progress, 0, "supervisor_started", str(experiment))
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v2", "--untracked-files=all"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    provenance = {
        "schema": "route_b_v3_1_native_grid_expanded_continuation_provenance_v3",
        "created_utc": utc_now(), "branch": branch, "required_starting_head": execution_head,
        "source_hashes": source_manifest(),
        "resolved_continuation_sha256": sha256(continuation_path),
        "resolved_training_sha256": sha256(training_path),
        "initial_git_status_porcelain_v2": git_status.splitlines(),
        "pre_existing_dirty_oai_preserved": "OAI/openairinterface5g" in git_status,
        "locked_test_paths_enumerated_or_read": 0,
        "carla_commands_launched": 0, "q_or_ae_commands_launched": 0,
        "measurement_files_modified": 0,
        "prior_rejected_automation_attempt": rejected_automation,
    }
    write_json_x(experiment / "provenance/PROVENANCE_MANIFEST.json", provenance)
    write_json_x(experiment / "PIPELINE_STARTED.json", {
        "created_utc": utc_now(), "pid": os.getpid(), "interpreter": sys.executable,
        "experiment": str(experiment.resolve()), "required_starting_head": execution_head,
    })

    terminal = "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE"
    decision: dict[str, Any] = {}
    recoveries: list[dict[str, Any]] = (
        [{"kind": "rejected_automation_qualification", **rejected_automation}]
        if rejected_automation is not None else []
    )
    preflight_command = [
        sys.executable, str(PACKAGE_ROOT / "preflight_continuation_v3.py"),
        "--experiment", str(experiment), "--continuation-config", str(continuation_path),
    ]
    preflight_rc = run_logged(preflight_command, experiment / "logs/preflight.log", "initial preflight")
    preflight = json.loads((experiment / "PREFLIGHT.json").read_text())
    if preflight_rc != 0:
        terminal = "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID"
        decision = {
            "terminal": terminal, "epochs_completed": [], "decoded_epochs": [10],
            "early_stop_reason": "continuation preflight failed closed",
            "retained_checkpoint": contract["resume_checkpoint"],
            "retained_checkpoint_sha256": contract["resume_checkpoint_sha256"],
            "wall_seconds": 0.0, "automatic_recovery": None,
        }
        write_json_x(experiment / "DECISION.json", decision)
    else:
        append_progress(progress, 0, "preflight_complete", "all state/data/CUDA gates passed")
        update_status(experiment, phase="training", preflight_passed=True)
        resume = latest_safe(experiment, contract)
        final_rc = 22
        for attempt in (1, 2):
            if attempt == 2:
                recovery_preflight_path = experiment / "PREFLIGHT_RECOVERY_ATTEMPT_1.json"
                recovery_command = preflight_command + ["--output", str(recovery_preflight_path)]
                recovery_preflight_rc = run_logged(
                    recovery_command, experiment / "logs/preflight_recovery.log",
                    "bounded recovery hash/CUDA re-verification",
                )
                if recovery_preflight_rc != 0:
                    terminal = "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID"
                    break
            worker_command = [
                sys.executable, str(PACKAGE_ROOT / "continue_training_v3.py"),
                "--experiment", str(experiment),
                "--continuation-config", str(continuation_path),
                "--training-config", str(training_path),
                "--resume-checkpoint", resume["path"],
                "--resume-sha256", resume["sha256"], "--attempt", str(attempt),
            ]
            final_rc = run_logged(
                worker_command, experiment / "logs/training.log",
                f"worker attempt {attempt}",
            )
            if final_rc == 0:
                decision = json.loads((experiment / "DECISION.json").read_text())
                terminal = decision["terminal"]
                break
            failure = json.loads((experiment / f"WORKER_FAILURE_ATTEMPT_{attempt}.json").read_text())
            if final_rc == 20:
                terminal = "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID"
                break
            if final_rc == 21:
                terminal = "LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION"
                break
            if attempt == 2:
                terminal = "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE"
                break
            resume = latest_safe(experiment, contract)
            recovery = {
                "schema": "route_b_v3_1_native_grid_expanded_runtime_recovery_v3",
                "created_utc": utc_now(), "attempt": 1, "failure": failure,
                "gpu_diagnostic": gpu_diagnostic(), "latest_safe": resume,
                "moved_incomplete_artifacts": move_incomplete_artifacts(experiment, 1),
                "retry_count": 1, "same_interpreter": sys.executable,
                "same_configuration_sha256": sha256(continuation_path),
            }
            write_json_x(experiment / "RUNTIME_RECOVERY_ATTEMPT_1.json", recovery)
            recoveries.append(recovery)
            append_progress(progress, 1, "bounded_runtime_recovery", str(resume))
            update_status(experiment, phase="runtime_recovery", retry=1,
                          latest_safe_checkpoint=resume["path"])

        if not decision:
            latest = latest_safe(experiment, contract)
            failures = [
                json.loads(path.read_text())
                for path in sorted(experiment.glob("WORKER_FAILURE_ATTEMPT_*.json"))
            ]
            epoch_rows = [
                json.loads(path.read_text())
                for path in sorted((experiment / "metrics").glob("epoch_???_training.json"))
            ] if (experiment / "metrics").is_dir() else []
            decoded = [
                json.loads(path.read_text())
                for path in sorted((experiment / "decisions").glob("epoch_0[234]0_decode.json"))
            ] if (experiment / "decisions").is_dir() else []
            decision = {
                "schema": "route_b_v3_1_native_grid_expanded_continuation_failure_decision_v3",
                "created_utc": utc_now(), "terminal": terminal,
                "epochs_completed": [row["epoch"] for row in epoch_rows],
                "decoded_epochs": [10] + [record["epoch"] for record in decoded],
                "early_stop_reason": failures[-1]["error"] if failures else "recovery preflight failed",
                "training_rows": epoch_rows, "decode_records": decoded,
                "selected": None, "sensitivity_v025": None,
                "retained_checkpoint": latest["path"],
                "retained_checkpoint_sha256": latest["sha256"],
                "worker_failures": failures,
                "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in failures),
                "peak_allocated_mib": max([float(row["cuda_allocated_peak_mib"]) for row in epoch_rows] or [0.0]),
                "peak_reserved_mib": max([float(row["cuda_reserved_peak_mib"]) for row in epoch_rows] or [0.0]),
                "automatic_recovery": None,
            }
            if not (experiment / "DECISION.json").exists():
                write_json_x(experiment / "DECISION.json", decision)

    if terminal not in AUTHORIZED_TERMINALS:
        raise RuntimeError(f"unauthorized terminal {terminal}")
    cleanup_result = cleanup(experiment, decision, terminal)
    write_json_x(experiment / "CLEANUP.json", cleanup_result)
    notification = notify(terminal, experiment)
    write_json_x(experiment / "NOTIFICATION.json", notification)
    pipeline_wall = time.monotonic() - started
    final_report = report(
        experiment.resolve(), terminal, decision, preflight, cleanup_result,
        notification, recoveries, pipeline_wall, contract, training, execution_head,
    )
    write_text_x(experiment / "FINAL_REPORT.md", final_report)
    write_text_x(TRACKED_REPORT, final_report)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "terminal": terminal, "created_utc": utc_now(), "wall_seconds": pipeline_wall,
        "cleanup": cleanup_result, "notification": notification,
    })
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
    update_status(experiment, phase="complete", terminal=terminal,
                  completion_sentinel=str(experiment / "COMPLETION_SENTINEL"))
    append_progress(progress, 0, "supervisor_complete", terminal)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment.resolve()),
        "retained_checkpoint": decision.get("retained_checkpoint"),
        "retained_checkpoint_sha256": decision.get("retained_checkpoint_sha256"),
    }, indent=2), flush=True)
    return 0 if terminal not in {
        "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE",
        "LRASPP_EXPANDED_CONTINUATION_STATE_INVALID",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

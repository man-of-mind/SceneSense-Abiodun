#!/usr/bin/env python3
"""Create-only supervisor for the final expanded native-grid sufficiency run."""

from __future__ import annotations

import argparse
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
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_native_grid_expanded_training_v2"
CONFIG_SOURCE = PACKAGE_ROOT / "configs/expanded_training_v2.json"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_NATIVE_GRID_EXPANDED_TRAINING_V2_REPORT.md"
POINTER = PACKAGE_ROOT / "NATIVE_GRID_EXPANDED_TRAIN_EXP_DIR.txt"
AUTHORIZED_TERMINALS = {
    "LRASPP_EXPANDED_LONGTRAIN_SERVICE_READY",
    "LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY",
    "LRASPP_EXPANDED_LONGTRAIN_NO_GAIN",
    "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY",
    "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH20_NO_PROGRESS",
    "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE",
}


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


def run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def notify(terminal: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["notify-send", "Route B expanded native-grid training", terminal],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return {
            "attempted": True, "available": result.returncode == 0,
            "returncode": result.returncode, "stderr": result.stderr[-1000:],
        }
    except Exception as exc:
        return {"attempted": True, "available": False,
                "error": f"{type(exc).__name__}: {exc}"}


def cleanup(experiment: Path, decision: dict[str, Any]) -> dict[str, Any]:
    terminal = decision["terminal"]
    destructive_cleanup = terminal in {
        "LRASPP_EXPANDED_LONGTRAIN_NO_GAIN",
        "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY",
        "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH20_NO_PROGRESS",
    }
    selected = decision["selected"]
    retained = selected or decision["best_ranked_regardless_of_eligibility"]
    retained_epoch = int(retained["epoch"])
    removed_checkpoints: list[dict[str, Any]] = []
    removed_predictions: list[dict[str, Any]] = []
    if destructive_cleanup:
        checkpoint_dir = experiment / "checkpoints"
        for path in sorted(checkpoint_dir.glob("*/*.pt")):
            if path.name == f"epoch_{retained_epoch:03d}.pt":
                continue
            removed_checkpoints.append({"path": str(path), "sha256": sha256(path)})
            path.unlink()
        predictions = experiment / "predictions"
        for root in sorted(path for path in predictions.iterdir() if path.is_dir()):
            manifest = json.loads((root / "inference_manifest.json").read_text())
            removed_predictions.append({
                "path": str(root), "checkpoint_epoch": manifest["checkpoint_epoch"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "detections_sha256": manifest["detections_sha256"],
                "prediction_set_sha256": manifest["prediction_set_sha256"],
            })
            shutil.rmtree(root)
    retained_path = Path(retained["checkpoint"])
    return {
        "schema": "route_b_v3_1_native_grid_expanded_training_cleanup_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": terminal, "cleanup_required": destructive_cleanup,
        "removed_nonselected_checkpoints": removed_checkpoints,
        "removed_raw_inference_payloads": removed_predictions,
        "retained_checkpoint_epoch": retained_epoch,
        "retained_checkpoint": str(retained_path),
        "retained_checkpoint_sha256": sha256(retained_path),
        "warm_start_or_corpus_payloads_removed": 0,
    }


def training_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Ep | Stage | Object LR start→end | Inherited LR start→end | Train / val loss | Centre / offset / XYZ / bbox | Dim / yaw / seg | V/P/FG IoU |",
        "|---:|:---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {row['stage']} | {row['object_lr_start']:.3g}→{row['object_lr_end']:.3g} | "
            f"{row['inherited_lr_start']:.3g}→{row['inherited_lr_end']:.3g} | "
            f"{row['train_total_loss']:.4f} / {row['validation_total_loss']:.4f} | "
            f"{row['center_loss']:.4f} / {row['offset_loss']:.4f} / {row['loc_loss']:.4f} / {row['bbox2d_loss']:.4f} | "
            f"{row['dim_loss']:.4f} / {row['yaw_loss']:.4f} / {row['seg_loss']:.4f} | "
            f"{row['vehicle_iou']:.4f} / {row['person_box_mask_iou']:.4f} / {row['foreground_miou']:.4f} |"
        )
    return "\n".join(lines)


def decode_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Ep | Vehicle P/R/F1 | V XY | V R@.02 | Person P/R/F1 | P XY | P R@.02 | Dup FP | Heatmap miss | V/P/FG IoU |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        metric = record["metrics"]
        lines.append(
            f"| {record['epoch']} | {metric['vehicle_precision']:.4f}/{metric['vehicle_recall']:.4f}/{metric['vehicle_f1']:.4f} | "
            f"{metric['vehicle_xy_mae_m']:.4f} | {metric['vehicle_recall_002']:.4f} | "
            f"{metric['person_precision']:.4f}/{metric['person_recall']:.4f}/{metric['person_f1']:.4f} | "
            f"{metric['person_xy_mae_m']:.4f} | {metric['person_recall_002']:.4f} | "
            f"{record['vehicle_duplicate_fp']} | {record['person_heatmap_center_miss']} | "
            f"{metric['vehicle_iou']:.4f}/{metric['person_box_mask_iou']:.4f}/{metric['foreground_miou']:.4f} |"
        )
    return "\n".join(lines)


def gate_text(gate: dict[str, Any] | None) -> str:
    if gate is None:
        return "Not reached."
    failed = [name for name, value in gate["gates"].items() if not value]
    return f"`{'PASS' if gate['pass'] else 'FAIL'}`; failed gates: `{failed}`."


def service_table(values: dict[str, bool] | None) -> str:
    if values is None:
        return "No selected checkpoint; all nine blocking service targets remain unassigned."
    lines = ["| Target | Pass |", "|---|:---:|"]
    lines.extend(f"| {key} | {'yes' if value else 'no'} |" for key, value in values.items())
    return "\n".join(lines)


def report(experiment: Path, decision: dict[str, Any], preflight: dict[str, Any],
           cleanup_result: dict[str, Any], pipeline_wall: float) -> str:
    config = decision["baseline"]
    selected = decision["selected"]
    best = decision["best_ranked_regardless_of_eligibility"]
    loss_best = decision["loss_best_checkpoint"]
    trace = json.loads((experiment / "LR_SCHEDULE_TRACE.json").read_text())
    first = trace["trace"][0]
    step500 = next(item for item in trace["trace"] if item["optimizer_step"] == 500)
    last = trace["trace"][-1]
    selected_text = (
        f"epoch {selected['epoch']}, `{selected['checkpoint']}`, SHA-256 `{selected['checkpoint_sha256']}`"
        if selected else "none"
    )
    best_record = next(
        record for record in decision["decode_records"] if record["epoch"] == best["epoch"]
    )
    base_tax = {
        "vehicle": {"PREDICTED_DUPLICATE": config["vehicle_duplicate_fp"],
                    "TWO_D_CORRECT_WORLD_WRONG": config["vehicle_two_d_correct_world_wrong"]},
        "person": {"CENTER_PRESENT_WORLD_WRONG": config["person_center_present_world_wrong"],
                   "HEATMAP_CENTER_MISS": config["person_heatmap_center_miss"]},
    }
    selected_tax = selected["taxonomy_v010"] if selected else None
    sensitivity = decision["sensitivity_v025"]
    return f"""# Route B v3.1 native-grid expanded training v2 report

Terminal: `{decision['terminal']}`

Experiment: `{experiment}`

## Expanded-view and immutable-source contract

- Train: 10 episodes / 16,827 frames; validation: 2 episodes / 3,345 frames; test absent.
- 40,132 symlinks; copied corpus payloads: 0.
- v0.10 train/validation positives and ignores: 64,516/290,498 and 13,597/57,601.
- Camera-plane localization-ignore v0.10 train/validation: 184/34; v0.25: 25/1.
- Expanded-view summary SHA-256: `{preflight['training_view_hashes']['expanded_view_summary']}`.
- Retained validation hashes: `{preflight['training_view_hashes']['retained_validation']}`.
- Warm start SHA-256: `1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed`.
- Imported native model/target/loss/decoder hashes: `{preflight['source_hashes']}`. No immutable native source was edited.

All eight bounded preflight checks passed using `/usr/bin/python3` on the sm_120 RTX 5090, including one real q=0 AMP Stage-H2 batch and exact explicit {{low,high}} split parity.

## Registered LR schedule and execution proof

H2 epochs 1–5 froze backbone/classifier and all BatchNorm, warmed the object group linearly for 500 optimizer steps to `1e-4`, then held it. J2 epochs 6–7 warmed object/inherited groups linearly to `5e-5/5e-6`; epochs 8–40 used cosine decay toward exactly 10% of each peak. AdamW weight decay was `1e-4`; batch 16; AMP cache disabled.

- First optimizer step: object `{first['object']:.12g}`, inherited `{first['inherited']:.12g}`.
- Step 500: object `{step500['object']:.12g}`, inherited `{step500['inherited']:.12g}`.
- Last executed step {last['optimizer_step']} (epoch {last['epoch']}): object `{last['object']:.12g}`, inherited `{last['inherited']:.12g}`.
- Checkpoints contain model, optimizer, scheduler, GradScaler, epoch, Python/NumPy/Torch/CUDA RNG states, resolved config, view hashes, and warm-start hash.

## Epoch-wise loss and LR

{training_table(decision['training_rows'])}

Loss-best decoded checkpoint: epoch {loss_best['epoch']} at validation loss `{loss_best['validation_total_loss']:.6f}`, SHA-256 `{loss_best['sha256']}`. It was not auto-promoted.

## Authorized validation decodes

{decode_table(decision['decode_records'])}

Decoded epochs were exactly `{decision['decoded_epochs']}`; each used one inference pass at score floor 0.02 for both registered thresholds.

## Decision gates and selection

- Epoch-10 stability: {gate_text(decision['epoch10_gate'])}
- Epoch-20 continuation: {gate_text(decision['epoch20_gate'])}
- Primary-eligible epochs in registered rank order: `{decision['primary_ranking']}`.
- Selected checkpoint: {selected_text}.
- Best-ranked regardless of eligibility, retained when no selected checkpoint exists: epoch {best['epoch']}, `{best['checkpoint']}`, SHA-256 `{best['checkpoint_sha256']}`.

Baseline taxonomy: `{base_tax}`. Selected taxonomy: `{selected_tax}`. Best-ranked diagnostic taxonomy: `{best_record['taxonomy_v010']}`.

Selected v0.10 metrics: `{selected['metrics_v010'] if selected else None}`.

Selected-only v0.25 sensitivity: `{sensitivity}`. Sensitivity reversal gates: `{decision['sensitivity_no_reversal_gates']}`. No substitute or additional checkpoint received v0.25 scoring.

Material-gain gates: `{decision['material_gain_gates']}`.

## Blocking service targets

{service_table(decision['service_targets'])}

q/AE was not started.

## Runtime, resources, cleanup, and scope

- Training/evaluation wall: `{decision['wall_seconds']:.3f} s`; full pipeline wall: `{pipeline_wall:.3f} s`.
- Peak CUDA allocated/reserved: `{decision['peak_allocated_mib']:.1f}/{decision['peak_reserved_mib']:.1f} MiB`.
- Cleanup: `{cleanup_result}`.
- Test, CARLA, OAI, containers, q/AE, feature drop, and 288 measurements were untouched. No decoder calibration, threshold/NMS sweep, loss sweep, or follow-up experiment ran.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    parser.add_argument("--resume-experiment", type=Path)
    parser.add_argument("--resume-reason")
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    started = time.monotonic()
    if args.resume_experiment is not None:
        experiment = args.resume_experiment.resolve(strict=True)
        if experiment.parent != EXPERIMENT_PARENT.resolve() or TRACKED_REPORT.exists():
            raise RuntimeError("resume target is outside the registered parent or report already exists")
        if POINTER.read_text().strip() != str(experiment):
            raise RuntimeError("resume pointer mismatch")
        if (experiment / "TRAINING_STARTED.json").exists():
            raise RuntimeError("refusing resume after any training launch")
        attempt = 1
        while (experiment / f"PIPELINE_ATTEMPT_{attempt}_FAILURE.json").exists():
            attempt += 1
        attempt_files = {
            "PREFLIGHT.json": f"PREFLIGHT_ATTEMPT_{attempt}_FAILURE.json",
            "PIPELINE_FAILURE.json": f"PIPELINE_ATTEMPT_{attempt}_FAILURE.json",
            "TERMINAL_VERDICT.txt": f"TERMINAL_VERDICT_ATTEMPT_{attempt}.txt",
            "COMPLETION_SENTINEL": f"COMPLETION_SENTINEL_ATTEMPT_{attempt}",
            "NOTIFICATION.json": f"NOTIFICATION_ATTEMPT_{attempt}.json",
            "FAILED.pid": f"FAILED_ATTEMPT_{attempt}.pid",
            "logs/preflight.log": f"logs/preflight_attempt_{attempt}.log",
        }
        for source, target in attempt_files.items():
            source_path, target_path = experiment / source, experiment / target
            if not source_path.exists() or target_path.exists():
                raise RuntimeError(f"preflight-only resume provenance mismatch: {source}")
            source_path.rename(target_path)
        write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
        write_json_x(experiment / f"PIPELINE_RESUMED_ATTEMPT_{attempt}.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "interpreter": sys.executable,
            "prior_training_launches": 0,
            "reason": args.resume_reason or "bounded preflight implementation repair",
        })
        config_path = experiment / "resolved_configs" / CONFIG_SOURCE.name
        logs = experiment / "logs"
    else:
        if TRACKED_REPORT.exists() or POINTER.exists():
            raise FileExistsError("tracked report or create-only experiment pointer already exists")
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = EXPERIMENT_PARENT / timestamp
        experiment.mkdir(parents=True, exist_ok=False)
        logs = experiment / "logs"
        logs.mkdir()
        resolved = experiment / "resolved_configs"
        resolved.mkdir()
        config_path = resolved / CONFIG_SOURCE.name
        shutil.copyfile(CONFIG_SOURCE, config_path)
        config = json.loads(config_path.read_text())
        view = (ROOT / config["training_view"]).resolve(strict=True)
        os.symlink(str((view / "dataset").resolve()), experiment / "dataset")
        os.symlink(str((view / "contracts").resolve()), experiment / "contracts")
        write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
        write_text_x(POINTER, str(experiment) + "\n")
        write_json_x(experiment / "PIPELINE_STARTED.json", {
            "schema": "route_b_v3_1_native_grid_expanded_training_pipeline_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "interpreter": sys.executable,
            "resolved_config_sha256": sha256(config_path),
            "training_view": str(view),
        })
    terminal = "LRASPP_EXPANDED_LONGTRAIN_RUNTIME_FAILURE"
    try:
        preflight_command = [
            sys.executable, str(PACKAGE_ROOT / "preflight_v2.py"),
            "--experiment", str(experiment), "--config", str(config_path),
        ]
        if run_logged(preflight_command, logs / "preflight.log") != 0:
            raise RuntimeError("preflight failed")
        train_command = [
            sys.executable, str(PACKAGE_ROOT / "train_long_v2.py"),
            "--experiment", str(experiment), "--config", str(config_path),
        ]
        if run_logged(train_command, logs / "training.log") != 0:
            raise RuntimeError("training/decision phase failed")
        decision = json.loads((experiment / "DECISION.json").read_text())
        terminal = decision["terminal"]
        if terminal not in AUTHORIZED_TERMINALS:
            raise RuntimeError(f"unauthorized terminal {terminal}")
        cleanup_result = cleanup(experiment, decision)
        write_json_x(experiment / "CLEANUP.json", cleanup_result)
        preflight = json.loads((experiment / "PREFLIGHT.json").read_text())
        pipeline_wall = time.monotonic() - started
        final_report = report(experiment, decision, preflight, cleanup_result, pipeline_wall)
        write_text_x(experiment / "FINAL_REPORT.md", final_report)
        write_text_x(TRACKED_REPORT, final_report)
        write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
        write_json_x(experiment / "PIPELINE_COMPLETE.json", {
            "terminal": terminal, "created_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": pipeline_wall, "cleanup": cleanup_result,
        })
        write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
        (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
        write_json_x(experiment / "NOTIFICATION.json", notify(terminal))
        print(json.dumps({
            "terminal": terminal, "experiment": str(experiment),
            "retained_checkpoint": cleanup_result["retained_checkpoint"],
        }, indent=2), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "terminal": terminal, "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.monotonic() - started,
        }
        write_json_x(experiment / "PIPELINE_FAILURE.json", failure)
        write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
        write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
        if (experiment / "RUNNING.pid").exists():
            (experiment / "RUNNING.pid").rename(experiment / "FAILED.pid")
        write_json_x(experiment / "NOTIFICATION.json", notify(terminal))
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

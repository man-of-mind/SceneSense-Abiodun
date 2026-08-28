#!/usr/bin/env python3
"""Sequential, fail-closed Phase-B launch/train/evaluate/cleanup supervisor."""

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
CONTRACT_EXPERIMENT = ROOT / "experiments/route_b_v3_1_camera_plane_contract_v1/20260828_060131"
EXPERIMENT_PARENT = ROOT / "experiments/route_b_v3_1_factorized_localization_v2"
TRAINING_SOURCE = PACKAGE_ROOT / "configs/factorized_localization_training_v2.json"
SELECTION_SOURCE = PACKAGE_ROOT / "configs/selection_contract_v2.json"
TRACKED_REPORT = PACKAGE_ROOT / "ROUTE_B_V3_1_FACTORIZED_LOCALIZATION_V2_REPORT.md"


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


def notification(experiment: Path, terminal: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["notify-send", "Route B v3.1 factorized localization", terminal],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return {"attempted": True, "available": completed.returncode == 0,
                "returncode": completed.returncode, "stderr": completed.stderr[-1000:]}
    except Exception as exc:
        return {"attempted": True, "available": False,
                "error": f"{type(exc).__name__}: {exc}"}


def metric_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Epoch | Eligible | Vehicle P/R/F1 | Vehicle XY m | Person P/R/F1 | Person XY m |",
        "|---:|:---:|---|---:|---|---:|",
    ]
    for record in records:
        metric = record["primary_v010"]["flat"]
        lines.append(
            f"| {record['epoch']} | {str(record['eligible']).lower()} | "
            f"{metric['vehicle_precision']:.6f} / {metric['vehicle_recall']:.6f} / {metric['vehicle_f1']:.6f} | "
            f"{metric['vehicle_xy_mae_m']:.6f} | "
            f"{metric['person_precision']:.6f} / {metric['person_recall']:.6f} / {metric['person_f1']:.6f} | "
            f"{metric['person_xy_mae_m']:.6f} |"
        )
    return "\n".join(lines)


def service_table(targets: dict[str, bool] | None) -> str:
    if targets is None:
        return "No checkpoint passed all eligibility gates; service gates were not assigned to a selected model."
    labels = {
        "vehicle_precision_ge_0_80": "Vehicle precision >= 0.80",
        "vehicle_recall_ge_0_85": "Vehicle recall >= 0.85",
        "person_precision_ge_0_80": "Person precision >= 0.80",
        "person_recall_ge_0_80": "Person recall >= 0.80",
        "vehicle_xy_mae_le_1_0m": "Vehicle XY MAE <= 1.0 m",
        "person_xy_mae_le_1_2m": "Person XY MAE <= 1.2 m",
        "vehicle_iou_ge_0_85": "Vehicle IoU >= 0.85",
        "person_box_mask_iou_ge_0_50": "Person box-mask IoU >= 0.50",
        "foreground_miou_ge_0_675": "Foreground mIoU >= 0.675",
    }
    lines = ["| Service target | Pass |", "|---|:---:|"]
    lines.extend(f"| {labels[key]} | {'yes' if value else 'no'} |" for key, value in targets.items())
    return "\n".join(lines)


def make_report(experiment: Path, terminal: str, pipeline_wall: float,
                cleanup: dict[str, Any]) -> str:
    contract = json.loads((CONTRACT_EXPERIMENT / "CAMERA_PLANE_CONTRACT_SUMMARY.json").read_text())
    amended = json.loads((CONTRACT_EXPERIMENT / "AMENDED_BASELINE.json").read_text())
    launch = json.loads((experiment / "LAUNCH_CHECKS.json").read_text())
    training = json.loads((experiment / "TRAINING_COMPLETE.json").read_text())
    evaluation = json.loads((experiment / "SELECTION.json").read_text())
    baseline = amended["amended"]["v010"]["flat"]
    selected = evaluation["selected"]
    best_epoch = cleanup["retained_checkpoint_epoch"]
    retained_path = cleanup["retained_checkpoint"]
    retained_sha = cleanup["retained_checkpoint_sha256"]
    selected_text = (
        f"epoch {selected['epoch']}: `{selected['checkpoint']}` (`{selected['checkpoint_sha256']}`)"
        if selected is not None else "none (no checkpoint passed every eligibility gate)"
    )
    taxonomy_base = amended["amended_taxonomy"]
    taxonomy_selected = selected["taxonomy_v010"] if selected else None
    sensitivity = selected["sensitivity_v025"]["flat"] if selected else None
    radar = evaluation["radar_stratified_localization"]
    parameters = training["parameter_report"]["model_total"]
    split = launch["checks"][6]["detail"]
    total_wall = (contract["wall_seconds"] + amended["wall_seconds"] + pipeline_wall)
    peak_allocated = max(
        training["peak_allocated_mib"],
        *(record["inference"]["peak_allocated_mib"] for record in evaluation["records"]),
    )
    peak_reserved = max(
        training["peak_reserved_mib"],
        *(record["inference"]["peak_reserved_mib"] for record in evaluation["records"]),
    )
    return f"""# Route B v3.1 factorized localization v2 report

Terminal: `{terminal}`

## Camera-plane contract

The reusable rule moves any localization-positive object with physical-centre camera-forward depth `<= 0` to localization-ignore with reason `CAMERA_PLANE_STRADDLING_CENTER_NONPOSITIVE_DEPTH`. Segmentation remains unchanged and the region is neutral, never background.

- v0.10 train exclusions: {contract['summaries']['v010']['train']['transition_records']}.
- v0.10 validation exclusions: {contract['summaries']['v010']['val']['transition_records']} (26 actor, 8 static-environment, 11 identities, zero person).
- v0.25 train/validation exclusions: {contract['summaries']['v025']['train']['transition_records']} / {contract['summaries']['v025']['val']['transition_records']}.
- All nine hard contract gates passed; test rows are absent and raw corpus files copied = 0.

## Amended native epoch-15 baseline (v0.10)

- Vehicle: P/R/F1 `{baseline['vehicle_precision']:.6f}/{baseline['vehicle_recall']:.6f}/{baseline['vehicle_f1']:.6f}`, XY MAE `{baseline['vehicle_xy_mae_m']:.6f} m`.
- Person: P/R/F1 `{baseline['person_precision']:.6f}/{baseline['person_recall']:.6f}/{baseline['person_f1']:.6f}`, XY MAE `{baseline['person_xy_mae_m']:.6f} m`.
- IoU: vehicle `{baseline['vehicle_iou']:.6f}`, person box-mask `{baseline['person_box_mask_iou']:.6f}`, foreground mIoU `{baseline['foreground_miou']:.6f}`.
- Re-score used retained detections only (`{amended['retained_detections_sha256']}`); new inference passes = 0. v0.10 and v0.25 ignore caches were independently keyed.

## Architecture, isolation, and split proof

The new tail-side path reads the frozen native stride-4 128-channel fused feature, applies two `3x3 Conv(64)+GroupNorm+ReLU` blocks, and emits one `log_depth` channel plus two projected-physical-centre offset channels. It unprojects positive `exp(log_depth)` using per-frame intrinsics and replaces only decoded XYZ. Legacy XYZ remains checkpoint-compatible but untrained.

The transported bundle remains exactly `{split['transported_feature_names']}`; tail raw-modality side channels are `{split['tail_raw_modality_side_channels']}` and monolithic/split outputs were bit-identical. Trainable parameters: {parameters['trainable']:,}; frozen parameters: {parameters['frozen']:,}; total: {parameters['total']:,}.

## Validation checkpoints (v0.10)

{metric_table(evaluation['records'])}

Exactly epochs 4, 8, and 12 were evaluated. Each checkpoint had one inference pass at score floor 0.02 supplying both fixed thresholds.

## Selection and taxonomy

Selected checkpoint: {selected_text}.

Best-ranked retained checkpoint regardless of promotion: epoch {best_epoch}, `{retained_path}` (`{retained_sha}`).

Baseline vehicle taxonomy: `{taxonomy_base['vehicle_fp_at_0_20']['counts']}`. Baseline person taxonomy: `{taxonomy_base['person_fn_at_0_02']['counts']}`.

Selected vehicle taxonomy: `{taxonomy_selected['vehicle_fp_at_0_20']['counts'] if taxonomy_selected else None}`. Selected person taxonomy: `{taxonomy_selected['person_fn_at_0_02']['counts'] if taxonomy_selected else None}`.

## Visibility and radar stratification

Selected v0.25 flat metrics: `{sensitivity}`. The selected-only v0.25 scorer used an independent cache; material selection required no class F1/XY reversal.

Radar-supported/unsupported localization at score 0.20 and 3 m matching: amended baseline `{radar['amended_baseline']}`; selected `{radar['selected']}`.

## Service targets

{service_table(selected['service_targets'] if selected else None)}

Frozen segmentation cannot improve in this localization-only run, so full service readiness is not claimed.

## Resources, cleanup, and safety

Measured combined contract/re-score/pipeline wall time: `{total_wall:.3f} s` (pipeline `{pipeline_wall:.3f} s`). Peak CUDA allocated/reserved: `{peak_allocated:.1f}/{peak_reserved:.1f} MiB`.

Hashes and metrics were recorded before removing all three inference payloads and the two non-retained new checkpoints. Canonical data, retained epoch-15 checkpoint/predictions, and prior experiments were not changed. Test remained unopened; CARLA, OAI, containers, q/AE, and 288 measurements were not run. No q/AE phase or follow-on experiment was started.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    started = time.monotonic()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = EXPERIMENT_PARENT / timestamp
    experiment.mkdir(parents=True, exist_ok=False)
    write_text_x(experiment / "RUNNING.pid", f"{os.getpid()}\n")
    logs = experiment / "logs"
    logs.mkdir()
    resolved = experiment / "resolved_configs"
    resolved.mkdir()
    training_config = resolved / TRAINING_SOURCE.name
    selection_contract = resolved / SELECTION_SOURCE.name
    shutil.copyfile(TRAINING_SOURCE, training_config)
    shutil.copyfile(SELECTION_SOURCE, selection_contract)
    os.symlink(str((CONTRACT_EXPERIMENT / "dataset").resolve()), experiment / "dataset")
    os.symlink(str((CONTRACT_EXPERIMENT / "contracts").resolve()), experiment / "contracts")
    write_json_x(experiment / "PIPELINE_STARTED.json", {
        "schema": "route_b_v3_1_factorized_localization_pipeline_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "contract_experiment": str(CONTRACT_EXPERIMENT),
        "training_config_sha256": sha256(training_config),
        "selection_contract_sha256": sha256(selection_contract),
        "selection_contract_registered_before_first_candidate_inference": True,
    })
    terminal = "LRASPP_FACTORIZED_LOCALIZATION_RUNTIME_FAILURE"
    try:
        phases = [
            ("launch", [
                sys.executable, str(PACKAGE_ROOT / "launch_check_v2.py"),
                "--experiment", str(experiment), "--training-config", str(training_config),
                "--selection-contract", str(selection_contract),
                "--contract-experiment", str(CONTRACT_EXPERIMENT),
            ]),
            ("training", [
                sys.executable, str(PACKAGE_ROOT / "train_v2.py"),
                "--experiment", str(experiment), "--config", str(training_config),
                "--contract-experiment", str(CONTRACT_EXPERIMENT),
            ]),
            ("evaluation", [
                sys.executable, str(PACKAGE_ROOT / "evaluate_v2.py"),
                "--experiment", str(experiment),
                "--contract-experiment", str(CONTRACT_EXPERIMENT),
                "--selection-contract", str(selection_contract),
                "--infer-script", str(PACKAGE_ROOT / "infer_v2.py"),
            ]),
        ]
        for name, command in phases:
            code = run_logged(command, logs / f"{name}.log")
            if code != 0:
                if name == "launch" and (experiment / "LAUNCH_CHECKS.json").is_file():
                    launch = json.loads((experiment / "LAUNCH_CHECKS.json").read_text())
                    if not launch.get("contract_valid", False):
                        terminal = "LRASPP_CAMERA_PLANE_CONTRACT_INVALID"
                raise RuntimeError(f"{name} phase exited {code}")

        evaluation = json.loads((experiment / "SELECTION.json").read_text())
        terminal = evaluation["terminal"]
        selected = evaluation["selected"]
        retain_epoch = (selected["epoch"] if selected is not None
                        else evaluation["best_ranked_epoch_regardless_of_eligibility"])
        checkpoint_dir = experiment / "checkpoints/route_b_v3_1_factorized_localization_v2"
        removed_checkpoints = []
        retained_checkpoint = checkpoint_dir / f"epoch_{retain_epoch:03d}.pt"
        for checkpoint in sorted(checkpoint_dir.glob("epoch_*.pt")):
            if checkpoint == retained_checkpoint:
                continue
            removed_checkpoints.append({"path": str(checkpoint), "sha256": sha256(checkpoint)})
            checkpoint.unlink()
        removed_predictions = []
        predictions_dir = experiment / "predictions"
        for prediction_root in sorted(predictions_dir.iterdir()):
            manifest = json.loads((prediction_root / "inference_manifest.json").read_text())
            removed_predictions.append({
                "path": str(prediction_root), "checkpoint_epoch": manifest["checkpoint_epoch"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "detections_sha256": manifest["detections_sha256"],
                "prediction_set_sha256": manifest["prediction_set_sha256"],
            })
            shutil.rmtree(prediction_root)
        cleanup = {
            "schema": "route_b_v3_1_factorized_localization_cleanup_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "removed_nonselected_checkpoints": removed_checkpoints,
            "removed_redundant_inference_payloads": removed_predictions,
            "retained_checkpoint_epoch": retain_epoch,
            "retained_checkpoint": str(retained_checkpoint),
            "retained_checkpoint_sha256": sha256(retained_checkpoint),
            "canonical_or_prior_artifacts_removed": 0,
        }
        write_json_x(experiment / "CLEANUP.json", cleanup)
        pipeline_wall = time.monotonic() - started
        report = make_report(experiment, terminal, pipeline_wall, cleanup)
        write_text_x(experiment / "FINAL_REPORT.md", report)
        if TRACKED_REPORT.exists():
            raise FileExistsError(f"refusing to overwrite tracked report {TRACKED_REPORT}")
        write_text_x(TRACKED_REPORT, report)
        write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
        write_json_x(experiment / "PIPELINE_COMPLETE.json", {
            "terminal": terminal, "created_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": pipeline_wall, "cleanup": cleanup,
        })
        write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
        (experiment / "RUNNING.pid").rename(experiment / "COMPLETED.pid")
        write_json_x(experiment / "NOTIFICATION.json", notification(experiment, terminal))
        print(json.dumps({"terminal": terminal, "experiment": str(experiment),
                          "retained_checkpoint": str(retained_checkpoint)}, indent=2), flush=True)
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
        write_json_x(experiment / "NOTIFICATION.json", notification(experiment, terminal))
        print(json.dumps(failure, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

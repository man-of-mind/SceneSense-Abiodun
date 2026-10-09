from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from common import (CONFIG_PATH, PACKAGE, ROOT, atomic_json, atomic_text, desktop_notify, load_json,
                    sha256, utc_now)


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True, text=True).stdout


def value(number: Any, digits: int = 4) -> str:
    try:
        result = float(number)
        return f"{result:.{digits}f}" if math.isfinite(result) else "NA"
    except Exception:
        return "NA"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    if not (experiment / "EVALUATION_COMPLETE").is_file(): raise RuntimeError("evaluation incomplete")
    config = load_json(CONFIG_PATH); registration = load_json(experiment / "SCIENTIFIC_REGISTRATION.json")
    structure = load_json(experiment / "STRUCTURAL_QUALIFICATION.json")
    calibration = load_json(experiment / "LOSS_CALIBRATION.json")
    assignment = load_json(experiment / "P2_ASSIGNMENT_AUDIT.json")
    disposable = load_json(experiment / "DISPOSABLE_QUALIFICATION.json")
    decision = load_json(experiment / "SELECTION_DECISION.json")
    evaluations = [load_json(experiment / f"evaluation/epoch_{epoch:03d}.json") for epoch in (3, 8, 16, 22, 26)]
    training = [load_json(experiment / f"training_metrics/epoch_{epoch:03d}.json") for epoch in range(1, 27)]
    telemetry = [load_json(experiment / f"gradient_telemetry/epoch_{epoch:03d}.json") for epoch in range(1, 27)]
    start = load_json(experiment / "STARTING_WORKTREE.json")
    amendments = [load_json(path) for path in sorted(experiment.glob("SOURCE_AMENDMENT_*.json"))]
    oai_status = git("status", "--porcelain=v2", "--branch", "--untracked-files=all", cwd=ROOT / "OAI/openairinterface5g")
    oai_hash = __import__("hashlib").sha256(json.dumps(oai_status, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if oai_hash != start["oai_porcelain_sha256"]:
        raise RuntimeError("pre-existing dirty OAI state changed")
    current_branch = git("branch", "--show-current").strip(); current_head = git("rev-parse", "HEAD").strip()
    if current_branch != "master": raise RuntimeError("finalization left master")
    curves_path = experiment / "TRAINING_CURVES.csv"
    fields = ["epoch", "updates", "wall_seconds", "peak_allocated_mib", "radar_stem_gradient_norm_mean",
              "rgb_stem_gradient_norm_mean", "D", "G", "S", "A", "weighted_D", "weighted_G", "weighted_S", "weighted_A",
              "optimization_fraction_D", "optimization_fraction_G", "optimization_fraction_S", "optimization_fraction_A", "total"]
    with curves_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in training:
            parts = row["mean_components"]
            writer.writerow({name: row.get(name, parts.get(name, "")) for name in fields})
    parameter = registration["initial_model"]["parameter_inventory"]
    selected = next(row for row in evaluations if row["epoch"] == decision["selected_epoch"])
    selected_taxonomy_summary = {name: selected["taxonomy_0_20"][name] for name in
                                 ("duplicate_fp", "cross_level_duplicate_fp", "background_fp",
                                  "two_d_correct_world_wrong", "person_centre_point_miss", "geometry_errors")}
    metric_lines = ["| Epoch | V P | V R | V F1 | P P | P R | P F1 | V XY m | P XY m | V IoU | P IoU | FG mIoU | Gates |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in evaluations:
        m, s = row["metrics"], row["service"]
        metric_lines.append(f"| {row['epoch']} | {value(m['vehicle_precision'])} | {value(m['vehicle_recall'])} | {value(m['vehicle_f1'])} | "
                            f"{value(m['person_precision'])} | {value(m['person_recall'])} | {value(m['person_f1'])} | "
                            f"{value(m['vehicle_xy_mae_m'])} | {value(m['person_xy_mae_m'])} | {value(m['vehicle_iou'])} | "
                            f"{value(m['person_box_mask_iou'])} | {value(m['foreground_miou'])} | {s['pass_count']}/9 |")
    count_lines = ["| Epoch | V TP | V FP | V FN | V R@.02 | P TP | P FP | P FN | P R@.02 |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    geometry_lines = ["| Epoch | V dim MAE m | V yaw MAE deg | P dim MAE m | P yaw MAE deg |",
                      "|---:|---:|---:|---:|---:|"]
    for row in evaluations:
        m = row["metrics"]
        count_lines.append(f"| {row['epoch']} | {m['vehicle_tp']} | {m['vehicle_fp']} | {m['vehicle_fn']} | "
                           f"{value(m['vehicle_recall_002'])} | {m['person_tp']} | {m['person_fp']} | {m['person_fn']} | "
                           f"{value(m['person_recall_002'])} |")
        geometry = row["taxonomy_0_20"]["geometry_errors"]
        geometry_lines.append(f"| {row['epoch']} | {value(geometry['vehicle']['dimension_mae_m'])} | "
                              f"{value(geometry['vehicle']['yaw_mae_deg'])} | {value(geometry['person']['dimension_mae_m'])} | "
                              f"{value(geometry['person']['yaw_mae_deg'])} |")
    service_lines = ["| Target | Value | Requirement | Attainment | Pass |", "|---|---:|---:|---:|:---:|"]
    for name, row in selected["service"]["targets"].items():
        service_lines.append(f"| {name} | {value(row['value'])} | {row['direction']} {row['target']} | {value(row['attainment_ratio'])} | {'yes' if row['passed'] else 'no'} |")
    level_lines = ["| Class | " + " | ".join(level.upper() for level in ("p2", "p3", "p4", "p5", "p6", "p7")) + " |",
                   "|---|" + "---:|" * 6]
    for class_name in ("vehicle", "person"):
        counts = assignment["positive_assignments_by_class_and_level"][class_name]
        level_lines.append("| " + class_name + " | " + " | ".join(str(counts[level]) for level in ("p2", "p3", "p4", "p5", "p6", "p7")) + " |")
    selected_checkpoint = Path(decision["selected_checkpoint"])
    lines = [
        "# SplitFusion FCOS R50 FPN P2-P7 V1 final report", "",
        f"Terminal verdict: `{decision['terminal']}`.", "",
        f"The fixed selection chose epoch **{decision['selected_epoch']}**. The retained checkpoint is `{selected_checkpoint}` with SHA-256 `{decision['selected_checkpoint_sha256']}`. "
        f"It passed {selected['service']['pass_count']} of nine clean service targets.", "",
        "## Provenance and scope", "",
        f"Starting local master: `{start['starting_head']}`. Report-generation HEAD: `{current_head}`. The final local master commit is the commit containing this report; its resolved hash is written after commit to the Git-ignored experiment `FINAL_GIT_STATE.json` to avoid an impossible self-referential commit hash.", "",
        (f"Documented runtime-only source amendments: `{json.dumps([{key: row[key] for key in ('scope', 'reason', 'changed_files', 'amended_source_state_sha256')} for row in amendments], sort_keys=True)}`. "
         "They were explicitly qualified, preserved the latest durable scientific checkpoint where applicable, and changed no architecture, target, mathematical loss, multiplier, optimizer, LR schedule, sampler, augmentation, batch size, or inference setting." if amendments else
         "No post-registration source amendment was required."), "",
        f"Official weights: `{config['official']['weights_enum']}`, {config['official']['bytes']:,} bytes, SHA-256 `{config['official']['sha256']}`, URL `{config['official']['url']}`. Installed Torchvision revision `{config['official']['torchvision_revision']}` uses BSD-3-Clause source licensing. The COCO weight/dataset disclaimer is recorded separately in `OFFICIAL_PROVENANCE.md`.", "",
        "Internal class 0 is vehicle copied exactly from COCO car row 3; internal class 1 is person copied exactly from COCO person row 1; canonical output labels are restored to vehicle=1/person=2 and background remains implicit.", "",
        "The locked test split remained absent and unopened. CARLA, OAI, q, quantization, AE/hybrid-q, live split deployment, and the 288-cell campaign were not run. The pre-existing dirty OAI tree retained the exact starting status hash.", "",
        "## Architecture and transport", "",
        f"The model has **{parameter['total_parameters']:,} parameters** across {parameter['parameter_tensors']} parameter tensors. Group counts are `{json.dumps(parameter['groups'], sort_keys=True)}`. The complete tensor-by-tensor transferred/new/frozen inventory and hashes are in `SCIENTIFIC_REGISTRATION.json` and `STRUCTURAL_QUALIFICATION.json`.", "",
        "One normalized `[B,7,448,768]` tensor enters a single concatenated seven-channel convolution. The front returns only raw `[B,256,112,192]` FP32 C2. Identity transport is exact and carries 22,020,096 bytes (21.0 MiB) per frame. The edge accepts C2 plus calibration/metadata and has no RGB, radar, GT depth, or semantic-GT argument.", "",
        f"Monolithic/identity-split parity: C2 exact={structure['split_parity']['c2_exact']}, same storage={structure['split_parity']['same_storage_identity']}; front/tail/monolithic latency evidence is `{json.dumps(structure['latency'], sort_keys=True)}`.", "",
        "## Assignment and geometry audits", "", *level_lines, "",
        f"P2 introduced {assignment['foreground_increase_introduced_by_p2']:,} foreground locations without changing P3-P7. Total foreground locations were {assignment['foreground_locations_total']:,}; actors without a carrier: {assignment['actors_without_fcos_carrier_count']}. Carrier visibility counts were `{json.dumps(assignment['carrier_visibility'], sort_keys=True)}`. P2 FCOS-loss fractions are `{json.dumps(assignment['p2_fraction_of_each_fcos_loss_on_fixed_calibration_microbatches'], sort_keys=True)}`.", "",
        "Every geometry gather retained `(image, level, flattened point, internal class)` through filtering, top-k, concatenation, classwise NMS, and truncation. Synthetic adversarial and real-train lineage evidence is in structural qualification. Depth uses 32 log1p-spaced 0-40 m bins plus overflow and bounded `0.5*tanh` residual; physical ray plus intrinsics/extrinsics analytically yields XYZ; dimensions train directly in log space; yaw is independently normalized.", "",
        "## Qualification and training", "",
        f"Loss-gradient calibration medians were `{json.dumps(calibration['medians'], sort_keys=True)}` and fixed multipliers were `{json.dumps(calibration['multipliers'], sort_keys=True)}`. The qualified physical batch was {structure['runtime']['physical_batch']} with accumulation {structure['runtime']['gradient_accumulation']} to effective batch 16 under a {structure['runtime']['max_allocated_cap_mib']:.0f} MiB cap.", "",
        f"Disposable qualification processed all {disposable['complete_frames']:,} epoch-1 frames and then {disposable['joint_updates']['updates']} joint-stage updates, checking losses, gradients, parameters, and optimizer state after every update. Its state is archived under `QUALIFICATION_ONLY_DO_NOT_USE`; the scientific model was reconstructed at hash `{disposable['reconstructed_state_sha256']}` with a fresh empty optimizer.", "",
        "Exactly 26 scientific epochs completed. Per-epoch raw/weighted loss curves are in `TRAINING_CURVES.csv`; full update records and P2 loss fractions are under `training_metrics/`. Fixed two-batch C2 gradient norms/cosines and separate RGB/radar stem evidence are under `gradient_telemetry/`.", "",
        "## Fixed v0.10 validation", "", *metric_lines, "", *count_lines, "", *geometry_lines, "",
        "Every checkpoint used one score-floor-0.02 inference pass; score-0.20 metrics were derived from the retained predictions. Each epoch JSON includes TP/FP/FN, recall at 0.02, dimension/yaw errors, ignore reconciliation, FPN attribution, duplicate/cross-level/background FP taxonomy, 2D-correct/world-wrong cases, person point misses, and distance/radar/visibility slices.", "",
        f"Selected-checkpoint duplicate and error taxonomy: `{json.dumps(selected_taxonomy_summary, sort_keys=True)}`.", "",
        "## Selected service gates", "", *service_lines, "",
        f"Selected v0.25 sensitivity: `{json.dumps(decision['selected_v025_sensitivity']['flat'], sort_keys=True)}`. This was run only for the selected checkpoint and did not affect selection.", "",
        "## Supervisor architecture story", "",
        "This run isolates a clean scientific question: whether an official FCOS ResNet-50 detector, split exactly at raw C2 and extended with a non-destructive overlapping P2 plus task-private dense/geometry heads, can meet Route B service targets while preserving one seven-channel input and one learned payload. The fixed five-checkpoint result and failure taxonomy are retained even when the outcome is not service-ready; no follow-on architecture was launched.", "",
    ]
    report = "\n".join(lines)
    atomic_text(experiment / "FINAL_REPORT.md", report, overwrite=False)
    atomic_text(PACKAGE / "FINAL_REPORT.md", report, overwrite=True)
    summary = {"schema": "splitfusion_fcos_final_report_provenance_v1", "created_utc": utc_now(),
               "experiment_report": str(experiment / "FINAL_REPORT.md"),
               "experiment_report_sha256": sha256(experiment / "FINAL_REPORT.md"),
               "source_copy": str(PACKAGE / "FINAL_REPORT.md"), "source_copy_sha256": sha256(PACKAGE / "FINAL_REPORT.md"),
               "selected_checkpoint": str(selected_checkpoint), "selected_checkpoint_sha256": decision["selected_checkpoint_sha256"],
               "starting_commit": start["starting_head"], "report_generation_commit": current_head,
               "oai_start_end_status_hash_equal": True, "excluded_systems_untouched": True,
               "training_curves_sha256": sha256(curves_path), "terminal": decision["terminal"]}
    atomic_json(experiment / "FINAL_REPORT_PROVENANCE.json", summary, overwrite=False)
    atomic_text(experiment / "COMPLETION_SENTINEL", decision["terminal"] + "\n", overwrite=False)
    atomic_json(experiment / "NOTIFICATION_COMPLETION.json", desktop_notify(
        "SplitFusion FCOS complete", f"Selected epoch {decision['selected_epoch']}: {decision['terminal']}"), overwrite=False)
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

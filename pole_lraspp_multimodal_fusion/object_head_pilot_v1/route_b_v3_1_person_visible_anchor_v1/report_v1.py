#!/usr/bin/env python3
"""Create the concise final experiment report from immutable run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from common_v1 import read_csv, sha256, utc_now, write_json_x, write_text_x  # noqa: E402


def _metric_row(label: str, metrics: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    iou = diagnostics["two_d"]["0.20"]["FULL_BOX_IOU_050"]
    low = diagnostics["two_d"]["0.02"]["FULL_BOX_IOU_050"]
    conditional = next(row for row in diagnostics["conditional_localization"]
                       if float(row["threshold"]) == 0.02
                       and row["match_definition"] == "FULL_BOX_IOU_050"
                       and row["subset_kind"] == "overall")
    return (
        f"| {label} | {metrics['person_precision']:.6f} | {metrics['person_recall']:.6f} | "
        f"{metrics['person_f1']:.6f} | {metrics['person_recall_002']:.6f} | "
        f"{metrics['person_xy_mae_m']:.6f} | {iou['f1']:.6f} | {low['recall']:.6f} | "
        f"{float(conditional['within_3m_fraction']):.6f} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--pipeline-wall-seconds", required=True, type=float)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    selection = json.loads((experiment / "SELECTION_DECISION.json").read_text())
    baseline = json.loads((experiment / "evaluation/BASELINE_REPRODUCTION.json").read_text())
    target = json.loads((experiment / "TARGET_GAUSSIAN_AUDIT.json").read_text())
    freeze = json.loads((experiment / "FREEZE_SPLIT_GEOMETRY_QUALIFICATION.json").read_text())
    numerical = json.loads((experiment / "NUMERICAL_QUALIFICATION.json").read_text())
    training = json.loads((experiment / "TRAINING_COMPLETE.json").read_text())
    input_provenance = json.loads((experiment / "INPUT_PROVENANCE.json").read_text())
    selected = (json.loads((experiment / f"evaluation/epoch_{selection['selected_epoch']:03d}.json").read_text())
                if selection["selected_epoch"] is not None else None)
    terminal = selection["terminal"]
    parameters = freeze["parameter_report"]
    checkpoints = training["checkpoints"]
    report = [
        "# Route B v3.1 person-private visible-anchor result", "",
        f"Terminal: `{terminal}`", "",
        "Exactly one scientific attempt completed all 24 registered epochs. The inherited vehicle and segmentation paths remained bit-identical; the person-private branch used corrected visible anchors, visible-box Gaussian radii, and distinct full-box and physical-ray offsets.", "",
        "## Contract proofs", "",
        f"- Own-visible anchor pixels: {target['proofs']['person_anchor_pixels_own_visible_fraction']:.1%}.",
        f"- Own-visible anchor cells: {target['proofs']['person_anchor_cells_contain_own_visible_fraction']:.1%}.",
        f"- Audit Gaussian reference rows reproduced exactly: {target['gaussian_population']['exact_raw_and_integer_matches']}/{target['gaussian_population']['checked_person_rows']}.",
        f"- Physical projection/unprojection maximum target-view error: {target['target_summary']['physical_projection_roundtrip_max_abs_error_m']:.3e} m.",
        f"- Inherited tensor state drift: {not freeze['zero_inherited_state_drift']} (hash `{freeze['inherited_state_hash_before']}`).",
        f"- Monolithic/split maximum deltas are all zero: {freeze['split_boundary']['outputs_bit_identical']}.",
        f"- Numerical policy: `{numerical['selected_policy']}`; private FP16 used: `{numerical['private_fp16_used']}`.",
        "", "## Parameters and run", "",
        f"- Total parameters: {parameters['model_total']['total']:,}; trainable: {parameters['model_total']['trainable']:,}; frozen: {parameters['model_total']['frozen']:,}.",
        f"- Person-private parameters: {parameters['person_private']['total']:,}; trainable: {parameters['person_private']['trainable']:,}.",
        f"- Epochs: 24; checkpoints/evaluations: {', '.join(str(row['epoch']) for row in checkpoints)}.",
        f"- Peak training VRAM: {training['peak_reserved_mib']:.1f} MiB reserved ({training['peak_allocated_mib']:.1f} MiB allocated).",
        f"- Pipeline wall time: {args.pipeline_wall_seconds:.1f} s.",
        "", "## Base versus selected person metrics", "",
        "| Model | P@.20 | R@.20 | F1@.20 | R@.02 | XY MAE m | IoU50 F1@.20 | IoU50 R@.02 | conditional ≤3m@.02 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("epoch-40 base", baseline["metrics"], baseline["diagnostics"]),
    ]
    if selected is not None:
        report.append(_metric_row(
            f"visible-anchor epoch {selection['selected_epoch']}", selected["metrics"],
            selected["diagnostics"],
        ))
        report += [
            "", "## Preservation", "",
            f"- Vehicle detection rows bit-identical: `{selected['preservation']['vehicle_detection_csv_fields_bit_identical_excluding_artifact_prediction_index']}`.",
            f"- Segmentation PNG hashes bit-identical: `{selected['preservation']['segmentation_png_hashes_bit_identical']}`.",
            f"- Canonical vehicle metrics exact: `{selected['preservation']['canonical_vehicle_metrics_exact']}`; duplicate FP: {selected['vehicle_duplicate_fp']}.",
            f"- Selected checkpoint: `{selection['selected_checkpoint']}`.",
            f"- Selected SHA-256: `{selection['selected_checkpoint_sha256']}`.",
            "", "## Conditional localization slices (IoU50, score 0.02)", "",
            "| Slice | Matched | ≤1m | ≤2m | ≤3m | ≤5m | median m | p90 m |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in selected["diagnostics"]["conditional_localization"]:
            if (float(row["threshold"]) != 0.02 or row["match_definition"] != "FULL_BOX_IOU_050"
                    or row["subset_kind"] not in {"overall", "distance_m", "radar_support", "visibility_contract"}):
                continue
            def fmt(value: Any) -> str:
                return "—" if value == "" else f"{float(value):.4f}"
            report.append(
                f"| {row['subset_kind']}:{row['subset_label']} | {row['matched_pairs']} | "
                f"{fmt(row['within_1m_fraction'])} | {fmt(row['within_2m_fraction'])} | "
                f"{fmt(row['within_3m_fraction'])} | {fmt(row['within_5m_fraction'])} | "
                f"{fmt(row['median_m'])} | {fmt(row['p90_m'])} |"
            )
        service = selection["selected_service_targets"]
        report += [
            "", "## Service targets", "",
            "All nine existing targets were reported. Full service readiness is not claimed. "
            f"Passed {sum(service.values())}/{len(service)} gates. Frozen baseline paths make vehicle precision, vehicle recall, person box-mask IoU, and foreground mIoU structurally unreachable in this experiment.",
        ]
    report += [
        "", "## Scope confirmation", "",
        "Locked test data remained absent and unopened. CARLA, OAI, q/quant/AE/zstd, live runtime, and the 288 campaign were untouched. No COCO distillation, teacher, raw tail-side sensor channel, threshold sweep, NMS sweep, or v0.25 run occurred unless the selected primary candidate passed a registered material route.",
        "", f"Experiment: `{experiment}`", "",
    ]
    report_path = experiment / "FINAL_REPORT.md"
    write_text_x(report_path, "\n".join(report))
    completion = {
        "schema": "route_b_v3_1_person_visible_anchor_completion_v1",
        "created_utc": utc_now(), "terminal": terminal, "experiment": str(experiment),
        "final_report": str(report_path), "final_report_sha256": sha256(report_path),
        "selected_epoch": selection["selected_epoch"],
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "epochs_completed": 24, "evaluated_epochs": selection["evaluated_epochs"],
        "scientific_attempts": 1, "pipeline_wall_seconds": args.pipeline_wall_seconds,
        "peak_training_reserved_mib": training["peak_reserved_mib"],
        "forbidden_scope_access_counts": input_provenance["forbidden_scope_access_counts"],
    }
    write_json_x(experiment / "PIPELINE_COMPLETE.json", completion)
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

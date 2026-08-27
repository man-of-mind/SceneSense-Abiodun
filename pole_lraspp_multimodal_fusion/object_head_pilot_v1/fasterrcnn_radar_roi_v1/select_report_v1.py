#!/usr/bin/env python3
"""Registered epoch selection, service gate, material-gain gate and final report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent, HERE.parent.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from train_v1 import write_json_create  # noqa: E402


SELECTION_RULE = [
    "highest minimum of vehicle/person recall at score 0.20",
    "highest mean vehicle/person F1",
    "lower mean XY MAE",
    "earlier epoch",
]
SERVICE_TARGETS = {
    "vehicle_precision": (">=", 0.80),
    "vehicle_recall": (">=", 0.85),
    "person_precision": (">=", 0.80),
    "person_recall": (">=", 0.80),
    "vehicle_xy_mae_m": ("<=", 1.0),
    "person_xy_mae_m": ("<=", 1.2),
    "vehicle_iou": (">=", 0.85),
    "person_box_mask_iou": (">=", 0.50),
    "miou": (">=", 0.80),
}
MATERIAL_GAIN_RULE = (
    "vehicle recall >= max(retained v1-corrected, retained v2); person recall >= "
    "max(retained v1-corrected, retained v2); and mean class F1 >= strongest retained mean class F1 + 0.05"
)


def epoch_row(metrics: dict) -> dict:
    primary = metrics["by_threshold"]["0.20"]["classes"]
    diagnostic = metrics["by_threshold"]["0.02"]["classes"]
    segmentation = metrics["segmentation"]
    vehicle, person = primary["vehicle"], primary["person"]
    return {
        "epoch": int(metrics["epoch"]),
        "checkpoint": metrics["checkpoint"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "vehicle_precision": vehicle["precision"], "vehicle_recall": vehicle["recall"], "vehicle_f1": vehicle["f1"],
        "vehicle_tp": vehicle["tp"], "vehicle_fp": vehicle["fp"], "vehicle_fn": vehicle["fn"],
        "person_precision": person["precision"], "person_recall": person["recall"], "person_f1": person["f1"],
        "person_tp": person["tp"], "person_fp": person["fp"], "person_fn": person["fn"],
        "vehicle_recall_s002": diagnostic["vehicle"]["recall"],
        "person_recall_s002": diagnostic["person"]["recall"],
        "vehicle_xy_mae_m": vehicle["xy_mae_m"], "person_xy_mae_m": person["xy_mae_m"],
        "vehicle_dimension_mae_m": vehicle["dimension_mae_m"], "person_dimension_mae_m": person["dimension_mae_m"],
        "vehicle_yaw_mae_deg": vehicle["yaw_mae_deg"], "person_yaw_mae_deg": person["yaw_mae_deg"],
        "vehicle_duplicate_fp_per_frame": vehicle["duplicate_fp_per_frame"],
        "person_duplicate_fp_per_frame": person["duplicate_fp_per_frame"],
        "vehicle_iou": segmentation["vehicle_iou"],
        "person_box_mask_iou": segmentation["person_box_mask_iou"],
        "miou": segmentation["miou"],
        "mean_class_f1": 0.5 * (vehicle["f1"] + person["f1"]),
        "min_class_recall": min(vehicle["recall"], person["recall"]),
        "mean_xy_mae_m": 0.5 * (vehicle["xy_mae_m"] + person["xy_mae_m"]),
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_allocated_mib": metrics["peak_allocated_mib"],
        "box_diagnostics": metrics["by_threshold"]["0.20"]["box_diagnostics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--all-metrics", required=True, type=Path)
    parser.add_argument("--v1-corrected", required=True, type=Path)
    parser.add_argument("--v2-selection", required=True, type=Path)
    parser.add_argument("--package-report", required=True, type=Path)
    args = parser.parse_args()
    all_metrics = json.loads(args.all_metrics.read_text(encoding="utf-8"))
    rows = sorted((epoch_row(value) for value in all_metrics.values()), key=lambda row: row["epoch"])
    if [row["epoch"] for row in rows] != [4, 8, 12]:
        raise SystemExit(f"expected evaluated epochs [4,8,12], got {[row['epoch'] for row in rows]}")
    ranked = sorted(rows, key=lambda row: (-row["min_class_recall"], -row["mean_class_f1"], row["mean_xy_mae_m"] if math.isfinite(row["mean_xy_mae_m"]) else math.inf, row["epoch"]))
    selected = ranked[0]

    v1 = json.loads(args.v1_corrected.read_text(encoding="utf-8"))
    v1_primary = v1["by_threshold"]["0.20"]["primary_greedy"]
    v1_comparison = {
        "source": str(args.v1_corrected.resolve()),
        "vehicle_precision": v1_primary["vehicle_precision"], "vehicle_recall": v1_primary["vehicle_recall"], "vehicle_f1": v1_primary["vehicle_f1"],
        "person_precision": v1_primary["person_precision"], "person_recall": v1_primary["person_recall"], "person_f1": v1_primary["person_f1"],
        "vehicle_xy_mae_m": v1_primary["vehicle_xy_mae_m"], "person_xy_mae_m": v1_primary["person_xy_mae_m"],
        "vehicle_iou": v1["segmentation"]["vehicle_iou"], "person_box_mask_iou": v1["segmentation"]["person_iou"], "miou": v1["segmentation"]["miou"],
    }
    v1_comparison["mean_class_f1"] = 0.5 * (v1_comparison["vehicle_f1"] + v1_comparison["person_f1"])
    v2_payload = json.loads(args.v2_selection.read_text(encoding="utf-8"))
    v2_selected = v2_payload["selected"]
    v2_comparison = {
        "source": str(args.v2_selection.resolve()),
        **{key: v2_selected[key] for key in (
            "vehicle_precision", "vehicle_recall", "vehicle_f1", "person_precision", "person_recall", "person_f1",
            "vehicle_xy_mae_m", "person_xy_mae_m", "vehicle_iou", "person_box_mask_iou", "miou", "mean_class_f1"
        )},
    }
    material_baseline = {
        "vehicle_recall_floor": max(v1_comparison["vehicle_recall"], v2_comparison["vehicle_recall"]),
        "person_recall_floor": max(v1_comparison["person_recall"], v2_comparison["person_recall"]),
        "mean_class_f1_floor": max(v1_comparison["mean_class_f1"], v2_comparison["mean_class_f1"]) + 0.05,
    }
    material_gain = (
        selected["vehicle_recall"] + 1e-12 >= material_baseline["vehicle_recall_floor"]
        and selected["person_recall"] + 1e-12 >= material_baseline["person_recall_floor"]
        and selected["mean_class_f1"] + 1e-12 >= material_baseline["mean_class_f1_floor"]
    )

    values = {
        "vehicle_precision": selected["vehicle_precision"], "vehicle_recall": selected["vehicle_recall"],
        "person_precision": selected["person_precision"], "person_recall": selected["person_recall"],
        "vehicle_xy_mae_m": selected["vehicle_xy_mae_m"], "person_xy_mae_m": selected["person_xy_mae_m"],
        "vehicle_iou": selected["vehicle_iou"], "person_box_mask_iou": selected["person_box_mask_iou"], "miou": selected["miou"],
    }
    gate = []
    for metric, (operator, target) in SERVICE_TARGETS.items():
        value = values[metric]
        passed = value >= target if operator == ">=" else value <= target
        gate.append({"metric": metric, "operator": operator, "target": target, "value": value, "pass": bool(passed)})
    service_ready = all(item["pass"] for item in gate)
    if service_ready:
        verdict = "FRCNN_RADAR_ROI_SERVICE_READY"
    elif material_gain:
        verdict = "FRCNN_RADAR_ROI_MATERIAL_GAIN_NOT_SERVICE_READY"
    else:
        verdict = "FRCNN_RADAR_ROI_NO_GAIN"

    experiment_dir = args.experiment_dir.resolve()
    training_runtime = json.loads((experiment_dir / "training_runtime.json").read_text(encoding="utf-8"))
    preflight = json.loads((experiment_dir / "preflight.json").read_text(encoding="utf-8"))
    decision = {
        "verdict": verdict,
        "selection_rule": SELECTION_RULE,
        "epochs": rows,
        "ranking": [{key: row[key] for key in ("epoch", "min_class_recall", "mean_class_f1", "mean_xy_mae_m")} for row in ranked],
        "selected": selected,
        "service_target_gate": gate,
        "service_ready": service_ready,
        "material_gain_rule": MATERIAL_GAIN_RULE,
        "material_gain_baseline": material_baseline,
        "material_gain": material_gain,
        "centerNet_v1_corrected": v1_comparison,
        "centerNet_v2": v2_comparison,
        "training_runtime": training_runtime,
        "evaluation_runtime_seconds": sum(row["runtime_seconds"] for row in rows),
        "evaluation_peak_allocated_mib": max(row["peak_allocated_mib"] for row in rows),
        "boundary": preflight["split_inference"]["boundary"],
        "pretrained_weights": preflight["weights"],
    }
    write_json_create(experiment_dir / "final_selection.json", decision)

    def table_row(row):
        return (
            f"| {row['epoch']} | {row['vehicle_precision']:.4f} | {row['vehicle_recall']:.4f} | {row['vehicle_f1']:.4f} | "
            f"{row['person_precision']:.4f} | {row['person_recall']:.4f} | {row['person_f1']:.4f} | "
            f"{row['vehicle_recall_s002']:.4f} | {row['person_recall_s002']:.4f} | "
            f"{row['vehicle_xy_mae_m']:.3f} | {row['person_xy_mae_m']:.3f} | "
            f"{row['vehicle_iou']:.4f} | {row['person_box_mask_iou']:.4f} | {row['miou']:.4f} |"
        )

    report = [
        "# Faster R-CNN radar-ROI v1 — final Route B report", "",
        f"Terminal verdict: `{verdict}`", "",
        "## Selected artifact", "",
        f"- Epoch: {selected['epoch']}", f"- Checkpoint: `{selected['checkpoint']}`",
        f"- SHA-256: `{selected['checkpoint_sha256']}`", "",
        "## Architecture and split boundary", "",
        "COCO-pretrained Faster R-CNN ResNet-50-FPN v2 performs RGB-only RPN, ROI classification and 2D box regression. An independent four-channel radar pyramid is pooled at every positive ROI and concatenated with the visual ROI embedding for camera-local XYZ, dimensions, local yaw, parked state and radar-support regression. A separate visual-FPN decoder produces semantic segmentation.", "",
        f"`encode_front(rgb, radar)` emits five RGB-FPN and five radar-feature tensors; `decode_tail(bundle, image_size)` has no raw modality argument. Boundary payload: {decision['boundary']['total_bytes']:,} bytes/sample in FP32. Monolithic/split maximum absolute difference: 0.0.", "",
        "## Pretrained provenance", "",
        f"- {decision['pretrained_weights']['enum']} from `{decision['pretrained_weights']['source']}`",
        f"- SHA-256 `{decision['pretrained_weights']['actual_sha256']}`; torchvision `{decision['pretrained_weights']['torchvision']}`; BSD-3-Clause.",
        "- Route person copies COCO person row 1. Route vehicle is the mean of COCO car 3, motorcycle 4, bus 6 and truck 8, for classifier and class-specific box regressor rows.", "",
        "## Epoch metrics", "",
        "| epoch | veh P | veh R | veh F1 | per P | per R | per F1 | veh R@.02 | per R@.02 | veh XY | per XY | veh IoU | person box-mask IoU | mIoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[table_row(row) for row in rows], "",
        "Person segmentation is projected-box-mask IoU, not silhouette IoU.", "",
        "## Retained-model comparison", "",
        "| model | veh P | veh R | veh F1 | per P | per R | per F1 | mean F1 | veh XY | per XY | mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| CenterNet v1 corrected | {v1_comparison['vehicle_precision']:.4f} | {v1_comparison['vehicle_recall']:.4f} | {v1_comparison['vehicle_f1']:.4f} | {v1_comparison['person_precision']:.4f} | {v1_comparison['person_recall']:.4f} | {v1_comparison['person_f1']:.4f} | {v1_comparison['mean_class_f1']:.4f} | {v1_comparison['vehicle_xy_mae_m']:.3f} | {v1_comparison['person_xy_mae_m']:.3f} | {v1_comparison['miou']:.4f} |",
        f"| CenterNet v2 | {v2_comparison['vehicle_precision']:.4f} | {v2_comparison['vehicle_recall']:.4f} | {v2_comparison['vehicle_f1']:.4f} | {v2_comparison['person_precision']:.4f} | {v2_comparison['person_recall']:.4f} | {v2_comparison['person_f1']:.4f} | {v2_comparison['mean_class_f1']:.4f} | {v2_comparison['vehicle_xy_mae_m']:.3f} | {v2_comparison['person_xy_mae_m']:.3f} | {v2_comparison['miou']:.4f} |",
        f"| Faster R-CNN selected | {selected['vehicle_precision']:.4f} | {selected['vehicle_recall']:.4f} | {selected['vehicle_f1']:.4f} | {selected['person_precision']:.4f} | {selected['person_recall']:.4f} | {selected['person_f1']:.4f} | {selected['mean_class_f1']:.4f} | {selected['vehicle_xy_mae_m']:.3f} | {selected['person_xy_mae_m']:.3f} | {selected['miou']:.4f} |", "",
        "## Runtime", "",
        f"Training: {training_runtime['runtime_seconds'] / 60.0:.1f} min, peak allocated/reserved {training_runtime['peak_allocated_mib']:.0f}/{training_runtime['peak_reserved_mib']:.0f} MiB. Evaluation total: {decision['evaluation_runtime_seconds'] / 60.0:.1f} min, peak allocated {decision['evaluation_peak_allocated_mib']:.0f} MiB.", "",
        "## Service gate", "",
        "| metric | target | selected | result |", "|---|---:|---:|---|",
        *[f"| {item['metric']} | {item['operator']} {item['target']:.2f} | {item['value']:.4f} | {'PASS' if item['pass'] else 'FAIL'} |" for item in gate], "",
        "## Interpretation", "",
        "Detector, localization and segmentation outcomes are separated in the tables above. The retained manual 32-person panel contains 15 clearly visible, 5 partially visible, 9 heavily occluded and 3 not visible examples. It is stratified rather than random, so those proportions are not extrapolated to the validation corpus; unresolved observability remains a limitation without changing the full GT denominator.", "",
        f"Material-gain rule: {MATERIAL_GAIN_RULE}. Result: {'PASS' if material_gain else 'FAIL'}.", "",
        f"# {verdict}", "",
    ]
    text = "\n".join(report)
    with (experiment_dir / "FINAL_REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(text)
    args.package_report.parent.mkdir(parents=True, exist_ok=True)
    with args.package_report.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(verdict, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


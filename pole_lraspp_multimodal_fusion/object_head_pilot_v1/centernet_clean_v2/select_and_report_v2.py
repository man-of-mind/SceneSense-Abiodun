#!/usr/bin/env python3
"""CenterNet v2 selection, service-target gate and final report.

Selection rule (fixed, in order):
  1. highest min(vehicle recall, person recall) at score 0.20
  2. highest mean class F1
  3. lowest mean XY MAE
  4. earlier epoch

Improvement rule - REGISTERED BEFORE ANY v2 EVALUATION WAS RUN.  The baseline is
the v1 epoch-12 checkpoint scored through the *corrected* decoder from
CENTERNET_EVALUATION_CONTRACT_AUDIT.md (the fairest v1 number available; the
published v1 decoder was defective).  v2 counts as improved iff either

  (a) min class recall improved AND mean class F1 did not fall by more than 0.01, or
  (b) mean class F1 improved AND min class recall did not fall by more than 0.01.

Service targets are absolute and are not relaxed.  The GT denominator is
unchanged in every table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

V1_BASELINE = {
    "source": (
        "experiments/route_b_centernet_clean_v1/20260826_175224/audit_decoder_contract_v1/"
        "CENTERNET_EVALUATION_CONTRACT_AUDIT.md - epoch 12, corrected (hybrid) decoder, score 0.20"
    ),
    "vehicle_precision": 0.81441,
    "vehicle_recall": 0.57488,
    "vehicle_f1": 0.67399,
    "person_precision": 0.54039,
    "person_recall": 0.48911,
    "person_f1": 0.51347,
    "vehicle_xy_mae_m": 0.9143,
    "person_xy_mae_m": 1.0949,
    "vehicle_recall_s002": 0.72968,
    "person_recall_s002": 0.62303,
    "miou": 0.6888252261434141,
    "vehicle_iou": 0.7192,
    "person_box_mask_iou": 0.3703,
}
V1_BASELINE["min_class_recall"] = min(
    V1_BASELINE["vehicle_recall"], V1_BASELINE["person_recall"]
)
V1_BASELINE["mean_class_f1"] = 0.5 * (V1_BASELINE["vehicle_f1"] + V1_BASELINE["person_f1"])

SERVICE_TARGETS = [
    ("vehicle_precision", ">=", 0.80),
    ("vehicle_recall", ">=", 0.85),
    ("person_precision", ">=", 0.80),
    ("person_recall", ">=", 0.80),
    ("vehicle_xy_mae_m", "<=", 1.0),
    ("person_xy_mae_m", "<=", 1.2),
    ("vehicle_iou", ">=", 0.85),
    ("person_box_mask_iou", ">=", 0.50),
    ("miou", ">=", 0.80),
]


def flatten(result: Dict) -> Dict[str, float]:
    s020 = result["by_threshold"]["0.20"]["primary_greedy"]
    s002 = result["by_threshold"]["0.02"]["primary_greedy"]
    seg = result["segmentation"]
    out = {
        "epoch": int(result["epoch"]),
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "miou": float(seg["miou"]),
        "background_iou": float(seg["background_iou"]),
        "vehicle_iou": float(seg["vehicle_iou"]),
        "person_box_mask_iou": float(seg["person_box_mask_iou"]),
        "pixel_accuracy": float(seg["pixel_accuracy"]),
    }
    for cls in ("vehicle", "person"):
        for metric in ("tp", "fp", "fn", "precision", "recall", "f1", "xy_mae_m", "dimension_mae_m"):
            out[f"{cls}_{metric}"] = s020.get(f"{cls}_{metric}", float("nan"))
        out[f"{cls}_recall_s002"] = s002.get(f"{cls}_recall", float("nan"))
        out[f"{cls}_precision_s002"] = s002.get(f"{cls}_precision", float("nan"))
        out[f"{cls}_tp_s002"] = s002.get(f"{cls}_tp", float("nan"))
        out[f"{cls}_fp_s002"] = s002.get(f"{cls}_fp", float("nan"))
        out[f"{cls}_fn_s002"] = s002.get(f"{cls}_fn", float("nan"))
    out["min_class_recall"] = min(out["vehicle_recall"], out["person_recall"])
    out["mean_class_f1"] = 0.5 * (out["vehicle_f1"] + out["person_f1"])
    out["mean_xy_mae_m"] = 0.5 * (out["vehicle_xy_mae_m"] + out["person_xy_mae_m"])
    return out


def fmt(value, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    if float(f).is_integer() and abs(f) < 1e9 and digits == 0:
        return str(int(f))
    return f"{f:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--eval-root", default="eval")
    args = parser.parse_args()
    exp_dir = args.experiment_dir.resolve()
    eval_root = exp_dir / args.eval_root

    all_metrics = json.loads((eval_root / "all_epochs_metrics.json").read_text(encoding="utf-8"))
    rows = sorted((flatten(v) for v in all_metrics.values()), key=lambda r: r["epoch"])

    ranked = sorted(
        rows,
        key=lambda r: (
            -r["min_class_recall"],
            -r["mean_class_f1"],
            r["mean_xy_mae_m"],
            r["epoch"],
        ),
    )
    selected = ranked[0]

    gate: List[Dict] = []
    all_pass = True
    for metric, op, target in SERVICE_TARGETS:
        value = float(selected[metric])
        ok = value >= target if op == ">=" else value <= target
        all_pass = all_pass and ok
        gate.append(
            {"metric": metric, "operator": op, "target": target, "value": value, "pass": bool(ok)}
        )

    d_recall = selected["min_class_recall"] - V1_BASELINE["min_class_recall"]
    d_f1 = selected["mean_class_f1"] - V1_BASELINE["mean_class_f1"]
    improved = (d_recall > 0.0 and d_f1 >= -0.01) or (d_f1 > 0.0 and d_recall >= -0.01)

    if all_pass:
        verdict = "CENTERNET_V2_SERVICE_READY"
    elif improved:
        verdict = "CENTERNET_V2_IMPROVED_NOT_SERVICE_READY"
    else:
        verdict = "CENTERNET_V2_NO_GAIN"

    binding = [g for g in gate if not g["pass"]]

    # runtime / VRAM from the training metrics table
    train_rows = []
    metrics_dir = exp_dir / "metrics"
    for path in sorted(metrics_dir.glob("*_metrics.csv")):
        with path.open("r", newline="", encoding="utf-8") as fh:
            train_rows = list(csv.DictReader(fh))
    total_seconds = sum(float(r["epoch_seconds"]) for r in train_rows)
    peak_alloc = max(float(r["cuda_max_memory_allocated_mib"]) for r in train_rows)
    peak_reserved = max(float(r["cuda_max_memory_reserved_mib"]) for r in train_rows)

    selection = {
        "verdict": verdict,
        "selected": selected,
        "selection_rule": [
            "1. highest min(vehicle recall, person recall) @0.20",
            "2. highest mean class F1",
            "3. lowest mean XY MAE",
            "4. earlier epoch",
        ],
        "ranking": [
            {k: r[k] for k in ("epoch", "min_class_recall", "mean_class_f1", "mean_xy_mae_m")}
            for r in ranked
        ],
        "service_target_gate": gate,
        "service_targets_all_pass": bool(all_pass),
        "binding_failures": binding,
        "v1_baseline": V1_BASELINE,
        "improvement_rule": (
            "registered before evaluation: improved iff (min class recall up and mean class F1 "
            "not down by more than 0.01) or (mean class F1 up and min class recall not down by "
            "more than 0.01)"
        ),
        "delta_min_class_recall_vs_v1_corrected": d_recall,
        "delta_mean_class_f1_vs_v1_corrected": d_f1,
        "improved": bool(improved),
        "runtime": {
            "training_epochs": len(train_rows),
            "training_wall_clock_seconds": total_seconds,
            "training_wall_clock_hours": total_seconds / 3600.0,
            "peak_cuda_memory_allocated_mib": peak_alloc,
            "peak_cuda_memory_reserved_mib": peak_reserved,
        },
        "epochs": rows,
    }
    (exp_dir / "selection_v2.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (exp_dir / "TERMINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")

    # ------------------------------------------------------------ markdown
    lines: List[str] = []
    lines.append("# CenterNet v2 - per-epoch validation table (frozen native decoder)\n")
    lines.append(
        "| epoch | veh P | veh R | veh F1 | per P | per R | per F1 | veh R@0.02 | per R@0.02 | "
        "veh XY MAE | per XY MAE | veh dim MAE | per dim MAE | veh IoU | person box-mask IoU | mIoU |"
    )
    lines.append("|---" * 16 + "|")
    for r in rows:
        lines.append(
            "| {epoch} | {vp} | {vr} | {vf} | {pp} | {pr} | {pf} | {vr2} | {pr2} | {vx} | {px} | "
            "{vd} | {pd} | {vi} | {pi} | {mi} |".format(
                epoch=r["epoch"],
                vp=fmt(r["vehicle_precision"]), vr=fmt(r["vehicle_recall"]), vf=fmt(r["vehicle_f1"]),
                pp=fmt(r["person_precision"]), pr=fmt(r["person_recall"]), pf=fmt(r["person_f1"]),
                vr2=fmt(r["vehicle_recall_s002"]), pr2=fmt(r["person_recall_s002"]),
                vx=fmt(r["vehicle_xy_mae_m"]), px=fmt(r["person_xy_mae_m"]),
                vd=fmt(r["vehicle_dimension_mae_m"]), pd=fmt(r["person_dimension_mae_m"]),
                vi=fmt(r["vehicle_iou"]), pi=fmt(r["person_box_mask_iou"]), mi=fmt(r["miou"]),
            )
        )
    lines.append("")
    lines.append("| epoch | veh TP | veh FP | veh FN | per TP | per FP | per FN |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['epoch']} | {int(r['vehicle_tp'])} | {int(r['vehicle_fp'])} | "
            f"{int(r['vehicle_fn'])} | {int(r['person_tp'])} | {int(r['person_fp'])} | "
            f"{int(r['person_fn'])} |"
        )
    (exp_dir / "PER_EPOCH_VALIDATION_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "selected_epoch": selected["epoch"],
                      "binding_failures": [b["metric"] for b in binding]}, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

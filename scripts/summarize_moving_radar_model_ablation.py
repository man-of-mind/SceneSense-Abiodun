#!/usr/bin/env python3
"""Summarize moving-ego radar point-rate/geometry model ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-tag", default="", help="Experiment date tag used by the ablation wrapper.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs/radar_model_ablation",
    )
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--cap", type=int, default=2200)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["5000:bbox", "5000:radius", "12000:bbox", "12000:radius"],
        help="Ablation configs as RADAR_PPS:MODE.",
    )
    return parser.parse_args()


def support_tag(mode: str) -> str:
    return "classaware" if mode == "radius" else "bboxsupport"


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_best_training(metrics_csv: Path) -> Dict[str, object]:
    if not metrics_csv.exists():
        return {}
    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best = max(rows, key=lambda row: to_float(row.get("miou"), -math.inf))
    return {
        "best_epoch": best.get("epoch", ""),
        "best_val_miou": to_float(best.get("miou")),
        "best_val_vehicle_iou": to_float(best.get("vehicle_iou")),
        "best_val_person_iou": to_float(best.get("person_iou")),
        "best_selection_score": to_float(best.get("selection_score")),
    }


def collect_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg in args.configs:
        pps_raw, mode = cfg.split(":", 1)
        pps = int(pps_raw)
        tag = support_tag(mode)
        prefix = f"moving_ego_radarpps{pps}_{tag}_{args.loops}loops_cap{args.cap}"
        exp_glob = f"{prefix}_fusion_train_{args.date_tag}" if args.date_tag else f"{prefix}_fusion_train_*"
        matches = sorted((ROOT / "experiments").glob(exp_glob))
        if not matches:
            rows.append({"radar_pps": pps, "support_mode": mode, "status": "missing"})
            continue
        exp = matches[-1]
        trial = f"{prefix}_768x432_lr1e-4_bs2"
        eval_json = exp / "eval_pilot_on_pilot_test/metrics/test_fusion_evaluation_metrics.json"
        train_csv = exp / f"metrics/{trial}_metrics.csv"
        row: Dict[str, object] = {
            "radar_pps": pps,
            "support_mode": mode,
            "support_tag": tag,
            "experiment_dir": str(exp),
            "status": "ok" if eval_json.exists() else "missing_eval",
        }
        if eval_json.exists():
            payload = read_json(eval_json)
            for key in (
                "samples",
                "miou",
                "vehicle_iou",
                "person_iou",
                "pixel_accuracy",
                "learned_object_f1",
                "learned_object_precision",
                "learned_object_recall",
                "learned_global_xy_mae_m",
                "learned_vehicle_object_f1",
                "learned_person_object_f1",
                "learned_vehicle_global_xy_mae_m",
                "learned_person_global_xy_mae_m",
                "device",
                "device_name",
            ):
                row[key] = payload.get(key, "")
        row.update(read_best_training(train_csv))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    labels = [f"{row['radar_pps'] / 1000:.0f}k\n{row['support_mode']}" for row in ok_rows]
    x = np.arange(len(labels))
    fields = [
        ("miou", "mIoU"),
        ("vehicle_iou", "Vehicle IoU"),
        ("person_iou", "Person IoU"),
        ("learned_object_f1", "Localization F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.3), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, fields):
        vals = [to_float(row.get(field)) for row in ok_rows]
        bars = ax.bar(x, vals, color=["#6f8797", "#8eb77f", "#3f6b93", "#2f9e75"][: len(vals)], width=0.62)
        ax.set_xticks(x, labels)
        ax.set_title(title, weight="bold")
        if field != "learned_global_xy_mae_m":
            ax.set_ylim(0, max(1.0, max(vals) * 1.2))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, vals):
            if math.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Moving-Ego Radar Model Ablation", weight="bold", fontsize=16)
    fig.savefig(output_dir / "moving_radar_model_ablation.png", dpi=220)
    fig.savefig(output_dir / "moving_radar_model_ablation.pdf")
    plt.close(fig)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Moving-Ego Radar Model Ablation",
        "",
        "| Radar pps | Person support | Status | mIoU | Vehicle IoU | Person IoU | Loc F1 | XY MAE (m) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pps} | {mode} | {status} | {miou:.3f} | {veh:.3f} | {person:.3f} | {f1:.3f} | {xy:.3f} |".format(
                pps=row.get("radar_pps", ""),
                mode=row.get("support_mode", ""),
                status=row.get("status", ""),
                miou=to_float(row.get("miou")),
                veh=to_float(row.get("vehicle_iou")),
                person=to_float(row.get("person_iou")),
                f1=to_float(row.get("learned_object_f1")),
                xy=to_float(row.get("learned_global_xy_mae_m")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `5k:bbox -> 5k:radius` isolates the geometry/association change.",
            "- `5k:radius -> 12k:radius` isolates radar point-density under the same person geometry.",
            "- `12k:bbox -> 12k:radius` checks whether geometry still matters when radar is denser.",
            "- Compare against the support-level factorial table before deciding whether the model learned to exploit the extra radar evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args)
    write_csv(output_dir / "moving_radar_model_ablation_summary.csv", rows)
    write_markdown(output_dir / "moving_radar_model_ablation_summary.md", rows)
    plot(rows, output_dir)
    print(f"Wrote radar model ablation summary to {output_dir}")


if __name__ == "__main__":
    main()

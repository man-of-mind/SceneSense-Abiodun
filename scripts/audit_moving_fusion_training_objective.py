#!/usr/bin/env python3
"""Audit whether moving-fusion checkpoint selection matches vehicle-IoU goals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis_outputs/moving_ego_fusion_objective_audit"
RUNS = {
    "moving8": {
        "label": "8-loop moving model",
        "csv": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/metrics/moving_fixedroute_8loops_cap6000_768x432_lr1e-4_bs2_metrics.csv",
    },
    "moving12": {
        "label": "12-loop moving model",
        "csv": ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/metrics/moving_fixedroute_12loops_cap9000_768x432_lr1e-4_bs2_metrics.csv",
    },
}
MAXIMIZE = ("selection_score", "miou", "vehicle_iou", "person_iou", "pixel_accuracy")
MINIMIZE = ("val_loss", "loc_loss", "dim_loss")


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def best_row(rows: Sequence[Mapping[str, str]], metric: str, *, maximize: bool) -> Mapping[str, str]:
    valid = [row for row in rows if math.isfinite(to_float(row.get(metric)))]
    if not valid:
        return {}
    key = lambda row: to_float(row.get(metric))
    return max(valid, key=key) if maximize else min(valid, key=key)


def summarize_runs() -> tuple[List[Dict[str, object]], Dict[str, List[Dict[str, str]]]]:
    summary_rows: List[Dict[str, object]] = []
    loaded: Dict[str, List[Dict[str, str]]] = {}
    for run_id, meta in RUNS.items():
        path = Path(meta["csv"])
        if not path.exists():
            continue
        rows = read_csv(path)
        loaded[run_id] = rows
        for metric in MAXIMIZE:
            row = best_row(rows, metric, maximize=True)
            if row:
                summary_rows.append(_summary_row(run_id, meta["label"], metric, row, "max"))
        for metric in MINIMIZE:
            row = best_row(rows, metric, maximize=False)
            if row:
                summary_rows.append(_summary_row(run_id, meta["label"], metric, row, "min"))
    return summary_rows, loaded


def _summary_row(run_id: str, label: str, selected_by: str, row: Mapping[str, str], direction: str) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "label": label,
        "selected_by": selected_by,
        "direction": direction,
        "epoch": row.get("epoch", ""),
        "selected_value": to_float(row.get(selected_by)),
        "selection_score": to_float(row.get("selection_score")),
        "miou": to_float(row.get("miou")),
        "vehicle_iou": to_float(row.get("vehicle_iou")),
        "person_iou": to_float(row.get("person_iou")),
        "pixel_accuracy": to_float(row.get("pixel_accuracy")),
        "val_loss": to_float(row.get("val_loss")),
        "loc_loss": to_float(row.get("loc_loss")),
        "dim_loss": to_float(row.get("dim_loss")),
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(loaded: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path) -> None:
    if not loaded:
        return
    fields = [
        ("selection_score", "Selection score"),
        ("miou", "Validation mIoU"),
        ("vehicle_iou", "Validation vehicle IoU"),
        ("person_iou", "Validation person IoU"),
        ("loc_loss", "Localization loss"),
        ("dim_loss", "Dimension loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), constrained_layout=True)
    colors = {"moving8": "#3b6fb6", "moving12": "#d08a1d"}
    for ax, (field, title) in zip(axes.flat, fields):
        for run_id, rows in loaded.items():
            epochs = [int(float(row.get("epoch", idx))) for idx, row in enumerate(rows)]
            values = [to_float(row.get(field)) for row in rows]
            ax.plot(epochs, values, linewidth=2.0, label=RUNS[run_id]["label"], color=colors.get(run_id))
            finite_pairs = [(epoch, value) for epoch, value in zip(epochs, values) if math.isfinite(value)]
            if finite_pairs:
                best_epoch, best_value = (
                    min(finite_pairs, key=lambda item: item[1])
                    if field.endswith("loss")
                    else max(finite_pairs, key=lambda item: item[1])
                )
                ax.scatter([best_epoch], [best_value], s=38, color=colors.get(run_id), edgecolor="black", zorder=5)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Moving-Ego Fusion Training Objective Audit", weight="bold", fontsize=16)
    fig.savefig(output_dir / "moving_fusion_training_objective_audit.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_training_objective_audit.pdf")
    plt.close(fig)


def fmt(value: object) -> str:
    number = to_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.4f}"


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    by_run_metric = {(row["run_id"], row["selected_by"]): row for row in rows}
    lines = [
        "# Moving-Ego Fusion Training Objective Audit",
        "",
        "## Readout",
        "",
    ]
    for run_id, meta in RUNS.items():
        selected = by_run_metric.get((run_id, "selection_score"))
        vehicle = by_run_metric.get((run_id, "vehicle_iou"))
        miou = by_run_metric.get((run_id, "miou"))
        if not selected:
            continue
        lines.extend(
            [
                f"### {meta['label']}",
                "",
                f"- Saved-selection epoch proxy: epoch `{selected['epoch']}`, selection score `{fmt(selected['selection_score'])}`, vehicle IoU `{fmt(selected['vehicle_iou'])}`, mIoU `{fmt(selected['miou'])}`.",
            ]
        )
        if vehicle:
            lines.append(
                f"- Best vehicle-IoU epoch: epoch `{vehicle['epoch']}`, vehicle IoU `{fmt(vehicle['vehicle_iou'])}`, mIoU `{fmt(vehicle['miou'])}`, selection score `{fmt(vehicle['selection_score'])}`."
            )
        if miou:
            lines.append(
                f"- Best mIoU epoch: epoch `{miou['epoch']}`, mIoU `{fmt(miou['miou'])}`, vehicle IoU `{fmt(miou['vehicle_iou'])}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            "- For the 8-loop model, the selection-score checkpoint is effectively aligned with the best vehicle-IoU epoch. Checkpoint selection is not the main reason vehicle IoU is below 0.90.",
            "- The 12-loop run has lower vehicle IoU even at its best vehicle-IoU epoch, so repeated loops on the same route are not the right standalone fix.",
            "- The next useful levers are objective weighting, lower object-head pressure during segmentation fine-tuning, and/or route/view diversity focused on low and medium density scenes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, loaded = summarize_runs()
    if not rows:
        raise FileNotFoundError("No moving-fusion training CSVs were found.")
    write_rows(output_dir / "moving_fusion_training_objective_audit.csv", rows)
    plot_curves(loaded, output_dir)
    write_markdown(output_dir / "moving_fusion_training_objective_audit.md", rows)
    (output_dir / "moving_fusion_training_objective_audit_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote objective audit to {output_dir}")


if __name__ == "__main__":
    main()

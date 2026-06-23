#!/usr/bin/env python3
"""Summarize moving-ego RGB+radar fusion model evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EVALS: Dict[str, Tuple[str, Path]] = {
    "moving8_on_moving": (
        "Moving model, 8 loops -> moving test",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
    ),
    "moving12_on_moving": (
        "Moving model, 12 loops -> moving test",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/eval_moving_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
    ),
    "moving8_on_viewA": (
        "Moving model, 8 loops -> parked View A",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_parked_viewA/metrics/test_fusion_evaluation_metrics.json",
    ),
    "moving8_on_viewB": (
        "Moving model, 8 loops -> parked View B",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_parked_viewB/metrics/test_fusion_evaluation_metrics.json",
    ),
    "moving8_on_parkedAB": (
        "Moving model, 8 loops -> parked A+B",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260617/eval_moving_model_on_parked_AB/metrics/test_fusion_evaluation_metrics.json",
    ),
    "parkedAB_on_moving12": (
        "Parked A+B model -> moving test",
        ROOT
        / "experiments/moving_ego_tl16_spawn80_fixedroute_speed60_fusion_train_20260618_moredata/eval_parked_AB_model_on_moving/metrics/test_fusion_evaluation_metrics.json",
    ),
}

SUMMARY_FIELDS = [
    "samples",
    "miou",
    "vehicle_iou",
    "person_iou",
    "learned_object_precision",
    "learned_object_recall",
    "learned_object_f1",
    "learned_global_xy_mae_m",
    "learned_vehicle_object_precision",
    "learned_vehicle_object_recall",
    "learned_vehicle_global_xy_mae_m",
    "learned_person_object_precision",
    "learned_person_object_recall",
    "learned_person_global_xy_mae_m",
]


def load_metrics() -> Dict[str, Dict[str, object]]:
    loaded: Dict[str, Dict[str, object]] = {}
    for run_id, (label, path) in EVALS.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_id"] = run_id
        payload["label"] = label
        payload["path"] = str(path)
        loaded[run_id] = payload
    return loaded


def metric(metrics: Dict[str, object], key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(rows: Dict[str, Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "label", *SUMMARY_FIELDS, "metrics_path"])
        writer.writeheader()
        for run_id, payload in rows.items():
            writer.writerow(
                {
                    "run_id": run_id,
                    "label": payload["label"],
                    **{field: payload.get(field, "") for field in SUMMARY_FIELDS},
                    "metrics_path": payload["path"],
                }
            )


def annotate(ax: plt.Axes, bars: Iterable[plt.Rectangle], *, precision: int = 2) -> None:
    for bar in bars:
        value = bar.get_height()
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.{precision}f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def save_moving_model_bars(rows: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    run_ids = [run_id for run_id in ("moving8_on_moving", "moving12_on_moving") if run_id in rows]
    labels = ["8 loops", "12 loops"]
    colors = ["#3b6fb6", "#d08a1d"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)
    fields = [
        ("miou", "3-class mIoU"),
        ("vehicle_iou", "Vehicle IoU"),
        ("person_iou", "Person IoU"),
    ]
    for ax, (field, title) in zip(axes, fields):
        values = [metric(rows[run_id], field) for run_id in run_ids]
        bars = ax.bar(labels[: len(values)], values, color=colors[: len(values)], width=0.58)
        annotate(ax, bars)
        ax.set_title(title, weight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Moving-Ego RGB+Radar Fusion: Extra Loops Did Not Improve Segmentation", weight="bold", fontsize=15)
    fig.savefig(output_dir / "moving_fusion_segmentation_8_vs_12loops.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_segmentation_8_vs_12loops.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.8), constrained_layout=True)
    fields = [
        ("learned_object_precision", "Localization precision", 1.0),
        ("learned_object_recall", "Localization recall", 1.0),
        ("learned_object_f1", "Localization F1", 1.0),
        ("learned_global_xy_mae_m", "XY error (m)", None),
    ]
    for ax, (field, title, ymax) in zip(axes, fields):
        values = [metric(rows[run_id], field) for run_id in run_ids]
        bars = ax.bar(labels[: len(values)], values, color=colors[: len(values)], width=0.58)
        annotate(ax, bars)
        ax.set_title(title, weight="bold")
        if ymax is not None:
            ax.set_ylim(0, ymax)
        else:
            finite = [value for value in values if np.isfinite(value)]
            ax.set_ylim(0, max(finite) * 1.25 if finite else 1.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Moving-Ego RGB+Radar Fusion: Localization Metrics", weight="bold", fontsize=15)
    fig.savefig(output_dir / "moving_fusion_localization_8_vs_12loops.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_localization_8_vs_12loops.pdf")
    plt.close(fig)


def save_domain_gap_bars(rows: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    run_ids = [
        run_id
        for run_id in (
            "moving8_on_moving",
            "moving8_on_viewA",
            "moving8_on_viewB",
            "moving8_on_parkedAB",
            "parkedAB_on_moving12",
        )
        if run_id in rows
    ]
    if len(run_ids) < 2:
        return
    labels = [
        "Moving\non moving",
        "Moving\non View A",
        "Moving\non View B",
        "Moving\non parked A+B",
        "Parked A+B\non moving",
    ][: len(run_ids)]
    colors = ["#2f9e75", "#7895c9", "#7895c9", "#7895c9", "#c76c5b"][: len(run_ids)]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=True)
    fields = [
        ("miou", "3-class mIoU"),
        ("vehicle_iou", "Vehicle IoU"),
        ("person_iou", "Person IoU"),
    ]
    for ax, (field, title) in zip(axes, fields):
        values = [metric(rows[run_id], field) for run_id in run_ids]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        annotate(ax, bars)
        ax.set_title(title, weight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("RGB+Radar Fusion Domain Gap: Moving and Parked Views Need Different Coverage", weight="bold", fontsize=15)
    fig.savefig(output_dir / "moving_fusion_domain_gap_segmentation.png", dpi=220)
    fig.savefig(output_dir / "moving_fusion_domain_gap_segmentation.pdf")
    plt.close(fig)


def write_summary(rows: Dict[str, Dict[str, object]], output: Path) -> None:
    def fmt(run_id: str, field: str) -> str:
        if run_id not in rows:
            return "n/a"
        value = metric(rows[run_id], field)
        return "n/a" if not np.isfinite(value) else f"{value:.3f}"

    lines: List[str] = [
        "# Moving-Ego Fusion Evaluation Summary",
        "",
        "## Main Readout",
        "",
        "- The 8-loop moving model is currently the stronger moving-domain checkpoint: "
        f"mIoU={fmt('moving8_on_moving', 'miou')}, vehicle IoU={fmt('moving8_on_moving', 'vehicle_iou')}, "
        f"person IoU={fmt('moving8_on_moving', 'person_iou')}.",
        "- The 12-loop/more-data run did not improve segmentation on its own: "
        f"mIoU={fmt('moving12_on_moving', 'miou')}, vehicle IoU={fmt('moving12_on_moving', 'vehicle_iou')}, "
        f"person IoU={fmt('moving12_on_moving', 'person_iou')}.",
        "- Localization improved slightly with the 12-loop run, but remains weak enough that it should be treated as an engineering target, not a solved metric: "
        f"F1 {fmt('moving8_on_moving', 'learned_object_f1')} -> {fmt('moving12_on_moving', 'learned_object_f1')}, "
        f"XY error {fmt('moving8_on_moving', 'learned_global_xy_mae_m')} m -> {fmt('moving12_on_moving', 'learned_global_xy_mae_m')} m.",
        "- Parked A+B model performance on moving data remains a negative control, not the main path forward: "
        f"mIoU={fmt('parkedAB_on_moving12', 'miou')}, vehicle IoU={fmt('parkedAB_on_moving12', 'vehicle_iou')}.",
        "",
        "## Next Model Direction",
        "",
        "- Keep the moving model as the main candidate for moving-domain work.",
        "- Do not assume more repeated loops are enough; the next improvement should add route/viewpoint diversity, sensor-processing improvements, or training loss/threshold tuning.",
        "- Evaluate the moving model on parked View A/B only as a domain-gap diagnostic, not as the success criterion.",
        "",
        "## Generated Artifacts",
        "",
        "- `moving_fusion_segmentation_8_vs_12loops.png`",
        "- `moving_fusion_localization_8_vs_12loops.png`",
        "- `moving_fusion_domain_gap_segmentation.png`",
        "- `moving_fusion_eval_summary.csv`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs/moving_ego_fusion_model_eval",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    if not rows:
        raise FileNotFoundError("No moving fusion evaluation metrics were found.")
    write_csv(rows, output_dir / "moving_fusion_eval_summary.csv")
    save_moving_model_bars(rows, output_dir)
    save_domain_gap_bars(rows, output_dir)
    write_summary(rows, output_dir / "moving_fusion_eval_summary.md")
    print(f"Wrote moving fusion evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()

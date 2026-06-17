#!/usr/bin/env python3
"""Create presentation plots for parked-ego fusion viewpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EVALS = {
    ("View A model", "View A"): ROOT
    / "experiments/parked_ego_tl16_right7_fusion_train_20260612/eval_A_model_on_viewA/metrics/test_fusion_evaluation_metrics.json",
    ("View A model", "View B"): ROOT
    / "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_A_model_on_viewB/metrics/test_fusion_evaluation_metrics.json",
    ("View B model", "View A"): ROOT
    / "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_B_model_on_viewA/metrics/test_fusion_evaluation_metrics.json",
    ("View B model", "View B"): ROOT
    / "experiments/parked_ego_tl16_viewB_fusion_train_20260612/eval_B_model_on_viewB/metrics/test_fusion_evaluation_metrics.json",
    ("Views A+B model", "View A"): ROOT
    / "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewA/metrics/test_fusion_evaluation_metrics.json",
    ("Views A+B model", "View B"): ROOT
    / "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewB/metrics/test_fusion_evaluation_metrics.json",
    ("Views A+B model", "Combined"): ROOT
    / "experiments/parked_ego_tl16_viewAB_fusion_train_20260612/eval_AB_model_on_viewAB/metrics/test_fusion_evaluation_metrics.json",
}

MODELS = ["View A model", "View B model", "Views A+B model"]
VIEWS = ["View A", "View B"]
MODEL_LABELS = {
    "View A model": "Train: View A only",
    "View B model": "Train: View B only",
    "Views A+B model": "Train: Views A+B",
}
VIEW_LABELS = {
    "View A": "Test: View A",
    "View B": "Test: View B",
    "Combined": "Test: A+B combined",
}
MODEL_COLORS = {
    "View A model": "#3b6fb6",
    "View B model": "#2f9e75",
    "Views A+B model": "#d08a1d",
}


def load_metrics() -> Dict[Tuple[str, str], Dict[str, float]]:
    metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    missing = []
    for key, path in EVALS.items():
        if not path.exists():
            missing.append(path)
            continue
        metrics[key] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        formatted = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing evaluation metrics:\n{formatted}")
    return metrics


def matrix(metrics: Dict[Tuple[str, str], Dict[str, float]], field: str) -> np.ndarray:
    values = np.full((len(MODELS), len(VIEWS)), np.nan, dtype=np.float64)
    for i, model in enumerate(MODELS):
        for j, view in enumerate(VIEWS):
            values[i, j] = float(metrics[(model, view)].get(field, np.nan))
    return values


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                label = "n/a"
            else:
                label = format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", color="white" if value < np.nanmean(values) else "#111111", fontsize=11)


def save_heatmap(
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    output: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".3f",
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(VIEWS)), VIEWS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_xlabel("Evaluation viewpoint")
    ax.set_ylabel("Training data")
    ax.set_title(title, fontsize=14, weight="bold")
    annotate_heatmap(ax, values, fmt)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_grouped_bars(metrics: Dict[Tuple[str, str], Dict[str, float]], output: Path) -> None:
    fields = [
        ("miou", "3-class mIoU", "Higher is better"),
        ("vehicle_iou", "Vehicle IoU", "Higher is better"),
        ("learned_object_f1", "Localization F1", "Higher is better"),
        ("learned_global_xy_mae_m", "XY error (m)", "Lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.6))
    x = np.arange(len(VIEWS))
    width = 0.22
    for ax, (field, label, subtitle) in zip(axes.flat, fields):
        for idx, model in enumerate(MODELS):
            vals = [float(metrics[(model, view)].get(field, np.nan)) for view in VIEWS]
            ax.bar(
                x + (idx - 1) * width,
                vals,
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            for xpos, val in zip(x + (idx - 1) * width, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [VIEW_LABELS[view] for view in VIEWS])
        ax.set_title(label, weight="bold", fontsize=14, pad=12)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=10)
        if field == "learned_global_xy_mae_m":
            ax.set_ylim(0, max(float(metrics[(model, view)].get(field, 0)) for model in MODELS for view in VIEWS) * 1.18)
        else:
            ax.set_ylim(0, 1.0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        title="Model training data",
        title_fontsize=11,
        fontsize=11,
    )
    fig.suptitle("Parked-Ego RGB+Radar Fusion: Viewpoint Generalization", fontsize=18, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.07, right=0.98, hspace=0.46, wspace=0.22)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_metric_bars(
    metrics: Dict[Tuple[str, str], Dict[str, float]],
    output: Path,
    *,
    fields: Sequence[Tuple[str, str, str]],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(7.2 * len(fields), 5.3), squeeze=False)
    x = np.arange(len(VIEWS))
    width = 0.22
    for ax, (field, label, subtitle) in zip(axes.flat, fields):
        for idx, model in enumerate(MODELS):
            vals = [float(metrics[(model, view)].get(field, np.nan)) for view in VIEWS]
            positions = x + (idx - 1) * width
            ax.bar(positions, vals, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [VIEW_LABELS[view] for view in VIEWS])
        ax.set_title(label, weight="bold", fontsize=14, pad=12)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#555555")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        if field == "learned_global_xy_mae_m":
            ax.set_ylim(0, max(float(metrics[(model, view)].get(field, 0)) for model in MODELS for view in VIEWS) * 1.18)
        else:
            ax.set_ylim(0, 1.0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        title="Model training data",
        title_fontsize=11,
        fontsize=11,
    )
    fig.suptitle(title, fontsize=17, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.75, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_precision_recall_bars(metrics: Dict[Tuple[str, str], Dict[str, float]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
    fields = (
        ("learned_object_precision", "Localization Precision", "Correct predictions / all predictions"),
        ("learned_object_recall", "Localization Recall", "Correct predictions / all GT objects"),
    )
    x = np.arange(len(VIEWS))
    width = 0.22
    for ax, (field, label, subtitle) in zip(axes.flat, fields):
        for idx, model in enumerate(MODELS):
            vals = [float(metrics[(model, view)].get(field, np.nan)) for view in VIEWS]
            positions = x + (idx - 1) * width
            ax.bar(positions, vals, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [VIEW_LABELS[view] for view in VIEWS])
        ax.set_ylim(0, 0.62)
        ax.set_title(label, weight="bold", fontsize=14, pad=22)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=10, color="#555555")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        frameon=False,
        title="Model training data",
        title_fontsize=11,
        fontsize=11,
    )
    fig.suptitle("Localization Precision and Recall Across Viewpoints", fontsize=17, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.69, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _class_metric_value(data: Dict[str, float], cls: str, metric: str) -> float:
    if metric == "precision":
        return float(data.get(f"learned_{cls}_object_precision", np.nan))
    if metric == "recall":
        return float(data.get(f"learned_{cls}_object_recall", np.nan))
    if metric == "f1":
        precision = float(data.get(f"learned_{cls}_object_precision", 0.0))
        recall = float(data.get(f"learned_{cls}_object_recall", 0.0))
        return _f1(precision, recall)
    if metric == "xy":
        return float(data.get(f"learned_{cls}_global_xy_mae_m", np.nan))
    raise KeyError(metric)


def save_class_localization_bars(metrics: Dict[Tuple[str, str], Dict[str, float]], output: Path) -> None:
    fields = (
        ("vehicle", "f1", "Vehicle localization F1", "Higher is better"),
        ("person", "f1", "Person localization F1", "Higher is better"),
        ("vehicle", "xy", "Vehicle XY error (m)", "Lower is better"),
        ("person", "xy", "Person XY error (m)", "Lower is better"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    x = np.arange(len(VIEWS))
    width = 0.22
    for ax, (cls, metric_name, label, subtitle) in zip(axes.flat, fields):
        max_val = 0.0
        for idx, model in enumerate(MODELS):
            vals = [_class_metric_value(metrics[(model, view)], cls, metric_name) for view in VIEWS]
            max_val = max(max_val, *[v for v in vals if not np.isnan(v)])
            positions = x + (idx - 1) * width
            ax.bar(positions, vals, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [VIEW_LABELS[view] for view in VIEWS])
        ax.set_title(label, weight="bold", fontsize=14, pad=12)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#555555")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        if metric_name == "xy":
            ax.set_ylim(0, max(0.1, max_val * 1.18))
        else:
            ax.set_ylim(0, 0.62)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        title="Model training data",
        title_fontsize=11,
        fontsize=11,
    )
    fig.suptitle("Vehicle vs Person Localization Across Viewpoints", fontsize=17, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.07, right=0.98, hspace=0.46, wspace=0.22)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_class_precision_recall_bars(metrics: Dict[Tuple[str, str], Dict[str, float]], output: Path) -> None:
    fields = (
        ("vehicle", "precision", "Vehicle precision", "Correct vehicle predictions / all vehicle predictions"),
        ("vehicle", "recall", "Vehicle recall", "Correct vehicle predictions / all GT vehicles"),
        ("person", "precision", "Person precision", "Correct person predictions / all person predictions"),
        ("person", "recall", "Person recall", "Correct person predictions / all GT persons"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    x = np.arange(len(VIEWS))
    width = 0.22
    for ax, (cls, metric_name, label, subtitle) in zip(axes.flat, fields):
        for idx, model in enumerate(MODELS):
            vals = [_class_metric_value(metrics[(model, view)], cls, metric_name) for view in VIEWS]
            positions = x + (idx - 1) * width
            ax.bar(positions, vals, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [VIEW_LABELS[view] for view in VIEWS])
        ax.set_ylim(0, 0.62)
        ax.set_title(label, weight="bold", fontsize=14, pad=14)
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color="#555555")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        title="Model training data",
        title_fontsize=11,
        fontsize=11,
    )
    fig.suptitle("Class-Specific Localization Precision and Recall", fontsize=17, weight="bold", y=0.985)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.07, right=0.98, hspace=0.46, wspace=0.22)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_compact_summary(metrics: Dict[Tuple[str, str], Dict[str, float]], output: Path) -> None:
    fields = [
        "model",
        "eval_view",
        "samples",
        "miou",
        "vehicle_iou",
        "person_iou",
        "pixel_accuracy",
        "learned_object_precision",
        "learned_object_recall",
        "learned_object_f1",
        "learned_global_xy_mae_m",
        "learned_vehicle_global_xy_mae_m",
        "learned_person_global_xy_mae_m",
        "learned_vehicle_object_precision",
        "learned_vehicle_object_recall",
        "learned_person_object_precision",
        "learned_person_object_recall",
    ]
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            for view in VIEWS:
                data = metrics[(model, view)]
                row = {"model": model, "eval_view": view}
                for field in fields[2:]:
                    row[field] = data.get(field, "")
                writer.writerow(row)
        data = metrics[("Views A+B model", "Combined")]
        row = {"model": "Views A+B model", "eval_view": "Combined"}
        for field in fields[2:]:
            row[field] = data.get(field, "")
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis_outputs/parked_ego_fusion_viewpoint_eval",
        help="Directory for generated figures and summary CSV.",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    save_heatmap(
        matrix(metrics, "miou"),
        title="Segmentation mIoU Across Viewpoints",
        cbar_label="3-class mIoU",
        output=out / "fusion_viewpoint_miou_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    save_heatmap(
        matrix(metrics, "vehicle_iou"),
        title="Vehicle IoU Across Viewpoints",
        cbar_label="Vehicle IoU",
        output=out / "fusion_viewpoint_vehicle_iou_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    save_heatmap(
        matrix(metrics, "learned_object_f1"),
        title="Localization F1 Across Viewpoints",
        cbar_label="F1",
        output=out / "fusion_viewpoint_localization_f1_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=0.55,
    )
    save_heatmap(
        matrix(metrics, "learned_global_xy_mae_m"),
        title="Localization XY Error Across Viewpoints",
        cbar_label="Mean XY error (m)",
        output=out / "fusion_viewpoint_xy_error_matrix.png",
        cmap="magma_r",
        vmin=1.0,
        vmax=1.8,
        fmt=".2f",
    )
    save_grouped_bars(metrics, out / "fusion_viewpoint_metric_bars.png")
    save_metric_bars(
        metrics,
        out / "fusion_viewpoint_segmentation_bars.png",
        fields=(
            ("miou", "3-class mIoU", "Higher is better"),
            ("vehicle_iou", "Vehicle IoU", "Higher is better"),
        ),
        title="Segmentation Accuracy Across Parked-Ego Viewpoints",
    )
    save_metric_bars(
        metrics,
        out / "fusion_viewpoint_localization_bars.png",
        fields=(
            ("learned_object_f1", "Localization F1", "Higher is better"),
            ("learned_global_xy_mae_m", "XY error (m)", "Lower is better"),
        ),
        title="Object Localization Across Parked-Ego Viewpoints",
    )
    save_precision_recall_bars(metrics, out / "fusion_viewpoint_localization_precision_recall_bars.png")
    save_class_localization_bars(metrics, out / "fusion_viewpoint_class_localization_bars.png")
    save_class_precision_recall_bars(metrics, out / "fusion_viewpoint_class_precision_recall_bars.png")
    save_compact_summary(metrics, out / "fusion_viewpoint_eval_compact_summary.csv")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

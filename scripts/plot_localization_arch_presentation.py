#!/usr/bin/env python3
"""Create presentation plots for localization-head/backbone architecture runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("analysis_outputs/localization_arch_presentation_20260625")


RUNS = [
    {
        "label": "Previous\nhead-only",
        "short": "Baseline",
        "object_f1": 0.345,
        "person_f1": 0.37,
        "vehicle_f1": 0.33,
        "person_seg_iou": 0.758,
        "vehicle_seg_iou": 0.916,
        "person_xy_mae": 2.16,
        "note": "Frozen backbone, reused object head",
    },
    {
        "label": "New head\nvariant",
        "short": "A",
        "object_f1": 0.254,
        "person_f1": 0.269,
        "vehicle_f1": 0.233,
        "person_seg_iou": 0.758,
        "vehicle_seg_iou": 0.916,
        "person_xy_mae": 2.16,
        "note": "Frozen backbone, new deeper head",
    },
    {
        "label": "Simple head\nfrom scratch",
        "short": "B",
        "object_f1": 0.302,
        "person_f1": 0.304,
        "vehicle_f1": 0.299,
        "person_seg_iou": 0.758,
        "vehicle_seg_iou": 0.916,
        "person_xy_mae": 2.32,
        "note": "Frozen backbone, new simple head",
    },
    {
        "label": "Joint training\nshared features",
        "short": "C",
        "object_f1": 0.387,
        "person_f1": 0.410,
        "vehicle_f1": 0.363,
        "person_seg_iou": 0.692,
        "vehicle_seg_iou": 0.913,
        "person_xy_mae": 2.04,
        "note": "Backbone adapts, seg weight 1",
    },
    {
        "label": "Joint training\nextra SEG weight",
        "short": "D",
        "object_f1": 0.382,
        "person_f1": 0.411,
        "vehicle_f1": 0.351,
        "person_seg_iou": 0.692,
        "vehicle_seg_iou": 0.913,
        "person_xy_mae": 2.08,
        "note": "Backbone adapts, seg weight 2",
    },
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_localization_f1() -> None:
    labels = [row["label"] for row in RUNS]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    ax.bar(x - width, [r["object_f1"] for r in RUNS], width, label="Overall object F1", color="#2454a6")
    ax.bar(x, [r["person_f1"] for r in RUNS], width, label="Person F1", color="#d65a31")
    ax.bar(x + width, [r["vehicle_f1"] for r in RUNS], width, label="Vehicle F1", color="#4f9d69")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Localization F1")
    ax.set_title("Localization Improved Only When Shared Features Were Allowed to Adapt")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3, frameon=False)
    for idx, row in enumerate(RUNS):
        ax.text(idx, 0.485, row["short"], ha="center", va="top", color="#555555", fontsize=9)
    fig.subplots_adjust(bottom=0.28)
    _save(fig, "localization_f1_architecture_comparison")


def plot_seg_localization_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    colors = ["#6f7785", "#2454a6", "#5470c6", "#d65a31", "#b8452e"]
    for row, color in zip(RUNS, colors):
        ax.scatter(row["person_seg_iou"], row["object_f1"], s=145, color=color, edgecolor="white", linewidth=1.5)
        ax.annotate(row["short"], (row["person_seg_iou"], row["object_f1"]), xytext=(7, 7), textcoords="offset points", weight="bold")
    ax.axvline(0.758, color="#6f7785", linestyle="--", linewidth=1.2, alpha=0.8, label="Best SEG-only person IoU")
    ax.set_xlim(0.66, 0.78)
    ax.set_ylim(0.22, 0.42)
    ax.set_xlabel("Person segmentation IoU")
    ax.set_ylabel("Object localization F1")
    ax.set_title("Shared Backbone Tradeoff: Localization Gain vs Person SEG Loss")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "segmentation_localization_tradeoff")


def plot_person_xy() -> None:
    labels = [row["label"] for row in RUNS]
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    bars = ax.bar(labels, [r["person_xy_mae"] for r in RUNS], color=["#6f7785", "#2454a6", "#5470c6", "#d65a31", "#b8452e"])
    ax.set_ylim(1.8, 2.45)
    ax.set_ylabel("Person XY MAE (m)")
    ax.set_title("New Head Variants Did Not Reduce Person XY Error")
    for bar, row in zip(bars, RUNS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{row['person_xy_mae']:.2f}m", ha="center", va="bottom")
    _save(fig, "person_xy_error_architecture_comparison")


def main() -> None:
    _style()
    plot_localization_f1()
    plot_seg_localization_tradeoff()
    plot_person_xy()
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()

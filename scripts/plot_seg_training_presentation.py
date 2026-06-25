#!/usr/bin/env python3
"""Build presentation-ready plots for the SceneSense SEG fusion breakthrough."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis_outputs/seg_training_presentation_20260624"

BASELINE_SUMMARY = ROOT / "analysis_outputs/r100k_segonly_ablation/r100k_r4_tw2_person_focus_summary.csv"
LOVASZ05_SUMMARY = ROOT / "analysis_outputs/r100k_segonly_lovasz05_cosine_bs24/r100k_r4_tw2_person_focus_summary.csv"
FINAL_SUMMARY = ROOT / "analysis_outputs/r100k_segonly_lovasz05_personselect_pat20_bs24/r100k_r4_tw2_person_focus_summary.csv"
FINAL_EXP = ROOT / "experiments/moving_ego_radarpps100000_bboxsupport_r4_tw2_segonly_ablation_20260624_segonly_lovasz05_personselect_pat20_bs24"
FINAL_TRIAL = "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20"
RADAR_PPS_SUMMARY = ROOT / "radar_pedestrian_diagnostic_runs/radar_ped_full_pps5k_to100k_dist2_to100_physics_20260623/summary_by_condition.csv"
RASTER_PLOT = ROOT / "analysis_outputs/radar_rasterization_sweep/moving_12k_bboxsupport_2loops_radii0_1_2_3_4_5_7/radar_rasterization_tradeoff_clean.png"
TEMPORAL_PLOT = ROOT / "analysis_outputs/radar_temporal_accumulation_sweep/moving_12k_bboxsupport_2loops_radius4_windows1_2_3_4_5/radar_temporal_accumulation_presentation_clean.png"


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: Sequence[Mapping[str, str]], run: str, density: str = "overall") -> Mapping[str, str]:
    for row in rows:
        if row.get("run") == run and row.get("density") == density:
            return row
    raise KeyError(f"Missing run={run!r} density={density!r}")


def save_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_progression(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    metrics = [
        ("miou", "mIoU", "#4567a9"),
        ("vehicle_iou", "Vehicle IoU", "#2e8b57"),
        ("person_iou", "Person IoU", "#c64e4e"),
    ]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.23
    fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=True)
    for idx, (field, label, color) in enumerate(metrics):
        values = [to_float(row[field]) for row in rows]
        positions = x + (idx - 1) * width
        bars = ax.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, [str(row["label"]) for row in rows], rotation=0)
    ax.set_ylabel("IoU")
    ax.set_ylim(0.56, 0.95)
    ax.set_title("RGB+Radar SEG Model Progression at Same 768x432 Split Input", weight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    style_axis(ax)
    fig.savefig(out_dir / "seg_model_progression_iou.png", dpi=240)
    fig.savefig(out_dir / "seg_model_progression_iou.pdf")
    plt.close(fig)


def plot_person_journey(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    values = [to_float(row["person_iou"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#7b7b7b", "#3b82c4", "#756bb1", "#c64e4e"], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", fontsize=10)
    ax.axhline(0.70, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(len(labels) - 0.25, 0.705, "0.70 target", ha="right", va="bottom", fontsize=9, color="#333333")
    ax.set_ylim(0.58, 0.80)
    ax.set_ylabel("Person IoU")
    ax.set_title("Person Segmentation Improvement Without Increasing Payload", weight="bold")
    style_axis(ax)
    fig.savefig(out_dir / "person_iou_breakthrough.png", dpi=240)
    fig.savefig(out_dir / "person_iou_breakthrough.pdf")
    plt.close(fig)


def plot_final_density(summary_rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    final_run = "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20"
    densities = ["low", "medium", "crowded", "overall"]
    rows = [find_row(summary_rows, final_run, density) for density in densities]
    metrics = [
        ("miou", "mIoU", "#4567a9"),
        ("vehicle_iou", "Vehicle IoU", "#2e8b57"),
        ("person_iou", "Person IoU", "#c64e4e"),
    ]
    x = np.arange(len(densities), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 6.1), constrained_layout=True)
    for idx, (field, label, color) in enumerate(metrics):
        values = [to_float(row[field]) for row in rows]
        positions = x + (idx - 1) * width
        bars = ax.bar(positions, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, [density.capitalize() for density in densities])
    ax.set_ylim(0.72, 0.96)
    ax.set_ylabel("IoU")
    ax.set_title("Final SEG Fusion Model Performance by Scene Density", weight="bold")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    style_axis(ax)
    fig.savefig(out_dir / "final_seg_iou_by_density.png", dpi=240)
    fig.savefig(out_dir / "final_seg_iou_by_density.pdf")
    plt.close(fig)


def plot_radar_pps(out_dir: Path) -> None:
    rows = read_rows(RADAR_PPS_SUMMARY)
    physics = "default"
    pps_values = [5000, 12000, 48000, 100000]
    colors = ["#777777", "#4c78a8", "#f58518", "#c64e4e"]
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    for pps, color in zip(pps_values, colors):
        selected = [
            row for row in rows
            if row.get("walker_physics") == physics
            and int(float(row.get("radar_pps", "0"))) == pps
            and to_float(row.get("target_distance_m")) >= 10.0
        ]
        selected.sort(key=lambda row: to_float(row.get("target_distance_m")))
        x = [to_float(row["target_distance_m"]) for row in selected]
        y = [100.0 * to_float(row["bbox_support_rate"]) for row in selected]
        ax.plot(x, y, marker="o", linewidth=2.2, label=f"{pps // 1000}k PPS", color=color)
    ax.set_xlim(10, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Pedestrian distance from radar (m)")
    ax.set_ylabel("Frames with pedestrian radar support (%)")
    ax.set_title("Radar PPS Improves Pedestrian Support at Distance", weight="bold")
    ax.legend(frameon=False, loc="lower left")
    style_axis(ax)
    fig.savefig(out_dir / "radar_pps_person_support_distance_clean.png", dpi=240)
    fig.savefig(out_dir / "radar_pps_person_support_distance_clean.pdf")
    plt.close(fig)


def copy_supporting_plots(out_dir: Path) -> None:
    copies = {
        RASTER_PLOT: "radar_raster_radius_tradeoff.png",
        TEMPORAL_PLOT: "radar_temporal_window_tradeoff.png",
        FINAL_EXP / "figures" / f"{FINAL_TRIAL}_training_curves.png": "final_seg_training_curves.png",
    }
    for src, dst_name in copies.items():
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def write_story(out_dir: Path, progression_rows: Sequence[Mapping[str, object]]) -> None:
    final_metrics = progression_rows[-1]
    checkpoint = FINAL_EXP / "checkpoints" / FINAL_TRIAL / "best.pt"
    lines = [
        "# SceneSense SEG Fusion Presentation Story - 2026-06-24",
        "",
        "## Core Message",
        "",
        "- Radar preprocessing improved pedestrian support, but the main SEG accuracy bottleneck was training architecture/objective.",
        "- Removing the localization head during SEG training, fixing augmentation, and selecting for person-aware IoU produced a strong 7-channel RGB+radar fusion SEG model.",
        "- The final model still uses `768x432`, so the accuracy gain did not come from increasing split-inference payload.",
        "",
        "## Recommended Slide Flow",
        "",
        "1. **Radar diagnostic:** PPS, raster radius, and short temporal accumulation increase pedestrian radar support.",
        "2. **Model bottleneck:** Multitask SEG+localization was suppressing segmentation; seg-only training revealed the ceiling was higher.",
        "3. **Training fix:** strong augmentation, BN freeze, larger batch, Lovasz loss, cosine schedule, and person-aware checkpoint selection.",
        "4. **Final SEG result:** strong vehicle and person segmentation at the same input resolution.",
        "5. **Next step:** Stage-2 localization head training on top of the protected SEG checkpoint.",
        "",
        "## Presentation Plots",
        "",
        "- `radar_pps_person_support_distance_clean.png` - PPS improves pedestrian support at distance.",
        "- `radar_raster_radius_tradeoff.png` - radius 4 gives a good support/spillover tradeoff.",
        "- `radar_temporal_window_tradeoff.png` - 2-frame accumulation improves support with limited staleness.",
        "- `seg_model_progression_iou.png` - main SEG breakthrough story.",
        "- `person_iou_breakthrough.png` - person IoU crossed the 0.70 target without payload increase.",
        "- `final_seg_iou_by_density.png` - final model by low/medium/crowded density.",
        "- `final_seg_training_curves.png` - optional evidence slide.",
        "",
        "## Final Model",
        "",
        f"- Best checkpoint: `{checkpoint}`",
        f"- Overall mIoU: `{to_float(final_metrics['miou']):.3f}`",
        f"- Vehicle IoU: `{to_float(final_metrics['vehicle_iou']):.3f}`",
        f"- Person IoU: `{to_float(final_metrics['person_iou']):.3f}`",
        "",
        "## Talking Points",
        "",
        "- The radar-side work was not wasted: it told us how to make pedestrian radar support denser and more stable.",
        "- But the largest jump came when we stopped forcing segmentation and localization to share gradients during the first stage.",
        "- The current best checkpoint is a SEG-first fusion model. Stage 2 should train localization as a head on top of this stable representation.",
        "- Object/localization F1 in these SEG-only runs is intentionally not meaningful because `object_total=0`.",
    ]
    (out_dir / "SEG_FUSION_PRESENTATION_STORY_20260624.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_rows(BASELINE_SUMMARY)
    lovasz05_rows = read_rows(LOVASZ05_SUMMARY)
    final_rows = read_rows(FINAL_SUMMARY)

    progression = [
        {
            "label": "Multitask\nbaseline",
            "run": "r100k_r4_tw2_base",
            **{k: to_float(find_row(baseline_rows, "r100k_r4_tw2_base")[k]) for k in ("miou", "vehicle_iou", "person_iou")},
        },
        {
            "label": "SEG-only\nfixed aug",
            "run": "segonly_stronggeo_bnfreeze_bs8_miou",
            **{k: to_float(find_row(baseline_rows, "segonly_stronggeo_bnfreeze_bs8_miou")[k]) for k in ("miou", "vehicle_iou", "person_iou")},
        },
        {
            "label": "Lovasz 0.5\ncosine",
            "run": "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_miou",
            **{k: to_float(find_row(lovasz05_rows, "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_miou")[k]) for k in ("miou", "vehicle_iou", "person_iou")},
        },
        {
            "label": "Person-select\nfull cosine",
            "run": "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20",
            **{k: to_float(find_row(final_rows, "segonly_stronggeo_bnfreeze_bs24_lovasz05_cosine_personmiou_pat20")[k]) for k in ("miou", "vehicle_iou", "person_iou")},
        },
    ]

    save_csv(OUTPUT_DIR / "seg_model_progression_summary.csv", progression)
    plot_progression(progression, OUTPUT_DIR)
    plot_person_journey(progression, OUTPUT_DIR)
    plot_final_density(final_rows, OUTPUT_DIR)
    plot_radar_pps(OUTPUT_DIR)
    copy_supporting_plots(OUTPUT_DIR)
    write_story(OUTPUT_DIR, progression)
    print(f"Wrote presentation plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

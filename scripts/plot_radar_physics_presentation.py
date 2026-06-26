#!/usr/bin/env python3
"""Presentation plots for the controlled radar pedestrian physics/PPS sweep."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


INPUT = Path("radar_pedestrian_diagnostic_runs/radar_ped_full_pps5k_to100k_dist2_to100_physics_20260623/summary_by_condition.csv")
OUT_DIR = Path("analysis_outputs/radar_physics_presentation_20260625")
MIN_DISTANCE_M = 10.0

PHYSICS_LABELS = {
    "default": "Default walker setting",
    "on": "Walker physics ON",
    "off": "Walker physics OFF",
}
PHYSICS_COLORS = {
    "default": "#2454a6",
    "on": "#4f9d69",
    "off": "#d65a31",
}


def read_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with INPUT.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "physics": str(row["walker_physics"]),
                    "pps": int(row["radar_pps"]),
                    "distance": float(row["target_distance_m"]),
                    "support": float(row["bbox_support_rate"]),
                    "mean_points": float(row["mean_bbox_points"]),
                    "mean_total": float(row["mean_total_radar_points"]),
                }
            )
    return [r for r in rows if float(r["distance"]) >= MIN_DISTANCE_M]


def setup_style() -> None:
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


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def subset(rows: Iterable[dict[str, float | str]], *, pps: int | None = None, physics: str | None = None) -> list[dict[str, float | str]]:
    result = list(rows)
    if pps is not None:
        result = [r for r in result if int(r["pps"]) == int(pps)]
    if physics is not None:
        result = [r for r in result if str(r["physics"]) == str(physics)]
    return sorted(result, key=lambda r: float(r["distance"]))


def plot_physics_comparison(rows: list[dict[str, float | str]], pps: int = 48000) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)
    for physics in ("default", "on", "off"):
        data = subset(rows, pps=pps, physics=physics)
        if not data:
            continue
        x = [float(r["distance"]) for r in data]
        support = [float(r["support"]) * 100.0 for r in data]
        points = [float(r["mean_points"]) for r in data]
        label = PHYSICS_LABELS.get(physics, physics)
        color = PHYSICS_COLORS.get(physics, None)
        axes[0].plot(x, support, marker="o", linewidth=2.3, label=label, color=color)
        axes[1].plot(x, points, marker="o", linewidth=2.3, label=label, color=color)

    axes[0].set_title(f"Pedestrian Radar Support at {pps//1000}k PPS")
    axes[0].set_ylabel("Frames with >=1 pedestrian radar point (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_title(f"Mean Radar Points on Pedestrian at {pps//1000}k PPS")
    axes[1].set_ylabel("Mean points inside pedestrian box")
    for ax in axes:
        ax.set_xlabel("Pedestrian distance from radar (m)")
        ax.set_xlim(MIN_DISTANCE_M, 100)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("CARLA Walker Physics Did Not Explain the Distance Falloff", y=1.04)
    save(fig, "radar_physics_on_off_distance_comparison_48k")


def plot_fixed_pps_distance(rows: list[dict[str, float | str]], physics: str = "default") -> None:
    selected_pps = [5000, 12000, 48000, 100000]
    colors = ["#6f7785", "#2454a6", "#d65a31", "#4f9d69"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)
    for pps, color in zip(selected_pps, colors):
        data = subset(rows, pps=pps, physics=physics)
        if not data:
            continue
        x = [float(r["distance"]) for r in data]
        support = [float(r["support"]) * 100.0 for r in data]
        points = [float(r["mean_points"]) for r in data]
        label = f"{pps//1000}k PPS"
        axes[0].plot(x, support, marker="o", linewidth=2.3, label=label, color=color)
        axes[1].plot(x, points, marker="o", linewidth=2.3, label=label, color=color)

    axes[0].set_title("Support Rate vs Distance")
    axes[0].set_ylabel("Frames with >=1 pedestrian radar point (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_title("Mean Pedestrian Radar Points vs Distance")
    axes[1].set_ylabel("Mean points inside pedestrian box")
    for ax in axes:
        ax.set_xlabel("Pedestrian distance from radar (m)")
        ax.set_xlim(MIN_DISTANCE_M, 100)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Higher PPS Helps, But Distance Still Reduces Pedestrian Radar Support", y=1.04)
    save(fig, "radar_fixed_pps_distance_sweep_default")


def main() -> None:
    setup_style()
    rows = read_rows()
    plot_physics_comparison(rows, pps=48000)
    plot_fixed_pps_distance(rows, physics="default")
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()

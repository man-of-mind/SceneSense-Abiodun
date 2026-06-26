#!/usr/bin/env python3
"""Presentation plots for moving-pedestrian radar diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


MOVING_SUMMARY = Path(
    "radar_pedestrian_diagnostic_runs/"
    "radar_ped_cross_pps5k_to100k_dist5_to55_scaledamp_default_20260625/"
    "summary_by_condition.csv"
)
STATIC_SUMMARY = Path(
    "radar_pedestrian_diagnostic_runs/"
    "radar_ped_full_pps5k_to100k_dist2_to100_physics_20260623/"
    "summary_by_condition.csv"
)
STATIC_FRAMES = Path(
    "radar_pedestrian_diagnostic_runs/"
    "radar_ped_full_pps5k_to100k_dist2_to100_physics_20260623/"
    "frame_metrics.csv"
)
MODEL_UTILITY = Path("analysis_outputs/model_utility_by_radar_points_smoke/model_utility_by_radar_point_bin.csv")
OUT_DIR = Path("analysis_outputs/radar_crossing_presentation_20260625")


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
            "grid.alpha": 0.24,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def subset(rows: Iterable[dict[str, str]], *, pps: int | None = None, physics: str | None = "default") -> list[dict[str, str]]:
    result = list(rows)
    if pps is not None:
        result = [r for r in result if int(r["radar_pps"]) == int(pps)]
    if physics is not None and "walker_physics" in result[0]:
        result = [r for r in result if str(r["walker_physics"]) == str(physics)]
    return sorted(result, key=lambda r: float(r["target_distance_m"]))


def plot_moving_points_and_useful_support(rows: list[dict[str, str]]) -> None:
    selected_pps = [5000, 12000, 48000, 100000]
    colors = ["#6f7785", "#2454a6", "#d65a31", "#4f9d69"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharex=True)

    for pps, color in zip(selected_pps, colors):
        data = subset(rows, pps=pps)
        x = [float(r["target_distance_m"]) for r in data]
        mean_points = [float(r["mean_radius_points"]) for r in data]
        useful_support = [float(r["radius_support_rate_ge10"]) * 100.0 for r in data]
        label = f"{pps//1000}k PPS"
        axes[0].plot(x, mean_points, marker="o", linewidth=2.3, label=label, color=color)
        axes[1].plot(x, useful_support, marker="o", linewidth=2.3, label=label, color=color)

    axes[0].axhline(10, color="#222222", linestyle="--", linewidth=1.2, alpha=0.65)
    axes[0].text(55, 11, "10-point reference", ha="right", va="bottom", fontsize=9)
    axes[0].set_title("Mean Pedestrian Radar Points")
    axes[0].set_ylabel("Mean points within pedestrian radius")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Useful-Support Rate")
    axes[1].set_ylabel("Frames with >=10 pedestrian points (%)")
    axes[1].set_ylim(-3, 103)
    for ax in axes:
        ax.set_xlabel("Pedestrian distance from radar (m)")
        ax.set_xlim(5, 55)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Moving Pedestrian Sweep: PPS Helps, But Distance Still Dominates", y=1.04)
    save(fig, "moving_pedestrian_points_and_useful_support")


def plot_binary_vs_useful(rows: list[dict[str, str]], pps: int = 100000) -> None:
    data = subset(rows, pps=pps)
    x = [float(r["target_distance_m"]) for r in data]
    binary = [float(r["radius_support_rate"]) * 100.0 for r in data]
    ge5 = [float(r["radius_support_rate_ge5"]) * 100.0 for r in data]
    ge10 = [float(r["radius_support_rate_ge10"]) * 100.0 for r in data]
    mean_points = [float(r["mean_radius_points"]) for r in data]

    fig, ax1 = plt.subplots(figsize=(9.8, 5.1))
    ax1.plot(x, binary, marker="o", linewidth=2.5, color="#8b8f97", label=">=1 point: contact")
    ax1.plot(x, ge5, marker="o", linewidth=2.5, color="#2454a6", label=">=5 points")
    ax1.plot(x, ge10, marker="o", linewidth=2.5, color="#d65a31", label=">=10 points: useful support")
    ax1.set_title(f"Binary Radar Support Overstates Useful Evidence at {pps//1000}k PPS")
    ax1.set_xlabel("Pedestrian distance from radar (m)")
    ax1.set_ylabel("Frames meeting threshold (%)")
    ax1.set_xlim(5, 55)
    ax1.set_ylim(-3, 103)
    ax1.legend(loc="upper right", frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_points, marker="s", linewidth=1.9, linestyle="--", color="#4f9d69", label="Mean points")
    ax2.set_ylabel("Mean pedestrian radar points")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    save(fig, "binary_vs_useful_support_100k")


def static_threshold_rows(frame_path: Path, pps: int = 100000, min_distance_m: float = 10.0) -> list[dict[str, float]]:
    groups: dict[float, list[int]] = {}
    with frame_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("walker_physics") != "default":
                continue
            if int(row["radar_pps"]) != int(pps):
                continue
            distance = float(row["target_distance_m"])
            if distance < min_distance_m:
                continue
            groups.setdefault(distance, []).append(int(row["person_radius_points"]))
    results: list[dict[str, float]] = []
    for distance, counts in sorted(groups.items()):
        n = max(1, len(counts))
        results.append(
            {
                "distance": distance,
                "mean_points": sum(counts) / n,
                "ge1": sum(count >= 1 for count in counts) / n * 100.0,
                "ge5": sum(count >= 5 for count in counts) / n * 100.0,
                "ge10": sum(count >= 10 for count in counts) / n * 100.0,
                "ge20": sum(count >= 20 for count in counts) / n * 100.0,
                "ge50": sum(count >= 50 for count in counts) / n * 100.0,
            }
        )
    return results


def plot_static_binary_vs_useful(frame_path: Path, pps: int = 100000) -> None:
    data = static_threshold_rows(frame_path, pps=pps)
    x = [r["distance"] for r in data]
    fig, ax1 = plt.subplots(figsize=(10.4, 5.1))
    ax1.plot(x, [r["ge1"] for r in data], marker="o", linewidth=2.5, color="#8b8f97", label=">=1 point: contact")
    ax1.plot(x, [r["ge10"] for r in data], marker="o", linewidth=2.5, color="#d65a31", label=">=10 points")
    ax1.plot(x, [r["ge20"] for r in data], marker="o", linewidth=2.5, color="#2454a6", label=">=20 points")
    ax1.plot(x, [r["ge50"] for r in data], marker="o", linewidth=2.5, color="#7b4ab8", label=">=50 points")
    ax1.set_title(f"Static Pedestrian: Useful Radar Evidence Also Falls With Distance at {pps//1000}k PPS")
    ax1.set_xlabel("Pedestrian distance from radar (m)")
    ax1.set_ylabel("Frames meeting threshold (%)")
    ax1.set_xlim(10, 100)
    ax1.set_ylim(-3, 103)
    ax1.legend(loc="upper right", frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(x, [r["mean_points"] for r in data], marker="s", linewidth=1.9, linestyle="--", color="#4f9d69")
    ax2.set_ylabel("Mean pedestrian radar points")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    save(fig, "static_binary_vs_useful_support_100k")


def plot_static_vs_crossing(static_rows: list[dict[str, str]], moving_rows: list[dict[str, str]], pps: int = 100000) -> None:
    static = {float(r["target_distance_m"]): float(r["mean_radius_points"]) for r in subset(static_rows, pps=pps)}
    moving = {float(r["target_distance_m"]): float(r["mean_radius_points"]) for r in subset(moving_rows, pps=pps)}
    distances = [d for d in sorted(moving) if d in static]
    x = list(range(len(distances)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.bar([i - width / 2 for i in x], [static[d] for d in distances], width, label="Static, centered pedestrian", color="#2454a6")
    ax.bar([i + width / 2 for i in x], [moving[d] for d in distances], width, label="Moving across FOV", color="#d65a31")
    ax.set_title(f"Moving Across the FOV Reduces Average Radar Evidence at {pps//1000}k PPS")
    ax.set_xlabel("Pedestrian distance from radar (m)")
    ax.set_ylabel("Mean points within pedestrian radius")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in distances])
    ax.legend(loc="upper right", frameon=False)
    save(fig, "static_vs_moving_mean_points_100k")


def plot_model_utility(rows: list[dict[str, str]]) -> None:
    person = [r for r in rows if r["label"] == "person" and not r["distance_bin_m"]]
    labels = [r["radar_point_bin"].replace("-inf", "+") for r in person]
    recall = [float(r["recall"]) * 100.0 for r in person]
    xy = [float(r["mean_xy_error_m"]) for r in person]

    fig, ax1 = plt.subplots(figsize=(9.8, 5.0))
    bars = ax1.bar(labels, recall, color="#2454a6", alpha=0.9, label="Localization recall")
    ax1.set_title("Model Utility: More Pedestrian Radar Points Improve Localization")
    ax1.set_xlabel("Radar points associated with ground-truth pedestrian")
    ax1.set_ylabel("Person localization recall (%)")
    ax1.set_ylim(0, 100)
    for bar, value in zip(bars, recall):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.0f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(labels, xy, marker="o", linewidth=2.2, color="#d65a31", label="Mean XY error")
    ax2.set_ylabel("Mean XY error when matched (m)")
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    lines, line_labels = ax1.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, line_labels + line_labels2, loc="upper left", frameon=False)
    save(fig, "model_utility_person_recall_by_radar_points")


def main() -> None:
    setup_style()
    moving = read_rows(MOVING_SUMMARY)
    static = read_rows(STATIC_SUMMARY)
    utility = read_rows(MODEL_UTILITY)
    plot_moving_points_and_useful_support(moving)
    plot_binary_vs_useful(moving)
    plot_static_binary_vs_useful(STATIC_FRAMES)
    plot_static_vs_crossing(static, moving)
    plot_model_utility(utility)
    print(f"Wrote radar crossing presentation plots to {OUT_DIR}")


if __name__ == "__main__":
    main()

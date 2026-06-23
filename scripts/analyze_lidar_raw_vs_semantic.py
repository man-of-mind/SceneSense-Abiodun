#!/usr/bin/env python3
"""Analyze SceneSense raw-vs-semantic LiDAR diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {
    "raw_bbox": "Raw LiDAR\ngeometry",
    "semantic_tag_bbox": "Semantic LiDAR\ntag filter",
    "semantic_object_id": "Semantic LiDAR\nobject ID",
}

CLASS_LABELS = {
    "vehicle": "Vehicle",
    "person": "Person",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create summary tables and plots for raw-vs-semantic LiDAR diagnostics."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more lidar_diagnostic_runs/raw_vs_semantic_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/lidar_raw_vs_semantic",
        help="Directory for summary outputs.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Raw vs Semantic LiDAR",
        help="Title prefix for generated plots.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_name(run_dir: Path) -> str:
    return run_dir.name.replace("raw_vs_semantic_", "")


def summarize_actor_metrics(actor_rows: Sequence[dict], name: str) -> List[dict]:
    rows: List[dict] = []
    modes = ["raw_bbox", "semantic_tag_bbox", "semantic_object_id"]
    classes = ["vehicle", "person"]
    for mode in modes:
        for actor_type in classes:
            subset = [
                row
                for row in actor_rows
                if row.get("mode") == mode and row.get("actor_type") == actor_type
            ]
            total = len(subset)
            hits = sum(int(row.get("hit", 0)) for row in subset)
            xy_errors = [
                safe_float(row.get("xy_error_m"))
                for row in subset
                if int(row.get("hit", 0)) and safe_float(row.get("xy_error_m")) is not None
            ]
            point_counts = [int(float(row.get("point_count", 0))) for row in subset]
            rows.append(
                {
                    "run": name,
                    "mode": mode,
                    "actor_type": actor_type,
                    "actor_observations": total,
                    "hits": hits,
                    "recall": hits / total if total else np.nan,
                    "xy_error_mean_m": float(np.mean(xy_errors)) if xy_errors else np.nan,
                    "xy_error_median_m": float(np.median(xy_errors)) if xy_errors else np.nan,
                    "points_per_actor_mean": float(np.mean(point_counts)) if point_counts else np.nan,
                    "points_per_actor_median": float(np.median(point_counts)) if point_counts else np.nan,
                }
            )
    return rows


def summarize_frame_metrics(frame_rows: Sequence[dict], name: str) -> dict:
    raw_points = [safe_float(row.get("raw_points")) for row in frame_rows]
    sem_points = [safe_float(row.get("semantic_points")) for row in frame_rows]
    raw_bytes = [safe_float(row.get("raw_bytes_est")) for row in frame_rows]
    sem_bytes = [safe_float(row.get("semantic_bytes_est")) for row in frame_rows]
    raw_points = [x for x in raw_points if x is not None]
    sem_points = [x for x in sem_points if x is not None]
    raw_bytes = [x for x in raw_bytes if x is not None]
    sem_bytes = [x for x in sem_bytes if x is not None]
    return {
        "run": name,
        "frames": len(frame_rows),
        "raw_points_mean": float(np.mean(raw_points)) if raw_points else np.nan,
        "semantic_points_mean": float(np.mean(sem_points)) if sem_points else np.nan,
        "raw_bytes_mean": float(np.mean(raw_bytes)) if raw_bytes else np.nan,
        "semantic_bytes_est_mean": float(np.mean(sem_bytes)) if sem_bytes else np.nan,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: Sequence[dict], run: str, mode: str, actor_type: str, metric: str) -> float:
    for row in rows:
        if row["run"] == run and row["mode"] == mode and row["actor_type"] == actor_type:
            return float(row[metric])
    return np.nan


def plot_grouped_metric(
    rows: Sequence[dict],
    runs: Sequence[str],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    modes = ["raw_bbox", "semantic_tag_bbox", "semantic_object_id"]
    classes = ["vehicle", "person"]
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 4.8), sharey=True)
    if len(classes) == 1:
        axes = [axes]
    width = 0.22
    x = np.arange(len(runs))
    for ax, actor_type in zip(axes, classes):
        for idx, mode in enumerate(modes):
            vals = [metric_lookup(rows, run, mode, actor_type, metric) for run in runs]
            ax.bar(x + (idx - 1) * width, vals, width=width, label=MODE_LABELS[mode])
        ax.set_title(CLASS_LABELS[actor_type])
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.84, 0.94))
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_points_per_frame(frame_rows: Sequence[dict], output_path: Path, title: str) -> None:
    if not frame_rows:
        return
    runs = [row["run"] for row in frame_rows]
    raw = [float(row["raw_points_mean"]) for row in frame_rows]
    sem = [float(row["semantic_points_mean"]) for row in frame_rows]
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - width / 2, raw, width=width, label="Raw LiDAR")
    ax.bar(x + width / 2, sem, width=width, label="Semantic LiDAR")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=20, ha="right")
    ax.set_ylabel("Mean points per frame")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(
    path: Path,
    actor_summary: Sequence[dict],
    frame_summary: Sequence[dict],
    run_dirs: Sequence[Path],
) -> None:
    lines: List[str] = []
    lines.append("# Raw vs Semantic LiDAR Diagnostic Summary")
    lines.append("")
    lines.append("## Runs")
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.append("")
    lines.append("## What The Modes Mean")
    lines.append("- `raw_bbox`: raw LiDAR geometry assigned to CARLA actor boxes for evaluation only.")
    lines.append("- `semantic_tag_bbox`: semantic LiDAR points filtered by semantic tag, then assigned to actor boxes.")
    lines.append("- `semantic_object_id`: semantic LiDAR grouped by CARLA object ID; this is oracle association.")
    lines.append("")
    lines.append("## Actor Coverage")
    lines.append("")
    lines.append("| Run | Mode | Class | Recall | XY error mean (m) | Points/actor mean | Observations |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in actor_summary:
        recall = float(row["recall"]) if not np.isnan(float(row["recall"])) else np.nan
        xy = float(row["xy_error_mean_m"]) if not np.isnan(float(row["xy_error_mean_m"])) else np.nan
        pts = float(row["points_per_actor_mean"]) if not np.isnan(float(row["points_per_actor_mean"])) else np.nan
        lines.append(
            f"| {row['run']} | {row['mode']} | {row['actor_type']} | "
            f"{recall:.3f} | {xy:.3f} | {pts:.1f} | {row['actor_observations']} |"
        )
    lines.append("")
    lines.append("## Frame-Level Sensor Load")
    lines.append("")
    lines.append("| Run | Frames | Raw pts/frame | Semantic pts/frame | Raw bytes/frame est. | Semantic bytes/frame est. |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in frame_summary:
        lines.append(
            f"| {row['run']} | {row['frames']} | {row['raw_points_mean']:.1f} | "
            f"{row['semantic_points_mean']:.1f} | {row['raw_bytes_mean']:.0f} | "
            f"{row['semantic_bytes_est_mean']:.0f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "Semantic LiDAR is expected to look better mainly because it exposes simulator-provided tags "
        "and object IDs. The useful radar-transfer ideas are the geometry-side behaviors: point density, "
        "actor-box association for training/evaluation, short temporal accumulation, and BEV/voxel features."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(item) for item in args.run_dirs]
    actor_summary: List[dict] = []
    frame_summary: List[dict] = []
    valid_runs: List[str] = []

    for run_dir in run_dirs:
        actor_path = run_dir / "actor_metrics.csv"
        frame_path = run_dir / "frame_metrics.csv"
        if not actor_path.exists() or not frame_path.exists():
            raise FileNotFoundError(
                f"Missing actor_metrics.csv or frame_metrics.csv in {run_dir}"
            )
        name = run_name(run_dir)
        valid_runs.append(name)
        actor_rows = read_csv(actor_path)
        frame_rows = read_csv(frame_path)
        actor_summary.extend(summarize_actor_metrics(actor_rows, name))
        frame_summary.append(summarize_frame_metrics(frame_rows, name))

    write_csv(output_dir / "lidar_raw_vs_semantic_actor_summary.csv", actor_summary)
    write_csv(output_dir / "lidar_raw_vs_semantic_frame_summary.csv", frame_summary)
    write_markdown(
        output_dir / "lidar_raw_vs_semantic_summary.md",
        actor_summary,
        frame_summary,
        run_dirs,
    )

    plot_grouped_metric(
        actor_summary,
        valid_runs,
        "recall",
        "Actor recall",
        f"{args.title_prefix}: actor recall",
        output_dir / "lidar_raw_vs_semantic_recall.png",
        ylim=(0.0, 1.05),
    )
    plot_grouped_metric(
        actor_summary,
        valid_runs,
        "xy_error_mean_m",
        "Mean XY error (m)",
        f"{args.title_prefix}: localization error",
        output_dir / "lidar_raw_vs_semantic_xy_error.png",
    )
    plot_grouped_metric(
        actor_summary,
        valid_runs,
        "points_per_actor_mean",
        "Mean points per actor observation",
        f"{args.title_prefix}: actor point support",
        output_dir / "lidar_raw_vs_semantic_points_per_actor.png",
    )
    plot_points_per_frame(
        frame_summary,
        output_dir / "lidar_raw_vs_semantic_points_per_frame.png",
        f"{args.title_prefix}: point load",
    )

    print(f"Wrote LiDAR raw-vs-semantic analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

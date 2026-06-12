#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


NUMERIC_COLUMNS = (
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "payload_bytes",
    "payload_bytes_uncompressed",
    "payload_chunks",
    "miou_binary",
    "miou_3class_macro",
    "miou_vehicle_iou",
    "miou_person_iou",
    "gt_vehicle_pixels",
    "gt_person_pixels",
)


def load_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No evaluation CSV files were provided.")
    df = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "payload_bytes_uncompressed" in df.columns and "payload_bytes" in df.columns:
        df["compression_ratio"] = df["payload_bytes_uncompressed"] / df["payload_bytes"].replace(0, np.nan)
    return df


def scalar_summary(series: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {"rows": int(len(df))}
    for column in (
        "front_ms",
        "back_ms",
        "round_trip_ms",
        "payload_bytes",
        "payload_bytes_uncompressed",
        "compression_ratio",
        "miou_3class_macro",
        "miou_vehicle_iou",
        "miou_person_iou",
        "miou_binary",
    ):
        if column in df.columns:
            summary[column] = scalar_summary(df[column])
    if "run_tag" in df.columns:
        summary["run_tags"] = sorted(str(value) for value in df["run_tag"].dropna().unique())
    return summary


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for column, label in (
        ("front_ms", "front half"),
        ("back_ms", "back half"),
        ("round_trip_ms", "round trip"),
    ):
        if column in df.columns:
            ax.plot(x, df[column], linewidth=1.0, label=label)
    ax.set_title("Split LR-ASPP Latency Over Time")
    ax.set_xlabel("Logged frame index")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "latency_timeseries")

    if "payload_bytes" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        ax.plot(x, df["payload_bytes"] / 1024.0, linewidth=1.0, label="compressed payload")
        if "payload_bytes_uncompressed" in df.columns:
            ax.plot(x, df["payload_bytes_uncompressed"] / 1024.0, linewidth=1.0, label="float16 baseline")
        ax.set_title("UDP Payload Size Over Time")
        ax.set_xlabel("Logged frame index")
        ax.set_ylabel("Payload (KiB)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_figure(fig, output_dir / "payload_timeseries")

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for column, label in (
        ("miou_3class_macro", "macro mIoU"),
        ("miou_vehicle_iou", "vehicle IoU"),
        ("miou_person_iou", "person IoU"),
    ):
        if column in df.columns:
            ax.plot(x, df[column], linewidth=1.0, label=label)
    ax.set_title("CARLA Semantic-GT Segmentation Quality Over Time")
    ax.set_xlabel("Logged frame index")
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "quality_timeseries")


def plot_grouped_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if "run_tag" not in df.columns:
        return
    grouped = df.groupby("run_tag", dropna=False)
    rows = []
    for run_tag, subset in grouped:
        rows.append(
            {
                "run_tag": str(run_tag),
                "front_ms_mean": float(subset["front_ms"].mean()) if "front_ms" in subset else float("nan"),
                "back_ms_mean": float(subset["back_ms"].mean()) if "back_ms" in subset else float("nan"),
                "round_trip_ms_mean": float(subset["round_trip_ms"].mean()) if "round_trip_ms" in subset else float("nan"),
                "payload_kib_mean": float(subset["payload_bytes"].mean() / 1024.0) if "payload_bytes" in subset else float("nan"),
                "miou_macro_mean": float(subset["miou_3class_macro"].mean()) if "miou_3class_macro" in subset else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values("run_tag")
    table.to_csv(output_dir / "grouped_summary.csv", index=False)
    if table.empty:
        return

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - 0.2, table["front_ms_mean"], width=0.2, label="front")
    ax.bar(x, table["back_ms_mean"], width=0.2, label="back")
    ax.bar(x + 0.2, table["round_trip_ms_mean"], width=0.2, label="round trip")
    ax.set_xticks(x, table["run_tag"], rotation=25, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Mean Latency by Run")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "latency_by_run")

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax1.bar(x - 0.18, table["payload_kib_mean"], width=0.36, color="#2563eb", label="payload")
    ax1.set_ylabel("Mean payload (KiB)")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, table["miou_macro_mean"], width=0.36, color="#f97316", label="macro mIoU")
    ax2.set_ylabel("Mean macro mIoU")
    ax2.set_ylim(0.0, 1.0)
    ax1.set_xticks(x, table["run_tag"], rotation=25, ha="right")
    ax1.set_title("Payload and Quality by Run")
    ax1.grid(True, axis="y", alpha=0.25)
    save_figure(fig, output_dir / "payload_quality_by_run")


def write_report(output_dir: Path, summary: Dict[str, object]) -> None:
    lines = ["Trained LR-ASPP Split-Inference Evaluation Report", ""]
    lines.append(f"Rows: {summary.get('rows', 0)}")
    for key in (
        "front_ms",
        "back_ms",
        "round_trip_ms",
        "payload_bytes",
        "compression_ratio",
        "miou_3class_macro",
        "miou_vehicle_iou",
        "miou_person_iou",
    ):
        value = summary.get(key)
        if not isinstance(value, dict):
            continue
        lines.append(
            f"{key}: mean={value.get('mean', float('nan')):.4f}, "
            f"median={value.get('median', float('nan')):.4f}, "
            f"p95={value.get('p95', float('nan')):.4f}"
        )
    if summary.get("run_tags"):
        lines.append("")
        lines.append("Run tags: " + ", ".join(summary["run_tags"]))  # type: ignore[index]
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=[], help="Evaluation CSV path. Can be repeated.")
    parser.add_argument("--glob", default="", help="Optional glob for evaluation CSVs.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = [Path(path).expanduser().resolve() for path in args.csv]
    if args.glob:
        paths.extend(Path(path).expanduser().resolve() for path in sorted(glob.glob(args.glob)))
    paths = [path for path in paths if path.exists()]
    df = load_csvs(paths)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, summary)
    plot_timeseries(df, output_dir)
    plot_grouped_bars(df, output_dir)
    print(f"Wrote trained split evaluation analysis to {output_dir}")


if __name__ == "__main__":
    main()

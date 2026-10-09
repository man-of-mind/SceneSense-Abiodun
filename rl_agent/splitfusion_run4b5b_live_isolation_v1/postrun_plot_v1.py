"""PNG-only aligned-frame visualization of offline Run-4B/Run-5B metrics.

The default x-axis is explicitly *Frame order*, not elapsed or wall-clock time.
Timeout markers sit at the 170-ms censoring boundary and are labelled as
censored; they are never presented as observed 170-ms latencies.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .postrun_evaluator_v1 import FRAME_FIELDS, FRAME_METRICS_SCHEMA


class PostRunPlotError(RuntimeError):
    """Frame metrics cannot support the requested PNG."""


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise PostRunPlotError("frame metrics contain a non-finite scalar")
    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FRAME_FIELDS:
            raise PostRunPlotError("frame metrics columns drifted")
        rows = [dict(row) for row in reader]
    if not rows:
        raise PostRunPlotError("frame metrics are empty")
    if any(row["schema"] != FRAME_METRICS_SCHEMA for row in rows):
        raise PostRunPlotError("frame metrics schema drifted")
    order = [int(row["frame_order"]) for row in rows]
    if order != list(range(len(rows))):
        raise PostRunPlotError("frame order is not contiguous from zero")
    return rows


def _series(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    return np.asarray([
        np.nan if (value := _optional_float(row[name])) is None else value
        for row in rows
    ], dtype=np.float64)


def plot_frame_metrics_png(*, frame_metrics_csv: Path, output_png: Path,
                           title: str = "SplitFusion offline validation metrics",
                           ) -> Path:
    """Create one create-only PNG; PDF and misleading time labels are refused."""

    output = Path(output_png)
    if output.suffix.lower() != ".png":
        raise PostRunPlotError("output must use the .png extension")
    if output.exists():
        raise PostRunPlotError(f"create-only plot already exists: {output}")
    if not output.parent.is_dir():
        raise PostRunPlotError("plot parent directory does not exist")
    rows = _read_rows(Path(frame_metrics_csv))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.arange(len(rows), dtype=np.int64)
    deadline = float(rows[0]["deadline_ms"])
    if any(float(row["deadline_ms"]) != deadline for row in rows):
        raise PostRunPlotError("deadline changes within the run")
    success = np.asarray([int(row["operational_success"]) == 1 for row in rows])
    timeout = ~success
    latency = _series(rows, "operational_latency_ms")

    fig, axes = plt.subplots(
        6, 1, figsize=(13.5, 17.0), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0, 1.0, 1.0, 0.9]},
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(x[success], latency[success], color="#1769aa", marker="o",
            markersize=3.5, linewidth=1.2, label="Observed successful ACK latency")
    ax.scatter(x[timeout], np.full(int(timeout.sum()), deadline), marker="x",
               s=34, linewidths=1.4, color="#c62828",
               label="Timeout (right-censored above 170 ms)")
    ax.axhline(deadline, color="#c62828", linestyle="--", linewidth=1.1,
               label=f"Operational deadline ({deadline:g} ms)")
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_title("Operational tail-output ACK latency", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=True)
    ax.grid(alpha=0.22)

    ax = axes[1]
    for field, label, color in (
        ("vehicle_xy_error_median_m", "Vehicle median XY error", "#1565c0"),
        ("person_xy_error_median_m", "Person median XY error", "#ef6c00"),
    ):
        ax.plot(x, _series(rows, field), marker=".", linewidth=1.0,
                label=label, color=color)
    ax.set_ylabel("Error (m)", fontweight="bold")
    ax.set_title("Matched-object localization error", fontweight="bold")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[2]
    for field, label, color in (
        ("segmentation_miou_3class", "3-class mIoU", "#2e7d32"),
        ("segmentation_vehicle_iou", "Vehicle IoU", "#1565c0"),
        ("segmentation_person_iou", "Person IoU", "#ef6c00"),
    ):
        ax.plot(x, _series(rows, field), linewidth=1.0, label=label, color=color)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("IoU", fontweight="bold")
    ax.set_title("Offline segmentation quality", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[3]
    ax.plot(x, _series(rows, "q_perc"), color="#6a1b9a", linewidth=1.25,
            label="Q_perc (offline CARLA evaluation)")
    ax.plot(x, _series(rows, "evaluation_reward"), color="#00838f",
            linewidth=1.05, label="Evaluation reward")
    ax.axhline(-1.0, color="#777777", linestyle=":", linewidth=0.9,
               label="Registered timeout reward")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Offline quality and evaluation reward", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[4]
    q = _series(rows, "q_exec")
    mode = _series(rows, "mode_id")
    q_line = ax.plot(x, q, color="#6a1b9a", linewidth=1.2,
                     label="Executed q (drop fraction)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("q", color="#6a1b9a", fontweight="bold")
    twin = ax.twinx()
    mode_line = twin.scatter(x, mode, s=12, color="#424242", alpha=0.65,
                             label="Mode ID")
    twin.set_ylim(-0.6, 11.6)
    twin.set_ylabel("Mode ID", color="#424242", fontweight="bold")
    ax.set_title("Executed policy actions", fontweight="bold")
    handles = list(q_line) + [mode_line]
    ax.legend(handles, [item.get_label() for item in handles],
              loc="upper center", ncol=2, fontsize=8)
    ax.grid(alpha=0.22)

    ax = axes[5]
    ax.plot(x, _series(rows, "payload_bytes") / 1024.0, color="#ad1457",
            linewidth=1.1, label="Transmitted payload")
    ax.set_ylabel("Payload (KiB)", fontweight="bold")
    ax.set_title("Action payload", fontweight="bold")
    ax.set_xlabel("Frame order", fontweight="bold")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(alpha=0.22)

    fig.savefig(output, dpi=180, format="png", facecolor="white")
    plt.close(fig)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot offline SplitFusion validation metrics (PNG only)."
    )
    parser.add_argument("--frame-metrics", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--title", default="SplitFusion offline validation metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plot_frame_metrics_png(
        frame_metrics_csv=args.frame_metrics,
        output_png=args.output_png,
        title=args.title,
    )
    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PostRunPlotError", "plot_frame_metrics_png", "main"]

#!/usr/bin/env python3
"""Compare SceneSense application metrics across multiple run groups."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_scenesense_app_metrics as app_metrics  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_analysis"
MPLCONFIG_DIR = Path("/tmp/scenesense_mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

SUMMARY_FIELDS = (
    "run_group",
    "stream_id",
    "transport_label",
    "rows",
    "received",
    "timeout_rate",
    "duration_s",
    "approx_fps",
    "avg_round_trip_ms",
    "p95_round_trip_ms",
    "avg_front_ms",
    "p95_front_ms",
    "avg_back_ms",
    "p95_back_ms",
    "avg_network_queue_ms",
    "p95_network_queue_ms",
    "avg_glass_to_result_ms",
    "p95_glass_to_result_ms",
    "avg_feature_payload_mb",
    "feature_goodput_mbps",
    "avg_feature_chunks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one combined CSV and a few comparison plots from SceneSense "
            "application metrics run groups."
        )
    )
    parser.add_argument(
        "--root",
        default=str(app_metrics.DEFAULT_RUN_ROOT),
        help="Root containing scenesense run folders.",
    )
    parser.add_argument(
        "--run-group",
        action="append",
        required=True,
        help="Run group to include. Repeat for each experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults under metrics_logs/scenesense_analysis/.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write only the combined CSV.",
    )
    return parser.parse_args()


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_stream(run_group: str, stream_id: str, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    received_rows = [row for row in rows if app_metrics.to_bool(row.get("result_received", ""))]
    elapsed = app_metrics.finite_values(rows, "elapsed_s")
    duration_s = (
        max(elapsed) - min(elapsed)
        if len(elapsed) >= 2
        else (max(elapsed) if elapsed else float("nan"))
    )
    if not (math.isfinite(duration_s) and duration_s > 0):
        duration_s = float("nan")

    rtt = app_metrics.finite_values(received_rows, "round_trip_ms")
    front_all = app_metrics.finite_values(rows, "front_ms")
    back = app_metrics.finite_values(received_rows, "back_ms")
    feature_payload = app_metrics.finite_values(rows, "feature_payload_bytes")
    chunks = app_metrics.finite_values(rows, "feature_payload_chunks")
    network_queue = finite(
        app_metrics.to_float(row.get("round_trip_ms", ""))
        - app_metrics.to_float(row.get("back_ms", ""))
        for row in received_rows
    )
    glass_to_result = finite(
        app_metrics.to_float(row.get("front_ms", ""))
        + app_metrics.to_float(row.get("round_trip_ms", ""))
        for row in received_rows
    )
    total_feature_bytes = sum(feature_payload)
    feature_goodput_mbps = (
        total_feature_bytes * 8.0 / duration_s / 1_000_000.0
        if math.isfinite(duration_s) and duration_s > 0
        else float("nan")
    )
    transport_labels = sorted(
        {row.get("transport_label", "") for row in rows if row.get("transport_label")}
    )
    return {
        "run_group": run_group,
        "stream_id": stream_id,
        "transport_label": ",".join(transport_labels),
        "rows": total,
        "received": len(received_rows),
        "timeout_rate": (total - len(received_rows)) / total if total else float("nan"),
        "duration_s": duration_s,
        "approx_fps": total / duration_s if math.isfinite(duration_s) and duration_s > 0 else float("nan"),
        "avg_round_trip_ms": mean(rtt),
        "p95_round_trip_ms": percentile(rtt, 95),
        "avg_front_ms": mean(front_all),
        "p95_front_ms": percentile(front_all, 95),
        "avg_back_ms": mean(back),
        "p95_back_ms": percentile(back, 95),
        "avg_network_queue_ms": mean(network_queue),
        "p95_network_queue_ms": percentile(network_queue, 95),
        "avg_glass_to_result_ms": mean(glass_to_result),
        "p95_glass_to_result_ms": percentile(glass_to_result, 95),
        "avg_feature_payload_mb": mean(feature_payload) / 1_000_000.0 if feature_payload else float("nan"),
        "feature_goodput_mbps": feature_goodput_mbps,
        "avg_feature_chunks": mean(chunks),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def plot(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['run_group']}\n{row['stream_id']}" for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    bar_specs: Sequence[Tuple[str, str, str]] = (
        ("avg_round_trip_ms", "Average RTT", "ms"),
        ("avg_network_queue_ms", "Average RTT minus back_ms", "ms"),
        ("avg_front_ms", "Average front half", "ms"),
        ("avg_back_ms", "Average back half", "ms"),
    )
    for ax, (field, title, ylabel) in zip(axes.ravel(), bar_specs):
        values = [app_metrics.to_float(row.get(field, "")) for row in rows]
        ax.bar(x, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    fig.suptitle("SceneSense fusion latency comparison")
    fig.savefig(out_dir / "latency_component_bars.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(14, 6), constrained_layout=True)
    rtt = [app_metrics.to_float(row.get("avg_round_trip_ms", "")) for row in rows]
    net = [app_metrics.to_float(row.get("avg_network_queue_ms", "")) for row in rows]
    payload = [app_metrics.to_float(row.get("avg_feature_payload_mb", "")) for row in rows]
    ax1.plot(x, rtt, marker="o", label="avg RTT")
    ax1.plot(x, net, marker="o", label="avg RTT - back_ms")
    ax1.set_ylabel("latency (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, payload, color="tab:green", marker="s", linestyle="--", label="payload MB/frame")
    ax2.set_ylabel("payload MB/frame")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Latency versus feature payload")
    fig.savefig(out_dir / "latency_payload_comparison.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = app_metrics.load_metrics(Path(args.root).expanduser().resolve())
    selected_groups = [str(group).strip() for group in args.run_group if str(group).strip()]
    summaries: List[Dict[str, object]] = []
    for run_group in selected_groups:
        group_rows = [row for row in rows if row.get("run_group") == run_group]
        if not group_rows:
            print(f"[warn] no rows found for run_group={run_group}", file=sys.stderr)
            continue
        for stream_id, stream_rows in sorted(app_metrics.group_rows(group_rows, "stream_id").items()):
            summaries.append(summarize_stream(run_group, stream_id, stream_rows))

    if not summaries:
        raise SystemExit("No matching run groups found.")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        safe_name = app_metrics.clean_token("_vs_".join(selected_groups[:3]), "comparison")
        out_dir = DEFAULT_OUTPUT_ROOT / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "combined_experiment_summary.csv", summaries)
    if not args.no_plots:
        plot(summaries, out_dir)
    print(f"Wrote: {out_dir / 'combined_experiment_summary.csv'}")
    if not args.no_plots:
        print(f"Wrote: {out_dir / 'latency_component_bars.png'}")
        print(f"Wrote: {out_dir / 'latency_payload_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

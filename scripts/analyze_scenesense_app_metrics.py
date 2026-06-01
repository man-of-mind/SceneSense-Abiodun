#!/usr/bin/env python3
"""Summarize and plot SceneSense application metrics CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_runs"
DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_analysis"
DEFAULT_NETWORK_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_network"
MPLCONFIG_DIR = Path("/tmp/scenesense_mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

SUMMARY_FIELDS = (
    "run_group",
    "stream_id",
    "transport_label",
    "source_csv_count",
    "rows",
    "received",
    "missed",
    "receive_rate",
    "timeout_rate",
    "duration_s",
    "approx_fps",
    "avg_round_trip_ms",
    "p50_round_trip_ms",
    "p95_round_trip_ms",
    "avg_front_ms",
    "p95_front_ms",
    "avg_back_ms",
    "p95_back_ms",
    "avg_feature_payload_bytes",
    "p95_feature_payload_bytes",
    "avg_feature_payload_mb",
    "feature_goodput_mbps",
    "compression_ratio",
    "avg_result_payload_bytes",
    "avg_feature_chunks",
    "avg_object_count",
    "avg_segmentation_class_count",
    "avg_radar_projected_points",
    "max_spatial_map_dropped_packets",
)

NETWORK_SUMMARY_FIELDS = (
    "run_group",
    "iface",
    "iface_label",
    "source_csv_count",
    "samples",
    "duration_s",
    "avg_tx_mbps",
    "p95_tx_mbps",
    "max_tx_mbps",
    "avg_rx_mbps",
    "p95_rx_mbps",
    "max_rx_mbps",
    "tx_bytes_delta",
    "rx_bytes_delta",
    "tx_packets_delta",
    "rx_packets_delta",
    "tx_drops_delta",
    "rx_drops_delta",
    "tx_errors_delta",
    "rx_errors_delta",
    "ping_attempts",
    "ping_success_rate",
    "avg_ping_rtt_ms",
    "p95_ping_rtt_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SceneSense application metrics. The helper scans per-stream "
            "CSV files, groups them by run_group, writes summary CSV/Markdown, "
            "and optionally saves plots."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_RUN_ROOT),
        help="Root containing scenesense run folders.",
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Run group to analyze. Defaults to the latest discovered group.",
    )
    parser.add_argument(
        "--transport-label",
        default="",
        help="Optional transport label filter, for example loopback or oai.",
    )
    parser.add_argument(
        "--stream-id",
        action="append",
        default=[],
        help="Optional stream id filter. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for summary files and plots. Defaults under metrics_logs/scenesense_analysis/.",
    )
    parser.add_argument(
        "--network-root",
        default=str(DEFAULT_NETWORK_ROOT),
        help="Root containing optional SceneSense network metrics grouped by run_group.",
    )
    parser.add_argument(
        "--network-run-group",
        default="",
        help=(
            "Optional network metrics run_group to load when it differs from the "
            "application run_group. Useful for rescuing mismatched manual labels."
        ),
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Do not load optional network metrics.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List discovered run groups and exit.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV/Markdown summaries without matplotlib plots.",
    )
    return parser.parse_args()


def to_float(value: object) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def to_int(value: object) -> int:
    number = to_float(value)
    return int(number) if math.isfinite(number) else 0


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_token(value: object, default: str = "run") -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value or ""))
    token = token.strip("_")
    return token or default


def finite_values(rows: Iterable[Dict[str, str]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = to_float(row.get(field, ""))
        if math.isfinite(value):
            values.append(value)
    return values


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


def parse_time_key(row: Dict[str, str], csv_path: Path) -> str:
    wall_time = str(row.get("wall_time_iso", "")).strip()
    if wall_time:
        return wall_time
    try:
        return datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def load_metrics(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(root.rglob("streams/*_metrics.csv")):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for raw_row in reader:
                    row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
                    source_run_dir = csv_path.parent.parent
                    row["_source_csv"] = str(csv_path)
                    row["_source_run_dir"] = str(source_run_dir)
                    row["_time_key"] = parse_time_key(row, csv_path)
                    if not row.get("run_id"):
                        row["run_id"] = source_run_dir.name
                    if not row.get("run_group"):
                        row["run_group"] = row.get("run_id") or source_run_dir.name
                    if not row.get("stream_id"):
                        row["stream_id"] = csv_path.stem.replace("_metrics", "")
                    rows.append(row)
        except (OSError, csv.Error) as exc:
            print(f"[warn] skipped {csv_path}: {exc}", file=sys.stderr)
    return rows


def load_network_metrics(root: Path, run_group: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not root.exists():
        return rows
    for csv_path in sorted(root.rglob("network_timeseries.csv")):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for raw_row in reader:
                    row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
                    if row.get("run_group") != run_group:
                        continue
                    row["_source_csv"] = str(csv_path)
                    row["_time_key"] = str(row.get("wall_time_iso", "")).strip()
                    rows.append(row)
        except (OSError, csv.Error) as exc:
            print(f"[warn] skipped network metrics {csv_path}: {exc}", file=sys.stderr)
    return rows


def filter_rows(rows: Iterable[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    stream_filter = set(args.stream_id or [])
    filtered: List[Dict[str, str]] = []
    for row in rows:
        if args.transport_label and row.get("transport_label") != args.transport_label:
            continue
        if stream_filter and row.get("stream_id") not in stream_filter:
            continue
        filtered.append(row)
    return filtered


def group_rows(rows: Iterable[Dict[str, str]], field: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "") or "unknown"].append(row)
    return dict(grouped)


def latest_run_group(rows: Sequence[Dict[str, str]]) -> Optional[str]:
    grouped = group_rows(rows, "run_group")
    if not grouped:
        return None
    return max(grouped, key=lambda key: max((row.get("_time_key", "") for row in grouped[key]), default=""))


def group_overview(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    overview: List[Dict[str, object]] = []
    for run_group, group in sorted(group_rows(rows, "run_group").items()):
        streams = sorted({row.get("stream_id", "unknown") for row in group})
        received = sum(1 for row in group if to_bool(row.get("result_received", "")))
        total = len(group)
        overview.append(
            {
                "run_group": run_group,
                "streams": ",".join(streams),
                "rows": total,
                "receive_rate": received / total if total else float("nan"),
                "first_time": min((row.get("_time_key", "") for row in group), default=""),
                "last_time": max((row.get("_time_key", "") for row in group), default=""),
            }
        )
    return overview


def summarize_stream(run_group: str, stream_id: str, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    received_rows = [row for row in rows if to_bool(row.get("result_received", ""))]
    received = len(received_rows)
    missed = total - received
    elapsed = finite_values(rows, "elapsed_s")
    duration_s = max(elapsed) - min(elapsed) if len(elapsed) >= 2 else (max(elapsed) if elapsed else float("nan"))
    duration_s = duration_s if math.isfinite(duration_s) and duration_s > 0 else float("nan")

    rtt = finite_values(received_rows, "round_trip_ms")
    front = finite_values(rows, "front_ms")
    back = finite_values(received_rows, "back_ms")
    feature_payload = finite_values(rows, "feature_payload_bytes")
    feature_uncompressed = finite_values(rows, "feature_payload_bytes_uncompressed")
    result_payload = finite_values(received_rows, "result_payload_bytes_estimate")
    chunks = finite_values(rows, "feature_payload_chunks")
    object_counts = finite_values(received_rows, "object_count")
    segmentation_counts = finite_values(received_rows, "segmentation_class_count")
    radar_points = finite_values(rows, "radar_projected_points")
    dropped = finite_values(rows, "spatial_map_dropped_packets")

    total_feature_bytes = sum(feature_payload)
    total_uncompressed_bytes = sum(feature_uncompressed)
    feature_goodput_mbps = (
        total_feature_bytes * 8.0 / duration_s / 1_000_000.0
        if math.isfinite(duration_s) and duration_s > 0
        else float("nan")
    )
    compression_ratio = (
        total_uncompressed_bytes / total_feature_bytes
        if total_feature_bytes > 0 and total_uncompressed_bytes > 0
        else float("nan")
    )

    transport_labels = sorted({row.get("transport_label", "") for row in rows if row.get("transport_label")})
    source_csvs = sorted({row.get("_source_csv", "") for row in rows if row.get("_source_csv")})

    return {
        "run_group": run_group,
        "stream_id": stream_id,
        "transport_label": ",".join(transport_labels),
        "source_csv_count": len(source_csvs),
        "rows": total,
        "received": received,
        "missed": missed,
        "receive_rate": received / total if total else float("nan"),
        "timeout_rate": missed / total if total else float("nan"),
        "duration_s": duration_s,
        "approx_fps": total / duration_s if math.isfinite(duration_s) and duration_s > 0 else float("nan"),
        "avg_round_trip_ms": mean(rtt),
        "p50_round_trip_ms": percentile(rtt, 50),
        "p95_round_trip_ms": percentile(rtt, 95),
        "avg_front_ms": mean(front),
        "p95_front_ms": percentile(front, 95),
        "avg_back_ms": mean(back),
        "p95_back_ms": percentile(back, 95),
        "avg_feature_payload_bytes": mean(feature_payload),
        "p95_feature_payload_bytes": percentile(feature_payload, 95),
        "avg_feature_payload_mb": mean(feature_payload) / 1_000_000.0 if feature_payload else float("nan"),
        "feature_goodput_mbps": feature_goodput_mbps,
        "compression_ratio": compression_ratio,
        "avg_result_payload_bytes": mean(result_payload),
        "avg_feature_chunks": mean(chunks),
        "avg_object_count": mean(object_counts),
        "avg_segmentation_class_count": mean(segmentation_counts),
        "avg_radar_projected_points": mean(radar_points),
        "max_spatial_map_dropped_packets": max(dropped) if dropped else 0,
    }


def summarize_group(run_group: str, rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for stream_id, stream_rows in sorted(group_rows(rows, "stream_id").items()):
        summaries.append(summarize_stream(run_group, stream_id, stream_rows))
    return summaries


def sorted_by_elapsed(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda row: (to_float(row.get("elapsed_s", "")), row.get("_time_key", "")))


def counter_delta(rows: Sequence[Dict[str, str]], field: str) -> int:
    ordered = sorted_by_elapsed(rows)
    values = [to_float(row.get(field, "")) for row in ordered]
    values = [value for value in values if math.isfinite(value)]
    if len(values) < 2:
        return 0
    return max(0, int(values[-1]) - int(values[0]))


def summarize_network_iface(
    run_group: str,
    iface_key: str,
    rows: Sequence[Dict[str, str]],
) -> Dict[str, object]:
    active_rows = [row for row in rows if to_bool(row.get("iface_up", ""))]
    elapsed = finite_values(active_rows, "elapsed_s")
    duration_s = max(elapsed) - min(elapsed) if len(elapsed) >= 2 else float("nan")
    tx = finite_values(active_rows, "tx_bitrate_mbps")
    rx = finite_values(active_rows, "rx_bitrate_mbps")
    ping_attempts = [row for row in active_rows if str(row.get("ping_ok", "")).strip() != ""]
    ping_success = [row for row in ping_attempts if to_bool(row.get("ping_ok", ""))]
    ping_rtt = finite_values(ping_success, "ping_rtt_ms")
    source_csvs = sorted({row.get("_source_csv", "") for row in rows if row.get("_source_csv")})
    labels = sorted({row.get("iface_label", "") for row in rows if row.get("iface_label")})

    return {
        "run_group": run_group,
        "iface": iface_key,
        "iface_label": labels[0] if labels else iface_key,
        "source_csv_count": len(source_csvs),
        "samples": len(active_rows),
        "duration_s": duration_s,
        "avg_tx_mbps": mean(tx),
        "p95_tx_mbps": percentile(tx, 95),
        "max_tx_mbps": max(tx) if tx else float("nan"),
        "avg_rx_mbps": mean(rx),
        "p95_rx_mbps": percentile(rx, 95),
        "max_rx_mbps": max(rx) if rx else float("nan"),
        "tx_bytes_delta": counter_delta(active_rows, "tx_bytes"),
        "rx_bytes_delta": counter_delta(active_rows, "rx_bytes"),
        "tx_packets_delta": counter_delta(active_rows, "tx_packets"),
        "rx_packets_delta": counter_delta(active_rows, "rx_packets"),
        "tx_drops_delta": counter_delta(active_rows, "tx_dropped"),
        "rx_drops_delta": counter_delta(active_rows, "rx_dropped"),
        "tx_errors_delta": counter_delta(active_rows, "tx_errors"),
        "rx_errors_delta": counter_delta(active_rows, "rx_errors"),
        "ping_attempts": len(ping_attempts),
        "ping_success_rate": (
            len(ping_success) / len(ping_attempts) if ping_attempts else float("nan")
        ),
        "avg_ping_rtt_ms": mean(ping_rtt),
        "p95_ping_rtt_ms": percentile(ping_rtt, 95),
    }


def summarize_network_group(run_group: str, rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for iface, iface_rows in sorted(group_rows(rows, "iface").items()):
        summaries.append(summarize_network_iface(run_group, iface, iface_rows))
    return summaries


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def write_combined_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    ignored = {"_source_csv", "_source_run_dir", "_time_key"}
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key in ignored:
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(path, fieldnames, rows)


def write_markdown_report(
    path: Path,
    run_group: str,
    summaries: Sequence[Dict[str, object]],
    network_summaries: Sequence[Dict[str, object]] = (),
) -> None:
    columns = (
        "stream_id",
        "rows",
        "receive_rate",
        "timeout_rate",
        "avg_round_trip_ms",
        "p95_round_trip_ms",
        "avg_feature_payload_mb",
        "feature_goodput_mbps",
        "approx_fps",
    )
    lines = [
        f"# SceneSense Application Metrics: {run_group}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(format_value(summary.get(column, "")) for column in columns) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `receive_rate` is the fraction of frames with a back-half result.")
    lines.append("- `timeout_rate` is the fraction of frames without a result before the front-half timeout.")
    lines.append("- `feature_goodput_mbps` is application feature bytes divided by stream duration.")
    if network_summaries:
        network_columns = (
            "iface_label",
            "samples",
            "avg_tx_mbps",
            "p95_tx_mbps",
            "avg_rx_mbps",
            "p95_rx_mbps",
            "ping_success_rate",
            "avg_ping_rtt_ms",
            "tx_drops_delta",
            "rx_drops_delta",
        )
        lines.extend(
            [
                "",
                "## Network Metrics",
                "",
                "| " + " | ".join(network_columns) + " |",
                "| " + " | ".join("---" for _ in network_columns) + " |",
            ]
        )
        for summary in network_summaries:
            lines.append(
                "| "
                + " | ".join(format_value(summary.get(column, "")) for column in network_columns)
                + " |"
            )
        lines.append("")
        lines.append("Network note: UE tunnel `tx` is approximately UE uplink traffic; `rx` is return/downlink traffic.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    run_group: str,
    rows: Sequence[Dict[str, str]],
    summaries: Sequence[Dict[str, object]],
    out_dir: Path,
    network_rows: Sequence[Dict[str, str]] = (),
    network_summaries: Sequence[Dict[str, object]] = (),
) -> List[Path]:
    try:
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipped plots: {exc}", file=sys.stderr)
        return []

    out_paths: List[Path] = []
    by_stream = group_rows(rows, "stream_id")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_rtt, ax_payload, ax_infer, ax_receive = axes.ravel()

    for stream_id, stream_rows in sorted(by_stream.items()):
        ordered = sorted_by_elapsed(stream_rows)
        elapsed = [to_float(row.get("elapsed_s", "")) for row in ordered]
        rtt = [to_float(row.get("round_trip_ms", "")) for row in ordered]
        payload_mb = [to_float(row.get("feature_payload_bytes", "")) / 1_000_000.0 for row in ordered]
        front = [to_float(row.get("front_ms", "")) for row in ordered]
        back = [to_float(row.get("back_ms", "")) for row in ordered]

        ax_rtt.plot(elapsed, rtt, marker=".", linewidth=1.0, label=stream_id)
        ax_payload.plot(elapsed, payload_mb, marker=".", linewidth=1.0, label=stream_id)
        ax_infer.plot(elapsed, front, linewidth=1.0, label=f"{stream_id} front")
        ax_infer.plot(elapsed, back, linewidth=1.0, linestyle="--", label=f"{stream_id} back")

    stream_labels = [str(summary["stream_id"]) for summary in summaries]
    receive_rates = [to_float(summary.get("receive_rate", "")) * 100.0 for summary in summaries]
    ax_receive.bar(stream_labels, receive_rates)
    ax_receive.set_ylim(0, 105)
    ax_receive.tick_params(axis="x", rotation=20)

    ax_rtt.set_title("Round-trip latency")
    ax_rtt.set_xlabel("elapsed seconds")
    ax_rtt.set_ylabel("ms")
    ax_rtt.legend(fontsize=8)

    ax_payload.set_title("Feature payload")
    ax_payload.set_xlabel("elapsed seconds")
    ax_payload.set_ylabel("MB/frame")
    ax_payload.legend(fontsize=8)

    ax_infer.set_title("Front/back inference time")
    ax_infer.set_xlabel("elapsed seconds")
    ax_infer.set_ylabel("ms")
    ax_infer.legend(fontsize=7)

    ax_receive.set_title("Result receive rate")
    ax_receive.set_ylabel("% frames with result")

    fig.suptitle(f"SceneSense application metrics: {run_group}")
    out_path = out_dir / "application_timeseries.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    bar_specs = (
        ("p95_round_trip_ms", "p95 RTT", "ms"),
        ("avg_feature_payload_mb", "avg feature payload", "MB/frame"),
        ("feature_goodput_mbps", "feature goodput", "Mbps"),
        ("timeout_rate", "timeout rate", "fraction"),
    )
    for ax, (field, title, ylabel) in zip(axes.ravel(), bar_specs):
        values = [to_float(summary.get(field, "")) for summary in summaries]
        if field == "timeout_rate":
            values = [value * 100.0 for value in values]
            ylabel = "% frames"
        ax.bar(stream_labels, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"SceneSense stream comparison: {run_group}")
    out_path = out_dir / "application_summary_bars.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)

    if network_rows:
        by_iface = group_rows(network_rows, "iface")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        ax_rate, ax_ping, ax_avg, ax_health = axes.ravel()

        for iface, iface_rows in sorted(by_iface.items()):
            ordered = sorted_by_elapsed(iface_rows)
            label = next((row.get("iface_label", "") for row in ordered if row.get("iface_label")), iface)
            elapsed = [to_float(row.get("elapsed_s", "")) for row in ordered]
            tx = [to_float(row.get("tx_bitrate_mbps", "")) for row in ordered]
            rx = [to_float(row.get("rx_bitrate_mbps", "")) for row in ordered]
            ping_elapsed = [
                to_float(row.get("elapsed_s", ""))
                for row in ordered
                if str(row.get("ping_ok", "")).lower() == "true"
            ]
            ping_rtt = [
                to_float(row.get("ping_rtt_ms", ""))
                for row in ordered
                if str(row.get("ping_ok", "")).lower() == "true"
            ]
            ax_rate.plot(elapsed, tx, linewidth=1.0, label=f"{label} tx/uplink")
            ax_rate.plot(elapsed, rx, linewidth=1.0, linestyle="--", label=f"{label} rx/downlink")
            if ping_rtt:
                ax_ping.plot(ping_elapsed, ping_rtt, marker=".", linewidth=1.0, label=label)

        labels = [str(summary["iface_label"]) for summary in network_summaries]
        avg_tx = [to_float(summary.get("avg_tx_mbps", "")) for summary in network_summaries]
        avg_rx = [to_float(summary.get("avg_rx_mbps", "")) for summary in network_summaries]
        x_values = list(range(len(labels)))
        width = 0.35
        ax_avg.bar([x - width / 2 for x in x_values], avg_tx, width=width, label="tx/uplink")
        ax_avg.bar([x + width / 2 for x in x_values], avg_rx, width=width, label="rx/downlink")
        ax_avg.set_xticks(x_values)
        ax_avg.set_xticklabels(labels, rotation=20)
        ax_avg.legend(fontsize=8)

        tx_drops = [to_float(summary.get("tx_drops_delta", "")) for summary in network_summaries]
        rx_drops = [to_float(summary.get("rx_drops_delta", "")) for summary in network_summaries]
        ax_health.bar([x - width / 2 for x in x_values], tx_drops, width=width, label="tx drops")
        ax_health.bar([x + width / 2 for x in x_values], rx_drops, width=width, label="rx drops")
        ax_health.set_xticks(x_values)
        ax_health.set_xticklabels(labels, rotation=20)
        ax_health.legend(fontsize=8)

        ax_rate.set_title("UE tunnel bitrate")
        ax_rate.set_xlabel("elapsed seconds")
        ax_rate.set_ylabel("Mbps")
        ax_rate.legend(fontsize=8)

        ax_ping.set_title("Ping RTT")
        ax_ping.set_xlabel("elapsed seconds")
        ax_ping.set_ylabel("ms")
        if any(to_float(summary.get("ping_attempts", 0)) > 0 for summary in network_summaries):
            ax_ping.legend(fontsize=8)

        ax_avg.set_title("Average tunnel bitrate")
        ax_avg.set_ylabel("Mbps")

        ax_health.set_title("Tunnel drops")
        ax_health.set_ylabel("packets")

        fig.suptitle(f"SceneSense network metrics: {run_group}")
        out_path = out_dir / "network_timeseries.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def print_group_overview(overview: Sequence[Dict[str, object]]) -> None:
    if not overview:
        print("No SceneSense application metrics groups found.")
        return
    print("Discovered run groups:")
    for item in overview:
        rate = to_float(item.get("receive_rate", "")) * 100.0
        print(
            f"- {item['run_group']} | streams={item['streams']} | rows={item['rows']} | "
            f"receive={rate:.1f}% | {item['first_time']} -> {item['last_time']}"
        )


def print_summary(summaries: Sequence[Dict[str, object]]) -> None:
    for summary in summaries:
        receive = to_float(summary.get("receive_rate", "")) * 100.0
        timeout = to_float(summary.get("timeout_rate", "")) * 100.0
        avg_rtt = format_value(summary.get("avg_round_trip_ms", ""))
        p95_rtt = format_value(summary.get("p95_round_trip_ms", ""))
        payload = format_value(summary.get("avg_feature_payload_mb", ""))
        print(
            f"- {summary['stream_id']}: rows={summary['rows']}, receive={receive:.1f}%, "
            f"timeout={timeout:.1f}%, avg_rtt={avg_rtt} ms, p95_rtt={p95_rtt} ms, "
            f"payload={payload} MB/frame"
        )


def print_network_summary(summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        return
    print("Network metrics:")
    for summary in summaries:
        avg_tx = format_value(summary.get("avg_tx_mbps", ""))
        avg_rx = format_value(summary.get("avg_rx_mbps", ""))
        p95_tx = format_value(summary.get("p95_tx_mbps", ""))
        p95_rx = format_value(summary.get("p95_rx_mbps", ""))
        ping_rate = to_float(summary.get("ping_success_rate", "")) * 100.0
        ping_text = f", ping_success={ping_rate:.1f}%" if math.isfinite(ping_rate) else ""
        print(
            f"- {summary['iface_label']} ({summary['iface']}): samples={summary['samples']}, "
            f"avg_tx={avg_tx} Mbps, p95_tx={p95_tx} Mbps, "
            f"avg_rx={avg_rx} Mbps, p95_rx={p95_rx} Mbps{ping_text}"
        )


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = filter_rows(load_metrics(root), args)
    if not rows:
        print(f"No SceneSense application metrics CSVs found under {root}.")
        return 0 if args.list_groups else 1

    if args.list_groups:
        print_group_overview(group_overview(rows))
        return 0

    run_group = args.run_group.strip() or latest_run_group(rows)
    if not run_group:
        print(f"No run group found under {root}.")
        return 1

    selected = [row for row in rows if row.get("run_group") == run_group]
    if not selected:
        print(f"No rows found for run_group={run_group!r} under {root}.")
        return 1

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_ANALYSIS_ROOT / clean_token(run_group)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = summarize_group(run_group, selected)
    network_rows: List[Dict[str, str]] = []
    network_summaries: List[Dict[str, object]] = []
    network_run_group = args.network_run_group.strip() or run_group
    if not args.skip_network:
        network_rows = load_network_metrics(Path(args.network_root).expanduser().resolve(), network_run_group)
        if network_rows:
            network_summaries = summarize_network_group(network_run_group, network_rows)
            write_csv(out_dir / "network_summary.csv", NETWORK_SUMMARY_FIELDS, network_summaries)
            write_combined_csv(out_dir / "network_combined_rows.csv", network_rows)

    write_csv(out_dir / "application_summary.csv", SUMMARY_FIELDS, summaries)
    write_combined_csv(out_dir / "application_combined_rows.csv", selected)
    write_markdown_report(out_dir / "application_summary.md", run_group, summaries, network_summaries)
    plot_paths = (
        []
        if args.no_plots
        else make_plots(
            run_group,
            selected,
            summaries,
            out_dir,
            network_rows=network_rows,
            network_summaries=network_summaries,
        )
    )

    print(f"Analyzed run_group={run_group}")
    if network_summaries and network_run_group != run_group:
        print(f"Loaded network_run_group={network_run_group}")
    print(f"Output directory: {out_dir}")
    print_summary(summaries)
    print_network_summary(network_summaries)
    print(f"Wrote: {out_dir / 'application_summary.csv'}")
    print(f"Wrote: {out_dir / 'application_summary.md'}")
    if network_summaries:
        print(f"Wrote: {out_dir / 'network_summary.csv'}")
    for path in plot_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

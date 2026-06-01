#!/usr/bin/env python3
"""Sample lightweight OAI UE tunnel metrics during a SceneSense run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "metrics_logs" / "scenesense_network"
SYS_CLASS_NET = Path("/sys/class/net")

STAT_NAMES = (
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
    "rx_dropped",
    "tx_dropped",
    "rx_errors",
    "tx_errors",
)

CSV_FIELDS = (
    "wall_time_iso",
    "elapsed_s",
    "sample_index",
    "run_group",
    "iface",
    "iface_label",
    "iface_up",
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
    "rx_dropped",
    "tx_dropped",
    "rx_errors",
    "tx_errors",
    "rx_bitrate_mbps",
    "tx_bitrate_mbps",
    "rx_packet_rate_pps",
    "tx_packet_rate_pps",
    "rx_drop_delta",
    "tx_drop_delta",
    "rx_error_delta",
    "tx_error_delta",
    "ping_host",
    "ping_ok",
    "ping_rtt_ms",
)

SUMMARY_FIELDS = (
    "run_group",
    "iface",
    "iface_label",
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
            "Sample Linux UE tunnel counters into a SceneSense network metrics CSV. "
            "For UE tunnels, tx is approximately UE uplink traffic and rx is "
            "approximately downlink/return traffic."
        )
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Run group shared with application metrics. Defaults to a 10-minute oai bucket.",
    )
    parser.add_argument(
        "--transport-label",
        default="oai",
        help="Label used when auto-generating a run group.",
    )
    parser.add_argument(
        "--interface",
        action="append",
        default=[],
        metavar="IFACE[:LABEL]",
        help=(
            "Interface to sample. Defaults to oaitun_ue1:ue1 and oaitun_ue2:ue2. "
            "Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Sample interval in seconds.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Optional run duration. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root for network metrics grouped by run_group.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--ping-host",
        default="",
        help="Optional host/IP to ping through each interface, for example 192.168.70.135.",
    )
    parser.add_argument(
        "--ping-every-s",
        type=float,
        default=5.0,
        help="Ping interval per interface. Ignored when --ping-host is empty.",
    )
    parser.add_argument(
        "--ping-timeout-s",
        type=float,
        default=1.0,
        help="Per-ping timeout.",
    )
    return parser.parse_args()


def clean_token(value: object, default: str = "run") -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value or ""))
    token = token.strip("_")
    return token or default


def default_run_group(transport_label: str) -> str:
    now = datetime.now()
    bucket_minute = (now.minute // 10) * 10
    bucket = now.replace(minute=bucket_minute, second=0, microsecond=0)
    return f"{bucket:%Y%m%d_%H%M}_{clean_token(transport_label)}"


def parse_interfaces(values: Sequence[str]) -> List[Tuple[str, str]]:
    if not values:
        return [("oaitun_ue1", "ue1"), ("oaitun_ue2", "ue2")]
    parsed: List[Tuple[str, str]] = []
    for raw in values:
        iface, _, label = str(raw).partition(":")
        iface = iface.strip()
        label = label.strip() or iface
        if iface:
            parsed.append((iface, label))
    return parsed


def read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_interface_stats(iface: str) -> Optional[Dict[str, int]]:
    stats_dir = SYS_CLASS_NET / iface / "statistics"
    if not stats_dir.exists():
        return None
    stats: Dict[str, int] = {}
    for name in STAT_NAMES:
        value = read_int(stats_dir / name)
        if value is None:
            return None
        stats[name] = value
    return stats


def rate_mbps(delta_value: Optional[int], delta_s: float) -> Optional[float]:
    if delta_value is None or delta_s <= 0:
        return None
    return float(delta_value) * 8.0 / delta_s / 1_000_000.0


def rate_pps(delta_value: Optional[int], delta_s: float) -> Optional[float]:
    if delta_value is None or delta_s <= 0:
        return None
    return float(delta_value) / delta_s


def stat_delta(
    current: Optional[Dict[str, int]],
    previous: Optional[Dict[str, int]],
    name: str,
) -> Optional[int]:
    if current is None or previous is None:
        return None
    return max(0, int(current[name]) - int(previous[name]))


PING_RE = re.compile(r"time[=<]([0-9.]+)\s*ms")


def ping_once(iface: str, host: str, timeout_s: float) -> Tuple[Optional[bool], Optional[float]]:
    if not host:
        return None, None
    command = [
        "ping",
        "-I",
        iface,
        "-c",
        "1",
        "-W",
        str(max(1, int(math.ceil(timeout_s)))),
        host,
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1.0, timeout_s + 0.5),
            check=False,
        )
    except Exception:
        return False, None
    if result.returncode != 0:
        return False, None
    match = PING_RE.search(result.stdout)
    if match:
        return True, float(match.group(1))
    return True, None


def value_or_blank(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def numeric_values(rows: Iterable[Dict[str, object]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        try:
            value = float(str(row.get(field, "")).strip())
        except (TypeError, ValueError):
            continue
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


def counter_delta(rows: Sequence[Dict[str, object]], field: str) -> int:
    values = numeric_values(rows, field)
    if len(values) < 2:
        return 0
    return max(0, int(values[-1]) - int(values[0]))


def summarize_rows(run_group: str, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_iface: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("iface", "")), str(row.get("iface_label", "")))
        by_iface.setdefault(key, []).append(row)

    summaries: List[Dict[str, object]] = []
    for (iface, label), iface_rows in sorted(by_iface.items()):
        active_rows = [row for row in iface_rows if str(row.get("iface_up", "")).lower() == "true"]
        elapsed = numeric_values(active_rows, "elapsed_s")
        duration_s = max(elapsed) - min(elapsed) if len(elapsed) >= 2 else float("nan")
        tx = numeric_values(active_rows, "tx_bitrate_mbps")
        rx = numeric_values(active_rows, "rx_bitrate_mbps")
        ping_attempts = [
            row for row in active_rows if str(row.get("ping_ok", "")).strip() != ""
        ]
        ping_success = [
            row for row in ping_attempts if str(row.get("ping_ok", "")).lower() == "true"
        ]
        ping_rtt = numeric_values(ping_success, "ping_rtt_ms")
        summaries.append(
            {
                "run_group": run_group,
                "iface": iface,
                "iface_label": label,
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
        )
    return summaries


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def write_manifest(
    path: Path,
    *,
    run_group: str,
    interfaces: Sequence[Tuple[str, str]],
    interval_s: float,
    duration_s: float,
    ping_host: str,
    csv_path: Path,
    summary_path: Path,
) -> None:
    manifest = {
        "schema": "scenesense_oai_network_metrics.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_group": run_group,
        "interfaces": [{"iface": iface, "label": label} for iface, label in interfaces],
        "interval_s": interval_s,
        "duration_s": duration_s,
        "ping_host": ping_host,
        "direction_note": "On UE tunnel interfaces, tx is approximately UE uplink and rx is return/downlink traffic.",
        "output_files": {
            "network_timeseries_csv": str(csv_path),
            "network_summary_csv": str(summary_path),
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_row(
    *,
    now: float,
    start_time: float,
    sample_index: int,
    run_group: str,
    iface: str,
    label: str,
    stats: Optional[Dict[str, int]],
    previous_stats: Optional[Dict[str, int]],
    previous_time: Optional[float],
    ping_host: str,
    ping_ok: Optional[bool],
    ping_rtt_ms: Optional[float],
) -> Dict[str, object]:
    delta_s = now - previous_time if previous_time is not None else 0.0
    row: Dict[str, object] = {
        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_s": now - start_time,
        "sample_index": sample_index,
        "run_group": run_group,
        "iface": iface,
        "iface_label": label,
        "iface_up": stats is not None,
        "ping_host": ping_host,
        "ping_ok": ping_ok,
        "ping_rtt_ms": ping_rtt_ms,
    }
    if stats is None:
        for field in CSV_FIELDS:
            row.setdefault(field, "")
        return row

    for name in STAT_NAMES:
        row[name] = stats[name]

    rx_bytes_delta = stat_delta(stats, previous_stats, "rx_bytes")
    tx_bytes_delta = stat_delta(stats, previous_stats, "tx_bytes")
    rx_packets_delta = stat_delta(stats, previous_stats, "rx_packets")
    tx_packets_delta = stat_delta(stats, previous_stats, "tx_packets")

    row.update(
        {
            "rx_bitrate_mbps": rate_mbps(rx_bytes_delta, delta_s),
            "tx_bitrate_mbps": rate_mbps(tx_bytes_delta, delta_s),
            "rx_packet_rate_pps": rate_pps(rx_packets_delta, delta_s),
            "tx_packet_rate_pps": rate_pps(tx_packets_delta, delta_s),
            "rx_drop_delta": stat_delta(stats, previous_stats, "rx_dropped"),
            "tx_drop_delta": stat_delta(stats, previous_stats, "tx_dropped"),
            "rx_error_delta": stat_delta(stats, previous_stats, "rx_errors"),
            "tx_error_delta": stat_delta(stats, previous_stats, "tx_errors"),
        }
    )
    return {field: value_or_blank(row.get(field, "")) for field in CSV_FIELDS}


def main() -> int:
    args = parse_args()
    run_group = args.run_group.strip() or default_run_group(args.transport_label)
    interfaces = parse_interfaces(args.interface)
    interval_s = max(0.1, float(args.interval_s))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.output_root).expanduser().resolve() / clean_token(run_group)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "network_timeseries.csv"
    summary_path = output_dir / "network_summary.csv"
    manifest_path = output_dir / "network_manifest.json"

    rows: List[Dict[str, object]] = []
    previous_stats: Dict[str, Optional[Dict[str, int]]] = {}
    previous_times: Dict[str, Optional[float]] = {}
    next_ping_time: Dict[str, float] = {iface: 0.0 for iface, _label in interfaces}
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(f"[network] run_group={run_group}")
    print(f"[network] output={csv_path}")
    print(f"[network] interfaces={', '.join(f'{iface}:{label}' for iface, label in interfaces)}")
    if args.ping_host:
        print(f"[network] ping_host={args.ping_host} every {args.ping_every_s}s")
    print("[network] Press Ctrl+C to stop.")

    start_time = time.monotonic()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        sample_index = 0
        try:
            while not stop_requested:
                now = time.monotonic()
                if args.duration_s > 0 and now - start_time >= args.duration_s:
                    break
                for iface, label in interfaces:
                    stats = read_interface_stats(iface)
                    ping_ok: Optional[bool] = None
                    ping_rtt_ms: Optional[float] = None
                    if args.ping_host and now >= next_ping_time.get(iface, 0.0):
                        ping_ok, ping_rtt_ms = ping_once(
                            iface,
                            args.ping_host,
                            float(args.ping_timeout_s),
                        )
                        next_ping_time[iface] = now + max(float(args.ping_every_s), interval_s)
                    row = build_row(
                        now=now,
                        start_time=start_time,
                        sample_index=sample_index,
                        run_group=run_group,
                        iface=iface,
                        label=label,
                        stats=stats,
                        previous_stats=previous_stats.get(iface),
                        previous_time=previous_times.get(iface),
                        ping_host=args.ping_host,
                        ping_ok=ping_ok,
                        ping_rtt_ms=ping_rtt_ms,
                    )
                    rows.append(row)
                    writer.writerow(row)
                    previous_stats[iface] = stats
                    previous_times[iface] = now
                handle.flush()
                sample_index += 1
                sleep_until = start_time + sample_index * interval_s
                time.sleep(max(0.0, sleep_until - time.monotonic()))
        finally:
            duration_s = time.monotonic() - start_time

    summaries = summarize_rows(run_group, rows)
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    write_manifest(
        manifest_path,
        run_group=run_group,
        interfaces=interfaces,
        interval_s=interval_s,
        duration_s=duration_s,
        ping_host=args.ping_host,
        csv_path=csv_path,
        summary_path=summary_path,
    )

    print(f"[network] samples={len(rows)} duration_s={duration_s:.1f}")
    for summary in summaries:
        print(
            "[network] "
            f"{summary['iface_label']} ({summary['iface']}): "
            f"avg_tx={format_value(summary['avg_tx_mbps'])} Mbps, "
            f"avg_rx={format_value(summary['avg_rx_mbps'])} Mbps, "
            f"tx_bytes={summary['tx_bytes_delta']}, rx_bytes={summary['rx_bytes_delta']}"
        )
    print(f"[network] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize TRACTOR raw packet traces for bursty-uplink replay planning.

The TRACTOR raw files are CSVs with columns:
  App name, No., Time, Source, Destination, Protocol, Length

By default this script treats the IP ending in `.250` as the UE/phone address,
matching the downloaded TRACTOR traces. Packets sourced by that IP are counted
as uplink; packets destined to that IP are counted as downlink.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from statistics import median


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def guess_ue_ip(rows: list[dict[str, str]]) -> str:
    ips: dict[str, int] = {}
    for row in rows:
        for col in ("Source", "Destination"):
            ip = row.get(col, "")
            ips[ip] = ips.get(ip, 0) + 1
    ending_250 = [ip for ip in ips if ip.endswith(".250")]
    if ending_250:
        return max(ending_250, key=lambda ip: ips[ip])
    return max(ips, key=ips.get)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def burst_stats(times: list[float], lengths: list[int], gap_s: float) -> tuple[int, float, float, float]:
    if not times:
        return 0, float("nan"), float("nan"), float("nan")
    bursts: list[int] = []
    current = 0
    prev_t = None
    for t, length in sorted(zip(times, lengths)):
        if prev_t is not None and t - prev_t > gap_s:
            bursts.append(current)
            current = 0
        current += length
        prev_t = t
    bursts.append(current)
    bursts_kb = [b / 1024.0 for b in bursts]
    return len(bursts), median(bursts_kb), percentile(bursts_kb, 95), max(bursts_kb)


def one_second_rates(times: list[float], lengths: list[int], start: float) -> tuple[list[float], int | None, float]:
    bins: dict[int, int] = {}
    for t, length in zip(times, lengths):
        sec = int(math.floor(t - start))
        bins[sec] = bins.get(sec, 0) + length
    if not bins:
        return [], None, 0.0
    peak_sec, peak_bytes = max(bins.items(), key=lambda kv: kv[1])
    return [(b * 8.0) / 1e6 for b in bins.values()], peak_sec, (peak_bytes * 8.0) / 1e6


def summarize_file(path: Path, ue_ip: str | None, burst_gap_ms: float) -> dict[str, object]:
    rows = load_rows(path)
    if not rows:
        return {"trace": path.name, "rows": 0}
    ue = ue_ip or guess_ue_ip(rows)
    all_times: list[float] = []
    ul_times: list[float] = []
    ul_lengths: list[int] = []
    dl_lengths: list[int] = []
    all_lengths: list[int] = []
    protocols: dict[str, int] = {}
    apps: dict[str, int] = {}

    for row in rows:
        try:
            t = float(row["Time"])
            length = int(float(row["Length"]))
        except (KeyError, ValueError):
            continue
        src = row.get("Source", "")
        dst = row.get("Destination", "")
        all_times.append(t)
        all_lengths.append(length)
        protocols[row.get("Protocol", "")] = protocols.get(row.get("Protocol", ""), 0) + 1
        apps[row.get("App name", "")] = apps.get(row.get("App name", ""), 0) + 1
        if src == ue:
            ul_times.append(t)
            ul_lengths.append(length)
        elif dst == ue:
            dl_lengths.append(length)

    start = min(all_times) if all_times else 0.0
    end = max(all_times) if all_times else 0.0
    duration = max(end - start, 1e-9)
    ul_bytes = sum(ul_lengths)
    dl_bytes = sum(dl_lengths)
    rates, peak_1s_offset_s, peak_1s_mbps = one_second_rates(ul_times, ul_lengths, start)
    mean_ul_mbps = ul_bytes * 8.0 / duration / 1e6
    p95_1s_mbps = percentile(rates, 95) if rates else 0.0
    sorted_ul = sorted(ul_times)
    gaps_ms = [(b - a) * 1000.0 for a, b in zip(sorted_ul, sorted_ul[1:])]
    burst_count, burst_p50_kb, burst_p95_kb, burst_max_kb = burst_stats(
        ul_times, ul_lengths, burst_gap_ms / 1000.0
    )

    return {
        "trace": path.name,
        "rows": len(rows),
        "ue_ip": ue,
        "duration_s": duration,
        "uplink_packets": len(ul_lengths),
        "downlink_packets": len(dl_lengths),
        "uplink_mb": ul_bytes / 1e6,
        "downlink_mb": dl_bytes / 1e6,
        "uplink_share_bytes": ul_bytes / max(ul_bytes + dl_bytes, 1),
        "mean_ul_mbps": mean_ul_mbps,
        "p95_1s_ul_mbps": p95_1s_mbps,
        "peak_1s_ul_mbps": peak_1s_mbps,
        "peak_1s_offset_s": peak_1s_offset_s if peak_1s_offset_s is not None else float("nan"),
        "peak_to_mean_ul_rate": peak_1s_mbps / mean_ul_mbps if mean_ul_mbps > 0 else float("nan"),
        "packet_len_p50": percentile([float(x) for x in all_lengths], 50),
        "packet_len_p95": percentile([float(x) for x in all_lengths], 95),
        "ul_gap_p50_ms": percentile(gaps_ms, 50),
        "ul_gap_p95_ms": percentile(gaps_ms, 95),
        "ul_gap_max_ms": max(gaps_ms) if gaps_ms else float("nan"),
        "burst_gap_ms": burst_gap_ms,
        "uplink_bursts": burst_count,
        "uplink_burst_p50_kb": burst_p50_kb,
        "uplink_burst_p95_kb": burst_p95_kb,
        "uplink_burst_max_kb": burst_max_kb,
        "top_app": max(apps, key=apps.get) if apps else "",
        "top_protocol": max(protocols, key=protocols.get) if protocols else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="TRACTOR/raw/*.csv")
    ap.add_argument("--out", default="analysis/tractor_trace_summary.csv")
    ap.add_argument("--ue-ip", default=None)
    ap.add_argument("--burst-gap-ms", type=float, default=100.0)
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.raw_glob))
    if not paths:
        raise SystemExit(f"No files matched {args.raw_glob!r}")
    rows = [summarize_file(p, args.ue_ip, args.burst_gap_ms) for p in paths]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

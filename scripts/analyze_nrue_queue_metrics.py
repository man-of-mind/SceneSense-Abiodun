#!/usr/bin/env python3
"""Summarize NR UE RLC/MAC/BSR queue traces for SceneSense OAI diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Iterable, Sequence


ABIODUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TTRACER_ROOT = ABIODUN_DIR / "metrics_logs" / "scenesense_ttracer"

LCG_FIELDS = tuple(f"lcg{i}_bytes" for i in range(8))

WINDOW_FIELDS = (
    "run_group",
    "window_index",
    "window_start_s",
    "window_end_s",
    "rlc_samples",
    "rlc_total_buffer_p50_bytes",
    "rlc_total_buffer_p95_bytes",
    "rlc_total_buffer_max_bytes",
    "bsr_samples",
    "bsr_sent",
    "bsr_sent_rate_hz",
    "bsr_total_lcg_p50_bytes",
    "bsr_total_lcg_p95_bytes",
    "bsr_total_lcg_max_bytes",
    "sdu_bytes",
    "sdu_mbps",
    "sdu_bytes_p50_per_grant",
    "padding_p50_bytes",
    "padding_p95_bytes",
)

SUMMARY_FIELDS = (
    "run_group",
    "duration_s",
    "rlc_samples",
    "rlc_total_buffer_p50_bytes",
    "rlc_total_buffer_p95_bytes",
    "rlc_total_buffer_max_bytes",
    "bsr_samples",
    "bsr_sent",
    "bsr_sent_rate_hz",
    "bsr_total_lcg_p50_bytes",
    "bsr_total_lcg_p95_bytes",
    "bsr_total_lcg_max_bytes",
    "sdu_bytes",
    "sdu_mbps",
    "sdu_bytes_p50_per_grant",
    "padding_p50_bytes",
    "padding_p95_bytes",
    "bsr_type_counts",
)

BSR_TYPE_LABELS = {
    0: "none",
    1: "long",
    2: "short",
    3: "short_trunc",
    4: "long_trunc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert NRUE_MAC_RLC_BUFFER_STATUS.csv and NRUE_MAC_BSR_STATUS.csv "
            "into windowed UE queue/drain metrics."
        )
    )
    parser.add_argument("--run-group", default="", help="Run group under metrics_logs/scenesense_ttracer.")
    parser.add_argument("--root", default=str(DEFAULT_TTRACER_ROOT), help="T-tracer metrics root.")
    parser.add_argument("--rlc-csv", default="", help="Explicit NRUE_MAC_RLC_BUFFER_STATUS.csv path.")
    parser.add_argument("--bsr-csv", default="", help="Explicit NRUE_MAC_BSR_STATUS.csv path.")
    parser.add_argument("--window-s", type=float, default=1.0, help="Aggregation window in seconds; default 1.0.")
    parser.add_argument("--output-dir", default="", help="Output directory; defaults to <run_group>/ue/analysis.")
    return parser.parse_args()


def to_int(value: object) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def parse_time_to_seconds(value: str, previous: float | None) -> float:
    timestamp = datetime.strptime(value.strip(), "%H:%M:%S.%f")
    seconds = (
        timestamp.hour * 3600.0
        + timestamp.minute * 60.0
        + timestamp.second
        + timestamp.microsecond / 1_000_000.0
    )
    if previous is not None and seconds + 12 * 3600 < previous:
        seconds += 24 * 3600
    return seconds


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


def fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_count(value: float) -> str:
    if not math.isfinite(value):
        return ""
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return fmt(value)


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    root = Path(args.root).expanduser().resolve()
    if args.rlc_csv:
        rlc_csv = Path(args.rlc_csv).expanduser().resolve()
    else:
        if not args.run_group:
            raise ValueError("provide --run-group or --rlc-csv/--bsr-csv")
        rlc_csv = root / args.run_group / "ue" / "csv" / "NRUE_MAC_RLC_BUFFER_STATUS.csv"

    if args.bsr_csv:
        bsr_csv = Path(args.bsr_csv).expanduser().resolve()
    else:
        if not args.run_group:
            raise ValueError("provide --run-group or --rlc-csv/--bsr-csv")
        bsr_csv = root / args.run_group / "ue" / "csv" / "NRUE_MAC_BSR_STATUS.csv"

    run_group = args.run_group
    if not run_group:
        try:
            run_group = bsr_csv.parents[2].name
        except IndexError:
            run_group = bsr_csv.stem

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = root / run_group / "ue" / "analysis"
    return rlc_csv, bsr_csv, output_dir, run_group


def load_rlc_samples(path: Path) -> list[dict[str, float]]:
    """Aggregate per-LCID rows into one instantaneous total per UE/frame/slot sample."""
    grouped: dict[tuple[str, int, int, int], dict[str, float]] = {}
    previous: float | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            t = parse_time_to_seconds(raw["time"], previous)
            previous = t
            key = (raw.get("rnti", ""), to_int(raw.get("ue_id")), to_int(raw.get("frame")), to_int(raw.get("slot")))
            sample = grouped.setdefault(key, {"time_s": t, "total_buffer_bytes": 0.0, "lcids": 0.0})
            sample["time_s"] = min(sample["time_s"], t)
            sample["total_buffer_bytes"] += to_int(raw.get("bytes_in_buffer"))
            sample["lcids"] += 1
    return sorted(grouped.values(), key=lambda row: row["time_s"])


def load_bsr_samples(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    previous: float | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            t = parse_time_to_seconds(raw["time"], previous)
            previous = t
            total_lcg = sum(to_int(raw.get(field)) for field in LCG_FIELDS)
            rows.append(
                {
                    "time_s": t,
                    "bsr_type": float(to_int(raw.get("bsr_type"))),
                    "bsr_sent": float(to_int(raw.get("bsr_sent"))),
                    "padding_len": float(to_int(raw.get("padding_len"))),
                    "num_sdus": float(to_int(raw.get("num_sdus"))),
                    "sdu_bytes": float(to_int(raw.get("sdu_bytes"))),
                    "total_lcg_bytes": float(total_lcg),
                }
            )
    return rows


def window_index(time_s: float, start_s: float, window_s: float) -> int:
    return max(0, int((time_s - start_s) // window_s))


def summarize_window(
    run_group: str,
    idx: int,
    window_s: float,
    rlc_rows: Sequence[dict[str, float]],
    bsr_rows: Sequence[dict[str, float]],
) -> dict[str, str]:
    rlc_totals = [row["total_buffer_bytes"] for row in rlc_rows]
    bsr_totals = [row["total_lcg_bytes"] for row in bsr_rows]
    sdu_values = [row["sdu_bytes"] for row in bsr_rows]
    padding_values = [row["padding_len"] for row in bsr_rows]
    sdu_sum = sum(sdu_values)
    bsr_sent = sum(row["bsr_sent"] for row in bsr_rows)
    return {
        "run_group": run_group,
        "window_index": str(idx),
        "window_start_s": fmt(idx * window_s),
        "window_end_s": fmt((idx + 1) * window_s),
        "rlc_samples": str(len(rlc_rows)),
        "rlc_total_buffer_p50_bytes": fmt_count(percentile(rlc_totals, 50)),
        "rlc_total_buffer_p95_bytes": fmt_count(percentile(rlc_totals, 95)),
        "rlc_total_buffer_max_bytes": fmt_count(max(rlc_totals) if rlc_totals else float("nan")),
        "bsr_samples": str(len(bsr_rows)),
        "bsr_sent": fmt_count(bsr_sent),
        "bsr_sent_rate_hz": fmt(bsr_sent / window_s),
        "bsr_total_lcg_p50_bytes": fmt_count(percentile(bsr_totals, 50)),
        "bsr_total_lcg_p95_bytes": fmt_count(percentile(bsr_totals, 95)),
        "bsr_total_lcg_max_bytes": fmt_count(max(bsr_totals) if bsr_totals else float("nan")),
        "sdu_bytes": fmt_count(sdu_sum),
        "sdu_mbps": fmt((sdu_sum * 8.0) / window_s / 1_000_000.0),
        "sdu_bytes_p50_per_grant": fmt_count(percentile(sdu_values, 50)),
        "padding_p50_bytes": fmt_count(percentile(padding_values, 50)),
        "padding_p95_bytes": fmt_count(percentile(padding_values, 95)),
    }


def make_windows(run_group: str, window_s: float, rlc_samples: Sequence[dict[str, float]], bsr_samples: Sequence[dict[str, float]]) -> list[dict[str, str]]:
    times = [row["time_s"] for row in rlc_samples] + [row["time_s"] for row in bsr_samples]
    if not times:
        return []
    start_s = min(times)
    grouped_rlc: DefaultDict[int, list[dict[str, float]]] = defaultdict(list)
    grouped_bsr: DefaultDict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rlc_samples:
        grouped_rlc[window_index(row["time_s"], start_s, window_s)].append(row)
    for row in bsr_samples:
        grouped_bsr[window_index(row["time_s"], start_s, window_s)].append(row)
    max_idx = max(set(grouped_rlc) | set(grouped_bsr))
    return [
        summarize_window(run_group, idx, window_s, grouped_rlc.get(idx, []), grouped_bsr.get(idx, []))
        for idx in range(max_idx + 1)
    ]


def make_summary(run_group: str, duration_s: float, rlc_samples: Sequence[dict[str, float]], bsr_samples: Sequence[dict[str, float]]) -> dict[str, str]:
    rlc_totals = [row["total_buffer_bytes"] for row in rlc_samples]
    bsr_totals = [row["total_lcg_bytes"] for row in bsr_samples]
    sdu_values = [row["sdu_bytes"] for row in bsr_samples]
    padding_values = [row["padding_len"] for row in bsr_samples]
    sdu_sum = sum(sdu_values)
    bsr_sent = sum(row["bsr_sent"] for row in bsr_samples)
    type_counts = Counter(to_int(row["bsr_type"]) for row in bsr_samples)
    type_counts_text = ", ".join(f"{BSR_TYPE_LABELS.get(key, str(key))}:{value}" for key, value in sorted(type_counts.items()))
    safe_duration = duration_s if duration_s > 0 else 1.0
    return {
        "run_group": run_group,
        "duration_s": fmt(duration_s),
        "rlc_samples": str(len(rlc_samples)),
        "rlc_total_buffer_p50_bytes": fmt_count(percentile(rlc_totals, 50)),
        "rlc_total_buffer_p95_bytes": fmt_count(percentile(rlc_totals, 95)),
        "rlc_total_buffer_max_bytes": fmt_count(max(rlc_totals) if rlc_totals else float("nan")),
        "bsr_samples": str(len(bsr_samples)),
        "bsr_sent": fmt_count(bsr_sent),
        "bsr_sent_rate_hz": fmt(bsr_sent / safe_duration),
        "bsr_total_lcg_p50_bytes": fmt_count(percentile(bsr_totals, 50)),
        "bsr_total_lcg_p95_bytes": fmt_count(percentile(bsr_totals, 95)),
        "bsr_total_lcg_max_bytes": fmt_count(max(bsr_totals) if bsr_totals else float("nan")),
        "sdu_bytes": fmt_count(sdu_sum),
        "sdu_mbps": fmt((sdu_sum * 8.0) / safe_duration / 1_000_000.0),
        "sdu_bytes_p50_per_grant": fmt_count(percentile(sdu_values, 50)),
        "padding_p50_bytes": fmt_count(percentile(padding_values, 50)),
        "padding_p95_bytes": fmt_count(percentile(padding_values, 95)),
        "bsr_type_counts": type_counts_text,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# NR UE Queue Metrics Summary\n\n")
        handle.write("BSR type labels: `0=none`, `1=long`, `2=short`, `3=short_trunc`, `4=long_trunc`.\n\n")
        handle.write("| Metric | Value |\n")
        handle.write("|---|---:|\n")
        for key in SUMMARY_FIELDS:
            handle.write(f"| {key} | {summary.get(key, '')} |\n")
        handle.write(
            "\nInterpretation hint: if RLC/LCG queue stays high while scheduled UL grants are "
            "bounded, the OAI delay is inside UE MAC/RLC drain / BSR-grant scheduling rather "
            "than edge compute or downlink.\n"
        )


def main() -> int:
    args = parse_args()
    try:
        rlc_csv, bsr_csv, output_dir, run_group = default_paths(args)
        require_file(rlc_csv, "RLC CSV")
        require_file(bsr_csv, "BSR CSV")
    except (OSError, ValueError) as exc:
        print(f"[analyze_nrue_queue_metrics] {exc}", file=sys.stderr)
        return 1

    rlc_samples = load_rlc_samples(rlc_csv)
    bsr_samples = load_bsr_samples(bsr_csv)
    times = [row["time_s"] for row in rlc_samples] + [row["time_s"] for row in bsr_samples]
    duration_s = max(times) - min(times) if len(times) > 1 else 0.0
    windows = make_windows(run_group, args.window_s, rlc_samples, bsr_samples)
    summary = make_summary(run_group, duration_s, rlc_samples, bsr_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    window_csv = output_dir / "nrue_queue_windows.csv"
    summary_csv = output_dir / "nrue_queue_summary.csv"
    summary_md = output_dir / "nrue_queue_summary.md"
    write_csv(window_csv, WINDOW_FIELDS, windows)
    write_csv(summary_csv, SUMMARY_FIELDS, [summary])
    write_markdown(summary_md, summary)
    print(f"[analyze_nrue_queue_metrics] wrote {window_csv}")
    print(f"[analyze_nrue_queue_metrics] wrote {summary_csv}")
    print(f"[analyze_nrue_queue_metrics] wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

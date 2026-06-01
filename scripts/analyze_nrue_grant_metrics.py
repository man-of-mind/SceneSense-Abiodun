#!/usr/bin/env python3
"""Summarize SceneSense NR UE decoded-grant T-tracer metrics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


ABIODUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TTRACER_ROOT = ABIODUN_DIR / "metrics_logs" / "scenesense_ttracer"

WINDOW_FIELDS = (
    "run_group",
    "rnti",
    "direction",
    "direction_label",
    "window_index",
    "window_start_s",
    "window_end_s",
    "grants",
    "grant_rate_hz",
    "total_tbs_bytes",
    "scheduled_mbps",
    "avg_mcs",
    "p50_mcs",
    "p95_mcs",
    "avg_rb_size",
    "p50_rb_size",
    "p95_rb_size",
    "avg_nr_symbols",
    "avg_tbs_bytes",
    "p95_tbs_bytes",
    "avg_qam_mod_order",
    "avg_target_code_rate",
    "retx_grants",
    "retx_rate",
    "new_data_grants",
    "avg_tpc",
    "avg_n_cce",
)

SUMMARY_FIELDS = (
    "run_group",
    "rnti",
    "direction",
    "direction_label",
    "duration_s",
    "windows",
    "grants",
    "grant_rate_hz",
    "total_tbs_bytes",
    "scheduled_mbps",
    "avg_mcs",
    "p50_mcs",
    "p95_mcs",
    "avg_rb_size",
    "p50_rb_size",
    "p95_rb_size",
    "avg_nr_symbols",
    "avg_tbs_bytes",
    "p95_tbs_bytes",
    "avg_qam_mod_order",
    "avg_target_code_rate",
    "retx_grants",
    "retx_rate",
    "new_data_grants",
    "avg_tpc",
    "avg_n_cce",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert NRUE_MAC_DCI_GRANT.csv into per-RNTI/window network-state "
            "features for SceneSense analysis and later RL inputs."
        )
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Run group under metrics_logs/scenesense_ttracer.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Explicit NRUE_MAC_DCI_GRANT.csv path. Overrides --run-group lookup.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_TTRACER_ROOT),
        help="T-tracer metrics root.",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=1.0,
        help="Aggregation window in seconds; default: 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run_group>/ue/analysis.",
    )
    return parser.parse_args()


def to_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


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


def direction_label(direction: int) -> str:
    if direction == 1:
        return "ul"
    if direction == 0:
        return "dl"
    return "unknown"


def normalize_tbs_bytes(direction: int, raw_tbs: int) -> float:
    # In this OAI tree, UE UL grant tb_size is bytes while UE DL tb_size is bits.
    # Normalize both directions to bytes before deriving scheduled bitrate.
    return float(raw_tbs) if direction == 1 else float(raw_tbs) / 8.0


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


def find_input_csv(args: argparse.Namespace) -> Tuple[Path, str]:
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        run_group = args.run_group
        if not run_group:
            try:
                run_group = csv_path.parents[2].name
            except IndexError:
                run_group = csv_path.stem
        return csv_path, run_group

    if not args.run_group:
        raise ValueError("provide --run-group or --csv")

    root = Path(args.root).expanduser().resolve()
    csv_path = root / args.run_group / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path, args.run_group


def default_output_dir(args: argparse.Namespace, csv_path: Path, run_group: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.csv:
        return csv_path.parents[1] / "analysis"
    return Path(args.root).expanduser().resolve() / run_group / "ue" / "analysis"


def summarize_rows(rows: List[Dict[str, int]], duration_s: float, run_group: str, rnti: int, direction: int, window_count: int) -> Dict[str, str]:
    grants = len(rows)
    tbs_values = [row["tbs"] for row in rows]
    mcs_values = [row["mcs"] for row in rows]
    rb_values = [row["rb_size"] for row in rows]
    symbol_values = [row["nr_symbols"] for row in rows]
    qam_values = [row["qam_mod_order"] for row in rows]
    code_rate_values = [row["target_code_rate"] for row in rows]
    tpc_values = [row["tpc"] for row in rows]
    cce_values = [row["n_cce"] for row in rows]
    total_tbs = sum(tbs_values)
    safe_duration = duration_s if duration_s > 0 else 1.0
    retx_grants = sum(1 for row in rows if row["round"] > 0 or row["rv"] > 0)
    new_data_grants = sum(1 for row in rows if row["ndi"] > 0)

    return {
        "run_group": run_group,
        "rnti": f"0x{rnti:04x}",
        "direction": str(direction),
        "direction_label": direction_label(direction),
        "duration_s": fmt(duration_s),
        "windows": str(window_count),
        "grants": str(grants),
        "grant_rate_hz": fmt(grants / safe_duration),
        "total_tbs_bytes": fmt_count(total_tbs),
        "scheduled_mbps": fmt((total_tbs * 8.0) / safe_duration / 1_000_000.0),
        "avg_mcs": fmt(mean(mcs_values)),
        "p50_mcs": fmt(percentile(mcs_values, 50)),
        "p95_mcs": fmt(percentile(mcs_values, 95)),
        "avg_rb_size": fmt(mean(rb_values)),
        "p50_rb_size": fmt(percentile(rb_values, 50)),
        "p95_rb_size": fmt(percentile(rb_values, 95)),
        "avg_nr_symbols": fmt(mean(symbol_values)),
        "avg_tbs_bytes": fmt(mean(tbs_values)),
        "p95_tbs_bytes": fmt(percentile(tbs_values, 95)),
        "avg_qam_mod_order": fmt(mean(qam_values)),
        "avg_target_code_rate": fmt(mean(code_rate_values)),
        "retx_grants": str(retx_grants),
        "retx_rate": fmt(retx_grants / grants if grants else float("nan")),
        "new_data_grants": str(new_data_grants),
        "avg_tpc": fmt(mean(tpc_values)),
        "avg_n_cce": fmt(mean(cce_values)),
    }


def load_rows(csv_path: Path) -> Tuple[List[Dict[str, int]], float, float]:
    rows: List[Dict[str, int]] = []
    first_time: float | None = None
    last_time: float | None = None
    previous_time: float | None = None

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "direction", "rnti", "mcs", "rb_size", "nr_symbols", "tbs"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

        for raw in reader:
            seconds = parse_time_to_seconds(raw["time"], previous_time)
            previous_time = seconds
            if first_time is None:
                first_time = seconds
            last_time = seconds
            direction = to_int(raw.get("direction"))
            raw_tbs = to_int(raw.get("tbs"))
            rows.append(
                {
                    "seconds": seconds,
                    "direction": direction,
                    "rnti": to_int(raw.get("rnti")),
                    "mcs": to_int(raw.get("mcs")),
                    "rb_size": to_int(raw.get("rb_size")),
                    "nr_symbols": to_int(raw.get("nr_symbols")),
                    "tbs": normalize_tbs_bytes(direction, raw_tbs),
                    "qam_mod_order": to_int(raw.get("qam_mod_order")),
                    "target_code_rate": to_int(raw.get("target_code_rate")),
                    "ndi": to_int(raw.get("ndi")),
                    "rv": to_int(raw.get("rv")),
                    "round": to_int(raw.get("round")),
                    "tpc": to_int(raw.get("tpc")),
                    "n_cce": to_int(raw.get("n_cce")),
                }
            )

    if first_time is None or last_time is None:
        return rows, 0.0, 0.0
    return rows, first_time, last_time


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: Sequence[Dict[str, str]], window_s: float, input_csv: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# NR UE Grant Summary\n\n")
        handle.write(f"- Input CSV: `{input_csv}`\n")
        handle.write(f"- Window size: `{window_s:g}s`\n\n")
        handle.write("| RNTI | Direction | Grants | Scheduled Mbps | Avg MCS | Avg RBs | Avg symbols | Avg TBS bytes | Retx rate |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                "| {rnti} | {direction_label} | {grants} | {scheduled_mbps} | "
                "{avg_mcs} | {avg_rb_size} | {avg_nr_symbols} | "
                "{avg_tbs_bytes} | {retx_rate} |\n".format(**row)
            )
        handle.write(
            "\n`scheduled_mbps` is derived from decoded TBS grants, not from an IP-layer "
            "throughput counter. It is the UE-visible scheduled data budget for that "
            "window/direction.\n"
        )


def main() -> int:
    args = parse_args()
    if args.window_s <= 0:
        print("[analyze_nrue_grant_metrics] --window-s must be > 0", file=sys.stderr)
        return 2

    try:
        input_csv, run_group = find_input_csv(args)
        output_dir = default_output_dir(args, input_csv, run_group)
        rows, first_time, last_time = load_rows(input_csv)
    except (OSError, ValueError) as exc:
        print(f"[analyze_nrue_grant_metrics] {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"[analyze_nrue_grant_metrics] no rows found in {input_csv}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    duration_s = max(last_time - first_time, args.window_s)

    buckets: DefaultDict[Tuple[int, int, int], List[Dict[str, int]]] = defaultdict(list)
    by_rnti_direction: DefaultDict[Tuple[int, int], List[Dict[str, int]]] = defaultdict(list)
    max_window = 0
    for row in rows:
        window_index = int((row["seconds"] - first_time) // args.window_s)
        max_window = max(max_window, window_index)
        key = (row["rnti"], row["direction"], window_index)
        buckets[key].append(row)
        by_rnti_direction[(row["rnti"], row["direction"])].append(row)

    window_rows: List[Dict[str, str]] = []
    for (rnti, direction, window_index), bucket_rows in sorted(buckets.items()):
        window_start = window_index * args.window_s
        window_end = window_start + args.window_s
        row = summarize_rows(bucket_rows, args.window_s, run_group, rnti, direction, 1)
        row["window_index"] = str(window_index)
        row["window_start_s"] = fmt(window_start)
        row["window_end_s"] = fmt(window_end)
        window_rows.append(row)

    summary_rows: List[Dict[str, str]] = []
    for (rnti, direction), group_rows in sorted(by_rnti_direction.items()):
        summary_rows.append(
            summarize_rows(group_rows, duration_s, run_group, rnti, direction, max_window + 1)
        )

    window_path = output_dir / "nrue_grant_windows.csv"
    summary_path = output_dir / "nrue_grant_summary.csv"
    md_path = output_dir / "nrue_grant_summary.md"
    write_csv(window_path, WINDOW_FIELDS, window_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_markdown(md_path, summary_rows, args.window_s, input_csv)

    print(f"[analyze_nrue_grant_metrics] rows={len(rows)} duration_s={duration_s:.3f}")
    print(f"[analyze_nrue_grant_metrics] wrote {window_path}")
    print(f"[analyze_nrue_grant_metrics] wrote {summary_path}")
    print(f"[analyze_nrue_grant_metrics] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

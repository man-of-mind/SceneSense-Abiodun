#!/usr/bin/env python3
"""Summarize downlink/result-return FPS sweep metrics.

Usage:
  python3 downlink_latency_fps/analyze_downlink_fps.py downlink_latency_fps/runs
  python3 downlink_latency_fps/analyze_downlink_fps.py downlink_latency_fps/runs --contains 20260717_ideal_one_loop
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable


NUMERIC_FIELDS = [
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "round_trip_result_recv_ms",
    "transport_round_trip_ms_estimate",
    "total_pipeline_ms_estimate",
    "result_wait_ms",
    "result_queue_wait_ms",
    "tail_done_to_result_recv_ms",
    "result_send_to_recv_ms_perf",
    "result_send_to_recv_ms_wall",
    "result_recv_to_display_ms",
    "tail_done_to_display_ms",
    "feature_payload_bytes",
    "feature_payload_bytes_uncompressed",
    "feature_payload_chunks",
    "result_payload_bytes_estimate",
    "result_payload_chunks_estimate",
    "ego_speed_mps",
]

RESULT_ONLY_FIELDS = {
    "back_ms",
    "round_trip_ms",
    "round_trip_result_recv_ms",
    "transport_round_trip_ms_estimate",
    "total_pipeline_ms_estimate",
    "result_queue_wait_ms",
    "tail_done_to_result_recv_ms",
    "result_send_to_recv_ms_perf",
    "result_send_to_recv_ms_wall",
    "result_recv_to_display_ms",
    "tail_done_to_display_ms",
    "result_payload_bytes_estimate",
    "result_payload_chunks_estimate",
}


REPORT_COLUMNS = [
    "condition",
    "fps",
    "frames",
    "received",
    "delivery",
    "ego_speed_p50_mps",
    "ego_speed_mean_mps",
    "ego_speed_p95_mps",
    "moving_gt0p5_frac",
    "feature_kb_p50",
    "feature_chunks_p50",
    "result_kb_p50",
    "result_chunks_p50",
    "capture_to_result_est_p50_ms",
    "front_p50_ms",
    "back_p50_ms",
    "rtt_recv_p50_ms",
    "rtt_recv_p95_ms",
    "feature_upload_payload_handling_p50_ms",
    "tail_to_recv_p50_ms",
    "result_send_to_recv_wall_p50_ms",
    "queue_wait_p50_ms",
]


def as_float(value: object) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return float("nan")
        out = float(text)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def clean(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def quantile(values: list[float], q: float) -> float:
    values = sorted(clean(values))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def p50(values: list[float]) -> float:
    values = clean(values)
    return statistics.median(values) if values else float("nan")


def infer_condition(path: Path, first: dict[str, str]) -> str:
    label = (first.get("transport_label") or "").strip()
    if label:
        return label
    for part in path.parts:
        if part in {"ideal_loopback", "bounded_loopback", "oai_default"}:
            return part
    return "unknown"


def infer_fps(path: Path, first: dict[str, str]) -> str:
    text = " ".join(
        [
            str(path),
            first.get("run_group", ""),
            first.get("run_id", ""),
        ]
    )
    match = re.search(r"fps[_-]?(\d+(?:\.\d+)?)", text)
    return match.group(1) if match else "unknown"


def summarize_csv(path: Path) -> dict[str, object] | None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    first = rows[0]
    received = [str(r.get("result_received", "")).strip().lower() in {"1", "true", "yes"} for r in rows]
    values = {}
    for name in NUMERIC_FIELDS:
        vals = []
        for row, ok in zip(rows, received):
            if name in RESULT_ONLY_FIELDS and not ok:
                continue
            vals.append(as_float(row.get(name, "")))
        values[name] = vals

    frames = len(rows)
    got = sum(1 for x in received if x)
    speeds = clean(values["ego_speed_mps"])
    moving_frac = (
        sum(1 for v in speeds if v > 0.5) / len(speeds)
        if speeds
        else float("nan")
    )
    capture_to_result = []
    feature_upload_payload_handling = []
    for row, ok in zip(rows, received):
        if not ok:
            continue
        front_ms = as_float(row.get("front_ms", ""))
        rtt_recv_ms = as_float(row.get("round_trip_result_recv_ms", ""))
        back_ms = as_float(row.get("back_ms", ""))
        tail_to_recv_ms = as_float(row.get("tail_done_to_result_recv_ms", ""))
        if math.isfinite(front_ms) and math.isfinite(rtt_recv_ms):
            # Current best estimate from existing logs. Note: because the
            # send timestamp is taken just before sender.send(), and front_ms
            # ends just after sender.send(), this can conservatively double
            # count the feature-send burst. Future runs should log front_start
            # and send_done explicitly.
            capture_to_result.append(front_ms + rtt_recv_ms)
        if (
            math.isfinite(rtt_recv_ms)
            and math.isfinite(back_ms)
            and math.isfinite(tail_to_recv_ms)
        ):
            feature_upload_payload_handling.append(
                max(0.0, rtt_recv_ms - back_ms - tail_to_recv_ms)
            )
    out: dict[str, object] = {
        "condition": infer_condition(path, first),
        "fps": infer_fps(path, first),
        "frames": frames,
        "received": got,
        "delivery": got / frames if frames else float("nan"),
        "ego_speed_p50_mps": p50(values["ego_speed_mps"]),
        "ego_speed_mean_mps": statistics.mean(speeds) if speeds else float("nan"),
        "ego_speed_p95_mps": quantile(values["ego_speed_mps"], 0.95),
        "moving_gt0p5_frac": moving_frac,
        "feature_kb_p50": p50(values["feature_payload_bytes"]) / 1024.0,
        "feature_chunks_p50": p50(values["feature_payload_chunks"]),
        "result_kb_p50": p50(values["result_payload_bytes_estimate"]) / 1024.0,
        "result_chunks_p50": p50(values["result_payload_chunks_estimate"]),
        "capture_to_result_est_p50_ms": p50(capture_to_result),
        "front_p50_ms": p50(values["front_ms"]),
        "back_p50_ms": p50(values["back_ms"]),
        "rtt_recv_p50_ms": p50(values["round_trip_result_recv_ms"]),
        "rtt_recv_p95_ms": quantile(values["round_trip_result_recv_ms"], 0.95),
        "feature_upload_payload_handling_p50_ms": p50(feature_upload_payload_handling),
        "tail_to_recv_p50_ms": p50(values["tail_done_to_result_recv_ms"]),
        "result_send_to_recv_wall_p50_ms": p50(values["result_send_to_recv_ms_wall"]),
        "queue_wait_p50_ms": p50(values["result_queue_wait_ms"]),
        "_source": str(path),
    }
    return out


def fmt(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        if abs(value) < 1:
            return f"{value:.3f}"
        return f"{value:.1f}"
    return str(value)


def safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "filtered"


def main(argv: list[str]) -> int:
    root_arg = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "downlink_latency_fps/runs"
    root = Path(root_arg)
    contains = ""
    if "--contains" in argv:
        idx = argv.index("--contains")
        try:
            contains = argv[idx + 1]
        except IndexError:
            print("--contains requires a substring", file=sys.stderr)
            return 2
    if not root.exists():
        print(f"Missing run root: {root}", file=sys.stderr)
        return 2

    summaries = []
    for path in sorted(root.rglob("*_metrics.csv")):
        if contains and contains not in str(path):
            continue
        item = summarize_csv(path)
        if item is not None:
            summaries.append(item)

    if not summaries:
        print(f"No metrics CSV files found under {root}", file=sys.stderr)
        return 1

    def sort_key(row: dict[str, object]) -> tuple[str, float, str]:
        try:
            fps = float(row["fps"])
        except Exception:
            fps = 9999.0
        return (str(row["condition"]), fps, str(row["_source"]))

    summaries.sort(key=sort_key)

    suffix = f"_{safe_token(contains)}" if contains else ""
    out_csv = root / f"downlink_fps_summary{suffix}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS + ["source"])
        writer.writeheader()
        for row in summaries:
            writer.writerow({**{k: row.get(k, "") for k in REPORT_COLUMNS}, "source": row.get("_source", "")})

    print(f"Wrote {out_csv}")
    print()
    print("| " + " | ".join(REPORT_COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(REPORT_COLUMNS)) + "|")
    for row in summaries:
        print("| " + " | ".join(fmt(row.get(k, "")) for k in REPORT_COLUMNS) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

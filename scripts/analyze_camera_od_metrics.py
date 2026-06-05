#!/usr/bin/env python3
"""Summarize camera-only object-detection quality and transport metrics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence


QUALITY_COLUMNS = (
    "od_recall",
    "od_precision",
    "od_vehicle_recall",
    "od_vehicle_precision",
    "od_person_recall",
    "od_person_precision",
    "od_mean_iou",
)
COUNT_COLUMNS = (
    "gt_object_count",
    "gt_vehicle_count",
    "gt_person_count",
    "pred_object_count",
    "pred_vehicle_count",
    "pred_person_count",
    "od_matched_count",
    "od_vehicle_matched_count",
    "od_person_matched_count",
)
TRAFFIC_COLUMNS = (
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "payload_kib",
    "payload_uncompressed_kib",
    "payload_chunks",
    "detections",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SceneSense camera-only OD CSVs. OD quality metrics are "
            "reported when the run was collected with --enable-od-gt."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="CSV files, directories, or quoted glob patterns. Directories expand to *.csv.",
    )
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--label", default="camera_od_metrics", help="Output label.")
    return parser.parse_args()


def expand_paths(raw_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in raw_paths:
        expanded = [Path(p) for p in glob.glob(raw)] if any(ch in raw for ch in "*?[]") else [Path(raw)]
        for path in expanded:
            path = path.expanduser()
            if path.is_dir():
                paths.extend(sorted(path.glob("*.csv")))
            elif path.exists():
                paths.append(path)
    unique: Dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path.resolve()
    return sorted(unique.values())


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: object) -> int:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else 0


def ratio(num: int, den: int) -> Optional[float]:
    return float(num / den) if den > 0 else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_values(values: Iterable[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p05": percentile(clean, 0.05),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def flatten(prefix: str, stats: Dict[str, Optional[float]]) -> Dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def analyze_rows(source: str, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    fieldnames = set(rows[0].keys()) if rows else set()
    quality_present = "gt_od_available" in fieldnames and all(col in fieldnames for col in COUNT_COLUMNS)

    totals = {column: sum(parse_int(row.get(column)) for row in rows) for column in COUNT_COLUMNS}
    frames_with_gt = sum(1 for row in rows if parse_int(row.get("gt_od_available")) > 0)
    frames_with_objects = sum(1 for row in rows if parse_int(row.get("gt_object_count")) > 0)
    frames_with_predictions = sum(1 for row in rows if parse_int(row.get("pred_object_count")) > 0)

    summary: Dict[str, object] = {
        "source": source,
        "frames_total": len(rows),
        "quality_columns_present": bool(quality_present),
        "frames_with_gt_od": frames_with_gt,
        "frames_with_gt_objects": frames_with_objects,
        "frames_with_predictions": frames_with_predictions,
        **totals,
        "global_recall": ratio(totals["od_matched_count"], totals["gt_object_count"]),
        "global_precision": ratio(totals["od_matched_count"], totals["pred_object_count"]),
        "global_vehicle_recall": ratio(totals["od_vehicle_matched_count"], totals["gt_vehicle_count"]),
        "global_vehicle_precision": ratio(totals["od_vehicle_matched_count"], totals["pred_vehicle_count"]),
        "global_person_recall": ratio(totals["od_person_matched_count"], totals["gt_person_count"]),
        "global_person_precision": ratio(totals["od_person_matched_count"], totals["pred_person_count"]),
    }

    for column in QUALITY_COLUMNS:
        values = [v for row in rows if (v := parse_float(row.get(column))) is not None]
        summary.update(flatten(column, summarize_values(values)))

    for column in TRAFFIC_COLUMNS:
        values = [v for row in rows if (v := parse_float(row.get(column))) is not None]
        summary.update(flatten(column, summarize_values(values)))

    thresholds = [v for row in rows if (v := parse_float(row.get("od_match_iou_threshold"))) is not None]
    summary["od_match_iou_threshold"] = median(thresholds) if thresholds else None
    return summary


def write_csv(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    keys: List[str] = []
    for summary in summaries:
        for key in summary:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    summaries = list(payload["summaries"])  # type: ignore[index]
    lines = [
        f"# {payload['label']}",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Source | Frames | GT Objects | Predictions | Matches | Recall | Precision | Vehicle Recall | Person Recall | RTT Median ms | Payload Median KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["source"]),
                    str(summary["frames_total"]),
                    str(summary["gt_object_count"]),
                    str(summary["pred_object_count"]),
                    str(summary["od_matched_count"]),
                    fmt(summary.get("global_recall")),
                    fmt(summary.get("global_precision")),
                    fmt(summary.get("global_vehicle_recall")),
                    fmt(summary.get("global_person_recall")),
                    fmt(summary.get("round_trip_ms_median")),
                    fmt(summary.get("payload_kib_median")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- GT boxes are CARLA vehicle/person actors projected into the RGB camera.",
            "- Predictions are COCO Faster R-CNN classes collapsed to vehicle/person.",
            "- Global recall is matched GT / selected GT; global precision is matched predictions / selected predictions.",
            "- Matching is greedy, class-aware 2D box IoU using the run's `od_match_iou_threshold`.",
            "- If `quality_columns_present=false`, the CSV was transport-only and was not collected with `--enable-od-gt`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_paths = expand_paths(args.csv_paths)
    if not csv_paths:
        raise SystemExit("No CSV files found for the provided path(s).")

    per_file: List[Dict[str, object]] = []
    all_rows: List[Dict[str, str]] = []
    for path in csv_paths:
        rows = read_csv(path)
        if not rows:
            continue
        all_rows.extend(rows)
        per_file.append(analyze_rows(path.name, rows))
    if not all_rows:
        raise SystemExit("CSV files were found, but none contained data rows.")

    summaries = [analyze_rows("OVERALL", all_rows), *per_file]
    payload = {
        "label": args.label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv_files": [str(path) for path in csv_paths],
        "summaries": summaries,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{args.label}_{stamp}"
        json_path = output_dir / f"{prefix}.json"
        csv_path = output_dir / f"{prefix}.csv"
        md_path = output_dir / f"{prefix}.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(csv_path, summaries)
        write_markdown(md_path, payload)
        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

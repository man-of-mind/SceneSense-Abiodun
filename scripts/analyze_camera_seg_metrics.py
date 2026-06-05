#!/usr/bin/env python3
"""Summarize camera-only semantic segmentation quality and transport metrics."""

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
    "miou_binary",
    "miou_3class_macro",
    "miou_vehicle_iou",
    "miou_person_iou",
)
TRAFFIC_COLUMNS = (
    "front_ms",
    "back_ms",
    "round_trip_ms",
    "payload_kib",
    "payload_uncompressed_kib",
    "payload_chunks",
    "mask_payload_bytes_estimate",
    "mask_foreground_pixels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SceneSense camera-only SEG CSVs. Quality metrics are "
            "reported when the run was collected with --enable-semantic-gt."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help=(
            "CSV files, directories, or quoted glob patterns. Directories are "
            "expanded to *.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for JSON/CSV/Markdown summary files.",
    )
    parser.add_argument(
        "--label",
        default="camera_seg_metrics",
        help="Label used in output filenames and the top-level JSON.",
    )
    parser.add_argument(
        "--visible-pixel-threshold",
        type=int,
        default=1,
        help=(
            "Minimum GT pixels for a class-specific visible-GT IoU summary. "
            "Default counts any visible pixel."
        ),
    )
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
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_int(value: object) -> int:
    parsed = parse_float(value)
    if parsed is None:
        return 0
    return int(parsed)


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
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p05": percentile(clean, 0.05),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
    }


def flatten_stats(prefix: str, stats: Dict[str, Optional[float]]) -> Dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def analyze_rows(source: str, rows: Sequence[Dict[str, str]], *, visible_threshold: int) -> Dict[str, object]:
    fieldnames = set(rows[0].keys()) if rows else set()
    quality_present = all(column in fieldnames for column in QUALITY_COLUMNS)
    gt_present = "gt_camera_available" in fieldnames

    summary: Dict[str, object] = {
        "source": source,
        "frames_total": len(rows),
        "quality_columns_present": bool(quality_present),
        "gt_column_present": bool(gt_present),
        "frames_with_mask": sum(1 for row in rows if parse_int(row.get("mask_available")) > 0),
        "frames_with_gt_camera": sum(1 for row in rows if parse_int(row.get("gt_camera_available")) > 0),
        "frames_with_vehicle_gt": sum(
            1 for row in rows if parse_int(row.get("gt_vehicle_pixels")) >= visible_threshold
        ),
        "frames_with_person_gt": sum(
            1 for row in rows if parse_int(row.get("gt_person_pixels")) >= visible_threshold
        ),
        "gt_vehicle_pixels_total": sum(parse_int(row.get("gt_vehicle_pixels")) for row in rows),
        "gt_person_pixels_total": sum(parse_int(row.get("gt_person_pixels")) for row in rows),
    }

    for column in QUALITY_COLUMNS:
        values = [v for row in rows if (v := parse_float(row.get(column))) is not None]
        summary.update(flatten_stats(column, summarize_values(values)))

    vehicle_visible = [
        v
        for row in rows
        if parse_int(row.get("gt_vehicle_pixels")) >= visible_threshold
        if (v := parse_float(row.get("miou_vehicle_iou"))) is not None
    ]
    person_visible = [
        v
        for row in rows
        if parse_int(row.get("gt_person_pixels")) >= visible_threshold
        if (v := parse_float(row.get("miou_person_iou"))) is not None
    ]
    summary.update(flatten_stats("miou_vehicle_iou_visible_gt", summarize_values(vehicle_visible)))
    summary.update(flatten_stats("miou_person_iou_visible_gt", summarize_values(person_visible)))

    for column in TRAFFIC_COLUMNS:
        values = [v for row in rows if (v := parse_float(row.get(column))) is not None]
        summary.update(flatten_stats(column, summarize_values(values)))

    return summary


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        return
    keys: List[str] = []
    for summary in summaries:
        for key in summary:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


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
        "| Source | Frames | GT Frames | Binary IoU Mean | 3-Class mIoU Mean | Vehicle IoU Mean | Person IoU Mean | RTT Median ms | Payload Median KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["source"]),
                    str(summary["frames_total"]),
                    str(summary["frames_with_gt_camera"]),
                    fmt(summary.get("miou_binary_mean")),
                    fmt(summary.get("miou_3class_macro_mean")),
                    fmt(summary.get("miou_vehicle_iou_visible_gt_mean")),
                    fmt(summary.get("miou_person_iou_visible_gt_mean")),
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
            "- `miou_binary` is foreground-vs-background IoU after collapsing vehicle and person into foreground.",
            "- `miou_3class_macro` averages background, vehicle, and person IoU for classes with non-empty union.",
            "- `*_visible_gt` class summaries include only frames where that GT class has at least the configured pixel threshold.",
            "- If `quality_columns_present=false`, the source CSV was transport-only and was not collected with semantic GT metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_paths = expand_paths(args.csv_paths)
    if not csv_paths:
        raise SystemExit("No CSV files found for the provided path(s).")

    per_file_rows: List[Dict[str, object]] = []
    all_rows: List[Dict[str, str]] = []
    for path in csv_paths:
        rows = read_csv(path)
        if not rows:
            continue
        all_rows.extend(rows)
        per_file_rows.append(
            analyze_rows(
                path.name,
                rows,
                visible_threshold=max(1, int(args.visible_pixel_threshold)),
            )
        )

    if not all_rows:
        raise SystemExit("CSV files were found, but none contained data rows.")

    overall = analyze_rows(
        "OVERALL",
        all_rows,
        visible_threshold=max(1, int(args.visible_pixel_threshold)),
    )
    summaries = [overall, *per_file_rows]
    payload = {
        "label": args.label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "visible_pixel_threshold": max(1, int(args.visible_pixel_threshold)),
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

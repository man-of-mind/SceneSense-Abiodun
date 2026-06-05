#!/usr/bin/env python3
"""Evaluate SceneSense fusion object-head predictions against CARLA vehicle GT."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_runs"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "metrics_logs" / "scenesense_analysis"
MPLCONFIG_DIR = Path("/tmp/scenesense_mplconfig")
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

BASE_SUMMARY_FIELDS = (
    "run_group",
    "stream_id",
    "platform",
    "frames",
    "gt_vehicle_count",
    "predicted_vehicle_count",
    "match_distance_m",
    "matched_vehicle_count",
    "precision_at_match_distance",
    "false_positives_per_frame",
    "mean_xy_error_m",
    "median_xy_error_m",
    "p95_xy_error_m",
    "mean_xyz_error_m",
    "mean_abs_yaw_error_deg",
    "mean_length_abs_error_m",
    "mean_width_abs_error_m",
    "mean_height_abs_error_m",
    "source_prediction_csv_count",
    "source_ground_truth_csv_count",
)

PLATFORM_SUMMARY_FIELDS = (
    "run_group",
    "platform",
    "streams",
    "frames",
    "gt_vehicle_count",
    "predicted_vehicle_count",
    "match_distance_m",
    "matched_vehicle_count",
    "recall_at_match_distance",
    "precision_at_match_distance",
    "false_positives_per_frame",
    "mean_xy_error_m",
    "mean_xyz_error_m",
    "mean_abs_yaw_error_deg",
)

GT_FILTER_SUMMARY_FIELDS = (
    "run_group",
    "stream_id",
    "platform",
    "raw_gt_vehicle_count",
    "selected_gt_vehicle_count",
    "dropped_gt_vehicle_count",
    "min_gt_bbox_area_px",
    "min_gt_bbox_width_px",
    "min_gt_bbox_height_px",
    "max_gt_distance_m",
    "require_gt_center_in_image",
    "camera_width",
    "camera_height",
)

MATCH_FIELDS = (
    "run_group",
    "stream_id",
    "platform",
    "frame_id",
    "gt_actor_id",
    "pred_object_index",
    "xy_error_m",
    "xyz_error_m",
    "yaw_abs_error_deg",
    "length_abs_error_m",
    "width_abs_error_m",
    "height_abs_error_m",
    "gt_world_x",
    "gt_world_y",
    "gt_world_z",
    "pred_world_x",
    "pred_world_y",
    "pred_world_z",
    "gt_length_m",
    "gt_width_m",
    "gt_height_m",
    "pred_length_m",
    "pred_width_m",
    "pred_height_m",
    "pred_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match per-object fusion predictions against CARLA vehicle ground "
            "truth and summarize object recall/localization transfer."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_RUN_ROOT),
        help="Root containing SceneSense run folders.",
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Run group to analyze. Defaults to latest group with object CSVs.",
    )
    parser.add_argument(
        "--stream-id",
        action="append",
        default=[],
        help="Optional stream id filter. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults under metrics_logs/scenesense_analysis/.",
    )
    parser.add_argument(
        "--distance-thresholds-m",
        default="1,2,3",
        help="Comma-separated XY match thresholds used for recall columns.",
    )
    parser.add_argument(
        "--match-distance-m",
        type=float,
        default=2.0,
        help="XY threshold used for error and false-positive summaries.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Drop predictions below this score before matching.",
    )
    parser.add_argument(
        "--include-out-of-frustum-gt",
        action="store_true",
        help="Include vehicle GT rows whose 3D bbox does not intersect the camera image.",
    )
    parser.add_argument(
        "--min-gt-bbox-area-px",
        type=float,
        default=0.0,
        help=(
            "Drop GT vehicles whose projected 2D bbox area is smaller than this. "
            "Useful for matching training-style object targets instead of every "
            "tiny/frustum-edge actor."
        ),
    )
    parser.add_argument(
        "--min-gt-bbox-width-px",
        type=float,
        default=0.0,
        help="Drop GT vehicles whose projected 2D bbox width is smaller than this.",
    )
    parser.add_argument(
        "--min-gt-bbox-height-px",
        type=float,
        default=0.0,
        help="Drop GT vehicles whose projected 2D bbox height is smaller than this.",
    )
    parser.add_argument(
        "--max-gt-distance-m",
        type=float,
        default=0.0,
        help="Drop GT vehicles farther than this from the camera. 0 disables the filter.",
    )
    parser.add_argument(
        "--require-gt-center-in-image",
        action="store_true",
        help=(
            "Require the projected GT bbox center to be inside the image. "
            "This mirrors the object-target builder more closely."
        ),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=854,
        help="Image width used by --require-gt-center-in-image.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Image height used by --require-gt-center-in-image.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV/Markdown summaries without plots.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List discovered run groups and exit.",
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
    return token.strip("_") or default


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


def parse_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("Distance thresholds must be positive.")
        thresholds.append(value)
    if not thresholds:
        thresholds = [1.0, 2.0, 3.0]
    return sorted(set(thresholds))


def threshold_field(threshold_m: float) -> str:
    label = f"{threshold_m:g}".replace(".", "p")
    return f"recall_at_{label}m"


def stream_platform(stream_id: str) -> str:
    token = stream_id.lower()
    if "ego" in token:
        return "parked_ego"
    if "tl" in token or "pole" in token:
        return "pole"
    return "unknown"


def load_csvs(root: Path, suffix: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(root.rglob(f"streams/*_{suffix}.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for raw in reader:
                    row = {str(key): "" if value is None else str(value) for key, value in raw.items()}
                    row["_source_csv"] = str(path)
                    row["_source_run_dir"] = str(path.parent.parent)
                    if not row.get("stream_id"):
                        row["stream_id"] = path.stem.replace(f"_{suffix}", "")
                    rows.append(row)
        except (OSError, csv.Error) as exc:
            print(f"[warn] skipped {path}: {exc}", file=sys.stderr)
    return rows


def latest_run_group(rows: Sequence[Dict[str, str]]) -> Optional[str]:
    if not rows:
        return None
    groups: Dict[str, Tuple[str, int]] = {}
    for row in rows:
        group = row.get("run_group", "")
        if not group:
            continue
        key = row.get("wall_time_iso", "") or row.get("_source_csv", "")
        _old_key, count = groups.get(group, ("", 0))
        groups[group] = (max(_old_key, key), count + 1)
    if not groups:
        return None
    return max(groups.items(), key=lambda item: (item[1][0], item[1][1]))[0]


def list_groups(pred_rows: Sequence[Dict[str, str]], gt_rows: Sequence[Dict[str, str]]) -> None:
    combined = list(pred_rows) + list(gt_rows)
    by_group: Dict[str, Dict[str, object]] = defaultdict(lambda: {"pred": 0, "gt": 0, "streams": set()})
    for row in pred_rows:
        group = row.get("run_group", "")
        if not group:
            continue
        by_group[group]["pred"] = int(by_group[group]["pred"]) + 1
        by_group[group]["streams"].add(row.get("stream_id", ""))  # type: ignore[union-attr]
    for row in gt_rows:
        group = row.get("run_group", "")
        if not group:
            continue
        by_group[group]["gt"] = int(by_group[group]["gt"]) + 1
        by_group[group]["streams"].add(row.get("stream_id", ""))  # type: ignore[union-attr]
    if not combined:
        print("No fusion object CSVs discovered.")
        return
    for group, info in sorted(by_group.items()):
        streams = ",".join(sorted(str(value) for value in info["streams"] if value))  # type: ignore[index]
        print(f"- {group} | streams={streams} | pred_rows={info['pred']} | gt_rows={info['gt']}")


def group_by_frame(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("stream_id", ""), to_int(row.get("frame_id", "")))].append(row)
    return grouped


def yaw_abs_error_deg(pred_deg: float, gt_deg: float) -> float:
    if not (math.isfinite(pred_deg) and math.isfinite(gt_deg)):
        return float("nan")
    return abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0)


def row_xyz(row: Dict[str, str]) -> Tuple[float, float, float]:
    return (
        to_float(row.get("world_x", "")),
        to_float(row.get("world_y", "")),
        to_float(row.get("world_z", "")),
    )


def is_valid_xyz(row: Dict[str, str]) -> bool:
    return all(math.isfinite(value) for value in row_xyz(row))


def bbox_width_height_area(row: Dict[str, str]) -> Tuple[float, float, float]:
    x1 = to_float(row.get("bbox_x1", ""))
    y1 = to_float(row.get("bbox_y1", ""))
    x2 = to_float(row.get("bbox_x2", ""))
    y2 = to_float(row.get("bbox_y2", ""))
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return (float("nan"), float("nan"), float("nan"))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (width, height, width * height)


def gt_center_in_image(row: Dict[str, str], *, camera_width: int, camera_height: int) -> bool:
    x = to_float(row.get("projected_x", ""))
    y = to_float(row.get("projected_y", ""))
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and 0.0 <= x < float(camera_width)
        and 0.0 <= y < float(camera_height)
    )


def gt_passes_filters(row: Dict[str, str], args: argparse.Namespace) -> bool:
    width, height, area = bbox_width_height_area(row)
    min_area = float(args.min_gt_bbox_area_px)
    min_width = float(args.min_gt_bbox_width_px)
    min_height = float(args.min_gt_bbox_height_px)
    if min_area > 0.0 and (not math.isfinite(area) or area < min_area):
        return False
    if min_width > 0.0 and (not math.isfinite(width) or width < min_width):
        return False
    if min_height > 0.0 and (not math.isfinite(height) or height < min_height):
        return False
    max_distance = float(args.max_gt_distance_m)
    distance = to_float(row.get("distance_m", ""))
    if max_distance > 0.0 and (not math.isfinite(distance) or distance > max_distance):
        return False
    if bool(args.require_gt_center_in_image) and not gt_center_in_image(
        row,
        camera_width=int(args.camera_width),
        camera_height=int(args.camera_height),
    ):
        return False
    return True


def match_rows(
    gt_rows: Sequence[Dict[str, str]],
    pred_rows: Sequence[Dict[str, str]],
    *,
    threshold_m: float,
) -> List[Tuple[Dict[str, str], Dict[str, str], float]]:
    candidates: List[Tuple[float, int, int]] = []
    for gt_index, gt in enumerate(gt_rows):
        gx, gy, _gz = row_xyz(gt)
        if not (math.isfinite(gx) and math.isfinite(gy)):
            continue
        for pred_index, pred in enumerate(pred_rows):
            px, py, _pz = row_xyz(pred)
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            distance = math.hypot(px - gx, py - gy)
            if distance <= threshold_m:
                candidates.append((distance, gt_index, pred_index))
    candidates.sort(key=lambda item: item[0])
    matched_gt = set()
    matched_pred = set()
    matches: List[Tuple[Dict[str, str], Dict[str, str], float]] = []
    for distance, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_rows[gt_index], pred_rows[pred_index], distance))
    return matches


def match_detail(
    run_group: str,
    stream_id: str,
    gt: Dict[str, str],
    pred: Dict[str, str],
    xy_error_m: float,
) -> Dict[str, object]:
    gx, gy, gz = row_xyz(gt)
    px, py, pz = row_xyz(pred)
    xyz_error_m = (
        math.sqrt((px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2)
        if all(math.isfinite(value) for value in (gx, gy, gz, px, py, pz))
        else float("nan")
    )
    length_error = abs(to_float(pred.get("length_m", "")) - to_float(gt.get("length_m", "")))
    width_error = abs(to_float(pred.get("width_m", "")) - to_float(gt.get("width_m", "")))
    height_error = abs(to_float(pred.get("height_m", "")) - to_float(gt.get("height_m", "")))
    return {
        "run_group": run_group,
        "stream_id": stream_id,
        "platform": stream_platform(stream_id),
        "frame_id": to_int(gt.get("frame_id", "")),
        "gt_actor_id": gt.get("actor_id", ""),
        "pred_object_index": pred.get("object_index", ""),
        "xy_error_m": xy_error_m,
        "xyz_error_m": xyz_error_m,
        "yaw_abs_error_deg": yaw_abs_error_deg(
            to_float(pred.get("yaw_deg", "")),
            to_float(gt.get("yaw_deg", "")),
        ),
        "length_abs_error_m": length_error,
        "width_abs_error_m": width_error,
        "height_abs_error_m": height_error,
        "gt_world_x": gx,
        "gt_world_y": gy,
        "gt_world_z": gz,
        "pred_world_x": px,
        "pred_world_y": py,
        "pred_world_z": pz,
        "gt_length_m": gt.get("length_m", ""),
        "gt_width_m": gt.get("width_m", ""),
        "gt_height_m": gt.get("height_m", ""),
        "pred_length_m": pred.get("length_m", ""),
        "pred_width_m": pred.get("width_m", ""),
        "pred_height_m": pred.get("height_m", ""),
        "pred_score": pred.get("score", ""),
    }


def summarize_stream(
    *,
    run_group: str,
    stream_id: str,
    gt_rows: Sequence[Dict[str, str]],
    pred_rows: Sequence[Dict[str, str]],
    thresholds_m: Sequence[float],
    match_distance_m: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    gt_by_frame = group_by_frame(gt_rows)
    pred_by_frame = group_by_frame(pred_rows)
    frame_keys = sorted(set(gt_by_frame) | set(pred_by_frame), key=lambda item: item[1])

    matched_counts = {threshold: 0 for threshold in thresholds_m}
    default_matches: List[Dict[str, object]] = []
    for key in frame_keys:
        frame_gt = [row for row in gt_by_frame.get(key, []) if is_valid_xyz(row)]
        frame_pred = [row for row in pred_by_frame.get(key, []) if is_valid_xyz(row)]
        for threshold in thresholds_m:
            matched_counts[threshold] += len(
                match_rows(frame_gt, frame_pred, threshold_m=threshold)
            )
        for gt, pred, xy_error_m in match_rows(
            frame_gt,
            frame_pred,
            threshold_m=match_distance_m,
        ):
            default_matches.append(match_detail(run_group, stream_id, gt, pred, xy_error_m))

    total_gt = sum(len(rows) for rows in gt_by_frame.values())
    total_pred = sum(len(rows) for rows in pred_by_frame.values())
    matched_default = len(default_matches)
    xy_errors = finite(to_float(row.get("xy_error_m", "")) for row in default_matches)
    xyz_errors = finite(to_float(row.get("xyz_error_m", "")) for row in default_matches)
    yaw_errors = finite(to_float(row.get("yaw_abs_error_deg", "")) for row in default_matches)
    length_errors = finite(to_float(row.get("length_abs_error_m", "")) for row in default_matches)
    width_errors = finite(to_float(row.get("width_abs_error_m", "")) for row in default_matches)
    height_errors = finite(to_float(row.get("height_abs_error_m", "")) for row in default_matches)

    source_prediction_csvs = sorted({row.get("_source_csv", "") for row in pred_rows if row.get("_source_csv")})
    source_ground_truth_csvs = sorted({row.get("_source_csv", "") for row in gt_rows if row.get("_source_csv")})
    summary: Dict[str, object] = {
        "run_group": run_group,
        "stream_id": stream_id,
        "platform": stream_platform(stream_id),
        "frames": len(frame_keys),
        "gt_vehicle_count": total_gt,
        "predicted_vehicle_count": total_pred,
        "match_distance_m": float(match_distance_m),
        "matched_vehicle_count": matched_default,
        "precision_at_match_distance": matched_default / total_pred if total_pred else float("nan"),
        "false_positives_per_frame": (
            max(0, total_pred - matched_default) / len(frame_keys) if frame_keys else float("nan")
        ),
        "mean_xy_error_m": mean(xy_errors),
        "median_xy_error_m": percentile(xy_errors, 50),
        "p95_xy_error_m": percentile(xy_errors, 95),
        "mean_xyz_error_m": mean(xyz_errors),
        "mean_abs_yaw_error_deg": mean(yaw_errors),
        "mean_length_abs_error_m": mean(length_errors),
        "mean_width_abs_error_m": mean(width_errors),
        "mean_height_abs_error_m": mean(height_errors),
        "source_prediction_csv_count": len(source_prediction_csvs),
        "source_ground_truth_csv_count": len(source_ground_truth_csvs),
    }
    for threshold, count in matched_counts.items():
        summary[threshold_field(threshold)] = count / total_gt if total_gt else float("nan")
    return summary, default_matches


def platform_summary(
    *,
    run_group: str,
    stream_summaries: Sequence[Dict[str, object]],
    match_rows_out: Sequence[Dict[str, object]],
    match_distance_m: float,
) -> List[Dict[str, object]]:
    by_platform: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for summary in stream_summaries:
        by_platform[str(summary.get("platform", "unknown"))].append(summary)
    matches_by_platform: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in match_rows_out:
        matches_by_platform[str(row.get("platform", "unknown"))].append(row)

    rows: List[Dict[str, object]] = []
    for platform, summaries in sorted(by_platform.items()):
        total_gt = sum(to_float(summary.get("gt_vehicle_count", 0)) for summary in summaries)
        total_pred = sum(to_float(summary.get("predicted_vehicle_count", 0)) for summary in summaries)
        total_matched = sum(to_float(summary.get("matched_vehicle_count", 0)) for summary in summaries)
        total_frames = sum(to_float(summary.get("frames", 0)) for summary in summaries)
        matches = matches_by_platform.get(platform, [])
        rows.append(
            {
                "run_group": run_group,
                "platform": platform,
                "streams": len(summaries),
                "frames": total_frames,
                "gt_vehicle_count": total_gt,
                "predicted_vehicle_count": total_pred,
                "match_distance_m": float(match_distance_m),
                "matched_vehicle_count": total_matched,
                "recall_at_match_distance": total_matched / total_gt if total_gt else float("nan"),
                "precision_at_match_distance": total_matched / total_pred if total_pred else float("nan"),
                "false_positives_per_frame": (
                    max(0.0, total_pred - total_matched) / total_frames
                    if total_frames
                    else float("nan")
                ),
                "mean_xy_error_m": mean(finite(to_float(row.get("xy_error_m", "")) for row in matches)),
                "mean_xyz_error_m": mean(finite(to_float(row.get("xyz_error_m", "")) for row in matches)),
                "mean_abs_yaw_error_deg": mean(
                    finite(to_float(row.get("yaw_abs_error_deg", "")) for row in matches)
                ),
            }
        )
    return rows


def gt_filter_summary(
    *,
    run_group: str,
    raw_gt_rows: Sequence[Dict[str, str]],
    selected_gt_rows: Sequence[Dict[str, str]],
    stream_ids: Sequence[str],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for stream_id in stream_ids:
        raw_count = sum(1 for row in raw_gt_rows if row.get("stream_id") == stream_id)
        selected_count = sum(1 for row in selected_gt_rows if row.get("stream_id") == stream_id)
        rows.append(
            {
                "run_group": run_group,
                "stream_id": stream_id,
                "platform": stream_platform(stream_id),
                "raw_gt_vehicle_count": raw_count,
                "selected_gt_vehicle_count": selected_count,
                "dropped_gt_vehicle_count": max(0, raw_count - selected_count),
                "min_gt_bbox_area_px": float(args.min_gt_bbox_area_px),
                "min_gt_bbox_width_px": float(args.min_gt_bbox_width_px),
                "min_gt_bbox_height_px": float(args.min_gt_bbox_height_px),
                "max_gt_distance_m": float(args.max_gt_distance_m),
                "require_gt_center_in_image": int(bool(args.require_gt_center_in_image)),
                "camera_width": int(args.camera_width),
                "camera_height": int(args.camera_height),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def format_value(value: object) -> str:
    number = to_float(value)
    if math.isfinite(number):
        return f"{number:.3f}"
    return str(value)


def write_markdown(
    path: Path,
    *,
    run_group: str,
    stream_summaries: Sequence[Dict[str, object]],
    platform_summaries: Sequence[Dict[str, object]],
    threshold_fields: Sequence[str],
) -> None:
    stream_columns = [
        "stream_id",
        "platform",
        "frames",
        "gt_vehicle_count",
        "predicted_vehicle_count",
        "matched_vehicle_count",
        *threshold_fields,
        "precision_at_match_distance",
        "mean_xy_error_m",
        "mean_abs_yaw_error_deg",
    ]
    platform_columns = [
        "platform",
        "streams",
        "gt_vehicle_count",
        "predicted_vehicle_count",
        "matched_vehicle_count",
        "recall_at_match_distance",
        "precision_at_match_distance",
        "mean_xy_error_m",
    ]
    lines = [f"# Fusion Object Transfer: {run_group}", ""]
    lines.append("## Streams")
    lines.append("| " + " | ".join(stream_columns) + " |")
    lines.append("| " + " | ".join("---" for _ in stream_columns) + " |")
    for summary in stream_summaries:
        lines.append("| " + " | ".join(format_value(summary.get(column, "")) for column in stream_columns) + " |")
    lines.append("")
    lines.append("## Platforms")
    lines.append("| " + " | ".join(platform_columns) + " |")
    lines.append("| " + " | ".join("---" for _ in platform_columns) + " |")
    for summary in platform_summaries:
        lines.append("| " + " | ".join(format_value(summary.get(column, "")) for column in platform_columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summaries(
    out_dir: Path,
    *,
    stream_summaries: Sequence[Dict[str, object]],
    platform_summaries: Sequence[Dict[str, object]],
    recall_field: str,
) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipped plots: {exc}", file=sys.stderr)
        return []

    paths: List[Path] = []
    ordered_streams = sorted(stream_summaries, key=lambda row: (str(row.get("platform", "")), str(row.get("stream_id", ""))))
    labels = [str(row.get("stream_id", "")) for row in ordered_streams]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    recall_values = [to_float(row.get(recall_field, "")) for row in ordered_streams]
    xy_values = [to_float(row.get("mean_xy_error_m", "")) for row in ordered_streams]
    axes[0].bar(labels, recall_values, color="#4c78a8")
    axes[0].set_title("Vehicle Recall")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, xy_values, color="#f58518")
    axes[1].set_title("Mean XY Error (m)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = out_dir / "fusion_object_stream_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if platform_summaries:
        ordered_platforms = sorted(platform_summaries, key=lambda row: str(row.get("platform", "")))
        labels = [str(row.get("platform", "")) for row in ordered_platforms]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
        axes[0].bar(
            labels,
            [to_float(row.get("recall_at_match_distance", "")) for row in ordered_platforms],
            color="#54a24b",
        )
        axes[0].set_title("Platform Recall")
        axes[0].set_ylim(0, 1.0)
        axes[1].bar(
            labels,
            [to_float(row.get("mean_xy_error_m", "")) for row in ordered_platforms],
            color="#e45756",
        )
        axes[1].set_title("Platform Mean XY Error (m)")
        fig.tight_layout()
        path = out_dir / "fusion_object_platform_summary.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    pred_rows = load_csvs(root, "object_predictions")
    gt_rows = load_csvs(root, "object_ground_truth")

    if args.list_groups:
        list_groups(pred_rows, gt_rows)
        return

    run_group = str(args.run_group).strip() or latest_run_group(pred_rows + gt_rows)
    if not run_group:
        print(f"No fusion object prediction/ground-truth rows found under {root}.", file=sys.stderr)
        raise SystemExit(1)

    stream_filter = {str(value).strip() for value in args.stream_id if str(value).strip()}
    pred_rows = [
        row
        for row in pred_rows
        if row.get("run_group") == run_group
        and (not stream_filter or row.get("stream_id") in stream_filter)
        and to_float(row.get("score", "")) >= float(args.score_threshold)
    ]
    gt_rows_unfiltered = [
        row
        for row in gt_rows
        if row.get("run_group") == run_group
        and (not stream_filter or row.get("stream_id") in stream_filter)
        and (args.include_out_of_frustum_gt or to_bool(row.get("in_camera_frustum", "")))
    ]
    gt_rows = [row for row in gt_rows_unfiltered if gt_passes_filters(row, args)]
    if not pred_rows and not gt_rows:
        print(f"No selected rows for run_group={run_group!r} under {root}.", file=sys.stderr)
        raise SystemExit(1)

    thresholds_m = parse_thresholds(args.distance_thresholds_m)
    recall_fields = [threshold_field(threshold) for threshold in thresholds_m]
    summary_fields = [*BASE_SUMMARY_FIELDS, *recall_fields]
    stream_ids = sorted({row.get("stream_id", "") for row in pred_rows + gt_rows if row.get("stream_id")})
    stream_summaries: List[Dict[str, object]] = []
    match_rows_out: List[Dict[str, object]] = []
    for stream_id in stream_ids:
        summary, matches = summarize_stream(
            run_group=run_group,
            stream_id=stream_id,
            gt_rows=[row for row in gt_rows if row.get("stream_id") == stream_id],
            pred_rows=[row for row in pred_rows if row.get("stream_id") == stream_id],
            thresholds_m=thresholds_m,
            match_distance_m=float(args.match_distance_m),
        )
        stream_summaries.append(summary)
        match_rows_out.extend(matches)

    platform_summaries = platform_summary(
        run_group=run_group,
        stream_summaries=stream_summaries,
        match_rows_out=match_rows_out,
        match_distance_m=float(args.match_distance_m),
    )
    gt_filter_rows = gt_filter_summary(
        run_group=run_group,
        raw_gt_rows=gt_rows_unfiltered,
        selected_gt_rows=gt_rows,
        stream_ids=stream_ids,
        args=args,
    )
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else DEFAULT_OUTPUT_ROOT / f"{clean_token(run_group)}_fusion_object_transfer"
    )
    write_csv(out_dir / "fusion_object_transfer_summary.csv", summary_fields, stream_summaries)
    write_csv(out_dir / "fusion_object_transfer_matches.csv", MATCH_FIELDS, match_rows_out)
    write_csv(out_dir / "fusion_object_transfer_platform_summary.csv", PLATFORM_SUMMARY_FIELDS, platform_summaries)
    write_csv(out_dir / "fusion_object_transfer_gt_filter_summary.csv", GT_FILTER_SUMMARY_FIELDS, gt_filter_rows)
    write_markdown(
        out_dir / "fusion_object_transfer_summary.md",
        run_group=run_group,
        stream_summaries=stream_summaries,
        platform_summaries=platform_summaries,
        threshold_fields=recall_fields,
    )
    plot_paths: List[Path] = []
    if not args.no_plots:
        plot_paths = plot_summaries(
            out_dir,
            stream_summaries=stream_summaries,
            platform_summaries=platform_summaries,
            recall_field=threshold_field(float(args.match_distance_m)),
        )

    print(f"Analyzed run_group={run_group}")
    for summary in stream_summaries:
        recall = format_value(summary.get(threshold_field(float(args.match_distance_m)), ""))
        xy = format_value(summary.get("mean_xy_error_m", ""))
        print(
            f"- {summary['stream_id']}: gt={summary['gt_vehicle_count']} "
            f"pred={summary['predicted_vehicle_count']} recall={recall} mean_xy={xy}m"
        )
    print(f"Wrote: {out_dir / 'fusion_object_transfer_summary.csv'}")
    print(f"Wrote: {out_dir / 'fusion_object_transfer_platform_summary.csv'}")
    print(f"Wrote: {out_dir / 'fusion_object_transfer_gt_filter_summary.csv'}")
    print(f"Wrote: {out_dir / 'fusion_object_transfer_summary.md'}")
    for path in plot_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

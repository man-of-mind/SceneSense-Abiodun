#!/usr/bin/env python3
"""Post-hoc FOV-position localization analysis on natural moving-ego data.

This script avoids the artificial Experiment-3 one-car scene. It reuses
existing evaluator/live logs and bins natural vehicle opportunities by
horizontal FOV position, with distance bands kept separate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-scenesense")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AB = Path(__file__).resolve().parents[1]
OUT = AB / "staleness" / "fov_posthoc"
OUT.mkdir(parents=True, exist_ok=True)

SCORE_THRESHOLD = 0.20
MATCH_GATE_M = 2.0
MAX_DISTANCE_M = 40.0
CLASS_NAME = "vehicle"
CAMERA_FOV_DEG = 120.0
CAMERA_WIDTH_PX = 1280.0

ABS_BINS = [
    (0.0, 10.0, "center 0-10"),
    (10.0, 25.0, "inner 10-25"),
    (25.0, 40.0, "outer 25-40"),
    (40.0, 60.0, "edge 40-60"),
]
SIGNED_BINS = [
    (-60.0, -40.0, "left edge"),
    (-40.0, -25.0, "left outer"),
    (-25.0, -10.0, "left inner"),
    (-10.0, 10.0, "center"),
    (10.0, 25.0, "right inner"),
    (25.0, 40.0, "right outer"),
    (40.0, 60.0, "right edge"),
]
DISTANCE_BINS = [
    (0.0, 15.0, "0-15m"),
    (15.0, 25.0, "15-25m"),
    (25.0, 40.0, "25-40m"),
]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bin_label(value: float, bins: Sequence[Tuple[float, float, str]]) -> Optional[str]:
    if not math.isfinite(value):
        return None
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    if bins and abs(value - bins[-1][1]) < 1e-9:
        return bins[-1][2]
    return None


def bin_center(label: str, bins: Sequence[Tuple[float, float, str]]) -> float:
    for lo, hi, candidate in bins:
        if label == candidate:
            return (lo + hi) / 2.0
    return float("nan")


def pixel_to_angle_deg(x_px: float, width_px: float, fov_deg: float) -> float:
    cx = width_px / 2.0
    fx = width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return math.degrees(math.atan2(x_px - cx, fx))


def safe_key_world(row: Dict[str, str]) -> Tuple[float, float]:
    return finite_float(row.get("gt_world_x")), finite_float(row.get("gt_world_y"))


def load_object_boxes(path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_sample: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("label", "")).lower() == CLASS_NAME:
                by_sample[row["sample_id"]].append(row)
    return by_sample


def nearest_box(row: Dict[str, str], boxes_by_sample: Dict[str, List[Dict[str, str]]]) -> Optional[Dict[str, str]]:
    gx, gy = safe_key_world(row)
    if not all(math.isfinite(value) for value in (gx, gy)):
        return None
    candidates = boxes_by_sample.get(row.get("sample_id", ""), [])
    if not candidates:
        return None
    box = min(
        candidates,
        key=lambda item: (
            finite_float(item.get("object_world_x")) - gx
        )
        ** 2
        + (
            finite_float(item.get("object_world_y")) - gy
        )
        ** 2,
    )
    wx = finite_float(box.get("object_world_x"))
    wy = finite_float(box.get("object_world_y"))
    if not all(math.isfinite(value) for value in (wx, wy)):
        return None
    if math.hypot(wx - gx, wy - gy) > 3.0:
        return None
    return box


def add_derived_fields(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    distance_m = float(record.get("distance_m", float("nan")))
    pixel_x = float(record.get("pixel_x", float("nan")))
    width = float(record.get("image_width_px", CAMERA_WIDTH_PX))
    fov = float(record.get("camera_fov_deg", CAMERA_FOV_DEG))
    if not (math.isfinite(distance_m) and 0.0 <= distance_m <= MAX_DISTANCE_M):
        return None
    if not (math.isfinite(pixel_x) and width > 0.0):
        return None
    angle = pixel_to_angle_deg(pixel_x, width, fov)
    if not -70.0 <= angle <= 70.0:
        return None
    abs_angle = abs(angle)
    abs_bin = bin_label(abs_angle, ABS_BINS)
    signed_bin = bin_label(angle, SIGNED_BINS)
    distance_bin = bin_label(distance_m, DISTANCE_BINS)
    if abs_bin is None or signed_bin is None or distance_bin is None:
        return None
    record.update(
        {
            "angle_deg": angle,
            "abs_angle_deg": abs_angle,
            "abs_fov_bin": abs_bin,
            "signed_fov_bin": signed_bin,
            "distance_bin": distance_bin,
        }
    )
    return record


def load_offline_records() -> List[Dict[str, object]]:
    boxes = load_object_boxes(AB / "staleness" / "egospeed_split_ds" / "object_boxes.csv")
    records: List[Dict[str, object]] = []
    for split, metrics_path in [
        ("offline_test", AB / "staleness" / "egospeed_eval" / "metrics" / "test_learned_object_metrics.csv"),
        ("offline_val", AB / "staleness" / "egospeed_eval" / "metrics" / "val_learned_object_metrics.csv"),
    ]:
        if not metrics_path.exists():
            continue
        for row in read_csv(metrics_path):
            status = str(row.get("match_status", "")).lower()
            gt_class = str(row.get("gt_class_name", "")).lower()
            if status not in {"tp", "fn"} or gt_class != CLASS_NAME:
                continue
            box = nearest_box(row, boxes)
            if box is None:
                continue
            matched = status == "tp"
            record = {
                "source": "offline",
                "subset": split,
                "sample_id": row.get("sample_id", ""),
                "frame_id": row.get("frame_id", ""),
                "matched": int(matched),
                "error_m": finite_float(row.get("global_xy_error_m")) if matched else float("nan"),
                "score": finite_float(row.get("score")) if matched else float("nan"),
                "pixel_x": finite_float(box.get("gt_center_x")),
                "pixel_y": finite_float(box.get("gt_center_y")),
                "distance_m": finite_float(box.get("gt_distance_m")),
                "depth_m": finite_float(box.get("gt_depth_m")),
                "object_speed_mps": finite_float(box.get("object_speed_mps")),
                "stationary_label": str(box.get("stationary_label", "")),
                "image_width_px": CAMERA_WIDTH_PX,
                "camera_fov_deg": CAMERA_FOV_DEG,
            }
            record = add_derived_fields(record)
            if record is not None:
                records.append(record)
    return records


def one_file(root: Path, pattern: str) -> Optional[Path]:
    paths = sorted(root.glob(pattern))
    return paths[0] if paths else None


def live_run_kind(run_dir: Path) -> Optional[str]:
    metrics_path = one_file(run_dir, "streams/*_metrics.csv")
    if metrics_path is None:
        return None
    try:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle), None)
    except (OSError, StopIteration):
        return None
    group = str((first or {}).get("run_group", ""))
    if group.startswith("speedsweep_"):
        return "live_speedsweep"
    if group == "ctrl_movingego_200k":
        return "live_moving_control"
    return None


def load_json(path: Optional[Path]) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_live_records() -> List[Dict[str, object]]:
    root = AB / "staleness" / "metrics_logs" / "scenesense_runs"
    records: List[Dict[str, object]] = []
    if not root.exists():
        return records
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        kind = live_run_kind(run_dir)
        if kind is None:
            continue
        gt_path = one_file(run_dir, "streams/*_object_ground_truth.csv")
        pred_path = one_file(run_dir, "streams/*_object_predictions.csv")
        config = load_json(one_file(run_dir, "manifests/*_resolved_config.json"))
        if gt_path is None or pred_path is None:
            continue
        width = float(config.get("camera_width", 0.0) or CAMERA_WIDTH_PX)
        fov = float(config.get("camera_fov", 0.0) or CAMERA_FOV_DEG)
        pps = int(float(config.get("radar_points_per_second", 0.0) or 0.0))
        preds_by_frame: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        for row in read_csv(pred_path):
            if str(row.get("class_name", "")).lower() != CLASS_NAME:
                continue
            if finite_float(row.get("score"), -1.0) < SCORE_THRESHOLD:
                continue
            preds_by_frame[int(finite_float(row.get("frame_id"), -1))].append(row)
        gt_by_frame: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        for row in read_csv(gt_path):
            if str(row.get("class_name", "")).lower() != CLASS_NAME:
                continue
            if not truthy(row.get("in_camera_frustum", "")):
                continue
            distance = finite_float(row.get("distance_m"))
            pixel_x = finite_float(row.get("projected_x"))
            if not (math.isfinite(distance) and math.isfinite(pixel_x)):
                continue
            if distance > MAX_DISTANCE_M:
                continue
            gt_by_frame[int(finite_float(row.get("frame_id"), -1))].append(row)

        for frame_id, gt_rows in gt_by_frame.items():
            pred_rows = preds_by_frame.get(frame_id, [])
            pairs = []
            for gt_idx, gt in enumerate(gt_rows):
                gx = finite_float(gt.get("origin_x"), finite_float(gt.get("world_x")))
                gy = finite_float(gt.get("origin_y"), finite_float(gt.get("world_y")))
                if not all(math.isfinite(value) for value in (gx, gy)):
                    continue
                for pred_idx, pred in enumerate(pred_rows):
                    px = finite_float(pred.get("world_x"))
                    py = finite_float(pred.get("world_y"))
                    if not all(math.isfinite(value) for value in (px, py)):
                        continue
                    error = math.hypot(px - gx, py - gy)
                    if error <= MATCH_GATE_M:
                        pairs.append((error, gt_idx, pred_idx, pred))
            pairs.sort(key=lambda item: item[0])
            matched_gt: Dict[int, Tuple[float, Dict[str, str]]] = {}
            used_preds = set()
            for error, gt_idx, pred_idx, pred in pairs:
                if gt_idx in matched_gt or pred_idx in used_preds:
                    continue
                matched_gt[gt_idx] = (error, pred)
                used_preds.add(pred_idx)

            for gt_idx, gt in enumerate(gt_rows):
                match = matched_gt.get(gt_idx)
                record = {
                    "source": kind,
                    "subset": run_dir.name,
                    "run_group": str(config.get("resolved_run_group", "")),
                    "radar_points_per_second": pps,
                    "sample_id": f"{run_dir.name}_frame{frame_id}_actor{gt.get('actor_id', '')}",
                    "frame_id": frame_id,
                    "matched": int(match is not None),
                    "error_m": match[0] if match is not None else float("nan"),
                    "score": finite_float(match[1].get("score")) if match is not None else float("nan"),
                    "pixel_x": finite_float(gt.get("projected_x")),
                    "pixel_y": finite_float(gt.get("projected_y")),
                    "distance_m": finite_float(gt.get("distance_m")),
                    "depth_m": finite_float(gt.get("distance_m")),
                    "image_width_px": width,
                    "camera_fov_deg": fov,
                }
                record = add_derived_fields(record)
                if record is not None:
                    records.append(record)
    return records


def summarize(records: Sequence[Dict[str, object]], bin_key: str) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[
            (
                str(record["source"]),
                str(record.get("distance_bin", "all")),
                str(record.get(bin_key, "")),
            )
        ].append(record)
    rows: List[Dict[str, object]] = []
    for (source, distance_bin, fov_bin), group in sorted(groups.items()):
        errors = [float(row["error_m"]) for row in group if int(row.get("matched", 0)) and math.isfinite(float(row["error_m"]))]
        scores = [float(row["score"]) for row in group if int(row.get("matched", 0)) and math.isfinite(float(row["score"]))]
        opportunities = len(group)
        matches = len(errors)
        rows.append(
            {
                "source": source,
                "distance_bin": distance_bin,
                "fov_bin": fov_bin,
                "opportunities": opportunities,
                "matches": matches,
                "availability": matches / opportunities if opportunities else 0.0,
                "error_mean_m": mean(errors),
                "error_median_m": median(errors),
                "error_p90_m": percentile(errors, 0.90),
                "score_mean": mean(scores),
                "distance_mean_m": mean([float(row["distance_m"]) for row in group]),
                "angle_abs_mean_deg": mean([float(row["abs_angle_deg"]) for row in group]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_label(source: str) -> str:
    return {
        "offline": "offline eval",
        "live_speedsweep": "live speed sweep",
        "live_moving_control": "live moving control",
    }.get(source, source)


def plot_metric(rows: Sequence[Dict[str, object]], source: str, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for dist_label in [label for _, _, label in DISTANCE_BINS]:
        series = [
            row
            for row in rows
            if row["source"] == source
            and row["distance_bin"] == dist_label
            and row["fov_bin"] in {label for _, _, label in ABS_BINS}
            and int(row["opportunities"]) >= 5
        ]
        if not series:
            continue
        series.sort(key=lambda row: bin_center(str(row["fov_bin"]), ABS_BINS))
        xs = [bin_center(str(row["fov_bin"]), ABS_BINS) for row in series]
        ys = [float(row[metric]) for row in series]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=dist_label)
        for x, y, row in zip(xs, ys, series):
            if math.isfinite(y):
                ax.text(x, y, f"n={int(row['opportunities'])}", fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("absolute horizontal FOV angle bin center (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{source_label(source)}: vehicle {ylabel.lower()} vs FOV position")
    ax.set_xticks([bin_center(label, ABS_BINS) for _, _, label in ABS_BINS])
    ax.set_xticklabels([label.replace(" ", "\n") for _, _, label in ABS_BINS])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title="distance")
    fig.tight_layout()
    fig.savefig(OUT / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def md_table(rows: Sequence[Dict[str, object]], source: str, distance_bin: str) -> List[str]:
    selected = [
        row
        for row in rows
        if row["source"] == source
        and row["distance_bin"] == distance_bin
        and row["fov_bin"] in {label for _, _, label in ABS_BINS}
    ]
    selected.sort(key=lambda row: bin_center(str(row["fov_bin"]), ABS_BINS))
    lines = [
        f"### {source_label(source)} — {distance_bin}",
        "",
        "| FOV bin | opportunities | matches | availability | median err | mean err | p90 err | mean score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {fov_bin} | {opportunities} | {matches} | {availability:.3f} | {median} | {mean} | {p90} | {score} |".format(
                fov_bin=row["fov_bin"],
                opportunities=int(row["opportunities"]),
                matches=int(row["matches"]),
                availability=float(row["availability"]),
                median=fmt(float(row["error_median_m"])),
                mean=fmt(float(row["error_mean_m"])),
                p90=fmt(float(row["error_p90_m"])),
                score=fmt(float(row["score_mean"])),
            )
        )
    lines.append("")
    return lines


def find_summary_row(
    rows: Sequence[Dict[str, object]], source: str, distance_bin: str, fov_bin: str
) -> Optional[Dict[str, object]]:
    for row in rows:
        if (
            row["source"] == source
            and row["distance_bin"] == distance_bin
            and row["fov_bin"] == fov_bin
        ):
            return row
    return None


def compact_row(row: Optional[Dict[str, object]]) -> str:
    if row is None:
        return "n/a"
    return (
        f"n={int(row['opportunities'])}, avail={float(row['availability']):.3f}, "
        f"median={fmt(float(row['error_median_m']))} m, score={fmt(float(row['score_mean']))}"
    )


def write_report(abs_rows: Sequence[Dict[str, object]], signed_rows: Sequence[Dict[str, object]], records: Sequence[Dict[str, object]]) -> None:
    counts = defaultdict(int)
    matches = defaultdict(int)
    for record in records:
        counts[str(record["source"])] += 1
        matches[str(record["source"])] += int(record.get("matched", 0))

    lines = [
        "# Post-hoc natural-scene FOV localization split",
        "",
        "Vehicle-only analysis at score >= 0.20, match gate <= 2 m, objects <= 40 m.",
        "The point is to test FOV position on natural moving-ego data, not the artificial one-car Experiment-3 scene.",
        "",
        "## Dataset counts",
        "",
        "| source | opportunities | matches | availability |",
        "|---|---:|---:|---:|",
    ]
    for source in sorted(counts):
        lines.append(
            f"| {source_label(source)} | {counts[source]} | {matches[source]} | "
            f"{(matches[source] / counts[source] if counts[source] else 0.0):.3f} |"
        )
    offline_near_center = find_summary_row(abs_rows, "offline", "0-15m", "center 0-10")
    offline_near_edge = find_summary_row(abs_rows, "offline", "0-15m", "edge 40-60")
    offline_mid_center = find_summary_row(abs_rows, "offline", "15-25m", "center 0-10")
    offline_mid_edge = find_summary_row(abs_rows, "offline", "15-25m", "edge 40-60")
    offline_far_center = find_summary_row(abs_rows, "offline", "25-40m", "center 0-10")
    offline_far_edge = find_summary_row(abs_rows, "offline", "25-40m", "edge 40-60")
    live_control_far_center = find_summary_row(abs_rows, "live_moving_control", "25-40m", "center 0-10")
    live_control_far_edge = find_summary_row(abs_rows, "live_moving_control", "25-40m", "edge 40-60")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The natural offline 200k split gives a meaningful FOV signal, but not a simple monotonic "
            "`center is always best` curve.",
            "- Near vehicles (0-15 m) are easy across the FOV; edge localization is not worse there. "
            f"Offline center: {compact_row(offline_near_center)}; edge: {compact_row(offline_near_edge)}.",
            "- For 15-40 m vehicles, edge bins lose availability and usually have worse matched localization. "
            f"Offline 15-25 m center: {compact_row(offline_mid_center)}; edge: {compact_row(offline_mid_edge)}. "
            f"Offline 25-40 m center: {compact_row(offline_far_center)}; edge: {compact_row(offline_far_edge)}.",
            "- The 200k live moving-control run is smaller/noisier but points the same way for availability: "
            f"25-40 m center: {compact_row(live_control_far_center)}; edge: {compact_row(live_control_far_edge)}.",
            "- The live speed-sweep logs are included only as a secondary check because they are 5k-PPS runs; "
            "the offline split and the 200k moving-control run are the best references for the current 200k model.",
            "- Practical takeaway for RL/FOV prioritization: use range-aware edge risk, not a blanket center prior. "
            "At close range, edge objects can be localized well; at medium/far range, edge objects are more likely "
            "to be missed and sometimes localize worse when matched.",
        ]
    )
    lines.extend(["", "## Absolute FOV bins", ""])
    for source in ["offline", "live_speedsweep", "live_moving_control"]:
        if source not in counts:
            continue
        for distance_bin in [label for _, _, label in DISTANCE_BINS]:
            lines.extend(md_table(abs_rows, source, distance_bin))
    lines.extend(
        [
            "## Files",
            "",
            "- `offline_live_vehicle_abs_fov_summary.csv` — absolute-angle bins by source and distance.",
            "- `offline_live_vehicle_signed_fov_summary.csv` — signed left/right bins by source and distance.",
            "- `*_fov_error_by_distance.{png,pdf}` — median localization error vs absolute FOV position.",
            "- `*_fov_availability_by_distance.{png,pdf}` — match availability vs absolute FOV position.",
            "",
        ]
    )
    (OUT / "FOV_POSTHOC_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    offline = load_offline_records()
    live = load_live_records()
    records = offline + live
    write_csv(OUT / "offline_live_vehicle_fov_records.csv", records)
    abs_rows = summarize(records, "abs_fov_bin")
    signed_rows = summarize(records, "signed_fov_bin")
    write_csv(OUT / "offline_live_vehicle_abs_fov_summary.csv", abs_rows)
    write_csv(OUT / "offline_live_vehicle_signed_fov_summary.csv", signed_rows)
    for source in sorted({str(row["source"]) for row in records}):
        plot_metric(abs_rows, source, "error_median_m", "median localization error (m)", f"{source}_fov_error_by_distance")
        plot_metric(abs_rows, source, "availability", "match availability", f"{source}_fov_availability_by_distance")
    write_report(abs_rows, signed_rows, records)
    print(f"offline records: {len(offline)}")
    print(f"live records: {len(live)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

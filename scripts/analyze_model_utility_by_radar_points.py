#!/usr/bin/env python3
"""Connect radar point count to localization utility.

Binary radar support can be misleading: one radar point on a pedestrian is
technically support, but may not be enough to improve a model prediction. This
script joins evaluation TP/FN rows with the dataset's `object_boxes.csv`, then
bins ground-truth objects by distance and associated radar points.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - CSV outputs still work.
    plt = None  # type: ignore


DEFAULT_POINT_BINS = "0,1,5,10,20,50,inf"
DEFAULT_DISTANCE_BINS = "0,10,20,30,40,60,80,100,inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-metrics",
        nargs="+",
        required=True,
        help="Evaluation CSV(s), usually metrics/test_learned_object_metrics.csv.",
    )
    parser.add_argument(
        "--dataset-dir",
        nargs="+",
        required=True,
        help="Fusion dataset dir(s) containing object_boxes.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="abiodun/analysis_outputs/model_utility_by_radar_points",
        help="Directory for CSV, plots, and Markdown summary.",
    )
    parser.add_argument("--point-bins", default=DEFAULT_POINT_BINS)
    parser.add_argument("--distance-bins-m", default=DEFAULT_DISTANCE_BINS)
    parser.add_argument(
        "--match-world-tolerance-m",
        type=float,
        default=3.0,
        help="Maximum GT world-XY distance used when joining eval rows to object_boxes.csv.",
    )
    parser.add_argument(
        "--min-objects-for-threshold",
        type=int,
        default=20,
        help="Minimum objects required before the script suggests a useful radar-point threshold.",
    )
    return parser.parse_args()


def parse_bins(raw: str) -> List[float]:
    bins: List[float] = []
    for part in raw.split(","):
        text = part.strip().lower()
        if not text:
            continue
        bins.append(math.inf if text in {"inf", "infinity"} else float(text))
    if len(bins) < 2 or bins != sorted(bins):
        raise ValueError(f"Invalid sorted bins: {raw!r}")
    return bins


def bin_label(value: float, bins: Sequence[float], *, integer_points: bool = False) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            if integer_points and lo == 0 and hi == 1:
                return "0"
            hi_text = "inf" if math.isinf(hi) else f"{int(hi) if hi.is_integer() else hi:g}"
            lo_text = f"{int(lo) if lo.is_integer() else lo:g}"
            if integer_points and math.isfinite(hi):
                return f"{lo_text}-{int(hi) - 1}"
            return f"{lo_text}-{hi_text}"
    return "unknown"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def to_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        raw = str(row.get(key, "")).strip()
        if raw == "":
            return default
        value = float(raw)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def short_name(path: Path) -> str:
    name = path.name
    for token in (
        "test_learned_object_metrics",
        "moving_ego_",
        "radarpps",
        "bboxsupport",
        "fusion_train",
    ):
        name = name.replace(token, "")
    return name.strip("_") or path.parent.parent.name


def load_object_boxes(dataset_dirs: Sequence[Path]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    boxes: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for dataset_dir in dataset_dirs:
        path = dataset_dir / "object_boxes.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing object_boxes.csv: {path}")
        for row in read_csv(path):
            label = str(row.get("label", "")).strip()
            if label not in {"vehicle", "person"}:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            boxes[(sample_id, label)].append(
                {
                    "dataset_dir": str(dataset_dir),
                    "sample_id": sample_id,
                    "label": label,
                    "gt_actor_id": str(row.get("gt_actor_id", "")),
                    "world_x": to_float(row, "object_world_x"),
                    "world_y": to_float(row, "object_world_y"),
                    "distance_m": to_float(row, "gt_distance_m"),
                    "depth_m": to_float(row, "gt_depth_m"),
                    "bbox_area_px": to_float(row, "gt_bbox_area_px"),
                    "radar_support_points": to_float(row, "radar_support_points", 0.0),
                    "object_speed_mps": to_float(row, "object_speed_mps"),
                    "stationary_label": to_float(row, "stationary_label", 0.0),
                    "parked_label": to_float(row, "parked_label", 0.0),
                }
            )
    return boxes


def nearest_box(
    boxes: Dict[Tuple[str, str], List[Dict[str, object]]],
    *,
    sample_id: str,
    label: str,
    world_x: float,
    world_y: float,
    tolerance_m: float,
) -> Optional[Dict[str, object]]:
    candidates = boxes.get((sample_id, label), [])
    if not candidates:
        return None
    best: Optional[Dict[str, object]] = None
    best_dist = float("inf")
    for box in candidates:
        bx = float(box.get("world_x", float("nan")))
        by = float(box.get("world_y", float("nan")))
        if not (math.isfinite(bx) and math.isfinite(by) and math.isfinite(world_x) and math.isfinite(world_y)):
            continue
        dist = math.hypot(world_x - bx, world_y - by)
        if dist < best_dist:
            best = box
            best_dist = dist
    if best is None or best_dist > float(tolerance_m):
        return None
    result = dict(best)
    result["join_xy_error_m"] = best_dist
    return result


def collect_eval_objects(
    metric_paths: Sequence[Path],
    boxes: Dict[Tuple[str, str], List[Dict[str, object]]],
    point_bins: Sequence[float],
    distance_bins: Sequence[float],
    tolerance_m: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric_path in metric_paths:
        run_label = short_name(metric_path.parent.parent)
        for row in read_csv(metric_path):
            status = str(row.get("match_status", "")).strip()
            if status not in {"tp", "fn"}:
                continue
            label = str(row.get("gt_class_name") or row.get("class_name") or "").strip()
            if label not in {"vehicle", "person"}:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            gt_x = to_float(row, "gt_world_x")
            gt_y = to_float(row, "gt_world_y")
            box = nearest_box(
                boxes,
                sample_id=sample_id,
                label=label,
                world_x=gt_x,
                world_y=gt_y,
                tolerance_m=tolerance_m,
            )
            if box is None:
                continue
            radar_points = float(box.get("radar_support_points", 0.0))
            distance = float(box.get("distance_m", float("nan")))
            if not math.isfinite(distance):
                continue
            matched = status == "tp"
            xy_error = to_float(row, "global_xy_error_m") if matched else float("nan")
            rows.append(
                {
                    "run": run_label,
                    "sample_id": sample_id,
                    "label": label,
                    "match_status": status,
                    "matched": int(matched),
                    "distance_m": distance,
                    "distance_bin_m": bin_label(distance, distance_bins),
                    "radar_support_points": radar_points,
                    "radar_point_bin": bin_label(radar_points, point_bins, integer_points=True),
                    "global_xy_error_m": xy_error,
                    "bbox_area_px": float(box.get("bbox_area_px", float("nan"))),
                    "depth_m": float(box.get("depth_m", float("nan"))),
                    "object_speed_mps": float(box.get("object_speed_mps", float("nan"))),
                    "join_xy_error_m": float(box.get("join_xy_error_m", float("nan"))),
                    "metric_csv": str(metric_path),
                    "dataset_dir": str(box.get("dataset_dir", "")),
                }
            )
    return rows


def safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else float("nan")


def safe_median(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def aggregate(rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in fields)].append(row)
    output: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        matched = [int(row["matched"]) for row in group]
        errors = [
            float(row["global_xy_error_m"])
            for row in group
            if int(row["matched"]) and math.isfinite(float(row["global_xy_error_m"]))
        ]
        radar_points = [float(row["radar_support_points"]) for row in group]
        out = {field: value for field, value in zip(fields, key)}
        out.update(
            {
                "gt_objects": len(group),
                "true_positives": int(sum(matched)),
                "recall": float(sum(matched) / len(group)) if group else float("nan"),
                "mean_xy_error_m": safe_mean(errors),
                "median_xy_error_m": safe_median(errors),
                "mean_radar_points": safe_mean(radar_points),
                "median_radar_points": safe_median(radar_points),
            }
        )
        output.append(out)
    return output


def point_bin_order(point_bins: Sequence[float]) -> List[str]:
    return [bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, point_bins, integer_points=True) for lo, hi in zip(point_bins[:-1], point_bins[1:])]


def distance_bin_order(distance_bins: Sequence[float]) -> List[str]:
    return [bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, distance_bins) for lo, hi in zip(distance_bins[:-1], distance_bins[1:])]


def plot_recall_by_point_bin(rows: Sequence[Dict[str, object]], point_bins: Sequence[float], output_dir: Path) -> None:
    if plt is None:
        return
    labels = point_bin_order(point_bins)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    for offset, label in [(-width / 2, "vehicle"), (width / 2, "person")]:
        values = []
        for bucket in labels:
            group = [row for row in rows if row["label"] == label and row["radar_point_bin"] == bucket]
            summary = aggregate(group, [])[0] if group else {"recall": float("nan")}
            values.append(float(summary["recall"]))
        ax.bar(x + offset, values, width=width, label=label.title())
    ax.set_title("Localization recall by radar point count")
    ax.set_xlabel("Radar points associated with GT object")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "localization_recall_by_radar_point_bin.png", dpi=240)
    fig.savefig(output_dir / "localization_recall_by_radar_point_bin.pdf")
    plt.close(fig)


def plot_xy_error_by_point_bin(rows: Sequence[Dict[str, object]], point_bins: Sequence[float], output_dir: Path) -> None:
    if plt is None:
        return
    labels = point_bin_order(point_bins)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    for class_name in ("vehicle", "person"):
        values = []
        for bucket in labels:
            errors = [
                float(row["global_xy_error_m"])
                for row in rows
                if row["label"] == class_name
                and row["radar_point_bin"] == bucket
                and int(row["matched"])
                and math.isfinite(float(row["global_xy_error_m"]))
            ]
            values.append(safe_mean(errors))
        ax.plot(labels, values, marker="o", linewidth=2.2, label=class_name.title())
    ax.set_title("Matched-object XY error by radar point count")
    ax.set_xlabel("Radar points associated with GT object")
    ax.set_ylabel("Mean XY error (m)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "xy_error_by_radar_point_bin.png", dpi=240)
    fig.savefig(output_dir / "xy_error_by_radar_point_bin.pdf")
    plt.close(fig)


def plot_person_heatmap(
    rows: Sequence[Dict[str, object]],
    point_bins: Sequence[float],
    distance_bins: Sequence[float],
    output_dir: Path,
) -> None:
    if plt is None:
        return
    point_labels = point_bin_order(point_bins)
    distance_labels = distance_bin_order(distance_bins)
    matrix = np.full((len(distance_labels), len(point_labels)), np.nan, dtype=np.float64)
    counts = np.zeros_like(matrix)
    for i, distance_bucket in enumerate(distance_labels):
        for j, point_bucket in enumerate(point_labels):
            group = [
                row
                for row in rows
                if row["label"] == "person"
                and row["distance_bin_m"] == distance_bucket
                and row["radar_point_bin"] == point_bucket
            ]
            if group:
                matrix[i, j] = float(sum(int(row["matched"]) for row in group) / len(group))
                counts[i, j] = len(group)
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_title("Person localization recall by distance and radar points")
    ax.set_xlabel("Radar points associated with GT person")
    ax.set_ylabel("Distance from ego (m)")
    ax.set_xticks(np.arange(len(point_labels)), point_labels)
    ax.set_yticks(np.arange(len(distance_labels)), distance_labels)
    for i in range(len(distance_labels)):
        for j in range(len(point_labels)):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}\nn={int(counts[i, j])}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Recall")
    fig.savefig(output_dir / "person_recall_distance_vs_radar_points.png", dpi=240)
    fig.savefig(output_dir / "person_recall_distance_vs_radar_points.pdf")
    plt.close(fig)


def mean_error_for_group(group: Sequence[Dict[str, object]]) -> float:
    errors = [
        float(row["global_xy_error_m"])
        for row in group
        if int(row["matched"]) and math.isfinite(float(row["global_xy_error_m"]))
    ]
    return safe_mean(errors)


def suggest_threshold(
    rows: Sequence[Dict[str, object]],
    class_name: str,
    min_objects: int,
    ordered_point_bins: Sequence[str],
) -> str:
    bucket_stats = []
    for bucket in ordered_point_bins:
        group = [row for row in rows if row["label"] == class_name and row["radar_point_bin"] == bucket]
        if len(group) >= min_objects:
            recall = sum(int(row["matched"]) for row in group) / len(group)
            bucket_stats.append((bucket, len(group), recall, mean_error_for_group(group)))
    zero = next(((recall, xy) for bucket, _, recall, xy in bucket_stats if bucket == "0"), None)
    if zero is None:
        return f"Not enough `{class_name}` objects in the zero-point bucket to estimate a threshold."
    zero_recall, zero_xy = zero
    for bucket, count, recall, xy_error in bucket_stats:
        xy_ok = not math.isfinite(zero_xy) or not math.isfinite(xy_error) or xy_error <= zero_xy
        if bucket != "0" and recall >= zero_recall + 0.05 and xy_ok:
            return (
                f"For `{class_name}`, `{bucket}` radar points is the first populated bucket "
                f"where recall improves by >=0.05 over zero-radar objects without increasing mean XY error "
                f"(recall {zero_recall:.3f} -> {recall:.3f}, xy {zero_xy:.2f} -> {xy_error:.2f} m, n={count})."
            )
    return (
        f"For `{class_name}`, no populated radar-point bucket improved recall by >=0.05 "
        "while also keeping mean XY error no worse than the zero-point bucket."
    )


def write_markdown(
    path: Path,
    *,
    metric_paths: Sequence[Path],
    dataset_dirs: Sequence[Path],
    rows: Sequence[Dict[str, object]],
    point_summary: Sequence[Dict[str, object]],
    distance_summary: Sequence[Dict[str, object]],
    min_objects: int,
    ordered_point_bins: Sequence[str],
) -> None:
    lines = [
        "# Model Utility vs Radar Points",
        "",
        "This analysis asks when radar evidence becomes useful to the localization model.",
        "It joins evaluation TP/FN rows to `object_boxes.csv`, then bins GT objects by distance and associated radar points.",
        "",
        "## Inputs",
        "",
        "Evaluation metrics:",
    ]
    lines.extend(f"- `{path}`" for path in metric_paths)
    lines.append("")
    lines.append("Datasets:")
    lines.extend(f"- `{path}`" for path in dataset_dirs)
    lines.extend(
        [
            "",
            "## Radar-Point Buckets",
            "",
            "| Class | Radar points | GT objects | Recall | Mean XY error (m) | Mean radar pts |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in point_summary:
        lines.append(
            "| {label} | {bucket} | {n} | {recall:.3f} | {xy:.3f} | {pts:.2f} |".format(
                label=row.get("label", ""),
                bucket=row.get("radar_point_bin", ""),
                n=int(row.get("gt_objects", 0)),
                recall=float(row.get("recall", float("nan"))),
                xy=float(row.get("mean_xy_error_m", float("nan"))),
                pts=float(row.get("mean_radar_points", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Person Distance Buckets",
            "",
            "| Distance (m) | GT persons | Recall | Mean XY error (m) | Mean radar pts |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in distance_summary:
        if row.get("label") != "person":
            continue
        lines.append(
            "| {bucket} | {n} | {recall:.3f} | {xy:.3f} | {pts:.2f} |".format(
                bucket=row.get("distance_bin_m", ""),
                n=int(row.get("gt_objects", 0)),
                recall=float(row.get("recall", float("nan"))),
                xy=float(row.get("mean_xy_error_m", float("nan"))),
                pts=float(row.get("mean_radar_points", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Empirical Threshold Notes",
            "",
            f"- {suggest_threshold(rows, 'person', min_objects, ordered_point_bins)}",
            f"- {suggest_threshold(rows, 'vehicle', min_objects, ordered_point_bins)}",
            "",
            "## Presentation Takeaway",
            "",
            "- `>=1 radar point` is only contact, not necessarily useful support.",
            "- Useful support should be defined by a model utility change: higher recall and/or lower XY error.",
            "- The threshold can differ by class because vehicles produce many more radar returns than pedestrians.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    metric_paths = [Path(raw).expanduser().resolve() for raw in args.object_metrics]
    dataset_dirs = [Path(raw).expanduser().resolve() for raw in args.dataset_dir]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_bins = parse_bins(str(args.point_bins))
    distance_bins = parse_bins(str(args.distance_bins_m))

    boxes = load_object_boxes(dataset_dirs)
    rows = collect_eval_objects(
        metric_paths,
        boxes,
        point_bins,
        distance_bins,
        tolerance_m=float(args.match_world_tolerance_m),
    )
    if not rows:
        raise SystemExit("No TP/FN rows could be joined to object_boxes.csv.")

    object_fields = [
        "run",
        "sample_id",
        "label",
        "match_status",
        "matched",
        "distance_m",
        "distance_bin_m",
        "radar_support_points",
        "radar_point_bin",
        "global_xy_error_m",
        "bbox_area_px",
        "depth_m",
        "object_speed_mps",
        "join_xy_error_m",
        "metric_csv",
        "dataset_dir",
    ]
    write_csv(output_dir / "model_utility_objects.csv", rows, object_fields)

    point_summary = aggregate(rows, ["label", "radar_point_bin"])
    distance_summary = aggregate(rows, ["label", "distance_bin_m"])
    joint_summary = aggregate(rows, ["label", "distance_bin_m", "radar_point_bin"])
    point_order = {bucket: index for index, bucket in enumerate(point_bin_order(point_bins))}
    distance_order = {bucket: index for index, bucket in enumerate(distance_bin_order(distance_bins))}
    class_order = {"vehicle": 0, "person": 1}
    point_summary.sort(
        key=lambda row: (
            class_order.get(str(row.get("label", "")), 99),
            point_order.get(str(row.get("radar_point_bin", "")), 99),
        )
    )
    distance_summary.sort(
        key=lambda row: (
            class_order.get(str(row.get("label", "")), 99),
            distance_order.get(str(row.get("distance_bin_m", "")), 99),
        )
    )
    joint_summary.sort(
        key=lambda row: (
            class_order.get(str(row.get("label", "")), 99),
            distance_order.get(str(row.get("distance_bin_m", "")), 99),
            point_order.get(str(row.get("radar_point_bin", "")), 99),
        )
    )
    summary_fields = [
        "label",
        "radar_point_bin",
        "distance_bin_m",
        "gt_objects",
        "true_positives",
        "recall",
        "mean_xy_error_m",
        "median_xy_error_m",
        "mean_radar_points",
        "median_radar_points",
    ]
    write_csv(output_dir / "model_utility_by_radar_point_bin.csv", point_summary, summary_fields)
    write_csv(output_dir / "model_utility_by_distance.csv", distance_summary, summary_fields)
    write_csv(output_dir / "model_utility_by_distance_and_radar_points.csv", joint_summary, summary_fields)

    plot_recall_by_point_bin(rows, point_bins, output_dir)
    plot_xy_error_by_point_bin(rows, point_bins, output_dir)
    plot_person_heatmap(rows, point_bins, distance_bins, output_dir)
    write_markdown(
        output_dir / "model_utility_by_radar_points_summary.md",
        metric_paths=metric_paths,
        dataset_dirs=dataset_dirs,
        rows=rows,
        point_summary=point_summary,
        distance_summary=distance_summary,
        min_objects=int(args.min_objects_for_threshold),
        ordered_point_bins=point_bin_order(point_bins),
    )
    print(f"Wrote model-utility analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

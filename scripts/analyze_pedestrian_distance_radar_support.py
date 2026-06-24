#!/usr/bin/env python3
"""Analyze how object distance affects radar support in fusion datasets.

The fusion training datasets already store ground-truth object distance from
the ego sensor plus the number of radar points associated with each object.
This diagnostic bins those rows by distance so we can explain whether poor
pedestrian segmentation/localization is caused by missing radar evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - CSV/Markdown outputs still work.
    plt = None  # type: ignore


DEFAULT_BINS = "0,10,20,30,40,60,80,120,inf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        help="Fusion dataset directories containing object_boxes.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="abiodun/analysis_outputs/pedestrian_distance_radar_support",
        help="Directory for CSV, Markdown, and plots.",
    )
    parser.add_argument(
        "--distance-bins-m",
        default=DEFAULT_BINS,
        help=f"Comma-separated distance bin edges in meters. Default: {DEFAULT_BINS}",
    )
    parser.add_argument(
        "--min-support-points",
        type=int,
        default=1,
        help="An object is radar-supported when radar_support_points >= this value.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=0,
        help="Optional smoke-test row cap per dataset; 0 means all rows.",
    )
    return parser.parse_args()


def parse_bins(raw: str) -> List[float]:
    bins: List[float] = []
    for part in raw.split(","):
        text = part.strip().lower()
        if not text:
            continue
        bins.append(math.inf if text in {"inf", "infinity"} else float(text))
    if len(bins) < 2:
        raise ValueError("At least two distance bin edges are required")
    if bins != sorted(bins):
        raise ValueError("Distance bins must be sorted ascending")
    return bins


def bin_label(value: float, bins: Sequence[float]) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            hi_text = "inf" if math.isinf(hi) else f"{hi:g}"
            return f"{lo:g}-{hi_text}"
    return "unknown"


def density_from_name(name: str) -> str:
    lowered = name.lower()
    for density in ("low", "medium", "crowded"):
        if f"_{density}_" in lowered or lowered.endswith(f"_{density}") or f"{density}_" in lowered:
            return density
    if "merged" in lowered:
        return "merged"
    return "unknown"


def short_dataset_label(dataset_dir: Path) -> str:
    name = dataset_dir.name
    replacements = [
        ("moving_ego_", ""),
        ("tl16_spawn80_", ""),
        ("fixedroute_speed60_", ""),
        ("2loops_cap2200_", ""),
        ("8loops_cap6000_", ""),
        ("stride2", ""),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    name = name.strip("_")
    return name or dataset_dir.name


def read_csv(path: Path, max_rows: int = 0) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
        return rows


def to_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = str(row.get(key, "")).strip()
        if value == "":
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def object_distance_m(row: Dict[str, str]) -> float:
    distance = to_float(row, "gt_distance_m")
    if math.isfinite(distance):
        return distance
    sx = to_float(row, "object_sensor_x", 0.0)
    sy = to_float(row, "object_sensor_y", 0.0)
    sz = to_float(row, "object_sensor_z", 0.0)
    return float(math.sqrt(sx * sx + sy * sy + sz * sz))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_median(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def summarize_group(rows: Sequence[Dict[str, object]], min_support_points: int) -> Dict[str, object]:
    radar_points = [float(row["radar_support_points"]) for row in rows]
    distances = [float(row["distance_m"]) for row in rows]
    bbox_areas = [float(row["bbox_area_px"]) for row in rows if math.isfinite(float(row["bbox_area_px"]))]
    supported = [points >= min_support_points for points in radar_points]
    supported_count = int(sum(1 for value in supported if value))
    zero_count = int(sum(1 for points in radar_points if points <= 0.0))
    return {
        "object_rows": len(rows),
        "supported_rows": supported_count,
        "support_rate": supported_count / len(rows) if rows else float("nan"),
        "zero_support_rate": zero_count / len(rows) if rows else float("nan"),
        "mean_distance_m": safe_mean(distances),
        "mean_radar_points": safe_mean(radar_points),
        "median_radar_points": safe_median(radar_points),
        "p75_radar_points": float(np.percentile(radar_points, 75)) if radar_points else float("nan"),
        "mean_bbox_area_px": safe_mean(bbox_areas),
        "median_bbox_area_px": safe_median(bbox_areas),
    }


def collect_rows(dataset_dirs: Sequence[Path], bins: Sequence[float], max_rows: int) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    for dataset_dir in dataset_dirs:
        object_boxes_path = dataset_dir / "object_boxes.csv"
        if not object_boxes_path.exists():
            raise FileNotFoundError(f"Missing object boxes CSV: {object_boxes_path}")
        dataset_label = short_dataset_label(dataset_dir)
        density = density_from_name(dataset_dir.name)
        for row in read_csv(object_boxes_path, max_rows=max_rows):
            label = str(row.get("label", "")).strip()
            if label not in {"person", "vehicle"}:
                continue
            distance = object_distance_m(row)
            if not math.isfinite(distance):
                continue
            radar_points = to_float(row, "radar_support_points", 0.0)
            collected.append(
                {
                    "dataset": dataset_label,
                    "dataset_dir": str(dataset_dir),
                    "density": density,
                    "label": label,
                    "distance_m": distance,
                    "distance_bin_m": bin_label(distance, bins),
                    "bbox_area_px": to_float(row, "gt_bbox_area_px"),
                    "radar_support_points": radar_points,
                    "sample_id": str(row.get("sample_id", "")),
                    "frame_id": str(row.get("frame_id", "")),
                    "gt_actor_id": str(row.get("gt_actor_id", "")),
                }
            )
    return collected


def aggregate(
    rows: Sequence[Dict[str, object]],
    *,
    group_fields: Sequence[str],
    min_support_points: int,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups[key].append(row)

    output_rows: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        summary = summarize_group(group, min_support_points)
        output_row = {field: value for field, value in zip(group_fields, key)}
        output_row.update(summary)
        output_rows.append(output_row)
    return output_rows


def plot_support_rate_by_distance(
    rows: Sequence[Dict[str, object]],
    bins: Sequence[float],
    output_path: Path,
    *,
    label_filter: str = "person",
    min_support_points: int = 1,
) -> None:
    if plt is None:
        return
    bin_labels = [bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, bins) for lo, hi in zip(bins[:-1], bins[1:])]
    datasets = sorted({str(row["dataset"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    for dataset in datasets:
        ys: List[float] = []
        ns: List[int] = []
        for bucket in bin_labels:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset and row["label"] == label_filter and row["distance_bin_m"] == bucket
            ]
            summary = summarize_group(group, min_support_points)
            ys.append(float(summary["support_rate"]) if group else float("nan"))
            ns.append(int(summary["object_rows"]) if group else 0)
        ax.plot(bin_labels, ys, marker="o", linewidth=2, label=dataset)
    ax.set_title(f"{label_filter.title()} radar support rate by distance")
    ax.set_ylabel("Objects with >=1 radar point")
    ax.set_xlabel("Distance from ego (m)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_vehicle_vs_person(
    rows: Sequence[Dict[str, object]],
    bins: Sequence[float],
    output_path: Path,
    *,
    dataset_name: str,
    min_support_points: int = 1,
) -> None:
    if plt is None:
        return
    bin_labels = [bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, bins) for lo, hi in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for label in ("vehicle", "person"):
        ys: List[float] = []
        for bucket in bin_labels:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset_name and row["label"] == label and row["distance_bin_m"] == bucket
            ]
            summary = summarize_group(group, min_support_points)
            ys.append(float(summary["support_rate"]) if group else float("nan"))
        ax.plot(bin_labels, ys, marker="o", linewidth=2, label=label)
    ax.set_title(f"Vehicle vs pedestrian radar support: {dataset_name}")
    ax.set_ylabel("Objects with >=1 radar point")
    ax.set_xlabel("Distance from ego (m)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_mean_points_by_distance(
    rows: Sequence[Dict[str, object]],
    bins: Sequence[float],
    output_path: Path,
    *,
    dataset_name: str,
) -> None:
    if plt is None:
        return
    bin_labels = [bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, bins) for lo, hi in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for label in ("vehicle", "person"):
        ys: List[float] = []
        for bucket in bin_labels:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset_name and row["label"] == label and row["distance_bin_m"] == bucket
            ]
            summary = summarize_group(group, 1)
            ys.append(float(summary["mean_radar_points"]) if group else float("nan"))
        ax.plot(bin_labels, ys, marker="o", linewidth=2, label=label)
    ax.set_title(f"Mean radar points per object: {dataset_name}")
    ax.set_ylabel("Mean associated radar points")
    ax.set_xlabel("Distance from ego (m)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(
    path: Path,
    *,
    dataset_dirs: Sequence[Path],
    overall_rows: Sequence[Dict[str, object]],
    distance_rows: Sequence[Dict[str, object]],
    min_support_points: int,
) -> None:
    lines = [
        "# Pedestrian Distance vs Radar Support",
        "",
        "This diagnostic reads `object_boxes.csv` and asks whether objects have enough radar evidence as their distance from the ego sensor changes.",
        "",
        f"- Radar-supported object definition: `radar_support_points >= {min_support_points}`",
        "- Distance source: `gt_distance_m` when present, otherwise sensor-relative `(x,y,z)` distance.",
        "- Key interpretation: if vehicles are supported at a distance but pedestrians are not, the issue is not simply range; it is pedestrian radar sparsity/association.",
        "",
        "## Datasets",
    ]
    for dataset_dir in dataset_dirs:
        lines.append(f"- `{dataset_dir}`")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Dataset | Class | Rows | Support rate | Zero-support rate | Mean radar pts | Median radar pts | Mean distance (m) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall_rows:
        lines.append(
            "| {dataset} | {label} | {rows} | {rate:.3f} | {zero:.3f} | {mean_pts:.2f} | {median_pts:.2f} | {mean_dist:.1f} |".format(
                dataset=row["dataset"],
                label=row["label"],
                rows=int(row["object_rows"]),
                rate=float(row["support_rate"]),
                zero=float(row["zero_support_rate"]),
                mean_pts=float(row["mean_radar_points"]),
                median_pts=float(row["median_radar_points"]),
                mean_dist=float(row["mean_distance_m"]),
            )
        )
    lines.extend(
        [
            "",
            "## Pedestrian Distance Buckets",
            "",
            "| Dataset | Distance (m) | Person rows | Support rate | Zero-support rate | Mean radar pts | Mean bbox area (px) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in distance_rows:
        if row.get("label") != "person":
            continue
        lines.append(
            "| {dataset} | {bucket} | {rows} | {rate:.3f} | {zero:.3f} | {mean_pts:.2f} | {area:.0f} |".format(
                dataset=row["dataset"],
                bucket=row["distance_bin_m"],
                rows=int(row["object_rows"]),
                rate=float(row["support_rate"]),
                zero=float(row["zero_support_rate"]),
                mean_pts=float(row["mean_radar_points"]),
                area=float(row["mean_bbox_area_px"]),
            )
        )
    lines.extend(
        [
            "",
            "## Presentation Takeaway",
            "",
            "- Use the vehicle line as the radar sanity check.",
            "- If pedestrian support stays low while vehicle support is high, increasing model training alone is unlikely to fix pedestrian IoU.",
            "- The next radar-side knobs to test are point density, FOV, temporal accumulation, and better rasterization/association, with latency and payload tracked explicitly.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dirs = [Path(raw).expanduser().resolve() for raw in args.dataset_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    bins = parse_bins(args.distance_bins_m)

    rows = collect_rows(dataset_dirs, bins, max_rows=int(args.max_rows_per_dataset))
    if not rows:
        raise SystemExit("No vehicle/person rows found in the supplied datasets.")

    per_object_fields = [
        "dataset",
        "density",
        "label",
        "distance_m",
        "distance_bin_m",
        "bbox_area_px",
        "radar_support_points",
        "sample_id",
        "frame_id",
        "gt_actor_id",
    ]
    write_csv(output_dir / "pedestrian_distance_radar_support_objects.csv", rows, per_object_fields)

    distance_rows = aggregate(
        rows,
        group_fields=["dataset", "density", "label", "distance_bin_m"],
        min_support_points=int(args.min_support_points),
    )
    distance_fields = [
        "dataset",
        "density",
        "label",
        "distance_bin_m",
        "object_rows",
        "supported_rows",
        "support_rate",
        "zero_support_rate",
        "mean_distance_m",
        "mean_radar_points",
        "median_radar_points",
        "p75_radar_points",
        "mean_bbox_area_px",
        "median_bbox_area_px",
    ]
    write_csv(output_dir / "pedestrian_distance_radar_support_by_distance.csv", distance_rows, distance_fields)

    overall_rows = aggregate(
        rows,
        group_fields=["dataset", "density", "label"],
        min_support_points=int(args.min_support_points),
    )
    overall_fields = [field for field in distance_fields if field != "distance_bin_m"]
    write_csv(output_dir / "pedestrian_distance_radar_support_overall.csv", overall_rows, overall_fields)

    if plt is not None:
        bin_order = {
            bin_label((lo + hi) / 2 if math.isfinite(hi) else lo + 1, bins): index
            for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]))
        }
        distance_rows.sort(
            key=lambda row: (
                str(row.get("dataset", "")),
                str(row.get("label", "")),
                int(bin_order.get(str(row.get("distance_bin_m", "")), 10_000)),
            )
        )
        plot_support_rate_by_distance(
            rows,
            bins,
            output_dir / "person_radar_support_rate_by_distance.png",
            label_filter="person",
            min_support_points=int(args.min_support_points),
        )
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset)
            plot_vehicle_vs_person(
                rows,
                bins,
                output_dir / f"vehicle_vs_person_support_by_distance_{safe_name}.png",
                dataset_name=dataset,
                min_support_points=int(args.min_support_points),
            )
            plot_mean_points_by_distance(
                rows,
                bins,
                output_dir / f"mean_radar_points_by_distance_{safe_name}.png",
                dataset_name=dataset,
            )

    write_markdown(
        output_dir / "pedestrian_distance_radar_support_summary.md",
        dataset_dirs=dataset_dirs,
        overall_rows=overall_rows,
        distance_rows=distance_rows,
        min_support_points=int(args.min_support_points),
    )
    print(f"Wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

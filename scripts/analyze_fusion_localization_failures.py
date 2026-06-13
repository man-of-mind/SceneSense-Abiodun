#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DISTANCE_BINS = [0.0, 20.0, 40.0, 60.0, 80.0, 120.0, math.inf]
BBOX_AREA_BINS = [0.0, 128.0, 512.0, 2048.0, 8192.0, 32768.0, math.inf]


def density_from_sample(sample_id: str) -> str:
    if "_low_" in sample_id:
        return "low"
    if "_medium_" in sample_id:
        return "medium"
    if "_crowded_" in sample_id:
        return "crowded"
    return "unknown"


def bin_label(value: float, bins: Sequence[float]) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            hi_text = "inf" if math.isinf(hi) else f"{hi:g}"
            return f"{lo:g}-{hi_text}"
    return "unknown"


def prf(counts: Dict[str, int]) -> Dict[str, float]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def object_lookup(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by_sample: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
    return by_sample


def as_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def nearest_gt_box(
    boxes_by_sample: Dict[str, List[Dict[str, str]]],
    metric_row: Dict[str, str],
    *,
    max_world_error_m: float = 0.35,
) -> Dict[str, str]:
    sample_id = metric_row.get("sample_id", "")
    label = metric_row.get("gt_class_name") or metric_row.get("class_name") or ""
    gx = as_float(metric_row, "gt_world_x")
    gy = as_float(metric_row, "gt_world_y")
    best: Tuple[float, Dict[str, str]] | None = None
    if math.isnan(gx) or math.isnan(gy):
        return {}
    for box in boxes_by_sample.get(sample_id, []):
        if box.get("label") != label:
            continue
        bx = as_float(box, "object_world_x")
        by = as_float(box, "object_world_y")
        if math.isnan(bx) or math.isnan(by):
            continue
        dist = math.hypot(gx - bx, gy - by)
        if best is None or dist < best[0]:
            best = (dist, box)
    if best is None or best[0] > max_world_error_m:
        return {}
    return best[1]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(
    values: Dict[str, Dict[str, float]],
    *,
    groups: Sequence[str],
    series: Sequence[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for i, name in enumerate(series):
        ys = [float(values.get(group, {}).get(name, 0.0)) for group in groups]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, ys, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels=groups)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else exp_dir / "dataset"
    metrics_csv = Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else exp_dir / "metrics" / "test_learned_object_metrics.csv"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else exp_dir / "analysis" / "localization_failures"

    metric_rows = read_csv(metrics_csv)
    boxes_by_sample = object_lookup(read_csv(dataset_dir / "object_boxes.csv"))

    overall_counts: Dict[str, int] = defaultdict(int)
    by_density: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_class: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_density_class: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    recall_by_distance: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    recall_by_area: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    recall_by_radar: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    enriched_gt_rows: List[Dict[str, object]] = []

    for row in metric_rows:
        status = row.get("match_status", "unknown")
        sample_id = row.get("sample_id", "")
        density = density_from_sample(sample_id)
        class_name = row.get("class_name") or row.get("gt_class_name") or row.get("pred_class_name") or "unknown"
        if status in {"tp", "fp", "fn"}:
            overall_counts[status] += 1
            by_density[density][status] += 1
            by_class[class_name][status] += 1
            by_density_class[density][class_name][status] += 1
        if status not in {"tp", "fn"}:
            continue

        box = nearest_gt_box(boxes_by_sample, row)
        if not box:
            continue
        gt_distance = as_float(box, "gt_distance_m")
        gt_area = as_float(box, "gt_bbox_area_px")
        radar_points = as_float(box, "radar_support_points", 0.0)
        distance_bucket = bin_label(gt_distance, DISTANCE_BINS)
        area_bucket = bin_label(gt_area, BBOX_AREA_BINS)
        radar_bucket = "radar>0" if radar_points > 0 else "radar=0"
        recall_by_distance[class_name][distance_bucket][status] += 1
        recall_by_area[class_name][area_bucket][status] += 1
        recall_by_radar[class_name][radar_bucket][status] += 1
        enriched_gt_rows.append(
            {
                "sample_id": sample_id,
                "density": density,
                "class_name": class_name,
                "match_status": status,
                "gt_distance_m": gt_distance,
                "gt_depth_m": as_float(box, "gt_depth_m"),
                "gt_bbox_area_px": gt_area,
                "radar_support_points": radar_points,
                "global_xy_error_m": as_float(row, "global_xy_error_m"),
                "score": as_float(row, "score"),
            }
        )

    summary = {
        "experiment_dir": str(exp_dir),
        "dataset_dir": str(dataset_dir),
        "metrics_csv": str(metrics_csv),
        "overall": prf(overall_counts),
        "by_density": {key: prf(value) for key, value in sorted(by_density.items())},
        "by_class": {key: prf(value) for key, value in sorted(by_class.items())},
        "by_density_class": {
            density: {label: prf(counts) for label, counts in sorted(labels.items())}
            for density, labels in sorted(by_density_class.items())
        },
        "recall_by_distance": {
            label: {bucket: prf(counts) for bucket, counts in sorted(buckets.items())}
            for label, buckets in sorted(recall_by_distance.items())
        },
        "recall_by_bbox_area": {
            label: {bucket: prf(counts) for bucket, counts in sorted(buckets.items())}
            for label, buckets in sorted(recall_by_area.items())
        },
        "recall_by_radar_support": {
            label: {bucket: prf(counts) for bucket, counts in sorted(buckets.items())}
            for label, buckets in sorted(recall_by_radar.items())
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fusion_localization_failure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "fusion_localization_gt_enriched.csv", enriched_gt_rows)

    density_values = {
        density: {label: stats["f1"] for label, stats in labels.items()}
        for density, labels in summary["by_density_class"].items()
    }
    plot_grouped_bars(
        density_values,
        groups=["low", "medium", "crowded"],
        series=["vehicle", "person"],
        title="Fusion localization F1 by traffic density",
        ylabel="F1 at 3 m match",
        output_path=output_dir / "localization_f1_by_density.png",
    )

    distance_values = {
        bucket: {label: stats.get(bucket, {}).get("recall", 0.0) for label, stats in summary["recall_by_distance"].items()}
        for bucket in [bin_label((lo + (hi if not math.isinf(hi) else lo + 40.0)) / 2.0, DISTANCE_BINS) for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])]
    }
    plot_grouped_bars(
        distance_values,
        groups=list(distance_values.keys()),
        series=["vehicle", "person"],
        title="Fusion localization recall by GT distance",
        ylabel="Recall at 3 m match",
        output_path=output_dir / "localization_recall_by_distance.png",
    )

    area_values = {
        bucket: {label: stats.get(bucket, {}).get("recall", 0.0) for label, stats in summary["recall_by_bbox_area"].items()}
        for bucket in [bin_label((lo + (hi if not math.isinf(hi) else lo * 2.0)) / 2.0, BBOX_AREA_BINS) for lo, hi in zip(BBOX_AREA_BINS[:-1], BBOX_AREA_BINS[1:])]
    }
    plot_grouped_bars(
        area_values,
        groups=list(area_values.keys()),
        series=["vehicle", "person"],
        title="Fusion localization recall by GT bbox area",
        ylabel="Recall at 3 m match",
        output_path=output_dir / "localization_recall_by_bbox_area.png",
    )

    radar_values = {
        bucket: {label: stats.get(bucket, {}).get("recall", 0.0) for label, stats in summary["recall_by_radar_support"].items()}
        for bucket in ["radar=0", "radar>0"]
    }
    plot_grouped_bars(
        radar_values,
        groups=["radar=0", "radar>0"],
        series=["vehicle", "person"],
        title="Fusion localization recall by radar support",
        ylabel="Recall at 3 m match",
        output_path=output_dir / "localization_recall_by_radar_support.png",
    )

    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "overall": summary["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

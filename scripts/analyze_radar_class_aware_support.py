#!/usr/bin/env python3
"""Compare bbox radar support with class-aware pedestrian radar support.

This is an offline diagnostic: it reads an existing fusion training dataset and
recomputes radar support from saved radar_points/*.npz plus object_boxes.csv.
Vehicles use the oriented object box. Pedestrians can use a radius/cylinder
gate, which mirrors the LiDAR diagnostic lesson without using semantic IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - CSV/Markdown outputs still work.
    plt = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Fusion dataset containing manifest.csv/object_boxes.csv.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output analysis directory. Defaults to analysis_outputs/radar_class_aware_support/<dataset-name>.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples for a quick smoke run; 0 means all.")
    parser.add_argument("--vehicle-box-margin-m", type=float, default=1.0)
    parser.add_argument("--person-mode", choices=("bbox", "radius"), default="radius")
    parser.add_argument("--person-radius-m", type=float, default=1.5)
    parser.add_argument("--person-z-down-m", type=float, default=0.5)
    parser.add_argument("--person-z-up-m", type=float, default=2.0)
    parser.add_argument(
        "--min-support-points",
        type=int,
        default=1,
        help="Count an object as radar-supported when support points are at least this value.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(str(row.get(key, "")).strip())
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(str(row.get(key, "")).strip()))
    except (TypeError, ValueError):
        return int(default)


def path_from_row(dataset_dir: Path, row: Dict[str, str], field: str) -> Path:
    raw = str(row.get(field, "")).strip()
    path = Path(raw)
    return path if path.is_absolute() else dataset_dir / path


def group_by_sample(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    return grouped


def local_points_from_row(points_world: np.ndarray, row: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(
        [
            to_float(row, "object_world_x"),
            to_float(row, "object_world_y"),
            to_float(row, "object_world_z"),
        ],
        dtype=np.float64,
    )
    delta = points_world.astype(np.float64, copy=False) - center[None, :]
    yaw = math.radians(to_float(row, "object_yaw_deg"))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    local_x = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    local_y = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    local = np.stack([local_x, local_y, delta[:, 2]], axis=1)
    return local, center


def box_support_count(points_world: np.ndarray, row: Dict[str, str], margin_m: float) -> int:
    if points_world.size == 0:
        return 0
    local, _ = local_points_from_row(points_world, row)
    extent_x = max(0.01, to_float(row, "gt_size_x_m") / 2.0)
    extent_y = max(0.01, to_float(row, "gt_size_y_m") / 2.0)
    extent_z = max(0.01, to_float(row, "gt_size_z_m") / 2.0)
    margin = max(0.0, float(margin_m))
    inside = (
        (np.abs(local[:, 0]) <= extent_x + margin)
        & (np.abs(local[:, 1]) <= extent_y + margin)
        & (np.abs(local[:, 2]) <= extent_z + margin)
    )
    return int(np.count_nonzero(inside))


def person_radius_support_count(
    points_world: np.ndarray,
    row: Dict[str, str],
    *,
    radius_m: float,
    z_down_m: float,
    z_up_m: float,
) -> int:
    if points_world.size == 0:
        return 0
    local, _ = local_points_from_row(points_world, row)
    extent_z = max(0.01, to_float(row, "gt_size_z_m") / 2.0)
    radius = max(0.05, float(radius_m))
    z_down = max(0.0, float(z_down_m))
    z_up = max(0.0, float(z_up_m))
    inside = (
        (local[:, 0] * local[:, 0] + local[:, 1] * local[:, 1] <= radius * radius)
        & (local[:, 2] >= -extent_z - z_down)
        & (local[:, 2] <= extent_z + z_up)
    )
    return int(np.count_nonzero(inside))


def class_aware_support_count(points_world: np.ndarray, row: Dict[str, str], args: argparse.Namespace) -> Tuple[int, str]:
    label = str(row.get("label", ""))
    if label == "person" and str(args.person_mode) == "radius":
        return (
            person_radius_support_count(
                points_world,
                row,
                radius_m=float(args.person_radius_m),
                z_down_m=float(args.person_z_down_m),
                z_up_m=float(args.person_z_up_m),
            ),
            "person_radius",
        )
    return box_support_count(points_world, row, float(args.vehicle_box_margin_m)), "bbox"


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def summarize(rows: Sequence[Dict[str, object]], min_support: int) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("label", "")), str(row.get("class_aware_mode", "")))].append(row)

    summary_rows: List[Dict[str, object]] = []
    for (label, mode), group in sorted(groups.items()):
        total = len(group)
        current_counts = np.asarray([float(row["current_support_points"]) for row in group], dtype=np.float64)
        class_counts = np.asarray([float(row["class_aware_support_points"]) for row in group], dtype=np.float64)
        current_supported = int(np.count_nonzero(current_counts >= int(min_support)))
        class_supported = int(np.count_nonzero(class_counts >= int(min_support)))
        gained = int(np.count_nonzero((current_counts < int(min_support)) & (class_counts >= int(min_support))))
        lost = int(np.count_nonzero((current_counts >= int(min_support)) & (class_counts < int(min_support))))
        summary_rows.append(
            {
                "label": label,
                "class_aware_mode": mode,
                "object_rows": total,
                "current_supported_rows": current_supported,
                "current_supported_rate": ratio(current_supported, total),
                "class_aware_supported_rows": class_supported,
                "class_aware_supported_rate": ratio(class_supported, total),
                "support_gain_rows": gained,
                "support_loss_rows": lost,
                "current_support_mean": float(current_counts.mean()) if total else float("nan"),
                "class_aware_support_mean": float(class_counts.mean()) if total else float("nan"),
                "current_support_median": float(np.median(current_counts)) if total else float("nan"),
                "class_aware_support_median": float(np.median(class_counts)) if total else float("nan"),
            }
        )
    return summary_rows


def write_markdown(
    path: Path,
    *,
    dataset_dir: Path,
    args: argparse.Namespace,
    sample_count: int,
    row_count: int,
    summary_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Radar Class-Aware Support Diagnostic",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Samples inspected: `{sample_count}`",
        f"- Object rows inspected: `{row_count}`",
        f"- Min support points: `{int(args.min_support_points)}`",
        f"- Vehicle box margin: `{float(args.vehicle_box_margin_m):.2f} m`",
        (
            "- Person association: "
            f"`{args.person_mode}`, radius `{float(args.person_radius_m):.2f} m`, "
            f"z-down `{float(args.person_z_down_m):.2f} m`, "
            f"z-up `{float(args.person_z_up_m):.2f} m`"
        ),
        "",
        "| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {mode} | {rows} | {cur_rate:.3f} | {new_rate:.3f} | {gain} | {loss} | {cur_mean:.2f} | {new_mean:.2f} |".format(
                label=row["label"],
                mode=row["class_aware_mode"],
                rows=int(row["object_rows"]),
                cur_rate=float(row["current_supported_rate"]),
                new_rate=float(row["class_aware_supported_rate"]),
                gain=int(row["support_gain_rows"]),
                loss=int(row["support_loss_rows"]),
                cur_mean=float(row["current_support_mean"]),
                new_mean=float(row["class_aware_support_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This diagnostic does not use semantic IDs or hidden inference-time ground truth. "
                "It recomputes support from saved radar points and the supervised-training actor labels. "
                "If the person support rate rises, it means the original actor-box association was too strict "
                "for sparse pedestrian radar returns."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(output_dir: Path, summary_rows: Sequence[Dict[str, object]]) -> List[str]:
    if plt is None or not summary_rows:
        return []
    labels = [str(row["label"]).title() for row in summary_rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    written: List[str] = []

    def finish(fig: object, path: Path) -> None:
        fig.tight_layout()  # type: ignore[attr-defined]
        fig.savefig(path, dpi=180)  # type: ignore[attr-defined]
        fig.savefig(path.with_suffix(".pdf"))  # type: ignore[attr-defined]
        plt.close(fig)
        written.append(str(path))

    current_rates = [float(row["current_supported_rate"]) for row in summary_rows]
    class_rates = [float(row["class_aware_supported_rate"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2.0, current_rates, width, label="Original actor-box support", color="#5b789e")
    ax.bar(x + width / 2.0, class_rates, width, label="Class-aware support", color="#d7823d")
    ax.set_ylabel("Radar-supported object rows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(0.15, max(current_rates + class_rates) * 1.25))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for xpos, value in zip(x - width / 2.0, current_rates):
        ax.text(xpos, value + 0.006, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2.0, class_rates):
        ax.text(xpos, value + 0.006, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    finish(fig, output_dir / "radar_class_aware_support_rate.png")

    current_means = [float(row["current_support_mean"]) for row in summary_rows]
    class_means = [float(row["class_aware_support_mean"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2.0, current_means, width, label="Original actor-box support", color="#5b789e")
    ax.bar(x + width / 2.0, class_means, width, label="Class-aware support", color="#d7823d")
    ax.set_ylabel("Mean radar points per object row")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(0.25, max(current_means + class_means) * 1.25))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for xpos, value in zip(x - width / 2.0, current_means):
        ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2.0, class_means):
        ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    finish(fig, output_dir / "radar_class_aware_mean_points.png")
    return written


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_rows = read_csv(dataset_dir / "manifest.csv")
    object_rows = read_csv(dataset_dir / "object_boxes.csv")
    if int(args.max_samples) > 0:
        manifest_rows = manifest_rows[: int(args.max_samples)]
    sample_ids = {str(row.get("sample_id", "")) for row in manifest_rows}
    objects_by_sample = group_by_sample(row for row in object_rows if str(row.get("sample_id", "")) in sample_ids)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir).strip()
        else Path("abiodun/analysis_outputs/radar_class_aware_support") / dataset_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    row_results: List[Dict[str, object]] = []
    missing_points = 0
    processed_samples = 0
    for manifest in manifest_rows:
        sample_id = str(manifest.get("sample_id", ""))
        rows = objects_by_sample.get(sample_id, [])
        if not rows:
            continue
        radar_path = path_from_row(dataset_dir, manifest, "radar_points_path")
        if not radar_path.exists():
            missing_points += 1
            continue
        with np.load(radar_path) as points:
            world_xyz = np.asarray(points.get("world_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        processed_samples += 1
        for row in rows:
            if str(row.get("gt_source", "")) != "actor":
                continue
            if row.get("object_world_x", "") == "":
                continue
            current_support = to_int(row, "radar_support_points")
            class_support, mode = class_aware_support_count(world_xyz, row, args)
            row_results.append(
                {
                    "sample_id": sample_id,
                    "frame_id": row.get("frame_id", ""),
                    "label": row.get("label", ""),
                    "gt_actor_id": row.get("gt_actor_id", ""),
                    "gt_distance_m": row.get("gt_distance_m", ""),
                    "gt_bbox_area_px": row.get("gt_bbox_area_px", ""),
                    "current_support_points": current_support,
                    "class_aware_support_points": class_support,
                    "support_delta": int(class_support) - int(current_support),
                    "class_aware_mode": mode,
                }
            )

    summary_rows = summarize(row_results, int(args.min_support_points))
    row_fields = (
        "sample_id",
        "frame_id",
        "label",
        "gt_actor_id",
        "gt_distance_m",
        "gt_bbox_area_px",
        "current_support_points",
        "class_aware_support_points",
        "support_delta",
        "class_aware_mode",
    )
    summary_fields = (
        "label",
        "class_aware_mode",
        "object_rows",
        "current_supported_rows",
        "current_supported_rate",
        "class_aware_supported_rows",
        "class_aware_supported_rate",
        "support_gain_rows",
        "support_loss_rows",
        "current_support_mean",
        "class_aware_support_mean",
        "current_support_median",
        "class_aware_support_median",
    )
    write_csv(output_dir / "radar_class_aware_support_rows.csv", row_results, row_fields)
    write_csv(output_dir / "radar_class_aware_support_summary.csv", summary_rows, summary_fields)
    write_markdown(
        output_dir / "radar_class_aware_support_summary.md",
        dataset_dir=dataset_dir,
        args=args,
        sample_count=processed_samples,
        row_count=len(row_results),
        summary_rows=summary_rows,
    )
    plot_paths = plot_summary(output_dir, summary_rows)
    payload = {
        "schema": "scenesense_radar_class_aware_support.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "processed_samples": processed_samples,
        "missing_radar_point_files": missing_points,
        "object_rows": len(row_results),
        "settings": {
            "vehicle_box_margin_m": float(args.vehicle_box_margin_m),
            "person_mode": str(args.person_mode),
            "person_radius_m": float(args.person_radius_m),
            "person_z_down_m": float(args.person_z_down_m),
            "person_z_up_m": float(args.person_z_up_m),
            "min_support_points": int(args.min_support_points),
        },
        "summary": summary_rows,
        "plots": plot_paths,
    }
    (output_dir / "radar_class_aware_support_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    payload = analyze(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

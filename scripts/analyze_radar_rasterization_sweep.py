#!/usr/bin/env python3
"""Analyze radar rasterization radius tradeoffs on saved fusion datasets.

The training dataset stores both the model-ready radar tensor and the raw
projected radar points. This script re-paints those saved points with different
pixel radii and measures:

- object-box radar coverage by class,
- object support rate by class,
- occupied radar pixels per frame,
- spillover outside visible GT object boxes.

It is analysis-only; it does not rewrite the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Fusion dataset directory containing manifest.csv, object_boxes.csv, and radar_points/.",
    )
    parser.add_argument(
        "--output-dir",
        default="abiodun/analysis_outputs/radar_rasterization_sweep",
        help="Directory where CSV summaries and plots are written.",
    )
    parser.add_argument("--run-name", default="", help="Optional subdirectory name under --output-dir.")
    parser.add_argument("--radii-px", default="0,1,2,3,4,5,7")
    parser.add_argument(
        "--distance-bins-m",
        default="0,10,20,30,40,60,80,120,200",
        help="Comma-separated distance bin edges for object coverage summaries.",
    )
    parser.add_argument("--min-box-area-px", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for faster pilot runs.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Analyze every Nth sample row from the manifest.",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else float(default)
    except (TypeError, ValueError):
        return float(default)


def load_radar_tensor_shape(dataset_dir: Path, manifest_rows: Sequence[Mapping[str, str]]) -> Tuple[int, int]:
    for row in manifest_rows:
        path = dataset_dir / str(row.get("radar_tensor_path", ""))
        if not path.exists():
            continue
        payload = np.load(path)
        if isinstance(payload, np.lib.npyio.NpzFile):
            radar = payload["radar"]
        else:
            radar = payload
        if radar.ndim != 3:
            continue
        return int(radar.shape[2]), int(radar.shape[1])
    raise FileNotFoundError("Could not infer radar tensor shape from manifest radar_tensor_path entries.")


def sample_groups(object_rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Mapping[str, str]]]:
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in object_rows:
        groups[str(row.get("sample_id", ""))].append(row)
    return groups


def distance_bin(distance_m: float, edges: Sequence[float]) -> str:
    if not edges:
        return "all"
    for left, right in zip(edges[:-1], edges[1:]):
        if float(left) <= distance_m < float(right):
            return f"{left:g}-{right:g}m"
    return f">={edges[-1]:g}m"


def paint_occupancy(
    *,
    width: int,
    height: int,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    if u.size == 0:
        return mask
    valid_mask = (
        valid.astype(bool)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    if not np.any(valid_mask):
        return mask
    radius = max(0, int(radius_px))
    pxs = np.rint(u[valid_mask]).astype(np.int32)
    pys = np.rint(v[valid_mask]).astype(np.int32)
    for px, py in zip(pxs, pys):
        if radius <= 0:
            if 0 <= px < width and 0 <= py < height:
                mask[py, px] = True
            continue
        y0, y1 = max(0, int(py) - radius), min(int(height), int(py) + radius + 1)
        x0, x1 = max(0, int(px) - radius), min(int(width), int(px) + radius + 1)
        if y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = True
    return mask


def scaled_box(
    row: Mapping[str, str],
    *,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int, float]]:
    x = to_float(row, "gt_bbox_x") * float(scale_x)
    y = to_float(row, "gt_bbox_y") * float(scale_y)
    w = to_float(row, "gt_bbox_w") * float(scale_x)
    h = to_float(row, "gt_bbox_h") * float(scale_y)
    if w <= 0.0 or h <= 0.0:
        return None
    x0 = max(0, min(int(width), int(math.floor(x))))
    y0 = max(0, min(int(height), int(math.floor(y))))
    x1 = max(0, min(int(width), int(math.ceil(x + w))))
    y1 = max(0, min(int(height), int(math.ceil(y + h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1, float((x1 - x0) * (y1 - y0))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def aggregate_object_rows(rows: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    out: List[Dict[str, object]] = []
    for key, vals in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        coverages = [float(v["box_coverage_fraction"]) for v in vals]
        covered_pixels = [float(v["box_covered_pixels"]) for v in vals]
        support_flags = [int(v["box_supported"]) for v in vals]
        support5_flags = [int(v["box_supported_5px"]) for v in vals]
        areas = [float(v["box_area_px"]) for v in vals]
        coverage_summary = summarize(coverages)
        pixels_summary = summarize(covered_pixels)
        row: Dict[str, object] = {name: value for name, value in zip(group_keys, key)}
        row.update(
            {
                "objects": len(vals),
                "support_rate_any": float(sum(support_flags) / max(1, len(support_flags))),
                "support_rate_5px": float(sum(support5_flags) / max(1, len(support5_flags))),
                "mean_box_coverage_fraction": coverage_summary["mean"],
                "median_box_coverage_fraction": coverage_summary["median"],
                "p75_box_coverage_fraction": coverage_summary["p75"],
                "p90_box_coverage_fraction": coverage_summary["p90"],
                "mean_box_covered_pixels": pixels_summary["mean"],
                "median_box_covered_pixels": pixels_summary["median"],
                "mean_box_area_px": float(mean(areas)) if areas else float("nan"),
            }
        )
        out.append(row)
    return out


def aggregate_frame_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["radius_px"])].append(row)
    out: List[Dict[str, object]] = []
    for radius, vals in sorted(groups.items()):
        occupied = [float(v["occupied_pixels"]) for v in vals]
        object_pixels = [float(v["occupied_object_pixels"]) for v in vals]
        spill_pixels = [float(v["occupied_spillover_pixels"]) for v in vals]
        spill_frac = [float(v["spillover_fraction"]) for v in vals if math.isfinite(float(v["spillover_fraction"]))]
        occupancy_pct = [float(v["occupancy_fraction"]) for v in vals]
        out.append(
            {
                "radius_px": radius,
                "frames": len(vals),
                "mean_occupied_pixels": float(mean(occupied)) if occupied else float("nan"),
                "mean_occupied_object_pixels": float(mean(object_pixels)) if object_pixels else float("nan"),
                "mean_occupied_spillover_pixels": float(mean(spill_pixels)) if spill_pixels else float("nan"),
                "mean_spillover_fraction": float(mean(spill_frac)) if spill_frac else float("nan"),
                "mean_occupancy_fraction": float(mean(occupancy_pct)) if occupancy_pct else float("nan"),
            }
        )
    return out


def plot_outputs(output_dir: Path, class_summary: Sequence[Mapping[str, object]], frame_summary: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return

    labels = sorted({str(row["label"]) for row in class_summary})
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for label in labels:
        rows = sorted([row for row in class_summary if str(row["label"]) == label], key=lambda r: int(r["radius_px"]))
        ax.plot(
            [int(row["radius_px"]) for row in rows],
            [float(row["mean_box_coverage_fraction"]) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_title("Radar Raster Radius: Object-Box Coverage")
    ax.set_xlabel("Radar raster radius (px)")
    ax.set_ylabel("Mean fraction of GT box covered")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "raster_radius_object_coverage.png", dpi=220)
    fig.savefig(output_dir / "raster_radius_object_coverage.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for label in labels:
        rows = sorted([row for row in class_summary if str(row["label"]) == label], key=lambda r: int(r["radius_px"]))
        ax.plot(
            [int(row["radius_px"]) for row in rows],
            [float(row["support_rate_5px"]) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_title("Radar Raster Radius: Object Support")
    ax.set_xlabel("Radar raster radius (px)")
    ax.set_ylabel("Objects with at least 5 occupied pixels")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "raster_radius_object_support.png", dpi=220)
    fig.savefig(output_dir / "raster_radius_object_support.pdf")
    plt.close(fig)

    rows = sorted(frame_summary, key=lambda r: int(r["radius_px"]))
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    x = [int(row["radius_px"]) for row in rows]
    ax1.plot(x, [float(row["mean_occupied_pixels"]) for row in rows], marker="o", linewidth=2, color="#0072B2", label="occupied pixels")
    ax1.set_xlabel("Radar raster radius (px)")
    ax1.set_ylabel("Mean occupied pixels / frame", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row["mean_spillover_fraction"]) for row in rows], marker="s", linewidth=2, color="#D55E00", label="spillover fraction")
    ax2.set_ylabel("Fraction of occupied pixels outside GT boxes", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0.0, 1.02)
    ax1.set_title("Radar Raster Radius: Coverage vs Background Spillover")
    fig.savefig(output_dir / "raster_radius_spillover_tradeoff.png", dpi=220)
    fig.savefig(output_dir / "raster_radius_spillover_tradeoff.pdf")
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    *,
    dataset_dir: Path,
    radii: Sequence[int],
    class_summary: Sequence[Mapping[str, object]],
    frame_summary: Sequence[Mapping[str, object]],
) -> None:
    by_radius_label = {(int(row["radius_px"]), str(row["label"])): row for row in class_summary}
    lines = [
        "# Radar Rasterization Sweep",
        "",
        f"Dataset: `{dataset_dir}`",
        "",
        f"Radii tested: `{', '.join(str(r) for r in radii)}`",
        "",
        "## Class Coverage",
        "",
        "| radius px | vehicle coverage | person coverage | vehicle support >=5px | person support >=5px | spillover fraction | occupied px/frame |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    frame_by_radius = {int(row["radius_px"]): row for row in frame_summary}
    for radius in radii:
        vehicle = by_radius_label.get((int(radius), "vehicle"), {})
        person = by_radius_label.get((int(radius), "person"), {})
        frame = frame_by_radius.get(int(radius), {})
        lines.append(
            "| {r} | {vc:.4f} | {pc:.4f} | {vs:.3f} | {ps:.3f} | {spill:.3f} | {occ:.1f} |".format(
                r=radius,
                vc=float(vehicle.get("mean_box_coverage_fraction", float("nan"))),
                pc=float(person.get("mean_box_coverage_fraction", float("nan"))),
                vs=float(vehicle.get("support_rate_5px", float("nan"))),
                ps=float(person.get("support_rate_5px", float("nan"))),
                spill=float(frame.get("mean_spillover_fraction", float("nan"))),
                occ=float(frame.get("mean_occupied_pixels", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Higher object-box coverage means the radar tensor gives the model more spatial evidence inside GT boxes.",
            "- Higher spillover means more radar pixels are painted outside visible GT boxes, which can become background noise.",
            "- A useful radius should improve person coverage/support more than it inflates spillover or harms vehicle clarity.",
        ]
    )
    (output_dir / "radar_rasterization_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    object_path = dataset_dir / "object_boxes.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not object_path.exists():
        raise FileNotFoundError(f"Missing object boxes: {object_path}")

    run_name = args.run_name or dataset_dir.name
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = parse_int_list(args.radii_px)
    distance_edges = parse_float_list(args.distance_bins_m)
    manifest_rows_all = read_csv(manifest_path)
    stride = max(1, int(args.sample_stride))
    manifest_rows = [row for index, row in enumerate(manifest_rows_all) if index % stride == 0]
    if int(args.max_samples) > 0:
        manifest_rows = manifest_rows[: int(args.max_samples)]

    object_rows = read_csv(object_path)
    objects_by_sample = sample_groups(object_rows)
    radar_width, radar_height = load_radar_tensor_shape(dataset_dir, manifest_rows_all)

    object_result_rows: List[Dict[str, object]] = []
    frame_result_rows: List[Dict[str, object]] = []

    for sample_index, manifest in enumerate(manifest_rows):
        sample_id = str(manifest["sample_id"])
        radar_path = dataset_dir / str(manifest.get("radar_points_path", ""))
        if not radar_path.exists():
            continue
        with np.load(radar_path) as radar_points:
            u = np.asarray(radar_points["u"], dtype=np.float32)
            v = np.asarray(radar_points["v"], dtype=np.float32)
            valid = np.asarray(radar_points["valid_projection"], dtype=np.uint8)

        camera_width = max(1.0, to_float(manifest, "camera_width", radar_width))
        camera_height = max(1.0, to_float(manifest, "camera_height", radar_height))
        scale_x = float(radar_width) / camera_width
        scale_y = float(radar_height) / camera_height

        boxes: List[Tuple[Mapping[str, str], Tuple[int, int, int, int, float]]] = []
        object_union = np.zeros((radar_height, radar_width), dtype=bool)
        for obj in objects_by_sample.get(sample_id, []):
            scaled = scaled_box(obj, scale_x=scale_x, scale_y=scale_y, width=radar_width, height=radar_height)
            if scaled is None:
                continue
            if float(scaled[4]) < float(args.min_box_area_px):
                continue
            boxes.append((obj, scaled))
            x0, y0, x1, y1, _ = scaled
            object_union[y0:y1, x0:x1] = True

        for radius in radii:
            occ = paint_occupancy(width=radar_width, height=radar_height, u=u, v=v, valid=valid, radius_px=int(radius))
            occupied_pixels = int(np.count_nonzero(occ))
            occupied_object_pixels = int(np.count_nonzero(occ & object_union))
            occupied_spillover_pixels = max(0, occupied_pixels - occupied_object_pixels)
            spillover_fraction = (
                float(occupied_spillover_pixels / occupied_pixels) if occupied_pixels > 0 else float("nan")
            )
            frame_result_rows.append(
                {
                    "sample_id": sample_id,
                    "radius_px": int(radius),
                    "occupied_pixels": occupied_pixels,
                    "occupied_object_pixels": occupied_object_pixels,
                    "occupied_spillover_pixels": occupied_spillover_pixels,
                    "spillover_fraction": spillover_fraction,
                    "occupancy_fraction": float(occupied_pixels / max(1, radar_width * radar_height)),
                    "objects": len(boxes),
                    "radar_points": int(u.size),
                }
            )
            for obj, scaled in boxes:
                x0, y0, x1, y1, area = scaled
                covered = int(np.count_nonzero(occ[y0:y1, x0:x1]))
                label = str(obj.get("label", "object"))
                distance_m = to_float(obj, "gt_distance_m", float("nan"))
                object_result_rows.append(
                    {
                        "sample_id": sample_id,
                        "radius_px": int(radius),
                        "label": label,
                        "distance_bin_m": distance_bin(distance_m, distance_edges),
                        "gt_distance_m": distance_m,
                        "box_area_px": float(area),
                        "box_covered_pixels": covered,
                        "box_coverage_fraction": float(covered / max(1.0, area)),
                        "box_supported": int(covered > 0),
                        "box_supported_5px": int(covered >= 5),
                        "radar_support_points_original": to_float(obj, "radar_support_points", 0.0),
                    }
                )
        if sample_index == 0 or (sample_index + 1) % 250 == 0 or sample_index + 1 == len(manifest_rows):
            print(f"Analyzed {sample_index + 1}/{len(manifest_rows)} samples")

    class_summary = aggregate_object_rows(object_result_rows, ["radius_px", "label"])
    distance_summary = aggregate_object_rows(object_result_rows, ["radius_px", "label", "distance_bin_m"])
    frame_summary = aggregate_frame_rows(frame_result_rows)

    write_csv(
        output_dir / "raster_object_rows.csv",
        object_result_rows,
        [
            "sample_id",
            "radius_px",
            "label",
            "distance_bin_m",
            "gt_distance_m",
            "box_area_px",
            "box_covered_pixels",
            "box_coverage_fraction",
            "box_supported",
            "box_supported_5px",
            "radar_support_points_original",
        ],
    )
    write_csv(
        output_dir / "raster_frame_rows.csv",
        frame_result_rows,
        [
            "sample_id",
            "radius_px",
            "occupied_pixels",
            "occupied_object_pixels",
            "occupied_spillover_pixels",
            "spillover_fraction",
            "occupancy_fraction",
            "objects",
            "radar_points",
        ],
    )
    write_csv(
        output_dir / "raster_radius_class_summary.csv",
        class_summary,
        [
            "radius_px",
            "label",
            "objects",
            "support_rate_any",
            "support_rate_5px",
            "mean_box_coverage_fraction",
            "median_box_coverage_fraction",
            "p75_box_coverage_fraction",
            "p90_box_coverage_fraction",
            "mean_box_covered_pixels",
            "median_box_covered_pixels",
            "mean_box_area_px",
        ],
    )
    write_csv(
        output_dir / "raster_radius_distance_summary.csv",
        distance_summary,
        [
            "radius_px",
            "label",
            "distance_bin_m",
            "objects",
            "support_rate_any",
            "support_rate_5px",
            "mean_box_coverage_fraction",
            "median_box_coverage_fraction",
            "p75_box_coverage_fraction",
            "p90_box_coverage_fraction",
            "mean_box_covered_pixels",
            "median_box_covered_pixels",
            "mean_box_area_px",
        ],
    )
    write_csv(
        output_dir / "raster_radius_frame_summary.csv",
        frame_summary,
        [
            "radius_px",
            "frames",
            "mean_occupied_pixels",
            "mean_occupied_object_pixels",
            "mean_occupied_spillover_pixels",
            "mean_spillover_fraction",
            "mean_occupancy_fraction",
        ],
    )

    if bool(args.plot):
        plot_outputs(output_dir, class_summary, frame_summary)
    write_markdown(output_dir, dataset_dir=dataset_dir, radii=radii, class_summary=class_summary, frame_summary=frame_summary)
    metadata = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "radii_px": radii,
        "radar_tensor_width": radar_width,
        "radar_tensor_height": radar_height,
        "samples_analyzed": len(manifest_rows),
        "object_rows_analyzed": len(object_result_rows),
    }
    (output_dir / "raster_sweep_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

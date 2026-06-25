#!/usr/bin/env python3
"""Analyze short temporal radar accumulation on saved fusion datasets.

The saved fusion datasets include projected radar points for each sample. This
script accumulates the current sample plus a short history of prior samples,
repaints those points into the model input plane, and measures the benefit/cost:

- object-box radar coverage by class,
- object support rate by class,
- occupied radar pixels per frame,
- spillover outside current-frame GT object boxes,
- history span in seconds, which is the staleness cost.

This is analysis-only. It does not rewrite the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

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
        default="abiodun/analysis_outputs/radar_temporal_accumulation_sweep",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--windows", default="1,2,3,4,5", help="Comma-separated number of saved radar frames to accumulate.")
    parser.add_argument("--radii-px", default="4", help="Comma-separated radar raster radii to evaluate.")
    parser.add_argument(
        "--distance-bins-m",
        default="0,10,20,30,40,60,80,120,200",
        help="Comma-separated distance bin edges.",
    )
    parser.add_argument("--min-box-area-px", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--reset-gap-s",
        type=float,
        default=1.0,
        help="Reset accumulation history when consecutive samples are separated by more than this many seconds.",
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
        if radar.ndim == 3:
            return int(radar.shape[2]), int(radar.shape[1])
    raise FileNotFoundError("Could not infer radar tensor shape from manifest radar_tensor_path entries.")


def sample_groups(object_rows: Sequence[Mapping[str, str]]) -> Dict[str, List[Mapping[str, str]]]:
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in object_rows:
        groups[str(row.get("sample_id", ""))].append(row)
    return groups


def distance_bin(distance_m: float, edges: Sequence[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if float(left) <= distance_m < float(right):
            return f"{left:g}-{right:g}m"
    return f">={edges[-1]:g}m" if edges else "all"


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
        y0 = max(0, int(py) - radius)
        y1 = min(int(height), int(py) + radius + 1)
        x0 = max(0, int(px) - radius)
        x1 = min(int(width), int(px) + radius + 1)
        if y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = True
    return mask


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p75": float("nan"), "p90": float("nan")}
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
        coverage = summarize([float(v["box_coverage_fraction"]) for v in vals])
        covered_pixels = summarize([float(v["box_covered_pixels"]) for v in vals])
        support = [int(v["box_supported"]) for v in vals]
        support5 = [int(v["box_supported_5px"]) for v in vals]
        areas = [float(v["box_area_px"]) for v in vals]
        row = {name: value for name, value in zip(group_keys, key)}
        row.update(
            {
                "objects": len(vals),
                "support_rate_any": float(sum(support) / max(1, len(support))),
                "support_rate_5px": float(sum(support5) / max(1, len(support5))),
                "mean_box_coverage_fraction": coverage["mean"],
                "median_box_coverage_fraction": coverage["median"],
                "p75_box_coverage_fraction": coverage["p75"],
                "p90_box_coverage_fraction": coverage["p90"],
                "mean_box_covered_pixels": covered_pixels["mean"],
                "median_box_covered_pixels": covered_pixels["median"],
                "mean_box_area_px": float(mean(areas)) if areas else float("nan"),
            }
        )
        out.append(row)
    return out


def aggregate_frame_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[int, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["radius_px"]), int(row["window_frames"]))].append(row)
    out: List[Dict[str, object]] = []
    for (radius, window), vals in sorted(groups.items()):
        occupied = [float(v["occupied_pixels"]) for v in vals]
        object_pixels = [float(v["occupied_object_pixels"]) for v in vals]
        spill_pixels = [float(v["occupied_spillover_pixels"]) for v in vals]
        spill_frac = [float(v["spillover_fraction"]) for v in vals if math.isfinite(float(v["spillover_fraction"]))]
        occ_frac = [float(v["occupancy_fraction"]) for v in vals]
        history_span = [float(v["history_span_s"]) for v in vals]
        out.append(
            {
                "radius_px": radius,
                "window_frames": window,
                "frames": len(vals),
                "mean_history_span_s": float(mean(history_span)) if history_span else float("nan"),
                "mean_occupied_pixels": float(mean(occupied)) if occupied else float("nan"),
                "mean_occupied_object_pixels": float(mean(object_pixels)) if object_pixels else float("nan"),
                "mean_occupied_spillover_pixels": float(mean(spill_pixels)) if spill_pixels else float("nan"),
                "mean_spillover_fraction": float(mean(spill_frac)) if spill_frac else float("nan"),
                "mean_occupancy_fraction": float(mean(occ_frac)) if occ_frac else float("nan"),
            }
        )
    return out


def plot_radius(output_dir: Path, radius: int, class_summary: Sequence[Mapping[str, object]], frame_summary: Sequence[Mapping[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return

    rows_for_radius = [row for row in class_summary if int(row["radius_px"]) == int(radius)]
    frame_for_radius = sorted(
        [row for row in frame_summary if int(row["radius_px"]) == int(radius)],
        key=lambda row: int(row["window_frames"]),
    )
    if not rows_for_radius or not frame_for_radius:
        return
    by_label: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows_for_radius:
        by_label[str(row["label"])].append(row)
    for rows in by_label.values():
        rows.sort(key=lambda row: int(row["window_frames"]))

    colors = {"vehicle": "#0072B2", "person": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)

    ax = axes[0]
    for label in ["vehicle", "person"]:
        rows = by_label.get(label, [])
        if not rows:
            continue
        ax.plot(
            [int(row["window_frames"]) for row in rows],
            [float(row["mean_box_coverage_fraction"]) for row in rows],
            marker="o",
            linewidth=2.5,
            label=label.title(),
            color=colors.get(label),
        )
    ax.set_title("Radar Coverage Inside GT Boxes")
    ax.set_xlabel("Accumulated saved radar frames")
    ax.set_ylabel("Mean box coverage")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for label in ["vehicle", "person"]:
        rows = by_label.get(label, [])
        if not rows:
            continue
        ax.plot(
            [int(row["window_frames"]) for row in rows],
            [float(row["support_rate_5px"]) for row in rows],
            marker="o",
            linewidth=2.5,
            label=label.title(),
            color=colors.get(label),
        )
    ax.set_title("Objects With >=5 Radar Pixels")
    ax.set_xlabel("Accumulated saved radar frames")
    ax.set_ylabel("Support rate")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    windows = [int(row["window_frames"]) for row in frame_for_radius]
    occupied_pct = [100.0 * float(row["mean_occupancy_fraction"]) for row in frame_for_radius]
    span_s = [float(row["mean_history_span_s"]) for row in frame_for_radius]
    ax.plot(windows, occupied_pct, marker="o", linewidth=2.5, color="#009E73", label="painted area")
    ax.set_title("Painted Area and Staleness Cost")
    ax.set_xlabel("Accumulated saved radar frames")
    ax.set_ylabel("Occupied input tensor (%)", color="#009E73")
    ax.tick_params(axis="y", labelcolor="#009E73")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(windows, span_s, marker="s", linewidth=2.0, color="#CC79A7", label="history span")
    ax2.set_ylabel("Mean oldest-frame age (s)", color="#CC79A7")
    ax2.tick_params(axis="y", labelcolor="#CC79A7")
    for x, y in zip(windows, span_s):
        ax2.text(x, y + 0.015, f"{y:.1f}s", ha="center", fontsize=8, color="#7A3F68")

    fig.suptitle(f"Radar Temporal Accumulation Tradeoff, Radius {radius}px", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / f"radar_temporal_accumulation_tradeoff_radius{radius}.png", dpi=240)
    fig.savefig(output_dir / f"radar_temporal_accumulation_tradeoff_radius{radius}.pdf")
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    *,
    dataset_dir: Path,
    windows: Sequence[int],
    radii: Sequence[int],
    class_summary: Sequence[Mapping[str, object]],
    frame_summary: Sequence[Mapping[str, object]],
) -> None:
    by_radius_window_label = {
        (int(row["radius_px"]), int(row["window_frames"]), str(row["label"])): row for row in class_summary
    }
    frame_by_radius_window = {(int(row["radius_px"]), int(row["window_frames"])): row for row in frame_summary}
    lines = [
        "# Radar Temporal Accumulation Sweep",
        "",
        f"Dataset: `{dataset_dir}`",
        "",
        f"Windows tested: `{', '.join(str(w) for w in windows)}` saved radar frames",
        f"Radii tested: `{', '.join(str(r) for r in radii)}` pixels",
        "",
        "Note: history span is based on saved dataset samples. With 10 FPS and sample stride 2, each extra saved frame is about 0.2 s.",
        "",
        "## Coverage and Cost",
        "",
        "| radius px | saved frames | history span s | vehicle coverage | person coverage | vehicle support >=5px | person support >=5px | occupied tensor % | spillover fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for radius in radii:
        for window in windows:
            vehicle = by_radius_window_label.get((int(radius), int(window), "vehicle"), {})
            person = by_radius_window_label.get((int(radius), int(window), "person"), {})
            frame = frame_by_radius_window.get((int(radius), int(window)), {})
            lines.append(
                "| {r} | {w} | {span:.3f} | {vc:.4f} | {pc:.4f} | {vs:.3f} | {ps:.3f} | {occ:.2f} | {spill:.3f} |".format(
                    r=radius,
                    w=window,
                    span=float(frame.get("mean_history_span_s", float("nan"))),
                    vc=float(vehicle.get("mean_box_coverage_fraction", float("nan"))),
                    pc=float(person.get("mean_box_coverage_fraction", float("nan"))),
                    vs=float(vehicle.get("support_rate_5px", float("nan"))),
                    ps=float(person.get("support_rate_5px", float("nan"))),
                    occ=100.0 * float(frame.get("mean_occupancy_fraction", float("nan"))),
                    spill=float(frame.get("mean_spillover_fraction", float("nan"))),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- More saved frames means more radar evidence, but also older evidence.",
            "- If person coverage improves while occupied tensor area grows moderately, accumulation may help.",
            "- If occupied area and spillover grow faster than object support, accumulation likely adds stale noise.",
            "- This analysis uses naive image-plane accumulation, so it is a conservative test without ego-motion compensation.",
        ]
    )
    (output_dir / "radar_temporal_accumulation_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    object_path = dataset_dir / "object_boxes.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not object_path.exists():
        raise FileNotFoundError(f"Missing object boxes: {object_path}")

    windows = sorted(set(parse_int_list(args.windows)))
    radii = sorted(set(parse_int_list(args.radii_px)))
    if not windows:
        raise ValueError("No accumulation windows supplied.")
    if not radii:
        raise ValueError("No radii supplied.")
    max_window = max(windows)
    distance_edges = parse_float_list(args.distance_bins_m)

    manifest_rows_all = read_csv(manifest_path)
    stride = max(1, int(args.sample_stride))
    manifest_rows = [row for index, row in enumerate(manifest_rows_all) if index % stride == 0]
    if int(args.max_samples) > 0:
        manifest_rows = manifest_rows[: int(args.max_samples)]
    object_rows = read_csv(object_path)
    objects_by_sample = sample_groups(object_rows)
    radar_width, radar_height = load_radar_tensor_shape(dataset_dir, manifest_rows_all)

    run_name = args.run_name or dataset_dir.name
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    history: Deque[Dict[str, object]] = deque(maxlen=max_window)
    previous_experiment_id: Optional[str] = None
    previous_timestamp: Optional[float] = None
    object_result_rows: List[Dict[str, object]] = []
    frame_result_rows: List[Dict[str, object]] = []

    for sample_index, manifest in enumerate(manifest_rows):
        sample_id = str(manifest["sample_id"])
        experiment_id = str(manifest.get("experiment_id", ""))
        timestamp = to_float(manifest, "timestamp", float("nan"))
        if (
            previous_experiment_id is not None
            and (
                experiment_id != previous_experiment_id
                or timestamp < float(previous_timestamp or 0.0)
                or (math.isfinite(timestamp) and math.isfinite(float(previous_timestamp or float("nan"))) and timestamp - float(previous_timestamp or timestamp) > float(args.reset_gap_s))
            )
        ):
            history.clear()
        previous_experiment_id = experiment_id
        previous_timestamp = timestamp

        radar_path = dataset_dir / str(manifest.get("radar_points_path", ""))
        if not radar_path.exists():
            continue
        with np.load(radar_path) as radar_points:
            history.append(
                {
                    "sample_id": sample_id,
                    "timestamp": timestamp,
                    "u": np.asarray(radar_points["u"], dtype=np.float32),
                    "v": np.asarray(radar_points["v"], dtype=np.float32),
                    "valid": np.asarray(radar_points["valid_projection"], dtype=np.uint8),
                }
            )

        camera_width = max(1.0, to_float(manifest, "camera_width", radar_width))
        camera_height = max(1.0, to_float(manifest, "camera_height", radar_height))
        scale_x = float(radar_width) / camera_width
        scale_y = float(radar_height) / camera_height
        boxes: List[Tuple[Mapping[str, str], Tuple[int, int, int, int, float]]] = []
        object_union = np.zeros((radar_height, radar_width), dtype=bool)
        for obj in objects_by_sample.get(sample_id, []):
            scaled = scaled_box(obj, scale_x=scale_x, scale_y=scale_y, width=radar_width, height=radar_height)
            if scaled is None or float(scaled[4]) < float(args.min_box_area_px):
                continue
            boxes.append((obj, scaled))
            x0, y0, x1, y1, _ = scaled
            object_union[y0:y1, x0:x1] = True

        for window in windows:
            if len(history) < int(window):
                continue
            entries = list(history)[-int(window) :]
            history_span_s = float(timestamp - float(entries[0]["timestamp"])) if math.isfinite(timestamp) else float("nan")
            u = np.concatenate([entry["u"] for entry in entries])
            v = np.concatenate([entry["v"] for entry in entries])
            valid = np.concatenate([entry["valid"] for entry in entries])
            for radius in radii:
                occ = paint_occupancy(width=radar_width, height=radar_height, u=u, v=v, valid=valid, radius_px=int(radius))
                occupied_pixels = int(np.count_nonzero(occ))
                occupied_object_pixels = int(np.count_nonzero(occ & object_union))
                occupied_spillover_pixels = max(0, occupied_pixels - occupied_object_pixels)
                spillover_fraction = float(occupied_spillover_pixels / occupied_pixels) if occupied_pixels > 0 else float("nan")
                frame_result_rows.append(
                    {
                        "sample_id": sample_id,
                        "radius_px": int(radius),
                        "window_frames": int(window),
                        "history_span_s": history_span_s,
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
                            "window_frames": int(window),
                            "label": label,
                            "distance_bin_m": distance_bin(distance_m, distance_edges),
                            "gt_distance_m": distance_m,
                            "box_area_px": float(area),
                            "box_covered_pixels": covered,
                            "box_coverage_fraction": float(covered / max(1.0, area)),
                            "box_supported": int(covered > 0),
                            "box_supported_5px": int(covered >= 5),
                        }
                    )
        if sample_index == 0 or (sample_index + 1) % 250 == 0 or sample_index + 1 == len(manifest_rows):
            print(f"Analyzed {sample_index + 1}/{len(manifest_rows)} samples")

    class_summary = aggregate_object_rows(object_result_rows, ["radius_px", "window_frames", "label"])
    distance_summary = aggregate_object_rows(object_result_rows, ["radius_px", "window_frames", "label", "distance_bin_m"])
    frame_summary = aggregate_frame_rows(frame_result_rows)

    write_csv(
        output_dir / "temporal_object_rows.csv",
        object_result_rows,
        [
            "sample_id",
            "radius_px",
            "window_frames",
            "label",
            "distance_bin_m",
            "gt_distance_m",
            "box_area_px",
            "box_covered_pixels",
            "box_coverage_fraction",
            "box_supported",
            "box_supported_5px",
        ],
    )
    write_csv(
        output_dir / "temporal_frame_rows.csv",
        frame_result_rows,
        [
            "sample_id",
            "radius_px",
            "window_frames",
            "history_span_s",
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
        output_dir / "temporal_radius_window_class_summary.csv",
        class_summary,
        [
            "radius_px",
            "window_frames",
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
        output_dir / "temporal_radius_window_distance_summary.csv",
        distance_summary,
        [
            "radius_px",
            "window_frames",
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
        output_dir / "temporal_radius_window_frame_summary.csv",
        frame_summary,
        [
            "radius_px",
            "window_frames",
            "frames",
            "mean_history_span_s",
            "mean_occupied_pixels",
            "mean_occupied_object_pixels",
            "mean_occupied_spillover_pixels",
            "mean_spillover_fraction",
            "mean_occupancy_fraction",
        ],
    )
    if bool(args.plot):
        for radius in radii:
            plot_radius(output_dir, int(radius), class_summary, frame_summary)
    write_markdown(
        output_dir,
        dataset_dir=dataset_dir,
        windows=windows,
        radii=radii,
        class_summary=class_summary,
        frame_summary=frame_summary,
    )
    metadata = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "windows": windows,
        "radii_px": radii,
        "radar_tensor_width": radar_width,
        "radar_tensor_height": radar_height,
        "samples_analyzed": len(manifest_rows),
        "object_rows_analyzed": len(object_result_rows),
    }
    (output_dir / "temporal_sweep_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

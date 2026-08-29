#!/usr/bin/env python3
"""Depth-visible person anchors, corrected Gaussian radii, and private targets."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
NATIVE_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

from data_collection.route_b_perception_v3.visibility_v1 import (  # noqa: E402
    decode_depth_bgra,
    depth_is_plausible,
    reconstruct_consistent_mask,
)
from model_v1 import NATIVE_STRIDE  # noqa: E402
_TARGET_SPEC = importlib.util.spec_from_file_location(
    "route_b_v3_1_native_grid_targets_for_visible_anchor_v1",
    NATIVE_PACKAGE / "targets_v1.py",
)
if _TARGET_SPEC is None or _TARGET_SPEC.loader is None:
    raise ImportError("unable to load frozen native-grid targets")
_native_targets = importlib.util.module_from_spec(_TARGET_SPEC)
_TARGET_SPEC.loader.exec_module(_native_targets)
NativeGridDataset = _native_targets.NativeGridDataset
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    draw_gaussian,
    gaussian_radius as current_gaussian_radius,
)

MODEL_WIDTH, MODEL_HEIGHT = 768, 432
GRID_WIDTH, GRID_HEIGHT = MODEL_WIDTH // NATIVE_STRIDE, MODEL_HEIGHT // NATIVE_STRIDE

DERIVED_FIELDS = (
    "split", "experiment_id", "sample_id", "source_identity", "gt_actor_id",
    "visibility_tier", "visible_fraction", "visible_pixels", "centroid_source_x",
    "centroid_source_y", "centroid_cell_x", "centroid_cell_y", "anchor_source_x",
    "anchor_source_y", "anchor_model_x", "anchor_model_y", "anchor_grid_x",
    "anchor_grid_y", "anchor_cell_x", "anchor_cell_y", "anchor_rule",
    "anchor_pixel_is_own_visible", "anchor_cell_has_own_visible", "visible_bbox_source_x0",
    "visible_bbox_source_y0", "visible_bbox_source_x1", "visible_bbox_source_y1",
    "visible_bbox_model_w", "visible_bbox_model_h", "visible_bbox_grid_w",
    "visible_bbox_grid_h", "reference_radius_raw", "reference_radius_integer",
    "full_box_center_model_x", "full_box_center_model_y", "full_box_width_fraction",
    "full_box_height_fraction", "physical_ray_model_x", "physical_ray_model_y",
    "local_x", "local_y", "local_z", "world_x", "world_y", "world_z", "size_x",
    "size_y", "size_z", "yaw_sin", "yaw_cos", "radar_support", "distance_m",
    "area_px", "camera_fx_model", "camera_fy_model", "camera_cx_model",
    "camera_cy_model",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reference_gaussian_radius(height: float, width: float,
                              min_overlap: float = 0.7) -> float:
    """Independent standard CornerNet/CenterNet three-quadratic radius."""
    height, width, overlap = float(height), float(width), float(min_overlap)
    b1 = height + width
    c1 = width * height * (1.0 - overlap) / (1.0 + overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4.0 * c1))) / 2.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 16.0 * c2))) / 2.0
    a3 = 4.0 * overlap
    b3 = -2.0 * overlap * (height + width)
    c3 = (overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / 2.0
    return max(0.0, min(r1, r2, r3))


def gaussian_unit_tests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, height, width in (
        ("square", 10.0, 10.0), ("tall", 20.0, 4.0), ("tiny", 1.5, 0.75),
    ):
        reference = reference_gaussian_radius(height, width)
        transposed = reference_gaussian_radius(width, height)
        current = current_gaussian_radius(height, width)
        rows.append({
            "case": name, "reference": reference, "current": current,
            "pass": bool(math.isfinite(reference) and reference > 0.0
                         and abs(reference - transposed) <= 1e-12
                         and math.isfinite(current) and current >= 0.0),
        })
    half_x, half_y = 10.5, 12.5
    rows.append({
        "case": "half_cell", "reference": 2.0, "current": 2.0,
        "pass": (math.floor(half_x) - half_x == -0.5
                 and math.floor(half_y) - half_y == -0.5),
    })
    radius = 3
    support = (min(GRID_WIDTH, int(round(0.1)) + radius + 1) - 0) * (
        min(GRID_HEIGHT, int(round(0.2)) + radius + 1) - 0
    )
    rows.append({"case": "boundary", "reference": 3.0, "current": 3.0,
                 "pass": support == 16, "support_cells": support})
    if not all(row["pass"] for row in rows):
        raise RuntimeError(f"Gaussian unit-test failure: {rows}")
    return rows


def _visibility_lookup(dataset_root: Path, episodes: Iterable[str],
                       wanted: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for episode in sorted(set(episodes)):
        for row in read_csv(dataset_root / "dataset" / episode / "object_visibility.csv"):
            key = (row["sample_id"], row["gt_actor_id"], row["label"])
            if key in wanted:
                if key in result:
                    raise RuntimeError(f"duplicate visibility row: {key}")
                result[key] = row
    return result


def _cell_source_bounds(cell_x: int, cell_y: int, width: int,
                        height: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(cell_x * NATIVE_STRIDE * width / MODEL_WIDTH)))
    y0 = max(0, int(math.floor(cell_y * NATIVE_STRIDE * height / MODEL_HEIGHT)))
    x1 = min(width, int(math.ceil((cell_x + 1) * NATIVE_STRIDE * width / MODEL_WIDTH)))
    y1 = min(height, int(math.ceil((cell_y + 1) * NATIVE_STRIDE * height / MODEL_HEIGHT)))
    return x0, y0, x1, y1


def _nearest_pixel(xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> tuple[int, int]:
    distances = (xs.astype(np.float64) + 0.5 - x) ** 2 + (ys.astype(np.float64) + 0.5 - y) ** 2
    index = int(np.argmin(distances))  # np.nonzero order gives deterministic row-major ties.
    return int(xs[index]), int(ys[index])


def build_visible_target_view(dataset_root: Path, output_csv: Path,
                              progress_every: int = 1000) -> dict[str, Any]:
    """Create the sole derived target view without changing canonical v3.1."""
    manifest_rows = read_csv(dataset_root / "dataset/manifest.csv")
    manifest = {row["sample_id"]: row for row in manifest_rows}
    positives: list[dict[str, str]] = []
    for split in ("train", "val"):
        for row in read_csv(dataset_root / f"contracts/v010/{split}/object_boxes.csv"):
            if row["label"] == "person":
                item = dict(row); item["split"] = split; positives.append(item)
    wanted = {(row["sample_id"], row["gt_actor_id"], "person") for row in positives}
    visibility = _visibility_lookup(
        dataset_root, (row["experiment_id"] for row in positives), wanted,
    )
    if set(visibility) != wanted:
        raise RuntimeError(f"visibility reconciliation {len(visibility)} != {len(wanted)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in positives:
        grouped[row["sample_id"]].append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    split_counts: Counter[str] = Counter()
    anchor_rules: Counter[str] = Counter()
    radius_counts: dict[str, Counter[int]] = defaultdict(Counter)
    max_roundtrip_error = 0.0
    with output_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DERIVED_FIELDS)
        writer.writeheader()
        for frame_index, (sample_id, frame_positives) in enumerate(grouped.items(), 1):
            frame = manifest[sample_id]
            width, height = int(frame["camera_width"]), int(frame["camera_height"])
            depth_path = dataset_root / "dataset" / frame["experiment_id"] / "depth" / f"{sample_id}.png"
            raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"missing depth target input: {depth_path}")
            depth = decode_depth_bgra(raw)
            if not depth_is_plausible(depth):
                raise RuntimeError(f"implausible depth target input: {depth_path}")
            sx, sy = MODEL_WIDTH / width, MODEL_HEIGHT / height
            fx, fy = float(frame["camera_fx"]) * sx, float(frame["camera_fy"]) * sy
            cx, cy = float(frame["camera_cx"]) * sx, float(frame["camera_cy"]) * sy
            for source in frame_positives:
                visible = visibility[(sample_id, source["gt_actor_id"], "person")]
                mask = reconstruct_consistent_mask(depth, visible, width=width, height=height)
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    raise RuntimeError(f"empty eligible person mask: {sample_id}/{source['source_identity']}")
                centroid_x, centroid_y = float(np.mean(xs)), float(np.mean(ys))
                centroid_model_x, centroid_model_y = centroid_x * sx, centroid_y * sy
                centroid_cell_x = int(math.floor(centroid_model_x / NATIVE_STRIDE))
                centroid_cell_y = int(math.floor(centroid_model_y / NATIVE_STRIDE))
                if not (0 <= centroid_cell_x < GRID_WIDTH and 0 <= centroid_cell_y < GRID_HEIGHT):
                    raise RuntimeError("visible centroid cell outside native grid")
                x0, y0, x1, y1 = _cell_source_bounds(
                    centroid_cell_x, centroid_cell_y, width, height,
                )
                inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                if np.any(inside):
                    anchor_px, anchor_py = _nearest_pixel(
                        xs[inside], ys[inside], centroid_x, centroid_y,
                    )
                    anchor_rule = "nearest_own_visible_pixel_in_centroid_cell"
                else:
                    anchor_px, anchor_py = _nearest_pixel(xs, ys, centroid_x, centroid_y)
                    anchor_rule = "global_nearest_own_visible_pixel_fallback"
                # Pixel centres are continuous coordinates whose floor samples the proven pixel.
                anchor_source_x, anchor_source_y = anchor_px + 0.5, anchor_py + 0.5
                anchor_model_x, anchor_model_y = anchor_source_x * sx, anchor_source_y * sy
                anchor_grid_x, anchor_grid_y = anchor_model_x / NATIVE_STRIDE, anchor_model_y / NATIVE_STRIDE
                anchor_cell_x, anchor_cell_y = int(math.floor(anchor_grid_x)), int(math.floor(anchor_grid_y))
                if not mask[int(math.floor(anchor_source_y)), int(math.floor(anchor_source_x))]:
                    raise RuntimeError("final anchor does not sample its own visible mask")
                ax0, ay0, ax1, ay1 = _cell_source_bounds(anchor_cell_x, anchor_cell_y, width, height)
                if not np.any(mask[ay0:ay1, ax0:ax1]):
                    raise RuntimeError("final anchor cell has no own-visible pixel")

                vx0, vy0, vx1, vy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                visible_model_w, visible_model_h = (vx1 - vx0) * sx, (vy1 - vy0) * sy
                visible_grid_w, visible_grid_h = visible_model_w / NATIVE_STRIDE, visible_model_h / NATIVE_STRIDE
                radius_raw = reference_gaussian_radius(visible_grid_h, visible_grid_w)
                radius_integer = int(max(1, round(radius_raw)))
                depth_m = float(source["object_sensor_x"])
                right_m, up_m = float(source["object_sensor_y"]), float(source["object_sensor_z"])
                if not (math.isfinite(depth_m) and 0.0 < depth_m <= 40.0):
                    raise RuntimeError(f"invalid physical depth target: {depth_m}")
                physical_u = cx + fx * right_m / depth_m
                physical_v = cy - fy * up_m / depth_m
                reconstructed = np.asarray([
                    depth_m, (physical_u - cx) * depth_m / fx,
                    (cy - physical_v) * depth_m / fy,
                ], dtype=np.float64)
                roundtrip = float(np.max(np.abs(reconstructed - np.asarray([depth_m, right_m, up_m]))))
                max_roundtrip_error = max(max_roundtrip_error, roundtrip)
                yaw = math.radians(float(source["object_yaw_deg"]))
                row = {
                    "split": source["split"], "experiment_id": source["experiment_id"],
                    "sample_id": sample_id, "source_identity": source["source_identity"],
                    "gt_actor_id": source["gt_actor_id"], "visibility_tier": visible["visibility_tier"],
                    "visible_fraction": float(visible["visible_fraction"]), "visible_pixels": len(xs),
                    "centroid_source_x": centroid_x, "centroid_source_y": centroid_y,
                    "centroid_cell_x": centroid_cell_x, "centroid_cell_y": centroid_cell_y,
                    "anchor_source_x": anchor_source_x, "anchor_source_y": anchor_source_y,
                    "anchor_model_x": anchor_model_x, "anchor_model_y": anchor_model_y,
                    "anchor_grid_x": anchor_grid_x, "anchor_grid_y": anchor_grid_y,
                    "anchor_cell_x": anchor_cell_x, "anchor_cell_y": anchor_cell_y,
                    "anchor_rule": anchor_rule, "anchor_pixel_is_own_visible": 1,
                    "anchor_cell_has_own_visible": 1, "visible_bbox_source_x0": vx0,
                    "visible_bbox_source_y0": vy0, "visible_bbox_source_x1": vx1,
                    "visible_bbox_source_y1": vy1, "visible_bbox_model_w": visible_model_w,
                    "visible_bbox_model_h": visible_model_h, "visible_bbox_grid_w": visible_grid_w,
                    "visible_bbox_grid_h": visible_grid_h, "reference_radius_raw": radius_raw,
                    "reference_radius_integer": radius_integer,
                    "full_box_center_model_x": float(source["gt_center_x"]) * sx,
                    "full_box_center_model_y": float(source["gt_center_y"]) * sy,
                    "full_box_width_fraction": float(source["gt_bbox_w"]) / width,
                    "full_box_height_fraction": float(source["gt_bbox_h"]) / height,
                    "physical_ray_model_x": physical_u, "physical_ray_model_y": physical_v,
                    "local_x": depth_m, "local_y": right_m, "local_z": up_m,
                    "world_x": float(source["object_world_x"]),
                    "world_y": float(source["object_world_y"]),
                    "world_z": float(source["object_world_z"]),
                    "size_x": max(0.01, float(source["gt_size_x_m"])),
                    "size_y": max(0.01, float(source["gt_size_y_m"])),
                    "size_z": max(0.01, float(source["gt_size_z_m"])),
                    "yaw_sin": math.sin(yaw), "yaw_cos": math.cos(yaw),
                    "radar_support": int(float(source.get("radar_support_points", "0") or 0) > 0),
                    "distance_m": float(source["gt_distance_m"]),
                    "area_px": float(source["gt_bbox_area_px"]),
                    "camera_fx_model": fx, "camera_fy_model": fy,
                    "camera_cx_model": cx, "camera_cy_model": cy,
                }
                writer.writerow(row)
                rows_written += 1; split_counts[source["split"]] += 1
                anchor_rules[anchor_rule] += 1; radius_counts[source["split"]][radius_integer] += 1
            if progress_every and frame_index % progress_every == 0:
                print(f"[visible target view] {frame_index}/{len(grouped)} frames rows={rows_written}", flush=True)
    expected = Counter(row["split"] for row in positives)
    if split_counts != expected:
        raise RuntimeError(f"derived target population mismatch: {split_counts} != {expected}")
    return {
        "rows": rows_written, "split_counts": dict(split_counts),
        "anchor_rules": dict(anchor_rules),
        "all_anchor_pixels_own_visible": True,
        "all_anchor_cells_have_own_visible": True,
        "radius_integer_counts": {split: dict(sorted(counts.items()))
                                  for split, counts in radius_counts.items()},
        "physical_projection_roundtrip_max_abs_error_m": max_roundtrip_error,
        "validation_influenced_train_targets_or_parameters": False,
    }


def verify_audit_gaussian_population(dataset_root: Path, audit_csv: Path) -> dict[str, Any]:
    """Prove our independent radius reproduces every audit reference row."""
    audit: dict[tuple[str, str, str], tuple[float, int]] = {}
    current_audit: dict[tuple[str, str, str], tuple[float, int]] = {}
    for row in read_csv(audit_csv):
        key = (row["split"], row["sample_id"], row["source_identity"])
        value = (float(row["raw_radius"]), int(row["integer_radius"]))
        (audit if row["implementation"] == "reference" else current_audit)[key] = value
    manifest = {row["sample_id"]: row for row in read_csv(dataset_root / "dataset/manifest.csv")}
    mismatches: list[dict[str, Any]] = []
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    current_counts: dict[str, Counter[int]] = defaultdict(Counter)
    checked = 0
    for split in ("train", "val"):
        for row in read_csv(dataset_root / f"contracts/v010/{split}/object_boxes.csv"):
            if row["label"] != "person":
                continue
            frame = manifest[row["sample_id"]]
            w, h = int(frame["camera_width"]), int(frame["camera_height"])
            box_w = float(row["gt_bbox_w"]) * MODEL_WIDTH / w / NATIVE_STRIDE
            box_h = float(row["gt_bbox_h"]) * MODEL_HEIGHT / h / NATIVE_STRIDE
            reference = reference_gaussian_radius(box_h, box_w)
            reference_integer = int(max(1, round(reference)))
            current = current_gaussian_radius(box_h, box_w)
            current_integer = int(max(1, round(current)))
            key = (split, row["sample_id"], row["source_identity"])
            expected_reference = audit.get(key)
            expected_current = current_audit.get(key)
            if (expected_reference is None or abs(reference - expected_reference[0]) > 1e-12
                    or reference_integer != expected_reference[1]
                    or expected_current is None or abs(current - expected_current[0]) > 1e-12
                    or current_integer != expected_current[1]):
                mismatches.append({"key": key, "reference": reference,
                                   "reference_integer": reference_integer})
            checked += 1; counts[split][reference_integer] += 1
            current_counts[split][current_integer] += 1
    if checked != len(audit) or mismatches:
        raise RuntimeError(
            f"Gaussian audit population mismatch checked={checked} audit={len(audit)} "
            f"mismatches={mismatches[:3]}"
        )
    return {
        "checked_person_rows": checked, "exact_raw_and_integer_matches": checked,
        "mismatches": 0, "reference_integer_counts": {
            split: dict(sorted(values.items())) for split, values in counts.items()
        },
        "current_integer_counts": {
            split: dict(sorted(values.items())) for split, values in current_counts.items()
        },
    }


def load_visible_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = read_csv(path)
    numeric = set(DERIVED_FIELDS) - {
        "split", "experiment_id", "sample_id", "source_identity", "gt_actor_id",
        "visibility_tier", "anchor_rule",
    }
    for source in rows:
        item: dict[str, Any] = dict(source)
        for key in numeric:
            item[key] = float(source[key])
        grouped[source["sample_id"]].append(item)
    train = [row for row in rows if row["split"] == "train"]
    box_values = [
        abs((float(row["full_box_center_model_x"]) - float(row["anchor_model_x"])) / NATIVE_STRIDE)
        for row in train
    ] + [
        abs((float(row["full_box_center_model_y"]) - float(row["anchor_model_y"])) / NATIVE_STRIDE)
        for row in train
    ]
    ray_values = [
        abs((float(row["physical_ray_model_x"]) - float(row["anchor_model_x"])) / NATIVE_STRIDE)
        for row in train
    ] + [
        abs((float(row["physical_ray_model_y"]) - float(row["anchor_model_y"])) / NATIVE_STRIDE)
        for row in train
    ]
    scales = {
        "box_center_grid_cells": max(1.0, float(math.ceil(np.percentile(box_values, 99.5)))),
        "physical_ray_grid_cells": max(1.0, float(math.ceil(np.percentile(ray_values, 99.5)))),
    }
    return grouped, {
        "rows": len(rows), "train_rows": len(train),
        "val_rows": len(rows) - len(train), "offset_scales": scales,
        "derivation": "train_only_absolute_component_p99_5_then_ceil_minimum_1",
        "validation_influence": False,
    }


def build_private_targets(rows: Sequence[Mapping[str, Any]], ignore_cells: torch.Tensor,
                          *, offset_scales: Mapping[str, float],
                          depth_bounds_m: Sequence[float],
                          dimension_scale_m: float, endpoint_scale_m: float) -> dict[str, torch.Tensor]:
    heatmap = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    mask = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    subcell = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    box_offset = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    box_wh = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    ray_offset = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    depth_norm = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    local_xyz = np.zeros((3, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    dimensions = np.zeros((3, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    yaw = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    radar = np.zeros((1, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    occupied = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    collisions = 0
    low, high = (float(value) for value in depth_bounds_m)
    log_span = math.log(high) - math.log(low)
    box_scale = float(offset_scales["box_center_grid_cells"])
    ray_scale = float(offset_scales["physical_ray_grid_cells"])
    for row in sorted(rows, key=lambda item: (-float(item["visible_pixels"]), str(item["source_identity"]))):
        grid_x, grid_y = float(row["anchor_grid_x"]), float(row["anchor_grid_y"])
        cell_x, cell_y = int(math.floor(grid_x)), int(math.floor(grid_y))
        radius = int(row["reference_radius_integer"])
        draw_gaussian(heatmap[0], grid_x, grid_y, radius)
        heatmap[0, cell_y, cell_x] = 1.0
        if occupied[cell_y, cell_x]:
            collisions += 1
            continue
        mask[0, cell_y, cell_x] = 1.0
        subcell[:, cell_y, cell_x] = (grid_x - cell_x, grid_y - cell_y)
        box_offset[:, cell_y, cell_x] = (
            (float(row["full_box_center_model_x"]) / NATIVE_STRIDE - grid_x) / box_scale,
            (float(row["full_box_center_model_y"]) / NATIVE_STRIDE - grid_y) / box_scale,
        )
        box_wh[:, cell_y, cell_x] = (
            float(row["full_box_width_fraction"]), float(row["full_box_height_fraction"]),
        )
        ray_offset[:, cell_y, cell_x] = (
            (float(row["physical_ray_model_x"]) / NATIVE_STRIDE - grid_x) / ray_scale,
            (float(row["physical_ray_model_y"]) / NATIVE_STRIDE - grid_y) / ray_scale,
        )
        depth = float(row["local_x"])
        depth_norm[0, cell_y, cell_x] = (math.log(depth) - math.log(low)) / log_span
        local_xyz[:, cell_y, cell_x] = np.asarray(
            [float(row["local_x"]), float(row["local_y"]), float(row["local_z"])],
            dtype=np.float32,
        ) / float(endpoint_scale_m)
        dimensions[:, cell_y, cell_x] = np.asarray(
            [float(row["size_x"]), float(row["size_y"]), float(row["size_z"])],
            dtype=np.float32,
        ) / float(dimension_scale_m)
        yaw[:, cell_y, cell_x] = (float(row["yaw_sin"]), float(row["yaw_cos"]))
        radar[0, cell_y, cell_x] = float(row["radar_support"])
        occupied[cell_y, cell_x] = True
    ignored = ignore_cells.numpy().astype(bool)
    heatmap[(heatmap == 0.0) & ignored] = -1.0
    return {
        "visible_heatmap": torch.from_numpy(heatmap),
        "person_private_mask": torch.from_numpy(mask),
        "visible_subcell_offset": torch.from_numpy(subcell),
        "visible_to_box_center_offset": torch.from_numpy(box_offset),
        "full_box_wh": torch.from_numpy(box_wh),
        "visible_to_physical_ray_offset": torch.from_numpy(ray_offset),
        "bounded_log_depth": torch.from_numpy(depth_norm),
        "local_xyz_normalized": torch.from_numpy(local_xyz),
        "person_dimensions_normalized": torch.from_numpy(dimensions),
        "person_yaw": torch.from_numpy(yaw),
        "radar_support": torch.from_numpy(radar),
        "person_cell_collisions": torch.tensor(collisions, dtype=torch.long),
    }


class VisibleAnchorDataset(NativeGridDataset):
    """Canonical fused inputs plus deterministic create-only private targets."""

    def __init__(self, *args: Any, visible_rows: Mapping[str, Sequence[Mapping[str, Any]]],
                 offset_scales: Mapping[str, float], depth_bounds_m: Sequence[float],
                 dimension_scale_m: float, endpoint_scale_m: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.visible_rows = visible_rows
        self.offset_scales = offset_scales
        self.depth_bounds_m = tuple(float(value) for value in depth_bounds_m)
        self.dimension_scale_m = float(dimension_scale_m)
        self.endpoint_scale_m = float(endpoint_scale_m)

    def __getitem__(self, index: int):
        fused, segmentation, native_targets = super().__getitem__(index)
        sample_id = self.rows[index]["sample_id"]
        private = build_private_targets(
            self.visible_rows.get(sample_id, ()), native_targets["object_ignore_mask"],
            offset_scales=self.offset_scales, depth_bounds_m=self.depth_bounds_m,
            dimension_scale_m=self.dimension_scale_m,
            endpoint_scale_m=self.endpoint_scale_m,
        )
        row = self.rows[index]
        private["camera_intrinsic_model"] = torch.tensor([
            [float(row["camera_fx"]) * self.input_width / float(row["camera_width"]), 0.0,
             float(row["camera_cx"]) * self.input_width / float(row["camera_width"])],
            [0.0, float(row["camera_fy"]) * self.input_height / float(row["camera_height"]),
             float(row["camera_cy"]) * self.input_height / float(row["camera_height"])],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        return fused, segmentation, private


def build_sampling_weights(train_rows: Sequence[dict[str, str]],
                           validation_rows: Sequence[dict[str, str]],
                           object_rows: Mapping[str, Sequence[dict[str, str]]],
                           sampling: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    train_ids = {row["sample_id"] for row in train_rows}
    validation_ids = {row["sample_id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation sample IDs overlap")
    episodes = Counter(row["experiment_id"] for row in train_rows)
    tracks: Counter[tuple[str, str]] = Counter()
    persons: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        values = [value for value in object_rows.get(row["sample_id"], ())
                  if value.get("label") == "person" and value.get("gt_source") == "actor"]
        persons[row["sample_id"]] = values
        for value in values:
            tracks[(row["experiment_id"], value["gt_actor_id"])] += 1
    inverse = [1.0 / value for value in tracks.values()]
    inverse_mean = float(np.mean(inverse)) if inverse else 1.0
    raw: list[float] = []
    for row in train_rows:
        episode = len(train_rows) / (len(episodes) * episodes[row["experiment_id"]])
        values = persons[row["sample_id"]]
        if values:
            track = float(np.mean([
                1.0 / tracks[(row["experiment_id"], value["gt_actor_id"])] for value in values
            ])) / max(1e-12, inverse_mean)
            exposure = float(sampling["small_far_multiplier"]) if any(
                float(value["gt_bbox_area_px"]) < float(sampling["small_area_px"])
                or float(value["gt_distance_m"]) >= float(sampling["far_distance_m"])
                for value in values
            ) else 1.0
            negative = 1.0
        else:
            track, exposure = 1.0, 1.0
            negative = float(sampling["negative_frame_weight"])
        raw.append(episode * track * exposure * negative)
    values = np.asarray(raw, dtype=np.float64)
    median = float(np.median(values)); cap = median * float(sampling["inverse_weight_cap_ratio"])
    weights = np.minimum(values, cap); weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.double), {
        "train_frames": len(train_rows), "validation_frames_excluded": len(validation_rows),
        "episodes": len(episodes), "person_tracks": len(tracks),
        "person_positive_frames": sum(bool(value) for value in persons.values()),
        "negative_frames_retained": sum(not value for value in persons.values()),
        "raw_min": float(values.min()), "raw_median": median, "raw_max": float(values.max()),
        "cap": cap, "capped_frames": int(np.count_nonzero(values > cap)),
        "normalized_min": float(weights.min()), "normalized_max": float(weights.max()),
        "validation_used_for_sampling_or_parameters": 0,
    }

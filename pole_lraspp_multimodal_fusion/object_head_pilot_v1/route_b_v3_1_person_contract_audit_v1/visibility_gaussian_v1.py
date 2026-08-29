"""Depth-consistent target observability and independent Gaussian-radius audit."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
FUSION = ROOT / "pole_lraspp_multimodal_fusion"
if str(FUSION) not in sys.path:
    sys.path.insert(0, str(FUSION))

from data_collection.route_b_perception_v3.visibility_v1 import (  # noqa: E402
    decode_depth_bgra,
    depth_is_plausible,
    reconstruct_consistent_mask,
)
from pole_lraspp_multimodal_fusion.object_targets import gaussian_radius as current_gaussian_radius  # noqa: E402


MODEL_WIDTH, MODEL_HEIGHT, STRIDE = 768, 432, 4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _person_positives(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in ("v010", "v025"):
        for split in ("train", "val"):
            path = dataset_root / f"contracts/{contract}/{split}/object_boxes.csv"
            for row in read_csv(path):
                if row["label"] != "person":
                    continue
                if row["source_kind"] != "actor" or ":actor:" not in row["source_identity"]:
                    raise RuntimeError(f"unexpected authoritative person source: {row['source_identity']}")
                item = dict(row)
                item["contract"] = contract
                item["split"] = split
                rows.append(item)
    return rows


def _manifest_map(dataset_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(dataset_root / "dataset/manifest.csv")
    mapping = {row["sample_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise RuntimeError("manifest sample-ID uniqueness failure")
    return mapping


def _raw_visibility_maps(
    dataset_root: Path, episodes: Iterable[str], wanted: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    mapping: dict[tuple[str, str, str], dict[str, str]] = {}
    for episode in sorted(set(episodes)):
        path = dataset_root / "dataset" / episode / "object_visibility.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["sample_id"], row["gt_actor_id"], row["label"])
                if key not in wanted:
                    continue
                if key in mapping:
                    raise RuntimeError(f"duplicate raw visibility key: {key}")
                mapping[key] = row
    return mapping


def _bin_label(value: float, edges: Sequence[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right:
            return f"[{left:g},{right:g})"
    return f"[{edges[-2]:g},{edges[-1]:g})"


def _model_xy(source_x: float, source_y: float, width: int, height: int) -> tuple[float, float]:
    return source_x * MODEL_WIDTH / width, source_y * MODEL_HEIGHT / height


def _cell_mask_support(mask: np.ndarray, model_x: float, model_y: float, width: int, height: int) -> tuple[bool, bool]:
    cell_x, cell_y = int(math.floor(model_x / STRIDE)), int(math.floor(model_y / STRIDE))
    in_bounds = 0 <= cell_x < MODEL_WIDTH // STRIDE and 0 <= cell_y < MODEL_HEIGHT // STRIDE
    if not in_bounds:
        return False, False
    source_x0 = int(math.floor(cell_x * STRIDE * width / MODEL_WIDTH))
    source_y0 = int(math.floor(cell_y * STRIDE * height / MODEL_HEIGHT))
    source_x1 = int(math.ceil((cell_x + 1) * STRIDE * width / MODEL_WIDTH))
    source_y1 = int(math.ceil((cell_y + 1) * STRIDE * height / MODEL_HEIGHT))
    source_x0, source_y0 = max(0, source_x0), max(0, source_y0)
    source_x1, source_y1 = min(width, source_x1), min(height, source_y1)
    return True, bool(np.any(mask[source_y0:source_y1, source_x0:source_x1]))


def _audit_one(
    row: Mapping[str, Any], frame: Mapping[str, str], visibility: Mapping[str, str],
    depth: np.ndarray,
) -> dict[str, Any]:
    width, height = int(frame["camera_width"]), int(frame["camera_height"])
    mask = reconstruct_consistent_mask(depth, dict(visibility), width=width, height=height)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError(f"eligible person has empty reconstructed mask: {row['sample_id']}/{row['source_identity']}")
    tx, ty = float(row["gt_center_x"]), float(row["gt_center_y"])
    mx, my = _model_xy(tx, ty, width, height)
    grid_x, grid_y = mx / STRIDE, my / STRIDE
    cell_x, cell_y = int(math.floor(grid_x)), int(math.floor(grid_y))
    in_frame = 0.0 <= tx < width and 0.0 <= ty < height
    x0, y0 = float(visibility["clipped_bbox_x"]), float(visibility["clipped_bbox_y"])
    x1 = x0 + float(visibility["clipped_bbox_w"])
    y1 = y0 + float(visibility["clipped_bbox_h"])
    in_box = x0 <= tx < x1 and y0 <= ty < y1
    sample_x = min(width - 1, max(0, int(math.floor(tx))))
    sample_y = min(height - 1, max(0, int(math.floor(ty))))
    if not in_frame:
        category = "OUTSIDE_FRAME"
    elif not in_box:
        category = "OUTSIDE_OWN_PROJECTED_BOX"
    elif mask[sample_y, sample_x]:
        category = "ON_OWN_VISIBLE_MASK"
    else:
        depth_value = float(depth[sample_y, sample_x])
        lower = float(visibility["actor_near_depth_m"]) - float(visibility["depth_tolerance_m"])
        category = "ON_CLOSER_OCCLUDER" if math.isfinite(depth_value) and depth_value < lower else "INSIDE_PROJECTED_BOX_BUT_NOT_VISIBLE"

    distances2 = (xs.astype(np.float64) - tx) ** 2 + (ys.astype(np.float64) - ty) ** 2
    nearest_index = int(np.argmin(distances2))
    nearest_x, nearest_y = float(xs[nearest_index]), float(ys[nearest_index])
    centroid_x, centroid_y = float(np.mean(xs)), float(np.mean(ys))
    visible_box_cx = (float(np.min(xs)) + float(np.max(xs))) / 2.0
    visible_box_cy = (float(np.min(ys)) + float(np.max(ys))) / 2.0
    sx, sy = MODEL_WIDTH / width, MODEL_HEIGHT / height
    nearest_distance_model = math.hypot((nearest_x - tx) * sx, (nearest_y - ty) * sy)
    centroid_dx, centroid_dy = (centroid_x - tx) * sx, (centroid_y - ty) * sy
    visible_box_dx, visible_box_dy = (visible_box_cx - tx) * sx, (visible_box_cy - ty) * sy
    centroid_mx, centroid_my = _model_xy(centroid_x, centroid_y, width, height)
    nearest_mx, nearest_my = _model_xy(nearest_x, nearest_y, width, height)
    centroid_valid, centroid_supported = _cell_mask_support(mask, centroid_mx, centroid_my, width, height)
    nearest_valid, nearest_supported = _cell_mask_support(mask, nearest_mx, nearest_my, width, height)
    return {
        "contract": row["contract"], "split": row["split"], "episode_id": row["experiment_id"],
        "sample_id": row["sample_id"], "source_identity": row["source_identity"],
        "gt_actor_id": row["gt_actor_id"], "visibility_tier": visibility["visibility_tier"],
        "visible_fraction": float(visibility["visible_fraction"]),
        "distance_m": float(row["gt_distance_m"]), "distance_bin_m": _bin_label(float(row["gt_distance_m"]), (0, 10, 20, 30, 40.000001)),
        "area_px": float(row["gt_bbox_area_px"]), "area_bin_px": _bin_label(float(row["gt_bbox_area_px"]), (0, 400, 1600, 6400, 1e9)),
        "radar_supported": int(float(row.get("radar_support_points", "0") or 0) > 0),
        "category": category,
        "target_source_x": tx, "target_source_y": ty, "target_model_x": mx, "target_model_y": my,
        "target_grid_x": grid_x, "target_grid_y": grid_y, "forced_peak_cell_x": cell_x, "forced_peak_cell_y": cell_y,
        "nearest_visible_source_x": nearest_x, "nearest_visible_source_y": nearest_y,
        "nearest_visible_model_distance_px": nearest_distance_model,
        "visible_centroid_source_x": centroid_x, "visible_centroid_source_y": centroid_y,
        "visible_centroid_offset_x_model_px": centroid_dx, "visible_centroid_offset_y_model_px": centroid_dy,
        "visible_centroid_offset_model_px": math.hypot(centroid_dx, centroid_dy),
        "visible_bbox_center_offset_x_model_px": visible_box_dx,
        "visible_bbox_center_offset_y_model_px": visible_box_dy,
        "visible_bbox_center_offset_model_px": math.hypot(visible_box_dx, visible_box_dy),
        "visible_centroid_stride4_cell_valid": int(centroid_valid),
        "visible_centroid_stride4_cell_has_own_visible_pixel": int(centroid_supported),
        "nearest_visible_stride4_cell_valid": int(nearest_valid),
        "nearest_visible_stride4_cell_has_own_visible_pixel": int(nearest_supported),
        "native_visible_px": int(visibility["native_visible_px"]),
        "model_input_visible_px": int(visibility["model_input_visible_px"]),
        "projected_bbox_x": float(row["gt_bbox_x"]), "projected_bbox_y": float(row["gt_bbox_y"]),
        "projected_bbox_w": float(row["gt_bbox_w"]), "projected_bbox_h": float(row["gt_bbox_h"]),
        "depth_path": str(dataset_depth_path(frame, row["sample_id"])),
        "rgb_path": str(frame["rgb_path"]),
    }


def dataset_depth_path(frame: Mapping[str, str], sample_id: str) -> Path:
    return Path(frame["experiment_id"]) / "depth" / f"{sample_id}.png"


def audit_target_visibility(dataset_root: Path, progress_every: int = 1000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = _person_positives(dataset_root)
    manifest = _manifest_map(dataset_root)
    wanted = {(row["sample_id"], row["source_identity"].rsplit(":actor:", 1)[1], "person") for row in positives}
    visibility = _raw_visibility_maps(dataset_root, (row["experiment_id"] for row in positives), wanted)
    if set(visibility) != wanted:
        raise RuntimeError(f"retained visibility key reconciliation failure: found={len(visibility)} wanted={len(wanted)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        grouped[row["sample_id"]].append(row)
    output: list[dict[str, Any]] = []
    unique_reconstructions: dict[tuple[str, str], dict[str, Any]] = {}
    for frame_index, (sample_id, frame_rows) in enumerate(grouped.items(), 1):
        frame = manifest[sample_id]
        depth_path = dataset_root / "dataset" / dataset_depth_path(frame, sample_id)
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"missing retained depth: {depth_path}")
        depth = decode_depth_bgra(raw)
        if not depth_is_plausible(depth):
            raise RuntimeError(f"implausible retained depth: {depth_path}")
        for row in frame_rows:
            actor_id = row["source_identity"].rsplit(":actor:", 1)[1]
            key = (sample_id, actor_id, "person")
            if key not in visibility:
                raise RuntimeError(f"missing retained visibility row: {key}")
            unique_key = (sample_id, row["source_identity"])
            if unique_key not in unique_reconstructions:
                unique_reconstructions[unique_key] = _audit_one(row, frame, visibility[key], depth)
            base = dict(unique_reconstructions[unique_key])
            base["contract"], base["split"] = row["contract"], row["split"]
            output.append(base)
        if progress_every and frame_index % progress_every == 0:
            print(f"[target visibility] frames {frame_index}/{len(grouped)} rows={len(output)}", flush=True)
    expected = {(row["contract"], row["split"]): 0 for row in positives}
    for row in positives:
        expected[(row["contract"], row["split"])] += 1
    actual: dict[tuple[str, str], int] = defaultdict(int)
    for row in output:
        actual[(row["contract"], row["split"])] += 1
    if dict(actual) != expected:
        raise RuntimeError(f"visibility population reconciliation failure: {actual} != {expected}")
    return output, {
        "contract_split_denominators": {f"{key[0]}_{key[1]}": value for key, value in sorted(expected.items())},
        "unique_actor_frame_reconstructions": len(unique_reconstructions),
        "rows": len(output),
    }


def reference_gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Standard CornerNet/CenterNet radius, independently transcribed."""
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


def _draw_support(grid_x: float, grid_y: float, radius: int) -> dict[str, Any]:
    rounded_x, rounded_y = int(round(grid_x)), int(round(grid_y))
    x0, x1 = max(0, rounded_x - radius), min(MODEL_WIDTH // STRIDE, rounded_x + radius + 1)
    y0, y1 = max(0, rounded_y - radius), min(MODEL_HEIGHT // STRIDE, rounded_y + radius + 1)
    support = max(0, x1 - x0) * max(0, y1 - y0)
    return {
        "positive_cells": 1,
        "support_cells": support,
        "left_extent_cells": grid_x - x0,
        "right_extent_cells": (x1 - 1) - grid_x,
        "up_extent_cells": grid_y - y0,
        "down_extent_cells": (y1 - 1) - grid_y,
        "left_right_asymmetry_cells": ((x1 - 1) - grid_x) - (grid_x - x0),
        "up_down_asymmetry_cells": ((y1 - 1) - grid_y) - (grid_y - y0),
        "forced_floor_peak_dx_cells": math.floor(grid_x) - grid_x,
        "forced_floor_peak_dy_cells": math.floor(grid_y) - grid_y,
    }


def gaussian_unit_tests() -> list[dict[str, Any]]:
    cases = (("square", 10.0, 10.0), ("tall", 20.0, 4.0), ("tiny_person", 1.5, 0.75))
    rows: list[dict[str, Any]] = []
    for name, height, width in cases:
        value = reference_gaussian_radius(height, width)
        swapped = reference_gaussian_radius(width, height)
        passed = math.isfinite(value) and value > 0 and abs(value - swapped) < 1e-12
        rows.append({"case": name, "passed": passed, "reference_radius": value, "detail": "finite_positive_and_transpose_symmetric"})
    half = _draw_support(10.5, 12.5, 2)
    rows.append({"case": "half_cell", "passed": half["positive_cells"] == 1 and half["forced_floor_peak_dx_cells"] == -0.5,
                 "reference_radius": 2.0, "detail": "forced floor peak is exactly one positive and offset -0.5/-0.5"})
    boundary = _draw_support(0.1, 0.2, 3)
    rows.append({"case": "boundary_cell", "passed": boundary["support_cells"] == 16,
                 "reference_radius": 3.0, "detail": f"clipped support={boundary['support_cells']} expected=16"})
    if not all(row["passed"] for row in rows):
        raise RuntimeError(f"Gaussian independent unit-test failure: {rows}")
    return rows


def audit_gaussian(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tests = gaussian_unit_tests()
    rows: list[dict[str, Any]] = []
    frame_geometry = {row["sample_id"]: (int(row["camera_width"]), int(row["camera_height"]))
                      for row in read_csv(dataset_root / "dataset/manifest.csv")}
    for split in ("train", "val"):
        for source in read_csv(dataset_root / f"contracts/v010/{split}/object_boxes.csv"):
            if source["label"] != "person":
                continue
            source_width, source_height = frame_geometry[source["sample_id"]]
            box_w_cells = float(source["gt_bbox_w"]) * MODEL_WIDTH / source_width / STRIDE
            box_h_cells = float(source["gt_bbox_h"]) * MODEL_HEIGHT / source_height / STRIDE
            grid_x = float(source["gt_center_x"]) * MODEL_WIDTH / source_width / STRIDE
            grid_y = float(source["gt_center_y"]) * MODEL_HEIGHT / source_height / STRIDE
            current_raw = current_gaussian_radius(box_h_cells, box_w_cells)
            reference_raw = reference_gaussian_radius(box_h_cells, box_w_cells)
            values = (("current", current_raw), ("reference", reference_raw))
            for implementation, raw_radius in values:
                integer_radius = int(max(1, round(raw_radius)))
                support = _draw_support(grid_x, grid_y, integer_radius)
                rows.append({
                    "split": split, "episode_id": source["experiment_id"], "sample_id": source["sample_id"],
                    "source_identity": source["source_identity"], "implementation": implementation,
                    "box_width_cells": box_w_cells, "box_height_cells": box_h_cells,
                    "distance_m": float(source["gt_distance_m"]),
                    "distance_bin_m": _bin_label(float(source["gt_distance_m"]), (0, 10, 20, 30, 40.000001)),
                    "area_px": float(source["gt_bbox_area_px"]),
                    "area_bin_px": _bin_label(float(source["gt_bbox_area_px"]), (0, 400, 1600, 6400, 1e9)),
                    "raw_radius": raw_radius, "integer_radius": integer_radius,
                    **support,
                })
    # Population tests establish complete, paired coverage.
    for split in ("train", "val"):
        current_count = sum(row["split"] == split and row["implementation"] == "current" for row in rows)
        reference_count = sum(row["split"] == split and row["implementation"] == "reference" for row in rows)
        passed = current_count == reference_count and current_count > 0
        tests.append({"case": f"{split}_person_population", "passed": passed,
                      "reference_radius": "", "detail": f"paired rows={current_count}"})
        if not passed:
            raise RuntimeError(f"Gaussian population pairing failure: {split}")
    return rows, tests


def _unique_candidates(rows: Sequence[Mapping[str, Any]], category: str) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    # Prefer v010 and validation, then largest offset for informative panels.
    ordered = sorted(rows, key=lambda row: (
        row["category"] != category, row["contract"] != "v010", row["split"] != "val",
        -float(row["visible_centroid_offset_model_px"]), row["sample_id"], row["source_identity"]
    ))
    for row in ordered:
        if row["category"] != category:
            continue
        key = (str(row["sample_id"]), str(row["source_identity"]))
        if key in seen:
            continue
        seen.add(key); result.append(dict(row))
    return result


def select_review_panels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    roles = (("ON_CLOSER_OCCLUDER", 8), ("INSIDE_PROJECTED_BOX_BUT_NOT_VISIBLE", 8), ("ON_OWN_VISIBLE_MASK", 4))
    for category, count in roles:
        candidates = _unique_candidates(rows, category)
        if len(candidates) < count:
            raise RuntimeError(f"insufficient review panels for {category}: {len(candidates)} < {count}")
        for row in candidates:
            key = (row["sample_id"], row["source_identity"])
            if key in used:
                continue
            row["review_role"] = category
            selections.append(row); used.add(key)
            if sum(item["review_role"] == category for item in selections) == count:
                break
    largest = sorted(rows, key=lambda row: (-float(row["visible_centroid_offset_model_px"]), row["sample_id"], row["source_identity"]))
    for source in largest:
        key = (source["sample_id"], source["source_identity"])
        if key in used:
            continue
        row = dict(source); row["review_role"] = "LARGEST_VISIBLE_CENTROID_OFFSET"
        selections.append(row); used.add(key)
        if sum(item["review_role"] == "LARGEST_VISIBLE_CENTROID_OFFSET" for item in selections) == 4:
            break
    if len(selections) != 24:
        raise RuntimeError(f"review panel selection count {len(selections)} != 24")
    return selections


def _load_panel_mask(dataset_root: Path, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(dataset_root / "dataset" / row["rgb_path"]), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(dataset_root / "dataset" / row["depth_path"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        raise RuntimeError(f"panel input missing: {row['sample_id']}")
    episode = row["episode_id"]
    actor_id = str(row["source_identity"]).rsplit(":actor:", 1)[1]
    visibility_rows = read_csv(dataset_root / "dataset" / episode / "object_visibility.csv")
    matches = [item for item in visibility_rows if item["sample_id"] == row["sample_id"]
               and item["gt_actor_id"] == actor_id and item["label"] == "person"]
    if len(matches) != 1:
        raise RuntimeError(f"panel visibility lookup failure: {row['sample_id']}/{actor_id}")
    depth = decode_depth_bgra(depth_raw)
    mask = reconstruct_consistent_mask(depth, matches[0], width=rgb.shape[1], height=rgb.shape[0])
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), mask


def render_review_panels(dataset_root: Path, selections: Sequence[Mapping[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rendered: list[dict[str, Any]] = []
    for index, row in enumerate(selections, 1):
        rgb, mask = _load_panel_mask(dataset_root, row)
        x0, y0 = float(row["projected_bbox_x"]), float(row["projected_bbox_y"])
        x1, y1 = x0 + float(row["projected_bbox_w"]), y0 + float(row["projected_bbox_h"])
        pad_x, pad_y = max(30.0, 0.5 * (x1 - x0)), max(30.0, 0.5 * (y1 - y0))
        crop_x0, crop_y0 = max(0, int(math.floor(x0 - pad_x))), max(0, int(math.floor(y0 - pad_y)))
        crop_x1 = min(rgb.shape[1], int(math.ceil(x1 + pad_x)))
        crop_y1 = min(rgb.shape[0], int(math.ceil(y1 + pad_y)))
        crop = rgb[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
        overlay = crop.copy()
        overlay[crop_mask] = np.asarray([0, 255, 80], dtype=np.uint8)
        crop = np.where(crop_mask[:, :, None], (0.55 * crop + 0.45 * overlay).astype(np.uint8), crop)
        image = Image.fromarray(crop).resize((720, 540), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (720, 680), "white")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        scale_x, scale_y = 720 / max(1, crop_x1 - crop_x0), 540 / max(1, crop_y1 - crop_y0)
        def point(source_x: float, source_y: float) -> tuple[float, float]:
            return (source_x - crop_x0) * scale_x, (source_y - crop_y0) * scale_y
        bx0, by0 = point(x0, y0); bx1, by1 = point(x1, y1)
        draw.rectangle((bx0, by0, bx1, by1), outline=(255, 215, 0), width=4)
        tx, ty = point(float(row["target_source_x"]), float(row["target_source_y"]))
        cx, cy = point(float(row["visible_centroid_source_x"]), float(row["visible_centroid_source_y"]))
        nx, ny = point(float(row["nearest_visible_source_x"]), float(row["nearest_visible_source_y"]))
        draw.line((tx - 9, ty, tx + 9, ty), fill=(255, 0, 0), width=4)
        draw.line((tx, ty - 9, tx, ty + 9), fill=(255, 0, 0), width=4)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=(0, 120, 255), width=4)
        draw.polygon(((nx, ny - 8), (nx + 8, ny), (nx, ny + 8), (nx - 8, ny)), outline=(255, 0, 255))
        lines = [
            f"{index:02d} {row['review_role']} | observed={row['category']}",
            f"person | {row['sample_id']} | {row['source_identity']}",
            f"visible={float(row['visible_fraction']):.3f} | range={float(row['distance_m']):.2f} m | radar={bool(row['radar_supported'])}",
            f"nearest={float(row['nearest_visible_model_distance_px']):.2f} model px | centroid offset={float(row['visible_centroid_offset_model_px']):.2f} model px",
            "yellow=full box  red=target  green=visible mask  blue=centroid  magenta=nearest",
        ]
        draw.multiline_text((12, 550), "\n".join(lines), fill="black", font=ImageFont.load_default(), spacing=4)
        safe_role = str(row["review_role"]).lower()
        path = output_dir / f"panel_{index:02d}_{safe_role}.png"
        canvas.save(path)
        rendered.append({"index": index, "path": str(path), "review_role": row["review_role"],
                         "sample_id": row["sample_id"], "source_identity": row["source_identity"]})
    thumbs = [Image.open(item["path"]).convert("RGB").resize((360, 340), Image.Resampling.LANCZOS) for item in rendered]
    sheet = Image.new("RGB", (360 * 4, 340 * 6), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 360, (index // 4) * 340))
    sheet.save(output_dir / "CONTACT_SHEET.png")
    return rendered

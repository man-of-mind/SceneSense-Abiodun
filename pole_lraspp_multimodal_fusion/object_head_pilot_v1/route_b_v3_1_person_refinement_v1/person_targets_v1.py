#!/usr/bin/env python3
"""Person-only binned-range and projected-centre targets on the native grid."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_v1 import NATIVE_STRIDE  # noqa: E402
from targets_v1 import NativeGridDataset  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import valid_localization_objects  # noqa: E402


def derive_range_edges(object_rows: dict[str, list[dict[str, str]]],
                       train_sample_ids: set[str], quantiles: Sequence[float],
                       floor_m: float, ceiling_m: float) -> dict[str, Any]:
    depths = np.asarray([
        float(row["object_sensor_x"])
        for sample_id in sorted(train_sample_ids)
        for row in object_rows.get(sample_id, [])
        if row.get("label") == "person" and row.get("gt_source") == "actor"
        and row.get("object_sensor_x", "") != ""
        and 0.0 < float(row["object_sensor_x"]) <= float(ceiling_m)
    ], dtype=np.float64)
    if depths.size < 1000 or not np.isfinite(depths).all():
        raise RuntimeError("person train range-bin population is invalid")
    edges = np.quantile(depths, np.asarray(quantiles, dtype=np.float64))
    edges[0], edges[-1] = float(floor_m), float(ceiling_m)
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError(f"non-increasing train-derived range edges: {edges}")
    counts, _ = np.histogram(depths, bins=edges)
    return {
        "source": "v010_train_person_object_sensor_x_quantiles",
        "quantiles": [float(value) for value in quantiles],
        "edges_m": [float(value) for value in edges],
        "centers_m": [float((edges[i] + edges[i + 1]) / 2.0) for i in range(len(edges) - 1)],
        "half_widths_m": [float((edges[i + 1] - edges[i]) / 2.0) for i in range(len(edges) - 1)],
        "counts": [int(value) for value in counts], "population": int(depths.size),
        "minimum_m": float(depths.min()), "maximum_m": float(depths.max()),
    }


def add_person_targets(targets: dict[str, torch.Tensor], *,
                       objects: Sequence[dict[str, float]],
                       original_size: tuple[int, int], input_size: tuple[int, int],
                       intrinsic_full: np.ndarray, range_edges: Sequence[float],
                       offset_caps: Sequence[float]) -> dict[str, torch.Tensor]:
    input_w, input_h = (int(value) for value in input_size)
    source_w, source_h = (int(value) for value in original_size)
    grid_h, grid_w = targets["regression_mask"].shape[-2:]
    scale_x, scale_y = input_w / source_w, input_h / source_h
    intrinsic_model = np.asarray(intrinsic_full, dtype=np.float64).copy()
    intrinsic_model[0, :] *= scale_x
    intrinsic_model[1, :] *= scale_y
    edges = np.asarray(range_edges, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    half_widths = (edges[1:] - edges[:-1]) / 2.0

    mask = np.zeros((1, grid_h, grid_w), dtype=np.float32)
    bin_index = np.full((grid_h, grid_w), -1, dtype=np.int64)
    residual = np.zeros((1, grid_h, grid_w), dtype=np.float32)
    projected_offset = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    box_center_offset = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    local_xyz = np.zeros((3, grid_h, grid_w), dtype=np.float32)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    clipped_offsets = 0
    for obj in sorted(
        (value for value in objects if int(value["class_index"]) == 1),
        key=lambda item: float(item.get("area", 0.0)), reverse=True,
    ):
        grid_x = float(obj["center_x"]) * scale_x / NATIVE_STRIDE
        grid_y = float(obj["center_y"]) * scale_y / NATIVE_STRIDE
        cell_x, cell_y = int(np.floor(grid_x)), int(np.floor(grid_y))
        if not (0 <= cell_x < grid_w and 0 <= cell_y < grid_h) or occupied[cell_y, cell_x]:
            continue
        depth = float(obj["local_x"])
        if not math.isfinite(depth) or depth <= 0.0 or depth > float(edges[-1]):
            raise RuntimeError(f"invalid person range target {depth}")
        index = int(np.searchsorted(edges, depth, side="right") - 1)
        index = min(max(index, 0), len(centers) - 1)
        right, up = float(obj["local_y"]), float(obj["local_z"])
        projected_u = intrinsic_model[0, 2] + intrinsic_model[0, 0] * right / depth
        projected_v = intrinsic_model[1, 2] - intrinsic_model[1, 1] * up / depth
        raw_offset = np.asarray([
            projected_u / NATIVE_STRIDE - grid_x,
            projected_v / NATIVE_STRIDE - grid_y,
        ], dtype=np.float64)
        caps = np.asarray(offset_caps, dtype=np.float64)
        clipped = np.clip(raw_offset, -caps, caps)
        clipped_offsets += int(not np.array_equal(raw_offset, clipped))
        mask[0, cell_y, cell_x] = 1.0
        bin_index[cell_y, cell_x] = index
        residual[0, cell_y, cell_x] = np.clip(
            (depth - centers[index]) / max(1e-6, half_widths[index]), -1.0, 1.0,
        )
        projected_offset[:, cell_y, cell_x] = clipped.astype(np.float32)
        box_center_offset[:, cell_y, cell_x] = (grid_x - cell_x, grid_y - cell_y)
        local_xyz[:, cell_y, cell_x] = (depth, right, up)
        occupied[cell_y, cell_x] = True
    targets.update({
        "person_regression_mask": torch.from_numpy(mask),
        "person_range_bin": torch.from_numpy(bin_index),
        "person_range_residual": torch.from_numpy(residual),
        "person_projected_center_offset": torch.from_numpy(projected_offset),
        "person_box_center_offset": torch.from_numpy(box_center_offset),
        "person_local_xyz": torch.from_numpy(local_xyz),
        "camera_intrinsic_model": torch.from_numpy(intrinsic_model.astype(np.float32)),
        "person_offset_clipped_count": torch.tensor(clipped_offsets, dtype=torch.long),
    })
    return targets


class PersonRefinementDataset(NativeGridDataset):
    def __init__(self, *args: Any, range_edges: Sequence[float],
                 offset_caps: Sequence[float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.range_edges = tuple(float(value) for value in range_edges)
        self.offset_caps = tuple(float(value) for value in offset_caps)

    def __getitem__(self, index: int):
        fused, segmentation, targets = super().__getitem__(index)
        row = self.rows[index]
        source_w, source_h = int(row["camera_width"]), int(row["camera_height"])
        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []), image_width=source_w,
            image_height=source_h,
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 12.0)),
            object_class_names=self.object_class_names,
            max_distance_m=float(self.object_cfg.get("max_gt_distance_m", 40.0)),
        )
        intrinsic = np.asarray([
            [float(row["camera_fx"]), 0.0, float(row["camera_cx"])],
            [0.0, float(row["camera_fy"]), float(row["camera_cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return fused, segmentation, add_person_targets(
            targets, objects=objects, original_size=(source_w, source_h),
            input_size=(self.input_width, self.input_height), intrinsic_full=intrinsic,
            range_edges=self.range_edges, offset_caps=self.offset_caps,
        )


def build_sampling_weights(train_rows: Sequence[dict[str, str]],
                           validation_rows: Sequence[dict[str, str]],
                           object_rows: dict[str, list[dict[str, str]]],
                           sampling: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    train_ids = {row["sample_id"] for row in train_rows}
    validation_ids = {row["sample_id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation sample IDs overlap")
    episode_counts = Counter(row["experiment_id"] for row in train_rows)
    track_counts: Counter[tuple[str, str]] = Counter()
    per_sample_person: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        persons = [value for value in object_rows.get(row["sample_id"], [])
                   if value.get("label") == "person" and value.get("gt_source") == "actor"]
        per_sample_person[row["sample_id"]] = persons
        for person in persons:
            track_counts[(row["experiment_id"], person["gt_actor_id"])] += 1
    inverse_values = [1.0 / count for count in track_counts.values()]
    inverse_mean = float(np.mean(inverse_values)) if inverse_values else 1.0
    raw: list[float] = []
    components: list[dict[str, float]] = []
    episode_number = len(episode_counts)
    for row in train_rows:
        episode_factor = len(train_rows) / (
            max(1, episode_number) * episode_counts[row["experiment_id"]]
        )
        persons = per_sample_person[row["sample_id"]]
        if persons:
            track_factor = float(np.mean([
                1.0 / track_counts[(row["experiment_id"], person["gt_actor_id"])]
                for person in persons
            ])) / max(1e-12, inverse_mean)
            small_far = any(
                float(person["gt_bbox_area_px"]) < float(sampling["small_area_px"])
                or float(person["gt_distance_m"]) >= float(sampling["far_distance_m"])
                for person in persons
            )
            exposure = float(sampling["small_far_multiplier"]) if small_far else 1.0
            negative = 1.0
        else:
            track_factor, exposure = 1.0, 1.0
            negative = float(sampling["negative_frame_weight"])
        value = episode_factor * track_factor * exposure * negative
        raw.append(value)
        components.append({
            "episode": episode_factor, "track": track_factor,
            "small_far": exposure, "negative": negative,
        })
    raw_array = np.asarray(raw, dtype=np.float64)
    median = float(np.median(raw_array))
    cap = median * float(sampling["inverse_weight_cap_ratio"])
    weights = np.minimum(raw_array, cap)
    weights /= weights.mean()
    report = {
        "train_samples": len(train_rows), "validation_samples_excluded": len(validation_rows),
        "train_validation_overlap": 0, "episodes": len(episode_counts),
        "person_actor_tracks": len(track_counts), "person_positive_frames": sum(bool(v) for v in per_sample_person.values()),
        "raw_min": float(raw_array.min()), "raw_median": median,
        "raw_max": float(raw_array.max()), "cap": cap,
        "capped_samples": int((raw_array > cap).sum()),
        "normalized_min": float(weights.min()), "normalized_max": float(weights.max()),
        "validation_rows_used_by_sampler_or_mining": 0,
        "deterministic_weight_formula": "episode_inverse*mean_track_inverse*small_far*negative_frame",
    }
    return torch.as_tensor(weights, dtype=torch.double), report

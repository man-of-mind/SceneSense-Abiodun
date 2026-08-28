#!/usr/bin/env python3
"""Positive-depth factorized targets layered onto the frozen native-grid targets."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PKG), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from targets_v1 import NativeGridDataset, build_native_object_targets  # noqa: E402
from model_v1 import NATIVE_STRIDE  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES, valid_localization_objects,
)


def add_factorized_targets(
    targets: Dict[str, torch.Tensor], *, objects: Sequence[Dict[str, float]],
    original_size: Tuple[int, int], input_size: Tuple[int, int],
    intrinsic_full: np.ndarray, stride: int = NATIVE_STRIDE,
) -> Dict[str, torch.Tensor]:
    input_w, input_h = (int(value) for value in input_size)
    source_w, source_h = (int(value) for value in original_size)
    grid_h, grid_w = targets["regression_mask"].shape[-2:]
    scale_x, scale_y = input_w / source_w, input_h / source_h
    intrinsic_model = np.asarray(intrinsic_full, dtype=np.float64).copy()
    intrinsic_model[0, :] *= scale_x
    intrinsic_model[1, :] *= scale_y

    log_depth = np.zeros((1, grid_h, grid_w), dtype=np.float32)
    projected_offset = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    local_xy = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    class_index = np.full((1, grid_h, grid_w), -1, dtype=np.int64)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for obj in sorted(objects, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        center_x_model = float(obj["center_x"]) * scale_x
        center_y_model = float(obj["center_y"]) * scale_y
        grid_x, grid_y = center_x_model / stride, center_y_model / stride
        cell_x, cell_y = int(np.floor(grid_x)), int(np.floor(grid_y))
        if cell_x < 0 or cell_y < 0 or cell_x >= grid_w or cell_y >= grid_h:
            continue
        if occupied[cell_y, cell_x]:
            continue
        depth = float(obj["local_x"])
        if not math.isfinite(depth) or depth <= 0.0:
            raise RuntimeError(f"non-positive localization target depth: {depth}")
        right, up = float(obj["local_y"]), float(obj["local_z"])
        projected_u = intrinsic_model[0, 2] + intrinsic_model[0, 0] * right / depth
        projected_v = intrinsic_model[1, 2] - intrinsic_model[1, 1] * up / depth
        log_depth[0, cell_y, cell_x] = math.log(depth)
        projected_offset[0, cell_y, cell_x] = projected_u / stride - grid_x
        projected_offset[1, cell_y, cell_x] = projected_v / stride - grid_y
        local_xy[:, cell_y, cell_x] = (depth, right)
        class_index[0, cell_y, cell_x] = int(obj["class_index"])
        occupied[cell_y, cell_x] = True
    expected = targets["regression_mask"].numpy()[0] > 0.5
    if not np.array_equal(occupied, expected):
        raise RuntimeError(
            f"factorized/native positive-cell mismatch: factorized={occupied.sum()} native={expected.sum()}"
        )
    targets["factorized_log_depth"] = torch.from_numpy(log_depth)
    targets["projected_3d_center_offset"] = torch.from_numpy(projected_offset)
    targets["factorized_local_xy"] = torch.from_numpy(local_xy)
    targets["factorized_class_index"] = torch.from_numpy(class_index)
    targets["camera_intrinsic_model"] = torch.from_numpy(intrinsic_model.astype(np.float32))
    return targets


class FactorizedLocalizationDataset(NativeGridDataset):
    """Native v3.1 q=0 inputs/targets plus localization-only factor targets."""

    def __getitem__(self, index: int):
        fused, segmentation, targets = super().__getitem__(index)
        row = self.rows[index]
        source_w, source_h = int(row["camera_width"]), int(row["camera_height"])
        objects = valid_localization_objects(
            self.object_rows.get(row["sample_id"], []),
            image_width=source_w, image_height=source_h,
            min_area_px=float(self.object_cfg.get("min_gt_area_px", 12.0)),
            object_class_names=self.object_class_names,
            max_distance_m=float(self.object_cfg.get("max_gt_distance_m", 40.0)),
        )
        intrinsic = np.asarray([
            [float(row["camera_fx"]), 0.0, float(row["camera_cx"])],
            [0.0, float(row["camera_fy"]), float(row["camera_cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        targets = add_factorized_targets(
            targets, objects=objects, original_size=(source_w, source_h),
            input_size=(self.input_width, self.input_height), intrinsic_full=intrinsic,
        )
        return fused, segmentation, targets


def synthetic_projection_case() -> Dict[str, Any]:
    obj = {
        "class_index": 1.0, "class_name": "person", "center_x": 700.0, "center_y": 350.0,
        "bbox_w": 20.0, "bbox_h": 40.0, "area": 800.0,
        "local_x": 20.0, "local_y": 2.0, "local_z": 1.0,
        "world_x": 0.0, "world_y": 0.0, "world_z": 0.0,
        "size_x": 0.4, "size_y": 0.4, "size_z": 1.8,
        "yaw_sin": 0.0, "yaw_cos": 1.0, "parked": 0.0, "radar_support": 1.0,
    }
    base = build_native_object_targets(
        objects=[obj], original_size=(1280, 720), input_size=(768, 432), max_objects=64,
        object_class_names=OBJECT_CLASS_NAMES,
    )
    intrinsic = np.asarray([[369.5041722813606, 0.0, 640.0],
                            [0.0, 369.5041722813606, 360.0], [0.0, 0.0, 1.0]])
    targets = add_factorized_targets(
        base, objects=[obj], original_size=(1280, 720), input_size=(768, 432),
        intrinsic_full=intrinsic,
    )
    cell = torch.nonzero(targets["regression_mask"][0] > 0.5, as_tuple=False)[0]
    y, x = int(cell[0]), int(cell[1])
    box_offset = targets["center_offset"][:, y, x].numpy()
    projected_offset = targets["projected_3d_center_offset"][:, y, x].numpy()
    k = targets["camera_intrinsic_model"].numpy()
    depth = float(np.exp(targets["factorized_log_depth"][0, y, x].item()))
    u = (x + float(box_offset[0]) + float(projected_offset[0])) * NATIVE_STRIDE
    v = (y + float(box_offset[1]) + float(projected_offset[1])) * NATIVE_STRIDE
    reconstructed = np.asarray([depth, (u - k[0, 2]) * depth / k[0, 0],
                                -(v - k[1, 2]) * depth / k[1, 1]])
    error = float(np.linalg.norm(reconstructed - np.asarray([20.0, 2.0, 1.0])))
    return {
        "cell_xy": [x, y], "depth_m": depth,
        "projected_offset_grid": projected_offset.tolist(),
        "reconstructed_forward_right_up_m": reconstructed.tolist(),
        "roundtrip_3d_error_m": error, "finite": bool(np.isfinite(reconstructed).all()),
        "pass": bool(np.isfinite(reconstructed).all() and error <= 1e-5),
    }

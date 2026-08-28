#!/usr/bin/env python3
"""Native-grid decoder that changes only XYZ via depth and camera unprojection."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PKG), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from decode_v1 import TOPK_PER_CLASS, native_local_maxima  # noqa: E402
from model_v1 import HEATMAP_CHANNELS, MODEL_SIZE, NATIVE_STRIDE, SL_OFFSET, SL_REG  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES, REG_BBOX_WH, REG_DIMS, REG_LOCAL_XYZ, REG_PARKED,
    REG_RADAR_SUPPORT, REG_YAW, transform_point,
)


def decode_factorized_objects(
    object_output: torch.Tensor, localization_output: torch.Tensor, *,
    camera_matrix: np.ndarray, intrinsic_model: np.ndarray, score_threshold: float,
    topk: int = TOPK_PER_CLASS, stride: int = NATIVE_STRIDE,
    model_size: Sequence[int] = MODEL_SIZE,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
) -> List[Dict[str, float]]:
    if object_output.ndim == 4:
        object_output = object_output[0]
    if localization_output.ndim == 4:
        localization_output = localization_output[0]
    heat = torch.sigmoid(object_output[:HEATMAP_CHANNELS].float())
    peaks = native_local_maxima(heat.unsqueeze(0))[0].detach().cpu()
    regs = object_output[SL_REG].float().detach().cpu().numpy()
    center_offsets = object_output[SL_OFFSET].float().clamp(0.0, 1.0).detach().cpu().numpy()
    factorized = localization_output.float().detach().cpu().numpy()
    raw = object_output.float().detach().cpu()
    model_w, model_h = int(model_size[0]), int(model_size[1])
    grid_h, grid_w = int(heat.shape[1]), int(heat.shape[2])
    predictions: List[Dict[str, float]] = []

    for class_index in range(HEATMAP_CHANNELS):
        flat = peaks[class_index].reshape(-1)
        scores, indices = torch.topk(flat, k=min(int(topk), int(flat.numel())))
        for score_tensor, index_tensor in zip(scores, indices):
            score = float(score_tensor.item())
            if score < float(score_threshold):
                break
            index = int(index_tensor.item())
            cell_y, cell_x = divmod(index, grid_w)
            box_center_x = (cell_x + float(center_offsets[0, cell_y, cell_x])) * stride
            box_center_y = (cell_y + float(center_offsets[1, cell_y, cell_x])) * stride
            projected_x = box_center_x + float(factorized[1, cell_y, cell_x]) * stride
            projected_y = box_center_y + float(factorized[2, cell_y, cell_x]) * stride
            depth = math.exp(float(factorized[0, cell_y, cell_x]))
            local = np.asarray([
                depth,
                (projected_x - intrinsic_model[0, 2]) * depth / intrinsic_model[0, 0],
                -(projected_y - intrinsic_model[1, 2]) * depth / intrinsic_model[1, 1],
            ], dtype=np.float64)
            world = transform_point(camera_matrix, local)
            legacy_local = regs[REG_LOCAL_XYZ, cell_y, cell_x]
            legacy_world = transform_point(camera_matrix, legacy_local)
            dims = np.maximum(regs[REG_DIMS, cell_y, cell_x], 0.0)
            yaw_sin, yaw_cos = regs[REG_YAW, cell_y, cell_x]
            yaw_norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))

            def softplus(value: float) -> float:
                return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))

            box_w = softplus(float(regs[REG_BBOX_WH.start, cell_y, cell_x])) * model_w
            box_h = softplus(float(regs[REG_BBOX_WH.start + 1, cell_y, cell_x])) * model_h
            predictions.append({
                "class_index": float(class_index),
                "class_name": (str(object_class_names[class_index])
                               if class_index < len(object_class_names) else f"object_{class_index}"),
                "score": score,
                "center_x_px": box_center_x, "center_y_px": box_center_y,
                "bbox_w_px": box_w, "bbox_h_px": box_h,
                "bbox_x0": box_center_x - box_w / 2.0,
                "bbox_y0": box_center_y - box_h / 2.0,
                "bbox_x1": box_center_x + box_w / 2.0,
                "bbox_y1": box_center_y + box_h / 2.0,
                "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
                "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
                "size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2]),
                "yaw_sin": float(yaw_sin / yaw_norm), "yaw_cos": float(yaw_cos / yaw_norm),
                "parked_score": float(torch.sigmoid(
                    raw[HEATMAP_CHANNELS + REG_PARKED, cell_y, cell_x]).item()),
                "radar_support_score": float(torch.sigmoid(
                    raw[HEATMAP_CHANNELS + REG_RADAR_SUPPORT, cell_y, cell_x]).item()),
                "projected_3d_center_x_px": projected_x,
                "projected_3d_center_y_px": projected_y,
                "predicted_depth_m": depth,
                "legacy_local_x": float(legacy_local[0]),
                "legacy_local_y": float(legacy_local[1]),
                "legacy_local_z": float(legacy_local[2]),
                "legacy_world_x": float(legacy_world[0]),
                "legacy_world_y": float(legacy_world[1]),
                "legacy_world_z": float(legacy_world[2]),
            })
    predictions.sort(key=lambda item: (-float(item["score"]), str(item["class_name"])))
    return predictions

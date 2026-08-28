#!/usr/bin/env python3
"""Native stride-4 decoder.

Peak selection and every field read happen on the 192x108 grid. Nothing is bilinearly
resized for decoding. The external record format is byte-identical to the v3.1 one, so
the downstream spatial-map object schema is untouched: the offset channels are internal
and are consumed here, never exported.

Per class:
  1. sigmoid on the native heatmap
  2. native 3x3 local-maximum suppression
  3. fixed score threshold
  4. top-k 120 after local maxima
  5. image centre = (cell + offset) * stride
  6. XYZ / dimensions / yaw read from that one selected native cell
  7. merge vehicle and person records

No world-distance NMS. No threshold sweep. This pilot isolates the target/output
geometry correction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
BASE_PKG = FUSION_ROOT / "object_head_pilot_v1/route_b_v3_1_clean_base_v1"
for _path in (str(PACKAGE_ROOT), str(BASE_PKG), str(FUSION_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402
    OBJECT_CLASS_NAMES,
    REG_DIMS,
    REG_LOCAL_XYZ,
    REG_PARKED,
    REG_RADAR_SUPPORT,
    REG_YAW,
    REG_BBOX_WH,
    transform_point,
)

from model_v1 import HEATMAP_CHANNELS, MODEL_SIZE, NATIVE_STRIDE, SL_OFFSET, SL_REG  # noqa: E402

TOPK_PER_CLASS = 120
LOCAL_MAXIMUM_KERNEL = 3


def native_local_maxima(scores: torch.Tensor, kernel: int = LOCAL_MAXIMUM_KERNEL) -> torch.Tensor:
    """Keep only cells that equal their kernel x kernel neighbourhood maximum."""
    pooled = F.max_pool2d(scores, kernel_size=kernel, stride=1, padding=kernel // 2)
    return scores * (pooled - scores).abs().le(1e-12).to(scores.dtype)


def decode_native_objects(
    object_output: torch.Tensor,
    *,
    camera_matrix: np.ndarray,
    score_threshold: float,
    topk: int = TOPK_PER_CLASS,
    stride: int = NATIVE_STRIDE,
    model_size: Sequence[int] = MODEL_SIZE,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
) -> List[Dict[str, float]]:
    if object_output.ndim == 4:
        object_output = object_output[0]
    heat = torch.sigmoid(object_output[:HEATMAP_CHANNELS].float())
    peaks = native_local_maxima(heat.unsqueeze(0))[0].detach().cpu()
    regs = object_output[SL_REG].float().detach().cpu().numpy()
    # The offset is trained as a fraction of one cell; clamping keeps a recovered
    # centre inside the cell that actually produced it, which is what the neutrality
    # and 2D-support lookups assume. It cannot move a centre onto a better cell.
    offsets = object_output[SL_OFFSET].float().clamp(0.0, 1.0).detach().cpu().numpy()
    raw = object_output.float().detach().cpu()

    model_width, model_height = int(model_size[0]), int(model_size[1])
    grid_h, grid_w = int(heat.shape[1]), int(heat.shape[2])
    predictions: List[Dict[str, float]] = []

    for class_index in range(HEATMAP_CHANNELS):
        flat = peaks[class_index].reshape(-1)
        count = min(int(topk), int(flat.numel()))
        if count <= 0:
            continue
        scores, indices = torch.topk(flat, k=count)
        for score_tensor, index_tensor in zip(scores, indices):
            score = float(score_tensor.item())
            if score < float(score_threshold):
                break  # topk is sorted descending
            index = int(index_tensor.item())
            cell_y, cell_x = divmod(index, grid_w)
            if not (0 <= cell_x < grid_w and 0 <= cell_y < grid_h):
                continue

            center_x_px = (float(cell_x) + float(offsets[0, cell_y, cell_x])) * float(stride)
            center_y_px = (float(cell_y) + float(offsets[1, cell_y, cell_x])) * float(stride)
            local = regs[REG_LOCAL_XYZ, cell_y, cell_x]
            dims = np.maximum(regs[REG_DIMS, cell_y, cell_x], 0.0)
            yaw_sin, yaw_cos = regs[REG_YAW, cell_y, cell_x]
            norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
            world = transform_point(camera_matrix, local)

            # The 2D-box channels are input-image FRACTIONS, so they convert against the
            # full model canvas, not the native grid. This reproduces the v3.1 pixel
            # semantics exactly while the peak itself comes from the native cell.
            def _softplus(value: float) -> float:
                return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))

            box_w = _softplus(float(regs[REG_BBOX_WH.start, cell_y, cell_x])) * float(model_width)
            box_h = _softplus(float(regs[REG_BBOX_WH.start + 1, cell_y, cell_x])) * float(model_height)

            class_name = (str(object_class_names[class_index]) if class_index < len(object_class_names)
                          else f"object_{class_index}")
            predictions.append({
                "class_index": float(class_index),
                "class_name": class_name,
                "score": score,
                "center_x_px": center_x_px,
                "center_y_px": center_y_px,
                "bbox_w_px": box_w,
                "bbox_h_px": box_h,
                "bbox_x0": center_x_px - box_w / 2.0,
                "bbox_y0": center_y_px - box_h / 2.0,
                "bbox_x1": center_x_px + box_w / 2.0,
                "bbox_y1": center_y_px + box_h / 2.0,
                "local_x": float(local[0]), "local_y": float(local[1]), "local_z": float(local[2]),
                "world_x": float(world[0]), "world_y": float(world[1]), "world_z": float(world[2]),
                "size_x": float(dims[0]), "size_y": float(dims[1]), "size_z": float(dims[2]),
                "yaw_sin": float(yaw_sin / norm), "yaw_cos": float(yaw_cos / norm),
                "parked_score": float(torch.sigmoid(raw[HEATMAP_CHANNELS + REG_PARKED, cell_y, cell_x]).item()),
                "radar_support_score": float(
                    torch.sigmoid(raw[HEATMAP_CHANNELS + REG_RADAR_SUPPORT, cell_y, cell_x]).item()),
                "native_cell_x": float(cell_x),
                "native_cell_y": float(cell_y),
            })
    predictions.sort(key=lambda item: (-float(item["score"]), str(item["class_name"])))
    return predictions

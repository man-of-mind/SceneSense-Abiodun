#!/usr/bin/env python3
"""FROZEN CenterNet v2 native decoder.

Frozen before any v2 training result was observed.  The contract, in order:

  1. sigmoid the heatmap at its native resolution (no interpolation, ever);
  2. per-class 3x3 local-maximum suppression;
  3. score threshold;
  4. top-k **after** local maxima (k = 120, per native branch);
  5. add the branch's private predicted centre offset;
  6. convert the centre to image coordinates with that branch's stride:
         centre_img = (integer_cell + offset) * stride
     which is the exact inverse of the target construction;
  7. read XYZ / dimensions / yaw / parked / radar-support / box-WH directly at
     the native cell;
  8. merge the vehicle (stride 4) and person (stride 2) decoded lists;
  9. the evaluator retains class-aware 3 m matching, the 40 m range gate and the
     12 px GT-area rule.

Operating point 0.20; 0.02 is a permissive recall diagnostic only.  No
threshold or top-k tuning.  The decoded object record keeps the v1 field schema
(class, score, image centre, 2D box, XYZ, world XYZ, dimensions, yaw, parked
score, radar-support score); ``native_x`` / ``native_y`` / ``branch_stride`` are
additive diagnostics and are not part of the spatial-map record.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_targets import (
    REG_BBOX_WH,
    REG_DIMS,
    REG_LOCAL_XYZ,
    REG_PARKED,
    REG_RADAR_SUPPORT,
    REG_YAW,
    transform_point,
)

DECODER_NAME = "centernet_clean_v2_native_dual_stride_localmax_before_topk"
TOPK_PER_BRANCH = 120
PEAK_KERNEL = 3
BRANCH_SPEC: Tuple[Tuple[str, str, int, int], ...] = (
    # (output prefix, class name, class index, stride)
    ("veh", "vehicle", 0, 4),
    ("per", "person", 1, 2),
)


def local_maxima_mask(heat: torch.Tensor, kernel: int = PEAK_KERNEL) -> torch.Tensor:
    pooled = F.max_pool2d(heat, kernel_size=int(kernel), stride=1, padding=int(kernel) // 2)
    return (pooled - heat).abs() < 1e-12


def _softplus(value: float) -> float:
    return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))


def _decode_branch(
    hm: torch.Tensor,
    off: torch.Tensor,
    reg: torch.Tensor,
    *,
    class_name: str,
    class_index: int,
    stride: int,
    camera_matrix: np.ndarray,
    topk: int,
    score_threshold: float,
    input_size: Tuple[int, int],
) -> List[Dict[str, float]]:
    """hm (1,gh,gw) logits, off (2,gh,gw), reg (12,gh,gw) - all native resolution."""
    in_w, in_h = int(input_size[0]), int(input_size[1])
    heat = torch.sigmoid(hm.detach().float())
    keep = local_maxima_mask(heat.unsqueeze(0), kernel=PEAK_KERNEL)[0]
    above = heat >= float(score_threshold)
    masked = torch.where(keep & above, heat, torch.zeros_like(heat))
    gh, gw = int(heat.shape[1]), int(heat.shape[2])
    flat = masked.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    if k <= 0:
        return []
    scores, indices = torch.topk(flat, k=k)
    scores = scores.cpu()
    indices = indices.cpu()
    off_np = off.detach().float().cpu().numpy()
    reg_np = reg.detach().float().cpu().numpy()

    predictions: List[Dict[str, float]] = []
    for score_t, index_t in zip(scores, indices):
        score = float(score_t.item())
        if score < float(score_threshold):
            continue
        idx = int(index_t.item())
        y, x = divmod(idx, gw)
        ox = float(off_np[0, y, x])
        oy = float(off_np[1, y, x])
        cx_img = (float(x) + ox) * float(stride)
        cy_img = (float(y) + oy) * float(stride)
        local = reg_np[REG_LOCAL_XYZ, y, x]
        dims = np.maximum(reg_np[REG_DIMS, y, x], 0.0)
        yaw_sin, yaw_cos = reg_np[REG_YAW, y, x]
        norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        world = transform_point(camera_matrix, local)
        bw = _softplus(float(reg_np[REG_BBOX_WH.start, y, x])) * float(in_w)
        bh = _softplus(float(reg_np[REG_BBOX_WH.start + 1, y, x])) * float(in_h)
        predictions.append(
            {
                "class_index": float(class_index),
                "class_name": str(class_name),
                "score": score,
                "center_x_px": cx_img,
                "center_y_px": cy_img,
                "bbox_w_px": bw,
                "bbox_h_px": bh,
                "bbox_x0": cx_img - bw / 2.0,
                "bbox_y0": cy_img - bh / 2.0,
                "bbox_x1": cx_img + bw / 2.0,
                "bbox_y1": cy_img + bh / 2.0,
                "local_x": float(local[0]),
                "local_y": float(local[1]),
                "local_z": float(local[2]),
                "world_x": float(world[0]),
                "world_y": float(world[1]),
                "world_z": float(world[2]),
                "size_x": float(dims[0]),
                "size_y": float(dims[1]),
                "size_z": float(dims[2]),
                "yaw_sin": float(yaw_sin / norm),
                "yaw_cos": float(yaw_cos / norm),
                "parked_score": float(1.0 / (1.0 + math.exp(-float(reg_np[REG_PARKED, y, x])))),
                "radar_support_score": float(
                    1.0 / (1.0 + math.exp(-float(reg_np[REG_RADAR_SUPPORT, y, x])))
                ),
                "native_x": int(x),
                "native_y": int(y),
                "branch_stride": int(stride),
            }
        )
    return predictions


def decode_objects_v2(
    outputs: Dict[str, torch.Tensor],
    *,
    camera_matrix: np.ndarray,
    input_size: Tuple[int, int],
    score_threshold: float,
    topk: int = TOPK_PER_BRANCH,
    sample_index: int = 0,
) -> List[Dict[str, float]]:
    """Decode one frame from a batched model output dict and merge both branches."""
    predictions: List[Dict[str, float]] = []
    for prefix, class_name, class_index, stride in BRANCH_SPEC:
        hm = outputs[f"{prefix}_hm"]
        off = outputs[f"{prefix}_off"]
        reg = outputs[f"{prefix}_reg"]
        if hm.ndim == 4:
            hm, off, reg = hm[sample_index], off[sample_index], reg[sample_index]
        predictions.extend(
            _decode_branch(
                hm,
                off,
                reg,
                class_name=class_name,
                class_index=class_index,
                stride=stride,
                camera_matrix=camera_matrix,
                topk=int(topk),
                score_threshold=float(score_threshold),
                input_size=input_size,
            )
        )
    predictions.sort(key=lambda item: -float(item["score"]))
    return predictions


def range_gate(
    predictions: Sequence[Dict[str, float]], camera_matrix: np.ndarray, max_distance_m: float
) -> List[Dict[str, float]]:
    cam_c = np.asarray(camera_matrix)[:3, 3]
    return [
        p
        for p in predictions
        if math.hypot(float(p["world_x"]) - cam_c[0], float(p["world_y"]) - cam_c[1])
        <= float(max_distance_m)
    ]

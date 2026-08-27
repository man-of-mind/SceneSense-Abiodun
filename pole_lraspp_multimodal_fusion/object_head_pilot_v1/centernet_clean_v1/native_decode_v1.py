#!/usr/bin/env python3
"""CenterNet-only native-grid decoder (v1) for the clean Route B CenterFusion model.

The legacy path in ``object_targets.decode_objects`` consumes object maps that
``CleanCenterFusionResNet34.decode_tail`` has already bilinearly enlarged from the
stride-4 grid (108x192) to the input resolution (432x768).  It then takes a *global*
top-k over the enlarged map and only afterwards applies a 2 px occupancy suppression.
Because a single stride-4 peak becomes a ~4x4 plateau of near-equal scores after 4x
bilinear enlargement, most of the top-k budget is spent on interpolated copies of a
handful of native peaks.

This module decodes on the native stride-4 grid instead:

  * per-class 3x3 local-maximum suppression **before** top-k;
  * global top-k **after** local maxima;
  * every regression channel (XYZ / dimensions / yaw / parked / radar-support / box WH)
    is read at the native cell and never interpolated;
  * only the image-space centre and the 2D box are converted to full resolution.

There is no centre-offset head in this checkpoint, so the image-space centre is the
exact geometric centre of the native cell under ``align_corners=False`` (stride*s+ (s-1)/2).
Nothing here is trained or added to the model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_targets import (
    OBJECT_CLASS_NAMES,
    OBJECT_REG_CHANNELS_BBOX,
    REG_BBOX_WH,
    REG_DIMS,
    REG_LOCAL_XYZ,
    REG_PARKED,
    REG_RADAR_SUPPORT,
    REG_YAW,
    object_reg_channels,
    transform_point,
)


def native_object_maps(model: torch.nn.Module, fused_tensor: torch.Tensor) -> torch.Tensor:
    """Pre-interpolation 14-channel object maps at the stride-4 grid.

    Mirrors ``decode_tail`` exactly up to (and excluding) the ``F.interpolate`` call, so
    ``F.interpolate(native, size=input_hw, mode='bilinear', align_corners=False)`` is
    bit-comparable with ``model(fused)['object']``.
    """
    rgb = fused_tensor[:, :3]
    radar = fused_tensor[:, 3 : 3 + int(model.radar_channels)]
    bundle = model.encode_front(rgb, radar)
    rgb_p2, radar_p2 = bundle["rgb_p2"], bundle["radar_p2"]
    primary = model.object_head(rgb_p2)
    refinement = model.refinement_head(rgb_p2, radar_p2, primary)
    return primary + refinement


def local_maxima_mask(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Per-class 3x3 local-maximum mask (standard CenterNet peak keep-mask)."""
    pooled = F.max_pool2d(heat, kernel_size=int(kernel), stride=1, padding=int(kernel) // 2)
    return (pooled - heat).abs() < 1e-12


def _softplus(value: float) -> float:
    # Numerically stable softplus, identical to the legacy decoder's encoding.
    return float(np.log1p(np.exp(-abs(value))) + max(value, 0.0))


def decode_objects_native(
    native_object_output: torch.Tensor,
    *,
    camera_matrix: np.ndarray,
    topk: int,
    score_threshold: float,
    input_size: Tuple[int, int],
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    predict_bbox2d: bool = False,
    peak_kernel: int = 3,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Native-grid decode.  Returns (predictions, diagnostics).

    ``input_size`` is (width, height) of the model input, used only to place the
    image-space centre and the 2D box in full-resolution pixels.
    """
    if native_object_output.ndim == 4:
        native_object_output = native_object_output[0]
    reg_count = object_reg_channels(predict_bbox2d)
    heatmap_channels = max(1, int(native_object_output.shape[0]) - reg_count)
    heat = torch.sigmoid(native_object_output[:heatmap_channels]).detach().float().cpu()
    regs = native_object_output[heatmap_channels:].detach().float().cpu().numpy()
    raw = native_object_output.detach().float().cpu()

    grid_h, grid_w = int(heat.shape[1]), int(heat.shape[2])
    in_w, in_h = int(input_size[0]), int(input_size[1])
    stride_x = float(in_w) / float(grid_w)
    stride_y = float(in_h) / float(grid_h)

    keep = local_maxima_mask(heat.unsqueeze(0), kernel=int(peak_kernel))[0]
    masked = torch.where(keep, heat, torch.zeros_like(heat))

    diagnostics: Dict[str, object] = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "stride_x": stride_x,
        "stride_y": stride_y,
    }
    for name, thr in (("020", 0.20), ("002", 0.02)):
        for cls in range(heatmap_channels):
            diagnostics[f"native_localmax_c{cls}_ge{name}"] = int(
                ((masked[cls] >= thr) & keep[cls]).sum().item()
            )

    flat = masked.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    if k <= 0:
        diagnostics["native_topk_saturated"] = 0
        return [], diagnostics
    scores, indices = torch.topk(flat, k=k)
    diagnostics["native_topk_saturated"] = int(float(scores[-1].item()) >= float(score_threshold))

    predictions: List[Dict[str, float]] = []
    for score_t, index_t in zip(scores, indices):
        score = float(score_t.item())
        if score < float(score_threshold):
            continue
        idx = int(index_t.item())
        class_index, rem = divmod(idx, grid_h * grid_w)
        y, x = divmod(rem, grid_w)
        # Regression is read at the native cell: never interpolated.
        local = regs[REG_LOCAL_XYZ, y, x]
        dims = np.maximum(regs[REG_DIMS, y, x], 0.0)
        yaw_sin, yaw_cos = regs[REG_YAW, y, x]
        norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        world = transform_point(camera_matrix, local)
        class_name = (
            str(object_class_names[class_index])
            if class_index < len(object_class_names)
            else f"object_{class_index}"
        )
        # Only the image-space centre is lifted to full resolution.  align_corners=False
        # maps native cell centre (x, y) to full-res ((x+0.5)*stride, (y+0.5)*stride).
        cx_full = (float(x) + 0.5) * stride_x
        cy_full = (float(y) + 0.5) * stride_y
        bbox2d: Dict[str, float] = {}
        if predict_bbox2d and regs.shape[0] >= OBJECT_REG_CHANNELS_BBOX:
            bw = _softplus(float(regs[REG_BBOX_WH.start, y, x])) * float(in_w)
            bh = _softplus(float(regs[REG_BBOX_WH.start + 1, y, x])) * float(in_h)
            bbox2d = {
                "bbox_w_px": bw,
                "bbox_h_px": bh,
                "bbox_x0": cx_full - bw / 2.0,
                "bbox_y0": cy_full - bh / 2.0,
                "bbox_x1": cx_full + bw / 2.0,
                "bbox_y1": cy_full + bh / 2.0,
            }
        predictions.append(
            {
                "class_index": float(class_index),
                "class_name": class_name,
                "score": score,
                "center_x_px": cx_full,
                "center_y_px": cy_full,
                "native_x": int(x),
                "native_y": int(y),
                **bbox2d,
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
                "parked_score": float(
                    torch.sigmoid(raw[heatmap_channels + REG_PARKED, y, x]).item()
                ),
                "radar_support_score": float(
                    torch.sigmoid(raw[heatmap_channels + REG_RADAR_SUPPORT, y, x]).item()
                ),
            }
        )
    return predictions, diagnostics


def max_cardinality_match(
    predictions: Sequence[Dict[str, float]],
    gt_objects: Sequence[Dict[str, float]],
    *,
    max_distance_m: float,
    class_aware: bool = True,
) -> List[Tuple[int, int, float]]:
    """Maximum-cardinality class-aware bipartite matching under the same distance gate.

    Ties on cardinality are broken toward the smallest total distance, so the result is
    never worse than greedy on either count or localization error.
    """
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = len(predictions), len(gt_objects)
    if n_pred == 0 or n_gt == 0:
        return []
    dist = np.full((n_pred, n_gt), np.inf, dtype=np.float64)
    for p_idx, pred in enumerate(predictions):
        for g_idx, gt in enumerate(gt_objects):
            if class_aware and str(pred.get("class_name", "")) != str(gt.get("class_name", "")):
                continue
            d = float(
                np.hypot(
                    float(pred["world_x"]) - float(gt["world_x"]),
                    float(pred["world_y"]) - float(gt["world_y"]),
                )
            )
            if d <= float(max_distance_m):
                dist[p_idx, g_idx] = d
    feasible = np.isfinite(dist)
    if not feasible.any():
        return []
    # Cardinality dominates: an edge costs (-BIG + distance), a non-edge costs 0.
    big = 1.0e6
    cost = np.where(feasible, dist - big, 0.0)
    rows, cols = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, float]] = []
    for r, c in zip(rows, cols):
        if feasible[r, c]:
            matches.append((int(r), int(c), float(dist[r, c])))
    return matches


def decode_objects_hybrid(
    native_object_output: torch.Tensor,
    full_object_output: torch.Tensor,
    *,
    camera_matrix: np.ndarray,
    topk: int,
    score_threshold: float,
    object_class_names: Sequence[str] = OBJECT_CLASS_NAMES,
    predict_bbox2d: bool = False,
    peak_kernel: int = 3,
) -> Tuple[List[Dict[str, float]], Dict[str, object]]:
    """Peak-selection-only correction: native local-max before top-k, legacy read location.

    This isolates the two things the legacy decoder conflates.  Peaks are selected on the
    native stride-4 grid with 3x3 local-maximum suppression *before* the global top-k, so
    one native peak can never consume more than one top-k slot and adjacent native peaks
    can never suppress each other.  But every value (score and all regression channels) is
    then read from the *interpolated* map at the arg-max full-resolution pixel inside that
    peak's own 4x4 block -- i.e. exactly where the legacy decoder would have read it.

    That matters because this checkpoint's regression was supervised at single full-resolution
    target pixels on the interpolated map, so the interpolated read location is the trained
    one and the native cell centre is not.  Nothing is interpolated that the deployed model
    did not already interpolate.
    """
    if native_object_output.ndim == 4:
        native_object_output = native_object_output[0]
    if full_object_output.ndim == 4:
        full_object_output = full_object_output[0]
    reg_count = object_reg_channels(predict_bbox2d)
    heatmap_channels = max(1, int(native_object_output.shape[0]) - reg_count)

    nat_heat = torch.sigmoid(native_object_output[:heatmap_channels]).detach().float().cpu()
    full_heat = torch.sigmoid(full_object_output[:heatmap_channels]).detach().float().cpu()
    full_raw = full_object_output.detach().float().cpu()
    full_regs = full_raw[heatmap_channels:].numpy()

    grid_h, grid_w = int(nat_heat.shape[1]), int(nat_heat.shape[2])
    f_h, f_w = int(full_heat.shape[1]), int(full_heat.shape[2])
    sy = f_h // grid_h
    sx = f_w // grid_w

    keep = local_maxima_mask(nat_heat.unsqueeze(0), kernel=int(peak_kernel))[0]
    masked = torch.where(keep, nat_heat, torch.zeros_like(nat_heat))
    flat = masked.reshape(-1)
    k = min(int(topk), int(flat.numel()))
    diagnostics: Dict[str, object] = {"hybrid_topk_saturated": 0}
    if k <= 0:
        return [], diagnostics
    scores, indices = torch.topk(flat, k=k)
    diagnostics["hybrid_topk_saturated"] = int(float(scores[-1].item()) >= float(score_threshold))

    predictions: List[Dict[str, float]] = []
    for score_t, index_t in zip(scores, indices):
        if float(score_t.item()) <= 0.0:
            continue
        idx = int(index_t.item())
        class_index, rem = divmod(idx, grid_h * grid_w)
        gy, gx = divmod(rem, grid_w)
        y0, y1 = gy * sy, min(f_h, (gy + 1) * sy)
        x0, x1 = gx * sx, min(f_w, (gx + 1) * sx)
        block = full_heat[class_index, y0:y1, x0:x1]
        off = int(torch.argmax(block.reshape(-1)).item())
        by, bx = divmod(off, int(block.shape[1]))
        y, x = y0 + by, x0 + bx
        score = float(full_heat[class_index, y, x].item())
        if score < float(score_threshold):
            continue
        local = full_regs[REG_LOCAL_XYZ, y, x]
        dims = np.maximum(full_regs[REG_DIMS, y, x], 0.0)
        yaw_sin, yaw_cos = full_regs[REG_YAW, y, x]
        norm = max(1e-6, float(np.hypot(yaw_sin, yaw_cos)))
        world = transform_point(camera_matrix, local)
        class_name = (
            str(object_class_names[class_index])
            if class_index < len(object_class_names)
            else f"object_{class_index}"
        )
        bbox2d: Dict[str, float] = {}
        if predict_bbox2d and full_regs.shape[0] >= OBJECT_REG_CHANNELS_BBOX:
            bw = _softplus(float(full_regs[REG_BBOX_WH.start, y, x])) * float(f_w)
            bh = _softplus(float(full_regs[REG_BBOX_WH.start + 1, y, x])) * float(f_h)
            bbox2d = {
                "bbox_w_px": bw,
                "bbox_h_px": bh,
                "bbox_x0": float(x) - bw / 2.0,
                "bbox_y0": float(y) - bh / 2.0,
                "bbox_x1": float(x) + bw / 2.0,
                "bbox_y1": float(y) + bh / 2.0,
            }
        predictions.append(
            {
                "class_index": float(class_index),
                "class_name": class_name,
                "score": score,
                "center_x_px": float(x),
                "center_y_px": float(y),
                "native_x": int(gx),
                "native_y": int(gy),
                **bbox2d,
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
                "parked_score": float(
                    torch.sigmoid(full_raw[heatmap_channels + REG_PARKED, y, x]).item()
                ),
                "radar_support_score": float(
                    torch.sigmoid(full_raw[heatmap_channels + REG_RADAR_SUPPORT, y, x]).item()
                ),
            }
        )
    return predictions, diagnostics

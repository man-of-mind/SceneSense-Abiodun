#!/usr/bin/env python3
"""Native-grid object loss and the unchanged v3.1 segmentation loss.

The object recipe is FIXED and registered in the resolved config. Only two things
differ from the frozen v3.1 object loss:

  1. every term is evaluated on the native 192x108 grid instead of an enlarged
     768x432 map;
  2. one added Smooth-L1 term on the private centre-offset channels at positive cells.

The heatmap term is the standard modified (CornerNet/CenterNet) focal loss, normalised
by the total number of valid object centres, with ignore cells (target == -1) excluded
from both the positive and the background sums. No person weighting, no class-balanced
flag. The XYZ / dimension / yaw / parked / radar-support / bbox2d terms and their
weights are carried over verbatim; none of them needed a unit conversion, because the
metric channels are raw metres and the 2D-box channels are input-image fractions - both
grid-independent. Only the DECODER converts units.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    REG_DIMS,
    REG_LOCAL_XYZ,
    REG_PARKED,
    REG_RADAR_SUPPORT,
    REG_YAW,
    REG_BBOX_WH,
)

from model_v1 import HEATMAP_CHANNELS, SL_OFFSET, SL_REG  # noqa: E402

DEFAULT_OBJECT_WEIGHTS = {
    "center": 4.0, "location": 1.5, "dimensions": 0.6, "yaw": 0.3,
    "parked": 0.2, "radar_support": 0.1, "bbox2d": 1.0, "offset": 1.0,
}


def focal_heatmap_loss_native(
    logits: torch.Tensor, target: torch.Tensor, *, alpha: float = 2.0, beta: float = 4.0
) -> Tuple[torch.Tensor, float]:
    """Modified focal loss with the exact v3.1 target == -1 ignore semantics."""
    valid = target.ge(0.0).to(logits.dtype)
    safe = target.clamp(min=0.0, max=1.0)
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = safe.ge(1.0 - 1e-3).to(logits.dtype) * valid
    neg = (1.0 - pos) * valid
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - safe, beta) * neg
    pos_count = pos.sum()
    denominator = pos_count.clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / denominator, float(pos_count.detach().item())


def native_object_loss(
    outputs: torch.Tensor, targets: Dict[str, torch.Tensor], weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    heatmap = targets["center_heatmap"].to(outputs.device)
    if int(heatmap.shape[1]) != HEATMAP_CHANNELS:
        raise ValueError(f"expected {HEATMAP_CHANNELS} heatmap channels, got {int(heatmap.shape[1])}")
    if tuple(outputs.shape[-2:]) != tuple(heatmap.shape[-2:]):
        raise ValueError(
            f"native grid mismatch: predictions {tuple(outputs.shape[-2:])} vs targets "
            f"{tuple(heatmap.shape[-2:])}. Native-grid training never resizes either side."
        )
    center_logits = outputs[:, :HEATMAP_CHANNELS]
    regs = outputs[:, SL_REG]
    offsets = outputs[:, SL_OFFSET]

    reg_target = targets["regression"].to(outputs.device)
    offset_target = targets["center_offset"].to(outputs.device)
    reg_mask = targets["regression_mask"].to(outputs.device)

    center_loss, positive_cells = focal_heatmap_loss_native(center_logits, heatmap)

    denom = reg_mask.sum().clamp(min=1.0)
    mask = reg_mask.expand_as(regs)
    loc_loss = F.smooth_l1_loss(regs[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ],
                                reg_target[:, REG_LOCAL_XYZ] * mask[:, REG_LOCAL_XYZ],
                                reduction="sum") / denom
    dim_loss = F.smooth_l1_loss(regs[:, REG_DIMS] * mask[:, REG_DIMS],
                                reg_target[:, REG_DIMS] * mask[:, REG_DIMS],
                                reduction="sum") / denom
    yaw_pred = F.normalize(regs[:, REG_YAW], dim=1)
    yaw_loss = F.smooth_l1_loss(yaw_pred * mask[:, REG_YAW],
                                reg_target[:, REG_YAW] * mask[:, REG_YAW],
                                reduction="sum") / denom
    parked_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_PARKED:REG_PARKED + 1], reg_target[:, REG_PARKED:REG_PARKED + 1],
        weight=reg_mask, reduction="sum") / denom
    radar_loss = F.binary_cross_entropy_with_logits(
        regs[:, REG_RADAR_SUPPORT:REG_RADAR_SUPPORT + 1], reg_target[:, REG_RADAR_SUPPORT:REG_RADAR_SUPPORT + 1],
        weight=reg_mask, reduction="sum") / denom

    eps = 1e-6
    pred_w = F.softplus(regs[:, REG_BBOX_WH.start:REG_BBOX_WH.start + 1])
    pred_h = F.softplus(regs[:, REG_BBOX_WH.start + 1:REG_BBOX_WH.start + 2])
    gt_w = reg_target[:, REG_BBOX_WH.start:REG_BBOX_WH.start + 1].clamp(min=0.0)
    gt_h = reg_target[:, REG_BBOX_WH.start + 1:REG_BBOX_WH.start + 2].clamp(min=0.0)
    inter = torch.min(pred_w, gt_w) * torch.min(pred_h, gt_h)
    union = pred_w * pred_h + gt_w * gt_h - inter + eps
    enclose = torch.max(pred_w, gt_w) * torch.max(pred_h, gt_h) + eps
    giou = inter / union - (enclose - union) / enclose
    bbox_loss = ((1.0 - giou) * reg_mask).sum() / denom

    # New: stride-quantization offset, supervised only at positive centre cells.
    offset_mask = reg_mask.expand_as(offsets)
    offset_loss = F.smooth_l1_loss(offsets * offset_mask, offset_target * offset_mask,
                                   reduction="sum") / denom

    total = (
        float(weights.get("center", 4.0)) * center_loss
        + float(weights.get("location", 1.5)) * loc_loss
        + float(weights.get("dimensions", 0.6)) * dim_loss
        + float(weights.get("yaw", 0.3)) * yaw_loss
        + float(weights.get("parked", 0.2)) * parked_loss
        + float(weights.get("radar_support", 0.1)) * radar_loss
        + float(weights.get("bbox2d", 1.0)) * bbox_loss
        + float(weights.get("offset", 1.0)) * offset_loss
    )
    parts = {
        "center_loss": float(center_loss.detach().item()),
        "loc_loss": float(loc_loss.detach().item()),
        "dim_loss": float(dim_loss.detach().item()),
        "yaw_loss": float(yaw_loss.detach().item()),
        "parked_loss": float(parked_loss.detach().item()),
        "radar_support_loss": float(radar_loss.detach().item()),
        "bbox2d_loss": float(bbox_loss.detach().item()),
        "offset_loss": float(offset_loss.detach().item()),
        "positive_cells": positive_cells,
    }
    return total, parts


def segmentation_loss(
    logits: torch.Tensor, masks: torch.Tensor, *,
    class_weights: Optional[torch.Tensor], lovasz_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """The v3.1 segmentation loss, unchanged: weighted CE + Lovasz-Softmax.

    Returns (loss, parts, logits_at_mask_resolution).

    Uses the same ignore-aware Lovasz implementation the clean-base runtime installs,
    so the segmentation objective is bit-for-bit the frozen one.
    """
    from runtime_v1 import lovasz_softmax_loss_v31

    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    ce_loss = F.cross_entropy(logits, masks, weight=class_weights)
    if float(lovasz_weight) > 0.0:
        lovasz_loss = lovasz_softmax_loss_v31(logits.float(), masks)
    else:
        lovasz_loss = logits.new_zeros(())
    total = ce_loss + float(lovasz_weight) * lovasz_loss
    # The upsampled logits are returned as well: the segmentation confusion matrix is
    # scored at MASK resolution, exactly as the frozen trainer does.
    return total, {
        "seg_loss": float(total.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "lovasz_loss": float(lovasz_loss.detach().item()),
    }, logits

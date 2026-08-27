#!/usr/bin/env python3
"""CenterNet v2 losses - one independent objective per native branch.

Each branch (vehicle @ stride 4, person @ stride 2) gets its own CornerNet focal
heatmap loss, its own private centre-offset L1, and its own 12-field regression
losses, all evaluated **only at that branch's native positive cells**.  Because
the branches are separate objectives, each already normalises by its own
positive count, which is what balances the two classes; the v1
``class_balanced_center`` flag is therefore unnecessary and unused.

Regression field weights are carried over unchanged from the v1 config
(center 4.0, location 1.5, dimensions 0.6, yaw 0.3, parked 0.2, radar_support
0.1, bbox2d 1.0); ``offset`` is the one new weight and is fixed at 1.0.
No loss sweep is performed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_targets import (
    REG_BBOX_WH,
    REG_DIMS,
    REG_LOCAL_XYZ,
    REG_PARKED,
    REG_RADAR_SUPPORT,
    REG_YAW,
    focal_heatmap_loss,
)
from pole_lraspp_multimodal_fusion.train_fusion import lovasz_softmax_loss

BRANCH_PREFIXES = ("veh", "per")
DEFAULT_OBJECT_WEIGHTS = {
    "center": 4.0,
    "offset": 1.0,
    "location": 1.5,
    "dimensions": 0.6,
    "yaw": 0.3,
    "parked": 0.2,
    "radar_support": 0.1,
    "bbox2d": 1.0,
}


def branch_object_loss(
    hm_logits: torch.Tensor,
    off_pred: torch.Tensor,
    reg_pred: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    prefix: str,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    hm_t = targets[f"{prefix}_hm"]
    off_t = targets[f"{prefix}_off"]
    reg_t = targets[f"{prefix}_reg"]
    mask = targets[f"{prefix}_mask"]

    center_loss = focal_heatmap_loss(hm_logits, hm_t)
    denom = mask.sum().clamp(min=1.0)

    off_loss = (F.l1_loss(off_pred * mask, off_t * mask, reduction="sum")) / denom

    m = mask.expand_as(reg_pred)
    loc_loss = F.smooth_l1_loss(
        reg_pred[:, REG_LOCAL_XYZ] * m[:, REG_LOCAL_XYZ],
        reg_t[:, REG_LOCAL_XYZ] * m[:, REG_LOCAL_XYZ],
        reduction="sum",
    ) / denom
    dim_loss = F.smooth_l1_loss(
        reg_pred[:, REG_DIMS] * m[:, REG_DIMS],
        reg_t[:, REG_DIMS] * m[:, REG_DIMS],
        reduction="sum",
    ) / denom
    yaw_pred = F.normalize(reg_pred[:, REG_YAW], dim=1)
    yaw_loss = F.smooth_l1_loss(
        yaw_pred * m[:, REG_YAW], reg_t[:, REG_YAW] * m[:, REG_YAW], reduction="sum"
    ) / denom
    parked_loss = F.binary_cross_entropy_with_logits(
        reg_pred[:, REG_PARKED : REG_PARKED + 1],
        reg_t[:, REG_PARKED : REG_PARKED + 1],
        weight=mask,
        reduction="sum",
    ) / denom
    radar_loss = F.binary_cross_entropy_with_logits(
        reg_pred[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        reg_t[:, REG_RADAR_SUPPORT : REG_RADAR_SUPPORT + 1],
        weight=mask,
        reduction="sum",
    ) / denom
    eps = 1e-6
    pw = F.softplus(reg_pred[:, REG_BBOX_WH.start : REG_BBOX_WH.start + 1])
    ph = F.softplus(reg_pred[:, REG_BBOX_WH.start + 1 : REG_BBOX_WH.start + 2])
    gw = reg_t[:, REG_BBOX_WH.start : REG_BBOX_WH.start + 1].clamp(min=0.0)
    ghh = reg_t[:, REG_BBOX_WH.start + 1 : REG_BBOX_WH.start + 2].clamp(min=0.0)
    inter = torch.min(pw, gw) * torch.min(ph, ghh)
    union = pw * ph + gw * ghh - inter + eps
    iou = inter / union
    enclose = torch.max(pw, gw) * torch.max(ph, ghh) + eps
    giou = iou - (enclose - union) / enclose
    bbox_loss = ((1.0 - giou) * mask).sum() / denom

    total = (
        float(weights.get("center", 4.0)) * center_loss
        + float(weights.get("offset", 1.0)) * off_loss
        + float(weights.get("location", 1.5)) * loc_loss
        + float(weights.get("dimensions", 0.6)) * dim_loss
        + float(weights.get("yaw", 0.3)) * yaw_loss
        + float(weights.get("parked", 0.2)) * parked_loss
        + float(weights.get("radar_support", 0.1)) * radar_loss
        + float(weights.get("bbox2d", 1.0)) * bbox_loss
    )
    parts = {
        f"{prefix}_center_loss": float(center_loss.detach().item()),
        f"{prefix}_offset_loss": float(off_loss.detach().item()),
        f"{prefix}_loc_loss": float(loc_loss.detach().item()),
        f"{prefix}_dim_loss": float(dim_loss.detach().item()),
        f"{prefix}_yaw_loss": float(yaw_loss.detach().item()),
        f"{prefix}_parked_loss": float(parked_loss.detach().item()),
        f"{prefix}_radar_support_loss": float(radar_loss.detach().item()),
        f"{prefix}_bbox2d_loss": float(bbox_loss.detach().item()),
        f"{prefix}_positives": float(mask.sum().detach().item()),
    }
    return total, parts


def compute_v2_losses(
    outputs: Dict[str, torch.Tensor],
    masks: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    *,
    object_weights: Dict[str, float],
    segmentation_weight: float,
    object_total_weight: float,
    class_weights: torch.Tensor | None,
    lovasz_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["out"]
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    ce_loss = F.cross_entropy(logits.float(), masks, weight=class_weights)
    lovasz = (
        lovasz_softmax_loss(logits.float(), masks)
        if float(lovasz_weight) > 0.0
        else logits.new_zeros(())
    )
    seg_loss = ce_loss + float(lovasz_weight) * lovasz

    object_total = seg_loss.new_zeros(())
    parts: Dict[str, float] = {}
    for prefix in BRANCH_PREFIXES:
        branch_loss, branch_parts = branch_object_loss(
            outputs[f"{prefix}_hm"].float(),
            outputs[f"{prefix}_off"].float(),
            outputs[f"{prefix}_reg"].float(),
            targets,
            prefix,
            object_weights,
        )
        object_total = object_total + branch_loss
        parts.update(branch_parts)
        parts[f"{prefix}_object_loss"] = float(branch_loss.detach().item())

    total = float(segmentation_weight) * seg_loss + float(object_total_weight) * object_total
    parts.update(
        {
            "seg_loss": float(seg_loss.detach().item()),
            "ce_loss": float(ce_loss.detach().item()),
            "lovasz_loss": float(lovasz.detach().item()),
            "object_loss": float(object_total.detach().item()),
        }
    )
    return total, parts

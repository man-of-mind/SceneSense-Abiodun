#!/usr/bin/env python3
"""Dimensionally normalized losses for the person-private visible-anchor branch."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

STRIDE = 4.0


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    valid = target.ge(0.0).to(logits.dtype)
    safe = target.clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = safe.ge(1.0 - 1e-3).to(logits.dtype) * valid
    negative = (1.0 - positive) * valid
    positive_loss = -torch.log(prediction) * (1.0 - prediction).pow(alpha) * positive
    negative_loss = (-torch.log(1.0 - prediction) * prediction.pow(alpha)
                     * (1.0 - safe).pow(beta) * negative)
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1.0)


def decode_bounded_depth(raw: torch.Tensor, bounds_m: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    low, high = (float(value) for value in bounds_m)
    normalized = torch.sigmoid(raw)
    log_depth = math.log(low) + normalized * (math.log(high) - math.log(low))
    return torch.exp(log_depth), normalized


def _masked_vector_loss(prediction: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # Prediction/target are [B,C,H,W], mask is [B,H,W].
    values_prediction = prediction.permute(0, 2, 3, 1)[mask]
    values_target = target.permute(0, 2, 3, 1)[mask]
    if values_prediction.numel() == 0:
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(values_prediction, values_target, beta=beta, reduction="mean")


def private_person_loss(outputs: Mapping[str, torch.Tensor],
                        targets: Mapping[str, torch.Tensor], *,
                        design: Mapping[str, Any],
                        offset_scales: Mapping[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["visible_heatmap"].device
    dtype = torch.float32
    values = {key: value.to(device=device, dtype=dtype) for key, value in targets.items()
              if isinstance(value, torch.Tensor) and value.is_floating_point()}
    positive = values["person_private_mask"][:, 0].gt(0.5)
    positive_count = int(positive.sum().item())

    heatmap = focal_heatmap_loss(outputs["visible_heatmap"].float(), values["visible_heatmap"])
    subcell_prediction = outputs["visible_subcell_offset"].float()
    subcell = _masked_vector_loss(
        subcell_prediction, values["visible_subcell_offset"], positive, beta=0.25,
    )
    box_offset = _masked_vector_loss(
        outputs["visible_to_box_center_offset"].float(),
        values["visible_to_box_center_offset"], positive, beta=0.25,
    )
    ray_offset = _masked_vector_loss(
        outputs["visible_to_physical_ray_offset"].float(),
        values["visible_to_physical_ray_offset"], positive, beta=0.25,
    )

    predicted_wh = F.softplus(outputs["full_box_wh"].float())
    target_wh = values["full_box_wh"].clamp(min=0.0)
    wh_smooth = _masked_vector_loss(predicted_wh, target_wh, positive, beta=0.05)
    eps = 1e-6
    pw, ph = predicted_wh[:, 0], predicted_wh[:, 1]
    tw, th = target_wh[:, 0], target_wh[:, 1]
    intersection = torch.minimum(pw, tw) * torch.minimum(ph, th)
    union = pw * ph + tw * th - intersection + eps
    enclosure = torch.maximum(pw, tw) * torch.maximum(ph, th) + eps
    giou = intersection / union - (enclosure - union) / enclosure
    wh_giou = (1.0 - giou)[positive].mean() if positive_count else predicted_wh.sum() * 0.0

    bounds = tuple(float(value) for value in design["depth_bounds_m"])
    depth, depth_normalized = decode_bounded_depth(
        outputs["positive_camera_forward_depth"].float()[:, 0], bounds,
    )
    target_depth_normalized = values["bounded_log_depth"][:, 0]
    depth_loss = (F.smooth_l1_loss(
        depth_normalized[positive], target_depth_normalized[positive], beta=0.05,
        reduction="mean",
    ) if positive_count else depth.sum() * 0.0)

    batch, _channels, height, width = outputs["visible_heatmap"].shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    visible_x = xx.unsqueeze(0) + subcell_prediction[:, 0].clamp(0.0, 1.0)
    visible_y = yy.unsqueeze(0) + subcell_prediction[:, 1].clamp(0.0, 1.0)
    ray_scale = float(offset_scales["physical_ray_grid_cells"])
    projected_u = (visible_x + outputs["visible_to_physical_ray_offset"].float()[:, 0]
                   * ray_scale) * STRIDE
    projected_v = (visible_y + outputs["visible_to_physical_ray_offset"].float()[:, 1]
                   * ray_scale) * STRIDE
    intrinsic = values["camera_intrinsic_model"]
    fx = intrinsic[:, 0, 0].view(batch, 1, 1)
    fy = intrinsic[:, 1, 1].view(batch, 1, 1)
    cx = intrinsic[:, 0, 2].view(batch, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch, 1, 1)
    endpoint_scale = float(design["endpoint_normalization_m"])
    predicted_local_xyz_normalized = torch.stack([
        depth,
        (projected_u - cx) * depth / fx,
        (cy - projected_v) * depth / fy,
    ], dim=1) / endpoint_scale
    endpoint = _masked_vector_loss(
        predicted_local_xyz_normalized, values["local_xyz_normalized"], positive, beta=0.05,
    )

    dimension_scale = float(design["dimension_normalization_m"])
    predicted_dimensions_normalized = outputs["person_dimensions"].float() / dimension_scale
    dimensions = _masked_vector_loss(
        predicted_dimensions_normalized, values["person_dimensions_normalized"], positive,
        beta=0.1,
    )
    predicted_yaw = F.normalize(outputs["person_yaw"].float(), dim=1, eps=1e-6)
    yaw = _masked_vector_loss(predicted_yaw, values["person_yaw"], positive, beta=0.25)
    radar_logits = outputs["radar_support"].float()[:, 0]
    radar = (F.binary_cross_entropy_with_logits(
        radar_logits[positive], values["radar_support"][:, 0][positive], reduction="mean",
    ) if positive_count else radar_logits.sum() * 0.0)

    unweighted = {
        "visible_heatmap": heatmap,
        "visible_subcell_offset": subcell,
        "visible_to_box_center_offset": box_offset,
        "full_box_wh_smooth_l1": wh_smooth,
        "full_box_wh_giou": wh_giou,
        "physical_ray_offset": ray_offset,
        "bounded_log_depth": depth_loss,
        "local_xyz_endpoint": endpoint,
        "person_dimensions": dimensions,
        "person_yaw": yaw,
        "radar_support": radar,
    }
    weights = design["loss_weights"]
    weighted = {name: float(weights[name]) * value for name, value in unweighted.items()}
    total = torch.stack(tuple(weighted.values())).sum()
    total_value = float(total.detach().item())
    parts: dict[str, float] = {
        "total_loss": total_value, "positive_cells": float(positive_count),
        "decoded_depth_min_m": float(depth[positive].min().detach().item()) if positive_count else 0.0,
        "decoded_depth_max_m": float(depth[positive].max().detach().item()) if positive_count else 0.0,
    }
    for name, value in unweighted.items():
        parts[f"unweighted_{name}"] = float(value.detach().item())
        parts[f"weighted_{name}"] = float(weighted[name].detach().item())
        parts[f"share_{name}"] = float(weighted[name].detach().item()) / max(1e-12, total_value)
    return total, parts

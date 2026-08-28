#!/usr/bin/env python3
"""Fixed class-macro factorized localization losses."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PKG = ROOT / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1"
if str(NATIVE_PKG) not in sys.path:
    sys.path.insert(0, str(NATIVE_PKG))

from model_v1 import NATIVE_STRIDE, SL_OFFSET  # noqa: E402


def factorized_localization_loss(
    localization: torch.Tensor, legacy_object: torch.Tensor,
    targets: Dict[str, torch.Tensor], weights: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predicted_log_depth = localization[:, 0]
    predicted_projected_offset = localization[:, 1:3]
    target_log_depth = targets["factorized_log_depth"].to(localization.device)[:, 0]
    target_projected_offset = targets["projected_3d_center_offset"].to(localization.device)
    target_local_xy = targets["factorized_local_xy"].to(localization.device)
    class_index = targets["factorized_class_index"].to(localization.device)[:, 0]
    positive = targets["regression_mask"].to(localization.device)[:, 0] > 0.5
    intrinsic = targets["camera_intrinsic_model"].to(localization.device, dtype=localization.dtype)
    legacy_offset = legacy_object[:, SL_OFFSET].detach().to(localization.dtype).clamp(0.0, 1.0)

    batch, _channels, height, width = localization.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=localization.device, dtype=localization.dtype),
        torch.arange(width, device=localization.device, dtype=localization.dtype),
        indexing="ij",
    )
    box_grid_x = xx.unsqueeze(0) + legacy_offset[:, 0]
    box_grid_y = yy.unsqueeze(0) + legacy_offset[:, 1]
    projected_u = (box_grid_x + predicted_projected_offset[:, 0]) * float(NATIVE_STRIDE)
    projected_v = (box_grid_y + predicted_projected_offset[:, 1]) * float(NATIVE_STRIDE)
    depth = torch.exp(predicted_log_depth)
    fx = intrinsic[:, 0, 0].view(batch, 1, 1)
    fy = intrinsic[:, 1, 1].view(batch, 1, 1)
    cx = intrinsic[:, 0, 2].view(batch, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch, 1, 1)
    predicted_local_xy = torch.stack([
        depth,
        (projected_u - cx) * depth / fx,
    ], dim=1)

    depth_terms = []
    offset_terms = []
    endpoint_terms = []
    counts: Dict[str, int] = {}
    for class_id, class_name in enumerate(("vehicle", "person")):
        mask = positive & class_index.eq(class_id)
        count = int(mask.sum().item())
        counts[class_name] = count
        if count == 0:
            continue
        depth_terms.append(F.smooth_l1_loss(
            predicted_log_depth[mask], target_log_depth[mask], beta=1.0, reduction="mean"
        ))
        predicted_offset_values = predicted_projected_offset.permute(0, 2, 3, 1)[mask]
        target_offset_values = target_projected_offset.permute(0, 2, 3, 1)[mask]
        offset_terms.append(F.smooth_l1_loss(
            predicted_offset_values, target_offset_values, beta=1.0, reduction="mean"
        ))
        predicted_endpoint_values = predicted_local_xy.permute(0, 2, 3, 1)[mask]
        target_endpoint_values = target_local_xy.permute(0, 2, 3, 1)[mask]
        endpoint_terms.append(F.smooth_l1_loss(
            predicted_endpoint_values, target_endpoint_values,
            beta=float(weights["local_xy_endpoint_smooth_l1_beta_m"]), reduction="mean",
        ))
    if not depth_terms:
        zero = localization.sum() * 0.0
        depth_loss = offset_loss = endpoint_loss = zero
    else:
        depth_loss = torch.stack(depth_terms).mean()
        offset_loss = torch.stack(offset_terms).mean()
        endpoint_loss = torch.stack(endpoint_terms).mean()
    total = (
        float(weights["log_depth_smooth_l1_weight"]) * depth_loss
        + float(weights["projected_center_offset_smooth_l1_weight"]) * offset_loss
        + float(weights["local_xy_endpoint_weight"]) * endpoint_loss
    )
    return total, {
        "total_loss": float(total.detach().item()),
        "log_depth_loss": float(depth_loss.detach().item()),
        "projected_center_offset_loss": float(offset_loss.detach().item()),
        "local_xy_endpoint_loss": float(endpoint_loss.detach().item()),
        "vehicle_positive_cells": counts.get("vehicle", 0),
        "person_positive_cells": counts.get("person", 0),
        "classes_present": len(depth_terms),
    }

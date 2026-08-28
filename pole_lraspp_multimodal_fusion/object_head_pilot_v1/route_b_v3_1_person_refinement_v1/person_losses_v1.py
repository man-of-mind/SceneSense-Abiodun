#!/usr/bin/env python3
"""Registered person-only refinement losses with bounded train-only hard negatives."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
for path in (str(NATIVE_PACKAGE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import model_v1 as native  # noqa: E402


def _bounded_center_loss(logits: torch.Tensor, target: torch.Tensor, *,
                         hard_negative_ratio: int, hard_negative_minimum: int) -> tuple[torch.Tensor, dict[str, float]]:
    valid = target.ge(0.0)
    safe = target.clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = safe.ge(1.0 - 1e-3) & valid
    negative = (~positive) & valid
    positive_values = -torch.log(prediction) * (1.0 - prediction).pow(2) * positive
    negative_values = (
        -torch.log(1.0 - prediction) * prediction.pow(2)
        * (1.0 - safe).pow(4) * negative
    )
    count = int(positive.sum().item())
    denominator = positive.sum().clamp(min=1).to(logits.dtype)
    flat_negative = negative_values[negative]
    selected_count = min(
        int(flat_negative.numel()),
        max(int(hard_negative_minimum), int(hard_negative_ratio) * max(1, count)),
    )
    selected = (
        torch.topk(flat_negative, selected_count, sorted=False).values.sum()
        if selected_count else logits.new_zeros(())
    )
    loss = (positive_values.sum() + selected) / denominator
    return loss, {
        "person_positive_cells": float(count),
        "center_hard_negatives_selected": float(selected_count),
        "center_valid_negatives": float(flat_negative.numel()),
    }


TensorObserver = Callable[[str, torch.Tensor], None]


def _observe(observer: TensorObserver | None, name: str, value: torch.Tensor) -> None:
    if observer is not None:
        observer(name, value)


def _mask_loss(logits: torch.Tensor, masks: torch.Tensor, *,
               hard_negative_ratio: int,
               tensor_observer: TensorObserver | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(logits.shape[-2:]) != tuple(masks.shape[-2:]):
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    _observe(tensor_observer, "loss.mask_interpolated_logits", logits)
    # The scored target is a filled projected-person box.  Compare the person
    # logit against the log-sum-exp of the two frozen alternatives.
    binary_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
    valid = masks.ne(-100)
    positive = masks.eq(2) & valid
    negative = masks.ne(2) & valid
    binary_target = positive.to(binary_logit.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(binary_logit, binary_target, reduction="none")
    positive_loss = per_pixel[positive].sum()
    negative_values = per_pixel[negative]
    positive_count = int(positive.sum().item())
    selected_count = min(int(negative_values.numel()), int(hard_negative_ratio) * max(1, positive_count))
    selected_negative = (
        torch.topk(negative_values, selected_count, sorted=False).values.sum()
        if selected_count else binary_logit.new_zeros(())
    )
    balanced_bce = (positive_loss + selected_negative) / positive.sum().clamp(min=1).to(binary_logit.dtype)
    probability = torch.sigmoid(binary_logit) * valid.to(binary_logit.dtype)
    target = binary_target * valid.to(binary_logit.dtype)
    intersection = (probability * target).sum()
    # Tversky weights false negatives more heavily without unbounded pixel weights.
    false_positive = (probability * (1.0 - target) * valid).sum()
    false_negative = ((1.0 - probability) * target).sum()
    tversky = (intersection + 1.0) / (intersection + 0.3 * false_positive + 0.7 * false_negative + 1.0)
    loss = 0.5 * balanced_bce + 0.5 * (1.0 - tversky)
    _observe(tensor_observer, "loss.person_mask", loss)
    return loss, {
        "person_mask_positive_pixels": float(positive_count),
        "mask_hard_negatives_selected": float(selected_count),
        "person_mask_tversky": float(tversky.detach().item()),
    }


def person_refinement_loss(outputs: dict[str, Any], masks: torch.Tensor,
                           targets: dict[str, torch.Tensor], *,
                           range_edges: Sequence[float], offset_caps: Sequence[float],
                           design: dict[str, Any],
                           tensor_observer: TensorObserver | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    refinement = outputs["person_refinement"]
    base_object = outputs["object"]
    target_heatmap = targets["center_heatmap"][:, 1:2].to(base_object.device)
    combined_objectness = base_object[:, 1:2] + refinement["objectness_residual"]
    _observe(tensor_observer, "loss.combined_objectness", combined_objectness)
    center, parts = _bounded_center_loss(
        combined_objectness, target_heatmap,
        hard_negative_ratio=int(design["hard_negative_ratio"]),
        hard_negative_minimum=int(design["hard_negative_minimum"]),
    )
    _observe(tensor_observer, "loss.center", center)
    positive = targets["person_regression_mask"].to(base_object.device)[:, 0].gt(0.5)
    indices = positive.nonzero(as_tuple=False)
    if indices.numel() == 0:
        zero = combined_objectness.sum() * 0.0
        total = center + zero
        for name in (
            "loss.range_bin", "loss.range_residual", "loss.projected_offset",
            "loss.local_xy_endpoint", "loss.quality", "loss.person_mask",
        ):
            _observe(tensor_observer, name, zero)
        _observe(tensor_observer, "loss.total", total)
        return total, {**parts, "range_bin_loss": 0.0, "range_residual_loss": 0.0,
                               "projected_offset_loss": 0.0, "local_xy_endpoint_loss": 0.0,
                               "quality_loss": 0.0, "person_mask_loss": 0.0}
    batch_index, cell_y, cell_x = indices[:, 0], indices[:, 1], indices[:, 2]
    bin_logits = refinement["range_bin_logits"][batch_index, :, cell_y, cell_x]
    _observe(tensor_observer, "loss.range_bin_logits_positive", bin_logits)
    bin_target = targets["person_range_bin"].to(base_object.device)[batch_index, cell_y, cell_x]
    range_bin_loss = F.cross_entropy(bin_logits, bin_target)
    _observe(tensor_observer, "loss.range_bin", range_bin_loss)
    predicted_residual = torch.tanh(
        refinement["range_residual"][batch_index, 0, cell_y, cell_x]
    )
    residual_target = targets["person_range_residual"].to(base_object.device)[batch_index, 0, cell_y, cell_x]
    range_residual_loss = F.smooth_l1_loss(predicted_residual, residual_target)
    _observe(tensor_observer, "loss.range_residual", range_residual_loss)
    caps = torch.as_tensor(offset_caps, dtype=torch.float32, device=base_object.device)
    predicted_offset = torch.tanh(
        refinement["projected_center_offset"][batch_index, :, cell_y, cell_x]
    ) * caps
    offset_target = targets["person_projected_center_offset"].to(base_object.device)[batch_index, :, cell_y, cell_x]
    projected_offset_loss = F.smooth_l1_loss(predicted_offset, offset_target)
    _observe(tensor_observer, "loss.projected_offset", projected_offset_loss)

    edges = torch.as_tensor(range_edges, dtype=torch.float32, device=base_object.device)
    centers = (edges[:-1] + edges[1:]) / 2.0
    half_widths = (edges[1:] - edges[:-1]) / 2.0
    probability = torch.softmax(bin_logits.float(), dim=1)
    candidate_ranges = centers.unsqueeze(0) + predicted_residual.unsqueeze(1) * half_widths.unsqueeze(0)
    predicted_depth = (probability * candidate_ranges).sum(dim=1).clamp(min=0.05, max=float(edges[-1]))
    _observe(tensor_observer, "loss.predicted_depth", predicted_depth)
    base_grid_offset = base_object[:, native.SL_OFFSET][batch_index, :, cell_y, cell_x].clamp(0.0, 1.0)
    u = (cell_x.to(torch.float32) + base_grid_offset[:, 0] + predicted_offset[:, 0]) * float(native.NATIVE_STRIDE)
    v = (cell_y.to(torch.float32) + base_grid_offset[:, 1] + predicted_offset[:, 1]) * float(native.NATIVE_STRIDE)
    intrinsic = targets["camera_intrinsic_model"].to(base_object.device)[batch_index]
    predicted_right = (u - intrinsic[:, 0, 2]) * predicted_depth / intrinsic[:, 0, 0]
    predicted_up = -(v - intrinsic[:, 1, 2]) * predicted_depth / intrinsic[:, 1, 1]
    _observe(tensor_observer, "loss.unprojection_u", u)
    _observe(tensor_observer, "loss.unprojection_v", v)
    _observe(tensor_observer, "loss.unprojection_right", predicted_right)
    _observe(tensor_observer, "loss.unprojection_up", predicted_up)
    local_target = targets["person_local_xyz"].to(base_object.device)[batch_index, :, cell_y, cell_x]
    local_xy_error = torch.sqrt(
        (predicted_depth - local_target[:, 0]).pow(2)
        + (predicted_right - local_target[:, 1]).pow(2) + 1e-8
    )
    local_xy_endpoint_loss = F.smooth_l1_loss(
        torch.stack([predicted_depth, predicted_right], dim=1), local_target[:, :2]
    )
    _observe(tensor_observer, "loss.local_xy_endpoint", local_xy_endpoint_loss)
    quality_target = (1.0 - local_xy_error.detach() / 3.0).clamp(0.0, 1.0)
    quality_logits = refinement["localization_quality"][batch_index, 0, cell_y, cell_x]
    quality_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_target)
    _observe(tensor_observer, "loss.quality", quality_loss)
    person_mask_loss, mask_parts = _mask_loss(
        outputs["out"], masks.to(base_object.device),
        hard_negative_ratio=int(design["mask_hard_negative_ratio"]),
        tensor_observer=tensor_observer,
    )
    weights = design["loss_weights"]
    total = (
        float(weights["center"]) * float(design["person_center_multiplier"]) * center
        + float(weights["quality"]) * quality_loss
        + float(weights["range_bin"]) * range_bin_loss
        + float(weights["range_residual"]) * range_residual_loss
        + float(weights["projected_offset"]) * projected_offset_loss
        + float(weights["local_xy_endpoint"]) * local_xy_endpoint_loss
        + float(weights["person_mask"]) * person_mask_loss
    )
    _observe(tensor_observer, "loss.total", total)
    details = {
        **parts, **mask_parts,
        "total_loss": float(total.detach().item()),
        "center_loss": float(center.detach().item()),
        "quality_loss": float(quality_loss.detach().item()),
        "range_bin_loss": float(range_bin_loss.detach().item()),
        "range_residual_loss": float(range_residual_loss.detach().item()),
        "projected_offset_loss": float(projected_offset_loss.detach().item()),
        "local_xy_endpoint_loss": float(local_xy_endpoint_loss.detach().item()),
        "person_mask_loss": float(person_mask_loss.detach().item()),
        "local_xy_error_mean_m": float(local_xy_error.detach().mean().item()),
        "quality_target_mean": float(quality_target.mean().item()),
        "person_positive_cells": float(indices.shape[0]),
        "clipped_offset_targets": float(targets["person_offset_clipped_count"].sum().item()),
    }
    return total, details

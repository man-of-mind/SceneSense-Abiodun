from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Mapping

import torch
import torch.nn.functional as F

CLASS_NAMES = ("vehicle", "person")


def _lovasz_grad(sorted_gt: torch.Tensor) -> torch.Tensor:
    positives = sorted_gt.sum()
    intersection = positives - sorted_gt.float().cumsum(0)
    union = positives + (1.0 - sorted_gt).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if jaccard.numel() > 1:
        # This spelling preserves the frozen equation while avoiding the overlapping
        # in-place view rejected by current PyTorch builds.
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    classes = probabilities.shape[1]
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
    labels = labels.reshape(-1)
    valid = labels.ge(0) & labels.lt(classes)
    probabilities, labels = probabilities[valid], labels[valid]
    if labels.numel() == 0:
        return logits.new_zeros(())
    losses = []
    for class_index in range(classes):
        foreground = (labels == class_index).to(probabilities.dtype)
        if foreground.sum() <= 0:
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        errors, permutation = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors, _lovasz_grad(foreground[permutation])))
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def segmentation_losses(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    weights = torch.tensor([0.5, 1.0, 4.0], device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, labels, weight=weights, ignore_index=-100)
    lovasz = lovasz_softmax(logits.float(), labels)
    combined = ce + 0.5 * lovasz
    return combined, {"segmentation_ce": ce, "segmentation_lovasz": lovasz, "segmentation": combined}, {
        "segmentation": int(labels.ne(-100).sum().item()),
    }


def focal_class(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    valid = target.ge(0.0)
    safe = target.clamp(0.0, 1.0)
    prediction = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = safe.ge(1.0 - 1e-3) & valid
    count = int(positive.sum().item())
    if count == 0:
        return None, 0
    negative = (~positive) & valid
    positive_loss = -torch.log(prediction) * (1.0 - prediction).pow(2.0) * positive
    negative_loss = -torch.log(1.0 - prediction) * prediction.pow(2.0) * (1.0 - safe).pow(4.0) * negative
    return (positive_loss.sum() + negative_loss.sum()) / count, count


def _gather(field: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    if cells.numel() == 0:
        return field.new_empty((0, field.shape[1]))
    return field[cells[:, 0], :, cells[:, 1], cells[:, 2]]


def _mean_available(values: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else reference.sum() * 0.0


def actor_object_losses(outputs: Mapping[str, Mapping[str, torch.Tensor]],
                        heatmap_target: torch.Tensor,
                        owners: Mapping[str, Mapping[str, torch.Tensor]],
                        depth_anchors: torch.Tensor, depth_delta: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    class_terms: dict[str, list[torch.Tensor]] = defaultdict_list()
    per_class: dict[str, torch.Tensor] = {}
    denominators: dict[str, int] = OrderedDict()
    reference = outputs["vehicle"]["heatmap"]
    for class_index, class_name in enumerate(CLASS_NAMES):
        branch = outputs[class_name]
        target = heatmap_target[:, class_index:class_index + 1]
        heat, heat_count = focal_class(branch["heatmap"], target)
        if heat is not None:
            class_terms["heatmap"].append(heat)
            per_class[f"heatmap_{class_name}"] = heat
        denominators[f"heatmap_{class_name}"] = heat_count
        target_fields = owners[class_name]
        cells = target_fields["cells"].to(branch["heatmap"].device)
        count = int(cells.shape[0])
        denominators[f"owners_{class_name}"] = count
        if count == 0:
            continue
        target_on_device = {name: value.to(branch["heatmap"].device, non_blocking=True)
                            for name, value in target_fields.items() if name != "cells"}
        subcell = torch.sigmoid(_gather(branch["subcell"], cells))
        class_terms["subcell"].append(F.smooth_l1_loss(subcell, target_on_device["subcell"], reduction="mean"))
        box_delta = _gather(branch["box_center_delta"], cells)
        class_terms["box_center_delta"].append(F.smooth_l1_loss(
            box_delta, target_on_device["box_center_delta"], reduction="mean"))
        box_wh = F.softplus(_gather(branch["box_wh"], cells))
        class_terms["box_wh"].append(F.smooth_l1_loss(box_wh, target_on_device["box_wh"], reduction="mean"))
        physical_delta = _gather(branch["physical_ray_delta"], cells)
        class_terms["physical_ray"].append(F.smooth_l1_loss(
            physical_delta, target_on_device["physical_ray_delta"], reduction="mean"))

        depth = target_on_device["depth"].squeeze(1)
        z_target = torch.log1p(depth)
        position = (z_target - depth_anchors[0]) / depth_delta
        lower = position.floor().long().clamp(0, len(depth_anchors) - 1)
        upper = (lower + 1).clamp(max=len(depth_anchors) - 1)
        upper_weight = (position - lower.float()).clamp(0.0, 1.0)
        upper_weight = torch.where(upper == lower, torch.zeros_like(upper_weight), upper_weight)
        lower_weight = 1.0 - upper_weight
        logits = _gather(branch["depth_bin_logits"], cells)
        log_probabilities = F.log_softmax(logits, dim=1)
        bin_loss = -(lower_weight * log_probabilities.gather(1, lower[:, None]).squeeze(1)
                     + upper_weight * log_probabilities.gather(1, upper[:, None]).squeeze(1)).mean()
        class_terms["depth_bin"].append(bin_loss)
        per_class[f"depth_bin_{class_name}"] = bin_loss
        residuals = _gather(branch["depth_bin_residuals"], cells)
        lower_target = (z_target - depth_anchors[lower]) / depth_delta
        upper_target = (z_target - depth_anchors[upper]) / depth_delta
        lower_loss = F.smooth_l1_loss(residuals.gather(1, lower[:, None]).squeeze(1), lower_target, reduction="none")
        upper_loss = F.smooth_l1_loss(residuals.gather(1, upper[:, None]).squeeze(1), upper_target, reduction="none")
        residual_loss = (lower_weight * lower_loss + upper_weight * upper_loss).mean()
        class_terms["depth_residual"].append(residual_loss)
        per_class[f"depth_residual_{class_name}"] = residual_loss

        probabilities = F.softmax(logits, dim=1)
        z_prediction = (probabilities * (depth_anchors[None, :] + depth_delta * residuals)).sum(dim=1)
        decoded_depth = torch.expm1(z_prediction).clamp_min(0.0)
        batch, cell_y, cell_x = cells[:, 0], cells[:, 1].float(), cells[:, 2].float()
        grid_anchor_x = cell_x + subcell[:, 0]
        grid_anchor_y = cell_y + subcell[:, 1]
        u_physical = 4.0 * (grid_anchor_x + physical_delta[:, 0])
        v_physical = 4.0 * (grid_anchor_y + physical_delta[:, 1])
        intrinsic = target_on_device["intrinsic"]
        local_prediction = torch.stack([
            decoded_depth,
            decoded_depth * (u_physical - intrinsic[:, 2]) / intrinsic[:, 0],
            decoded_depth * (intrinsic[:, 3] - v_physical) / intrinsic[:, 1],
        ], dim=1)
        class_terms["endpoint"].append(F.smooth_l1_loss(
            local_prediction / 3.0, target_on_device["local_xyz"] / 3.0, reduction="mean"))
        dimensions = torch.exp(_gather(branch["log_dimensions"], cells))
        class_terms["dimensions"].append(F.smooth_l1_loss(
            torch.log(dimensions.clamp_min(1e-6)), torch.log(target_on_device["dimensions"]), reduction="mean"))
        yaw = F.normalize(_gather(branch["yaw_sincos"], cells), dim=1, eps=1e-6)
        class_terms["yaw"].append(F.smooth_l1_loss(yaw, target_on_device["yaw"], reduction="mean"))
        radar = _gather(branch["radar_support"], cells)
        class_terms["radar_support"].append(F.binary_cross_entropy_with_logits(
            radar, target_on_device["radar_support"], reduction="mean"))
        if class_name == "vehicle":
            parked = _gather(branch["parked"], cells)
            class_terms["parked"].append(F.binary_cross_entropy_with_logits(
                parked, target_on_device["parked"], reduction="mean"))
    result = {name: _mean_available(values, reference) for name, values in class_terms.items()}
    result.update(per_class)
    for name in ("heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray", "depth_bin",
                 "depth_residual", "endpoint", "dimensions", "yaw", "parked", "radar_support"):
        result.setdefault(name, reference.sum() * 0.0)
    return result, denominators


def defaultdict_list() -> dict[str, list[torch.Tensor]]:
    return {name: [] for name in (
        "heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray", "depth_bin",
        "depth_residual", "endpoint", "dimensions", "yaw", "parked", "radar_support",
    )}


def dense_losses(prediction: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor,
                 radar_points: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    valid = valid.to(prediction.device, non_blocking=True)
    depth = depth.to(prediction.device, non_blocking=True)
    count = int(valid.sum().item())
    if count:
        dense = F.smooth_l1_loss(prediction[:, 0][valid], torch.log1p(depth[valid]), reduction="mean")
    else:
        dense = prediction.sum() * 0.0
    values = []
    point_count = 0
    for batch_index, points_cpu in enumerate(radar_points):
        if points_cpu.numel() == 0:
            continue
        points = points_cpu.to(prediction.device, non_blocking=True)
        grid = points[:, :2].view(1, 1, -1, 2)
        sampled = F.grid_sample(prediction[batch_index:batch_index + 1], grid,
                                mode="bilinear", padding_mode="zeros", align_corners=False).reshape(-1)
        values.append(F.smooth_l1_loss(sampled, points[:, 2], reduction="sum"))
        point_count += len(points)
    radar = torch.stack(values).sum() / point_count if point_count else prediction.sum() * 0.0
    return {"dense_depth": dense, "radar_consistency": radar}, {
        "dense_depth": count, "radar_consistency": point_count,
    }


def compute_losses(model: torch.nn.Module, batch: Mapping[str, Any], weights: Mapping[str, float]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], dict[str, Any]]:
    inputs = batch["input"].to(next(model.parameters()).device, non_blocking=True)
    labels = batch["segmentation"].to(inputs.device, non_blocking=True)
    outputs = model(inputs, dense=True)
    segmentation, segmentation_parts, segmentation_denoms = segmentation_losses(outputs["out"], labels)
    object_parts, object_denoms = actor_object_losses(
        outputs["objects"], batch["heatmap"].to(inputs.device, non_blocking=True), batch["owners"],
        model.depth_anchors, model.depth_delta,
    )
    depth_parts, depth_denoms = dense_losses(
        outputs["dense_depth_log1p"], batch["dense_depth"], batch["dense_valid"], batch["radar_points"],
    )
    parts: dict[str, torch.Tensor] = {"segmentation": segmentation}
    parts.update(segmentation_parts); parts.update(object_parts); parts.update(depth_parts)
    scientific_names = (
        "segmentation", "heatmap", "subcell", "box_center_delta", "box_wh", "physical_ray",
        "depth_bin", "depth_residual", "endpoint", "dimensions", "yaw", "parked",
        "radar_support", "dense_depth", "radar_consistency",
    )
    total = sum(float(weights[name]) * parts[name] for name in scientific_names)
    denominators = {**segmentation_denoms, **object_denoms, **depth_denoms}
    return total, parts, denominators, outputs


def decode_depth_distribution(logits: torch.Tensor, residuals: torch.Tensor,
                              anchors: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    z = (F.softmax(logits.float(), dim=0) * (anchors + delta * residuals.float())).sum()
    return torch.expm1(z).clamp_min(0.0)

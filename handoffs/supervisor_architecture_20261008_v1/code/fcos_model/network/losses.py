from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from data import CONTENT_H, CONTENT_W, DEPTH_BINS
from model import LEVELS, SplitFusionFCOS

SCALE_RANGES = ((0.0, 32.0), (0.0, 64.0), (64.0, 128.0), (128.0, 256.0),
                (256.0, 512.0), (512.0, float("inf")))
GEOMETRY_INTERNAL = OrderedDict((
    ("depth_bin", 1.5), ("depth_residual", 0.75), ("endpoint", 0.10),
    ("physical_ray", 1.0), ("dimensions", 0.60), ("yaw", 0.15),
))


def _lovasz_grad(sorted_gt: torch.Tensor) -> torch.Tensor:
    positives = sorted_gt.sum()
    intersection = positives - sorted_gt.float().cumsum(0)
    union = positives + (1.0 - sorted_gt).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    classes = probabilities.shape[1]
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
    labels = labels.reshape(-1)
    valid = labels.ge(0) & labels.lt(classes)
    probabilities, labels = probabilities[valid], labels[valid]
    if labels.numel() == 0:
        return logits.sum() * 0.0
    values = []
    for class_index in range(classes):
        foreground = labels.eq(class_index).to(probabilities.dtype)
        if not bool(foreground.any()):
            continue
        error = (foreground - probabilities[:, class_index]).abs()
        error, order = torch.sort(error, descending=True)
        values.append(torch.dot(error, _lovasz_grad(foreground[order])))
    return torch.stack(values).mean() if values else logits.sum() * 0.0


def semantic_loss(logits: torch.Tensor, targets: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = torch.stack([target["segmentation"] for target in targets]).to(logits.device, non_blocking=True)
    weights = torch.tensor([0.5, 1.0, 4.0], device=logits.device, dtype=torch.float32)
    ce = F.cross_entropy(logits.float(), labels, weight=weights, ignore_index=-100)
    lovasz = lovasz_softmax(logits, labels)
    return ce + 0.5 * lovasz, {"semantic_ce": ce, "semantic_lovasz": lovasz}


def match_anchors(anchors: torch.Tensor, target: Mapping[str, Any],
                  num_per_level: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    device = anchors.device
    boxes = target["boxes"].to(device, non_blocking=True)
    labels = target["labels"].to(device, non_blocking=True)
    if boxes.numel() == 0:
        matched = torch.full((len(anchors),), -1, dtype=torch.int64, device=device)
    else:
        gt_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        sizes = anchors[:, 2] - anchors[:, 0]
        pairwise = (centers[:, None] - gt_centers[None]).abs().amax(dim=2) < 1.5 * sizes[:, None]
        x, y = centers[:, 0, None], centers[:, 1, None]
        x0, y0, x1, y1 = boxes.T
        distances = torch.stack((x - x0, y - y0, x1 - x, y1 - y), dim=2)
        pairwise &= distances.amin(dim=2) > 0
        maximum = distances.amax(dim=2)
        offset = 0
        for count, (lower, upper) in zip(num_per_level, SCALE_RANGES):
            level = slice(offset, offset + count)
            pairwise[level] &= maximum[level].gt(lower) & maximum[level].lt(upper)
            offset += count
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        preference = pairwise.float() * (1e8 - areas[None])
        value, matched = preference.max(dim=1)
        matched[value < 1e-5] = -1
    centers = (anchors[:, :2] + anchors[:, 2:]) / 2
    ix = centers[:, 0].floor().long()
    iy = centers[:, 1].floor().long()
    padding = iy.ge(CONTENT_H) | ix.lt(0) | ix.ge(CONTENT_W) | iy.lt(0)
    ignore = target["ignore_mask"].to(device, non_blocking=True)
    in_content = ~padding
    neutral = torch.zeros_like(padding)
    neutral[in_content] = ignore[iy[in_content], ix[in_content]]
    matched[(matched < 0) & (padding | neutral)] = -2

    level_index = torch.cat([torch.full((count,), level, dtype=torch.int64, device=device)
                             for level, count in enumerate(num_per_level)])
    positive = matched >= 0
    counts = Counter()
    per_class_level = {name: {level: 0 for level in LEVELS} for name in ("vehicle", "person")}
    actor_carriers = torch.zeros(len(boxes), dtype=torch.int64, device=device)
    if bool(positive.any()):
        actor_carriers.scatter_add_(0, matched[positive], torch.ones_like(matched[positive]))
        gt_labels = labels[matched[positive]]
        positive_levels = level_index[positive]
        for class_index, class_name in enumerate(("vehicle", "person")):
            for level, level_name in enumerate(LEVELS):
                per_class_level[class_name][level_name] = int(((gt_labels == class_index) & (positive_levels == level)).sum())
        segmentation = target["segmentation"].to(device, non_blocking=True)
        depths = target["depth"].to(device, non_blocking=True)
        positive_points = torch.where(positive)[0]
        for point_index in positive_points.tolist():
            actor = int(matched[point_index])
            px, py = int(ix[point_index]), int(iy[point_index])
            label = int(labels[actor])
            if int(segmentation[py, px]) == label + 1:
                inside = ((boxes[:, 0] <= px) & (px < boxes[:, 2]) & (boxes[:, 1] <= py) &
                          (py < boxes[:, 3]) & labels.eq(label))
                candidates = torch.where(inside)[0]
                owner = int(candidates[depths[candidates].argmin()]) if candidates.numel() else -1
                counts["own_visible" if owner == actor else "occluder"] += 1
            elif int(segmentation[py, px]) not in (0, -100):
                counts["occluder"] += 1
            else:
                counts["elsewhere"] += 1
    return matched, {
        "per_class_level": per_class_level,
        "actors_without_carrier": [target["source_identity"][index] for index in torch.where(actor_carriers == 0)[0].tolist()],
        "carrier_visibility": {name: counts[name] for name in ("own_visible", "occluder", "elsewhere")},
        "foreground": int(positive.sum()),
        "ignored_locations": int((matched == -2).sum()),
    }


def _detection_loss_level(model: SplitFusionFCOS, cls: torch.Tensor, box: torch.Tensor, ctr: torch.Tensor,
                          anchors: torch.Tensor, matched: torch.Tensor, targets: Sequence[Mapping[str, Any]],
                          denominator: int) -> dict[str, torch.Tensor]:
    batch, points, _classes = cls.shape
    gt_classes = torch.full((batch, points), -1, dtype=torch.int64, device=cls.device)
    gt_boxes = torch.zeros((batch, points, 4), dtype=torch.float32, device=cls.device)
    for image_index, target in enumerate(targets):
        match = matched[image_index]
        positive = match >= 0
        if bool(positive.any()):
            labels = target["labels"].to(cls.device)
            boxes = target["boxes"].to(cls.device)
            gt_classes[image_index, positive] = labels[match[positive]]
            gt_boxes[image_index, positive] = boxes[match[positive]]
        gt_classes[image_index, match == -2] = -2
    positive = gt_classes >= 0
    valid = gt_classes != -2
    classification_target = torch.zeros_like(cls)
    if bool(positive.any()):
        classification_target[positive, gt_classes[positive]] = 1.0
    loss_cls = sigmoid_focal_loss(cls[valid].float(), classification_target[valid].float(), reduction="sum") / max(1, denominator)
    expanded_anchors = anchors.unsqueeze(0).expand(batch, -1, -1)
    decoded = model.box_coder.decode(box.float(), expanded_anchors)
    loss_box = generalized_box_iou_loss(decoded[positive], gt_boxes[positive], reduction="sum") / max(1, denominator)
    encoded = model.box_coder.encode(expanded_anchors, gt_boxes)
    left_right = encoded[..., [0, 2]]
    top_bottom = encoded[..., [1, 3]]
    target_ctr = torch.sqrt((left_right.amin(dim=-1) / left_right.amax(dim=-1).clamp_min(1e-12)) *
                            (top_bottom.amin(dim=-1) / top_bottom.amax(dim=-1).clamp_min(1e-12))).nan_to_num(0.0)
    loss_ctr = F.binary_cross_entropy_with_logits(ctr.squeeze(2)[positive].float(), target_ctr[positive],
                                                  reduction="sum") / max(1, denominator)
    return {"classification": loss_cls, "bbox_regression": loss_box, "bbox_ctrness": loss_ctr}


def detection_losses(model: SplitFusionFCOS, outputs: Mapping[str, Any],
                     targets: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    per = outputs["detection"]["per_level"]
    num = [value.shape[1] for value in per["cls_logits"]]
    split_anchors = [list(value.split(num)) for value in outputs["anchors"]]
    matched_images, audits = [], []
    for anchors, target in zip(outputs["anchors"], targets):
        matched, audit = match_anchors(anchors, target, num)
        matched_images.append(matched); audits.append(audit)
    denominator = sum(int((matched >= 0).sum()) for matched in matched_images)
    level_parts = []
    offset = 0
    for level_index, count in enumerate(num):
        matched = torch.stack([value[offset:offset + count] for value in matched_images])
        anchors = split_anchors[0][level_index]
        level_parts.append(_detection_loss_level(
            model, per["cls_logits"][level_index], per["bbox_regression"][level_index],
            per["bbox_ctrness"][level_index], anchors, matched, targets, denominator,
        ))
        offset += count
    parts = {name: sum(level[name] for level in level_parts)
             for name in ("classification", "bbox_regression", "bbox_ctrness")}
    total = sum(parts.values())
    audit = {
        "foreground": denominator,
        "per_image": audits,
        "per_level_losses": {LEVELS[index]: {name: float(value.detach()) for name, value in level.items()}
                             for index, level in enumerate(level_parts)},
        "p2_loss_fraction": {name: float(level_parts[0][name].detach() / parts[name].detach().clamp_min(1e-12))
                             for name in parts},
    }
    return total, parts, matched_images, audit


def geometry_losses(model: SplitFusionFCOS, outputs: Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
                    matched_images: Sequence[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    num = [value.shape[1] for value in outputs["detection"]["per_level"]["cls_logits"]]
    anchors_by_image = [list(value.split(num)) for value in outputs["anchors"]]
    actor_terms: dict[str, list[torch.Tensor]] = {name: [] for name in GEOMETRY_INTERNAL}
    carrier_counts = []
    for image_index, target in enumerate(targets):
        offset = 0
        target_on = {name: value.to(outputs["c2"].device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for name, value in target.items()}
        image_actor_losses: dict[str, list[torch.Tensor]] = {name: [] for name in GEOMETRY_INTERNAL}
        image_actor_indices = []
        for level_index, count in enumerate(num):
            match = matched_images[image_index][offset:offset + count]
            offset += count
            positive = match >= 0
            if not bool(positive.any()):
                continue
            point = torch.where(positive)[0]
            actor = match[positive]
            label = target_on["labels"][actor]
            raw = outputs["geometry"][level_index]
            row = torch.arange(len(point), device=point.device)
            gathered = {name: value[image_index, point, label] for name, value in raw.items()}
            logits = gathered["depth_bin_logits"].float()
            target_bin = target_on["depth_bin"][actor]
            bin_loss = F.cross_entropy(logits, target_bin, reduction="none")
            bounded_residuals = 0.5 * torch.tanh(gathered["depth_bin_residuals"].float())
            selected = bounded_residuals[row, target_bin]
            residual_loss = F.smooth_l1_loss(selected, target_on["depth_residual"][actor], reduction="none")
            anchors = anchors_by_image[image_index][level_index]
            centers = (anchors[point, :2] + anchors[point, 2:]) / 2
            sizes = anchors[point, 2] - anchors[point, 0]
            ray_target = (target_on["physical_uv"][actor] - centers) / sizes[:, None]
            ray_loss = F.smooth_l1_loss(gathered["physical_ray"].float(), ray_target, reduction="none").mean(1)
            edges = model.depth_edges_m.float().to(point.device)
            zl, zu = torch.log1p(edges[:-1]), torch.log1p(edges[1:])
            decoded_bins = torch.expm1(0.5 * (zl + zu)[None] + bounded_residuals * (zu - zl)[None]).clamp(0.0, 40.0)
            probabilities = F.softmax(logits, dim=1)
            decoded_depth = (probabilities[:, :DEPTH_BINS] * decoded_bins).sum(1) + probabilities[:, DEPTH_BINS] * 40.0
            uv = centers + sizes[:, None] * gathered["physical_ray"].float()
            intrinsic = target_on["intrinsic"].float()
            local = torch.stack((decoded_depth,
                                 decoded_depth * (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                                 decoded_depth * (intrinsic[1, 2] - uv[:, 1]) / intrinsic[1, 1]), dim=1)
            endpoint = F.smooth_l1_loss(local / 3.0, target_on["local_xyz"][actor] / 3.0,
                                        reduction="none").mean(1)
            dimensions = F.smooth_l1_loss(gathered["log_dimensions"].float(),
                                           torch.log(target_on["dimensions"][actor]), reduction="none").mean(1)
            yaw = F.normalize(gathered["yaw"].float(), dim=1, eps=1e-6)
            yaw_loss = F.smooth_l1_loss(yaw, target_on["yaw"][actor], reduction="none").mean(1)
            for name, value in (("depth_bin", bin_loss), ("depth_residual", residual_loss),
                                ("physical_ray", ray_loss), ("endpoint", endpoint),
                                ("dimensions", dimensions), ("yaw", yaw_loss)):
                image_actor_losses[name].append(value)
            image_actor_indices.append(actor)
        if image_actor_indices:
            actors = torch.cat(image_actor_indices)
            carrier_counts.append(int(len(actors)))
            for name in GEOMETRY_INTERNAL:
                carrier_loss = torch.cat(image_actor_losses[name])
                for actor in range(len(target_on["labels"])):
                    mask = actors == actor
                    if bool(mask.any()):
                        actor_terms[name].append(carrier_loss[mask].mean())
        else:
            carrier_counts.append(0)
    reference = outputs["c2"]
    parts = {name: torch.stack(values).mean() if values else reference.sum() * 0.0
             for name, values in actor_terms.items()}
    total = sum(weight * parts[name] for name, weight in GEOMETRY_INTERNAL.items())
    return total, parts, {"carriers_per_image": carrier_counts, "normalization": "per_actor_over_complete_batch"}


def dense_losses(outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    prediction = outputs["dense_depth_log1p_stride4"][:, 0, :CONTENT_H // 4].float()
    depth = batch["dense_depth"].to(prediction.device, non_blocking=True)
    valid = batch["dense_valid"].to(prediction.device, non_blocking=True)
    dense = F.smooth_l1_loss(prediction[valid], torch.log1p(depth[valid]), reduction="mean")
    radar_values, radar_count = [], 0
    for image_index, points_cpu in enumerate(batch["radar_points"]):
        if points_cpu.numel() == 0:
            continue
        points = points_cpu.to(prediction.device, non_blocking=True)
        sampled = F.grid_sample(prediction[image_index:image_index + 1, None],
                                points[:, :2].view(1, 1, -1, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=False).reshape(-1)
        radar_values.append(F.smooth_l1_loss(sampled, points[:, 2].float(), reduction="sum"))
        radar_count += len(points)
    radar = torch.stack(radar_values).sum() / max(1, radar_count) if radar_values else prediction.sum() * 0.0
    return dense + 0.5 * radar, {"dense_depth": dense, "radar_consistency": radar}, {
        "dense_valid_pixels": int(valid.sum()), "radar_points": radar_count,
        "padded_stride4_rows_ignored": outputs["dense_depth_log1p_stride4"].shape[-2] - CONTENT_H // 4,
    }


def compute_loss_groups(model: SplitFusionFCOS, batch: Mapping[str, Any],
                        multipliers: Mapping[str, float] | None = None, *,
                        use_amp: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any], Mapping[str, Any]]:
    inputs = batch["input"].to(next(model.parameters()).device, non_blocking=True)
    targets = batch["targets"]
    amp = inputs.device.type == "cuda" and use_amp
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = model(inputs, dense=True)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        detection, detection_parts, matched, assignment = detection_losses(model, outputs, targets)
        geometry, geometry_parts, geometry_audit = geometry_losses(model, outputs, targets, matched)
        semantic, semantic_parts = semantic_loss(outputs["semantic_logits"], targets)
        auxiliary, auxiliary_parts, auxiliary_audit = dense_losses(outputs, batch)
    groups = {"D": detection, "G": geometry, "S": semantic, "A": auxiliary}
    weights = {"D": 1.0, "G": 1.0, "S": 1.0, "A": 1.0}
    if multipliers is not None:
        weights.update({name: float(multipliers[name]) for name in ("G", "S", "A")})
    total = sum(weights[name] * groups[name] for name in ("D", "G", "S", "A"))
    total_pressure = total.detach().abs().clamp_min(1e-12)
    components = {**{f"fcos_{name}": value for name, value in detection_parts.items()},
                  **{f"geometry_{name}": value for name, value in geometry_parts.items()},
                  **semantic_parts, **auxiliary_parts, **groups,
                  **{f"weighted_{name}": weights[name] * groups[name] for name in groups},
                  **{f"optimization_fraction_{name}": (weights[name] * groups[name]).detach() / total_pressure
                     for name in groups}, "total": total}
    audit = {"assignment": assignment, "geometry": geometry_audit, "auxiliary": auxiliary_audit,
             "multipliers": weights}
    return total, components, audit, outputs


def scalar_components(parts: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().float().item()) for name, value in parts.items()}

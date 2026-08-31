from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.labeling import (
    label_candidates,
)

CONTENT_HEIGHT = 432
CONTENT_WIDTH = 768
PERSON_INTERNAL_CLASS = 1
PERSON_SEMANTIC_CHANNEL = 2
CANONICAL_SCORE_THRESHOLD = 0.20
WORLD_MATCH_RADIUS_M = 3.0
SEMANTIC_SUPPORT_THRESHOLDS = (None, 0.01, 0.025, 0.05, 0.10, 0.20)
GROUP_IOU_THRESHOLDS = (None, 0.05, 0.10, 0.20, 0.30, 0.40)
HOLDOUT_EXPERIMENT_IDS = frozenset((
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
))


def grid_configurations() -> tuple[dict[str, float | int | None], ...]:
    return tuple({
        "grid_index": support_index * len(GROUP_IOU_THRESHOLDS) + group_index,
        "semantic_support_threshold": support,
        "group_box_iou_threshold": group,
    } for support_index, support in enumerate(SEMANTIC_SUPPORT_THRESHOLDS)
      for group_index, group in enumerate(GROUP_IOU_THRESHOLDS))


def validate_configuration(configuration: Mapping[str, Any]) -> None:
    support = configuration.get("semantic_support_threshold")
    group = configuration.get("group_box_iou_threshold")
    if support not in SEMANTIC_SUPPORT_THRESHOLDS or group not in GROUP_IOU_THRESHOLDS:
        raise ValueError("consolidation configuration is outside the preregistered grid")
    expected_index = (SEMANTIC_SUPPORT_THRESHOLDS.index(support) * len(GROUP_IOU_THRESHOLDS)
                      + GROUP_IOU_THRESHOLDS.index(group))
    if "grid_index" in configuration and int(configuration["grid_index"]) != expected_index:
        raise ValueError("consolidation grid index does not match its thresholds")


def person_mask_from_logits(semantic_logits: torch.Tensor, *, image_index: int = 0) -> torch.Tensor:
    if semantic_logits.ndim != 4 or semantic_logits.shape[1] <= PERSON_SEMANTIC_CHANNEL:
        raise ValueError("semantic logits must have shape [B,C,H,W] with person channel 2")
    mask = semantic_logits.argmax(dim=1)[image_index].eq(PERSON_SEMANTIC_CHANNEL)
    return mask.detach().bool().cpu()


def connected_person_components(person_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return deterministic row-major IDs for 8-connected foreground components."""
    if person_mask.ndim != 2:
        raise ValueError("person mask must be two-dimensional")
    mask = np.ascontiguousarray(person_mask.detach().bool().cpu().numpy(), dtype=np.uint8)
    if hasattr(cv2, "connectedComponentsWithAlgorithm"):
        count, raw = cv2.connectedComponentsWithAlgorithm(mask, 8, cv2.CV_32S, cv2.CCL_SAUF)
    else:
        count, raw = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
    flat = raw.reshape(-1)
    ids, first = np.unique(flat, return_index=True)
    foreground = ids != 0
    ids, first = ids[foreground], first[foreground]
    order = np.argsort(first, kind="stable")
    lookup = np.zeros(max(count, 1), dtype=np.int32)
    for new_id, old_id in enumerate(ids[order], start=1):
        lookup[int(old_id)] = new_id
    labels = lookup[raw]
    return torch.from_numpy(labels.copy()).long(), len(ids)


def _box_pixel_bounds(box: torch.Tensor, height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise FloatingPointError("non-finite candidate box")
    left = max(0, min(width, math.ceil(x0 - 0.5)))
    top = max(0, min(height, math.ceil(y0 - 0.5)))
    right = max(0, min(width, math.ceil(x1 - 0.5)))
    bottom = max(0, min(height, math.ceil(y1 - 0.5)))
    return left, top, right, bottom


def assign_components(component_labels: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign each box the intersecting component with maximum raster-mask IoU."""
    labels = component_labels.detach().long().cpu()
    boxes = boxes.detach().double().cpu()
    if labels.ndim != 2 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("component-label/box shape drift")
    height, width = labels.shape
    component_count = int(labels.max()) if labels.numel() else 0
    areas = torch.bincount(labels.flatten(), minlength=component_count + 1).double()
    assignments = torch.full((boxes.shape[0],), -1, dtype=torch.int32)
    support = torch.zeros((boxes.shape[0],), dtype=torch.float32)
    for index, box in enumerate(boxes):
        left, top, right, bottom = _box_pixel_bounds(box, height, width)
        box_area = (right - left) * (bottom - top)
        if box_area <= 0 or component_count == 0:
            continue
        intersections = torch.bincount(
            labels[top:bottom, left:right].flatten(), minlength=component_count + 1,
        ).double()
        intersections[0] = 0.0
        unions = float(box_area) + areas - intersections
        ious = torch.where(unions > 0, intersections / unions, torch.zeros_like(unions))
        ious[0] = 0.0
        best = int(torch.argmax(ious))
        if best > 0 and float(intersections[best]) > 0.0:
            assignments[index] = best
            support[index] = float(ious[best])
    return assignments, support


def candidate_ignore_flags(boxes: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    boxes = boxes.detach().double().cpu()
    mask = ignore_mask.detach().bool().cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 4 or mask.ndim != 2:
        raise ValueError("box/ignore-mask shape drift")
    height, width = mask.shape
    centres = torch.round((boxes[:, :2] + boxes[:, 2:]) / 2.0).long()
    valid = ((centres[:, 0] >= 0) & (centres[:, 0] < width)
             & (centres[:, 1] >= 0) & (centres[:, 1] < height))
    x = centres[:, 0].clamp(0, width - 1)
    y = centres[:, 1].clamp(0, height - 1)
    return valid & mask[y, x]


def build_frame_record(
    *,
    outputs: Mapping[str, Any],
    detections: Mapping[str, torch.Tensor],
    ignore_mask: torch.Tensor,
    gt_person_world_xy: torch.Tensor,
    sample_id: str,
    experiment_id: str,
) -> dict[str, Any]:
    person_indices = torch.where(detections["labels_internal"].long() == PERSON_INTERNAL_CLASS)[0]
    boxes = detections["boxes"].index_select(0, person_indices).detach().float().cpu()
    mask = person_mask_from_logits(outputs["semantic_logits"])
    if mask.shape != (CONTENT_HEIGHT, CONTENT_WIDTH):
        raise RuntimeError("full-resolution semantic mask shape drift")
    components, component_count = connected_person_components(mask)
    component_ids, support = assign_components(components, boxes)
    frame = {
        "sample_id": str(sample_id),
        "experiment_id": str(experiment_id),
        "original_indices": person_indices.detach().to(torch.int32).cpu(),
        "scores": detections["scores"].index_select(0, person_indices).detach().float().cpu(),
        "boxes": boxes,
        "world_xy": detections["world_xyz"].index_select(0, person_indices)[:, :2].detach().double().cpu(),
        "component_ids": component_ids,
        "semantic_support": support,
        "ignore_flags": candidate_ignore_flags(boxes, ignore_mask),
        "gt_world_xy": gt_person_world_xy.detach().double().cpu().reshape(-1, 2),
        "semantic_component_count": int(component_count),
    }
    count = person_indices.numel()
    if not all(frame[name].shape[0] == count for name in (
        "original_indices", "scores", "boxes", "world_xy", "component_ids", "semantic_support", "ignore_flags",
    )):
        raise RuntimeError("person frame-record count drift")
    return frame


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    x0, y0 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    x1, y1 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def consolidate_person_candidates(
    *,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    world_xy: torch.Tensor,
    component_ids: torch.Tensor,
    semantic_support: torch.Tensor,
    original_indices: torch.Tensor | None = None,
    semantic_support_threshold: float | None,
    group_box_iou_threshold: float | None,
) -> torch.Tensor:
    """Return retained positions in original candidate order for one fixed rule."""
    if semantic_support_threshold not in SEMANTIC_SUPPORT_THRESHOLDS:
        raise ValueError("semantic-support threshold is outside the preregistered grid")
    if group_box_iou_threshold not in GROUP_IOU_THRESHOLDS:
        raise ValueError("group box-IoU threshold is outside the preregistered grid")
    scores = scores.detach().float().cpu()
    boxes = boxes.detach().double().cpu()
    world_xy = world_xy.detach().double().cpu()
    component_ids = component_ids.detach().long().cpu()
    semantic_support = semantic_support.detach().float().cpu()
    count = scores.numel()
    if (boxes.shape != (count, 4) or world_xy.shape != (count, 2)
            or component_ids.shape != (count,) or semantic_support.shape != (count,)):
        raise ValueError("person consolidation input shape drift")
    if not bool(torch.isfinite(scores).all() and torch.isfinite(boxes).all()
                and torch.isfinite(world_xy).all() and torch.isfinite(semantic_support).all()):
        raise FloatingPointError("non-finite person consolidation input")
    original = (torch.arange(count, dtype=torch.long) if original_indices is None
                else original_indices.detach().long().cpu())
    if original.shape != (count,) or len(set(original.tolist())) != count:
        raise ValueError("original candidate indices must be unique and aligned")
    eligible = scores >= CANONICAL_SCORE_THRESHOLD
    if semantic_support_threshold is not None:
        eligible &= semantic_support >= float(semantic_support_threshold)
    positions = torch.where(eligible)[0].tolist()
    if group_box_iou_threshold is None or len(positions) < 2:
        return torch.tensor(sorted(positions, key=lambda index: int(original[index])), dtype=torch.long)

    parent = {index: index for index in positions}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for offset, left in enumerate(positions):
        for right in positions[offset + 1:]:
            if int(component_ids[left]) < 0 or int(component_ids[left]) != int(component_ids[right]):
                continue
            if float(torch.linalg.vector_norm(world_xy[left] - world_xy[right])) > WORLD_MATCH_RADIUS_M:
                continue
            if _box_iou(boxes[left], boxes[right]) >= float(group_box_iou_threshold):
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in positions:
        groups[find(index)].append(index)
    winners = [min(members, key=lambda index: (-float(scores[index]), int(original[index])))
               for members in groups.values()]
    return torch.tensor(sorted(winners, key=lambda index: int(original[index])), dtype=torch.long)


def _synthetic_ignore_mask(boxes: torch.Tensor, ignore_flags: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((CONTENT_HEIGHT, CONTENT_WIDTH), dtype=torch.bool)
    centres = torch.round((boxes.double()[:, :2] + boxes.double()[:, 2:]) / 2.0).long()
    for centre, ignored in zip(centres, ignore_flags.bool()):
        x, y = int(centre[0]), int(centre[1])
        if bool(ignored) and 0 <= x < CONTENT_WIDTH and 0 <= y < CONTENT_HEIGHT:
            mask[y, x] = True
    return mask


def rematch_person_frame(frame: Mapping[str, Any], retained_positions: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    positions = retained_positions.detach().long().cpu()
    boxes = frame["boxes"].index_select(0, positions).double()
    world_xy = frame["world_xy"].index_select(0, positions).double()
    ignored = frame["ignore_flags"].index_select(0, positions).bool()
    gt_world_xy = frame["gt_world_xy"].double()
    return label_candidates(
        candidate_world_xy=world_xy,
        candidate_classes=torch.ones(positions.numel(), dtype=torch.long),
        candidate_boxes=boxes,
        gt_world_xy=gt_world_xy,
        gt_classes=torch.ones(gt_world_xy.shape[0], dtype=torch.long),
        ignore_mask=_synthetic_ignore_mask(boxes, ignored),
        match_radius_m=WORLD_MATCH_RADIUS_M,
    )


def evaluate_frames(frames: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any]) -> dict[str, Any]:
    validate_configuration(configuration)
    totals = {"tp": 0, "fp": 0, "fn": 0, "ignored": 0, "retained_predictions": 0}
    for frame in frames:
        retained = consolidate_person_candidates(
            scores=frame["scores"], boxes=frame["boxes"], world_xy=frame["world_xy"],
            component_ids=frame["component_ids"], semantic_support=frame["semantic_support"],
            original_indices=frame["original_indices"],
            semantic_support_threshold=configuration["semantic_support_threshold"],
            group_box_iou_threshold=configuration["group_box_iou_threshold"],
        )
        labels, summary = rematch_person_frame(frame, retained)
        totals["tp"] += int(summary["tp"])
        totals["fp"] += int((labels == 0).sum())
        totals["fn"] += int(summary["fn"])
        totals["ignored"] += int((labels == -1).sum())
        totals["retained_predictions"] += retained.numel()
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        **dict(configuration),
        **totals,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def partition_experiment_ids(experiment_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    episode_ids = set(experiment_ids)
    if (len(episode_ids) != 10 or not HOLDOUT_EXPERIMENT_IDS.issubset(episode_ids)
            or any("_train_" not in experiment_id for experiment_id in episode_ids)):
        raise RuntimeError("expected exactly ten canonical training episodes including both holdouts")
    fit_ids = tuple(sorted(episode_ids - HOLDOUT_EXPERIMENT_IDS))
    holdout_ids = tuple(sorted(episode_ids & HOLDOUT_EXPERIMENT_IDS))
    if len(fit_ids) != 8 or len(holdout_ids) != 2 or set(fit_ids) & set(holdout_ids):
        raise RuntimeError("consolidation episode split is not exact and disjoint 8/2")
    return fit_ids, holdout_ids


def partition_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], tuple[str, ...], tuple[str, ...]]:
    fit_ids, holdout_ids = partition_experiment_ids([str(frame["experiment_id"]) for frame in frames])
    fit = [frame for frame in frames if frame["experiment_id"] in fit_ids]
    holdout = [frame for frame in frames if frame["experiment_id"] in holdout_ids]
    return fit, holdout, fit_ids, holdout_ids


def select_fit_configuration(fit_frames: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    reports = [evaluate_frames(fit_frames, configuration) for configuration in grid_configurations()]
    eligible = [report for report in reports if report["recall"] >= 0.80]
    if not eligible:
        return reports, None
    selected = min(eligible, key=lambda report: (
        -report["precision"], -report["recall"], report["retained_predictions"], report["grid_index"],
    ))
    return reports, selected


def retained_detection_indices(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor], configuration: Mapping[str, Any],
) -> torch.Tensor:
    """Return original post-NMS indices retained by one selected person rule."""
    validate_configuration(configuration)
    classes = detections["labels_internal"].long()
    person_indices = torch.where(classes == PERSON_INTERNAL_CLASS)[0]
    mask = person_mask_from_logits(outputs["semantic_logits"])
    components, _count = connected_person_components(mask)
    component_ids, support = assign_components(components, detections["boxes"].index_select(0, person_indices))
    retained_positions = consolidate_person_candidates(
        scores=detections["scores"].index_select(0, person_indices),
        boxes=detections["boxes"].index_select(0, person_indices),
        world_xy=detections["world_xyz"].index_select(0, person_indices)[:, :2],
        component_ids=component_ids,
        semantic_support=support,
        original_indices=person_indices.cpu(),
        semantic_support_threshold=configuration["semantic_support_threshold"],
        group_box_iou_threshold=configuration["group_box_iou_threshold"],
    )
    retained_person = person_indices.index_select(0, retained_positions.to(person_indices.device))
    vehicle = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    return torch.cat((vehicle, retained_person)).sort().values


def apply_rule_to_detections(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor], configuration: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Select retained tensors without changing any vehicle or retained-person value."""
    classes = detections["labels_internal"].long()
    vehicle = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    keep = retained_detection_indices(outputs, detections, configuration)
    result = {name: value.index_select(0, keep) for name, value in detections.items()}
    if not all(torch.equal(result[name][torch.where(result["labels_internal"] != PERSON_INTERNAL_CLASS)[0]],
                           value.index_select(0, vehicle)) for name, value in detections.items()):
        raise RuntimeError("vehicle field changed during person instance consolidation")
    return result

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

MATCH_RADIUS_M = 3.0
IGNORE_LABEL = -1
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


def _ignored_box_centres(boxes: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4 or ignore_mask.ndim != 2:
        raise ValueError("box/ignore-mask shape drift")
    boxes_cpu = boxes.detach().double().cpu()
    mask_cpu = ignore_mask.detach().bool().cpu()
    height, width = mask_cpu.shape
    ignored = []
    for box in boxes_cpu:
        x = int(round(float((box[0] + box[2]) / 2.0)))
        y = int(round(float((box[1] + box[3]) / 2.0)))
        ignored.append(0 <= x < width and 0 <= y < height and bool(mask_cpu[y, x]))
    return torch.tensor(ignored, dtype=torch.bool)


def label_candidates(
    *, candidate_world_xy: torch.Tensor, candidate_classes: torch.Tensor, candidate_boxes: torch.Tensor,
    gt_world_xy: torch.Tensor, gt_classes: torch.Tensor, ignore_mask: torch.Tensor,
    match_radius_m: float = MATCH_RADIUS_M,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Label post-base-NMS candidates with deterministic evaluator-style matching.

    Ignore-centred candidates are removed before matching because their label is
    categorically excluded from candidate-quality loss. Remaining same-class
    pairs are greedily selected in ``(world distance, candidate index, GT index)``
    order, identical to the canonical evaluator's one-to-one nearest rule.
    """
    candidate_world_xy = candidate_world_xy.detach().double().cpu()
    candidate_classes = candidate_classes.detach().long().cpu()
    candidate_boxes = candidate_boxes.detach().double().cpu()
    gt_world_xy = gt_world_xy.detach().double().cpu()
    gt_classes = gt_classes.detach().long().cpu()
    count, eligible_gt = candidate_classes.numel(), gt_classes.numel()
    if candidate_world_xy.shape != (count, 2) or candidate_boxes.shape != (count, 4):
        raise ValueError("candidate label input shape drift")
    if gt_world_xy.shape != (eligible_gt, 2):
        raise ValueError("GT label input shape drift")
    if not bool(torch.isfinite(candidate_world_xy).all()) or not bool(torch.isfinite(gt_world_xy).all()):
        raise FloatingPointError("non-finite world coordinate in candidate labeler")

    ignored = _ignored_box_centres(candidate_boxes, ignore_mask)
    labels = torch.full((count,), NEGATIVE_LABEL, dtype=torch.int8)
    labels[ignored] = IGNORE_LABEL
    pairs: list[tuple[float, int, int]] = []
    for candidate_index in range(count):
        if bool(ignored[candidate_index]):
            continue
        for gt_index in range(eligible_gt):
            if int(candidate_classes[candidate_index]) != int(gt_classes[gt_index]):
                continue
            delta = candidate_world_xy[candidate_index] - gt_world_xy[gt_index]
            distance = math.hypot(float(delta[0]), float(delta[1]))
            if distance <= float(match_radius_m):
                pairs.append((distance, candidate_index, gt_index))
    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    for _distance, candidate_index, gt_index in sorted(pairs):
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        labels[candidate_index] = POSITIVE_LABEL

    true_positive = len(used_gt)
    false_negative = eligible_gt - true_positive
    if true_positive + false_negative != eligible_gt:
        raise RuntimeError("TP + FN does not equal eligible GT count")
    summary = {
        "eligible_gt": eligible_gt,
        "tp": true_positive,
        "fn": false_negative,
        "negative": int((labels == NEGATIVE_LABEL).sum()),
        "ignored": int((labels == IGNORE_LABEL).sum()),
        "tp_plus_fn_reconciles": True,
    }
    return labels, summary


def contract_world_targets(
    rows: Sequence[Mapping[str, str]], class_names: Sequence[str] = ("vehicle", "person"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read evaluator world coordinates directly from eligible v0.10 rows."""
    eligible = [row for row in rows if row["label"] in class_names]
    world_xy = torch.tensor(
        [[float(row["object_world_x"]), float(row["object_world_y"])] for row in eligible],
        dtype=torch.float64,
    ).reshape(-1, 2)
    classes = torch.tensor([class_names.index(row["label"]) for row in eligible], dtype=torch.int64)
    return world_xy, classes

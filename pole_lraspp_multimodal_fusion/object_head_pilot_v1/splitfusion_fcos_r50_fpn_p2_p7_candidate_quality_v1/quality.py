from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

FPN_CHANNELS = 256
LEVEL_NAMES = ("p2", "p3", "p4", "p5", "p6", "p7")
EXTRA_FEATURE_NAMES = (
    "base_score",
    "class_identity",
    "normalized_fpn_level",
    "semantic_probability_at_point",
    "semantic_probability_max_in_box",
    "depth_bin_max_probability",
    "normalized_depth_bin_entropy",
)
FEATURE_DIM = FPN_CHANNELS + len(EXTRA_FEATURE_NAMES)
HIDDEN_DIM = 64
SCORE_EPS = 1e-6


class QualityMLP(nn.Module):
    """The sole trainable component in the refinement package."""

    def __init__(self, *, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.normalization = nn.LayerNorm(FEATURE_DIM) if self.normalize else nn.Identity()
        self.hidden = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.output = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected [N,{FEATURE_DIM}] candidate features, got {tuple(features.shape)}")
        return self.output(self.activation(self.hidden(self.normalization(features)))).squeeze(1)


def refined_logits(base_scores: torch.Tensor, quality_delta: torch.Tensor) -> torch.Tensor:
    """Return the prescribed frozen-base logit plus learned quality residual."""
    if base_scores.shape != quality_delta.shape:
        raise ValueError("base-score and quality-delta shapes differ")
    base = base_scores.float().clamp(SCORE_EPS, 1.0 - SCORE_EPS)
    return torch.logit(base) + quality_delta.float()


def refine_scores(base_scores: torch.Tensor, quality_delta: torch.Tensor) -> torch.Tensor:
    """Apply a residual in logit space; a zero residual is numerically neutral."""
    return torch.sigmoid(refined_logits(base_scores, quality_delta))


def _semantic_box_max(probability: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    height, width = probability.shape
    x0 = max(0, min(width - 1, int(math.floor(float(box[0])))))
    y0 = max(0, min(height - 1, int(math.floor(float(box[1])))))
    x1 = max(x0 + 1, min(width, int(math.ceil(float(box[2])))))
    y1 = max(y0 + 1, min(height, int(math.ceil(float(box[3])))))
    return probability[y0:y1, x0:x1].amax()


def extract_candidate_features(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor], *, image_index: int = 0,
) -> torch.Tensor:
    """Gather the frozen feature and exactly seven scalar candidate attributes."""
    levels = detections["level_indices"].long()
    points = detections["point_indices"].long()
    classes = detections["labels_internal"].long()
    scores = detections["scores"].float()
    boxes = detections["boxes"].float()
    identities = detections["candidate_identity"].long()
    count = int(scores.numel())
    device = scores.device
    if not all(value.shape[0] == count for value in (levels, points, classes, boxes, identities)):
        raise ValueError("candidate field count drift")
    if count == 0:
        return torch.empty((0, FEATURE_DIM), device=device, dtype=torch.float32)
    expected_identity = torch.stack((
        torch.full_like(levels, int(image_index)), levels, points, classes,
    ), dim=1)
    if not torch.equal(identities, expected_identity):
        raise RuntimeError("candidate identity drift before quality extraction")

    semantic = torch.softmax(outputs["semantic_logits"][image_index].float(), dim=0)
    depth_probabilities = detections.get("depth_bin_probabilities")
    if depth_probabilities is None:
        raise KeyError("recovered detector must expose candidate depth-bin probabilities")
    depth_probabilities = depth_probabilities.float()
    if depth_probabilities.ndim != 2 or depth_probabilities.shape[0] != count:
        raise ValueError("candidate depth probability shape drift")
    depth_count = depth_probabilities.shape[1]
    if depth_count < 2:
        raise ValueError("depth distribution must contain at least two bins")

    rows: list[torch.Tensor] = []
    for index in range(count):
        level_index = int(levels[index])
        if not 0 <= level_index < len(LEVEL_NAMES):
            raise ValueError(f"invalid FPN level index {level_index}")
        feature_map = outputs["features"][LEVEL_NAMES[level_index]][image_index]
        _channels, height, width = feature_map.shape
        point_index = int(points[index])
        if not 0 <= point_index < height * width:
            raise ValueError(f"invalid flattened point {point_index} for {LEVEL_NAMES[level_index]}")
        row, column = divmod(point_index, width)
        frozen_feature = feature_map[:, row, column].float()

        # FPN cells cover the padded 448x768 detector input. The semantic output
        # covers the 432x768 content, so padded-bottom points clamp to its last row.
        semantic_y = min(semantic.shape[1] - 1, int((row + 0.5) * 448.0 / height))
        semantic_x = min(semantic.shape[2] - 1, int((column + 0.5) * 768.0 / width))
        semantic_class = int(classes[index]) + 1
        class_probability = semantic[semantic_class]
        point_probability = class_probability[semantic_y, semantic_x]
        box_probability = _semantic_box_max(class_probability, boxes[index])

        depth_probability = depth_probabilities[index]
        depth_max = depth_probability.amax()
        entropy = -(depth_probability * depth_probability.clamp_min(1e-12).log()).sum()
        normalized_entropy = entropy / math.log(depth_count)
        extras = torch.stack((
            scores[index],
            classes[index].float(),
            levels[index].float() / float(len(LEVEL_NAMES) - 1),
            point_probability,
            box_probability,
            depth_max,
            normalized_entropy,
        ))
        rows.append(torch.cat((frozen_feature, extras)))
    features = torch.stack(rows)
    if features.shape != (count, FEATURE_DIM) or not bool(torch.isfinite(features).all()):
        raise FloatingPointError("non-finite or malformed candidate feature vector")
    return features


def deterministic_class_aware_nms(
    boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou_threshold: float = 0.60,
) -> torch.Tensor:
    """Stable score-first NMS over all levels, suppressing only within a class."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("NMS IoU threshold must be in [0,1]")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.numel() or scores.shape != classes.shape:
        raise ValueError("NMS input shape drift")
    boxes_cpu = boxes.detach().double().cpu()
    scores_cpu = scores.detach().double().cpu()
    classes_cpu = classes.detach().long().cpu()
    order = sorted(range(scores_cpu.numel()), key=lambda index: (-float(scores_cpu[index]), index))
    kept: list[int] = []
    for index in order:
        candidate = boxes_cpu[index]
        suppress = False
        for prior_index in kept:
            if int(classes_cpu[index]) != int(classes_cpu[prior_index]):
                continue
            prior = boxes_cpu[prior_index]
            x0, y0 = max(float(candidate[0]), float(prior[0])), max(float(candidate[1]), float(prior[1]))
            x1, y1 = min(float(candidate[2]), float(prior[2])), min(float(candidate[3]), float(prior[3]))
            intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            candidate_area = max(0.0, float(candidate[2] - candidate[0])) * max(0.0, float(candidate[3] - candidate[1]))
            prior_area = max(0.0, float(prior[2] - prior[0])) * max(0.0, float(prior[3] - prior[1]))
            iou = intersection / max(1e-12, candidate_area + prior_area - intersection)
            if iou > float(iou_threshold):
                suppress = True
                break
        if not suppress:
            kept.append(index)
    return torch.tensor(kept, dtype=torch.long, device=boxes.device)


def apply_refinement(
    detections: Mapping[str, torch.Tensor], refined_scores: torch.Tensor,
    *, nms_iou: float = 0.60, limit: int = 100,
) -> dict[str, torch.Tensor]:
    """Re-rank existing candidates, apply one cross-level NMS, and create nothing."""
    if refined_scores.shape != detections["scores"].shape:
        raise ValueError("refined-score count drift")
    rescored = dict(detections)
    rescored["scores"] = refined_scores
    keep = deterministic_class_aware_nms(
        rescored["boxes"], refined_scores, rescored["labels_internal"], nms_iou,
    )[: int(limit)]
    return {name: value[keep] for name, value in rescored.items()}


def sigmoid_focal_loss(
    logits: torch.Tensor, labels: torch.Tensor, *, alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    if logits.shape != labels.shape:
        raise ValueError("focal-loss shape drift")
    targets = labels.float()
    cross_entropy = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    probability = torch.sigmoid(logits.float())
    probability_target = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_target * (1.0 - probability_target).pow(gamma) * cross_entropy).mean()


def build_quality_optimizer(head: QualityMLP) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    head_ids = {id(parameter) for parameter in head.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != head_ids:
        raise RuntimeError("optimizer is not exactly restricted to quality-head parameters")
    return optimizer

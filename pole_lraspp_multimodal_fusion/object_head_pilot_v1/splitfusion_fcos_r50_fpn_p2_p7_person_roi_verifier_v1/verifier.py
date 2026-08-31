from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign, roi_pool

FPN_CHANNELS = 256
LEVEL_NAMES = ("p2", "p3", "p4", "p5", "p6", "p7")
ROI_LEVEL_NAMES = ("p2", "p3")
PADDED_IMAGE_SIZE = (448, 768)
ROI_OUTPUT_SIZE = (7, 7)
ROI_POOLED_SIZE = (2, 2)
ROI_DESCRIPTOR_DIM = FPN_CHANNELS * ROI_POOLED_SIZE[0] * ROI_POOLED_SIZE[1]
SCALAR_FEATURE_NAMES = (
    "base_fcos_score",
    "normalized_source_fpn_level",
    "person_semantic_probability_at_candidate_point",
    "maximum_person_semantic_probability_inside_box",
    "depth_bin_maximum_probability",
    "normalized_depth_bin_entropy",
    "normalized_box_width",
    "normalized_box_height",
    "normalized_box_area",
    "log_aspect_ratio",
)
FEATURE_DIM = ROI_DESCRIPTOR_DIM + len(SCALAR_FEATURE_NAMES)
HIDDEN_DIM = 128
PERSON_CLASS = 1
PERSON_SEMANTIC_CHANNEL = 2
SCORE_EPS = 1e-6
INITIAL_CALIBRATION_BIAS = 0.0
VERIFIER_ARCHITECTURE = {
    "layers": ["LayerNorm(1034)", "Linear(1034,128)", "ReLU", "Linear(128,1)"],
    "input": FEATURE_DIM,
    "hidden": HIDDEN_DIM,
    "output": 1,
    "final_layer_zero_initialized": True,
}
HOLDOUT_EXPERIMENT_IDS = frozenset((
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
))


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("logit input must be in (0,1)")
    return math.log(value / (1.0 - value))


def fp16_round_trip_roi_descriptors(value: torch.Tensor) -> torch.Tensor:
    """Return the exact FP32 representation seen after FP16 cache storage."""
    value = value.float()
    if value.ndim != 2 or value.shape[1] != ROI_DESCRIPTOR_DIM:
        raise ValueError(f"expected [N,{ROI_DESCRIPTOR_DIM}] ROI descriptors, got {tuple(value.shape)}")
    if (not bool(torch.isfinite(value).all())
            or (value.numel() and float(value.abs().amax()) > torch.finfo(torch.float16).max)):
        raise FloatingPointError("ROI descriptor is not safe for an FP16 round trip")
    return value.to(torch.float16).to(torch.float32)


class PersonRoIDescriptor(nn.Module):
    """Parameter-free vectorized descriptor for post-NMS person candidates."""

    def __init__(self) -> None:
        super().__init__()
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=list(ROI_LEVEL_NAMES), output_size=ROI_OUTPUT_SIZE, sampling_ratio=2,
        )

    def forward(
        self, outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor], *, image_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = detections["scores"].float()
        levels = detections["level_indices"].long()
        points = detections["point_indices"].long()
        classes = detections["labels_internal"].long()
        boxes = detections["boxes"].float()
        identities = detections["candidate_identity"].long()
        count = scores.numel()
        if boxes.shape != (count, 4) or identities.shape != (count, 4):
            raise ValueError("candidate box/identity shape drift")
        if not all(value.shape == (count,) for value in (levels, points, classes)):
            raise ValueError("candidate field count drift")
        expected_identity = torch.stack((
            torch.full_like(levels, int(image_index)), levels, points, classes,
        ), dim=1)
        if not torch.equal(identities, expected_identity):
            raise RuntimeError("candidate identity drift before person ROI extraction")
        if bool(((levels < 0) | (levels >= len(LEVEL_NAMES))).any()):
            raise ValueError("invalid source FPN level index")

        person_indices = torch.where(classes == PERSON_CLASS)[0]
        person_boxes = boxes.index_select(0, person_indices)
        feature_maps = outputs["features"]
        roi_inputs: dict[str, torch.Tensor] = {}
        for name in ROI_LEVEL_NAMES:
            value = feature_maps[name]
            if value.ndim != 4 or value.shape[1] != FPN_CHANNELS or image_index >= value.shape[0]:
                raise ValueError(f"{name} frozen FPN shape drift")
            roi_inputs[name] = value[image_index:image_index + 1]

        # This is the only FPN ROI operation: every person box in the frame is
        # submitted together in one MultiScaleRoIAlign call.
        raw_rois = self.roi_align(roi_inputs, [person_boxes], [PADDED_IMAGE_SIZE])
        if raw_rois.shape != (person_indices.numel(), FPN_CHANNELS, *ROI_OUTPUT_SIZE):
            raise ValueError("7x7 frozen ROI shape drift")
        roi_descriptors = F.adaptive_avg_pool2d(raw_rois.float(), ROI_POOLED_SIZE).flatten(1)

        person_levels = levels.index_select(0, person_indices)
        person_points = points.index_select(0, person_indices)
        person_scores = scores.index_select(0, person_indices)
        semantic = torch.softmax(outputs["semantic_logits"][image_index].float(), dim=0)
        if semantic.ndim != 3 or semantic.shape[0] <= PERSON_SEMANTIC_CHANNEL:
            raise ValueError("semantic-logit shape drift")
        semantic_y = torch.empty_like(person_points)
        semantic_x = torch.empty_like(person_points)
        for level_index, level_name in enumerate(LEVEL_NAMES):
            mask = person_levels == level_index
            level_points = person_points[mask]
            _batch, _channels, height, width = feature_maps[level_name].shape
            if bool(((level_points < 0) | (level_points >= height * width)).any()):
                raise ValueError(f"invalid flattened point for {level_name}")
            rows = torch.div(level_points, width, rounding_mode="floor")
            columns = level_points.remainder(width)
            semantic_y[mask] = ((rows.float() + 0.5) * (PADDED_IMAGE_SIZE[0] / height)).long().clamp_max(
                semantic.shape[1] - 1,
            )
            semantic_x[mask] = ((columns.float() + 0.5) * (PADDED_IMAGE_SIZE[1] / width)).long().clamp_max(
                semantic.shape[2] - 1,
            )
        point_probability = semantic[PERSON_SEMANTIC_CHANNEL, semantic_y, semantic_x]
        semantic_rois = torch.cat((torch.zeros_like(person_boxes[:, :1]), person_boxes), dim=1)
        box_probability = roi_pool(
            semantic[PERSON_SEMANTIC_CHANNEL].unsqueeze(0).unsqueeze(0),
            semantic_rois,
            output_size=(1, 1),
            spatial_scale=1.0,
        ).flatten()

        depth_probabilities = detections.get("depth_bin_probabilities")
        if depth_probabilities is None:
            raise KeyError("recovered detector must expose candidate depth-bin probabilities")
        person_depth = depth_probabilities.index_select(0, person_indices).float()
        if person_depth.ndim != 2 or person_depth.shape[1] < 2:
            raise ValueError("candidate depth probability shape drift")
        depth_max = person_depth.amax(dim=1)
        entropy = -(person_depth * person_depth.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(person_depth.shape[1])

        width = (person_boxes[:, 2] - person_boxes[:, 0]).clamp_min(SCORE_EPS)
        height = (person_boxes[:, 3] - person_boxes[:, 1]).clamp_min(SCORE_EPS)
        scalars = torch.stack((
            person_scores,
            person_levels.float() / float(len(LEVEL_NAMES) - 1),
            point_probability,
            box_probability,
            depth_max,
            normalized_entropy,
            width / float(PADDED_IMAGE_SIZE[1]),
            height / float(PADDED_IMAGE_SIZE[0]),
            (width * height) / float(PADDED_IMAGE_SIZE[0] * PADDED_IMAGE_SIZE[1]),
            torch.log(width / height),
        ), dim=1)
        if (roi_descriptors.shape != (person_indices.numel(), ROI_DESCRIPTOR_DIM)
                or scalars.shape != (person_indices.numel(), len(SCALAR_FEATURE_NAMES))
                or not bool(torch.isfinite(roi_descriptors).all())
                or not bool(torch.isfinite(scalars).all())):
            raise FloatingPointError("non-finite or malformed person ROI descriptor")
        return roi_descriptors, scalars, person_indices


class PersonVerifier(nn.Module):
    """The only trainable component in this package."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(FEATURE_DIM)
        self.hidden = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.output = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"expected [N,{FEATURE_DIM}] verifier input, got {tuple(features.shape)}")
        return self.output(self.activation(self.hidden(self.normalization(features)))).squeeze(1)


def refined_person_logits(
    base_scores: torch.Tensor, verifier_delta: torch.Tensor, calibration_bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    if base_scores.shape != verifier_delta.shape:
        raise ValueError("base-score and verifier-delta shapes differ")
    base = base_scores.float().clamp(SCORE_EPS, 1.0 - SCORE_EPS)
    bias = torch.as_tensor(calibration_bias, dtype=torch.float32, device=base.device)
    if bias.numel() != 1 or not bool(torch.isfinite(bias)):
        raise ValueError("calibration bias must be one finite scalar")
    return torch.logit(base) + verifier_delta.float() + bias


def refined_person_scores(
    base_scores: torch.Tensor, verifier_delta: torch.Tensor, calibration_bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Apply the deployment FP32 logit residual and sigmoid arithmetic."""
    return torch.sigmoid(refined_person_logits(base_scores, verifier_delta, calibration_bias))


def apply_person_refinement(
    detections: Mapping[str, torch.Tensor], verifier_delta: torch.Tensor,
    *, calibration_bias: float | torch.Tensor = 0.0,
) -> dict[str, torch.Tensor]:
    """Replace person scores in place-by-index while preserving every candidate."""
    classes = detections["labels_internal"].long()
    person_indices = torch.where(classes == PERSON_CLASS)[0]
    if verifier_delta.shape != person_indices.shape:
        raise ValueError("person verifier output count drift")
    refined = dict(detections)
    scores = detections["scores"].clone()
    person_base = scores.index_select(0, person_indices)
    scores[person_indices] = refined_person_scores(person_base, verifier_delta, calibration_bias)
    refined["scores"] = scores
    vehicle = classes != PERSON_CLASS
    if not torch.equal(refined["scores"][vehicle], detections["scores"][vehicle]):
        raise RuntimeError("vehicle score changed during person-only refinement")
    return refined


def build_verifier_optimizer(head: PersonVerifier) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    head_ids = {id(parameter) for parameter in head.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != head_ids:
        raise RuntimeError("optimizer is not exactly restricted to verifier parameters")
    return optimizer


def partition_experiment_ids(experiment_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unique = set(experiment_ids)
    if len(unique) != 10 or not HOLDOUT_EXPERIMENT_IDS.issubset(unique):
        raise RuntimeError("expected the canonical ten training episodes including both holdouts")
    if any("_train_" not in experiment_id for experiment_id in unique):
        raise RuntimeError("verifier split may contain only training episodes")
    fit = tuple(sorted(unique - HOLDOUT_EXPERIMENT_IDS))
    holdout = tuple(sorted(unique & HOLDOUT_EXPERIMENT_IDS))
    if len(fit) != 8 or len(holdout) != 2 or set(fit) & set(holdout):
        raise RuntimeError("verifier episode split is not exact and disjoint 8/2")
    return fit, holdout


def _metrics_at_threshold(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float, eligible_positive_count: int,
) -> dict[str, float | int]:
    predicted = scores >= float(threshold)
    positive = labels == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = eligible_positive_count - tp
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def exact_pr_report(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    eligible_positive_count: int | None = None,
    canonical_threshold: float = 0.20,
) -> dict[str, Any]:
    """Compute a tie-correct exact holdout PR frontier and joint intervals."""
    scores = scores.detach().double().cpu().flatten()
    labels = labels.detach().long().cpu().flatten()
    if scores.shape != labels.shape or scores.numel() == 0 or not bool(torch.isfinite(scores).all()):
        raise ValueError("holdout scores/labels are empty, non-finite, or misaligned")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("exact PR accepts only non-ignored binary labels")
    matched_positive = int((labels == 1).sum())
    total_positive = matched_positive if eligible_positive_count is None else int(eligible_positive_count)
    if total_positive < matched_positive:
        raise RuntimeError("eligible positive count is below matched positive candidate count")
    if total_positive == 0:
        raise RuntimeError("holdout contains no positive person candidates")

    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores = scores[order]
    ordered_positive = (labels[order] == 1).long()
    _values, group_counts = torch.unique_consecutive(ordered_scores, return_counts=True)
    group_ends = group_counts.cumsum(0) - 1
    tp = ordered_positive.cumsum(0)[group_ends]
    predicted = group_ends + 1
    fp = predicted - tp
    precision = tp.double() / predicted.double()
    recall = tp.double() / total_positive
    thresholds = ordered_scores[group_ends]
    joint = (precision >= 0.80) & (recall >= 0.80)

    feasible_intervals: list[dict[str, float]] = []
    feasible_indices = torch.where(joint)[0].tolist()
    if feasible_indices:
        runs: list[tuple[int, int]] = []
        start = previous = feasible_indices[0]
        for index in feasible_indices[1:]:
            if index != previous + 1:
                runs.append((start, previous))
                start = index
            previous = index
        runs.append((start, previous))
        tiny = torch.finfo(torch.float64).eps
        for start, end in runs:
            upper = float(thresholds[start])
            lower = float(thresholds[end])
            if not 0.0 <= lower <= upper <= 1.0:
                raise RuntimeError("feasible holdout score interval is outside [0,1]")
            lower_for_logit = min(1.0 - tiny, max(tiny, lower))
            upper_for_logit = min(1.0 - tiny, max(tiny, upper))
            midpoint_logit = 0.5 * (_logit(lower_for_logit) + _logit(upper_for_logit))
            feasible_intervals.append({
                "lower_score_inclusive": lower,
                "upper_score_inclusive": upper,
                "midpoint_logit": midpoint_logit,
                "midpoint_score": 1.0 / (1.0 + math.exp(-midpoint_logit)),
            })

    recall_mask = recall >= 0.80
    precision_mask = precision >= 0.80
    return {
        "candidates": scores.numel(),
        "eligible_positive": total_positive,
        "matched_positive_candidates": matched_positive,
        "negative": int((labels == 0).sum()),
        "at_0_20": _metrics_at_threshold(scores, labels, canonical_threshold, total_positive),
        "maximum_precision_at_recall_gte_0_80": float(precision[recall_mask].max()) if bool(recall_mask.any()) else 0.0,
        "maximum_recall_at_precision_gte_0_80": float(recall[precision_mask].max()) if bool(precision_mask.any()) else 0.0,
        "joint_precision_recall_0_80_exists": bool(joint.any()),
        "feasible_intervals": feasible_intervals,
        "selected_interval": feasible_intervals[0] if feasible_intervals else None,
        "selected_interval_rule": "highest-score contiguous feasible interval",
    }


def calibration_bias_for_interval(interval: Mapping[str, float], canonical_threshold: float = 0.20) -> float:
    midpoint_logit = float(interval["midpoint_logit"])
    bias = _logit(float(canonical_threshold)) - midpoint_logit
    if not math.isfinite(bias):
        raise FloatingPointError("non-finite person calibration bias")
    return bias

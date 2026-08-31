from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    assign_components,
    connected_person_components,
    person_mask_from_logits,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    PersonRoIDescriptor,
    fp16_round_trip_roi_descriptors,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
)

from .cache_join import PERSON_INTERNAL_CLASS, build_relational_features
from .provenance import (
    CONSOLIDATION_MANIFEST_SHA256,
    FIT_EXPERIMENT_IDS,
    FROZEN_CHECKPOINT_SHA256,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_MANIFEST_SHA256,
    load_locked_config,
)
from .selector import ARCHITECTURE, PersonRelationalSelector, refined_person_scores

CANONICAL_PERSON_THRESHOLD = 0.20
COUNT_FIELDS = ("tp", "fp", "fn", "ignored")
THRESHOLD_SOURCE = "one fixed epoch and one joint two-episode holdout threshold"


def _passes_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(float(metrics.get("precision", -1.0)) >= 0.80
                and float(metrics.get("recall", -1.0)) >= 0.80)


def _bounded_logit(value: float) -> float:
    epsilon = 2.220446049250313e-16
    clamped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _valid_metrics(metrics: Any, expected_threshold: float) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    try:
        threshold = float(metrics["threshold"])
        counts = {name: int(metrics[name]) for name in COUNT_FIELDS}
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return (all(value >= 0 for value in counts.values())
            and math.isclose(threshold, float(expected_threshold), rel_tol=0.0, abs_tol=1e-12)
            and math.isfinite(precision) and math.isfinite(recall)
            and math.isclose(precision, tp / max(1, tp + fp), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(recall, tp / max(1, tp + fn), rel_tol=0.0, abs_tol=1e-12))


def _valid_metric_bundle(bundle: Any, expected_threshold: float) -> bool:
    if not isinstance(bundle, Mapping):
        return False
    aggregate = bundle.get("aggregate")
    episodes = bundle.get("episodes")
    if (not isinstance(episodes, Mapping)
            or set(episodes) != set(HOLDOUT_EXPERIMENT_IDS)
            or not _valid_metrics(aggregate, expected_threshold)
            or not all(_valid_metrics(episodes[name], expected_threshold)
                       for name in HOLDOUT_EXPERIMENT_IDS)):
        return False
    return all(
        int(aggregate[field]) == sum(int(episodes[name][field]) for name in HOLDOUT_EXPERIMENT_IDS)
        for field in COUNT_FIELDS
    )


def _same_counts(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    reports = [(left["aggregate"], right["aggregate"])] + [
        (left["episodes"][name], right["episodes"][name]) for name in HOLDOUT_EXPERIMENT_IDS
    ]
    return all(all(int(a[field]) == int(b[field]) for field in COUNT_FIELDS) for a, b in reports)


def _fixed_training_contract(training: Any) -> bool:
    if not isinstance(training, Mapping):
        return False
    fit = training.get("fit_episodes")
    holdout = training.get("holdout_episodes")
    losses = training.get("epoch_losses")
    sampling = training.get("epoch_sampling")
    if (fit != list(FIT_EXPERIMENT_IDS)
            or holdout != list(HOLDOUT_EXPERIMENT_IDS)
            or training.get("epochs") != 5
            or training.get("selected_epoch") != 5
            or training.get("batch_frames") != 16
            or training.get("optimizer") != "Adam"
            or training.get("learning_rate") != 1e-3
            or training.get("positive_to_negative_loss_sampling") != "1:3"
            or training.get("all_candidates_retained_in_attention_context") is not True
            or training.get("ignored_labels_excluded_from_loss") is not True
            or training.get("sampling_plan_scans") != 1
            or training.get("seed") != 20260831
            or not isinstance(losses, list) or len(losses) != 5
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in losses)
            or not isinstance(sampling, list) or len(sampling) != 5):
        return False
    try:
        pairs = [(int(value["positive"]), int(value["negative"])) for value in sampling]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (len(set(pairs)) == 1
            and all(positive > 0 and negative == 3 * positive for positive, negative in pairs))


def _calibration_contract(holdout: Any) -> bool:
    if not isinstance(holdout, Mapping) or holdout.get("threshold_source") != THRESHOLD_SOURCE:
        return False
    before = holdout.get("before_calibration")
    interval = holdout.get("joint_feasible_interval")
    selected = holdout.get("selected_threshold_metrics")
    deployment = holdout.get("deployment_at_0_20")
    if (not isinstance(before, Mapping)
            or before.get("candidate_scores_computed_once") is not True
            or before.get("tie_processing") != "all_equal_scores_added_before_affected_frames_are_rematched"
            or before.get("joint_precision_recall_0_80_exists") is not True
            or not isinstance(interval, Mapping)
            or before.get("selected_interval") != interval):
        return False
    try:
        score_boundaries = int(before["score_boundaries"])
        lower = float(interval["lower_score_exclusive"])
        upper = float(interval["upper_score_inclusive"])
        midpoint_logit = float(interval["midpoint_logit"])
        selected_threshold = float(interval["selected_threshold"])
        attempted_bias = float(holdout["attempted_calibration_bias"])
        calibration_bias = float(holdout["calibration_bias"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not 0.0 <= lower < selected_threshold <= upper <= 1.0:
        return False
    expected_midpoint_logit = 0.5 * (_bounded_logit(lower) + _bounded_logit(upper))
    expected_threshold = _sigmoid(midpoint_logit)
    expected_bias = _bounded_logit(CANONICAL_PERSON_THRESHOLD) - midpoint_logit
    if (score_boundaries <= 0
            or not all(math.isfinite(value) for value in (
                midpoint_logit, selected_threshold, attempted_bias, calibration_bias,
            ))
            or not math.isclose(midpoint_logit, expected_midpoint_logit, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(selected_threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(attempted_bias, expected_bias, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(calibration_bias, attempted_bias, rel_tol=0.0, abs_tol=1e-12)
            or not _valid_metric_bundle(selected, selected_threshold)
            or not _valid_metric_bundle(deployment, CANONICAL_PERSON_THRESHOLD)
            or not _passes_gate(selected["aggregate"])
            or not all(_passes_gate(selected["episodes"][name]) for name in HOLDOUT_EXPERIMENT_IDS)
            or not _passes_gate(deployment["aggregate"])
            or not all(_passes_gate(deployment["episodes"][name]) for name in HOLDOUT_EXPERIMENT_IDS)
            or holdout.get("selected_deployment_counts_agree") is not True
            or not _same_counts(selected, deployment)):
        return False
    return True


def _validate_selector_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if (checkpoint.get("schema") != "splitfusion_fcos_person_relational_selector_v1"
            or checkpoint.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or checkpoint.get("roi_manifest_sha256") != ROI_MANIFEST_SHA256
            or checkpoint.get("consolidation_manifest_sha256") != CONSOLIDATION_MANIFEST_SHA256
            or checkpoint.get("architecture") != ARCHITECTURE
            or not _fixed_training_contract(checkpoint.get("training"))
            or not _calibration_contract(checkpoint.get("holdout"))
            or checkpoint.get("status") != "train_feasible"
            or checkpoint.get("validation_allowed") is not True
            or checkpoint.get("validation_or_test_accessed") is not False):
        raise RuntimeError("relational-selector checkpoint provenance or feasibility contract drift")


def load_selector_checkpoint(
    path: Path, device: torch.device,
) -> tuple[PersonRelationalSelector, float]:
    """Load only a selector whose fixed holdout threshold passed every gate."""
    load_locked_config()
    checkpoint = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=True)
    _validate_selector_checkpoint(checkpoint)
    calibration_bias = float(checkpoint["holdout"]["calibration_bias"])
    if not torch.isfinite(torch.tensor(calibration_bias, dtype=torch.float32)):
        raise FloatingPointError("non-finite relational-selector calibration bias")
    selector = PersonRelationalSelector()
    selector.load_state_dict(checkpoint["selector"], strict=True)
    selector.to(device).eval()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in selector.parameters()):
        raise FloatingPointError("non-finite relational-selector checkpoint parameter")
    return selector, calibration_bias


def extract_live_person_features(
    outputs: Mapping[str, Any],
    detections: Mapping[str, torch.Tensor],
    extractor: PersonRoIDescriptor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recreate the cache representation and relational features without filtering."""
    descriptors, scalars, person_indices = extractor(outputs, detections)
    descriptors = fp16_round_trip_roi_descriptors(descriptors)
    cached_features = torch.cat((descriptors, scalars.float()), dim=1)
    person_boxes = detections["boxes"].index_select(0, person_indices)
    person_world = detections["world_xyz"].index_select(0, person_indices)[:, :2]
    person_scores = detections["scores"].index_select(0, person_indices)
    component_labels, _count = connected_person_components(
        person_mask_from_logits(outputs["semantic_logits"]),
    )
    component_ids, support = assign_components(component_labels, person_boxes)
    features = build_relational_features(
        cached_features,
        person_boxes,
        person_world,
        component_ids,
        support,
        person_scores,
        person_indices,
    )
    return features, person_indices


def select_live_person_scores(
    selector: PersonRelationalSelector,
    features: torch.Tensor,
    base_scores: torch.Tensor,
    *,
    calibration_bias: float,
) -> torch.Tensor:
    count = int(base_scores.numel())
    if features.shape[0] != count:
        raise RuntimeError("live feature/base-score count drift")
    if count == 0:
        return base_scores.float()
    padding = torch.zeros((1, count), dtype=torch.bool, device=features.device)
    residual = selector(features.unsqueeze(0), padding)[0]
    return refined_person_scores(base_scores, residual, calibration_bias)


def apply_relational_service_policy(
    detections: Mapping[str, torch.Tensor],
    person_indices: torch.Tensor,
    person_scores: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Use relational person selection while preserving locked vehicle behavior."""
    base_scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = int(base_scores.numel())
    if (base_scores.dtype != torch.float32 or classes.shape != (count,)
            or any(not isinstance(value, torch.Tensor) or value.shape[0] != count
                   for value in detections.values())):
        raise RuntimeError("frozen post-NMS detection alignment drift")
    expected_person = torch.where(classes == PERSON_INTERNAL_CLASS)[0]
    if (not torch.equal(person_indices.to(expected_person.device).long(), expected_person)
            or person_scores.shape != expected_person.shape
            or not bool(torch.isfinite(person_scores).all())):
        raise RuntimeError("relational person score/order drift")

    keep_mask = classes != PERSON_INTERNAL_CLASS
    keep_mask = keep_mask.clone()
    keep_mask[expected_person] = person_scores.to(keep_mask.device) >= CANONICAL_PERSON_THRESHOLD
    keep = torch.where(keep_mask)[0]
    result = {name: value.index_select(0, keep) for name, value in detections.items()}
    retained_classes = result["labels_internal"].long()
    vehicle_positions = torch.where(retained_classes != PERSON_INTERNAL_CLASS)[0]
    person_positions = torch.where(retained_classes == PERSON_INTERNAL_CLASS)[0]
    retained_person_source = keep.index_select(0, person_positions)

    scores = base_scores.index_select(0, keep).clone()
    vehicle_source = keep.index_select(0, vehicle_positions)
    scores[vehicle_positions] = calibrate_vehicle_scores(base_scores.index_select(0, vehicle_source))
    if retained_person_source.numel():
        source_to_person_position = torch.full(
            (count,), -1, dtype=torch.long, device=expected_person.device,
        )
        source_to_person_position[expected_person] = torch.arange(
            expected_person.numel(), device=expected_person.device,
        )
        scores[person_positions] = person_scores.index_select(
            0, source_to_person_position.index_select(0, retained_person_source),
        ).to(scores.device)
    result["scores"] = scores

    original_vehicle = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    if (not torch.equal(vehicle_source, original_vehicle)
            or not torch.equal(result["scores"].index_select(0, vehicle_positions),
                               calibrate_vehicle_scores(base_scores.index_select(0, original_vehicle)))):
        raise RuntimeError("vehicle behavior differs from locked service-candidate runtime")
    for name, value in detections.items():
        if name != "scores" and not torch.equal(result[name], value.index_select(0, keep)):
            raise RuntimeError(f"non-score detection field changed: {name}")
    return result, keep

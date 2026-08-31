from __future__ import annotations

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
from .provenance import FROZEN_CHECKPOINT_SHA256, HOLDOUT_EXPERIMENT_IDS, load_locked_config
from .selector import ARCHITECTURE, PersonRelationalSelector, refined_person_scores

CANONICAL_PERSON_THRESHOLD = 0.20


def _passes_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(float(metrics.get("precision", -1.0)) >= 0.80
                and float(metrics.get("recall", -1.0)) >= 0.80)


def load_selector_checkpoint(
    path: Path, device: torch.device,
) -> tuple[PersonRelationalSelector, float]:
    """Load only a selector whose fixed holdout threshold passed every gate."""
    load_locked_config()
    checkpoint = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=True)
    deployment = checkpoint.get("holdout", {}).get("deployment_at_0_20")
    episodes = deployment.get("episodes", {}) if isinstance(deployment, Mapping) else {}
    aggregate = deployment.get("aggregate", {}) if isinstance(deployment, Mapping) else {}
    allowed = (_passes_gate(aggregate)
               and set(episodes) == set(HOLDOUT_EXPERIMENT_IDS)
               and all(_passes_gate(episodes[name]) for name in HOLDOUT_EXPERIMENT_IDS))
    if (checkpoint.get("schema") != "splitfusion_fcos_person_relational_selector_v1"
            or checkpoint.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("status") != "train_feasible"
            or checkpoint.get("validation_allowed") is not True
            or checkpoint.get("validation_or_test_accessed") is not False
            or not allowed):
        raise RuntimeError("relational selector is not holdout-feasible; inference is prohibited")
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

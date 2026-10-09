from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    PERSON_INTERNAL_CLASS,
    retained_detection_indices,
)

from .provenance import (
    PERSON_RULE,
    VEHICLE_CLAMP_EPSILON,
    VEHICLE_LOGIT_BIAS,
)


def calibrate_vehicle_scores(base_scores: torch.Tensor) -> torch.Tensor:
    """Apply the locked monotonic calibration with deployment FP32 arithmetic."""
    if not base_scores.is_floating_point() or not bool(torch.isfinite(base_scores).all()):
        raise FloatingPointError("vehicle base scores must be finite floating-point values")
    scores = base_scores.float()
    bias = torch.tensor(VEHICLE_LOGIT_BIAS, dtype=torch.float32, device=scores.device)
    calibrated = torch.sigmoid(torch.logit(scores.clamp(
        min=VEHICLE_CLAMP_EPSILON, max=1.0 - VEHICLE_CLAMP_EPSILON,
    )) + bias)
    if calibrated.dtype != torch.float32 or not bool(torch.isfinite(calibrated).all()):
        raise FloatingPointError("non-finite or non-FP32 calibrated vehicle score")
    return calibrated


def apply_combined_service_policy(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Consolidate persons and calibrate vehicles without another NMS or reordering."""
    scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = scores.numel()
    if scores.dtype != torch.float32 or classes.shape != (count,):
        raise ValueError("frozen post-NMS score/class contract drift")
    if any(not isinstance(value, torch.Tensor) or value.shape[0] != count for value in detections.values()):
        raise ValueError("frozen post-NMS detection field alignment drift")

    keep = retained_detection_indices(outputs, detections, PERSON_RULE)
    if (keep.ndim != 1 or keep.dtype != torch.long
            or (keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()))):
        raise RuntimeError("reviewed person consolidation changed original candidate ordering")
    result = {name: value.index_select(0, keep) for name, value in detections.items()}
    retained_classes = result["labels_internal"].long()
    vehicle_positions = torch.where(retained_classes != PERSON_INTERNAL_CLASS)[0]
    person_positions = torch.where(retained_classes == PERSON_INTERNAL_CLASS)[0]
    original_vehicle_indices = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    retained_vehicle_indices = keep.index_select(0, vehicle_positions)
    if not torch.equal(retained_vehicle_indices, original_vehicle_indices):
        raise RuntimeError("vehicle candidate was filtered or reordered")

    original_selected_scores = scores.index_select(0, keep)
    combined_scores = original_selected_scores.clone()
    combined_scores[vehicle_positions] = calibrate_vehicle_scores(
        original_selected_scores.index_select(0, vehicle_positions),
    )
    result["scores"] = combined_scores
    for name, value in detections.items():
        if name != "scores" and not torch.equal(result[name], value.index_select(0, keep)):
            raise RuntimeError(f"non-authorized detection field changed: {name}")
    if not torch.equal(result["scores"].index_select(0, person_positions),
                       original_selected_scores.index_select(0, person_positions)):
        raise RuntimeError("retained person score changed")
    return result, keep


def combined_records(
    base: Any,
    row: dict[str, str],
    detections: Mapping[str, torch.Tensor],
    original_indices: torch.Tensor,
) -> list[dict[str, Any]]:
    """Serialize retained records with original post-NMS prediction indices."""
    count = detections["scores"].numel()
    indices = original_indices.detach().long().cpu()
    if (indices.shape != (count,) or bool((indices < 0).any())
            or (count > 1 and not bool((indices[1:] > indices[:-1]).all()))):
        raise ValueError("retained original prediction-index contract drift")
    records = []
    for retained_index, original_index in enumerate(indices.tolist()):
        record = dict(base.infer.record(dict(detections), row, retained_index))
        record["prediction_index"] = int(original_index)
        records.append(record)
    return records

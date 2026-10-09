from __future__ import annotations

from collections.abc import Mapping

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    PERSON_INTERNAL_CLASS,
)


PERSON_SCORE_THRESHOLD = 0.25


def filter_consolidated_person_outputs(
    detections: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Drop only consolidated person rows below 0.25, preserving every value."""
    if "scores" not in detections or "labels_internal" not in detections:
        raise ValueError("scores and labels_internal are required")
    scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = scores.numel()
    if scores.dtype != torch.float32 or classes.shape != (count,):
        raise ValueError("frozen consolidated score/class contract drift")
    if any(
        not isinstance(value, torch.Tensor) or value.shape[0] != count
        for value in detections.values()
    ):
        raise ValueError("consolidated detection field alignment drift")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite consolidated detection score")

    person = classes == PERSON_INTERNAL_CLASS
    keep = torch.where(~person | (scores >= PERSON_SCORE_THRESHOLD))[0]
    result = {name: value.index_select(0, keep) for name, value in detections.items()}

    vehicle = torch.where(~person)[0]
    retained_classes = result["labels_internal"].long()
    retained_vehicle_positions = torch.where(
        retained_classes != PERSON_INTERNAL_CLASS
    )[0]
    retained_vehicle_indices = keep.index_select(0, retained_vehicle_positions)
    if not torch.equal(retained_vehicle_indices, vehicle):
        raise RuntimeError("person threshold filtered or reordered a vehicle")
    for name, value in detections.items():
        if not torch.equal(
            result[name].index_select(0, retained_vehicle_positions),
            value.index_select(0, vehicle),
        ):
            raise RuntimeError(f"person threshold changed vehicle field: {name}")
        if not torch.equal(result[name], value.index_select(0, keep)):
            raise RuntimeError(f"person threshold changed retained field: {name}")
    return result, keep

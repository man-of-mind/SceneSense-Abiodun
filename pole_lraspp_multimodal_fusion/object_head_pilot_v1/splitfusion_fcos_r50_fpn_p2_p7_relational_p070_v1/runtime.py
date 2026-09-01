from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.runtime import (
    apply_relational_service_policy,
    extract_live_person_features,
    select_live_person_scores,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    PersonRoIDescriptor,
)

from .contract import RevisedSelectorRuntime


def apply_relational_p070_policy(
    outputs: Mapping[str, Any],
    detections: Mapping[str, torch.Tensor],
    runtime: RevisedSelectorRuntime,
    extractor: PersonRoIDescriptor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Score persons relationally; preserve service vehicles and all non-score fields."""
    features, person_indices = extract_live_person_features(outputs, detections, extractor)
    base_person_scores = detections["scores"].index_select(0, person_indices)
    scores = select_live_person_scores(
        runtime.selector,
        features,
        base_person_scores,
        calibration_bias=runtime.deployment_bias,
    )
    return apply_relational_service_policy(detections, person_indices, scores)

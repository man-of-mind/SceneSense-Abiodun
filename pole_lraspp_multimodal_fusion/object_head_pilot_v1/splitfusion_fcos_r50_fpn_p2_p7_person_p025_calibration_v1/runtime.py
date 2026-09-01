from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    apply_combined_service_policy,
)

from .policy import filter_consolidated_person_outputs
from .provenance import load_candidate_contract


_LOCKED_CONTRACT = load_candidate_contract()


def apply_p025_service_policy(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the frozen p020 service, then remove only person scores below 0.25."""
    consolidated, p020_indices = apply_combined_service_policy(outputs, detections)
    filtered, positions = filter_consolidated_person_outputs(consolidated)
    p025_indices = p020_indices.index_select(0, positions.to(p020_indices.device))
    if not torch.equal(p025_indices, p020_indices.index_select(0, positions)):
        raise RuntimeError("p025 original-index subset drift")
    return filtered, p025_indices

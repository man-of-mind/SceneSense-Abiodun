"""Parameter-free person instance consolidation for frozen SplitFusion FCOS."""

from .core import (
    GROUP_IOU_THRESHOLDS,
    HOLDOUT_EXPERIMENT_IDS,
    SEMANTIC_SUPPORT_THRESHOLDS,
    apply_rule_to_detections,
    assign_components,
    connected_person_components,
    consolidate_person_candidates,
    evaluate_frames,
    grid_configurations,
    partition_experiment_ids,
    partition_frames,
)

__all__ = (
    "GROUP_IOU_THRESHOLDS",
    "HOLDOUT_EXPERIMENT_IDS",
    "SEMANTIC_SUPPORT_THRESHOLDS",
    "apply_rule_to_detections",
    "assign_components",
    "connected_person_components",
    "consolidate_person_candidates",
    "evaluate_frames",
    "grid_configurations",
    "partition_experiment_ids",
    "partition_frames",
)

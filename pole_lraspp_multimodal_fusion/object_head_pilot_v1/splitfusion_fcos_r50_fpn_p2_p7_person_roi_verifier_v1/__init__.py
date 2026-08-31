"""Frozen-base, person-only ROI verifier for SplitFusion FCOS."""

from .verifier import (
    FEATURE_DIM,
    HOLDOUT_EXPERIMENT_IDS,
    PersonRoIDescriptor,
    PersonVerifier,
    apply_person_refinement,
    build_verifier_optimizer,
    partition_experiment_ids,
    refined_person_logits,
)

__all__ = (
    "FEATURE_DIM",
    "HOLDOUT_EXPERIMENT_IDS",
    "PersonRoIDescriptor",
    "PersonVerifier",
    "apply_person_refinement",
    "build_verifier_optimizer",
    "partition_experiment_ids",
    "refined_person_logits",
)

"""Frozen-base person-only per-frame relational candidate selector."""

from .selector import (
    ARCHITECTURE,
    INPUT_DIM,
    MAX_CANDIDATES_PER_FRAME,
    PersonRelationalSelector,
    build_selector_optimizer,
    refined_person_logits,
    refined_person_scores,
)

__all__ = (
    "ARCHITECTURE",
    "INPUT_DIM",
    "MAX_CANDIDATES_PER_FRAME",
    "PersonRelationalSelector",
    "build_selector_optimizer",
    "refined_person_logits",
    "refined_person_scores",
)

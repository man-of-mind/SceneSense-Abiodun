"""Reviewed-contract wrapper for the frozen relational person selector."""

from .contract import (
    CANONICAL_PERSON_THRESHOLD,
    DEPLOYMENT_LOGIT_BIAS,
    RAW_RELATIONAL_THRESHOLD,
    load_revised_selector,
)

__all__ = [
    "CANONICAL_PERSON_THRESHOLD",
    "DEPLOYMENT_LOGIT_BIAS",
    "RAW_RELATIONAL_THRESHOLD",
    "load_revised_selector",
]

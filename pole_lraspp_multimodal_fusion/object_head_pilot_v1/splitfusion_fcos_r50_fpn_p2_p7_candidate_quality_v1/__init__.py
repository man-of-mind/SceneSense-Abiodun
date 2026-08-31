"""Frozen SplitFusion FCOS candidate-quality refinement, version 1."""

from .quality import FEATURE_DIM, QualityMLP, refine_scores

__all__ = ("FEATURE_DIM", "QualityMLP", "refine_scores")

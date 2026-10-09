"""Offline exact scorer candidate; no live integration or wire authority."""

from .scorer import NumericalScorer, ScoreResult, reconstruct_evidence, archive_payloads

__all__ = ["NumericalScorer", "ScoreResult", "reconstruct_evidence", "archive_payloads"]

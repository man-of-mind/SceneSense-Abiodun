"""Hybrid-q transport-only extension at the frozen SplitFusion-FCOS C2 split.

Phase 1 is implementation only: ranker, exact q semantics, sparse wire codec,
training-path primitives and fail-closed guards. Nothing here loads the frozen
checkpoint, touches real data or trains.
"""

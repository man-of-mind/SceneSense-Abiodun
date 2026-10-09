"""Fail-closed exact offline SplitFusion quality-grid producer.

This package is deliberately inert on import.  In particular it never opens a
dataset, imports torch, queries CUDA, or loads a checkpoint.  Use ``cli`` for
metadata preflight, deterministic selection, and an explicitly authorized run.
"""

from .contract import (
    EVIDENCE_LABEL,
    EXPECTED_GRID_ROWS,
    FAMILIES,
    Q_E4_GRID,
    QUANTIZERS,
)

__all__ = [
    "EVIDENCE_LABEL",
    "EXPECTED_GRID_ROWS",
    "FAMILIES",
    "Q_E4_GRID",
    "QUANTIZERS",
]

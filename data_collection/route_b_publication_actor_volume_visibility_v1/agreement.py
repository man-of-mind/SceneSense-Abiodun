"""Ordinal agreement statistics against the human visibility bands.

Only non-``ambiguous`` annotator-A rows enter any agreement score.  The AI
annotation is never consulted here; it stays diagnostic-only elsewhere.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import BAND_ORDER, BINARY_DECISION_THRESHOLD


def confusion_matrix(
    human: Sequence[str], auto: Sequence[str], labels: Sequence[str] = BAND_ORDER
) -> np.ndarray:
    """Rows are human bands, columns are automatic bands, in ordinal order."""
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for h, a in zip(human, auto):
        matrix[index[h], index[a]] += 1
    return matrix


def exact_agreement(matrix: np.ndarray) -> float:
    total = int(matrix.sum())
    return float(np.trace(matrix)) / total if total else float("nan")


def linear_weighted_kappa(matrix: np.ndarray) -> float:
    """Cohen's kappa with linear disagreement weights on the ordinal bands."""
    total = float(matrix.sum())
    if total <= 0.0:
        return float("nan")
    observed = matrix.astype(np.float64) / total
    rows = observed.sum(axis=1, keepdims=True)
    cols = observed.sum(axis=0, keepdims=True)
    expected = rows @ cols
    size = matrix.shape[0]
    idx = np.arange(size, dtype=np.float64)
    weights = np.abs(idx[:, None] - idx[None, :]) / float(size - 1)
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return float("nan")
    return 1.0 - numerator / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    from scipy import stats

    result = stats.spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return float(result.statistic), float(result.pvalue)


def binary_confusion(
    human_positive: Sequence[bool], auto_positive: Sequence[bool]
) -> dict[str, int]:
    h = np.asarray(list(human_positive), dtype=bool)
    a = np.asarray(list(auto_positive), dtype=bool)
    return {
        "tp": int(np.count_nonzero(h & a)),
        "fn": int(np.count_nonzero(h & ~a)),
        "fp": int(np.count_nonzero(~h & a)),
        "tn": int(np.count_nonzero(~h & ~a)),
    }


def balanced_accuracy(counts: dict[str, int]) -> float:
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    if positives == 0 or negatives == 0:
        return float("nan")
    sensitivity = counts["tp"] / positives
    specificity = counts["tn"] / negatives
    return float(0.5 * (sensitivity + specificity))


def human_band_is_visible(band: str) -> bool:
    """Human counterpart of the automatic >=0.65 decision."""
    return band in ("partial_65_90", "bare_90_100")


def evaluate(
    human_bands: Sequence[str],
    scores: Sequence[float],
    auto_bands: Sequence[str],
    *,
    threshold: float = BINARY_DECISION_THRESHOLD,
) -> dict[str, object]:
    matrix = confusion_matrix(human_bands, auto_bands)
    human_rank = [BAND_ORDER.index(b) for b in human_bands]
    rho, pvalue = spearman(human_rank, scores)
    binary = binary_confusion(
        [human_band_is_visible(b) for b in human_bands],
        [float(s) >= float(threshold) for s in scores],
    )
    return {
        "n": int(len(human_bands)),
        "labels": list(BAND_ORDER),
        "confusion_matrix": matrix.tolist(),
        "exact_agreement": exact_agreement(matrix),
        "linear_weighted_cohen_kappa": linear_weighted_kappa(matrix),
        "spearman_rho": rho,
        "spearman_p_value": pvalue,
        "binary_threshold": float(threshold),
        "binary_confusion": binary,
        "balanced_accuracy": balanced_accuracy(binary),
    }

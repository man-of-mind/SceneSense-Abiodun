from __future__ import annotations

from dataclasses import dataclass

import torch

from . import contract, guards


@dataclass(frozen=True)
class CellSelection:
    """Result of applying exact q semantics to one frame of ranker scores."""

    q: float
    cells: int
    keep_count: int
    drop_count: int
    keep_indices: torch.Tensor  # int64, strictly ascending row-major cell indices
    keep_mask: torch.Tensor  # bool [H, W]


def _select_cells(
    scores: torch.Tensor, q: float, *, registered_only: bool = True
) -> CellSelection:
    """Private generic selection over any [H,W] score map (tests and internals)."""
    value = guards.require_valid_q(q, registered_only=registered_only)
    if scores.dim() != 2:
        raise guards.HybridQPayloadError(
            f"scores must be [H,W], got {list(scores.shape)}"
        )
    guards.require_finite(scores, "ranker scores")

    cells = int(scores.numel())
    keep = contract.keep_count(value, cells)
    drop = contract.drop_count(value, cells)
    flat = scores.reshape(-1).detach().to(torch.float32)

    if keep == cells:
        keep_indices = torch.arange(cells, dtype=torch.int64, device=scores.device)
    else:
        order = torch.argsort(flat, descending=True, stable=True)
        keep_indices = torch.sort(order[:keep]).values.to(torch.int64)

    guards.require_keep_cardinality(int(keep_indices.numel()), keep)
    guards.require_sorted_unique_indices(keep_indices, cells)

    keep_mask = torch.zeros(cells, dtype=torch.bool, device=scores.device)
    keep_mask[keep_indices] = True
    return CellSelection(
        q=value,
        cells=cells,
        keep_count=keep,
        drop_count=drop,
        keep_indices=keep_indices,
        keep_mask=keep_mask.reshape(scores.shape),
    )


def select_cells(scores: torch.Tensor, q: float) -> CellSelection:
    """Production selection over one frozen frame of scores: [112,192], registered q.

    Keeps the highest-scoring cells; ties prefer the lower row-major index. A
    stable descending sort of the row-major flattened scores makes this
    deterministic and repeatable.
    """
    guards.require_frozen_scores(scores)
    value = guards.require_valid_q(q)
    selection = _select_cells(scores, value)
    guards.require_selection_integrity(
        selection,
        value,
        cells=contract.SPLIT_CELLS,
        spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
    )
    return selection


def _apply_selection(c2: torch.Tensor, selection: CellSelection) -> torch.Tensor:
    """Private generic masking: zero every dropped cell across all channels."""
    guards.require_generic_c2(c2, channels=int(c2.shape[0]), what="C2 tensor")
    if tuple(selection.keep_mask.shape) != tuple(c2.shape[1:]):
        raise guards.HybridQPayloadError("selection mask shape does not match C2 tensor")
    mask = selection.keep_mask.to(device=c2.device).unsqueeze(0)
    return c2 * mask.to(c2.dtype)


def apply_selection(c2: torch.Tensor, selection: CellSelection) -> torch.Tensor:
    """Production masking at the frozen boundary.

    All 256 channels of a spatial cell are retained or removed together;
    dropped cells become exact zeros.
    """
    guards.require_frozen_c2(c2)
    guards.require_selection_integrity(
        selection,
        selection.q,
        cells=contract.SPLIT_CELLS,
        spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
    )
    return _apply_selection(c2, selection)


def select_and_apply(
    c2: torch.Tensor, ranker, q: float
) -> tuple[torch.Tensor, CellSelection | None]:
    """Full production masking path at the frozen boundary.

    q=0 validates the tensor, then bypasses ranking and masking entirely and
    returns the input tensor object itself, so dense identity is exact by
    construction. Callers must not mutate the returned tensor in place.
    """
    guards.require_frozen_c2(c2)
    value = guards.require_valid_q(q)
    if contract.drop_count(value, contract.SPLIT_CELLS) == 0:
        return c2, None
    scores = ranker.score_cells(c2)
    selection = select_cells(scores, value)
    return apply_selection(c2, selection), selection

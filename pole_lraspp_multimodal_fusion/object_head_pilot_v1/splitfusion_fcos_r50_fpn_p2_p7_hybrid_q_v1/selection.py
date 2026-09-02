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


def select_cells(
    scores: torch.Tensor, q: float, *, registered_only: bool = True
) -> CellSelection:
    """Keep the highest-scoring cells; ties prefer the lower row-major index.

    `scores` is [H, W]. Selection is deterministic: a stable descending sort
    over the row-major flattened scores means equal scores are taken in
    ascending index order.
    """
    value = guards.require_valid_q(q, registered_only=registered_only)
    if scores.dim() != 2:
        raise guards.HybridQPayloadError(
            f"scores must be [H,W], got {tuple(scores.shape)}"
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


def apply_selection(c2: torch.Tensor, selection: CellSelection) -> torch.Tensor:
    """Zero every dropped cell across all 256 channels; retained cells are exact.

    All channels of a spatial cell are retained or removed together.
    """
    guards.require_c2_tensor(c2, channels=c2.shape[0], what="C2 tensor")
    if selection.keep_mask.shape != c2.shape[1:]:
        raise guards.HybridQPayloadError("selection mask shape does not match C2 tensor")
    mask = selection.keep_mask.to(device=c2.device).unsqueeze(0)
    return c2 * mask.to(c2.dtype)


def select_and_apply(
    c2: torch.Tensor,
    ranker,
    q: float,
    *,
    registered_only: bool = True,
) -> tuple[torch.Tensor, CellSelection | None]:
    """Full masking path. q=0 bypasses ranking and returns the dense tensor.

    At q=0 the ranker is never invoked and the returned tensor is the input
    object itself, so dense identity is exact by construction.
    """
    value = guards.require_valid_q(q, registered_only=registered_only)
    if contract.drop_count(value, int(c2.shape[1] * c2.shape[2])) == 0:
        return c2, None
    scores = ranker.score_cells(c2)
    selection = select_cells(scores, value, registered_only=registered_only)
    return apply_selection(c2, selection), selection

"""Composition of the frozen ranker, the AE encoder and the spatial keep mask.

The registered order is fixed and this module is the only place that expresses
it:

  1. the ranker scores the **original FP32 C2**, never the latent;
  2. the AE encoder runs on the complete frame, **before** spatial dropping;
  3. the resulting keep indices drop latent cells, all B channels of a retained
     cell together.

Because the ranker never sees the latent, one q-independent cell ordering
induces the keep set for *every* AE family: AE128, AE64 and AE32 transport
exactly the same cells at the same q. q=0 bypasses the ranker (there is nothing
to rank when nothing is dropped) but never bypasses the AE, and the reconstruction
at q=0 is therefore **not** an identity - channel compression is lossy.

Selection, q semantics and the 1e-4 wire grid are reused unchanged from the
frozen `continuous_q` interface, so any q in [0, 0.98] is mechanically
supported here. Executability is not measured accuracy: q=0.90 and q=0.98
remain evaluation/emergency values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, continuous_q, guards
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.selection import CellSelection
from . import ae_contract
from .ae_model import SplitFeatureAE


@dataclass(frozen=True)
class LatentComposition:
    """One frame resolved at one q, ready for the AE latent wire."""

    plan: continuous_q.ContinuousQ
    bottleneck: int
    latent: torch.Tensor  # dense [B,112,192] FP32, before dropping
    selection: CellSelection | None  # None exactly at q=0
    keep_mask: torch.Tensor  # bool [112,192]; all True at q=0
    masked_latent: torch.Tensor  # zero-scattered [B,112,192] FP32

    @property
    def keep_count(self) -> int:
        return int(self.plan.keep_count)

    @property
    def ranker_used(self) -> bool:
        return self.selection is not None


def all_keep_mask(
    *, device: torch.device | str = "cpu", frames: int | None = None
) -> torch.Tensor:
    """The q=0 all-one mask: nothing is dropped, so every cell is retained."""
    shape = ae_contract.AE_LATENT_SPATIAL_SHAPE
    if frames is not None:
        shape = (int(frames), *shape)
    return torch.ones(shape, dtype=torch.bool, device=device)


def mask_from_selection(selection: CellSelection) -> torch.Tensor:
    """Validated bool [112,192] keep mask carried by one continuous-q selection."""
    guards.require_selection_integrity(
        selection,
        selection.q,
        cells=ae_contract.AE_LATENT_CELLS,
        spatial_shape=ae_contract.AE_LATENT_SPATIAL_SHAPE,
    )
    return ae_contract.require_keep_mask(
        selection.keep_mask, expect_keep=int(selection.keep_count)
    )


def apply_latent_mask(latent: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Zero every dropped cell across all B latent channels.

    The mask is hard and detached, so this drops values without ever routing a
    gradient into the ranker; gradient still reaches the AE encoder through the
    retained cells.
    """
    ae_contract.require_latent(latent, int(latent.shape[0]), what="latent to mask")
    ae_contract.require_keep_mask(keep_mask)
    plane = keep_mask.detach().to(device=latent.device, dtype=latent.dtype).unsqueeze(0)
    return latent * plane


def detached_hard_mask(scores: torch.Tensor, q: float) -> tuple[CellSelection | None, torch.Tensor]:
    """Training-side masking interface: hard, detached, no gradient to the ranker.

    Returns `(None, all-one mask)` at q=0 and `(selection, mask)` otherwise. The
    scores are detached before selection, and selection is a non-differentiable
    top-k, so the ranker can receive no gradient through this path.
    """
    plan = continuous_q.quantize_q(q)
    if plan.is_bypass:
        return None, all_keep_mask(device=scores.device)
    selection = continuous_q.select_cells(scores.detach(), plan.wire_q)
    return selection, mask_from_selection(selection)


def compose(
    c2: torch.Tensor, autoencoder: SplitFeatureAE, ranker, q: float
) -> LatentComposition:
    """Rank on the original FP32 C2, encode the full frame, then drop cells."""
    guards.require_frozen_c2(c2)
    if not isinstance(autoencoder, SplitFeatureAE):
        raise guards.HybridQConfigError("compose requires a SplitFeatureAE")
    plan = continuous_q.quantize_q(q)

    # 1. The encoder always runs, and always on the complete frame: q=0
    #    bypasses the ranker, never the AE.
    latent = autoencoder.encode(c2)
    ae_contract.require_latent(latent, autoencoder.bottleneck, what="AE encoder output")

    # 2. Ranking reads the original FP32 tensor object. The AE latent is never
    #    an input to the ranker at any q.
    if plan.is_bypass:
        selection = None
        keep_mask = all_keep_mask(device=c2.device)
    else:
        scores = ranker.score_cells(c2)
        selection = continuous_q.select_cells(scores, plan.wire_q)
        keep_mask = mask_from_selection(selection)

    # 3. Only now are cells dropped, all B latent channels of a cell together.
    masked = apply_latent_mask(latent, keep_mask)
    guards.require_keep_cardinality(int(keep_mask.sum()), plan.keep_count)
    return LatentComposition(
        plan=plan,
        bottleneck=autoencoder.bottleneck,
        latent=latent,
        selection=selection,
        keep_mask=keep_mask,
        masked_latent=masked,
    )


def registered_keep_counts() -> dict[float, int]:
    """Keep counts at the registered q anchors, from the frozen formula."""
    return {
        float(q): contract.keep_count(float(q), ae_contract.AE_LATENT_CELLS)
        for q in contract.REGISTERED_Q_VALUES
    }

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

`compose` resolves one frame; `compose_batch` is the training-side batched form
and selects independently per frame, never across the batch.

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


@dataclass(frozen=True)
class BatchedLatentComposition:
    """One training batch resolved at one q, with per-frame keep masks."""

    plan: continuous_q.ContinuousQ
    bottleneck: int
    latent: torch.Tensor  # dense [N,B,112,192] FP32, before dropping
    selections: tuple[CellSelection, ...] | None  # None exactly at q=0
    keep_mask: torch.Tensor  # bool [N,112,192]; all True at q=0
    masked_latent: torch.Tensor  # zero-scattered [N,B,112,192] FP32

    @property
    def frames(self) -> int:
        return int(self.latent.shape[0])

    @property
    def keep_count(self) -> int:
        """Per-frame keep count; identical for every frame at one q."""
        return int(self.plan.keep_count)


def compose_batch(
    c2: torch.Tensor,
    autoencoder: SplitFeatureAE,
    ranker,
    q: float,
    *,
    stage_b_only: bool = True,
) -> BatchedLatentComposition:
    """Batched training composition: encode the batch, then drop per frame.

    One q applies to the whole batch, matching the locked Stage-B cycle, and by
    default only a Stage-B q is accepted (`ae_loss.stage_b_q_for_update` is the
    intended source). Selection is then run **independently for every frame**:
    each frame gets its own stable top-K over its own scores, so every frame
    keeps exactly `keep_count` cells. Candidates are never flattened across the
    batch and no single global top-K set is ever formed -- that would let a
    high-scoring frame spend another frame's budget.

    All masks are hard, boolean and detached, and the ranker runs under
    `no_grad`, so no gradient can reach it. Gradient still flows to the AE
    encoder through the retained cells.
    """
    if not isinstance(autoencoder, SplitFeatureAE):
        raise guards.HybridQConfigError("compose_batch requires a SplitFeatureAE")
    if not isinstance(c2, torch.Tensor) or c2.dim() != 4:
        raise guards.HybridQPayloadError(
            "compose_batch requires a batched [N,256,112,192] C2 tensor"
        )
    guards.require_frozen_batched_c2(c2, what="batched C2")
    plan = continuous_q.quantize_q(q)
    if stage_b_only and contract._q_to_e4(plan.wire_q) not in {
        contract._q_to_e4(float(value)) for value in ae_contract.AE_STAGE_B_Q_CYCLE
    }:
        raise guards.HybridQConfigError(
            f"q={plan.wire_q!r} is not in the locked Stage-B cycle "
            f"{ae_contract.AE_STAGE_B_Q_CYCLE}"
        )
    frames = int(c2.shape[0])

    # 1. The encoder always runs, on the complete batch.
    latent = autoencoder.encode(c2)
    if tuple(latent.shape) != (
        frames,
        autoencoder.bottleneck,
        ae_contract.AE_LATENT_HEIGHT,
        ae_contract.AE_LATENT_WIDTH,
    ):
        raise guards.HybridQPayloadError(
            f"AE encoder returned {list(latent.shape)} for a {frames}-frame batch"
        )

    # 2. Ranking reads the original FP32 C2 batch, under no_grad.
    if plan.is_bypass:
        selections = None
        keep_mask = all_keep_mask(device=c2.device, frames=frames)
    else:
        with torch.no_grad():
            scores = ranker(c2.detach())
        if tuple(scores.shape) != (frames, *ae_contract.AE_LATENT_SPATIAL_SHAPE):
            raise guards.HybridQPayloadError(
                f"batched ranker returned {list(scores.shape)}, expected "
                f"{[frames, *ae_contract.AE_LATENT_SPATIAL_SHAPE]}"
            )
        # Independent per-frame stable top-K. No cross-frame candidate pool.
        per_frame = [
            continuous_q.select_cells(scores[index].detach(), plan.wire_q)
            for index in range(frames)
        ]
        selections = tuple(per_frame)
        keep_mask = torch.stack(
            [mask_from_selection(selection) for selection in per_frame]
        ).to(device=c2.device)

    ae_contract.require_keep_mask(keep_mask[0], what="per-frame keep mask")
    if tuple(keep_mask.shape) != (frames, *ae_contract.AE_LATENT_SPATIAL_SHAPE):
        raise guards.HybridQPayloadError("stacked keep mask does not cover the batch")
    counts = keep_mask.reshape(frames, -1).sum(dim=1)
    for index in range(frames):
        guards.require_keep_cardinality(int(counts[index]), plan.keep_count)

    # 3. Drop cells, all B latent channels of a cell together, per frame.
    plane = keep_mask.detach().to(dtype=latent.dtype).unsqueeze(1)
    masked = latent * plane
    return BatchedLatentComposition(
        plan=plan,
        bottleneck=autoencoder.bottleneck,
        latent=latent,
        selections=selections,
        keep_mask=keep_mask,
        masked_latent=masked,
    )


def registered_keep_counts() -> dict[float, int]:
    """Keep counts at the registered q anchors, from the frozen formula."""
    return {
        float(q): contract.keep_count(float(q), ae_contract.AE_LATENT_CELLS)
        for q in contract.REGISTERED_Q_VALUES
    }

from __future__ import annotations

import torch
from torch import nn

from . import contract, guards


class SpatialRanker(nn.Module):
    """Lightweight per-cell importance scorer over the frozen fused C2 tensor.

    Exactly: 1x1 Conv 256->8, ReLU, depthwise 3x3 Conv 8->8 (pad 1), ReLU,
    1x1 Conv 8->1 with **no bias**. No normalization, no attention, no second
    backbone and no object-level ROI model.

    The final layer carries no bias because a single global scalar added to
    every cell score is unidentifiable: it cannot change the cell ranking, the
    exact-cardinality selection or the hard mask, and listwise softmax
    distillation is invariant to it. A straight-through gradient on such a bias
    would therefore not correspond to any change in the transported cell set.

    The module consumes detached C2 only. It has no runtime access to RGB,
    radar, ground truth, detections, segmentation, geometry or any
    target-region side channel.
    """

    def __init__(
        self,
        in_channels: int = contract.SPLIT_CHANNELS,
        hidden_channels: int = contract.RANKER_HIDDEN_CHANNELS,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.reduce = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1)
        self.act1 = nn.ReLU(inplace=False)
        self.depthwise = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
        )
        self.act2 = nn.ReLU(inplace=False)
        self.score = nn.Conv2d(self.hidden_channels, 1, kernel_size=1, bias=False)

    def forward(self, c2: torch.Tensor) -> torch.Tensor:
        """Score cells of the frozen C2 tensor.

        Accepts [256,112,192] or [B,256,112,192] FP32 and returns [112,192] or
        [B,112,192]. The input is detached inside the forward pass, so gradients
        reach the ranker parameters only and never the frozen perception trunk.
        """
        guards.require_frozen_batched_c2(c2, what="ranker input")
        return self._score_any(c2)

    def _score_any(self, c2: torch.Tensor) -> torch.Tensor:
        """Private generic scorer with no frozen-shape enforcement (tests only)."""
        if not isinstance(c2, torch.Tensor):
            raise guards.HybridQPayloadError("ranker input must be a torch.Tensor")
        if c2.dim() == 3:
            batched = c2.unsqueeze(0)
            squeeze = True
        elif c2.dim() == 4:
            batched = c2
            squeeze = False
        else:
            raise guards.HybridQPayloadError(
                f"ranker input must be [C,H,W] or [B,C,H,W], got {list(c2.shape)}"
            )
        if batched.shape[1] != self.in_channels:
            raise guards.HybridQPayloadError(
                f"ranker expects {self.in_channels} channels, got {batched.shape[1]}"
            )
        features = batched.detach()
        hidden = self.act1(self.reduce(features))
        hidden = self.act2(self.depthwise(hidden))
        scores = self.score(hidden).squeeze(1)
        return scores.squeeze(0) if squeeze else scores

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        """Guarded production entry point: frozen shape in, finite [112,192] out."""
        guards.require_frozen_c2(c2, what="ranker input")
        scores = self._score_any(c2)
        return guards.require_frozen_scores(scores)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def mac_count(self, height: int, width: int) -> int:
        cells = int(height) * int(width)
        return (
            self.in_channels * self.hidden_channels * cells
            + self.hidden_channels * 3 * 3 * cells
            + self.hidden_channels * 1 * cells
        )


def build_ranker(*, seed: int | None = contract.RANKER_INIT_SEED) -> SpatialRanker:
    """Construct the contract ranker deterministically at the registered seed.

    Initialization runs inside `torch.random.fork_rng`, so the caller's global
    RNG state is restored and the construction does not advance the caller's
    random sequence. `seed=None` uses ambient RNG (diagnostic use only).
    """
    if seed is None:
        ranker = SpatialRanker()
    else:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(seed))
            ranker = SpatialRanker()
    observed = ranker.parameter_count()
    if observed != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError(
            f"ranker parameter count {observed} != contract "
            f"{contract.RANKER_PARAMETER_COUNT}"
        )
    return ranker

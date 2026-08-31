from __future__ import annotations

import torch
from torch import nn

CACHED_FEATURE_DIM = 1034
RELATIONAL_FEATURE_DIM = 10
INPUT_DIM = CACHED_FEATURE_DIM + RELATIONAL_FEATURE_DIM
HIDDEN_DIM = 128
ATTENTION_HEADS = 4
FEEDFORWARD_DIM = 256
TRANSFORMER_LAYERS = 2
MAX_CANDIDATES_PER_FRAME = 97
SCORE_EPSILON = 1e-6

ARCHITECTURE = {
    "input_dimension": INPUT_DIM,
    "normalization": f"LayerNorm({INPUT_DIM})",
    "projection": f"Linear({INPUT_DIM},{HIDDEN_DIM})",
    "hidden_dimension": HIDDEN_DIM,
    "layers": TRANSFORMER_LAYERS,
    "self_attention_heads": ATTENTION_HEADS,
    "feedforward_dimension": FEEDFORWARD_DIM,
    "dropout": 0.0,
    "output": "one residual logit per candidate",
    "zero_initialized_output": True,
}


class PersonRelationalSelector(nn.Module):
    """Permutation-equivariant per-frame selector with no positional encoding."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(INPUT_DIM)
        self.projection = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.output = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if (features.ndim != 3 or features.shape[2] != INPUT_DIM
                or padding_mask.shape != features.shape[:2]
                or padding_mask.dtype != torch.bool):
            raise ValueError(f"expected features [B,N,{INPUT_DIM}] and boolean padding mask [B,N]")
        if features.shape[1] > MAX_CANDIDATES_PER_FRAME:
            raise RuntimeError("candidate frame exceeds locked maximum; truncation is prohibited")
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError("non-finite relational-selector input")

        # PyTorch attention has no defined softmax for an entirely masked row.
        # Temporarily expose one token for such empty frames and zero it below.
        safe_mask = padding_mask.clone()
        empty = safe_mask.all(dim=1)
        if bool(empty.any()) and safe_mask.shape[1] > 0:
            safe_mask[empty, 0] = False
        hidden = self.projection(self.normalization(features))
        hidden = self.encoder(hidden, src_key_padding_mask=safe_mask)
        delta = self.output(hidden).squeeze(-1)
        return delta.masked_fill(padding_mask, 0.0)


def refined_person_logits(
    base_scores: torch.Tensor,
    residual_logits: torch.Tensor,
    calibration_bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    if base_scores.shape != residual_logits.shape:
        raise ValueError("base-score and residual-logit shapes differ")
    base = base_scores.float()
    residual = residual_logits.float()
    if (not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(residual).all())
            or bool(((base < 0.0) | (base > 1.0)).any())):
        raise FloatingPointError("person scores and residuals must be finite with scores in [0,1]")
    bias = torch.as_tensor(calibration_bias, dtype=torch.float32, device=base.device)
    if bias.numel() != 1 or not bool(torch.isfinite(bias)):
        raise ValueError("calibration bias must be one finite scalar")
    return torch.logit(base.clamp(SCORE_EPSILON, 1.0 - SCORE_EPSILON)) + residual + bias


def refined_person_scores(
    base_scores: torch.Tensor,
    residual_logits: torch.Tensor,
    calibration_bias: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Use deployment FP32 arithmetic, preserving exact neutral scores."""
    base = base_scores.float()
    residual = residual_logits.float()
    bias = torch.as_tensor(calibration_bias, dtype=torch.float32, device=base.device)
    scores = torch.sigmoid(refined_person_logits(base, residual, bias))
    neutral = residual.eq(0.0) & bias.eq(0.0)
    return torch.where(neutral, base, scores)


def build_selector_optimizer(selector: PersonRelationalSelector) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(selector.parameters(), lr=1e-3)
    selector_ids = {id(parameter) for parameter in selector.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != selector_ids:
        raise RuntimeError("optimizer is not restricted exactly to selector parameters")
    return optimizer

"""Task-aware AE reconstruction loss and the masking interfaces a trainer needs.

Phase 9A implements these interfaces and locks the intended protocol in
documentation. It runs no scientific training: nothing here builds a dataset,
opens a checkpoint, steps an optimizer or launches a fit.

Locked intended later protocol
------------------------------
- Only AE parameters are trainable. Frozen perception and the stable epoch-4
  ranker stay in eval mode with `requires_grad=False`, and the optimizer is
  permitted to own AE parameters only.
- Fit episodes are used for optimization; the reserved train-holdout is used for
  selection. No evaluation split is touched.
- Stage A: dense q=0 reconstruction.
- Stage B: balanced batch-level cycle over q in {0, 0.30, 0.50, 0.70}, one q per
  batch, so the four conditions receive equal numbers of updates.
- q=0.90 and q=0.98 are excluded from optimization entirely and retained as
  later evaluation/emergency values.
- No fake quantization and no zstd inside training. The AE is trained in FP32,
  so one AE checkpoint can later serve UINT8, UINT6 or UINT4 without retraining.
  This module accordingly refuses non-float inputs.
- Ranker masks are hard and detached (`ae_composition.detached_hard_mask`), so
  no gradient can enter the ranker.

Objective
---------
Two unit-weighted components, both scale-free, reported separately:

    plain      = ||C2_hat - C2||^2 / ||C2||^2
    group_t    = sum_hw I_t(h,w) * e(h,w) / sum_hw I_t(h,w) * g(h,w)
    importance = mean over valid t of group_t
    total      = plain + importance

with e(h,w) = sum_c (C2_hat - C2)^2 and g(h,w) = sum_c C2^2, and I_t the
existing Phase-4 L1-normalized D/G/S/A importance map. Groups are combined with
equal weight, an unavailable map is ignored, and at least three of the four
groups must be valid. No raw multitask detection/segmentation loss weight is
introduced anywhere: the task maps enter only as spatial weightings of the same
reconstruction error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards, training
from . import ae_contract
from .ae_model import SplitFeatureAE, ae_parameters


@dataclass
class AeReconstructionLoss:
    """Every component reported separately; `total` is the only scalar to step."""

    total: torch.Tensor
    plain: torch.Tensor
    importance: torch.Tensor
    group_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_groups: tuple[str, ...] = ()
    excluded_groups: dict[str, str] = field(default_factory=dict)

    def report(self) -> dict[str, float | list[str] | dict[str, float] | dict[str, str]]:
        """Detached scalars for logging; never used as part of the graph."""
        return {
            "total": float(self.total.detach()),
            "plain_reconstruction": float(self.plain.detach()),
            "importance_reconstruction": float(self.importance.detach()),
            "group_terms": {
                name: float(value.detach()) for name, value in self.group_terms.items()
            },
            "valid_groups": list(self.valid_groups),
            "excluded_groups": dict(self.excluded_groups),
        }


def _require_loss_tensor(tensor: torch.Tensor, what: str) -> torch.Tensor:
    """Reconstruction operates on FP32 features only, never on codes."""
    if not isinstance(tensor, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tensor.dtype is not torch.float32:
        raise guards.HybridQPayloadError(
            f"{what} must be float32; quantized or compressed tensors never "
            f"enter the training objective, got {tensor.dtype}"
        )
    if tensor.dim() not in (3, 4):
        raise guards.HybridQPayloadError(
            f"{what} must be [C,H,W] or [N,C,H,W], got {list(tensor.shape)}"
        )
    if tensor.shape[-3] != contract.SPLIT_CHANNELS:
        raise guards.HybridQPayloadError(
            f"{what} must carry {contract.SPLIT_CHANNELS} channels, got {tensor.shape[-3]}"
        )
    if tuple(tensor.shape[-2:]) != contract.SPLIT_SPATIAL_SHAPE:
        raise guards.HybridQPayloadError(
            f"{what} spatial shape must be {list(contract.SPLIT_SPATIAL_SHAPE)}, "
            f"got {list(tensor.shape[-2:])}"
        )
    return tensor if tensor.dim() == 4 else tensor.unsqueeze(0)


def _collect_group_maps(
    importance_maps: Mapping[str, torch.Tensor | None] | training.TeacherMapResult,
    frames: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Normalize and validate the Phase-4 D/G/S/A maps, recording exclusions."""
    if isinstance(importance_maps, training.TeacherMapResult):
        raw: Mapping[str, torch.Tensor | None] = {
            name: importance_maps.group_maps[name]
            for name in importance_maps.valid_groups
        }
        excluded = dict(importance_maps.excluded_groups)
    else:
        if not isinstance(importance_maps, Mapping):
            raise guards.HybridQConfigError(
                "importance maps must be a mapping or a TeacherMapResult"
            )
        unknown = set(importance_maps) - set(ae_contract.AE_TASK_GROUPS)
        if unknown:
            raise guards.HybridQConfigError(
                f"unregistered task group(s) {sorted(unknown)}; only "
                f"{list(ae_contract.AE_TASK_GROUPS)} are locked"
            )
        raw = importance_maps
        excluded = {}

    maps: dict[str, torch.Tensor] = {}
    for group in ae_contract.AE_TASK_GROUPS:
        candidate = raw.get(group) if hasattr(raw, "get") else None
        if candidate is None:
            excluded.setdefault(group, "absent")
            continue
        if not isinstance(candidate, torch.Tensor):
            excluded[group] = "not_a_tensor"
            continue
        weights = candidate.detach().to(torch.float32)
        if weights.dim() == 2:
            weights = weights.unsqueeze(0).expand(frames, -1, -1)
        elif weights.dim() != 3 or int(weights.shape[0]) != frames:
            excluded[group] = "frame_count_mismatch"
            continue
        if tuple(weights.shape[-2:]) != contract.SPLIT_SPATIAL_SHAPE:
            excluded[group] = "wrong_spatial_shape"
            continue
        if not bool(torch.isfinite(weights).all()):
            excluded[group] = "non_finite"
            continue
        if bool((weights < 0).any()):
            excluded[group] = "negative_importance"
            continue
        mass = weights.reshape(frames, -1).sum(dim=1)
        if not bool((mass > 0).all()):
            excluded[group] = "zero_mass"
            continue
        # Locked per-frame L1 normalization, matching the Phase-4 interface.
        maps[group] = weights / mass.reshape(frames, 1, 1)
        excluded.pop(group, None)
    return maps, excluded


def task_aware_reconstruction_loss(
    c2: torch.Tensor,
    reconstructed: torch.Tensor,
    importance_maps: Mapping[str, torch.Tensor | None] | training.TeacherMapResult,
    *,
    min_valid_groups: int = ae_contract.AE_MIN_VALID_TASK_GROUPS,
) -> AeReconstructionLoss:
    """Feature reconstruction loss, plain plus equal-weight task-importance."""
    target = _require_loss_tensor(c2, "reference C2").detach()
    estimate = _require_loss_tensor(reconstructed, "reconstructed C2")
    if target.shape != estimate.shape:
        raise guards.HybridQPayloadError(
            f"reconstruction shape {list(estimate.shape)} != reference "
            f"{list(target.shape)}"
        )
    guards.require_finite(target, "reference C2")

    frames = int(target.shape[0])
    squared_error = (estimate - target).pow(2)
    cell_error = squared_error.sum(dim=1)  # [N,H,W]
    cell_energy = target.pow(2).sum(dim=1)  # [N,H,W]

    reference_energy = cell_energy.sum()
    if float(reference_energy) <= 0.0:
        raise guards.HybridQNumericalError(
            "reference C2 has zero energy; the normalized loss is undefined"
        )
    plain = squared_error.sum() / reference_energy

    maps, excluded = _collect_group_maps(importance_maps, frames)
    group_terms: dict[str, torch.Tensor] = {}
    for group in ae_contract.AE_TASK_GROUPS:
        weights = maps.get(group)
        if weights is None:
            excluded.setdefault(group, "absent")
            continue
        denominator = (weights * cell_energy).sum()
        if float(denominator) <= 0.0:
            # The map's mass sits entirely where the reference has no energy;
            # that group carries no usable normalization for this frame.
            excluded[group] = "zero_reference_energy"
            continue
        group_terms[group] = (weights * cell_error).sum() / denominator
        excluded.pop(group, None)

    valid = tuple(group for group in ae_contract.AE_TASK_GROUPS if group in group_terms)
    if len(valid) < int(min_valid_groups):
        raise guards.HybridQConfigError(
            f"task-aware reconstruction requires at least {int(min_valid_groups)} "
            f"valid D/G/S/A maps, got {len(valid)} {list(valid)}; "
            f"excluded={dict(sorted(excluded.items()))}"
        )

    weight = 1.0 / len(valid)
    importance = sum(group_terms[group] * weight for group in valid)
    total = plain + importance
    guards.require_finite(total.detach(), "task-aware reconstruction loss")
    return AeReconstructionLoss(
        total=total,
        plain=plain,
        importance=importance,
        group_terms=group_terms,
        valid_groups=valid,
        excluded_groups=dict(sorted(excluded.items())),
    )


# ---------------------------------------------------------------------------
# Trainer-facing schedule and ownership interfaces (declarative; nothing runs)
# ---------------------------------------------------------------------------


def stage_a_q() -> float:
    """Stage A trains dense reconstruction at q=0 only."""
    return float(ae_contract.AE_STAGE_A_Q)


def stage_b_q_for_update(update_index: int) -> float:
    """Balanced batch-level cycle over q in {0, 0.30, 0.50, 0.70}, one q per batch.

    A deterministic round robin makes the four conditions exactly balanced over
    any whole number of cycles, with no sampling variance to account for later.
    """
    if isinstance(update_index, bool) or not isinstance(update_index, int):
        raise guards.HybridQConfigError("update index must be an int")
    if update_index < 0:
        raise guards.HybridQConfigError("update index must be non-negative")
    cycle = ae_contract.AE_STAGE_B_Q_CYCLE
    return float(cycle[update_index % len(cycle)])


def require_optimization_q(q: float) -> float:
    """q=0.90 and q=0.98 are evaluation/emergency values, never optimized."""
    value = guards.require_valid_q(q, registered_only=False)
    if contract._q_to_e4(value) in {
        contract._q_to_e4(float(excluded))
        for excluded in ae_contract.AE_OPTIMIZATION_EXCLUDED_Q
    }:
        raise guards.HybridQConfigError(
            f"q={value!r} is excluded from optimization "
            f"{ae_contract.AE_OPTIMIZATION_EXCLUDED_Q} and is evaluation-only"
        )
    return value


def require_ae_only_optimizer(
    optimizer: torch.optim.Optimizer, autoencoder: SplitFeatureAE
) -> None:
    """The optimizer may own AE parameters and nothing else."""
    guards.require_optimizer_owns_only(optimizer, ae_parameters(autoencoder))


def require_frozen_companions(modules) -> None:
    """Frozen perception and the stable ranker: eval mode, no trainable state."""
    modules = list(modules)
    guards.require_frozen_perception(modules)
    guards.require_eval_mode(modules)


def training_protocol() -> dict[str, object]:
    """The locked intended later protocol, as data rather than prose only."""
    return {
        "trainable_scope": ae_contract.AE_TRAINABLE_SCOPE,
        "frozen_modules": ("splitfusion_perception", "stable_epoch4_ranker"),
        "optimization_split": "fit_episodes",
        "selection_split": "reserved_train_holdout",
        "stage_a": {"objective": "dense_reconstruction", "q": stage_a_q()},
        "stage_b": {
            "objective": "balanced_batch_level_q_cycle",
            "q_cycle": ae_contract.AE_STAGE_B_Q_CYCLE,
            "assignment": "one_q_per_batch_round_robin",
        },
        "excluded_from_optimization_q": ae_contract.AE_OPTIMIZATION_EXCLUDED_Q,
        "fake_quantization_in_training": False,
        "zstd_in_training": False,
        "checkpoint_reusable_for": ("uint8", "uint6", "uint4"),
        "ranker_masks": "hard_and_detached_no_gradient_to_ranker",
        "task_groups": ae_contract.AE_TASK_GROUPS,
        "min_valid_task_groups": ae_contract.AE_MIN_VALID_TASK_GROUPS,
        "group_combination": "equal_weight",
        "raw_multitask_loss_weights_introduced": False,
        "executed_in_phase_9a": False,
    }

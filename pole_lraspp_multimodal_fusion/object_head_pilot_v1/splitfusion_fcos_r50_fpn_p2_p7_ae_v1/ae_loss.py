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
- Ranker masks are hard and detached (`ae_composition.detached_hard_mask`,
  `ae_composition.compose_batch`), so no gradient can enter the ranker.

Teacher supervision comes from the **existing** Phase-4 teacher cache, consumed
exactly as it is stored. A shard holds, per frame, one combined FP32 importance
map plus `valid_groups` / `excluded_groups` metadata. It does **not** hold four
separate D/G/S/A maps, so this module neither expects nor fabricates them, and
reports no per-group reconstruction term. The cache is not rebuilt.

Objective
---------
Two unit-weighted, scale-free components, reported separately:

    e(h,w)     = sum_c (C2_hat - C2)^2        g(h,w) = sum_c C2^2
    plain      = sum e / sum g
    combined   = sum_hw I * e / sum_hw I * g
    total      = plain + combined

`I` is the cached combined map, which the Phase-4 producer already L1-normalized
per frame; it is re-normalized here defensively, which is a no-op on a
well-formed cache entry. Both ratios are taken over the whole batch, so the two
components are normalized consistently.

Per-frame `valid_groups` decides admissibility: **every** frame in the batch
must carry at least three valid D/G/S/A groups, or the call fails closed. Group
availability and exclusions are reported as metadata only. No raw multitask
detection/segmentation loss weight is introduced anywhere: the cached map enters
only as a spatial weighting of the same reconstruction error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards, training
from . import ae_contract
from .ae_model import SplitFeatureAE, ae_parameters


@dataclass(frozen=True)
class CachedTeacherBatch:
    """Exactly the Phase-4 teacher-cache fields, for one training batch.

    Mirrors a shard slice: one combined importance map per frame plus that
    frame's validity metadata. There is deliberately no per-group map field,
    because the cache does not store one.
    """

    importance: torch.Tensor  # [N,112,192] FP32 combined map
    valid_groups: tuple[tuple[str, ...], ...]
    excluded_groups: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.importance, torch.Tensor):
            raise guards.HybridQPayloadError("cached importance must be a torch.Tensor")
        if self.importance.dim() != 3:
            raise guards.HybridQPayloadError(
                f"cached importance must be [N,H,W], got {list(self.importance.shape)}"
            )
        if self.importance.dtype is not torch.float32:
            raise guards.HybridQPayloadError(
                f"cached importance must be float32, got {self.importance.dtype}"
            )
        if tuple(self.importance.shape[1:]) != contract.SPLIT_SPATIAL_SHAPE:
            raise guards.HybridQPayloadError(
                f"cached importance maps must be {list(contract.SPLIT_SPATIAL_SHAPE)}, "
                f"got {list(self.importance.shape[1:])}"
            )
        frames = int(self.importance.shape[0])
        if len(self.valid_groups) != frames or len(self.excluded_groups) != frames:
            raise guards.HybridQPayloadError(
                "cached validity metadata does not cover every frame in the batch"
            )
        for groups in self.valid_groups:
            unknown = set(groups) - set(ae_contract.AE_TASK_GROUPS)
            if unknown:
                raise guards.HybridQConfigError(
                    f"unregistered valid task group(s) {sorted(unknown)}; only "
                    f"{list(ae_contract.AE_TASK_GROUPS)} are locked"
                )
            if len(set(groups)) != len(groups):
                raise guards.HybridQPayloadError("cached valid groups contain duplicates")
        for excluded in self.excluded_groups:
            unknown = set(excluded) - set(ae_contract.AE_TASK_GROUPS)
            if unknown:
                raise guards.HybridQConfigError(
                    f"unregistered excluded task group(s) {sorted(unknown)}"
                )

    @property
    def frames(self) -> int:
        return int(self.importance.shape[0])

    def group_availability(self) -> dict[str, int]:
        """How many frames of this batch counted each group as valid."""
        return {
            group: sum(1 for groups in self.valid_groups if group in groups)
            for group in ae_contract.AE_TASK_GROUPS
        }

    def exclusion_reasons(self) -> dict[str, dict[str, int]]:
        """Recorded exclusion reasons per group, as counts over the batch."""
        reasons: dict[str, dict[str, int]] = {}
        for excluded in self.excluded_groups:
            for group, reason in excluded.items():
                bucket = reasons.setdefault(group, {})
                bucket[str(reason)] = bucket.get(str(reason), 0) + 1
        return {group: dict(sorted(bucket.items())) for group, bucket in sorted(reasons.items())}

    @classmethod
    def from_shard(
        cls, payload: Mapping[str, object], offsets: Sequence[int] | None = None
    ) -> "CachedTeacherBatch":
        """Slice one loaded Phase-4 shard (`teacher_cache.load_shard`) into a batch.

        Reads only `importance`, `valid_groups` and `excluded_groups` — the
        fields the shard actually stores. `offsets` selects the frames of this
        batch; `None` takes the whole shard. Nothing is written and no cache is
        rebuilt.
        """
        missing = {"importance", "valid_groups", "excluded_groups"} - set(payload)
        if missing:
            raise guards.HybridQPayloadError(
                f"teacher shard is missing required field(s) {sorted(missing)}"
            )
        maps = payload["importance"]
        valid = list(payload["valid_groups"])
        excluded = list(payload["excluded_groups"])
        if not isinstance(maps, torch.Tensor):
            raise guards.HybridQPayloadError("shard importance must be a torch.Tensor")
        if len(valid) != int(maps.shape[0]) or len(excluded) != int(maps.shape[0]):
            raise guards.HybridQPayloadError("shard metadata length disagrees with maps")
        if offsets is not None:
            index = [int(offset) for offset in offsets]
            if not index:
                raise guards.HybridQConfigError("a teacher batch must hold a frame")
            if any(not 0 <= offset < int(maps.shape[0]) for offset in index):
                raise guards.HybridQPayloadError("shard offset out of range")
            maps = maps.index_select(0, torch.tensor(index, dtype=torch.int64))
            valid = [valid[offset] for offset in index]
            excluded = [excluded[offset] for offset in index]
        return cls(
            importance=maps.to(torch.float32).contiguous(),
            valid_groups=tuple(tuple(str(name) for name in groups) for groups in valid),
            excluded_groups=tuple(
                {str(key): str(value) for key, value in dict(entry).items()}
                for entry in excluded
            ),
        )

    @classmethod
    def from_teacher_maps(
        cls, results: Sequence[training.TeacherMapResult]
    ) -> "CachedTeacherBatch":
        """Build a batch from in-memory Phase-4 results, as the cache writer does.

        `build_teacher_maps` produces exactly one combined map plus validity
        metadata per frame, which is what a shard stores; this is the same
        representation without a round trip through disk.
        """
        if not results:
            raise guards.HybridQConfigError("a teacher batch must hold a frame")
        maps = []
        for result in results:
            if not isinstance(result, training.TeacherMapResult):
                raise guards.HybridQPayloadError("expected Phase-4 TeacherMapResult objects")
            if result.importance is None:
                raise guards.HybridQPayloadError(
                    "a frame with no valid teacher group is not supervisable"
                )
            maps.append(result.importance.detach().to(torch.float32))
        return cls(
            importance=torch.stack(maps).contiguous(),
            valid_groups=tuple(tuple(result.valid_groups) for result in results),
            excluded_groups=tuple(dict(result.excluded_groups) for result in results),
        )


@dataclass
class AeReconstructionLoss:
    """Both components reported separately; `total` is the only scalar to step.

    There is no per-group reconstruction term, because the cache stores one
    combined map. Group availability and exclusions are metadata.
    """

    total: torch.Tensor
    plain: torch.Tensor
    combined_importance: torch.Tensor
    frames: int = 0
    group_availability: dict[str, int] = field(default_factory=dict)
    excluded_groups: dict[str, dict[str, int]] = field(default_factory=dict)
    min_valid_groups_observed: int = 0

    def report(self) -> dict[str, object]:
        """Detached scalars and metadata for logging; never part of the graph."""
        return {
            "total": float(self.total.detach()),
            "plain_reconstruction": float(self.plain.detach()),
            "combined_importance_reconstruction": float(
                self.combined_importance.detach()
            ),
            "frames": int(self.frames),
            "group_availability": dict(self.group_availability),
            "excluded_groups": {
                group: dict(reasons) for group, reasons in self.excluded_groups.items()
            },
            "min_valid_groups_observed": int(self.min_valid_groups_observed),
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


def task_aware_reconstruction_loss(
    c2: torch.Tensor,
    reconstructed: torch.Tensor,
    teacher: CachedTeacherBatch,
    *,
    min_valid_groups: int = ae_contract.AE_MIN_VALID_TASK_GROUPS,
) -> AeReconstructionLoss:
    """Plain plus cached-combined-importance normalized C2 reconstruction."""
    if not isinstance(teacher, CachedTeacherBatch):
        raise guards.HybridQConfigError(
            "teacher supervision must be a CachedTeacherBatch built from the "
            "Phase-4 cache representation"
        )
    target = _require_loss_tensor(c2, "reference C2").detach()
    estimate = _require_loss_tensor(reconstructed, "reconstructed C2")
    if target.shape != estimate.shape:
        raise guards.HybridQPayloadError(
            f"reconstruction shape {list(estimate.shape)} != reference "
            f"{list(target.shape)}"
        )
    guards.require_finite(target, "reference C2")

    frames = int(target.shape[0])
    if teacher.frames != frames:
        raise guards.HybridQPayloadError(
            f"teacher batch covers {teacher.frames} frames, C2 batch has {frames}"
        )

    # Every frame must carry enough valid groups; a thin frame is refused rather
    # than quietly averaged in.
    per_frame_valid = [len(groups) for groups in teacher.valid_groups]
    observed_minimum = min(per_frame_valid)
    if observed_minimum < int(min_valid_groups):
        thin = [
            position
            for position, count in enumerate(per_frame_valid)
            if count < int(min_valid_groups)
        ]
        raise guards.HybridQConfigError(
            f"task-aware reconstruction requires at least {int(min_valid_groups)} "
            f"valid D/G/S/A groups per frame; batch positions {thin} carry "
            f"{[per_frame_valid[position] for position in thin]}"
        )

    weights = teacher.importance.detach().to(
        device=target.device, dtype=torch.float32
    )
    guards.require_finite(weights, "cached combined importance map")
    if bool((weights < 0).any()):
        raise guards.HybridQNumericalError(
            "cached combined importance map must be non-negative"
        )
    mass = weights.reshape(frames, -1).sum(dim=1)
    if not bool((mass > 0).all()):
        raise guards.HybridQNumericalError(
            "a cached combined importance map has no positive mass"
        )
    # Defensive re-normalization; the Phase-4 producer already L1-normalizes.
    weights = weights / mass.reshape(frames, 1, 1)

    squared_error = (estimate - target).pow(2)
    cell_error = squared_error.sum(dim=1)  # [N,H,W]
    cell_energy = target.pow(2).sum(dim=1)  # [N,H,W]

    reference_energy = cell_energy.sum()
    if float(reference_energy) <= 0.0:
        raise guards.HybridQNumericalError(
            "reference C2 has zero energy; the normalized loss is undefined"
        )
    plain = squared_error.sum() / reference_energy

    weighted_energy = (weights * cell_energy).sum()
    if float(weighted_energy) <= 0.0:
        raise guards.HybridQNumericalError(
            "the cached importance mass sits where the reference C2 has no energy"
        )
    combined = (weights * cell_error).sum() / weighted_energy

    total = plain + combined
    guards.require_finite(total.detach(), "task-aware reconstruction loss")
    return AeReconstructionLoss(
        total=total,
        plain=plain,
        combined_importance=combined,
        frames=frames,
        group_availability=teacher.group_availability(),
        excluded_groups=teacher.exclusion_reasons(),
        min_valid_groups_observed=observed_minimum,
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

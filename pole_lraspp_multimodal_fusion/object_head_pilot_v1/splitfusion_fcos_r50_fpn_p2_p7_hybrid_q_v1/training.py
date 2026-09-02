"""Training-path primitives for hybrid-q. Phase 2 locks them; it executes none of them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from . import contract, guards
from .selection import _select_cells


# --------------------------------------------------------------------------
# Teacher importance maps over the registered frozen-model loss groups
# --------------------------------------------------------------------------


def teacher_importance_map(c2: torch.Tensor, task_grad: torch.Tensor) -> torch.Tensor:
    """I_t(h,w) = sum_c |C2(c,h,w) * grad_t(c,h,w)|, returned as [H,W]."""
    if c2.shape != task_grad.shape:
        raise guards.HybridQPayloadError(
            f"gradient shape {list(task_grad.shape)} != C2 shape {list(c2.shape)}"
        )
    if c2.dim() != 3:
        raise guards.HybridQPayloadError("teacher maps are computed per frame on [C,H,W]")
    return (c2.detach() * task_grad.detach()).abs().sum(dim=0)


def normalize_importance_l1(importance: torch.Tensor) -> torch.Tensor:
    """Locked L1 normalization, applied to one group map independently."""
    guards.require_finite(importance, "teacher importance map")
    if (importance < 0).any():
        raise guards.HybridQNumericalError("teacher importance map must be non-negative")
    mass = importance.sum()
    if not torch.isfinite(mass) or float(mass) <= 0.0:
        raise guards.HybridQNumericalError("teacher importance map has no positive mass")
    return importance / mass


@dataclass
class TeacherMapResult:
    """Per-frame teacher supervision with explicit group validity bookkeeping.

    `gradient_mass` is the raw pre-normalization L1 mass of each group map and is
    diagnostic only. `task_losses` holds the dense task-loss values, kept
    separately for future train-only reference-scale construction; it is never
    used as an importance weight here.
    """

    importance: torch.Tensor | None
    group_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_groups: tuple[str, ...] = ()
    excluded_groups: dict[str, str] = field(default_factory=dict)
    gradient_mass: dict[str, float] = field(default_factory=dict)
    task_losses: dict[str, float] = field(default_factory=dict)
    normalization: str = contract.TEACHER_NORMALIZATION
    combination: str = contract.TEACHER_GROUP_COMBINATION

    @property
    def is_supervisable(self) -> bool:
        return self.importance is not None and len(self.valid_groups) > 0


def build_teacher_maps(
    c2: torch.Tensor,
    group_grads: Mapping[str, torch.Tensor | None],
    *,
    task_losses: Mapping[str, float] | None = None,
) -> TeacherMapResult:
    """Build one combined teacher map from the registered loss-group gradients.

    Groups are exactly the registered frozen-model loss groups D, G, S and A.
    Each valid group is L1-normalized independently, then valid groups are
    combined with equal weight. Absent, zero-gradient and non-finite groups are
    recorded in `excluded_groups` with a reason and excluded; they are never
    silently treated as valid supervision. A frame with no valid group yields
    `importance=None`.
    """
    unknown = set(group_grads) - set(contract.TEACHER_GROUPS)
    if unknown:
        raise guards.HybridQConfigError(
            f"unregistered teacher loss group(s) {sorted(unknown)}; "
            f"only {list(contract.TEACHER_GROUPS)} are locked"
        )

    group_maps: dict[str, torch.Tensor] = {}
    excluded: dict[str, str] = {}
    gradient_mass: dict[str, float] = {}

    for group in contract.TEACHER_GROUPS:
        if group not in group_grads:
            excluded[group] = "absent"
            continue
        grad = group_grads[group]
        if grad is None:
            excluded[group] = "absent"
            continue
        raw = teacher_importance_map(c2, grad)
        if not torch.isfinite(raw).all():
            excluded[group] = "non_finite"
            continue
        mass = float(raw.sum())
        if mass <= 0.0:
            excluded[group] = "zero_gradient"
            continue
        gradient_mass[group] = mass
        group_maps[group] = normalize_importance_l1(raw)

    valid = tuple(group for group in contract.TEACHER_GROUPS if group in group_maps)
    retained_losses = dict(task_losses) if task_losses else {}
    if retained_losses:
        unknown_losses = set(retained_losses) - set(contract.TEACHER_GROUPS)
        if unknown_losses:
            raise guards.HybridQConfigError(
                f"unregistered task-loss group(s) {sorted(unknown_losses)}"
            )

    if not valid:
        return TeacherMapResult(
            importance=None,
            valid_groups=(),
            excluded_groups=excluded,
            task_losses=retained_losses,
        )

    weight = 1.0 / len(valid)
    combined = sum(group_maps[group] * weight for group in valid)
    return TeacherMapResult(
        importance=normalize_importance_l1(combined),
        group_maps=group_maps,
        valid_groups=valid,
        excluded_groups=excluded,
        gradient_mass=gradient_mass,
        task_losses=retained_losses,
    )


@dataclass(frozen=True)
class TeacherCacheRecord:
    """Contract for a future teacher-cache artifact.

    Full C2 tensors are deliberately absent: caches hold teacher maps, validity
    flags, identifiers, diagnostic gradient mass and dense task losses only.
    """

    frame_id: str
    sequence_id: str
    importance: torch.Tensor
    valid_groups: tuple[str, ...]
    excluded_groups: dict[str, str]
    gradient_mass: dict[str, float]
    task_losses: dict[str, float]
    normalization: str = contract.TEACHER_NORMALIZATION
    perception_checkpoint_sha256: str = contract.FROZEN_CHECKPOINT_SHA256

    def __post_init__(self) -> None:
        if self.importance.dim() != 2:
            raise guards.HybridQPayloadError("cached teacher importance must be [H,W]")


# --------------------------------------------------------------------------
# Locked distillation objective
# --------------------------------------------------------------------------


def ranker_distillation_loss(
    scores: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = contract.DISTILLATION_TEMPERATURE,
) -> torch.Tensor:
    """Listwise soft cross-entropy against the L1-normalized teacher distribution.

    The temperature is locked at `contract.DISTILLATION_TEMPERATURE` (1.0); the
    argument exists so a corrupted configuration still fails closed.
    """
    tau = guards.require_positive_temperature(temperature)
    if scores.shape != teacher.shape:
        raise guards.HybridQPayloadError("score and teacher map shapes differ")
    guards.require_finite(scores, "ranker scores")
    guards.require_finite(teacher, "teacher importance map")
    target = teacher.reshape(-1)
    mass = target.sum()
    if float(mass) <= 0.0:
        raise guards.HybridQNumericalError("teacher map has no positive mass")
    target = target / mass
    log_prob = F.log_softmax(scores.reshape(-1) / tau, dim=0)
    return -(target * log_prob).sum()


# --------------------------------------------------------------------------
# Locked q-aware stage
# --------------------------------------------------------------------------


def q_for_update(update_index: int) -> float:
    """Deterministic repeated training cycle 0.30, 0.50, 0.70 — one q per update."""
    if int(update_index) < 0:
        raise guards.HybridQConfigError("update index must be non-negative")
    cycle = contract.Q_AWARE_TRAINING_CYCLE
    return cycle[int(update_index) % len(cycle)]


def q_aware_schedule(num_updates: int) -> tuple[float, ...]:
    return tuple(q_for_update(index) for index in range(int(num_updates)))


@dataclass(frozen=True)
class ReferenceMedians:
    """Frozen per-task reference medians for the q-aware objective.

    `source` must be `fit_train`: no validation- or test-derived scale is allowed.
    """

    medians: Mapping[str, float]
    source: str = contract.REFERENCE_MEDIAN_SOURCE

    def __post_init__(self) -> None:
        if self.source != contract.REFERENCE_MEDIAN_SOURCE:
            raise guards.HybridQConfigError(
                f"reference medians must come from {contract.REFERENCE_MEDIAN_SOURCE!r}, "
                f"got {self.source!r}"
            )
        unknown = set(self.medians) - set(contract.TEACHER_GROUPS)
        if unknown:
            raise guards.HybridQConfigError(
                f"unregistered reference-median group(s) {sorted(unknown)}"
            )
        for group, value in self.medians.items():
            scale = float(value)
            if not scale > 0.0 or scale != scale or scale == float("inf"):
                raise guards.HybridQConfigError(
                    f"reference median for {group!r} must be finite and positive"
                )

    def require(self, group: str) -> float:
        if group not in self.medians:
            raise guards.HybridQConfigError(
                f"no frozen train-reference median registered for group {group!r}"
            )
        return float(self.medians[group])


def q_aware_objective(
    masked_task_losses: Mapping[str, torch.Tensor | None],
    distillation_loss: torch.Tensor,
    references: ReferenceMedians,
) -> torch.Tensor:
    """mean(valid masked task loss / train-reference median) + 0.1 * distillation.

    Interface only: Phase 2 does not execute it.
    """
    unknown = set(masked_task_losses) - set(contract.TEACHER_GROUPS)
    if unknown:
        raise guards.HybridQConfigError(
            f"unregistered task-loss group(s) {sorted(unknown)}"
        )
    scaled = []
    for group in contract.TEACHER_GROUPS:
        loss = masked_task_losses.get(group)
        if loss is None:
            continue
        guards.require_finite(loss, f"masked task loss for group {group}")
        scaled.append(loss / references.require(group))
    if not scaled:
        raise guards.HybridQConfigError("q-aware objective requires at least one valid task")
    guards.require_finite(distillation_loss, "distillation loss")
    task_term = torch.stack([value.reshape(()) for value in scaled]).mean()
    return task_term + contract.Q_AWARE_DISTILLATION_WEIGHT * distillation_loss


# --------------------------------------------------------------------------
# Hard-mask forward with the locked straight-through surrogate
# --------------------------------------------------------------------------


def _straight_through_mask(
    scores: torch.Tensor, q: float, *, registered_only: bool = True
) -> torch.Tensor:
    """Private generic surrogate over any [H,W] score map (tests and internals)."""
    value = guards.require_valid_q(q, registered_only=registered_only)
    tau = guards.require_positive_temperature(contract.STRAIGHT_THROUGH_TEMPERATURE)
    selection = _select_cells(scores, value, registered_only=registered_only)

    if selection.drop_count == 0:
        # q=0 parity: explicit identity bypass, detached so no ranker gradient flows.
        return torch.ones_like(scores).detach()

    hard = selection.keep_mask.to(scores.dtype)
    flat = scores.reshape(-1)
    kept = flat[selection.keep_indices]
    dropped = flat[~selection.keep_mask.reshape(-1)]
    threshold = ((kept.min() + dropped.max()) * 0.5).detach()
    soft = torch.sigmoid((scores - threshold) / tau)
    # Grouped so the forward value is exactly `hard`: (soft - soft.detach()) is
    # exactly zero, whereas (hard + soft) - soft.detach() rounds in float32.
    return hard + (soft - soft.detach())


def straight_through_mask(scores: torch.Tensor, q: float) -> torch.Tensor:
    """Hard exact-cardinality mask forward, sigmoid surrogate backward.

    The boundary is the locked midpoint between the lowest retained and the
    highest dropped score, at the locked temperature 1.0. The returned tensor
    equals the hard 0/1 mask numerically. q=0 is an explicit identity bypass
    that carries no surrogate gradient, so parity monitoring cannot train the
    ranker.
    """
    guards.require_frozen_scores(scores)
    return _straight_through_mask(scores, guards.require_valid_q(q))


def masked_c2_forward(
    c2: torch.Tensor, mask: torch.Tensor, *, detach_features: bool = True
) -> torch.Tensor:
    """Apply a [H,W] mask to every channel of C2, keeping the mask differentiable.

    `detach_features=True` keeps gradient flowing to the ranker through the
    mask only, never into the frozen perception trunk. The edge still receives
    a dense zero-scattered tensor.
    """
    if mask.shape != c2.shape[-2:]:
        raise guards.HybridQPayloadError("mask shape does not match C2 spatial shape")
    features = c2.detach() if detach_features else c2
    return features * mask.unsqueeze(0).to(features.dtype)


# --------------------------------------------------------------------------
# Locked optimizer construction and frozen ownership
# --------------------------------------------------------------------------


def build_ranker_optimizer(
    ranker: nn.Module,
    *,
    lr: float = contract.LEARNING_RATE,
    weight_decay: float = contract.WEIGHT_DECAY,
    frozen_modules: Sequence[nn.Module] = (),
) -> torch.optim.Optimizer:
    """Locked AdamW over ranker parameters only, at constant LR.

    The frozen perception stack must be non-trainable and in evaluation mode.
    """
    frozen = tuple(frozen_modules)
    guards.require_frozen_perception(frozen)
    guards.require_eval_mode(frozen)
    frozen_ids = {id(p) for module in frozen for p in module.parameters()}
    params = [p for p in ranker.parameters() if p.requires_grad]
    if not params:
        raise guards.HybridQConfigError("ranker has no trainable parameters")
    for parameter in params:
        if id(parameter) in frozen_ids:
            raise guards.HybridQOwnershipError(
                "ranker parameter list overlaps the frozen perception stack"
            )
    optimizer = torch.optim.AdamW(
        params, lr=float(lr), weight_decay=float(weight_decay)
    )
    guards.require_optimizer_owns_only(optimizer, params)
    return optimizer


def clip_ranker_gradients(ranker: nn.Module) -> torch.Tensor:
    """Locked global ranker gradient-norm clip at 5.0."""
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in ranker.parameters() if p.requires_grad],
        contract.GRAD_CLIP_GLOBAL_NORM,
    )
    if not torch.isfinite(norm):
        raise guards.HybridQNumericalError("ranker gradient norm is non-finite")
    return norm


def require_post_step_health(
    ranker: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """After an optimizer step: ranker parameters and optimizer state stay finite."""
    guards.require_module_parameters_finite(ranker, "ranker")
    guards.require_finite_optimizer_state(optimizer)


# --------------------------------------------------------------------------
# Gradient qualification policy
# --------------------------------------------------------------------------


@dataclass
class GradientQualification:
    """Per-named-tensor gradient policy over a complete qualification window.

    Finiteness of the loss and of every observed gradient is required on every
    update. Every named trainable ranker tensor must receive a gradient and be
    nonzero at least once across the completed window. Isolated zero-gradient
    batches are logged, never treated as numerical failures; parameters that are
    still missing or disconnected at the end of the window fail.
    """

    window: int
    parameter_names: tuple[str, ...]
    seen: int = 0
    present_updates: dict[str, int] = field(default_factory=dict)
    nonzero_updates: dict[str, int] = field(default_factory=dict)
    zero_gradient_batches: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    missing_gradient_batches: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise guards.HybridQConfigError("qualification window must be at least 1")
        if not self.parameter_names:
            raise guards.HybridQConfigError("qualification requires named parameters")
        self.parameter_names = tuple(self.parameter_names)
        self.present_updates = {name: 0 for name in self.parameter_names}
        self.nonzero_updates = {name: 0 for name in self.parameter_names}

    @classmethod
    def for_module(cls, module: nn.Module, window: int) -> "GradientQualification":
        names = tuple(
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        )
        return cls(window=int(window), parameter_names=names)

    def observe(self, module: nn.Module, *, loss: torch.Tensor) -> bool:
        """Record one update. Returns whether every tracked tensor was nonzero."""
        if not torch.isfinite(torch.as_tensor(loss).detach()).all():
            raise guards.HybridQNumericalError("non-finite training loss")
        named = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if tuple(named) != self.parameter_names:
            raise guards.HybridQQualificationError(
                "trainable ranker tensor set changed during the qualification window"
            )

        self.seen += 1
        missing: list[str] = []
        zero: list[str] = []
        for name in self.parameter_names:
            grad = named[name].grad
            if grad is None:
                missing.append(name)
                continue
            if not torch.isfinite(grad).all():
                raise guards.HybridQNumericalError(
                    f"non-finite gradient on ranker tensor '{name}'"
                )
            self.present_updates[name] += 1
            if bool(grad.abs().sum() > 0):
                self.nonzero_updates[name] += 1
            else:
                zero.append(name)
        if missing:
            self.missing_gradient_batches.append((self.seen, tuple(missing)))
        if zero:
            self.zero_gradient_batches.append((self.seen, tuple(zero)))
        return not missing and not zero

    def window_complete(self) -> bool:
        return self.seen >= self.window

    def disconnected(self) -> tuple[str, ...]:
        return tuple(name for name in self.parameter_names if self.present_updates[name] == 0)

    def never_nonzero(self) -> tuple[str, ...]:
        return tuple(name for name in self.parameter_names if self.nonzero_updates[name] == 0)

    def qualified(self) -> bool:
        return (
            self.window_complete()
            and not self.disconnected()
            and not self.never_nonzero()
        )

    def require_qualified(self) -> None:
        if not self.window_complete():
            raise guards.HybridQQualificationError(
                f"qualification window incomplete: {self.seen}/{self.window} updates"
            )
        disconnected = self.disconnected()
        if disconnected:
            raise guards.HybridQQualificationError(
                f"ranker tensor(s) never received a gradient: {list(disconnected)}"
            )
        silent = self.never_nonzero()
        if silent:
            raise guards.HybridQQualificationError(
                f"ranker tensor(s) never had a nonzero gradient: {list(silent)}"
            )

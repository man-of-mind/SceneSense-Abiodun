"""Training-path primitives for hybrid-q. Phase 1 defines them; it runs none of them."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from . import contract, guards
from .selection import select_cells


NORMALIZATIONS = ("l1", "max")


# --------------------------------------------------------------------------
# Teacher importance maps
# --------------------------------------------------------------------------


def teacher_importance_map(c2: torch.Tensor, task_grad: torch.Tensor) -> torch.Tensor:
    """I_t(h,w) = sum_c |C2(c,h,w) * grad_t(c,h,w)|, returned as [H,W]."""
    if c2.shape != task_grad.shape:
        raise guards.HybridQPayloadError(
            f"gradient shape {tuple(task_grad.shape)} != C2 shape {tuple(c2.shape)}"
        )
    if c2.dim() != 3:
        raise guards.HybridQPayloadError("teacher maps are computed per frame on [C,H,W]")
    return (c2.detach() * task_grad.detach()).abs().sum(dim=0)


def normalize_importance(importance: torch.Tensor, scheme: str = "l1") -> torch.Tensor:
    """Normalize one task map independently of every other task."""
    if scheme not in NORMALIZATIONS:
        raise guards.HybridQConfigError(f"unknown normalization scheme {scheme!r}")
    guards.require_finite(importance, "teacher importance map")
    if (importance < 0).any():
        raise guards.HybridQNumericalError("teacher importance map must be non-negative")
    scale = importance.sum() if scheme == "l1" else importance.max()
    if not torch.isfinite(scale) or float(scale) <= 0.0:
        raise guards.HybridQNumericalError("teacher importance map has no positive mass")
    return importance / scale


@dataclass
class TeacherMapResult:
    """Per-frame teacher supervision with explicit task validity bookkeeping."""

    importance: torch.Tensor | None
    task_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_tasks: tuple[str, ...] = ()
    excluded_tasks: dict[str, str] = field(default_factory=dict)
    loss_scales: dict[str, float] = field(default_factory=dict)
    normalization: str = "l1"

    @property
    def is_supervisable(self) -> bool:
        return self.importance is not None and len(self.valid_tasks) > 0


def build_teacher_maps(
    c2: torch.Tensor,
    task_grads: Mapping[str, torch.Tensor | None],
    *,
    normalization: str = "l1",
    task_weights: Mapping[str, float] | None = None,
) -> TeacherMapResult:
    """Build one combined teacher map from per-task gradients.

    Each valid task is normalized independently before combination. Absent
    tasks, zero-gradient tasks and non-finite tasks are recorded in
    `excluded_tasks` and excluded; they are never silently treated as valid
    supervision. A frame with no valid task yields `importance=None`.
    """
    if normalization not in NORMALIZATIONS:
        raise guards.HybridQConfigError(f"unknown normalization scheme {normalization!r}")

    task_maps: dict[str, torch.Tensor] = {}
    excluded: dict[str, str] = {}
    loss_scales: dict[str, float] = {}

    for task in sorted(task_grads):
        grad = task_grads[task]
        if grad is None:
            excluded[task] = "absent"
            continue
        raw = teacher_importance_map(c2, grad)
        if not torch.isfinite(raw).all():
            excluded[task] = "non_finite"
            continue
        mass = float(raw.sum())
        if mass <= 0.0:
            excluded[task] = "zero_gradient"
            continue
        loss_scales[task] = mass
        task_maps[task] = normalize_importance(raw, normalization)

    valid = tuple(sorted(task_maps))
    if not valid:
        return TeacherMapResult(
            importance=None,
            task_maps={},
            valid_tasks=(),
            excluded_tasks=excluded,
            loss_scales={},
            normalization=normalization,
        )

    weights = {task: 1.0 for task in valid}
    if task_weights is not None:
        for task in valid:
            weight = float(task_weights.get(task, 1.0))
            if not weight > 0.0:
                raise guards.HybridQConfigError(f"task weight for {task!r} must be positive")
            weights[task] = weight

    total = sum(weights[task] for task in valid)
    combined = sum(task_maps[task] * (weights[task] / total) for task in valid)
    return TeacherMapResult(
        importance=normalize_importance(combined, normalization),
        task_maps=task_maps,
        valid_tasks=valid,
        excluded_tasks=excluded,
        loss_scales=loss_scales,
        normalization=normalization,
    )


@dataclass(frozen=True)
class TeacherCacheRecord:
    """Contract for a future teacher-cache artifact.

    Full C2 tensors are deliberately absent: caches hold teacher maps,
    validity flags, identifiers and loss-scale metadata only.
    """

    frame_id: str
    sequence_id: str
    importance: torch.Tensor
    valid_tasks: tuple[str, ...]
    excluded_tasks: dict[str, str]
    loss_scales: dict[str, float]
    normalization: str
    perception_checkpoint_sha256: str = contract.FROZEN_CHECKPOINT_SHA256

    def __post_init__(self) -> None:
        if self.importance.dim() != 2:
            raise guards.HybridQPayloadError("cached teacher importance must be [H,W]")


# --------------------------------------------------------------------------
# Ranker distillation loss
# --------------------------------------------------------------------------


def ranker_distillation_loss(
    scores: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Listwise soft cross-entropy between the ranker and the teacher map.

    `teacher` must already be an independently normalized distribution over
    cells. `temperature` is required from configuration.
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
# Hard-mask forward with straight-through surrogate
# --------------------------------------------------------------------------


def straight_through_mask(
    scores: torch.Tensor,
    q: float,
    *,
    temperature: float,
    registered_only: bool = True,
) -> torch.Tensor:
    """Hard exact-cardinality mask forward, sigmoid surrogate backward.

    The returned tensor equals the hard 0/1 mask numerically; only its
    gradient comes from the temperature-scaled surrogate. The temperature must
    be supplied explicitly by configuration.
    """
    tau = guards.require_positive_temperature(temperature)
    selection = select_cells(scores, q, registered_only=registered_only)
    hard = selection.keep_mask.to(scores.dtype)
    if selection.keep_count in (0, selection.cells):
        threshold = scores.reshape(-1).min().detach()
    else:
        kept = scores.reshape(-1)[selection.keep_indices]
        dropped = scores.reshape(-1)[~selection.keep_mask.reshape(-1)]
        threshold = ((kept.min() + dropped.max()) * 0.5).detach()
    soft = torch.sigmoid((scores - threshold) / tau)
    # Grouped so the forward value is exactly `hard`: (soft - soft.detach()) is
    # exactly zero, whereas (hard + soft) - soft.detach() rounds in float32.
    return hard + (soft - soft.detach())


def masked_c2_forward(
    c2: torch.Tensor, mask: torch.Tensor, *, detach_features: bool = True
) -> torch.Tensor:
    """Apply a [H,W] mask to every channel of C2, keeping the mask differentiable.

    `detach_features=True` keeps gradient flowing to the ranker through the
    mask only, never into the frozen perception trunk.
    """
    if mask.shape != c2.shape[-2:]:
        raise guards.HybridQPayloadError("mask shape does not match C2 spatial shape")
    features = c2.detach() if detach_features else c2
    return features * mask.unsqueeze(0).to(features.dtype)


# --------------------------------------------------------------------------
# Optimizer construction and frozen ownership
# --------------------------------------------------------------------------


def build_ranker_optimizer(
    ranker: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    frozen_modules: Iterable[nn.Module] = (),
) -> torch.optim.Optimizer:
    """Adam over ranker parameters only, with frozen-ownership checks applied."""
    frozen = tuple(frozen_modules)
    guards.require_frozen_perception(frozen)
    frozen_ids = {id(p) for module in frozen for p in module.parameters()}
    params = [p for p in ranker.parameters() if p.requires_grad]
    if not params:
        raise guards.HybridQConfigError("ranker has no trainable parameters")
    for parameter in params:
        if id(parameter) in frozen_ids:
            raise guards.HybridQOwnershipError(
                "ranker parameter list overlaps the frozen perception stack"
            )
    optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    guards.require_optimizer_owns_only(optimizer, params)
    return optimizer


# --------------------------------------------------------------------------
# Gradient qualification policy
# --------------------------------------------------------------------------


@dataclass
class GradientQualification:
    """Runtime gradient policy over a qualification window.

    Finiteness is required on every update. Nonzero-gradient evidence is
    required across the window, not per batch: an isolated zero-gradient batch
    is logged, never treated as a numerical failure.
    """

    window: int
    seen: int = 0
    nonzero_batches: int = 0
    zero_gradient_batches: list[int] = field(default_factory=list)

    def observe(self, parameters: Sequence[nn.Parameter]) -> bool:
        """Record one update. Returns whether this batch had nonzero gradient."""
        grads = [p.grad for p in parameters if p.grad is not None]
        for grad in grads:
            if not torch.isfinite(grad).all():
                raise guards.HybridQNumericalError("non-finite ranker gradient")
        nonzero = any(bool(grad.abs().sum() > 0) for grad in grads)
        self.seen += 1
        if nonzero:
            self.nonzero_batches += 1
        else:
            self.zero_gradient_batches.append(self.seen)
        return nonzero

    def window_complete(self) -> bool:
        return self.seen >= self.window

    def qualified(self) -> bool:
        """Nonzero-gradient evidence must exist somewhere in a completed window."""
        return self.window_complete() and self.nonzero_batches > 0

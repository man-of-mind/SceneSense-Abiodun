"""Exactly one Hybrid-SAC update (Phase 4b.2, part C).

``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.  This module performs one critic
step, one actor step and one Polyak target update against an
already-sampled :class:`ReplayTensorBatchV1`.  It exercises numerical and
contract mechanics.  It is **not** deployable policy training, not a
convergence result, and not evidence that Hybrid SAC improves SplitFusion.

Deliberately absent: any sampling loop, rollout, checkpoint system,
evaluation loop, prioritized replay, automatic entropy tuning, learning-rate
schedule, or long-running trainer.  Replay sampling stays caller-side with
its own generator; this trainer never draws a batch.

The discount rule
-----------------

The SMDP discount is taken **verbatim** from :meth:`ReplayTensorBatchV1.discount`,
which returns the contract's own derived ``discount_multiplier`` converted
once into the replay dtype.

The trainer must never re-derive it.  Rounding ``gamma`` into float32 and
*then* exponentiating is a different number from rounding the contract's
float64 power afterwards: at ``gamma = 0.99999999`` and ``d = 150`` the
re-derived value is exactly ``1.0`` and the correct one is
``0.9999985098838806``, so re-deriving silently deletes the discount.  For
that reason this module:

* calls ``batch.discount()`` and uses the result as-is;
* uses ``batch.bootstrap`` as-is;
* never calls :func:`hybrid_sac_models.critic_target`, whose present form
  builds ``torch.pow(tensor(gamma, dtype), duration)``; and
* contains no ``gamma ** duration`` anywhere.

``gamma_per_tensor`` survives only as binding metadata: the trainer proves its
configured value is exactly equal to the batch's binding before touching any
parameter, then never computes with it.  ``duration`` is likewise validation
and diagnostic metadata only.

The target
----------

Under ``no_grad``, with ``V`` evaluated only on bootstrap-eligible rows and
scattered back into a zero vector::

    next_value = zeros_like(reward)
    next_value[bootstrap] = soft_state_value(next_state[bootstrap])
    y = reward + where(bootstrap, discount * next_value, 0)

A row that does not bootstrap never reaches ``soft_state_value`` at all, so
the zero-filled sentinel in ``next_state`` is never evaluated as if it were a
real observation.

Update order
------------

1. Preflight every shape, dtype, device, binding, mask, range and finiteness
   check.  Nothing mutates until all of them pass, so a malformed batch leaves
   both the parameters and the optimizer state untouched.
2. Form the target under ``no_grad``.
3. One critic optimizer step on ``MSE(Q1, y) + MSE(Q2, y)``.
4. One actor optimizer step on the exact 12-mode enumeration, with the online
   critics' parameters frozen so no critic gradient is accumulated while
   ``dQ/dq`` still reaches the actor.
5. One Polyak target update.

There is no target actor, and no target parameter is ever in an optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX, Q_E4_MIN
from .hybrid_sac_models import (
    ConditionalHybridActor,
    TwinHybridCritics,
    actor_objective,
    mode_one_hot,
    soft_state_value,
)
from .replay_buffer import (
    Q_CRITIC_NORMALIZER,
    ReplayTensorBatchV1,
)
from .state_reward_transition_contract import POLICY_FEATURE_COUNT
from .transaction_identity import MINIMUM_HOLD_TENSORS

__all__ = [
    "HybridSacTrainerV1",
    "PHASE_LABEL",
    "PROVISIONAL_HYPERPARAMETERS",
    "TrainerConfigV1",
    "TrainerError",
    "TrainerPreflightError",
    "TrainerStateError",
    "UpdateMetricsV1",
]


PHASE_LABEL = "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY"

#: Smoke-test starting points.  **Provisional, not scientifically frozen.**
#: They are ordinary SAC defaults chosen so one update runs, not values
#: selected by any calibration study on this system.
PROVISIONAL_HYPERPARAMETERS: Dict[str, Any] = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "batch_size": 256,
    "tau": 0.005,
    "status": "PROVISIONAL_SMOKE_HYPOTHESIS_NOT_FROZEN",
}


class TrainerError(Exception):
    """Base class for every trainer failure."""


class TrainerPreflightError(TrainerError):
    """A batch or configuration failed validation; nothing was mutated."""


class TrainerStateError(TrainerError):
    """The trainer's own wiring is invalid (e.g. a target in an optimizer)."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TrainerConfigV1:
    """Immutable trainer configuration.

    ``gamma_per_tensor`` is carried **only** to be proved equal to the batch's
    replay binding.  No discount is computed from it; see the module
    docstring.  ``batch_size`` is a recorded provisional hypothesis, not an
    enforced constraint: :meth:`HybridSacTrainerV1.update_once` consumes
    whatever batch the caller sampled.
    """

    gamma_per_tensor: float
    alpha_d: float
    alpha_c: float
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    float_dtype: torch.dtype = torch.float32
    hyperparameter_status: str = "PROVISIONAL_SMOKE_HYPOTHESIS_NOT_FROZEN"

    def __post_init__(self) -> None:
        for name in ("alpha_d", "alpha_c"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
                raise TrainerPreflightError(
                    f"{name} must be a fixed finite positive float (this phase "
                    f"has no automatic entropy tuning), got {value!r}"
                )
        gamma = self.gamma_per_tensor
        if not isinstance(gamma, float) or not math.isfinite(gamma):
            raise TrainerPreflightError(
                f"gamma_per_tensor must be a finite float, got {gamma!r}"
            )
        if not 0.0 < gamma <= 1.0:
            raise TrainerPreflightError(
                f"gamma_per_tensor must lie in (0, 1], got {gamma}"
            )
        if not isinstance(self.tau, float) or not 0.0 < self.tau <= 1.0:
            raise TrainerPreflightError(
                f"tau must be a float in (0, 1], got {self.tau!r}"
            )
        for name in ("actor_lr", "critic_lr"):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
                raise TrainerPreflightError(
                    f"{name} must be a finite positive float, got {value!r}"
                )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TrainerPreflightError("batch_size must be an integer")
        if self.batch_size < 1:
            raise TrainerPreflightError("batch_size must be positive")
        if self.float_dtype is not torch.float32:
            raise TrainerPreflightError(
                f"this phase trains in float32 to match the replay binding, "
                f"got {self.float_dtype}"
            )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class UpdateMetricsV1:
    """Finite scalar diagnostics for exactly one update.

    Diagnostics only.  None of these is a learning-progress claim: a single
    update cannot show convergence, and this phase makes no such claim.
    """

    batch_size: int
    bootstrap_count: int
    bootstrap_fraction: float
    duration_mean: float
    duration_min: int
    duration_max: int

    reward_mean: float
    reward_min: float
    reward_max: float
    target_mean: float
    target_min: float
    target_max: float
    next_value_mean: float
    discount_min: float
    discount_max: float

    critic_1_loss: float
    critic_2_loss: float
    critic_loss_total: float
    actor_loss: float

    q1_mean: float
    q2_mean: float
    twin_gap_mean: float

    discrete_entropy: float
    conditional_logprob_mean: float
    conditional_entropy_estimate: float

    q_requested_min: float
    q_requested_max: float
    q_executed_min: float
    q_executed_max: float
    q_saturation_fraction: float

    critic_grad_norm: float
    actor_grad_norm: float
    actor_param_delta_norm: float
    online_critic_param_delta_norm: float
    target_param_delta_norm: float

    gamma_per_tensor: float
    phase_label: str = PHASE_LABEL

    def as_dict(self) -> Dict[str, Any]:
        """Plain serializable mapping of every diagnostic."""
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }

    def assert_finite(self) -> None:
        """Fail closed if any numeric diagnostic is not finite."""
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise TrainerError(f"diagnostic {name} is not finite: {value!r}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _param_snapshot(parameters) -> List[Tensor]:
    """Detached clones of parameters, for a later delta norm."""
    return [parameter.detach().clone() for parameter in parameters]


def _delta_norm(before: List[Tensor], after) -> float:
    """L2 norm of the concatenated parameter difference."""
    total = 0.0
    for old, new in zip(before, after):
        total += float(torch.sum((new.detach() - old) ** 2))
    return math.sqrt(total)


def _grad_norm(parameters) -> float:
    """L2 norm of the concatenated gradient, treating absent grads as zero."""
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad**2))
    return math.sqrt(total)


class _FrozenParameters:
    """Temporarily clear ``requires_grad`` on a set of parameters.

    Used so the actor step accumulates **no** critic gradient while gradient
    still propagates through the critics' activations to the actor -- freezing
    a parameter stops its ``.grad`` accumulation, it does not cut the graph to
    the critic's inputs, which is exactly the ``dQ/dq`` path the actor needs.
    """

    def __init__(self, parameters) -> None:
        self._parameters = list(parameters)
        self._previous: List[bool] = []

    def __enter__(self) -> "_FrozenParameters":
        self._previous = [p.requires_grad for p in self._parameters]
        for parameter in self._parameters:
            parameter.requires_grad_(False)
        return self

    def __exit__(self, *exc: Any) -> None:
        for parameter, previous in zip(self._parameters, self._previous):
            parameter.requires_grad_(previous)


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #


class HybridSacTrainerV1:
    """One-update Hybrid-SAC trainer over an already-sampled replay batch.

    Not thread-safe.  Holds no replay buffer and never samples: the caller
    owns replay sampling and its generator, and hands in a batch.
    """

    def __init__(
        self,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: TrainerConfigV1,
        *,
        target_generator: torch.Generator,
        actor_generator: torch.Generator,
    ) -> None:
        if not isinstance(actor, ConditionalHybridActor):
            raise TrainerStateError("actor must be a ConditionalHybridActor")
        if not isinstance(critics, TwinHybridCritics):
            raise TrainerStateError("critics must be TwinHybridCritics")
        if not isinstance(config, TrainerConfigV1):
            raise TrainerStateError("config must be a TrainerConfigV1")
        for name, generator in (
            ("target_generator", target_generator),
            ("actor_generator", actor_generator),
        ):
            if not isinstance(generator, torch.Generator):
                raise TrainerStateError(
                    f"{name} must be an explicit torch.Generator"
                )
            if generator is torch.default_generator:
                raise TrainerStateError(
                    f"{name} must not be the global default generator"
                )
            if generator.device.type != "cpu":
                raise TrainerStateError(f"{name} must be a CPU generator")
        if target_generator is actor_generator:
            raise TrainerStateError(
                "the target-q and actor-q streams must be separate generators"
            )
        for name, module in (("actor", actor), ("critics", critics)):
            if module.config.dtype is not config.float_dtype:
                raise TrainerStateError(
                    f"{name} is built in {module.config.dtype} but the trainer "
                    f"and the replay binding use {config.float_dtype}"
                )
            for parameter in module.parameters():
                if parameter.device.type != "cpu":
                    raise TrainerStateError(f"{name} must be on CPU")

        self.actor = actor
        self.critics = critics
        self.config = config
        self._target_generator = target_generator
        self._actor_generator = actor_generator

        self._online_critic_parameters = list(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self._online_critic_parameters, lr=config.critic_lr
        )
        self._assert_no_target_parameters_in_optimizers()
        self.update_count = 0

    # -- wiring proof ------------------------------------------------------ #

    def _assert_no_target_parameters_in_optimizers(self) -> None:
        """Fail closed if any Polyak target parameter is optimizer-visible."""
        target_ids = {
            id(parameter)
            for target in (self.critics.target_1, self.critics.target_2)
            for parameter in target.parameters()
        }
        for label, optimizer in (
            ("actor_optimizer", self.actor_optimizer),
            ("critic_optimizer", self.critic_optimizer),
        ):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in target_ids:
                        raise TrainerStateError(
                            f"{label} contains a Polyak target parameter; "
                            f"targets are updated only by polyak_update"
                        )
        # There is no target actor to guard against; assert that too.
        if hasattr(self.actor, "target"):  # pragma: no cover - defensive
            raise TrainerStateError("this phase has no target actor")

    # -- preflight --------------------------------------------------------- #

    def _preflight(self, batch: Any) -> None:
        """Validate everything before any parameter or optimizer mutation."""
        E = TrainerPreflightError
        if type(batch) is not ReplayTensorBatchV1:
            raise E(
                f"update_once consumes an exact ReplayTensorBatchV1, got "
                f"{type(batch).__name__}"
            )
        if batch.float_dtype is not self.config.float_dtype:
            raise E(
                f"batch dtype {batch.float_dtype} does not match the trainer's "
                f"{self.config.float_dtype}"
            )
        size = batch.batch_size
        if size < 1:
            raise E("batch is empty")

        # gamma is binding metadata only: prove exact equality, never compute.
        binding_gamma = batch.binding.gamma_per_tensor
        if binding_gamma != self.config.gamma_per_tensor:
            raise E(
                f"trainer gamma_per_tensor {self.config.gamma_per_tensor!r} is "
                f"not exactly the batch binding's {binding_gamma!r}"
            )
        if batch.binding.policy_feature_count != POLICY_FEATURE_COUNT:
            raise E("batch binding declares a foreign policy feature count")

        float_fields = {
            "state": (batch.state, (size, POLICY_FEATURE_COUNT)),
            "next_state": (batch.next_state, (size, POLICY_FEATURE_COUNT)),
            "reward": (batch.reward, (size,)),
            "discount": (batch.discount(), (size,)),
            "q_normalized_executed": (batch.q_normalized_executed, (size,)),
        }
        for name, (tensor, shape) in float_fields.items():
            if tuple(tensor.shape) != shape:
                raise E(
                    f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype is not self.config.float_dtype:
                raise E(
                    f"{name} must be {self.config.float_dtype}, got "
                    f"{tensor.dtype}"
                )
            if tensor.device.type != "cpu":
                raise E(f"{name} must be on CPU, got {tensor.device}")
            if not bool(torch.isfinite(tensor).all()):
                raise E(f"{name} contains a non-finite value")

        int_fields = {
            "mode_id": batch.mode_id,
            "q_e4": batch.q_e4,
            "duration": batch.duration,
        }
        for name, tensor in int_fields.items():
            if tuple(tensor.shape) != (size,):
                raise E(f"{name} must have shape ({size},)")
            if tensor.dtype is not torch.int64:
                raise E(f"{name} must be int64, got {tensor.dtype}")
            if tensor.device.type != "cpu":
                raise E(f"{name} must be on CPU")

        mask_fields = {
            "has_next_state": batch.has_next_state,
            "bootstrap": batch.bootstrap,
            "terminated": batch.terminated,
            "truncated": batch.truncated,
        }
        for name, tensor in mask_fields.items():
            if tuple(tensor.shape) != (size,):
                raise E(f"{name} must have shape ({size},)")
            if tensor.dtype is not torch.bool:
                raise E(f"{name} must be bool, got {tensor.dtype}")
            if tensor.device.type != "cpu":
                raise E(f"{name} must be on CPU")

        # The bootstrap rule is re-derived rather than trusted.
        expected = batch.has_next_state & (~batch.terminated)
        if not bool(torch.equal(batch.bootstrap, expected)):
            raise E(
                "batch.bootstrap is not has_next_state AND NOT terminated"
            )
        if bool((batch.terminated & batch.truncated).any()):
            raise E("a row is both terminated and truncated")

        discount = batch.discount()
        if bool((discount < 0.0).any()) or bool((discount > 1.0).any()):
            raise E("discount must lie in [0, 1]")

        modes = batch.mode_id
        if bool((modes < 0).any()) or bool((modes >= EXPECTED_MODE_COUNT).any()):
            raise E(f"mode_id must lie in [0, {EXPECTED_MODE_COUNT - 1}]")
        q_e4 = batch.q_e4
        if bool((q_e4 < Q_E4_MIN).any()) or bool((q_e4 > Q_E4_MAX).any()):
            raise E(f"q_e4 must lie in [{Q_E4_MIN}, {Q_E4_MAX}]")
        duration = batch.duration
        if bool((duration < MINIMUM_HOLD_TENSORS).any()):
            raise E(
                f"duration must be at least {MINIMUM_HOLD_TENSORS} tensors"
            )

    # -- the single update ------------------------------------------------- #

    def update_once(self, batch: ReplayTensorBatchV1) -> UpdateMetricsV1:
        """Run exactly one critic step, one actor step and one Polyak update.

        Args:
            batch: A batch the caller already sampled from a replay buffer.

        Returns:
            Finite scalar diagnostics for this update.

        Raises:
            TrainerPreflightError: on any invalid batch or binding.  Raised
                before any parameter or optimizer state changes.
        """
        self._preflight(batch)

        dtype = self.config.float_dtype
        state = batch.state
        next_state = batch.next_state
        reward = batch.reward
        bootstrap = batch.bootstrap
        discount = batch.discount()
        q_normalized = batch.q_normalized_executed
        modes = batch.mode_id

        actor_before = _param_snapshot(self.actor.parameters())
        online_before = _param_snapshot(self._online_critic_parameters)
        target_parameters = list(
            chain(
                self.critics.target_1.parameters(),
                self.critics.target_2.parameters(),
            )
        )
        target_before = _param_snapshot(target_parameters)

        # -- 1. target, under no_grad, bootstrap rows only ----------------- #
        with torch.no_grad():
            next_value = torch.zeros_like(reward)
            bootstrap_index = bootstrap.nonzero(as_tuple=False).squeeze(1)
            if int(bootstrap_index.numel()) > 0:
                # Only bootstrap-eligible successor states are evaluated, so a
                # zero-filled sentinel row never reaches the target critics.
                evaluated = soft_state_value(
                    self.actor,
                    self.critics,
                    next_state.index_select(0, bootstrap_index),
                    self.config.alpha_d,
                    self.config.alpha_c,
                    generator=self._target_generator,
                ).value
                next_value = next_value.index_copy(
                    0, bootstrap_index, evaluated.to(dtype)
                )
            # The discount is the batch's emitted value, used as-is.
            target = reward + torch.where(
                bootstrap,
                discount * next_value,
                torch.zeros_like(next_value),
            )
        if not bool(torch.isfinite(target).all()):
            raise TrainerError("the critic target is not finite")

        # -- 2. one critic step -------------------------------------------- #
        one_hot = mode_one_hot(modes, EXPECTED_MODE_COUNT, dtype)
        q1, q2 = self.critics.q_values(state, one_hot, q_normalized)
        critic_1_loss = torch.mean((q1 - target) ** 2)
        critic_2_loss = torch.mean((q2 - target) ** 2)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for parameter in self._online_critic_parameters:
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise TrainerError("a critic gradient is not finite")
        critic_grad_norm = _grad_norm(self._online_critic_parameters)
        self.critic_optimizer.step()

        # -- 3. one actor step --------------------------------------------- #
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        with _FrozenParameters(self._online_critic_parameters):
            objective = actor_objective(
                self.actor,
                self.critics,
                state,
                self.config.alpha_d,
                self.config.alpha_c,
                generator=self._actor_generator,
            )
            objective.objective.backward()
            contaminated = [
                index
                for index, parameter in enumerate(self._online_critic_parameters)
                if parameter.grad is not None
            ]
            if contaminated:
                raise TrainerError(
                    f"the actor step accumulated gradient on online critic "
                    f"parameters {contaminated}; the critic step's gradient "
                    f"would be corrupted"
                )
        for parameter in self.actor.parameters():
            if parameter.grad is None or not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise TrainerError("an actor gradient is absent or not finite")
        actor_grad_norm = _grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        # -- 4. one Polyak target update ----------------------------------- #
        self.critics.polyak_update(self.config.tau)
        self.update_count += 1

        metrics = self._metrics(
            batch=batch,
            target=target,
            next_value=next_value,
            discount=discount,
            bootstrap=bootstrap,
            q1=q1,
            q2=q2,
            critic_1_loss=critic_1_loss,
            critic_2_loss=critic_2_loss,
            critic_loss=critic_loss,
            objective=objective,
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=actor_grad_norm,
            actor_before=actor_before,
            online_before=online_before,
            target_before=target_before,
            target_parameters=target_parameters,
        )
        metrics.assert_finite()
        return metrics

    # -- diagnostics ------------------------------------------------------- #

    def _metrics(self, **parts: Any) -> UpdateMetricsV1:
        """Assemble the finite scalar diagnostics for one update."""
        batch: ReplayTensorBatchV1 = parts["batch"]
        target: Tensor = parts["target"]
        bootstrap: Tensor = parts["bootstrap"]
        discount: Tensor = parts["discount"]
        next_value: Tensor = parts["next_value"]
        q1: Tensor = parts["q1"].detach()
        q2: Tensor = parts["q2"].detach()
        objective = parts["objective"]
        sample = objective.sample
        probs = objective.probs.detach()

        size = batch.batch_size
        bootstrap_count = int(bootstrap.sum())
        duration = batch.duration.to(torch.float64)
        reward = batch.reward
        q_requested = sample.q.detach()
        q_executed = sample.q_executed.detach()
        q_e4_sampled = sample.q_e4
        saturated = (q_e4_sampled == Q_E4_MIN) | (q_e4_sampled == Q_E4_MAX)

        log_probs_d = sample.log_prob_discrete.detach()
        log_probs_c = sample.log_prob_continuous.detach()
        discrete_entropy = float(
            (-(probs * log_probs_d).sum(dim=-1)).mean()
        )
        conditional_logprob = float((probs * log_probs_c).sum(dim=-1).mean())

        return UpdateMetricsV1(
            batch_size=size,
            bootstrap_count=bootstrap_count,
            bootstrap_fraction=bootstrap_count / size,
            duration_mean=float(duration.mean()),
            duration_min=int(batch.duration.min()),
            duration_max=int(batch.duration.max()),
            reward_mean=float(reward.mean()),
            reward_min=float(reward.min()),
            reward_max=float(reward.max()),
            target_mean=float(target.mean()),
            target_min=float(target.min()),
            target_max=float(target.max()),
            next_value_mean=float(next_value.mean()),
            discount_min=float(discount.min()),
            discount_max=float(discount.max()),
            critic_1_loss=float(parts["critic_1_loss"].detach()),
            critic_2_loss=float(parts["critic_2_loss"].detach()),
            critic_loss_total=float(parts["critic_loss"].detach()),
            actor_loss=float(objective.objective.detach()),
            q1_mean=float(q1.mean()),
            q2_mean=float(q2.mean()),
            twin_gap_mean=float((q1 - q2).abs().mean()),
            discrete_entropy=discrete_entropy,
            conditional_logprob_mean=conditional_logprob,
            conditional_entropy_estimate=-conditional_logprob,
            q_requested_min=float(q_requested.min()),
            q_requested_max=float(q_requested.max()),
            q_executed_min=float(q_executed.min()),
            q_executed_max=float(q_executed.max()),
            q_saturation_fraction=float(saturated.to(torch.float64).mean()),
            critic_grad_norm=parts["critic_grad_norm"],
            actor_grad_norm=parts["actor_grad_norm"],
            actor_param_delta_norm=_delta_norm(
                parts["actor_before"], self.actor.parameters()
            ),
            online_critic_param_delta_norm=_delta_norm(
                parts["online_before"], self._online_critic_parameters
            ),
            target_param_delta_norm=_delta_norm(
                parts["target_before"], parts["target_parameters"]
            ),
            gamma_per_tensor=self.config.gamma_per_tensor,
        )

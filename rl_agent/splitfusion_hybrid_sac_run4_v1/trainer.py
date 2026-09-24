"""Fail-closed Hybrid-SAC update mechanics for the Run-4 contract.

This module adapts the already-tested Hybrid-SAC equations to the Run-4
21-feature state and verifier-gated replay boundary.  It deliberately does
not read evidence, create replay data, launch a runtime, initialize CUDA, or
start a training loop.

The production trainer accepts only a :class:`ReplayBindingV1` whose
calibration and queue-kernel evidence were attested by the Run-4 verifier.
At the time this module was added, no public attestation issuer existed; that
means production construction is intentionally blocked until the calibration
verifier is integrated.  A private mechanics-only trainer exists solely for
unit tests.  Its TEST_ONLY_MECHANICS batches can never enter the exported
production trainer.

The semi-Markov discount is consumed verbatim from ``batch.discount()``.
``duration`` is checked for validity and reported diagnostically, but is never
used to derive another discount.  Only rows carrying a real successor and an
eligible bootstrap mask reach the target critics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from itertools import chain
from typing import Any, Iterable, List

import torch
from torch import Tensor, nn

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    TwinHybridCritics,
    actor_objective,
    mode_one_hot,
    soft_state_value,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    MINIMUM_HOLD_TENSORS,
)

from .models import validate_run4_models
from .replay import ReplayBindingV1, ReplayTensorBatchV1
from .run4_contract import POLICY_FEATURE_COUNT, POLICY_FEATURE_ORDER

__all__ = [
    "RUN4_TRAINER_SCHEMA_ID",
    "Run4HybridSacTrainerV1",
    "TrainerConfigV1",
    "TrainerError",
    "TrainerPreflightError",
    "TrainerStateError",
    "UpdateMetricsV1",
]


RUN4_TRAINER_SCHEMA_ID = "splitfusion.run4.hybrid_sac_trainer.v1"
_TEST_ONLY_EVIDENCE_CLASS = "TEST_ONLY_MECHANICS"


class TrainerError(RuntimeError):
    """Base class for Run-4 trainer failures."""


class TrainerPreflightError(TrainerError):
    """A batch failed before any parameter, optimizer, or RNG mutation."""


class TrainerStateError(TrainerError):
    """The trainer's models, binding, generators, or optimizers are invalid."""


@dataclass(frozen=True, slots=True)
class TrainerConfigV1:
    """Explicit fixed-alpha SAC hyperparameters for one Run-4 update.

    There is intentionally no gamma field.  Gamma belongs to the replay
    binding and the contract-derived per-row discount belongs to each batch.
    Keeping it out of this configuration removes an accidental second source
    from which a discount could be reconstructed.
    """

    alpha_d: float
    alpha_c: float
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    nominal_batch_size: int = 256
    float_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        for name in ("alpha_d", "alpha_c", "tau", "actor_lr", "critic_lr"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TrainerStateError(f"{name} must be an exact finite float")
        if self.alpha_d <= 0.0 or self.alpha_c <= 0.0:
            raise TrainerStateError("alpha_d and alpha_c must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise TrainerStateError("tau must lie in (0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise TrainerStateError("actor_lr and critic_lr must be positive")
        if (
            type(self.nominal_batch_size) is not int
            or self.nominal_batch_size < 1
        ):
            raise TrainerStateError("nominal_batch_size must be a positive int")
        if self.float_dtype is not torch.float32:
            raise TrainerStateError("Run-4 replay and models require float32")


@dataclass(frozen=True, slots=True)
class UpdateMetricsV1:
    """Finite diagnostics for one update; not a convergence claim."""

    batch_size: int
    bootstrap_count: int
    reward_mean: float
    target_mean: float
    discount_min: float
    discount_max: float
    critic_loss: float
    actor_loss: float
    critic_grad_norm: float
    actor_grad_norm: float
    actor_delta_norm: float
    critic_delta_norm: float
    target_delta_norm: float
    discrete_entropy: float
    q_executed_mean: float
    update_index: int
    schema_id: str = RUN4_TRAINER_SCHEMA_ID

    def require_finite(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise TrainerError(
                    f"diagnostic {item.name} is not finite: {value!r}"
                )


def _snapshot(parameters: Iterable[nn.Parameter]) -> List[Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _delta_norm(before: List[Tensor], after: Iterable[nn.Parameter]) -> float:
    total = 0.0
    current = list(after)
    if len(before) != len(current):
        raise TrainerStateError("parameter collection changed during an update")
    for old, new in zip(before, current):
        difference = (new.detach() - old).to(torch.float64)
        total += float(torch.sum(difference * difference))
    return math.sqrt(total)


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(torch.float64)
            total += float(torch.sum(gradient * gradient))
    return math.sqrt(total)


def _require_finite_tensor(value: Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TrainerError(f"{label} contains a non-finite value")


class _FrozenParameters:
    """Freeze critic parameters while preserving the actor's dQ/dq path."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        self._parameters = list(parameters)
        self._previous: List[bool] = []

    def __enter__(self) -> "_FrozenParameters":
        self._previous = [item.requires_grad for item in self._parameters]
        for item in self._parameters:
            item.requires_grad_(False)
        return self

    def __exit__(self, *exc: Any) -> None:
        for item, previous in zip(self._parameters, self._previous):
            item.requires_grad_(previous)


class _Run4TrainerCore:
    """Shared equations; subclasses define the accepted evidence class."""

    def __init__(
        self,
        *,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: TrainerConfigV1,
        expected_binding: ReplayBindingV1,
        target_generator: torch.Generator,
        actor_generator: torch.Generator,
    ) -> None:
        if type(config) is not TrainerConfigV1:
            raise TrainerStateError("config must be an exact TrainerConfigV1")
        if type(expected_binding) is not ReplayBindingV1:
            raise TrainerStateError(
                "expected_binding must be an exact ReplayBindingV1"
            )
        self._require_evidence_class(expected_binding)
        if expected_binding.policy_feature_count != POLICY_FEATURE_COUNT:
            raise TrainerStateError("binding feature count differs from Run 4")
        if expected_binding.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise TrainerStateError("binding feature order differs from Run 4")
        try:
            validate_run4_models(actor, critics)
        except Exception as exc:
            raise TrainerStateError("models do not satisfy the Run-4 binding") from exc
        for name, generator in (
            ("target_generator", target_generator),
            ("actor_generator", actor_generator),
        ):
            if not isinstance(generator, torch.Generator):
                raise TrainerStateError(f"{name} must be a torch.Generator")
            if generator is torch.default_generator:
                raise TrainerStateError(f"{name} must not be the global generator")
            if generator.device.type != "cpu":
                raise TrainerStateError(f"{name} must be a CPU generator")
        if target_generator is actor_generator:
            raise TrainerStateError("target and actor generators must be distinct")

        self.actor = actor
        self.critics = critics
        self.config = config
        self.expected_binding = expected_binding
        self._target_generator = target_generator
        self._actor_generator = actor_generator
        self._online_critic_parameters = list(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self._online_critic_parameters, lr=config.critic_lr
        )
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _require_evidence_class(self, binding: ReplayBindingV1) -> None:
        raise NotImplementedError

    def _assert_optimizer_wiring(self) -> None:
        target_ids = {
            id(parameter)
            for target in (self.critics.target_1, self.critics.target_2)
            for parameter in target.parameters()
        }
        expected = (
            (self.actor_optimizer, tuple(self.actor.parameters()), "actor"),
            (
                self.critic_optimizer,
                tuple(self._online_critic_parameters),
                "critic",
            ),
        )
        for optimizer, parameters, label in expected:
            actual = [
                item
                for group in optimizer.param_groups
                for item in group["params"]
            ]
            actual_ids = [id(item) for item in actual]
            expected_ids = [id(item) for item in parameters]
            if len(actual_ids) != len(set(actual_ids)):
                raise TrainerStateError(f"{label} optimizer duplicates a parameter")
            if set(actual_ids) != set(expected_ids):
                raise TrainerStateError(
                    f"{label} optimizer does not own exactly its model parameters"
                )
            if target_ids.intersection(actual_ids):
                raise TrainerStateError(
                    f"{label} optimizer contains a Polyak target parameter"
                )

    def _preflight(self, batch: Any) -> None:
        """Validate every batch invariant before an update mutates anything."""

        error = TrainerPreflightError
        self._assert_optimizer_wiring()
        try:
            validate_run4_models(self.actor, self.critics)
        except Exception as exc:
            raise TrainerStateError("models changed after trainer construction") from exc
        if type(batch) is not ReplayTensorBatchV1:
            raise error("update_once requires an exact ReplayTensorBatchV1")
        try:
            self._require_evidence_class(batch.binding)
            self.expected_binding.assert_exactly(batch.binding)
        except Exception as exc:
            raise error("batch replay evidence/binding differs") from exc
        if batch.float_dtype is not torch.float32:
            raise error("batch must use CPU float32 replay tensors")
        size = batch.batch_size
        if size < 1:
            raise error("batch must not be empty")

        floats = {
            "state": (batch.state, (size, POLICY_FEATURE_COUNT)),
            "next_state": (batch.next_state, (size, POLICY_FEATURE_COUNT)),
            "reward": (batch.reward, (size,)),
            "discount": (batch.discount(), (size,)),
            "q_normalized_executed": (
                batch.q_normalized_executed,
                (size,),
            ),
        }
        for name, (tensor, shape) in floats.items():
            if tuple(tensor.shape) != shape:
                raise error(f"{name} must have shape {shape}")
            if tensor.device.type != "cpu" or tensor.dtype is not torch.float32:
                raise error(f"{name} must be a CPU float32 tensor")
            if not bool(torch.isfinite(tensor).all()):
                raise error(f"{name} contains a non-finite value")

        integers = {
            "mode_id": batch.mode_id,
            "q_e4": batch.q_e4,
            "duration": batch.duration,
        }
        for name, tensor in integers.items():
            if tuple(tensor.shape) != (size,):
                raise error(f"{name} must have shape ({size},)")
            if tensor.device.type != "cpu" or tensor.dtype is not torch.int64:
                raise error(f"{name} must be a CPU int64 tensor")

        masks = {
            "has_next_state": batch.has_next_state,
            "bootstrap": batch.bootstrap,
            "terminated": batch.terminated,
            "truncated": batch.truncated,
        }
        for name, tensor in masks.items():
            if tuple(tensor.shape) != (size,):
                raise error(f"{name} must have shape ({size},)")
            if tensor.device.type != "cpu" or tensor.dtype is not torch.bool:
                raise error(f"{name} must be a CPU bool tensor")

        has_next = batch.has_next_state
        terminated = batch.terminated
        truncated = batch.truncated
        bootstrap = batch.bootstrap
        if bool((terminated & truncated).any()):
            raise error("a row cannot be both terminated and truncated")
        expected_bootstrap = has_next & (~terminated) & (~truncated)
        if not bool(torch.equal(bootstrap, expected_bootstrap)):
            raise error("bootstrap is not an eligible real-successor mask")
        if not bool(torch.equal(has_next, ~(terminated | truncated))):
            raise error("successor presence disagrees with episode boundary")
        absent = (~has_next).nonzero(as_tuple=False).squeeze(1)
        if int(absent.numel()) and not bool(
            torch.equal(
                batch.next_state.index_select(0, absent),
                torch.zeros(
                    (int(absent.numel()), POLICY_FEATURE_COUNT),
                    dtype=torch.float32,
                ),
            )
        ):
            raise error("rows without successors must retain the zero sentinel")

        if bool((batch.mode_id < 0).any()) or bool(
            (batch.mode_id >= EXPECTED_MODE_COUNT).any()
        ):
            raise error("mode_id is outside the 12-mode action catalog")
        if bool((batch.q_e4 < Q_E4_MIN).any()) or bool(
            (batch.q_e4 > Q_E4_MAX).any()
        ):
            raise error("q_e4 is outside the registered execution range")
        if bool((batch.duration < MINIMUM_HOLD_TENSORS).any()):
            raise error("duration violates the minimum action hold")
        discount = batch.discount()
        if bool((discount <= 0.0).any()) or bool((discount > 1.0).any()):
            raise error("stored discount must lie in (0, 1]")

    def update_once(self, batch: ReplayTensorBatchV1) -> UpdateMetricsV1:
        """Perform one critic, actor, and Polyak update after full preflight."""

        self._preflight(batch)
        state = batch.state
        reward = batch.reward
        discount = batch.discount()
        bootstrap = batch.bootstrap

        actor_before = _snapshot(self.actor.parameters())
        critic_before = _snapshot(self._online_critic_parameters)
        target_parameters = list(
            chain(
                self.critics.target_1.parameters(),
                self.critics.target_2.parameters(),
            )
        )
        target_before = _snapshot(target_parameters)

        with torch.no_grad():
            next_value = torch.zeros_like(reward)
            eligible = bootstrap.nonzero(as_tuple=False).squeeze(1)
            if int(eligible.numel()):
                evaluated = soft_state_value(
                    self.actor,
                    self.critics,
                    batch.next_state.index_select(0, eligible),
                    self.config.alpha_d,
                    self.config.alpha_c,
                    generator=self._target_generator,
                ).value
                next_value = next_value.index_copy(0, eligible, evaluated)
            target = reward + torch.where(
                bootstrap,
                discount * next_value,
                torch.zeros_like(next_value),
            )
        _require_finite_tensor(target, "critic target")

        one_hot = mode_one_hot(
            batch.mode_id, EXPECTED_MODE_COUNT, torch.float32
        )
        q1, q2 = self.critics.q_values(
            state, one_hot, batch.q_normalized_executed
        )
        _require_finite_tensor(q1.detach(), "critic q1")
        _require_finite_tensor(q2.detach(), "critic q2")
        critic_loss = torch.mean((q1 - target) ** 2) + torch.mean(
            (q2 - target) ** 2
        )
        _require_finite_tensor(critic_loss, "critic loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for index, parameter in enumerate(self._online_critic_parameters):
            if parameter.grad is not None:
                _require_finite_tensor(parameter.grad, f"critic gradient {index}")
        critic_grad_norm = _gradient_norm(self._online_critic_parameters)
        if not math.isfinite(critic_grad_norm):
            raise TrainerError("critic gradient norm is non-finite")
        self.critic_optimizer.step()

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
            _require_finite_tensor(objective.objective, "actor loss")
            _require_finite_tensor(
                objective.per_mode_term.detach(), "actor per-mode term"
            )
            _require_finite_tensor(objective.probs.detach(), "actor probabilities")
            for name in (
                "log_prob_discrete",
                "log_prob_continuous",
                "q",
                "q_executed",
                "q_normalized_straight_through",
            ):
                _require_finite_tensor(
                    getattr(objective.sample, name).detach(),
                    f"actor sample {name}",
                )
            objective.objective.backward()
            if any(
                parameter.grad is not None
                for parameter in self._online_critic_parameters
            ):
                raise TrainerError("actor step accumulated critic gradients")
        for index, parameter in enumerate(self.actor.parameters()):
            if parameter.grad is None:
                raise TrainerError(f"actor gradient {index} is absent")
            _require_finite_tensor(parameter.grad, f"actor gradient {index}")
        actor_grad_norm = _gradient_norm(self.actor.parameters())
        if not math.isfinite(actor_grad_norm):
            raise TrainerError("actor gradient norm is non-finite")
        self.actor_optimizer.step()
        self.critics.polyak_update(self.config.tau)

        probabilities = objective.probs.detach()
        entropy = float(
            (-(probabilities * objective.sample.log_prob_discrete.detach()).sum(-1))
            .mean()
        )
        metrics = UpdateMetricsV1(
            batch_size=batch.batch_size,
            bootstrap_count=int(bootstrap.sum()),
            reward_mean=float(reward.mean()),
            target_mean=float(target.mean()),
            discount_min=float(discount.min()),
            discount_max=float(discount.max()),
            critic_loss=float(critic_loss.detach()),
            actor_loss=float(objective.objective.detach()),
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=actor_grad_norm,
            actor_delta_norm=_delta_norm(actor_before, self.actor.parameters()),
            critic_delta_norm=_delta_norm(
                critic_before, self._online_critic_parameters
            ),
            target_delta_norm=_delta_norm(target_before, target_parameters),
            discrete_entropy=entropy,
            q_executed_mean=float(objective.sample.q_executed.detach().mean()),
            update_index=self.update_count + 1,
        )
        metrics.require_finite()
        self.update_count += 1
        return metrics


class Run4HybridSacTrainerV1(_Run4TrainerCore):
    """Production trainer; only verifier-attested evidence may construct it."""

    def _require_evidence_class(self, binding: ReplayBindingV1) -> None:
        binding.require_training_eligible()


class _TestOnlyRun4HybridSacTrainerV1(_Run4TrainerCore):
    """Private equation harness; never accepts production replay evidence."""

    def _require_evidence_class(self, binding: ReplayBindingV1) -> None:
        binding.revalidate()
        if binding.evidence_eligibility != _TEST_ONLY_EVIDENCE_CLASS:
            raise TrainerStateError(
                "mechanics-only trainer requires TEST_ONLY_MECHANICS evidence"
            )

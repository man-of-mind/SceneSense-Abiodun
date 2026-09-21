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

Precision
---------

Everything here is float32: :class:`HybridSacModelConfig` defaults to
``torch.float32``, the replay binding is float32, and the trainer requires the
two to agree.  Sampled-``q`` quantization inside the actor therefore runs in
float32; that is a *training proposal*, and the exact wire quantization at the
execution boundary remains the registered decimal half-up rule in
``action_contract``, which this module never performs.

Diagnostic reductions -- gradient norms and parameter-delta norms -- are
accumulated in float64 instead.  Squaring a float32 gradient can overflow
while the gradient itself is perfectly finite: at a reward of ``-1e19`` the
critic gradients are all finite and the float64 norm is ``4.14e19``, but
squaring in float32 yields ``inf``.  A diagnostic must never veto an update
that is numerically sound.

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

from .action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)
from .hybrid_sac_models import (
    ConditionalHybridActor,
    HybridSacModelError,
    NORMALIZED_Z_DENSITY,
    PHYSICAL_Q_DENSITY,
    TwinHybridCritics,
    actor_objective,
    mode_one_hot,
    soft_state_value,
)
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT_SHA256
from .replay_buffer import (
    Q_CRITIC_NORMALIZER,
    ReplayBindingV1,
    ReplayTensorBatchV1,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
)
from .transaction_identity import MINIMUM_HOLD_TENSORS

__all__ = [
    "HybridSacTrainerV1",
    "PHASE_LABEL",
    "PROVISIONAL_HYPERPARAMETERS",
    "TrainerConfigV1",
    "TrainerError",
    "TrainerPreflightError",
    "TrainerStateError",
    "ModeledSmokeUpdateMetricsV1",
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
            for name in UpdateMetricsV1.__dataclass_fields__
        }

    @property
    def continuous_log_prob_coordinate(self) -> str:
        """Default metrics retain the legacy physical-q density coordinate."""
        return PHYSICAL_Q_DENSITY

    @property
    def modeled_smoke_support_sha256(self) -> Optional[str]:
        """Default full-range updates have no modeled-smoke support binding."""
        return None

    def assert_finite(self) -> None:
        """Fail closed if any numeric diagnostic is not finite."""
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise TrainerError(f"diagnostic {name} is not finite: {value!r}")
        if self.continuous_log_prob_coordinate not in (
            PHYSICAL_Q_DENSITY,
            NORMALIZED_Z_DENSITY,
        ):
            raise TrainerError(
                "diagnostic continuous_log_prob_coordinate is unknown"
            )
        if self.continuous_log_prob_coordinate == PHYSICAL_Q_DENSITY:
            if self.modeled_smoke_support_sha256 is not None:
                raise TrainerError(
                    "physical-q diagnostics cannot claim modeled-smoke support"
                )
        elif self.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise TrainerError(
                "normalized-z diagnostics require a canonical support SHA-256"
            )


@dataclass(frozen=True, slots=True)
class ModeledSmokeUpdateMetricsV1(UpdateMetricsV1):
    """Support-bound diagnostics without changing legacy checkpoint fields."""

    _density_coordinate_record: str = NORMALIZED_Z_DENSITY
    _support_sha256_record: str = MODELED_SMOKE_SUPPORT_SHA256

    @property
    def continuous_log_prob_coordinate(self) -> str:
        return self._density_coordinate_record

    @property
    def modeled_smoke_support_sha256(self) -> Optional[str]:
        return self._support_sha256_record

    def as_dict(self) -> Dict[str, Any]:
        document = UpdateMetricsV1.as_dict(self)
        document["continuous_log_prob_coordinate"] = (
            self.continuous_log_prob_coordinate
        )
        document["modeled_smoke_support_sha256"] = (
            self.modeled_smoke_support_sha256
        )
        return document


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _param_snapshot(parameters) -> List[Tensor]:
    """Detached clones of parameters, for a later delta norm."""
    return [parameter.detach().clone() for parameter in parameters]


def _delta_norm(before: List[Tensor], after) -> float:
    """L2 norm of the concatenated parameter difference, summed in float64."""
    total = 0.0
    for old, new in zip(before, after):
        difference = (new.detach() - old).to(torch.float64)
        total += float(torch.sum(difference * difference))
    return math.sqrt(total)


def _grad_norm(parameters) -> float:
    """L2 norm of the concatenated gradient, summed in float64.

    The squaring is done after widening to float64 on purpose.  A float32
    gradient near ``1e19`` is finite, but its square overflows float32 to
    ``inf``; accumulating in float32 would then fail an update whose
    gradients are entirely sound.  Absent gradients count as zero.
    """
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            widened = parameter.grad.detach().to(torch.float64)
            total += float(torch.sum(widened * widened))
    return math.sqrt(total)


def _require_finite_tensor(tensor: Tensor, name: str) -> None:
    """Fail closed unless every entry of ``tensor`` is finite."""
    if not bool(torch.isfinite(tensor).all()):
        raise TrainerError(f"{name} is not finite")


def _require_finite_scalar(value: float, name: str) -> None:
    """Fail closed unless ``value`` is a finite Python float."""
    if not math.isfinite(float(value)):
        raise TrainerError(f"{name} is not finite: {value!r}")


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
        expected_binding: ReplayBindingV1,
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
        self._validate_expected_binding(expected_binding, config)
        # Structure is validated against the real modules, before any
        # optimizer exists, so a malformed pair can never own optimizer state.
        self._validate_model_structure(actor, critics, config)

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
            actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self._online_critic_parameters, lr=config.critic_lr
        )
        self._assert_no_target_parameters_in_optimizers()
        self.update_count = 0

    # -- binding and structure --------------------------------------------- #

    @staticmethod
    def _validate_expected_binding(
        expected_binding: Any, config: TrainerConfigV1
    ) -> None:
        """Freeze the complete replay binding this trainer will ever accept.

        Every static field is checked against what this process actually has
        compiled in, so a binding carrying a foreign schema, a foreign catalog
        or a permuted feature order is refused at construction rather than
        learned from.  ``policy_feature_order`` is compared element by element,
        not merely by length: a reversed order has the right count and the
        wrong semantics for all 31 features.
        """
        if type(expected_binding) is not ReplayBindingV1:
            raise TrainerStateError(
                f"expected_binding must be an exact ReplayBindingV1, got "
                f"{type(expected_binding).__name__}"
            )
        static = (
            ("schema_id", expected_binding.schema_id, SCHEMA_ID),
            ("schema_version", expected_binding.schema_version, SCHEMA_VERSION),
            ("schema_sha256", expected_binding.schema_sha256, SCHEMA_SHA256),
            ("catalog_sha256", expected_binding.catalog_sha256, CATALOG_SHA256),
            (
                "policy_feature_order",
                expected_binding.policy_feature_order,
                tuple(POLICY_FEATURE_ORDER),
            ),
            (
                "policy_feature_count",
                expected_binding.policy_feature_count,
                POLICY_FEATURE_COUNT,
            ),
        )
        for name, declared, current in static:
            if declared != current:
                raise TrainerStateError(
                    f"expected_binding.{name} is {declared!r}, but this "
                    f"process is built against {current!r}"
                )
        if config.gamma_per_tensor != expected_binding.gamma_per_tensor:
            raise TrainerStateError(
                f"config.gamma_per_tensor {config.gamma_per_tensor!r} is not "
                f"exactly expected_binding.gamma_per_tensor "
                f"{expected_binding.gamma_per_tensor!r}"
            )

    @staticmethod
    def _first_linear(container: Any, label: str) -> nn.Linear:
        """Return the first ``nn.Linear`` inside a sequential trunk."""
        for module in container:
            if isinstance(module, nn.Linear):
                return module
        raise TrainerStateError(f"{label} contains no Linear layer")

    @classmethod
    def _validate_model_structure(
        cls,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: TrainerConfigV1,
    ) -> None:
        """Validate the real modules, not their declared configuration.

        ``module.config.dtype`` is metadata and can go stale: calling
        ``actor.double()`` changes every parameter while leaving the config
        saying ``float32``.  Every floating parameter and buffer is therefore
        inspected directly.
        """
        if not isinstance(actor, ConditionalHybridActor):
            raise TrainerStateError("actor must be a ConditionalHybridActor")
        if not isinstance(critics, TwinHybridCritics):
            raise TrainerStateError("critics must be TwinHybridCritics")

        modules = (
            ("actor", actor),
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        )
        for label, module in modules:
            for name, tensor in chain(
                module.named_parameters(), module.named_buffers()
            ):
                if tensor.device.type != "cpu":
                    raise TrainerStateError(
                        f"{label}.{name} is on {tensor.device}; this phase is "
                        f"CPU-only"
                    )
                if (
                    tensor.is_floating_point()
                    and tensor.dtype is not config.float_dtype
                ):
                    raise TrainerStateError(
                        f"{label}.{name} is {tensor.dtype}, but the trainer "
                        f"and the replay binding use {config.float_dtype}; "
                        f"declared config metadata is not trusted here"
                    )

        # Declared dimensions must match the frozen contracts ...
        for label, module in (
            ("actor", actor),
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        ):
            if module.config.state_dim != POLICY_FEATURE_COUNT:
                raise TrainerStateError(
                    f"{label} declares state_dim {module.config.state_dim}, "
                    f"but the frozen policy feature count is "
                    f"{POLICY_FEATURE_COUNT}"
                )
            if module.config.mode_count != EXPECTED_MODE_COUNT:
                raise TrainerStateError(
                    f"{label} declares mode_count {module.config.mode_count}, "
                    f"but the frozen catalog has {EXPECTED_MODE_COUNT} modes"
                )

        # ... and the real layer shapes must agree with them.
        encoder_input = cls._first_linear(actor.encoder, "actor.encoder")
        if encoder_input.in_features != POLICY_FEATURE_COUNT:
            raise TrainerStateError(
                f"the actor encoder accepts {encoder_input.in_features} "
                f"features, but the frozen policy state has "
                f"{POLICY_FEATURE_COUNT}"
            )
        for name in ("logit_head", "mean_head", "log_std_head"):
            head = getattr(actor, name)
            if head.out_features != EXPECTED_MODE_COUNT:
                raise TrainerStateError(
                    f"actor.{name} emits {head.out_features} outputs, but the "
                    f"frozen catalog has {EXPECTED_MODE_COUNT} joint modes"
                )
        critic_input_width = POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1
        for label, critic in (
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        ):
            trunk_input = cls._first_linear(critic.trunk, f"{label}.trunk")
            if trunk_input.in_features != critic_input_width:
                raise TrainerStateError(
                    f"{label} accepts {trunk_input.in_features} inputs, but "
                    f"[state, one_hot(mode), q] is {critic_input_width} wide"
                )
            if critic.value_head.out_features != 1:
                raise TrainerStateError(
                    f"{label} emits {critic.value_head.out_features} values; "
                    f"a critic returns one scalar"
                )
        if hasattr(actor, "target") or hasattr(critics, "target_actor"):
            raise TrainerStateError("this phase has no target actor")
        try:
            lower, upper = actor.active_q_e4_bounds()
            coordinate = actor.continuous_density_coordinate
            support_sha256 = actor.modeled_smoke_support_sha256
        except HybridSacModelError as exc:
            raise TrainerStateError(
                "actor support contract or registered buffers are invalid"
            ) from exc
        if lower.shape != (EXPECTED_MODE_COUNT,) or upper.shape != (
            EXPECTED_MODE_COUNT,
        ):
            raise TrainerStateError("actor support bounds have the wrong shape")
        if coordinate == PHYSICAL_Q_DENSITY and support_sha256 is not None:
            raise TrainerStateError(
                "physical-q actor unexpectedly declares modeled-smoke support"
            )
        if coordinate == NORMALIZED_Z_DENSITY and (
            support_sha256 != MODELED_SMOKE_SUPPORT_SHA256
        ):
            raise TrainerStateError(
                "normalized-z actor lacks registered modeled-smoke provenance"
            )

    # -- wiring proof ------------------------------------------------------ #

    def _assert_no_target_parameters_in_optimizers(self) -> None:
        """Prove each optimizer owns exactly its intended parameter set.

        Re-run at the start of every update, not only at construction: a
        parameter group added later -- a Polyak target slipped into the critic
        optimizer, say -- would otherwise be stepped like an online parameter
        and silently destroy the target's role.
        """
        target_ids = {
            id(parameter)
            for target in (self.critics.target_1, self.critics.target_2)
            for parameter in target.parameters()
        }
        expected = {
            "actor_optimizer": (
                self.actor_optimizer,
                [id(p) for p in self.actor.parameters()],
            ),
            "critic_optimizer": (
                self.critic_optimizer,
                [id(p) for p in self._online_critic_parameters],
            ),
        }
        for label, (optimizer, expected_ids) in expected.items():
            observed = [
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            if len(observed) != len(set(observed)):
                raise TrainerStateError(
                    f"{label} holds a duplicated parameter"
                )
            leaked = sorted(target_ids.intersection(observed))
            if leaked:
                raise TrainerStateError(
                    f"{label} contains {len(leaked)} Polyak target "
                    f"parameter(s); targets are updated only by polyak_update"
                )
            if set(observed) != set(expected_ids):
                foreign = len(set(observed) - set(expected_ids))
                missing = len(set(expected_ids) - set(observed))
                raise TrainerStateError(
                    f"{label} does not hold exactly its intended parameter "
                    f"set: {foreign} foreign, {missing} missing"
                )
        if hasattr(self.actor, "target") or hasattr(self.critics, "target_actor"):
            raise TrainerStateError("this phase has no target actor")

    # -- preflight --------------------------------------------------------- #

    def _preflight(self, batch: Any) -> None:
        """Validate everything before any parameter or optimizer mutation."""
        E = TrainerPreflightError
        try:
            self.actor.active_q_e4_bounds()
            self.actor.continuous_density_coordinate
            self.actor.modeled_smoke_support_sha256
        except HybridSacModelError as exc:
            raise E(
                "actor support contract or registered buffers changed after "
                "trainer construction"
            ) from exc
        # Wiring is re-proved before anything else touches a parameter.
        self._assert_no_target_parameters_in_optimizers()
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

        # The whole binding is frozen, not just gamma: reward spec,
        # normalization, freshness, schema, catalog, feature order and gamma
        # must all be the identical learning problem on every update.
        if type(batch.binding) is not ReplayBindingV1:
            raise E(
                f"batch.binding must be an exact ReplayBindingV1, got "
                f"{type(batch.binding).__name__}"
            )
        if batch.binding != self.expected_binding:
            differing = [
                name
                for name in (
                    "reward_spec_sha256",
                    "state_normalization_spec_sha256",
                    "freshness_policy_sha256",
                    "gamma_per_tensor",
                    "schema_id",
                    "schema_version",
                    "schema_sha256",
                    "catalog_sha256",
                    "policy_feature_order",
                    "policy_feature_count",
                )
                if getattr(batch.binding, name)
                != getattr(self.expected_binding, name)
            ]
            raise E(
                f"batch binding differs from the trainer's frozen binding in "
                f"{differing}; a trainer may not learn across two replay "
                f"learning problems"
            )
        # gamma is binding metadata only: proved equal, never computed with.
        if batch.binding.gamma_per_tensor != self.config.gamma_per_tensor:
            raise E(
                f"trainer gamma_per_tensor {self.config.gamma_per_tensor!r} is "
                f"not exactly the batch binding's "
                f"{batch.binding.gamma_per_tensor!r}"
            )

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
        _require_finite_tensor(target, "the critic target")

        # -- 2. one critic step -------------------------------------------- #
        one_hot = mode_one_hot(modes, EXPECTED_MODE_COUNT, dtype)
        q1, q2 = self.critics.q_values(state, one_hot, q_normalized)
        _require_finite_tensor(q1.detach(), "q1")
        _require_finite_tensor(q2.detach(), "q2")
        critic_1_loss = torch.mean((q1 - target) ** 2)
        critic_2_loss = torch.mean((q2 - target) ** 2)
        critic_loss = critic_1_loss + critic_2_loss
        # A finite reward can still square to infinity in float32: at a
        # reward near -1e20 the MSE overflows even though every input was
        # valid.  Refuse before the backward pass rather than step on inf.
        _require_finite_tensor(critic_1_loss.detach(), "critic_1_loss")
        _require_finite_tensor(critic_2_loss.detach(), "critic_2_loss")
        _require_finite_tensor(critic_loss.detach(), "the total critic loss")

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for index, parameter in enumerate(self._online_critic_parameters):
            if parameter.grad is not None:
                _require_finite_tensor(
                    parameter.grad, f"critic gradient {index}"
                )
        critic_grad_norm = _grad_norm(self._online_critic_parameters)
        _require_finite_scalar(critic_grad_norm, "critic_grad_norm")
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
            _require_finite_tensor(
                objective.objective.detach(), "the actor objective"
            )
            for name in (
                "per_mode_term",
                "probs",
            ):
                _require_finite_tensor(
                    getattr(objective, name).detach(), f"actor {name}"
                )
            for name in (
                "log_prob_discrete",
                "log_prob_continuous",
                "q",
                "q_normalized_straight_through",
            ):
                _require_finite_tensor(
                    getattr(objective.sample, name).detach(),
                    f"actor sample {name}",
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
        for index, parameter in enumerate(self.actor.parameters()):
            if parameter.grad is None:
                raise TrainerError(f"actor gradient {index} is absent")
            _require_finite_tensor(parameter.grad, f"actor gradient {index}")
        actor_grad_norm = _grad_norm(self.actor.parameters())
        _require_finite_scalar(actor_grad_norm, "actor_grad_norm")
        self.actor_optimizer.step()

        # -- 4. one Polyak target update ----------------------------------- #
        self.critics.polyak_update(self.config.tau)

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
        # Diagnostics are validated before the update is recorded, so a
        # non-finite diagnostic never leaves behind a counted update.
        metrics.assert_finite()
        self.update_count += 1
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
        saturation_lower, saturation_upper = self.actor.active_q_e4_bounds()
        if saturation_lower.shape != (EXPECTED_MODE_COUNT,) or (
            saturation_upper.shape != (EXPECTED_MODE_COUNT,)
        ):
            raise TrainerStateError(
                "actor returned malformed per-mode q_e4 saturation bounds"
            )
        saturated = (
            q_e4_sampled == saturation_lower.unsqueeze(0)
        ) | (q_e4_sampled == saturation_upper.unsqueeze(0))

        log_probs_d = sample.log_prob_discrete.detach()
        log_probs_c = sample.log_prob_continuous.detach()
        if (
            sample.continuous_log_prob_coordinate
            != self.actor.continuous_density_coordinate
        ):
            raise TrainerStateError(
                "actor sample density coordinate differs from actor semantics"
            )
        discrete_entropy = float(
            (-(probs * log_probs_d).sum(dim=-1)).mean()
        )
        conditional_logprob = float((probs * log_probs_c).sum(dim=-1).mean())

        metrics_type = (
            ModeledSmokeUpdateMetricsV1
            if self.actor.uses_modeled_smoke_support
            else UpdateMetricsV1
        )
        return metrics_type(
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

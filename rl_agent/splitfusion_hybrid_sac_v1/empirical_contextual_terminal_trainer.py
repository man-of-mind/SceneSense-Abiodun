"""One numerical Hybrid-SAC update for terminal D1 contextual samples.

The fitted target is the observed scalar reward verbatim.  The Polyak copies
are retained only so checkpoints stay structurally compatible with the
existing twin-critic model; they are never evaluated while forming the
target.  The actor objective enumerates all 12 modes exactly and samples only
the one-dimensional conditional action in the bounded normalized-z density.

This is a CPU-only, private empirical-pilot path.  It does not alter or adapt
the production trainer and does not accept a production replay batch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn

from .action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX
from .empirical_contextual_terminal_replay import (
    EmpiricalTerminalBatchV1,
    EmpiricalTerminalBindingV1,
    PHASE_LABEL as REPLAY_PHASE_LABEL,
)
from .hybrid_sac_models import (
    ConditionalHybridActor,
    HybridSacModelConfig,
    HybridSacModelError,
    NORMALIZED_Z_DENSITY,
    TwinHybridCritics,
    actor_objective,
    mode_one_hot,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "EmpiricalTerminalHybridSacTrainerV1",
    "EmpiricalTerminalTrainerConfigV1",
    "EmpiricalTerminalUpdateMetricsV1",
    "PHASE_LABEL",
    "TrainerError",
    "TrainerPreflightError",
    "TrainerStateError",
]


PHASE_LABEL = "D2A_EMPIRICAL_TERMINAL_CONTEXTUAL_HYBRID_SAC_TRAINING_ONLY"


class TrainerError(Exception):
    """Base class for the private terminal contextual trainer."""


class TrainerPreflightError(TrainerError):
    """A batch failed before any parameter or optimizer mutation."""


class TrainerStateError(TrainerError):
    """The trainer's own model, random stream, or optimizer wiring is invalid."""


@dataclass(frozen=True, slots=True)
class EmpiricalTerminalTrainerConfigV1:
    """Provisional smoke hyperparameters for one terminal contextual update."""

    alpha_d: float = 0.10
    alpha_c: float = 0.05
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    float_dtype: torch.dtype = torch.float32
    hyperparameter_status: str = "PROVISIONAL_EMPIRICAL_SMOKE_NOT_FROZEN"

    def __post_init__(self) -> None:
        for name in ("alpha_d", "alpha_c", "tau", "actor_lr", "critic_lr"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise TrainerPreflightError(
                    f"{name} must be an exact finite positive float"
                )
        if self.tau > 1.0:
            raise TrainerPreflightError("tau must lie in (0, 1]")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise TrainerPreflightError(
                "batch_size must be an exact positive integer"
            )
        if self.float_dtype is not torch.float32:
            raise TrainerPreflightError("D2a is fixed to torch.float32")
        if (
            type(self.hyperparameter_status) is not str
            or not self.hyperparameter_status
        ):
            raise TrainerPreflightError("hyperparameter status must be non-empty")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "actor_lr": self.actor_lr,
            "alpha_c": self.alpha_c,
            "alpha_d": self.alpha_d,
            "batch_size": self.batch_size,
            "critic_lr": self.critic_lr,
            "float_dtype": str(self.float_dtype),
            "hyperparameter_status": self.hyperparameter_status,
            "record": "empirical_terminal_trainer_config_v1",
            "tau": self.tau,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class EmpiricalTerminalUpdateMetricsV1:
    batch_size: int
    reward_mean: float
    reward_min: float
    reward_max: float
    target_mean: float
    target_min: float
    target_max: float
    target_reward_max_abs_diff: float
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
    continuous_log_prob_coordinate: str
    modeled_smoke_support_sha256: str
    replay_binding_sha256: str
    trainer_config_sha256: str
    replay_phase_label: str = REPLAY_PHASE_LABEL
    phase_label: str = PHASE_LABEL

    def as_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def assert_finite(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise TrainerError(f"diagnostic {name} is not finite: {value!r}")
        if self.target_reward_max_abs_diff != 0.0:
            raise TrainerError("terminal critic target differs from reward")
        if self.continuous_log_prob_coordinate != NORMALIZED_Z_DENSITY:
            raise TrainerError("actor density is not normalized-z")
        if self.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise TrainerError("actor modeled-support digest drift")


def _snapshot(parameters: Iterable[nn.Parameter]) -> List[Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _delta_norm(before: List[Tensor], after: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for old, new in zip(before, after):
        delta = (new.detach() - old).to(torch.float64)
        total += float(torch.sum(delta * delta))
    return math.sqrt(total)


def _grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(torch.float64)
            total += float(torch.sum(gradient * gradient))
    return math.sqrt(total)


def _require_finite_tensor(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TrainerError(f"{name} contains a non-finite value")


def _require_finite_scalar(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise TrainerError(f"{name} is not finite: {value!r}")


class EmpiricalTerminalHybridSacTrainerV1:
    """One-update Hybrid-SAC trainer over a sampled D1 terminal batch."""

    def __init__(
        self,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: EmpiricalTerminalTrainerConfigV1,
        *,
        expected_binding: EmpiricalTerminalBindingV1,
        actor_generator: torch.Generator,
    ) -> None:
        if type(actor) is not ConditionalHybridActor:
            raise TrainerStateError("actor must be an exact ConditionalHybridActor")
        if type(critics) is not TwinHybridCritics:
            raise TrainerStateError("critics must be exact TwinHybridCritics")
        if type(config) is not EmpiricalTerminalTrainerConfigV1:
            raise TrainerStateError("config has a foreign type")
        if type(expected_binding) is not EmpiricalTerminalBindingV1:
            raise TrainerStateError("expected_binding has a foreign type")
        if not isinstance(actor_generator, torch.Generator):
            raise TrainerStateError("actor_generator must be explicit")
        if actor_generator is torch.default_generator:
            raise TrainerStateError("global default generator is forbidden")
        if actor_generator.device.type != "cpu":
            raise TrainerStateError("actor_generator must be a CPU generator")
        self._validate_binding(expected_binding)
        self._validate_models(actor, critics, config)

        self.actor = actor
        self.critics = critics
        self.config = config
        self._config_sha256 = config.canonical_sha256()
        self.expected_binding = expected_binding
        self._expected_binding_sha256 = expected_binding.canonical_sha256()
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
        self._assert_optimizer_wiring()
        self.update_count = 0

    @staticmethod
    def _validate_binding(binding: EmpiricalTerminalBindingV1) -> None:
        if binding.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise TrainerStateError("binding feature order drift")
        if binding.policy_feature_count != POLICY_FEATURE_COUNT:
            raise TrainerStateError("binding feature width drift")
        if binding.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise TrainerStateError("binding modeled support digest drift")
        if binding.float_dtype != str(torch.float32):
            raise TrainerStateError("binding dtype is not torch.float32")

    @staticmethod
    def _first_linear(container: Any, name: str) -> nn.Linear:
        for module in container:
            if isinstance(module, nn.Linear):
                return module
        raise TrainerStateError(f"{name} contains no Linear layer")

    @classmethod
    def _validate_models(
        cls,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: EmpiricalTerminalTrainerConfigV1,
    ) -> None:
        modules = (
            ("actor", actor),
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        )
        parameter_ids: Dict[str, set[int]] = {}
        for label, module in modules:
            if type(module.config) is not HybridSacModelConfig:
                raise TrainerStateError(f"{label} config has a foreign type")
            try:
                module.config.__post_init__()
            except HybridSacModelError as exc:
                raise TrainerStateError(f"{label} config is invalid") from exc
            if module.config != actor.config:
                raise TrainerStateError(
                    f"{label} config differs from the bounded actor config"
                )
            if module.config.state_dim != POLICY_FEATURE_COUNT:
                raise TrainerStateError(f"{label} state width drift")
            if module.config.mode_count != EXPECTED_MODE_COUNT:
                raise TrainerStateError(f"{label} mode count drift")
            for name, value in chain(
                module.named_parameters(), module.named_buffers()
            ):
                if value.device.type != "cpu":
                    raise TrainerStateError(f"{label}.{name} is not on CPU")
                if value.is_floating_point() and value.dtype is not config.float_dtype:
                    raise TrainerStateError(
                        f"{label}.{name} is {value.dtype}, not torch.float32"
                    )
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise TrainerStateError(f"{label}.{name} is non-finite")
            parameters = tuple(module.parameters())
            parameter_ids[label] = {id(parameter) for parameter in parameters}
            should_require_gradient = label in ("actor", "critic_1", "critic_2")
            for index, parameter in enumerate(parameters):
                if parameter.requires_grad is not should_require_gradient:
                    required = "trainable" if should_require_gradient else "frozen"
                    raise TrainerStateError(
                        f"{label} parameter {index} must remain {required}"
                    )
        labels = tuple(parameter_ids)
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                shared = parameter_ids[left].intersection(parameter_ids[right])
                if shared:
                    raise TrainerStateError(
                        f"{left} and {right} share {len(shared)} parameter "
                        "identity/identities; actor, twins and targets must "
                        "be pairwise disjoint"
                    )
        if cls._first_linear(actor.encoder, "actor.encoder").in_features != (
            POLICY_FEATURE_COUNT
        ):
            raise TrainerStateError("real actor input width drift")
        for name in ("logit_head", "mean_head", "log_std_head"):
            if getattr(actor, name).out_features != EXPECTED_MODE_COUNT:
                raise TrainerStateError(f"actor {name} output width drift")
        expected_critic_width = POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1
        for label, critic in (
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        ):
            if cls._first_linear(critic.trunk, label).in_features != expected_critic_width:
                raise TrainerStateError(f"{label} real input width drift")
            if critic.value_head.out_features != 1:
                raise TrainerStateError(f"{label} output width drift")
        try:
            if not actor.uses_modeled_smoke_support:
                raise TrainerStateError("D2a actor must be bounded by modeled support")
            if actor.continuous_density_coordinate != NORMALIZED_Z_DENSITY:
                raise TrainerStateError("D2a actor must use normalized-z density")
            if actor.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
                raise TrainerStateError("actor modeled support digest drift")
            lower, upper = actor.active_q_e4_bounds()
        except HybridSacModelError as exc:
            raise TrainerStateError("actor support validation failed") from exc
        expected_bounds = torch.tensor(
            MODELED_SMOKE_SUPPORT.mode_q_e4_bounds, dtype=torch.int64
        )
        if not torch.equal(lower, expected_bounds[:, 0]) or not torch.equal(
            upper, expected_bounds[:, 1]
        ):
            raise TrainerStateError("actor support bounds differ from contract")
        # The loop above proves online parameters trainable, target parameters
        # frozen, and all five networks pairwise identity-disjoint.

    def _assert_optimizer_wiring(self) -> None:
        live_online = list(
            chain(
                self.critics.critic_1.parameters(),
                self.critics.critic_2.parameters(),
            )
        )
        captured_ids = [id(parameter) for parameter in self._online_critic_parameters]
        live_ids = [id(parameter) for parameter in live_online]
        if captured_ids != live_ids:
            raise TrainerStateError(
                "the live online-critic parameter sequence differs from the "
                "constructor-captured sequence; a critic module or parameter "
                "was replaced after optimizer construction"
            )
        targets = {
            id(parameter)
            for module in (self.critics.target_1, self.critics.target_2)
            for parameter in module.parameters()
        }
        expected = (
            (
                "actor_optimizer",
                self.actor_optimizer,
                [id(parameter) for parameter in self.actor.parameters()],
            ),
            (
                "critic_optimizer",
                self.critic_optimizer,
                live_ids,
            ),
        )
        for label, optimizer, required in expected:
            observed = [
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            if len(observed) != len(set(observed)):
                raise TrainerStateError(f"{label} contains a duplicate parameter")
            if targets.intersection(observed):
                raise TrainerStateError(f"{label} contains a Polyak target")
            if set(observed) != set(required):
                raise TrainerStateError(f"{label} parameter set drift")

    def _preflight(self, batch: Any) -> None:
        try:
            self.config.__post_init__()
        except TrainerPreflightError as exc:
            raise TrainerStateError("trainer config was mutated") from exc
        if self.config.canonical_sha256() != self._config_sha256:
            raise TrainerStateError(
                "trainer config differs from its construction-time digest"
            )
        self._validate_models(self.actor, self.critics, self.config)
        self._assert_optimizer_wiring()
        if self.expected_binding.canonical_sha256() != (
            self._expected_binding_sha256
        ):
            raise TrainerStateError("trainer expected binding was mutated")
        if type(batch) is not EmpiricalTerminalBatchV1:
            raise TrainerPreflightError(
                "update_once requires an exact EmpiricalTerminalBatchV1"
            )
        if type(batch.binding) is not EmpiricalTerminalBindingV1:
            raise TrainerPreflightError("batch binding has a foreign type")
        if batch.binding != self.expected_binding or (
            batch.binding.canonical_sha256() != self._expected_binding_sha256
        ):
            raise TrainerPreflightError("batch binding differs from trainer binding")
        if batch.float_dtype is not torch.float32:
            raise TrainerPreflightError("batch dtype is not torch.float32")
        size = batch.batch_size
        if size < 1:
            raise TrainerPreflightError("batch is empty")

        state = batch.state
        reward = batch.reward
        q_normalized = batch.q_normalized_executed
        mode = batch.mode_id
        q_e4 = batch.q_e4
        for name, tensor, shape in (
            ("state", state, (size, POLICY_FEATURE_COUNT)),
            ("reward", reward, (size,)),
            ("q_normalized_executed", q_normalized, (size,)),
        ):
            if tuple(tensor.shape) != shape:
                raise TrainerPreflightError(f"{name} has shape {tuple(tensor.shape)}")
            if tensor.dtype is not torch.float32 or tensor.device.type != "cpu":
                raise TrainerPreflightError(f"{name} is not CPU float32")
            if not bool(torch.isfinite(tensor).all()):
                raise TrainerPreflightError(f"{name} contains a non-finite value")
        for name, tensor in (("mode_id", mode), ("q_e4", q_e4)):
            if tuple(tensor.shape) != (size,) or tensor.dtype is not torch.int64:
                raise TrainerPreflightError(f"{name} is not [B] int64")
            if tensor.device.type != "cpu":
                raise TrainerPreflightError(f"{name} is not on CPU")
        if bool((mode < 0).any()) or bool((mode >= EXPECTED_MODE_COUNT).any()):
            raise TrainerPreflightError("mode_id lies outside [0, 11]")
        bounds = torch.tensor(
            MODELED_SMOKE_SUPPORT.mode_q_e4_bounds, dtype=torch.int64
        )
        lower = bounds[:, 0].gather(0, mode)
        upper = bounds[:, 1].gather(0, mode)
        if bool(((q_e4 < lower) | (q_e4 > upper)).any()):
            raise TrainerPreflightError("q_e4 is outside its mode-specific support")
        expected_q = q_e4.to(torch.float32) / float(Q_E4_MAX)
        if not torch.equal(q_normalized, expected_q):
            raise TrainerPreflightError(
                "critic action is not exactly executed q_e4 / 9800"
            )

    def _transaction_snapshot(self) -> Dict[str, Any]:
        """Capture all mutable update state for exception-safe rollback."""
        parameters = tuple(
            chain(self.actor.parameters(), self.critics.parameters())
        )
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critics": copy.deepcopy(self.critics.state_dict()),
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic_optimizer": copy.deepcopy(self.critic_optimizer.state_dict()),
            "actor_generator": self._actor_generator.get_state().clone(),
            "gradients": tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in parameters
            ),
            "parameters": parameters,
            "update_count": self.update_count,
        }

    def _restore_transaction(self, snapshot: Dict[str, Any]) -> None:
        self.actor.load_state_dict(snapshot["actor"], strict=True)
        self.critics.load_state_dict(snapshot["critics"], strict=True)
        self.actor_optimizer.load_state_dict(snapshot["actor_optimizer"])
        self.critic_optimizer.load_state_dict(snapshot["critic_optimizer"])
        self._actor_generator.set_state(snapshot["actor_generator"])
        for parameter, gradient in zip(
            snapshot["parameters"], snapshot["gradients"]
        ):
            parameter.grad = None if gradient is None else gradient.clone()
        self.update_count = snapshot["update_count"]

    def update_once(
        self, batch: EmpiricalTerminalBatchV1
    ) -> EmpiricalTerminalUpdateMetricsV1:
        """Run one atomic critic, actor and structural target-copy update."""
        self._preflight(batch)
        snapshot = self._transaction_snapshot()
        try:
            return self._update_once_after_preflight(batch)
        except BaseException:
            try:
                self._restore_transaction(snapshot)
            except BaseException as rollback_error:
                raise TrainerStateError(
                    "an update failed and its transactional rollback also failed"
                ) from rollback_error
            raise

    def _update_once_after_preflight(
        self, batch: EmpiricalTerminalBatchV1
    ) -> EmpiricalTerminalUpdateMetricsV1:
        """Mutation section entered only after complete preflight."""
        state = batch.state
        reward = batch.reward
        modes = batch.mode_id
        q_normalized = batch.q_normalized_executed

        actor_before = _snapshot(self.actor.parameters())
        online_before = _snapshot(self._online_critic_parameters)
        target_parameters = list(
            chain(
                self.critics.target_1.parameters(),
                self.critics.target_2.parameters(),
            )
        )
        target_before = _snapshot(target_parameters)

        # This assignment is the entire one-step terminal Bellman target.
        target = reward.clone()
        if not torch.equal(target, reward):
            raise TrainerError("terminal target is not bit-identical to reward")

        one_hot = mode_one_hot(modes, EXPECTED_MODE_COUNT, torch.float32)
        q1, q2 = self.critics.q_values(state, one_hot, q_normalized)
        _require_finite_tensor(q1.detach(), "q1")
        _require_finite_tensor(q2.detach(), "q2")
        critic_1_loss = torch.mean((q1 - target) ** 2)
        critic_2_loss = torch.mean((q2 - target) ** 2)
        critic_loss = critic_1_loss + critic_2_loss
        _require_finite_tensor(critic_1_loss.detach(), "critic_1_loss")
        _require_finite_tensor(critic_2_loss.detach(), "critic_2_loss")
        _require_finite_tensor(critic_loss.detach(), "critic_loss_total")

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for index, parameter in enumerate(self._online_critic_parameters):
            if parameter.grad is not None:
                _require_finite_tensor(parameter.grad, f"critic gradient {index}")
        critic_grad_norm = _grad_norm(self._online_critic_parameters)
        _require_finite_scalar(critic_grad_norm, "critic_grad_norm")
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        objective = actor_objective(
            self.actor,
            self.critics,
            state,
            self.config.alpha_d,
            self.config.alpha_c,
            generator=self._actor_generator,
        )
        if objective.sample.continuous_log_prob_coordinate != NORMALIZED_Z_DENSITY:
            raise TrainerError("actor sample is not a normalized-z density")
        for name, tensor in (
            ("actor objective", objective.objective),
            ("actor per-mode term", objective.per_mode_term),
            ("actor probabilities", objective.probs),
            ("actor discrete log probability", objective.sample.log_prob_discrete),
            ("actor conditional log probability", objective.sample.log_prob_continuous),
            ("actor requested q", objective.sample.q),
            (
                "actor straight-through q",
                objective.sample.q_normalized_straight_through,
            ),
        ):
            _require_finite_tensor(tensor.detach(), name)
        objective.objective.backward()
        contaminated = [
            index
            for index, parameter in enumerate(self._online_critic_parameters)
            if parameter.grad is not None
        ]
        if contaminated:
            raise TrainerError(
                f"actor backward contaminated critic gradients {contaminated}"
            )
        for index, parameter in enumerate(self.actor.parameters()):
            if parameter.grad is None:
                raise TrainerError(f"actor gradient {index} is absent")
            _require_finite_tensor(parameter.grad, f"actor gradient {index}")
        actor_grad_norm = _grad_norm(self.actor.parameters())
        _require_finite_scalar(actor_grad_norm, "actor_grad_norm")
        self.actor_optimizer.step()

        # These copies remain checkpoint structure only; no target copy is
        # evaluated above while forming the fitted scalar.
        self.critics.polyak_update(self.config.tau)

        sample = objective.sample
        probabilities = objective.probs.detach()
        log_d = sample.log_prob_discrete.detach()
        log_c = sample.log_prob_continuous.detach()
        sampled_q_e4 = sample.q_e4.detach()
        lower, upper = self.actor.active_q_e4_bounds()
        saturated = (sampled_q_e4 == lower.unsqueeze(0)) | (
            sampled_q_e4 == upper.unsqueeze(0)
        )
        q1_detached = q1.detach()
        q2_detached = q2.detach()
        metrics = EmpiricalTerminalUpdateMetricsV1(
            batch_size=batch.batch_size,
            reward_mean=float(reward.mean()),
            reward_min=float(reward.min()),
            reward_max=float(reward.max()),
            target_mean=float(target.mean()),
            target_min=float(target.min()),
            target_max=float(target.max()),
            target_reward_max_abs_diff=float((target - reward).abs().max()),
            critic_1_loss=float(critic_1_loss.detach()),
            critic_2_loss=float(critic_2_loss.detach()),
            critic_loss_total=float(critic_loss.detach()),
            actor_loss=float(objective.objective.detach()),
            q1_mean=float(q1_detached.mean()),
            q2_mean=float(q2_detached.mean()),
            twin_gap_mean=float((q1_detached - q2_detached).abs().mean()),
            discrete_entropy=float(
                (-(probabilities * log_d).sum(dim=-1)).mean()
            ),
            conditional_logprob_mean=float(
                (probabilities * log_c).sum(dim=-1).mean()
            ),
            conditional_entropy_estimate=float(
                -(probabilities * log_c).sum(dim=-1).mean()
            ),
            q_requested_min=float(sample.q.detach().min()),
            q_requested_max=float(sample.q.detach().max()),
            q_executed_min=float(sample.q_executed.detach().min()),
            q_executed_max=float(sample.q_executed.detach().max()),
            q_saturation_fraction=float(saturated.to(torch.float64).mean()),
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=actor_grad_norm,
            actor_param_delta_norm=_delta_norm(
                actor_before, self.actor.parameters()
            ),
            online_critic_param_delta_norm=_delta_norm(
                online_before, self._online_critic_parameters
            ),
            target_param_delta_norm=_delta_norm(
                target_before, target_parameters
            ),
            continuous_log_prob_coordinate=(
                sample.continuous_log_prob_coordinate
            ),
            modeled_smoke_support_sha256=(
                self.actor.modeled_smoke_support_sha256 or ""
            ),
            replay_binding_sha256=self._expected_binding_sha256,
            trainer_config_sha256=self._config_sha256,
        )
        metrics.assert_finite()
        self.update_count += 1
        return metrics

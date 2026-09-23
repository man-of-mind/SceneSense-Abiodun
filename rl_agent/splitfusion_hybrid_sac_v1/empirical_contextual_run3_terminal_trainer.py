"""CPU-only terminal Hybrid-SAC update for authenticated Run-3 batches.

Every row is terminal, therefore the critic target is exactly ``y = r``.
Polyak targets are retained for architectural continuity/checkpointing but are
never consulted while constructing the Run-3 fitted target.
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
from .empirical_contextual_run3_reward import Run3TerminalOutcome
from .empirical_contextual_run3_terminal_replay import (
    RUN3_REPLAY_PHASE_LABEL,
    Run3ReplayBindingV1,
    Run3TerminalBatchV1,
)
from .hybrid_sac_models import (
    ConditionalHybridActor,
    HybridSacModelConfig,
    NORMALIZED_Z_DENSITY,
    TwinHybridCritics,
    actor_objective,
    mode_one_hot,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .state_reward_transition_contract import POLICY_FEATURE_COUNT
from .transaction_identity import canonical_sha256

__all__ = [
    "Run3TerminalHybridSacTrainerV1",
    "Run3TerminalTrainerConfigV1",
    "Run3TerminalUpdateMetricsV1",
    "Run3TrainerError",
]


RUN3_TRAINER_PHASE_LABEL = "RUN3_REALIZED_TERMINAL_HYBRID_SAC_TRAINER_V1"


class Run3TrainerError(RuntimeError):
    """Run-3 trainer preflight or numerical update failed closed."""


@dataclass(frozen=True, slots=True)
class Run3TerminalTrainerConfigV1:
    alpha_d: float = 0.10
    alpha_c: float = 0.05
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    float_dtype: torch.dtype = torch.float32
    status: str = "RUN3_FIXED_PRIMARY_TRAINING_HYPOTHESIS"

    def __post_init__(self) -> None:
        for name in ("alpha_d", "alpha_c", "tau", "actor_lr", "critic_lr"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise Run3TrainerError(f"{name} must be positive finite float")
        if self.tau > 1.0:
            raise Run3TrainerError("tau must lie in (0,1]")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise Run3TrainerError("batch_size must be positive int")
        if self.float_dtype is not torch.float32:
            raise Run3TrainerError("Run-3 is fixed to CPU float32")
        if not self.status:
            raise Run3TrainerError("trainer status must be non-empty")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "actor_lr": self.actor_lr,
            "alpha_c": self.alpha_c,
            "alpha_d": self.alpha_d,
            "batch_size": self.batch_size,
            "critic_lr": self.critic_lr,
            "float_dtype": str(self.float_dtype),
            "record": "run3_terminal_trainer_config_v1",
            "status": self.status,
            "tau": self.tau,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class Run3TerminalUpdateMetricsV1:
    update: int
    batch_size: int
    reward_mean: float
    reward_min: float
    reward_max: float
    target_reward_bit_mismatch_count: int
    q_loc_mean: float
    q_seg_mean: float
    q_perc_mean: float
    success_latency_count: int
    success_latency_mean_ms: float
    success_latency_max_ms: float
    terminal_outcome_counts: Tuple[int, ...]
    executed_mode_counts: Tuple[int, ...]
    executed_q_e4_mean: float
    executed_q_e4_min: int
    executed_q_e4_max: int
    critic_1_loss: float
    critic_2_loss: float
    critic_loss_total: float
    actor_loss: float
    q1_mean: float
    q2_mean: float
    twin_gap_mean: float
    discrete_entropy: float
    conditional_entropy_estimate: float
    critic_grad_norm: float
    actor_grad_norm: float
    actor_param_delta_norm: float
    online_critic_param_delta_norm: float
    target_param_delta_norm: float
    replay_binding_sha256: str
    trainer_config_sha256: str
    continuous_log_prob_coordinate: str = NORMALIZED_Z_DENSITY
    modeled_smoke_support_sha256: str = MODELED_SMOKE_SUPPORT_SHA256
    replay_phase_label: str = RUN3_REPLAY_PHASE_LABEL
    phase_label: str = RUN3_TRAINER_PHASE_LABEL

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def assert_valid(self) -> None:
        if len(self.terminal_outcome_counts) != len(Run3TerminalOutcome):
            raise Run3TrainerError("terminal count inventory drift")
        if sum(self.terminal_outcome_counts) != self.batch_size:
            raise Run3TrainerError("terminal counts do not sum to batch")
        if len(self.executed_mode_counts) != EXPECTED_MODE_COUNT:
            raise Run3TrainerError("mode count inventory drift")
        if sum(self.executed_mode_counts) != self.batch_size:
            raise Run3TrainerError("mode counts do not sum to batch")
        if self.target_reward_bit_mismatch_count != 0:
            raise Run3TrainerError("terminal target is not reward")
        if self.success_latency_count < 0 or self.success_latency_count > self.batch_size:
            raise Run3TrainerError("success latency count drift")
        for name, value in self.as_dict().items():
            if isinstance(value, float) and not math.isfinite(value):
                raise Run3TrainerError(f"metric {name} is non-finite")


def _snapshot(parameters: Iterable[nn.Parameter]) -> List[Tensor]:
    return [item.detach().clone() for item in parameters]


def _delta_norm(before: List[Tensor], after: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for old, new in zip(before, after):
        delta = new.detach().to(torch.float64) - old.to(torch.float64)
        total += float(torch.sum(delta * delta))
    return math.sqrt(total)


def _grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().to(torch.float64)
            total += float(torch.sum(gradient * gradient))
    return math.sqrt(total)


def _finite_tensor(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise Run3TrainerError(f"{name} contains non-finite values")


def _bit_mismatches(left: Tensor, right: Tensor) -> int:
    if left.dtype is not torch.float32 or right.dtype is not torch.float32:
        raise Run3TrainerError("bit comparison requires float32")
    return int(torch.count_nonzero(left.contiguous().view(torch.int32) != right.contiguous().view(torch.int32)))


class Run3TerminalHybridSacTrainerV1:
    """Atomic terminal critic/actor update with isolated critic gradients."""

    def __init__(
        self,
        actor: ConditionalHybridActor,
        critics: TwinHybridCritics,
        config: Run3TerminalTrainerConfigV1,
        *,
        expected_binding: Run3ReplayBindingV1,
        expected_batch_issuer_capability: object,
        actor_generator: torch.Generator,
    ) -> None:
        if type(actor) is not ConditionalHybridActor or type(critics) is not TwinHybridCritics:
            raise Run3TrainerError("trainer requires exact Hybrid-SAC models")
        if type(config) is not Run3TerminalTrainerConfigV1:
            raise Run3TrainerError("trainer config has foreign type")
        config.__post_init__()
        if type(expected_binding) is not Run3ReplayBindingV1:
            raise Run3TrainerError("replay binding has foreign type")
        expected_binding.require_valid()
        if (
            not isinstance(actor_generator, torch.Generator)
            or actor_generator is torch.default_generator
            or actor_generator.device.type != "cpu"
        ):
            raise Run3TrainerError("actor generator must be private CPU state")
        self._validate_models(actor, critics)
        self.actor = actor
        self.critics = critics
        self.config = config
        self.expected_binding = expected_binding
        if expected_batch_issuer_capability is None:
            raise Run3TrainerError("trainer requires an opaque replay issuer capability")
        self._expected_batch_issuer_capability = expected_batch_issuer_capability
        self._binding_sha256 = expected_binding.canonical_sha256()
        self._config_sha256 = config.canonical_sha256()
        self._model_config_snapshot = copy.deepcopy(actor.config)
        self._actor_generator = actor_generator
        self._online_critics = list(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._online_critics, lr=config.critic_lr)
        self._assert_optimizer_wiring()
        self.update_count = 0

    @staticmethod
    def _validate_models(actor: ConditionalHybridActor, critics: TwinHybridCritics) -> None:
        config = actor.config
        if type(config) is not HybridSacModelConfig or config.dtype is not torch.float32:
            raise Run3TrainerError("actor model config must be float32")
        if config.state_dim != POLICY_FEATURE_COUNT or config.mode_count != EXPECTED_MODE_COUNT:
            raise Run3TrainerError("model dimensions drift")
        if not actor.uses_modeled_smoke_support:
            raise Run3TrainerError("actor is not bound to modeled support")
        if actor.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise Run3TrainerError("actor support hash drift")
        for label, module in (
            ("actor", actor),
            ("critic_1", critics.critic_1),
            ("critic_2", critics.critic_2),
            ("target_1", critics.target_1),
            ("target_2", critics.target_2),
        ):
            if module.config != config:
                raise Run3TrainerError(f"{label} config drift")
            for value in chain(module.parameters(), module.buffers()):
                if value.device.type != "cpu":
                    raise Run3TrainerError(f"{label} escaped CPU")
                if value.is_floating_point() and (
                    value.dtype is not torch.float32 or not bool(torch.isfinite(value).all())
                ):
                    raise Run3TrainerError(f"{label} dtype/finiteness drift")
            expected_trainable = label in ("actor", "critic_1", "critic_2")
            if any(parameter.requires_grad is not expected_trainable for parameter in module.parameters()):
                raise Run3TrainerError(f"{label} trainability drift")
        inventories = {
            label: {id(parameter) for parameter in module.parameters()}
            for label, module in (
                ("actor", actor), ("critic_1", critics.critic_1),
                ("critic_2", critics.critic_2), ("target_1", critics.target_1),
                ("target_2", critics.target_2),
            )
        }
        labels = tuple(inventories)
        for index, left in enumerate(labels):
            for right in labels[index + 1:]:
                if inventories[left].intersection(inventories[right]):
                    raise Run3TrainerError(f"{left}/{right} share parameters")

    def _assert_optimizer_wiring(self) -> None:
        live_critics = list(chain(self.critics.critic_1.parameters(), self.critics.critic_2.parameters()))
        if [id(item) for item in live_critics] != [id(item) for item in self._online_critics]:
            raise Run3TrainerError("captured online critic parameter sequence drift")
        target_ids = {
            id(item)
            for module in (self.critics.target_1, self.critics.target_2)
            for item in module.parameters()
        }
        for label, optimizer, expected, learning_rate in (
            ("actor", self.actor_optimizer, list(self.actor.parameters()), self.config.actor_lr),
            ("critic", self.critic_optimizer, live_critics, self.config.critic_lr),
        ):
            if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
                raise Run3TrainerError(f"{label} optimizer structure drift")
            observed = [item for group in optimizer.param_groups for item in group["params"]]
            if len(observed) != len({id(item) for item in observed}):
                raise Run3TrainerError(f"{label} optimizer duplicates parameters")
            if {id(item) for item in observed} != {id(item) for item in expected}:
                raise Run3TrainerError(f"{label} optimizer parameter ownership drift")
            if target_ids.intersection(id(item) for item in observed):
                raise Run3TrainerError(f"{label} optimizer contains target parameter")
            if optimizer.param_groups[0].get("lr") != learning_rate:
                raise Run3TrainerError(f"{label} optimizer learning rate drift")
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, Tensor) and (
                        value.device.type != "cpu"
                        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
                    ):
                        raise Run3TrainerError(f"{label} optimizer state drift")

    def _preflight(self, batch: Run3TerminalBatchV1) -> None:
        if type(batch) is not Run3TerminalBatchV1:
            raise Run3TrainerError("trainer accepts only exact Run3 batches")
        batch.revalidate(
            expected_issuer_capability=self._expected_batch_issuer_capability
        )
        self.config.__post_init__()
        if self.config.canonical_sha256() != self._config_sha256:
            raise Run3TrainerError("trainer config digest drift")
        self._validate_models(self.actor, self.critics)
        if self.actor.config != self._model_config_snapshot:
            raise Run3TrainerError("model config drift")
        self._assert_optimizer_wiring()
        self.expected_binding.require_valid()
        if self.expected_binding.canonical_sha256() != self._binding_sha256:
            raise Run3TrainerError("expected binding digest drift")
        if batch.binding != self.expected_binding:
            raise Run3TrainerError("batch binding differs from trainer binding")
        if batch.batch_size != self.config.batch_size:
            raise Run3TrainerError("batch size differs from trainer config")
        if batch.next_state is not None or bool(batch.bootstrap.any()):
            raise Run3TrainerError("Run-3 batch must never bootstrap")
        if not bool(batch.terminated.all()) or not bool((batch.duration == 1).all()):
            raise Run3TrainerError("Run-3 terminal/duration semantics drift")
        if not bool((batch.discount == 0.0).all()):
            raise Run3TrainerError("Run-3 discount must be zero")
        tensors = batch.learner_tensors()
        if set(tensors) != {"state", "mode_id", "q_e4", "q_normalized", "reward"}:
            raise Run3TrainerError("learner tensor inventory drift")
        _finite_tensor(tensors["state"], "state")
        _finite_tensor(tensors["reward"], "reward")
        _finite_tensor(tensors["q_normalized"], "q_normalized")

    def _snapshot_transaction(self) -> Dict[str, Any]:
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critics": copy.deepcopy(self.critics.state_dict()),
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic_optimizer": copy.deepcopy(self.critic_optimizer.state_dict()),
            "actor_rng": self._actor_generator.get_state().clone(),
            "update_count": self.update_count,
        }

    def _restore_transaction(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"], strict=True)
        self.critics.load_state_dict(state["critics"], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._actor_generator.set_state(state["actor_rng"])
        self.update_count = state["update_count"]

    def update_once(self, batch: Run3TerminalBatchV1) -> Run3TerminalUpdateMetricsV1:
        self._preflight(batch)
        transaction = self._snapshot_transaction()
        try:
            return self._update(batch)
        except BaseException:
            self._restore_transaction(transaction)
            raise

    def _update(self, batch: Run3TerminalBatchV1) -> Run3TerminalUpdateMetricsV1:
        state = batch.state
        modes = batch.mode_id
        q_e4 = batch.q_e4
        reward = batch.reward
        target = batch.terminal_target()
        mismatch = _bit_mismatches(target, reward)
        if mismatch:
            raise Run3TrainerError("terminal target/reward bit mismatch")
        actor_before = _snapshot(self.actor.parameters())
        critic_before = _snapshot(self._online_critics)
        target_parameters = list(chain(self.critics.target_1.parameters(), self.critics.target_2.parameters()))
        target_before = _snapshot(target_parameters)

        one_hot = mode_one_hot(modes, EXPECTED_MODE_COUNT, torch.float32)
        q_normalized = q_e4.to(torch.float32) / float(Q_E4_MAX)
        q1, q2 = self.critics.q_values(state, one_hot, q_normalized)
        critic_1_loss = torch.mean((q1 - target) ** 2)
        critic_2_loss = torch.mean((q2 - target) ** 2)
        critic_loss = critic_1_loss + critic_2_loss
        _finite_tensor(critic_loss, "critic loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        for gradient_index, parameter in enumerate(self._online_critics):
            if parameter.grad is not None:
                _finite_tensor(parameter.grad, f"critic gradient {gradient_index}")
        critic_grad = _grad_norm(self._online_critics)
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
        _finite_tensor(objective.objective, "actor loss")
        objective.objective.backward()
        if any(parameter.grad is not None for parameter in self._online_critics):
            raise Run3TrainerError("actor backward contaminated critic gradients")
        if any(parameter.grad is None for parameter in self.actor.parameters()):
            raise Run3TrainerError("actor gradient missing")
        for index, parameter in enumerate(self.actor.parameters()):
            _finite_tensor(parameter.grad, f"actor gradient {index}")
        actor_grad = _grad_norm(self.actor.parameters())
        self.actor_optimizer.step()
        self.critics.polyak_update(self.config.tau)

        # Diagnostics consume the sealed compact admission records.  The rich
        # provenance rows remain available for checkpoints/final deep audit,
        # but are not traversed in the trainer hot path.
        rows = batch.compact_records
        outcome_order = tuple(Run3TerminalOutcome)
        outcome_counts = tuple(
            sum(row.terminal_outcome is outcome for row in rows)
            for outcome in outcome_order
        )
        mode_counts = tuple(int((modes == mode).sum()) for mode in range(EXPECTED_MODE_COUNT))
        latencies = tuple(row.latency_ms for row in rows if row.latency_ms is not None)
        probs = objective.probs.detach()
        log_d = objective.sample.log_prob_discrete.detach()
        log_c = objective.sample.log_prob_continuous.detach()
        self.update_count += 1
        metrics = Run3TerminalUpdateMetricsV1(
            update=self.update_count,
            batch_size=batch.batch_size,
            reward_mean=float(reward.mean()),
            reward_min=float(reward.min()),
            reward_max=float(reward.max()),
            target_reward_bit_mismatch_count=mismatch,
            q_loc_mean=sum(row.q_loc for row in rows) / len(rows),
            q_seg_mean=sum(row.q_seg for row in rows) / len(rows),
            q_perc_mean=sum(row.q_perc for row in rows) / len(rows),
            success_latency_count=len(latencies),
            success_latency_mean_ms=(0.0 if not latencies else sum(latencies) / len(latencies)),
            success_latency_max_ms=(0.0 if not latencies else max(latencies)),
            terminal_outcome_counts=outcome_counts,
            executed_mode_counts=mode_counts,
            executed_q_e4_mean=float(q_e4.to(torch.float64).mean()),
            executed_q_e4_min=int(q_e4.min()),
            executed_q_e4_max=int(q_e4.max()),
            critic_1_loss=float(critic_1_loss.detach()),
            critic_2_loss=float(critic_2_loss.detach()),
            critic_loss_total=float(critic_loss.detach()),
            actor_loss=float(objective.objective.detach()),
            q1_mean=float(q1.detach().mean()),
            q2_mean=float(q2.detach().mean()),
            twin_gap_mean=float((q1.detach() - q2.detach()).abs().mean()),
            discrete_entropy=float((-(probs * log_d).sum(dim=-1)).mean()),
            conditional_entropy_estimate=float(-(probs * log_c).sum(dim=-1).mean()),
            critic_grad_norm=critic_grad,
            actor_grad_norm=actor_grad,
            actor_param_delta_norm=_delta_norm(actor_before, self.actor.parameters()),
            online_critic_param_delta_norm=_delta_norm(critic_before, self._online_critics),
            target_param_delta_norm=_delta_norm(target_before, target_parameters),
            replay_binding_sha256=self._binding_sha256,
            trainer_config_sha256=self._config_sha256,
        )
        metrics.assert_valid()
        return metrics

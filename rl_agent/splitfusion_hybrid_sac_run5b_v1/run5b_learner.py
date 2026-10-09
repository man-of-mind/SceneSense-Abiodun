"""Run-5B replay buffer and Hybrid-SAC trainer (21-feature state).

DERIVED MECHANICALLY from ``splitfusion_hybrid_sac_run4b_v1/learner.py`` by
``derive_from_run4b.py``; the only differences are the listed substitutions.

The SAC numerical update is not forked: ``Run4BTrainerV1.update_once`` is
inherited unchanged from the tested Run-4 ``_Run4TrainerCore``.  Only
construction and preflight are Run-4B-specific (20-feature state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain

import torch
from torch import Tensor

from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as run4_trainer
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    Q_E4_MAX,
    Q_E4_MIN,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    TwinHybridCritics,
)

from . import run5b_state_contract as C
from .run5b_models import validate_models

TrainerConfigV1 = run4_trainer.TrainerConfigV1
UpdateMetricsV1 = run4_trainer.UpdateMetricsV1
REPLAY_CAPACITY = 65536


class LearnerError(RuntimeError):
    """Replay or trainer invariant failure."""


@dataclass(frozen=True, slots=True)
class Run4BBatchV1:
    """Minibatch exposing exactly what the inherited update consumes."""

    state: Tensor
    next_state: Tensor
    mode_id: Tensor
    q_e4: Tensor
    reward: Tensor
    _discount: Tensor
    bootstrap: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.state.shape[0])

    def discount(self) -> Tensor:
        return self._discount

    @property
    def q_normalized_executed(self) -> Tensor:
        return self.q_e4.to(torch.float32) / float(Q_E4_MAX)


class ReplayBufferV1:
    """FIFO replay with explicit, complete, checkpointable contents."""

    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        if type(capacity) is not int or capacity < 1:
            raise LearnerError("capacity must be a positive int")
        self.capacity = capacity
        self._state = torch.zeros((capacity, C.FEATURE_COUNT), dtype=torch.float32)
        self._next = torch.zeros((capacity, C.FEATURE_COUNT), dtype=torch.float32)
        self._mode = torch.zeros(capacity, dtype=torch.int64)
        self._q = torch.zeros(capacity, dtype=torch.int64)
        self._reward = torch.zeros(capacity, dtype=torch.float32)
        self._discount = torch.zeros(capacity, dtype=torch.float32)
        self._start = 0
        self._size = 0
        self.accepted = 0

    def __len__(self) -> int:
        return self._size

    def add(self, *, state, next_state, mode_id: int, q_e4: int,
            reward: float, discount: float) -> None:
        if len(state) != C.FEATURE_COUNT or len(next_state) != C.FEATURE_COUNT:
            raise LearnerError("replay rows must carry 21 features")
        values = (*state, *next_state, reward, discount)
        if not all(math.isfinite(float(v)) for v in values):
            raise LearnerError("non-finite replay row")
        if not (0 <= mode_id < C.MODE_COUNT and Q_E4_MIN <= q_e4 <= Q_E4_MAX):
            raise LearnerError("replay action out of range")
        if not 0.0 < discount <= 1.0:
            raise LearnerError("discount must lie in (0, 1]")
        if self._size < self.capacity:
            slot = (self._start + self._size) % self.capacity
            self._size += 1
        else:
            slot = self._start
            self._start = (self._start + 1) % self.capacity
        self._state[slot] = torch.tensor(state, dtype=torch.float32)
        self._next[slot] = torch.tensor(next_state, dtype=torch.float32)
        self._mode[slot] = mode_id
        self._q[slot] = q_e4
        self._reward[slot] = reward
        self._discount[slot] = discount
        self.accepted += 1

    def _physical(self, logical: Tensor) -> Tensor:
        return (logical + self._start) % self.capacity

    def sample(self, batch_size: int, generator: torch.Generator) -> Run4BBatchV1:
        if not isinstance(generator, torch.Generator) or (
                generator is torch.default_generator):
            raise LearnerError("sampling requires a private generator")
        if batch_size > self._size:
            raise LearnerError(f"cannot sample {batch_size} of {self._size}")
        permutation = torch.randperm(self._size, generator=generator,
                                     device="cpu")
        index = self._physical(permutation[:batch_size])
        return Run4BBatchV1(
            state=self._state.index_select(0, index),
            next_state=self._next.index_select(0, index),
            mode_id=self._mode.index_select(0, index),
            q_e4=self._q.index_select(0, index),
            reward=self._reward.index_select(0, index),
            _discount=self._discount.index_select(0, index),
            bootstrap=torch.ones(batch_size, dtype=torch.bool))

    def state_dict(self) -> dict[str, Tensor | int]:
        """Logical-order contents; only resident rows are stored."""
        index = self._physical(torch.arange(self._size, dtype=torch.int64))
        return {"state": self._state.index_select(0, index).clone(),
                "next_state": self._next.index_select(0, index).clone(),
                "mode_id": self._mode.index_select(0, index).clone(),
                "q_e4": self._q.index_select(0, index).clone(),
                "reward": self._reward.index_select(0, index).clone(),
                "discount": self._discount.index_select(0, index).clone(),
                "size": self._size, "accepted": self.accepted,
                "capacity": self.capacity}

    def load_state_dict(self, value: dict) -> None:
        if int(value["capacity"]) != self.capacity:
            raise LearnerError("replay capacity differs")
        size = int(value["size"])
        if size > self.capacity:
            raise LearnerError("replay size exceeds capacity")
        for name, width in (("state", C.FEATURE_COUNT),
                            ("next_state", C.FEATURE_COUNT)):
            if tuple(value[name].shape) != (size, width):
                raise LearnerError(f"replay {name} shape differs")
        self._state.zero_(); self._next.zero_(); self._mode.zero_()
        self._q.zero_(); self._reward.zero_(); self._discount.zero_()
        self._state[:size] = value["state"]
        self._next[:size] = value["next_state"]
        self._mode[:size] = value["mode_id"]
        self._q[:size] = value["q_e4"]
        self._reward[:size] = value["reward"]
        self._discount[:size] = value["discount"]
        self._start = 0
        self._size = size
        self.accepted = int(value["accepted"])


class Run4BTrainerV1(run4_trainer._Run4TrainerCore):
    """Run-4B construction/preflight around the inherited SAC update."""

    def __init__(self, *, actor: ConditionalHybridActor,
                 critics: TwinHybridCritics, config: TrainerConfigV1,
                 target_generator: torch.Generator,
                 actor_generator: torch.Generator) -> None:
        if type(config) is not TrainerConfigV1:
            raise LearnerError("config must be an exact TrainerConfigV1")
        validate_models(actor, critics)
        for generator in (target_generator, actor_generator):
            if not isinstance(generator, torch.Generator) or (
                    generator is torch.default_generator):
                raise LearnerError("trainer generators must be private")
        if target_generator is actor_generator:
            raise LearnerError("trainer generators must be distinct")
        self.actor = actor
        self.critics = critics
        self.config = config
        self.expected_binding = None
        self._target_generator = target_generator
        self._actor_generator = actor_generator
        self._online_critic_parameters = list(chain(
            critics.critic_1.parameters(), critics.critic_2.parameters()))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self._online_critic_parameters, lr=config.critic_lr)
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _require_evidence_class(self, binding) -> None:  # pragma: no cover
        raise LearnerError("Run-4B does not use the Run-4 replay binding")

    def _preflight(self, batch: Run4BBatchV1) -> None:
        error = run4_trainer.TrainerPreflightError
        self._assert_optimizer_wiring()
        validate_models(self.actor, self.critics)
        if type(batch) is not Run4BBatchV1:
            raise error("update_once requires an exact Run4BBatchV1")
        size = batch.batch_size
        if size < 1:
            raise error("empty batch")
        for name, tensor, shape in (
                ("state", batch.state, (size, C.FEATURE_COUNT)),
                ("next_state", batch.next_state, (size, C.FEATURE_COUNT)),
                ("reward", batch.reward, (size,)),
                ("discount", batch.discount(), (size,))):
            if tuple(tensor.shape) != shape or tensor.dtype is not torch.float32:
                raise error(f"{name} must be float32 {shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise error(f"{name} contains a non-finite value")
        for name, tensor in (("mode_id", batch.mode_id), ("q_e4", batch.q_e4)):
            if tuple(tensor.shape) != (size,) or tensor.dtype is not torch.int64:
                raise error(f"{name} must be int64 ({size},)")
        if bool((batch.mode_id < 0).any()) or bool(
                (batch.mode_id >= C.MODE_COUNT).any()):
            raise error("mode_id outside the 12-mode catalog")
        if bool((batch.q_e4 < Q_E4_MIN).any()) or bool(
                (batch.q_e4 > Q_E4_MAX).any()):
            raise error("q_e4 outside the registered range")
        if batch.bootstrap.dtype is not torch.bool or not bool(
                batch.bootstrap.all()):
            raise error("every Run-4B transition continues the session")
        discount = batch.discount()
        if bool((discount <= 0.0).any()) or bool((discount > 1.0).any()):
            raise error("discount must lie in (0, 1]")

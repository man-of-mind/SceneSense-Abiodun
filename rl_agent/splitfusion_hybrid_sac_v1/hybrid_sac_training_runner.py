"""Bounded Hybrid-SAC algorithm qualification runner.

``HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY``

This module answers one narrow question: can the Phase-C actor, twin critics,
replay tensor schema and one-update trainer execute a deterministic multi-step
learning experiment, including an exact stop/resume boundary?  It does *not*
train on SplitFusion evidence and it makes no CARLA, OAI, model-quality or
deployment claim.

The qualification environment is deliberately analytic.  It exposes all 12
discrete modes, assigns every mode a distinct continuous optimum strictly
inside ``q in (0, 0.98)``, and directly encodes both the target mode and target
continuous optimum in its 31-value observation.  The fixed evaluation stream
therefore tests training mechanics on the same finite analytic state family;
it is not an independent data split and makes no generalization claim.  Its
replay is a separate in-memory store which creates
``ReplayTensorBatchV1`` directly; it never inserts a positive synthetic reward
into ``ReplayBufferV1``.  The production fail-closed gate therefore remains
untouched.

Five independent local CPU RNG streams are mandatory:

* collection (environment transitions and exploratory actions),
* replay sampling,
* target-policy continuous samples,
* actor-update continuous samples, and
* fixed analytic evaluation.

No operation in this module intentionally consumes ``torch.default_generator``.
Checkpoints include models, Polyak targets, both optimizers, all five active
RNGs, the initial fixed-evaluation RNG state, qualification replay, environment
state, counters and audit history.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .action_contract import CATALOG_SHA256, EXPECTED_MODE_COUNT, Q_E4_MAX
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)
from .hybrid_sac_trainer import HybridSacTrainerV1, TrainerConfigV1, UpdateMetricsV1
from .replay_buffer import ReplayBindingV1, ReplayTensorBatchV1
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
)
from .transaction_identity import MINIMUM_HOLD_TENSORS

__all__ = [
    "AcceptanceThresholdsV1",
    "EvaluationMetricsV1",
    "HybridSacAlgorithmQualificationRunnerV1",
    "QualificationEnvironmentV1",
    "QualificationError",
    "QualificationReplayV1",
    "QualificationRunnerConfigV1",
    "SeedQualificationResultV1",
    "build_qualification_binding",
    "run_seed_suite",
]


PHASE_LABEL = "HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY"
CHECKPOINT_SCHEMA = "splitfusion.hybrid_sac.algorithm_qualification_checkpoint.v1"
CHECKPOINT_FILE_SCHEMA = (
    "splitfusion.hybrid_sac.algorithm_qualification_checkpoint_file.v1"
)
QUALIFICATION_REPLAY_SCHEMA = (
    "splitfusion.hybrid_sac.algorithm_qualification_replay.v1"
)


class QualificationError(Exception):
    """Any invalid qualification configuration, state or checkpoint."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise QualificationError(f"{name} must be finite, got {value!r}")
    return numeric


def _require_exact_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _require_canonical_binding(value: Any, binding: ReplayBindingV1) -> None:
    expected = binding.to_canonical_dict()
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise QualificationError("checkpoint replay binding fields differ")
    string_fields = {
        "catalog_sha256",
        "freshness_policy_sha256",
        "reward_spec_sha256",
        "schema_id",
        "schema_sha256",
        "state_normalization_spec_sha256",
    }
    if any(type(value[name]) is not str for name in string_fields):
        raise QualificationError("checkpoint replay binding string type differs")
    if type(value["gamma_per_tensor"]) is not float:
        raise QualificationError("checkpoint replay gamma type differs")
    for name in ("schema_version", "policy_feature_count"):
        if type(value[name]) is not int:
            raise QualificationError(f"checkpoint replay {name} type differs")
    order = value["policy_feature_order"]
    if not isinstance(order, list) or any(type(item) is not str for item in order):
        raise QualificationError("checkpoint replay feature order type differs")
    if dict(value) != expected:
        raise QualificationError("checkpoint replay binding differs")


def _new_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if generator is torch.default_generator:  # pragma: no cover - defensive
        raise QualificationError("a qualification RNG aliased the global RNG")
    return generator


def build_qualification_binding(gamma_per_tensor: float) -> ReplayBindingV1:
    """Build a binding explicitly scoped to analytic qualification data."""
    return ReplayBindingV1(
        reward_spec_sha256=_sha256_text(
            "HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY/reward/v1"
        ),
        state_normalization_spec_sha256=_sha256_text(
            "HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY/state/v1"
        ),
        freshness_policy_sha256=_sha256_text(
            "HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY/no-freshness-claim/v1"
        ),
        gamma_per_tensor=float(gamma_per_tensor),
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        schema_sha256=SCHEMA_SHA256,
        catalog_sha256=CATALOG_SHA256,
        policy_feature_order=tuple(POLICY_FEATURE_ORDER),
        policy_feature_count=POLICY_FEATURE_COUNT,
    )


@dataclass(frozen=True, slots=True)
class QualificationRunnerConfigV1:
    """Small bounded-run configuration; none of these are scientific defaults."""

    seed: int
    max_updates: int = 2_000
    replay_capacity: int = 8_192
    batch_size: int = 64
    warmup_transitions: int = 256
    collect_per_update: int = 2
    episode_horizon: int = 24
    gamma_per_tensor: float = 0.99
    alpha_d: float = 0.10
    alpha_c: float = 0.05
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    def __post_init__(self) -> None:
        integer_fields = (
            "seed",
            "max_updates",
            "replay_capacity",
            "batch_size",
            "warmup_transitions",
            "collect_per_update",
            "episode_horizon",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise QualificationError(f"{name} must be an integer")
        if self.seed < 0:
            raise QualificationError("seed must be non-negative")
        if not 1 <= self.max_updates <= 1_000_000:
            raise QualificationError("max_updates must lie in [1, 1000000]")
        if self.batch_size < 1:
            raise QualificationError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise QualificationError("replay_capacity must cover one batch")
        if self.warmup_transitions < self.batch_size:
            raise QualificationError("warmup_transitions must cover one batch")
        if self.warmup_transitions > self.replay_capacity:
            raise QualificationError("warmup_transitions exceeds replay capacity")
        if self.collect_per_update < 1:
            raise QualificationError("collect_per_update must be positive")
        if self.episode_horizon < 2:
            raise QualificationError("episode_horizon must be at least two")
        for name in (
            "gamma_per_tensor",
            "alpha_d",
            "alpha_c",
            "tau",
            "actor_lr",
            "critic_lr",
        ):
            _require_finite(getattr(self, name), name)
        if not 0.0 < self.gamma_per_tensor <= 1.0:
            raise QualificationError("gamma_per_tensor must lie in (0, 1]")
        if self.alpha_d <= 0.0 or self.alpha_c <= 0.0:
            raise QualificationError("entropy temperatures must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise QualificationError("tau must lie in (0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise QualificationError("learning rates must be positive")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class AcceptanceThresholdsV1:
    """Optional algorithm-qualification checks, not paper acceptance gates."""

    minimum_improvement_over_random: float = 0.05
    maximum_oracle_regret: float = 0.35

    def __post_init__(self) -> None:
        _require_finite(
            self.minimum_improvement_over_random,
            "minimum_improvement_over_random",
        )
        _require_finite(self.maximum_oracle_regret, "maximum_oracle_regret")
        if self.maximum_oracle_regret < 0.0:
            raise QualificationError("maximum_oracle_regret must be non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationMetricsV1:
    """Fixed analytic policy, random, fixed-action and oracle comparison."""

    steps: int
    policy_mean_reward: float
    random_mean_reward: float
    fixed_mean_reward: float
    oracle_mean_reward: float
    improvement_over_random: float
    oracle_regret: float
    policy_mode_accuracy: float
    policy_mean_absolute_q_error: float
    policy_selected_mode_count: int
    policy_interior_q_fraction: float
    fixed_mode: int
    fixed_q_e4: int
    minimum_improvement_over_random: float
    maximum_oracle_regret: float
    accepted: bool
    phase_label: str = PHASE_LABEL

    def assert_finite(self) -> None:
        if self.steps < 1:
            raise QualificationError("evaluation steps must be positive")
        for name in (
            "policy_mean_reward",
            "random_mean_reward",
            "fixed_mean_reward",
            "oracle_mean_reward",
            "improvement_over_random",
            "oracle_regret",
            "policy_mode_accuracy",
            "policy_mean_absolute_q_error",
            "policy_interior_q_fraction",
            "minimum_improvement_over_random",
            "maximum_oracle_regret",
        ):
            _require_finite(getattr(self, name), name)
        if not 0 <= self.policy_selected_mode_count <= EXPECTED_MODE_COUNT:
            raise QualificationError("selected mode count is outside [0, 12]")
        if not 0.0 <= self.policy_interior_q_fraction <= 1.0:
            raise QualificationError("interior-q fraction is outside [0, 1]")
        if not 0 <= self.fixed_mode < EXPECTED_MODE_COUNT:
            raise QualificationError("fixed baseline mode is outside [0, 11]")
        if not 0 <= self.fixed_q_e4 <= Q_E4_MAX:
            raise QualificationError("fixed baseline q_e4 is outside [0, 9800]")

    def as_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class SeedQualificationResultV1:
    seed: int
    update_count: int
    collected_transitions: int
    evaluation: EvaluationMetricsV1


class QualificationEnvironmentV1:
    """Analytic 31-D contextual task with 12 interior hybrid optima.

    Context ``m`` has optimal discrete mode ``m`` and optimal wire quality
    ``800 + 750*m``.  Therefore all 12 modes are exercised and every optimum
    is strictly inside ``[0, 9800]``.  The observation deliberately exposes
    the target mode as a 12-way one-hot vector and the normalized target
    continuous optimum in slot 12; the remaining values are bounded
    deterministic context descriptors.  These target-bearing qualification
    features test mechanics only: they are neither an independent data split
    nor evidence of generalization or production-state semantics.
    """

    Q_OPTIMA_E4: Tuple[int, ...] = tuple(800 + 750 * mode for mode in range(12))

    def __init__(self, horizon: int) -> None:
        if horizon < 2:
            raise QualificationError("environment horizon must be at least two")
        if len(self.Q_OPTIMA_E4) != EXPECTED_MODE_COUNT:
            raise QualificationError("qualification optima do not cover all modes")
        if not all(0 < q < Q_E4_MAX for q in self.Q_OPTIMA_E4):
            raise QualificationError("every qualification q optimum must be interior")
        self.horizon = int(horizon)
        self.episode_index = 0
        self.step_in_episode = 0
        self.context_id: Optional[int] = None
        self.needs_reset = True

    @staticmethod
    def observation(context_id: int, progress: float) -> Tensor:
        if not 0 <= context_id < EXPECTED_MODE_COUNT:
            raise QualificationError("context_id is outside the 12-mode task")
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise QualificationError("progress must lie in [0, 1]")
        values = torch.zeros(POLICY_FEATURE_COUNT, dtype=torch.float32)
        values[context_id] = 1.0
        q_opt = QualificationEnvironmentV1.Q_OPTIMA_E4[context_id]
        values[12] = float(q_opt) / float(Q_E4_MAX)
        values[13] = float(context_id) / float(EXPECTED_MODE_COUNT - 1)
        values[14] = float(progress)
        # Bounded deterministic distractors prevent the task from being a
        # literal 13-column lookup while keeping replay exactly reproducible.
        for index in range(15, POLICY_FEATURE_COUNT):
            values[index] = (
                ((context_id + 1) * (index - 11)) % 17
            ) / 16.0
        return values

    @classmethod
    def reward_for(cls, context_id: int, mode_id: int, q_e4: int) -> float:
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise QualificationError("mode_id is outside [0, 11]")
        if not 0 <= q_e4 <= Q_E4_MAX:
            raise QualificationError("q_e4 is outside [0, 9800]")
        q_error = (float(q_e4) - cls.Q_OPTIMA_E4[context_id]) / Q_E4_MAX
        mode_penalty = 1.0 if mode_id != context_id else 0.0
        reward = 1.0 - mode_penalty - 2.0 * q_error * q_error
        return _require_finite(reward, "qualification reward")

    @classmethod
    def oracle_action(cls, context_id: int) -> Tuple[int, int]:
        return context_id, cls.Q_OPTIMA_E4[context_id]

    def reset(self, generator: torch.Generator) -> Tensor:
        self.context_id = int(
            torch.randint(
                EXPECTED_MODE_COUNT, (1,), generator=generator, dtype=torch.int64
            ).item()
        )
        self.step_in_episode = 0
        self.needs_reset = False
        return self.observation(self.context_id, 0.0)

    def step(
        self, mode_id: int, q_e4: int, generator: torch.Generator
    ) -> Tuple[Tensor, float, bool]:
        if self.needs_reset or self.context_id is None:
            raise QualificationError("reset must be called before step")
        reward = self.reward_for(self.context_id, int(mode_id), int(q_e4))
        self.step_in_episode += 1
        terminated = self.step_in_episode >= self.horizon
        next_context = int(
            torch.randint(
                EXPECTED_MODE_COUNT, (1,), generator=generator, dtype=torch.int64
            ).item()
        )
        progress = min(1.0, self.step_in_episode / self.horizon)
        next_state = self.observation(next_context, progress)
        self.context_id = next_context
        if terminated:
            self.episode_index += 1
            self.needs_reset = True
        return next_state, reward, terminated

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": "qualification_environment.v1",
            "horizon": self.horizon,
            "episode_index": self.episode_index,
            "step_in_episode": self.step_in_episode,
            "context_id": self.context_id,
            "needs_reset": self.needs_reset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "schema",
            "horizon",
            "episode_index",
            "step_in_episode",
            "context_id",
            "needs_reset",
        }:
            raise QualificationError("qualification environment fields differ")
        if state.get("schema") != "qualification_environment.v1":
            raise QualificationError("unknown qualification environment schema")
        if type(state.get("horizon")) is not int or state.get("horizon") != self.horizon:
            raise QualificationError("checkpoint environment horizon differs")
        episode_index = _require_exact_int(
            state.get("episode_index"), "checkpoint episode_index"
        )
        step = _require_exact_int(
            state.get("step_in_episode"), "checkpoint step_in_episode"
        )
        context = state.get("context_id")
        needs_reset = state.get("needs_reset")
        if step > self.horizon:
            raise QualificationError("invalid checkpoint step_in_episode")
        if context is not None and (
            type(context) is not int
            or not 0 <= context < EXPECTED_MODE_COUNT
        ):
            raise QualificationError("invalid checkpoint context_id")
        if not isinstance(needs_reset, bool):
            raise QualificationError("invalid checkpoint needs_reset")
        if not needs_reset and context is None:
            raise QualificationError("active environment has no context")
        if needs_reset and not (
            (episode_index == 0 and step == 0 and context is None)
            or (episode_index > 0 and step == self.horizon and context is not None)
        ):
            raise QualificationError("reset-required environment state is inconsistent")
        self.episode_index = episode_index
        self.step_in_episode = step
        self.context_id = context
        self.needs_reset = needs_reset


@dataclass(frozen=True, slots=True)
class _QualificationTransition:
    transition_id: int
    state: Tensor
    next_state: Tensor
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    discount: float
    terminated: bool


class QualificationReplayV1:
    """Private analytic replay; intentionally not ``ReplayBufferV1``."""

    def __init__(
        self, capacity: int, binding: ReplayBindingV1, dtype: torch.dtype
    ) -> None:
        if capacity < 1:
            raise QualificationError("qualification replay capacity is invalid")
        if dtype is not torch.float32:
            raise QualificationError("qualification replay is CPU float32 only")
        self.capacity = int(capacity)
        self.binding = binding
        self.dtype = dtype
        self._rows: List[_QualificationTransition] = []
        self.total_inserted = 0

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        state: Tensor,
        next_state: Tensor,
        mode_id: int,
        q_e4: int,
        reward: float,
        duration: int,
        terminated: bool,
    ) -> None:
        for name, tensor in (("state", state), ("next_state", next_state)):
            if tensor.shape != (POLICY_FEATURE_COUNT,):
                raise QualificationError(f"{name} has the wrong shape")
            if tensor.dtype is not self.dtype or tensor.device.type != "cpu":
                raise QualificationError(f"{name} must be CPU float32")
            if not bool(torch.isfinite(tensor).all()):
                raise QualificationError(f"{name} contains non-finite values")
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise QualificationError("replay mode is outside [0, 11]")
        if not 0 <= q_e4 <= Q_E4_MAX:
            raise QualificationError("replay q_e4 is outside [0, 9800]")
        _require_finite(reward, "replay reward")
        if duration < MINIMUM_HOLD_TENSORS:
            raise QualificationError("qualification duration violates hold floor")
        discount = float(self.binding.gamma_per_tensor ** duration)
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise QualificationError("derived qualification discount is invalid")
        row = _QualificationTransition(
            transition_id=self.total_inserted,
            state=state.detach().clone(),
            next_state=next_state.detach().clone(),
            mode_id=int(mode_id),
            q_e4=int(q_e4),
            reward=float(reward),
            duration=int(duration),
            discount=discount,
            terminated=bool(terminated),
        )
        if len(self._rows) == self.capacity:
            self._rows.pop(0)
        self._rows.append(row)
        self.total_inserted += 1

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ReplayTensorBatchV1:
        if not isinstance(generator, torch.Generator):
            raise QualificationError("replay sampling needs an explicit generator")
        if generator is torch.default_generator or generator.device.type != "cpu":
            raise QualificationError("replay generator must be local CPU RNG")
        if not 1 <= batch_size <= len(self._rows):
            raise QualificationError("qualification replay cannot supply batch")
        indices = torch.randperm(len(self._rows), generator=generator)[
            :batch_size
        ].tolist()
        rows = [self._rows[index] for index in indices]
        state = torch.stack([row.state for row in rows])
        next_state = torch.stack([row.next_state for row in rows])
        terminated = torch.tensor(
            [row.terminated for row in rows], dtype=torch.bool
        )
        has_next = torch.ones(batch_size, dtype=torch.bool)
        bootstrap = has_next & (~terminated)
        return ReplayTensorBatchV1(
            _state=state,
            _next_state=next_state,
            _mode_id=torch.tensor([row.mode_id for row in rows], dtype=torch.int64),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor([row.reward for row in rows], dtype=self.dtype),
            _duration=torch.tensor(
                [row.duration for row in rows], dtype=torch.int64
            ),
            _discount=torch.tensor(
                [row.discount for row in rows], dtype=self.dtype
            ),
            _has_next_state=has_next,
            _bootstrap=bootstrap,
            _terminated=terminated,
            _truncated=torch.zeros(batch_size, dtype=torch.bool),
            binding=self.binding,
            audit=tuple(
                {
                    "evidence_class": PHASE_LABEL,
                    "qualification_transition_id": row.transition_id,
                    "source": "ANALYTIC_TOY_ENVIRONMENT_NOT_SPLITFUSION_EVIDENCE",
                }
                for row in rows
            ),
            float_dtype=self.dtype,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": QUALIFICATION_REPLAY_SCHEMA,
            "capacity": self.capacity,
            "binding": self.binding.to_canonical_dict(),
            "dtype": str(self.dtype),
            "total_inserted": self.total_inserted,
            "rows": [
                {
                    "transition_id": row.transition_id,
                    "state": row.state.clone(),
                    "next_state": row.next_state.clone(),
                    "mode_id": row.mode_id,
                    "q_e4": row.q_e4,
                    "reward": row.reward,
                    "duration": row.duration,
                    "discount": row.discount,
                    "terminated": row.terminated,
                }
                for row in self._rows
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise QualificationError("checkpoint replay state is not a mapping")
        expected_keys = {
            "schema",
            "capacity",
            "binding",
            "dtype",
            "total_inserted",
            "rows",
        }
        if set(state) != expected_keys:
            raise QualificationError("checkpoint replay fields differ")
        if state.get("schema") != QUALIFICATION_REPLAY_SCHEMA:
            raise QualificationError("unknown qualification replay schema")
        if type(state.get("capacity")) is not int or state.get("capacity") != self.capacity:
            raise QualificationError("checkpoint replay capacity differs")
        _require_canonical_binding(state.get("binding"), self.binding)
        if state.get("dtype") != str(self.dtype):
            raise QualificationError("checkpoint replay dtype differs")
        rows = state.get("rows")
        total_inserted = state.get("total_inserted")
        if not isinstance(rows, list) or len(rows) > self.capacity:
            raise QualificationError("invalid checkpoint replay rows")
        total_inserted = _require_exact_int(
            total_inserted, "checkpoint replay counter"
        )
        if total_inserted < len(rows):
            raise QualificationError("checkpoint replay counter precedes its rows")
        first_expected_id = total_inserted - len(rows)
        rebuilt: List[_QualificationTransition] = []
        row_keys = {
            "transition_id",
            "state",
            "next_state",
            "mode_id",
            "q_e4",
            "reward",
            "duration",
            "discount",
            "terminated",
        }
        for offset, item in enumerate(rows):
            if not isinstance(item, Mapping):
                raise QualificationError("invalid replay row in checkpoint")
            if set(item) != row_keys:
                raise QualificationError("checkpoint replay row fields differ")
            transition_id = _require_exact_int(
                item["transition_id"], "checkpoint transition_id"
            )
            if transition_id != first_expected_id + offset:
                raise QualificationError(
                    "checkpoint replay transition IDs are not the retained "
                    "contiguous suffix"
                )
            tensors: Dict[str, Tensor] = {}
            for name in ("state", "next_state"):
                tensor = item[name]
                if (
                    not isinstance(tensor, Tensor)
                    or tensor.shape != (POLICY_FEATURE_COUNT,)
                    or tensor.dtype is not self.dtype
                    or tensor.device.type != "cpu"
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise QualificationError(
                        f"checkpoint replay {name} is not finite CPU float32 "
                        f"with shape ({POLICY_FEATURE_COUNT},)"
                    )
                tensors[name] = tensor.detach().clone()
            mode_id = _require_exact_int(item["mode_id"], "checkpoint mode_id")
            if mode_id >= EXPECTED_MODE_COUNT:
                raise QualificationError("checkpoint mode_id is outside [0, 11]")
            q_e4 = _require_exact_int(item["q_e4"], "checkpoint q_e4")
            if q_e4 > Q_E4_MAX:
                raise QualificationError("checkpoint q_e4 is outside [0, 9800]")
            if type(item["reward"]) is not float:
                raise QualificationError("checkpoint reward type differs")
            reward = _require_finite(item["reward"], "checkpoint reward")
            reward_tensor = torch.tensor(reward, dtype=self.dtype)
            if not bool(torch.isfinite(reward_tensor)):
                raise QualificationError(
                    "checkpoint reward is not finite after float32 conversion"
                )
            duration = _require_exact_int(
                item["duration"],
                "checkpoint duration",
                MINIMUM_HOLD_TENSORS,
            )
            if type(item["discount"]) is not float:
                raise QualificationError("checkpoint discount type differs")
            discount = _require_finite(item["discount"], "checkpoint discount")
            expected_discount = float(self.binding.gamma_per_tensor ** duration)
            if discount != expected_discount:
                raise QualificationError(
                    "checkpoint discount is not the binding-derived "
                    "gamma_per_tensor ** duration value"
                )
            converted_discount = torch.tensor(discount, dtype=self.dtype)
            if (
                not bool(torch.isfinite(converted_discount))
                or not 0.0 <= float(converted_discount) <= 1.0
            ):
                raise QualificationError(
                    "checkpoint discount is invalid after float32 conversion"
                )
            terminated = item["terminated"]
            if not isinstance(terminated, bool):
                raise QualificationError("checkpoint terminated must be bool")
            rebuilt.append(
                _QualificationTransition(
                    transition_id=transition_id,
                    state=tensors["state"],
                    next_state=tensors["next_state"],
                    mode_id=mode_id,
                    q_e4=q_e4,
                    reward=reward,
                    duration=duration,
                    discount=discount,
                    terminated=terminated,
                )
            )
        self._rows = rebuilt
        self.total_inserted = total_inserted


class HybridSacAlgorithmQualificationRunnerV1:
    """Bounded CPU runner around :class:`HybridSacTrainerV1`."""

    _STREAM_OFFSETS = {
        "collection": 101,
        "replay": 211,
        "target": 307,
        "actor": 401,
        "evaluation": 503,
    }

    def __init__(self, config: QualificationRunnerConfigV1) -> None:
        if type(config) is not QualificationRunnerConfigV1:
            raise QualificationError("config must be QualificationRunnerConfigV1")
        self.config = config
        self.binding = build_qualification_binding(config.gamma_per_tensor)
        self.generators: Dict[str, torch.Generator] = {
            name: _new_generator(config.seed * 10_000 + offset)
            for name, offset in self._STREAM_OFFSETS.items()
        }
        if len({id(generator) for generator in self.generators.values()}) != 5:
            raise QualificationError("qualification RNG streams are not distinct")
        # Preserve the seed-derived start of the analytic evaluation stream.
        # The active evaluation generator advances when evaluate() is called;
        # this immutable snapshot makes any recorded fixed-stream evaluation
        # exactly reproducible after a checkpoint is loaded.
        self.fixed_evaluation_rng_state = (
            self.generators["evaluation"].get_state().clone()
        )

        model_config = HybridSacModelConfig(dtype=torch.float32)
        # The builders use fork_rng, making initialization deterministic while
        # restoring the caller's global RNG on exit.
        self.actor = build_actor(model_config, seed=config.seed * 10_000 + 601)
        self.critics = build_twin_critics(
            model_config, seed=config.seed * 10_000 + 701
        )
        trainer_config = TrainerConfigV1(
            gamma_per_tensor=config.gamma_per_tensor,
            alpha_d=config.alpha_d,
            alpha_c=config.alpha_c,
            tau=config.tau,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            batch_size=config.batch_size,
            float_dtype=torch.float32,
            hyperparameter_status=(
                "HYBRID_SAC_ALGORITHM_QUALIFICATION_ONLY_NOT_FROZEN"
            ),
        )
        self.trainer = HybridSacTrainerV1(
            self.actor,
            self.critics,
            trainer_config,
            expected_binding=self.binding,
            target_generator=self.generators["target"],
            actor_generator=self.generators["actor"],
        )
        self.environment = QualificationEnvironmentV1(config.episode_horizon)
        self.replay = QualificationReplayV1(
            config.replay_capacity, self.binding, torch.float32
        )
        self.current_state: Optional[Tensor] = None
        self.collected_transitions = 0
        self.update_history: List[Dict[str, Any]] = []

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    def _random_action(self) -> Tuple[int, int]:
        generator = self.generators["collection"]
        mode = int(
            torch.randint(
                EXPECTED_MODE_COUNT, (1,), generator=generator, dtype=torch.int64
            ).item()
        )
        q_e4 = int(
            torch.randint(
                Q_E4_MAX + 1, (1,), generator=generator, dtype=torch.int64
            ).item()
        )
        return mode, q_e4

    @torch.no_grad()
    def _policy_action(self, state: Tensor) -> Tuple[int, int]:
        sample = self.actor.sample_all_modes(
            state.unsqueeze(0), generator=self.generators["collection"]
        )
        mode = int(
            torch.multinomial(
                sample.probs[0],
                num_samples=1,
                replacement=True,
                generator=self.generators["collection"],
            ).item()
        )
        return mode, int(sample.q_e4[0, mode].item())

    def collect_one(self, *, force_random: bool = False) -> None:
        if self.current_state is None or self.environment.needs_reset:
            self.current_state = self.environment.reset(
                self.generators["collection"]
            )
        state = self.current_state
        use_random = force_random or len(self.replay) < self.config.warmup_transitions
        mode_id, q_e4 = (
            self._random_action() if use_random else self._policy_action(state)
        )
        next_state, reward, terminated = self.environment.step(
            mode_id, q_e4, self.generators["collection"]
        )
        self.replay.append(
            state=state,
            next_state=next_state,
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            duration=MINIMUM_HOLD_TENSORS,
            terminated=terminated,
        )
        self.collected_transitions += 1
        self.current_state = None if terminated else next_state

    def _ensure_warmup(self) -> None:
        while len(self.replay) < self.config.warmup_transitions:
            self.collect_one(force_random=True)

    def advance(self, updates: int) -> Tuple[UpdateMetricsV1, ...]:
        """Run a bounded number of updates; long training is caller-authorized."""
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise QualificationError("updates must be a non-negative integer")
        if self.update_count + updates > self.config.max_updates:
            raise QualificationError("requested updates exceed max_updates")
        if updates == 0:
            return tuple()
        self._ensure_warmup()
        emitted: List[UpdateMetricsV1] = []
        for _ in range(updates):
            for _ in range(self.config.collect_per_update):
                self.collect_one()
            batch = self.replay.sample(
                self.config.batch_size, self.generators["replay"]
            )
            raw_metric = self.trainer.update_once(batch)
            # The shared trainer retains its conservative smoke-test label.
            # Records emitted by this wrapper are normalized to this narrower
            # qualification run's evidence class.
            metric = replace(raw_metric, phase_label=PHASE_LABEL)
            metric.assert_finite()
            record = metric.as_dict()
            for name, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    _require_finite(float(value), f"update metric {name}")
            self.update_history.append(record)
            emitted.append(metric)
        return tuple(emitted)

    def reset_fixed_evaluation_stream(self) -> None:
        """Reset evaluation to its persisted seed-derived initial RNG state."""
        self.generators["evaluation"].set_state(
            self.fixed_evaluation_rng_state.clone()
        )

    def _fixed_evaluation_contexts(self, steps: int) -> Tensor:
        if not isinstance(steps, int) or steps < EXPECTED_MODE_COUNT:
            raise QualificationError(
                f"fixed evaluation steps must be at least {EXPECTED_MODE_COUNT}"
            )
        base = torch.arange(steps, dtype=torch.int64) % EXPECTED_MODE_COUNT
        order = torch.randperm(steps, generator=self.generators["evaluation"])
        return base.index_select(0, order)

    @torch.no_grad()
    def evaluate(
        self,
        steps: int = 240,
        *,
        fixed_mode: int = 0,
        fixed_q_e4: int = Q_E4_MAX // 2,
        thresholds: Optional[AcceptanceThresholdsV1] = None,
    ) -> EvaluationMetricsV1:
        """Compare policy and baselines on the disclosed analytic state family.

        Call :meth:`reset_fixed_evaluation_stream` before evaluation when the
        seed's registered fixed stream is required.  The states directly encode
        their target mode and q; these metrics make no generalization claim.
        """
        thresholds = thresholds or AcceptanceThresholdsV1()
        if not 0 <= fixed_mode < EXPECTED_MODE_COUNT:
            raise QualificationError("fixed_mode is outside [0, 11]")
        if not 0 <= fixed_q_e4 <= Q_E4_MAX:
            raise QualificationError("fixed_q_e4 is outside [0, 9800]")
        contexts = self._fixed_evaluation_contexts(steps)
        states = torch.stack(
            [
                self.environment.observation(int(context), 0.5)
                for context in contexts.tolist()
            ]
        )
        training = self.actor.training
        self.actor.eval()
        execution = self.actor.deterministic_execution(states)
        self.actor.train(training)

        random_modes = torch.randint(
            EXPECTED_MODE_COUNT,
            (steps,),
            generator=self.generators["evaluation"],
            dtype=torch.int64,
        )
        random_q = torch.randint(
            Q_E4_MAX + 1,
            (steps,),
            generator=self.generators["evaluation"],
            dtype=torch.int64,
        )
        policy_rewards: List[float] = []
        random_rewards: List[float] = []
        fixed_rewards: List[float] = []
        selected_modes: set[int] = set()
        interior_q_count = 0
        mode_correct = 0
        q_absolute_error = 0.0
        for index, context in enumerate(contexts.tolist()):
            policy_mode = int(execution.mode_index[index])
            policy_q = int(execution.q_e4[index])
            selected_modes.add(policy_mode)
            interior_q_count += int(0 < policy_q < Q_E4_MAX)
            policy_rewards.append(
                self.environment.reward_for(context, policy_mode, policy_q)
            )
            random_rewards.append(
                self.environment.reward_for(
                    context, int(random_modes[index]), int(random_q[index])
                )
            )
            fixed_rewards.append(
                self.environment.reward_for(context, fixed_mode, fixed_q_e4)
            )
            mode_correct += int(policy_mode == context)
            q_absolute_error += abs(
                policy_q - self.environment.Q_OPTIMA_E4[context]
            ) / Q_E4_MAX

        policy_mean = sum(policy_rewards) / steps
        random_mean = sum(random_rewards) / steps
        fixed_mean = sum(fixed_rewards) / steps
        oracle_mean = 1.0
        improvement = policy_mean - random_mean
        regret = oracle_mean - policy_mean
        selected_mode_count = len(selected_modes)
        interior_q_fraction = interior_q_count / steps
        accepted = (
            improvement >= thresholds.minimum_improvement_over_random
            and regret <= thresholds.maximum_oracle_regret
            and selected_mode_count == EXPECTED_MODE_COUNT
            and interior_q_fraction == 1.0
        )
        result = EvaluationMetricsV1(
            steps=steps,
            policy_mean_reward=policy_mean,
            random_mean_reward=random_mean,
            fixed_mean_reward=fixed_mean,
            oracle_mean_reward=oracle_mean,
            improvement_over_random=improvement,
            oracle_regret=regret,
            policy_mode_accuracy=mode_correct / steps,
            policy_mean_absolute_q_error=q_absolute_error / steps,
            policy_selected_mode_count=selected_mode_count,
            policy_interior_q_fraction=interior_q_fraction,
            fixed_mode=fixed_mode,
            fixed_q_e4=fixed_q_e4,
            minimum_improvement_over_random=(
                thresholds.minimum_improvement_over_random
            ),
            maximum_oracle_regret=thresholds.maximum_oracle_regret,
            accepted=accepted,
        )
        result.assert_finite()
        return result

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return a deep, deterministic snapshot of every mutable component."""
        return copy.deepcopy(
            {
                "schema": CHECKPOINT_SCHEMA,
                "phase_label": PHASE_LABEL,
                "config": self.config.to_canonical_dict(),
                "binding": self.binding.to_canonical_dict(),
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
                "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
                "generators": {
                    name: generator.get_state().clone()
                    for name, generator in self.generators.items()
                },
                "fixed_evaluation_rng_state": (
                    self.fixed_evaluation_rng_state.clone()
                ),
                "replay": self.replay.state_dict(),
                "environment": self.environment.state_dict(),
                "current_state": (
                    None
                    if self.current_state is None
                    else self.current_state.detach().clone()
                ),
                "collected_transitions": self.collected_transitions,
                "trainer_update_count": self.trainer.update_count,
                "update_history": copy.deepcopy(self.update_history),
                "module_training": {
                    "actor": self.actor.training,
                    "critics": self.critics.training,
                },
            }
        )

    @staticmethod
    def _validate_module_state(
        payload: Any, expected: Mapping[str, Tensor], label: str
    ) -> None:
        if not isinstance(payload, Mapping) or set(payload) != set(expected):
            raise QualificationError(f"checkpoint {label} state fields differ")
        for name, reference in expected.items():
            value = payload[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != reference.shape
                or value.dtype is not reference.dtype
                or value.device.type != "cpu"
            ):
                raise QualificationError(
                    f"checkpoint {label}.{name} tensor metadata differs"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise QualificationError(
                    f"checkpoint {label}.{name} contains a non-finite value"
                )

    @staticmethod
    def _validate_optimizer_payload(
        payload: Any, optimizer: torch.optim.Optimizer, label: str
    ) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "state",
            "param_groups",
        }:
            raise QualificationError(f"checkpoint {label} optimizer fields differ")
        observed_groups = payload["param_groups"]
        expected_groups = optimizer.state_dict()["param_groups"]
        if not isinstance(observed_groups, list) or len(observed_groups) != len(
            expected_groups
        ):
            raise QualificationError(f"checkpoint {label} parameter groups differ")
        known_parameter_ids: List[int] = []
        for observed, expected in zip(observed_groups, expected_groups):
            if not isinstance(observed, Mapping) or set(observed) != set(expected):
                raise QualificationError(
                    f"checkpoint {label} parameter-group fields differ"
                )
            for name, expected_value in expected.items():
                if name == "params":
                    params = observed[name]
                    if not isinstance(params, list) or len(params) != len(
                        expected_value
                    ):
                        raise QualificationError(
                            f"checkpoint {label} parameter inventory differs"
                        )
                    if any(isinstance(item, bool) or not isinstance(item, int) for item in params):
                        raise QualificationError(
                            f"checkpoint {label} parameter IDs are invalid"
                        )
                    known_parameter_ids.extend(params)
                elif observed[name] != expected_value:
                    raise QualificationError(
                        f"checkpoint {label} option {name} differs"
                    )
        if len(known_parameter_ids) != len(set(known_parameter_ids)):
            raise QualificationError(
                f"checkpoint {label} contains duplicate parameter IDs"
            )
        state = payload["state"]
        if not isinstance(state, Mapping) or not set(state).issubset(
            set(known_parameter_ids)
        ):
            raise QualificationError(f"checkpoint {label} optimizer state differs")

    @staticmethod
    def _validate_loaded_adam(
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[Tensor],
        update_count: int,
        label: str,
    ) -> None:
        for index, parameter in enumerate(parameters):
            state = optimizer.state.get(parameter, {})
            if update_count == 0:
                if state:
                    raise QualificationError(
                        f"checkpoint {label} state exists before any update"
                    )
                continue
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise QualificationError(
                    f"checkpoint {label} parameter {index} has incomplete Adam state"
                )
            step = state["step"]
            if (
                not isinstance(step, Tensor)
                or step.numel() != 1
                or step.device.type != "cpu"
                or not bool(torch.isfinite(step).all())
                or float(step) != float(update_count)
            ):
                raise QualificationError(
                    f"checkpoint {label} parameter {index} has wrong Adam step"
                )
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if (
                    not isinstance(value, Tensor)
                    or value.shape != parameter.shape
                    or value.dtype is not parameter.dtype
                    or value.device.type != "cpu"
                    or not bool(torch.isfinite(value).all())
                ):
                    raise QualificationError(
                        f"checkpoint {label} parameter {index} {name} is invalid"
                    )

    def _load_checkpoint_state_inplace(
        self, checkpoint: Mapping[str, Any]
    ) -> None:
        """Validate and load into a disposable fresh runner."""
        if not isinstance(checkpoint, Mapping):
            raise QualificationError("checkpoint is not a mapping")
        expected_fields = {
            "schema",
            "phase_label",
            "config",
            "binding",
            "actor",
            "critics",
            "actor_optimizer",
            "critic_optimizer",
            "generators",
            "fixed_evaluation_rng_state",
            "replay",
            "environment",
            "current_state",
            "collected_transitions",
            "trainer_update_count",
            "update_history",
            "module_training",
        }
        if set(checkpoint) != expected_fields:
            raise QualificationError("checkpoint fields differ")
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise QualificationError("unknown qualification checkpoint schema")
        if checkpoint.get("phase_label") != PHASE_LABEL:
            raise QualificationError("checkpoint is not algorithm qualification")
        config_mapping = checkpoint.get("config")
        if not isinstance(config_mapping, Mapping):
            raise QualificationError("checkpoint runner configuration is invalid")
        try:
            restored_config = QualificationRunnerConfigV1(**dict(config_mapping))
        except Exception as exc:
            raise QualificationError(
                "checkpoint runner configuration is invalid"
            ) from exc
        if restored_config != self.config:
            raise QualificationError("checkpoint runner configuration differs")
        _require_canonical_binding(checkpoint.get("binding"), self.binding)

        collected = _require_exact_int(
            checkpoint.get("collected_transitions"),
            "checkpoint collection counter",
        )
        update_count = _require_exact_int(
            checkpoint.get("trainer_update_count"),
            "checkpoint update counter",
        )
        if update_count > self.config.max_updates:
            raise QualificationError("checkpoint update counter exceeds max_updates")
        history = checkpoint.get("update_history")
        if not isinstance(history, list) or len(history) != update_count:
            raise QualificationError("checkpoint update history is inconsistent")
        metric_fields = set(UpdateMetricsV1.__dataclass_fields__)
        metric_integer_fields = {
            "batch_size",
            "bootstrap_count",
            "duration_min",
            "duration_max",
        }
        validated_history: List[Dict[str, Any]] = []
        for record in history:
            if not isinstance(record, Mapping) or set(record) != metric_fields:
                raise QualificationError("checkpoint metric record fields differ")
            for name in metric_integer_fields:
                if type(record[name]) is not int:
                    raise QualificationError(
                        f"checkpoint metric {name} type differs"
                    )
            for name in metric_fields - metric_integer_fields - {"phase_label"}:
                if type(record[name]) is not float:
                    raise QualificationError(
                        f"checkpoint metric {name} type differs"
                    )
            if record["phase_label"] != PHASE_LABEL:
                raise QualificationError("checkpoint metric phase label differs")
            try:
                metric = UpdateMetricsV1(**dict(record))
                metric.assert_finite()
            except Exception as exc:
                raise QualificationError(
                    f"checkpoint metric record is invalid: {exc}"
                ) from exc
            validated_history.append(metric.as_dict())

        module_training = checkpoint.get("module_training")
        if (
            not isinstance(module_training, Mapping)
            or set(module_training) != {"actor", "critics"}
            or not all(isinstance(value, bool) for value in module_training.values())
        ):
            raise QualificationError("checkpoint module training modes are invalid")

        generator_states = checkpoint.get("generators")
        if not isinstance(generator_states, Mapping) or set(generator_states) != set(
            self._STREAM_OFFSETS
        ):
            raise QualificationError("checkpoint RNG inventory differs")
        validated_generator_states: Dict[str, Tensor] = {}
        for name, state in generator_states.items():
            if (
                not isinstance(state, Tensor)
                or state.dtype is not torch.uint8
                or state.device.type != "cpu"
                or state.ndim != 1
            ):
                raise QualificationError(f"checkpoint RNG {name} state is invalid")
            probe = _new_generator(0)
            try:
                probe.set_state(state.detach().clone())
            except Exception as exc:
                raise QualificationError(
                    f"checkpoint RNG {name} state is invalid"
                ) from exc
            validated_generator_states[name] = state.detach().clone()

        fixed_evaluation_rng_state = checkpoint.get(
            "fixed_evaluation_rng_state"
        )
        if (
            not isinstance(fixed_evaluation_rng_state, Tensor)
            or fixed_evaluation_rng_state.dtype is not torch.uint8
            or fixed_evaluation_rng_state.device.type != "cpu"
            or fixed_evaluation_rng_state.ndim != 1
        ):
            raise QualificationError(
                "checkpoint fixed evaluation RNG state is invalid"
            )
        fixed_evaluation_rng_state = (
            fixed_evaluation_rng_state.detach().clone()
        )
        fixed_probe = _new_generator(0)
        try:
            fixed_probe.set_state(fixed_evaluation_rng_state)
        except Exception as exc:
            raise QualificationError(
                "checkpoint fixed evaluation RNG state is invalid"
            ) from exc
        if not torch.equal(
            fixed_evaluation_rng_state, self.fixed_evaluation_rng_state
        ):
            raise QualificationError(
                "checkpoint fixed evaluation RNG state differs from seed"
            )

        self._validate_module_state(
            checkpoint["actor"], self.actor.state_dict(), "actor"
        )
        self._validate_module_state(
            checkpoint["critics"], self.critics.state_dict(), "critics"
        )
        self._validate_optimizer_payload(
            checkpoint["actor_optimizer"],
            self.trainer.actor_optimizer,
            "actor",
        )
        self._validate_optimizer_payload(
            checkpoint["critic_optimizer"],
            self.trainer.critic_optimizer,
            "critic",
        )

        # Replay and environment loaders fully validate into this disposable
        # runner.  Any failure is invisible to the caller's live runner.
        self.replay.load_state_dict(checkpoint["replay"])
        self.environment.load_state_dict(checkpoint["environment"])
        if self.replay.total_inserted != collected:
            raise QualificationError(
                "checkpoint collection counter differs from replay history"
            )
        expected_collected = (
            self.environment.episode_index * self.environment.horizon
        )
        if not self.environment.needs_reset:
            expected_collected += self.environment.step_in_episode
        if collected != expected_collected:
            raise QualificationError(
                "checkpoint environment progress differs from collection counter"
            )

        current_state = checkpoint.get("current_state")
        if self.environment.needs_reset:
            if current_state is not None:
                raise QualificationError(
                    "checkpoint has a current state while reset is required"
                )
            validated_current_state = None
        else:
            if (
                not isinstance(current_state, Tensor)
                or current_state.shape != (POLICY_FEATURE_COUNT,)
                or current_state.dtype is not torch.float32
                or current_state.device.type != "cpu"
                or not bool(torch.isfinite(current_state).all())
            ):
                raise QualificationError("checkpoint current_state is invalid")
            expected_state = self.environment.observation(
                int(self.environment.context_id),
                min(
                    1.0,
                    self.environment.step_in_episode / self.environment.horizon,
                ),
            )
            if not torch.equal(current_state, expected_state):
                raise QualificationError(
                    "checkpoint current_state differs from environment state"
                )
            validated_current_state = current_state.detach().clone()

        try:
            self.actor.load_state_dict(checkpoint["actor"], strict=True)
            self.critics.load_state_dict(checkpoint["critics"], strict=True)
            self.trainer.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer"]
            )
            self.trainer.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )
        except Exception as exc:
            raise QualificationError("checkpoint model/optimizer load failed") from exc

        self._validate_loaded_adam(
            self.trainer.actor_optimizer,
            self.actor.parameters(),
            update_count,
            "actor",
        )
        self._validate_loaded_adam(
            self.trainer.critic_optimizer,
            self.trainer._online_critic_parameters,
            update_count,
            "critic",
        )
        for name, state in validated_generator_states.items():
            self.generators[name].set_state(state)
        self.fixed_evaluation_rng_state = fixed_evaluation_rng_state
        self.current_state = validated_current_state
        self.collected_transitions = collected
        self.trainer.update_count = update_count
        self.update_history = copy.deepcopy(validated_history)
        self.actor.train(bool(module_training["actor"]))
        self.critics.train(bool(module_training["critics"]))
        self.trainer._assert_no_target_parameters_in_optimizers()

    def load_checkpoint_state(self, checkpoint: Mapping[str, Any]) -> None:
        """Transactionally restore a fully validated qualification snapshot.

        Validation and deserialization occur on a disposable runner.  The
        live runner is changed only after every replay, model, optimizer, RNG,
        environment and counter invariant has passed.
        """
        staged = type(self)(self.config)
        staged._load_checkpoint_state_inplace(checkpoint)
        self.binding = staged.binding
        self.generators = staged.generators
        self.fixed_evaluation_rng_state = staged.fixed_evaluation_rng_state
        self.actor = staged.actor
        self.critics = staged.critics
        self.trainer = staged.trainer
        self.environment = staged.environment
        self.replay = staged.replay
        self.current_state = staged.current_state
        self.collected_transitions = staged.collected_transitions
        self.update_history = staged.update_history

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any]
    ) -> "HybridSacAlgorithmQualificationRunnerV1":
        config_mapping = checkpoint.get("config")
        if not isinstance(config_mapping, Mapping):
            raise QualificationError("checkpoint has no runner configuration")
        runner = cls(QualificationRunnerConfigV1(**dict(config_mapping)))
        runner.load_checkpoint_state(checkpoint)
        return runner

    def save_checkpoint(self, path: Path) -> str:
        """Atomically save a data-only checkpoint and return its file hash."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        payload_buffer = io.BytesIO()
        torch.save(self.checkpoint_state(), payload_buffer)
        payload = payload_buffer.getvalue()
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        envelope = {
            "schema": CHECKPOINT_FILE_SCHEMA,
            "payload_sha256": payload_sha256,
            "payload": payload,
        }
        file_buffer = io.BytesIO()
        torch.save(envelope, file_buffer)
        file_bytes = file_buffer.getvalue()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        with temporary.open("wb") as handle:
            handle.write(file_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return file_sha256

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, expected_sha256: str
    ) -> "HybridSacAlgorithmQualificationRunnerV1":
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise QualificationError(
                "expected_sha256 must be a lowercase 64-character SHA-256"
            )
        file_bytes = Path(path).read_bytes()
        observed_sha256 = hashlib.sha256(file_bytes).hexdigest()
        if observed_sha256 != expected_sha256:
            raise QualificationError("checkpoint file SHA-256 differs")
        try:
            envelope = torch.load(
                io.BytesIO(file_bytes), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise QualificationError("safe checkpoint envelope load failed") from exc
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "schema",
            "payload_sha256",
            "payload",
        }:
            raise QualificationError("checkpoint file envelope fields differ")
        if envelope["schema"] != CHECKPOINT_FILE_SCHEMA:
            raise QualificationError("unknown checkpoint file schema")
        payload = envelope["payload"]
        if not isinstance(payload, bytes):
            raise QualificationError("checkpoint payload is not bytes")
        if hashlib.sha256(payload).hexdigest() != envelope["payload_sha256"]:
            raise QualificationError("checkpoint payload SHA-256 differs")
        try:
            checkpoint = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise QualificationError("safe checkpoint payload load failed") from exc
        if not isinstance(checkpoint, Mapping):
            raise QualificationError("checkpoint file is not a mapping")
        return cls.from_checkpoint(checkpoint)


def run_seed_suite(
    base_config: QualificationRunnerConfigV1,
    seeds: Sequence[int],
    *,
    updates: int,
    evaluation_steps: int = 120,
) -> Tuple[SeedQualificationResultV1, ...]:
    """Run an explicitly bounded independent-seed qualification suite.

    Three or more seeds are required so a caller cannot accidentally present a
    single initialization as a seed suite.  This helper still carries only the
    algorithm-qualification label; it does not upgrade the toy task to
    SplitFusion evidence.
    """
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise QualificationError("a seed suite requires at least three unique seeds")
    results: List[SeedQualificationResultV1] = []
    base = base_config.to_canonical_dict()
    for seed in seeds:
        config_values = dict(base)
        config_values["seed"] = int(seed)
        runner = HybridSacAlgorithmQualificationRunnerV1(
            QualificationRunnerConfigV1(**config_values)
        )
        runner.advance(updates)
        runner.reset_fixed_evaluation_stream()
        evaluation = runner.evaluate(evaluation_steps)
        results.append(
            SeedQualificationResultV1(
                seed=int(seed),
                update_count=runner.update_count,
                collected_transitions=runner.collected_transitions,
                evaluation=evaluation,
            )
        )
    return tuple(results)

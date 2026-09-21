"""D2b mechanics-only runner over the complete registered D1 fit sampler.

This is deliberately *not* an experiment runner.  It does not import the
fit train/validation partition, evaluate a policy, select a checkpoint, or
make a convergence/generalization claim.  Its only purpose is to exercise the
mechanical path from a synchronous D1 one-step episode through the private
terminal replay and one D2a numerical update.

The registered smoke consists of three independent runs (seeds 17, 29 and
43).  Every run uses the full registered D1 fit distribution.  Consequently,
its metrics are training-mechanics diagnostics only.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import struct
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .empirical_contextual_contract import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
    PILOT_UTILITY_SPEC_SHA256,
    EmpiricalActionV1,
    require_supported_action,
)
from .empirical_contextual_environment import (
    EmpiricalEnvironmentStateV1,
    EmpiricalOneStepEnvironmentV1,
    EmpiricalPolicyObservationV1,
    EmpiricalStepResultV1,
)
from .empirical_contextual_terminal_replay import (
    PHASE_LABEL as REPLAY_PHASE_LABEL,
    EmpiricalTerminalReplayV1,
    EmpiricalTerminalTransitionV1,
)
from .empirical_contextual_terminal_trainer import (
    PHASE_LABEL as TRAINER_PHASE_LABEL,
    EmpiricalTerminalHybridSacTrainerV1,
    EmpiricalTerminalTrainerConfigV1,
    EmpiricalTerminalUpdateMetricsV1,
)
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "EmpiricalContextualSmokeRunnerV1",
    "EmpiricalSmokeCheckpointV1",
    "EmpiricalSmokeConfigV1",
    "EmpiricalSmokeError",
    "EmpiricalSmokeSummaryV1",
    "EmpiricalThreeSeedSmokeReportV1",
    "PHASE_LABEL",
    "REGISTERED_SMOKE_CONFIG",
    "registered_warmup_mode_counts",
    "run_registered_three_seed_smoke",
]


PHASE_LABEL = (
    "D2B_MECHANICAL_SMOKE_FULL_D1_FIT_ONLY_"
    "NO_VALIDATION_NO_CONVERGENCE_OR_GENERALIZATION_CLAIM"
)
SCOPE_DISCLOSURE = (
    "Uses the complete registered D1 fit sampler, not the committed "
    "train/fit-validation partition. Metrics test mechanics only and must "
    "not be reported as validation, convergence, or generalization evidence."
)
RUNNER_SCHEMA = "splitfusion.empirical_contextual_mechanical_smoke.v1"
SEED_DERIVATION_SCHEMA = "splitfusion.empirical_contextual_smoke.rng.v1"
_COLLECTION_NAMESPACE = uuid.UUID("78e41f91-4ee8-50a5-b063-11c198faf181")


class EmpiricalSmokeError(RuntimeError):
    """The mechanics-only runner encountered a contract violation."""


def _exact_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise EmpiricalSmokeError(f"{name} must be an exact positive integer")
    return value


def _derive_seed(master_seed: int, stream: str) -> int:
    document = f"{SEED_DERIVATION_SCHEMA}:{master_seed}:{stream}".encode("ascii")
    # torch.Generator.manual_seed accepts signed/unsigned 64-bit values.  Keep
    # the high bit clear for equally portable Python and Torch construction.
    return int.from_bytes(hashlib.sha256(document).digest()[:8], "big") & (
        (1 << 63) - 1
    )


@dataclass(frozen=True, slots=True)
class EmpiricalSmokeConfigV1:
    """Frozen mechanics-smoke schedule; smaller values are test-only."""

    seeds: Tuple[int, ...] = (17, 29, 43)
    warmup_transitions: int = 1024
    batch_size: int = 256
    collect_per_update: int = 4
    update_count: int = 500
    replay_capacity: int = 8192
    cpu_threads: int = 1
    scope: str = PHASE_LABEL

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise EmpiricalSmokeError(
                "seeds must be a non-empty tuple of unique non-negative integers"
            )
        for name in (
            "warmup_transitions",
            "batch_size",
            "collect_per_update",
            "update_count",
            "replay_capacity",
            "cpu_threads",
        ):
            _exact_positive_int(getattr(self, name), name)
        if self.cpu_threads != 1:
            raise EmpiricalSmokeError("D2b is fixed to one CPU thread")
        if self.warmup_transitions < self.batch_size:
            raise EmpiricalSmokeError("warmup must contain at least one full batch")
        if self.total_transitions > self.replay_capacity:
            raise EmpiricalSmokeError(
                "the mechanics smoke must retain its full history without eviction"
            )
        if self.scope != PHASE_LABEL:
            raise EmpiricalSmokeError("mechanics-only scope label drift")

    @property
    def total_transitions(self) -> int:
        return self.warmup_transitions + self.collect_per_update * self.update_count

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "collect_per_update": self.collect_per_update,
            "cpu_threads": self.cpu_threads,
            "record": "empirical_contextual_smoke_config_v1",
            "replay_capacity": self.replay_capacity,
            "scope": self.scope,
            "seeds": list(self.seeds),
            "update_count": self.update_count,
            "warmup_transitions": self.warmup_transitions,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


REGISTERED_SMOKE_CONFIG = EmpiricalSmokeConfigV1()


def registered_warmup_mode_counts(seed: int) -> Tuple[int, ...]:
    """Audit the exact registered warmup RNG schedule without loading D1."""
    if type(seed) is not int or seed not in REGISTERED_SMOKE_CONFIG.seeds:
        raise EmpiricalSmokeError("warmup inventory seed is not registered")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_derive_seed(seed, "collection"))
    counts = [0] * len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)
    for _ in range(REGISTERED_SMOKE_CONFIG.warmup_transitions):
        mode = int(torch.randint(0, len(counts), (1,), generator=generator).item())
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode]
        # Consume the exact second draw used by _warmup_action.
        torch.randint(lower, upper + 1, (1,), generator=generator)
        counts[mode] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class _SynchronousEpisodeV1:
    """Immediate local handoff that makes an observation/result join explicit."""

    collection_seq: int
    observation: EmpiricalPolicyObservationV1
    action: EmpiricalActionV1
    result: EmpiricalStepResultV1

    def assert_exact(self, *, expected_seq: int, d1_binding_sha256: str) -> None:
        if self.collection_seq != expected_seq:
            raise EmpiricalSmokeError("local episode sequence changed before insertion")
        if self.observation.environment_binding_sha256 != d1_binding_sha256:
            raise EmpiricalSmokeError("observation belongs to another D1 binding")
        if self.result.audit.executed_mode_id != self.action.mode_id or (
            self.result.audit.executed_q_e4 != self.action.q_e4
        ):
            raise EmpiricalSmokeError(
                "D1 result/action mismatch; refusing an accidental cross-join"
            )
        if (
            not self.result.policy.terminated
            or self.result.policy.truncated
            or self.result.policy.reward is None
        ):
            raise EmpiricalSmokeError("D1 episode is not a reward-bearing terminal row")
        require_supported_action(self.action.mode_id, self.action.q_e4)


def _hash_state(value: Any) -> str:
    """Deterministic digest for nested checkpoint state, including tensors."""

    digest = hashlib.sha256()

    def emit(item: Any) -> None:
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            data = str(item).encode("ascii")
            digest.update(b"I" + len(data).to_bytes(8, "big") + data)
        elif type(item) is float:
            digest.update(b"F" + struct.pack(">d", item))
        elif type(item) is str:
            data = item.encode("utf-8")
            digest.update(b"S" + len(data).to_bytes(8, "big") + data)
        elif isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            emit(str(tensor.dtype))
            emit(tuple(tensor.shape))
            raw = tensor.numpy().tobytes(order="C")
            digest.update(b"T" + len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, Mapping):
            digest.update(b"M")
            keyed = sorted(item.items(), key=lambda pair: repr(pair[0]))
            emit(len(keyed))
            for key, child in keyed:
                emit(key)
                emit(child)
        elif isinstance(item, (tuple, list)):
            digest.update(b"Q" if isinstance(item, tuple) else b"L")
            emit(len(item))
            for child in item:
                emit(child)
        elif hasattr(item, "__dataclass_fields__"):
            digest.update(b"D")
            emit(type(item).__module__ + "." + type(item).__qualname__)
            for field in fields(item):
                if field.name != "checkpoint_sha256":
                    emit(field.name)
                    emit(getattr(item, field.name))
        else:
            raise EmpiricalSmokeError(
                f"checkpoint contains unsupported {type(item).__name__}"
            )

    emit(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EmpiricalSmokeCheckpointV1:
    """Complete between-episode state for deterministic smoke resumption."""

    config: EmpiricalSmokeConfigV1
    seed: int
    runner_binding_sha256: str
    collection_session_uuid: str
    actor_state: Mapping[str, Any]
    critics_state: Mapping[str, Any]
    actor_optimizer_state: Mapping[str, Any]
    critic_optimizer_state: Mapping[str, Any]
    init_rng_state: object
    collection_rng_state: Tensor
    replay_rng_state: Tensor
    actor_update_rng_state: Tensor
    environment_state: EmpiricalEnvironmentStateV1
    transition_history: Tuple[EmpiricalTerminalTransitionV1, ...]
    metrics: Tuple[EmpiricalTerminalUpdateMetricsV1, ...]
    collection_seq: int
    warmup_collected: int
    post_warmup_collected: int
    update_count: int
    support_violations: int
    checkpoint_sha256: str

    def _document(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "checkpoint_sha256"
        }

    def require_valid(self) -> None:
        if type(self.config) is not EmpiricalSmokeConfigV1:
            raise EmpiricalSmokeError("checkpoint config has a foreign type")
        self.config.__post_init__()
        if type(self.seed) is not int or self.seed not in self.config.seeds:
            raise EmpiricalSmokeError("checkpoint seed is not configured")
        if self.checkpoint_sha256 != _hash_state(self._document()):
            raise EmpiricalSmokeError("checkpoint digest mismatch")
        if self.collection_seq != len(self.transition_history):
            raise EmpiricalSmokeError("checkpoint history/sequence mismatch")
        if self.warmup_collected != min(
            self.collection_seq, self.config.warmup_transitions
        ):
            raise EmpiricalSmokeError("checkpoint warmup counter mismatch")
        expected_post = self.update_count * self.config.collect_per_update
        if self.post_warmup_collected != expected_post:
            raise EmpiricalSmokeError("checkpoint post-warmup counter mismatch")
        if self.collection_seq != self.warmup_collected + expected_post:
            raise EmpiricalSmokeError("checkpoint collection counters disagree")
        if self.update_count != len(self.metrics):
            raise EmpiricalSmokeError("checkpoint metric/update count mismatch")
        if not 0 <= self.update_count <= self.config.update_count:
            raise EmpiricalSmokeError("checkpoint update count is outside schedule")
        if self.support_violations != 0:
            raise EmpiricalSmokeError("checkpoint reports a support violation")


@dataclass(frozen=True, slots=True)
class EmpiricalSmokeSummaryV1:
    phase_label: str
    scope_disclosure: str
    seed: int
    configured_updates: int
    completed_updates: int
    transition_count: int
    warmup_transition_count: int
    post_warmup_transition_count: int
    replay_resident_count: int
    replay_eviction_count: int
    support_violation_count: int
    warmup_mode_counts: Tuple[int, ...]
    warmup_all_modes_exercised: bool
    target_reward_max_abs_diff: float
    reward_min: float
    reward_max: float
    reward_mean: float
    all_metrics_finite: bool
    actor_parameter_delta_from_init: float
    critic_1_parameter_delta_from_init: float
    critic_2_parameter_delta_from_init: float
    global_python_rng_unchanged: bool
    global_torch_rng_unchanged: bool
    cuda_initialized_by_runner: bool
    mechanical_acceptance_passed: bool
    config_sha256: str
    runner_binding_sha256: str
    d1_binding_sha256: str
    replay_binding_sha256: str
    trainer_config_sha256: str
    modeled_smoke_support_sha256: str
    pilot_utility_spec_sha256: str
    transition_history_sha256: str
    checkpoint_sha256: str
    replay_phase_label: str = REPLAY_PHASE_LABEL
    trainer_phase_label: str = TRAINER_PHASE_LABEL

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class EmpiricalThreeSeedSmokeReportV1:
    phase_label: str
    scope_disclosure: str
    config_sha256: str
    summaries: Tuple[EmpiricalSmokeSummaryV1, ...]

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "phase_label": self.phase_label,
            "scope_disclosure": self.scope_disclosure,
            "summaries": [item.to_canonical_dict() for item in self.summaries],
        }


@contextmanager
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


class EmpiricalContextualSmokeRunnerV1:
    """Trusted synchronous D1 -> terminal replay -> D2a mechanics runner."""

    def __init__(
        self,
        *,
        seed: int,
        config: EmpiricalSmokeConfigV1 = REGISTERED_SMOKE_CONFIG,
        project_root: Optional[Path] = None,
    ) -> None:
        if type(config) is not EmpiricalSmokeConfigV1:
            raise EmpiricalSmokeError("config must be exact EmpiricalSmokeConfigV1")
        config.__post_init__()
        if type(seed) is not int or seed not in config.seeds:
            raise EmpiricalSmokeError("seed must be one of config.seeds")
        self.config = config
        self.seed = seed
        self._global_python_rng_baseline = random.getstate()
        self._global_torch_rng_baseline = torch.get_rng_state().clone()
        self._cuda_initialized_at_entry = torch.cuda.is_initialized()
        self._stream_seeds = {
            name: _derive_seed(seed, name)
            for name in ("init", "collection", "replay", "actor_update", "environment")
        }
        self._init_rng = random.Random(self._stream_seeds["init"])
        actor_seed = self._init_rng.randrange(0, 1 << 63)
        critic_seed = self._init_rng.randrange(0, 1 << 63)
        self._collection_rng = torch.Generator(device="cpu")
        self._collection_rng.manual_seed(self._stream_seeds["collection"])
        self._replay_rng = torch.Generator(device="cpu")
        self._replay_rng.manual_seed(self._stream_seeds["replay"])
        self._actor_update_rng = torch.Generator(device="cpu")
        self._actor_update_rng.manual_seed(self._stream_seeds["actor_update"])

        model_config = HybridSacModelConfig(
            dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
        )
        self.actor = build_actor(model_config, seed=actor_seed)
        self.critics = build_twin_critics(model_config, seed=critic_seed)
        self._initial_actor_state = copy.deepcopy(self.actor.state_dict())
        self._initial_critic_1_state = copy.deepcopy(
            self.critics.critic_1.state_dict()
        )
        self._initial_critic_2_state = copy.deepcopy(
            self.critics.critic_2.state_dict()
        )
        self.environment = EmpiricalOneStepEnvironmentV1.load_registered(
            seed=self._stream_seeds["environment"], project_root=project_root
        )
        self.replay = EmpiricalTerminalReplayV1(config.replay_capacity)
        self.trainer_config = EmpiricalTerminalTrainerConfigV1(
            batch_size=config.batch_size
        )
        self.trainer: Optional[EmpiricalTerminalHybridSacTrainerV1] = None
        self._history: list[EmpiricalTerminalTransitionV1] = []
        self._metrics: list[EmpiricalTerminalUpdateMetricsV1] = []
        self._collection_seq = 0
        self._warmup_collected = 0
        self._post_warmup_collected = 0
        self._support_violations = 0
        self._closed = False
        self._d1_binding_sha256 = self.environment.binding.canonical_sha256()
        self._runner_binding_sha256 = canonical_sha256(self._binding_document())
        self._collection_session_uuid = str(
            uuid.uuid5(
                _COLLECTION_NAMESPACE,
                f"{self._runner_binding_sha256}:{self.seed}",
            )
        )
        self._assert_process_rng_isolation()

    def _binding_document(self) -> Dict[str, Any]:
        return {
            "config_sha256": self.config.canonical_sha256(),
            "d1_binding": self.environment.binding.to_canonical_dict(),
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "phase_label": PHASE_LABEL,
            "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "replay_phase_label": REPLAY_PHASE_LABEL,
            "rng_seed_derivation_schema": SEED_DERIVATION_SCHEMA,
            "runner_schema": RUNNER_SCHEMA,
            "seed": self.seed,
            "trainer_config_sha256": self.trainer_config.canonical_sha256(),
            "trainer_phase_label": TRAINER_PHASE_LABEL,
        }

    @property
    def transition_history(self) -> Tuple[EmpiricalTerminalTransitionV1, ...]:
        return tuple(self._history)

    @property
    def metrics(self) -> Tuple[EmpiricalTerminalUpdateMetricsV1, ...]:
        return tuple(self._metrics)

    @property
    def completed_updates(self) -> int:
        return len(self._metrics)

    def _require_open(self) -> None:
        if self._closed:
            raise EmpiricalSmokeError("runner is closed")

    def _assert_process_rng_isolation(self) -> None:
        if random.getstate() != self._global_python_rng_baseline:
            raise EmpiricalSmokeError("runner advanced module-global Python RNG")
        if not torch.equal(torch.get_rng_state(), self._global_torch_rng_baseline):
            raise EmpiricalSmokeError("runner advanced Torch's global CPU RNG")
        if not self._cuda_initialized_at_entry and torch.cuda.is_initialized():
            raise EmpiricalSmokeError("runner initialized CUDA")

    def _warmup_action(self) -> EmpiricalActionV1:
        mode = int(
            torch.randint(
                0, len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds), (1,),
                generator=self._collection_rng,
            ).item()
        )
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode]
        q_e4 = int(
            torch.randint(
                lower, upper + 1, (1,), generator=self._collection_rng
            ).item()
        )
        return require_supported_action(mode, q_e4)

    def _actor_action(self, observation: EmpiricalPolicyObservationV1) -> EmpiricalActionV1:
        state = torch.tensor([observation.values], dtype=torch.float32)
        with torch.no_grad():
            _log_probs, probabilities = self.actor.mode_log_probs(state)
            mode = int(
                torch.multinomial(
                    probabilities[0], 1, generator=self._collection_rng
                ).item()
            )
            conditional = self.actor.sample_all_modes(
                state, generator=self._collection_rng
            )
            q_e4 = int(conditional.q_e4[0, mode].item())
        return require_supported_action(mode, q_e4)

    def _ensure_trainer(self) -> EmpiricalTerminalHybridSacTrainerV1:
        if self.trainer is None:
            if self.replay.binding is None:
                raise EmpiricalSmokeError("cannot bind trainer before first transition")
            self.trainer = EmpiricalTerminalHybridSacTrainerV1(
                self.actor,
                self.critics,
                self.trainer_config,
                expected_binding=self.replay.binding,
                actor_generator=self._actor_update_rng,
            )
        return self.trainer

    def _collect_one(self, *, warmup: bool) -> None:
        seq = self._collection_seq
        if len(self._history) != seq or self.replay.accepted_count != seq:
            raise EmpiricalSmokeError("collection sequence/replay history drift")
        observation = self.environment.reset()
        action = self._warmup_action() if warmup else self._actor_action(observation)
        result = self.environment.step(action)
        episode = _SynchronousEpisodeV1(seq, observation, action, result)
        episode.assert_exact(
            expected_seq=seq, d1_binding_sha256=self._d1_binding_sha256
        )
        transition = EmpiricalTerminalTransitionV1.from_d1(
            collection_session_uuid=self._collection_session_uuid,
            collection_seq=seq,
            observation=episode.observation,
            action=episode.action,
            result=episode.result,
            d1_binding=self.environment.binding,
        )
        # Identity assertions are intentionally adjacent to construction and
        # insertion: no later reset or mutable staging table can intervene.
        if (
            transition.collection_seq != seq
            or transition.observation is not observation
            or transition.action is not action
            or transition.result is not result
        ):
            raise EmpiricalSmokeError("immediate episode tuple identity was lost")
        self.replay.insert(transition)
        self._history.append(transition)
        self._collection_seq += 1
        if warmup:
            self._warmup_collected += 1
        else:
            self._post_warmup_collected += 1
        if self.replay.evicted_count != 0:
            raise EmpiricalSmokeError("smoke replay unexpectedly evicted a transition")

    def run_until_updates(self, target_updates: int) -> EmpiricalSmokeSummaryV1:
        """Advance to an absolute update count and return mechanics diagnostics."""
        self._require_open()
        if (
            type(target_updates) is not int
            or not self.completed_updates <= target_updates <= self.config.update_count
        ):
            raise EmpiricalSmokeError("target update count is invalid or goes backwards")
        with _one_cpu_thread():
            while self._warmup_collected < self.config.warmup_transitions:
                self._collect_one(warmup=True)
            self._ensure_trainer()
            while self.completed_updates < target_updates:
                for _ in range(self.config.collect_per_update):
                    self._collect_one(warmup=False)
                batch = self.replay.sample(
                    self.config.batch_size, generator=self._replay_rng
                )
                metric = self._ensure_trainer().update_once(batch)
                metric.assert_finite()
                if metric.target_reward_max_abs_diff != 0.0:
                    raise EmpiricalSmokeError("terminal target differs from reward")
                self._metrics.append(metric)
                if self.trainer is None or self.trainer.update_count != len(self._metrics):
                    raise EmpiricalSmokeError("trainer/runner update counter drift")
        self._assert_process_rng_isolation()
        return self.summary()

    def run(self) -> EmpiricalSmokeSummaryV1:
        return self.run_until_updates(self.config.update_count)

    def checkpoint(self) -> EmpiricalSmokeCheckpointV1:
        self._require_open()
        trainer = self._ensure_trainer()
        document = dict(
            config=self.config,
            seed=self.seed,
            runner_binding_sha256=self._runner_binding_sha256,
            collection_session_uuid=self._collection_session_uuid,
            actor_state=copy.deepcopy(self.actor.state_dict()),
            critics_state=copy.deepcopy(self.critics.state_dict()),
            actor_optimizer_state=copy.deepcopy(trainer.actor_optimizer.state_dict()),
            critic_optimizer_state=copy.deepcopy(trainer.critic_optimizer.state_dict()),
            init_rng_state=copy.deepcopy(self._init_rng.getstate()),
            collection_rng_state=self._collection_rng.get_state().clone(),
            replay_rng_state=self._replay_rng.get_state().clone(),
            actor_update_rng_state=self._actor_update_rng.get_state().clone(),
            environment_state=copy.deepcopy(self.environment.state_dict()),
            transition_history=tuple(self._history),
            metrics=tuple(self._metrics),
            collection_seq=self._collection_seq,
            warmup_collected=self._warmup_collected,
            post_warmup_collected=self._post_warmup_collected,
            update_count=self.completed_updates,
            support_violations=self._support_violations,
        )
        checkpoint = EmpiricalSmokeCheckpointV1(
            **document, checkpoint_sha256=_hash_state(document)
        )
        checkpoint.require_valid()
        return checkpoint

    def load_checkpoint(self, checkpoint: EmpiricalSmokeCheckpointV1) -> None:
        """Restore a complete checkpoint; replay is rebuilt from full history."""
        self._require_open()
        if type(checkpoint) is not EmpiricalSmokeCheckpointV1:
            raise EmpiricalSmokeError("checkpoint has a foreign type")
        checkpoint.require_valid()
        if checkpoint.config != self.config or checkpoint.seed != self.seed:
            raise EmpiricalSmokeError("checkpoint schedule/seed mismatch")
        if checkpoint.runner_binding_sha256 != self._runner_binding_sha256:
            raise EmpiricalSmokeError("checkpoint runner binding mismatch")
        if checkpoint.collection_session_uuid != self._collection_session_uuid:
            raise EmpiricalSmokeError("checkpoint collection session mismatch")

        rebuilt = EmpiricalTerminalReplayV1(self.config.replay_capacity)
        for expected_seq, transition in enumerate(checkpoint.transition_history):
            transition.revalidate()
            if (
                transition.collection_seq != expected_seq
                or transition.collection_session_uuid
                != self._collection_session_uuid
            ):
                raise EmpiricalSmokeError("checkpoint transition sequence drift")
            rebuilt.insert(transition)
        if rebuilt.evicted_count != 0 or len(rebuilt) != checkpoint.collection_seq:
            raise EmpiricalSmokeError("checkpoint history cannot reconstruct replay")
        if rebuilt.binding is None:
            raise EmpiricalSmokeError("checkpoint history is unexpectedly empty")

        # The digest and replay reconstruction above validate the offered
        # object before mutable runner state is replaced.
        self.environment.load_state_dict(checkpoint.environment_state)
        self.actor.load_state_dict(copy.deepcopy(checkpoint.actor_state), strict=True)
        self.critics.load_state_dict(
            copy.deepcopy(checkpoint.critics_state), strict=True
        )
        self._init_rng.setstate(copy.deepcopy(checkpoint.init_rng_state))
        self._collection_rng.set_state(checkpoint.collection_rng_state.clone())
        self._replay_rng.set_state(checkpoint.replay_rng_state.clone())
        self._actor_update_rng.set_state(checkpoint.actor_update_rng_state.clone())
        restored_trainer = EmpiricalTerminalHybridSacTrainerV1(
            self.actor,
            self.critics,
            self.trainer_config,
            expected_binding=rebuilt.binding,
            actor_generator=self._actor_update_rng,
        )
        restored_trainer.actor_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.actor_optimizer_state)
        )
        restored_trainer.critic_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.critic_optimizer_state)
        )
        restored_trainer.update_count = checkpoint.update_count
        self.replay = rebuilt
        self.trainer = restored_trainer
        self._history = list(checkpoint.transition_history)
        self._metrics = list(checkpoint.metrics)
        self._collection_seq = checkpoint.collection_seq
        self._warmup_collected = checkpoint.warmup_collected
        self._post_warmup_collected = checkpoint.post_warmup_collected
        self._support_violations = checkpoint.support_violations
        self._assert_process_rng_isolation()

    @staticmethod
    def _parameter_delta_norm(
        module: torch.nn.Module, initial: Mapping[str, Any]
    ) -> float:
        total = 0.0
        current = module.state_dict()
        if tuple(current) != tuple(initial):
            raise EmpiricalSmokeError("model state inventory drift")
        for name, value in current.items():
            reference = initial[name]
            if value.is_floating_point():
                delta = value.detach().to(torch.float64) - reference.to(torch.float64)
                total += float(torch.sum(delta * delta))
            elif not torch.equal(value, reference):
                raise EmpiricalSmokeError(f"non-floating model buffer {name} drift")
        return math.sqrt(total)

    def summary(self) -> EmpiricalSmokeSummaryV1:
        self._require_open()
        if not self._history:
            raise EmpiricalSmokeError("summary requires collected transitions")
        if self.trainer is None or self.replay.binding is None:
            raise EmpiricalSmokeError("summary requires a bound trainer and replay")
        rewards = tuple(transition.reward for transition in self._history)
        all_finite = all(
            all(
                not isinstance(value, (int, float)) or math.isfinite(float(value))
                for value in metric.as_dict().values()
            )
            for metric in self._metrics
        )
        target_diff = max(
            (metric.target_reward_max_abs_diff for metric in self._metrics),
            default=0.0,
        )
        checkpoint = self.checkpoint()
        warmup_counts = [0] * len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)
        for transition in self._history[: self._warmup_collected]:
            warmup_counts[transition.action.mode_id] += 1
        all_modes = all(count > 0 for count in warmup_counts)
        if self.config == REGISTERED_SMOKE_CONFIG and not all_modes:
            raise EmpiricalSmokeError("registered warmup did not exercise all 12 modes")
        actor_delta = self._parameter_delta_norm(
            self.actor, self._initial_actor_state
        )
        critic_1_delta = self._parameter_delta_norm(
            self.critics.critic_1, self._initial_critic_1_state
        )
        critic_2_delta = self._parameter_delta_norm(
            self.critics.critic_2, self._initial_critic_2_state
        )
        python_rng_unchanged = random.getstate() == self._global_python_rng_baseline
        torch_rng_unchanged = torch.equal(
            torch.get_rng_state(), self._global_torch_rng_baseline
        )
        cuda_initialized_by_runner = (
            not self._cuda_initialized_at_entry and torch.cuda.is_initialized()
        )
        completed_schedule = self.completed_updates == self.config.update_count
        mechanical_acceptance = (
            completed_schedule
            and self.replay.evicted_count == 0
            and self._support_violations == 0
            and target_diff == 0.0
            and min(rewards) < max(rewards)
            and all_finite
            and actor_delta > 0.0
            and critic_1_delta > 0.0
            and critic_2_delta > 0.0
            and python_rng_unchanged
            and torch_rng_unchanged
            and not cuda_initialized_by_runner
            and (self.config != REGISTERED_SMOKE_CONFIG or all_modes)
        )
        if completed_schedule and not mechanical_acceptance:
            raise EmpiricalSmokeError("completed smoke failed mechanical acceptance")
        return EmpiricalSmokeSummaryV1(
            phase_label=PHASE_LABEL,
            scope_disclosure=SCOPE_DISCLOSURE,
            seed=self.seed,
            configured_updates=self.config.update_count,
            completed_updates=self.completed_updates,
            transition_count=len(self._history),
            warmup_transition_count=self._warmup_collected,
            post_warmup_transition_count=self._post_warmup_collected,
            replay_resident_count=len(self.replay),
            replay_eviction_count=self.replay.evicted_count,
            support_violation_count=self._support_violations,
            warmup_mode_counts=tuple(warmup_counts),
            warmup_all_modes_exercised=all_modes,
            target_reward_max_abs_diff=target_diff,
            reward_min=min(rewards),
            reward_max=max(rewards),
            reward_mean=sum(rewards) / len(rewards),
            all_metrics_finite=all_finite,
            actor_parameter_delta_from_init=actor_delta,
            critic_1_parameter_delta_from_init=critic_1_delta,
            critic_2_parameter_delta_from_init=critic_2_delta,
            global_python_rng_unchanged=python_rng_unchanged,
            global_torch_rng_unchanged=torch_rng_unchanged,
            cuda_initialized_by_runner=cuda_initialized_by_runner,
            mechanical_acceptance_passed=mechanical_acceptance,
            config_sha256=self.config.canonical_sha256(),
            runner_binding_sha256=self._runner_binding_sha256,
            d1_binding_sha256=self._d1_binding_sha256,
            replay_binding_sha256=self.replay.binding.canonical_sha256(),
            trainer_config_sha256=self.trainer_config.canonical_sha256(),
            modeled_smoke_support_sha256=MODELED_SMOKE_SUPPORT_SHA256,
            pilot_utility_spec_sha256=PILOT_UTILITY_SPEC_SHA256,
            transition_history_sha256=canonical_sha256(
                [transition.canonical_sha256() for transition in self._history]
            ),
            checkpoint_sha256=checkpoint.checkpoint_sha256,
        )

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True

    def __enter__(self) -> "EmpiricalContextualSmokeRunnerV1":
        self._require_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def run_registered_three_seed_smoke(
    *, project_root: Optional[Path] = None
) -> EmpiricalThreeSeedSmokeReportV1:
    """Run the exact registered three-seed mechanical schedule serially."""
    summaries = []
    for seed in REGISTERED_SMOKE_CONFIG.seeds:
        with EmpiricalContextualSmokeRunnerV1(
            seed=seed,
            config=REGISTERED_SMOKE_CONFIG,
            project_root=project_root,
        ) as runner:
            summaries.append(runner.run())
    return EmpiricalThreeSeedSmokeReportV1(
        phase_label=PHASE_LABEL,
        scope_disclosure=SCOPE_DISCLOSURE,
        config_sha256=REGISTERED_SMOKE_CONFIG.canonical_sha256(),
        summaries=tuple(summaries),
    )

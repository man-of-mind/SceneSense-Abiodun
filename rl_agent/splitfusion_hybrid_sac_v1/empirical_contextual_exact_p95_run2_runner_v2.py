"""Isolated train-only campaign runner for exact-P95 Run 2 v2.

This module is intentionally independent of the historical D1 smoke and
baseline runners.  It collects synchronous terminal D1 episodes from only the
registered train partition, immediately adapts each authenticated D1 row to
the exact Run-2-v2 reward contract, and trains only through the authenticated
v2 replay and terminal trainer.

No fit-validation panel or evaluator is imported here.  Evaluation and
artifact publication belong to separate phases.
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
from itertools import chain
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
    EmpiricalPolicyObservationV1,
    EmpiricalStepResultV1,
)
from .empirical_contextual_exact_p95_run2_replay_v2 import (
    EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
    GATE_V2_CANONICAL_CONTENT_SHA256,
    GATE_V2_CONTEXT_ORACLES_SHA256,
    GATE_V2_DECISION_SHA256,
    GATE_V2_IMPLEMENTATION_SHA256,
    GATE_V2_PROFILE_SUMMARY_SHA256,
    GATE_V2_REPORT_SHA256,
    GATE_V2_SUMMARY_SHA256,
    GATE_V2_TEST_SHA256,
    PREREG_V2_FILE_SHA256,
    PREREG_V2_HASH_MANIFEST_SHA256,
    PREREG_V2_REPORT_SHA256,
    TRAIN_V2_CANONICAL_CONTENT_SHA256,
    TRAIN_V2_DECISION_SHA256,
    TRAIN_V2_IMPLEMENTATION_SHA256,
    TRAIN_V2_ORACLES_SHA256,
    TRAIN_V2_REPORT_SHA256,
    TRAIN_V2_SUMMARY_SHA256,
    ExactP95Run2ReplayBindingV2,
    ExactP95Run2ReplayV2,
    ExactP95Run2RewardBindingV2,
    ExactP95Run2ShapedTransitionV2,
)
from .empirical_contextual_exact_p95_run2_terminal_trainer_v2 import (
    PHASE_LABEL as TRAINER_PHASE_LABEL,
    ExactP95Run2TerminalHybridSacTrainerV2,
    ExactP95Run2TerminalTrainerConfigV2,
    ExactP95Run2TerminalUpdateMetricsV2,
)
from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
)
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalEnvironmentStateV1,
    PartitionedEmpiricalOneStepEnvironmentV1,
)
from .empirical_contextual_terminal_replay import EmpiricalTerminalTransitionV1
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2",
    "EmpiricalContextualExactP95Run2RunnerV2",
    "ExactP95Run2CheckpointV2",
    "ExactP95Run2ConfigV2",
    "ExactP95Run2RunnerConfigV2",
    "ExactP95Run2RunnerErrorV2",
    "ExactP95Run2RunnerSummaryV2",
    "ExactP95Run2RunnerV2",
    "PHASE_LABEL",
    "REGISTERED_EXACT_P95_RUN2_CONFIG_V2",
    "REGISTERED_RUN2_V2_CONFIG",
    "RUN2_V2_CHECKPOINT_INTERVAL_UPDATES",
    "RUNNER_SCHEMA",
    "registered_warmup_mode_counts_v2",
]


PHASE_LABEL = "EXACT_P95_RUN2_V2_REGISTERED_TRAIN_PARTITION_TRAINING_ONLY"
RUNNER_SCHEMA = "splitfusion.empirical_contextual_exact_p95_run2_runner.v2"
CHECKPOINT_SCHEMA = "splitfusion.empirical_contextual_exact_p95_run2_checkpoint.v2"
SUMMARY_SCHEMA = "splitfusion.empirical_contextual_exact_p95_run2_summary.v2"
SEED_DERIVATION_SCHEMA = "splitfusion.exact_p95_run2_runner_rng.v2"
EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2 = 500
RUN2_V2_CHECKPOINT_INTERVAL_UPDATES = (
    EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2
)
_COLLECTION_NAMESPACE = uuid.UUID("c26262df-767c-5fb4-b8d0-7b3f8362ea53")


class ExactP95Run2RunnerErrorV2(RuntimeError):
    """The isolated Run-2-v2 training runner failed closed."""


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ExactP95Run2RunnerErrorV2(
            f"{name} must be an exact positive integer"
        )
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactP95Run2RunnerErrorV2(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _derive_seed(master_seed: int, stream: str) -> int:
    payload = f"{SEED_DERIVATION_SCHEMA}:{master_seed}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_state(value: Any) -> str:
    """Digest nested checkpoint state without pickle-version dependence."""

    digest = hashlib.sha256()

    def emit(item: Any) -> None:
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            raw = str(item).encode("ascii")
            digest.update(b"I" + len(raw).to_bytes(8, "big") + raw)
        elif type(item) is float:
            digest.update(b"F" + struct.pack(">d", item))
        elif type(item) is str:
            raw = item.encode("utf-8")
            digest.update(b"S" + len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            emit(str(tensor.dtype))
            emit(tuple(tensor.shape))
            raw = tensor.numpy().tobytes(order="C")
            digest.update(b"T" + len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, Mapping):
            digest.update(b"M")
            entries = sorted(item.items(), key=lambda pair: repr(pair[0]))
            emit(len(entries))
            for key, child in entries:
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
            raise ExactP95Run2RunnerErrorV2(
                f"checkpoint contains unsupported {type(item).__name__}"
            )

    emit(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExactP95Run2RunnerConfigV2:
    """Frozen Run-2-v2 schedule; smaller schedules are test-only."""

    seeds: Tuple[int, ...] = (17, 29, 43)
    warmup_transitions: int = 1024
    batch_size: int = 256
    collect_per_update: int = 4
    update_count: int = 5000
    replay_capacity: int = 32768
    cpu_threads: int = 1
    scope: str = PHASE_LABEL

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ExactP95Run2RunnerErrorV2(
                "seeds must be unique non-negative exact integers"
            )
        for name in (
            "warmup_transitions",
            "batch_size",
            "collect_per_update",
            "update_count",
            "replay_capacity",
            "cpu_threads",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.cpu_threads != 1:
            raise ExactP95Run2RunnerErrorV2("Run-2 v2 is fixed to one CPU thread")
        if self.warmup_transitions < self.batch_size:
            raise ExactP95Run2RunnerErrorV2("warmup must contain a full batch")
        if self.total_transitions > self.replay_capacity:
            raise ExactP95Run2RunnerErrorV2(
                "Run-2 v2 must retain all transitions without eviction"
            )
        if self.scope != PHASE_LABEL:
            raise ExactP95Run2RunnerErrorV2("Run-2-v2 phase label drift")

    @property
    def total_transitions(self) -> int:
        return self.warmup_transitions + self.collect_per_update * self.update_count

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "collect_per_update": self.collect_per_update,
            "cpu_threads": self.cpu_threads,
            "record": "exact_p95_run2_runner_config_v2",
            "replay_capacity": self.replay_capacity,
            "scope": self.scope,
            "seeds": list(self.seeds),
            "update_count": self.update_count,
            "warmup_transitions": self.warmup_transitions,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


REGISTERED_EXACT_P95_RUN2_CONFIG_V2 = ExactP95Run2RunnerConfigV2()
REGISTERED_RUN2_V2_CONFIG = REGISTERED_EXACT_P95_RUN2_CONFIG_V2
ExactP95Run2ConfigV2 = ExactP95Run2RunnerConfigV2


def registered_warmup_mode_counts_v2(seed: int) -> Tuple[int, ...]:
    """Audit the registered warmup draws without loading empirical evidence."""
    if type(seed) is not int or seed not in REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds:
        raise ExactP95Run2RunnerErrorV2("warmup inventory seed is not registered")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_derive_seed(seed, "collection"))
    counts = [0] * len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)
    for _ in range(REGISTERED_EXACT_P95_RUN2_CONFIG_V2.warmup_transitions):
        mode = int(torch.randint(0, len(counts), (1,), generator=generator).item())
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode]
        torch.randint(lower, upper + 1, (1,), generator=generator)
        counts[mode] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class _SynchronousEpisodeV2:
    collection_seq: int
    observation: EmpiricalPolicyObservationV1
    action: EmpiricalActionV1
    result: EmpiricalStepResultV1

    def require_valid(self, *, expected_seq: int, d1_binding_sha256: str) -> None:
        if self.collection_seq != expected_seq:
            raise ExactP95Run2RunnerErrorV2("episode sequence drift")
        if self.observation.environment_binding_sha256 != d1_binding_sha256:
            raise ExactP95Run2RunnerErrorV2("observation D1 binding drift")
        if (
            self.result.audit.executed_mode_id != self.action.mode_id
            or self.result.audit.executed_q_e4 != self.action.q_e4
        ):
            raise ExactP95Run2RunnerErrorV2("synchronous action/result join drift")
        if (
            not self.result.policy.terminated
            or self.result.policy.truncated
            or self.result.policy.reward is None
        ):
            raise ExactP95Run2RunnerErrorV2("episode is not reward-bearing terminal")
        require_supported_action(self.action.mode_id, self.action.q_e4)


@dataclass(frozen=True, slots=True)
class ExactP95Run2CheckpointV2:
    """Complete between-episode state for exact deterministic resumption."""

    config: ExactP95Run2RunnerConfigV2
    seed: int
    runner_binding_document: Mapping[str, Any]
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
    environment_state: PartitionedEmpiricalEnvironmentStateV1
    d1_transition_history: Tuple[EmpiricalTerminalTransitionV1, ...]
    run2_v2_transition_history: Tuple[ExactP95Run2ShapedTransitionV2, ...]
    metrics: Tuple[ExactP95Run2TerminalUpdateMetricsV2, ...]
    replay_binding: Optional[ExactP95Run2ReplayBindingV2]
    collection_seq: int
    warmup_collected: int
    post_warmup_collected: int
    update_count: int
    replay_accepted_count: int
    replay_evicted_count: int
    support_violation_count: int
    trainer_initialized: bool
    schema: str
    checkpoint_sha256: str

    def _document(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "checkpoint_sha256"
        }

    def require_valid(self) -> None:
        if type(self) is not ExactP95Run2CheckpointV2:
            raise ExactP95Run2RunnerErrorV2("checkpoint has a foreign type")
        if type(self.config) is not ExactP95Run2RunnerConfigV2:
            raise ExactP95Run2RunnerErrorV2("checkpoint config has a foreign type")
        self.config.__post_init__()
        if self.schema != CHECKPOINT_SCHEMA:
            raise ExactP95Run2RunnerErrorV2("checkpoint schema drift")
        if type(self.seed) is not int or self.seed not in self.config.seeds:
            raise ExactP95Run2RunnerErrorV2("checkpoint seed is not configured")
        _require_sha256(self.runner_binding_sha256, "runner_binding_sha256")
        if (
            not isinstance(self.runner_binding_document, Mapping)
            or canonical_sha256(dict(self.runner_binding_document))
            != self.runner_binding_sha256
            or self.runner_binding_document.get("runner_schema") != RUNNER_SCHEMA
            or self.runner_binding_document.get("phase_label") != PHASE_LABEL
            or self.runner_binding_document.get("sampling_split") != TRAIN_SPLIT
            or self.runner_binding_document.get("reward_spec_v2_sha256")
            != EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256
            or self.runner_binding_document.get("trainer_phase_label")
            != TRAINER_PHASE_LABEL
        ):
            raise ExactP95Run2RunnerErrorV2("checkpoint runner binding drift")
        try:
            parsed_uuid = uuid.UUID(self.collection_session_uuid)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ExactP95Run2RunnerErrorV2("checkpoint UUID is malformed") from exc
        if str(parsed_uuid) != self.collection_session_uuid:
            raise ExactP95Run2RunnerErrorV2("checkpoint UUID is not canonical")
        if self.checkpoint_sha256 != _hash_state(self._document()):
            raise ExactP95Run2RunnerErrorV2("checkpoint digest mismatch")
        if self.collection_seq != len(self.d1_transition_history) or (
            self.collection_seq != len(self.run2_v2_transition_history)
        ):
            raise ExactP95Run2RunnerErrorV2("checkpoint history/sequence mismatch")
        if self.warmup_collected != min(
            self.collection_seq, self.config.warmup_transitions
        ):
            raise ExactP95Run2RunnerErrorV2("checkpoint warmup counter mismatch")
        expected_post = self.update_count * self.config.collect_per_update
        if self.post_warmup_collected != expected_post:
            raise ExactP95Run2RunnerErrorV2("checkpoint post-warmup counter mismatch")
        if self.collection_seq != self.warmup_collected + expected_post:
            raise ExactP95Run2RunnerErrorV2("checkpoint collection counters disagree")
        if self.update_count != len(self.metrics) or not (
            0 <= self.update_count <= self.config.update_count
        ):
            raise ExactP95Run2RunnerErrorV2("checkpoint update counter mismatch")
        if self.replay_accepted_count != self.collection_seq:
            raise ExactP95Run2RunnerErrorV2("checkpoint replay accepted-count drift")
        if self.replay_evicted_count != 0 or self.support_violation_count != 0:
            raise ExactP95Run2RunnerErrorV2("checkpoint records a hard-gate violation")
        if self.trainer_initialized is not bool(self.collection_seq):
            raise ExactP95Run2RunnerErrorV2("checkpoint trainer initialization drift")
        if self.environment_state.d1_state.reset_count != self.collection_seq:
            raise ExactP95Run2RunnerErrorV2("checkpoint environment/reset drift")
        if not all(isinstance(item, Mapping) for item in (
            self.actor_state,
            self.critics_state,
            self.actor_optimizer_state,
            self.critic_optimizer_state,
        )):
            raise ExactP95Run2RunnerErrorV2("checkpoint state mapping drift")
        for state_name, state in (
            ("collection_rng_state", self.collection_rng_state),
            ("replay_rng_state", self.replay_rng_state),
            ("actor_update_rng_state", self.actor_update_rng_state),
        ):
            if (
                type(state) is not Tensor
                or state.device.type != "cpu"
                or state.dtype is not torch.uint8
                or state.ndim != 1
            ):
                raise ExactP95Run2RunnerErrorV2(f"{state_name} is malformed")
            probe = torch.Generator(device="cpu")
            try:
                probe.set_state(state.clone())
            except RuntimeError as exc:
                raise ExactP95Run2RunnerErrorV2(
                    f"{state_name} cannot restore a CPU generator"
                ) from exc
        python_probe = random.Random()
        try:
            python_probe.setstate(self.init_rng_state)
        except (TypeError, ValueError) as exc:
            raise ExactP95Run2RunnerErrorV2("init RNG state is malformed") from exc
        for index, (source, shaped) in enumerate(
            zip(self.d1_transition_history, self.run2_v2_transition_history)
        ):
            if type(source) is not EmpiricalTerminalTransitionV1:
                raise ExactP95Run2RunnerErrorV2("D1 checkpoint history type drift")
            if type(shaped) is not ExactP95Run2ShapedTransitionV2:
                raise ExactP95Run2RunnerErrorV2("v2 checkpoint history type drift")
            source.revalidate()
            shaped.revalidate()
            if (
                source.collection_seq != index
                or source.collection_session_uuid != self.collection_session_uuid
                or shaped.source_d1_transition.canonical_sha256()
                != source.canonical_sha256()
                or shaped.source_d1_reward64 != source.reward
            ):
                raise ExactP95Run2RunnerErrorV2("checkpoint reward-history drift")
        for metric in self.metrics:
            if type(metric) is not ExactP95Run2TerminalUpdateMetricsV2:
                raise ExactP95Run2RunnerErrorV2("checkpoint metric type drift")
            metric.assert_finite()
        if self.collection_seq == 0:
            if self.replay_binding is not None:
                raise ExactP95Run2RunnerErrorV2("empty checkpoint has replay binding")
        else:
            if type(self.replay_binding) is not ExactP95Run2ReplayBindingV2:
                raise ExactP95Run2RunnerErrorV2("checkpoint replay binding missing")
            self.replay_binding.require_valid()
            expected = ExactP95Run2ReplayBindingV2.from_transition(
                self.run2_v2_transition_history[0]
            )
            expected.assert_matches(self.replay_binding)


@dataclass(frozen=True, slots=True)
class ExactP95Run2RunnerSummaryV2:
    schema: str
    phase_label: str
    seed: int
    sampling_split: str
    configured_updates: int
    completed_updates: int
    transition_count: int
    warmup_transition_count: int
    post_warmup_transition_count: int
    replay_resident_count: int
    replay_eviction_count: int
    support_violation_count: int
    raw_d1_reward_min: Optional[float]
    raw_d1_reward_max: Optional[float]
    raw_d1_reward_mean: Optional[float]
    shaped_reward64_min: Optional[float]
    shaped_reward64_max: Optional[float]
    shaped_reward64_mean: Optional[float]
    emitted_reward_float32_min: Optional[float]
    emitted_reward_float32_max: Optional[float]
    emitted_reward_float32_mean: Optional[float]
    target_reward_signed_bit_mismatch_count: int
    all_metrics_finite: bool
    actor_parameter_delta_from_init: float
    critic_1_parameter_delta_from_init: float
    critic_2_parameter_delta_from_init: float
    global_python_rng_unchanged: bool
    global_torch_rng_unchanged: bool
    cuda_initialized_by_runner: bool
    completed_training_hard_gates_passed: bool
    config_sha256: str
    runner_binding_sha256: str
    d1_binding_sha256: str
    fit_partition_sha256: str
    sampling_contract_sha256: str
    replay_binding_sha256: Optional[str]
    reward_binding_sha256: Optional[str]
    trainer_config_sha256: str
    modeled_smoke_support_sha256: str
    pilot_utility_spec_sha256: str
    reward_spec_v2_sha256: str
    d1_transition_history_sha256: str
    run2_v2_transition_history_sha256: str
    metrics_history_sha256: str
    checkpoint_sha256: str
    trainer_phase_label: str = TRAINER_PHASE_LABEL

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


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


def _mean(values: Tuple[float, ...]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


class ExactP95Run2RunnerV2:
    """Standalone synchronous D1-train -> Run-2-v2 replay/trainer runner."""

    def __init__(
        self,
        *,
        seed: int,
        config: ExactP95Run2RunnerConfigV2 = (
            REGISTERED_EXACT_P95_RUN2_CONFIG_V2
        ),
        project_root: Optional[Path] = None,
    ) -> None:
        if type(config) is not ExactP95Run2RunnerConfigV2:
            raise ExactP95Run2RunnerErrorV2(
                "config must be exact ExactP95Run2RunnerConfigV2"
            )
        config.__post_init__()
        if type(seed) is not int or seed not in config.seeds:
            raise ExactP95Run2RunnerErrorV2("seed must be one of config.seeds")
        self.config = config
        self.seed = seed
        self._project_root = (
            None if project_root is None else Path(project_root).resolve(strict=True)
        )
        self._global_python_rng_baseline = random.getstate()
        self._global_torch_rng_baseline = torch.get_rng_state().clone()
        self._cuda_initialized_at_entry = torch.cuda.is_initialized()
        self._stream_seeds = {
            name: _derive_seed(seed, name)
            for name in (
                "init",
                "collection",
                "replay",
                "actor_update",
                "environment",
            )
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

        self.environment = PartitionedEmpiricalOneStepEnvironmentV1.load_registered(
            seed=self._stream_seeds["environment"],
            split=TRAIN_SPLIT,
            project_root=self._project_root,
        )
        self.replay = ExactP95Run2ReplayV2(config.replay_capacity)
        self.trainer_config = ExactP95Run2TerminalTrainerConfigV2(
            batch_size=config.batch_size,
            hyperparameter_status=(
                "RUN2_V2_REGISTERED_PRETRAINING_GATE_GO_TRAINING"
            ),
        )
        self.trainer: Optional[ExactP95Run2TerminalHybridSacTrainerV2] = None
        self._pending_actor_optimizer_state, (
            self._pending_critic_optimizer_state
        ) = self._new_optimizer_states(self.actor, self.critics)
        self._d1_history: list[EmpiricalTerminalTransitionV1] = []
        self._run2_history: list[ExactP95Run2ShapedTransitionV2] = []
        self._metrics: list[ExactP95Run2TerminalUpdateMetricsV2] = []
        self._collection_seq = 0
        self._warmup_collected = 0
        self._post_warmup_collected = 0
        self._support_violations = 0
        self._closed = False
        self._d1_binding_sha256 = self.environment.binding.canonical_sha256()
        self._train_scene_ids = frozenset(
            assignment.sample_id
            for assignment in self.environment._fit_partition.scene_assignments
            if assignment.split == TRAIN_SPLIT
        )
        self._train_radio_rows = frozenset(
            assignment.csv_row_number
            for assignment in self.environment._fit_partition.radio_assignments
            if assignment.split == TRAIN_SPLIT
        )
        if len(self._train_scene_ids) != self.environment.scene_population_count:
            raise ExactP95Run2RunnerErrorV2("train scene inventory drift")
        if sum(self.environment.radio_profile_population_counts.values()) != len(
            self._train_radio_rows
        ):
            raise ExactP95Run2RunnerErrorV2("train radio inventory drift")
        self._runner_binding_document = self._make_runner_binding_document()
        self._runner_binding_sha256 = canonical_sha256(
            self._runner_binding_document
        )
        self._collection_session_uuid = str(
            uuid.uuid5(
                _COLLECTION_NAMESPACE,
                f"{self._runner_binding_sha256}:{self.seed}",
            )
        )
        self._assert_process_rng_isolation()

    def _new_optimizer_states(self, actor, critics):
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=self.trainer_config.actor_lr
        )
        critic_optimizer = torch.optim.Adam(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters()),
            lr=self.trainer_config.critic_lr,
        )
        return (
            copy.deepcopy(actor_optimizer.state_dict()),
            copy.deepcopy(critic_optimizer.state_dict()),
        )

    def _make_runner_binding_document(self) -> Dict[str, Any]:
        module_directory = Path(__file__).resolve().parent
        evidence_hashes = {
            "gate_canonical_content_sha256": GATE_V2_CANONICAL_CONTENT_SHA256,
            "gate_context_oracles_sha256": GATE_V2_CONTEXT_ORACLES_SHA256,
            "gate_decision_sha256": GATE_V2_DECISION_SHA256,
            "gate_implementation_sha256": GATE_V2_IMPLEMENTATION_SHA256,
            "gate_profile_summary_sha256": GATE_V2_PROFILE_SUMMARY_SHA256,
            "gate_report_sha256": GATE_V2_REPORT_SHA256,
            "gate_summary_sha256": GATE_V2_SUMMARY_SHA256,
            "gate_test_sha256": GATE_V2_TEST_SHA256,
            "preregistration_file_sha256": PREREG_V2_FILE_SHA256,
            "preregistration_hash_manifest_sha256": (
                PREREG_V2_HASH_MANIFEST_SHA256
            ),
            "preregistration_report_sha256": PREREG_V2_REPORT_SHA256,
            "train_canonical_content_sha256": TRAIN_V2_CANONICAL_CONTENT_SHA256,
            "train_decision_sha256": TRAIN_V2_DECISION_SHA256,
            "train_implementation_sha256": TRAIN_V2_IMPLEMENTATION_SHA256,
            "train_oracles_sha256": TRAIN_V2_ORACLES_SHA256,
            "train_report_sha256": TRAIN_V2_REPORT_SHA256,
            "train_summary_sha256": TRAIN_V2_SUMMARY_SHA256,
        }
        return {
            "config_sha256": self.config.canonical_sha256(),
            "d1_binding": self.environment.binding.to_canonical_dict(),
            "d1_binding_sha256": self._d1_binding_sha256,
            "fit_partition_sha256": self.environment.fit_partition_sha256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "phase_label": PHASE_LABEL,
            "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "replay_v2_implementation_sha256": _sha256_file(
                module_directory
                / "empirical_contextual_exact_p95_run2_replay_v2.py"
            ),
            "reward_spec_v2_sha256": EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            "rng_seed_derivation_schema": SEED_DERIVATION_SCHEMA,
            "runner_schema": RUNNER_SCHEMA,
            "sampling_contract_sha256": self.environment.sampling_contract_sha256,
            "sampling_split": self.environment.sampling_split,
            "seed": self.seed,
            "trainer_config": self.trainer_config.to_canonical_dict(),
            "trainer_config_sha256": self.trainer_config.canonical_sha256(),
            "trainer_phase_label": TRAINER_PHASE_LABEL,
            "trainer_v2_implementation_sha256": _sha256_file(
                module_directory
                / "empirical_contextual_exact_p95_run2_terminal_trainer_v2.py"
            ),
            "v2_evidence_hashes": evidence_hashes,
        }

    @property
    def runner_binding_document(self) -> Dict[str, Any]:
        return copy.deepcopy(self._runner_binding_document)

    @property
    def runner_binding_sha256(self) -> str:
        return self._runner_binding_sha256

    @property
    def collection_session_uuid(self) -> str:
        return self._collection_session_uuid

    @property
    def d1_transition_history(self) -> Tuple[EmpiricalTerminalTransitionV1, ...]:
        return tuple(self._d1_history)

    @property
    def run2_v2_transition_history(
        self,
    ) -> Tuple[ExactP95Run2ShapedTransitionV2, ...]:
        return tuple(self._run2_history)

    @property
    def transition_history(
        self,
    ) -> Tuple[ExactP95Run2ShapedTransitionV2, ...]:
        """Convenience alias for the authoritative v2 learning history."""
        return self.run2_v2_transition_history

    @property
    def metrics(self) -> Tuple[ExactP95Run2TerminalUpdateMetricsV2, ...]:
        return tuple(self._metrics)

    @property
    def completed_updates(self) -> int:
        return len(self._metrics)

    @property
    def sampled_scene_ids(self) -> Tuple[str, ...]:
        return tuple(item.result.audit.sample_id for item in self._d1_history)

    @property
    def sampled_radio_csv_row_numbers(self) -> Tuple[int, ...]:
        return tuple(
            item.result.audit.hidden_radio_csv_row_number
            for item in self._d1_history
        )

    def _require_open(self) -> None:
        if self._closed:
            raise ExactP95Run2RunnerErrorV2("runner is closed")

    def _assert_process_rng_isolation(self) -> None:
        if random.getstate() != self._global_python_rng_baseline:
            raise ExactP95Run2RunnerErrorV2("runner advanced global Python RNG")
        if not torch.equal(torch.get_rng_state(), self._global_torch_rng_baseline):
            raise ExactP95Run2RunnerErrorV2("runner advanced global Torch RNG")
        if not self._cuda_initialized_at_entry and torch.cuda.is_initialized():
            raise ExactP95Run2RunnerErrorV2("runner initialized CUDA")

    def require_registered_train_binding(self) -> None:
        if self.environment.sampling_split != TRAIN_SPLIT:
            raise ExactP95Run2RunnerErrorV2("environment is not train-only")
        if (
            self.environment.fit_partition_sha256
            != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
        ):
            raise ExactP95Run2RunnerErrorV2("fit-partition binding drift")
        if self._runner_binding_sha256 != canonical_sha256(
            self._runner_binding_document
        ):
            raise ExactP95Run2RunnerErrorV2("runner binding drift")

    def _warmup_action(self) -> EmpiricalActionV1:
        mode = int(
            torch.randint(
                0,
                len(MODELED_SMOKE_SUPPORT.mode_q_e4_bounds),
                (1,),
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

    def _actor_action(
        self, observation: EmpiricalPolicyObservationV1
    ) -> EmpiricalActionV1:
        state = torch.tensor([observation.values], dtype=torch.float32, device="cpu")
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

    def _ensure_trainer(self) -> ExactP95Run2TerminalHybridSacTrainerV2:
        if self.trainer is None:
            if self.replay.binding is None:
                raise ExactP95Run2RunnerErrorV2(
                    "cannot bind v2 trainer before a v2 transition"
                )
            trainer = ExactP95Run2TerminalHybridSacTrainerV2(
                self.actor,
                self.critics,
                self.trainer_config,
                expected_binding=self.replay.binding,
                actor_generator=self._actor_update_rng,
            )
            trainer.actor_optimizer.load_state_dict(
                copy.deepcopy(self._pending_actor_optimizer_state)
            )
            trainer.critic_optimizer.load_state_dict(
                copy.deepcopy(self._pending_critic_optimizer_state)
            )
            trainer.update_count = len(self._metrics)
            trainer._assert_optimizer_wiring()
            self.trainer = trainer
        return self.trainer

    def _require_train_identity(self, result: EmpiricalStepResultV1) -> None:
        if result.audit.sample_id not in self._train_scene_ids:
            raise ExactP95Run2RunnerErrorV2("non-train scene reached Run-2 v2")
        if result.audit.hidden_radio_csv_row_number not in self._train_radio_rows:
            raise ExactP95Run2RunnerErrorV2("non-train radio row reached Run-2 v2")

    def _collect_one(self, *, warmup: bool) -> None:
        seq = self._collection_seq
        if (
            len(self._d1_history) != seq
            or len(self._run2_history) != seq
            or self.replay.accepted_count != seq
        ):
            raise ExactP95Run2RunnerErrorV2("collection/history/replay drift")
        observation = self.environment.reset()
        action = self._warmup_action() if warmup else self._actor_action(observation)
        result = self.environment.step(action)
        episode = _SynchronousEpisodeV2(seq, observation, action, result)
        episode.require_valid(
            expected_seq=seq, d1_binding_sha256=self._d1_binding_sha256
        )
        self._require_train_identity(result)
        d1_transition = EmpiricalTerminalTransitionV1.from_d1(
            collection_session_uuid=self._collection_session_uuid,
            collection_seq=seq,
            observation=observation,
            action=action,
            result=result,
            d1_binding=self.environment.binding,
        )
        reward_binding: Optional[ExactP95Run2RewardBindingV2] = None
        if self._run2_history:
            reward_binding = self._run2_history[0].reward_binding
        shaped = ExactP95Run2ShapedTransitionV2.from_validated_d1(
            d1_transition,
            project_root=self._project_root,
            reward_binding=reward_binding,
        )
        if (
            d1_transition.observation is not observation
            or d1_transition.action is not action
            or d1_transition.result is not result
            or shaped.source_d1_transition is not d1_transition
            or shaped.source_d1_reward64 != d1_transition.reward
        ):
            raise ExactP95Run2RunnerErrorV2(
                "synchronous D1-to-v2 transition identity was lost"
            )
        self.replay.insert(shaped)
        self._d1_history.append(d1_transition)
        self._run2_history.append(shaped)
        self._collection_seq += 1
        if warmup:
            self._warmup_collected += 1
        else:
            self._post_warmup_collected += 1
        if self.replay.evicted_count != 0:
            raise ExactP95Run2RunnerErrorV2("Run-2-v2 replay evicted a row")

    def run_until_updates(
        self, target_updates: int
    ) -> ExactP95Run2RunnerSummaryV2:
        """Advance to an absolute update count; target zero collects nothing."""
        self._require_open()
        self.require_registered_train_binding()
        if (
            type(target_updates) is not int
            or not self.completed_updates <= target_updates <= self.config.update_count
        ):
            raise ExactP95Run2RunnerErrorV2(
                "target update count is invalid or goes backwards"
            )
        if target_updates == self.completed_updates:
            self._assert_process_rng_isolation()
            return self.summary()
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
                if metric.target_reward_signed_bit_mismatch_count != 0:
                    raise ExactP95Run2RunnerErrorV2(
                        "authenticated v2 target/reward bits differ"
                    )
                self._metrics.append(metric)
                if self.trainer.update_count != len(self._metrics):
                    raise ExactP95Run2RunnerErrorV2(
                        "v2 trainer/runner update counter drift"
                    )
        self._assert_process_rng_isolation()
        return self.summary()

    def run(self) -> ExactP95Run2RunnerSummaryV2:
        return self.run_until_updates(self.config.update_count)

    def checkpoint(self) -> ExactP95Run2CheckpointV2:
        self._require_open()
        if self._run2_history:
            trainer = self._ensure_trainer()
            actor_optimizer_state = copy.deepcopy(
                trainer.actor_optimizer.state_dict()
            )
            critic_optimizer_state = copy.deepcopy(
                trainer.critic_optimizer.state_dict()
            )
        else:
            actor_optimizer_state = copy.deepcopy(
                self._pending_actor_optimizer_state
            )
            critic_optimizer_state = copy.deepcopy(
                self._pending_critic_optimizer_state
            )
        document = dict(
            config=self.config,
            seed=self.seed,
            runner_binding_document=copy.deepcopy(
                self._runner_binding_document
            ),
            runner_binding_sha256=self._runner_binding_sha256,
            collection_session_uuid=self._collection_session_uuid,
            actor_state=copy.deepcopy(self.actor.state_dict()),
            critics_state=copy.deepcopy(self.critics.state_dict()),
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            init_rng_state=copy.deepcopy(self._init_rng.getstate()),
            collection_rng_state=self._collection_rng.get_state().clone(),
            replay_rng_state=self._replay_rng.get_state().clone(),
            actor_update_rng_state=self._actor_update_rng.get_state().clone(),
            environment_state=copy.deepcopy(self.environment.state_dict()),
            d1_transition_history=tuple(self._d1_history),
            run2_v2_transition_history=tuple(self._run2_history),
            metrics=tuple(self._metrics),
            replay_binding=self.replay.binding,
            collection_seq=self._collection_seq,
            warmup_collected=self._warmup_collected,
            post_warmup_collected=self._post_warmup_collected,
            update_count=self.completed_updates,
            replay_accepted_count=self.replay.accepted_count,
            replay_evicted_count=self.replay.evicted_count,
            support_violation_count=self._support_violations,
            trainer_initialized=self.trainer is not None,
            schema=CHECKPOINT_SCHEMA,
        )
        checkpoint = ExactP95Run2CheckpointV2(
            **document, checkpoint_sha256=_hash_state(document)
        )
        checkpoint.require_valid()
        self._assert_process_rng_isolation()
        return checkpoint

    def _validate_environment_state(
        self,
        state: PartitionedEmpiricalEnvironmentStateV1,
        *,
        expected_reset_count: int,
    ) -> None:
        if type(state) is not PartitionedEmpiricalEnvironmentStateV1:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint environment state has a foreign type"
            )
        if (
            state.sampling_split != TRAIN_SPLIT
            or state.fit_partition_sha256 != self.environment.fit_partition_sha256
            or state.sampling_contract_sha256
            != self.environment.sampling_contract_sha256
        ):
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint train-environment binding mismatch"
            )
        d1_state = state.d1_state
        if (
            d1_state.environment_binding_sha256 != self._d1_binding_sha256
            or d1_state.master_seed != self._stream_seeds["environment"]
            or d1_state.reset_count != expected_reset_count
        ):
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint D1 environment state mismatch"
            )
        context_probe = random.Random()
        try:
            context_probe.setstate(d1_state.context_rng_state)
            self.environment._radio_sampler.validate_state_dict(
                d1_state.radio_sampler_state
            )
        except (TypeError, ValueError) as exc:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint environment RNG state is malformed"
            ) from exc
        if d1_state.radio_sampler_state.draw_count != expected_reset_count:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint radio/environment draw counter mismatch"
            )

    def _stage_unbound_optimizer_states(
        self,
        actor,
        critics,
        actor_state: Mapping[str, Any],
        critic_state: Mapping[str, Any],
    ) -> None:
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=self.trainer_config.actor_lr
        )
        critic_optimizer = torch.optim.Adam(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters()),
            lr=self.trainer_config.critic_lr,
        )
        actor_optimizer.load_state_dict(copy.deepcopy(actor_state))
        critic_optimizer.load_state_dict(copy.deepcopy(critic_state))
        for label, optimizer, expected_lr in (
            ("actor", actor_optimizer, self.trainer_config.actor_lr),
            ("critic", critic_optimizer, self.trainer_config.critic_lr),
        ):
            if (
                type(optimizer) is not torch.optim.Adam
                or len(optimizer.param_groups) != 1
                or optimizer.param_groups[0].get("lr") != expected_lr
            ):
                raise ExactP95Run2RunnerErrorV2(
                    f"checkpoint {label} optimizer structure drift"
                )
            for optimizer_item in optimizer.state.values():
                for value in optimizer_item.values():
                    if isinstance(value, Tensor) and (
                        value.device.type != "cpu"
                        or (
                            value.is_floating_point()
                            and not bool(torch.isfinite(value).all())
                        )
                    ):
                        raise ExactP95Run2RunnerErrorV2(
                            f"checkpoint {label} optimizer tensor drift"
                        )

    def load_checkpoint(self, checkpoint: ExactP95Run2CheckpointV2) -> None:
        """Stage and validate the complete checkpoint before live mutation."""
        self._require_open()
        if type(checkpoint) is not ExactP95Run2CheckpointV2:
            raise ExactP95Run2RunnerErrorV2("checkpoint has a foreign type")
        checkpoint.require_valid()
        if checkpoint.config != self.config or checkpoint.seed != self.seed:
            raise ExactP95Run2RunnerErrorV2("checkpoint schedule/seed mismatch")
        if checkpoint.runner_binding_sha256 != self._runner_binding_sha256:
            raise ExactP95Run2RunnerErrorV2("checkpoint runner binding mismatch")
        if dict(checkpoint.runner_binding_document) != self._runner_binding_document:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint runner binding document mismatch"
            )
        if checkpoint.collection_session_uuid != self._collection_session_uuid:
            raise ExactP95Run2RunnerErrorV2("checkpoint collection UUID mismatch")

        rebuilt = ExactP95Run2ReplayV2(self.config.replay_capacity)
        for expected_seq, (source, shaped) in enumerate(
            zip(
                checkpoint.d1_transition_history,
                checkpoint.run2_v2_transition_history,
            )
        ):
            source.revalidate()
            shaped.revalidate()
            self._require_train_identity(source.result)
            if (
                source.collection_seq != expected_seq
                or source.collection_session_uuid != self._collection_session_uuid
                or shaped.source_d1_transition.canonical_sha256()
                != source.canonical_sha256()
            ):
                raise ExactP95Run2RunnerErrorV2(
                    "checkpoint transition sequence/source drift"
                )
            rebuilt.insert(shaped)
        if (
            rebuilt.evicted_count != 0
            or len(rebuilt) != checkpoint.collection_seq
            or rebuilt.accepted_count != checkpoint.replay_accepted_count
        ):
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint history cannot reconstruct v2 replay"
            )
        if rebuilt.binding != checkpoint.replay_binding:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint reconstructed replay binding drift"
            )

        staged_actor = copy.deepcopy(self.actor)
        staged_critics = copy.deepcopy(self.critics)
        try:
            staged_actor.load_state_dict(
                copy.deepcopy(checkpoint.actor_state), strict=True
            )
            staged_critics.load_state_dict(
                copy.deepcopy(checkpoint.critics_state), strict=True
            )
        except (RuntimeError, ValueError) as exc:
            raise ExactP95Run2RunnerErrorV2(
                "checkpoint model state is incompatible"
            ) from exc
        staged_collection_rng = torch.Generator(device="cpu")
        staged_replay_rng = torch.Generator(device="cpu")
        staged_actor_update_rng = torch.Generator(device="cpu")
        staged_collection_rng.set_state(checkpoint.collection_rng_state.clone())
        staged_replay_rng.set_state(checkpoint.replay_rng_state.clone())
        staged_actor_update_rng.set_state(
            checkpoint.actor_update_rng_state.clone()
        )
        staged_init_rng = random.Random()
        staged_init_rng.setstate(copy.deepcopy(checkpoint.init_rng_state))
        staged_trainer: Optional[
            ExactP95Run2TerminalHybridSacTrainerV2
        ] = None
        if checkpoint.trainer_initialized:
            if rebuilt.binding is None:
                raise ExactP95Run2RunnerErrorV2(
                    "checkpoint trainer lacks a replay binding"
                )
            staged_trainer = ExactP95Run2TerminalHybridSacTrainerV2(
                staged_actor,
                staged_critics,
                self.trainer_config,
                expected_binding=rebuilt.binding,
                actor_generator=staged_actor_update_rng,
            )
            try:
                staged_trainer.actor_optimizer.load_state_dict(
                    copy.deepcopy(checkpoint.actor_optimizer_state)
                )
                staged_trainer.critic_optimizer.load_state_dict(
                    copy.deepcopy(checkpoint.critic_optimizer_state)
                )
                staged_trainer.update_count = checkpoint.update_count
                staged_trainer._validate_models(
                    staged_actor, staged_critics, self.trainer_config
                )
                staged_trainer._assert_optimizer_wiring()
            except (RuntimeError, ValueError) as exc:
                raise ExactP95Run2RunnerErrorV2(
                    "checkpoint trainer/optimizer state is incompatible"
                ) from exc
        else:
            self._stage_unbound_optimizer_states(
                staged_actor,
                staged_critics,
                checkpoint.actor_optimizer_state,
                checkpoint.critic_optimizer_state,
            )
        self._validate_environment_state(
            checkpoint.environment_state,
            expected_reset_count=checkpoint.collection_seq,
        )

        # All fallible digest, history, replay, model, optimizer, RNG and
        # environment validation has completed.  Environment restoration is
        # itself fail-atomic; the assignments following it are non-failing.
        self.environment.load_state_dict(checkpoint.environment_state)
        self.actor = staged_actor
        self.critics = staged_critics
        self._init_rng = staged_init_rng
        self._collection_rng = staged_collection_rng
        self._replay_rng = staged_replay_rng
        self._actor_update_rng = staged_actor_update_rng
        self.replay = rebuilt
        self.trainer = staged_trainer
        self._pending_actor_optimizer_state = copy.deepcopy(
            checkpoint.actor_optimizer_state
        )
        self._pending_critic_optimizer_state = copy.deepcopy(
            checkpoint.critic_optimizer_state
        )
        self._d1_history = list(checkpoint.d1_transition_history)
        self._run2_history = list(checkpoint.run2_v2_transition_history)
        self._metrics = list(checkpoint.metrics)
        self._collection_seq = checkpoint.collection_seq
        self._warmup_collected = checkpoint.warmup_collected
        self._post_warmup_collected = checkpoint.post_warmup_collected
        self._support_violations = checkpoint.support_violation_count
        self._assert_process_rng_isolation()

    @staticmethod
    def _parameter_delta_norm(
        module: torch.nn.Module, initial: Mapping[str, Any]
    ) -> float:
        total = 0.0
        current = module.state_dict()
        if tuple(current) != tuple(initial):
            raise ExactP95Run2RunnerErrorV2("model state inventory drift")
        for name, value in current.items():
            reference = initial[name]
            if value.is_floating_point():
                delta = value.detach().to(torch.float64) - reference.to(torch.float64)
                total += float(torch.sum(delta * delta))
            elif not torch.equal(value, reference):
                raise ExactP95Run2RunnerErrorV2(
                    f"non-floating model buffer {name} drift"
                )
        return math.sqrt(total)

    def summary(self) -> ExactP95Run2RunnerSummaryV2:
        self._require_open()
        raw_rewards = tuple(item.reward for item in self._d1_history)
        shaped_rewards = tuple(item.shaped_reward64 for item in self._run2_history)
        emitted_rewards = tuple(
            item.emitted_reward_float32 for item in self._run2_history
        )
        metrics_finite = all(
            all(
                not isinstance(value, (int, float))
                or math.isfinite(float(value))
                for value in metric.as_dict().values()
            )
            for metric in self._metrics
        )
        mismatch_count = sum(
            metric.target_reward_signed_bit_mismatch_count
            for metric in self._metrics
        )
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
        complete = self.completed_updates == self.config.update_count
        hard_gates = (
            complete
            and len(self._run2_history) == self.config.total_transitions
            and self.replay.evicted_count == 0
            and self._support_violations == 0
            and mismatch_count == 0
            and metrics_finite
            and actor_delta > 0.0
            and critic_1_delta > 0.0
            and critic_2_delta > 0.0
            and python_rng_unchanged
            and torch_rng_unchanged
            and not cuda_initialized_by_runner
        )
        if complete and not hard_gates:
            raise ExactP95Run2RunnerErrorV2(
                "completed Run-2-v2 training failed a hard gate"
            )
        replay_binding = self.replay.binding
        checkpoint = self.checkpoint()
        return ExactP95Run2RunnerSummaryV2(
            schema=SUMMARY_SCHEMA,
            phase_label=PHASE_LABEL,
            seed=self.seed,
            sampling_split=self.environment.sampling_split,
            configured_updates=self.config.update_count,
            completed_updates=self.completed_updates,
            transition_count=len(self._run2_history),
            warmup_transition_count=self._warmup_collected,
            post_warmup_transition_count=self._post_warmup_collected,
            replay_resident_count=len(self.replay),
            replay_eviction_count=self.replay.evicted_count,
            support_violation_count=self._support_violations,
            raw_d1_reward_min=None if not raw_rewards else min(raw_rewards),
            raw_d1_reward_max=None if not raw_rewards else max(raw_rewards),
            raw_d1_reward_mean=_mean(raw_rewards),
            shaped_reward64_min=(
                None if not shaped_rewards else min(shaped_rewards)
            ),
            shaped_reward64_max=(
                None if not shaped_rewards else max(shaped_rewards)
            ),
            shaped_reward64_mean=_mean(shaped_rewards),
            emitted_reward_float32_min=(
                None if not emitted_rewards else min(emitted_rewards)
            ),
            emitted_reward_float32_max=(
                None if not emitted_rewards else max(emitted_rewards)
            ),
            emitted_reward_float32_mean=_mean(emitted_rewards),
            target_reward_signed_bit_mismatch_count=mismatch_count,
            all_metrics_finite=metrics_finite,
            actor_parameter_delta_from_init=actor_delta,
            critic_1_parameter_delta_from_init=critic_1_delta,
            critic_2_parameter_delta_from_init=critic_2_delta,
            global_python_rng_unchanged=python_rng_unchanged,
            global_torch_rng_unchanged=torch_rng_unchanged,
            cuda_initialized_by_runner=cuda_initialized_by_runner,
            completed_training_hard_gates_passed=hard_gates,
            config_sha256=self.config.canonical_sha256(),
            runner_binding_sha256=self._runner_binding_sha256,
            d1_binding_sha256=self._d1_binding_sha256,
            fit_partition_sha256=self.environment.fit_partition_sha256,
            sampling_contract_sha256=self.environment.sampling_contract_sha256,
            replay_binding_sha256=(
                None
                if replay_binding is None
                else replay_binding.canonical_sha256()
            ),
            reward_binding_sha256=(
                None
                if replay_binding is None
                else replay_binding.reward_binding.canonical_sha256()
            ),
            trainer_config_sha256=self.trainer_config.canonical_sha256(),
            modeled_smoke_support_sha256=MODELED_SMOKE_SUPPORT_SHA256,
            pilot_utility_spec_sha256=PILOT_UTILITY_SPEC_SHA256,
            reward_spec_v2_sha256=EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            d1_transition_history_sha256=canonical_sha256(
                [item.canonical_sha256() for item in self._d1_history]
            ),
            run2_v2_transition_history_sha256=canonical_sha256(
                [item.canonical_sha256() for item in self._run2_history]
            ),
            metrics_history_sha256=canonical_sha256(
                [item.canonical_sha256() for item in self._metrics]
            ),
            checkpoint_sha256=checkpoint.checkpoint_sha256,
        )

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True

    def __enter__(self) -> "ExactP95Run2RunnerV2":
        self._require_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


EmpiricalContextualExactP95Run2RunnerV2 = ExactP95Run2RunnerV2

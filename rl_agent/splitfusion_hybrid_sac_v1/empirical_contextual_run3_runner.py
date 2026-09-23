"""Bounded train-only Run-3 realized-outcome Hybrid-SAC runner.

The registered environment revalidates the already-frozen full evidence
qualification while loading.  Collection itself is restricted to the 391
registered training scene identities and training radio rows.  No validation
panel, validation evaluator, held checkpoint selection, CARLA, OAI, CUDA, or
network service is imported or launched here.
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
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .empirical_contextual_contract import EmpiricalActionV1, require_supported_action
from .empirical_contextual_environment import EmpiricalPolicyObservationV1
from .empirical_contextual_fit_partition import (
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
)
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalEnvironmentStateV1,
    PartitionedEmpiricalOneStepEnvironmentV1,
)
from .empirical_contextual_run3_reward import (
    RUN3_KERNEL_SPEC_SHA256,
    RUN3_REWARD_SPEC_SHA256,
    QuantileLatencyProxyV1,
    Run3CounterRngV1,
    sample_run3_simulator_outcome,
)
from .empirical_contextual_run3_terminal_replay import (
    Run3ReplayBindingV1,
    Run3TerminalReplayV1,
    Run3TerminalTransitionV1,
)
from .empirical_contextual_run3_terminal_trainer import (
    Run3TerminalHybridSacTrainerV1,
    Run3TerminalTrainerConfigV1,
    Run3TerminalUpdateMetricsV1,
)
from .empirical_contextual_terminal_replay import EmpiricalTerminalTransitionV1
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
    quantize_q_e4,
)
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256
from .transaction_identity import canonical_sha256

__all__ = [
    "RUN3_PREFLIGHT_RESULT_SHA256",
    "RUN3_REGISTERED_CONFIG",
    "Run3CheckpointV1",
    "Run3RunnerConfigV1",
    "Run3RunnerError",
    "Run3RunnerSummaryV1",
    "Run3TrainingRunnerV1",
]


RUN3_PREFLIGHT_RESULT_SHA256 = "52801ce029eaac4dfde4834a2d8031429663a3efb383deed7ec1ed06d50ebab7"
RUN3_RUNNER_SCHEMA = "splitfusion.run3_realized_training_runner.v1"
RUN3_CHECKPOINT_SCHEMA = "splitfusion.run3_realized_training_checkpoint.v1"
RUN3_SUMMARY_SCHEMA = "splitfusion.run3_realized_training_summary.v1"
RUN3_MODEL_SNAPSHOT_SCHEMA = "splitfusion.run3_model_only_snapshot.v1"
RUN3_PHASE_LABEL = "RUN3_REALIZED_REWARD_TRAIN_ONLY_HYBRID_SAC"
RUN3_SEED_SCHEMA = "splitfusion.run3_rng_streams.v1"
_SESSION_NAMESPACE = uuid.UUID("5b8c1332-6d6c-59a4-8ec1-df34f8c2f6db")
_REGISTERED_DECISION_LINEAGE_IMPLEMENTATION_HASHES = (
    ("empirical_contextual_run3_reward.py", "37309fd539a7a2db3365c7c040a4ae9c8b74b85611baecd8574537e57cd921b6"),
    ("empirical_contextual_run3_terminal_replay.py", "bfdf700e11433a629e1ec399a1ac4ef8032a610db8f1672300e3a419f15ab781"),
    ("empirical_contextual_run3_terminal_trainer.py", "90176f108b2f412efa0cb07c074bd2c3289f1f71813a422c4f4315994a582095"),
    ("empirical_contextual_run3_runner.py", "ece95021f283f592ce2b39989be73e255c918b8d6e68575781c166723a03baab"),
)
_REGISTERED_DECISION_LINEAGE_SEED17_BINDING_SHA256 = (
    "80ed51b8e02bc4c36224451cc0a9d1d66876e2a0076f16da46205d435d7f3e34"
)
_REGISTERED_DECISION_LINEAGE_SEED17_SESSION_UUID = (
    "6dc674b2-4fd5-5720-84b4-e59ffe1330a1"
)


class Run3RunnerError(RuntimeError):
    """Run-3 schedule, collection, checkpoint, or hard gate failed."""


def _derive_seed(master_seed: int, stream: str) -> int:
    raw = f"{RUN3_SEED_SCHEMA}:{master_seed}:{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decision_lineage_binding_sha256(
    current_binding_document: Mapping[str, Any],
) -> str:
    """Reconstruct the registered d11095c decision-key lineage.

    Current implementation hashes remain authoritative evidence metadata.
    This separate registered lineage prevents a performance-only refactor from
    changing decision keys and therefore counter-kernel realized outcomes.
    """

    document = copy.deepcopy(dict(current_binding_document))
    document.pop("registered_decision_lineage", None)
    document["implementation_hashes"] = dict(
        _REGISTERED_DECISION_LINEAGE_IMPLEMENTATION_HASHES
    )
    return canonical_sha256(document)


def _hash_state(value: Any) -> str:
    """Stable digest of nested checkpoint state, independent of pickle."""
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
        elif isinstance(item, Enum):
            emit(type(item).__module__ + "." + type(item).__qualname__)
            emit(item.value)
        elif isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            emit(str(tensor.dtype))
            emit(tuple(tensor.shape))
            data = tensor.numpy().tobytes(order="C")
            digest.update(b"T" + len(data).to_bytes(8, "big") + data)
        elif isinstance(item, Mapping):
            entries = sorted(item.items(), key=lambda pair: repr(pair[0]))
            digest.update(b"M")
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
            for descriptor in fields(item):
                if descriptor.name != "checkpoint_sha256":
                    emit(descriptor.name)
                    emit(getattr(item, descriptor.name))
        else:
            raise Run3RunnerError(f"unsupported checkpoint type {type(item).__name__}")

    emit(value)
    return digest.hexdigest()


@contextmanager
def _one_cpu_thread() -> Iterator[None]:
    prior = torch.get_num_threads()
    if prior != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if prior != 1:
            torch.set_num_threads(prior)


@dataclass(frozen=True, slots=True)
class Run3RunnerConfigV1:
    seeds: Tuple[int, ...] = (17, 29, 43)
    warmup_transitions: int = 1024
    batch_size: int = 256
    collect_per_update: int = 4
    update_count: int = 10_000
    matched_horizon_update: int = 5_000
    replay_capacity: int = 65_536
    model_snapshot_interval: int = 500
    cpu_threads: int = 1
    scope: str = RUN3_PHASE_LABEL

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not tuple
            or not self.seeds
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise Run3RunnerError("seeds must be unique non-negative integers")
        for name in (
            "warmup_transitions", "batch_size", "collect_per_update",
            "update_count", "matched_horizon_update", "replay_capacity",
            "model_snapshot_interval", "cpu_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise Run3RunnerError(f"{name} must be positive int")
        if self.cpu_threads != 1:
            raise Run3RunnerError("Run-3 is fixed to one CPU thread")
        if self.warmup_transitions < self.batch_size:
            raise Run3RunnerError("warmup must provide a full batch")
        if self.matched_horizon_update >= self.update_count:
            raise Run3RunnerError("matched horizon must precede primary endpoint")
        if self.total_transitions > self.replay_capacity:
            raise Run3RunnerError("schedule would evict replay transitions")
        if self.scope != RUN3_PHASE_LABEL:
            raise Run3RunnerError("Run-3 scope drift")

    @property
    def total_transitions(self) -> int:
        return self.warmup_transitions + self.collect_per_update * self.update_count

    @property
    def full_checkpoint_updates(self) -> Tuple[int, int, int]:
        return 0, self.matched_horizon_update, self.update_count

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "collect_per_update": self.collect_per_update,
            "cpu_threads": self.cpu_threads,
            "full_checkpoint_updates": list(self.full_checkpoint_updates),
            "matched_horizon_update": self.matched_horizon_update,
            "model_snapshot_interval": self.model_snapshot_interval,
            "primary_endpoint_selection": (
                f"FIXED_CONFIGURED_UPDATE_{self.update_count}_NO_PEAK_SELECTION"
            ),
            "record": "run3_runner_config_v1",
            "replay_capacity": self.replay_capacity,
            "scope": self.scope,
            "seeds": list(self.seeds),
            "update_count": self.update_count,
            "warmup_transitions": self.warmup_transitions,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


RUN3_REGISTERED_CONFIG = Run3RunnerConfigV1()


@dataclass(frozen=True, slots=True)
class Run3CheckpointV1:
    config: Run3RunnerConfigV1
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
    transitions: Tuple[Run3TerminalTransitionV1, ...]
    metrics: Tuple[Run3TerminalUpdateMetricsV1, ...]
    replay_binding: Optional[Run3ReplayBindingV1]
    collection_seq: int
    warmup_collected: int
    post_warmup_collected: int
    update_count: int
    replay_accepted_count: int
    replay_evicted_count: int
    excluded_fault_count: int
    trainer_initialized: bool
    schema: str
    checkpoint_sha256: str

    def _document(self) -> Dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if descriptor.name != "checkpoint_sha256"
        }

    def require_valid(self) -> None:
        if type(self.config) is not Run3RunnerConfigV1:
            raise Run3RunnerError("checkpoint config type drift")
        self.config.__post_init__()
        if self.schema != RUN3_CHECKPOINT_SCHEMA:
            raise Run3RunnerError("checkpoint schema drift")
        if self.seed not in self.config.seeds:
            raise Run3RunnerError("checkpoint seed drift")
        if canonical_sha256(dict(self.runner_binding_document)) != self.runner_binding_sha256:
            raise Run3RunnerError("checkpoint binding digest drift")
        if self.runner_binding_document.get("preflight_result_sha256") != RUN3_PREFLIGHT_RESULT_SHA256:
            raise Run3RunnerError("checkpoint preflight binding drift")
        if self.runner_binding_document.get("reward_spec_sha256") != RUN3_REWARD_SPEC_SHA256:
            raise Run3RunnerError("checkpoint reward binding drift")
        if self.runner_binding_document.get("kernel_spec_sha256") != RUN3_KERNEL_SPEC_SHA256:
            raise Run3RunnerError("checkpoint kernel binding drift")
        if self.checkpoint_sha256 != _hash_state(self._document()):
            raise Run3RunnerError("checkpoint content digest mismatch")
        if self.collection_seq != len(self.transitions):
            raise Run3RunnerError("checkpoint transition count drift")
        if self.update_count != len(self.metrics):
            raise Run3RunnerError("checkpoint metrics/update drift")
        if self.warmup_collected != min(self.collection_seq, self.config.warmup_transitions):
            raise Run3RunnerError("checkpoint warmup count drift")
        if self.post_warmup_collected != self.update_count * self.config.collect_per_update:
            raise Run3RunnerError("checkpoint post-warmup count drift")
        if self.collection_seq != self.warmup_collected + self.post_warmup_collected:
            raise Run3RunnerError("checkpoint collection arithmetic drift")
        if self.replay_accepted_count != self.collection_seq or self.replay_evicted_count != 0:
            raise Run3RunnerError("checkpoint replay count/eviction drift")
        if self.excluded_fault_count != 0:
            raise Run3RunnerError("completed checkpoint contains excluded faults")
        if self.trainer_initialized is not bool(self.collection_seq):
            raise Run3RunnerError("checkpoint trainer initialization drift")
        for index, row in enumerate(self.transitions):
            if type(row) is not Run3TerminalTransitionV1:
                raise Run3RunnerError("checkpoint transition type drift")
            row.revalidate()
            if row.collection_session_uuid != self.collection_session_uuid or row.collection_seq != index:
                raise Run3RunnerError("checkpoint transition sequence drift")
        if self.transitions:
            if type(self.replay_binding) is not Run3ReplayBindingV1:
                raise Run3RunnerError("checkpoint replay binding missing")
            self.replay_binding.require_valid()
            if self.replay_binding.d1_environment_binding_sha256 != self.transitions[0].d1_environment_binding_sha256:
                raise Run3RunnerError("checkpoint replay binding mismatch")
        elif self.replay_binding is not None:
            raise Run3RunnerError("empty checkpoint has replay binding")
        for metric in self.metrics:
            metric.assert_valid()


@dataclass(frozen=True, slots=True)
class Run3RunnerSummaryV1:
    schema: str
    seed: int
    configured_updates: int
    completed_updates: int
    transition_count: int
    replay_resident_count: int
    replay_eviction_count: int
    excluded_fault_count: int
    reward_mean: Optional[float]
    success_count: int
    reassembly_failure_count: int
    admission_failure_count: int
    service_timeout_count: int
    actor_parameter_delta_norm: float
    critic_1_parameter_delta_norm: float
    critic_2_parameter_delta_norm: float
    checkpoint_sha256: str
    completed_training_hard_gates_passed: bool


class Run3TrainingRunnerV1:
    """Deterministic 391-scene train-partition collector and trainer."""

    def __init__(
        self,
        *,
        seed: int,
        config: Run3RunnerConfigV1 = RUN3_REGISTERED_CONFIG,
        project_root: Optional[Path] = None,
    ) -> None:
        if type(config) is not Run3RunnerConfigV1:
            raise Run3RunnerError("runner config has foreign type")
        config.__post_init__()
        if type(seed) is not int or seed not in config.seeds:
            raise Run3RunnerError("seed is not registered in config")
        self.config = config
        self.seed = seed
        self._project_root = None if project_root is None else Path(project_root).resolve(strict=True)
        self._python_rng_baseline = random.getstate()
        self._torch_rng_baseline = torch.get_rng_state().clone()
        self._cuda_at_entry = torch.cuda.is_initialized()
        stream_names = ("init", "collection", "replay", "actor_update", "environment", "counter_kernel")
        self._stream_seeds = {name: _derive_seed(seed, name) for name in stream_names}
        self._init_rng = random.Random(self._stream_seeds["init"])
        actor_seed = self._init_rng.randrange(0, 1 << 63)
        critic_seed = self._init_rng.randrange(0, 1 << 63)
        self._collection_rng = torch.Generator(device="cpu")
        self._collection_rng.manual_seed(self._stream_seeds["collection"])
        self._replay_rng = torch.Generator(device="cpu")
        self._replay_rng.manual_seed(self._stream_seeds["replay"])
        self._actor_update_rng = torch.Generator(device="cpu")
        self._actor_update_rng.manual_seed(self._stream_seeds["actor_update"])
        self._counter_rng = Run3CounterRngV1(self._stream_seeds["counter_kernel"])

        model_config = HybridSacModelConfig(dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT)
        self.actor = build_actor(model_config, seed=actor_seed)
        self.critics = build_twin_critics(model_config, seed=critic_seed)
        self._initial_actor = copy.deepcopy(self.actor.state_dict())
        self._initial_critic_1 = copy.deepcopy(self.critics.critic_1.state_dict())
        self._initial_critic_2 = copy.deepcopy(self.critics.critic_2.state_dict())
        self.environment = PartitionedEmpiricalOneStepEnvironmentV1.load_registered(
            seed=self._stream_seeds["environment"], split=TRAIN_SPLIT,
            project_root=self._project_root,
        )
        self.replay = Run3TerminalReplayV1(
            config.replay_capacity, partition=self.environment._fit_partition
        )
        self.trainer_config = Run3TerminalTrainerConfigV1(batch_size=config.batch_size)
        self.trainer: Optional[Run3TerminalHybridSacTrainerV1] = None
        self._pending_actor_optimizer_state, self._pending_critic_optimizer_state = self._fresh_optimizer_states()
        self._transitions: list[Run3TerminalTransitionV1] = []
        self._metrics: list[Run3TerminalUpdateMetricsV1] = []
        self._collection_seq = 0
        self._warmup_collected = 0
        self._post_warmup_collected = 0
        self._excluded_fault_count = 0
        self._closed = False
        self._d1_binding_sha256 = self.environment.binding.canonical_sha256()
        partition = self.environment._fit_partition
        self._train_scene_ids = frozenset(row.sample_id for row in partition.scene_assignments if row.split == TRAIN_SPLIT)
        self._train_radio_rows = frozenset(row.csv_row_number for row in partition.radio_assignments if row.split == TRAIN_SPLIT)
        self._runner_binding_document = self._make_binding()
        self._runner_binding_sha256 = canonical_sha256(self._runner_binding_document)
        self._decision_lineage_binding_sha256 = _decision_lineage_binding_sha256(
            self._runner_binding_document
        )
        self._collection_session_uuid = str(
            uuid.uuid5(
                _SESSION_NAMESPACE,
                f"{self._decision_lineage_binding_sha256}:{seed}",
            )
        )
        if self.config == RUN3_REGISTERED_CONFIG and seed == 17 and (
            self._decision_lineage_binding_sha256
            != _REGISTERED_DECISION_LINEAGE_SEED17_BINDING_SHA256
            or self._collection_session_uuid
            != _REGISTERED_DECISION_LINEAGE_SEED17_SESSION_UUID
        ):
            raise Run3RunnerError("registered seed-17 decision lineage drift")
        self._assert_process_isolation()

    def _fresh_optimizer_states(self):
        actor = torch.optim.Adam(self.actor.parameters(), lr=self.trainer_config.actor_lr)
        critic = torch.optim.Adam(chain(self.critics.critic_1.parameters(), self.critics.critic_2.parameters()), lr=self.trainer_config.critic_lr)
        return copy.deepcopy(actor.state_dict()), copy.deepcopy(critic.state_dict())

    def _make_binding(self) -> Dict[str, Any]:
        directory = Path(__file__).resolve().parent
        return {
            "config": self.config.to_canonical_dict(),
            "config_sha256": self.config.canonical_sha256(),
            "d1_environment_binding_sha256": self._d1_binding_sha256,
            "fit_partition_sha256": self.environment.fit_partition_sha256,
            "frozen_full_evidence_load_disclosure": (
                "LOAD_REVALIDATES_FROZEN_FULL_SURFACE_QUALIFICATION;"
                "COLLECTION_SAMPLES_ONLY_391_REGISTERED_TRAIN_IDENTITIES"
            ),
            "implementation_hashes": {
                name: _sha256_file(directory / name)
                for name in (
                    "empirical_contextual_run3_reward.py",
                    "empirical_contextual_run3_terminal_replay.py",
                    "empirical_contextual_run3_terminal_trainer.py",
                    "empirical_contextual_run3_runner.py",
                )
            },
            "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "phase_label": RUN3_PHASE_LABEL,
            "preflight_result_sha256": RUN3_PREFLIGHT_RESULT_SHA256,
            "registered_decision_lineage": {
                "baseline_commit": "d11095ce0077e89a414c63e6489e90bade4bd400",
                "implementation_hashes": dict(
                    _REGISTERED_DECISION_LINEAGE_IMPLEMENTATION_HASHES
                ),
                "purpose": (
                    "PRESERVE_DECISION_KEYS_AND_COUNTER_KERNEL_DRAWS_ACROSS_"
                    "PERFORMANCE_ONLY_REFACTOR"
                ),
            },
            "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
            "rng_stream_seeds": dict(self._stream_seeds),
            "runner_schema": RUN3_RUNNER_SCHEMA,
            "sampling_contract_sha256": self.environment.sampling_contract_sha256,
            "sampling_split": self.environment.sampling_split,
            "seed": self.seed,
            "trainer_config_sha256": self.trainer_config.canonical_sha256(),
        }

    @property
    def metrics(self) -> Tuple[Run3TerminalUpdateMetricsV1, ...]:
        return tuple(self._metrics)

    @property
    def transitions(self) -> Tuple[Run3TerminalTransitionV1, ...]:
        return tuple(self._transitions)

    @property
    def completed_updates(self) -> int:
        return len(self._metrics)

    @property
    def collection_session_uuid(self) -> str:
        return self._collection_session_uuid

    def _assert_process_isolation(self) -> None:
        if random.getstate() != self._python_rng_baseline:
            raise Run3RunnerError("runner advanced global Python RNG")
        if not torch.equal(torch.get_rng_state(), self._torch_rng_baseline):
            raise Run3RunnerError("runner advanced global Torch RNG")
        if not self._cuda_at_entry and torch.cuda.is_initialized():
            raise Run3RunnerError("runner initialized CUDA")

    def _require_open(self) -> None:
        if self._closed:
            raise Run3RunnerError("runner is closed")

    def _warmup_action(self) -> Tuple[EmpiricalActionV1, float]:
        mode = int(torch.randint(0, 12, (1,), generator=self._collection_rng).item())
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[mode]
        q_e4 = int(torch.randint(lower, upper + 1, (1,), generator=self._collection_rng).item())
        requested_q = q_e4 / 10_000.0
        canonical = int(quantize_q_e4(torch.tensor([requested_q], dtype=torch.float32))[0])
        if canonical != q_e4:
            raise Run3RunnerError("warmup canonical quantization drift")
        return require_supported_action(mode, canonical), requested_q

    def _actor_action(self, observation: EmpiricalPolicyObservationV1) -> Tuple[EmpiricalActionV1, float]:
        state = torch.tensor([observation.values], dtype=torch.float32)
        with torch.no_grad():
            _log_probs, probabilities = self.actor.mode_log_probs(state)
            mode = int(torch.multinomial(probabilities[0], 1, generator=self._collection_rng).item())
            proposal = self.actor.sample_all_modes(state, generator=self._collection_rng)
            requested_q = float(proposal.q[0, mode])
            proposed_q_e4 = int(proposal.q_e4[0, mode])
            canonical_q_e4 = int(quantize_q_e4(torch.tensor([requested_q], dtype=torch.float32))[0])
        if proposed_q_e4 != canonical_q_e4:
            raise Run3RunnerError("actor proposal/canonical q_e4 mismatch")
        return require_supported_action(mode, canonical_q_e4), requested_q

    def _quality_components(self, sample_id: str, action: EmpiricalActionV1):
        query = self.environment._surface.query_fit_q_e4(sample_id, action.mode_id, action.q_e4)
        records = tuple((item.name, item.value, item.valid, item.status) for item in query.policy.quality)
        values = {item[0]: item[1] for item in records}
        for name in ("q_loc", "q_seg", "q_perc"):
            if type(values.get(name)) is not float:
                raise Run3RunnerError(f"quality component {name} unavailable")
        return query, records, float(values["q_loc"]), float(values["q_seg"]), float(values["q_perc"])

    def _collect_one(self, *, warmup: bool) -> None:
        seq = self._collection_seq
        if seq != len(self._transitions) or seq != self.replay.accepted_count:
            raise Run3RunnerError("collection/history/replay drift")
        # Identity is frozen before observing or selecting an action; it cannot
        # depend on mode, q, sample, profile, or any realized outcome.
        decision_key = f"run3:{self._collection_session_uuid}:{seq}"
        observation = self.environment.reset()
        action, requested_q = self._warmup_action() if warmup else self._actor_action(observation)
        source = self.environment.step(action)
        audit = source.audit
        if audit.sample_id not in self._train_scene_ids or audit.hidden_radio_csv_row_number not in self._train_radio_rows:
            raise Run3RunnerError("non-training identity reached Run-3")
        query, records, q_loc, q_seg, q_perc = self._quality_components(audit.sample_id, action)
        policy = source.policy
        required = (
            policy.p_complete_reassembly_given_sent,
            policy.p_edge_admission_given_reassembled,
            policy.latency_proxy_ms,
            policy.latency_proxy_p95_ms,
            policy.latency_proxy_p99_ms,
        )
        if any(type(value) is not float for value in required):
            # This is evidence/evaluator unavailability, not a service failure.
            self._excluded_fault_count += 1
            raise Run3RunnerError("evaluator/source fault excluded; no replay row emitted")
        proxy = QuantileLatencyProxyV1(
            p50_ms=float(policy.latency_proxy_ms),
            p95_ms=float(policy.latency_proxy_p95_ms),
            p99_ms=float(policy.latency_proxy_p99_ms),
        )
        realized = sample_run3_simulator_outcome(
            q_perc=q_perc,
            p_complete_reassembly_given_sent=float(policy.p_complete_reassembly_given_sent),
            p_edge_admission_given_reassembled=float(policy.p_edge_admission_given_reassembled),
            latency_proxy=proxy,
            random_draws=self._counter_rng.draws(decision_key),
        )
        authenticated_d1 = EmpiricalTerminalTransitionV1.from_d1(
            collection_session_uuid=self._collection_session_uuid,
            collection_seq=seq,
            observation=observation,
            action=action,
            result=source,
            d1_binding=self.environment.binding,
        )
        transition = Run3TerminalTransitionV1.issue(
            collection_session_uuid=self._collection_session_uuid,
            collection_seq=seq,
            decision_key=decision_key,
            observation=observation,
            action=action,
            requested_q=requested_q,
            source_result=source,
            source_d1_transition=authenticated_d1,
            quality_query=query,
            quality_components=records,
            q_loc=q_loc,
            q_seg=q_seg,
            q_perc=q_perc,
            realized=realized,
            d1_environment_binding_sha256=self._d1_binding_sha256,
        )
        self.replay.insert(transition)
        self._transitions.append(transition)
        self._collection_seq += 1
        if warmup:
            self._warmup_collected += 1
        else:
            self._post_warmup_collected += 1
        if self.replay.evicted_count:
            raise Run3RunnerError("zero-eviction acceptance gate failed")

    def _ensure_trainer(self) -> Run3TerminalHybridSacTrainerV1:
        if self.trainer is None:
            if self.replay.binding is None:
                raise Run3RunnerError("cannot bind trainer before collection")
            trainer = Run3TerminalHybridSacTrainerV1(
                self.actor, self.critics, self.trainer_config,
                expected_binding=self.replay.binding,
                expected_batch_issuer_capability=(
                    self.replay.trainer_issuer_capability
                ),
                actor_generator=self._actor_update_rng,
            )
            trainer.actor_optimizer.load_state_dict(copy.deepcopy(self._pending_actor_optimizer_state))
            trainer.critic_optimizer.load_state_dict(copy.deepcopy(self._pending_critic_optimizer_state))
            trainer.update_count = len(self._metrics)
            self.trainer = trainer
        return self.trainer

    def run_until_updates(self, target_updates: int) -> Run3RunnerSummaryV1:
        self._require_open()
        if type(target_updates) is not int or not self.completed_updates <= target_updates <= self.config.update_count:
            raise Run3RunnerError("target update is invalid or goes backward")
        if target_updates == self.completed_updates:
            return self.summary()
        with _one_cpu_thread():
            while self._warmup_collected < self.config.warmup_transitions:
                self._collect_one(warmup=True)
            self._ensure_trainer()
            while self.completed_updates < target_updates:
                for _ in range(self.config.collect_per_update):
                    self._collect_one(warmup=False)
                batch = self.replay.sample(self.config.batch_size, generator=self._replay_rng)
                metric = self._ensure_trainer().update_once(batch)
                metric.assert_valid()
                self._metrics.append(metric)
        self._assert_process_isolation()
        return self.summary()

    def run(self) -> Run3RunnerSummaryV1:
        return self.run_until_updates(self.config.update_count)

    def _optimizer_states(self):
        if self.trainer is None:
            return copy.deepcopy(self._pending_actor_optimizer_state), copy.deepcopy(self._pending_critic_optimizer_state)
        return copy.deepcopy(self.trainer.actor_optimizer.state_dict()), copy.deepcopy(self.trainer.critic_optimizer.state_dict())

    def checkpoint(self) -> Run3CheckpointV1:
        self._require_open()
        actor_optimizer, critic_optimizer = self._optimizer_states()
        document = dict(
            config=self.config,
            seed=self.seed,
            runner_binding_document=copy.deepcopy(self._runner_binding_document),
            runner_binding_sha256=self._runner_binding_sha256,
            collection_session_uuid=self._collection_session_uuid,
            actor_state=copy.deepcopy(self.actor.state_dict()),
            critics_state=copy.deepcopy(self.critics.state_dict()),
            actor_optimizer_state=actor_optimizer,
            critic_optimizer_state=critic_optimizer,
            init_rng_state=copy.deepcopy(self._init_rng.getstate()),
            collection_rng_state=self._collection_rng.get_state().clone(),
            replay_rng_state=self._replay_rng.get_state().clone(),
            actor_update_rng_state=self._actor_update_rng.get_state().clone(),
            environment_state=copy.deepcopy(self.environment.state_dict()),
            transitions=tuple(self._transitions),
            metrics=tuple(self._metrics),
            replay_binding=self.replay.binding,
            collection_seq=self._collection_seq,
            warmup_collected=self._warmup_collected,
            post_warmup_collected=self._post_warmup_collected,
            update_count=self.completed_updates,
            replay_accepted_count=self.replay.accepted_count,
            replay_evicted_count=self.replay.evicted_count,
            excluded_fault_count=self._excluded_fault_count,
            trainer_initialized=self.trainer is not None,
            schema=RUN3_CHECKPOINT_SCHEMA,
        )
        result = Run3CheckpointV1(**document, checkpoint_sha256=_hash_state(document))
        result.require_valid()
        self._assert_process_isolation()
        return result

    def model_only_snapshot(self) -> Dict[str, Any]:
        if self.completed_updates % self.config.model_snapshot_interval != 0:
            raise Run3RunnerError("model snapshot requested off cadence")
        document = {
            "actor_state": copy.deepcopy(self.actor.state_dict()),
            "binding_sha256": self._runner_binding_sha256,
            "critic_1_state": copy.deepcopy(self.critics.critic_1.state_dict()),
            "critic_2_state": copy.deepcopy(self.critics.critic_2.state_dict()),
            "schema": RUN3_MODEL_SNAPSHOT_SCHEMA,
            "seed": self.seed,
            "update": self.completed_updates,
        }
        document["snapshot_sha256"] = _hash_state(document)
        return document

    def load_checkpoint(self, checkpoint: Run3CheckpointV1) -> None:
        self._require_open()
        if type(checkpoint) is not Run3CheckpointV1:
            raise Run3RunnerError("checkpoint has foreign type")
        checkpoint.require_valid()
        if checkpoint.config != self.config or checkpoint.seed != self.seed:
            raise Run3RunnerError("checkpoint schedule/seed mismatch")
        if checkpoint.runner_binding_sha256 != self._runner_binding_sha256:
            raise Run3RunnerError("checkpoint runner binding mismatch")
        if checkpoint.collection_session_uuid != self._collection_session_uuid:
            raise Run3RunnerError("checkpoint session mismatch")
        rebuilt = Run3TerminalReplayV1(
            self.config.replay_capacity, partition=self.environment._fit_partition
        )
        for row in checkpoint.transitions:
            rebuilt.insert(row)
        if rebuilt.evicted_count or rebuilt.accepted_count != checkpoint.collection_seq:
            raise Run3RunnerError("checkpoint cannot rebuild zero-eviction replay")
        staged_actor = copy.deepcopy(self.actor)
        staged_critics = copy.deepcopy(self.critics)
        staged_actor.load_state_dict(copy.deepcopy(checkpoint.actor_state), strict=True)
        staged_critics.load_state_dict(copy.deepcopy(checkpoint.critics_state), strict=True)
        collection_rng = torch.Generator(device="cpu")
        replay_rng = torch.Generator(device="cpu")
        actor_rng = torch.Generator(device="cpu")
        collection_rng.set_state(checkpoint.collection_rng_state.clone())
        replay_rng.set_state(checkpoint.replay_rng_state.clone())
        actor_rng.set_state(checkpoint.actor_update_rng_state.clone())
        init_rng = random.Random()
        init_rng.setstate(copy.deepcopy(checkpoint.init_rng_state))
        staged_trainer = None
        if checkpoint.trainer_initialized:
            if rebuilt.binding is None:
                raise Run3RunnerError("checkpoint trainer lacks replay binding")
            staged_trainer = Run3TerminalHybridSacTrainerV1(
                staged_actor, staged_critics, self.trainer_config,
                expected_binding=rebuilt.binding,
                expected_batch_issuer_capability=(
                    rebuilt.trainer_issuer_capability
                ),
                actor_generator=actor_rng,
            )
            staged_trainer.actor_optimizer.load_state_dict(copy.deepcopy(checkpoint.actor_optimizer_state))
            staged_trainer.critic_optimizer.load_state_dict(copy.deepcopy(checkpoint.critic_optimizer_state))
            staged_trainer.update_count = checkpoint.update_count
        self.environment.load_state_dict(checkpoint.environment_state)
        self.actor = staged_actor
        self.critics = staged_critics
        self._init_rng = init_rng
        self._collection_rng = collection_rng
        self._replay_rng = replay_rng
        self._actor_update_rng = actor_rng
        self.replay = rebuilt
        self.trainer = staged_trainer
        self._pending_actor_optimizer_state = copy.deepcopy(checkpoint.actor_optimizer_state)
        self._pending_critic_optimizer_state = copy.deepcopy(checkpoint.critic_optimizer_state)
        self._transitions = list(checkpoint.transitions)
        self._metrics = list(checkpoint.metrics)
        self._collection_seq = checkpoint.collection_seq
        self._warmup_collected = checkpoint.warmup_collected
        self._post_warmup_collected = checkpoint.post_warmup_collected
        self._excluded_fault_count = checkpoint.excluded_fault_count
        self._assert_process_isolation()

    @staticmethod
    def _parameter_delta(module: torch.nn.Module, initial: Mapping[str, Any]) -> float:
        total = 0.0
        for name, value in module.state_dict().items():
            reference = initial[name]
            if value.is_floating_point():
                delta = value.to(torch.float64) - reference.to(torch.float64)
                total += float(torch.sum(delta * delta))
        return math.sqrt(total)

    def summary(self) -> Run3RunnerSummaryV1:
        outcomes = [row.terminal_outcome.value for row in self._transitions]
        rewards = [row.reward for row in self._transitions]
        complete = self.completed_updates == self.config.update_count
        actor_delta = self._parameter_delta(self.actor, self._initial_actor)
        critic_1_delta = self._parameter_delta(self.critics.critic_1, self._initial_critic_1)
        critic_2_delta = self._parameter_delta(self.critics.critic_2, self._initial_critic_2)
        hard_gates = (
            complete
            and len(self._transitions) == self.config.total_transitions
            and self.replay.evicted_count == 0
            and self._excluded_fault_count == 0
            and actor_delta > 0.0 and critic_1_delta > 0.0 and critic_2_delta > 0.0
        )
        checkpoint = self.checkpoint()
        return Run3RunnerSummaryV1(
            schema=RUN3_SUMMARY_SCHEMA,
            seed=self.seed,
            configured_updates=self.config.update_count,
            completed_updates=self.completed_updates,
            transition_count=len(self._transitions),
            replay_resident_count=len(self.replay),
            replay_eviction_count=self.replay.evicted_count,
            excluded_fault_count=self._excluded_fault_count,
            reward_mean=None if not rewards else sum(rewards) / len(rewards),
            success_count=outcomes.count("SUCCESS_WITHIN_DEADLINE"),
            reassembly_failure_count=outcomes.count("REASSEMBLY_FAILURE"),
            admission_failure_count=outcomes.count("EDGE_ADMISSION_FAILURE"),
            service_timeout_count=outcomes.count("SIMULATED_SERVICE_TIMEOUT"),
            actor_parameter_delta_norm=actor_delta,
            critic_1_parameter_delta_norm=critic_1_delta,
            critic_2_parameter_delta_norm=critic_2_delta,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            completed_training_hard_gates_passed=hard_gates,
        )

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True
        self._assert_process_isolation()

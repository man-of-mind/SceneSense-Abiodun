"""Run-5 training mechanics: 22-D replay, Run-4 SAC update, event checkpoints, sidecars.

Reused from Run 4 without modification
--------------------------------------
* the SAC numerical update ``trainer._Run4TrainerCore.update_once`` (critic,
  actor, Polyak), inherited verbatim; only construction and preflight change
  width from 21 to 22, exactly as Run 4's own modeled trainer did;
* ``smoke_preregistration.FROZEN_CONFIG``: gamma 0.99/tensor, alpha_d 0.05,
  alpha_c 0.02, lr 3e-4, tau 0.005, batch 256, capacity 65,536, 4 environment
  transitions per update, 4 intra-op threads, seeds (17, 29, 43);
* ``RunnerSeedPlanV1`` stream derivation and the frozen 288-decision
  12x6x4 stratified warm-up schedule;
* stochastic training actions (categorical mode + conditional q from
  ``sample_all_modes``), identical to Run 4's ``_actor_request``;
* uniform-without-replacement replay sampling on a private generator, FIFO
  eviction and lifetime duplicate indexes.

Checkpoints
-----------
Every registered boundary emits (1) an event-sourced JSON checkpoint with a
full boundary fingerprint including the joint-channel state digest and (2) a
materialized sidecar (actor, online and target critics, both Adam states, all
five generators, and the joint-channel document).  Two resume paths are
provided and both must reproduce the fingerprint bit-for-bit: replay from
genesis, and the sidecar fast path (tensors and generators from the sidecar,
replay contents and channel state reconstructed from the ledger).

Importing performs no I/O and initializes no accelerator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import stat
from collections import deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_checkpoint_sidecar_v1 import sidecar as SC
from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as T
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT, Q_E4_MAX, Q_E4_MIN, Q_E4_SCALE, load_contract,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, TwinHybridCritics, build_actor,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ACTION_IDENTITY_SCHEMA_SHA256, MINIMUM_HOLD_TENSORS, ExecutedActionIdentity,
)

from . import run5_collector as RC
from . import run5_models as RM
from . import run5_snr_v2 as SNR

CONFIG = smoke_preregistration.FROZEN_CONFIG
WIDTH = 22
SCHEMA = "splitfusion.run5.training.v1"
CHECKPOINT_SCHEMA = "splitfusion.run5.event_checkpoint.v1"
SIDECAR_SCHEMA = "splitfusion.run5.materialized_checkpoint_sidecar.v1"
SMOKE_CHECKPOINTS = (0, 100, 250, 500)
SNR_INDEX = 21
_tree_sha256 = orch._tree_sha256
_tensor_sha256 = orch._tensor_sha256


class Run5TrainingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Run5TrainingError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False)
                          .encode("ascii")).hexdigest()


def trainer_config() -> T.TrainerConfigV1:
    return T.TrainerConfigV1(alpha_d=CONFIG.alpha_d, alpha_c=CONFIG.alpha_c,
                             tau=CONFIG.polyak_tau, actor_lr=CONFIG.actor_learning_rate,
                             critic_lr=CONFIG.critic_learning_rate,
                             nominal_batch_size=CONFIG.batch_size)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run5ReplayBatchV1:
    _state: torch.Tensor
    _next_state: torch.Tensor
    _mode_id: torch.Tensor
    _q_e4: torch.Tensor
    _reward: torch.Tensor
    _duration: torch.Tensor
    _discount: torch.Tensor
    _has_next_state: torch.Tensor
    _bootstrap: torch.Tensor
    _terminated: torch.Tensor
    _truncated: torch.Tensor
    binding_sha256: str

    float_dtype = torch.float32

    @property
    def batch_size(self) -> int:
        return int(self._state.shape[0])

    state = property(lambda self: self._state.clone())
    next_state = property(lambda self: self._next_state.clone())
    mode_id = property(lambda self: self._mode_id.clone())
    q_e4 = property(lambda self: self._q_e4.clone())
    reward = property(lambda self: self._reward.clone())
    duration = property(lambda self: self._duration.clone())
    has_next_state = property(lambda self: self._has_next_state.clone())
    bootstrap = property(lambda self: self._bootstrap.clone())
    terminated = property(lambda self: self._terminated.clone())
    truncated = property(lambda self: self._truncated.clone())

    @property
    def q_normalized_executed(self) -> torch.Tensor:
        return self._q_e4.to(torch.float32) / float(Q_E4_MAX)

    def discount(self) -> torch.Tensor:
        return self._discount.clone()


class Run5ReplayBufferV1:
    def __init__(self, capacity: int, binding_sha256: str) -> None:
        require(type(capacity) is int and capacity > 0, "capacity must be > 0")
        self.capacity = capacity
        self.binding_sha256 = binding_sha256
        self._rows: deque[RC.Run5CollectedTransitionV1] = deque()
        self._seen_digests: set[str] = set()
        self._seen_identities: dict[tuple[str, int], str] = {}
        self.accepted_count = 0
        self.evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    def insert(self, record: RC.Run5CollectedTransitionV1) -> None:
        require(type(record) is RC.Run5CollectedTransitionV1, "replay accepts Run-5 records only")
        digest = record.digest
        key = (record.session_uuid, record.decision_seq)
        require(digest not in self._seen_digests, "duplicate transition digest")
        require(key not in self._seen_identities, "duplicate logical decision identity")
        require(record.duration >= MINIMUM_HOLD_TENSORS, "duration violates minimum hold")
        for value in (record.reward, record.discount):
            require(math.isfinite(float(torch.tensor(value, dtype=torch.float32))),
                    "reward/discount is not finite in float32")
        self._seen_digests.add(digest)
        self._seen_identities[key] = digest
        self._rows.append(record)
        self.accepted_count += 1
        while len(self._rows) > self.capacity:
            self._rows.popleft()
            self.evicted_count += 1

    def resident_digests(self) -> Tuple[str, ...]:
        return tuple(row.digest for row in self._rows)

    def sample(self, batch_size: int, generator: torch.Generator) -> Run5ReplayBatchV1:
        require(isinstance(generator, torch.Generator)
                and generator is not torch.default_generator
                and generator.device.type == "cpu", "sampling needs a private CPU generator")
        require(0 < batch_size <= len(self._rows), "cannot sample that many rows")
        permutation = torch.randperm(len(self._rows), generator=generator, device="cpu")
        rows = [self._rows[int(i)] for i in permutation[:batch_size]]
        next_state = torch.zeros((len(rows), WIDTH), dtype=torch.float32)
        for index, row in enumerate(rows):
            if row.next_state is not None:
                next_state[index] = torch.tensor(row.next_state, dtype=torch.float32)
        return Run5ReplayBatchV1(
            _state=torch.tensor([row.state for row in rows], dtype=torch.float32),
            _next_state=next_state,
            _mode_id=torch.tensor([row.mode_id for row in rows], dtype=torch.int64),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor([row.reward for row in rows], dtype=torch.float32),
            _duration=torch.tensor([row.duration for row in rows], dtype=torch.int64),
            _discount=torch.tensor([row.discount for row in rows], dtype=torch.float32),
            _has_next_state=torch.tensor([row.has_next_state for row in rows], dtype=torch.bool),
            _bootstrap=torch.tensor([row.bootstrap for row in rows], dtype=torch.bool),
            _terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
            _truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
            binding_sha256=self.binding_sha256)


# ---------------------------------------------------------------------------
# Trainer: Run-4 update_once, Run-5 construction/preflight
# ---------------------------------------------------------------------------


class Run5HybridSacTrainerV1(T._Run4TrainerCore):
    def __init__(self, *, actor: ConditionalHybridActor, critics: TwinHybridCritics,
                 config: T.TrainerConfigV1, binding_sha256: str,
                 target_generator: torch.Generator, actor_generator: torch.Generator) -> None:
        if type(config) is not T.TrainerConfigV1:
            raise T.TrainerStateError("config must be an exact TrainerConfigV1")
        RM.validate_run5_models(actor, critics)
        for name, generator in (("target_generator", target_generator),
                                ("actor_generator", actor_generator)):
            if (not isinstance(generator, torch.Generator)
                    or generator is torch.default_generator or generator.device.type != "cpu"):
                raise T.TrainerStateError(f"{name} must be a private CPU generator")
        if target_generator is actor_generator:
            raise T.TrainerStateError("target and actor generators must be distinct")
        self.actor = actor
        self.critics = critics
        self.config = config
        self.expected_binding = binding_sha256
        self._target_generator = target_generator
        self._actor_generator = actor_generator
        self._online_critic_parameters = list(
            chain(critics.critic_1.parameters(), critics.critic_2.parameters()))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._online_critic_parameters,
                                                 lr=config.critic_lr)
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _require_evidence_class(self, binding: Any) -> None:
        if binding != self.expected_binding:
            raise T.TrainerPreflightError("batch binding differs from the trainer binding")

    def _preflight(self, batch: Any) -> None:
        error = T.TrainerPreflightError
        self._assert_optimizer_wiring()
        RM.validate_run5_models(self.actor, self.critics)
        if type(batch) is not Run5ReplayBatchV1:
            raise error("update_once requires an exact Run5ReplayBatchV1")
        self._require_evidence_class(batch.binding_sha256)
        size = batch.batch_size
        if size < 1:
            raise error("batch must not be empty")
        for name, tensor, shape in (
                ("state", batch.state, (size, WIDTH)),
                ("next_state", batch.next_state, (size, WIDTH)),
                ("reward", batch.reward, (size,)), ("discount", batch.discount(), (size,)),
                ("q_normalized_executed", batch.q_normalized_executed, (size,))):
            if tuple(tensor.shape) != shape or tensor.dtype is not torch.float32:
                raise error(f"{name} must be float32 with shape {shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise error(f"{name} contains a non-finite value")
        has_next, bootstrap = batch.has_next_state, batch.bootstrap
        terminated, truncated = batch.terminated, batch.truncated
        if not bool(torch.equal(bootstrap, has_next & ~terminated & ~truncated)):
            raise error("bootstrap is not an eligible real-successor mask")
        if not bool(torch.equal(has_next, ~(terminated | truncated))):
            raise error("successor presence disagrees with episode boundary")
        absent = (~has_next).nonzero(as_tuple=False).squeeze(1)
        if int(absent.numel()) and bool(batch.next_state.index_select(0, absent).abs().sum() > 0):
            raise error("rows without successors must retain the zero sentinel")
        if bool((batch.mode_id < 0).any()) or bool((batch.mode_id >= EXPECTED_MODE_COUNT).any()):
            raise error("mode_id is outside the 12-mode action catalog")
        if bool((batch.q_e4 < Q_E4_MIN).any()) or bool((batch.q_e4 > Q_E4_MAX).any()):
            raise error("q_e4 is outside the registered execution range")
        if bool((batch.duration < MINIMUM_HOLD_TENSORS).any()):
            raise error("duration violates the minimum action hold")
        discount = batch.discount()
        if bool((discount <= 0.0).any()) or bool((discount > 1.0).any()):
            raise error("stored discount must lie in (0, 1]")


# ---------------------------------------------------------------------------
# Event checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run5CheckpointV1:
    document: Mapping[str, Any]

    @property
    def update_count(self) -> int:
        return int(self.document["update_count"])

    @property
    def decision_count(self) -> int:
        return int(self.document["decision_count"])

    @property
    def boundary(self) -> Mapping[str, Any]:
        return self.document["boundary"]

    @property
    def canonical_sha256(self) -> str:
        return _sha(dict(self.document))

    def to_bytes(self) -> bytes:
        return json.dumps(dict(self.document), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")


def write_event_checkpoint(path: Path, checkpoint: Run5CheckpointV1) -> str:
    data = checkpoint.to_bytes()
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def read_event_checkpoint(path: Path) -> Run5CheckpointV1:
    data = Path(path).read_bytes()
    document = json.loads(data)
    require(document.get("schema") == CHECKPOINT_SCHEMA,
            "not a Run-5 event checkpoint (a Run-4 checkpoint is refused)")
    checkpoint = Run5CheckpointV1(document)
    require(checkpoint.to_bytes() == data, "event checkpoint is not canonical")
    return checkpoint


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Run5OrchestratorV1:
    def __init__(self, *, collector_factory: Callable[[], RC.Run5ModeledCollectorV1],
                 seed: int, checkpoint_updates: Sequence[int] = SMOKE_CHECKPOINTS) -> None:
        require(torch.get_num_threads() == CONFIG.torch_intraop_threads,
                "torch intra-op threads must be exactly 4")
        self.seed_plan = orch.RunnerSeedPlanV1.for_registered_seed(seed)
        self.schedule = orch.build_frozen_warmup_schedule(seed)
        self.checkpoint_updates = tuple(checkpoint_updates)
        require(set(self.checkpoint_updates) <= set(CONFIG.checkpoint_updates),
                "checkpoint updates must be registered Run-4 boundaries")
        self.collector_factory = collector_factory
        self.collector = collector_factory()
        require(type(self.collector) is RC.Run5ModeledCollectorV1, "collector is foreign")
        require(self.collector.decision_count == 0, "collector is not at genesis")
        plan = self.seed_plan
        self.actor, self.critics = RM.build_run5_models(actor_seed=plan.actor_seed,
                                                       critic_seed=plan.critic_seed)
        self.binding_document = {
            "schema": SCHEMA, "model_binding_sha256": RM.RUN5_V2_MODEL_BINDING_SHA256,
            "feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256,
            "collector_binding_sha256": self.collector.collector_binding_sha256,
            "preregistration_sha256": smoke_preregistration.PREREGISTRATION_SHA256,
            "gamma_per_tensor": CONFIG.gamma_per_tensor, "capacity": CONFIG.replay_capacity,
            "trainer_config": {"alpha_d": CONFIG.alpha_d, "alpha_c": CONFIG.alpha_c,
                               "tau": CONFIG.polyak_tau, "lr": CONFIG.actor_learning_rate,
                               "batch_size": CONFIG.batch_size},
            "seed_plan": plan.to_dict(), "schedule_id": self.schedule.config.schedule_id,
            "action_identity_schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
        }
        self.binding_sha256 = _sha(self.binding_document)
        self.replay = Run5ReplayBufferV1(CONFIG.replay_capacity, self.binding_sha256)
        self.generators = {}
        for name, value in (("decision_q", plan.decision_q_seed),
                            ("decision_mode", plan.decision_mode_seed),
                            ("replay", plan.replay_seed), ("trainer_target", plan.target_seed),
                            ("trainer_actor", plan.trainer_actor_seed)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(value)
            self.generators[name] = generator
        self.trainer = Run5HybridSacTrainerV1(
            actor=self.actor, critics=self.critics, config=trainer_config(),
            binding_sha256=self.binding_sha256,
            target_generator=self.generators["trainer_target"],
            actor_generator=self.generators["trainer_actor"])
        self._catalog = load_contract()
        self.ledger: list[dict[str, Any]] = []
        self.history: list[RC.Run5CollectedTransitionV1] = []
        self.preflight: Optional[dict[str, Any]] = None
        self.metrics: list[T.UpdateMetricsV1] = []
        self._model_before = self._model_sha256()

    @property
    def decision_count(self) -> int:
        return len(self.ledger)

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    def _model_sha256(self) -> str:
        return _sha({"actor": _tree_sha256(self.actor.state_dict()),
                     "critics": _tree_sha256(self.critics.state_dict()),
                     "actor_opt": _tree_sha256(self.trainer.actor_optimizer.state_dict()),
                     "critic_opt": _tree_sha256(self.trainer.critic_optimizer.state_dict())})

    # -- action selection (identical to Run 4) --------------------------
    def _execution(self, mode_id: int, q_e4: int) -> None:
        executable = self._catalog.resolve(mode_id, q_e4 / float(Q_E4_SCALE))
        action = ExecutedActionIdentity.from_executable_action(executable, self._catalog)
        require((action.mode_id, action.q_e4) == (mode_id, q_e4), "catalog changed the action")

    def _warmup_request(self, ordinal: int) -> orch.ModeledActionRequestV1:
        selected = self.schedule.action_at(ordinal)
        return orch.ModeledActionRequestV1(
            decision_ordinal=ordinal, mode_id=selected.mode_id, q_e4=selected.q_e4,
            source="STRATIFIED_WARMUP", warmup_q_bin_index=selected.q_bin_index)

    def _actor_request(self, ordinal: int, state: Tuple[float, ...]) -> orch.ModeledActionRequestV1:
        tensor = torch.tensor((state,), dtype=torch.float32)
        with torch.no_grad():
            sample = self.actor.sample_all_modes(tensor, generator=self.generators["decision_q"])
            mode_id = int(torch.multinomial(sample.probs[0], 1,
                                            generator=self.generators["decision_mode"])[0])
            q_e4 = int(sample.q_e4[0, mode_id])
        self._execution(mode_id, q_e4)
        return orch.ModeledActionRequestV1(decision_ordinal=ordinal, mode_id=mode_id,
                                           q_e4=q_e4, source="STOCHASTIC_ACTOR",
                                           warmup_q_bin_index=None)

    def _next_request(self) -> orch.ModeledActionRequestV1:
        ordinal = self.decision_count
        if ordinal < len(self.schedule):
            return self._warmup_request(ordinal)
        return self._actor_request(ordinal, self.collector.current_state_features())

    def collect_one(self) -> RC.Run5CollectedTransitionV1:
        request = self._next_request()
        record = self.collector.collect(request)
        require(record.request == request, "collector substituted the request")
        self.replay.insert(record)
        self.history.append(record)
        self.ledger.append({"request": request.to_dict(), "digest": record.digest})
        require(self.collector.decision_count == self.decision_count, "count mismatch")
        return record

    # -- preflight ------------------------------------------------------
    def run_preflight(self) -> dict[str, Any]:
        require(self.decision_count == 0 and self.update_count == 0, "preflight runs at genesis")
        for _ in range(len(self.schedule)):
            self.collect_one()
        report = preflight_report(self.history, self.collector.diagnostics())
        require(self._model_sha256() == self._model_before,
                "models/optimizers changed during the no-gradient warm-up")
        require(report["passed"], f"preflight failed: {report['failures']}")
        self.preflight = report
        return report

    # -- boundary / checkpoint ------------------------------------------
    def boundary(self) -> dict[str, Any]:
        collector_checkpoint = self.collector.checkpoint()
        return {
            "update_count": self.update_count, "decision_count": self.decision_count,
            "actor_sha256": _tree_sha256(self.actor.state_dict()),
            "critics_sha256": _tree_sha256(self.critics.state_dict()),
            "actor_optimizer_sha256": _tree_sha256(self.trainer.actor_optimizer.state_dict()),
            "critic_optimizer_sha256": _tree_sha256(self.trainer.critic_optimizer.state_dict()),
            **{f"{name}_rng_sha256": _tensor_sha256(generator.get_state())
               for name, generator in sorted(self.generators.items())},
            "replay_resident_sha256": _sha(list(self.replay.resident_digests())),
            "replay_accepted_count": self.replay.accepted_count,
            "replay_evicted_count": self.replay.evicted_count,
            "collector_checkpoint_sha256": _sha(collector_checkpoint.to_dict()),
            "joint_channel_sha256": self.collector._channel.checkpoint_sha256(),
            "ledger_sha256": _sha(self.ledger),
        }

    def checkpoint(self) -> Run5CheckpointV1:
        require(self.preflight is not None, "preflight has not passed")
        require(self.decision_count == len(self.schedule)
                + CONFIG.environment_transitions_per_update * self.update_count,
                "checkpoint ratio differs")
        return Run5CheckpointV1({
            "schema": CHECKPOINT_SCHEMA, "binding": self.binding_document,
            "binding_sha256": self.binding_sha256, "seed": self.seed_plan.master_seed,
            "update_count": self.update_count, "decision_count": self.decision_count,
            "ledger": list(self.ledger),
            "collector_checkpoint": self.collector.checkpoint().to_dict(),
            "preflight_sha256": _sha(self.preflight), "preflight": self.preflight,
            "boundary": self.boundary()})

    def train_once(self) -> T.UpdateMetricsV1:
        batch = self.replay.sample(CONFIG.batch_size, self.generators["replay"])
        metrics = self.trainer.update_once(batch)
        metrics.require_finite()
        self.metrics.append(metrics)
        return metrics

    def run_to(self, target_update: int, *,
               checkpoint_callback: Callable[[Run5CheckpointV1], None],
               emit_current: bool = False) -> None:
        require(target_update in self.checkpoint_updates, "target is not a checkpoint boundary")
        require(target_update >= self.update_count, "target precedes current state")
        if self.preflight is None:
            self.run_preflight()
        if emit_current and self.update_count in self.checkpoint_updates:
            checkpoint_callback(self.checkpoint())
        while self.update_count < target_update:
            before = self.update_count
            for _ in range(CONFIG.environment_transitions_per_update):
                self.collect_one()
            self.train_once()
            require(self.update_count == before + 1, "one loop did not make one update")
            if self.update_count in self.checkpoint_updates:
                checkpoint_callback(self.checkpoint())

    # -- restore -------------------------------------------------------
    @classmethod
    def _fresh_with_history(cls, checkpoint: Run5CheckpointV1, collector_factory,
                            checkpoint_updates) -> "Run5OrchestratorV1":
        document = checkpoint.document
        require(document["schema"] == CHECKPOINT_SCHEMA, "foreign checkpoint")
        candidate = cls(collector_factory=collector_factory, seed=int(document["seed"]),
                        checkpoint_updates=checkpoint_updates)
        require(candidate.binding_sha256 == document["binding_sha256"],
                "checkpoint binding differs from this configuration")
        candidate.collector.restore(RC.Run5CollectorCheckpointV1.from_dict(
            document["collector_checkpoint"]))
        history = candidate.collector.history()
        require(len(history) == checkpoint.decision_count, "collector history incomplete")
        return candidate

    @classmethod
    def restore_by_replay(cls, checkpoint: Run5CheckpointV1, *, collector_factory,
                          checkpoint_updates=SMOKE_CHECKPOINTS) -> "Run5OrchestratorV1":
        """Event-sourced restore: replay actions and updates from genesis."""
        candidate = cls._fresh_with_history(checkpoint, collector_factory, checkpoint_updates)
        history = candidate.collector.history()
        for ordinal, (record, row) in enumerate(zip(history, checkpoint.document["ledger"])):
            if ordinal < len(candidate.schedule):
                expected = candidate._warmup_request(ordinal)
            else:
                expected = candidate._actor_request(ordinal, record.state)
            require(expected.to_dict() == row["request"], "replayed policy action differs")
            require(record.digest == row["digest"], "replayed transition differs")
            candidate.replay.insert(record)
            candidate.history.append(record)
            candidate.ledger.append(row)
            completed = ordinal + 1 - len(candidate.schedule)
            if completed > 0 and completed % CONFIG.environment_transitions_per_update == 0:
                candidate.train_once()
        candidate.preflight = checkpoint.document["preflight"]
        require(candidate.boundary() == dict(checkpoint.boundary),
                "event-sourced restore is not bit-identical")
        return candidate

    @classmethod
    def restore_from_sidecar(cls, checkpoint: Run5CheckpointV1, sidecar_dir: Path, *,
                             collector_factory,
                             checkpoint_updates=SMOKE_CHECKPOINTS) -> "Run5OrchestratorV1":
        """Fast path: sidecar tensors/optimizers/generators + ledger-rebuilt replay."""
        material = read_sidecar(sidecar_dir, checkpoint)
        candidate = cls._fresh_with_history(checkpoint, collector_factory, checkpoint_updates)
        history = candidate.collector.history()
        for record, row in zip(history, checkpoint.document["ledger"]):
            require(record.digest == row["digest"], "rebuilt transition differs")
            candidate.replay.insert(record)
            candidate.history.append(record)
            candidate.ledger.append(row)
        require(candidate.collector.channel_checkpoint() == material["joint_channel"],
                "rebuilt joint-channel state differs from the sidecar")
        candidate.actor.load_state_dict(material["actor"], strict=True)
        candidate.critics.load_state_dict({**material["online_critics"],
                                           **material["target_critics"]}, strict=True)
        candidate.trainer.actor_optimizer.load_state_dict(material["actor_optimizer"])
        candidate.trainer.critic_optimizer.load_state_dict(material["critic_optimizer"])
        for name, state in material["generators"].items():
            candidate.generators[name].set_state(state)
        candidate.trainer.update_count = checkpoint.update_count
        candidate.preflight = checkpoint.document["preflight"]
        require(candidate.boundary() == dict(checkpoint.boundary),
                "sidecar resume is not bit-identical to the event boundary")
        return candidate


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

PREVIOUS_SLICE = slice(4, 21)


def preflight_report(history: Sequence[RC.Run5CollectedTransitionV1],
                     diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    strata = {}
    for record in history:
        key = (record.mode_id, record.request.warmup_q_bin_index)
        strata[key] = strata.get(key, 0) + 1
    if len(strata) != 72 or set(strata.values()) != {4}:
        failures.append("warm-up is not exactly 12 modes x 6 q strata x 4")
    terminals = {record.terminal for record in history}
    if "SUCCESS" not in terminals or len(terminals) < 2:
        failures.append("warm-up lacks both success and failure/timeout")
    columns = {name: [record.state[i] for record in history]
               for name, i in (("camera_si", 0), ("radar_p40", 1), ("mcs", 2), ("backlog", 3),
                               ("snr", SNR_INDEX), ("prev_q", 16), ("prev_quality", 17),
                               ("prev_latency", 18), ("prev_present", 19), ("prev_success", 20))}
    variation = {name: {"distinct": len(set(values)), "span": max(values) - min(values)}
                 for name, values in columns.items()}
    for name, item in variation.items():
        if item["distinct"] < 2 or not item["span"] > 0:
            failures.append(f"{name} does not vary")
    mismatches = 0
    for previous, record in zip(history, history[1:]):
        prev = record.state[PREVIOUS_SLICE]
        one_hot = [0.0] * EXPECTED_MODE_COUNT
        one_hot[previous.mode_id] = 1.0
        success = previous.terminal == "SUCCESS"
        expected = (*one_hot, previous.q_e4 / float(Q_E4_MAX),
                    float(previous.q_perc) if success else 0.0,
                    float(previous.latency_ms) / 170.0 if success else 0.0, 1.0,
                    1.0 if success else 0.0)
        mismatches += int(tuple(prev) != tuple(float(v) for v in expected))
    if mismatches:
        failures.append(f"{mismatches} previous-outcome encodings differ from the prior record")
    future = sum(1 for d in diagnostics if not d["generated_ticks_after_observed"])
    if future:
        failures.append("a generated SNR tick was not after the observed tick")
    return {"decisions": len(history), "strata": len(strata),
            "terminals": sorted(terminals), "variation": variation,
            "previous_outcome_mismatches": mismatches, "future_tick_violations": future,
            "failures": failures, "passed": not failures}


# ---------------------------------------------------------------------------
# Materialized sidecar (Run-5 22-D; primitives reused from the Run-4 sidecar)
# ---------------------------------------------------------------------------

GENERATOR_NAMES = SC.GENERATOR_NAMES
ARTIFACT_NAMES = SC.ARTIFACT_NAMES


def _fixtures() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260929)
    rows = torch.rand((20, WIDTH), generator=generator, dtype=torch.float32)
    rows[:, 4:16] = 0.0
    for index in range(20):
        rows[index, 4 + index % EXPECTED_MODE_COUNT] = 1.0
    rows[:, 19] = 1.0
    rows[:, 20] = (torch.arange(20) % 2).to(torch.float32)
    return rows


def actor_fixture_outputs(actor: ConditionalHybridActor) -> list[dict[str, Any]]:
    with torch.no_grad():
        execution = actor.deterministic_execution(_fixtures())
        logits = actor(_fixtures()).logits
    return [{"mode_id": int(execution.mode_index[i]), "q_e4": int(execution.q_e4[i]),
             "logits_sha256": _tensor_sha256(logits[i].contiguous())} for i in range(20)]


def capture_material(orchestrator: Run5OrchestratorV1) -> dict[str, Any]:
    online, target = SC._split_critics(orchestrator.critics.state_dict())
    return {
        "actor": SC._cpu_tree(orchestrator.actor.state_dict()),
        "online_critics": SC._cpu_tree(online), "target_critics": SC._cpu_tree(target),
        "actor_optimizer": SC._cpu_tree(orchestrator.trainer.actor_optimizer.state_dict()),
        "critic_optimizer": SC._cpu_tree(orchestrator.trainer.critic_optimizer.state_dict()),
        "generators": {name: orchestrator.generators[name].get_state().clone()
                       for name in GENERATOR_NAMES},
    }


def _boundary_view(material: Mapping[str, Any]) -> dict[str, str]:
    return {
        "actor_sha256": _tree_sha256(material["actor"]),
        "critics_sha256": _tree_sha256({**material["online_critics"],
                                        **material["target_critics"]}),
        "actor_optimizer_sha256": _tree_sha256(material["actor_optimizer"]),
        "critic_optimizer_sha256": _tree_sha256(material["critic_optimizer"]),
        **{f"{name}_rng_sha256": _tensor_sha256(material["generators"][name])
           for name in GENERATOR_NAMES},
    }


def write_sidecar(directory: Path, orchestrator: Run5OrchestratorV1,
                  checkpoint: Run5CheckpointV1) -> str:
    target = Path(directory)
    require(not (target.exists() or target.is_symlink()), "sidecar target already exists")
    material = capture_material(orchestrator)
    for name, digest in _boundary_view(material).items():
        require(checkpoint.boundary[name] == digest, f"{name} differs from the event boundary")
    channel = orchestrator.collector.channel_checkpoint()
    require(_sha(channel) == checkpoint.boundary["joint_channel_sha256"],
            "joint-channel state differs from the event boundary")
    staging = target.parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        staging.mkdir(mode=0o700)
        artifacts = {}
        for name in ARTIFACT_NAMES:
            path = staging / SC.ARTIFACT_FILENAMES[name]
            SC._write_create_only(path, lambda stream, n=name: torch.save(material[n], stream))
            digest, size = checkpoint_io._sha256_file(path)
            artifacts[name] = {"filename": SC.ARTIFACT_FILENAMES[name], "sha256": digest,
                               "size_bytes": size, "tree_sha256": _tree_sha256(material[name]),
                               "tensors": SC._tensor_leaves(material[name])}
        manifest = {
            "schema_id": SIDECAR_SCHEMA, "loader": SC.LOADER,
            "identity": {"seed": orchestrator.seed_plan.master_seed,
                         "seed_plan": orchestrator.seed_plan.to_dict(),
                         "update_count": checkpoint.update_count,
                         "decision_count": checkpoint.decision_count,
                         "event_checkpoint_sha256": checkpoint.canonical_sha256,
                         "boundary": dict(checkpoint.boundary)},
            "schema_identity": {"model_binding": dict(RM.RUN5_V2_MODEL_BINDING)
                                | {"policy_feature_order":
                                   list(RM.RUN5_V2_MODEL_BINDING["policy_feature_order"])},
                                "model_binding_sha256": RM.RUN5_V2_MODEL_BINDING_SHA256,
                                "feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256,
                                "action_identity_schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
                                "event_checkpoint_schema": CHECKPOINT_SCHEMA},
            "artifacts": artifacts, "generator_names": list(GENERATOR_NAMES),
            "joint_channel": channel,
            "actor_fixtures": actor_fixture_outputs(orchestrator.actor),
            "scope": "tensors, optimizers, generators and joint-channel state at the "
                     "event boundary; replay contents are rebuilt from the ledger",
        }
        manifest_bytes = checkpoint_io._canonical_json_bytes(manifest)
        SC._write_create_only(staging / SC.MANIFEST_FILENAME,
                              lambda stream: stream.write(manifest_bytes))
        SC._fsync_directory(staging)
        require(not (target.exists() or target.is_symlink()), "sidecar target appeared")
        os.rename(staging, target)
        SC._fsync_directory(target.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return checkpoint_io._sha256_bytes(manifest_bytes)


def read_sidecar(directory: Path, checkpoint: Run5CheckpointV1) -> dict[str, Any]:
    root = Path(directory)
    manifest_bytes = (root / SC.MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schema_id") == SIDECAR_SCHEMA,
            "not a Run-5 sidecar (Run-4 21-D sidecars are refused)")
    require(checkpoint_io._canonical_json_bytes(manifest) == manifest_bytes,
            "sidecar manifest is not canonical")
    require(manifest["identity"]["event_checkpoint_sha256"] == checkpoint.canonical_sha256,
            "sidecar names a different event checkpoint")
    require(manifest["schema_identity"]["model_binding_sha256"]
            == RM.RUN5_V2_MODEL_BINDING_SHA256, "sidecar model binding is not Run-5 v2")
    names = {item.name for item in root.iterdir()}
    require(names == {SC.MANIFEST_FILENAME, *SC.ARTIFACT_FILENAMES.values()},
            "sidecar member set differs")
    material: dict[str, Any] = {}
    for name in ARTIFACT_NAMES:
        path = root / SC.ARTIFACT_FILENAMES[name]
        require(stat.S_ISREG(path.lstat().st_mode), f"{name} is not a regular file")
        digest, size = checkpoint_io._sha256_file(path)
        entry = manifest["artifacts"][name]
        require(digest == entry["sha256"] and size == entry["size_bytes"], f"{name} tampered")
        material[name] = torch.load(path, map_location="cpu", weights_only=True)
        require(_tree_sha256(material[name]) == entry["tree_sha256"], f"{name} tree differs")
    for name, digest in _boundary_view(material).items():
        require(checkpoint.boundary[name] == digest, f"sidecar {name} differs from boundary")
    material["joint_channel"] = manifest["joint_channel"]
    require(_sha(material["joint_channel"]) == checkpoint.boundary["joint_channel_sha256"],
            "sidecar joint-channel state differs from boundary")
    material["manifest"] = manifest
    return material


def cold_load_actor(directory: Path, checkpoint: Run5CheckpointV1) -> ConditionalHybridActor:
    """Fresh 22-D actor from the sidecar alone; refuses any 21-D state."""
    material = read_sidecar(directory, checkpoint)
    critics = {**material["online_critics"], **material["target_critics"]}
    actor, _ = RM.load_run5_model_state(
        binding=RM.RUN5_V2_MODEL_BINDING, actor_state=material["actor"],
        critic_state=critics, expected_binding_sha256=RM.RUN5_V2_MODEL_BINDING_SHA256)
    require(actor_fixture_outputs(actor) == material["manifest"]["actor_fixtures"],
            "cold-loaded actor fixtures differ from the manifest")
    return actor

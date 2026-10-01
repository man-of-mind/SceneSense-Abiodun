"""Run-5B training mechanics: 21-D replay, Run-4 SAC update, atomic boundary bundles.

Reused from Run 4 without modification
--------------------------------------
* the SAC numerical update ``trainer._Run4TrainerCore.update_once`` (critic,
  actor, Polyak), inherited verbatim; construction and preflight are Run-5B's,
  the pattern Run 4's own modeled trainer used;
* the stratified warm-up schedule class and the registered continuous-q
  action support (the action space is unchanged);
* the Run-5 RNG stream derivation (same per-seed decision/replay/trainer
  streams, so the comparison with Run 5 differs only in state and network);
* stochastic training actions (categorical mode + conditional q from
  ``sample_all_modes``), identical to Run 4's ``_actor_request``;
* uniform-without-replacement replay sampling on a private generator, FIFO
  eviction and lifetime duplicate indexes.

Every number (gamma, alphas, learning rates, tau, batch, capacity, warm-up,
transitions per update, seeds, checkpoint cadence) comes from the Run-5B
preregistration and equals the Run-4/Run-5 value; no Run-4 or Run-5
preregistration identity is bound.

Checkpoints and resume
----------------------
A boundary bundle (``run5b_bundle``) holds the event record, the actor, both
online and target critics, both Adam states, all five private generators, the
joint-channel state and every identity.  ``restore_from_bundle`` performs no
gradient step: tensors, optimizers and generators are loaded from the bundle;
the collector (environment, queue, scene streams, channel) and the replay
contents are rebuilt by re-executing the recorded *actions* through the
environment, which is deterministic and verified against every digest in the
bundle.  Restoration first applies the exact Run-5B identity checks of
``run5b_models`` (old Run-4 21-D and Run-5 22-D identities are refused).

Importing performs no I/O and initializes no accelerator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import exploration
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as T
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT, Q_E4_MAX, Q_E4_MIN, Q_E4_SCALE, load_contract,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, TwinHybridCritics,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ACTION_IDENTITY_SCHEMA_SHA256, MINIMUM_HOLD_TENSORS, ExecutedActionIdentity,
)

from . import run5b_bundle as B
from . import run5b_collector as RC
from . import run5b_models as RM
from . import run5b_preregistration as PR
from . import run5b_state_contract as C

CONFIG = PR.CONFIG
WIDTH = C.RUN5B_POLICY_FEATURE_COUNT
SCHEMA = "splitfusion.run5b.training.v1"
EVENT_SCHEMA = "splitfusion.run5b.event_record.v1"
TRAINING_STATE_SCHEMA = "splitfusion.run5b.training_state.v1"
SNR_INDEX = C.SNR_FEATURE_INDEX
GENERATOR_NAMES = ("decision_q", "decision_mode", "replay", "trainer_target", "trainer_actor")
ONLINE_PREFIXES = ("critic_1.", "critic_2.")
TARGET_PREFIXES = ("target_1.", "target_2.")
_tree_sha256 = orch._tree_sha256
_tensor_sha256 = orch._tensor_sha256


class Run5BTrainingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Run5BTrainingError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(B.canonical_bytes(value)).hexdigest()


def trainer_config() -> T.TrainerConfigV1:
    return T.TrainerConfigV1(alpha_d=CONFIG.alpha_d, alpha_c=CONFIG.alpha_c,
                             tau=CONFIG.polyak_tau, actor_lr=CONFIG.actor_learning_rate,
                             critic_lr=CONFIG.critic_learning_rate,
                             nominal_batch_size=CONFIG.batch_size)


def derive_seed(master: int, label: str) -> int:
    material = B.canonical_bytes({"domain": "RUN5_TRAINING_RNG_STREAM_V1", "label": label,
                                  "master_seed": master})
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def seed_plan(master: int) -> dict[str, int]:
    require(master in CONFIG.seed_order, "seed is not registered in the Run-5B preregistration")
    plan = {"master_seed": master}
    for name in ("actor", "critics", *GENERATOR_NAMES):
        plan[f"{name}_seed"] = derive_seed(master, name)
    require(len(set(plan.values()) - {master}) == len(plan) - 1, "RNG stream seeds collide")
    return plan


def warmup_schedule(master: int) -> exploration.StratifiedWarmupSchedule:
    schedule = exploration.StratifiedWarmupSchedule(exploration.WarmupScheduleConfig(
        mode_q_e4_bounds=MODELED_SMOKE_SUPPORT.mode_q_e4_bounds,
        q_bin_count=CONFIG.warmup_q_bin_count,
        samples_per_q_bin=CONFIG.warmup_samples_per_mode_q_bin,
        master_seed=master, support_contract_id=MODELED_SMOKE_SUPPORT_SHA256))
    require(len(schedule) == CONFIG.warmup_decision_count, "warm-up is not 288 decisions")
    return schedule


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run5BReplayBatchV1:
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


class Run5BReplayBufferV1:
    def __init__(self, capacity: int, binding_sha256: str) -> None:
        require(type(capacity) is int and capacity > 0, "capacity must be > 0")
        self.capacity = capacity
        self.binding_sha256 = binding_sha256
        self._rows: deque[RC.Run5BCollectedTransitionV1] = deque()
        self._seen_digests: set[str] = set()
        self._seen_identities: dict[tuple[str, int], str] = {}
        self.accepted_count = 0
        self.evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    def insert(self, record: RC.Run5BCollectedTransitionV1) -> None:
        require(type(record) is RC.Run5BCollectedTransitionV1, "replay accepts Run-5B records only")
        digest = record.digest
        key = (record.session_uuid, record.decision_seq)
        require(digest not in self._seen_digests, "duplicate transition digest")
        require(key not in self._seen_identities, "duplicate logical decision identity")
        require(record.duration >= MINIMUM_HOLD_TENSORS, "duration violates minimum hold")
        self._seen_digests.add(digest)
        self._seen_identities[key] = digest
        self._rows.append(record)
        self.accepted_count += 1
        while len(self._rows) > self.capacity:
            self._rows.popleft()
            self.evicted_count += 1

    def resident_digests(self) -> Tuple[str, ...]:
        return tuple(row.digest for row in self._rows)

    def sample(self, batch_size: int, generator: torch.Generator) -> Run5BReplayBatchV1:
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
        return Run5BReplayBatchV1(
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
# Trainer: Run-4 update_once, Run-5B construction/preflight
# ---------------------------------------------------------------------------


class Run5BHybridSacTrainerV1(T._Run4TrainerCore):
    def __init__(self, *, actor: ConditionalHybridActor, critics: TwinHybridCritics,
                 config: T.TrainerConfigV1, binding_sha256: str,
                 target_generator: torch.Generator, actor_generator: torch.Generator) -> None:
        if type(config) is not T.TrainerConfigV1:
            raise T.TrainerStateError("config must be an exact TrainerConfigV1")
        RM.validate_run5b_models(actor, critics)
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
        RM.validate_run5b_models(self.actor, self.critics)
        if type(batch) is not Run5BReplayBatchV1:
            raise error("update_once requires an exact Run5BReplayBatchV1")
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
# Orchestrator
# ---------------------------------------------------------------------------


class Run5BOrchestratorV1:
    def __init__(self, *, collector_factory: Callable[[], RC.Run5BModeledCollectorV1],
                 seed: int, checkpoint_updates: Sequence[int],
                 preregistration_sha256: str) -> None:
        require(torch.get_num_threads() == CONFIG.torch_intraop_threads,
                "torch intra-op threads must be exactly 4")
        registered = set(CONFIG.smoke_checkpoints) | set(CONFIG.deep_checkpoints)
        require(set(checkpoint_updates) <= registered and 0 in checkpoint_updates,
                "checkpoint updates must be registered Run-5B boundaries")
        self.checkpoint_updates = tuple(sorted(checkpoint_updates))
        self.seed = seed
        self.seed_plan = seed_plan(seed)
        self.schedule = warmup_schedule(seed)
        self.collector = collector_factory()
        require(type(self.collector) is RC.Run5BModeledCollectorV1, "collector is foreign")
        require(self.collector.decision_count == 0, "collector is not at genesis")
        self.actor, self.critics = RM.build_run5b_models(
            actor_seed=self.seed_plan["actor_seed"], critic_seed=self.seed_plan["critics_seed"])
        self.binding_document = {
            "schema": SCHEMA, "preregistration_sha256": preregistration_sha256,
            "model_binding_sha256": RM.RUN5B_TRAINING_MODEL_BINDING_SHA256,
            "feature_schema_id": C.FEATURE_SCHEMA_ID,
            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
            "feature_order_sha256": C.FEATURE_ORDER_SHA256,
            "reward_schema_sha256": R4.REWARD_SCHEMA_SHA256,
            "collector_binding_sha256": self.collector.collector_binding_sha256,
            "action_identity_schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
            "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "seed_plan": self.seed_plan, "schedule_id": self.schedule.config.schedule_id,
        }
        self.binding_sha256 = _sha(self.binding_document)
        self.replay = Run5BReplayBufferV1(CONFIG.replay_capacity, self.binding_sha256)
        self.generators: dict[str, torch.Generator] = {}
        for name in GENERATOR_NAMES:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed_plan[f"{name}_seed"])
            self.generators[name] = generator
        self.trainer = Run5BHybridSacTrainerV1(
            actor=self.actor, critics=self.critics, config=trainer_config(),
            binding_sha256=self.binding_sha256,
            target_generator=self.generators["trainer_target"],
            actor_generator=self.generators["trainer_actor"])
        self._catalog = load_contract()
        self.ledger: list[dict[str, Any]] = []
        self.history: list[RC.Run5BCollectedTransitionV1] = []
        self.preflight: Optional[dict[str, Any]] = None
        self.last_metrics: Optional[T.UpdateMetricsV1] = None
        self.stop_requested = False
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

    def collect_one(self) -> RC.Run5BCollectedTransitionV1:
        request = self._next_request()
        record = self.collector.collect(request)
        require(record.request == request, "collector substituted the request")
        self.replay.insert(record)
        self.history.append(record)
        self.ledger.append({"request": request.to_dict(), "digest": record.digest})
        require(self.collector.decision_count == self.decision_count, "count mismatch")
        return record

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

    def train_once(self) -> T.UpdateMetricsV1:
        batch = self.replay.sample(CONFIG.batch_size, self.generators["replay"])
        metrics = self.trainer.update_once(batch)
        metrics.require_finite()
        self.last_metrics = metrics
        return metrics

    def run_to(self, target_update: int, *,
               on_decision: Callable[[int, RC.Run5BCollectedTransitionV1], None],
               on_update: Callable[[T.UpdateMetricsV1], None],
               on_boundary: Callable[[str], None]) -> str:
        """Advance to ``target_update``; returns 'TARGET' or 'STOPPED'.

        ``on_boundary(kind)`` is called at every registered boundary with
        kind 'checkpoint', and once with 'emergency' when a stop request is
        honoured after a completed update.
        """
        require(target_update >= self.update_count, "target precedes current state")
        if self.preflight is None:
            self.run_preflight()
            for ordinal, record in enumerate(self.history):
                on_decision(ordinal, record)
            on_boundary("checkpoint")
        while self.update_count < target_update:
            if self.stop_requested:
                if self.update_count not in self.checkpoint_updates:
                    on_boundary("emergency")
                return "STOPPED"
            before = self.update_count
            for _ in range(CONFIG.environment_transitions_per_update):
                record = self.collect_one()
                on_decision(self.decision_count - 1, record)
            metrics = self.train_once()
            require(self.update_count == before + 1, "one loop did not make one update")
            on_update(metrics)
            if self.update_count in self.checkpoint_updates:
                on_boundary("checkpoint")
        return "TARGET"

    # -- boundary ---------------------------------------------------------
    def boundary(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count, "decision_count": self.decision_count,
            "actor_sha256": _tree_sha256(self.actor.state_dict()),
            "critics_sha256": _tree_sha256(self.critics.state_dict()),
            "actor_optimizer_sha256": _tree_sha256(self.trainer.actor_optimizer.state_dict()),
            "critic_optimizer_sha256": _tree_sha256(self.trainer.critic_optimizer.state_dict()),
            **{f"{name}_rng_sha256": _tensor_sha256(self.generators[name].get_state())
               for name in GENERATOR_NAMES},
            "replay_resident_sha256": _sha(list(self.replay.resident_digests())),
            "replay_accepted_count": self.replay.accepted_count,
            "replay_evicted_count": self.replay.evicted_count,
            "collector_checkpoint_sha256": _sha(self.collector.checkpoint().to_dict()),
            "joint_channel_sha256": self.collector._channel.checkpoint_sha256(),
            "ledger_sha256": _sha(self.ledger),
        }

    def event_record(self) -> dict[str, Any]:
        require(self.preflight is not None, "preflight has not passed")
        require(self.decision_count == len(self.schedule)
                + CONFIG.environment_transitions_per_update * self.update_count,
                "checkpoint ratio differs")
        return {"schema": EVENT_SCHEMA, "binding": self.binding_document,
                "binding_sha256": self.binding_sha256, "seed": self.seed,
                "update_count": self.update_count, "decision_count": self.decision_count,
                "checkpoint_updates": list(self.checkpoint_updates),
                "ledger": list(self.ledger),
                "collector_checkpoint": self.collector.checkpoint().to_dict(),
                "preflight": self.preflight, "boundary": self.boundary()}

    def training_state(self) -> dict[str, Any]:
        online, target = {}, {}
        for name, value in self.critics.state_dict().items():
            destination = online if name.startswith(ONLINE_PREFIXES) else target
            require(name.startswith(ONLINE_PREFIXES + TARGET_PREFIXES),
                    f"critic tensor {name} has no registered prefix")
            destination[name] = value.detach().clone()
        return {"schema": TRAINING_STATE_SCHEMA, "seed": self.seed,
                "update_count": self.update_count,
                "online_critics": online, "target_critics": target,
                "actor_optimizer": self.trainer.actor_optimizer.state_dict(),
                "critic_optimizer": self.trainer.critic_optimizer.state_dict(),
                "generators": {name: self.generators[name].get_state().clone()
                               for name in GENERATOR_NAMES}}

    def bundle_payloads(self) -> tuple[dict[str, bytes], dict[str, Any]]:
        event = self.event_record()
        payloads = {
            "event.json": B.canonical_bytes(event),
            "training_state.pt": B.torch_bytes(self.training_state()),
            "actor_state_dict.pt": B.torch_bytes(
                {k: v.detach().clone() for k, v in self.actor.state_dict().items()}),
            "channel_state.json": B.canonical_bytes(self.collector.channel_checkpoint()),
        }
        identity = {
            "seed": self.seed, "update_count": self.update_count,
            "decision_count": self.decision_count, "binding_sha256": self.binding_sha256,
            "preregistration_sha256": self.binding_document["preregistration_sha256"],
            "model_binding": _plain(RM.RUN5B_TRAINING_MODEL_BINDING),
            "model_binding_sha256": RM.RUN5B_TRAINING_MODEL_BINDING_SHA256,
            "feature_schema_id": C.FEATURE_SCHEMA_ID,
            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
            "feature_order": list(C.RUN5B_POLICY_FEATURE_ORDER),
            "feature_order_sha256": C.FEATURE_ORDER_SHA256,
            "reward_schema_sha256": R4.REWARD_SCHEMA_SHA256,
            "event_sha256": _sha(event), "boundary": event["boundary"],
            "actor_fixtures": actor_fixture_outputs(self.actor),
            "replay_continuation": {
                "method": "HASH_BOUND_DETERMINISTIC_RECONSTRUCTION_FROM_EVENT_ACTION_LEDGER",
                "resident_sha256": event["boundary"]["replay_resident_sha256"],
                "accepted": self.replay.accepted_count, "evicted": self.replay.evicted_count},
        }
        return payloads, identity

    # -- restore (no gradient replay) ------------------------------------
    @classmethod
    def restore_from_bundle(cls, bundle: B.VerifiedBundle, *, collector_factory,
                            preregistration_sha256: str) -> "Run5BOrchestratorV1":
        manifest = bundle.manifest
        RM.require_run5b_identity(manifest, preregistration_sha256=preregistration_sha256)
        require(manifest.get("feature_schema_sha256") == C.FEATURE_SCHEMA_SHA256
                and manifest.get("feature_order") == list(C.RUN5B_POLICY_FEATURE_ORDER)
                and _plain(manifest.get("model_binding"))
                == _plain(RM.RUN5B_TRAINING_MODEL_BINDING),
                "bundle feature schema, order or model binding differs")
        event = json.loads(bundle.payload("event.json"))
        require(event.get("schema") == EVENT_SCHEMA, "bundle event record is foreign")
        require(_sha(event) == manifest["event_sha256"], "event record differs from manifest")
        state = B.torch_from_bytes(bundle.payload("training_state.pt"))
        actor_state = B.torch_from_bytes(bundle.payload("actor_state_dict.pt"))
        channel = json.loads(bundle.payload("channel_state.json"))
        require(state.get("schema") == TRAINING_STATE_SCHEMA and state["seed"] == event["seed"]
                and state["update_count"] == event["update_count"],
                "training state identity differs from the event record")
        candidate = cls(collector_factory=collector_factory, seed=int(event["seed"]),
                        checkpoint_updates=tuple(event["checkpoint_updates"]),
                        preregistration_sha256=preregistration_sha256)
        require(candidate.binding_sha256 == event["binding_sha256"],
                "bundle binding differs from this configuration")
        candidate.actor, candidate.critics = RM.require_run5b_checkpoint(
            manifest=manifest, actor_state=actor_state,
            critic_state={**state["online_critics"], **state["target_critics"]},
            preregistration_sha256=preregistration_sha256,
            expected_tree_sha256=manifest["boundary"]["actor_sha256"])
        candidate.trainer = Run5BHybridSacTrainerV1(
            actor=candidate.actor, critics=candidate.critics, config=trainer_config(),
            binding_sha256=candidate.binding_sha256,
            target_generator=candidate.generators["trainer_target"],
            actor_generator=candidate.generators["trainer_actor"])
        candidate.trainer.actor_optimizer.load_state_dict(state["actor_optimizer"])
        candidate.trainer.critic_optimizer.load_state_dict(state["critic_optimizer"])
        require(set(state["generators"]) == set(GENERATOR_NAMES), "generator set differs")
        for name in GENERATOR_NAMES:
            candidate.generators[name].set_state(state["generators"][name])
        candidate.trainer.update_count = int(event["update_count"])
        candidate.collector.restore(RC.RC.Run5CollectorCheckpointV1.from_dict(
            event["collector_checkpoint"]))
        for record, row in zip(candidate.collector.history(), event["ledger"]):
            require(record.digest == row["digest"], "rebuilt transition differs")
            candidate.replay.insert(record)
            candidate.history.append(record)
            candidate.ledger.append(row)
        require(candidate.collector.channel_checkpoint() == channel,
                "rebuilt joint-channel state differs from the bundle")
        candidate.preflight = event["preflight"]
        require(candidate.boundary() == event["boundary"],
                "bundle resume is not bit-identical to the recorded boundary")
        require(actor_fixture_outputs(candidate.actor) == manifest["actor_fixtures"],
                "restored actor fixture outputs differ")
        return candidate


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

PREVIOUS_SLICE = C.PREVIOUS_SLICE


def expected_previous_features(previous: RC.Run5BCollectedTransitionV1) -> tuple[float, ...]:
    """Transport-only encoding of the prior record (Q_perc is not part of it)."""
    one_hot = [0.0] * EXPECTED_MODE_COUNT
    one_hot[previous.mode_id] = 1.0
    success = previous.terminal == "SUCCESS"
    return tuple(float(v) for v in (
        *one_hot, previous.q_e4 / float(Q_E4_MAX),
        float(previous.latency_ms) / R4.REWARD_DEADLINE_MS if success else 0.0,
        1.0, 1.0 if success else 0.0))


def qperc_leaks(history: Sequence[RC.Run5BCollectedTransitionV1]) -> int:
    """Transitions whose state contains the predecessor's exact Q_perc value.

    Only interior values ``0 < Q_perc < 1`` are informative: 0.0 and 1.0 occur
    structurally in every state (one-hot, presence, empty backlog).
    """
    return sum(int(prev.q_perc is not None and 0.0 < float(prev.q_perc) < 1.0
                   and float(prev.q_perc) in record.state)
               for prev, record in zip(history, history[1:]))


def preflight_report(history: Sequence[RC.Run5BCollectedTransitionV1],
                     diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    strata: dict[tuple, int] = {}
    for record in history:
        key = (record.mode_id, record.request.warmup_q_bin_index)
        strata[key] = strata.get(key, 0) + 1
    if len(strata) != 72 or set(strata.values()) != {4}:
        failures.append("warm-up is not exactly 12 modes x 6 q strata x 4")
    terminals = {record.terminal for record in history}
    if "SUCCESS" not in terminals or len(terminals) < 2:
        failures.append("warm-up lacks both success and failure/timeout")
    variation = state_variation(history)
    for name, item in variation.items():
        if item["distinct"] < 2 or not item["span"] > 0:
            failures.append(f"{name} does not vary")
    mismatches = sum(int(tuple(record.state[PREVIOUS_SLICE]) != expected_previous_features(prev))
                     for prev, record in zip(history, history[1:]))
    if mismatches:
        failures.append(f"{mismatches} previous-outcome encodings differ from the prior record")
    leaks = qperc_leaks(history)
    if leaks:
        failures.append(f"{leaks} states contain the previous Q_perc")
    reward_errors = sum(int(not reward_matches(record)) for record in history)
    if reward_errors:
        failures.append(f"{reward_errors} rewards differ from the registered formula")
    future = sum(1 for d in diagnostics if not d["generated_ticks_after_observed"])
    if future:
        failures.append("a generated SNR tick was not after the observed tick")
    return {"decisions": len(history), "strata": len(strata),
            "terminals": sorted(terminals), "variation": variation,
            "previous_outcome_mismatches": mismatches, "previous_qperc_leaks": leaks,
            "reward_formula_mismatches": reward_errors,
            "future_tick_violations": future, "failures": failures, "passed": not failures}


FEATURE_COLUMNS = {"camera_si": 0, "radar_p40": 1, "mcs": 2, "backlog": 3, "snr": SNR_INDEX,
                   "prev_q": C.PREV_Q_INDEX, "prev_latency": C.PREV_LATENCY_INDEX,
                   "prev_present": C.PREV_PRESENT_INDEX, "prev_success": C.PREV_SUCCESS_INDEX}


def state_variation(history: Sequence[RC.Run5BCollectedTransitionV1]) -> dict[str, dict[str, Any]]:
    out = {name: {"distinct": len({r.state[i] for r in history}),
                  "span": max(r.state[i] for r in history) - min(r.state[i] for r in history)}
           for name, i in FEATURE_COLUMNS.items()}
    modes = [max(range(EXPECTED_MODE_COUNT), key=lambda k: r.state[4 + k])
             for r in history if r.state[C.PREV_PRESENT_INDEX] == 1.0]
    out["prev_mode"] = {"distinct": len(set(modes)),
                        "span": (max(modes) - min(modes)) if modes else 0}
    return out


def reward_matches(record: RC.Run5BCollectedTransitionV1) -> bool:
    if record.terminal == "SUCCESS":
        expected = record.q_perc - R4.REWARD_LATENCY_WEIGHT * (record.latency_ms
                                                               / R4.REWARD_DEADLINE_MS)
        return (record.reward == expected and 0.0 <= record.latency_ms <= R4.REWARD_DEADLINE_MS)
    return record.reward == R4.REGISTERED_FAILURE_REWARD and record.q_perc is None


# ---------------------------------------------------------------------------
# Deterministic actor probe
# ---------------------------------------------------------------------------


def _fixtures() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260930)
    rows = torch.rand((20, WIDTH), generator=generator, dtype=torch.float32)
    rows[:, 4:16] = 0.0
    for index in range(20):
        rows[index, 4 + index % EXPECTED_MODE_COUNT] = 1.0
    rows[:, C.PREV_PRESENT_INDEX] = 1.0
    rows[:, C.PREV_SUCCESS_INDEX] = (torch.arange(20) % 2).to(torch.float32)
    return rows


def actor_fixture_outputs(actor: ConditionalHybridActor) -> list[dict[str, Any]]:
    with torch.no_grad():
        execution = actor.deterministic_execution(_fixtures())
        logits = actor(_fixtures()).logits
    return [{"mode_id": int(execution.mode_index[i]), "q_e4": int(execution.q_e4[i]),
             "logits_sha256": _tensor_sha256(logits[i].contiguous())} for i in range(20)]

"""Pilot orchestrator: original Run-5 SAC mechanics with the restricted-support actor and the pilot collector.

Reused unchanged: ``trainer._Run4TrainerCore.update_once`` (critic, actor, Polyak), the Run-5 replay buffer and
transition record, the Run-5 seed plan and RNG streams, the preregistered hyperparameters (gamma 0.99/tensor,
alpha_d 0.05, alpha_c 0.02, lr 3e-4/3e-4, tau 0.005, batch 256, replay 65,536, 4 transitions per update, 4 intra-op
threads), stochastic training actions (categorical mode + conditional q from ``sample_all_modes``), checkpoint
boundaries. Changes mechanically required by the masking (documented in the assumptions record):
  * actor = ``support.PilotRestrictedActor`` (same network, pilot brackets + mask); model validation accordingly;
  * warm-up = the original stratified schedule restricted to modes 5/8/11; samples per mode x q-bin 4 -> 16 so
    the warm-up keeps its registered 288 decisions (>= the 256-row first batch);
  * every executed action is checked to lie in the pilot support.
Fresh replay per run; nothing from earlier runs or the other reward arm is loaded.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as T
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_preregistration as PR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (TwinHybridCritics, _local_torch_seed,
                                                                   build_twin_critics)

from . import support as S

CONFIG = PR.CONFIG
WARMUP_SAMPLES_PER_MODE_Q_BIN = 16          # 3 modes x 6 bins x 16 = 288 = CONFIG.warmup_decision_count
CHECKPOINTS = tuple(CONFIG.deep_checkpoints)
SCHEMA = "splitfusion.run5_restricted_pilot.training.v1"
CANDIDATE_REWARD_SCHEMA = dict(schema="splitfusion.run5_restricted_pilot.candidate_reward.v1",
                               success="Y * (0.7 + 0.3 * Q_seg); Y = class-weighted within-tolerance recall "
                                       "(person 0.6 @1.2 m, vehicle 0.4 @1.0 m) over classes with eligible GT",
                               timeout=0.0, timely_undefined="EXCLUDED_EVALUATOR_FAULT_SESSION_RESET",
                               late_undefined="TIMEOUT", deadline_ms_inclusive=170)
CANDIDATE_REWARD_SCHEMA_SHA256 = hashlib.sha256(json.dumps(CANDIDATE_REWARD_SCHEMA, sort_keys=True).encode()).hexdigest()


class PilotTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotTrainingError(message)


def validate_pilot_models(actor, critics) -> None:
    _require(type(actor) is S.PilotRestrictedActor, "actor must be an exact PilotRestrictedActor")
    _require(type(critics) is TwinHybridCritics, "critics must be exact TwinHybridCritics")
    expected = RM.run5_model_config()
    for name, module in (("actor", actor), ("critic_1", critics.critic_1), ("critic_2", critics.critic_2),
                         ("target_1", critics.target_1), ("target_2", critics.target_2)):
        _require(module.config == expected, f"{name} configuration differs from Run 5")
    _require(RM._first_linear(actor.encoder, "actor").in_features == 22, "actor input width differs from 22")
    for critic in (critics.critic_1, critics.critic_2, critics.target_1, critics.target_2):
        _require(RM._first_linear(critic.trunk, "critic").in_features == 22 + 12 + 1, "critic input width differs")
    actor._validate_registered_support_buffers()


class PilotTrainerV1(T._Run4TrainerCore):
    """Run-5 trainer construction and preflight with pilot model validation; update_once is inherited unchanged."""

    def __init__(self, *, actor, critics, config, binding_sha256, target_generator, actor_generator) -> None:
        validate_pilot_models(actor, critics)
        _require(target_generator is not actor_generator, "target and actor generators must be distinct")
        self.actor, self.critics, self.config = actor, critics, config
        self.expected_binding = binding_sha256
        self._target_generator, self._actor_generator = target_generator, actor_generator
        self._online_critic_parameters = list(chain(critics.critic_1.parameters(), critics.critic_2.parameters()))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self._online_critic_parameters, lr=config.critic_lr)
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _require_evidence_class(self, binding: Any) -> None:
        if binding != self.expected_binding:
            raise T.TrainerPreflightError("batch binding differs from the trainer binding")

    def _preflight(self, batch: Any) -> None:
        """The Run-5 tensor preflight (run5_training.Run5HybridSacTrainerV1._preflight), with pilot model validation
        and the pilot support check added."""
        error = T.TrainerPreflightError
        self._assert_optimizer_wiring()
        validate_pilot_models(self.actor, self.critics)
        if type(batch) is not RT.Run5ReplayBatchV1:
            raise error("update_once requires an exact Run5ReplayBatchV1")
        self._require_evidence_class(batch.binding_sha256)
        size = batch.batch_size
        if size < 1:
            raise error("batch must not be empty")
        for name, tensor, shape in (("state", batch.state, (size, RT.WIDTH)), ("next_state", batch.next_state, (size, RT.WIDTH)),
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
        if not all(S.in_support(m, q) for m, q in zip(batch.mode_id.tolist(), batch.q_e4.tolist())):
            raise error("batch action outside the pilot support")
        if bool((batch.duration < 2).any()):
            raise error("duration violates the minimum action hold")
        discount = batch.discount()
        if bool((discount <= 0.0).any()) or bool((discount > 1.0).any()):
            raise error("stored discount must lie in (0, 1]")


def _tree(state: dict) -> str:
    buffer = io.BytesIO()
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class PilotOrchestratorV1:
    def __init__(self, *, collector_factory: Callable[[], Any], seed: int, arm: str, authorization_sha256: str) -> None:
        _require(torch.get_num_threads() == CONFIG.torch_intraop_threads, "torch intra-op threads must be exactly 4")
        _require(arm in ("CONTROL", "CANDIDATE"), "unknown arm")
        self.seed, self.arm = seed, arm
        self.seed_plan = RT.seed_plan(seed)
        self.warmup, self.schedule_id = S.warmup_actions(seed, CONFIG.warmup_q_bin_count, WARMUP_SAMPLES_PER_MODE_Q_BIN)
        _require(len(self.warmup) == CONFIG.warmup_decision_count, "warm-up is not 288 decisions")
        self.collector = collector_factory()
        _require(self.collector.decision_count == 0 and self.collector.arm == arm, "collector not at genesis for this arm")
        config = RM.run5_model_config()
        with _local_torch_seed(self.seed_plan["actor_seed"]):
            self.actor = S.PilotRestrictedActor(config)
        self.critics = build_twin_critics(config, seed=self.seed_plan["critics_seed"])
        self.binding_document = dict(
            schema=SCHEMA, arm=arm, support_sha256=S.SUPPORT_SHA256, model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
            reward_schema_sha256=R4.REWARD_SCHEMA_SHA256 if arm == "CONTROL" else CANDIDATE_REWARD_SCHEMA_SHA256,
            collector_binding_sha256=self.collector.collector_binding_sha256, seed_plan=self.seed_plan,
            warmup_schedule_id=self.schedule_id, warmup_samples_per_mode_q_bin=WARMUP_SAMPLES_PER_MODE_Q_BIN,
            authorization_sha256=authorization_sha256, preregistration_config=str(CONFIG))
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding_document, sort_keys=True).encode()).hexdigest()
        self.replay = RT.Run5ReplayBufferV1(CONFIG.replay_capacity, self.binding_sha256)
        self.generators = {}
        for name in RT.GENERATOR_NAMES:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed_plan[f"{name}_seed"])
            self.generators[name] = generator
        self.trainer = PilotTrainerV1(actor=self.actor, critics=self.critics, config=RT.trainer_config(),
                                      binding_sha256=self.binding_sha256,
                                      target_generator=self.generators["trainer_target"],
                                      actor_generator=self.generators["trainer_actor"])
        self.ledger: list[list] = []
        self.transition_digests: list[Optional[str]] = []
        self.last_metrics = None

    @property
    def decision_count(self) -> int:
        return len(self.ledger)

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    def _request(self) -> orch.ModeledActionRequestV1:
        ordinal = self.decision_count
        if ordinal < len(self.warmup):
            a = self.warmup[ordinal]
            return orch.ModeledActionRequestV1(decision_ordinal=ordinal, mode_id=a.mode_id, q_e4=a.q_e4,
                                               source="STRATIFIED_WARMUP", warmup_q_bin_index=a.q_bin_index)
        state = torch.tensor((self.collector.current_state_features(),), dtype=torch.float32)
        with torch.no_grad():
            sample = self.actor.sample_all_modes(state, generator=self.generators["decision_q"])
            mode_id = int(torch.multinomial(sample.probs[0], 1, generator=self.generators["decision_mode"])[0])
            q_e4 = int(sample.q_e4[0, mode_id])
        _require(S.in_support(mode_id, q_e4), f"actor produced an action outside the support ({mode_id}, {q_e4})")
        return orch.ModeledActionRequestV1(decision_ordinal=ordinal, mode_id=mode_id, q_e4=q_e4,
                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None)

    def _execute(self, request: orch.ModeledActionRequestV1):
        transition = self.collector.collect_transition(request)
        if transition is not None:
            self.replay.insert(transition)
        self.ledger.append([request.mode_id, request.q_e4, request.source, request.warmup_q_bin_index])
        self.transition_digests.append(None if transition is None else transition.digest)
        return transition

    def collect_one(self):
        return self._execute(self._request())

    def warmup_phase(self) -> None:
        _require(self.decision_count == 0, "warm-up runs at genesis")
        before = self._model_sha256()
        for _ in range(len(self.warmup)):
            self.collect_one()
        _require(self._model_sha256() == before, "models changed during the no-gradient warm-up")
        _require(len(self.replay) >= CONFIG.batch_size, f"warm-up produced {len(self.replay)} < {CONFIG.batch_size} transitions")

    def train_once(self):
        batch = self.replay.sample(CONFIG.batch_size, self.generators["replay"])
        metrics = self.trainer.update_once(batch)
        metrics.require_finite()
        self.last_metrics = metrics
        return metrics

    def step_update(self):
        before = self.update_count
        for _ in range(CONFIG.environment_transitions_per_update):
            self.collect_one()
        metrics = self.train_once()
        _require(self.update_count == before + 1, "one loop did not make one update")
        return metrics

    # ---------------------------------------------------------------- bundles
    def _model_sha256(self) -> str:
        return hashlib.sha256(json.dumps({"actor": _tree(self.actor.state_dict()), "critics": _tree(self.critics.state_dict()),
                                          "a_opt": _tree(self.trainer.actor_optimizer.state_dict()),
                                          "c_opt": _tree(self.trainer.critic_optimizer.state_dict())}).encode()).hexdigest()

    def state_digest(self) -> dict[str, Any]:
        return dict(update_count=self.update_count, decision_count=self.decision_count, model=self._model_sha256(),
                    generators={k: hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest() for k, g in self.generators.items()},
                    ledger=hashlib.sha256(json.dumps(self.ledger).encode()).hexdigest(),
                    transitions=hashlib.sha256(json.dumps(self.transition_digests).encode()).hexdigest(),
                    replay=hashlib.sha256(json.dumps(list(self.replay.resident_digests())).encode()).hexdigest(),
                    channel=dict(pos=self.collector._pos, busy=self.collector.score_channel_busy_until_ns,
                                 fallback_frames=self.collector.fallback_frames_total))

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        _require(not path.exists(), "bundles are create-only")
        payload = dict(actor=self.actor.state_dict(), critics=self.critics.state_dict(),
                       actor_optimizer=self.trainer.actor_optimizer.state_dict(),
                       critic_optimizer=self.trainer.critic_optimizer.state_dict(),
                       generators={k: g.get_state() for k, g in self.generators.items()},
                       update_count=self.update_count, ledger=list(self.ledger), binding_sha256=self.binding_sha256,
                       digest=self.state_digest())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        meta = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), **payload["digest"])
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        return meta

    @classmethod
    def restore(cls, path: Path, *, collector_factory, seed: int, arm: str, authorization_sha256: str) -> "PilotOrchestratorV1":
        raw = Path(path).read_bytes()
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        _require(hashlib.sha256(raw).hexdigest() == meta["sha256"], "bundle bytes differ from its manifest")
        payload = torch.load(io.BytesIO(raw), weights_only=False)
        fresh = cls(collector_factory=collector_factory, seed=seed, arm=arm, authorization_sha256=authorization_sha256)
        _require(fresh.binding_sha256 == payload["binding_sha256"], "bundle belongs to a different binding")
        for mode_id, q_e4, source, q_bin in payload["ledger"]:          # deterministic re-execution, no sampling
            fresh._execute(orch.ModeledActionRequestV1(decision_ordinal=fresh.decision_count, mode_id=mode_id,
                                                       q_e4=q_e4, source=source, warmup_q_bin_index=q_bin))
        fresh.actor.load_state_dict(payload["actor"])
        fresh.critics.load_state_dict(payload["critics"])
        fresh.trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        fresh.trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        for name, state in payload["generators"].items():
            fresh.generators[name].set_state(state)
        fresh.trainer.update_count = int(payload["update_count"])
        _require(fresh.state_digest() == payload["digest"], "restored state digest differs from the bundle")
        return fresh

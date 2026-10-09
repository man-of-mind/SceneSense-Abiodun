"""Full Run-5 orchestrator: the original Run-5 networks, trainer, replay, warm-up and SAC settings, fresh per run.

Reused unchanged from ``splitfusion_hybrid_sac_run5_v1``:
* models: ``run5_models.build_run5_models`` (exact ``ConditionalHybridActor`` / ``TwinHybridCritics``, 22 inputs,
  2 x 128, 12 logits + 12 conditional means/log-stds, original ``MODELED_SMOKE_SUPPORT`` q bounds, no mask);
* trainer: ``run5_training.Run5HybridSacTrainerV1`` (``update_once`` and the Run-5 tensor preflight);
* replay: ``Run5ReplayBufferV1`` / ``Run5CollectedTransitionV1``; seed plan and RNG streams ``seed_plan``;
* warm-up: ``run5_training.warmup_schedule`` (12 modes x 6 q bins x 4 = 288 stratified, no-gradient decisions);
* preregistered settings ``run5_preregistration.CONFIG``: gamma 0.99/tensor, alpha_d 0.05, alpha_c 0.02,
  lr 3e-4/3e-4, tau 0.005, batch 256, replay 65,536, 4 transitions per update, 4 intra-op threads, checkpoints at
  0, 100, 250 and every 500 updates;
* stochastic training actions: categorical mode + conditional q from ``sample_all_modes``.
Bundles are create-only; restore re-executes the logged action ledger deterministically (no sampling), then loads
weights, optimiser and generator state and requires the full state digest to equal the bundle's.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_preregistration as PR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import Q_E4_SCALE, load_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity

from . import quality as Q
from . import support as S

CONFIG = PR.CONFIG
CHECKPOINTS = tuple(CONFIG.deep_checkpoints)
SCHEMA = "splitfusion.run5_full.training.v1"


class FullTrainingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTrainingError(message)


def _tree(state: dict) -> str:
    buffer = io.BytesIO()
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class FullOrchestratorV1:
    def __init__(self, *, collector_factory: Callable[[], Any], seed: int, authorization_sha256: str) -> None:
        _require(torch.get_num_threads() == CONFIG.torch_intraop_threads, "torch intra-op threads must be exactly 4")
        self.seed = seed
        self.seed_plan = RT.seed_plan(seed)
        self.schedule = RT.warmup_schedule(seed)
        self.collector = collector_factory()
        _require(self.collector.decision_count == 0, "collector not at genesis")
        self.actor, self.critics = RM.build_run5_models(actor_seed=self.seed_plan["actor_seed"],
                                                        critic_seed=self.seed_plan["critics_seed"])
        _require(tuple(tuple(int(v) for v in p) for p in zip(self.actor._support_q_e4_lower.tolist(),
                                                              self.actor._support_q_e4_upper.tolist())) == S.BOUNDS,
                 "actor support differs from the original Run-5 support")
        self.binding_document = dict(
            schema=SCHEMA, model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
            continuous_q_support_sha256=S.SUPPORT_SHA256, reward_sha256=Q.REWARD_SHA256,
            collector_binding_sha256=self.collector.collector_binding_sha256, seed_plan=self.seed_plan,
            warmup_schedule_id=self.schedule.config.schedule_id, authorization_sha256=authorization_sha256,
            preregistration_config=str(CONFIG))
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding_document, sort_keys=True).encode()).hexdigest()
        self.replay = RT.Run5ReplayBufferV1(CONFIG.replay_capacity, self.binding_sha256)
        self.generators = {}
        for name in RT.GENERATOR_NAMES:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed_plan[f"{name}_seed"])
            self.generators[name] = generator
        self.trainer = RT.Run5HybridSacTrainerV1(actor=self.actor, critics=self.critics, config=RT.trainer_config(),
                                                 binding_sha256=self.binding_sha256,
                                                 target_generator=self.generators["trainer_target"],
                                                 actor_generator=self.generators["trainer_actor"])
        self._catalog = load_contract()
        self.ledger: list[list] = []
        self.transition_digests: list[Optional[str]] = []
        self.last_metrics = None

    @property
    def decision_count(self) -> int:
        return len(self.ledger)

    @property
    def update_count(self) -> int:
        return self.trainer.update_count

    def _execution_check(self, mode_id: int, q_e4: int) -> None:
        executable = self._catalog.resolve(mode_id, q_e4 / float(Q_E4_SCALE))
        action = ExecutedActionIdentity.from_executable_action(executable, self._catalog)
        _require((action.mode_id, action.q_e4) == (mode_id, q_e4), "catalogue changed the action")
        _require(S.in_support(mode_id, q_e4), f"action outside the original support ({mode_id}, {q_e4})")

    def _request(self) -> orch.ModeledActionRequestV1:
        ordinal = self.decision_count
        if ordinal < len(self.schedule):
            a = self.schedule.action_at(ordinal)
            return orch.ModeledActionRequestV1(decision_ordinal=ordinal, mode_id=a.mode_id, q_e4=a.q_e4,
                                               source="STRATIFIED_WARMUP", warmup_q_bin_index=a.q_bin_index)
        state = torch.tensor((self.collector.current_state_features(),), dtype=torch.float32)
        with torch.no_grad():
            sample = self.actor.sample_all_modes(state, generator=self.generators["decision_q"])
            mode_id = int(torch.multinomial(sample.probs[0], 1, generator=self.generators["decision_mode"])[0])
            q_e4 = int(sample.q_e4[0, mode_id])
        self._execution_check(mode_id, q_e4)
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
        for _ in range(len(self.schedule)):
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
        c = self.collector
        return dict(update_count=self.update_count, decision_count=self.decision_count, model=self._model_sha256(),
                    generators={k: hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest() for k, g in self.generators.items()},
                    ledger=hashlib.sha256(json.dumps(self.ledger).encode()).hexdigest(),
                    transitions=hashlib.sha256(json.dumps(self.transition_digests).encode()).hexdigest(),
                    replay=hashlib.sha256(json.dumps(list(self.replay.resident_digests())).encode()).hexdigest(),
                    collector=dict(busy=c.score_channel_busy_until_ns, backlog=c._backlog_bytes, session=c._session_index,
                                   channel=hashlib.sha256(json.dumps(c._channel.checkpoint(), sort_keys=True,
                                                                     default=str).encode()).hexdigest()))

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
    def restore(cls, path: Path, *, collector_factory, seed: int, authorization_sha256: str) -> "FullOrchestratorV1":
        raw = Path(path).read_bytes()
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        _require(hashlib.sha256(raw).hexdigest() == meta["sha256"], "bundle bytes differ from its manifest")
        payload = torch.load(io.BytesIO(raw), weights_only=False)
        fresh = cls(collector_factory=collector_factory, seed=seed, authorization_sha256=authorization_sha256)
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

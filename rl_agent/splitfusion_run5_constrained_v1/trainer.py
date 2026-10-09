"""Constrained orchestrator: the original Run-5 networks, SAC trainer, warm-up, collector kernel, state, support and
checkpoint/restore mechanism (``splitfusion_run5_full_v1.trainer.FullOrchestratorV1``), with three additions:

* ConstrainedCollectorV1 - the reward-trial collector serving r^P = D*Y and recording each decision's normalised
  constraint vector g/kappa (from the catalogue's expected counts; reference cell offline only, never a state input);
* dual update - deterministic in the decision stream: after every DUAL_EVERY-th stochastic decision, lambda is updated
  from the mean g/kappa over the last WINDOW stochastic decisions (excluded decisions included: all-scored retention).
  Because it runs inside ``_execute``, restore by deterministic re-execution reproduces lambda exactly;
* ConstrainedReplayV1 - same rows, same generator consumption as Run5ReplayBufferV1.sample; the minibatch reward is
  recomputed at sampling time as record.reward + lambda . g/kappa (record.reward = r^P: D*Y on SUCCESS, 0 on TIMEOUT).
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import replace
from typing import Any, Callable

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import load_contract
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_full_v1 import trainer as FT
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as TC

from . import objective as O

CONFIG = FT.CONFIG
SCHEMA = "splitfusion.run5_constrained_v1.training.v1"
CHECKPOINTS = (0, 250, 2500, 5000, 7500, 10_000)
_require = FT._require


class ConstrainedCollectorV1(TC.TrialRun5CollectorV1):
    SCHEMA = "splitfusion.run5_constrained_v1.collector.v1"

    def __init__(self, *, latency, seed: int, shared_sources, objective_sha256: str, targets: dict, constants: dict):
        catalog = shared_sources["catalog"]
        _require(isinstance(catalog, O.LocCatalogV1) and catalog.reward_sha256 == objective_sha256, "foreign catalogue")
        catalog.arm = "C0"                                   # pass the parent's arm check, then restore the label
        try:
            super().__init__(latency=latency, seed=seed, shared_sources=shared_sources, arm="C0")
        finally:
            catalog.arm = O.OBJECTIVE_ID
        self.arm = O.OBJECTIVE_ID
        self.reward_sha256 = objective_sha256
        self.targets, self.constants = dict(targets), constants
        self.collector_binding_sha256 = O.sha({"schema": self.SCHEMA, "corrected": self.binding_sha256,
                                               "objective": objective_sha256, "reward": catalog.reward_binding_sha256,
                                               "split": catalog.split, "latency": latency.binding_sha256,
                                               "pool": latency.pool, "support": S.SUPPORT_SHA256})
        self.g_by_identity: dict[tuple, np.ndarray] = {}
        self.last_g = None

    def collect_transition(self, request):
        transition = super().collect_transition(request)
        d = self.diagnostics[-1]
        g = O.g_vector(self.catalog, d["reward_scene_key"], d["mode_id"], d["q_e4"], self.targets, self.constants)
        d["g"] = g.tolist()
        self.last_g = g
        if transition is not None:
            self.g_by_identity[(transition.session_uuid, transition.decision_seq)] = g
        return transition


class ConstrainedReplayV1(RT.Run5ReplayBufferV1):
    def __init__(self, capacity, binding_sha256, *, owner):
        super().__init__(capacity, binding_sha256)
        self._owner = owner

    def sample(self, batch_size, generator):
        require = RT.require
        require(isinstance(generator, torch.Generator) and generator is not torch.default_generator
                and generator.device.type == "cpu", "sampling needs a private CPU generator")
        require(0 < batch_size <= len(self._rows), "cannot sample that many rows")
        permutation = torch.randperm(len(self._rows), generator=generator, device="cpu")
        rows = [self._rows[int(i)] for i in permutation[:batch_size]]
        lam = self._owner.lam
        g = self._owner.collector.g_by_identity
        rewards = [row.reward + float(lam @ g[(row.session_uuid, row.decision_seq)]) for row in rows]
        next_state = torch.zeros((len(rows), RT.WIDTH), dtype=torch.float32)
        for index, row in enumerate(rows):
            if row.next_state is not None:
                next_state[index] = torch.tensor(row.next_state, dtype=torch.float32)
        return RT.Run5ReplayBatchV1(
            _state=torch.tensor([row.state for row in rows], dtype=torch.float32), _next_state=next_state,
            _mode_id=torch.tensor([row.mode_id for row in rows], dtype=torch.int64),
            _q_e4=torch.tensor([row.q_e4 for row in rows], dtype=torch.int64),
            _reward=torch.tensor(rewards, dtype=torch.float32),
            _duration=torch.tensor([row.duration for row in rows], dtype=torch.int64),
            _discount=torch.tensor([row.discount for row in rows], dtype=torch.float32),
            _has_next_state=torch.tensor([row.has_next_state for row in rows], dtype=torch.bool),
            _bootstrap=torch.tensor([row.bootstrap for row in rows], dtype=torch.bool),
            _terminated=torch.tensor([row.terminated for row in rows], dtype=torch.bool),
            _truncated=torch.tensor([row.truncated for row in rows], dtype=torch.bool),
            binding_sha256=self.binding_sha256)


class ConstrainedOrchestratorV1(FT.FullOrchestratorV1):
    def __init__(self, *, collector_factory: Callable[[], Any], seed: int, authorization_sha256: str,
                 lambda0: np.ndarray = None, objective_sha256: str = None) -> None:
        _require(torch.get_num_threads() == CONFIG.torch_intraop_threads, "torch intra-op threads must be exactly 4")
        self.seed = seed
        self.seed_plan = RT.seed_plan(seed)
        self.schedule = RT.warmup_schedule(seed)
        self.collector = collector_factory()
        c = self.collector
        _require(c.decision_count == 0, "collector not at genesis")
        _require(c.latency.pool == "TRAIN_GCV2" and c.catalog.split == "fit", "training must use TRAIN_GCV2 and FIT")
        _require(c.reward_sha256 == objective_sha256 == c.catalog.reward_sha256, "objective binding mismatch")
        self.actor, self.critics = RM.build_run5_models(actor_seed=self.seed_plan["actor_seed"],
                                                        critic_seed=self.seed_plan["critics_seed"])
        _require(tuple(tuple(int(v) for v in p) for p in zip(self.actor._support_q_e4_lower.tolist(),
                                                              self.actor._support_q_e4_upper.tolist())) == S.BOUNDS,
                 "actor support differs from the original Run-5 support")
        self.lambda0 = np.asarray(lambda0, dtype=np.float64)
        self.lam = self.lambda0.copy()
        self.dual_log: list[dict] = []
        self._window: deque = deque(maxlen=O.WINDOW)
        self._stochastic = 0
        self.binding_document = dict(
            schema=SCHEMA, model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
            continuous_q_support_sha256=S.SUPPORT_SHA256, objective=O.OBJECTIVE_ID, objective_sha256=objective_sha256,
            targets=c.targets, lambda0=self.lambda0.tolist(), collector_binding_sha256=c.collector_binding_sha256,
            latency_pool=c.latency.pool, scene_split=c.catalog.split, seed_plan=self.seed_plan,
            warmup_schedule_id=self.schedule.config.schedule_id, authorization_sha256=authorization_sha256,
            preregistration_config=str(CONFIG))
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding_document, sort_keys=True).encode()).hexdigest()
        self.replay = ConstrainedReplayV1(CONFIG.replay_capacity, self.binding_sha256, owner=self)
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
        self.transition_digests: list = []
        self.last_metrics = None

    def _execute(self, request):
        transition = super()._execute(request)
        if request.source == "STOCHASTIC_ACTOR":
            self._window.append(self.collector.last_g)
            self._stochastic += 1
            if self._stochastic % O.DUAL_EVERY == 0:
                mean = np.mean(np.stack(self._window), axis=0)
                self.lam = np.clip(self.lam - O.ETA * mean, 0.0, O.LAMBDA_MAX)
                self.dual_log.append(dict(stochastic_decisions=self._stochastic, decision_count=self.decision_count,
                                          window_mean=mean.tolist(), lam=self.lam.tolist()))
        return transition

    def state_digest(self) -> dict[str, Any]:
        d = super().state_digest()
        d["dual"] = hashlib.sha256(json.dumps(dict(lam=self.lam.tolist(), log=self.dual_log,
                                                   stochastic=self._stochastic)).encode()).hexdigest()
        return d

    @classmethod
    def restore(cls, path, *, collector_factory, seed, authorization_sha256, lambda0=None, objective_sha256=None):
        factory = lambda **kw: cls(lambda0=lambda0, objective_sha256=objective_sha256, **kw)
        return FT.FullOrchestratorV1.restore.__func__(_Bound(factory), path, collector_factory=collector_factory,
                                                      seed=seed, authorization_sha256=authorization_sha256)


class _Bound:
    """Lets the inherited classmethod body construct this subclass with its extra registered arguments."""
    def __init__(self, factory):
        self._factory = factory

    def __call__(self, **kw):
        return self._factory(**kw)

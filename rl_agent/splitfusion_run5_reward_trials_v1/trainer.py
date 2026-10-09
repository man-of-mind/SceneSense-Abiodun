"""Reward-trial orchestrator: ``splitfusion_run5_full_v1.trainer.FullOrchestratorV1`` (original Run-5 networks,
trainer, replay, warm-up, SAC settings; fresh per run; create-only bundles; deterministic restore), with the binding
document naming the arm's actual reward definition/hash, the training pool and the FIT split. Only ``__init__``
is restated (the binding is fixed at construction); every other method is inherited unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import torch

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import load_contract
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_full_v1 import trainer as FT

from . import rewards as R

CONFIG = FT.CONFIG
SCHEMA = "splitfusion.run5_reward_trials.training.v1"
TRAIN_POOL = "TRAIN_GCV2"
TRAIN_SPLIT = "fit"
CHECKPOINTS = (0, 250, 2500, 5000, 7500, 10_000)          # bounded retention (subset of the registered schedule)
TrialTrainingError = FT.FullTrainingError
_require = FT._require
assert set(CHECKPOINTS) <= set(FT.CHECKPOINTS) | {10_000}


class TrialOrchestratorV1(FT.FullOrchestratorV1):
    def __init__(self, *, collector_factory: Callable[[], Any], seed: int, authorization_sha256: str) -> None:
        _require(torch.get_num_threads() == CONFIG.torch_intraop_threads, "torch intra-op threads must be exactly 4")
        self.seed = seed
        self.seed_plan = RT.seed_plan(seed)
        self.schedule = RT.warmup_schedule(seed)
        self.collector = collector_factory()
        c = self.collector
        _require(c.decision_count == 0, "collector not at genesis")
        _require(c.latency.pool == TRAIN_POOL and c.catalog.split == TRAIN_SPLIT,
                 "training must use the TRAIN_GCV2 pool and the FIT scene split")
        _require(c.reward_sha256 == R.reward_sha256(c.arm) == c.catalog.reward_sha256, "reward binding mismatch")
        self.actor, self.critics = RM.build_run5_models(actor_seed=self.seed_plan["actor_seed"],
                                                        critic_seed=self.seed_plan["critics_seed"])
        _require(tuple(tuple(int(v) for v in p) for p in zip(self.actor._support_q_e4_lower.tolist(),
                                                              self.actor._support_q_e4_upper.tolist())) == S.BOUNDS,
                 "actor support differs from the original Run-5 support")
        self.binding_document = dict(
            schema=SCHEMA, model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256,
            continuous_q_support_sha256=S.SUPPORT_SHA256, arm=c.arm, reward_definition=R.definition(c.arm),
            reward_sha256=c.reward_sha256, collector_binding_sha256=c.collector_binding_sha256,
            latency_pool=c.latency.pool, scene_split=c.catalog.split, seed_plan=self.seed_plan,
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
        self.transition_digests: list = []
        self.last_metrics = None

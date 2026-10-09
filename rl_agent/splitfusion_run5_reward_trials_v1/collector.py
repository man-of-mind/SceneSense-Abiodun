"""Reward-trial collector: ``splitfusion_run5_full_v1.collector.FullRun5CollectorV1`` with the arm's catalogue and a
binding_02 pool latency model. The kernel, state (22-D incl. causal SNR and previous Q_perc), channel (all four
profiles), backlog, outcome semantics and session handling are inherited unchanged; the kernel reads the reward
through ``catalog.reward_value``, which serves exactly the catalogue's arm.
"""
from __future__ import annotations

from typing import Any, Mapping

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_run5_full_v1 import collector as FC

from . import catalog as K
from . import latency as L
from . import rewards as R

EVIDENCE_LABEL = "RUN5_REWARD_TRIAL_MODELED"
TrialCollectorError = FC.FullCollectorError
_require = FC._require
_sha = FC._sha


def load_shared_sources(arm: str, split: str = "fit") -> dict[str, Any]:
    sources = CR.load_shared_sources()           # transport, MCS model, SI scaling check against FIT (unchanged)
    fit_keys = sources["catalog"].keys
    catalog = K.TrialSceneCatalogV1(split, arm)
    if split == "fit":
        _require(catalog.keys == fit_keys and catalog.binding_sha256 == sources["catalog"].binding_sha256,
                 "trial FIT catalogue differs from the corrected FIT catalogue")
    sources["catalog"] = catalog
    return sources


class TrialRun5CollectorV1(FC.FullRun5CollectorV1):
    SCHEMA = "splitfusion.run5_reward_trials.collector.v1"

    def __init__(self, *, latency, seed: int, shared_sources: Mapping[str, Any], arm: str) -> None:
        catalog = shared_sources.get("catalog")
        _require(isinstance(catalog, K.TrialSceneCatalogV1) and catalog.arm == arm,
                 "shared sources lack the arm's trial catalogue")
        _require(isinstance(latency, L.PoolLatencyModelV1), "latency must be a binding_02 pool model")
        self.arm = arm
        self.reward_sha256 = R.reward_sha256(arm)
        self.latency = latency
        self.decision_count = 0
        self.diagnostics: list[dict[str, Any]] = []
        CR.CorrectedRun5CollectorV1.__init__(self, calibration=FC._Adapter(latency), seed=seed, mode="FIXTURE",
                                             shared_sources=shared_sources)
        self.collector_binding_sha256 = _sha({
            "schema": self.SCHEMA, "corrected": self.binding_sha256, "arm": arm, "reward_sha256": self.reward_sha256,
            "reward": self.catalog.reward_binding_sha256, "split": self.catalog.split,
            "latency": latency.binding_sha256, "pool": latency.pool, "support": FC.S.SUPPORT_SHA256,
            "evidence_label": EVIDENCE_LABEL})

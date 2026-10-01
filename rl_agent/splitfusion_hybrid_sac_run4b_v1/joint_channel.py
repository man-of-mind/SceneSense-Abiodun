"""RUN4B_JOINT_CHANNEL_COMPARATOR_V1: Run-4B with the Run-5B joint SNR/MCS channel.

Only the MCS process changes.  The SNR-free Run-4 Markov provider is replaced
by the frozen Run-5 ``JointSnrMcsChannelV1``, imported byte-for-byte from
``run5b-no-qperc-v1`` (pinned below), with Run-5B's accepted SNR-tilted
kernel, registered network-profile design and seed rule
``derive_seed(seed, "train-channel")``.  It is advanced exactly as Run-5B
advances it: observe at genesis, then ``advance(2)`` and ``observe()`` once
per decision after the outcome is resolved.  The channel tape is
action-independent, so for each seed it is identical to Run-5B's.

SNR is generated inside the channel to produce MCS.  It is never stored in
the decision context, the observation or the 20-feature state; the state,
reward, SAC settings, operational-latency provider, seeds and checkpoint
rules are the unchanged Run-4B ones.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL

from . import contract as C
from . import environment as E

LABEL = "RUN4B_JOINT_CHANNEL_COMPARATOR_V1"
PILOT_LABEL = "RUN4B_SNR_FREE_CHANNEL_PILOT"
JOINT_ENV_SCHEMA = "splitfusion.run4b.joint_channel_environment.v1"
CHANNEL_SEED_LABEL = "train-channel"
RUN5B_SOURCE_BRANCH = "run5b-no-qperc-v1"
RUN5B_SOURCE_COMMIT = "49430a8e4753387958093841c1896db25bf4c3c1"
REGISTERED_CHANNEL_BINDING_SHA256 = (
    "870bf3558722291f51fac594eda41fba6aca3d9225165fb645de636287518e63")
# Byte-for-byte imports from RUN5B_SOURCE_COMMIT (sha256 of file bytes).
RUN5_IMPORTED_FILES: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_run5_v1/__init__.py":
        "8468acc07597a41bccb957bf851ffef44a86ec93ef96e55a48cb5d9f224b83d4",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_channel.py":
        "6ed62c78ac8b6a1a0e29e8f72525589ed7274780a54fbf458b84b35ae1578118",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py":
        "94d68d5a0e048c35b02a04932c3594185da4977eb2f1d18f62e371bd9a9c1846",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/successor_mcs_snr_audit.py":
        "08758309fa4d380777ce797aa84c86d13059d277f4d470d97cb245a16a037ebb",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py":
        "718315b65a08b71c519b2328539d274800398794c46db363cf4a78b5a272e5ae",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/SUCCESSOR_MCS_SNR_AUDIT.json":
        "295ca76206d9dc9a5181a871c7644f9545edff384213d3e3a01c0886e4b683a6",
}


class JointChannelError(RuntimeError):
    """A joint-channel binding or causality invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JointChannelError(message)


def verify_imported_files(root: Path = E.C2.ROOT) -> None:
    for relpath, expected in RUN5_IMPORTED_FILES.items():
        observed = hashlib.sha256((Path(root) / relpath).read_bytes()).hexdigest()
        _require(observed == expected, f"imported Run-5 file drifted: {relpath}")


@dataclass
class JointSharedSourcesV1(E.SharedSourcesV1):
    snr_kernel: Any = None
    design: Any = None

    @classmethod
    def load(cls, provider: Optional[OPL.OperationalLatencyProviderV1] = None
             ) -> "JointSharedSourcesV1":
        verify_imported_files()
        base = E.SharedSourcesV1.load(provider)
        kernel = J.load_accepted_kernel(E.C2.ROOT)
        _require(kernel.base.binding_sha256 == base.mcs_model.binding_sha256,
                 "tilted kernel base differs from the Run-4 MCS kernel")
        sources = cls(**{name: getattr(base, name) for name in (
            "provider", "catalog", "mcs_model", "mcs_evidence_sha256",
            "mcs_acceptance_sha256", "action_catalog")},
            snr_kernel=kernel, design=J.load_design())
        _require(sources.channel_binding_sha256
                 == REGISTERED_CHANNEL_BINDING_SHA256,
                 "joint channel binding differs from Run-5B's")
        return sources

    @property
    def channel_binding_sha256(self) -> str:
        probe = J.JointSnrMcsChannelV1(kernel=self.snr_kernel,
                                       design=self.design, seed=0)
        return probe.binding_sha256

    def binding_document(self) -> dict[str, Any]:
        document = super().binding_document()
        document.update({
            "schema": JOINT_ENV_SCHEMA,
            "variant": LABEL,
            "mcs_process": "RUN5B_JOINT_SNR_MCS_CHANNEL",
            "channel_binding_sha256": self.channel_binding_sha256,
            "channel_schema": J.SCHEMA_ID,
            "channel_seed_rule": f"derive_seed(seed, {CHANNEL_SEED_LABEL!r})",
            "channel_advance": "observe at genesis; advance(2) then observe per "
                               "decision after the outcome",
            "run5b_source_branch": RUN5B_SOURCE_BRANCH,
            "run5b_source_commit": RUN5B_SOURCE_COMMIT,
            "run5_imported_files": dict(RUN5_IMPORTED_FILES),
            "snr_in_actor_state": False,
        })
        return document


class _ChannelMcsProcessV1:
    """Stands in for the Run-4 Markov provider inside ``step()``."""

    def __init__(self, env: "Run4BJointChannelEnvironmentV1") -> None:
        self._env = env
        self.last_step: Optional[J.ChannelStepV1] = None

    def reset(self) -> int:
        return int(self._env._channel.observe().mcs)

    def step(self) -> J.ChannelStepV1:
        env = self._env
        before = env._channel.observe()
        _require(before.mcs == env.context["prior_ul_mcs"],
                 "channel MCS differs from the decision's observed MCS")
        step = env._channel.advance(J.SUPPORTED_DURATION_TENSORS)
        _require(step.current_mcs == before.mcs
                 and step.current_snr_db == before.snr_db,
                 "channel advanced from a state the decision did not observe")
        env._channel.observe()
        self.last_step = step
        return step


class Run4BJointChannelEnvironmentV1(E.Run4BEnvironmentV1):
    """Run-4B environment whose only change is the joint SNR/MCS channel."""

    def __init__(self, sources: JointSharedSourcesV1, *, seed: int) -> None:
        _require(type(sources) is JointSharedSourcesV1,
                 "joint environment requires JointSharedSourcesV1")
        _require(type(seed) is int and seed >= 0, "seed must be an int >= 0")
        self.sources = sources
        self.seed = seed
        self.scaling = sources.scaling()
        base = seed * 1_000_003
        self._rng = {name: random.Random(base + offset)
                     for name, offset in E.STREAM_OFFSETS.items()
                     if name != "mcs"}
        self._channel = J.JointSnrMcsChannelV1(
            kernel=sources.snr_kernel, design=sources.design,
            seed=J.derive_seed(seed, CHANNEL_SEED_LABEL))
        self._mcs = _ChannelMcsProcessV1(self)
        self._latency_streams = sources.provider.streams(seed)
        self._backlog = 0
        self._mcs_current = self._mcs.reset()
        self.prior = C.OperationalPriorV1.genesis()
        self.decision_seq = 0
        self.context = self._new_context()

    def step(self, mode_id: int, q_e4: int) -> E.TransitionV1:
        transition = super().step(mode_id, q_e4)
        step = self._mcs.last_step
        # Diagnostics only (logged, never part of any state vector).
        transition.diagnostics["channel_snr_db_at_decision"] = step.current_snr_db
        transition.diagnostics["channel_tick_at_decision"] = step.current_tick
        transition.diagnostics["successor_mcs"] = step.successor_mcs
        return transition

    def channel_tape_entry(self) -> list:
        """Current (MCS, SNR) for tape comparison with Run-5B; not a feature."""
        observation = self._channel.observe()
        return [int(observation.mcs), float(observation.snr_db).hex()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": JOINT_ENV_SCHEMA, "seed": self.seed,
            "decision_seq": self.decision_seq,
            "backlog_bytes": self._backlog, "mcs_current": self._mcs_current,
            "prior": self.prior.to_dict(), "context": dict(self.context),
            "rng": {name: E._rng_state_to_json(rng.getstate())
                    for name, rng in sorted(self._rng.items())},
            "channel": self._channel.checkpoint(),
            "latency_streams": self._latency_streams.get_state(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        _require(value["schema"] == JOINT_ENV_SCHEMA,
                 "joint environment schema differs")
        _require(int(value["seed"]) == self.seed, "environment seed differs")
        _require(set(value["rng"]) == set(self._rng), "RNG stream set differs")
        for name, state in value["rng"].items():
            self._rng[name].setstate(E._rng_state_from_json(state))
        self._channel.restore(value["channel"])
        self._latency_streams.set_state(value["latency_streams"])
        self._backlog = int(value["backlog_bytes"])
        self._mcs_current = int(value["mcs_current"])
        self.prior = C.OperationalPriorV1.from_dict(dict(value["prior"]))
        self.decision_seq = int(value["decision_seq"])
        self.context = dict(value["context"])
        _require(self._channel.observe().mcs == self.context["prior_ul_mcs"],
                 "restored channel MCS differs from the decision context")

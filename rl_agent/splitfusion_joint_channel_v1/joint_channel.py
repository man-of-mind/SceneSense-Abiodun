"""Shared joint SNR/MCS channel authority for Run-4B-Joint and Run-5B.

Decision ``USE_JOINT_SNR_MCS_CHANNEL`` (2026-09-30).  Both the Run-4B-Joint
comparator (20-D actor state, SNR withheld) and Run-5B (21-D, SNR exposed)
import this module unchanged and bind :data:`JOINT_CHANNEL_BINDING_SHA256`.

What it is
----------
:class:`JointChannelEnvironmentV1` *is* the Run-4B modeled environment
(``splitfusion_hybrid_sac_run4b_v1.environment.Run4BEnvironmentV1``): its
``step`` (scene draw, shared operational-latency provider
``L_op = A + S + T + E + D``, inclusive 170-ms deadline, reward, operational
prior, backlog advance) is inherited unchanged.  Exactly one source differs:
the uplink MCS process is the frozen Run-5 joint SNR/MCS channel
(``run5_channel.JointSnrMcsChannelV1``: registered network-profile design v2,
four balanced profiles in 300-tick segments, accepted SNR-tilted kernel whose
base is the same Run-4 MCS model Run-4B binds), seeded
``derive(seed, 'train-channel')`` exactly as Run 5.

Causality
---------
The channel exposes only the current ``(snr_db, mcs)``.  It is advanced by
two ticks inside the inherited ``step`` only after the outcome is resolved
(at the point Run 4B advanced its MCS provider); every generated tick is
asserted to be in the future.  The current SNR is recorded in the decision
context, which no reward/outcome code reads; whether the policy may see it is
the caller's observation-space choice.  Scene, transport, A/E/D latency and
channel streams are separate RNG streams; none is consumed by observing SNR.

Evidence is read from an explicit ``evidence_root`` (the main checkout),
because several pinned sources are untracked evidence.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as MCSEV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_acceptance as MCSA
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as MCSP
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import environment as E4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
from rl_agent.splitfusion_hybrid_sac_run5_v1 import successor_mcs_snr_audit as SA
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

SCHEMA = "splitfusion.joint_snr_mcs_channel_environment.v1"
DECISION = "USE_JOINT_SNR_MCS_CHANNEL"
TRAINING_CHANNEL_LABEL = "train-channel"
REGISTERED_PROVIDER_BINDING_SHA256 = (
    "3cc1e6e36ae6e4f43dc8110077c62efbf5c719aaa179935b0f6de4c5bb1f6c29")
REGISTERED_RUN4B_ENVIRONMENT_BINDING_SHA256 = (
    "387903e1cac258dba2766848ecb418da447289d6b3806f8106ec1cb07a67be99")
CHANNEL_ADVANCE_TENSORS = J.SUPPORTED_DURATION_TENSORS
SHARED_SOURCE_FILES = (
    "rl_agent/splitfusion_joint_channel_v1/joint_channel.py",
    "rl_agent/splitfusion_hybrid_sac_run4b_v1/environment.py",
    "rl_agent/splitfusion_hybrid_sac_run4b_v1/contract.py",
    "rl_agent/splitfusion_operational_latency_v1/provider.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_channel.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/successor_mcs_snr_audit.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/SUCCESSOR_MCS_SNR_AUDIT.json",
    "rl_agent/configs/network_profile_design_v2.json",
)


class JointChannelError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JointChannelError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False)
                          .encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Sources (Run-4B's, loaded from an explicit root) + the frozen Run-5 channel
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class JointSourcesV1:
    run4b: E4.SharedSourcesV1
    snr_kernel: Any
    design: Mapping[str, Any]
    evidence_root: str

    def channel(self, seed: int) -> J.JointSnrMcsChannelV1:
        return J.JointSnrMcsChannelV1(kernel=self.snr_kernel, design=self.design, seed=seed)

    def binding_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "decision": DECISION,
            "run4b_environment_binding": self.run4b.binding_document(),
            "run4b_environment_binding_sha256": self.run4b.binding_sha256,
            "replaced_component": {
                "run4b": "MCSP.FitMcsMarkovProviderV1 (SNR-free Run-4 MCS Markov provider)",
                "joint": "run5_channel.JointSnrMcsChannelV1 successor MCS (SNR-tilted kernel)",
                "kernel_base_binding_sha256": SA.REGISTERED_RUN4_KERNEL_BINDING,
            },
            "channel_binding_sha256": self.channel(0).binding_sha256,
            "channel_accepted_audit_sha256": J.ACCEPTED_AUDIT_SHA256,
            "network_profile_design_sha256": J.SNR.NETWORK_PROFILE_DESIGN_SHA256,
            "profiles": list(J.TRAINING_PROFILES),
            "profile_kernel_status": dict(J.PROFILE_KERNEL_STATUS),
            "segment_ticks": J.SEGMENT_TICKS,
            "ticks_per_decision": CHANNEL_ADVANCE_TENSORS * J.TICKS_PER_TENSOR,
            "training_channel_seed": f"run5_channel.derive_seed(seed, '{TRAINING_CHANNEL_LABEL}')",
            "advance_point": ("inside the inherited Run-4B step, after the outcome and reward "
                              "are resolved, where Run 4B advanced its MCS provider"),
            "snr_in_reward_or_outcome": False,
            "snr_in_context": "current causal SNR only; policy exposure is the caller's choice",
            "operational_latency_provider_sha256": self.run4b.provider.binding_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha(self.binding_document())


def load_sources(evidence_root: Path) -> JointSourcesV1:
    """Run-4B ``SharedSourcesV1.load`` with an explicit root, plus the channel."""
    root = Path(evidence_root).resolve()
    C2.verify_preserved(root)
    C2.verify_oai_ceiling(root)
    # The sealed MCS acceptance re-derives its evidence from the default root;
    # run the unchanged check with that one loader pointed at ``root``.
    original = MCSEV.load_dynamic_mcs_273prb_evidence
    with mock.patch.object(MCSA.evidence_module, "load_dynamic_mcs_273prb_evidence",
                           functools.partial(original, repository_root=root)):
        MCSA.load_registered_acceptance()
    evidence = MCSEV.load_dynamic_mcs_273prb_evidence(repository_root=root)
    provider = OPL.OperationalLatencyProviderV1.load(root)
    require(provider.binding_sha256 == REGISTERED_PROVIDER_BINDING_SHA256,
            "operational-latency provider binding differs from the registered digest")
    run4b = E4.SharedSourcesV1(
        provider=provider, catalog=SS.FitSceneCatalog(root),
        mcs_model=MCSP.fit_mcs_markov_model(evidence),
        mcs_evidence_sha256=evidence.canonical_evidence_sha256,
        mcs_acceptance_sha256=C2.sha256_file(C2.ROOT / E4.MCS_ACCEPTANCE_RELPATH),
        action_catalog=action_contract.load_contract())
    require(run4b.binding_sha256 == REGISTERED_RUN4B_ENVIRONMENT_BINDING_SHA256,
            "Run-4B environment sources differ from the registered Run-4B binding")
    kernel = J.load_accepted_kernel(root)
    require(kernel.base.binding_sha256 == run4b.mcs_model.binding_sha256,
            "joint-channel kernel base is not the Run-4B MCS model")
    return JointSourcesV1(run4b=run4b, snr_kernel=kernel, design=J.load_design(),
                          evidence_root=str(root))


def shared_source_hashes(root: Path) -> dict[str, str]:
    return {p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest()
            for p in SHARED_SOURCE_FILES}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ChannelStep:
    successor_mcs: int


class _ChannelMcsSource:
    """Presents the joint channel through the MCS-provider calls Run-4B's step makes."""

    def __init__(self, channel: J.JointSnrMcsChannelV1) -> None:
        self.channel = channel
        self.last_step: J.ChannelStepV1 | None = None

    def reset(self) -> int:
        return int(self.channel.observe().mcs)

    def step(self) -> _ChannelStep:
        step = self.channel.advance(CHANNEL_ADVANCE_TENSORS)
        require(all(t > step.current_tick for t in step.generated_ticks),
                "a generated SNR tick is not in the future")
        self.last_step = step
        observed = self.channel.observe()
        require(observed.mcs == step.successor_mcs, "channel successor/observation differ")
        return _ChannelStep(int(step.successor_mcs))


class JointChannelEnvironmentV1(E4.Run4BEnvironmentV1):
    """Run-4B environment whose MCS source is the frozen joint SNR/MCS channel."""

    def __init__(self, sources: JointSourcesV1, *, seed: int) -> None:
        require(type(sources) is JointSourcesV1, "sources must be JointSourcesV1")
        require(type(seed) is int and seed >= 0, "seed must be an int >= 0")
        # Mirrors Run4BEnvironmentV1.__init__ stream-for-stream; only the MCS
        # source is the joint channel instead of the SNR-free Markov provider.
        self.joint_sources = sources
        self.sources = sources.run4b
        self.seed = seed
        self.scaling = self.sources.scaling()
        base = seed * 1_000_003
        self._rng = {name: random.Random(base + offset)
                     for name, offset in E4.STREAM_OFFSETS.items() if name != "mcs"}
        self._channel = sources.channel(J.derive_seed(seed, TRAINING_CHANNEL_LABEL))
        self._mcs = _ChannelMcsSource(self._channel)
        self._latency_streams = self.sources.provider.streams(seed)
        self._backlog = 0
        self._mcs_current = int(self._mcs.reset())
        self.prior = C4.OperationalPriorV1.genesis()
        self.decision_seq = 0
        self.context = self._new_context()

    def _new_context(self) -> dict[str, Any]:
        context = super()._new_context()
        observed = self._channel.observe()
        require(observed.mcs == context["prior_ul_mcs"], "context MCS is not the channel MCS")
        context["snr_db"] = float(observed.snr_db)
        return context

    def current_snr_db(self) -> float:
        """The current causal SNR (the value at the decision's observed tick)."""
        return float(self.context["snr_db"])

    def step(self, mode_id: int, q_e4: int) -> E4.TransitionV1:
        snr_before = self.current_snr_db()
        transition = super().step(mode_id, q_e4)
        channel_step = self._mcs.last_step
        require(channel_step is not None and channel_step.current_snr_db == snr_before,
                "channel advanced from a state the decision did not observe")
        diagnostics = {**transition.diagnostics, "snr_db": snr_before,
                       "successor_mcs": channel_step.successor_mcs,
                       "successor_snr_db": channel_step.successor_snr_db,
                       "generated_ticks_after_observed": all(
                           t > channel_step.current_tick for t in channel_step.generated_ticks)}
        return dataclasses.replace(transition, diagnostics=diagnostics)

    def profile_balance(self) -> dict[str, int]:
        """Audit-only; never an actor input."""
        return self._channel.profile_balance()

    @property
    def future_sample_violations(self) -> int:
        return self._channel.future_sample_violations

    # -- durable state ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "run4b_schema": E4.ENV_SCHEMA, "seed": self.seed,
            "decision_seq": self.decision_seq,
            "backlog_bytes": self._backlog, "mcs_current": self._mcs_current,
            "prior": self.prior.to_dict(), "context": dict(self.context),
            "rng": {name: E4._rng_state_to_json(rng.getstate())
                    for name, rng in sorted(self._rng.items())},
            "joint_channel": self._channel.checkpoint(),
            "latency_streams": self._latency_streams.get_state(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        require(value["schema"] == SCHEMA, "environment schema differs")
        require(int(value["seed"]) == self.seed, "environment seed differs")
        require(set(value["rng"]) == set(self._rng), "RNG stream set differs")
        for name, state in value["rng"].items():
            self._rng[name].setstate(E4._rng_state_from_json(state))
        self._channel.restore(value["joint_channel"])
        self._latency_streams.set_state(value["latency_streams"])
        self._backlog = int(value["backlog_bytes"])
        self._mcs_current = int(value["mcs_current"])
        self.prior = C4.OperationalPriorV1.from_dict(dict(value["prior"]))
        self.decision_seq = int(value["decision_seq"])
        self.context = dict(value["context"])
        require(self._channel.observe().mcs == self._mcs_current,
                "restored channel MCS differs from the restored context")

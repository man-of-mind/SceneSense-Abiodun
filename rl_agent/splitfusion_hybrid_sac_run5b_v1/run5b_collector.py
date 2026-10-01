"""Run-5B modeled collector: the frozen Run-5 session with a native 21-D state.

Reused unchanged (by inheritance from the frozen Run-5 collector)
-----------------------------------------------------------------
The persistent causal session, the Run-4 sequential environment and kernel
(immediate deadline/latency/reward from the v2 transport model), the queue
advance, the scene streams and RNG seeds, the joint SNR/MCS channel with its
four balanced profiles and channel tapes, and the modeled twin of the live
lease adapter (``_snr_observation``: ACK 20 ms and heartbeat 5 ms before
state commit, then the live ``observe`` rule).

What changes
------------
The actor state.  For each decision the collector assembles a
:class:`Run5BPolicyStateV1` from the guarded measurement observations and a
:class:`TransportPriorOutcomeV1` projected from the preceding cycle's
resolution (Q_perc is never read), runs :func:`guard_run5b_state` with the
frozen lease policy and builds the vector with the native
:func:`build_run5b_policy_features`.

Audit (not construction): after building, the Run-5B values are compared bit
for bit with the transport fields of the Run-4 vector the environment computes
internally, and feature 20 with the Run-5 SNR scaling.  A mismatch is a hard
failure.  The Run-4 ``prev_quality_qperc`` slot has no Run-5B counterpart.

Reward evidence (``q_perc``, ``latency_ms``, ``reward``) is kept on the
transition record for the critic target only.

PENDING: the operational latency is still the inherited Run-4 v2 kernel
composition (retained residual + send span + transport + actor reserve), whose
residual includes ground-truth evaluation time.  It must be replaced by the
Run-4B hash-bound operational-latency provider, imported unchanged, before any
Run-5B smoke or deep training (``run5b_campaign`` refuses until then).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as RC
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR

from . import run5b_state_contract as C

SCHEMA = "scenesense.run5b.modeled_collector.v1"
CHECKPOINT_SCHEMA = "scenesense.run5b.modeled_collector_checkpoint.v1"
WIDTH = C.RUN5B_POLICY_FEATURE_COUNT
build_shared_sources = RC.build_shared_sources
lease_policy = RC.lease_policy

# Run-4 vector positions holding the same transport quantity as Run-5B 0-19.
RUN4_POSITION_OF = tuple(R4.POLICY_FEATURE_ORDER.index(name)
                         for name in C.RUN4B_POLICY_FEATURE_ORDER)


class Run5BCollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Run5BCollectorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False)
                          .encode("ascii")).hexdigest()


def _hex(values: Optional[Tuple[float, ...]]) -> Optional[list[str]]:
    return None if values is None else [float(v).hex() for v in values]


@dataclass(frozen=True, slots=True)
class Run5BCollectedTransitionV1:
    request: orch.ModeledActionRequestV1
    run4_transition_sha256: str
    session_uuid: str
    decision_seq: int
    state: Tuple[float, ...]
    next_state: Optional[Tuple[float, ...]]
    prior_sha256: Optional[str]
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    discount: float
    has_next_state: bool
    terminated: bool
    truncated: bool
    terminal: str
    # Reward evidence only (critic target); never part of any actor state.
    q_perc: Optional[float]
    latency_ms: Optional[float]

    def __post_init__(self) -> None:
        require(len(self.state) == WIDTH, "Run-5B state must be 21-D")
        require(self.next_state is None or len(self.next_state) == WIDTH,
                "Run-5B successor must be 21-D")
        require(self.has_next_state == (self.next_state is not None),
                "successor presence differs from has_next_state")
        require(all(math.isfinite(v) for v in self.state), "non-finite state")
        require(self.next_state is None or all(math.isfinite(v) for v in self.next_state),
                "non-finite successor")
        require(self.duration == 2, "the joint channel supports duration 2 only")
        require(0.0 < self.discount <= 1.0, "discount must lie in (0, 1]")
        require((self.decision_seq == 0) == (self.prior_sha256 is None),
                "only genesis may lack a transport prior")

    @property
    def bootstrap(self) -> bool:
        return self.has_next_state and not self.terminated and not self.truncated

    def ledger_dict(self) -> dict[str, Any]:
        return {"decision_seq": self.decision_seq, "discount": float(self.discount).hex(),
                "duration": self.duration, "has_next_state": self.has_next_state,
                "latency_ms": None if self.latency_ms is None else float(self.latency_ms).hex(),
                "mode_id": self.mode_id, "next_state": _hex(self.next_state),
                "prior_sha256": self.prior_sha256, "q_e4": self.q_e4,
                "q_perc": None if self.q_perc is None else float(self.q_perc).hex(),
                "request": self.request.to_dict(), "reward": float(self.reward).hex(),
                "run4_transition_sha256": self.run4_transition_sha256,
                "session_uuid": self.session_uuid, "state": _hex(self.state),
                "terminal": self.terminal, "terminated": self.terminated,
                "truncated": self.truncated}

    @property
    def digest(self) -> str:
        return _sha(self.ledger_dict())


class Run5BModeledCollectorV1(RC.Run5ModeledCollectorV1):
    SCHEMA = SCHEMA

    def __init__(self, *, artifact_path: Path, seed: int, evidence_root: Path,
                 shared_sources: Mapping[str, Any] | None = None) -> None:
        super().__init__(artifact_path=artifact_path, seed=seed, evidence_root=evidence_root,
                         shared_sources=shared_sources)
        probe = J.JointSnrMcsChannelV1(kernel=self._snr_kernel, design=self._design, seed=0)
        self.run5b_binding_sha256 = _sha({
            "schema": SCHEMA, "run4_collector_binding": self._collector_binding,
            "channel_binding": probe.binding_sha256,
            "lease_policy": self._lease.canonical_sha256(),
            "feature_schema": C.FEATURE_SCHEMA_SHA256,
            "transport_prior": "TransportPriorOutcomeV1.from_reward_resolution"})

    @property
    def collector_binding_sha256(self) -> str:
        return self.run5b_binding_sha256

    def _start_session(self) -> None:
        super()._start_session()
        self._reset_run5b_state()

    def _reset_run5b_state(self) -> None:
        self._last_resolution: Optional[R4.RewardResolutionV1] = None
        self._priors: dict[int, Optional[C.TransportPriorOutcomeV1]] = {}

    # -- native 21-D state ---------------------------------------------------
    def _transport_prior(self, sequence: int,
                         run4_state: R4.PolicyStateV2) -> Optional[C.TransportPriorOutcomeV1]:
        if sequence == 0:
            return None
        resolution = self._last_resolution
        require(resolution is not None
                and resolution.identity.decision_seq + 1 == sequence,
                "the transport prior is not the immediately preceding resolution")
        # Same outcome the environment carried forward (binding check only).
        require(run4_state.previous.reward_resolution_sha256 == resolution.canonical_sha256(),
                "transport prior and environment previous outcome differ")
        return C.TransportPriorOutcomeV1.from_reward_resolution(resolution)

    def _run5b_features(self, bundle, snr_db: float) -> Tuple[float, ...]:
        run4_state = bundle.state.state
        sequence = run4_state.identity.decision_seq
        prior = self._transport_prior(sequence, run4_state)
        state = C.Run5BPolicyStateV1(
            identity=run4_state.identity, camera_si=run4_state.camera_si,
            radar_p40=run4_state.radar_p40, prior_ul_mcs=run4_state.prior_ul_mcs,
            pre_action_rlc_backlog=run4_state.pre_action_rlc_backlog, previous=prior)
        guarded = C.guard_run5b_state(state, self._snr_observation(bundle, snr_db),
                                      bundle.state.boundary, self._provider.freshness,
                                      self._lease)
        values = C.build_run5b_policy_features(guarded, self._provider.scaling).as_tuple()
        self._audit_against_environment(values, bundle.features.as_tuple(), snr_db)
        self._priors[sequence] = prior
        return values

    @staticmethod
    def _audit_against_environment(values, run4_values, snr_db: float) -> None:
        mirrored = [float(run4_values[i]).hex() for i in RUN4_POSITION_OF]
        require([v.hex() for v in values[:C.RUN4B_POLICY_FEATURE_COUNT]] == mirrored,
                "Run-5B transport features differ from the environment's measured values")
        require(values[C.SNR_FEATURE_INDEX] == SNR.scale_snr_db(snr_db),
                "Run-5B SNR feature differs from the frozen Run-5 scaling")

    def current_state_features(self) -> Tuple[float, ...]:
        bundle = self._env.current_state
        sequence = bundle.state.state.identity.decision_seq
        if sequence not in self._feature_cache:
            self._feature_cache[sequence] = self._run5b_features(bundle, self.context.snr_db)
        return self._feature_cache[sequence]

    def transport_prior(self, sequence: int) -> Optional[C.TransportPriorOutcomeV1]:
        return self._priors[sequence]

    # -- one decision ------------------------------------------------------
    def collect(self, request: orch.ModeledActionRequestV1) -> Run5BCollectedTransitionV1:
        require(type(request) is orch.ModeledActionRequestV1, "collector request is foreign")
        require(request.decision_ordinal == self.decision_count, "out-of-order decision ordinal")
        state = self.current_state_features()
        current = self._env.current_state.state.state
        sequence = current.identity.decision_seq
        require(sequence == request.decision_ordinal,
                "environment decision_seq diverged from the decision ordinal")
        prior = self._priors[sequence]
        action = self._action_identity(request.mode_id, request.q_e4)
        cycle = self._env.step(action)
        require(type(cycle) is environment.CalibratedEmpiricalCycleV1,
                f"environment returned {type(cycle).__name__}")
        transition = cycle.export_for_replay()
        resolution = transition.reward_resolution
        require(resolution.identity.decision_seq == sequence, "resolution identity differs")
        self._last_resolution = resolution
        context = self.previous_context
        require(context is not None and context.resolved is not None and context.advanced,
                "the completed cycle has no resolved, advanced context")
        boundary = transition.episode_boundary
        next_state = None
        if boundary is R4.EpisodeBoundary.CONTINUES:
            next_state = self.current_state_features()
        record = Run5BCollectedTransitionV1(
            request=request, run4_transition_sha256=transition.canonical_sha256(),
            session_uuid=current.identity.session_uuid, decision_seq=sequence,
            state=state, next_state=next_state,
            prior_sha256=None if prior is None else prior.canonical_sha256(),
            mode_id=action.mode_id, q_e4=action.q_e4, reward=float(resolution.reward),
            duration=int(transition.duration), discount=float(transition.discount),
            has_next_state=next_state is not None,
            terminated=boundary is R4.EpisodeBoundary.TERMINATED,
            truncated=boundary is R4.EpisodeBoundary.TRUNCATED,
            terminal=resolution.terminal.value, q_perc=resolution.q_perc,
            latency_ms=resolution.latency_ms)
        resolved = context.resolved
        step = context.channel_step
        self._diagnostics.append({
            "decision_ordinal": request.decision_ordinal, "mode_id": action.mode_id,
            "q_e4": action.q_e4, "camera_si": context.camera_si,
            "radar_p40": context.radar_p40, "prior_ul_mcs": context.prior_ul_mcs,
            "snr_db": context.snr_db, "successor_mcs": step.successor_mcs,
            "successor_snr_db": step.successor_snr_db,
            "generated_ticks_after_observed": all(t > step.current_tick
                                                  for t in step.generated_ticks),
            "pre_enqueue_backlog_bytes": context.backlog_bytes,
            "next_backlog_bytes": self._backlog_bytes,
            "reward_frame_wire_bytes": int(resolved["reward_draw"].wire_bytes),
            "ingress_bytes": resolved["ingress_bytes"],
            "on_time_probability": resolved["prediction"].on_time_probability,
            "on_time": resolved["on_time"], "reward": float(resolution.reward),
            "terminal": resolution.terminal.value,
            "reward_scene_key": resolved["reward_draw"].scene_key,
            "held_scene_key": resolved["held_draw"].scene_key,
            "previous_present": prior is not None,
            "previous_mode_id": prior.action.mode_id if prior else None,
            "previous_q_e4": prior.action.q_e4 if prior else None,
            "previous_success": prior.success if prior else None,
            "previous_operational_latency_ms": prior.operational_latency_ms if prior else None,
        })
        self._history.append(record)
        self._actions.append(request)
        return record

    # -- durable state -----------------------------------------------------
    def checkpoint(self) -> RC.Run5CollectorCheckpointV1:
        payload = {
            "seed": self._seed, "backlog_bytes": self._backlog_bytes,
            "mcs_current": self._mcs_current, "snr_current_hex": float(self._snr_current).hex(),
            "channel_checkpoint_sha256": self._channel.checkpoint_sha256(),
            "actions": [item.to_dict() for item in self._actions],
            "diagnostics_sha256": _sha(self._diagnostics),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
        return RC.Run5CollectorCheckpointV1(
            CHECKPOINT_SCHEMA, self.run5b_binding_sha256, self.decision_count,
            tuple(item.digest for item in self._history), text,
            hashlib.sha256(text.encode("ascii")).hexdigest())

    def restore(self, checkpoint: RC.Run5CollectorCheckpointV1) -> None:
        require(type(checkpoint) is RC.Run5CollectorCheckpointV1, "checkpoint has a foreign type")
        require(checkpoint.schema == CHECKPOINT_SCHEMA, "collector checkpoint schema differs")
        require(checkpoint.collector_binding_sha256 == self.run5b_binding_sha256,
                "collector checkpoint binding differs")
        payload = json.loads(checkpoint.payload_json)
        require(int(payload["seed"]) == self._seed, "collector checkpoint seed differs")
        self._start_session()
        for item in payload["actions"]:
            self.collect(orch.ModeledActionRequestV1(**item))
        require(self.decision_count == checkpoint.decision_count, "restored count differs")
        require(tuple(item.digest for item in self._history) == checkpoint.transition_digests,
                "restored transition digests differ")
        require(self._backlog_bytes == int(payload["backlog_bytes"]), "restored backlog differs")
        require(self._mcs_current == int(payload["mcs_current"]), "restored MCS differs")
        require(float(self._snr_current).hex() == payload["snr_current_hex"],
                "restored SNR differs")
        require(self._channel.checkpoint_sha256() == payload["channel_checkpoint_sha256"],
                "restored joint-channel state differs")
        require(_sha(self._diagnostics) == payload["diagnostics_sha256"],
                "restored diagnostics differ")

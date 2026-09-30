"""Run-5 modeled collector: the Run-4 v2 collector plus the joint SNR/MCS channel.

What is reused unchanged
------------------------
The Run-4 ``RealModeledTransitionCollectorV1`` machinery: one persistent causal
session, the Run-4 sequential environment, the Run-4 ``_RealStateProvider``
(features 0-20), the Run-4 ``_RealKernel`` (the immediate deadline / latency /
reward draw from the v2 transport model over backlog, wire bytes and prior
MCS), the queue advance, the scene streams and the RNG seeds.

What changes
------------
* The MCS advance.  ``advance_once`` calls :class:`JointSnrMcsChannelV1`
  instead of the Run-4 SNR-free Markov provider.  The call happens after the
  cycle outcome is resolved and before the successor state is built, which is
  exactly where Run 4 advanced MCS.
* Feature 22.  Each current and successor state gets the v2 SNR lease
  observation and the v2 22-D vector; positions 0-20 are asserted bit-equal
  to the attested Run-4 transition's features.

The immediate outcome never reads SNR: ``_RealKernel.execute_cycle`` is Run
4's and its transport prediction takes only backlog, bytes and MCS.

Evidence is read from ``evidence_root`` (the main checkout, read-only) because
several registered sources are untracked there.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as EV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import collector_v1 as CV
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import run5_channel as J
from . import run5_snr_v2 as SNR

SCHEMA = "scenesense.run5.modeled_collector.v1"
CHECKPOINT_SCHEMA = "scenesense.run5.modeled_collector_checkpoint.v1"
REGISTERED_ARTIFACT_SHA256 = "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"
# Two 100-ms controller periods.  A training bound, not a live-qualified one.
LEASE_MAX_HEARTBEAT_AGE_NS = 200_000_000
SNR_EFFECTIVE_LEAD_NS = 20_000_000
SNR_HEARTBEAT_LEAD_NS = 5_000_000


class Run5CollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Run5CollectorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False)
                          .encode("ascii")).hexdigest()


def _hex(values: Optional[Tuple[float, ...]]) -> Optional[list[str]]:
    return None if values is None else [float(v).hex() for v in values]


def lease_policy() -> SNR.SnrLeasePolicyV1:
    return SNR.SnrLeasePolicyV1(
        policy_id="run5-modeled-controller-lease", policy_version=1,
        evidence_sha256=SNR.NETWORK_PROFILE_DESIGN_SHA256,
        max_heartbeat_age_ns=LEASE_MAX_HEARTBEAT_AGE_NS)


def build_shared_sources(artifact_path: Path, evidence_root: Path) -> dict[str, Any]:
    root = Path(evidence_root).resolve()
    require(hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
            == REGISTERED_ARTIFACT_SHA256, "transport artifact is not the registered v2b model")
    C2.verify_preserved(root)
    C2.verify_oai_ceiling(root)
    kernel = J.load_accepted_kernel(root)
    return {
        "model": A2.ProductionTransportModelV2.load(Path(artifact_path)),
        "catalog": SS.FitSceneCatalog(root),
        "retained_residuals": CV.load_retained_residuals(root),
        "actor_reserve_ns": CV.load_actor_reserve_ns(root),
        "action_catalog": action_contract.load_contract(),
        "mcs_model": kernel.base,
        "mcs_evidence_sha256": EV.EXPECTED_CANONICAL_EVIDENCE_SHA256,
        "snr_kernel": kernel,
        "design": J.load_design(),
        "evidence_root": str(root),
    }


@dataclass(frozen=True, slots=True)
class Run5CollectedTransitionV1:
    request: orch.ModeledActionRequestV1
    run4_transition_sha256: str
    session_uuid: str
    decision_seq: int
    state: Tuple[float, ...]
    next_state: Optional[Tuple[float, ...]]
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    discount: float
    has_next_state: bool
    terminated: bool
    truncated: bool
    terminal: str
    q_perc: Optional[float]
    latency_ms: Optional[float]

    def __post_init__(self) -> None:
        require(len(self.state) == 22, "Run-5 state must be 22-D")
        require(self.next_state is None or len(self.next_state) == 22,
                "Run-5 successor must be 22-D")
        require(self.has_next_state == (self.next_state is not None),
                "successor presence differs from has_next_state")
        require(all(math.isfinite(v) for v in self.state), "non-finite state")
        require(self.next_state is None or all(math.isfinite(v) for v in self.next_state),
                "non-finite successor")
        require(self.duration == 2, "Run-5 channel supports duration 2 only")
        require(0.0 < self.discount <= 1.0, "discount must lie in (0, 1]")

    @property
    def bootstrap(self) -> bool:
        return self.has_next_state and not self.terminated and not self.truncated

    def ledger_dict(self) -> dict[str, Any]:
        return {"decision_seq": self.decision_seq, "discount": float(self.discount).hex(),
                "duration": self.duration, "has_next_state": self.has_next_state,
                "latency_ms": None if self.latency_ms is None else float(self.latency_ms).hex(),
                "mode_id": self.mode_id, "next_state": _hex(self.next_state),
                "q_e4": self.q_e4,
                "q_perc": None if self.q_perc is None else float(self.q_perc).hex(),
                "request": self.request.to_dict(), "reward": float(self.reward).hex(),
                "run4_transition_sha256": self.run4_transition_sha256,
                "session_uuid": self.session_uuid, "state": _hex(self.state),
                "terminal": self.terminal, "terminated": self.terminated,
                "truncated": self.truncated}

    @property
    def digest(self) -> str:
        return _sha(self.ledger_dict())


@dataclass(frozen=True, slots=True)
class Run5CollectorCheckpointV1:
    schema: str
    collector_binding_sha256: str
    decision_count: int
    transition_digests: Tuple[str, ...]
    payload_json: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"collector_binding_sha256": self.collector_binding_sha256,
                "decision_count": self.decision_count, "payload_json": self.payload_json,
                "payload_sha256": self.payload_sha256, "schema": self.schema,
                "transition_digests": list(self.transition_digests)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Run5CollectorCheckpointV1":
        require(set(value) == {"collector_binding_sha256", "decision_count", "payload_json",
                               "payload_sha256", "schema", "transition_digests"},
                "collector checkpoint fields differ")
        require(hashlib.sha256(value["payload_json"].encode("ascii")).hexdigest()
                == value["payload_sha256"], "collector checkpoint payload tampered")
        return cls(value["schema"], value["collector_binding_sha256"],
                   int(value["decision_count"]), tuple(value["transition_digests"]),
                   value["payload_json"], value["payload_sha256"])


class Run5ModeledCollectorV1(CV.RealModeledTransitionCollectorV1):
    SCHEMA = SCHEMA

    def __init__(self, *, artifact_path: Path, seed: int, evidence_root: Path,
                 shared_sources: Mapping[str, Any] | None = None) -> None:
        root = Path(evidence_root).resolve()
        if shared_sources is None:
            shared_sources = build_shared_sources(Path(artifact_path), root)
        self._snr_kernel = shared_sources["snr_kernel"]
        self._design = shared_sources["design"]
        self._lease = lease_policy()
        self._shared = dict(shared_sources)
        super().__init__(artifact_path=Path(artifact_path), seed=seed, repo_root=root,
                         shared_sources=shared_sources)
        probe = J.JointSnrMcsChannelV1(kernel=self._snr_kernel, design=self._design, seed=0)
        self.run5_binding_sha256 = _sha({
            "schema": SCHEMA, "run4_collector_binding": self._collector_binding,
            "channel_binding": probe.binding_sha256,
            "lease_policy": self._lease.canonical_sha256(),
            "feature_schema_v2": SNR.FEATURE_SCHEMA_SHA256})

    def shared_sources(self) -> dict[str, Any]:
        return dict(self._shared)

    @property
    def collector_binding_sha256(self) -> str:
        return self.run5_binding_sha256

    # -- session / advance ----------------------------------------------
    def _start_session(self) -> None:
        self._scene_rng = random.Random(self._seed * 1_000_003 + 11)
        self._held_scene_rng = random.Random(self._seed * 1_000_003 + 67)
        self._transport_rng = random.Random(self._seed * 1_000_003 + 23)
        self._residual_rng = random.Random(self._seed * 1_000_003 + 37)
        self._channel = J.JointSnrMcsChannelV1(
            kernel=self._snr_kernel, design=self._design,
            seed=J.derive_seed(self._seed, "train-channel"))
        # The same causal interface the live controller uses; only the event
        # times come from the modeled CLOCK_MONOTONIC_RAW timeline.
        self._snr_adapter = SNR.ModeledLeaseSnrAdapterV1(
            provider_id=SCHEMA, session_uuid=self._session_uuid(), ue_id=CV.UE_ID)
        self._take_channel_observation()
        self._backlog_bytes = 0
        self._history = []
        self._actions = []
        self._diagnostics = []
        self._feature_cache: dict[int, Tuple[float, ...]] = {}
        self.previous_context = None
        self.context = self._new_context()
        self._env = self._build_env()
        self._env.reset(session_uuid=self._session_uuid(), ue_id=CV.UE_ID)

    def _take_channel_observation(self) -> None:
        observation = self._channel.observe()
        self._mcs_current = observation.mcs
        self._snr_current = observation.snr_db

    def _new_context(self):
        context = super()._new_context()
        context.snr_db = float(self._snr_current)
        return context

    def advance_once(self) -> None:
        context = self.context
        require(context is not None, "no decision context to advance from")
        if context.advanced:
            return
        resolved = context.resolved
        require(resolved is not None, "cannot advance before the kernel resolved the cycle")
        self._backlog_bytes = int(round(self.model.predict_next_backlog_bytes(
            pre_enqueue_backlog_bytes=float(context.backlog_bytes),
            deterministic_action_ingress_bytes=resolved["ingress_bytes"],
            prior_ul_mcs=int(context.prior_ul_mcs))))
        step = self._channel.advance(J.SUPPORTED_DURATION_TENSORS)
        require(step.current_mcs == context.prior_ul_mcs
                and step.current_snr_db == context.snr_db,
                "channel advanced from a state the decision did not observe")
        self._take_channel_observation()
        context.channel_step = step
        context.advanced = True
        self.previous_context = context
        self.context = self._new_context()

    # -- 22-D features ---------------------------------------------------
    def _snr_observation(self, bundle, snr_db: float) -> SNR.UlSnrLeaseObservationV1:
        """Modeled controller: ACK the active command, renew its lease, then observe.

        The command carrying the current tick's target is ACKed 20 ms and the
        lease renewed 5 ms before state commit; ``observe`` then applies the
        exact live selection rule at the commit cutoff.
        """
        boundary = bundle.state.boundary
        require(boundary.clock_domain == SNR.LIVE_CLOCK_DOMAIN,
                "modeled decisions must use CLOCK_MONOTONIC_RAW")
        commit = boundary.state_commit_timestamp_ns
        command_id = f"run5-modeled-effective-command-{boundary.identity.decision_seq}"
        self._snr_adapter.record_command_ack_at(
            at_ns=commit - SNR_EFFECTIVE_LEAD_NS, command_id=command_id, status="ACK",
            clamped=False, target_snr_db=float(snr_db))
        self._snr_adapter.record_heartbeat_at(at_ns=commit - SNR_HEARTBEAT_LEAD_NS,
                                              active_command_id=command_id)
        return self._snr_adapter.observe(boundary)

    def _run5_features(self, bundle, snr_db: float) -> Tuple[float, ...]:
        guarded = SNR.guard_run5_state_v2(
            bundle.state.state, self._snr_observation(bundle, snr_db), bundle.state.boundary,
            self._provider.freshness, self._lease)
        values = SNR.build_run5_features_v2(guarded, self._provider.scaling).as_tuple()
        run4 = bundle.features.as_tuple()
        require([v.hex() for v in values[:21]] == [float(v).hex() for v in run4],
                "Run-5 positions 0-20 differ from the Run-4 vector")
        return values

    def current_state_features(self) -> Tuple[float, ...]:
        bundle = self._env.current_state
        sequence = bundle.state.state.identity.decision_seq
        if sequence not in self._feature_cache:
            self._feature_cache[sequence] = self._run5_features(bundle, self.context.snr_db)
        return self._feature_cache[sequence]

    # -- one decision ------------------------------------------------------
    def collect(self, request: orch.ModeledActionRequestV1) -> Run5CollectedTransitionV1:
        require(type(request) is orch.ModeledActionRequestV1, "collector request is foreign")
        require(request.decision_ordinal == self.decision_count, "out-of-order decision ordinal")
        state = self.current_state_features()
        current = self._env.current_state.state.state
        require(current.identity.decision_seq == request.decision_ordinal,
                "environment decision_seq diverged from the decision ordinal")
        previous_state = current.previous
        action = self._action_identity(request.mode_id, request.q_e4)
        cycle = self._env.step(action)
        require(type(cycle) is environment.CalibratedEmpiricalCycleV1,
                f"environment returned {type(cycle).__name__}")
        transition = cycle.export_for_replay()
        require([float(v).hex() for v in transition.state_features.as_tuple()]
                == [v.hex() for v in state[:21]], "Run-4 transition state differs from prefix")
        context = self.previous_context
        require(context is not None and context.resolved is not None and context.advanced,
                "the completed cycle has no resolved, advanced context")
        boundary = transition.episode_boundary
        next_state = None
        if boundary is R4.EpisodeBoundary.CONTINUES:
            next_state = self.current_state_features()
            require([float(v).hex() for v in transition.next_state_features.as_tuple()]
                    == [v.hex() for v in next_state[:21]],
                    "Run-4 successor features differ from the Run-5 prefix")
        resolution = transition.reward_resolution
        record = Run5CollectedTransitionV1(
            request=request, run4_transition_sha256=transition.canonical_sha256(),
            session_uuid=current.identity.session_uuid,
            decision_seq=current.identity.decision_seq, state=state, next_state=next_state,
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
            "previous_present": previous_state is not None,
            "previous_mode_id": previous_state.action.mode_id if previous_state else None,
            "previous_q_e4": previous_state.action.q_e4 if previous_state else None,
            "previous_success": bool(previous_state.success) if previous_state else None,
        })
        self._history.append(record)
        self._actions.append(request)
        return record

    # -- durable state -----------------------------------------------------
    def checkpoint(self) -> Run5CollectorCheckpointV1:
        payload = {
            "seed": self._seed, "backlog_bytes": self._backlog_bytes,
            "mcs_current": self._mcs_current, "snr_current_hex": float(self._snr_current).hex(),
            "channel_checkpoint_sha256": self._channel.checkpoint_sha256(),
            "actions": [item.to_dict() for item in self._actions],
            "diagnostics_sha256": _sha(self._diagnostics),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
        return Run5CollectorCheckpointV1(
            CHECKPOINT_SCHEMA, self.run5_binding_sha256, self.decision_count,
            tuple(item.digest for item in self._history), text,
            hashlib.sha256(text.encode("ascii")).hexdigest())

    def channel_checkpoint(self) -> dict[str, Any]:
        return self._channel.checkpoint()

    def restore(self, checkpoint: Run5CollectorCheckpointV1) -> None:
        require(type(checkpoint) is Run5CollectorCheckpointV1, "checkpoint has a foreign type")
        require(checkpoint.schema == CHECKPOINT_SCHEMA, "collector checkpoint schema differs")
        require(checkpoint.collector_binding_sha256 == self.run5_binding_sha256,
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

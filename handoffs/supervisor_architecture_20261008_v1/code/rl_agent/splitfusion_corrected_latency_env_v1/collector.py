"""Corrected modeled collectors for the original quality-aware Run-4 (21-D) and Run-5 (22-D) actors.

Replaces only the latency/outcome kernel of ``ue_production_transport_model_v2.collector_v1`` (imported and left
unchanged). State definition, feature scaling, quality targets, reward and the inclusive 170-ms deadline are the
original ones:

* action-open -> UE receipt is a measured joint stage vector (``chains``), drawn whole from one chain in the same
  exact action stratum and scene neighbourhood, never from pooled marginals; the stale action-50 residual and its
  September-16 GT tail are gone;
* the scoring request waits for both the prediction and the exact GT (single score worker), so GT preparation is
  concurrent and is never added to the prediction path;
* the uplink is moved to the training payload/channel state only through the pinned transport model inside its
  bound validation scope;
* outcomes follow the live controller (``outcome``): timely defined -> SUCCESS, timely undefined -> excluded
  EVALUATOR_FAULT and a fresh-session genesis, late or missing -> TIMEOUT at 170 ms + 1 ns, holds create no state.

Unsupported terms raise ``UncalibratedTermError``. Mode ``FIXTURE`` runs the unchanged environment as labelled
synthetic mechanics (nothing replay-exportable); mode ``CALIBRATED`` requires an explicit approval and is blocked.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2
from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as MCSEV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as MCSP
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2

from . import CANDIDATE_VERSION
from .calibration import (ROOT, TRANSPORT_ARTIFACT_RELPATH, TRANSPORT_ARTIFACT_SHA256, CalibrationSetV1,
                          require_launch_approval, sha256_file)
from .chains import compose
from .outcome import classify, receipt_elapsed_ns
from .scene_catalog import FitSceneCatalogWithUndefinedV1

CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
UE_ID = "ue-modeled-corrected-v1"
MODES = ("FIXTURE", "CALIBRATED")
GENESIS_OPEN_NS = 2_000_000_000


class CollectorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False, default=str).encode("ascii")).hexdigest()


def load_shared_sources(repo_root: Path = ROOT) -> dict[str, Any]:
    """Read-only sources shared by every collector instance (load once, reuse across seeds/tests)."""
    artifact = repo_root / TRANSPORT_ARTIFACT_RELPATH
    _require(sha256_file(artifact) == TRANSPORT_ARTIFACT_SHA256, "transport artifact drifted")
    catalog = FitSceneCatalogWithUndefinedV1(repo_root)
    values = [catalog.scene_descriptors(key)[0] for key in catalog.defined_keys]
    mean = sum(values) / len(values)
    scale = max(1e-6, (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)
    scaling = live_state_v2.TRAINING_SCALING
    # The state definition is unchanged: scaling stays the original 476-scene value even though undefined
    # scenes are now drawable.
    _require(abs(mean - scaling.camera_si_center) < 1e-9 and abs(scale - scaling.camera_si_scale) < 1e-9,
             "catalog-derived SI scaling differs from the training scaling")
    mcs_evidence = MCSEV.load_dynamic_mcs_273prb_evidence()
    return {"model": A2.ProductionTransportModelV2.load(artifact), "catalog": catalog,
            "action_catalog": action_contract.load_contract(),
            "mcs_model": MCSP.fit_mcs_markov_model(mcs_evidence)}


@dataclass
class _Context:
    reward_scene_key: str
    held_scene_key: str
    camera_si: float
    radar_p40: float
    prior_ul_mcs: int
    backlog_bytes: int
    profile: Optional[str]
    action_open_ns: int = 0
    cycle_end_ns: int = 0
    resolution_ns: int = 0
    advanced: bool = False
    resolved: Optional[dict[str, Any]] = None


class _StateProvider:
    def __init__(self, collector: "CorrectedRun4CollectorV1") -> None:
        self._c = collector
        self.freshness = contract.FreshnessPolicyV2(
            policy_id="corrected-latency-env-v1-freshness", policy_version=1,
            evidence_sha256=collector.binding_sha256,
            camera_si_max_age_ns=C2.MAX_CAUSAL_AGE_NS, radar_p40_max_age_ns=C2.MAX_CAUSAL_AGE_NS,
            prior_ul_mcs_max_age_ns=C2.MAX_CAUSAL_AGE_NS, pre_action_rlc_backlog_max_age_ns=C2.MAX_CAUSAL_AGE_NS)
        self.scaling = live_state_v2.TRAINING_SCALING   # unchanged state definition

    def _metadata(self, identity, *, sample_seq, kind, observer, direction, source_ns, available_ns):
        return contract.MeasurementMetadataV1(
            identity=contract.SampleIdentityV1(identity.session_uuid, identity.ue_id, sample_seq),
            kind=kind, observer=observer, link_direction=direction, source=self._c.SCHEMA,
            source_timestamp_ns=source_ns, available_timestamp_ns=available_ns, clock_domain=CLOCK_DOMAIN, valid=True)

    def build_state(self, request) -> environment.DecisionStateBundleV1:
        c = self._c
        if c.context.resolved is not None:
            c.advance_once()                     # after the outcome, before the state that observes it
        previous = c.previous_context            # None only for the very first decision of the collector
        if previous is None:
            open_ns = GENESIS_OPEN_NS
            commit_ns = open_ns - 10_000_000
        else:
            minimum_commit = max(request.minimum_state_commit_timestamp_ns, previous.resolution_ns)
            cadence_open = previous.action_open_ns + contract.TRANSMIT_PERIOD_NS * contract.MINIMUM_HOLD_TENSORS
            target_open = max(minimum_commit + 40_000_000, cadence_open + 10_000_000, previous.cycle_end_ns + 10_000_000)
            commit_ns = max(minimum_commit, target_open - 40_000_000)
            open_ns = max(target_open, commit_ns + 1)
        context = c.context
        context.action_open_ns = open_ns
        source_ns, available_ns = commit_ns - 20_000_000, commit_ns - 10_000_000
        seq = request.identity.decision_seq
        meta = lambda kind, observer, direction, offset: self._metadata(
            request.identity, sample_seq=offset + seq, kind=kind, observer=observer, direction=direction,
            source_ns=source_ns, available_ns=available_ns)
        state = contract.PolicyStateV2(
            identity=request.identity,
            camera_si=contract.ScalarObservationV1(context.camera_si, meta(
                contract.MeasurementKind.CAMERA_SI, contract.Observer.SCENE_PIPELINE,
                contract.LinkDirection.NOT_APPLICABLE, 100), None),
            radar_p40=contract.ScalarObservationV1(context.radar_p40, meta(
                contract.MeasurementKind.RADAR_P40, contract.Observer.SCENE_PIPELINE,
                contract.LinkDirection.NOT_APPLICABLE, 100), None),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=contract.ScalarObservationV1(context.prior_ul_mcs, meta(
                    contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX, contract.Observer.UE,
                    contract.LinkDirection.UPLINK, 200), None),
                mcs_table=contract.UL_MCS_TABLE_ID, harq_round=0, new_data_indicator=1,
                grant_identity=f"modeled-ul-grant-{request.identity.session_uuid}-{seq}",
                scheduler_policy_id=contract.UL_MCS_POLICY_ID, selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID),
            pre_action_rlc_backlog=contract.ScalarObservationV1(context.backlog_bytes, meta(
                contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES, contract.Observer.UE,
                contract.LinkDirection.UPLINK, 300), None),
            previous=request.previous)
        boundary = contract.DecisionBoundaryV1(identity=request.identity, state_commit_timestamp_ns=commit_ns,
                                               action_open_timestamp_ns=open_ns, clock_domain=CLOCK_DOMAIN)
        guarded = contract.guard_state_for_action(state, boundary, self.freshness)
        return environment.DecisionStateBundleV1(guarded, contract.build_policy_features(guarded, self.scaling))


class _Kernel:
    def __init__(self, collector: "CorrectedRun4CollectorV1") -> None:
        self._c = collector

    def execute_cycle(self, request) -> environment.KernelCycleResultV1:
        c = self._c
        context = c.context
        identity = request.state.state.state.identity
        opened = request.state.state.boundary.action_open_timestamp_ns
        action = request.action
        q_reward, total_reward, wire_reward, anchor, low, high = c.catalog.draw(
            context.reward_scene_key, mode_id=action.mode_id, q_e4=action.q_e4)
        _, total_held, wire_held, _, held_low, held_high = c.catalog.draw(
            context.held_scene_key, mode_id=action.mode_id, q_e4=action.q_e4)
        draw = c.calibration.draw(
            mode_id=action.mode_id, q_e4=action.q_e4, profile=context.profile, camera_si=context.camera_si,
            radar_p40=context.radar_p40, prior_ul_mcs=context.prior_ul_mcs, backlog_bytes=context.backlog_bytes,
            wire_bytes=wire_reward, rng=c.chain_rng)
        busy = c.score_channel_busy_until_ns
        carry_in = None if busy is None else (busy - opened) / 1e6     # explicit cross-decision channel queueing
        composed = compose(draw.vector, carry_in=carry_in)
        c.score_channel_busy_until_ns = opened + int(round(composed.channel_busy_until * 1_000_000))
        outcome = classify(receipt_elapsed_ns(composed.receipt), q_reward)
        resolution_ns = opened + outcome.resolution_elapsed_ns
        event = contract.RewardEventV1(
            identity=identity, action=action, kind=outcome.kind, action_open_timestamp_ns=opened,
            resolution_timestamp_ns=resolution_ns, clock_domain=CLOCK_DOMAIN, source=c.SCHEMA,
            q_perc=outcome.q_perc if outcome.kind is contract.RewardEventKind.DELIVERED_SUCCESS else None)
        base = identity.decision_seq * 10
        tensors = tuple(
            contract.HoldTensorV1(
                tensor_seq=base + i, offered_payload_bytes=int(total),
                payload_evidence_class=contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE,
                payload_provenance_sha256=_sha({"scene": key, "row_low": lo, "row_high": hi}),
                reward_requested=i == 0)
            for i, (total, key, lo, hi) in enumerate(((total_reward, context.reward_scene_key, low, high),
                                                      (total_held, context.held_scene_key, held_low, held_high))))
        hold = contract.ActionHoldV1(identity=identity, action=action, tensors=tensors)
        cycle_end = max(resolution_ns + 1, opened + V1.DURATION_NS)
        context.cycle_end_ns = cycle_end
        context.resolution_ns = resolution_ns
        context.resolved = {
            "q_perc_grid": q_reward, "q_defined": q_reward is not None, "is_anchor": anchor,
            "wire_reward": wire_reward, "wire_held": wire_held, "ingress_bytes": int(wire_reward) + int(wire_held),
            "chain_id": draw.chain_id, "donor_chain_id": draw.donor_chain_id, "stratum_size": draw.stratum_size,
            "scene_distance": draw.scene_distance, "uplink_delta_ms": draw.uplink_delta_ms,
            "ordering": composed.ordering.value, "prediction_at_service_ms": composed.prediction_at_service,
            "gt_ready_ms": composed.gt_ready, "scoring_end_ms": composed.scoring_end,
            "score_channel_carry_in_ms": carry_in, "score_channel_busy_until_ms": composed.channel_busy_until,
            "receipt_ms": composed.receipt, "critical_path": [list(x) for x in composed.critical_path],
            "event_kind": outcome.kind.value, "feedback_class": outcome.feedback_class,
            "resolution_elapsed_ns": outcome.resolution_elapsed_ns, "opened_ns": opened}
        return environment.KernelCycleResultV1(hold=hold, reward_event=event, cycle_end_timestamp_ns=cycle_end,
                                               episode_boundary=contract.EpisodeBoundary.CONTINUES)


class CorrectedRun4CollectorV1:
    """One causal session per uninterrupted run; an excluded fault starts a fresh-session genesis, as live."""

    SCHEMA = "splitfusion.corrected_latency_env.run4_collector.v1"

    def __init__(self, *, calibration: CalibrationSetV1, seed: int, mode: str = "FIXTURE",
                 shared_sources: Optional[Mapping[str, Any]] = None) -> None:
        _require(mode in MODES, f"mode must be one of {MODES}")
        _require(type(seed) is int and seed >= 0, "seed must be an exact int >= 0")
        if mode == "CALIBRATED":
            require_launch_approval(calibration)       # raises LaunchBlockedError until explicitly approved
        self.mode = mode
        self.calibration = calibration
        self._seed = seed
        sources = dict(shared_sources) if shared_sources is not None else load_shared_sources()
        self._shared = sources
        self.model = sources["model"]
        self.catalog = sources["catalog"]
        self.action_catalog = sources["action_catalog"]
        self._mcs_model = sources["mcs_model"]
        self.binding_sha256 = _sha({"schema": self.SCHEMA, "candidate": CANDIDATE_VERSION,
                                    "calibration": calibration.binding_sha256,
                                    "catalog": self.catalog.binding_sha256,
                                    "transport": self.model.document_sha256,
                                    "mcs_model": self._mcs_model.binding_sha256, "seed": seed})
        self._provider = _StateProvider(self)
        self._kernel = _Kernel(self)
        self._start()

    # ------------------------------------------------------------------ channel hooks (Run-5 overrides)
    def _channel_reset(self) -> None:
        self._mcs = MCSP.FitMcsMarkovProviderV1(self._mcs_model, seed=self._seed * 1_000_003 + 53)
        self._mcs_current = self._mcs.reset()

    def _channel_advance(self, context: _Context) -> None:
        self._mcs_current = self._mcs.step().successor_mcs

    def _channel_profile(self) -> Optional[str]:
        return None                         # the Run-4 MCS Markov channel carries no profile label

    def _features(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self._env.current_state.features.as_tuple())

    # ------------------------------------------------------------------ session
    def _start(self) -> None:
        base = self._seed * 1_000_003
        self._scene_rng = random.Random(base + 11)
        self._held_scene_rng = random.Random(base + 67)
        self.chain_rng = random.Random(base + 41)
        self._channel_reset()
        self._backlog_bytes = 0
        self.score_channel_busy_until_ns: Optional[int] = None
        self._session_index = 0
        self._records: list[dict[str, Any]] = []
        self._actions: list[tuple[int, int]] = []
        self.previous_context: Optional[_Context] = None
        self.context: Optional[_Context] = None
        self.context = self._new_context()
        self._env = self._build_env()
        self._open_session()

    def _session_uuid(self) -> str:
        return f"00000000-0000-4c00-8{self._session_index:03d}-{self._seed % 10**12:012d}"

    def _open_session(self) -> None:
        self._env.reset(session_uuid=self._session_uuid(), ue_id=UE_ID)

    def _build_env(self) -> environment.Run4SequentialEnvironmentV1:
        _require(self.mode == "FIXTURE", "calibrated environments are blocked in this candidate")
        return environment.Run4SequentialEnvironmentV1(
            state_provider=self._provider, kernel=self._kernel, gamma=0.99,
            evidence_class=environment.EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE)

    def _new_context(self) -> _Context:
        reward_key = self.catalog.keys[self._scene_rng.randrange(self.catalog.scene_count)]
        held_key = self.catalog.keys[self._held_scene_rng.randrange(self.catalog.scene_count)]
        si, p40 = self.catalog.scene_descriptors(reward_key)
        return _Context(reward_scene_key=reward_key, held_scene_key=held_key, camera_si=si, radar_p40=p40,
                        prior_ul_mcs=int(self._mcs_current), backlog_bytes=int(self._backlog_bytes),
                        profile=self._channel_profile())

    def advance_once(self) -> None:
        context = self.context
        _require(context is not None and context.resolved is not None, "cannot advance an unresolved cycle")
        if context.advanced:
            return
        self._backlog_bytes = int(round(self.model.predict_next_backlog_bytes(
            pre_enqueue_backlog_bytes=float(context.backlog_bytes),
            deterministic_action_ingress_bytes=context.resolved["ingress_bytes"],
            prior_ul_mcs=int(context.prior_ul_mcs))))
        self._channel_advance(context)
        context.advanced = True
        self.previous_context = context
        self.context = self._new_context()

    # ------------------------------------------------------------------ one decision
    def current_state_features(self) -> tuple[float, ...]:
        return self._features()

    def _action_identity(self, mode_id: int, q_e4: int) -> ExecutedActionIdentity:
        executable = self.action_catalog.resolve(mode_id, q_e4 / float(action_contract.Q_E4_SCALE))
        return ExecutedActionIdentity.from_executable_action(executable, self.action_catalog)

    def collect(self, mode_id: int, q_e4: int) -> dict[str, Any]:
        current = self._env.current_state.state.state
        features = self.current_state_features()
        context = self.context
        cycle = self._env.step(self._action_identity(mode_id, q_e4))
        resolved = dict(context.resolved)
        excluded = isinstance(cycle, environment.ExcludedCycleV1)
        record = {
            "ordinal": len(self._records), "session_uuid": current.identity.session_uuid,
            "decision_seq": current.identity.decision_seq, "mode_id": mode_id, "q_e4": q_e4,
            "reward_scene_key": context.reward_scene_key, "held_scene_key": context.held_scene_key,
            "camera_si": context.camera_si, "radar_p40": context.radar_p40, "prior_ul_mcs": context.prior_ul_mcs,
            "backlog_bytes": context.backlog_bytes, "profile": context.profile,
            "state_features": [float(v).hex() for v in features],
            "previous_present": current.previous is not None,
            "terminal": cycle.terminal.value, "learning_included": not excluded,
            "reward": None if excluded else float(cycle.reward),
            "reward_request_flags": None if excluded else list(cycle.reward_request_flags), **resolved}
        record["latency_ms"] = (resolved["resolution_elapsed_ns"] / 1e6
                                if record["terminal"] == contract.RewardTerminal.SUCCESS.value else None)
        self._records.append(record)
        self._actions.append((mode_id, q_e4))
        if excluded:                 # live: the session breaks; the next decision is a fresh-session genesis
            self._session_index += 1
            self._open_session()
        return record

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    # ------------------------------------------------------------------ durable state
    def checkpoint(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "binding_sha256": self.binding_sha256, "seed": self._seed,
                "actions": [list(a) for a in self._actions], "records_sha256": _sha(self._records),
                "backlog_bytes": self._backlog_bytes, "session_index": self._session_index,
                "score_channel_busy_until_ns": self.score_channel_busy_until_ns,
                "chain_rng_state_sha256": _sha(self.chain_rng.getstate())}

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Deterministic replay of the action ledger; any digest difference is a hard failure."""
        _require(checkpoint.get("schema") == self.SCHEMA, "checkpoint schema differs")
        _require(checkpoint.get("binding_sha256") == self.binding_sha256, "checkpoint binding differs")
        _require(checkpoint.get("seed") == self._seed, "checkpoint seed differs")
        self._start()
        for mode_id, q_e4 in checkpoint["actions"]:
            self.collect(int(mode_id), int(q_e4))
        _require(_sha(self._records) == checkpoint["records_sha256"], "restored records differ")
        _require(self._backlog_bytes == checkpoint["backlog_bytes"], "restored backlog differs")
        _require(self._session_index == checkpoint["session_index"], "restored session index differs")
        _require(self.score_channel_busy_until_ns == checkpoint["score_channel_busy_until_ns"],
                 "restored score-channel occupancy differs")
        _require(_sha(self.chain_rng.getstate()) == checkpoint["chain_rng_state_sha256"], "restored chain RNG differs")


class CorrectedRun5CollectorV1(CorrectedRun4CollectorV1):
    """22-D Run-5 variant: the joint SNR/MCS channel, the causal SNR lease and its profile label for coverage."""

    SCHEMA = "splitfusion.corrected_latency_env.run5_collector.v1"
    # Copied verbatim from the original Run-5 collector (abiodun_run5_wt run5_collector.py:55-57, a3c462f).
    LEASE_MAX_HEARTBEAT_AGE_NS = 200_000_000
    SNR_EFFECTIVE_LEAD_NS = 20_000_000
    SNR_HEARTBEAT_LEAD_NS = 5_000_000

    def __init__(self, *, calibration: CalibrationSetV1, seed: int, mode: str = "FIXTURE",
                 shared_sources: Optional[Mapping[str, Any]] = None) -> None:
        from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
        from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
        self._J, self._SNR = J, SNR
        sources = dict(shared_sources) if shared_sources is not None else load_shared_sources()
        if "snr_kernel" not in sources:
            sources["snr_kernel"] = J.load_accepted_kernel(ROOT)
            sources["design"] = J.load_design()
        self._lease = SNR.SnrLeasePolicyV1(
            policy_id="run5-modeled-controller-lease", policy_version=1,
            evidence_sha256=SNR.NETWORK_PROFILE_DESIGN_SHA256, max_heartbeat_age_ns=self.LEASE_MAX_HEARTBEAT_AGE_NS)
        self._feature_cache: dict[tuple[str, int], tuple[float, ...]] = {}
        super().__init__(calibration=calibration, seed=seed, mode=mode, shared_sources=sources)

    def _channel_reset(self) -> None:
        J = self._J
        self._channel = J.JointSnrMcsChannelV1(kernel=self._shared["snr_kernel"], design=self._shared["design"],
                                               seed=J.derive_seed(self._seed, "train-channel"))
        observation = self._channel.observe()
        self._mcs_current, self._snr_current = observation.mcs, observation.snr_db

    def _channel_advance(self, context: _Context) -> None:
        step = self._channel.advance(self._J.SUPPORTED_DURATION_TENSORS)
        _require(step.current_mcs == context.prior_ul_mcs, "channel advanced from an unobserved state")
        observation = self._channel.observe()
        self._mcs_current, self._snr_current = observation.mcs, observation.snr_db

    def _channel_profile(self) -> Optional[str]:
        return self._J.TRAINING_PROFILES[int(self._channel.checkpoint()["hidden_profile_index"])]

    def _new_context(self) -> _Context:
        context = super()._new_context()
        context.snr_db = float(self._snr_current)
        return context

    def _open_session(self) -> None:
        self._snr_adapter = self._SNR.ModeledLeaseSnrAdapterV1(
            provider_id=self.SCHEMA, session_uuid=self._session_uuid(), ue_id=UE_ID)
        super()._open_session()

    def _features(self) -> tuple[float, ...]:
        bundle = self._env.current_state
        identity = bundle.state.state.identity
        key = (identity.session_uuid, identity.decision_seq)
        if key not in self._feature_cache:
            boundary = bundle.state.boundary
            commit = boundary.state_commit_timestamp_ns
            command_id = f"run5-modeled-effective-command-{identity.session_uuid}-{identity.decision_seq}"
            self._snr_adapter.record_command_ack_at(at_ns=commit - self.SNR_EFFECTIVE_LEAD_NS, command_id=command_id,
                                                   status="ACK", clamped=False, target_snr_db=float(self.context.snr_db))
            self._snr_adapter.record_heartbeat_at(at_ns=commit - self.SNR_HEARTBEAT_LEAD_NS,
                                                 active_command_id=command_id)
            guarded = self._SNR.guard_run5_state_v2(bundle.state.state, self._snr_adapter.observe(boundary), boundary,
                                                    self._provider.freshness, self._lease)
            values = self._SNR.build_run5_features_v2(guarded, self._provider.scaling).as_tuple()
            run4 = bundle.features.as_tuple()
            _require([v.hex() for v in values[:21]] == [float(v).hex() for v in run4],
                     "Run-5 positions 0-20 differ from the Run-4 vector")
            self._feature_cache[key] = tuple(float(v) for v in values)
        return self._feature_cache[key]

    def _start(self) -> None:
        self._feature_cache = {}
        super()._start()

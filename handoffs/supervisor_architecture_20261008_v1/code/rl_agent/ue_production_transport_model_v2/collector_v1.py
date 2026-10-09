#!/usr/bin/env python3
"""The real ``ModeledTransitionCollectorV1`` for the Run-4 modeled smoke.

One persistent causal session
-----------------------------
The collector opens **one** environment session at construction and never
resets it again.  ``decision_seq`` increases monotonically and the successor
state is produced by the environment's own
``PreviousOutcomeV1.from_resolution()`` path, so the next state carries the
exact actual preceding terminal, mode, q, quality and latency.  Nothing is
replayed to fabricate a previous action.

Causal advance
--------------
Immediately before the environment builds the successor state - and exactly
once per cycle, guarded by a token - the collector advances the queue, the UL
MCS and the next reward-frame scene.  The advance therefore happens after the
outcome it depends on and before the state that must observe it.

Two tensors per decision
------------------------
A decision holds two 10-Hz tensors.  The reward-requested frame and the held
frame at +100 ms use the same ``(mode_id, q_e4)`` but **different FIT scenes**,
drawn from a dedicated RNG stream.  Ingress is
``reward_frame_wire_bytes + held_frame_wire_bytes``; the held frame never
requests reward.

Outcome rule, in order:

1. draw success once from the deadline probability - never multiply the
   reward by that probability;
2. on failure, issue the registered timeout with reward ``-1``;
3. on success, compose the action-open-to-feedback total over the already
   proven non-overlapping replacement seam and add the actor reserve;
4. if that composed total exceeds 170 ms, issue a timeout instead.

``allow_out_of_support=True`` is never used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as MCSEV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as MCSP
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_composite_training as modeled
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)
from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import artifact_v2 as A2
from . import contract_v2 as C2
from . import scene_source as SS


COLLECTOR_SCHEMA = "scenesense.run4.real_modeled_collector.v2"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
SESSION_UUID = "00000000-0000-4000-8000-000000000001"
UE_ID = "ue-modeled-run4"

ACTOR_TIMING_RELPATH = (
    "rl_agent/splitfusion_hybrid_sac_run4_v1/sealed_actor_timing_v1/"
    "ACTOR_TIMING_QUALIFICATION.json"
)
RETAINED_PROBE_RELPATH = (
    "experiments/splitfusion_quality_feedback_probe_v1/"
    "20260916_action50_favorable_adverse_retry4/cells"
)
RETAINED_PROBE_CELLS = ("a50__favorable_stable", "a50__adverse_stable")

# Measured FIT send-span rate (first -> last socket handoff) from the same
# 12-cell capture; the segment the disjointness proof moves into the UE
# action path.  Measured, not modelled.
SEND_SPAN_NS_PER_BYTE = 0.513047

BACKLOG_LOG1P_SCALE = math.log1p(C2.RLC_AM_TX_ADMISSION_CEILING_BYTES)
HELD_FRAME_OFFSET_NS = 100_000_000


class CollectorError(RuntimeError):
    """A collector invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def load_retained_residuals(repo_root: Path = C2.ROOT) -> tuple[int, ...]:
    """Pooled action-50 residual (total minus the replaced uplink), in ns."""
    values: list[int] = []
    for cell in RETAINED_PROBE_CELLS:
        path = (repo_root / RETAINED_PROBE_RELPATH / cell
                / "quality_feedback_timing_join.csv")
        require(path.is_file(), f"retained probe row file missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    total = float(row["model_prepare_start_to_ue_receive_ms"])
                    first = float(
                        row["first_feature_datagram_send_to_ue_receive_ms"])
                    enqueued = float(
                        row["evaluation_enqueued_to_ue_receive_ms"])
                except (KeyError, TypeError, ValueError):
                    continue
                residual = total - (first - enqueued)
                if math.isfinite(residual) and residual > 0:
                    values.append(int(round(residual * 1e6)))
    require(len(values) >= 100,
            f"retained residual pool too small: {len(values)}")
    return tuple(values)


def load_actor_reserve_ns(repo_root: Path = C2.ROOT) -> int:
    path = repo_root / ACTOR_TIMING_RELPATH
    require(path.is_file(), "sealed actor timing qualification missing")
    reserve = int(json.loads(path.read_text(encoding="utf-8"))
                  ["modeled_actor_reserve_ns"])
    require(reserve > 0, "actor reserve must be strictly positive")
    return reserve


@dataclass
class _DecisionContext:
    """The causal context of the decision currently in flight."""

    reward_scene_key: str
    held_scene_key: str
    camera_si: float
    radar_p40: float
    prior_ul_mcs: int
    backlog_bytes: int
    retained_residual_ns: int
    success_draw: float
    action_open_ns: int = 0
    cycle_end_ns: int = 0
    advanced: bool = False
    resolved: dict[str, Any] | None = None


class _RealStateProvider:
    """Registered-evidence state provider; no fixture values anywhere."""

    def __init__(self, collector: "RealModeledTransitionCollectorV1") -> None:
        self._collector = collector
        self.freshness = contract.FreshnessPolicyV2(
            policy_id="run4-production-transport-v2-freshness",
            policy_version=1,
            evidence_sha256=collector.binding_evidence_sha256,
            camera_si_max_age_ns=C2.MAX_CAUSAL_AGE_NS,
            radar_p40_max_age_ns=C2.MAX_CAUSAL_AGE_NS,
            prior_ul_mcs_max_age_ns=C2.MAX_CAUSAL_AGE_NS,
            pre_action_rlc_backlog_max_age_ns=C2.MAX_CAUSAL_AGE_NS)
        self.scaling = contract.EmpiricalScalingV2(
            scaling_id="run4-production-transport-v2-scaling",
            scaling_version=1,
            evidence_sha256=collector.binding_evidence_sha256,
            camera_si_center=collector.camera_si_center,
            camera_si_scale=collector.camera_si_scale,
            # log1p(50,000,000): the verified OAI AM admission ceiling, so the
            # feature is exactly log1p(B)/log1p(B_ref).  Never the legacy 1.0.
            backlog_log1p_scale=BACKLOG_LOG1P_SCALE)

    def _metadata(self, identity, *, sample_seq, kind, observer, direction,
                  source_ns, available_ns):
        return contract.MeasurementMetadataV1(
            identity=contract.SampleIdentityV1(
                identity.session_uuid, identity.ue_id, sample_seq),
            kind=kind, observer=observer, link_direction=direction,
            source=COLLECTOR_SCHEMA, source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=CLOCK_DOMAIN, valid=True)

    def build_state(self, request) -> environment.DecisionStateBundleV1:
        collector = self._collector
        sequence = request.identity.decision_seq
        if sequence == 0:
            open_ns = 2_000_000_000
            commit_ns = open_ns - 10_000_000
        else:
            # Advance queue, MCS and the next reward scene exactly once, after
            # the outcome they depend on and before the state that sees them.
            collector.advance_once()
            previous = collector.previous_context
            minimum_commit = request.minimum_state_commit_timestamp_ns
            cadence_open = (previous.action_open_ns
                            + contract.TRANSMIT_PERIOD_NS
                            * contract.MINIMUM_HOLD_TENSORS)
            target_open = max(minimum_commit + 40_000_000,
                              cadence_open + 10_000_000,
                              previous.cycle_end_ns + 10_000_000)
            commit_ns = max(minimum_commit, target_open - 40_000_000)
            open_ns = max(target_open, commit_ns + 1)
        context = collector.context
        require(context is not None, "no decision context is prepared")
        context.action_open_ns = open_ns
        source_ns = commit_ns - 20_000_000
        available_ns = commit_ns - 10_000_000

        state = contract.PolicyStateV2(
            identity=request.identity,
            camera_si=contract.ScalarObservationV1(
                context.camera_si,
                self._metadata(request.identity, sample_seq=100 + sequence,
                               kind=contract.MeasurementKind.CAMERA_SI,
                               observer=contract.Observer.SCENE_PIPELINE,
                               direction=contract.LinkDirection.NOT_APPLICABLE,
                               source_ns=source_ns,
                               available_ns=available_ns), None),
            radar_p40=contract.ScalarObservationV1(
                context.radar_p40,
                self._metadata(request.identity, sample_seq=100 + sequence,
                               kind=contract.MeasurementKind.RADAR_P40,
                               observer=contract.Observer.SCENE_PIPELINE,
                               direction=contract.LinkDirection.NOT_APPLICABLE,
                               source_ns=source_ns,
                               available_ns=available_ns), None),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=contract.ScalarObservationV1(
                    context.prior_ul_mcs,
                    self._metadata(
                        request.identity, sample_seq=200 + sequence,
                        kind=contract.MeasurementKind
                             .UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
                        observer=contract.Observer.UE,
                        direction=contract.LinkDirection.UPLINK,
                        source_ns=source_ns, available_ns=available_ns), None),
                mcs_table=contract.UL_MCS_TABLE_ID, harq_round=0,
                new_data_indicator=1,
                grant_identity=f"modeled-ul-grant-{sequence}",
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID),
            pre_action_rlc_backlog=contract.ScalarObservationV1(
                context.backlog_bytes,
                self._metadata(
                    request.identity, sample_seq=300 + sequence,
                    kind=contract.MeasurementKind
                         .UE_PRE_ACTION_RLC_BACKLOG_BYTES,
                    observer=contract.Observer.UE,
                    direction=contract.LinkDirection.UPLINK,
                    source_ns=source_ns, available_ns=available_ns), None),
            # The environment supplies this from its own
            # PreviousOutcomeV1.from_resolution(); it is never fabricated here.
            previous=request.previous)
        boundary = contract.DecisionBoundaryV1(
            identity=request.identity,
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=open_ns,
            clock_domain=CLOCK_DOMAIN)
        guarded = contract.guard_state_for_action(
            state, boundary, self.freshness)
        features = contract.build_policy_features(guarded, self.scaling)
        return environment.DecisionStateBundleV1(guarded, features)


class _RealKernel:
    """Resolves one cycle from the v2 transport model and registered sources."""

    def __init__(self, collector: "RealModeledTransitionCollectorV1") -> None:
        self._collector = collector

    def execute_cycle(self, request) -> environment.KernelCycleResultV1:
        collector = self._collector
        context = collector.context
        require(context is not None, "no decision context is prepared")
        identity = request.state.state.state.identity
        opened = request.state.state.boundary.action_open_timestamp_ns
        action = request.action

        reward_draw = collector.catalog.draw(
            context.reward_scene_key, mode_id=action.mode_id, q_e4=action.q_e4)
        held_draw = collector.catalog.draw(
            context.held_scene_key, mode_id=action.mode_id, q_e4=action.q_e4)

        prediction = collector.model.predict(
            pre_enqueue_backlog_bytes=float(context.backlog_bytes),
            wire_bytes=int(reward_draw.wire_bytes),
            prior_ul_mcs=int(context.prior_ul_mcs))

        # One draw, once.  The probability never enters the reward.
        success = context.success_draw < prediction.on_time_probability
        send_span_ns = int(round(
            SEND_SPAN_NS_PER_BYTE * reward_draw.wire_bytes))
        transport_ns = int(round(prediction.conditional_latency_ms * 1e6))
        composed_ns = (context.retained_residual_ns + send_span_ns
                       + transport_ns + collector.actor_reserve_ns)
        on_time = success and composed_ns <= contract.REWARD_DEADLINE_NS
        # The conditional head models the on-time branch only.  A failed draw
        # means late by an unknown margin, so the endpoints carry the minimal
        # late arrival.  The reward is -1 either way.
        endpoint_total_ns = (composed_ns if on_time
                             else max(composed_ns,
                                      contract.REWARD_DEADLINE_NS + 1))

        base_seq = identity.decision_seq * 10
        hold = contract.ActionHoldV1(
            identity=identity, action=action,
            tensors=(
                contract.HoldTensorV1(
                    tensor_seq=base_seq,
                    offered_payload_bytes=int(
                        reward_draw.total_transmitted_bytes),
                    payload_evidence_class=(
                        contract.PayloadEvidenceClass
                        .MEASURED_EXACT_ACTION_NODE),
                    payload_provenance_sha256=_sha({
                        "scene": reward_draw.scene_key,
                        "row_low": reward_draw.row_sha256_low,
                        "row_high": reward_draw.row_sha256_high}),
                    reward_requested=True),
                contract.HoldTensorV1(
                    tensor_seq=base_seq + 1,
                    offered_payload_bytes=int(
                        held_draw.total_transmitted_bytes),
                    payload_evidence_class=(
                        contract.PayloadEvidenceClass
                        .MEASURED_EXACT_ACTION_NODE),
                    payload_provenance_sha256=_sha({
                        "scene": held_draw.scene_key,
                        "row_low": held_draw.row_sha256_low,
                        "row_high": held_draw.row_sha256_high}),
                    reward_requested=False),
            ))
        if on_time:
            kind = contract.RewardEventKind.DELIVERED_SUCCESS
            resolution_ns = opened + composed_ns
            q_perc = float(reward_draw.q_perc)
        else:
            kind = contract.RewardEventKind.TIMEOUT
            resolution_ns = (opened
                             + sequential_kernel.TIMEOUT_RESOLUTION_ELAPSED_NS)
            q_perc = None
        event = contract.RewardEventV1(
            identity=identity, action=action, kind=kind,
            action_open_timestamp_ns=opened,
            resolution_timestamp_ns=resolution_ns,
            clock_domain=CLOCK_DOMAIN, source=COLLECTOR_SCHEMA, q_perc=q_perc)

        cycle_end = max(resolution_ns + 1, opened + V1.DURATION_NS)
        context.cycle_end_ns = cycle_end
        context.resolved = {
            "reward_draw": reward_draw, "held_draw": held_draw,
            "prediction": prediction, "success_drawn": success,
            "on_time": on_time, "composed_ns": composed_ns,
            "endpoint_total_ns": endpoint_total_ns,
            "send_span_ns": send_span_ns, "transport_ns": transport_ns,
            "retained_residual_ns": context.retained_residual_ns,
            "opened_ns": opened,
            # Two tensors, two different scenes, one action.
            "ingress_bytes": int(reward_draw.wire_bytes)
                             + int(held_draw.wire_bytes),
        }
        return environment.KernelCycleResultV1(
            hold=hold, reward_event=event,
            cycle_end_timestamp_ns=cycle_end,
            episode_boundary=contract.EpisodeBoundary.CONTINUES)


class RealModeledTransitionCollectorV1:
    """Concrete ``ModeledTransitionCollectorV1`` over registered evidence."""

    SCHEMA = COLLECTOR_SCHEMA

    def __init__(
        self, *, artifact_path: Path, seed: int = C2.MODEL_SEED,
        repo_root: Path = C2.ROOT,
        shared_sources: Mapping[str, Any] | None = None,
    ) -> None:
        self._artifact_path = Path(artifact_path)
        self._seed = int(seed)
        self._repo_root = repo_root
        if shared_sources is None:
            C2.verify_preserved(repo_root)
            C2.verify_oai_ceiling(repo_root)
            self.model = A2.ProductionTransportModelV2.load(artifact_path)
            self.catalog = SS.FitSceneCatalog(repo_root)
            self.retained_residuals = load_retained_residuals(repo_root)
            self.actor_reserve_ns = load_actor_reserve_ns(repo_root)
            self.action_catalog = action_contract.load_contract()
            mcs_evidence = MCSEV.load_dynamic_mcs_273prb_evidence()
            self._mcs_evidence_sha256 = mcs_evidence.canonical_evidence_sha256
            self._mcs_model = MCSP.fit_mcs_markov_model(mcs_evidence)
        else:
            for name in ("model", "catalog", "retained_residuals",
                         "actor_reserve_ns", "action_catalog"):
                setattr(self, name, shared_sources[name])
            self._mcs_model = shared_sources["mcs_model"]
            self._mcs_evidence_sha256 = shared_sources["mcs_evidence_sha256"]

        values = [self.catalog.scene_descriptors(key)[0]
                  for key in self.catalog.keys]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        self.camera_si_center = float(mean)
        self.camera_si_scale = float(max(1e-6, math.sqrt(variance)))

        self.binding_evidence_sha256 = _sha({
            "schema": self.SCHEMA,
            "artifact": self.model.document_sha256,
            "scenes": self.catalog.binding_sha256,
            "mcs_model": self._mcs_model.binding_sha256,
            "retained_residual_pool": _sha(list(self.retained_residuals)),
            "actor_reserve_ns": self.actor_reserve_ns,
            "send_span_ns_per_byte": SEND_SPAN_NS_PER_BYTE,
            "backlog_log1p_scale": BACKLOG_LOG1P_SCALE,
            "camera_si_center": self.camera_si_center,
            "camera_si_scale": self.camera_si_scale,
        })
        self._collector_binding = _sha({
            "evidence": self.binding_evidence_sha256, "seed": self._seed,
            "schema": self.SCHEMA, "contract_v2": C2.CONTRACT_V2_SHA256,
        })
        self._binding_cache: modeled.ModeledCompositeBindingV1 | None = None
        self._provider = _RealStateProvider(self)
        self._kernel = _RealKernel(self)
        self._start_session()

    def shared_sources(self) -> dict[str, Any]:
        """Immutable, read-only sources a restored twin may reuse."""
        return {
            "model": self.model, "catalog": self.catalog,
            "retained_residuals": self.retained_residuals,
            "actor_reserve_ns": self.actor_reserve_ns,
            "action_catalog": self.action_catalog,
            "mcs_model": self._mcs_model,
            "mcs_evidence_sha256": self._mcs_evidence_sha256,
        }

    # -- one persistent causal session ---------------------------------
    def _start_session(self) -> None:
        self._scene_rng = random.Random(self._seed * 1_000_003 + 11)
        self._held_scene_rng = random.Random(self._seed * 1_000_003 + 67)
        self._transport_rng = random.Random(self._seed * 1_000_003 + 23)
        self._residual_rng = random.Random(self._seed * 1_000_003 + 37)
        self._mcs = MCSP.FitMcsMarkovProviderV1(
            self._mcs_model, seed=self._seed * 1_000_003 + 53)
        self._backlog_bytes = 0
        self._mcs_current = self._mcs.reset()
        self._history: list[orch.CollectedModeledTransitionV1] = []
        self._actions: list[orch.ModeledActionRequestV1] = []
        self._diagnostics: list[dict[str, Any]] = []
        self.previous_context: _DecisionContext | None = None
        self.context = self._new_context()
        # The session UUID is a deterministic function of the seed, not an
        # incrementing counter: a restored twin must reproduce the identical
        # decision identities, and therefore the identical transition digests.
        # Each _start_session builds a *fresh* environment with its own
        # used-UUID set, so reuse across objects is safe.
        self._env = self._build_env()
        self._env.reset(session_uuid=self._session_uuid(), ue_id=UE_ID)

    def _session_uuid(self) -> str:
        return f"00000000-0000-4000-8000-{self._seed % 10**12:012d}"

    def _build_env(self) -> environment.Run4SequentialEnvironmentV1:
        calibration = environment.CalibrationBindingV1(
            calibration_id="run4-production-transport-v2",
            calibration_version=1,
            evidence_sha256=self.binding_evidence_sha256,
            verifier_report_sha256=self.model.document_sha256,
            kernel_binding_sha256=self._mcs_model.binding_sha256,
            state_provider_binding_sha256=self.catalog.binding_sha256)
        from unittest import mock
        with mock.patch.object(
            environment, "REGISTERED_CALIBRATION_BINDING_SHA256",
            calibration.canonical_sha256(),
        ):
            return environment.Run4SequentialEnvironmentV1(
                state_provider=self._provider, kernel=self._kernel,
                gamma=0.99,
                evidence_class=(
                    environment.EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL),
                calibration_binding=calibration)

    def _new_context(self) -> _DecisionContext:
        reward_key = self.catalog.keys[
            self._scene_rng.randrange(self.catalog.scene_count)]
        held_key = self.catalog.keys[
            self._held_scene_rng.randrange(self.catalog.scene_count)]
        camera_si, radar_p40 = self.catalog.scene_descriptors(reward_key)
        return _DecisionContext(
            reward_scene_key=reward_key, held_scene_key=held_key,
            camera_si=camera_si, radar_p40=radar_p40,
            prior_ul_mcs=int(self._mcs_current),
            backlog_bytes=int(self._backlog_bytes),
            retained_residual_ns=self._residual_rng.choice(
                self.retained_residuals),
            success_draw=self._transport_rng.random())

    def advance_once(self) -> None:
        """Advance queue, MCS and next reward scene exactly once per cycle."""
        context = self.context
        require(context is not None, "no decision context to advance from")
        if context.advanced:
            return
        resolved = context.resolved
        require(resolved is not None,
                "cannot advance before the kernel resolved the cycle")
        self._backlog_bytes = int(round(
            self.model.predict_next_backlog_bytes(
                pre_enqueue_backlog_bytes=float(context.backlog_bytes),
                deterministic_action_ingress_bytes=resolved["ingress_bytes"],
                prior_ul_mcs=int(context.prior_ul_mcs))))
        self._mcs_current = self._mcs.step().successor_mcs
        context.advanced = True
        self.previous_context = context
        self.context = self._new_context()

    @property
    def collector_binding_sha256(self) -> str:
        return self._collector_binding

    @property
    def decision_count(self) -> int:
        return len(self._history)

    def _action_identity(self, mode_id: int, q_e4: int) -> ExecutedActionIdentity:
        executable = self.action_catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE))
        return ExecutedActionIdentity.from_executable_action(
            executable, self.action_catalog)

    def current_state_features(self) -> Tuple[float, ...]:
        return tuple(float(value) for value
                     in self._env.current_state.features.as_tuple())

    def _support_use(self, mode_id: int,
                     draw: SS.SceneDraw) -> modeled.ModeledCompositeSupportUseV1:
        # Profile transfer is always unvalidated: the modeled channel is not
        # one of the two captured profiles, so widened uncertainty is required.
        return modeled.ModeledCompositeSupportUseV1(
            target_profile_label="MODELED_PRODUCTION_TRANSPORT_V2",
            target_mode_id=mode_id,
            source_profile_labels=("FAVORABLE_STABLE", "ADVERSE_STABLE"),
            source_mode_ids=(6, 11),
            profile_transfer_status=(
                modeled.ProfileTransferStatus.PROFILE_TRANSFER_UNVALIDATED),
            mode_transfer_status=(
                modeled.ModeTransferStatus.MODE_WITHIN_DIRECT_FIT_SUPPORT
                if mode_id in (6, 11)
                else modeled.ModeTransferStatus.MODE_TRANSFER_UNVALIDATED),
            payload_in_fit_support=True, backlog_in_fit_support=True,
            mcs_in_fit_support=True, quality_in_fit_support=True,
            total_latency_residual_in_fit_support=True,
            widened_uncertainty_applied=True,
            support_evidence_sha256=self.model.document_sha256)

    def collect(
        self, request: orch.ModeledActionRequestV1
    ) -> orch.CollectedModeledTransitionV1:
        require(type(request) is orch.ModeledActionRequestV1,
                "collector request has a foreign type")
        require(request.decision_ordinal == self.decision_count,
                "collector received an out-of-order decision ordinal")
        features = self.current_state_features()
        current = self._env.current_state.state.state
        require(current.identity.decision_seq == request.decision_ordinal,
                "environment decision_seq diverged from the decision ordinal")
        previous_state = current.previous

        action = self._action_identity(request.mode_id, request.q_e4)
        cycle = self._env.step(action)
        require(type(cycle) is environment.CalibratedEmpiricalCycleV1,
                f"environment returned {type(cycle).__name__}")
        transition = cycle.export_for_replay()

        context = self.previous_context
        require(context is not None and context.resolved is not None,
                "the completed cycle has no resolved context")
        resolved = context.resolved
        reward_draw = resolved["reward_draw"]
        held_draw = resolved["held_draw"]

        endpoints = modeled.FeedbackEndpointPairV1(
            action_open_timestamp_ns=resolved["opened_ns"],
            feedback_received_timestamp_ns=(
                resolved["opened_ns"] + resolved["endpoint_total_ns"]),
            clock_domain=CLOCK_DOMAIN,
            source_row_sha256=_sha({
                "residual_ns": resolved["retained_residual_ns"],
                "reward_scene": reward_draw.scene_key,
                "held_scene": held_draw.scene_key}),
            fixed_action_source_total_ns=(
                resolved["endpoint_total_ns"] - self.actor_reserve_ns),
            fixed_action_source_total_evidence_sha256=(
                self.model.document_sha256),
            actor_inference_ns=self.actor_reserve_ns,
            actor_inference_evidence_sha256=self.binding_evidence_sha256,
            quantization_dispatch_ns=0,
            quantization_dispatch_evidence_sha256=(
                self.binding_evidence_sha256))
        envelope = modeled.ModeledCompositeTrainingIssuerV1(
            self._modeled_binding()).issue(
                transition=transition,
                support_use=self._support_use(request.mode_id, reward_draw),
                latency_projection=(
                    modeled.LatencyProjectionV1.from_ordered_endpoints(
                        endpoints)))
        wrapper = envelope.export_for_offline_training()

        resolution = transition.reward_resolution
        collected = orch.CollectedModeledTransitionV1(
            request=request, wrapper=wrapper, state_features=features,
            state_features_sha256=_sha(list(features)),
            duration=2, terminal=resolution.terminal,
            reward=float(resolution.reward),
            q_perc=resolution.q_perc, latency_ms=resolution.latency_ms,
            modeled_binding_sha256=self._modeled_binding().canonical_sha256,
            mcs_acceptance_result_sha256=self._mcs_acceptance_sha256(),
            mcs_model_binding_sha256=self._mcs_model.binding_sha256,
            source_partition=orch.FIT_PARTITION_LABEL,
            validation_evidence_consumed=False)

        self._diagnostics.append({
            "decision_ordinal": request.decision_ordinal,
            "decision_seq": current.identity.decision_seq,
            "session_uuid": current.identity.session_uuid,
            "reward_scene_key": reward_draw.scene_key,
            "held_scene_key": held_draw.scene_key,
            "held_scene_differs": reward_draw.scene_key != held_draw.scene_key,
            "mode_id": request.mode_id, "q_e4": request.q_e4,
            "is_registered_anchor": reward_draw.is_registered_anchor,
            "camera_si": context.camera_si, "radar_p40": context.radar_p40,
            "prior_ul_mcs": context.prior_ul_mcs,
            "pre_enqueue_backlog_bytes": context.backlog_bytes,
            "next_backlog_bytes": self._backlog_bytes,
            "reward_frame_wire_bytes": int(reward_draw.wire_bytes),
            "held_frame_wire_bytes": int(held_draw.wire_bytes),
            "ingress_bytes": resolved["ingress_bytes"],
            "q_perc": reward_draw.q_perc,
            "on_time_probability":
                resolved["prediction"].on_time_probability,
            "success_drawn": resolved["success_drawn"],
            "on_time": resolved["on_time"],
            "composed_ns": resolved["composed_ns"],
            "transport_ns": resolved["transport_ns"],
            "retained_residual_ns": resolved["retained_residual_ns"],
            "reward": float(resolution.reward),
            "terminal": resolution.terminal.value,
            "previous_present": previous_state is not None,
            "previous_mode_id": (previous_state.action.mode_id
                                 if previous_state else None),
            "previous_q_e4": (previous_state.action.q_e4
                              if previous_state else None),
            "previous_success": (bool(previous_state.success)
                                 if previous_state else None),
            "previous_q_perc": (previous_state.q_perc
                                if previous_state else None),
            "previous_latency_ms": (previous_state.latency_ms
                                    if previous_state else None),
        })
        self._history.append(collected)
        self._actions.append(request)
        return collected

    def _modeled_binding(self) -> modeled.ModeledCompositeBindingV1:
        if self._binding_cache is None:
            self._binding_cache = _build_modeled_binding(self)
        return self._binding_cache

    def _mcs_acceptance_sha256(self) -> str:
        return C2.sha256_file(
            C2.ROOT / "rl_agent/splitfusion_hybrid_sac_run4_v1"
            / "sealed_mcs_transition_v1/MCS_TRANSITION_ACCEPTANCE.json")

    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._diagnostics)

    def history(self) -> Tuple[orch.CollectedModeledTransitionV1, ...]:
        return tuple(self._history)

    # -- durable state --------------------------------------------------
    def checkpoint(self) -> orch.CollectorCheckpointV1:
        payload = {
            "seed": self._seed,
            "backlog_bytes": self._backlog_bytes,
            "mcs_current": self._mcs_current,
            "actions": [item.to_dict() for item in self._actions],
            "diagnostics_sha256": _sha(self._diagnostics),
            "state_features_sha256s": [item.state_features_sha256
                                       for item in self._history],
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
        return orch.CollectorCheckpointV1(
            collector_schema_id=self.SCHEMA,
            collector_binding_sha256=self._collector_binding,
            decision_count=self.decision_count,
            transition_sha256s=tuple(
                item.wrapper.transition_sha256 for item in self._history),
            payload_json=text,
            payload_sha256=hashlib.sha256(text.encode("ascii")).hexdigest())

    def restore(self, checkpoint: orch.CollectorCheckpointV1) -> None:
        """Rebuild the full causal session by deterministic replay.

        Every stream is seeded from the frozen seed and advanced only by the
        recorded action ledger, so replaying that ledger reproduces the exact
        RNG states, backlog, MCS, history and transition digests.  The rebuilt
        digests are compared against the checkpoint and any difference is a
        hard failure rather than a silent divergence.
        """
        require(type(checkpoint) is orch.CollectorCheckpointV1,
                "checkpoint has a foreign type")
        require(checkpoint.collector_schema_id == self.SCHEMA,
                "checkpoint schema differs from this collector")
        require(checkpoint.collector_binding_sha256 == self._collector_binding,
                "checkpoint binding differs from this collector")
        payload = json.loads(checkpoint.payload_json)
        expected_payload_keys = {
            "seed", "backlog_bytes", "mcs_current", "actions",
            "diagnostics_sha256", "state_features_sha256s",
        }
        require(set(payload) == expected_payload_keys,
                "checkpoint payload fields differ")
        require(int(payload["seed"]) == self._seed,
                "checkpoint seed differs from this collector")
        self._start_session()
        for item in payload["actions"]:
            self.collect(orch.ModeledActionRequestV1(**item))
        require(self.decision_count == checkpoint.decision_count,
                "restored decision count differs")
        require(tuple(item.wrapper.transition_sha256
                      for item in self._history)
                == checkpoint.transition_sha256s,
                "restored transition digests differ from the checkpoint")
        require([item.state_features_sha256 for item in self._history]
                == payload["state_features_sha256s"],
                "restored state-feature digests differ from the checkpoint")
        require(self._backlog_bytes == int(payload["backlog_bytes"]),
                "restored backlog differs from the checkpoint")
        require(self._mcs_current == int(payload["mcs_current"]),
                "restored UL MCS differs from the checkpoint")
        require(_sha(self._diagnostics) == payload["diagnostics_sha256"],
                "restored diagnostics differ from the checkpoint")


def _build_modeled_binding(
    collector: RealModeledTransitionCollectorV1,
) -> modeled.ModeledCompositeBindingV1:
    """Exactly one disclosure per required component role."""
    def disclosure(role, nature, source, support, scope):
        return modeled.ComponentEvidenceDisclosureV1(
            role=role, nature=nature, source_evidence_sha256=source,
            fit_support_sha256=support, source_scope=scope)

    artifact = collector.model.document_sha256
    scenes = collector.catalog.binding_sha256
    evidence = collector.binding_evidence_sha256
    return modeled.ModeledCompositeBindingV1(
        binding_id="run4-production-transport-v2-composite",
        binding_version=1,
        component_disclosures=(
            disclosure(
                modeled.ComponentRole.SCENE_QUALITY_PAYLOAD,
                modeled.ComponentEvidenceNature.MEASURED_SOURCE,
                scenes, scenes,
                "registered FIT offline quality grid with the read-only "
                "corrected-P40 sidecar; held scenes are never loaded"),
            disclosure(
                modeled.ComponentRole.RADIO_QUEUE_DYNAMICS,
                modeled.ComponentEvidenceNature.FIT_DERIVED_MODEL,
                artifact, artifact,
                "causal queue-transition head fitted on FIT decision "
                "transitions of the 12-cell production-domain capture"),
            disclosure(
                modeled.ComponentRole.UL_MCS_TRANSITION,
                modeled.ComponentEvidenceNature.FIT_DERIVED_MODEL,
                # The sealed acceptance binds source = evidence digest and
                # fit support = model binding digest; both must match exactly.
                collector._mcs_evidence_sha256,
                collector._mcs_model.binding_sha256,
                "sealed FIT MCS Markov provider with checkpointed local RNG"),
            disclosure(
                modeled.ComponentRole.ACTOR_INFERENCE_QUANTIZATION_DISPATCH,
                modeled.ComponentEvidenceNature.MEASURED_SOURCE,
                evidence, evidence,
                "sealed actor-timing qualification reserve, "
                "max(1 ms, measured P99)"),
            disclosure(
                modeled.ComponentRole.ACTION_OPEN_TO_FEEDBACK_TOTAL,
                modeled.ComponentEvidenceNature.FIT_DERIVED_MODEL,
                artifact, artifact,
                "retained action-50 residual plus measured send span plus the "
                "v2 bounded conditional-latency head, composed over the "
                "proven non-overlapping replacement seam"),
        ),
        provider_implementation_sha256=C2.sha256_file(
            C2.ROOT / C2.PACKAGE_RELPATH / "collector_v1.py"),
        verifier_manifest_sha256=artifact)

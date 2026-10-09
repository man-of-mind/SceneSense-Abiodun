"""Full-support Run-5 collector: the corrected Run-5 collector with the full-support latency kernel and the agreed
reward. Emits original ``Run5CollectedTransitionV1`` records.

Unchanged from ``CorrectedRun5CollectorV1`` (corrected latency candidate) and hence from the original Run-5:
* the 22-D state: positions 0-20 = the Run-4 vector (camera SI, radar P40, prior UL MCS, RLC backlog, previous
  mode one-hot, previous q, previous Q_perc, previous latency, previous present, previous success), position 21 =
  SNR through the causal controller lease; training scaling unchanged;
* the original joint SNR/MCS channel: all four registered profiles in balanced 300-tick blocks, accepted
  SNR-tilted MCS kernel, two ticks per decision, nothing future-dependent observable;
* the RLC backlog transition (T_v2b queue head on the deterministic two-tensor ingress);
* live-parity outcomes: timely defined -> SUCCESS; timely undefined -> excluded EVALUATOR_FAULT and fresh-session
  genesis; late or missing -> TIMEOUT at 170 ms + 1 ns; holds (two tensors, one reward request); inclusive
  170-ms deadline; cadence rule.

Changed:
* latency: ``latency.FullSupportLatencyModelV1`` (every stage once, explicit GT queue, uplink from the channel's
  MCS and backlog through T_v2b);
* reward: the agreed definition (``quality.REWARD_DEFINITION``): value = Y * (0.7 + 0.3 Q_seg) on a timely
  defined reply, 0 on TIMEOUT; previous-Q state feature stays Q_perc;
* session identity: the inherited UUID formatter is invalid from session index 1000; indices 0-999 keep the
  inherited string, later ones a disjoint valid form (the pilot's fix, unchanged).
Fallback frames are not modelled (the original Run-5 environment had none; the channel kernel supports exactly two
ticks per decision).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_corrected_latency_env_v1.outcome import classify, receipt_elapsed_ns
from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as RC
from rl_agent.ue_production_queue_capture_v1 import contract as V1

from . import quality as Q
from . import support as S

GAMMA = 0.99
EVIDENCE_LABEL = "FULL_RUN5_MODELED_TRAINING"


class FullCollectorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullCollectorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_shared_sources() -> dict[str, Any]:
    sources = CR.load_shared_sources()
    catalog = Q.FullSceneCatalogV1()
    _require(catalog.keys == sources["catalog"].keys, "full catalogue scenes differ from the corrected FIT catalogue")
    sources["catalog"] = catalog
    return sources


class _Adapter:
    def __init__(self, latency) -> None:
        self.latency = latency
        self.binding_sha256 = _sha({"latency": latency.binding_sha256, "support": S.SUPPORT_SHA256})


class _FullKernel:
    def __init__(self, collector: "FullRun5CollectorV1") -> None:
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
        draw = c.latency.draw(mode_id=action.mode_id, q_e4=action.q_e4, camera_si=context.camera_si,
                              radar_p40=context.radar_p40, prior_ul_mcs=context.prior_ul_mcs,
                              backlog_bytes=context.backlog_bytes, wire_bytes=wire_reward, rng=c.chain_rng)
        busy = c.score_channel_busy_until_ns
        carry_in = None if busy is None else (busy - opened) / 1e6       # explicit cross-decision queueing
        composed = compose(draw.vector, carry_in=carry_in)
        c.score_channel_busy_until_ns = opened + int(round(composed.channel_busy_until * 1_000_000))
        outcome = classify(receipt_elapsed_ns(composed.receipt), q_reward)
        resolution_ns = opened + outcome.resolution_elapsed_ns
        event = contract.RewardEventV1(
            identity=identity, action=action, kind=outcome.kind, action_open_timestamp_ns=opened,
            resolution_timestamp_ns=resolution_ns, clock_domain=CR.CLOCK_DOMAIN, source=c.SCHEMA,
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
        value = (c.catalog.reward_value(context.reward_scene_key, mode_id=action.mode_id, q_e4=action.q_e4)
                 if q_reward is not None else None)
        context.resolved = {
            "q_perc_grid": q_reward, "q_defined": q_reward is not None, "reward_value": value, "is_grid_point": anchor,
            "wire_reward": wire_reward, "wire_held": wire_held, "ingress_bytes": int(wire_reward) + int(wire_held),
            "base_chain_id": draw.base_chain_id, "donor_chain_id": draw.donor_chain_id,
            "base_action": list(draw.base_action), "delta_ms": dict(draw.delta_ms),
            "uplink_on_time_decision": draw.uplink_on_time_decision, "uplink_late_probability": draw.uplink_late_probability,
            "uplink_late": draw.uplink_late, "scene_distance": draw.scene_distance, "floors": list(draw.floors),
            "stage_ms": {s: getattr(draw.vector, s) for s in "ABCDEFH"}, "gt_ready_ms": composed.gt_ready,
            "ordering": composed.ordering.value, "prediction_at_service_ms": composed.prediction_at_service,
            "scoring_end_ms": composed.scoring_end, "carry_in_ms": carry_in, "receipt_ms": composed.receipt,
            "critical_path": [list(x) for x in composed.critical_path], "event_kind": outcome.kind.value,
            "resolution_elapsed_ns": outcome.resolution_elapsed_ns}
        return environment.KernelCycleResultV1(hold=hold, reward_event=event, cycle_end_timestamp_ns=cycle_end,
                                               episode_boundary=contract.EpisodeBoundary.CONTINUES)


class FullRun5CollectorV1(CR.CorrectedRun5CollectorV1):
    SCHEMA = "splitfusion.run5_full.collector.v1"

    def __init__(self, *, latency, seed: int, shared_sources: Mapping[str, Any]) -> None:
        _require(isinstance(shared_sources.get("catalog"), Q.FullSceneCatalogV1), "shared sources lack the full catalogue")
        self.latency = latency
        self.decision_count = 0
        self.diagnostics: list[dict[str, Any]] = []
        super().__init__(calibration=_Adapter(latency), seed=seed, mode="FIXTURE", shared_sources=shared_sources)
        self.collector_binding_sha256 = _sha({"schema": self.SCHEMA, "corrected": self.binding_sha256,
                                              "reward": self.catalog.reward_binding_sha256,
                                              "latency": latency.binding_sha256, "support": S.SUPPORT_SHA256,
                                              "evidence_label": EVIDENCE_LABEL})

    def _session_uuid(self) -> str:
        if self._session_index < 1000:
            return super()._session_uuid()
        return f"{self._session_index:08x}-0000-4c00-8fff-{self._seed % 10**12:012d}"

    def _build_env(self) -> environment.Run4SequentialEnvironmentV1:
        return environment.Run4SequentialEnvironmentV1(
            state_provider=self._provider, kernel=_FullKernel(self), gamma=GAMMA,
            evidence_class=environment.EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE)

    def collect_transition(self, request: orch.ModeledActionRequestV1):
        _require(type(request) is orch.ModeledActionRequestV1, "foreign request")
        _require(request.decision_ordinal == self.decision_count, "out-of-order decision ordinal")
        _require(S.in_support(request.mode_id, request.q_e4), "action outside the original Run-5 support")
        state = self.current_state_features()
        current = self._env.current_state.state.state
        context = self.context
        action = self._action_identity(request.mode_id, request.q_e4)
        _require((action.mode_id, action.q_e4) == (request.mode_id, request.q_e4), "catalogue changed the action")
        cycle = self._env.step(action)
        resolved = dict(context.resolved)
        excluded = isinstance(cycle, environment.ExcludedCycleV1)
        terminal = cycle.terminal
        success = terminal is contract.RewardTerminal.SUCCESS
        if not excluded:
            _require(terminal in (contract.RewardTerminal.SUCCESS, contract.RewardTerminal.TIMEOUT),
                     f"unexpected terminal {terminal}")
        if success:
            _require(resolved["reward_value"] is not None and resolved["q_perc_grid"] is not None,
                     "SUCCESS without defined quality")
        reward = None if excluded else (float(resolved["reward_value"]) if success else 0.0)
        transition = None
        if excluded:
            self._session_index += 1
            self._open_session()
        else:
            next_state = self.current_state_features()
            transition = RC.Run5CollectedTransitionV1(
                request=request, run4_transition_sha256=cycle.transition_sha256,
                session_uuid=current.identity.session_uuid, decision_seq=current.identity.decision_seq,
                state=state, next_state=next_state, mode_id=action.mode_id, q_e4=action.q_e4, reward=reward,
                duration=int(cycle.duration), discount=GAMMA ** int(cycle.duration), has_next_state=True,
                terminated=False, truncated=False, terminal=terminal.value,
                q_perc=resolved["q_perc_grid"] if success else None,
                latency_ms=resolved["resolution_elapsed_ns"] / 1e6 if success else None)
        self.diagnostics.append(dict(
            ordinal=self.decision_count, source=request.source, mode_id=action.mode_id, q_e4=action.q_e4,
            terminal=terminal.value, excluded=excluded, reward=reward, q_perc=resolved["q_perc_grid"],
            reward_value=resolved["reward_value"], receipt_ms=resolved["receipt_ms"], profile=context.profile,
            prior_ul_mcs=context.prior_ul_mcs, snr_db=context.snr_db, backlog_bytes=context.backlog_bytes,
            camera_si=context.camera_si, radar_p40=context.radar_p40, reward_scene_key=context.reward_scene_key,
            wire_bytes=resolved["wire_reward"], base_chain_id=resolved["base_chain_id"],
            delta_ms=resolved["delta_ms"], uplink_late=resolved["uplink_late"],
            uplink_late_probability=resolved["uplink_late_probability"], floors=resolved["floors"], stage_ms=resolved["stage_ms"],
            gt_ready_ms=resolved["gt_ready_ms"], prediction_at_service_ms=resolved["prediction_at_service_ms"],
            ordering=resolved["ordering"], carry_in_ms=resolved["carry_in_ms"], state=state))
        self.decision_count += 1
        return transition

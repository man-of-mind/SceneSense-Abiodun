"""Phase 6: per-frame policy / hold / fallback decisions (pure, no I/O).

Per 10-Hz frame, in this order:

1. **Registered hold.**  While the current policy ticket is unresolved or has
   transmitted fewer than ``k_min = 2`` tensors, the frame reuses the exact
   policy action (``RewardHoldControllerV2.next_frame``).  No guard refusal
   can interrupt it.
2. **Decision opportunity.**  Otherwise the caller's state builder is asked
   for the exact 21-D state.  If it raises ``ExternalFallbackRequired`` the
   actor is **not** called, no ticket opens, and the frame carries the fixed
   external fallback (mode 11, q_e4 9800, anchor action 71,
   ``split_ae32_uint4_q9800``) with ``reward_requested = False``.  Every typed
   reason is logged.  The controller -- last policy action and outcome -- is
   untouched, so the next frame retries a guarded decision.  There is no TTL.
3. **Admitted.**  The frozen actor chooses ``(mode_id, q_e4)``; the dynamic
   contract resolves it; the identity seam attests it; a ticket opens and the
   frame is the reward-requested tensor.

Session semantics follow the Run-4 contract.  An excluded *evaluator* outcome
(e.g. undefined Q_perc) cannot become a previous outcome, so the next policy
decision opens a new decision session at genesis (explicitly counted).  A
transport failure after a tensor was assigned is an excluded
*infrastructure* fault: the engine faults and the qualification stops; it is
never converted into a policy timeout.
"""

from __future__ import annotations

import dataclasses
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec

from . import continuous_execution_v2 as X
from . import reward_hold_controller_v2 as R
from . import ue_telemetry_provider_v2 as T

__all__ = [
    "EngineError",
    "InfrastructureFault",
    "FALLBACK",
    "FALLBACK_SEQ",
    "FrameKind",
    "FramePlanV2",
    "Run4DecisionEngineV2",
    "adopt_snapshot",
    "policy_coverage",
]

FALLBACK = {"mode_id": 11, "q_e4": 9800, "anchor_action_id": 71,
            "family": "AE32", "quantizer": "UINT4",
            "profile_id": "split_ae32_uint4_q9800"}
FALLBACK_SEQ = (1 << 64) - 1      # decision/ticket sentinel: no policy ticket
COVERAGE_WARMUP_OPPORTUNITIES = 10
COVERAGE_MIN_FRACTION = 0.95      # Phase-2 G2 threshold, reused unchanged


class EngineError(RuntimeError):
    pass


class InfrastructureFault(EngineError):
    """Post-assignment transport failure: excluded; the run must stop."""


class DecisionCapacityExhausted(EngineError):
    """Addendum 6: no complete k_min decision cycle fits; nothing is assigned."""


class FrameKind(str, enum.Enum):
    POLICY_DECISION = "POLICY_DECISION"
    POLICY_HOLD = "POLICY_HOLD"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True, slots=True)
class FramePlanV2:
    kind: FrameKind
    frame_id: int
    tensor_seq: int
    capture_wall_ns: int
    profile: dec.ExecutableDispatchProfile
    frame_identity: X.FrameIdentityV2
    assignment: Optional[R.FrameAssignmentV2]
    fallback_reasons: Tuple[str, ...]
    decision_session_uuid: str
    opportunity_index: Optional[int]
    actor_decision: Optional[Any] = None


def adopt_snapshot(snapshot: T.TelemetrySnapshotV2, *, root_session: str,
                   decision_session: str) -> T.TelemetrySnapshotV2:
    """Re-scope one UE-attach telemetry snapshot to a child decision session.

    Only samples of the provider's own root session are adopted, and the
    sample sequence numbers are kept, so selection and causality are
    unchanged.  A foreign snapshot is refused rather than re-labelled.
    """
    if snapshot.session_uuid != root_session:
        raise EngineError("snapshot does not belong to the provider root session")

    def scoped(sample):
        return dataclasses.replace(sample, identity=contract.SampleIdentityV1(
            decision_session, sample.identity.ue_id, sample.identity.sample_seq))
    return dataclasses.replace(
        snapshot, session_uuid=decision_session,
        dci=tuple(scoped(item) for item in snapshot.dci if
                  item.identity.session_uuid == root_session),
        rlc=tuple(scoped(item) for item in snapshot.rlc if
                  item.identity.session_uuid == root_session))


def policy_coverage(opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    """Non-vacuous Phase-6 gate: after 10 opportunities, >=95% admitted."""
    scored = [row for row in opportunities
              if row["opportunity_index"] >= COVERAGE_WARMUP_OPPORTUNITIES]
    admitted = sum(1 for row in scored if row["admitted"])
    fraction = admitted / len(scored) if scored else 0.0
    verdict = ("PASS" if scored and fraction >= COVERAGE_MIN_FRACTION
               else "INCONCLUSIVE_OR_FAILED")
    return {"scored_opportunities": len(scored), "admitted": admitted,
            "admitted_fraction": fraction,
            "fallback_fraction": (1.0 - fraction) if scored else 1.0,
            "min_fraction": COVERAGE_MIN_FRACTION, "verdict": verdict}


StateBuilder = Callable[
    [contract.DecisionIdentityV1, Optional[contract.PreviousOutcomeV1]],
    Tuple[contract.GuardedPolicyStateV2, contract.PolicyFeatureVectorV2]]


@dataclass
class _Counters:
    policy_decisions: int = 0
    policy_holds: int = 0
    fallbacks: int = 0
    actor_calls: int = 0
    session_rollovers: int = 0


class Run4DecisionEngineV2:
    def __init__(self, *, contract_: dec.DynamicExecutionContract, actor: Any,
                 ue_id: str, controller_lineage_sha256: str,
                 clock_domain: str = T.CLOCK_DOMAIN,
                 session_factory: Callable[[], str] = lambda: str(uuid.uuid4())
                 ) -> None:
        self._contract = contract_
        self._actor = actor
        self.ue_id = ue_id
        self.lineage = controller_lineage_sha256
        self.clock_domain = clock_domain
        self._session_factory = session_factory
        self.fallback_profile = contract_.resolve_q_e4(
            FALLBACK["mode_id"], FALLBACK["q_e4"])
        fp = self.fallback_profile
        if (fp.action_id, fp.profile_id, fp.family, fp.quantizer) != (
                FALLBACK["anchor_action_id"], FALLBACK["profile_id"],
                FALLBACK["family"], FALLBACK["quantizer"]):
            raise EngineError("registered fallback identity does not reconcile")
        self.fallback_action = X.executed_identity_from_profile(fp, contract_)
        self.controllers: list[R.RewardHoldControllerV2] = []
        self._new_session()
        self._tensor_seq = 0
        self._opportunity = 0
        self.counters = _Counters()
        self.opportunities: list[dict[str, Any]] = []
        self.fallback_log: list[dict[str, Any]] = []
        self.faulted: Optional[str] = None
        # Addendum 6: the caller clears this when a complete k_min cycle can no
        # longer fit (frame budget / decision cap). Holds are never refused.
        self.new_opportunity_allowed = True

    # -- sessions ---------------------------------------------------------------
    def _new_session(self) -> None:
        self.controllers.append(R.RewardHoldControllerV2(
            session_uuid=self._session_factory(), ue_id=self.ue_id,
            controller_lineage_sha256=self.lineage, clock_domain=self.clock_domain))

    @property
    def controller(self) -> R.RewardHoldControllerV2:
        return self.controllers[-1]

    def controller_for(self, session_uuid: str) -> Optional[R.RewardHoldControllerV2]:
        for candidate in self.controllers:
            if candidate.session_uuid == session_uuid:
                return candidate
        return None

    # -- per frame ---------------------------------------------------------------
    def _frame_identity(self, assignment: Optional[R.FrameAssignmentV2], *,
                        frame_id: int, tensor_seq: int, capture_wall_ns: int
                        ) -> X.FrameIdentityV2:
        if assignment is None:
            return X.FrameIdentityV2(
                session_uuid=self.controller.session_uuid,
                controller_lineage_sha256=self.lineage, decision_seq=FALLBACK_SEQ,
                ticket_seq=FALLBACK_SEQ, frame_id=frame_id, tensor_seq=tensor_seq,
                capture_timestamp_ns=capture_wall_ns, reward_requested=False)
        return X.FrameIdentityV2(
            session_uuid=self.controller.session_uuid,
            controller_lineage_sha256=self.lineage,
            decision_seq=assignment.decision_seq, ticket_seq=assignment.ticket_seq,
            frame_id=assignment.frame_id, tensor_seq=assignment.tensor_seq,
            capture_timestamp_ns=assignment.capture_timestamp_ns,
            reward_requested=assignment.reward_requested)

    def plan_frame(self, *, frame_id: int, capture_wall_ns: int, now_raw_ns: int,
                   build_state: StateBuilder) -> FramePlanV2:
        if self.faulted is not None:
            raise EngineError(f"engine faulted: {self.faulted}")
        controller = self.controller
        if controller.current is not None and not controller.can_decide(now_raw_ns):
            self._sync_tensor_seq(controller)
            assignment = controller.next_frame(
                frame_id=frame_id, capture_timestamp_ns=capture_wall_ns,
                now_ns=now_raw_ns)
            self.counters.policy_holds += 1
            profile = self._contract.resolve_q_e4(assignment.mode_id, assignment.q_e4)
            return self._plan(FrameKind.POLICY_HOLD, assignment, profile, frame_id,
                              capture_wall_ns, (), None)

        if not self.new_opportunity_allowed:
            raise DecisionCapacityExhausted(
                "no complete k_min decision cycle fits; stopping at a closed cycle")
        index = self._opportunity
        self._opportunity += 1
        try:
            previous = controller.previous_for_next_decision()
        except R.SessionBreakRequired:
            self._new_session()
            self.counters.session_rollovers += 1
            controller = self.controller
            previous = None
        identity = controller.next_identity()
        try:
            guarded, features = build_state(identity, previous)
        except contract.ExternalFallbackRequired as exc:
            reasons = tuple(str(exc).split("; ")) or ("UNSPECIFIED",)
            return self._fallback(frame_id, capture_wall_ns, reasons, index)
        decision = self._actor.act(features)
        self.counters.actor_calls += 1
        profile = self._contract.resolve_q_e4(decision.mode_id, decision.q_e4)
        action = X.executed_identity_from_profile(profile, self._contract)
        controller.open_decision(
            identity=identity, action=action,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            action_open_ns=guarded.boundary.action_open_timestamp_ns)
        self._sync_tensor_seq(controller)
        assignment = controller.next_frame(
            frame_id=frame_id, capture_timestamp_ns=capture_wall_ns,
            now_ns=max(now_raw_ns, guarded.boundary.action_open_timestamp_ns))
        self.counters.policy_decisions += 1
        self.opportunities.append({"opportunity_index": index, "admitted": True,
                                   "frame_id": frame_id,
                                   "decision_seq": identity.decision_seq,
                                   "session_uuid": identity.session_uuid})
        return self._plan(FrameKind.POLICY_DECISION, assignment, profile, frame_id,
                          capture_wall_ns, (), index, decision)

    def _fallback(self, frame_id: int, capture_wall_ns: int,
                  reasons: Tuple[str, ...], index: int) -> FramePlanV2:
        self.counters.fallbacks += 1
        tensor_seq = self._tensor_seq
        record = {"opportunity_index": index, "admitted": False,
                  "frame_id": frame_id, "tensor_seq": tensor_seq,
                  "reasons": list(reasons), "fallback": dict(FALLBACK),
                  "execution_bundle_sha256":
                      self.fallback_profile.execution_bundle_sha256,
                  "session_uuid": self.controller.session_uuid}
        self.opportunities.append(record)
        self.fallback_log.append(record)
        return self._plan(FrameKind.FALLBACK, None, self.fallback_profile, frame_id,
                          capture_wall_ns, reasons, index)

    def _sync_tensor_seq(self, controller: R.RewardHoldControllerV2) -> None:
        # One strictly increasing wire tensor sequence across fallback frames
        # and decision sessions (the edge frame-context validator requires a
        # monotone sequence per stream).  The controller keys its reward frame
        # on exactly this number.
        controller._tensor_seq = self._tensor_seq

    def _plan(self, kind, assignment, profile, frame_id, capture_wall_ns, reasons,
              index, decision=None) -> FramePlanV2:
        tensor_seq = (assignment.tensor_seq if assignment is not None
                      else self._tensor_seq)
        if tensor_seq != self._tensor_seq:
            raise EngineError("wire tensor sequence drifted from the controller")
        self._tensor_seq = tensor_seq + 1
        return FramePlanV2(
            kind=kind, frame_id=frame_id, tensor_seq=tensor_seq,
            capture_wall_ns=capture_wall_ns, profile=profile,
            frame_identity=self._frame_identity(
                assignment, frame_id=frame_id, tensor_seq=tensor_seq,
                capture_wall_ns=capture_wall_ns),
            assignment=assignment, fallback_reasons=tuple(reasons),
            decision_session_uuid=self.controller.session_uuid,
            opportunity_index=index, actor_decision=decision)

    # -- transport outcome ---------------------------------------------------------
    def transport_failed(self, plan: FramePlanV2, detail: str) -> None:
        """Post-assignment send failure: excluded infrastructure fault, stop."""
        self.faulted = (f"INFRASTRUCTURE_FAULT after assignment of frame "
                        f"{plan.frame_id} ({plan.kind.value}): {detail}")
        raise InfrastructureFault(self.faulted)

    # -- feedback ------------------------------------------------------------------
    def on_feedback(self, fb: R.RewardFeedbackV2, *, receipt_raw_ns: int
                    ) -> R.FeedbackClass:
        controller = self.controller_for(fb.session_uuid)
        target = controller if controller is not None else self.controller
        return target.on_feedback(fb, receipt_ns=receipt_raw_ns)

    def on_registered_terminal(self, terminal: R.RegisteredTerminalV2, *,
                               receipt_raw_ns: int) -> R.FeedbackClass:
        """Addendum 5: exact edge service terminal -> its own session's controller."""
        controller = self.controller_for(terminal.session_uuid)
        target = controller if controller is not None else self.controller
        return target.on_registered_terminal(terminal, receipt_ns=receipt_raw_ns)

    def coverage(self) -> dict[str, Any]:
        return policy_coverage(self.opportunities)

"""Phase 4: Run-4 action hold and 170-ms reward ticket (one serialized loop).

Bound directly to the registered Run-4 semantics:

* ``run4_contract.REWARD_DEADLINE_NS == 170_000_000``: action-open to
  feedback, inclusive.  This is the Run-4 share of the ~200-ms
  capture-to-feedback budget.  The obsolete Run-3 200-ms ticket controller is
  neither edited nor reused.
* ``MINIMUM_HOLD_TENSORS == 2``: a decision's first transmitted tensor
  requests the reward; every later tensor reuses the exact action with
  ``reward_requested = False``.  A new decision is permitted only after the
  ticket is resolved (feedback or timeout) **and** at least two tensors of
  the action were transmitted.
* Resolution goes through ``run4_contract.resolve_reward``.  Timeout resolves
  at ``action_open + TIMEOUT_RESOLUTION_ELAPSED_NS`` with reward ``-1``.
  Feedback after the boundary is a ``LATE_ORPHAN`` and never attaches to a
  newer ticket.  Byte-identical duplicates are ignored; conflicting
  duplicates fail closed.  Infrastructure/evaluator faults are excluded (not
  policy failures) and break the previous-outcome chain, so the next decision
  must start a new session.  A timeout alone is not an episode termination.
* Map installation and map ACKs are recorded in a separate ledger and never
  touch reward state.

All state changes happen in the caller's single event-loop thread.  Worker
threads may only post events into :meth:`post` / drain via :meth:`run_pending`.
Importing this module starts nothing.
"""

from __future__ import annotations

import enum
import hashlib
import json
import queue
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1.sequential_kernel import (
    TIMEOUT_RESOLUTION_ELAPSED_NS,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    MINIMUM_HOLD_TENSORS,
    ExecutedActionIdentity,
)

__all__ = [
    "ControllerError",
    "ConflictingFeedbackError",
    "SessionBreakRequired",
    "FeedbackClass",
    "RewardFeedbackV2",
    "RegisteredTerminalV2",
    "SERVICE_FAILURE_EDGE_OUTCOMES",
    "FrameAssignmentV2",
    "RewardHoldControllerV2",
    "REWARD_DEADLINE_NS",
    "K_MIN",
]

REWARD_DEADLINE_NS = contract.REWARD_DEADLINE_NS
K_MIN = MINIMUM_HOLD_TENSORS
if REWARD_DEADLINE_NS != 170_000_000 or K_MIN != 2:
    raise RuntimeError("registered Run-4 hold/deadline constants drifted")
if TIMEOUT_RESOLUTION_ELAPSED_NS <= REWARD_DEADLINE_NS:
    raise RuntimeError("timeout resolution must be strictly after the deadline")
SOURCE = "splitfusion_run4_live_v2:reward_hold_controller"
FEEDBACK_SCHEMA = "scenesense.run4_live_v2.reward_feedback.v1"


class ControllerError(RuntimeError):
    """The controller was driven outside its registered protocol."""


class ConflictingFeedbackError(ControllerError):
    """Two different feedback messages claim the same reward request."""


class SessionBreakRequired(ControllerError):
    """An excluded fault broke the previous-outcome chain."""


_FEEDBACK_KINDS = {
    "DELIVERED_SUCCESS": contract.RewardEventKind.DELIVERED_SUCCESS,
    "REGISTERED_DELIVERY_FAILURE": contract.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
    "REGISTERED_SERVICE_FAILURE": contract.RewardEventKind.REGISTERED_SERVICE_FAILURE,
    "INFRASTRUCTURE_FAULT": contract.RewardEventKind.INFRASTRUCTURE_FAULT,
    "EVALUATOR_FAULT": contract.RewardEventKind.EVALUATOR_FAULT,
}


@dataclass(frozen=True, slots=True)
class RewardFeedbackV2:
    """Evaluator -> UE feedback, bound to one exact reward request."""

    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    mode_id: int
    q_e4: int
    execution_bundle_sha256: str
    anchor_action_id: Optional[int]
    reward_requested: bool
    kind: str
    q_perc: Optional[float]

    def __post_init__(self) -> None:
        if self.kind not in _FEEDBACK_KINDS:
            raise ControllerError(f"unknown feedback kind {self.kind!r}")
        if (self.kind == "DELIVERED_SUCCESS") != (self.q_perc is not None):
            raise ControllerError("only DELIVERED_SUCCESS carries q_perc")

    def canonical_bytes(self) -> bytes:
        body = {"schema": FEEDBACK_SCHEMA,
                **{name: getattr(self, name) for name in self.__dataclass_fields__}}
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")

    @property
    def request_key(self) -> Tuple[str, str, int, int]:
        return (self.session_uuid, self.controller_lineage_sha256,
                self.decision_seq, self.ticket_seq)


# Addendum 5: exact edge terminals for an active reward-requested policy frame.
# The Phase-6 edge emits these only for frames it never processed, so no
# quality feedback can exist for them; each resolves the ticket as the
# existing REGISTERED_SERVICE_FAILURE (reward -1, no q_perc). Agent credit
# (e.g. CREDIT_SUPERSEDED_BY_FRESHER) is recorded separately, never as loss.
SERVICE_FAILURE_EDGE_OUTCOMES = ("SUPERSEDED_PENDING", "STALE_BEFORE_EDGE",
                                 "STALE_BEFORE_MAP")
TERMINAL_SCHEMA = "scenesense.run4_live_v2.registered_terminal.v1"


@dataclass(frozen=True, slots=True)
class RegisteredTerminalV2:
    """An exact registered edge terminal for one reward-requested policy frame."""

    session_uuid: str
    controller_lineage_sha256: str
    decision_seq: int
    ticket_seq: int
    frame_id: int
    tensor_seq: int
    capture_timestamp_ns: int
    mode_id: int
    q_e4: int
    execution_bundle_sha256: str
    anchor_action_id: Optional[int]
    reward_requested: bool
    outcome: str
    agent_credit: str
    stage: str

    def __post_init__(self) -> None:
        if self.outcome not in SERVICE_FAILURE_EDGE_OUTCOMES:
            raise ControllerError(f"terminal outcome {self.outcome!r} is not a "
                                  "registered service failure")

    def canonical_bytes(self) -> bytes:
        body = {"schema": TERMINAL_SCHEMA,
                **{name: getattr(self, name) for name in self.__dataclass_fields__}}
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")

    @property
    def request_key(self) -> Tuple[str, str, int, int]:
        return (self.session_uuid, self.controller_lineage_sha256,
                self.decision_seq, self.ticket_seq)


class FeedbackClass(str, enum.Enum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE_IGNORED = "DUPLICATE_IGNORED"
    LATE_ORPHAN = "LATE_ORPHAN"
    UNKNOWN_ORPHAN = "UNKNOWN_ORPHAN"


@dataclass(frozen=True, slots=True)
class FrameAssignmentV2:
    """What one transmitted tensor carries; every tensor goes to map/tail."""

    decision_seq: int
    ticket_seq: int
    tensor_seq: int
    frame_id: int
    capture_timestamp_ns: int
    reward_requested: bool
    action: ExecutedActionIdentity
    execution_bundle_sha256: str
    mode_id: int
    q_e4: int
    anchor_action_id: Optional[int]


@dataclass
class _Ticket:
    identity: contract.DecisionIdentityV1
    ticket_seq: int
    action: ExecutedActionIdentity
    bundle_sha256: str
    action_open_ns: int
    transmitted: int = 0
    reward_frame: Optional[FrameAssignmentV2] = None
    resolution: Optional[contract.RewardResolutionV1] = None
    accepted_bytes: Optional[bytes] = None
    resolved_by: Optional[str] = None


@dataclass
class _Ledger:
    feedback: list[dict[str, Any]] = field(default_factory=list)
    map_acks: list[dict[str, Any]] = field(default_factory=list)
    frames: list[FrameAssignmentV2] = field(default_factory=list)


class RewardHoldControllerV2:
    def __init__(self, *, session_uuid: str, ue_id: str,
                 controller_lineage_sha256: str, clock_domain: str) -> None:
        contract.DecisionIdentityV1(session_uuid, ue_id, 0)
        self.session_uuid = session_uuid
        self.ue_id = ue_id
        self.lineage = controller_lineage_sha256
        self.clock_domain = clock_domain
        self._ticket: Optional[_Ticket] = None
        self._closed: dict[Tuple[str, str, int, int], _Ticket] = {}
        self._next_decision_seq = 0
        self._tensor_seq = 0
        self._faulted: Optional[str] = None
        self._session_broken = False
        self.ledger = _Ledger()
        self._events: "queue.Queue[Tuple[str, Any, int]]" = queue.Queue()

    # -- decision gating --------------------------------------------------------
    @property
    def current(self) -> Optional[_Ticket]:
        return self._ticket

    def _require_healthy(self) -> None:
        if self._faulted is not None:
            raise ControllerError(f"controller faulted: {self._faulted}")

    def can_decide(self, now_ns: int) -> bool:
        self._require_healthy()
        self.poll(now_ns)
        ticket = self._ticket
        return ticket is None or (ticket.resolution is not None
                                  and ticket.transmitted >= K_MIN)

    def next_identity(self) -> contract.DecisionIdentityV1:
        return contract.DecisionIdentityV1(
            self.session_uuid, self.ue_id, self._next_decision_seq)

    def previous_for_next_decision(self) -> Optional[contract.PreviousOutcomeV1]:
        """Exact previous outcome for the next state; None only at genesis."""
        self._require_healthy()
        if self._session_broken:
            raise SessionBreakRequired(
                "an excluded infrastructure/evaluator fault closed the previous "
                "ticket; the next decision must open a new session")
        ticket = self._ticket
        if ticket is None:
            return None
        if ticket.resolution is None:
            raise ControllerError("previous ticket is unresolved")
        return contract.PreviousOutcomeV1.from_resolution(ticket.resolution)

    def open_decision(self, *, identity: contract.DecisionIdentityV1,
                      action: ExecutedActionIdentity, execution_bundle_sha256: str,
                      action_open_ns: int) -> _Ticket:
        if not self.can_decide(action_open_ns):
            raise ControllerError("hold/ticket not complete; decision refused")
        if self._session_broken:
            raise SessionBreakRequired("session broken by an excluded fault")
        if identity != self.next_identity():
            raise ControllerError("decision identity is not the next in sequence")
        action.require_reconciled()
        if self._ticket is not None:
            closed = self._ticket
            self._closed[(self.session_uuid, self.lineage,
                          closed.identity.decision_seq, closed.ticket_seq)] = closed
        self._ticket = _Ticket(identity=identity, ticket_seq=identity.decision_seq,
                               action=action, bundle_sha256=execution_bundle_sha256,
                               action_open_ns=action_open_ns)
        self._next_decision_seq += 1
        return self._ticket

    # -- per-frame --------------------------------------------------------------
    def next_frame(self, *, frame_id: int, capture_timestamp_ns: int,
                   now_ns: int) -> FrameAssignmentV2:
        self._require_healthy()
        ticket = self._ticket
        if ticket is None:
            raise ControllerError("no action has been decided yet")
        self.poll(now_ns)
        request = ticket.transmitted == 0
        assignment = FrameAssignmentV2(
            decision_seq=ticket.identity.decision_seq, ticket_seq=ticket.ticket_seq,
            tensor_seq=self._tensor_seq, frame_id=frame_id,
            capture_timestamp_ns=capture_timestamp_ns, reward_requested=request,
            action=ticket.action, execution_bundle_sha256=ticket.bundle_sha256,
            mode_id=ticket.action.mode_id, q_e4=ticket.action.q_e4,
            anchor_action_id=ticket.action.action_id)
        self._tensor_seq += 1
        ticket.transmitted += 1
        if request:
            ticket.reward_frame = assignment
        self.ledger.frames.append(assignment)
        return assignment

    # -- resolution -------------------------------------------------------------
    def _resolve(self, ticket: _Ticket, *, kind: contract.RewardEventKind,
                 resolution_ns: int, q_perc: Optional[float], by: str) -> None:
        event = contract.RewardEventV1(
            identity=ticket.identity, action=ticket.action, kind=kind,
            action_open_timestamp_ns=ticket.action_open_ns,
            resolution_timestamp_ns=resolution_ns, clock_domain=self.clock_domain,
            source=SOURCE, q_perc=q_perc)
        ticket.resolution = contract.resolve_reward(event)
        ticket.resolved_by = by
        if not ticket.resolution.learning_included:
            self._session_broken = True

    def poll(self, now_ns: int) -> None:
        ticket = self._ticket
        if ticket is None or ticket.resolution is not None:
            return
        if now_ns > ticket.action_open_ns + REWARD_DEADLINE_NS:
            self._resolve(ticket, kind=contract.RewardEventKind.TIMEOUT,
                          resolution_ns=ticket.action_open_ns
                          + TIMEOUT_RESOLUTION_ELAPSED_NS,
                          q_perc=None, by="TIMEOUT")

    def _matches(self, ticket: _Ticket, fb: RewardFeedbackV2) -> bool:
        frame = ticket.reward_frame
        return (frame is not None and fb.reward_requested is True
                and (fb.frame_id, fb.tensor_seq, fb.capture_timestamp_ns)
                == (frame.frame_id, frame.tensor_seq, frame.capture_timestamp_ns)
                and (fb.mode_id, fb.q_e4, fb.anchor_action_id)
                == (ticket.action.mode_id, ticket.action.q_e4, ticket.action.action_id)
                and fb.execution_bundle_sha256 == ticket.bundle_sha256)

    def on_feedback(self, fb: RewardFeedbackV2, *, receipt_ns: int) -> FeedbackClass:
        self._require_healthy()
        self.poll(receipt_ns)
        payload = fb.canonical_bytes()
        record = {"receipt_ns": receipt_ns, "request_key": list(fb.request_key),
                  "sha256": hashlib.sha256(payload).hexdigest()}
        key = fb.request_key
        ticket = self._ticket
        live_key = None if ticket is None else (
            self.session_uuid, self.lineage, ticket.identity.decision_seq,
            ticket.ticket_seq)
        target = ticket if key == live_key else self._closed.get(key)
        if target is None:
            record["class"] = FeedbackClass.UNKNOWN_ORPHAN.value
            self.ledger.feedback.append(record)
            return FeedbackClass.UNKNOWN_ORPHAN
        if target.accepted_bytes is not None:
            if payload == target.accepted_bytes:
                record["class"] = FeedbackClass.DUPLICATE_IGNORED.value
                self.ledger.feedback.append(record)
                return FeedbackClass.DUPLICATE_IGNORED
            self._faulted = f"conflicting feedback for {key}"
            raise ConflictingFeedbackError(self._faulted)
        if not self._matches(target, fb):
            self._faulted = f"feedback identity disagrees with request {key}"
            raise ConflictingFeedbackError(self._faulted)
        if target.resolution is not None:   # timed out before this arrived
            record["class"] = FeedbackClass.LATE_ORPHAN.value
            self.ledger.feedback.append(record)
            return FeedbackClass.LATE_ORPHAN
        self._resolve(target, kind=_FEEDBACK_KINDS[fb.kind], resolution_ns=receipt_ns,
                      q_perc=fb.q_perc, by="FEEDBACK")
        target.accepted_bytes = payload
        record["class"] = FeedbackClass.ACCEPTED.value
        self.ledger.feedback.append(record)
        return FeedbackClass.ACCEPTED

    def on_registered_terminal(self, terminal: RegisteredTerminalV2, *,
                               receipt_ns: int) -> FeedbackClass:
        """Resolve an exact edge service terminal as REGISTERED_SERVICE_FAILURE.

        Same discipline as :meth:`on_feedback`: full identity match (frame ID
        alone is insufficient), identical duplicates ignored, conflicting
        duplicates fail closed, and a terminal after the timeout is a late
        orphan that never attaches to a newer action. No q_perc is fabricated.
        """
        self._require_healthy()
        if type(terminal) is not RegisteredTerminalV2:
            raise ControllerError("terminal has a foreign type")
        self.poll(receipt_ns)
        payload = terminal.canonical_bytes()
        record = {"receipt_ns": receipt_ns, "request_key": list(terminal.request_key),
                  "sha256": hashlib.sha256(payload).hexdigest(),
                  "source": "REGISTERED_TERMINAL", "outcome": terminal.outcome,
                  "agent_credit": terminal.agent_credit}
        key = terminal.request_key
        ticket = self._ticket
        live_key = None if ticket is None else (
            self.session_uuid, self.lineage, ticket.identity.decision_seq,
            ticket.ticket_seq)
        target = ticket if key == live_key else self._closed.get(key)
        if target is None:
            record["class"] = FeedbackClass.UNKNOWN_ORPHAN.value
            self.ledger.feedback.append(record)
            return FeedbackClass.UNKNOWN_ORPHAN
        if target.accepted_bytes is not None:
            if payload == target.accepted_bytes:
                record["class"] = FeedbackClass.DUPLICATE_IGNORED.value
                self.ledger.feedback.append(record)
                return FeedbackClass.DUPLICATE_IGNORED
            self._faulted = f"conflicting terminal for {key}"
            raise ConflictingFeedbackError(self._faulted)
        frame = target.reward_frame
        if not (frame is not None and terminal.reward_requested is True
                and (terminal.frame_id, terminal.tensor_seq, terminal.capture_timestamp_ns)
                == (frame.frame_id, frame.tensor_seq, frame.capture_timestamp_ns)
                and (terminal.mode_id, terminal.q_e4, terminal.anchor_action_id)
                == (target.action.mode_id, target.action.q_e4, target.action.action_id)
                and terminal.execution_bundle_sha256 == target.bundle_sha256):
            self._faulted = f"terminal identity disagrees with request {key}"
            raise ConflictingFeedbackError(self._faulted)
        if target.resolution is not None:   # timed out before this arrived
            record["class"] = FeedbackClass.LATE_ORPHAN.value
            self.ledger.feedback.append(record)
            return FeedbackClass.LATE_ORPHAN
        self._resolve(target, kind=contract.RewardEventKind.REGISTERED_SERVICE_FAILURE,
                      resolution_ns=receipt_ns, q_perc=None, by="REGISTERED_TERMINAL")
        target.accepted_bytes = payload
        record["class"] = FeedbackClass.ACCEPTED.value
        self.ledger.feedback.append(record)
        return FeedbackClass.ACCEPTED

    def on_map_ack(self, *, frame_id: int, receipt_ns: int) -> None:
        """Asynchronous map install/ACK: recorded only, never reward state."""
        self.ledger.map_acks.append({"frame_id": frame_id, "receipt_ns": receipt_ns})

    # -- serialized event posting ----------------------------------------------
    def post(self, kind: str, payload: Any, receipt_ns: int) -> None:
        """Thread-safe: workers enqueue; only the loop thread mutates state."""
        self._events.put((kind, payload, receipt_ns))

    def run_pending(self) -> list[Tuple[str, Any]]:
        handled = []
        while True:
            try:
                kind, payload, receipt = self._events.get_nowait()
            except queue.Empty:
                return handled
            if kind == "feedback":
                handled.append((kind, self.on_feedback(payload, receipt_ns=receipt)))
            elif kind == "map_ack":
                self.on_map_ack(frame_id=payload, receipt_ns=receipt)
                handled.append((kind, payload))
            else:
                raise ControllerError(f"unknown event kind {kind!r}")

    def resolutions(self) -> list[contract.RewardResolutionV1]:
        tickets = list(self._closed.values()) + ([self._ticket] if self._ticket else [])
        return [t.resolution for t in sorted(tickets, key=lambda t: t.ticket_seq)
                if t.resolution is not None]

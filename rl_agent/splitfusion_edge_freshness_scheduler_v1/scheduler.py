"""CPU-pure contract for freshness-first SplitFusion edge scheduling.

This module is deliberately independent of the deployed CUDA/runtime path. It
defines the identities, terminal outcomes, queue policy, and agent-credit view
that must be qualified before the policy is integrated with the live edge.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any


SCHEMA = "scenesense.splitfusion.edge_freshness_feedback.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Stage(str, Enum):
    PENDING = "PENDING"
    BEFORE_DECODE = "BEFORE_DECODE"
    BEFORE_TAIL = "BEFORE_TAIL"
    BEFORE_PUBLICATION = "BEFORE_PUBLICATION"
    MAP_INSTALL = "MAP_INSTALL"


class TerminalReason(str, Enum):
    MAP_INSTALLED = "MAP_INSTALLED"
    SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
    SUPERSEDED_BEFORE_DECODE = "SUPERSEDED_BEFORE_DECODE"
    SUPERSEDED_BEFORE_TAIL = "SUPERSEDED_BEFORE_TAIL"
    SUPERSEDED_BEFORE_PUBLICATION = "SUPERSEDED_BEFORE_PUBLICATION"
    QUEUE_WAIT_BUDGET_EXCEEDED = "QUEUE_WAIT_BUDGET_EXCEEDED"
    PROCESSING_HORIZON_EXPIRED = "PROCESSING_HORIZON_EXPIRED"
    OUT_OF_ORDER_ARRIVAL = "OUT_OF_ORDER_ARRIVAL"
    TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    IDENTITY_REJECTED = "IDENTITY_REJECTED"


class OutcomeClass(str, Enum):
    INSTALLED_UTILITY = "INSTALLED_UTILITY"
    INTENTIONAL_FRESHNESS_DROP = "INTENTIONAL_FRESHNESS_DROP"
    EXPIRED_WORK = "EXPIRED_WORK"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"


_SUPERSEDED_REASON = {
    Stage.PENDING: TerminalReason.SUPERSEDED_PENDING,
    Stage.BEFORE_DECODE: TerminalReason.SUPERSEDED_BEFORE_DECODE,
    Stage.BEFORE_TAIL: TerminalReason.SUPERSEDED_BEFORE_TAIL,
    Stage.BEFORE_PUBLICATION: TerminalReason.SUPERSEDED_BEFORE_PUBLICATION,
}


@dataclass(frozen=True)
class FrameTicket:
    run_id: str
    cell_id: str
    stream_id: str
    frame_id: int
    sequence_id: int
    action_id: int
    capture_timestamp_ns: int
    edge_arrival_timestamp_ns: int
    feature_bytes: int

    def __post_init__(self) -> None:
        _require(bool(self.run_id), "run_id is required")
        _require(bool(self.cell_id), "cell_id is required")
        _require(bool(self.stream_id), "stream_id is required")
        _require(self.frame_id >= 0, "frame_id must be non-negative")
        _require(self.sequence_id >= 0, "sequence_id must be non-negative")
        _require(self.action_id >= 0, "action_id must be non-negative")
        _require(self.capture_timestamp_ns >= 0, "capture timestamp is invalid")
        _require(
            self.edge_arrival_timestamp_ns >= self.capture_timestamp_ns,
            "edge arrival precedes capture",
        )
        _require(self.feature_bytes >= 0, "feature_bytes must be non-negative")

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        return (
            self.run_id,
            self.cell_id,
            self.stream_id,
            self.frame_id,
            self.sequence_id,
        )


@dataclass(frozen=True)
class TerminalFeedback:
    ticket: FrameTicket
    reason: TerminalReason
    outcome_class: OutcomeClass
    stage: Stage
    observed_timestamp_ns: int
    queue_wait_ns: int
    bytes_already_sent: int
    compute_spent_ns: int = 0
    replacing_frame_id: int | None = None
    replacing_sequence_id: int | None = None

    def __post_init__(self) -> None:
        _require(
            self.observed_timestamp_ns >= self.ticket.capture_timestamp_ns,
            "feedback time precedes capture",
        )
        _require(self.queue_wait_ns >= 0, "queue_wait_ns must be non-negative")
        _require(self.bytes_already_sent >= 0, "bytes_already_sent is invalid")
        _require(self.compute_spent_ns >= 0, "compute_spent_ns is invalid")
        if self.reason.value.startswith("SUPERSEDED_"):
            _require(
                self.replacing_frame_id is not None
                and self.replacing_sequence_id is not None,
                "supersession feedback requires the replacing identity",
            )

    @property
    def age_ns(self) -> int:
        return self.observed_timestamp_ns - self.ticket.capture_timestamp_ns

    def agent_credit(self) -> "AgentCredit":
        installed = self.reason is TerminalReason.MAP_INSTALLED
        intentional = self.outcome_class is OutcomeClass.INTENTIONAL_FRESHNESS_DROP
        return AgentCredit(
            installation_utility_eligible=installed,
            count_as_transport_failure=(
                self.outcome_class is OutcomeClass.TRANSPORT_FAILURE
            ),
            count_as_structural_failure=(
                self.outcome_class is OutcomeClass.STRUCTURAL_FAILURE
            ),
            intentional_freshness_drop=intentional,
            charge_feature_bytes=self.bytes_already_sent,
            charge_compute_ns=self.compute_spent_ns,
            wasted_feature_bytes=(self.bytes_already_sent if not installed else 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "run_id": self.ticket.run_id,
            "cell_id": self.ticket.cell_id,
            "stream_id": self.ticket.stream_id,
            "frame_id": self.ticket.frame_id,
            "sequence_id": self.ticket.sequence_id,
            "action_id": self.ticket.action_id,
            "capture_timestamp_ns": self.ticket.capture_timestamp_ns,
            "edge_arrival_timestamp_ns": self.ticket.edge_arrival_timestamp_ns,
            "terminal_observed_timestamp_ns": self.observed_timestamp_ns,
            "terminal_reason": self.reason.value,
            "outcome_class": self.outcome_class.value,
            "stage": self.stage.value,
            "age_ns": self.age_ns,
            "queue_wait_ns": self.queue_wait_ns,
            "feature_bytes": self.ticket.feature_bytes,
            "bytes_already_sent": self.bytes_already_sent,
            "compute_spent_ns": self.compute_spent_ns,
            "replacing_frame_id": self.replacing_frame_id,
            "replacing_sequence_id": self.replacing_sequence_id,
            "agent_credit": self.agent_credit().to_dict(),
        }


@dataclass(frozen=True)
class AgentCredit:
    installation_utility_eligible: bool
    count_as_transport_failure: bool
    count_as_structural_failure: bool
    intentional_freshness_drop: bool
    charge_feature_bytes: int
    charge_compute_ns: int
    wasted_feature_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "installation_utility_eligible": self.installation_utility_eligible,
            "count_as_transport_failure": self.count_as_transport_failure,
            "count_as_structural_failure": self.count_as_structural_failure,
            "intentional_freshness_drop": self.intentional_freshness_drop,
            "charge_feature_bytes": self.charge_feature_bytes,
            "charge_compute_ns": self.charge_compute_ns,
            "wasted_feature_bytes": self.wasted_feature_bytes,
        }


@dataclass(frozen=True)
class Admission:
    admitted: bool
    feedback: tuple[TerminalFeedback, ...]


@dataclass(frozen=True)
class TakeResult:
    ticket: FrameTicket | None
    feedback: tuple[TerminalFeedback, ...]


class FreshnessScheduler:
    """Deterministic latest-frame scheduler with explicit terminal outcomes.

    ``queue_wait_budget_ns`` is deliberately optional. A value such as 10 ms
    belongs in a prospective sweep; it is not a scientific default.

    Running CUDA kernels are not preempted. ``stage_gate`` makes already-started
    work cancellable only at safe boundaries before decode, tail, or publication.
    """

    def __init__(
        self,
        *,
        processing_horizon_ns: int,
        queue_wait_budget_ns: int | None = None,
    ) -> None:
        _require(processing_horizon_ns > 0, "processing horizon must be positive")
        if queue_wait_budget_ns is not None:
            _require(queue_wait_budget_ns >= 0, "queue wait budget is invalid")
        self.processing_horizon_ns = int(processing_horizon_ns)
        self.queue_wait_budget_ns = (
            None if queue_wait_budget_ns is None else int(queue_wait_budget_ns)
        )
        self._pending: "OrderedDict[str, FrameTicket]" = OrderedDict()
        self._latest: dict[str, FrameTicket] = {}
        self._inflight: dict[str, tuple[FrameTicket, int]] = {}
        self._terminal: set[tuple[str, str, str, int, int]] = set()

    def _terminalize(
        self,
        ticket: FrameTicket,
        *,
        reason: TerminalReason,
        outcome_class: OutcomeClass,
        stage: Stage,
        now_ns: int,
        compute_spent_ns: int = 0,
        replacement: FrameTicket | None = None,
    ) -> TerminalFeedback:
        _require(ticket.identity not in self._terminal, "frame already terminal")
        _require(now_ns >= ticket.edge_arrival_timestamp_ns, "time precedes arrival")
        inflight = self._inflight.get(ticket.stream_id)
        queue_finished_ns = (
            int(now_ns)
            if inflight is None or inflight[0] != ticket
            else int(inflight[1])
        )
        feedback = TerminalFeedback(
            ticket=ticket,
            reason=reason,
            outcome_class=outcome_class,
            stage=stage,
            observed_timestamp_ns=int(now_ns),
            queue_wait_ns=queue_finished_ns - ticket.edge_arrival_timestamp_ns,
            bytes_already_sent=ticket.feature_bytes,
            compute_spent_ns=int(compute_spent_ns),
            replacing_frame_id=(None if replacement is None else replacement.frame_id),
            replacing_sequence_id=(
                None if replacement is None else replacement.sequence_id
            ),
        )
        self._terminal.add(ticket.identity)
        if inflight is not None and inflight[0] == ticket:
            del self._inflight[ticket.stream_id]
        return feedback

    def offer(self, ticket: FrameTicket, *, now_ns: int) -> Admission:
        _require(now_ns >= ticket.edge_arrival_timestamp_ns, "time precedes arrival")
        latest = self._latest.get(ticket.stream_id)
        if latest is not None and latest.sequence_id >= ticket.sequence_id:
            feedback = self._terminalize(
                ticket,
                reason=TerminalReason.OUT_OF_ORDER_ARRIVAL,
                outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
                stage=Stage.PENDING,
                now_ns=now_ns,
                replacement=latest,
            )
            return Admission(False, (feedback,))

        self._latest[ticket.stream_id] = ticket
        displaced = self._pending.get(ticket.stream_id)
        self._pending[ticket.stream_id] = ticket
        self._pending.move_to_end(ticket.stream_id)
        if displaced is None:
            return Admission(True, ())
        feedback = self._terminalize(
            displaced,
            reason=TerminalReason.SUPERSEDED_PENDING,
            outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
            stage=Stage.PENDING,
            now_ns=now_ns,
            replacement=ticket,
        )
        return Admission(True, (feedback,))

    def take(self, *, now_ns: int) -> TakeResult:
        feedback: list[TerminalFeedback] = []
        while self._pending:
            stream_id, ticket = self._pending.popitem(last=False)
            queue_wait = int(now_ns) - ticket.edge_arrival_timestamp_ns
            if (
                self.queue_wait_budget_ns is not None
                and queue_wait > self.queue_wait_budget_ns
            ):
                feedback.append(
                    self._terminalize(
                        ticket,
                        reason=TerminalReason.QUEUE_WAIT_BUDGET_EXCEEDED,
                        outcome_class=OutcomeClass.EXPIRED_WORK,
                        stage=Stage.PENDING,
                        now_ns=now_ns,
                    )
                )
                continue
            if int(now_ns) - ticket.capture_timestamp_ns > self.processing_horizon_ns:
                feedback.append(
                    self._terminalize(
                        ticket,
                        reason=TerminalReason.PROCESSING_HORIZON_EXPIRED,
                        outcome_class=OutcomeClass.EXPIRED_WORK,
                        stage=Stage.PENDING,
                        now_ns=now_ns,
                    )
                )
                continue
            _require(stream_id not in self._inflight, "stream already has inflight work")
            self._inflight[stream_id] = (ticket, int(now_ns))
            return TakeResult(ticket, tuple(feedback))
        return TakeResult(None, tuple(feedback))

    def stage_gate(
        self,
        ticket: FrameTicket,
        *,
        stage: Stage,
        now_ns: int,
        compute_spent_ns: int,
    ) -> TerminalFeedback | None:
        _require(
            stage in (Stage.BEFORE_DECODE, Stage.BEFORE_TAIL, Stage.BEFORE_PUBLICATION),
            "stage is not a cancellable boundary",
        )
        inflight = self._inflight.get(ticket.stream_id)
        _require(inflight is not None and inflight[0] == ticket, "frame not inflight")
        latest = self._latest[ticket.stream_id]
        if latest.sequence_id > ticket.sequence_id:
            return self._terminalize(
                ticket,
                reason=_SUPERSEDED_REASON[stage],
                outcome_class=OutcomeClass.INTENTIONAL_FRESHNESS_DROP,
                stage=stage,
                now_ns=now_ns,
                compute_spent_ns=compute_spent_ns,
                replacement=latest,
            )
        if int(now_ns) - ticket.capture_timestamp_ns > self.processing_horizon_ns:
            return self._terminalize(
                ticket,
                reason=TerminalReason.PROCESSING_HORIZON_EXPIRED,
                outcome_class=OutcomeClass.EXPIRED_WORK,
                stage=stage,
                now_ns=now_ns,
                compute_spent_ns=compute_spent_ns,
            )
        return None

    def installed(
        self,
        ticket: FrameTicket,
        *,
        now_ns: int,
        compute_spent_ns: int,
    ) -> TerminalFeedback:
        inflight = self._inflight.get(ticket.stream_id)
        _require(inflight is not None and inflight[0] == ticket, "frame not inflight")
        _require(
            self._latest[ticket.stream_id].sequence_id == ticket.sequence_id,
            "newer frame exists; supersession gate is required before install",
        )
        return self._terminalize(
            ticket,
            reason=TerminalReason.MAP_INSTALLED,
            outcome_class=OutcomeClass.INSTALLED_UTILITY,
            stage=Stage.MAP_INSTALL,
            now_ns=now_ns,
            compute_spent_ns=compute_spent_ns,
        )

    @property
    def pending_depth(self) -> int:
        return len(self._pending)


class OutcomeAccounting:
    """Reconcile frame-level outcomes without conflating scheduling and loss."""

    def __init__(self) -> None:
        self._outcomes: dict[tuple[str, str, str, int, int], TerminalFeedback] = {}

    def add(self, feedback: TerminalFeedback) -> None:
        _require(feedback.ticket.identity not in self._outcomes, "duplicate outcome")
        self._outcomes[feedback.ticket.identity] = feedback

    def summary(self) -> dict[str, Any]:
        values = tuple(self._outcomes.values())
        installed = sum(value.reason is TerminalReason.MAP_INSTALLED for value in values)
        superseded = sum(
            value.outcome_class is OutcomeClass.INTENTIONAL_FRESHNESS_DROP
            for value in values
        )
        return {
            "terminal_frames": len(values),
            "installed_frames": installed,
            "intentional_freshness_drops": superseded,
            "expired_frames": sum(
                value.outcome_class is OutcomeClass.EXPIRED_WORK for value in values
            ),
            "transport_failures": sum(
                value.outcome_class is OutcomeClass.TRANSPORT_FAILURE
                for value in values
            ),
            "structural_failures": sum(
                value.outcome_class is OutcomeClass.STRUCTURAL_FAILURE
                for value in values
            ),
            "raw_installed_over_terminal": (
                None if not values else installed / len(values)
            ),
            "feature_bytes_charged": sum(
                value.agent_credit().charge_feature_bytes for value in values
            ),
            "wasted_feature_bytes": sum(
                value.agent_credit().wasted_feature_bytes for value in values
            ),
        }

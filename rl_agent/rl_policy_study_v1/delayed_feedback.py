"""Causal tail-feedback accounting for the first recurrent-PPO experiment.

This is an offline-tested contract skeleton. It does not open sockets or change
the deployed UE/edge runtime. Version ``tail_only_v1`` ends the control reward
at compact model-tail feedback received by the UE; spatial-map outcomes remain
an independent audit stream.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Iterable


TAIL_ONLY_V1_DEADLINE_MS = 140.0
RECENT_OUTCOME_WINDOW = 16


class FeedbackContractError(RuntimeError):
    pass


class TerminalOutcome(str, Enum):
    TAIL_COMPLETED_ACK_RECEIVED = "TAIL_COMPLETED_ACK_RECEIVED"
    SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
    REASSEMBLY_FAILED = "REASSEMBLY_FAILED"
    STALE_BEFORE_EDGE = "STALE_BEFORE_EDGE"
    ACTION_EXECUTION_FAILED = "ACTION_EXECUTION_FAILED"
    FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION = (
        "FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION"
    )


class QualitySource(str, Enum):
    FROZEN_ACTION_VALIDATION = "FROZEN_ACTION_VALIDATION"
    QUALIFIED_LIVE_PROXY = "QUALIFIED_LIVE_PROXY"
    LABELLED_ENVIRONMENT = "LABELLED_ENVIRONMENT"


@dataclass(frozen=True, order=True)
class TicketKey:
    session_id: str
    ue_id: str
    frame_id: int
    action_id: int


@dataclass(frozen=True)
class PendingTicket:
    key: TicketKey
    decision_epoch: int
    opened_at_ms: float
    service_started_at_ms: float
    previous_action_id: int | None
    segmentation_quality_anchor: float
    localization_quality_anchor: float
    quality_source: QualitySource
    quality_version: str
    quality_catalog_sha256: str
    payload_bytes: int
    compute_ms_charged: float
    old_log_probability: float
    old_value: float
    incoming_lstm_h: tuple[float, ...]
    incoming_lstm_c: tuple[float, ...]
    deadline_ms: float = TAIL_ONLY_V1_DEADLINE_MS


@dataclass(frozen=True)
class TerminalFeedback:
    key: TicketKey
    outcome: TerminalOutcome
    observed_at_ms: float
    payload_bytes_charged: int
    compute_ms_charged: float
    resource_charges_observed: bool = True
    feedback_latency_ms: float | None = None
    deadline_ms: float | None = None
    deadline_met: bool | None = None
    segmentation_quality: float | None = None
    localization_quality: float | None = None
    quality_source: QualitySource | None = None
    quality_version: str | None = None
    quality_catalog_sha256: str | None = None
    quality_available: bool = False
    replacement_frame_id: int | None = None
    cumulative_terminal_frame_watermark: int | None = None


@dataclass(frozen=True)
class RewardWeights:
    """Candidate tail-only scalar reward coefficients.

    Payload and compute are represented by named cost critics in the v1 model,
    so their scalar coefficients default to zero to avoid optimizing the same
    cost twice.  Non-zero values remain available for an explicitly registered
    scalar-only ablation.
    """

    weight_segmentation: float = 0.35
    weight_localization: float = 0.35
    weight_latency: float = 0.30
    beta_deadline_miss: float = 0.50
    beta_transport_failure: float = 1.0
    beta_bytes: float = 0.0
    beta_compute: float = 0.0
    beta_action_switch: float = 0.0
    latency_tau_ms: float = 140.0
    maximum_payload_bytes: int = 1
    maximum_compute_ms: float = 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeedbackContractError(message)


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _tail_feedback_fields(feedback: TerminalFeedback) -> tuple[float, float, float]:
    _require(feedback.feedback_latency_ms is not None, "feedback latency is absent")
    _require(
        type(feedback.quality_available) is bool,
        "quality availability must be boolean",
    )
    _require(feedback.quality_available, "tail quality is marked unavailable")
    _require(feedback.segmentation_quality is not None, "segmentation quality is absent")
    _require(feedback.localization_quality is not None, "localization quality is absent")
    _require(
        isinstance(feedback.quality_source, QualitySource),
        "quality provenance is absent or unregistered",
    )
    _require(
        feedback.quality_source
        in (
            QualitySource.FROZEN_ACTION_VALIDATION,
            QualitySource.QUALIFIED_LIVE_PROXY,
        ),
        "labelled-environment quality belongs to the evaluator correction stream",
    )
    _require(_is_real(feedback.feedback_latency_ms), "feedback latency is not numeric")
    _require(_is_real(feedback.segmentation_quality), "segmentation quality is not numeric")
    _require(_is_real(feedback.localization_quality), "localization quality is not numeric")
    latency = float(feedback.feedback_latency_ms)
    segmentation = float(feedback.segmentation_quality)
    localization = float(feedback.localization_quality)
    _require(math.isfinite(latency) and latency >= 0.0, "invalid feedback latency")
    _require(
        math.isfinite(segmentation) and 0.0 <= segmentation <= 1.0,
        "invalid segmentation quality",
    )
    _require(
        math.isfinite(localization) and 0.0 <= localization <= 1.0,
        "invalid localization quality",
    )
    return latency, segmentation, localization


def _validate_tail_against_ticket(
    ticket: PendingTicket,
    feedback: TerminalFeedback,
) -> tuple[float, float, float]:
    latency, segmentation, localization = _tail_feedback_fields(feedback)
    _require(feedback.deadline_ms is not None, "feedback deadline is absent")
    _require(feedback.deadline_met is not None, "feedback deadline result is absent")
    _require(_is_real(feedback.deadline_ms), "feedback deadline is not numeric")
    _require(type(feedback.deadline_met) is bool, "deadline result must be boolean")
    _require(
        math.isclose(
            float(feedback.deadline_ms),
            ticket.deadline_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "feedback deadline disagrees with ticket",
    )
    _require(
        bool(feedback.deadline_met) == (latency <= ticket.deadline_ms),
        "feedback deadline result disagrees with latency",
    )
    _require(
        feedback.quality_source is ticket.quality_source,
        "feedback quality source disagrees with ticket",
    )
    _require(
        feedback.quality_version == ticket.quality_version,
        "feedback quality version disagrees with ticket",
    )
    _require(
        feedback.quality_catalog_sha256 == ticket.quality_catalog_sha256,
        "feedback quality catalog disagrees with ticket",
    )
    if ticket.quality_source is QualitySource.FROZEN_ACTION_VALIDATION:
        _require(
            math.isclose(
                segmentation,
                ticket.segmentation_quality_anchor,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "feedback segmentation anchor disagrees with ticket",
        )
        _require(
            math.isclose(
                localization,
                ticket.localization_quality_anchor,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "feedback localization anchor disagrees with ticket",
        )
    expected_latency = feedback.observed_at_ms - ticket.service_started_at_ms
    _require(
        math.isclose(latency, expected_latency, rel_tol=0.0, abs_tol=1e-6),
        "feedback latency disagrees with same-clock UE timestamps",
    )
    return latency, segmentation, localization


def _validate_resource_charges(feedback: TerminalFeedback) -> None:
    _require(
        type(feedback.resource_charges_observed) is bool,
        "resource-charge provenance must be boolean",
    )
    _require(
        isinstance(feedback.payload_bytes_charged, int)
        and not isinstance(feedback.payload_bytes_charged, bool)
        and feedback.payload_bytes_charged >= 0,
        "invalid charged payload bytes",
    )
    _require(
        _is_real(feedback.compute_ms_charged)
        and
        math.isfinite(feedback.compute_ms_charged)
        and feedback.compute_ms_charged >= 0.0,
        "invalid charged compute",
    )


def transition_reward(
    ticket: PendingTicket,
    feedback: TerminalFeedback,
    weights: RewardWeights,
) -> float:
    """Calculate the ``tail_only_v1`` reward for one reconciled ticket."""

    _require(ticket.key == feedback.key, "feedback identity does not match ticket")
    _require(isinstance(feedback.outcome, TerminalOutcome), "unregistered outcome")
    _require(
        feedback.outcome is not TerminalOutcome.ACTION_EXECUTION_FAILED,
        "structural action-execution failures terminate/exclude the episode; "
        "they are not reward samples",
    )
    _validate_resource_charges(feedback)
    _require(ticket.deadline_ms > 0.0, "deadline must be positive")
    _require(weights.latency_tau_ms > 0.0, "latency tau must be positive")
    _require(weights.maximum_payload_bytes > 0, "payload normalizer must be positive")
    _require(weights.maximum_compute_ms > 0.0, "compute normalizer must be positive")
    coefficients = (
        weights.weight_segmentation,
        weights.weight_localization,
        weights.weight_latency,
        weights.beta_deadline_miss,
        weights.beta_transport_failure,
        weights.beta_bytes,
        weights.beta_compute,
        weights.beta_action_switch,
    )
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in coefficients),
        "reward weights must be finite and non-negative",
    )
    _require(
        math.isclose(
            weights.weight_segmentation
            + weights.weight_localization
            + weights.weight_latency,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "primary reward weights must sum to one",
    )

    elapsed_to_feedback = feedback.observed_at_ms - ticket.service_started_at_ms
    _require(
        math.isfinite(elapsed_to_feedback) and elapsed_to_feedback >= 0.0,
        "feedback predates service start",
    )
    utility = 0.0
    deadline_penalty = (
        weights.beta_deadline_miss
        if elapsed_to_feedback > ticket.deadline_ms
        else 0.0
    )
    if feedback.outcome is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED:
        latency, segmentation, localization = _validate_tail_against_ticket(
            ticket, feedback
        )
        latency_cost = 1.0 - math.exp(-latency / weights.latency_tau_ms)
        utility = (
            weights.weight_segmentation * segmentation
            + weights.weight_localization * localization
            - weights.weight_latency * latency_cost
        )
    transport_penalty = (
        weights.beta_transport_failure
        if feedback.outcome is TerminalOutcome.REASSEMBLY_FAILED
        else 0.0
    )
    byte_penalty = weights.beta_bytes * (
        feedback.payload_bytes_charged / weights.maximum_payload_bytes
    )
    compute_penalty = weights.beta_compute * (
        feedback.compute_ms_charged / weights.maximum_compute_ms
    )
    switch_penalty = (
        weights.beta_action_switch
        if ticket.previous_action_id is not None
        and int(ticket.previous_action_id) != ticket.key.action_id
        else 0.0
    )
    return (
        utility
        - deadline_penalty
        - transport_penalty
        - byte_penalty
        - compute_penalty
        - switch_penalty
    )


class PendingTransitionLedger:
    """Identity-safe tail-feedback ledger with contiguous PPO emission."""

    def __init__(self) -> None:
        self._open: dict[TicketKey, PendingTicket] = {}
        self._closed: dict[TicketKey, tuple[PendingTicket, TerminalFeedback]] = {}
        self._order: list[TicketKey] = []
        self._emitted = 0
        self._frame_index: set[tuple[str, str, int]] = set()
        self._latest_opened: dict[tuple[str, str], tuple[int, int, float]] = {}
        self._identity: tuple[str, str] | None = None

    def open(self, ticket: PendingTicket) -> None:
        key = ticket.key
        identity = (key.session_id, key.ue_id)
        frame_identity = (*identity, key.frame_id)
        _require(bool(key.session_id) and bool(key.ue_id), "empty session or UE identity")
        _require(
            self._identity is None or identity == self._identity,
            "use one pending-transition ledger per session and UE",
        )
        _require(
            isinstance(key.frame_id, int)
            and not isinstance(key.frame_id, bool)
            and isinstance(key.action_id, int)
            and not isinstance(key.action_id, bool)
            and key.frame_id >= 0
            and 0 <= key.action_id < 72,
            "invalid identity field",
        )
        _require(
            isinstance(ticket.decision_epoch, int)
            and not isinstance(ticket.decision_epoch, bool)
            and ticket.decision_epoch >= 0,
            "invalid decision epoch",
        )
        _require(
            ticket.previous_action_id is None
            or (
                isinstance(ticket.previous_action_id, int)
                and not isinstance(ticket.previous_action_id, bool)
                and 0 <= ticket.previous_action_id < 72
            ),
            "invalid previous action",
        )
        _require(
            _is_real(ticket.segmentation_quality_anchor)
            and math.isfinite(ticket.segmentation_quality_anchor)
            and 0.0 <= ticket.segmentation_quality_anchor <= 1.0,
            "invalid segmentation anchor",
        )
        _require(
            _is_real(ticket.localization_quality_anchor)
            and math.isfinite(ticket.localization_quality_anchor)
            and 0.0 <= ticket.localization_quality_anchor <= 1.0,
            "invalid localization anchor",
        )
        _require(isinstance(ticket.quality_source, QualitySource), "invalid quality source")
        _require(
            ticket.quality_source
            in (
                QualitySource.FROZEN_ACTION_VALIDATION,
                QualitySource.QUALIFIED_LIVE_PROXY,
            ),
            "ticket quality source is unavailable to the deployable tail contract",
        )
        _require(bool(ticket.quality_version), "empty quality version")
        _require(_is_sha256(ticket.quality_catalog_sha256), "invalid quality catalog hash")
        _require(
            isinstance(ticket.payload_bytes, int)
            and not isinstance(ticket.payload_bytes, bool)
            and ticket.payload_bytes >= 0,
            "invalid payload",
        )
        _require(
            _is_real(ticket.compute_ms_charged)
            and math.isfinite(ticket.compute_ms_charged)
            and ticket.compute_ms_charged >= 0.0,
            "invalid compute charge",
        )
        _require(
            _is_real(ticket.service_started_at_ms)
            and _is_real(ticket.opened_at_ms)
            and math.isfinite(ticket.service_started_at_ms)
            and math.isfinite(ticket.opened_at_ms)
            and ticket.opened_at_ms >= ticket.service_started_at_ms,
            "invalid service-start timestamp",
        )
        _require(
            _is_real(ticket.deadline_ms)
            and math.isfinite(ticket.deadline_ms)
            and ticket.deadline_ms > 0.0,
            "invalid deadline",
        )
        _require(
            math.isclose(
                ticket.deadline_ms,
                TAIL_ONLY_V1_DEADLINE_MS,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "tail_only_v1 requires the registered 140-ms deadline",
        )
        _require(
            _is_real(ticket.old_log_probability)
            and math.isfinite(ticket.old_log_probability),
            "invalid old log probability",
        )
        _require(
            _is_real(ticket.old_value) and math.isfinite(ticket.old_value),
            "invalid old value",
        )
        _require(
            len(ticket.incoming_lstm_h) > 0
            and len(ticket.incoming_lstm_h) == len(ticket.incoming_lstm_c)
            and all(_is_real(value) and math.isfinite(value) for value in ticket.incoming_lstm_h)
            and all(_is_real(value) and math.isfinite(value) for value in ticket.incoming_lstm_c),
            "invalid incoming LSTM state",
        )
        _require(key not in self._open and key not in self._closed, "duplicate ticket")
        _require(frame_identity not in self._frame_index, "multiple actions for one frame")
        previous = self._latest_opened.get(identity)
        if previous is not None:
            _require(
                ticket.decision_epoch > previous[0]
                and key.frame_id > previous[1]
                and ticket.opened_at_ms >= previous[2],
                "non-monotone per-UE decision order",
            )
        if self._identity is None:
            self._identity = identity
        self._open[key] = ticket
        self._order.append(key)
        self._frame_index.add(frame_identity)
        self._latest_opened[identity] = (
            ticket.decision_epoch,
            key.frame_id,
            ticket.opened_at_ms,
        )

    def close(self, feedback: TerminalFeedback) -> bool:
        """Close one service ticket; identical duplicate feedback is idempotent."""

        key = feedback.key
        if key in self._closed:
            _require(self._closed[key][1] == feedback, "conflicting duplicate feedback")
            return False
        _require(key in self._open, "feedback has no open ticket")
        ticket = self._open[key]
        _require(isinstance(feedback.outcome, TerminalOutcome), "unregistered outcome")
        _validate_resource_charges(feedback)
        _require(
            _is_real(feedback.observed_at_ms)
            and math.isfinite(feedback.observed_at_ms)
            and feedback.observed_at_ms >= ticket.opened_at_ms,
            "feedback predates action",
        )
        if feedback.outcome is TerminalOutcome.SUPERSEDED_PENDING:
            _require(
                feedback.replacement_frame_id is not None
                and int(feedback.replacement_frame_id) > key.frame_id,
                "supersession lacks a newer replacement frame",
            )
        if feedback.outcome is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED:
            _validate_tail_against_ticket(ticket, feedback)
        self._open.pop(key)
        self._closed[key] = (ticket, feedback)
        return True

    def close_many(self, feedback: Iterable[TerminalFeedback]) -> int:
        return sum(int(self.close(item)) for item in feedback)

    def expire_after_reconciliation(
        self,
        key: TicketKey,
        *,
        observed_at_ms: float,
        durable_ledger_proves_absent: bool,
    ) -> None:
        _require(
            durable_ledger_proves_absent,
            "missing feedback alone cannot expire a pending action",
        )
        _require(key in self._open, "reconciliation has no open ticket")
        ticket = self._open[key]
        self.close(
            TerminalFeedback(
                key=key,
                outcome=TerminalOutcome.FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION,
                observed_at_ms=observed_at_ms,
                payload_bytes_charged=ticket.payload_bytes,
                compute_ms_charged=ticket.compute_ms_charged,
                resource_charges_observed=False,
            )
        )

    def observation(
        self,
        *,
        session_id: str,
        ue_id: str,
        current_frame_id: int,
        now_ms: float,
    ) -> dict[str, float | int | bool]:
        """Build a causal observation using only events visible by ``now_ms``.

        The ledger deliberately supports historical cutoffs for deterministic
        replay.  A ticket that has closed in the present is reconstructed as
        pending when its terminal event occurred after the requested cutoff.
        """

        _require(current_frame_id >= 0, "negative current frame")
        _require(math.isfinite(now_ms), "invalid observation cutoff")
        identity = (session_id, ue_id)
        visible_pending = [
            item
            for item in self._open.values()
            if (item.key.session_id, item.key.ue_id) == identity
            and item.opened_at_ms <= now_ms
        ]
        visible_terminal: list[tuple[PendingTicket, TerminalFeedback]] = []
        for ticket, feedback in self._closed.values():
            if (ticket.key.session_id, ticket.key.ue_id) != identity:
                continue
            if ticket.opened_at_ms > now_ms:
                continue
            if feedback.observed_at_ms <= now_ms:
                visible_terminal.append((ticket, feedback))
            else:
                visible_pending.append(ticket)

        visible_tail = [
            (ticket, feedback)
            for ticket, feedback in visible_terminal
            if feedback.outcome is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED
        ]
        latest_pair = (
            max(
                visible_tail,
                key=lambda pair: (
                    pair[0].key.frame_id,
                    pair[1].observed_at_ms,
                ),
            )
            if visible_tail
            else None
        )
        latest_ticket = latest_pair[0] if latest_pair is not None else None
        latest = latest_pair[1] if latest_pair is not None else None
        last_terminal_pair = (
            max(
                visible_terminal,
                key=lambda pair: (
                    pair[1].observed_at_ms,
                    pair[0].key.frame_id,
                ),
            )
            if visible_terminal
            else None
        )
        last_terminal = (
            last_terminal_pair[1] if last_terminal_pair is not None else None
        )

        overdue_pending = [
            item
            for item in visible_pending
            if now_ms - item.service_started_at_ms > item.deadline_ms
        ]
        eligible_attempts: list[
            tuple[PendingTicket, TerminalFeedback | None]
        ] = list(visible_terminal) + [(item, None) for item in overdue_pending]
        recent = sorted(
            eligible_attempts,
            key=lambda pair: (
                pair[0].decision_epoch,
                pair[0].key.frame_id,
                pair[0].opened_at_ms,
            ),
        )[-RECENT_OUTCOME_WINDOW:]
        recent_count = len(recent)
        recent_terminal_count = sum(feedback is not None for _, feedback in recent)
        on_time_count = sum(
            1
            for _, feedback in recent
            if feedback is not None
            if feedback.outcome is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED
            and bool(feedback.deadline_met)
        )
        late_count = sum(
            1
            for _, feedback in recent
            if feedback is not None
            if feedback.outcome is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED
            and not bool(feedback.deadline_met)
        )
        superseded_count = sum(
            1
            for _, feedback in recent
            if feedback is not None
            if feedback.outcome is TerminalOutcome.SUPERSEDED_PENDING
        )
        overdue_count = recent_count - recent_terminal_count
        deadline_miss_count = overdue_count + sum(
            1
            for ticket, feedback in recent
            if feedback is not None
            and feedback.observed_at_ms - ticket.service_started_at_ms
            > ticket.deadline_ms
        )
        failure_count = (
            recent_terminal_count - on_time_count - late_count - superseded_count
        )
        denominator = float(recent_count) if recent_count else 1.0

        pending = visible_pending
        oldest = min(pending, key=lambda item: item.opened_at_ms) if pending else None
        missed = [
            item
            for item in pending
            if now_ms - item.service_started_at_ms > item.deadline_ms
        ]
        return {
            "has_tail_feedback": latest is not None,
            "latest_tail_frame_lag": (
                max(0, current_frame_id - latest_ticket.key.frame_id)
                if latest_ticket is not None
                else 0
            ),
            "time_since_latest_tail_feedback_ms": (
                max(0.0, now_ms - latest.observed_at_ms)
                if latest is not None
                else 0.0
            ),
            "has_terminal_event": last_terminal_pair is not None,
            "last_terminal_frame_lag": (
                max(0, current_frame_id - last_terminal_pair[0].key.frame_id)
                if last_terminal_pair is not None
                else 0
            ),
            "last_terminal_action_id": (
                last_terminal_pair[0].key.action_id
                if last_terminal_pair is not None
                else 0
            ),
            "time_since_last_terminal_event_ms": (
                max(0.0, now_ms - last_terminal_pair[1].observed_at_ms)
                if last_terminal_pair is not None
                else 0.0
            ),
            "last_terminal_is_on_time_tail": (
                last_terminal is not None
                and last_terminal.outcome
                is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED
                and bool(last_terminal.deadline_met)
            ),
            "last_terminal_is_late_tail": (
                last_terminal is not None
                and last_terminal.outcome
                is TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED
                and not bool(last_terminal.deadline_met)
            ),
            "last_terminal_is_superseded": (
                last_terminal is not None
                and last_terminal.outcome is TerminalOutcome.SUPERSEDED_PENDING
            ),
            "last_terminal_is_reassembly_failure": (
                last_terminal is not None
                and last_terminal.outcome is TerminalOutcome.REASSEMBLY_FAILED
            ),
            "last_terminal_is_stale_before_edge": (
                last_terminal is not None
                and last_terminal.outcome is TerminalOutcome.STALE_BEFORE_EDGE
            ),
            "last_terminal_is_action_failure": (
                last_terminal is not None
                and last_terminal.outcome is TerminalOutcome.ACTION_EXECUTION_FAILED
            ),
            "last_terminal_is_reconciled_expiry": (
                last_terminal is not None
                and last_terminal.outcome
                is TerminalOutcome.FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION
            ),
            "latest_tail_action_id": (
                latest_ticket.key.action_id if latest_ticket is not None else 0
            ),
            "latest_feedback_latency_ms": (
                latest.feedback_latency_ms if latest is not None else 0.0
            ),
            "latest_feedback_latency_ratio": (
                latest.feedback_latency_ms / latest.deadline_ms
                if latest is not None
                else 0.0
            ),
            "latest_segmentation_quality": (
                latest.segmentation_quality if latest is not None else 0.0
            ),
            "latest_localization_quality": (
                latest.localization_quality if latest is not None else 0.0
            ),
            "latest_quality_is_frozen_anchor": (
                latest is not None
                and latest.quality_source is QualitySource.FROZEN_ACTION_VALIDATION
            ),
            "latest_quality_is_live_proxy": (
                latest is not None
                and latest.quality_source is QualitySource.QUALIFIED_LIVE_PROXY
            ),
            "latest_quality_available": (
                bool(latest.quality_available) if latest is not None else False
            ),
            "latest_deadline_met": latest.deadline_met if latest is not None else False,
            "recent_attempt_count": recent_count,
            "recent_terminal_count": recent_terminal_count,
            "recent_on_time_tail_rate": on_time_count / denominator,
            "recent_late_tail_rate": late_count / denominator,
            "recent_superseded_rate": superseded_count / denominator,
            "recent_service_failure_rate": failure_count / denominator,
            "recent_deadline_missed_pending_rate": overdue_count / denominator,
            "recent_deadline_miss_rate": deadline_miss_count / denominator,
            "pending_count": len(pending),
            "deadline_missed_pending_count": len(missed),
            "oldest_pending_frame_lag": (
                current_frame_id - oldest.key.frame_id if oldest is not None else 0
            ),
            "oldest_pending_age_ms": (
                max(0.0, now_ms - oldest.opened_at_ms) if oldest is not None else 0.0
            ),
        }

    def terminal_prefix(self) -> list[tuple[PendingTicket, TerminalFeedback]]:
        """Return newly closed contiguous service transitions in decision order."""

        result: list[tuple[PendingTicket, TerminalFeedback]] = []
        while self._emitted < len(self._order):
            key = self._order[self._emitted]
            if key not in self._closed:
                break
            result.append(self._closed[key])
            self._emitted += 1
        return result

    @property
    def pending_count(self) -> int:
        return len(self._open)

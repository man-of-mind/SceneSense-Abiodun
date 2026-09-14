"""Causal pending-ticket accounting for recurrent PPO.

This module is an offline-tested contract skeleton. It does not open sockets or
change the deployed UE/edge runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class FeedbackContractError(RuntimeError):
    pass


class TerminalOutcome(str, Enum):
    MAP_INSTALLED = "MAP_INSTALLED"
    SUPERSEDED_PENDING = "SUPERSEDED_PENDING"
    REASSEMBLY_FAILED = "REASSEMBLY_FAILED"
    STALE_BEFORE_EDGE_OR_MAP = "STALE_BEFORE_EDGE_OR_MAP"
    ACTION_EXECUTION_FAILED = "ACTION_EXECUTION_FAILED"
    FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION = (
        "FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION"
    )


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
    capture_at_ms: float
    payload_bytes: int
    compute_ms_charged: float
    old_log_probability: float
    old_value: float
    incoming_lstm_h: tuple[float, ...]
    incoming_lstm_c: tuple[float, ...]


@dataclass(frozen=True)
class TerminalFeedback:
    key: TicketKey
    outcome: TerminalOutcome
    observed_at_ms: float
    install_latency_ms: float | None = None
    installed_quality: float | None = None
    replacement_frame_id: int | None = None
    cumulative_terminal_frame_watermark: int | None = None


@dataclass(frozen=True)
class RewardWeights:
    alpha_quality: float = 1.0
    beta_transport_failure: float = 1.0
    beta_bytes: float = 0.05
    beta_compute: float = 0.05
    beta_action_switch: float = 0.01
    latency_tau_ms: float = 250.0
    maximum_payload_bytes: int = 1
    maximum_compute_ms: float = 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeedbackContractError(message)


def transition_reward(
    ticket: PendingTicket,
    feedback: TerminalFeedback,
    weights: RewardWeights,
    *,
    previous_action_id: int | None,
) -> float:
    """Calculate reward only after a ticket has a proven terminal outcome."""

    _require(ticket.key == feedback.key, "feedback identity does not match ticket")
    _require(weights.latency_tau_ms > 0.0, "latency tau must be positive")
    _require(weights.maximum_payload_bytes > 0, "payload normalizer must be positive")
    _require(weights.maximum_compute_ms > 0.0, "compute normalizer must be positive")
    coefficients = (
        weights.alpha_quality,
        weights.beta_transport_failure,
        weights.beta_bytes,
        weights.beta_compute,
        weights.beta_action_switch,
    )
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in coefficients),
        "reward weights must be finite and non-negative",
    )
    utility = 0.0
    if feedback.outcome is TerminalOutcome.MAP_INSTALLED:
        _require(feedback.install_latency_ms is not None, "install latency is absent")
        _require(feedback.installed_quality is not None, "installed quality is absent")
        latency = float(feedback.install_latency_ms)
        quality = float(feedback.installed_quality)
        _require(math.isfinite(latency) and latency >= 0.0, "invalid install latency")
        _require(math.isfinite(quality) and 0.0 <= quality <= 1.0, "invalid quality")
        utility = weights.alpha_quality * quality * math.exp(
            -latency / weights.latency_tau_ms
        )
    transport_penalty = (
        weights.beta_transport_failure
        if feedback.outcome is TerminalOutcome.REASSEMBLY_FAILED
        else 0.0
    )
    byte_penalty = weights.beta_bytes * (
        ticket.payload_bytes / weights.maximum_payload_bytes
    )
    compute_penalty = weights.beta_compute * (
        ticket.compute_ms_charged / weights.maximum_compute_ms
    )
    switch_penalty = (
        weights.beta_action_switch
        if previous_action_id is not None
        and int(previous_action_id) != ticket.key.action_id
        else 0.0
    )
    return utility - transport_penalty - byte_penalty - compute_penalty - switch_penalty


class PendingTransitionLedger:
    """Identity-safe delayed-feedback ledger with contiguous PPO emission."""

    def __init__(self) -> None:
        self._open: dict[TicketKey, PendingTicket] = {}
        self._closed: dict[TicketKey, tuple[PendingTicket, TerminalFeedback]] = {}
        self._order: list[TicketKey] = []
        self._emitted = 0
        self._latest_installed: dict[tuple[str, str], tuple[int, float]] = {}

    def open(self, ticket: PendingTicket) -> None:
        key = ticket.key
        _require(key.frame_id >= 0 and key.action_id >= 0, "negative identity field")
        _require(ticket.decision_epoch >= 0, "negative decision epoch")
        _require(ticket.payload_bytes >= 0, "negative payload")
        _require(ticket.compute_ms_charged >= 0.0, "negative compute charge")
        _require(key not in self._open and key not in self._closed, "duplicate ticket")
        self._open[key] = ticket
        self._order.append(key)

    def close(self, feedback: TerminalFeedback) -> bool:
        """Close one ticket; return False for an identical duplicate feedback."""

        key = feedback.key
        if key in self._closed:
            _require(self._closed[key][1] == feedback, "conflicting duplicate feedback")
            return False
        _require(key in self._open, "feedback has no open ticket")
        ticket = self._open.pop(key)
        _require(
            feedback.observed_at_ms >= ticket.opened_at_ms,
            "feedback predates action",
        )
        if feedback.outcome is TerminalOutcome.SUPERSEDED_PENDING:
            _require(
                feedback.replacement_frame_id is not None
                and int(feedback.replacement_frame_id) > key.frame_id,
                "supersession lacks a newer replacement frame",
            )
        if feedback.outcome is TerminalOutcome.MAP_INSTALLED:
            _require(feedback.install_latency_ms is not None, "install latency absent")
            identity = (key.session_id, key.ue_id)
            previous = self._latest_installed.get(identity)
            if previous is None or key.frame_id > previous[0]:
                self._latest_installed[identity] = (
                    key.frame_id,
                    feedback.observed_at_ms,
                )
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
        self.close(
            TerminalFeedback(
                key=key,
                outcome=TerminalOutcome.FEEDBACK_HORIZON_EXPIRED_AFTER_RECONCILIATION,
                observed_at_ms=observed_at_ms,
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
        identity = (session_id, ue_id)
        pending = [
            ticket
            for ticket in self._open.values()
            if (ticket.key.session_id, ticket.key.ue_id) == identity
        ]
        latest = self._latest_installed.get(identity)
        oldest = min(pending, key=lambda item: item.opened_at_ms) if pending else None
        return {
            "has_installed_frame": latest is not None,
            "latest_installed_frame_lag": (
                current_frame_id - latest[0] if latest is not None else current_frame_id + 1
            ),
            "time_since_latest_install_ms": (
                max(0.0, now_ms - latest[1]) if latest is not None else now_ms
            ),
            "pending_count": len(pending),
            "oldest_pending_frame_lag": (
                current_frame_id - oldest.key.frame_id if oldest is not None else 0
            ),
            "oldest_pending_age_ms": (
                max(0.0, now_ms - oldest.opened_at_ms) if oldest is not None else 0.0
            ),
        }

    def terminal_prefix(self) -> list[tuple[PendingTicket, TerminalFeedback]]:
        """Return newly closed contiguous transitions in decision order."""

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

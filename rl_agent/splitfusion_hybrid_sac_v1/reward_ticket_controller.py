"""Deterministic one-ticket action-hold and reward-feedback state machine.

Phase 3b of the SplitFusion conditional Hybrid-SAC foundation.  This module
implements the *control contract only* of DESIGN.md section 3 ("Runtime control
contract"), the terminal-classification rules of section 8, and the identity
rules of section 9.  It is a state machine over already-frozen identity
records; it computes nothing measured.

Frozen runtime semantics
------------------------
* Exactly **one** outstanding reward ticket per controller/session.
* ``B = 200_000_000 ns`` (200 ms), measured on an injected **monotonic**
  nanosecond clock from ticket opening (the policy decision) to exact feedback
  receipt at the UE.  The boundary is inclusive: receipt at ``opened_ns + B`` is
  timely; ``opened_ns + B + 1`` is not.
* ``k_min = 2`` transmitted tensors per selected action.  The first tensor of a
  decision carries ``reward_requested=True``; every later tensor governed by
  that decision reuses the *exact* reconciled
  :class:`~.transaction_identity.ExecutedActionIdentity` with
  ``reward_requested=False``.
* Valid feedback arriving before tensor 2 resolves the ticket but does **not**
  release the gate: the ticket sits in ``RESOLVED_WAITING_MIN_HOLD`` until
  tensor 2 has reused the action.  The next decision may then open only on a
  *future* frame -- never retroactively on a frame already admitted under the
  old action.
* An unresolved ticket past the deadline closes as ``FEEDBACK_TIMEOUT``,
  subject to the same minimum-two-tensor hold
  (``TIMED_OUT_WAITING_MIN_HOLD``).  Later matching feedback is
  ``LATE_ORPHAN`` and can never attach to a newer decision.
* A byte/field-identical duplicate of already accepted feedback is
  ``DUPLICATE_IGNORED``.  A conflicting duplicate, wrong session, wrong
  ``decision_seq``, wrong reward ``tensor_seq``, wrong ``carla_frame_id``,
  wrong action identity, or feedback naming a non-reward-requested tensor
  **fails closed** and neither closes nor mutates the current ticket.
* ``tensor_seq`` is the only sender chronology.  ``carla_frame_id`` is
  validation metadata, never an ordering key; arrival order, input order and
  timestamps are never used to join.  No sequence number is ever silently
  allocated: the caller supplies ``decision_seq`` and ``tensor_seq`` and they
  are validated strictly monotonically.
* The realized hold duration ``d`` is retained as the number of transmitted
  tensors governed by the decision.  No scalar reward and no ``gamma**d`` is
  computed here; both belong to the later SMDP/replay layer.

Scope
-----
State machine and terminal classification only.  This module does **not**
implement SAC, actor/critic networks, the scalar reward of DESIGN.md section 8,
replay storage or its schema, the causal state vector, scene descriptors, or
any CARLA/OAI/UDP/Docker/CUDA integration.  It never converts an
infrastructure/evaluator fault into a penalty: it only labels it.

Feedback-only control-plane loss (DESIGN.md section 8 case 2) cannot be
distinguished at runtime from a true missing terminal event, so a deadline
expiry is classified ``FEEDBACK_TIMEOUT`` and carries the learning disposition
``censored_pending_post_run_reconciliation``.  The adjudication itself is
explicitly out of scope for this phase.

Concurrency
-----------
:class:`RewardTicketController` is **not thread-safe**.  It holds mutable gate
state and enforces a single-outstanding-ticket invariant plus strictly
increasing ``decision_seq``/``tensor_seq``/clock, none of which is defended by a
lock.  Two concurrent callers can interleave a frame admission with a feedback
submission and observe a state the declared transition table never permits.

Future runtime integration must therefore **serialize every event through one
event loop or queue**: frame admission, feedback receipt, deadline/clock
observation and infrastructure-fault recording must all be delivered from a
single thread of control.  Broad locking is deliberately *not* added in this
phase: the correct boundary is the runtime's event loop, and adding locks here
would only hide a concurrent caller that is already violating the contract.

Importing this module performs no filesystem access, mutation or other runtime
side effect.  The schema hash below is computed from in-module literals.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .action_contract import CATALOG_SCHEMA, CATALOG_SHA256, EXECUTION_MODE
from .transaction_identity import (
    ACTION_IDENTITY_SCHEMA_ID,
    ACTION_IDENTITY_SCHEMA_SHA256,
    ActionHoldManifest,
    ExecutedActionIdentity,
    MINIMUM_HOLD_TENSORS,
    RewardFeedbackIdentity,
    SCHEMA_ID as TRANSACTION_SCHEMA_ID,
    SCHEMA_SHA256 as TRANSACTION_SCHEMA_SHA256,
    SCHEMA_VERSION as TRANSACTION_SCHEMA_VERSION,
    TensorTransactionId,
    TensorTransmissionEnvelope,
    TransactionIdentityError,
    canonical_json_bytes,
    canonical_sha256,
)

__all__ = [
    "RewardTicketControllerError",
    "ClockRegressionError",
    "GateUnavailableError",
    "NoOpenTicketError",
    "SequenceOrderError",
    "FrameReadmissionError",
    "IllegalTransitionError",
    "B_REWARD_DEADLINE_NS",
    "K_MIN_TENSORS",
    "CONTROLLER_SCHEMA_DESCRIPTOR",
    "CONTROLLER_SCHEMA_ID",
    "CONTROLLER_SCHEMA_SHA256",
    "CONTROLLER_SCHEMA_VERSION",
    "TERMINAL_LEARNING_DISPOSITION",
    "ControllerState",
    "TicketEvent",
    "AdmissionDisposition",
    "FeedbackDisposition",
    "FeedbackTerminalStatus",
    "TerminalClass",
    "RewardFeedbackMessage",
    "AdmissionResult",
    "FeedbackOutcome",
    "CompletedTicket",
    "GateStatus",
    "RewardTicketController",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class RewardTicketControllerError(TransactionIdentityError):
    """Base class for every reward-ticket control-contract violation."""


class ClockRegressionError(RewardTicketControllerError):
    """The injected clock moved backwards, or is not an exact non-negative int.

    The deadline is defined on a monotonic clock, so a regression is an
    infrastructure/programming fault.  It is raised, never folded into a
    terminal classification and never converted into a penalty.
    """


class GateUnavailableError(RewardTicketControllerError):
    """A new decision was attempted while a reward ticket was still held."""


class NoOpenTicketError(RewardTicketControllerError):
    """A held-action reuse or terminal event was attempted with no open ticket."""


class SequenceOrderError(RewardTicketControllerError):
    """A supplied ``decision_seq``/``tensor_seq`` violates the frozen ordering."""


class FrameReadmissionError(RewardTicketControllerError):
    """A ``carla_frame_id`` already admitted under a retained decision recurred.

    A decision may only ever open on a *future* frame, so a frame already
    governed by the active ticket -- or by any ticket still inside the bounded
    terminal history -- can never be re-admitted.
    """


class IllegalTransitionError(RewardTicketControllerError):
    """An event is not a declared transition out of the current state."""


# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #

#: DESIGN.md section 2 reward deadline, in nanoseconds: ``B = 200 ms`` at UE
#: receipt of exact ``REWARD_FINAL``.  Inclusive boundary.
B_REWARD_DEADLINE_NS: int = 200_000_000

#: DESIGN.md section 2 minimum hold ``k_min``, bound to the Phase-2 constant so
#: the two contracts cannot drift apart.
K_MIN_TENSORS: int = MINIMUM_HOLD_TENSORS

if K_MIN_TENSORS != 2:  # pragma: no cover - guards a dependency edit
    raise RewardTicketControllerError(
        f"the frozen minimum hold is k_min=2; the Phase-2 contract now "
        f"declares {K_MIN_TENSORS}"
    )


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ControllerState(Enum):
    """The one-ticket gate state.

    ``READY`` and ``CLOSED`` are the two gate-available states; they differ only
    in whether any ticket has already completed, so ``CLOSED`` is the observable
    resting state of the just-closed ticket and is retained until the next
    decision opens.  The three ``*_WAITING_MIN_HOLD`` states are terminal-pending:
    the ticket's terminal classification is already decided, but ``k_min = 2``
    tensors have not yet been transmitted, so the gate stays closed.
    """

    READY = "READY"
    OPEN_UNRESOLVED = "OPEN_UNRESOLVED"
    RESOLVED_WAITING_MIN_HOLD = "RESOLVED_WAITING_MIN_HOLD"
    TIMED_OUT_WAITING_MIN_HOLD = "TIMED_OUT_WAITING_MIN_HOLD"
    FAULTED_WAITING_MIN_HOLD = "FAULTED_WAITING_MIN_HOLD"
    CLOSED = "CLOSED"


#: The gate-available states: a new decision may open only from one of these.
GATE_AVAILABLE_STATES: Tuple[ControllerState, ...] = (
    ControllerState.READY,
    ControllerState.CLOSED,
)

#: States in which a ticket exists and its terminal class is already decided.
TERMINAL_PENDING_STATES: Tuple[ControllerState, ...] = (
    ControllerState.RESOLVED_WAITING_MIN_HOLD,
    ControllerState.TIMED_OUT_WAITING_MIN_HOLD,
    ControllerState.FAULTED_WAITING_MIN_HOLD,
)


class TicketEvent(Enum):
    """The declared events that may mutate the gate state."""

    OPEN_DECISION = "OPEN_DECISION"
    REUSE_TENSOR = "REUSE_TENSOR"
    ACCEPT_FEEDBACK = "ACCEPT_FEEDBACK"
    DEADLINE_EXPIRY = "DEADLINE_EXPIRY"
    INFRASTRUCTURE_FAULT = "INFRASTRUCTURE_FAULT"


class AdmissionDisposition(Enum):
    """What the gate did with one prepared frame."""

    OPENED_NEW_DECISION = "OPENED_NEW_DECISION"
    REUSED_HELD_ACTION = "REUSED_HELD_ACTION"


class FeedbackTerminalStatus(Enum):
    """The terminal status carried by an exact feedback message.

    ``REWARD_FINAL`` is the section-3 exact terminal event.
    ``ACTION_PATH_FAILURE`` is the section-8 case 1 *proven* feature-delivery,
    decode or tail failure, which is an action-path outcome rather than an
    instrument fault.
    """

    REWARD_FINAL = "REWARD_FINAL"
    ACTION_PATH_FAILURE = "ACTION_PATH_FAILURE"


class TerminalClass(Enum):
    """The single documented terminal classification of a completed ticket.

    DESIGN.md section 8 requires exactly one classification per transition, and
    requires instrument faults to stay distinguishable from action-path failure
    and from a missing terminal event.
    """

    REWARD_FINAL_EXACT = "REWARD_FINAL_EXACT"
    ACTION_PATH_FAILURE = "ACTION_PATH_FAILURE"
    FEEDBACK_TIMEOUT = "FEEDBACK_TIMEOUT"
    INFRASTRUCTURE_FAULT_EXCLUDED = "INFRASTRUCTURE_FAULT_EXCLUDED"


class FeedbackDisposition(Enum):
    """The exact classification of one submitted terminal event.

    Every ``REJECTED_*`` disposition fails closed: it neither closes nor
    mutates any ticket.
    """

    ACCEPTED_RESOLVED = "ACCEPTED_RESOLVED"
    ACCEPTED_INFRASTRUCTURE_FAULT = "ACCEPTED_INFRASTRUCTURE_FAULT"
    DUPLICATE_IGNORED = "DUPLICATE_IGNORED"
    LATE_ORPHAN = "LATE_ORPHAN"
    REJECTED_WRONG_SESSION = "REJECTED_WRONG_SESSION"
    REJECTED_UNKNOWN_DECISION = "REJECTED_UNKNOWN_DECISION"
    REJECTED_WRONG_REWARD_TENSOR_SEQ = "REJECTED_WRONG_REWARD_TENSOR_SEQ"
    REJECTED_NON_REWARD_TENSOR = "REJECTED_NON_REWARD_TENSOR"
    REJECTED_WRONG_CARLA_FRAME = "REJECTED_WRONG_CARLA_FRAME"
    REJECTED_ACTION_MISMATCH = "REJECTED_ACTION_MISMATCH"
    REJECTED_CONFLICTING_DUPLICATE = "REJECTED_CONFLICTING_DUPLICATE"


#: How the later learning layer must treat each terminal class.  These are
#: *labels*, not values: this module computes no scalar reward.
TERMINAL_LEARNING_DISPOSITION: Mapping[TerminalClass, str] = MappingProxyType(
    {
        TerminalClass.REWARD_FINAL_EXACT: "included",
        TerminalClass.ACTION_PATH_FAILURE: (
            "included_registered_negative_service_reward"
        ),
        TerminalClass.FEEDBACK_TIMEOUT: (
            "censored_pending_post_run_reconciliation"
        ),
        TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED: (
            "excluded_reported_as_experimental_failure"
        ),
    }
)

#: The terminal classes reachable only by accepting exact feedback.  Their
#: complement -- FEEDBACK_TIMEOUT and INFRASTRUCTURE_FAULT_EXCLUDED -- closes a
#: ticket with no accepted feedback at all.
_FEEDBACK_TERMINAL_CLASSES: Tuple[TerminalClass, ...] = (
    TerminalClass.REWARD_FINAL_EXACT,
    TerminalClass.ACTION_PATH_FAILURE,
)

_FEEDBACK_TERMINAL_CLASS: Mapping[FeedbackTerminalStatus, TerminalClass] = (
    MappingProxyType(
        {
            FeedbackTerminalStatus.REWARD_FINAL: TerminalClass.REWARD_FINAL_EXACT,
            FeedbackTerminalStatus.ACTION_PATH_FAILURE: (
                TerminalClass.ACTION_PATH_FAILURE
            ),
        }
    )
)


# --------------------------------------------------------------------------- #
# Declared transition table: anything absent fails closed
# --------------------------------------------------------------------------- #

_S = ControllerState
_E = TicketEvent

#: The complete set of legal ``(state, event) -> allowed next states``
#: transitions.  ``_transition`` refuses any pair or target absent from this
#: table, so an impossible transition raises instead of silently reshaping the
#: gate.  The branch inside a two-target entry is decided solely by whether the
#: hold has already reached ``k_min`` tensors.
_ALLOWED_TRANSITIONS: Mapping[
    Tuple[ControllerState, TicketEvent], Tuple[ControllerState, ...]
] = MappingProxyType(
    {
        (_S.READY, _E.OPEN_DECISION): (_S.OPEN_UNRESOLVED,),
        (_S.CLOSED, _E.OPEN_DECISION): (_S.OPEN_UNRESOLVED,),
        (_S.OPEN_UNRESOLVED, _E.REUSE_TENSOR): (_S.OPEN_UNRESOLVED,),
        (_S.RESOLVED_WAITING_MIN_HOLD, _E.REUSE_TENSOR): (_S.CLOSED,),
        (_S.TIMED_OUT_WAITING_MIN_HOLD, _E.REUSE_TENSOR): (_S.CLOSED,),
        (_S.FAULTED_WAITING_MIN_HOLD, _E.REUSE_TENSOR): (_S.CLOSED,),
        (_S.OPEN_UNRESOLVED, _E.ACCEPT_FEEDBACK): (
            _S.CLOSED,
            _S.RESOLVED_WAITING_MIN_HOLD,
        ),
        (_S.OPEN_UNRESOLVED, _E.DEADLINE_EXPIRY): (
            _S.CLOSED,
            _S.TIMED_OUT_WAITING_MIN_HOLD,
        ),
        (_S.OPEN_UNRESOLVED, _E.INFRASTRUCTURE_FAULT): (
            _S.CLOSED,
            _S.FAULTED_WAITING_MIN_HOLD,
        ),
    }
)


# --------------------------------------------------------------------------- #
# Canonical schema descriptor
# --------------------------------------------------------------------------- #


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze a literal into read-only mappings and tuples.

    Written locally rather than imported from the Phase-2 module: that module is
    a frozen dependency and its freezer is private, so depending on it here
    would couple this phase to a non-public name.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _declared_transition_literal() -> Dict[str, Tuple[str, ...]]:
    """Render the transition table as a canonical JSON-able literal."""
    return {
        f"{state.value}|{event.value}": tuple(
            target.value for target in targets
        )
        for (state, event), targets in _ALLOWED_TRANSITIONS.items()
    }


#: Semantic descriptor of the Phase-3b control contract.  Pure literal: it
#: states semantics and frozen constants, binds the exact schema ids/hashes of
#: both dependencies, and embeds the declared transition table.  Its hash
#: therefore changes only when the control contract or one of its dependencies
#: changes.
CONTROLLER_SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": "splitfusion_hybrid_sac_reward_ticket_controller_v1",
        "version": 2,
        "revision_note": (
            "v2 (phase 3b.1) corrects min_hold_satisfied_ns to the admission "
            "that first reached k_min rather than the closure instant, makes "
            "frame non-readmission absolute over the session lifetime rather "
            "than only over the retained terminal history, and hardens the "
            "completed-ticket timestamp/terminal-consistency invariants"
        ),
        "phase": (
            "one-ticket action-hold and feedback state machine only; no reward, "
            "state vector, replay storage, SAC or live integration"
        ),
        "design_reference": (
            "DESIGN.md section 3 runtime control contract, section 8 terminal "
            "accounting, section 9 transaction identity"
        ),
        "constants": {
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "minimum_hold_tensors": K_MIN_TENSORS,
            "outstanding_tickets_max": 1,
            "deadline_boundary": (
                "receipt at opened_ns + reward_deadline_ns is timely; "
                "opened_ns + reward_deadline_ns + 1 is a timeout"
            ),
            "clock": (
                "injected monotonic nanosecond clock; a regression is an "
                "infrastructure fault and is raised, never classified"
            ),
            "duration_semantics": (
                "d is the number of transmitted tensors governed by the "
                "decision; it is retained only, and no scalar reward or "
                "gamma**d is computed in this phase"
            ),
        },
        "dependencies": {
            "action_catalog": {
                "schema": CATALOG_SCHEMA,
                "sha256": CATALOG_SHA256,
                "execution_mode": EXECUTION_MODE,
            },
            "executed_action_identity": {
                "schema_id": ACTION_IDENTITY_SCHEMA_ID,
                "schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
            },
            "transaction_identity": {
                "schema_id": TRANSACTION_SCHEMA_ID,
                "schema_sha256": TRANSACTION_SCHEMA_SHA256,
                "schema_version": TRANSACTION_SCHEMA_VERSION,
            },
        },
        "chronology": (
            "tensor_seq is the only sender chronology; carla_frame_id is "
            "validation metadata, and arrival order, input order and "
            "timestamps are never used to order or to join"
        ),
        "sequence_allocation": (
            "the caller supplies decision_seq and tensor_seq; both are "
            "validated strictly increasing and are never silently allocated, "
            "nearest-matched or inferred"
        ),
        "serializable_boundary": (
            "every serialized record embeds a catalog-reconciled executed "
            "action identity and fails closed otherwise"
        ),
        "states": tuple(state.value for state in ControllerState),
        "gate_available_states": tuple(
            state.value for state in GATE_AVAILABLE_STATES
        ),
        "terminal_pending_states": tuple(
            state.value for state in TERMINAL_PENDING_STATES
        ),
        "events": tuple(event.value for event in TicketEvent),
        "transitions": _declared_transition_literal(),
        "transition_rule": (
            "any (state, event) pair or target absent from the declared table "
            "raises IllegalTransitionError"
        ),
        "admission_dispositions": tuple(
            disposition.value for disposition in AdmissionDisposition
        ),
        "feedback_dispositions": tuple(
            disposition.value for disposition in FeedbackDisposition
        ),
        "feedback_terminal_statuses": tuple(
            status.value for status in FeedbackTerminalStatus
        ),
        "terminal_classes": {
            terminal.value: TERMINAL_LEARNING_DISPOSITION[terminal]
            for terminal in TerminalClass
        },
        "terminal_rules": (
            "exactly one terminal classification per completed ticket; a "
            "deadline expiry is censored pending post-run reconciliation "
            "because feedback-only control-plane loss is not separable at "
            "runtime; an instrument fault is excluded and reported as an "
            "experimental failure, never converted into a penalty"
        ),
        "duplicate_rules": (
            "a byte-identical duplicate of accepted feedback is ignored "
            "idempotently; a conflicting duplicate fails closed; late matching "
            "feedback is LATE_ORPHAN and never attaches to a newer decision"
        ),
        "message_identity": (
            "the feedback duplicate key is the canonical hash of the wire "
            "message; the local receipt instant is deliberately excluded from "
            "message identity"
        ),
        "bounded_memory": (
            "closed-ticket identities are retained in an insertion-ordered "
            "bounded history of max_terminal_history completed tickets, "
            "evicted oldest-first; an evicted decision_seq is rejected as "
            "REJECTED_UNKNOWN_DECISION and can never mutate an active ticket"
        ),
        "frame_readmission": (
            "admitted carla_frame_id values are retained for the whole "
            "controller session, independently of the bounded terminal "
            "history, so an already-admitted frame stays refused even after "
            "its completed ticket has been evicted; a frame id is recorded "
            "only after its admission has fully succeeded"
        ),
        "min_hold_timestamp": (
            "min_hold_satisfied_ns is stamped exactly once, by the admission "
            "that first brings the governed tensor count to k_min, and is "
            "never overwritten by a later reused frame, feedback, deadline "
            "expiry, fault or closure"
        ),
        "concurrency": (
            "not thread-safe and intentionally unlocked; runtime integration "
            "must serialize frame admission, feedback, clock/deadline "
            "observation and fault recording through one event loop or queue"
        ),
    }
)

CONTROLLER_SCHEMA_ID: str = str(CONTROLLER_SCHEMA_DESCRIPTOR["schema_id"])
CONTROLLER_SCHEMA_VERSION: int = int(CONTROLLER_SCHEMA_DESCRIPTOR["version"])
CONTROLLER_SCHEMA_SHA256: str = canonical_sha256(CONTROLLER_SCHEMA_DESCRIPTOR)

#: Default bound on retained closed-ticket identities.
DEFAULT_MAX_TERMINAL_HISTORY: int = 8


# --------------------------------------------------------------------------- #
# Local validators: fail closed, never normalize
# --------------------------------------------------------------------------- #


def _exact_non_negative_int(value: Any, field_name: str) -> int:
    """Validate an exact non-negative Python ``int`` (``bool`` is rejected)."""
    if isinstance(value, bool):
        raise SequenceOrderError(
            f"{field_name} must be a non-negative int, not a bool: {value!r}"
        )
    if type(value) is not int:
        raise SequenceOrderError(
            f"{field_name} must be an exact int, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise SequenceOrderError(f"{field_name} must be >= 0, got {value}")
    return value


def _require_ns(value: Any, field_name: str) -> int:
    """Validate an exact non-negative nanosecond timestamp on a record field."""
    if isinstance(value, bool):
        raise RewardTicketControllerError(
            f"{field_name} must be a non-negative int, not a bool: {value!r}"
        )
    if type(value) is not int:
        raise RewardTicketControllerError(
            f"{field_name} must be an exact int nanosecond reading, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise RewardTicketControllerError(
            f"{field_name} must be >= 0, got {value}"
        )
    return value


def _require_sha256_hex(value: Any, field_name: str) -> str:
    """Validate a 64-character lowercase hex SHA-256 digest."""
    if not isinstance(value, str):
        raise RewardTicketControllerError(
            f"{field_name} must be a str digest, got "
            f"{type(value).__name__}: {value!r}"
        )
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RewardTicketControllerError(
            f"{field_name} must be 64 lowercase hex characters, got {value!r}"
        )
    return value


def _canonical_uuid(value: Any) -> str:
    """Validate the canonical lowercase hyphenated UUID form, as in Phase 2."""
    if not isinstance(value, str):
        raise RewardTicketControllerError(
            f"session_uuid must be a str, got {type(value).__name__}: {value!r}"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise RewardTicketControllerError(
            f"session_uuid is not a parsable UUID: {value!r}"
        ) from exc
    if str(parsed) != value:
        raise RewardTicketControllerError(
            f"session_uuid must be the canonical lowercase hyphenated form "
            f"{str(parsed)!r}, got {value!r}"
        )
    return value


# --------------------------------------------------------------------------- #
# Immutable records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RewardFeedbackMessage:
    """One submitted terminal event for a decision's registered reward tensor.

    The message is the wire payload only: it carries the Phase-2
    :class:`~.transaction_identity.RewardFeedbackIdentity` and a
    :class:`FeedbackTerminalStatus`.  The **local receipt instant is not a
    field** -- it is supplied separately to
    :meth:`RewardTicketController.submit_feedback` as the authoritative clock
    observation.  Keeping it out of the message means the duplicate key is a
    pure function of the wire bytes, and that no arrival timestamp can ever
    participate in a join.
    """

    identity: RewardFeedbackIdentity
    terminal_status: FeedbackTerminalStatus

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RewardFeedbackIdentity):
            raise RewardTicketControllerError(
                f"identity must be a Phase-2 RewardFeedbackIdentity, got "
                f"{type(self.identity).__name__}"
            )
        if not isinstance(self.terminal_status, FeedbackTerminalStatus):
            raise RewardTicketControllerError(
                f"terminal_status must be a FeedbackTerminalStatus, got "
                f"{type(self.terminal_status).__name__}: "
                f"{self.terminal_status!r}"
            )
        # Phase 2 already requires a reconciled action; restate it so the
        # serializable boundary of this phase is self-evidently guarded.
        self.identity.action.require_reconciled()

    # -- derived identity -------------------------------------------------- #

    @property
    def session_uuid(self) -> str:
        return self.identity.session_uuid

    @property
    def decision_seq(self) -> int:
        return self.identity.decision_seq

    @property
    def reward_tensor_seq(self) -> int:
        return self.identity.reward_tensor_seq

    @property
    def carla_frame_id(self) -> int:
        return self.identity.carla_frame_id

    @property
    def action(self) -> ExecutedActionIdentity:
        return self.identity.action

    @property
    def terminal_class(self) -> TerminalClass:
        """The terminal classification this message asserts."""
        return _FEEDBACK_TERMINAL_CLASS[self.terminal_status]

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping of the wire message, free of any local timestamp."""
        return {
            "controller_schema_id": CONTROLLER_SCHEMA_ID,
            "controller_schema_sha256": CONTROLLER_SCHEMA_SHA256,
            "controller_schema_version": CONTROLLER_SCHEMA_VERSION,
            "feedback_identity": self.identity.to_canonical_dict(),
            "record": "reward_feedback_message",
            "terminal_status": self.terminal_status.value,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """The duplicate key: byte/field identity of the wire message."""
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class CompletedTicket:
    """Immutable summary of exactly one completed reward ticket.

    The hold itself is carried as a Phase-2
    :class:`~.transaction_identity.ActionHoldManifest`, so the ``k_min``,
    shared-identity, unique-``tensor_seq`` and earliest-tensor-requests-reward
    invariants are re-proved by the frozen dependency rather than restated here.

    ``hold_duration_tensors`` is the realized ``d``.  No scalar reward,
    discount or ``gamma**d`` appears: those belong to the later SMDP/replay
    layer.

    ``min_hold_satisfied_ns`` is the instant of the admission that first brought
    the hold to ``k_min`` tensors.  It is **not** the closure instant: on the
    normal path the ticket closes later, when exact feedback arrives, so
    ``min_hold_satisfied_ns < closed_ns``.  The two coincide only when the
    ``k_min``-satisfying reuse is itself what releases an already-terminal
    ticket.

    ``__post_init__`` enforces the full timestamp and terminal-consistency
    algebra: the deadline is exactly ``opened_ns + B``, the timestamps are
    ordered ``opened_ns <= min_hold_satisfied_ns <= closed_ns``, and the
    terminal class, the resolution timestamp and the accepted-feedback digest
    either all describe accepted feedback or all describe its absence.
    """

    session_uuid: str
    decision_seq: int
    hold: ActionHoldManifest
    terminal_class: TerminalClass
    opened_ns: int
    deadline_ns: int
    closed_ns: int
    resolution_ns: Optional[int]
    accepted_feedback_sha256: Optional[str]
    min_hold_satisfied_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.hold, ActionHoldManifest):
            raise RewardTicketControllerError(
                f"hold must be a Phase-2 ActionHoldManifest, got "
                f"{type(self.hold).__name__}"
            )
        if not isinstance(self.terminal_class, TerminalClass):
            raise RewardTicketControllerError(
                f"terminal_class must be a TerminalClass, got "
                f"{type(self.terminal_class).__name__}"
            )
        # -- formats first, so a malformed field is reported as itself rather
        # -- than as a downstream hold disagreement ------------------------ #
        _canonical_uuid(self.session_uuid)
        if isinstance(self.decision_seq, bool) or type(self.decision_seq) is not int:
            raise RewardTicketControllerError(
                f"decision_seq must be an exact int, got "
                f"{type(self.decision_seq).__name__}: {self.decision_seq!r}"
            )
        if self.decision_seq < 0:
            raise RewardTicketControllerError(
                f"decision_seq must be >= 0, got {self.decision_seq}"
            )
        _require_ns(self.opened_ns, "opened_ns")
        _require_ns(self.deadline_ns, "deadline_ns")
        _require_ns(self.closed_ns, "closed_ns")
        _require_ns(self.min_hold_satisfied_ns, "min_hold_satisfied_ns")
        if self.resolution_ns is not None:
            _require_ns(self.resolution_ns, "resolution_ns")
        if self.accepted_feedback_sha256 is not None:
            _require_sha256_hex(
                self.accepted_feedback_sha256, "accepted_feedback_sha256"
            )

        # -- cross-record consistency -------------------------------------- #
        if self.hold.session_uuid != self.session_uuid:
            raise RewardTicketControllerError(
                "completed ticket session_uuid disagrees with its hold"
            )
        if self.hold.decision_seq != self.decision_seq:
            raise RewardTicketControllerError(
                "completed ticket decision_seq disagrees with its hold"
            )

        # -- timestamp algebra --------------------------------------------- #
        if self.deadline_ns != self.opened_ns + B_REWARD_DEADLINE_NS:
            raise RewardTicketControllerError(
                f"deadline_ns must be exactly opened_ns + "
                f"{B_REWARD_DEADLINE_NS} (the frozen B); got "
                f"{self.deadline_ns} for opened_ns {self.opened_ns}, which "
                f"implies {self.deadline_ns - self.opened_ns}"
            )
        if not (
            self.opened_ns <= self.min_hold_satisfied_ns <= self.closed_ns
        ):
            raise RewardTicketControllerError(
                f"timestamps must satisfy opened_ns <= min_hold_satisfied_ns "
                f"<= closed_ns; got {self.opened_ns} / "
                f"{self.min_hold_satisfied_ns} / {self.closed_ns}"
            )

        # -- terminal consistency ------------------------------------------ #
        resolved_by_feedback = self.terminal_class in _FEEDBACK_TERMINAL_CLASSES
        if resolved_by_feedback:
            if self.resolution_ns is None or self.accepted_feedback_sha256 is None:
                raise RewardTicketControllerError(
                    f"terminal class {self.terminal_class.value} is reached "
                    f"only by accepted exact feedback, so it requires both a "
                    f"resolution_ns and an accepted_feedback_sha256; got "
                    f"{self.resolution_ns!r} and "
                    f"{self.accepted_feedback_sha256!r}"
                )
            if not self.opened_ns <= self.resolution_ns <= self.closed_ns:
                raise RewardTicketControllerError(
                    f"resolution_ns {self.resolution_ns} must lie within "
                    f"[opened_ns {self.opened_ns}, closed_ns {self.closed_ns}]"
                )
            if self.resolution_ns > self.deadline_ns:
                raise RewardTicketControllerError(
                    f"resolution_ns {self.resolution_ns} is past deadline_ns "
                    f"{self.deadline_ns}; feedback accepted after the deadline "
                    f"is a LATE_ORPHAN and can never close a ticket"
                )
        else:
            if self.resolution_ns is not None or (
                self.accepted_feedback_sha256 is not None
            ):
                raise RewardTicketControllerError(
                    f"terminal class {self.terminal_class.value} closes a "
                    f"ticket with no accepted feedback, so resolution_ns and "
                    f"accepted_feedback_sha256 must both be null; got "
                    f"{self.resolution_ns!r} and "
                    f"{self.accepted_feedback_sha256!r}"
                )
            if (
                self.terminal_class is TerminalClass.FEEDBACK_TIMEOUT
                and self.closed_ns <= self.deadline_ns
            ):
                raise RewardTicketControllerError(
                    f"a FEEDBACK_TIMEOUT closes only after the deadline has "
                    f"passed, so closed_ns {self.closed_ns} must exceed "
                    f"deadline_ns {self.deadline_ns}"
                )

    # -- derived ----------------------------------------------------------- #

    @property
    def action(self) -> ExecutedActionIdentity:
        """The reconciled executed action held for the whole decision."""
        return self.hold.action

    @property
    def hold_duration_tensors(self) -> int:
        """The realized hold duration ``d``, in transmitted tensors."""
        return self.hold.tensor_count

    @property
    def reward_tensor_seq(self) -> int:
        return self.hold.reward_tensor_seq

    @property
    def reward_carla_frame_id(self) -> int:
        return self.hold.reward_tensor.transaction.carla_frame_id

    @property
    def tensor_seqs(self) -> Tuple[int, ...]:
        """Ascending governed ``tensor_seq`` values (the sender chronology)."""
        return self.hold.tensor_seqs

    @property
    def governed_frame_ids(self) -> Tuple[int, ...]:
        """Governed ``carla_frame_id`` values, in ``tensor_seq`` order."""
        return tuple(
            member.transaction.carla_frame_id for member in self.hold.tensors
        )

    @property
    def timed_out(self) -> bool:
        return self.terminal_class is TerminalClass.FEEDBACK_TIMEOUT

    @property
    def learning_disposition(self) -> str:
        """How the later learning layer must treat this transition."""
        return TERMINAL_LEARNING_DISPOSITION[self.terminal_class]

    @property
    def feedback_latency_ns(self) -> Optional[int]:
        """Exact open-to-receipt latency, or ``None`` with no exact receipt.

        This is the raw ``L_t`` input of DESIGN.md section 8; it is retained,
        never normalized by ``B`` and never scored here.
        """
        if self.resolution_ns is None:
            return None
        return self.resolution_ns - self.opened_ns

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical mapping; fails closed on an unreconciled action identity."""
        return {
            "accepted_feedback_sha256": self.accepted_feedback_sha256,
            "closed_ns": self.closed_ns,
            "controller_schema_id": CONTROLLER_SCHEMA_ID,
            "controller_schema_sha256": CONTROLLER_SCHEMA_SHA256,
            "controller_schema_version": CONTROLLER_SCHEMA_VERSION,
            "deadline_ns": self.deadline_ns,
            "decision_seq": self.decision_seq,
            "executed_action": self.action.to_canonical_dict(),
            "feedback_latency_ns": self.feedback_latency_ns,
            "hold": self.hold.to_canonical_dict(),
            "hold_duration_tensors": self.hold_duration_tensors,
            "learning_disposition": self.learning_disposition,
            "min_hold_satisfied_ns": self.min_hold_satisfied_ns,
            "minimum_hold_tensors": K_MIN_TENSORS,
            "opened_ns": self.opened_ns,
            "record": "reward_ticket_completion",
            "resolution_ns": self.resolution_ns,
            "reward_carla_frame_id": self.reward_carla_frame_id,
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "reward_tensor_seq": self.reward_tensor_seq,
            "session_uuid": self.session_uuid,
            "terminal_class": self.terminal_class.value,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class GateStatus:
    """Immutable view of the gate at one clock observation."""

    observed_ns: int
    state: ControllerState
    gate_available: bool
    held_decision_seq: Optional[int]
    governed_tensor_count: int
    min_hold_satisfied: bool
    min_hold_satisfied_ns: Optional[int]
    opened_ns: Optional[int]
    deadline_ns: Optional[int]
    remaining_ns: Optional[int]
    pending_terminal_class: Optional[TerminalClass]
    completed_ticket: Optional[CompletedTicket] = None


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Immutable result of admitting one prepared frame to the data path."""

    disposition: AdmissionDisposition
    envelope: TensorTransmissionEnvelope
    observed_ns: int
    state_before: ControllerState
    state_after: ControllerState
    governed_tensor_index: int
    deadline_ns: int
    #: The ticket this call completed, either by reaching ``k_min`` on this
    #: reuse or by the deadline evaluated at this call's clock observation
    #: before a new decision opened.
    completed_ticket: Optional[CompletedTicket] = None

    @property
    def transaction(self) -> TensorTransactionId:
        return self.envelope.transaction

    @property
    def decision_seq(self) -> int:
        return self.envelope.transaction.decision_seq

    @property
    def tensor_seq(self) -> int:
        return self.envelope.transaction.tensor_seq

    @property
    def carla_frame_id(self) -> int:
        return self.envelope.transaction.carla_frame_id

    @property
    def reward_requested(self) -> bool:
        return self.envelope.reward_requested

    @property
    def action(self) -> ExecutedActionIdentity:
        return self.envelope.action


@dataclass(frozen=True, slots=True)
class FeedbackOutcome:
    """Immutable classification of one submitted terminal event.

    ``state_before == state_after`` for every non-accepting disposition, which
    is the fail-closed guarantee: a rejected, duplicate or late-orphan message
    neither closes nor mutates a ticket.
    """

    disposition: FeedbackDisposition
    observed_ns: int
    state_before: ControllerState
    state_after: ControllerState
    decision_seq: int
    message_sha256: Optional[str]
    matched: Optional[str]
    terminal_class: Optional[TerminalClass]
    detail: str
    completed_ticket: Optional[CompletedTicket] = None

    @property
    def accepted(self) -> bool:
        """True only when this event resolved or faulted the active ticket."""
        return self.disposition in (
            FeedbackDisposition.ACCEPTED_RESOLVED,
            FeedbackDisposition.ACCEPTED_INFRASTRUCTURE_FAULT,
        )

    @property
    def rejected(self) -> bool:
        return self.disposition.value.startswith("REJECTED_")


# --------------------------------------------------------------------------- #
# Private active-ticket state
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _ActiveTicket:
    """Mutable bookkeeping for the single outstanding ticket.

    Private and never exported: every record leaving this module is immutable.
    ``envelopes`` grows only with the realized hold ``d``, which is bounded by
    the frames arriving inside ``B`` plus the one frame needed to satisfy
    ``k_min`` after a terminal event.
    """

    decision_seq: int
    action: ExecutedActionIdentity
    opened_ns: int
    deadline_ns: int
    envelopes: List[TensorTransmissionEnvelope]
    #: The instant of the admission that first brought the hold to ``k_min``
    #: tensors.  ``None`` until that admission happens, and written exactly
    #: once thereafter -- never by feedback, a deadline, a fault or closure.
    min_hold_satisfied_ns: Optional[int] = None
    pending_terminal_class: Optional[TerminalClass] = None
    resolution_ns: Optional[int] = None
    accepted_feedback_sha256: Optional[str] = None

    @property
    def tensor_count(self) -> int:
        return len(self.envelopes)

    def stamp_min_hold(self, now_ns: int) -> None:
        """Record the ``k_min``-satisfying instant, at most once per ticket.

        Called after every successful admission.  The ``is None`` guard is what
        makes the stamp immune to later reused frames: a hold of three or more
        tensors keeps the timestamp of its *second* tensor.
        """
        if self.min_hold_satisfied_ns is None and (
            self.tensor_count >= K_MIN_TENSORS
        ):
            self.min_hold_satisfied_ns = now_ns

    @property
    def reward_tensor_seq(self) -> int:
        return self.envelopes[0].transaction.tensor_seq

    @property
    def reward_carla_frame_id(self) -> int:
        return self.envelopes[0].transaction.carla_frame_id

    @property
    def governed_tensor_seqs(self) -> Tuple[int, ...]:
        return tuple(e.transaction.tensor_seq for e in self.envelopes)

    @property
    def governed_frame_ids(self) -> Tuple[int, ...]:
        return tuple(e.transaction.carla_frame_id for e in self.envelopes)


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #


class RewardTicketController:
    """Deterministic one-ticket action-hold and feedback gate for one session.

    All time is injected: every mutating entry point takes an exact
    ``now_ns``, which must be a non-negative Python ``int`` and must never
    regress.  The controller holds no clock of its own and never sleeps.

    Usage is either the two primitives -- :meth:`open_decision` and
    :meth:`reuse_held_action`, each of which fails closed when the gate
    disagrees -- or :meth:`admit_frame`, which observes the clock and routes to
    exactly one of them.  :meth:`admit_frame` invokes the supplied actor
    callable **only** when the gate is available, so the actor is provably not
    consulted during a hold.

    **This class is not thread-safe and is intentionally unlocked.**  Frame
    admission, feedback submission, clock/deadline observation and
    infrastructure-fault recording all mutate the same gate state, and the
    single-outstanding-ticket, strictly-increasing-sequence and monotonic-clock
    invariants are enforced without any mutual exclusion.  A runtime embedding
    this controller must serialize all four event kinds through one event loop
    or queue; see the module docstring.

    Memory: the terminal history is bounded by ``max_terminal_history``, but the
    set of admitted ``carla_frame_id`` values is **session-lifetime** state, by
    design -- frame non-readmission must not expire when a ticket is evicted.
    It grows by one integer per admitted frame and is released with the
    controller.
    """

    __slots__ = (
        "_session_uuid",
        "_max_terminal_history",
        "_state",
        "_ticket",
        "_history",
        "_admitted_frame_ids",
        "_last_observed_ns",
        "_last_decision_seq",
        "_last_tensor_seq",
        "_completed_count",
    )

    def __init__(
        self,
        session_uuid: str,
        *,
        max_terminal_history: int = DEFAULT_MAX_TERMINAL_HISTORY,
    ) -> None:
        self._session_uuid = _canonical_uuid(session_uuid)
        if isinstance(max_terminal_history, bool) or type(
            max_terminal_history
        ) is not int:
            raise RewardTicketControllerError(
                f"max_terminal_history must be an exact int, got "
                f"{type(max_terminal_history).__name__}: "
                f"{max_terminal_history!r}"
            )
        if max_terminal_history < 1:
            raise RewardTicketControllerError(
                f"max_terminal_history must be >= 1 so that a just-closed "
                f"ticket can still classify its own duplicates; got "
                f"{max_terminal_history}"
            )
        self._max_terminal_history = max_terminal_history
        self._state: ControllerState = ControllerState.READY
        self._ticket: Optional[_ActiveTicket] = None
        self._history: "OrderedDict[int, CompletedTicket]" = OrderedDict()
        # Session-lifetime identity set, deliberately *not* bounded by
        # max_terminal_history: frame non-readmission must survive eviction.
        self._admitted_frame_ids: "set[int]" = set()
        self._last_observed_ns: Optional[int] = None
        self._last_decision_seq: Optional[int] = None
        self._last_tensor_seq: Optional[int] = None
        self._completed_count = 0

    # -- read-only introspection ------------------------------------------ #

    @property
    def session_uuid(self) -> str:
        return self._session_uuid

    @property
    def state(self) -> ControllerState:
        return self._state

    @property
    def gate_available(self) -> bool:
        """True when a new decision may open on the next future frame."""
        return self._state in GATE_AVAILABLE_STATES

    @property
    def max_terminal_history(self) -> int:
        """The explicit bound on retained closed-ticket identities."""
        return self._max_terminal_history

    @property
    def completed_tickets(self) -> Tuple[CompletedTicket, ...]:
        """Retained closed tickets, oldest retained first."""
        return tuple(self._history.values())

    @property
    def retained_decision_seqs(self) -> Tuple[int, ...]:
        """``decision_seq`` values still inside the bounded history."""
        return tuple(self._history.keys())

    @property
    def completed_count(self) -> int:
        """Total tickets completed, including those evicted from history."""
        return self._completed_count

    @property
    def last_observed_ns(self) -> Optional[int]:
        return self._last_observed_ns

    def snapshot(self) -> GateStatus:
        """Non-mutating gate view at the last observed clock instant.

        Unlike :meth:`observe`, this never advances the clock and therefore
        never fires a deadline.  It is the tool for proving that a rejected
        message left the controller untouched.
        """
        return self._status(
            self._last_observed_ns if self._last_observed_ns is not None else 0
        )

    # -- clock ------------------------------------------------------------- #

    def observe(self, now_ns: int) -> GateStatus:
        """Advance the injected clock and report the gate.

        Advancing the clock is itself a contract event: if the active ticket is
        unresolved and ``now_ns`` is past its deadline, the ticket closes as
        ``FEEDBACK_TIMEOUT`` (or enters ``TIMED_OUT_WAITING_MIN_HOLD`` when
        fewer than ``k_min`` tensors have been transmitted) before the status is
        built.
        """
        completed = self._advance_clock(now_ns)
        return self._status(now_ns, completed_ticket=completed)

    def _require_clock(self, now_ns: Any) -> int:
        if isinstance(now_ns, bool) or type(now_ns) is not int:
            raise ClockRegressionError(
                f"now_ns must be an exact int nanosecond reading, got "
                f"{type(now_ns).__name__}: {now_ns!r}"
            )
        if now_ns < 0:
            raise ClockRegressionError(f"now_ns must be >= 0, got {now_ns}")
        if self._last_observed_ns is not None and now_ns < self._last_observed_ns:
            raise ClockRegressionError(
                f"the deadline is defined on a monotonic clock: now_ns "
                f"{now_ns} regresses below the last observation "
                f"{self._last_observed_ns}"
            )
        return now_ns

    def _advance_clock(self, now_ns: Any) -> Optional[CompletedTicket]:
        """Validate and record the clock, firing a due deadline exactly once."""
        observed = self._require_clock(now_ns)
        self._last_observed_ns = observed
        ticket = self._ticket
        if (
            ticket is None
            or self._state is not ControllerState.OPEN_UNRESOLVED
            or observed <= ticket.deadline_ns
        ):
            return None
        return self._fire_deadline(observed)

    def _fire_deadline(self, now_ns: int) -> Optional[CompletedTicket]:
        ticket = self._ticket
        assert ticket is not None  # guarded by _advance_clock
        ticket.pending_terminal_class = TerminalClass.FEEDBACK_TIMEOUT
        if ticket.tensor_count >= K_MIN_TENSORS:
            return self._close_ticket(TicketEvent.DEADLINE_EXPIRY, now_ns)
        # The minimum-two-tensor hold applies to a timeout exactly as it does to
        # a resolved ticket: the gate stays closed until tensor 2 has reused the
        # action.
        self._transition(
            TicketEvent.DEADLINE_EXPIRY,
            ControllerState.TIMED_OUT_WAITING_MIN_HOLD,
        )
        return None

    # -- state machine ----------------------------------------------------- #

    def _transition(self, event: TicketEvent, target: ControllerState) -> None:
        allowed = _ALLOWED_TRANSITIONS.get((self._state, event))
        if allowed is None or target not in allowed:
            raise IllegalTransitionError(
                f"{event.value} is not a declared transition from "
                f"{self._state.value}"
                + (
                    f" to {target.value}; declared targets are "
                    f"{[s.value for s in allowed]}"
                    if allowed is not None
                    else ""
                )
            )
        self._state = target

    def _close_ticket(
        self,
        event: TicketEvent,
        now_ns: int,
    ) -> CompletedTicket:
        ticket = self._ticket
        assert ticket is not None  # every caller checks
        terminal = ticket.pending_terminal_class
        if terminal is None:  # pragma: no cover - guarded by every caller
            raise IllegalTransitionError(
                "a ticket cannot close without exactly one terminal class"
            )
        if ticket.tensor_count < K_MIN_TENSORS:  # pragma: no cover - guarded
            raise IllegalTransitionError(
                f"a ticket cannot close with {ticket.tensor_count} governed "
                f"tensors; the frozen minimum hold is k_min={K_MIN_TENSORS}"
            )
        min_hold_satisfied_ns = ticket.min_hold_satisfied_ns
        if min_hold_satisfied_ns is None:  # pragma: no cover - guarded above
            raise IllegalTransitionError(
                f"decision {ticket.decision_seq} has {ticket.tensor_count} "
                f"governed tensors but no recorded k_min-satisfying instant; a "
                f"ticket cannot close without one"
            )
        completed = CompletedTicket(
            session_uuid=self._session_uuid,
            decision_seq=ticket.decision_seq,
            hold=ActionHoldManifest.build(ticket.envelopes),
            terminal_class=terminal,
            opened_ns=ticket.opened_ns,
            deadline_ns=ticket.deadline_ns,
            closed_ns=now_ns,
            resolution_ns=ticket.resolution_ns,
            accepted_feedback_sha256=ticket.accepted_feedback_sha256,
            min_hold_satisfied_ns=min_hold_satisfied_ns,
        )
        self._transition(event, ControllerState.CLOSED)
        self._ticket = None
        self._remember(completed)
        self._completed_count += 1
        return completed

    def _remember(self, completed: CompletedTicket) -> None:
        """Insert into the bounded history, evicting oldest-first.

        Insertion order is ``decision_seq`` order, because ``decision_seq`` is
        validated strictly increasing, so eviction is deterministic.
        """
        self._history[completed.decision_seq] = completed
        while len(self._history) > self._max_terminal_history:
            self._history.popitem(last=False)

    def _status(
        self,
        observed_ns: int,
        *,
        completed_ticket: Optional[CompletedTicket] = None,
    ) -> GateStatus:
        ticket = self._ticket
        pending = ticket.pending_terminal_class if ticket is not None else None
        remaining: Optional[int] = None
        if ticket is not None and self._state is ControllerState.OPEN_UNRESOLVED:
            remaining = max(0, ticket.deadline_ns - observed_ns)
        return GateStatus(
            observed_ns=observed_ns,
            state=self._state,
            gate_available=self.gate_available,
            held_decision_seq=None if ticket is None else ticket.decision_seq,
            governed_tensor_count=0 if ticket is None else ticket.tensor_count,
            min_hold_satisfied=(
                False if ticket is None else ticket.tensor_count >= K_MIN_TENSORS
            ),
            min_hold_satisfied_ns=(
                None if ticket is None else ticket.min_hold_satisfied_ns
            ),
            opened_ns=None if ticket is None else ticket.opened_ns,
            deadline_ns=None if ticket is None else ticket.deadline_ns,
            remaining_ns=remaining,
            pending_terminal_class=pending,
            completed_ticket=completed_ticket,
        )

    # -- frame admission --------------------------------------------------- #

    def _reject_readmitted_frame(self, carla_frame_id: int) -> None:
        """Refuse any frame already admitted anywhere in this session.

        A decision may open only on a *future* frame.  ``carla_frame_id`` is not
        chronology, so this is a set membership test, never an ordering
        comparison.  The set spans the **whole controller session** and is
        independent of ``max_terminal_history``: a frame stays refused after its
        completed ticket has been evicted from the bounded terminal history.
        The decision named in the message is best-effort diagnostic detail
        drawn from whatever is still retained; the refusal itself does not
        depend on it.
        """
        if carla_frame_id not in self._admitted_frame_ids:
            return
        if self._ticket is not None and carla_frame_id in (
            self._ticket.governed_frame_ids
        ):
            raise FrameReadmissionError(
                f"carla_frame_id {carla_frame_id} is already governed by the "
                f"active decision {self._ticket.decision_seq}; a frame already "
                f"admitted under an action can never be re-admitted"
            )
        for completed in self._history.values():
            if carla_frame_id in completed.governed_frame_ids:
                raise FrameReadmissionError(
                    f"carla_frame_id {carla_frame_id} was already governed by "
                    f"the completed decision {completed.decision_seq}; a new "
                    f"decision may open only on a future frame, never "
                    f"retroactively on one already admitted under the old "
                    f"action"
                )
        raise FrameReadmissionError(
            f"carla_frame_id {carla_frame_id} was already admitted earlier in "
            f"this controller session; its completed ticket has since been "
            f"evicted from the bounded terminal history, but frame "
            f"non-readmission is session-lifetime and does not expire with it"
        )

    def open_decision(
        self,
        *,
        decision_seq: int,
        tensor_seq: int,
        carla_frame_id: int,
        action: ExecutedActionIdentity,
        now_ns: int,
    ) -> AdmissionResult:
        """Open the single reward ticket on a frame that reached a ready gate.

        The returned envelope carries ``reward_requested=True``: DESIGN.md
        section 3 opens the ticket on the first frame of the hold.

        Raises:
            ClockRegressionError: if ``now_ns`` is malformed or regresses.
            GateUnavailableError: if a ticket is still held, including one that
                is resolved or timed out but has not yet reached ``k_min``.
            SequenceOrderError: if ``decision_seq`` or ``tensor_seq`` is
                malformed or not strictly increasing.
            FrameReadmissionError: if this frame was already admitted.
            UnreconciledActionIdentityError: if ``action`` was never reconciled
                against the frozen catalog.
        """
        completed_by_clock = self._advance_clock(now_ns)
        _exact_non_negative_int(decision_seq, "decision_seq")
        _exact_non_negative_int(tensor_seq, "tensor_seq")
        _exact_non_negative_int(carla_frame_id, "carla_frame_id")
        if not self.gate_available:
            raise GateUnavailableError(
                f"decision {decision_seq} cannot open while the gate is "
                f"{self._state.value}: exactly one reward ticket may be "
                f"outstanding, and a resolved or timed-out ticket still holds "
                f"the gate until k_min={K_MIN_TENSORS} tensors have reused the "
                f"action"
            )
        if not isinstance(action, ExecutedActionIdentity):
            raise RewardTicketControllerError(
                f"action must be a Phase-2 ExecutedActionIdentity, got "
                f"{type(action).__name__}"
            )
        action.require_reconciled()
        if (
            self._last_decision_seq is not None
            and decision_seq <= self._last_decision_seq
        ):
            raise SequenceOrderError(
                f"decision_seq must strictly increase within a session; "
                f"{decision_seq} does not follow {self._last_decision_seq}.  "
                f"Sequence numbers are supplied by the caller and never "
                f"silently allocated or reused"
            )
        self._require_tensor_seq(tensor_seq)
        self._reject_readmitted_frame(carla_frame_id)

        envelope = TensorTransmissionEnvelope(
            transaction=TensorTransactionId(
                session_uuid=self._session_uuid,
                decision_seq=decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
            ),
            reward_requested=True,
            action=action,
        )
        state_before = self._state
        self._ticket = _ActiveTicket(
            decision_seq=decision_seq,
            action=action,
            opened_ns=now_ns,
            deadline_ns=now_ns + B_REWARD_DEADLINE_NS,
            envelopes=[envelope],
        )
        self._transition(TicketEvent.OPEN_DECISION, ControllerState.OPEN_UNRESOLVED)
        self._last_decision_seq = decision_seq
        self._last_tensor_seq = tensor_seq
        # k_min is frozen at 2, so opening never satisfies the minimum hold;
        # the call is kept unconditional so the stamp has exactly one writer.
        self._ticket.stamp_min_hold(now_ns)
        # Recorded only now that the admission has fully succeeded.
        self._admitted_frame_ids.add(carla_frame_id)
        return AdmissionResult(
            disposition=AdmissionDisposition.OPENED_NEW_DECISION,
            envelope=envelope,
            observed_ns=now_ns,
            state_before=state_before,
            state_after=self._state,
            governed_tensor_index=1,
            deadline_ns=self._ticket.deadline_ns,
            completed_ticket=completed_by_clock,
        )

    def reuse_held_action(
        self,
        *,
        tensor_seq: int,
        carla_frame_id: int,
        now_ns: int,
        expect_decision_seq: Optional[int] = None,
    ) -> AdmissionResult:
        """Transmit one more tensor under the currently held decision.

        The envelope reuses the ticket's *exact* reconciled action identity and
        carries ``reward_requested=False``.  If the ticket's terminal class was
        already decided -- by feedback, by the deadline or by an instrument
        fault -- this reuse is the tensor that satisfies ``k_min`` and closes
        the ticket.

        Raises:
            NoOpenTicketError: if no ticket is held.
            SequenceOrderError: on a malformed or non-increasing ``tensor_seq``,
                or an ``expect_decision_seq`` that disagrees with the hold.
            FrameReadmissionError: if this frame was already admitted.
        """
        completed_by_clock = self._advance_clock(now_ns)
        _exact_non_negative_int(tensor_seq, "tensor_seq")
        _exact_non_negative_int(carla_frame_id, "carla_frame_id")
        ticket = self._ticket
        if ticket is None:
            raise NoOpenTicketError(
                f"no reward ticket is held, so there is no action to reuse; the "
                f"gate is {self._state.value}"
                + (
                    " (the previous ticket closed on this call's deadline "
                    "evaluation)"
                    if completed_by_clock is not None
                    else ""
                )
            )
        if expect_decision_seq is not None:
            _exact_non_negative_int(expect_decision_seq, "expect_decision_seq")
            if expect_decision_seq != ticket.decision_seq:
                raise SequenceOrderError(
                    f"expect_decision_seq {expect_decision_seq} disagrees with "
                    f"the held decision {ticket.decision_seq}; a tensor is "
                    f"never re-attributed to another decision"
                )
        self._require_tensor_seq(tensor_seq)
        self._reject_readmitted_frame(carla_frame_id)

        envelope = TensorTransmissionEnvelope(
            transaction=TensorTransactionId(
                session_uuid=self._session_uuid,
                decision_seq=ticket.decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
            ),
            reward_requested=False,
            action=ticket.action,
        )
        state_before = self._state
        ticket.envelopes.append(envelope)
        self._last_tensor_seq = tensor_seq
        governed_index = ticket.tensor_count
        deadline_ns = ticket.deadline_ns
        # Stamp before any closure, so a terminal-pending ticket released by
        # this very tensor carries this instant rather than its closure time --
        # which for that one path are the same instant anyway.
        ticket.stamp_min_hold(now_ns)
        # Recorded only now that the admission has fully succeeded.
        self._admitted_frame_ids.add(carla_frame_id)

        if state_before is ControllerState.OPEN_UNRESOLVED:
            self._transition(
                TicketEvent.REUSE_TENSOR, ControllerState.OPEN_UNRESOLVED
            )
            completed: Optional[CompletedTicket] = None
        else:
            # One of the three terminal-pending states: this tensor satisfies
            # k_min, so the already-decided terminal class now closes the ticket.
            completed = self._close_ticket(TicketEvent.REUSE_TENSOR, now_ns)
        return AdmissionResult(
            disposition=AdmissionDisposition.REUSED_HELD_ACTION,
            envelope=envelope,
            observed_ns=now_ns,
            state_before=state_before,
            state_after=self._state,
            governed_tensor_index=governed_index,
            deadline_ns=deadline_ns,
            completed_ticket=completed if completed is not None else None,
        )

    def admit_frame(
        self,
        *,
        tensor_seq: int,
        carla_frame_id: int,
        now_ns: int,
        next_decision_seq: int,
        select_action: Callable[[], ExecutedActionIdentity],
    ) -> AdmissionResult:
        """Route one prepared frame to exactly one admission primitive.

        ``next_decision_seq`` and ``select_action`` are consumed **only** when
        the gate is available at ``now_ns``.  During a hold the actor is not
        invoked and the candidate ``decision_seq`` is not consumed, which is the
        section-3 rule that the policy may not run while a ticket is
        outstanding.
        """
        if not callable(select_action):
            raise RewardTicketControllerError(
                f"select_action must be callable, got "
                f"{type(select_action).__name__}"
            )
        status = self.observe(now_ns)
        if status.gate_available:
            action = select_action()
            result = self.open_decision(
                decision_seq=next_decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
                action=action,
                now_ns=now_ns,
            )
        else:
            result = self.reuse_held_action(
                tensor_seq=tensor_seq,
                carla_frame_id=carla_frame_id,
                now_ns=now_ns,
            )
        if result.completed_ticket is None and status.completed_ticket is not None:
            return replace(result, completed_ticket=status.completed_ticket)
        return result

    def _require_tensor_seq(self, tensor_seq: int) -> None:
        if self._last_tensor_seq is not None and tensor_seq <= self._last_tensor_seq:
            raise SequenceOrderError(
                f"tensor_seq is the frozen sender chronology and must strictly "
                f"increase within a session; {tensor_seq} does not follow "
                f"{self._last_tensor_seq}.  carla_frame_id, timestamps and "
                f"arrival order are never used to infer order"
            )

    # -- terminal events --------------------------------------------------- #

    def submit_feedback(
        self,
        message: RewardFeedbackMessage,
        now_ns: int,
    ) -> FeedbackOutcome:
        """Classify one submitted reward-feedback message exactly once.

        ``now_ns`` is the authoritative UE receipt instant against which the
        200 ms deadline is evaluated; the message itself carries no timestamp.
        The clock is advanced first, so a message arriving after the deadline
        finds its ticket already timed out and is classified ``LATE_ORPHAN``.

        Every non-accepting disposition leaves the controller untouched.  Note
        that the clock advance itself is a real contract event: a deadline that
        was already due fires on this call regardless of the message's fate.
        """
        if not isinstance(message, RewardFeedbackMessage):
            raise RewardTicketControllerError(
                f"message must be a RewardFeedbackMessage, got "
                f"{type(message).__name__}"
            )
        completed_by_clock = self._advance_clock(now_ns)
        state_before = self._state
        digest = message.canonical_sha256()

        def outcome(
            disposition: FeedbackDisposition,
            detail: str,
            *,
            matched: Optional[str] = None,
            terminal_class: Optional[TerminalClass] = None,
            completed: Optional[CompletedTicket] = None,
        ) -> FeedbackOutcome:
            return FeedbackOutcome(
                disposition=disposition,
                observed_ns=now_ns,
                state_before=state_before,
                state_after=self._state,
                decision_seq=message.decision_seq,
                message_sha256=digest,
                matched=matched,
                terminal_class=terminal_class,
                detail=detail,
                completed_ticket=(
                    completed if completed is not None else completed_by_clock
                ),
            )

        if message.session_uuid != self._session_uuid:
            return outcome(
                FeedbackDisposition.REJECTED_WRONG_SESSION,
                f"feedback session {message.session_uuid} is not this "
                f"controller's session {self._session_uuid}",
            )

        ticket = self._ticket
        if ticket is not None and ticket.decision_seq == message.decision_seq:
            return self._classify_against_active(message, now_ns, digest, outcome)
        historical = self._history.get(message.decision_seq)
        if historical is not None:
            return self._classify_against_history(
                message, historical, digest, outcome
            )
        return outcome(
            FeedbackDisposition.REJECTED_UNKNOWN_DECISION,
            f"decision_seq {message.decision_seq} is neither the held decision "
            f"("
            + (
                "none held"
                if ticket is None
                else str(ticket.decision_seq)
            )
            + f") nor inside the retained terminal history "
            f"{list(self._history.keys())}; feedback is matched by exact "
            f"decision identity only and is never attached to another decision",
        )

    def _classify_against_active(
        self,
        message: RewardFeedbackMessage,
        now_ns: int,
        digest: str,
        outcome: Callable[..., FeedbackOutcome],
    ) -> FeedbackOutcome:
        ticket = self._ticket
        assert ticket is not None  # caller checked

        mismatch = self._identity_mismatch(
            message,
            reward_tensor_seq=ticket.reward_tensor_seq,
            reward_carla_frame_id=ticket.reward_carla_frame_id,
            governed_tensor_seqs=ticket.governed_tensor_seqs,
            action=ticket.action,
            where=f"held decision {ticket.decision_seq}",
        )
        if mismatch is not None:
            disposition, detail = mismatch
            return outcome(disposition, detail, matched="active")

        if self._state is ControllerState.OPEN_UNRESOLVED:
            terminal = message.terminal_class
            ticket.pending_terminal_class = terminal
            ticket.resolution_ns = now_ns
            ticket.accepted_feedback_sha256 = digest
            if ticket.tensor_count >= K_MIN_TENSORS:
                completed = self._close_ticket(
                    TicketEvent.ACCEPT_FEEDBACK, now_ns
                )
                detail = (
                    f"exact feedback closed decision "
                    f"{completed.decision_seq} as {terminal.value} with "
                    f"d={completed.hold_duration_tensors}"
                )
                return outcome(
                    FeedbackDisposition.ACCEPTED_RESOLVED,
                    detail,
                    matched="active",
                    terminal_class=terminal,
                    completed=completed,
                )
            self._transition(
                TicketEvent.ACCEPT_FEEDBACK,
                ControllerState.RESOLVED_WAITING_MIN_HOLD,
            )
            return outcome(
                FeedbackDisposition.ACCEPTED_RESOLVED,
                f"exact feedback resolved decision {ticket.decision_seq} as "
                f"{terminal.value}, but only {ticket.tensor_count} of "
                f"k_min={K_MIN_TENSORS} tensors have been transmitted, so the "
                f"gate stays closed until tensor 2 reuses the action",
                matched="active",
                terminal_class=terminal,
            )

        if self._state is ControllerState.RESOLVED_WAITING_MIN_HOLD:
            if digest == ticket.accepted_feedback_sha256:
                return outcome(
                    FeedbackDisposition.DUPLICATE_IGNORED,
                    f"byte-identical duplicate of the feedback already "
                    f"accepted for decision {ticket.decision_seq}; ignored "
                    f"idempotently with no second effect",
                    matched="active",
                    terminal_class=ticket.pending_terminal_class,
                )
            return outcome(
                FeedbackDisposition.REJECTED_CONFLICTING_DUPLICATE,
                f"decision {ticket.decision_seq} already accepted feedback "
                f"{ticket.accepted_feedback_sha256}; this message has the same "
                f"exact identity but a conflicting payload ({digest}), so it "
                f"fails closed and does not mutate the resolved ticket",
                matched="active",
            )

        # TIMED_OUT_WAITING_MIN_HOLD or FAULTED_WAITING_MIN_HOLD: the terminal
        # class is already decided and is not superseded by a late arrival.
        return outcome(
            FeedbackDisposition.LATE_ORPHAN,
            f"decision {ticket.decision_seq} is already "
            f"{self._state.value} with terminal class "
            f"{ticket.pending_terminal_class.value}; this matching message is "
            f"diagnostic only and can never be attached to this or a newer "
            f"decision",
            matched="active",
        )

    def _classify_against_history(
        self,
        message: RewardFeedbackMessage,
        historical: CompletedTicket,
        digest: str,
        outcome: Callable[..., FeedbackOutcome],
    ) -> FeedbackOutcome:
        mismatch = self._identity_mismatch(
            message,
            reward_tensor_seq=historical.reward_tensor_seq,
            reward_carla_frame_id=historical.reward_carla_frame_id,
            governed_tensor_seqs=historical.tensor_seqs,
            action=historical.action,
            where=f"closed decision {historical.decision_seq}",
        )
        if mismatch is not None:
            disposition, detail = mismatch
            return outcome(disposition, detail, matched="terminal_history")

        if historical.accepted_feedback_sha256 is not None:
            if digest == historical.accepted_feedback_sha256:
                return outcome(
                    FeedbackDisposition.DUPLICATE_IGNORED,
                    f"byte-identical duplicate of the feedback that already "
                    f"closed decision {historical.decision_seq} as "
                    f"{historical.terminal_class.value}; ignored idempotently",
                    matched="terminal_history",
                    terminal_class=historical.terminal_class,
                )
            return outcome(
                FeedbackDisposition.REJECTED_CONFLICTING_DUPLICATE,
                f"closed decision {historical.decision_seq} accepted "
                f"{historical.accepted_feedback_sha256}; this conflicting "
                f"payload ({digest}) fails closed",
                matched="terminal_history",
            )

        return outcome(
            FeedbackDisposition.LATE_ORPHAN,
            f"decision {historical.decision_seq} already closed as "
            f"{historical.terminal_class.value} with no accepted feedback; "
            f"this matching message is a diagnostic LATE_ORPHAN and is never "
            f"attached to a newer decision",
            matched="terminal_history",
            terminal_class=historical.terminal_class,
        )

    @staticmethod
    def _identity_mismatch(
        message: RewardFeedbackMessage,
        *,
        reward_tensor_seq: int,
        reward_carla_frame_id: int,
        governed_tensor_seqs: Tuple[int, ...],
        action: ExecutedActionIdentity,
        where: str,
    ) -> Optional[Tuple[FeedbackDisposition, str]]:
        """Exact-identity check; returns the rejection, or ``None`` when exact.

        Order is deliberate: a tensor/frame/action identity disagreement is more
        specific than a payload conflict, so it is reported as itself.
        """
        if message.reward_tensor_seq != reward_tensor_seq:
            if message.reward_tensor_seq in governed_tensor_seqs:
                return (
                    FeedbackDisposition.REJECTED_NON_REWARD_TENSOR,
                    f"tensor_seq {message.reward_tensor_seq} is governed by "
                    f"{where} but is not its registered reward tensor "
                    f"({reward_tensor_seq}); only the reward-requested tensor "
                    f"may generate that decision's reward",
                )
            return (
                FeedbackDisposition.REJECTED_WRONG_REWARD_TENSOR_SEQ,
                f"reward tensor_seq {message.reward_tensor_seq} is not the "
                f"registered reward tensor {reward_tensor_seq} of {where}; no "
                f"nearest-match lookup is performed",
            )
        if message.carla_frame_id != reward_carla_frame_id:
            return (
                FeedbackDisposition.REJECTED_WRONG_CARLA_FRAME,
                f"carla_frame_id {message.carla_frame_id} does not validate "
                f"the reward tensor of {where} ({reward_carla_frame_id})",
            )
        if message.action != action:
            return (
                FeedbackDisposition.REJECTED_ACTION_MISMATCH,
                f"feedback carries executed action "
                f"{message.action.canonical_mode} q_e4={message.action.q_e4}, "
                f"but {where} executed {action.canonical_mode} "
                f"q_e4={action.q_e4}",
            )
        return None

    def record_infrastructure_fault(
        self,
        *,
        decision_seq: int,
        now_ns: int,
        detail: str,
    ) -> FeedbackOutcome:
        """Close the held ticket as an excluded evaluator/infrastructure fault.

        DESIGN.md section 8 case 3: an instrument fault is an experimental
        failure, excluded from learning and reported separately.  It is kept
        distinguishable from ``ACTION_PATH_FAILURE`` and from
        ``FEEDBACK_TIMEOUT``, is never converted into a scalar penalty, and --
        because section 8 allows exactly one terminal classification per
        transition -- may be recorded only while the ticket is still
        ``OPEN_UNRESOLVED``.  The same ``k_min`` hold applies.
        """
        self._advance_clock(now_ns)
        _exact_non_negative_int(decision_seq, "decision_seq")
        if not isinstance(detail, str) or detail == "":
            raise RewardTicketControllerError(
                "an infrastructure fault must carry a non-empty detail string "
                "so it can be reported as an experimental failure"
            )
        ticket = self._ticket
        if ticket is None:
            raise NoOpenTicketError(
                f"no reward ticket is held, so no infrastructure fault can be "
                f"attached; the gate is {self._state.value}"
            )
        if ticket.decision_seq != decision_seq:
            raise SequenceOrderError(
                f"decision_seq {decision_seq} is not the held decision "
                f"{ticket.decision_seq}; a fault is never re-attributed"
            )
        state_before = self._state
        terminal = TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED
        if state_before is not ControllerState.OPEN_UNRESOLVED:
            # Fails closed through the declared table rather than silently
            # superseding an already-decided terminal classification.
            self._transition(
                TicketEvent.INFRASTRUCTURE_FAULT, ControllerState.CLOSED
            )
        ticket.pending_terminal_class = terminal
        if ticket.tensor_count >= K_MIN_TENSORS:
            completed = self._close_ticket(
                TicketEvent.INFRASTRUCTURE_FAULT, now_ns
            )
            state_after = self._state
        else:
            self._transition(
                TicketEvent.INFRASTRUCTURE_FAULT,
                ControllerState.FAULTED_WAITING_MIN_HOLD,
            )
            completed = None
            state_after = self._state
        return FeedbackOutcome(
            disposition=FeedbackDisposition.ACCEPTED_INFRASTRUCTURE_FAULT,
            observed_ns=now_ns,
            state_before=state_before,
            state_after=state_after,
            decision_seq=decision_seq,
            message_sha256=None,
            matched="active",
            terminal_class=terminal,
            detail=detail,
            completed_ticket=completed,
        )

    # -- repr -------------------------------------------------------------- #

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"RewardTicketController(session_uuid={self._session_uuid!r}, "
            f"state={self._state.value}, "
            f"held_decision_seq="
            f"{None if self._ticket is None else self._ticket.decision_seq}, "
            f"completed={self._completed_count})"
        )

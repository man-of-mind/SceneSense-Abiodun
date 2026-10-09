"""Offline, bounded accepted-frame transaction protocol.

This module neither schedules threads nor calls a producer, scorer, network, or
archive. A driver pulls one computation lease at a time and acknowledges each
consumer. History is an opaque immutable replacement, NOT an implementation of
the retained tracker's cloning/delta semantics. A real adapter must independently
prove that its tentative history is isolated and numerically equivalent.

Admission reservations must conservatively include *all* frame-owned storage,
including front/worker/validation/archive storage outside this object. The checks
here additionally bound bytes retained by the controller. History has its own
explicit byte limit. Consumer completion acknowledges relinquishment of their
references; the controller cannot revoke references retained by a caller.

There is no unbounded event/tombstone log: duplicate events on active tickets are
idempotent, and events on retired tickets raise RetiredTransaction. The accepted
frame index is a strictly increasing order within one controller/run, not a
replacement for the complete canonical plan identity.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import threading
from typing import Callable, Optional


def _bytes(value: bytes, name: str, *, nonempty: bool = False) -> None:
    if type(value) is not bytes or (nonempty and not value):
        raise ValueError(f"{name} must be immutable bytes" + (" and nonempty" if nonempty else ""))


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PlanKey:
    frame_index: int
    canonical_identity: bytes

    def __post_init__(self) -> None:
        _integer(self.frame_index, "frame_index")
        _bytes(self.canonical_identity, "canonical_identity", nonempty=True)


@dataclass(frozen=True, slots=True)
class WireBinding:
    plan_key: PlanKey
    canonical_wire_identity: bytes

    def __post_init__(self) -> None:
        if type(self.plan_key) is not PlanKey:
            raise ValueError("plan_key must be a PlanKey")
        _bytes(self.canonical_wire_identity, "canonical_wire_identity", nonempty=True)


class PlanKind(Enum):
    ACCEPTED = "accepted_reward_frame"
    HOLD = "hold"
    FALLBACK = "fallback"
    DISPLACED = "displaced"


@dataclass(frozen=True, slots=True)
class FramePlan:
    key: PlanKey
    expected_wire: WireBinding
    kind: PlanKind
    input_payload: bytes
    reserved_bytes: int

    def __post_init__(self) -> None:
        if type(self.key) is not PlanKey or type(self.expected_wire) is not WireBinding:
            raise ValueError("plan and wire identities must be immutable typed keys")
        if self.expected_wire.plan_key != self.key:
            raise ValueError("wire binding belongs to another plan")
        if type(self.kind) is not PlanKind:
            raise ValueError("kind must explicitly identify accepted/hold/fallback/displaced")
        _bytes(self.input_payload, "input_payload")
        _integer(self.reserved_bytes, "reserved_bytes", 1)
        if self.owned_bytes > self.reserved_bytes:
            raise ValueError("plan identity/input bytes exceed reservation")

    @property
    def owned_bytes(self) -> int:
        return (len(self.key.canonical_identity) + len(self.expected_wire.canonical_wire_identity)
                + len(self.input_payload))


@dataclass(frozen=True, slots=True)
class HistoryState:
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        _bytes(self.payload, "history payload")
        if type(self.sha256) is not str or self.sha256 != _sha(self.payload):
            raise ValueError("history digest mismatch")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "HistoryState":
        _bytes(payload, "history payload")
        return cls(payload, _sha(payload))


@dataclass(frozen=True, slots=True)
class TransactionTicket:
    serial: int
    plan_key: PlanKey
    _owner: object = field(repr=False)

    def __post_init__(self) -> None:
        _integer(self.serial, "ticket serial", 1)
        if type(self.plan_key) is not PlanKey or type(self._owner) is not object:
            raise ValueError("ticket requires a typed plan key and opaque controller token")


@dataclass(frozen=True, slots=True)
class ComputationLease:
    ticket: TransactionTicket
    plan: FramePlan
    base_history: HistoryState
    history_version: int

    def __post_init__(self) -> None:
        if (type(self.ticket) is not TransactionTicket or type(self.plan) is not FramePlan or
                type(self.base_history) is not HistoryState or self.ticket.plan_key != self.plan.key):
            raise ValueError("lease requires consistent immutable ticket, plan, and history")
        _integer(self.history_version, "history_version")


@dataclass(frozen=True, slots=True)
class TentativeResult:
    plan_key: PlanKey
    wire_binding: WireBinding
    base_history_sha256: str
    next_history: HistoryState
    numerical_payload: bytes
    numerical_sha256: str

    def __post_init__(self) -> None:
        if type(self.plan_key) is not PlanKey or type(self.wire_binding) is not WireBinding:
            raise ValueError("result needs immutable typed identities")
        if self.wire_binding.plan_key != self.plan_key:
            raise ValueError("result wire binding belongs to another plan")
        if type(self.next_history) is not HistoryState:
            raise ValueError("next_history must be immutable HistoryState")
        if type(self.base_history_sha256) is not str:
            raise ValueError("base history digest must be a string")
        _bytes(self.numerical_payload, "numerical_payload")
        if type(self.numerical_sha256) is not str or self.numerical_sha256 != _sha(self.numerical_payload):
            raise ValueError("numerical digest mismatch")

    @classmethod
    def for_lease(cls, lease: ComputationLease, next_history: bytes, numerical_payload: bytes) -> "TentativeResult":
        _bytes(numerical_payload, "numerical_payload")
        return cls(lease.plan.key, lease.plan.expected_wire, lease.base_history.sha256,
                   HistoryState.from_bytes(next_history), numerical_payload, _sha(numerical_payload))


@dataclass(frozen=True, slots=True)
class ReadyPayload:
    ticket: TransactionTicket
    result: TentativeResult
    history_version: int


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    run_valid: bool
    first_failure: Optional[str]
    failure_count: int
    closed: bool
    pending_jobs: int
    reserved_bytes: int
    committed_frames: int
    feedback_emitted: int
    released_frames: int
    history: HistoryState
    history_version: int

    @property
    def drained(self) -> bool:
        return self.pending_jobs == 0


class TransactionError(RuntimeError):
    pass


class ProtocolError(TransactionError):
    pass


class CapacityExceeded(TransactionError):
    pass


class RunInvalid(TransactionError):
    pass


class RetiredTransaction(TransactionError):
    """No references remain for this ticket; it cannot be resurrected."""


@dataclass
class _Transaction:
    ticket: TransactionTicket
    plan: FramePlan
    lease: Optional[ComputationLease] = None
    worker_pending: bool = False
    result: Optional[TentativeResult] = None
    validation_pending: bool = False
    validated: bool = False
    send_confirmed: bool = False
    cancelled: bool = False
    committed_version: Optional[int] = None
    ready_taken: bool = False
    scoring_done: bool = False
    feedback_terminal: bool = False
    feedback_emitted: bool = False
    archive_done: bool = False
    archive_succeeded: bool = False


class TransactionController:
    """Single ordered tentative history, with independently draining consumers.

    ``validate_computation`` is the only caller callback. It runs outside locks
    and must perform full numerical/identity admission, raising on rejection and
    returning None on success. A boolean ``already_validated`` shortcut is not
    provided. This generic module does not itself know the numerical DTO schema.

    An invalid run rejects new admission/leases/commits but permits outstanding
    work to finish or cancel so reservations can drain. Already published feedback
    is counted permanently, including after a later archive failure.
    """

    def __init__(self, *, max_jobs: int, max_bytes: int, max_history_bytes: int,
                 initial_history: bytes = b"") -> None:
        _integer(max_jobs, "max_jobs", 1)
        _integer(max_bytes, "max_bytes", 1)
        _integer(max_history_bytes, "max_history_bytes", 1)
        _bytes(initial_history, "initial_history")
        if len(initial_history) > max_history_bytes:
            raise ValueError("initial history exceeds its explicit capacity")
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._max_history_bytes = max_history_bytes
        self._history = HistoryState.from_bytes(initial_history)
        self._history_version = 0
        self._owner = object()
        self._condition = threading.Condition(threading.Lock())
        self._states: dict[int, _Transaction] = {}
        self._next_serial = 1
        self._last_frame_index = -1
        self._reserved_bytes = 0
        self._first_failure: Optional[str] = None
        self._failure_count = 0
        self._closed = False
        self._feedback_count = 0
        self._released_count = 0

    def _fault(self, reason: str) -> None:
        self._failure_count += 1
        if self._first_failure is None:
            self._first_failure = reason[:1024]

    def _protocol(self, reason: str) -> None:
        self._fault(reason)
        raise ProtocolError(reason)

    def _healthy(self) -> None:
        if self._first_failure is not None:
            raise RunInvalid(self._first_failure)

    def _get(self, ticket: TransactionTicket) -> _Transaction:
        if type(ticket) is not TransactionTicket or ticket._owner is not self._owner:
            self._protocol("foreign transaction ticket")
        state = self._states.get(ticket.serial)
        if state is None:
            raise RetiredTransaction("transaction has retired or was never admitted")
        if state.ticket != ticket:
            self._protocol("transaction ticket identity mismatch")
        return state

    def _release_if_done(self, state: _Transaction) -> None:
        cancelled_done = state.cancelled and not state.worker_pending and not state.validation_pending
        consumers_done = (state.committed_version is not None and state.scoring_done
                          and state.feedback_terminal and state.archive_done)
        if cancelled_done or consumers_done:
            del self._states[state.ticket.serial]
            self._reserved_bytes -= state.plan.reserved_bytes
            self._released_count += 1
            self._condition.notify_all()

    def _commit_if_ready(self, state: _Transaction) -> None:
        if (state.cancelled or state.committed_version is not None or
                not state.validated or not state.send_confirmed):
            return
        # Infrastructure failure stops new history mutation, without preventing
        # callers from acknowledging and releasing outstanding reservations.
        if self._first_failure is not None:
            state.cancelled = True
            self._release_if_done(state)
            return
        assert state.result is not None and state.lease is not None
        if state.lease.history_version != self._history_version or state.lease.base_history != self._history:
            self._protocol("tentative result is based on uncommitted or stale history")
        self._history = state.result.next_history
        self._history_version += 1
        state.committed_version = self._history_version

    def admit(self, plan: FramePlan) -> Optional[TransactionTicket]:
        """Reserve before front computation; excluded populations get no ticket."""
        if type(plan) is not FramePlan:
            raise ValueError("admission requires an immutable FramePlan")
        with self._condition:
            if plan.kind is not PlanKind.ACCEPTED:
                return None
            self._healthy()
            if self._closed:
                raise TransactionError("admission is closed")
            for state in self._states.values():
                if state.plan.key.frame_index == plan.key.frame_index:
                    if state.plan != plan:
                        self._protocol("conflicting duplicate plan admission")
                    return state.ticket
            if plan.key.frame_index <= self._last_frame_index:
                self._protocol("accepted frame index must strictly increase")
            if len(self._states) >= self._max_jobs or self._reserved_bytes + plan.reserved_bytes > self._max_bytes:
                self._fault("accepted frame admission overflow")
                raise CapacityExceeded("accepted frame admission overflow; no existing job was evicted")
            ticket = TransactionTicket(self._next_serial, plan.key, self._owner)
            self._next_serial += 1
            self._last_frame_index = plan.key.frame_index
            self._states[ticket.serial] = _Transaction(ticket, plan)
            self._reserved_bytes += plan.reserved_bytes
            return ticket

    def claim_next(self) -> Optional[ComputationLease]:
        """Only the earliest unresolved accepted frame can compute history."""
        with self._condition:
            self._healthy()
            for state in self._states.values():
                if state.committed_version is not None:
                    continue
                if state.cancelled:
                    if state.worker_pending or state.validation_pending:
                        return None
                    continue
                if state.lease is not None:
                    return None
                if state.plan.owned_bytes + len(self._history.payload) > state.plan.reserved_bytes:
                    self._fault("retained base history exceeded the frame reservation")
                    state.cancelled = True
                    self._release_if_done(state)
                    raise CapacityExceeded("retained base history exceeded the frame reservation")
                state.lease = ComputationLease(state.ticket, state.plan, self._history, self._history_version)
                state.worker_pending = True
                return state.lease
            return None

    def complete_computation(self, lease: ComputationLease, result: TentativeResult) -> bool:
        """Retain a provisional result; this alone never advances history."""
        if type(lease) is not ComputationLease:
            raise ValueError("completion requires the issued lease")
        with self._condition:
            state = self._get(lease.ticket)
            if state.lease is not lease:
                self._protocol("completion does not match the issued lease")
            if not state.worker_pending:
                if type(result) is TentativeResult and state.result == result:
                    return False
                self._protocol("conflicting duplicate computation completion")
            if state.cancelled:
                state.worker_pending = False
                self._release_if_done(state)
                return False
            if (type(result) is not TentativeResult or result.plan_key != state.plan.key or
                    result.wire_binding != state.plan.expected_wire or
                    result.base_history_sha256 != lease.base_history.sha256):
                self._fault("computed result identity/history binding mismatch")
                state.worker_pending = False
                state.cancelled = True
                self._release_if_done(state)
                raise ProtocolError("computed result identity/history binding mismatch")
            retained = (state.plan.owned_bytes + len(lease.base_history.payload)
                        + len(result.numerical_payload) + len(result.next_history.payload))
            if retained > state.plan.reserved_bytes or len(result.next_history.payload) > self._max_history_bytes:
                self._fault("computed result exceeded its byte reservation")
                state.worker_pending = False
                state.cancelled = True
                self._release_if_done(state)
                raise CapacityExceeded("computed result exceeded its byte reservation")
            state.worker_pending = False
            state.result = result
            return True

    def validate_computation(self, ticket: TransactionTicket,
                             validator: Callable[[TentativeResult], None]) -> bool:
        """Run the real adapter's admission validator, always outside the lock."""
        with self._condition:
            state = self._get(ticket)
            if state.cancelled:
                return False
            if state.validated or state.validation_pending:
                return False
            if state.result is None:
                self._protocol("validation precedes numerical completion")
            state.validation_pending = True
            result = state.result
        try:
            if validator(result) is not None:
                raise ValueError("validator must return None, never an already_validated boolean")
        except BaseException:
            with self._condition:
                state.validation_pending = False
                state.cancelled = True
                self._fault("numerical admission validation failed")
                self._release_if_done(state)
            raise
        with self._condition:
            state.validation_pending = False
            if state.cancelled:
                self._release_if_done(state)
                return False
            state.validated = True
            self._commit_if_ready(state)
            return True

    def confirm_send(self, ticket: TransactionTicket, binding: WireBinding) -> bool:
        with self._condition:
            state = self._get(ticket)
            if type(binding) is not WireBinding or binding != state.plan.expected_wire:
                self._protocol("send confirmation differs from the exact admitted plan/wire binding")
            if state.send_confirmed:
                return False
            if state.cancelled:
                self._protocol("send confirmation arrived after cancellation")
            state.send_confirmed = True
            self._commit_if_ready(state)
            return True

    def cancel(self, ticket: TransactionTicket) -> bool:
        """Cancel speculation; an active worker must still acknowledge release."""
        with self._condition:
            state = self._get(ticket)
            if state.committed_version is not None:
                self._protocol("a committed accepted frame cannot be cancelled")
            if state.cancelled:
                return False
            if state.send_confirmed:
                self._fault("sent accepted frame cancelled before a valid history commit")
            state.cancelled = True
            self._release_if_done(state)
            return True

    def acknowledge_cancelled(self, lease: ComputationLease) -> None:
        with self._condition:
            state = self._get(lease.ticket)
            if state.lease is not lease or not state.cancelled or not state.worker_pending:
                self._protocol("worker cancellation acknowledgement does not match an active cancelled lease")
            state.worker_pending = False
            self._release_if_done(state)

    def send_failed(self, ticket: TransactionTicket, reason: str) -> None:
        if type(reason) is not str:
            raise ValueError("failure reason must be a string")
        with self._condition:
            state = self._get(ticket)
            self._fault("send failed: " + reason[:1024])
            if state.send_confirmed or state.committed_version is not None:
                raise ProtocolError("send failure conflicts with prior successful confirmation")
            state.cancelled = True
            self._release_if_done(state)

    def worker_failed(self, lease: ComputationLease, reason: str) -> None:
        if type(reason) is not str:
            raise ValueError("failure reason must be a string")
        with self._condition:
            state = self._get(lease.ticket)
            if state.lease is not lease or not state.worker_pending:
                self._protocol("worker failure does not match an active lease")
            self._fault("numerical worker failed: " + reason[:1024])
            state.worker_pending = False
            state.cancelled = True
            self._release_if_done(state)

    def take_ready(self) -> Optional[ReadyPayload]:
        """Publish each committed result at most once, without invoking callbacks."""
        with self._condition:
            for state in self._states.values():
                if state.committed_version is not None and not state.ready_taken:
                    state.ready_taken = True
                    assert state.result is not None
                    return ReadyPayload(state.ticket, state.result, state.committed_version)
            return None

    def _ready_state(self, ticket: TransactionTicket) -> _Transaction:
        state = self._get(ticket)
        if not state.ready_taken:
            self._protocol("consumer completion precedes ready publication")
        return state

    def complete_scoring(self, ticket: TransactionTicket) -> bool:
        """Acknowledge scoring references released; no quality value is invented."""
        with self._condition:
            state = self._ready_state(ticket)
            if state.scoring_done:
                return False
            state.scoring_done = True
            self._release_if_done(state)
            return True

    def confirm_feedback(self, ticket: TransactionTicket) -> bool:
        with self._condition:
            state = self._ready_state(ticket)
            if not state.scoring_done:
                self._protocol("feedback confirmation precedes scoring completion")
            if state.feedback_terminal:
                if state.feedback_emitted:
                    return False
                self._protocol("feedback success conflicts with feedback failure")
            state.feedback_terminal = True
            state.feedback_emitted = True
            self._feedback_count += 1
            self._release_if_done(state)
            return True

    def scoring_failed(self, ticket: TransactionTicket, reason: str) -> None:
        if type(reason) is not str:
            raise ValueError("failure reason must be a string")
        with self._condition:
            state = self._ready_state(ticket)
            self._fault("scoring failed: " + reason[:1024])
            if state.scoring_done:
                raise ProtocolError("scoring failure conflicts with prior completion")
            state.scoring_done = True
            state.feedback_terminal = True
            self._release_if_done(state)

    def feedback_failed(self, ticket: TransactionTicket, reason: str) -> None:
        if type(reason) is not str:
            raise ValueError("failure reason must be a string")
        with self._condition:
            state = self._ready_state(ticket)
            self._fault("feedback failed: " + reason[:1024])
            if not state.scoring_done or state.feedback_emitted:
                raise ProtocolError("feedback failure conflicts with lifecycle state")
            state.feedback_terminal = True
            self._release_if_done(state)

    def complete_archive(self, ticket: TransactionTicket) -> bool:
        with self._condition:
            state = self._ready_state(ticket)
            if state.archive_done:
                if state.archive_succeeded:
                    return False
                self._protocol("archive success conflicts with archive failure")
            state.archive_done = True
            state.archive_succeeded = True
            self._release_if_done(state)
            return True

    def archive_failed(self, ticket: TransactionTicket, reason: str) -> None:
        if type(reason) is not str:
            raise ValueError("failure reason must be a string")
        with self._condition:
            # Even a late fault report for a retired frame invalidates the run;
            # retirement must not disguise loss of previously acknowledged data.
            if type(ticket) is not TransactionTicket or ticket._owner is not self._owner:
                self._protocol("foreign archive failure ticket")
            self._fault("archive failed: " + reason[:1024])
            state = self._get(ticket)
            if not state.ready_taken:
                raise ProtocolError("archive failure precedes ready publication")
            if state.archive_done and state.archive_succeeded:
                raise ProtocolError("archive failure conflicts with prior archive success")
            state.archive_done = True
            self._release_if_done(state)

    def close(self) -> None:
        """Stop admission while allowing every reserved transaction to drain."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wait_drained(self, timeout: float = 0.0) -> bool:
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("drain timeout must be finite and nonnegative")
        with self._condition:
            return self._condition.wait_for(lambda: not self._states, timeout=timeout)

    def snapshot(self) -> ControllerSnapshot:
        with self._condition:
            return ControllerSnapshot(self._first_failure is None, self._first_failure,
                                      self._failure_count, self._closed, len(self._states),
                                      self._reserved_bytes, self._history_version,
                                      self._feedback_count, self._released_count,
                                      self._history, self._history_version)

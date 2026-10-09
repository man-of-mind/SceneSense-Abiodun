"""Bounded offline target/prediction ownership handoff for registered identities.

This is an event join, not a prediction validator, scorer, transport, or history
controller. Targets must already carry the shared contract's admitted capability.
Predictions remain opaque immutable bytes: their checksum is checked here, while
the external scorer must still perform its complete record/numerical admission.

Reservations count registered identity bytes, two raw masks, metadata bytes, and
prediction bytes. They are payload-storage bounds, not a Python heap/RSS claim;
the admitted contract separately bounds decoded metadata. The driver must reserve
any additional worker/consumer storage in its transaction controller. No artifact
restoration, numerical target revalidation, or target rehashing happens here.

``release`` acknowledges that BOTH scoring and archive consumers relinquished
their references. Timeouts and close never evict accepted data. An incomplete or
failed frame can only be removed by explicit ``abandon``, which invalidates the
join. There are no tombstones: the owning monotonic transaction controller must
not re-register a retired full identity. Late arrivals cannot auto-register it.
"""

from dataclasses import dataclass
import hashlib
import math
import threading
from typing import Optional

from .contract import (
    ContractError, MASK_BYTES, MAX_METADATA_BYTES, NumericalTarget,
    is_digest, require_admitted,
)


class JoinError(RuntimeError):
    pass


class JoinProtocolError(JoinError):
    pass


class UnknownIdentity(JoinProtocolError):
    pass


class JoinCapacityExceeded(JoinError):
    pass


class JoinInvalid(JoinError):
    pass


class JoinClosed(JoinError):
    pass


class PairAlreadyTaken(JoinError):
    pass


@dataclass(frozen=True, slots=True)
class JoinedPair:
    identity_bytes: bytes
    target: NumericalTarget
    prediction_bytes: bytes
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class JoinSnapshot:
    run_valid: bool
    first_failure: Optional[str]
    failure_count: int
    closed: bool
    pending_jobs: int
    reserved_bytes: int
    retained_payload_bytes: int
    pairs_taken: int
    released_frames: int

    @property
    def drained(self) -> bool:
        return self.pending_jobs == 0


@dataclass(slots=True)
class _Slot:
    identity: bytes
    reserved_bytes: int
    target: Optional[NumericalTarget] = None
    prediction: Optional[bytes] = None
    prediction_sha256: Optional[str] = None
    taken: bool = False
    retained_bytes: int = 0


class BoundedTargetPredictionJoin:
    def __init__(self, *, max_jobs: int, max_bytes: int) -> None:
        for name, value in (("max_jobs", max_jobs), ("max_bytes", max_bytes)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._condition = threading.Condition(threading.Lock())
        self._slots: dict[bytes, _Slot] = {}
        self._reserved_bytes = 0
        self._retained_bytes = 0
        self._pairs_taken = 0
        self._released_frames = 0
        self._closed = False
        self._first_failure: Optional[str] = None
        self._failure_count = 0

    def _fault(self, message: str) -> None:
        self._failure_count += 1
        if self._first_failure is None:
            self._first_failure = message[:1024]
        self._condition.notify_all()

    def _reject(self, message: str) -> None:
        with self._condition:
            self._fault(message)
        raise JoinProtocolError(message)

    def _check_identity(self, identity: bytes) -> None:
        if type(identity) is not bytes or not 0 < len(identity) <= MAX_METADATA_BYTES:
            self._reject("complete identity must be bounded nonempty immutable bytes")

    def _healthy(self) -> None:
        if self._first_failure is not None:
            raise JoinInvalid(self._first_failure)

    def _open_for_arrivals(self) -> None:
        self._healthy()
        if self._closed:
            raise JoinClosed("join is closed to registrations and arrivals")

    def _get(self, identity: bytes) -> _Slot:
        slot = self._slots.get(identity)
        if slot is None:
            self._fault("unregistered or retired target/prediction identity")
            raise UnknownIdentity("unregistered or retired target/prediction identity")
        return slot

    def _reserve_payload(self, slot: _Slot, additional: int) -> None:
        if slot.retained_bytes + additional > slot.reserved_bytes:
            self._fault("target/prediction payload exceeded its slot reservation")
            raise JoinCapacityExceeded("target/prediction payload exceeded its slot reservation")
        slot.retained_bytes += additional
        self._retained_bytes += additional

    def register(self, identity: bytes, *, reserved_bytes: int) -> bool:
        """Register the coordinator's complete exact key before either arrival."""
        self._check_identity(identity)
        if type(reserved_bytes) is not int or reserved_bytes < len(identity) + 2 * MASK_BYTES:
            self._reject("reservation must include the identity and at least two raw masks")
        with self._condition:
            self._open_for_arrivals()
            existing = self._slots.get(identity)
            if existing is not None:
                if existing.reserved_bytes != reserved_bytes:
                    self._fault("conflicting duplicate identity reservation")
                    raise JoinProtocolError("conflicting duplicate identity reservation")
                return False
            if (len(self._slots) >= self._max_jobs or
                    self._reserved_bytes + reserved_bytes > self._max_bytes):
                self._fault("registered target/prediction join capacity exceeded")
                raise JoinCapacityExceeded("registered target/prediction join capacity exceeded; no eviction")
            self._slots[identity] = _Slot(identity, reserved_bytes, retained_bytes=len(identity))
            self._reserved_bytes += reserved_bytes
            self._retained_bytes += len(identity)
            return True

    def put_target(self, target: NumericalTarget) -> bool:
        """Transfer an admitted immutable target; never restore/revalidate masks."""
        try:
            target = require_admitted(target)
        except ContractError:
            self._reject("target arrival lacks the admitted numerical target capability")
        identity = target.identity_bytes
        self._check_identity(identity)
        owned_bytes = (len(target.metadata_bytes) + len(target.segmentation_bytes)
                       + len(target.object_ignore_bytes))
        with self._condition:
            self._open_for_arrivals()
            slot = self._get(identity)
            if slot.target is not None:
                prior = slot.target
                if (prior.semantic_sha256 != target.semantic_sha256 or
                        prior.metadata_bytes != target.metadata_bytes or
                        prior.segmentation_bytes != target.segmentation_bytes or
                        prior.object_ignore_bytes != target.object_ignore_bytes):
                    self._fault("conflicting duplicate target for the registered identity")
                    raise JoinProtocolError("conflicting duplicate target for the registered identity")
                return False
            self._reserve_payload(slot, owned_bytes)
            slot.target = target
            self._condition.notify_all()
            return True

    def put_prediction(self, identity: bytes, payload: bytes, supplied_sha256: str) -> bool:
        """Transfer checked immutable bytes, WITHOUT claiming numerical admission."""
        self._check_identity(identity)
        if type(payload) is not bytes or len(payload) > self._max_bytes:
            self._reject("prediction must be bounded immutable bytes")
        if not is_digest(supplied_sha256) or hashlib.sha256(payload).hexdigest() != supplied_sha256:
            self._reject("prediction payload checksum mismatch")
        with self._condition:
            self._open_for_arrivals()
            slot = self._get(identity)
            if slot.prediction is not None:
                if slot.prediction_sha256 != supplied_sha256 or slot.prediction != payload:
                    self._fault("conflicting duplicate prediction for the registered identity")
                    raise JoinProtocolError("conflicting duplicate prediction for the registered identity")
                return False
            self._reserve_payload(slot, len(payload))
            slot.prediction = payload
            slot.prediction_sha256 = supplied_sha256
            self._condition.notify_all()
            return True

    def wait_take(self, identity: bytes, *, timeout: Optional[float] = None) -> Optional[JoinedPair]:
        """Condition wait for both sides; timeout returns None without eviction."""
        self._check_identity(identity)
        if timeout is not None and (type(timeout) not in (int, float) or
                                    not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be None or finite and nonnegative")
        with self._condition:
            self._healthy()
            slot = self._get(identity)

            def ready_or_terminal() -> bool:
                return (slot.taken or self._closed or self._first_failure is not None or
                        self._slots.get(identity) is not slot or
                        (slot.target is not None and slot.prediction is not None))

            if not self._condition.wait_for(ready_or_terminal, timeout=timeout):
                return None
            self._healthy()
            if self._slots.get(identity) is not slot:
                self._fault("registered identity was explicitly abandoned while waiting")
                raise UnknownIdentity("registered identity retired while waiting")
            if slot.taken:
                raise PairAlreadyTaken("the registered target/prediction pair was already taken")
            if slot.target is not None and slot.prediction is not None:
                slot.taken = True
                self._pairs_taken += 1
                self._condition.notify_all()
                assert slot.prediction_sha256 is not None
                return JoinedPair(slot.identity, slot.target, slot.prediction, slot.prediction_sha256)
            raise JoinClosed("join closed before both registered inputs arrived; reservation retained")

    def _drop(self, slot: _Slot) -> None:
        del self._slots[slot.identity]
        self._reserved_bytes -= slot.reserved_bytes
        self._retained_bytes -= slot.retained_bytes
        self._released_frames += 1
        self._condition.notify_all()

    def release(self, identity: bytes) -> None:
        """Acknowledge both scoring AND archive consumers released their refs."""
        self._check_identity(identity)
        with self._condition:
            slot = self._get(identity)
            if not slot.taken:
                self._fault("cannot release an untaken accepted pair without explicit abandonment")
                raise JoinProtocolError("use explicit abandonment for an untaken accepted pair")
            self._drop(slot)

    def abandon(self, identity: bytes, reason: str) -> None:
        """Explicitly relinquish all consumer refs on failure; never a valid run."""
        self._check_identity(identity)
        if type(reason) is not str or not reason:
            raise ValueError("abandonment requires an explicit string reason")
        with self._condition:
            slot = self._get(identity)
            self._fault("accepted target/prediction pair abandoned: " + reason[:1024])
            self._drop(slot)

    def close(self) -> None:
        """Wake waiters and stop arrivals; preserve reservations and joined data."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self) -> JoinSnapshot:
        with self._condition:
            return JoinSnapshot(self._first_failure is None, self._first_failure,
                                self._failure_count, self._closed, len(self._slots),
                                self._reserved_bytes, self._retained_bytes,
                                self._pairs_taken, self._released_frames)

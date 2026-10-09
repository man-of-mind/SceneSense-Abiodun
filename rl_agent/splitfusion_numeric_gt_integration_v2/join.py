"""Bounded event-driven join of actual immutable v2 target/prediction capabilities.

The coordinator registers the complete key before either arrival. This is an
in-process condition, not live IPC or a substitute numerical trust boundary.
Predictions must come from the exact scorer's original admission factory.
Serialized-payload accounting is explicit; it is not Python heap/RSS accounting.
Release acknowledges all downstream references have been relinquished. Neither
timeouts nor close discard accepted data. No implicit registration or eviction.
"""
from dataclasses import dataclass
import math
import threading

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S, shared_admission as P
from . import ready as R


class JoinError(RuntimeError):
    pass


class JoinProtocolError(JoinError):
    pass


class JoinCapacityExceeded(JoinError):
    pass


class JoinClosed(JoinError):
    pass


class JoinInvalid(JoinError):
    pass


class PairAlreadyTaken(JoinError):
    pass


@dataclass(frozen=True, slots=True)
class JoinedPair:
    identity_bytes: bytes
    target: R.ReadyTarget
    prediction: object


@dataclass
class _Slot:
    identity: bytes
    document: dict
    reservation: int
    payload_bytes: int
    target: object = None
    prediction: object = None
    taken: bool = False


class BoundedTargetPredictionJoin:
    def __init__(self, scorer, *, max_jobs, max_bytes):
        if type(scorer) is not S.NumericalScorer:
            raise JoinProtocolError("actual numerical scorer context required")
        if (type(max_jobs) is not int or not 1 <= max_jobs <= 1000
                or type(max_bytes) is not int or max_bytes <= 0):
            raise JoinProtocolError("positive bounded join capacity required")
        self._owner = scorer._owner
        self._max_jobs, self._max_bytes = max_jobs, max_bytes
        self._condition = threading.Condition(threading.Lock())
        self._slots = {}
        self._reserved = self._retained = self._taken = self._released = 0
        self._failure = None
        self._closed = False
        self._last_frame = -1
        self._stream = None

    def _fault(self, message, error=JoinProtocolError):
        self._failure = self._failure or str(message)[:1024]
        self._condition.notify_all()
        raise error(str(message))

    def _open(self):
        if self._failure is not None:
            raise JoinInvalid(self._failure)
        if self._closed:
            raise JoinClosed("join closed to new registrations and arrivals")

    def _slot(self, identity):
        if type(identity) is not bytes or identity not in self._slots:
            self._fault("unregistered or retired full join identity")
        return self._slots[identity]

    def register(self, identity_bytes, *, reserved_bytes):
        with self._condition:
            self._open()
            try:
                document = C._parse(identity_bytes)
                if C.canonical(document) != identity_bytes or set(document) != {
                        "capture_identity", "action_identity", "plan_sha256", "run_binding",
                        "producer_binding", "frame_context", "prediction_contract"}:
                    raise JoinProtocolError("complete canonical registered join identity required")
                C._capture(document["capture_identity"])
                R._action(document["action_identity"], document["capture_identity"])
                if not all(C.is_digest(document[k]) for k in ("plan_sha256", "run_binding", "producer_binding")):
                    raise JoinProtocolError("registered binding digests missing")
                if type(reserved_bytes) is not int or reserved_bytes < len(identity_bytes) + 2 * C.MASK_BYTES:
                    raise JoinProtocolError("reservation must include identity and both target masks")
            except Exception as error:
                self._fault("invalid join registration: " + str(error))
            old = self._slots.get(identity_bytes)
            if old is not None:
                if old.reservation != reserved_bytes:
                    self._fault("conflicting duplicate join reservation")
                return False
            capture, action = document["capture_identity"], document["action_identity"]
            frame = capture["frame_id"]
            # An explicitly registered new session may coexist with a prior
            # session draining after undefined quality. Its full action UUID
            # remains part of the per-slot key and every prediction comparison.
            stream = tuple(capture[k] for k in ("run_id", "cell_id", "stream_id")) + (
                action["controller_lineage_sha256"], document["run_binding"],
                document["producer_binding"])
            if frame <= self._last_frame or (self._stream is not None and stream != self._stream):
                self._fault("regressing/reused frame or foreign registered stream")
            if len(self._slots) >= self._max_jobs or self._reserved + reserved_bytes > self._max_bytes:
                self._fault("join capacity exceeded; no eviction", JoinCapacityExceeded)
            self._slots[identity_bytes] = _Slot(identity_bytes, document, reserved_bytes, len(identity_bytes))
            self._reserved += reserved_bytes
            self._retained += len(identity_bytes)
            self._last_frame, self._stream = frame, stream
            return True

    def _account(self, slot, additional):
        if slot.payload_bytes + additional > slot.reservation:
            self._fault("join payload exceeds registered reservation", JoinCapacityExceeded)
        slot.payload_bytes += additional
        self._retained += additional

    def put_target(self, target):
        with self._condition:
            self._open()
            try:
                R.require_ready(target)
            except Exception as error:
                self._fault("target lacks actual ready admission: " + str(error))
            slot = self._slot(target.identity_bytes)
            if slot.target is not None:
                if (slot.target.metadata_bytes != target.metadata_bytes
                        or slot.target.segmentation_bytes != target.segmentation_bytes
                        or slot.target.object_ignore_bytes != target.object_ignore_bytes
                        or slot.target.semantic_sha256 != target.semantic_sha256):
                    self._fault("conflicting duplicate target")
                return False
            self._account(slot, len(target.metadata_bytes) + len(target.segmentation_bytes)
                          + len(target.object_ignore_bytes))
            slot.target = target
            self._condition.notify_all()
            return True

    def put_prediction(self, identity_bytes, prediction):
        with self._condition:
            self._open()
            if (type(prediction) is not P._AdmittedPrediction or prediction.issuer is not P._PREDICTION_ISSUER
                    or prediction.owner is not self._owner):
                self._fault("prediction lacks this scorer's actual admission capability")
            slot = self._slot(identity_bytes)
            identity = prediction.identity
            if (C.canonical(identity["gt_identity"]) != C.canonical(slot.document["capture_identity"])
                    or C.canonical(identity["action_identity"]) != C.canonical(slot.document["action_identity"])
                    or C.canonical(prediction.contract.document()) != C.canonical(slot.document["prediction_contract"])):
                self._fault("prediction full identity/calibration differs from registered plan")
            if slot.prediction is not None:
                prior = slot.prediction
                if (prior.labels != prediction.labels or prior.raw_records != prediction.raw_records
                        or C.canonical(prior.identity) != C.canonical(prediction.identity)):
                    self._fault("conflicting duplicate prediction")
                return False
            size = (len(prediction.labels) + len(prediction.raw_records) + len(C.canonical(identity))
                    + len(C.canonical(prediction.contract.document())))
            self._account(slot, size)
            slot.prediction = prediction
            self._condition.notify_all()
            return True

    def wait_take(self, identity_bytes, *, timeout=None):
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("finite nonnegative timeout or None required")
        with self._condition:
            if self._failure is not None:
                raise JoinInvalid(self._failure)
            slot = self._slot(identity_bytes)
            def terminal():
                return (self._failure is not None or self._closed or slot.taken
                    or self._slots.get(identity_bytes) is not slot
                    or (slot.target is not None and slot.prediction is not None))
            if not self._condition.wait_for(terminal, timeout=timeout):
                return None
            if self._failure is not None:
                raise JoinInvalid(self._failure)
            if self._slots.get(identity_bytes) is not slot:
                self._fault("join identity retired while waiting")
            if slot.taken:
                raise PairAlreadyTaken("joined inputs have already been taken")
            if slot.target is not None and slot.prediction is not None:
                slot.taken = True
                self._taken += 1
                self._condition.notify_all()
                return JoinedPair(identity_bytes, slot.target, slot.prediction)
            raise JoinClosed("join closed before both inputs arrived; reservation retained")

    def _drop(self, slot):
        del self._slots[slot.identity]
        self._reserved -= slot.reservation
        self._retained -= slot.payload_bytes
        self._released += 1
        self._condition.notify_all()

    def release(self, identity_bytes):
        """Coordinator confirms scoring/feedback/archive ownership is terminal."""
        with self._condition:
            slot = self._slot(identity_bytes)
            if not slot.taken:
                self._fault("cannot release an untaken pair; explicit abandonment required")
            self._drop(slot)

    def abandon(self, identity_bytes, reason):
        with self._condition:
            slot = self._slot(identity_bytes)
            if type(reason) is not str or not reason:
                self._fault("explicit abandonment reason required")
            self._failure = self._failure or ("join abandoned: " + reason)[:1024]
            self._drop(slot)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return dict(valid=self._failure is None, failure=self._failure, closed=self._closed,
                pending_jobs=len(self._slots), reserved_bytes=self._reserved,
                retained_payload_bytes=self._retained, pairs_taken=self._taken, released_frames=self._released,
                accounting="SERIALIZED_PAYLOAD_BYTES_NOT_PYTHON_RSS", live_ipc=False)

"""Actual producer transactions and bounded, single-worker artifact ownership.

This is an in-process integration primitive, not live IPC. One owner holds the
process lease because the sealed producer temporarily modifies shared helper
namespaces. Every operation, including deferred artifact construction, belongs
to its creating thread. Private assembly access is a trusted coordinator seam;
Staged and numerical products must never escape to another thread. Public
consumers receive only the separately admitted immutable ReadyTarget.

Reservations account for frame-owned logical payload storage, not Python RSS or
transient numerical scratch. No history clone is made here: the real producer's
commit/cancel protocol owns its tracker and conflict transaction.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
import threading
from types import FunctionType
import weakref

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v1.transactions import PlanKind
from rl_agent.splitfusion_numeric_gt_producer_v1.source import NumericSourceProducer


_PROCESS_OWNER = threading.Lock()
_INVALID_PRODUCERS = weakref.WeakSet()


class OwnershipError(RuntimeError):
    pass


class CapacityError(OwnershipError):
    pass


@dataclass(frozen=True, slots=True)
class ProducerToken:
    """Opaque identity; possessing a lookalike token conveys no ownership."""
    sequence: int
    _owner: object


@dataclass
class _Frame:
    token: ProducerToken
    plan: bytes
    action: dict
    capture: dict
    reservation: int
    context: dict = None
    staged: object = None
    products: tuple = None
    ready: object = None
    payload_bytes: int = 0
    send_confirmed: bool = False
    producer_terminal: str = None
    commit_result: dict = None
    scoring_done: bool = False
    feedback_terminal: bool = False
    archive_terminal: bool = False


def payload_storage_bytes(*values):
    """Conservative logical storage: buffers, UTF-8, scalars and framing.

    Shared containers are counted once, ndarray views conservatively count their
    visible bytes. Includes Staged rollback deltas and closure-owned class masks.
    It deliberately makes no claim about allocator/object overhead or peak RSS.
    """
    seen = set()

    def size(value):
        if value is None:
            return 1
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return 8 + (len(value.encode("utf-8")) if isinstance(value, str) else
                        value.nbytes if isinstance(value, memoryview) else len(value))
        if isinstance(value, (bool, int, float, np.generic)):
            return 16
        ident = id(value)
        if ident in seen:
            return 0
        seen.add(ident)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise OwnershipError("object ndarray cannot be retained")
            return 32 + value.nbytes
        if isinstance(value, Mapping):
            return 16 + sum(8 + size(k) + size(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            return 16 + sum(8 + size(v) for v in value)
        if is_dataclass(value):
            return 16 + sum(size(getattr(value, f.name)) for f in fields(value))
        if isinstance(value, FunctionType):
            # Staged.same_class owns a dict of image masks and semantic ndarray.
            return 16 + sum(size(c.cell_contents) for c in (value.__closure__ or ()))
        raise OwnershipError("unsupported retained payload type: " + type(value).__name__)

    return sum(size(v) for v in values)


class ProducerOwner:
    def __init__(self, producer, *, max_retained_jobs, max_retained_bytes):
        if type(producer) is not NumericSourceProducer:
            raise OwnershipError("actual NumericSourceProducer required")
        if producer in _INVALID_PRODUCERS:
            raise OwnershipError("failed producer cannot be rebound to a fresh owner")
        if (type(max_retained_jobs) is not int or not 1 <= max_retained_jobs <= 1000
                or type(max_retained_bytes) is not int or max_retained_bytes <= 0):
            raise OwnershipError("positive bounded retention limits required")
        if producer.staged is not None:
            raise OwnershipError("producer already has an unowned transaction")
        if not _PROCESS_OWNER.acquire(blocking=False):
            raise OwnershipError("another process-wide producer owner is active")
        self._producer = producer
        self._thread = threading.current_thread()
        self._issuer = object()
        self._max_jobs, self._max_bytes = max_retained_jobs, max_retained_bytes
        self._frames = {}
        self._pending = None
        self._sequence = self._reserved = self._released = 0
        self._commits = self._cancels = 0
        self._last_frame = producer.previous_frame
        self._failure = None
        self._closed = False

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None and self._failure is None:
            self._failure = f"owner scope failed: {type(error).__name__}: {error}"[:1024]
        self.close(abandon=error is not None)

    def _thread_check(self):
        if threading.current_thread() is not self._thread:
            # Do not touch the producer from the foreign thread, even to cancel.
            self._failure = self._failure or "producer access from a foreign thread"
            raise OwnershipError(self._failure)

    def _check(self, *, healthy=True):
        self._thread_check()
        if self._closed:
            raise OwnershipError("producer owner is closed")
        if healthy and self._failure is not None:
            raise OwnershipError(self._failure)

    def _fail(self, message, exc_type=OwnershipError):
        self._failure = self._failure or str(message)[:1024]
        raise exc_type(str(message))

    def _frame(self, token, *, healthy=True):
        self._check(healthy=healthy)
        if (type(token) is not ProducerToken or token._owner is not self._issuer
                or self._frames.get(token.sequence) is None
                or self._frames[token.sequence].token is not token):
            self._fail("foreign or retired producer token")
        return self._frames[token.sequence]

    def register_population(self, ids, evidence_sha256, *, before_frame_id):
        self._check()
        if self._pending is not None:
            self._fail("population registration while computation is pending")
        try:
            self._producer.register_population(ids, evidence_sha256, before_frame_id)
        except BaseException as error:
            self._failure = self._failure or f"population registration failed: {error}"[:1024]
            raise

    @property
    def population(self):
        self._check()
        return self._producer.population

    def compute(self, plan_identity_bytes, sourcekwargs, *, reserved_bytes,
                kind=PlanKind.ACCEPTED):
        self._check()
        if kind is not PlanKind.ACCEPTED:
            raise OwnershipError("hold/fallback/displaced work cannot enter the producer")
        if self._pending is not None:
            self._fail("another real producer transaction is pending")
        try:
            plan = C._parse(plan_identity_bytes)
            if C.canonical(plan) != plan_identity_bytes or not isinstance(plan, dict):
                raise OwnershipError("canonical full plan identity bytes required")
            capture, action = plan["capture"], plan["action"]
            C._capture(capture)
            if type(plan.get("frame_context")) is not dict:
                raise OwnershipError("full immutable plan must include its frame context")
            from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
            MP.validate_run4_identity(action)
            if action["reward_requested"] is not True or action["frame_id"] != capture["frame_id"]:
                raise OwnershipError("only accepted reward-requested frame plans are admitted")
            if C.canonical(capture) != C.canonical(sourcekwargs["identity"]):
                raise OwnershipError("source identity differs from the full plan")
            frame_id = capture["frame_id"]
            if type(frame_id) is not int or frame_id <= self._last_frame:
                raise OwnershipError("regressing or reused producer frame")
            if type(reserved_bytes) is not int or reserved_bytes <= len(plan_identity_bytes):
                raise OwnershipError("explicit frame payload reservation required")
            if len(self._frames) >= self._max_jobs or self._reserved + reserved_bytes > self._max_bytes:
                self._fail("producer retention capacity exceeded; no eviction", CapacityError)
        except BaseException as error:
            self._failure = self._failure or f"invalid producer plan: {error}"[:1024]
            raise
        self._sequence += 1
        token = ProducerToken(self._sequence, self._issuer)
        row = _Frame(token, plan_identity_bytes, action, capture, reserved_bytes,
                     context=plan["frame_context"])
        self._frames[token.sequence] = row
        self._reserved += reserved_bytes
        self._pending = row
        try:
            # These are the source producer's retained caller aliases. Native
            # image buffers are decoded/copied by its own compute implementation.
            owned = dict(sourcekwargs)
            for name in ("camera_matrix", "camera_inverse", "radar_world_xyz"):
                value = np.asarray(owned[name])
                if value.dtype.hasobject:
                    raise OwnershipError("object source array is forbidden")
                owned[name] = np.frombuffer(value.tobytes(order="C"), value.dtype).reshape(value.shape)
            for name in ("identity", "dynamic_snapshot_evidence"):
                owned[name] = copy.deepcopy(owned[name])
            row.staged = self._producer.compute(**owned)
            if row.staged is not self._producer.staged:
                self._fail("producer returned a foreign staged transaction")
            self._account(row)
            self._last_frame = frame_id
            return token
        except BaseException as error:
            self._failure = self._failure or f"producer compute failed: {type(error).__name__}: {error}"[:1024]
            if row.staged is not None and row.producer_terminal is None:
                self._cancel_real(row)
            self._discard(row)
            raise

    def _account(self, row):
        row.payload_bytes = payload_storage_bytes(row.plan, row.staged, row.products)
        if row.ready is not None:
            row.payload_bytes += (len(row.ready.metadata_bytes) + len(row.ready.segmentation_bytes)
                                  + len(row.ready.object_ignore_bytes))
        if row.payload_bytes > row.reservation:
            self._fail("frame payload exceeds its reserved storage", CapacityError)

    def _staged_for_assembly(self, token):
        """Trusted coordinator only; borrow synchronously, never retain elsewhere."""
        row = self._frame(token)
        if row.producer_terminal is not None or row.ready is not None:
            self._fail("staged assembly access after admission or terminal")
        return row.staged

    def stage_products(self, token, *, target, avo, payload):
        row = self._frame(token)
        if row.products is not None or row.ready is not None:
            self._fail("numerical products already transferred")
        from rl_agent.splitfusion_numeric_gt_producer_v1 import payload as P
        try:
            P.verify_payload(payload)
            if (payload.capture_identity != row.capture or payload.sample_id != row.staged.sid
                    or payload.history_before_sha256 != row.staged.history_before_sha256
                    or payload.history_after_sha256 != row.staged.history_after_sha256
                    or payload.input_content_sha256 != row.staged.inputs.provenance["input_content_sha256"]
                    or avo.frame_id != row.staged.frame_id):
                self._fail("numerical products differ from owned producer transaction")
            if P.A.canonical(P.json_value(avo.avo)) != C.canonical(payload.avo):
                self._fail("retained numerical AVO differs from its ready payload")
            if (not np.array_equal(target.segmentation, payload.segmentation)
                    or not np.array_equal(target.object_ignore, payload.object_ignore)):
                self._fail("numerical target differs from its payload")
            row.products = (target, avo, payload)
            self._account(row)
        except BaseException as error:
            self._failure = self._failure or f"numerical product admission failed: {error}"[:1024]
            raise

    def set_ready(self, token, ready):
        row = self._frame(token)
        from .ready import require_ready
        try:
            require_ready(ready)
            if row.ready is not None or row.products is None:
                self._fail("ready admission requires one owned numerical product set")
            metadata = ready.metadata
            payload = row.products[2]
            checks = (("capture_identity", row.capture), ("action_identity", row.action),
                ("frame_context", row.context),
                ("plan_sha256", hashlib.sha256(row.plan).hexdigest()),
                ("history_before_sha256", row.staged.history_before_sha256),
                ("history_after_sha256", row.staged.history_after_sha256),
                ("input_content_sha256", payload.input_content_sha256),
                ("producer_semantic_sha256", payload.semantic_sha256),
                ("producer_binding", payload.binding_sha256))
            if any(C.canonical(metadata[k]) != C.canonical(v) for k, v in checks):
                self._fail("admitted ready target differs from the full producer plan")
            if (ready.segmentation_bytes != payload.segmentation_bytes
                    or ready.object_ignore_bytes != payload.object_ignore_bytes):
                self._fail("admitted ready masks differ from owned numerical products")
            row.ready = ready
            self._account(row)
            return self._maybe_commit(row)
        except BaseException as error:
            self._failure = self._failure or f"ready admission failed: {error}"[:1024]
            raise

    def confirm_send(self, token, plan_identity_bytes):
        row = self._frame(token)
        if type(plan_identity_bytes) is not bytes or plan_identity_bytes != row.plan:
            self._fail("send confirmation differs from the exact full plan")
        if row.send_confirmed:
            return row.commit_result
        if row.producer_terminal is not None:
            self._fail("send confirmation after cancellation")
        row.send_confirmed = True
        return self._maybe_commit(row)

    def _maybe_commit(self, row):
        if row.ready is None or not row.send_confirmed:
            return None
        if row.producer_terminal is not None:
            return row.commit_result
        # Mark attempted BEFORE calling: the sealed implementation clears its
        # pending pointer before checking its final digest. Never retry it.
        row.producer_terminal = "COMMIT_ATTEMPTED"
        self._commits += 1
        try:
            row.commit_result = self._producer.commit(row.staged)
            if row.commit_result["history_after_sha256"] != row.ready.metadata["history_after_sha256"]:
                self._fail("producer commit differs from admitted numerical history")
            row.producer_terminal = "COMMITTED"
            self._pending = None
            return dict(row.commit_result)
        except BaseException as error:
            self._failure = self._failure or f"producer commit failed: {error}"[:1024]
            raise

    def _cancel_real(self, row):
        if row.producer_terminal is not None:
            self._fail("cannot cancel a terminal or attempted producer transaction")
        row.producer_terminal = "CANCEL_ATTEMPTED"
        self._cancels += 1
        try:
            result = self._producer.cancel(row.staged)
            row.producer_terminal = "CANCELLED"
            self._pending = None
            return result
        except BaseException as error:
            self._failure = self._failure or f"producer cancellation failed: {error}"[:1024]
            raise

    def cancel(self, token, reason):
        row = self._frame(token, healthy=False)
        if type(reason) is not str or not reason:
            self._fail("cancellation needs an explicit reason")
        if row.send_confirmed:
            self._failure = self._failure or "sent transaction cancelled before numerical readiness"
        result = self._cancel_real(row)
        # No consumer can acquire a ReadyTarget before commit through this API.
        self._discard(row)
        return result

    def ready_target(self, token):
        row = self._frame(token)
        if row.producer_terminal != "COMMITTED":
            self._fail("numerical target publication precedes confirmed-send commit")
        return row.ready

    def _archive_for_owner(self, token):
        """Private same-worker reconstruction borrow; no cross-thread handoff."""
        row = self._frame(token, healthy=False)
        if row.producer_terminal != "COMMITTED" or row.archive_terminal:
            self._fail("archive reconstruction outside committed owned lifetime")
        return (row.staged, *row.products)

    def complete_scoring(self, token):
        row = self._committed(token)
        row.scoring_done = True
        self._release_if_done(row)

    def confirm_feedback(self, token):
        row = self._committed(token)
        if not row.scoring_done:
            self._fail("feedback confirmation precedes scoring completion")
        row.feedback_terminal = True
        self._release_if_done(row)

    def resolve_live_scored_terminal(self, token, *, terminal):
        """Release a scored frame after the coordinator verifies durable live proof.

        This records a controller terminal, not a feedback-receipt claim. The
        producer runtime owns proof validation before calling this method.
        """
        row = self._committed(token)
        if terminal not in ("FEEDBACK_ON_TIME", "FEEDBACK_TIMEOUT") or not row.scoring_done:
            self._fail("live scored terminal requires actual scoring and known UE outcome")
        if row.feedback_terminal or row.archive_terminal:
            self._fail("live scored terminal already resolved or mixed with offline completion")
        row.feedback_terminal = row.archive_terminal = True
        self._release_if_done(row)

    def resolve_live_unscored_terminal(self, token, *, terminal):
        """Release a durable committed no-score terminal without poisoning history.

        No scoring, feedback, or archive bit is synthesized. The producer
        runtime must validate a durable terminal sidecar before invoking this.
        """
        row = self._committed(token)
        if terminal not in ("FEEDBACK_TIMEOUT", "FEEDBACK_ON_TIME"):
            self._fail("unknown unscored controller terminal")
        if row.scoring_done or row.feedback_terminal or row.archive_terminal:
            self._fail("unscored terminal conflicts with a scored or resolved frame")
        self._discard(row)

    def resolve_live_unscored_timeout(self, token):
        """Compatibility entry point for a durable ordinary no-score timeout."""
        self.resolve_live_unscored_terminal(token, terminal="FEEDBACK_TIMEOUT")

    def complete_archive(self, token):
        row = self._committed(token)
        row.archive_terminal = True
        self._release_if_done(row)

    def consumer_failed(self, token, *, consumer, reason):
        row = self._committed(token)
        if consumer not in ("scoring", "feedback", "archive") or type(reason) is not str or not reason:
            self._fail("explicit known consumer failure required")
        self._failure = self._failure or f"{consumer} failed: {reason}"[:1024]
        if consumer == "scoring":
            row.scoring_done = row.feedback_terminal = True
        elif consumer == "feedback":
            if not row.scoring_done:
                self._fail("feedback failure precedes scoring completion")
            row.feedback_terminal = True
        else:
            row.archive_terminal = True
        self._release_if_done(row)

    def _committed(self, token):
        row = self._frame(token, healthy=False)
        if row.producer_terminal != "COMMITTED":
            self._fail("consumer completion precedes producer commit")
        return row

    def _release_if_done(self, row):
        if row.scoring_done and row.feedback_terminal and row.archive_terminal:
            self._discard(row)

    def _discard(self, row):
        self._frames.pop(row.token.sequence)
        self._reserved -= row.reservation
        self._released += 1
        if self._pending is row:
            self._pending = None
        row.staged = row.products = row.ready = None

    def snapshot(self):
        self._thread_check()
        return dict(valid=self._failure is None, failure=self._failure, closed=self._closed,
            retained_jobs=len(self._frames), reserved_bytes=self._reserved,
            payload_bytes=sum(r.payload_bytes for r in self._frames.values()),
            max_retained_jobs=self._max_jobs, max_retained_bytes=self._max_bytes,
            commit_calls=self._commits, cancel_calls=self._cancels, released=self._released,
            accounting="LOGICAL_PAYLOAD_STORAGE_NOT_RSS_OR_PEAK_SCRATCH", live_ipc=False)

    def close(self, *, abandon=False):
        self._thread_check()
        if self._closed:
            return
        cleanup_error = None
        if self._pending is not None and self._pending.producer_terminal is None:
            row = self._pending
            if row.send_confirmed:
                self._failure = self._failure or "sent transaction closed before numerical readiness"
            try:
                self._cancel_real(row)
                self._discard(row)
            except BaseException as error:
                cleanup_error = error
                if not abandon:
                    raise
        if self._frames:
            self._failure = self._failure or "owner closed before scoring/feedback/archive terminal"
            if not abandon:
                # Refusal keeps the lease and API alive so consumers can drain
                # or the coordinator can explicitly relinquish failed work.
                raise OwnershipError(self._failure)
            for row in tuple(self._frames.values()):
                self._discard(row)
        self._closed = True
        if self._failure is not None:
            _INVALID_PRODUCERS.add(self._producer)
        _PROCESS_OWNER.release()
        if cleanup_error is not None:
            raise cleanup_error

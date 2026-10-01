"""W10275 HIGH-worker sender for the split-host Run-4 GT transport.

The integration wraps the writer pair *after* ``GtWriteRecorderV2.wrap``.
Semantic writes only publish their two returned paths into a bounded in-memory
join.  The object writer resolves the full reward-ticket identity and performs
the sole network send, guarded by an explicit existing-HIGH-worker predicate.

Importing this module opens no socket, starts no thread and performs no I/O.
The live owner must call :meth:`connect` and :meth:`close` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable, Mapping, Optional

from . import contract as C
from . import gt_transport as GT
from . import remote_edge_gt_entry_v1 as RGT


REGISTERED_HOST = C.default_topology().edge_ip
REGISTERED_PORT = RGT.REGISTERED_GT_PORT
MAX_PENDING_IDENTITIES = RGT.MAX_GT_TICKETS
GT_IDENTITY_FIELDS = {
    "run_id", "cell_id", "stream_id", "frame_id", "action_id", "profile_id",
    "capture_timestamp_ns",
}
MAX_COMPONENT_WAIT_S = 1.0


class GtSenderIntegrationError(RuntimeError):
    pass


class WrongWorkerError(GtSenderIntegrationError):
    pass


class SenderLifecycleError(GtSenderIntegrationError):
    pass


class SenderDeliveryError(GtSenderIntegrationError):
    pass


def _require(condition: bool, message: str,
             error: type[GtSenderIntegrationError] = GtSenderIntegrationError) -> None:
    if not condition:
        raise error(message)


def identity_from_ue_maps(*, gt_identity: Mapping[str, Any],
                          run4_identity: Mapping[str, Any]) -> Optional[GT.GtTransportIdentityV1]:
    """Resolve the full ticket from the two maps already held by the UE runtime.

    LOW/hold/fallback frames return ``None``.  A reward frame is accepted only
    when all shared frame/action fields agree.  No catalog identity is invented
    for an off-anchor action.
    """
    _require(set(gt_identity) == GT_IDENTITY_FIELDS, "GT identity field-set drift")
    required = {
        "session_uuid", "controller_lineage_sha256", "decision_seq", "ticket_seq",
        "frame_id", "tensor_seq", "anchor_action_id", "reward_requested", "frame_kind",
    }
    _require(required <= set(run4_identity), "Run-4 ticket identity is incomplete")
    if run4_identity["reward_requested"] is not True:
        _require(run4_identity["frame_kind"] in {"POLICY_HOLD", "FALLBACK"},
                 "non-reward frame kind drift")
        return None
    _require(run4_identity["frame_kind"] == "POLICY_DECISION",
             "reward ticket is not a policy decision")
    _require(int(run4_identity["frame_id"]) == int(gt_identity["frame_id"]),
             "GT/Run-4 frame identity drift")
    _require(run4_identity["anchor_action_id"] == gt_identity["action_id"],
             "GT/Run-4 anchor identity drift")
    return GT.GtTransportIdentityV1(
        run_id=str(gt_identity["run_id"]), cell_id=str(gt_identity["cell_id"]),
        stream_id=str(gt_identity["stream_id"]),
        session_uuid=str(run4_identity["session_uuid"]),
        controller_lineage_sha256=str(run4_identity["controller_lineage_sha256"]),
        decision_seq=int(run4_identity["decision_seq"]),
        ticket_seq=int(run4_identity["ticket_seq"]),
        frame_id=int(gt_identity["frame_id"]), tensor_seq=int(run4_identity["tensor_seq"]),
        capture_timestamp_ns=int(gt_identity["capture_timestamp_ns"]),
        action_id=gt_identity["action_id"], profile_id=gt_identity["profile_id"],
    )


def _gt_key(identity: Mapping[str, Any]) -> str:
    _require(set(identity) == GT_IDENTITY_FIELDS, "GT identity field-set drift")
    try:
        payload = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GtSenderIntegrationError("GT identity is not canonicalizable") from exc
    import hashlib

    return hashlib.sha256(payload).hexdigest()


@dataclass
class _PendingPaths:
    gt_identity: dict[str, Any]
    paths: dict[str, Path] = field(default_factory=dict)
    ack: Optional[GT.GtAckV1] = None
    sending: bool = False


class HighWorkerGtSenderV1:
    """Join writer results and send exactly from the existing HIGH GT worker."""

    def __init__(
        self, *,
        resolve_high_identity: Callable[[Mapping[str, Any]],
                                        Optional[GT.GtTransportIdentityV1]],
        is_high_worker: Callable[[], bool],
        host: str = REGISTERED_HOST, port: int = REGISTERED_PORT,
        connect_timeout_s: float = 2.0, ack_timeout_s: float = 2.0,
        component_wait_s: float = 0.25,
        connector: Callable[[tuple[str, int], float], Any] = socket.create_connection,
    ) -> None:
        _require((str(host), int(port)) == (REGISTERED_HOST, REGISTERED_PORT),
                 "sender endpoint is not the registered L10319 GT endpoint")
        for value, label, maximum in (
            (connect_timeout_s, "connect", GT.MAX_SOCKET_TIMEOUT_S),
            (ack_timeout_s, "ACK", GT.MAX_SOCKET_TIMEOUT_S),
            (component_wait_s, "component", MAX_COMPONENT_WAIT_S),
        ):
            _require(type(value) in (int, float) and 0 < float(value) <= maximum,
                     f"{label} timeout is outside its bound")
        self.host, self.port = str(host), int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.ack_timeout_s = float(ack_timeout_s)
        self.component_wait_s = float(component_wait_s)
        self._resolve = resolve_high_identity
        self._is_high_worker = is_high_worker
        self._connector = connector
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._stream: Any = None
        self._transport: Optional[GT.PersistentGtSenderV1] = None
        self._closed = False
        self._fault: Optional[str] = None
        self._pending: dict[str, _PendingPaths] = {}
        self._counters = {
            "connections": 0, "reconnections_after_lost_ack": 0,
            "stored": 0, "duplicate_identical": 0, "low_not_sent": 0,
            "semantic_path_publications": 0, "object_path_publications": 0,
            "failures": 0, "unknown_future_ticket_refusals": 0,
        }

    def connect(self) -> None:
        """Open the sole active persistent connection before route admission."""
        with self._send_lock:
            _require(not self._closed, "GT sender is closed", SenderLifecycleError)
            _require(self._stream is None and self._transport is None,
                     "GT sender is already connected", SenderLifecycleError)
            self._open_connection()

    def _open_connection(self) -> None:
        stream = self._connector((self.host, self.port), self.connect_timeout_s)
        try:
            transport = GT.PersistentGtSenderV1(stream, timeout_s=self.ack_timeout_s)
        except BaseException:
            stream.close()
            raise
        self._stream, self._transport = stream, transport
        self._counters["connections"] += 1

    def _drop_connection(self) -> None:
        stream = self._stream
        self._stream, self._transport = None, None
        if stream is None:
            return
        try:
            stream.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        try:
            stream.close()
        except OSError:
            pass

    def close(self) -> Mapping[str, Any]:
        """Idempotently stop future sends and close the active connection."""
        with self._send_lock:
            if not self._closed:
                self._closed = True
                self._drop_connection()
        with self._condition:
            self._condition.notify_all()
        return self.snapshot()

    def wrap_after_recorder(self, write_objects: Callable[..., Any],
                            write_semantic: Callable[..., Any]):
        """Wrap the already-recorded writers; ordering is an integration gate."""
        def semantic(directory, *, identity, **kwargs):
            result = write_semantic(directory, identity=identity, **kwargs)
            paths = list(result) if isinstance(result, (tuple, list)) else [result]
            _require(len(paths) == 2, "semantic writer did not return two paths")
            mapped = {"semantic.npy": Path(paths[0]), "semantic.json": Path(paths[1])}
            self._publish_paths(identity, mapped, kind="semantic")
            return result

        def objects(directory, *, identity, **kwargs):
            result = write_objects(directory, identity=identity, **kwargs)
            self._publish_paths(identity, {"objects.json": Path(result)}, kind="object")
            full = self._resolve(dict(identity))
            if full is None:
                with self._condition:
                    self._counters["low_not_sent"] += 1
                return result
            try:
                _require(full.gt_identity() == dict(identity),
                         "resolved full identity disagrees with writer GT identity")
                _require(bool(self._is_high_worker()),
                         "GT network send attempted outside the existing HIGH worker",
                         WrongWorkerError)
                self._send_joined(full)
            except GT.RemoteTicketRefusalError as exc:
                with self._condition:
                    self._counters["failures"] += 1
                    if exc.error_code == "UNKNOWN_OR_FUTURE_TICKET":
                        # The edge already waited for exact authorization. A
                        # tensor that never completed cannot earn a reward;
                        # fail this ticket without poisoning later identities.
                        self._counters["unknown_future_ticket_refusals"] += 1
                    else:
                        self._fault = f"{type(exc).__name__}: {exc}"[:500]
                    self._condition.notify_all()
                raise
            except BaseException as exc:
                with self._condition:
                    self._fault = f"{type(exc).__name__}: {exc}"[:500]
                    self._counters["failures"] += 1
                    self._condition.notify_all()
                raise
            return result

        return objects, semantic

    def _publish_paths(self, identity: Mapping[str, Any], paths: Mapping[str, Path],
                       *, kind: str) -> None:
        key = _gt_key(identity)
        with self._condition:
            _require(not self._closed, "GT paths published after sender close",
                     SenderLifecycleError)
            entry = self._pending.get(key)
            if entry is None:
                _require(len(self._pending) < MAX_PENDING_IDENTITIES,
                         "GT sender pending-identity registry is full",
                         SenderDeliveryError)
                entry = _PendingPaths(gt_identity=dict(identity))
                self._pending[key] = entry
            _require(entry.gt_identity == dict(identity), "GT join identity conflict")
            for name, path in paths.items():
                _require(name in GT.COMPONENT_NAMES, "foreign GT component name")
                existing = entry.paths.get(name)
                _require(existing is None or existing == path,
                         f"conflicting path for {name}")
                entry.paths[name] = path
            self._counters[f"{kind}_path_publications"] += 1
            self._condition.notify_all()

    def _send_joined(self, identity: GT.GtTransportIdentityV1) -> GT.GtAckV1:
        key, deadline = _gt_key(identity.gt_identity()), time.monotonic() + self.component_wait_s
        with self._condition:
            entry = self._pending[key]
            if entry.ack is not None:
                return entry.ack
            _require(not entry.sending, "duplicate concurrent HIGH GT send")
            while set(entry.paths) != set(GT.COMPONENT_NAMES):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SenderDeliveryError("timed out joining the three recorded GT paths")
                self._condition.wait(remaining)
                _require(not self._closed, "GT sender closed during component join",
                         SenderLifecycleError)
            entry.sending = True
            paths = dict(entry.paths)
        try:
            bundle = GT.bundle_from_phase6_paths(identity, paths)
            ack = self._send_with_one_lost_ack_retry(bundle)
        finally:
            with self._condition:
                entry.sending = False
                self._condition.notify_all()
        with self._condition:
            entry.ack = ack
            counter = "stored" if ack.status == "STORED" else "duplicate_identical"
            self._counters[counter] += 1
        return ack

    @staticmethod
    def _lost_ack_failure(exc: BaseException) -> bool:
        return (isinstance(exc, (socket.timeout, ConnectionError, BrokenPipeError))
                or (isinstance(exc, GT.ProtocolError)
                    and "stream closed during framed message" in str(exc)))

    def _send_with_one_lost_ack_retry(self, bundle: GT.GtBundleV1) -> GT.GtAckV1:
        with self._send_lock:
            _require(not self._closed, "GT sender is closed", SenderLifecycleError)
            _require(self._fault is None, f"GT sender is faulted: {self._fault}",
                     SenderLifecycleError)
            _require(self._transport is not None, "GT sender was not explicitly connected",
                     SenderLifecycleError)
            try:
                return self._transport.send(bundle)
            except BaseException as first:
                if not self._lost_ack_failure(first):
                    raise
                # A timeout leaves framing uncertain. Close it, open exactly one
                # replacement connection, and resend the bit-identical bundle.
                self._drop_connection()
                self._counters["reconnections_after_lost_ack"] += 1
                try:
                    self._open_connection()
                    return self._transport.send(bundle)
                except BaseException as second:
                    self._drop_connection()
                    raise SenderDeliveryError(
                        f"GT delivery failed after one identical retry: "
                        f"{type(second).__name__}: {second}") from second

    def snapshot(self) -> Mapping[str, Any]:
        with self._condition:
            return {
                "schema": "scenesense.run4.split_host.gt_sender.v1",
                "endpoint": f"{self.host}:{self.port}",
                "connected": self._transport is not None,
                "closed": self._closed, "fault": self._fault,
                "counters": dict(self._counters),
                "pending_keys": len(self._pending),
                "cross_host_clock_subtraction": False,
                "policy_deadline_clock_owner": "W10275",
            }


__all__ = [
    "REGISTERED_HOST", "REGISTERED_PORT", "MAX_PENDING_IDENTITIES", "GtSenderIntegrationError",
    "WrongWorkerError", "SenderLifecycleError", "SenderDeliveryError",
    "identity_from_ue_maps", "HighWorkerGtSenderV1",
]

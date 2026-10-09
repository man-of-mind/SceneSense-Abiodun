"""Causal, run-scoped SNR IPC for the original quality-aware Run-5 actor.

The unchanged target-SNR controller owns the actuator.  Its validated-ACK
writer hook stamps CLOCK_MONOTONIC_RAW *in that process*, then publishes a
small Unix datagram.  This is not a clock conversion or a CSV/file poll.
The UE admits a snapshot only after both its source event and IPC receipt.

The packet contains only the current acknowledged target and a matching
lease.  It contains no profile, trace, step, Markov state, RFsim noise value,
future target or measured gNB SNR.  The physical controller UUID is fixed for
the run; explicit decision-session adoption preserves it in audit evidence
and never restamps or renews a command/lease.

Controller invocation (normal target-SNR arguments follow these options)::

  python -m rl_agent.splitfusion_run5_live_avo_v1.snr_live --controller \
    --snr-socket /tmp/run5-unique.sock --snr-controller-id <uuid> ...

No import starts a thread, opens a file, creates a socket, or touches radio.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import socket
import stat
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as V1


CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
SCHEMA = "run5_live_causal_snr_snapshot_v1"
LEASE_NS = 200_000_000
PROVIDER_ID = "run5_live_raw_ack_unix_datagram_v1"
MAX_PACKET_BYTES = 4096


def raw_now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _uuid(value: str) -> str:
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("controller/session UUID must be canonical")
    return value


@dataclass(frozen=True)
class SnrSnapshot:
    schema: str
    controller_uuid: str
    sequence: int
    clock_domain: str
    command_id: str | None
    effective_raw_ns: int | None
    heartbeat_raw_ns: int
    target_snr_db: float | None
    valid: bool
    missing_reason: str | None

    def validate(self) -> None:
        if self.schema != SCHEMA or self.clock_domain != CLOCK_DOMAIN:
            raise ValueError("SNR schema/clock mismatch")
        _uuid(self.controller_uuid)
        for value in (self.sequence, self.heartbeat_raw_ns):
            if type(value) is not int or value < 0:
                raise ValueError("sequence/timestamp must be exact nonnegative integers")
        if type(self.valid) is not bool:
            raise ValueError("valid must be an exact bool")
        if self.effective_raw_ns is not None and (
                type(self.effective_raw_ns) is not int or
                not 0 <= self.effective_raw_ns <= self.heartbeat_raw_ns):
            raise ValueError("effective command time must precede the heartbeat")
        if self.command_id is not None and not isinstance(self.command_id, str):
            raise ValueError("command identity must be text")
        if self.valid:
            if (not self.command_id or self.effective_raw_ns is None or
                    type(self.target_snr_db) not in (float, int) or
                    not math.isfinite(self.target_snr_db) or self.missing_reason is not None):
                raise ValueError("valid SNR needs command, finite target and ACK time")
        elif (self.target_snr_db is not None or not isinstance(self.missing_reason, str)
              or not self.missing_reason):
            raise ValueError("invalid SNR requires a reason and no value")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def encode(self) -> bytes:
        self.validate()
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("ascii")

    @classmethod
    def decode(cls, packet: bytes) -> "SnrSnapshot":
        if len(packet) > MAX_PACKET_BYTES:
            raise ValueError("oversize SNR packet")
        value = json.loads(packet)
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("unexpected SNR packet fields")
        result = cls(**value)
        result.validate()
        return result


class ControllerPublisher:
    """Observer of validated controller rows; does not send radio commands.

    A skipped obsolete tick renews only the latest command already ACKed.
    It never makes that tick's unsent target effective.  The legacy mapping
    raises on out-of-range interpolation rather than clamping; its validated
    ACK rows therefore represent explicitly unclamped targets.  Quantization
    of the RFsim command is unchanged and is not support clipping.
    """

    def __init__(self, socket_path: Path | str, controller_id: str, *,
                 clock: Callable[[], int] = raw_now_ns, audit_path: Path | None = None):
        self.controller_id = _uuid(controller_id)
        self.socket_path = str(socket_path)
        self.clock = clock
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.sequence = 0
        self.current: SnrSnapshot | None = None
        self.stats = collections.Counter()
        self.audit = None if audit_path is None else Path(audit_path).open("x", encoding="utf-8")
        self.audit_failed = False

    def _publish(self, value: SnrSnapshot) -> None:
        if self.audit_failed:
            value = replace(value, valid=False, target_snr_db=None,
                            missing_reason="SNR_EVIDENCE_WRITER_FAILED")
        if self.audit is not None and not self.audit_failed:
            try:
                self.audit.write(value.encode().decode("ascii") + "\n")
                self.audit.flush()
            except Exception:
                self.audit_failed = True
                self.stats["audit_errors"] += 1
                value = replace(value, valid=False, target_snr_db=None,
                                missing_reason="SNR_EVIDENCE_WRITER_FAILED")
        self.current = value
        try:
            self.socket.sendto(value.encode(), self.socket_path)
            self.stats["sent"] += 1
        except OSError:
            # Missing receiver or full IPC buffer cannot kill the radio loop.
            # No received heartbeat means the UE lease expires fail-closed.
            self.stats["send_errors"] += 1

    def on_row(self, row: Mapping[str, Any]) -> None:
        """Called after legacy response validation and CSV flush, never before."""
        now = self.clock()
        if self.current is not None and now < self.current.heartbeat_raw_ns:
            raise ValueError("CLOCK_MONOTONIC_RAW went backwards")
        self.sequence += 1
        status = row.get("command_timing_status")
        if status in ("ACK_ON_TIME", "ACK_LATE"):
            # The original controller has already validated this response.
            if not row.get("command_ack_monotonic_ns"):
                raise ValueError("validated ACK row has no acknowledgement")
            target = row.get("target_snr_db")
            usable = type(target) in (float, int) and math.isfinite(target)
            value = SnrSnapshot(SCHEMA, self.controller_id, self.sequence,
                                CLOCK_DOMAIN, f"ack-{self.sequence}", now, now,
                                float(target) if usable else None, usable,
                                None if usable else "ACTIVE_COMMAND_HAS_NO_TARGET")
        elif status == "SKIP_OBSOLETE_NEVER_BURST":
            if self.current is None:
                value = SnrSnapshot(SCHEMA, self.controller_id, self.sequence,
                                    CLOCK_DOMAIN, None, None, now, None, False,
                                    "NO_EFFECTIVE_COMMAND_BEFORE_CUTOFF")
            else:
                value = replace(self.current, sequence=self.sequence, heartbeat_raw_ns=now)
        else:
            value = SnrSnapshot(SCHEMA, self.controller_id, self.sequence,
                                CLOCK_DOMAIN, None, None, now, None, False,
                                "CONTROLLER_ROW_NOT_ACKNOWLEDGED")
        self._publish(value)

    def close(self, *, reason: str = "CONTROLLER_STOPPED") -> None:
        self.sequence += 1
        self._publish(SnrSnapshot(SCHEMA, self.controller_id, self.sequence,
                                  CLOCK_DOMAIN, None, None, self.clock(), None, False, reason))
        self.socket.close()
        if self.audit is not None:
            try:
                self.audit.close()
            except Exception:
                self.stats["audit_errors"] += 1


@dataclass(frozen=True)
class ReceivedSnapshot:
    source: SnrSnapshot
    receipt_raw_ns: int


class LiveSnrProvider:
    """Bounded asynchronous IPC, with a nonblocking in-memory actor lookup.

    Use ``adopt_session`` on each explicit engine-session transition.  The
    controller identity is *not* changed; adoption only binds its current
    causal observation to the new decision session for the original guard.
    """

    def __init__(self, socket_path: Path | str, controller_id: str, ue_id: str, *,
                 capacity: int = 8192, clock: Callable[[], int] = raw_now_ns,
                 start_receiver: bool = True):
        self.controller_id = _uuid(controller_id)
        if not ue_id or type(capacity) is not int or capacity < 2:
            raise ValueError("UE identity and bounded history are required")
        self.ue_id = ue_id
        self.clock = clock
        self.session_uuid: str | None = None
        self._sample_seq = 0
        self._history = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_sequence = 0
        self.stats = collections.Counter()
        self.adoptions: list[dict[str, Any]] = []
        self.path = Path(socket_path)
        self.socket = None
        self.thread = None
        self._socket_inode = None
        if start_receiver:
            if not self.path.is_absolute() or len(os.fsencode(self.path)) >= 104:
                raise ValueError("use a short, absolute, create-only Unix socket path")
            if self.path.exists() or self.path.is_symlink():
                raise FileExistsError(self.path)
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                self.socket.bind(str(self.path))
                os.chmod(self.path, 0o600)
                self._socket_inode = self.path.stat().st_ino
                self.socket.settimeout(0.1)
            except Exception:
                self.socket.close()
                raise
            self.thread = threading.Thread(target=self._receive, name="run5-snr-receiver", daemon=True)
            self.thread.start()

    def accept_packet(self, packet: bytes, receipt_raw_ns: int) -> bool:
        """Receiver seam; receipt is stamped immediately after recv, never converted."""
        try:
            value = SnrSnapshot.decode(packet)
            if type(receipt_raw_ns) is not int or receipt_raw_ns < value.heartbeat_raw_ns:
                raise ValueError("source event was in the future at packet receipt")
            if value.controller_uuid != self.controller_id:
                self.stats["foreign_controller"] += 1
                return False
            with self._lock:
                if value.sequence <= self._last_sequence:
                    self.stats["duplicate_or_reordered"] += 1
                    return False
                if self._history and (
                        value.heartbeat_raw_ns < self._history[-1].source.heartbeat_raw_ns or
                        receipt_raw_ns < self._history[-1].receipt_raw_ns):
                    raise ValueError("source/receipt clock went backwards")
                self._history.append(ReceivedSnapshot(value, receipt_raw_ns))
                self._last_sequence = value.sequence
                self.stats["accepted"] += 1
            return True
        except (ValueError, TypeError, KeyError, UnicodeError):
            self.stats["invalid_packet"] += 1
            return False

    def _receive(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.socket.recv(MAX_PACKET_BYTES + 1)
                received = self.clock()
                self.accept_packet(packet, received)
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    self.stats["receiver_errors"] += 1
                return

    def adopt_session(self, session_uuid: str) -> None:
        session_uuid = _uuid(session_uuid)
        if session_uuid != self.session_uuid:
            self.adoptions.append({"decision_session_uuid": session_uuid,
                                   "source_controller_uuid": self.controller_id,
                                   "adopted_raw_ns": self.clock()})
            self.session_uuid = session_uuid
            self._sample_seq = 0

    def observe(self, boundary: R4.DecisionBoundaryV1) -> R5.UlSnrLeaseObservationV1:
        if not isinstance(boundary, R4.DecisionBoundaryV1):
            raise TypeError("boundary must be DecisionBoundaryV1")
        reason = None
        cutoff = boundary.state_commit_timestamp_ns
        with self._lock:
            selected = next((item for item in reversed(self._history)
                             if item.receipt_raw_ns < cutoff and
                             item.source.heartbeat_raw_ns < cutoff), None)
        if boundary.identity.ue_id != self.ue_id or boundary.identity.session_uuid != self.session_uuid:
            reason = "DECISION_IDENTITY_NOT_EXPLICITLY_ADOPTED"
        elif boundary.clock_domain != CLOCK_DOMAIN:
            reason = "DECISION_CLOCK_IS_NOT_CLOCK_MONOTONIC_RAW"
        elif selected is None:
            reason = "NO_RECEIVED_CONTROLLER_LEASE_BEFORE_CUTOFF"
        elif not selected.source.valid:
            reason = selected.source.missing_reason
        elif selected.source.effective_raw_ns >= cutoff:
            reason = "COMMAND_NOT_EFFECTIVE_BEFORE_CUTOFF"
        else:
            age = boundary.action_open_timestamp_ns - selected.source.heartbeat_raw_ns
            if not 0 < age <= LEASE_NS:
                reason = "CONTROLLER_LEASE_STALE"
            else:
                try:
                    R5.scale_snr_db(selected.source.target_snr_db)
                except R5.ExternalFallbackRequired:
                    reason = "SNR_OUT_OF_REGISTERED_SUPPORT"
        source = selected.source if selected else None
        # Invalid observation remains scoped to the *requested* identity, but
        # the explicit-adoption failure above prevents it being admitted.
        identity = R4.SampleIdentityV1(boundary.identity.session_uuid, self.ue_id, self._sample_seq)
        self._sample_seq += 1
        return R5.UlSnrLeaseObservationV1(
            identity=identity, kind=V1.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB,
            provider_id=PROVIDER_ID, controller_session_uuid=boundary.identity.session_uuid,
            value_db=source.target_snr_db if reason is None else None,
            active_command_id=source.command_id if source else None,
            effective_since_ns=source.effective_raw_ns if source else None,
            heartbeat_command_id=source.command_id if source else None,
            heartbeat_ns=source.heartbeat_raw_ns if source else None,
            clock_domain=CLOCK_DOMAIN, valid=reason is None, missing_reason=reason)

    def evidence(self) -> dict[str, Any]:
        with self._lock:
            rows = [{**item.source.to_dict(), "receipt_raw_ns": item.receipt_raw_ns}
                    for item in self._history]
        return {"schema": SCHEMA, "controller_uuid": self.controller_id,
                "lease_ns": LEASE_NS, "stats": dict(self.stats),
                "session_adoptions": list(self.adoptions), "received_snapshots": rows}

    def close(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            if self.thread.is_alive():
                raise RuntimeError("SNR receiver did not stop")
        if self.socket is not None:
            self.socket.close()
        # Never delete a replacement/foreign socket or a broad directory.
        if self._socket_inode is not None:
            try:
                current = self.path.lstat()
            except FileNotFoundError:
                return
            if stat.S_ISSOCK(current.st_mode) and current.st_ino == self._socket_inode:
                self.path.unlink()


def controller_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--controller", action="store_true", required=True)
    parser.add_argument("--snr-socket", type=Path, required=True)
    parser.add_argument("--snr-controller-id", required=True)
    parser.add_argument("--snr-audit", type=Path)
    args, legacy_args = parser.parse_known_args(argv)
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_target_snr_runtime as legacy
    publisher = ControllerPublisher(args.snr_socket, args.snr_controller_id,
                                    audit_path=args.snr_audit)
    original = legacy.write_row

    def observed_write(handle, writer, row):
        original(handle, writer, row)
        publisher.on_row(row)

    legacy.write_row = observed_write
    try:
        return legacy.main(legacy_args)
    finally:
        legacy.write_row = original
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(controller_main())

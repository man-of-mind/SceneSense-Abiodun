"""Direct edge-to-map ingestion, installation and compact feedback.

This is the spatial map's half of the corrected architecture. It binds an
edge-local endpoint (the host's address on the CN5G bridge), validates every
inbound object-map update against the frozen cell identity, installs accepted
updates under the map's own authoritative ``state_lock``, and only then emits a
compact ACK/agent-credit message to the UE.

Ordering is a contract, not an accident: ``_install_locked`` returns only after
the update is inside ``latest_streams``/``installed_frame_history``, and the
feedback datagram is constructed from its return value. There is no code path
that emits ``RESULT_INSTALLED`` without a completed installation.

Physical map freshness ends at ``install_timestamp``. The feedback emission and
its arrival at the UE are recorded separately and are never folded into
map-installation age.
"""

from __future__ import annotations

import csv
import queue
import socket
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from phase2_map_sharing.transport import ChunkReassembler

from . import protocol
from .cpu_reservation import apply_thread_reservation, describe_reservation
from .protocol import (
    DirectMapProtocolError,
    OUTCOME_MAP_REJECTED,
    OUTCOME_RESULT_INSTALLED,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_SUPERSEDED_PENDING,
)


INGEST_FIELDS = (
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "capture_id",
    "action_id",
    "profile_id",
    "capture_timestamp",
    "service_deadline_at",
    "ack_timeout_at",
    "first_datagram_at",
    "map_ingest_at",
    "map_install_at",
    "install_timestamp",
    "feedback_emit_at",
    "outcome",
    "agent_credit",
    "terminal",
    "accepted",
    "map_age_at_install_ms",
    "install_latency_from_publish_ms",
    "install_latency_from_tail_ms",
    "direct_update_bytes",
    "direct_update_datagrams",
    "feedback_bytes",
    "record_count",
    "superseded_by_frame_id",
    "rejection_reason",
    "edge_publish_start_wall_s",
    "edge_tail_complete_wall_s",
    "edge_reassembly_complete_wall_s",
    "edge_admission_wall_s",
    "edge_compute_start_wall_s",
    "edge_compute_finish_wall_s",
    "edge_publication_ticket_ready_wall_s",
    "edge_serialization_start_wall_s",
    "last_datagram_at",
    "reassembly_complete_at",
    "map_worker_start_at",
    "map_lock_request_at",
    "map_lock_acquired_at",
    "map_lock_released_at",
    "association_start_at",
    "association_end_at",
    "ack_emit_at",
    "ack_sent_at",
    "ingest_queue_wait_ms",
    "ingest_queue_depth_at_enqueue",
)


_INGEST_STOP = object()


class DirectMapIngestService:
    """UDP ingest -> identity validation -> authoritative install -> feedback."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        feedback_host: str,
        feedback_port: int,
        install: Callable[[Mapping[str, Any], float], dict[str, Any]],
        ingest_csv: Path | None = None,
        expected_run_id: str = "",
        expected_cell_id: str = "",
        allowed_action_ids: tuple[int, ...] = (),
        processing_horizon_s: float = 0.5,
        socket_buffer_request_bytes: int = 8 << 20,
        chunk_timeout_s: float = 2.0,
        ingest_queue_capacity: int = 64,
        ingest_cpus: str = "",
        receive_cpus: str = "",
    ) -> None:
        protocol._require(
            not protocol.is_ue_address(bind_host, ("10.0.0.2",)),
            "the direct map ingest endpoint must not bind the UE tunnel address",
        )
        self.bind_host = str(bind_host)
        self.bind_port = int(bind_port)
        self.feedback_remote = (str(feedback_host), int(feedback_port))
        self._install = install
        self.expected_run_id = str(expected_run_id)
        self.expected_cell_id = str(expected_cell_id)
        self.allowed_action_ids = tuple(int(value) for value in allowed_action_ids)
        self.processing_horizon_s = float(processing_horizon_s)
        self.counters: Counter[str] = Counter()
        self.failures: list[str] = []
        self._lock = threading.Lock()
        # Exactly-one-terminal bookkeeping. ``_terminal`` holds every identity
        # that already produced a terminal outcome, so a duplicate update can
        # never install twice nor emit a second terminal.
        self._terminal: set[tuple[str, str, str, int]] = set()
        self._newest_capture_ns: dict[str, int] = {}
        self._newest_frame_id: dict[str, int] = {}
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, int(socket_buffer_request_bytes)
        )
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, int(socket_buffer_request_bytes)
        )
        self.socket.bind((self.bind_host, self.bind_port))
        self.socket.settimeout(0.25)
        self.reassembler = ChunkReassembler(timeout_s=float(chunk_timeout_s), max_chunks=4096)
        self._expired_seen = 0
        self.stop_event = threading.Event()
        self._started = False
        self._first_datagram_at: dict[int, float] = {}
        self._last_datagram_at: dict[int, float] = {}
        self._rows: list[dict[str, Any]] = []
        # The receive owner must never be inside validation, installation or
        # feedback when a datagram arrives, or the recorded arrival instant is
        # really a scheduling artefact of its own previous install. A bounded
        # hand-off keeps the arrival stamp honest. It is a queue, not a drop
        # policy: on overflow the receive owner blocks and the event is
        # counted, so nothing is silently discarded.
        self.ingest_queue_capacity = int(ingest_queue_capacity)
        protocol._require(
            self.ingest_queue_capacity >= 2,
            "the map ingest hand-off must hold at least two completions",
        )
        self._ingest_queue: "queue.Queue[Any]" = queue.Queue(
            maxsize=self.ingest_queue_capacity
        )
        self.ingest_csv = Path(ingest_csv) if ingest_csv is not None else None
        self._handle = None
        self._writer = None
        if self.ingest_csv is not None:
            self.ingest_csv.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.ingest_csv.open("x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=list(INGEST_FIELDS))
            self._writer.writeheader()
            self._handle.flush()
        self.ingest_cpus = str(ingest_cpus)
        self.receive_cpus = str(receive_cpus)
        self.reservations: list[dict[str, Any]] = []
        self.thread = threading.Thread(
            target=self._loop, name="direct-map-receive", daemon=True
        )
        self.ingest_thread = threading.Thread(
            target=self._ingest_loop, name="direct-map-ingest", daemon=True
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._started = True
        self.ingest_thread.start()
        self.thread.start()

    def close(self) -> dict[str, Any]:
        self.stop_event.set()
        # ``close`` must be safe on a service that was constructed but never
        # started (an aborted cell launch, or synchronous ``ingest`` use).
        if self._started:
            self.thread.join(timeout=10.0)
            self._ingest_queue.put(_INGEST_STOP)
            self.ingest_thread.join(timeout=10.0)
        try:
            self.socket.close()
        except OSError:
            pass
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None
        return self.report()

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "splitfusion_direct_map_ingest_report.v1",
                "bind_host": self.bind_host,
                "bind_port": self.bind_port,
                "feedback_remote_host": self.feedback_remote[0],
                "feedback_remote_port": self.feedback_remote[1],
                "counters": dict(self.counters),
                "terminal_identities": len(self._terminal),
                "failures": list(self.failures[:16]),
                "rows": len(self._rows),
                "ingest_queue_capacity": self.ingest_queue_capacity,
                "cpu_reservation": describe_reservation(self.reservations),
            }

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]

    # -- receive path ------------------------------------------------------

    def _loop(self) -> None:
        """Receive owner: stamp arrival, reassemble, hand off. Nothing else."""

        self.reservations.append(
            apply_thread_reservation(self.receive_cpus, label="direct-map-receive")
        )
        while not self.stop_event.is_set():
            try:
                datagram, address = self.socket.recvfrom(65535)
            except socket.timeout:
                self.reassembler.expire(time.monotonic())
                self._reconcile_expiries()
                continue
            except OSError:
                return
            arrived_at = time.time()
            self.counters["direct_datagrams_received"] += 1
            self.counters["direct_datagram_bytes_received"] += len(datagram)
            message_id = int(self._peek_message_id(datagram))
            self._first_datagram_at.setdefault(message_id, arrived_at)
            self._last_datagram_at[message_id] = arrived_at
            try:
                complete = self.reassembler.ingest(
                    str(address), datagram, received_at_s=time.monotonic()
                )
            except ValueError:
                self.counters["direct_datagrams_malformed"] += 1
                continue
            self._reconcile_expiries()
            if complete is None:
                continue
            reassembly_complete_at = time.time()
            first_at = self._first_datagram_at.pop(
                int(complete.message_id), arrived_at
            )
            last_at = self._last_datagram_at.pop(int(complete.message_id), arrived_at)
            self.counters["direct_updates_reassembled"] += 1
            stamps = {
                "first_datagram_at": first_at,
                "last_datagram_at": last_at,
                "reassembly_complete_at": reassembly_complete_at,
                "enqueued_at": time.time(),
                "queue_depth_at_enqueue": self._ingest_queue.qsize(),
            }
            if self._ingest_queue.full():
                # Back-pressure, not a drop: the obligation is preserved and
                # the event is counted so it can never pass unnoticed.
                self.counters["direct_ingest_queue_blocked"] += 1
            self._ingest_queue.put((complete, stamps))

    def _ingest_loop(self) -> None:
        """Ingest owner: validate, install and acknowledge, off the receive path."""

        self.reservations.append(
            apply_thread_reservation(self.ingest_cpus, label="direct-map-ingest")
        )
        while True:
            item = self._ingest_queue.get()
            if item is _INGEST_STOP:
                return
            complete, stamps = item
            try:
                self._handle_complete(complete, stamps)
            except Exception as exc:  # recorded, never fatal to the map
                self.counters["direct_ingest_errors"] += 1
                with self._lock:
                    self.failures.append(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _peek_message_id(datagram: bytes) -> int:
        import struct

        if len(datagram) < 8:
            return -1
        try:
            message_id, _total, _index = struct.unpack("!IHH", datagram[:8])
        except struct.error:
            return -1
        return int(message_id)

    def _reconcile_expiries(self) -> None:
        observed = int(self.reassembler.expired_messages)
        if observed != self._expired_seen:
            self.counters["direct_incomplete_reassemblies_expired"] += (
                observed - self._expired_seen
            )
            self._expired_seen = observed

    def _handle_complete(self, complete: Any, stamps: Mapping[str, Any]) -> None:
        worker_start_at = time.time()
        try:
            document = protocol.decode(zlib.decompress(complete.payload))
        except (zlib.error, DirectMapProtocolError) as exc:
            self.counters["direct_updates_undecodable"] += 1
            with self._lock:
                self.failures.append(f"undecodable direct update: {exc}")
            return
        self.ingest(
            document,
            ingest_at=worker_start_at,
            first_datagram_at=float(stamps["first_datagram_at"]),
            update_bytes=len(complete.payload),
            update_datagrams=int(complete.chunk_count),
            stage_stamps={
                "last_datagram_at": float(stamps["last_datagram_at"]),
                "reassembly_complete_at": float(stamps["reassembly_complete_at"]),
                "map_worker_start_at": worker_start_at,
                "ingest_queue_wait_ms": (
                    worker_start_at - float(stamps["enqueued_at"])
                )
                * 1000.0,
                "ingest_queue_depth_at_enqueue": int(
                    stamps["queue_depth_at_enqueue"]
                ),
            },
        )

    # -- validation, installation and feedback -----------------------------

    def ingest(
        self,
        document: Mapping[str, Any],
        *,
        ingest_at: float,
        first_datagram_at: float | None = None,
        update_bytes: int = 0,
        update_datagrams: int = 0,
        emit: bool = True,
        stage_stamps: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate, install and acknowledge one direct object-map update."""

        first_at = ingest_at if first_datagram_at is None else float(first_datagram_at)
        stamps: dict[str, Any] = dict(stage_stamps or {})
        stamps.setdefault("last_datagram_at", first_at)
        stamps.setdefault("reassembly_complete_at", ingest_at)
        stamps.setdefault("map_worker_start_at", ingest_at)
        try:
            protocol.validate_object_map_update(document)
            self._validate_binding(document)
        except DirectMapProtocolError as exc:
            self.counters["direct_updates_rejected"] += 1
            return self._respond(
                document,
                outcome=OUTCOME_MAP_REJECTED,
                terminal=True,
                install_timestamp=None,
                map_ingest_at=ingest_at,
                first_datagram_at=first_at,
                rejection_reason=str(exc)[:200],
                update_bytes=update_bytes,
                update_datagrams=update_datagrams,
                emit=emit,
                stamps=stamps,
            )

        identity = protocol.update_identity(document)
        stream_id = str(document["stream_id"])
        capture_ns = int(document["capture_timestamp_ns"])

        with self._lock:
            duplicate = identity in self._terminal
            newest_capture = self._newest_capture_ns.get(stream_id, -1)
            newest_frame = self._newest_frame_id.get(stream_id)

        if duplicate:
            # A repeat arrival must never install twice and must never create a
            # second terminal for the same transmission obligation.
            self.counters["direct_updates_duplicate"] += 1
            return self._respond(
                document,
                outcome=OUTCOME_MAP_REJECTED,
                terminal=False,
                install_timestamp=None,
                map_ingest_at=ingest_at,
                first_datagram_at=first_at,
                rejection_reason="DUPLICATE_UPDATE_IGNORED",
                update_bytes=update_bytes,
                update_datagrams=update_datagrams,
                emit=emit,
                stamps=stamps,
            )

        age_s = ingest_at - capture_ns / 1_000_000_000.0
        if age_s > self.processing_horizon_s:
            self.counters["direct_updates_stale_before_map"] += 1
            return self._respond(
                document,
                outcome=OUTCOME_STALE_BEFORE_MAP,
                terminal=True,
                install_timestamp=None,
                map_ingest_at=ingest_at,
                first_datagram_at=first_at,
                rejection_reason=f"CAPTURE_AGE_MS={age_s * 1000.0:.3f}",
                update_bytes=update_bytes,
                update_datagrams=update_datagrams,
                emit=emit,
                stamps=stamps,
            )

        if capture_ns <= newest_capture:
            # The map already holds strictly fresher work for this stream. This
            # is deliberate replacement, not a network failure.
            self.counters["direct_updates_superseded"] += 1
            return self._respond(
                document,
                outcome=OUTCOME_SUPERSEDED_PENDING,
                terminal=True,
                install_timestamp=None,
                map_ingest_at=ingest_at,
                first_datagram_at=first_at,
                rejection_reason="NEWER_CAPTURE_ALREADY_INSTALLED",
                update_bytes=update_bytes,
                update_datagrams=update_datagrams,
                superseded_by_frame_id=newest_frame,
                emit=emit,
                stamps=stamps,
            )

        association_start_at = time.time()
        installed = self._install(document, ingest_at)
        install_timestamp = float(installed["install_timestamp"])
        stamps["association_start_at"] = association_start_at
        for name in (
            "association_end_at",
            "map_lock_request_at",
            "map_lock_acquired_at",
            "map_lock_released_at",
        ):
            if name in installed:
                stamps[name] = installed[name]
        stamps.setdefault("association_end_at", install_timestamp)
        with self._lock:
            self._newest_capture_ns[stream_id] = capture_ns
            self._newest_frame_id[stream_id] = int(document["frame_id"])
        self.counters["direct_updates_installed"] += 1
        self.counters["direct_records_installed"] += int(document["record_count"])
        return self._respond(
            document,
            outcome=OUTCOME_RESULT_INSTALLED,
            terminal=True,
            install_timestamp=install_timestamp,
            map_ingest_at=ingest_at,
            first_datagram_at=first_at,
            update_bytes=update_bytes,
            update_datagrams=update_datagrams,
            emit=emit,
            stamps=stamps,
        )

    def _validate_binding(self, document: Mapping[str, Any]) -> None:
        if self.expected_run_id:
            protocol._require(
                str(document["run_id"]) == self.expected_run_id,
                f"run identity drift: {document['run_id']!r}",
            )
        if self.expected_cell_id:
            protocol._require(
                str(document["cell_id"]) == self.expected_cell_id,
                f"cell identity drift: {document['cell_id']!r}",
            )
        if self.allowed_action_ids:
            protocol._require(
                int(document["action_id"]) in self.allowed_action_ids,
                f"action identity outside the cell allowlist: {document['action_id']}",
            )

    def _respond(
        self,
        document: Mapping[str, Any],
        *,
        outcome: str,
        terminal: bool,
        install_timestamp: float | None,
        map_ingest_at: float,
        first_datagram_at: float,
        update_bytes: int,
        update_datagrams: int,
        rejection_reason: str = "",
        superseded_by_frame_id: int | None = None,
        emit: bool = True,
        stamps: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit feedback (strictly after installation) and record the row."""

        stage = dict(stamps or {})

        map_age_ms = None
        if install_timestamp is not None:
            map_age_ms = (
                install_timestamp - int(document["capture_timestamp_ns"]) / 1_000_000_000.0
            ) * 1000.0
        feedback_emit_at = time.time()
        try:
            message = protocol.build_map_feedback(
                update=document,
                outcome=outcome,
                terminal=terminal,
                install_timestamp=install_timestamp,
                map_ingest_at=map_ingest_at,
                feedback_emit_at=feedback_emit_at,
                map_age_at_install_ms=map_age_ms,
                rejection_reason=rejection_reason,
                direct_update_bytes=int(update_bytes),
                direct_update_datagrams=int(update_datagrams),
                superseded_by_frame_id=superseded_by_frame_id,
            )
        except DirectMapProtocolError as exc:
            # The update was already installed if it got that far; a feedback
            # construction fault must never undo or hide that installation.
            self.counters["feedback_construction_failed"] += 1
            with self._lock:
                self.failures.append(f"feedback construction: {exc}")
            return {"outcome": outcome, "feedback": None, "feedback_bytes": 0}

        payload = protocol.encode(message)
        ack_sent_at = ""
        if emit:
            try:
                self.socket.sendto(payload, self.feedback_remote)
                ack_sent_at = time.time()
                self.counters["feedback_messages_emitted"] += 1
                self.counters["feedback_bytes_emitted"] += len(payload)
            except OSError as exc:
                # Installation has happened; the UE observes TIMEOUT_NO_ACK.
                self.counters["feedback_send_failed"] += 1
                with self._lock:
                    self.failures.append(f"feedback send: {exc}")
        if terminal:
            with self._lock:
                self._terminal.add(protocol.update_identity(document))

        edge_timing = dict(document.get("edge_timing") or {})
        publish_start = edge_timing.get("publish_start_wall_s")
        tail_complete = edge_timing.get("tail_complete_wall_s")
        row = {
            "run_id": str(document.get("run_id", "")),
            "cell_id": str(document.get("cell_id", "")),
            "stream_id": str(document.get("stream_id", "")),
            "frame_id": document.get("frame_id", ""),
            "capture_id": message["capture_id"],
            "action_id": document.get("action_id", ""),
            "profile_id": str(document.get("profile_id", "")),
            "capture_timestamp": message["capture_timestamp"],
            "service_deadline_at": document.get("service_deadline_at", ""),
            "ack_timeout_at": document.get("ack_timeout_at", ""),
            "first_datagram_at": first_datagram_at,
            "map_ingest_at": map_ingest_at,
            "map_install_at": ("" if install_timestamp is None else install_timestamp),
            "install_timestamp": ("" if install_timestamp is None else install_timestamp),
            "feedback_emit_at": feedback_emit_at,
            "outcome": outcome,
            "agent_credit": message["agent_credit"],
            "terminal": int(bool(terminal)),
            "accepted": int(outcome == OUTCOME_RESULT_INSTALLED),
            "map_age_at_install_ms": ("" if map_age_ms is None else map_age_ms),
            "install_latency_from_publish_ms": (
                (install_timestamp - float(publish_start)) * 1000.0
                if install_timestamp is not None and publish_start not in (None, "")
                else ""
            ),
            "install_latency_from_tail_ms": (
                (install_timestamp - float(tail_complete)) * 1000.0
                if install_timestamp is not None and tail_complete not in (None, "")
                else ""
            ),
            "direct_update_bytes": int(update_bytes),
            "direct_update_datagrams": int(update_datagrams),
            "feedback_bytes": len(payload),
            "record_count": document.get("record_count", ""),
            "superseded_by_frame_id": message["superseded_by_frame_id"],
            "rejection_reason": rejection_reason,
            "edge_publish_start_wall_s": edge_timing.get("publish_start_wall_s", ""),
            "edge_tail_complete_wall_s": edge_timing.get("tail_complete_wall_s", ""),
            "edge_reassembly_complete_wall_s": edge_timing.get(
                "reassembly_complete_wall_s", ""
            ),
            "edge_admission_wall_s": edge_timing.get("admission_wall_s", ""),
            "edge_compute_start_wall_s": edge_timing.get("compute_start_wall_s", ""),
            "edge_compute_finish_wall_s": edge_timing.get("compute_finish_wall_s", ""),
            "edge_publication_ticket_ready_wall_s": edge_timing.get(
                "publication_ticket_ready_wall_s", ""
            ),
            "edge_serialization_start_wall_s": edge_timing.get(
                "serialization_start_wall_s", ""
            ),
            "last_datagram_at": stage.get("last_datagram_at", ""),
            "reassembly_complete_at": stage.get("reassembly_complete_at", ""),
            "map_worker_start_at": stage.get("map_worker_start_at", ""),
            "map_lock_request_at": stage.get("map_lock_request_at", ""),
            "map_lock_acquired_at": stage.get("map_lock_acquired_at", ""),
            "map_lock_released_at": stage.get("map_lock_released_at", ""),
            "association_start_at": stage.get("association_start_at", ""),
            "association_end_at": stage.get("association_end_at", ""),
            "ack_emit_at": feedback_emit_at,
            "ack_sent_at": ack_sent_at,
            "ingest_queue_wait_ms": stage.get("ingest_queue_wait_ms", ""),
            "ingest_queue_depth_at_enqueue": stage.get(
                "ingest_queue_depth_at_enqueue", ""
            ),
        }
        with self._lock:
            self._rows.append(row)
            if self._writer is not None:
                self._writer.writerow(
                    {field: row.get(field, "") for field in INGEST_FIELDS}
                )
                self._handle.flush()
        return {
            "outcome": outcome,
            "feedback": message,
            "feedback_bytes": len(payload),
            "install_timestamp": install_timestamp,
            "row": row,
        }

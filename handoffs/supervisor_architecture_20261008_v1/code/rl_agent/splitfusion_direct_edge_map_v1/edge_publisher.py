"""Edge-side direct map publication and compact UE control transmission.

Two sockets with deliberately different jobs and different audits:

``DirectMapPublisher``
    Carries object records, and only ever to the audited edge-local map
    endpoint. Its constructor refuses a UE tunnel address, a UE result port,
    loopback and the unspecified address, so the removed edge -> UE -> map
    detour cannot be reintroduced by configuration.

``UEControlSender``
    Carries compact ACK/agent-credit/control messages to the UE. Every document
    is passed through ``assert_no_object_records`` before it is encoded, so a
    control message can never smuggle object records or a dense mask back onto
    the radio.
"""

from __future__ import annotations

import socket
import threading
import time
import zlib
from collections import Counter
from typing import Any, Mapping

from phase2_map_sharing.transport import chunk_payload

from . import protocol
from .endpoint import audit_publisher_destination


class DirectMapPublisher:
    """Publish versioned object-map updates straight to the edge spatial map."""

    def __init__(
        self,
        *,
        map_host: str,
        map_port: int,
        chunk_bytes: int = 12500,
        socket_buffer_request_bytes: int = 8 << 20,
        bind_host: str = "",
    ) -> None:
        audit_publisher_destination(map_host, map_port)
        self.remote = (str(map_host), int(map_port))
        self.chunk_bytes = int(chunk_bytes)
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, int(socket_buffer_request_bytes)
        )
        if str(bind_host).strip():
            self.socket.bind((str(bind_host), 0))

    def publish(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Send one validated update; return its publication accounting.

        Every stage boundary is stamped on the same wall clock the map uses, so
        publication can be decomposed into validation, serialization and the
        datagram burst instead of being attributed as one opaque interval.
        """

        publish_start_wall_s = time.time()
        publish_start_ns = time.perf_counter_ns()
        protocol.validate_object_map_update(document)
        serialization_start_wall_s = time.time()
        # The publish-start instant is stamped into the message before it is
        # encoded, so the map can report install latency measured from the
        # moment publication work began rather than inferring it.
        stamped = dict(document)
        timing = dict(stamped.get("edge_timing") or {})
        timing["publish_start_wall_s"] = publish_start_wall_s
        timing["serialization_start_wall_s"] = serialization_start_wall_s
        stamped["edge_timing"] = timing
        payload = zlib.compress(protocol.encode(stamped), level=1)
        chunks = chunk_payload(
            payload,
            message_id=int(stamped["frame_id"]),
            chunk_bytes=self.chunk_bytes,
        )
        serialization_end_wall_s = time.time()
        first_datagram_send_wall_s = 0.0
        for index, chunk in enumerate(chunks):
            if index == 0:
                first_datagram_send_wall_s = time.time()
            self.socket.sendto(chunk, self.remote)
        last_datagram_send_wall_s = time.time()
        publish_finish_ns = time.perf_counter_ns()
        datagram_bytes = sum(len(chunk) for chunk in chunks)
        with self._lock:
            self.counters["direct_map_updates_published"] += 1
            self.counters["direct_map_datagrams_transmitted"] += len(chunks)
            self.counters["direct_map_payload_bytes"] += len(payload)
            self.counters["direct_map_datagram_bytes"] += datagram_bytes
            self.counters["direct_map_records_published"] += int(stamped["record_count"])
        return {
            "publish_start_wall_s": publish_start_wall_s,
            "serialization_start_wall_s": serialization_start_wall_s,
            "serialization_end_wall_s": serialization_end_wall_s,
            "first_datagram_send_wall_s": first_datagram_send_wall_s,
            "last_datagram_send_wall_s": last_datagram_send_wall_s,
            "publish_finish_wall_s": last_datagram_send_wall_s,
            "publish_start_ns": publish_start_ns,
            "publish_finish_ns": publish_finish_ns,
            "publish_duration_ms": (publish_finish_ns - publish_start_ns) / 1e6,
            "direct_map_payload_bytes": len(payload),
            "direct_map_datagram_bytes": datagram_bytes,
            "direct_map_datagrams": len(chunks),
        }

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass


class UEControlSender:
    """Send compact, record-free control messages to the UE."""

    def __init__(
        self,
        *,
        ue_host: str,
        ue_port: int,
        socket_buffer_request_bytes: int = 1 << 20,
    ) -> None:
        protocol._require(bool(str(ue_host).strip()), "UE control host is empty")
        protocol._require(int(ue_port) > 0, "UE control port is invalid")
        protocol._require(
            int(ue_port) not in {51004, 51104},
            "the UE control port must not reuse the superseded result port",
        )
        self.remote = (str(ue_host), int(ue_port))
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, int(socket_buffer_request_bytes)
        )

    def send(self, document: Mapping[str, Any]) -> int:
        protocol.assert_no_object_records(document)
        payload = protocol.encode(document)
        protocol._require(
            len(payload) <= 8192,
            f"UE control message is not compact ({len(payload)} bytes)",
        )
        try:
            self.socket.sendto(payload, self.remote)
        except OSError:
            with self._lock:
                self.counters["ue_control_send_failed"] += 1
            return 0
        with self._lock:
            self.counters["ue_control_messages_sent"] += 1
            self.counters["ue_control_bytes_sent"] += len(payload)
        return len(payload)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass

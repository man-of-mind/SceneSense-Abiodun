#!/usr/bin/env python3
"""Receive and account for the structured UDP stream used by the UE-N3 screen.

The existing CARLA-shaped sender prefixes every datagram with ``!8sIIII``:
``SSBURST``, frame index, chunk index, chunks per frame, and the complete
datagram size.  This receiver consumes that wire format exactly.  It writes
packet and one-second interval evidence incrementally to JSONL and retains only
a bounded reorder window for each source-IP/source-port stream.

This utility does not start OAI or CARLA and does not decide an operational SNR
bound.  It only provides transport evidence for a separately controlled run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


HEADER = struct.Struct("!8sIIII")
MAGIC = b"SSBURST"
UINT32_LIMIT = 2**32
MAX_DURATION_S = 3_600.0
MAX_STREAMS_LIMIT = 16
MAX_REORDER_WINDOW_FRAMES_LIMIT = 1_024
MAX_CHUNKS_PER_FRAME_LIMIT = 1_024


class PacketContractError(ValueError):
    """Raised when a datagram does not satisfy the frozen SSBURST contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class BurstHeader:
    frame_index: int
    chunk_index: int
    chunks_per_frame: int
    declared_datagram_bytes: int


def parse_ssburst_datagram(data: bytes, *, max_chunks_per_frame: int) -> BurstHeader:
    """Parse one sender datagram and reject ambiguous sequence evidence."""

    if len(data) < HEADER.size:
        raise PacketContractError("DATAGRAM_SHORTER_THAN_HEADER")
    raw_magic, frame_index, chunk_index, chunks_per_frame, declared_size = HEADER.unpack_from(data)
    if raw_magic.rstrip(b"\0") != MAGIC:
        raise PacketContractError("MAGIC_MISMATCH")
    if chunks_per_frame < 1:
        raise PacketContractError("ZERO_CHUNKS_PER_FRAME")
    if chunks_per_frame > int(max_chunks_per_frame):
        raise PacketContractError("CHUNKS_PER_FRAME_EXCEEDS_BOUND")
    if chunk_index >= chunks_per_frame:
        raise PacketContractError("CHUNK_INDEX_OUT_OF_RANGE")
    if declared_size != len(data):
        raise PacketContractError("DECLARED_DATAGRAM_SIZE_MISMATCH")
    return BurstHeader(
        frame_index=frame_index,
        chunk_index=chunk_index,
        chunks_per_frame=chunks_per_frame,
        declared_datagram_bytes=declared_size,
    )


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stream_id(address: tuple[str, int]) -> str:
    return f"{address[0]}:{address[1]}"


def _empty_bins(length: int) -> list[int]:
    return [0 for _ in range(length)]


@dataclass
class StreamAccounting:
    identity: str
    expected_first_frame: int
    expected_frames: int
    expected_chunks_per_frame: int
    reorder_window_frames: int
    interval_bin_count: int
    next_finalize_frame: int = field(init=False)
    pending: dict[int, set[int]] = field(default_factory=dict)
    highest_seen_frame: int | None = None
    maximum_sequence_seen: int | None = None
    first_unique_monotonic_ns: int | None = None
    last_unique_monotonic_ns: int | None = None
    max_interarrival_gap_ns: int = 0
    interarrival_gaps_over_one_second: int = 0
    datagrams_with_valid_header: int = 0
    wire_datagram_bytes: int = 0
    unique_chunks: int = 0
    unique_datagram_bytes: int = 0
    unique_payload_bytes: int = 0
    duplicate_chunks: int = 0
    late_after_finalize: int = 0
    out_of_order_unique_chunks: int = 0
    contract_mismatch_datagrams: int = 0
    outside_expected_range_datagrams: int = 0
    finalized_frames: int = 0
    complete_frames: int = 0
    incomplete_frames: int = 0
    wholly_missing_frames: int = 0
    expected_chunks_finalized: int = 0
    received_chunks_finalized: int = 0
    lost_chunks: int = 0
    maximum_pending_frames_observed: int = 0
    interval_unique_chunks: list[int] = field(init=False)
    interval_unique_datagram_bytes: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.next_finalize_frame = self.expected_first_frame
        self.interval_unique_chunks = _empty_bins(self.interval_bin_count)
        self.interval_unique_datagram_bytes = _empty_bins(self.interval_bin_count)

    @property
    def expected_end_frame(self) -> int:
        return self.expected_first_frame + self.expected_frames

    def _finalize_present(self, chunks: set[int]) -> None:
        received = len(chunks)
        expected = self.expected_chunks_per_frame
        self.finalized_frames += 1
        self.expected_chunks_finalized += expected
        self.received_chunks_finalized += received
        self.lost_chunks += expected - received
        if received == expected:
            self.complete_frames += 1
        else:
            self.incomplete_frames += 1

    def _finalize_missing_bulk(self, frames: int) -> None:
        if frames <= 0:
            return
        expected_chunks = frames * self.expected_chunks_per_frame
        self.finalized_frames += frames
        self.incomplete_frames += frames
        self.wholly_missing_frames += frames
        self.expected_chunks_finalized += expected_chunks
        self.lost_chunks += expected_chunks

    def _advance_to(self, target_frame: int) -> None:
        """Finalize ``[next_finalize_frame, target_frame)`` in bounded work."""

        target_frame = min(target_frame, self.expected_end_frame)
        if target_frame <= self.next_finalize_frame:
            return
        present = sorted(
            frame for frame in self.pending
            if self.next_finalize_frame <= frame < target_frame
        )
        span = target_frame - self.next_finalize_frame
        self._finalize_missing_bulk(span - len(present))
        for frame in present:
            self._finalize_present(self.pending.pop(frame))
        self.next_finalize_frame = target_frame

    def _drain_complete_prefix(self) -> None:
        while self.next_finalize_frame < self.expected_end_frame:
            chunks = self.pending.get(self.next_finalize_frame)
            if chunks is None or len(chunks) != self.expected_chunks_per_frame:
                break
            self.pending.pop(self.next_finalize_frame)
            self._finalize_present(chunks)
            self.next_finalize_frame += 1

    def account(
        self,
        header: BurstHeader,
        *,
        datagram_bytes: int,
        monotonic_ns: int,
        interval_index: int | None,
    ) -> tuple[str, bool]:
        """Account one valid-header datagram and return status/out-of-order."""

        self.datagrams_with_valid_header += 1
        self.wire_datagram_bytes += datagram_bytes
        if header.chunks_per_frame != self.expected_chunks_per_frame:
            self.contract_mismatch_datagrams += 1
            return "CHUNKS_PER_FRAME_CONTRACT_MISMATCH", False
        if not self.expected_first_frame <= header.frame_index < self.expected_end_frame:
            self.outside_expected_range_datagrams += 1
            return "FRAME_OUTSIDE_EXPECTED_RANGE", False
        if header.frame_index < self.next_finalize_frame:
            self.late_after_finalize += 1
            return "LATE_AFTER_FINALIZE", True

        window_end = self.next_finalize_frame + self.reorder_window_frames
        if header.frame_index >= window_end:
            self._advance_to(header.frame_index - self.reorder_window_frames + 1)

        chunks = self.pending.setdefault(header.frame_index, set())
        if header.chunk_index in chunks:
            self.duplicate_chunks += 1
            return "DUPLICATE_CHUNK", False

        sequence = header.frame_index * self.expected_chunks_per_frame + header.chunk_index
        out_of_order = self.maximum_sequence_seen is not None and sequence < self.maximum_sequence_seen
        if out_of_order:
            self.out_of_order_unique_chunks += 1
        self.maximum_sequence_seen = max(sequence, self.maximum_sequence_seen or sequence)
        chunks.add(header.chunk_index)
        self.highest_seen_frame = max(header.frame_index, self.highest_seen_frame or header.frame_index)
        self.unique_chunks += 1
        self.unique_datagram_bytes += datagram_bytes
        self.unique_payload_bytes += datagram_bytes - HEADER.size
        if self.last_unique_monotonic_ns is not None:
            gap_ns = monotonic_ns - self.last_unique_monotonic_ns
            self.max_interarrival_gap_ns = max(self.max_interarrival_gap_ns, gap_ns)
            if gap_ns >= 1_000_000_000:
                self.interarrival_gaps_over_one_second += 1
        else:
            self.first_unique_monotonic_ns = monotonic_ns
        self.last_unique_monotonic_ns = monotonic_ns
        if interval_index is not None:
            self.interval_unique_chunks[interval_index] += 1
            self.interval_unique_datagram_bytes[interval_index] += datagram_bytes
        self.maximum_pending_frames_observed = max(
            self.maximum_pending_frames_observed, len(self.pending)
        )
        self._drain_complete_prefix()
        return "ACCEPTED_UNIQUE", out_of_order

    def finalize(self) -> None:
        self._advance_to(self.expected_end_frame)
        self.pending.clear()

    def summary(self, *, observed_duration_s: float) -> dict[str, Any]:
        observed_bins = min(
            len(self.interval_unique_chunks), int(math.ceil(observed_duration_s))
        )
        interval_chunks = self.interval_unique_chunks[:observed_bins]
        empty_bins = [count == 0 for count in interval_chunks]
        longest_empty_run = 0
        current_empty_run = 0
        for empty in empty_bins:
            current_empty_run = current_empty_run + 1 if empty else 0
            longest_empty_run = max(longest_empty_run, current_empty_run)
        expected = self.expected_chunks_finalized
        return {
            "stream_id": self.identity,
            "bounded_pending_frames_after_finalize": len(self.pending),
            "maximum_pending_frames_observed": self.maximum_pending_frames_observed,
            "reorder_window_frames": self.reorder_window_frames,
            "expected_first_frame": self.expected_first_frame,
            "expected_frames": self.expected_frames,
            "expected_chunks_per_frame": self.expected_chunks_per_frame,
            "datagrams_with_valid_header": self.datagrams_with_valid_header,
            "wire_datagram_bytes": self.wire_datagram_bytes,
            "unique_chunks": self.unique_chunks,
            "unique_datagram_bytes": self.unique_datagram_bytes,
            "unique_payload_bytes": self.unique_payload_bytes,
            "duplicate_chunks": self.duplicate_chunks,
            "late_after_finalize": self.late_after_finalize,
            "out_of_order_unique_chunks": self.out_of_order_unique_chunks,
            "contract_mismatch_datagrams": self.contract_mismatch_datagrams,
            "outside_expected_range_datagrams": self.outside_expected_range_datagrams,
            "finalized_frames": self.finalized_frames,
            "complete_frames": self.complete_frames,
            "incomplete_frames": self.incomplete_frames,
            "wholly_missing_frames": self.wholly_missing_frames,
            "expected_chunks": expected,
            "received_unique_chunks": self.received_chunks_finalized,
            "lost_chunks": self.lost_chunks,
            "chunk_delivery_ratio": (
                self.received_chunks_finalized / expected if expected else None
            ),
            "complete_frame_ratio": (
                self.complete_frames / self.finalized_frames
                if self.finalized_frames else None
            ),
            "unique_datagram_goodput_mbps": (
                self.unique_datagram_bytes * 8.0 / observed_duration_s / 1e6
                if observed_duration_s > 0 else None
            ),
            "unique_payload_goodput_mbps": (
                self.unique_payload_bytes * 8.0 / observed_duration_s / 1e6
                if observed_duration_s > 0 else None
            ),
            "max_interarrival_gap_s": self.max_interarrival_gap_ns / 1e9,
            "interarrival_gaps_over_one_second": self.interarrival_gaps_over_one_second,
            "one_second_bins": observed_bins,
            "empty_one_second_bins": sum(empty_bins),
            "max_consecutive_empty_one_second_bins": longest_empty_run,
        }


class ReceiverAccounting:
    """Bounded in-memory accounting shared by the live receiver and tests."""

    def __init__(
        self,
        *,
        measurement_start_monotonic_ns: int,
        duration_s: float,
        expected_first_frame: int,
        expected_frames: int,
        expected_chunks_per_frame: int,
        max_streams: int,
        reorder_window_frames: int,
        max_chunks_per_frame: int,
    ) -> None:
        if not 0 < duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be in (0, {MAX_DURATION_S:g}]")
        if not 0 < max_streams <= MAX_STREAMS_LIMIT:
            raise ValueError(f"max_streams must be in 1..{MAX_STREAMS_LIMIT}")
        if not 0 < reorder_window_frames <= MAX_REORDER_WINDOW_FRAMES_LIMIT:
            raise ValueError(
                "reorder_window_frames must be in "
                f"1..{MAX_REORDER_WINDOW_FRAMES_LIMIT}"
            )
        if not 0 < max_chunks_per_frame <= MAX_CHUNKS_PER_FRAME_LIMIT:
            raise ValueError(
                f"max_chunks_per_frame must be in 1..{MAX_CHUNKS_PER_FRAME_LIMIT}"
            )
        if not 0 < expected_chunks_per_frame <= max_chunks_per_frame:
            raise ValueError(
                "expected_chunks_per_frame must be positive and no greater than its bound"
            )
        if expected_frames <= 0:
            raise ValueError("expected_frames must be positive")
        if expected_first_frame < 0 or expected_first_frame + expected_frames > UINT32_LIMIT:
            raise ValueError("expected frame range must fit uint32")
        self.measurement_start_monotonic_ns = measurement_start_monotonic_ns
        self.duration_s = duration_s
        self.expected_first_frame = expected_first_frame
        self.expected_frames = expected_frames
        self.expected_chunks_per_frame = expected_chunks_per_frame
        self.max_streams = max_streams
        self.reorder_window_frames = reorder_window_frames
        self.max_chunks_per_frame = max_chunks_per_frame
        self.interval_bin_count = max(1, int(math.ceil(duration_s)))
        self.streams: dict[str, StreamAccounting] = {}
        self.datagrams_total = 0
        self.wire_bytes_total = 0
        self.malformed_datagrams = 0
        self.stream_limit_exceeded_datagrams = 0
        self.global_interval_unique_chunks = _empty_bins(self.interval_bin_count)
        self.global_interval_unique_datagram_bytes = _empty_bins(self.interval_bin_count)

    def _interval_index(self, monotonic_ns: int) -> int | None:
        elapsed_ns = monotonic_ns - self.measurement_start_monotonic_ns
        if elapsed_ns < 0:
            return None
        index = int(elapsed_ns // 1_000_000_000)
        return index if 0 <= index < self.interval_bin_count else None

    def ingest(
        self,
        data: bytes,
        address: tuple[str, int],
        *,
        wall_time_ns: int,
        monotonic_ns: int,
    ) -> dict[str, Any]:
        self.datagrams_total += 1
        self.wire_bytes_total += len(data)
        event: dict[str, Any] = {
            "event_type": "datagram",
            "receiver_wall_time_ns": wall_time_ns,
            "receiver_monotonic_ns": monotonic_ns,
            "elapsed_s": (monotonic_ns - self.measurement_start_monotonic_ns) / 1e9,
            "source_ip": address[0],
            "source_port": address[1],
            "stream_id": stream_id(address),
            "datagram_bytes": len(data),
        }
        try:
            header = parse_ssburst_datagram(
                data, max_chunks_per_frame=self.max_chunks_per_frame
            )
        except PacketContractError as exc:
            self.malformed_datagrams += 1
            event.update({"status": "MALFORMED", "reason": exc.reason})
            return event

        event.update({
            "frame_index": header.frame_index,
            "chunk_index": header.chunk_index,
            "chunks_per_frame": header.chunks_per_frame,
            "declared_datagram_bytes": header.declared_datagram_bytes,
            "payload_bytes": len(data) - HEADER.size,
        })
        identity = stream_id(address)
        tracker = self.streams.get(identity)
        if tracker is None:
            if len(self.streams) >= self.max_streams:
                self.stream_limit_exceeded_datagrams += 1
                event.update({
                    "status": "STREAM_LIMIT_EXCEEDED",
                    "reason": "MAX_STREAMS_BOUND",
                })
                return event
            tracker = StreamAccounting(
                identity=identity,
                expected_first_frame=self.expected_first_frame,
                expected_frames=self.expected_frames,
                expected_chunks_per_frame=self.expected_chunks_per_frame,
                reorder_window_frames=self.reorder_window_frames,
                interval_bin_count=self.interval_bin_count,
            )
            self.streams[identity] = tracker

        interval_index = self._interval_index(monotonic_ns)
        status, out_of_order = tracker.account(
            header,
            datagram_bytes=len(data),
            monotonic_ns=monotonic_ns,
            interval_index=interval_index,
        )
        event.update({"status": status, "out_of_order": out_of_order})
        if status == "ACCEPTED_UNIQUE" and interval_index is not None:
            self.global_interval_unique_chunks[interval_index] += 1
            self.global_interval_unique_datagram_bytes[interval_index] += len(data)
        return event

    def finalize(
        self,
        *,
        end_monotonic_ns: int,
        stop_reason: str,
        clean_shutdown: bool,
    ) -> dict[str, Any]:
        observed_duration_s = max(
            0.0,
            min(
                self.duration_s,
                (end_monotonic_ns - self.measurement_start_monotonic_ns) / 1e9,
            ),
        )
        for tracker in self.streams.values():
            tracker.finalize()
        observed_bins = min(
            len(self.global_interval_unique_chunks), int(math.ceil(observed_duration_s))
        )
        empty_bins = [
            count == 0
            for count in self.global_interval_unique_chunks[:observed_bins]
        ]
        longest_empty_run = 0
        current_empty_run = 0
        for empty in empty_bins:
            current_empty_run = current_empty_run + 1 if empty else 0
            longest_empty_run = max(longest_empty_run, current_empty_run)
        return {
            "schema": "scenesense.ue_n3_structured_udp_receiver_summary.v1",
            "status": "CAPTURED" if clean_shutdown else "FAILED",
            "stop_reason": stop_reason,
            "clean_shutdown": clean_shutdown,
            "created_at": utc_now(),
            "wire_contract": {
                "struct_format": HEADER.format,
                "header_bytes": HEADER.size,
                "magic_ascii": MAGIC.decode("ascii"),
                "stream_identity": "SOURCE_IPV4_AND_UDP_PORT",
            },
            "bounds": {
                "max_streams": self.max_streams,
                "reorder_window_frames": self.reorder_window_frames,
                "max_chunks_per_frame": self.max_chunks_per_frame,
                "one_second_bins": self.interval_bin_count,
            },
            "measurement": {
                "configured_duration_s": self.duration_s,
                "observed_duration_s": observed_duration_s,
                "expected_first_frame": self.expected_first_frame,
                "expected_frames_per_stream": self.expected_frames,
                "expected_chunks_per_frame": self.expected_chunks_per_frame,
            },
            "datagrams_total": self.datagrams_total,
            "wire_bytes_total": self.wire_bytes_total,
            "malformed_datagrams": self.malformed_datagrams,
            "stream_limit_exceeded_datagrams": self.stream_limit_exceeded_datagrams,
            "valid_stream_count": len(self.streams),
            "no_valid_stream_observed": not self.streams,
            "observed_one_second_bins": observed_bins,
            "empty_one_second_bins_global": sum(empty_bins),
            "max_consecutive_empty_one_second_bins_global": longest_empty_run,
            "streams": [
                self.streams[key].summary(observed_duration_s=observed_duration_s)
                for key in sorted(self.streams)
            ],
            "claim_boundary": (
                "TRANSPORT_SCREEN_EVIDENCE_ONLY_NOT_CONNECTION_PROOF_"
                "OR_OPERATIONAL_SNR_BOUND"
            ),
        }

    def interval_events(self, *, observed_duration_s: float | None = None) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        bin_count = len(self.global_interval_unique_chunks)
        if observed_duration_s is not None:
            bin_count = min(bin_count, int(math.ceil(observed_duration_s)))
        for index, chunks in enumerate(self.global_interval_unique_chunks[:bin_count]):
            events.append({
                "event_type": "one_second_interval",
                "scope": "GLOBAL",
                "interval_index": index,
                "interval_start_s": float(index),
                "interval_end_s": float(index + 1),
                "unique_chunks": chunks,
                "unique_datagram_bytes": self.global_interval_unique_datagram_bytes[index],
                "empty_interval": chunks == 0,
            })
        for identity in sorted(self.streams):
            tracker = self.streams[identity]
            for index, chunks in enumerate(tracker.interval_unique_chunks[:bin_count]):
                events.append({
                    "event_type": "one_second_interval",
                    "scope": "STREAM",
                    "stream_id": identity,
                    "interval_index": index,
                    "interval_start_s": float(index),
                    "interval_end_s": float(index + 1),
                    "unique_chunks": chunks,
                    "unique_datagram_bytes": tracker.interval_unique_datagram_bytes[index],
                    "empty_interval": chunks == 0,
                })
        return events


class StructuredUdpReceiver:
    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        events_jsonl: Path,
        summary_json: Path,
        ready_json: Path | None,
        duration_s: float,
        expected_first_frame: int,
        expected_frames: int,
        expected_chunks_per_frame: int,
        max_streams: int,
        reorder_window_frames: int,
        max_chunks_per_frame: int,
        socket_receive_buffer_bytes: int,
        poll_timeout_s: float = 0.2,
    ) -> None:
        # Validate every resource bound before creating directories or opening a
        # socket.  This keeps invalid invocations side-effect free and avoids a
        # leaked socket if ReceiverAccounting would otherwise reject the values
        # only after run() installs signal handlers.
        if not 0 < duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be in (0, {MAX_DURATION_S:g}]")
        if not 0 <= port <= 65_535:
            raise ValueError("port must be in 0..65535")
        if expected_frames <= 0:
            raise ValueError("expected_frames must be positive")
        if expected_first_frame < 0 or expected_first_frame + expected_frames > UINT32_LIMIT:
            raise ValueError("expected frame range must fit uint32")
        if not 0 < max_streams <= MAX_STREAMS_LIMIT:
            raise ValueError(f"max_streams must be in 1..{MAX_STREAMS_LIMIT}")
        if not 0 < reorder_window_frames <= MAX_REORDER_WINDOW_FRAMES_LIMIT:
            raise ValueError(
                "reorder_window_frames must be in "
                f"1..{MAX_REORDER_WINDOW_FRAMES_LIMIT}"
            )
        if not 0 < max_chunks_per_frame <= MAX_CHUNKS_PER_FRAME_LIMIT:
            raise ValueError(
                f"max_chunks_per_frame must be in 1..{MAX_CHUNKS_PER_FRAME_LIMIT}"
            )
        if not 0 < expected_chunks_per_frame <= max_chunks_per_frame:
            raise ValueError(
                "expected_chunks_per_frame must be positive and no greater than its bound"
            )
        if socket_receive_buffer_bytes <= 0:
            raise ValueError("socket_receive_buffer_bytes must be positive")
        if not math.isfinite(poll_timeout_s) or poll_timeout_s <= 0:
            raise ValueError("poll_timeout_s must be finite and positive")
        for output in (events_jsonl, summary_json, ready_json):
            if output is not None and output.exists():
                raise FileExistsError(f"create-only output already exists: {output}")
        self.events_jsonl = events_jsonl.resolve()
        self.summary_json = summary_json.resolve()
        self.ready_json = ready_json.resolve() if ready_json is not None else None
        self.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.summary_json.parent.mkdir(parents=True, exist_ok=True)
        if self.ready_json is not None:
            self.ready_json.parent.mkdir(parents=True, exist_ok=True)
        self.duration_s = duration_s
        self.stop_requested = threading.Event()
        self.stop_reason = "DURATION_COMPLETE"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, int(socket_receive_buffer_bytes)
        )
        self.socket.bind((bind_host, port))
        self.socket.settimeout(poll_timeout_s)
        self.local_address = self.socket.getsockname()
        self.accounting: ReceiverAccounting | None = None
        self.accounting_kwargs = {
            "duration_s": duration_s,
            "expected_first_frame": expected_first_frame,
            "expected_frames": expected_frames,
            "expected_chunks_per_frame": expected_chunks_per_frame,
            "max_streams": max_streams,
            "reorder_window_frames": reorder_window_frames,
            "max_chunks_per_frame": max_chunks_per_frame,
        }

    def request_stop(self, reason: str = "STOP_REQUESTED") -> None:
        self.stop_reason = reason
        self.stop_requested.set()

    def run(self, *, install_signal_handlers: bool = True) -> dict[str, Any]:
        previous_handlers: dict[signal.Signals, Any] = {}

        def stop_from_signal(signum: int, _frame: Any) -> None:
            self.request_stop(f"SIGNAL_{signal.Signals(signum).name}")

        if install_signal_handlers:
            for caught in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous_handlers[caught] = signal.getsignal(caught)
                signal.signal(caught, stop_from_signal)

        start_mono = time.monotonic_ns()
        self.accounting = ReceiverAccounting(
            measurement_start_monotonic_ns=start_mono,
            **self.accounting_kwargs,
        )
        deadline_ns = start_mono + int(self.duration_s * 1e9)
        summary: dict[str, Any]
        failure: Exception | None = None
        self.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.events_jsonl.open("x", encoding="utf-8", buffering=1) as events:
                if self.ready_json is not None:
                    atomic_json(self.ready_json, {
                        "schema": "scenesense.ue_n3_structured_udp_receiver_ready.v1",
                        "status": "READY",
                        "bind_host": self.local_address[0],
                        "port": self.local_address[1],
                        "measurement_start_monotonic_ns": start_mono,
                        "duration_s": self.duration_s,
                    })
                while not self.stop_requested.is_set():
                    now_mono = time.monotonic_ns()
                    if now_mono >= deadline_ns:
                        self.stop_reason = "DURATION_COMPLETE"
                        break
                    try:
                        data, address = self.socket.recvfrom(65_535)
                    except socket.timeout:
                        continue
                    event = self.accounting.ingest(
                        data,
                        (str(address[0]), int(address[1])),
                        wall_time_ns=time.time_ns(),
                        monotonic_ns=time.monotonic_ns(),
                    )
                    events.write(json.dumps(event, sort_keys=True) + "\n")
                end_mono = time.monotonic_ns()
                summary = self.accounting.finalize(
                    end_monotonic_ns=end_mono,
                    stop_reason=self.stop_reason,
                    clean_shutdown=True,
                )
                for event in self.accounting.interval_events(
                    observed_duration_s=float(summary["measurement"]["observed_duration_s"])
                ):
                    events.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception as exc:
            failure = exc
            end_mono = time.monotonic_ns()
            summary = self.accounting.finalize(
                end_monotonic_ns=end_mono,
                stop_reason=f"EXCEPTION_{type(exc).__name__}",
                clean_shutdown=False,
            )
            summary["error"] = str(exc)
        finally:
            self.socket.close()
            if install_signal_handlers:
                for caught, previous in previous_handlers.items():
                    signal.signal(caught, previous)
        atomic_json(self.summary_json, summary)
        if failure is not None:
            raise failure
        return summary


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed >= UINT32_LIMIT:
        raise argparse.ArgumentTypeError("value must be in 0..2^32-1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=nonnegative_int, default=56_130)
    parser.add_argument("--events-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--ready-json")
    parser.add_argument("--duration-s", type=positive_float, default=60.0)
    parser.add_argument("--expected-first-frame", type=nonnegative_int, default=0)
    parser.add_argument("--expected-frames", type=positive_int, default=600)
    parser.add_argument("--expected-chunks-per-frame", type=positive_int, default=1)
    parser.add_argument("--max-streams", type=positive_int, default=1)
    parser.add_argument("--reorder-window-frames", type=positive_int, default=64)
    parser.add_argument("--max-chunks-per-frame", type=positive_int, default=1_024)
    parser.add_argument("--socket-receive-buffer-bytes", type=positive_int, default=8 * 1024 * 1024)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.port > 65_535:
        raise SystemExit("--port must be in 0..65535")
    if args.expected_first_frame + args.expected_frames > UINT32_LIMIT:
        raise SystemExit("expected frame range exceeds uint32")
    if args.expected_chunks_per_frame > args.max_chunks_per_frame:
        raise SystemExit("expected chunks per frame exceeds configured maximum")
    receiver = StructuredUdpReceiver(
        bind_host=args.bind_host,
        port=args.port,
        events_jsonl=Path(args.events_jsonl),
        summary_json=Path(args.summary_json),
        ready_json=Path(args.ready_json) if args.ready_json else None,
        duration_s=args.duration_s,
        expected_first_frame=args.expected_first_frame,
        expected_frames=args.expected_frames,
        expected_chunks_per_frame=args.expected_chunks_per_frame,
        max_streams=args.max_streams,
        reorder_window_frames=args.reorder_window_frames,
        max_chunks_per_frame=args.max_chunks_per_frame,
        socket_receive_buffer_bytes=args.socket_receive_buffer_bytes,
    )
    summary = receiver.run()
    print(json.dumps({
        "status": summary["status"],
        "summary_json": str(receiver.summary_json),
        "events_jsonl": str(receiver.events_jsonl),
        "stop_reason": summary["stop_reason"],
    }, sort_keys=True))
    return 0 if summary["status"] == "CAPTURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

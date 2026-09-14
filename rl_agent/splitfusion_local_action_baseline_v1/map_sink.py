#!/usr/bin/env python3
"""Minimal edge map sink for one-shot LOCAL object-result uploads.

The sink runs inside the OAI external-data-network namespace. It validates the
real compact LOCAL schema, installs only monotonically newer frames, and emits
an authoritative ACK/NACK to the source socket. It never runs perception and
never reconstructs or awards segmentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence


CHUNK_HEADER = struct.Struct("!IHH")
CLOCK_RAW = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
LOCAL_SCHEMA = "scenesense.splitfusion.local_object_result.v1"
LOCAL_PROFILE_ID = "local_fcos_r50_fpn_p2_p7_p025_v1"
ACK_SCHEMA = "scenesense.splitfusion.local_map_ack.v1"
FIELDS = (
    "frame_id",
    "stream_id",
    "source_ip",
    "capture_raw_ns",
    "local_result_available_raw_ns",
    "first_datagram_raw_ns",
    "complete_raw_ns",
    "edge_install_raw_ns",
    "feedback_emit_raw_ns",
    "payload_bytes",
    "chunks",
    "object_count",
    "status",
    "rejection_reason",
)


def raw_ns() -> int:
    return time.clock_gettime_ns(CLOCK_RAW)


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return False


def decode_and_validate(payload: bytes, message_id: int) -> dict[str, Any]:
    value = json.loads(zlib.decompress(payload).decode("utf-8"))
    if value.get("schema") != LOCAL_SCHEMA:
        raise ValueError("schema")
    if value.get("profile_id") != LOCAL_PROFILE_ID:
        raise ValueError("profile")
    if int(value.get("frame_id", -1)) != int(message_id):
        raise ValueError("frame_identity")
    if not str(value.get("stream_id") or ""):
        raise ValueError("stream_identity")
    capture = int(value.get("capture_timestamp_ns", -1))
    available = int(value.get("local_result_available_ns", -1))
    if capture < 0 or available < capture:
        raise ValueError("timestamp_order")
    objects = value.get("objects")
    if not isinstance(objects, list) or not finite_tree(objects):
        raise ValueError("objects")
    segmentation = value.get("segmentation") or {}
    if segmentation.get("transported") is not False or segmentation.get("edge_map_credit") is not False:
        raise ValueError("segmentation_boundary")
    for record in objects:
        if (
            not isinstance(record, dict)
            or int(record.get("frame_id", -1)) != message_id
            or str(record.get("stream_id")) != str(value["stream_id"])
            or int(record.get("capture_timestamp_ns", -1)) != capture
        ):
            raise ValueError("object_identity")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stop = Path(args.stop_file).resolve()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(args.socket_buffer_bytes))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(args.socket_buffer_bytes))
    sock.bind((str(args.bind_host), int(args.port)))
    sock.settimeout(0.1)
    partial: dict[tuple[str, int], dict[str, Any]] = {}
    latest_by_stream: dict[str, tuple[int, int]] = {}
    rows: list[dict[str, Any]] = []
    datagrams = duplicates = malformed = expired = 0
    while not stop.exists():
        now = raw_ns()
        stale_keys = [
            key for key, item in partial.items()
            if now - int(item["first_raw_ns"]) > int(float(args.reassembly_timeout_s) * 1e9)
        ]
        for key in stale_keys:
            expired += 1
            partial.pop(key, None)
        try:
            packet, address = sock.recvfrom(65535)
        except socket.timeout:
            continue
        received = raw_ns()
        datagrams += 1
        if len(packet) < CHUNK_HEADER.size:
            malformed += 1
            continue
        message_id, chunk_index, total_chunks = CHUNK_HEADER.unpack_from(packet)
        if total_chunks <= 0 or chunk_index >= total_chunks:
            malformed += 1
            continue
        key = (str(address[0]), int(message_id))
        item = partial.setdefault(
            key,
            {
                "first_raw_ns": received,
                "total_chunks": int(total_chunks),
                "chunks": {},
                "address": address,
            },
        )
        if int(item["total_chunks"]) != int(total_chunks):
            malformed += 1
            partial.pop(key, None)
            continue
        chunks = item["chunks"]
        if chunk_index in chunks:
            duplicates += 1
            continue
        chunks[int(chunk_index)] = packet[CHUNK_HEADER.size :]
        if len(chunks) != total_chunks:
            continue
        payload = b"".join(chunks[index] for index in range(total_chunks))
        status = "NACK_REJECTED"
        reason = ""
        value: dict[str, Any] = {}
        installed = 0
        emitted = 0
        try:
            if str(address[0]) != str(args.expected_source_ip):
                raise ValueError("source_ip")
            value = decode_and_validate(payload, int(message_id))
            stream = str(value["stream_id"])
            candidate = (int(value["capture_timestamp_ns"]), int(message_id))
            if candidate <= latest_by_stream.get(stream, (-1, -1)):
                raise ValueError("stale_or_out_of_order")
            installed = raw_ns()
            latest_by_stream[stream] = candidate
            status = "ACK_INSTALLED"
        except Exception as exc:
            reason = str(exc)
        ack = {
            "schema": ACK_SCHEMA,
            "frame_id": int(message_id),
            "stream_id": str(value.get("stream_id") or ""),
            "capture_timestamp_ns": int(value.get("capture_timestamp_ns") or 0),
            "local_result_available_ns": int(value.get("local_result_available_ns") or 0),
            "edge_install_raw_ns": int(installed),
            "status": status,
            "accepted": status == "ACK_INSTALLED",
            "rejection_reason": reason,
        }
        emitted = raw_ns()
        ack["feedback_emit_raw_ns"] = emitted
        sock.sendto(
            json.dumps(ack, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            address,
        )
        rows.append(
            {
                "frame_id": int(message_id),
                "stream_id": str(value.get("stream_id") or ""),
                "source_ip": str(address[0]),
                "capture_raw_ns": int(value.get("capture_timestamp_ns") or 0),
                "local_result_available_raw_ns": int(value.get("local_result_available_ns") or 0),
                "first_datagram_raw_ns": int(item["first_raw_ns"]),
                "complete_raw_ns": received,
                "edge_install_raw_ns": installed,
                "feedback_emit_raw_ns": emitted,
                "payload_bytes": len(payload),
                "chunks": int(total_chunks),
                "object_count": len(value.get("objects") or []),
                "status": status,
                "rejection_reason": reason,
            }
        )
        partial.pop(key, None)
    sock.close()
    with (output / "map_sink_frames.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output / "map_sink_summary.json",
        {
            "schema": "scenesense.splitfusion.local_map_sink_summary.v1",
            "complete_messages": len(rows),
            "ack_installed": sum(row["status"] == "ACK_INSTALLED" for row in rows),
            "nack_rejected": sum(row["status"] == "NACK_REJECTED" for row in rows),
            "datagrams_received": datagrams,
            "duplicate_datagrams": duplicates,
            "malformed_datagrams": malformed,
            "expired_reassemblies": expired,
            "partial_at_stop": len(partial),
        },
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--expected-source-ip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--socket-buffer-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--reassembly-timeout-s", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

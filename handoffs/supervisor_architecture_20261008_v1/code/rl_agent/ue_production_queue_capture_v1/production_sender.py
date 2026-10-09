#!/usr/bin/env python3
"""Production-packetized (`!IHH`, 12,500-byte) tagged sender for the UE queue capture.

This is the deployed SplitFusion uplink packetization, not a 1,200-byte
calibration chunker.  Each frame replays the exact ``total_transmitted_bytes``
of one registered quality-grid row.

Frozen socket semantics
-----------------------
The socket is **blocking** with a large send buffer.  Every datagram handed to
the socket therefore enters the UE IP/PDCP/RLC path, which is what makes the
sender -> PDCP -> RLC byte residual closable.  A frame is never skipped and a
datagram is never silently dropped; ``datagrams_dropped_at_socket`` must be 0.

Because the guard byte role deliberately offers more than the link can carry,
the sender WILL fall behind the nominal 10-Hz grid once the UE queue
saturates.  That is expected, is recorded per frame as ``schedule_lag_ms``,
and is not a failure.  The authoritative ingress instants are the measured
socket-handoff stamps, never the nominal schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

from . import contract as C


CHUNK_HEADER = struct.Struct(C.UDP_CHUNK_HEADER_STRUCT)

SUMMARY_SCHEMA = "scenesense.production_queue_capture_sender_summary.v1"
FRAME_SCHEMA = "scenesense.production_queue_capture_sender_frame.v1"
READY_SCHEMA = "scenesense.production_queue_capture_sender_ready.v1"
EPOCH_SCHEMA = "scenesense.production_queue_capture_shared_epoch.v1"

FRAME_FIELDS = (
    "schema", "cell_id", "frame_index", "block_index", "tier", "action_id",
    "mode_id", "q_e4", "frame_index_in_block", "is_first_frame_of_block",
    "previous_tier", "row_sha256", "epoch_monotonic_ns",
    "scheduled_monotonic_ns", "frame_open_monotonic_ns", "frame_open_wall_ns",
    "first_send_monotonic_ns", "last_send_monotonic_ns",
    "total_transmitted_bytes", "datagram_count",
    "datagrams_handed_to_socket", "datagrams_dropped_at_socket",
    "application_payload_bytes_handed_to_socket",
    "udp_application_bytes_handed_to_socket", "send_span_ms",
    "schedule_lag_ms", "terminal_reason",
)


class SenderError(RuntimeError):
    """A frozen epoch, schedule, packetization or create-only gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SenderError(message)


def build_frame_bytes(total_transmitted_bytes: int, row_sha256: str) -> bytes:
    """Deterministic incompressible-enough filler of the exact registered length.

    The capture is byte-only: content carries no perception meaning.  Bytes are
    derived from the authority row digest so the wire image is reproducible.
    """
    seed = bytes.fromhex(row_sha256)
    out = bytearray()
    counter = 0
    while len(out) < total_transmitted_bytes:
        import hashlib
        out += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:total_transmitted_bytes])


def chunk_frame(payload: bytes, *, message_id: int) -> list[bytes]:
    """Exactly the deployed `!IHH` chunker at the live 12,500-byte binding."""
    capacity = C.UDP_PAYLOAD_BYTES_PER_DATAGRAM
    total = (len(payload) + capacity - 1) // capacity
    require(0 <= message_id <= 0xFFFF_FFFF, "message id exceeds the header range")
    require(1 <= total <= 0xFFFF, "chunk count exceeds the header range")
    return [
        CHUNK_HEADER.pack(message_id, index, total)
        + payload[index * capacity : (index + 1) * capacity]
        for index in range(total)
    ]


def _wait_until(epoch_ns: int) -> None:
    while True:
        remaining = epoch_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.005))


def _write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _wait_for_epoch_contract(
    path: Path, *, cell_id: str, period_ns: int, sender_ready_sha256: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    value: Any = None
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                pass
        time.sleep(0.005)
    require(type(value) is dict, "shared epoch contract was not published in time")
    expected = {
        "schema", "cell_id", "epoch_monotonic_ns", "period_ns",
        "created_monotonic_ns", "sender_ready_sha256",
    }
    require(set(value) == expected, "shared epoch contract fields drifted")
    require(value["schema"] == EPOCH_SCHEMA, "shared epoch schema drifted")
    require(value["cell_id"] == cell_id, "shared epoch cell drifted")
    require(value["period_ns"] == period_ns, "shared epoch period drifted")
    require(value["sender_ready_sha256"] == sender_ready_sha256,
            "shared epoch was not bound to this sender READY record")
    require(value["created_monotonic_ns"] < value["epoch_monotonic_ns"],
            "shared epoch was not future at publication")
    require(time.monotonic_ns() < value["epoch_monotonic_ns"],
            "sender received the shared epoch after its release instant")
    return value


def run(args: argparse.Namespace) -> int:
    schedule = json.loads(Path(args.schedule_json).read_text(encoding="utf-8"))
    require(schedule.get("schema")
            == "scenesense.production_queue_capture_payload_schedule.v1",
            "payload schedule schema drifted")
    require(schedule["cell_id"] == args.cell_id, "payload schedule cell drifted")
    frames_plan = schedule["frames"]
    require(len(frames_plan) == C.FRAMES_PER_CELL,
            "payload schedule frame count drifted")
    for plan in frames_plan:
        require(type(plan.get("port")) is int and 0 < int(plan["port"]) < 65536,
                "every scheduled frame must carry its block receiver port")
    require(args.chunk_bytes == C.UDP_CHUNK_BYTES_INCLUDING_HEADER,
            f"chunk bytes {args.chunk_bytes} is not the live production binding")
    require(CHUNK_HEADER.size == C.UDP_CHUNK_HEADER_BYTES,
            "production chunk header size drifted")

    log_path = Path(args.log_csv)
    summary_path = Path(args.summary_json)
    ready_path = Path(args.ready_json)
    epoch_path = Path(args.epoch_contract)
    require(not log_path.exists() and not summary_path.exists()
            and not ready_path.exists() and not epoch_path.exists(),
            "sender outputs are create-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Materialize every frame image up front so wire construction never competes
    # with the send schedule.
    images: dict[int, list[bytes]] = {}
    for plan in frames_plan:
        total = int(plan["total_transmitted_bytes"])
        payload = build_frame_bytes(total, str(plan["row_sha256"]))
        chunks = chunk_frame(payload, message_id=int(plan["frame_index"]))
        require(len(chunks) == int(plan["datagram_count"]),
                "materialized datagram count disagrees with the schedule")
        require(sum(len(chunk) for chunk in chunks)
                == total + C.UDP_CHUNK_HEADER_BYTES * len(chunks),
                "materialized wire bytes disagree with the production identity")
        images[int(plan["frame_index"])] = chunks

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.socket_sendbuf)
    effective_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    if args.bind_host:
        sock.bind((args.bind_host, 0))
    sock.setblocking(True)          # frozen: backpressure, never a silent drop

    period_ns = int(1e9 / C.FPS)
    ready = {
        "schema": READY_SCHEMA, "cell_id": args.cell_id,
        "ready_monotonic_ns": time.monotonic_ns(),
        "bind_host": args.bind_host, "remote_host": args.remote_host,
        "block_ports": sorted({int(plan["port"]) for plan in frames_plan}),
        "socket_sendbuf_requested": args.socket_sendbuf,
        "socket_sendbuf_effective": effective_sndbuf,
        "period_ns": period_ns,
        "schedule_sha256": schedule["schedule_sha256"],
        "chunk_bytes": args.chunk_bytes,
        "packetization_identity": C.PACKETIZATION_IDENTITY,
    }
    _write_json_create(ready_path, ready)
    ready_sha256 = C.sha256_file(ready_path)
    epoch_contract = _wait_for_epoch_contract(
        epoch_path, cell_id=args.cell_id, period_ns=period_ns,
        sender_ready_sha256=ready_sha256,
        timeout_s=args.epoch_contract_timeout_s,
    )
    epoch_ns = int(epoch_contract["epoch_monotonic_ns"])
    remote_host = args.remote_host

    rows: list[dict[str, Any]] = []
    previous_tier: str | None = None
    hard_deadline_ns = epoch_ns + int(args.cell_wall_clock_guard_s * 1e9)

    _wait_until(epoch_ns)
    actual_release_ns = time.monotonic_ns()
    start_wall_ns = time.time_ns()
    try:
        for plan in frames_plan:
            frame_index = int(plan["frame_index"])
            scheduled = epoch_ns + frame_index * period_ns
            _wait_until(scheduled)
            require(time.monotonic_ns() < hard_deadline_ns,
                    "cell exceeded its frozen wall-clock guard")
            open_mono = time.monotonic_ns()
            open_wall = time.time_ns()
            chunks = images[frame_index]
            remote = (remote_host, int(plan["port"]))
            handed = dropped = payload_handed = wire_handed = 0
            first_send: int | None = None
            last_send: int | None = None
            for chunk in chunks:
                try:
                    sock.sendto(chunk, remote)
                except OSError:
                    dropped += 1
                    continue
                stamp = time.monotonic_ns()
                first_send = stamp if first_send is None else first_send
                last_send = stamp
                handed += 1
                payload_handed += len(chunk) - C.UDP_CHUNK_HEADER_BYTES
                wire_handed += len(chunk)
            terminal = (
                "ALL_DATAGRAMS_HANDED_TO_SOCKET" if dropped == 0
                else "SOCKET_ERROR_ALL_DATAGRAMS_LOST" if handed == 0
                else "SOCKET_ERROR_PARTIAL_FRAME"
            )
            rows.append({
                "schema": FRAME_SCHEMA, "cell_id": args.cell_id,
                "frame_index": frame_index,
                "block_index": int(plan["block_index"]),
                "tier": plan["tier"], "action_id": int(plan["action_id"]),
                "mode_id": int(plan["mode_id"]), "q_e4": int(plan["q_e4"]),
                "frame_index_in_block": frame_index % C.FRAMES_PER_BLOCK,
                "is_first_frame_of_block":
                    frame_index % C.FRAMES_PER_BLOCK == 0,
                "previous_tier": (previous_tier
                                  if frame_index % C.FRAMES_PER_BLOCK == 0
                                  else plan["tier"]),
                "row_sha256": plan["row_sha256"],
                "epoch_monotonic_ns": epoch_ns,
                "scheduled_monotonic_ns": scheduled,
                "frame_open_monotonic_ns": open_mono,
                "frame_open_wall_ns": open_wall,
                "first_send_monotonic_ns": first_send,
                "last_send_monotonic_ns": last_send,
                "total_transmitted_bytes": int(plan["total_transmitted_bytes"]),
                "datagram_count": int(plan["datagram_count"]),
                "datagrams_handed_to_socket": handed,
                "datagrams_dropped_at_socket": dropped,
                "application_payload_bytes_handed_to_socket": payload_handed,
                "udp_application_bytes_handed_to_socket": wire_handed,
                "send_span_ms": ((last_send - first_send) / 1e6
                                 if first_send is not None else None),
                "schedule_lag_ms": (open_mono - scheduled) / 1e6,
                "terminal_reason": terminal,
            })
            if frame_index % C.FRAMES_PER_BLOCK == C.FRAMES_PER_BLOCK - 1:
                previous_tier = str(plan["tier"])
    finally:
        sock.close()
    end_ns = time.monotonic_ns()

    with log_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": SUMMARY_SCHEMA, "cell_id": args.cell_id,
        "frames": len(rows), "fps": C.FPS,
        "epoch_monotonic_ns": epoch_ns,
        "actual_epoch_release_monotonic_ns": actual_release_ns,
        "start_wall_ns": start_wall_ns, "end_monotonic_ns": end_ns,
        "sender_ready_sha256": ready_sha256,
        "epoch_contract_sha256": C.sha256_file(epoch_path),
        "schedule_sha256": schedule["schedule_sha256"],
        "packetization": {
            "identity": C.PACKETIZATION_IDENTITY,
            "chunk_bytes_including_header": args.chunk_bytes,
            "header_struct": C.UDP_CHUNK_HEADER_STRUCT,
            "header_bytes": C.UDP_CHUNK_HEADER_BYTES,
            "payload_bytes_per_datagram": C.UDP_PAYLOAD_BYTES_PER_DATAGRAM,
            "retransmission": C.RETRANSMISSION,
        },
        "socket": {
            "blocking": True,
            "sendbuf_requested": args.socket_sendbuf,
            "sendbuf_effective": effective_sndbuf,
            "bind_host": args.bind_host, "remote_host": args.remote_host,
            "block_ports": sorted({int(plan["port"]) for plan in frames_plan}),
        },
        "totals": {
            "datagrams_handed_to_socket":
                sum(row["datagrams_handed_to_socket"] for row in rows),
            "datagrams_dropped_at_socket":
                sum(row["datagrams_dropped_at_socket"] for row in rows),
            "application_payload_bytes_handed_to_socket":
                sum(row["application_payload_bytes_handed_to_socket"]
                    for row in rows),
            "udp_application_bytes_handed_to_socket":
                sum(row["udp_application_bytes_handed_to_socket"]
                    for row in rows),
            "planned_total_transmitted_bytes":
                sum(int(plan["total_transmitted_bytes"])
                    for plan in frames_plan),
            "planned_datagrams":
                sum(int(plan["datagram_count"]) for plan in frames_plan),
        },
        "schedule_lag_ms": {
            "max": max(row["schedule_lag_ms"] for row in rows),
            "final": rows[-1]["schedule_lag_ms"],
        },
        "log_csv_sha256": C.sha256_file(log_path),
    }
    _write_json_create(summary_path, summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--bind-host", default="")
    parser.add_argument("--chunk-bytes", type=int,
                        default=C.UDP_CHUNK_BYTES_INCLUDING_HEADER)
    parser.add_argument("--socket-sendbuf", type=int, default=64 << 20)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--ready-json", required=True)
    parser.add_argument("--epoch-contract", required=True)
    parser.add_argument("--epoch-contract-timeout-s", type=float, default=30.0)
    parser.add_argument("--cell-wall-clock-guard-s", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

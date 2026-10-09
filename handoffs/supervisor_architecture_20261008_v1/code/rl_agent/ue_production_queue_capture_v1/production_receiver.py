#!/usr/bin/env python3
"""Edge-side production `!IHH` receiver with exact reassembly timing.

Records, per sent frame, the first datagram arrival, the last datagram
arrival, and the instant the frame became completely reassembled.  Frames that
never complete are emitted explicitly as incomplete terminals; they are never
dropped, and they can never be promoted to a success.

IPv4 fragment accounting is observed from the kernel counters around the
capture window rather than assumed, because a 12,500-byte UDP application
datagram is far above the 1,500-byte path MTU and therefore IS fragmented.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

from . import contract as C


CHUNK_HEADER = struct.Struct(C.UDP_CHUNK_HEADER_STRUCT)

SUMMARY_SCHEMA = "scenesense.production_queue_capture_receiver_summary.v1"
FRAME_SCHEMA = "scenesense.production_queue_capture_receiver_frame.v1"

FRAME_FIELDS = (
    "schema", "cell_id", "message_id", "expected_datagrams",
    "datagrams_received", "duplicate_datagrams",
    "application_payload_bytes_received", "udp_application_bytes_received",
    "first_datagram_monotonic_ns", "last_datagram_monotonic_ns",
    "complete_reassembly_monotonic_ns", "complete", "terminal_reason",
)

SNMP_KEYS = (
    "FragOKs", "FragFails", "FragCreates",
    "ReasmReqds", "ReasmOKs", "ReasmFails", "ReasmTimeout",
)


class ReceiverError(RuntimeError):
    """A frozen receiver invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiverError(message)


def read_ip_counters(path: str = "/proc/net/snmp") -> dict[str, int]:
    """Snapshot the kernel IPv4 fragmentation/reassembly counters."""
    text = Path(path).read_text(encoding="utf-8").splitlines()
    counters: dict[str, int] = {}
    for index in range(0, len(text) - 1):
        if not text[index].startswith("Ip: "):
            continue
        names = text[index].split()[1:]
        values = text[index + 1].split()[1:]
        if len(names) != len(values):
            continue
        for name, value in zip(names, values):
            if name in SNMP_KEYS:
                counters[name] = int(value)
        break
    return counters


def _write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class _Pending:
    __slots__ = ("expected", "seen", "duplicates", "payload_bytes",
                 "wire_bytes", "first_ns", "last_ns", "complete_ns")

    def __init__(self, expected: int, now_ns: int) -> None:
        self.expected = expected
        self.seen: set[int] = set()
        self.duplicates = 0
        self.payload_bytes = 0
        self.wire_bytes = 0
        self.first_ns = now_ns
        self.last_ns = now_ns
        self.complete_ns: int | None = None


def run(args: argparse.Namespace) -> int:
    log_path = Path(args.log_csv)
    summary_path = Path(args.summary_json)
    ready_path = Path(args.ready_json)
    require(not log_path.exists() and not summary_path.exists()
            and not ready_path.exists(), "receiver outputs are create-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.socket_recvbuf)
    effective_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    sock.bind((args.bind_host, int(args.bind_port)))
    sock.settimeout(0.5)

    counters_before = read_ip_counters()
    _write_json_create(ready_path, {
        "schema": "scenesense.production_queue_capture_receiver_ready.v1",
        "cell_id": args.cell_id, "bind_host": args.bind_host,
        "bind_port": int(args.bind_port),
        "socket_recvbuf_requested": args.socket_recvbuf,
        "socket_recvbuf_effective": effective_rcvbuf,
        "ready_monotonic_ns": time.monotonic_ns(),
        "ip_counters_before": counters_before,
        "packetization_identity": C.PACKETIZATION_IDENTITY,
    })

    stopping = {"flag": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stopping["flag"] = True

    # Installing handlers only works on the main thread.  The live receiver is
    # always a main process; in-process tests are not, and must not crash.
    signals_installed = False
    try:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        signals_installed = True
    except ValueError:
        pass

    pending: dict[int, _Pending] = {}
    completed_order: list[int] = []
    malformed = 0
    total_datagrams = 0
    total_wire_bytes = 0
    first_arrival_ns: int | None = None
    last_arrival_ns: int | None = None
    idle_deadline = time.monotonic() + args.initial_idle_timeout_s

    while not stopping["flag"]:
        if time.monotonic() > idle_deadline:
            break
        try:
            datagram = sock.recv(args.recv_bufsize)
        except socket.timeout:
            continue
        except OSError:
            break
        now_ns = time.monotonic_ns()
        total_datagrams += 1
        total_wire_bytes += len(datagram)
        first_arrival_ns = first_arrival_ns or now_ns
        last_arrival_ns = now_ns
        idle_deadline = time.monotonic() + args.idle_timeout_s
        if len(datagram) < CHUNK_HEADER.size:
            malformed += 1
            continue
        message_id, index, total = CHUNK_HEADER.unpack_from(datagram)
        if total == 0 or index >= total:
            malformed += 1
            continue
        item = pending.get(message_id)
        if item is None:
            item = _Pending(total, now_ns)
            pending[message_id] = item
        if item.expected != total:
            malformed += 1
            continue
        item.last_ns = now_ns
        if index in item.seen:
            item.duplicates += 1
            continue
        item.seen.add(index)
        item.payload_bytes += len(datagram) - CHUNK_HEADER.size
        item.wire_bytes += len(datagram)
        if item.complete_ns is None and len(item.seen) == item.expected:
            item.complete_ns = now_ns
            completed_order.append(message_id)

    sock.close()
    counters_after = read_ip_counters()

    rows: list[dict[str, Any]] = []
    for message_id in sorted(pending):
        item = pending[message_id]
        complete = item.complete_ns is not None
        rows.append({
            "schema": FRAME_SCHEMA, "cell_id": args.cell_id,
            "message_id": message_id, "expected_datagrams": item.expected,
            "datagrams_received": len(item.seen),
            "duplicate_datagrams": item.duplicates,
            "application_payload_bytes_received": item.payload_bytes,
            "udp_application_bytes_received": item.wire_bytes,
            "first_datagram_monotonic_ns": item.first_ns,
            "last_datagram_monotonic_ns": item.last_ns,
            "complete_reassembly_monotonic_ns": item.complete_ns,
            "complete": complete,
            "terminal_reason": ("COMPLETE_REASSEMBLY" if complete
                                else "INCOMPLETE_REASSEMBLY"),
        })

    with log_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    delta = {
        key: counters_after.get(key, 0) - counters_before.get(key, 0)
        for key in SNMP_KEYS
    }
    summary = {
        "schema": SUMMARY_SCHEMA, "cell_id": args.cell_id,
        "messages_observed": len(rows),
        "messages_complete": sum(1 for row in rows if row["complete"]),
        "messages_incomplete": sum(1 for row in rows if not row["complete"]),
        "malformed_datagrams": malformed,
        "total_datagrams_received": total_datagrams,
        "total_udp_application_bytes_received": total_wire_bytes,
        "total_application_payload_bytes_received":
            sum(row["application_payload_bytes_received"] for row in rows),
        "duplicate_datagrams":
            sum(row["duplicate_datagrams"] for row in rows),
        "first_arrival_monotonic_ns": first_arrival_ns,
        "last_arrival_monotonic_ns": last_arrival_ns,
        "ip_counters_before": counters_before,
        "ip_counters_after": counters_after,
        "ip_counters_delta": delta,
        "observed_ip_fragmentation": delta.get("ReasmReqds", 0) > 0,
        "expected_fragments_per_full_datagram":
            C.FRAGMENTS_PER_FULL_DATAGRAM,
        "socket_recvbuf_effective": effective_rcvbuf,
        "signal_handlers_installed": signals_installed,
        "log_csv_sha256": C.sha256_file(log_path),
    }
    _write_json_create(summary_path, summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--socket-recvbuf", type=int, default=256 << 20)
    parser.add_argument("--recv-bufsize", type=int, default=65_535)
    parser.add_argument("--idle-timeout-s", type=float, default=15.0)
    parser.add_argument("--initial-idle-timeout-s", type=float, default=180.0)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--ready-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""DN-side ACK responder for the UE-N3E degraded/fallback service probe.

Runs inside the ``oai-ext-dn`` network namespace.  Every received datagram is
answered immediately with a small ACK carrying the request sequence number and
the client's original send timestamp, so the client alone can compute the
round-trip application latency without any cross-host clock alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
from pathlib import Path


MAGIC = b"N3EQ"
ACK_MAGIC = b"N3EA"
HEADER = struct.Struct("!4sIQ")  # magic, seq, client send monotonic ns
READY_SCHEMA = "scenesense.ue_n3e_fallback_ack_responder_ready.v1"
SUMMARY_SCHEMA = "scenesense.ue_n3e_fallback_ack_responder_summary.v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--ready-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--socket-buffer-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(args.socket_buffer_bytes))
    sock.bind((args.bind_host, int(args.port)))
    sock.settimeout(0.25)

    events = Path(args.events_csv)
    events.parent.mkdir(parents=True, exist_ok=True)
    handle = events.open("w", encoding="utf-8")
    handle.write("seq,request_bytes,source_ip,recv_wall_ns,recv_monotonic_ns\n")

    atomic_json(Path(args.ready_json), {
        "schema": READY_SCHEMA,
        "status": "READY",
        "bind_host": args.bind_host,
        "port": int(args.port),
        "pid": os.getpid(),
    })

    received = 0
    acked = 0
    malformed = 0
    sources: set[str] = set()
    deadline = time.monotonic() + float(args.duration_s)
    try:
        while time.monotonic() < deadline:
            try:
                payload, peer = sock.recvfrom(65535)
            except socket.timeout:
                continue
            recv_mono = time.monotonic_ns()
            recv_wall = time.time_ns()
            if len(payload) < HEADER.size:
                malformed += 1
                continue
            magic, seq, client_send_ns = HEADER.unpack_from(payload, 0)
            if magic != MAGIC:
                malformed += 1
                continue
            received += 1
            sources.add(peer[0])
            handle.write(f"{seq},{len(payload)},{peer[0]},{recv_wall},{recv_mono}\n")
            try:
                sock.sendto(HEADER.pack(ACK_MAGIC, seq, client_send_ns), peer)
                acked += 1
            except OSError:
                pass
    finally:
        handle.flush()
        handle.close()
        sock.close()
        atomic_json(Path(args.summary_json), {
            "schema": SUMMARY_SCHEMA,
            "status": "COMPLETE",
            "requests_received": received,
            "acks_sent": acked,
            "malformed_datagrams": malformed,
            "observed_source_ips": sorted(sources),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

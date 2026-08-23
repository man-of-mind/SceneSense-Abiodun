#!/usr/bin/env python3
"""UE-side request/ACK client for the UE-N3E degraded/fallback service probe.

Sends one fixed-size application payload per decision interval from the UE
tunnel address to the DN responder and records, per sequence number, the send
timestamp and the ACK arrival timestamp.  Latency is a single-clock monotonic
round trip measured entirely on the UE side.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path


MAGIC = b"N3EQ"
ACK_MAGIC = b"N3EA"
HEADER = struct.Struct("!4sIQ")
SUMMARY_SCHEMA = "scenesense.ue_n3e_fallback_ack_client_summary.v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--payload-bytes", type=int, default=2048)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--ack-drain-s", type=float, default=1.0)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    payload_bytes = int(args.payload_bytes)
    if payload_bytes < HEADER.size:
        raise SystemExit(f"--payload-bytes must be >= {HEADER.size}")
    filler = bytes(payload_bytes - HEADER.size)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((args.bind_host, 0))
    sock.settimeout(0.2)
    remote = (args.remote_host, int(args.remote_port))

    count = int(args.count)
    ack_mono: list[int | None] = [None] * count
    ack_source_ok: list[bool] = [False] * count
    lock = threading.Lock()
    stop = threading.Event()
    unexpected_acks = 0
    malformed_acks = 0

    def drain() -> None:
        nonlocal unexpected_acks, malformed_acks
        while not stop.is_set():
            try:
                data, peer = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                continue
            now = time.monotonic_ns()
            if len(data) < HEADER.size:
                with lock:
                    malformed_acks += 1
                continue
            magic, seq, _sent_ns = HEADER.unpack_from(data, 0)
            if magic != ACK_MAGIC or not 0 <= seq < count:
                with lock:
                    malformed_acks += 1
                continue
            with lock:
                if ack_mono[seq] is None:
                    ack_mono[seq] = now
                    ack_source_ok[seq] = peer[0] == args.remote_host
                else:
                    unexpected_acks += 1

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    interval_ns = int(float(args.interval_ms) * 1e6)
    send_mono: list[int] = [0] * count
    send_wall: list[int] = [0] * count
    send_ok: list[bool] = [False] * count
    send_errors: dict[str, int] = {}

    anchor = time.monotonic_ns()
    for seq in range(count):
        scheduled = anchor + seq * interval_ns
        while True:
            remaining = scheduled - time.monotonic_ns()
            if remaining <= 0:
                break
            time.sleep(min(remaining / 1e9, 0.005))
        now_mono = time.monotonic_ns()
        now_wall = time.time_ns()
        send_mono[seq] = now_mono
        send_wall[seq] = now_wall
        try:
            sock.sendto(HEADER.pack(MAGIC, seq, now_mono) + filler, remote)
            send_ok[seq] = True
        except OSError as exc:
            name = type(exc).__name__ + ":" + str(getattr(exc, "errno", "?"))
            send_errors[name] = send_errors.get(name, 0) + 1

    drain_deadline = time.monotonic() + float(args.ack_drain_s)
    while time.monotonic() < drain_deadline:
        time.sleep(0.02)
    stop.set()
    reader.join(timeout=2.0)
    sock.close()

    log = Path(args.log_csv)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(
            "seq,send_monotonic_ns,send_wall_ns,send_ok,acked,"
            "ack_monotonic_ns,ack_source_expected,ack_latency_ms\n"
        )
        for seq in range(count):
            got = ack_mono[seq]
            latency = "" if got is None else f"{(got - send_mono[seq]) / 1e6:.4f}"
            handle.write(
                f"{seq},{send_mono[seq]},{send_wall[seq]},{int(send_ok[seq])},"
                f"{int(got is not None)},{'' if got is None else got},"
                f"{int(ack_source_ok[seq])},{latency}\n"
            )

    delivered = sum(1 for value in ack_mono if value is not None)
    atomic_json(Path(args.summary_json), {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETE",
        "attempted": count,
        "send_ok": sum(send_ok),
        "acked": delivered,
        "send_errors": send_errors,
        "unexpected_duplicate_acks": unexpected_acks,
        "malformed_acks": malformed_acks,
        "payload_bytes": payload_bytes,
        "interval_ms": float(args.interval_ms),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One fixed SplitFusion payload on an exact 100-ms action-open grid.

The sender is intentionally small.  It exists because the earlier tagged
sender starts its grid at process entry; this experiment needs the traffic and
the RF-profile replay to share a preregistered future monotonic boundary.
Late frames are recorded and skipped, never burst to catch up.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import random
import socket
import time
from pathlib import Path
from typing import Any, Sequence

from rl_agent.ue_dynamic_mcs_273prb_v1 import contract as C
from rl_agent.ue_n3_structured_udp_receiver import HEADER, MAGIC


FIELDS = (
    "profile_id", "trace_id", "decision_index", "partition", "action_id",
    "scheduled_monotonic_ns", "decision_monotonic_ns", "decision_wall_ns",
    "first_send_monotonic_ns", "last_send_monotonic_ns", "payload_bytes",
    "chunks_per_frame", "chunks_sent", "chunks_dropped", "bytes_sent",
    "send_span_ms", "schedule_lag_ms", "terminal_reason",
)


def _open_create(path: Path, mode: str = "w") -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x" if mode == "w" else mode, newline="", encoding="utf-8")


def _wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.01))


def run(args: argparse.Namespace) -> int:
    if args.frames != C.FRAMES_PER_PROFILE:
        raise SystemExit(
            f"frames must equal registered {C.FRAMES_PER_PROFILE}, got {args.frames}"
        )
    if args.payload_bytes != C.PROBE_PAYLOAD_BYTES:
        raise SystemExit("payload bytes do not match the registered probe")
    if args.period_ns != C.PERIOD_NS:
        raise SystemExit("period does not match the registered 100-ms grid")
    if args.profile_id not in C.PROFILE_IDS:
        raise SystemExit(f"unregistered profile {args.profile_id!r}")

    payload = random.Random(args.payload_seed).randbytes(args.payload_bytes)
    spans: list[tuple[int, int]] = []
    offset = 0
    while offset < len(payload):
        size = min(C.PROBE_CHUNK_BYTES, len(payload) - offset)
        spans.append((offset, size))
        offset += size
    if len(spans) != C.PROBE_CHUNKS_PER_FRAME:
        raise SystemExit("registered probe chunk count drifted")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.send_buffer_bytes)
    sock.bind((args.bind_host, 0))
    sock.setblocking(False)

    rows: list[dict[str, Any]] = []
    try:
        for decision_index in range(args.frames):
            scheduled = args.start_monotonic_ns + decision_index * args.period_ns
            _wait_until(scheduled)
            decision_mono = time.monotonic_ns()
            decision_wall = time.time_ns()
            lag_ns = decision_mono - scheduled

            sent = dropped = byte_count = 0
            first_send: int | None = None
            last_send: int | None = None
            if lag_ns >= args.period_ns:
                terminal = "SCHEDULE_MISS_NO_BURST"
            else:
                for chunk_index, (offset, size) in enumerate(spans):
                    header = HEADER.pack(
                        MAGIC, decision_index, chunk_index, len(spans),
                        HEADER.size + size,
                    )
                    try:
                        sock.sendto(
                            header + payload[offset:offset + size],
                            (args.remote_host, args.remote_port),
                        )
                    except OSError as exc:
                        if exc.errno in (
                            errno.EWOULDBLOCK, errno.EAGAIN, errno.ENOBUFS
                        ):
                            dropped += 1
                            continue
                        raise
                    stamp = time.monotonic_ns()
                    if first_send is None:
                        first_send = stamp
                    last_send = stamp
                    sent += 1
                    byte_count += HEADER.size + size
                if dropped == 0:
                    terminal = "ALL_CHUNKS_HANDED_TO_SOCKET"
                elif sent == 0:
                    terminal = "SOCKET_BACKPRESSURE_ALL_CHUNKS_DROPPED"
                else:
                    terminal = "SOCKET_BACKPRESSURE_PARTIAL_FRAME"

            rows.append({
                "profile_id": args.profile_id,
                "trace_id": args.trace_id,
                "decision_index": decision_index,
                "partition": C.partition_for(decision_index),
                "action_id": C.PROBE_ACTION_ID,
                "scheduled_monotonic_ns": scheduled,
                "decision_monotonic_ns": decision_mono,
                "decision_wall_ns": decision_wall,
                "first_send_monotonic_ns": first_send,
                "last_send_monotonic_ns": last_send,
                "payload_bytes": len(payload),
                "chunks_per_frame": len(spans),
                "chunks_sent": sent,
                "chunks_dropped": dropped,
                "bytes_sent": byte_count,
                "send_span_ms": (
                    None if first_send is None or last_send is None
                    else (last_send - first_send) / 1e6
                ),
                "schedule_lag_ms": lag_ns / 1e6,
                "terminal_reason": terminal,
            })
    finally:
        sock.close()

    log_path = args.log_csv.resolve()
    with _open_create(log_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "scenesense.ue_dynamic_mcs_probe_sender.v1",
        "profile_id": args.profile_id,
        "trace_id": args.trace_id,
        "action_id": C.PROBE_ACTION_ID,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "frames": len(rows),
        "start_monotonic_ns": args.start_monotonic_ns,
        "period_ns": args.period_ns,
        "chunks_per_frame": len(spans),
        "chunks_sent": sum(int(row["chunks_sent"]) for row in rows),
        "chunks_dropped": sum(int(row["chunks_dropped"]) for row in rows),
        "schedule_misses": sum(
            row["terminal_reason"] == "SCHEDULE_MISS_NO_BURST" for row in rows
        ),
        "max_schedule_lag_ms": max(float(row["schedule_lag_ms"]) for row in rows),
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "bind_host": args.bind_host,
        "wire_contract": {
            "magic_ascii": MAGIC.decode("ascii"),
            "header_format": HEADER.format,
            "header_bytes": HEADER.size,
            "source": "rl_agent/ue_n3_structured_udp_receiver.py",
        },
    }
    with _open_create(args.summary_json.resolve()) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--start-monotonic-ns", type=int, required=True)
    parser.add_argument("--period-ns", type=int, default=C.PERIOD_NS)
    parser.add_argument("--frames", type=int, default=C.FRAMES_PER_PROFILE)
    parser.add_argument("--payload-bytes", type=int, default=C.PROBE_PAYLOAD_BYTES)
    parser.add_argument("--payload-seed", type=int, default=2026092402)
    parser.add_argument("--send-buffer-bytes", type=int, default=8_388_608)
    parser.add_argument("--log-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

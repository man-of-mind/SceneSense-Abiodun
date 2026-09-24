#!/usr/bin/env python3
"""Tagged UDP sender for the MCS/backlog calibration (amended, block-structured).

Emits deterministic synthetic payload bytes through the **frozen production
SSBURST chunking contract**, imported from
``rl_agent/ue_n3_structured_udp_receiver.py`` rather than restated, so the wire
format provably cannot drift from the production receiver that reassembles it.

No CARLA, no CUDA, no perception model: payloads are seeded pseudo-random bytes
sized to pinned catalogue actions.

**Load varies inside a single run.** One continuous 10 Hz decision timeline
walks a preregistered sequence of constant-load blocks. At a block boundary the
payload size and destination port change on the very next frame, with no
process restart and no gap, so the transition is sharp at a decision boundary
and the queue is never given a pause to drain that the design did not ask for.
Each block has its own receiver, because the production receiver is
instantiated with one ``expected_chunks_per_frame``; that is the only reason
blocks use different ports.

``decision_monotonic_ns`` is read immediately **before** the frame's first
datagram reaches the socket, so every state value joined to it (backlog,
previous MCS) is by construction an observation that already existed.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_n3_structured_udp_receiver import HEADER, MAGIC  # noqa: E402

FRAME_ROW_FIELDS = (
    "cell_id", "decision_index", "block_index", "tier", "action_id",
    "frame_index_in_block", "is_first_frame_of_block",
    "decisions_since_transition", "previous_tier",
    "scheduled_monotonic_ns", "decision_monotonic_ns", "decision_wall_ns",
    "first_send_monotonic_ns", "last_send_monotonic_ns",
    "payload_bytes", "chunks_per_frame", "chunks_sent", "chunks_dropped",
    "bytes_sent", "send_span_ms", "schedule_lag_ms", "terminal_reason",
)


def build_payload(payload_bytes: int, seed: int) -> bytes:
    """Deterministic payload body for one tier.

    One buffer per tier, hashed into the summary. Frames are distinguished on
    the wire by the SSBURST header's frame index, so the body need not change
    per frame; holding it fixed also keeps the offered byte count exactly equal
    across every frame of a block, which is what makes the tier a clean level.
    """
    return random.Random(seed).randbytes(payload_bytes)


def chunk_spans(payload_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    spans, offset = [], 0
    while offset < payload_bytes:
        size = min(chunk_bytes, payload_bytes - offset)
        spans.append((offset, size))
        offset += size
    return spans


def run(args: argparse.Namespace) -> int:
    blocks = json.loads(Path(args.block_plan).read_text())
    if not blocks:
        raise SystemExit("block plan is empty")

    payloads: dict[str, bytes] = {}
    spans: dict[str, list[tuple[int, int]]] = {}
    for block in blocks:
        tier = block["tier"]
        if tier in payloads:
            continue
        # Seed is per (campaign, tier): identical bytes for the same tier in
        # every cell, so the payload is never a hidden variable across cells.
        seed = args.payload_seed + block["action_id"]
        payloads[tier] = build_payload(int(block["payload_bytes"]), seed)
        spans[tier] = chunk_spans(int(block["payload_bytes"]), args.chunk_bytes)
        expected = int(block["chunks_per_frame"])
        if len(spans[tier]) != expected:
            raise SystemExit(
                f"tier {tier}: plan says {expected} chunks/frame, chunking gives "
                f"{len(spans[tier])}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.socket_sendbuf)
    if args.bind_host:
        sock.bind((args.bind_host, 0))
    # Non-blocking: a saturated uplink must appear as explicitly accounted
    # socket backpressure, never as a silently stretched decision cadence. The
    # 10 Hz schedule is the experiment's independent variable, and a block
    # transition must land exactly on its scheduled decision.
    sock.setblocking(False)

    rows: list[dict[str, Any]] = []
    period_ns = int(1e9 / args.fps)
    start_mono = time.monotonic_ns()
    start_wall = time.time_ns()
    decision_index = 0
    previous_tier: str | None = None

    for block in blocks:
        tier = block["tier"]
        port = int(block["port"])
        remote = (args.remote_host, port)
        payload = payloads[tier]
        block_spans = spans[tier]
        chunks_per_frame = len(block_spans)

        for frame_index in range(int(block["frames"])):
            scheduled = start_mono + decision_index * period_ns
            now = time.monotonic_ns()
            if now < scheduled:
                time.sleep((scheduled - now) / 1e9)

            decision_mono = time.monotonic_ns()
            decision_wall = time.time_ns()

            sent = dropped = bytes_sent = 0
            first_send = last_send = None
            for chunk_index, (offset, size) in enumerate(block_spans):
                header = HEADER.pack(MAGIC, frame_index, chunk_index,
                                     chunks_per_frame, HEADER.size + size)
                datagram = header + payload[offset:offset + size]
                try:
                    sock.sendto(datagram, remote)
                except OSError as exc:
                    if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.ENOBUFS):
                        dropped += 1
                        continue
                    raise
                stamp = time.monotonic_ns()
                first_send = first_send if first_send is not None else stamp
                last_send = stamp
                sent += 1
                bytes_sent += len(datagram)

            if dropped == 0:
                terminal = "ALL_CHUNKS_HANDED_TO_SOCKET"
            elif sent == 0:
                terminal = "SOCKET_BACKPRESSURE_ALL_CHUNKS_DROPPED"
            else:
                terminal = "SOCKET_BACKPRESSURE_PARTIAL_FRAME"

            rows.append({
                "cell_id": args.cell_id,
                "decision_index": decision_index,
                "block_index": int(block["block_index"]),
                "tier": tier,
                "action_id": int(block["action_id"]),
                "frame_index_in_block": frame_index,
                "is_first_frame_of_block": frame_index == 0,
                "decisions_since_transition": frame_index,
                "previous_tier": previous_tier if frame_index == 0 else tier,
                "scheduled_monotonic_ns": scheduled,
                "decision_monotonic_ns": decision_mono,
                "decision_wall_ns": decision_wall,
                "first_send_monotonic_ns": first_send,
                "last_send_monotonic_ns": last_send,
                "payload_bytes": int(block["payload_bytes"]),
                "chunks_per_frame": chunks_per_frame,
                "chunks_sent": sent,
                "chunks_dropped": dropped,
                "bytes_sent": bytes_sent,
                "send_span_ms": (
                    (last_send - first_send) / 1e6 if first_send is not None else None),
                "schedule_lag_ms": (decision_mono - scheduled) / 1e6,
                "terminal_reason": terminal,
            })
            decision_index += 1
        previous_tier = tier

    sock.close()
    end_mono = time.monotonic_ns()

    out = Path(args.log_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cell_id": args.cell_id,
        "decisions": len(rows),
        "fps": args.fps,
        "blocks": [
            {"block_index": b["block_index"], "tier": b["tier"],
             "action_id": b["action_id"], "payload_bytes": b["payload_bytes"],
             "frames": b["frames"], "port": b["port"],
             "payload_sha256": hashlib.sha256(payloads[b["tier"]]).hexdigest()}
            for b in blocks
        ],
        "sequence": [b["tier"] for b in blocks],
        "chunk_bytes": args.chunk_bytes,
        "payload_seed_base": args.payload_seed,
        "wire_contract": {
            "magic": MAGIC.decode("ascii"), "header_struct": HEADER.format,
            "header_bytes": HEADER.size,
            "imported_from": "rl_agent/ue_n3_structured_udp_receiver.py",
        },
        "remote_host": args.remote_host,
        "bind_host": args.bind_host,
        "start_monotonic_ns": start_mono, "start_wall_ns": start_wall,
        "end_monotonic_ns": end_mono,
        "per_tier": {
            tier: {
                "decisions": sum(1 for r in rows if r["tier"] == tier),
                "datagrams_handed_to_socket": sum(
                    r["chunks_sent"] for r in rows if r["tier"] == tier),
                "datagrams_dropped_at_socket": sum(
                    r["chunks_dropped"] for r in rows if r["tier"] == tier),
            }
            for tier in payloads
        },
        "datagrams_handed_to_socket": sum(r["chunks_sent"] for r in rows),
        "datagrams_dropped_at_socket": sum(r["chunks_dropped"] for r in rows),
        "bytes_handed_to_socket": sum(r["bytes_sent"] for r in rows),
        "max_schedule_lag_ms": max((r["schedule_lag_ms"] for r in rows), default=None),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "cell_id", "decisions", "sequence", "datagrams_handed_to_socket",
        "datagrams_dropped_at_socket", "max_schedule_lag_ms")}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--bind-host", default="10.0.0.2")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--block-plan", required=True,
                        help="JSON list of blocks: tier, payload_bytes, frames, port")
    parser.add_argument("--payload-seed", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=60_000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--socket-sendbuf", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

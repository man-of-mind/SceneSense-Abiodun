#!/usr/bin/env python3
"""Replay a packet-length/timestamp CSV trace as UDP traffic.

This is intentionally simple: it does not spoof source IPs or protocols. It
preserves the offered packet timing and payload size pattern toward a UDP sink,
which is what we need to stress OAI scheduling/MCS/RLC behavior.
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path


def guess_ue_ip(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        for col in ("Source", "Destination"):
            ip = row.get(col, "")
            counts[ip] = counts.get(ip, 0) + 1
    ending_250 = [ip for ip in counts if ip.endswith(".250")]
    if ending_250:
        return max(ending_250, key=lambda ip: counts[ip])
    return max(counts, key=counts.get)


def packet_chunks(length: int, max_payload: int, mode: str) -> list[int]:
    length = max(1, length)
    if mode == "cap":
        return [min(length, max_payload)]
    chunks: list[int] = []
    remaining = length
    while remaining > 0:
        n = min(remaining, max_payload)
        chunks.append(n)
        remaining -= n
    return chunks


def load_events(
    path: Path,
    direction: str,
    ue_ip: str | None,
    length_scale: float,
    max_payload: int,
    large_packet_mode: str,
):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ue = ue_ip or guess_ue_ip(rows)
    events = []
    for row in rows:
        try:
            t = float(row["Time"])
            length = int(float(row["Length"]) * length_scale)
        except (KeyError, ValueError):
            continue
        src = row.get("Source", "")
        dst = row.get("Destination", "")
        is_ul = src == ue
        is_dl = dst == ue
        if direction == "uplink" and not is_ul:
            continue
        if direction == "downlink" and not is_dl:
            continue
        if direction == "both" and not (is_ul or is_dl):
            continue
        for chunk_len in packet_chunks(length, max_payload, large_packet_mode):
            events.append((t, chunk_len))
    events.sort()
    return ue, events


def make_payload(seq: int, trace_time: float, size: int) -> bytes:
    payload = bytearray(size)
    if size >= 20:
        struct.pack_into("!QdI", payload, 0, seq, trace_time, size)
    return bytes(payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_csv")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--port", type=int, default=55000)
    ap.add_argument("--bind-ip", default=None)
    ap.add_argument("--direction", choices=["uplink", "downlink", "both"], default="uplink")
    ap.add_argument("--ue-ip", default=None)
    ap.add_argument("--time-scale", type=float, default=1.0, help="<1 speeds up, >1 slows down")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--max-payload", type=int, default=1400)
    ap.add_argument("--large-packet-mode", choices=["split", "cap"], default="split",
                    help="split preserves raw byte volume using multiple UDP datagrams; cap preserves raw row count")
    ap.add_argument("--start-offset-s", type=float, default=0.0,
                    help="Skip events before this offset from the first replayable event")
    ap.add_argument("--max-duration-s", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ue, events = load_events(
        Path(args.trace_csv),
        args.direction,
        args.ue_ip,
        args.length_scale,
        args.max_payload,
        args.large_packet_mode,
    )
    if not events:
        raise SystemExit("No replayable events after filtering")
    original_first_t = events[0][0]
    if args.start_offset_s:
        start_t = original_first_t + args.start_offset_s
        events = [(t, n) for t, n in events if t >= start_t]
        if not events:
            raise SystemExit("No replayable events after start-offset filtering")
    first_t = events[0][0]
    if args.max_duration_s is not None:
        events = [(t, n) for t, n in events if t - first_t <= args.max_duration_s]
        if not events:
            raise SystemExit("No replayable events after max-duration filtering")
    total_bytes = sum(n for _, n in events)
    duration = max(events[-1][0] - events[0][0], 1e-9) * args.time_scale
    print(
        f"trace={args.trace_csv} ue_ip={ue} direction={args.direction} "
        f"start_offset_s={args.start_offset_s:.3f} packet_mode={args.large_packet_mode} "
        f"udp_datagrams={len(events)} bytes={total_bytes} duration_s={duration:.3f} "
        f"mean_mbps={total_bytes * 8 / duration / 1e6:.3f}"
    )
    if args.dry_run:
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.bind_ip:
        sock.bind((args.bind_ip, 0))
    target = (args.dst, args.port)
    wall_start = time.monotonic()
    sent_bytes = 0
    for seq, (trace_t, size) in enumerate(events):
        send_at = wall_start + (trace_t - first_t) * args.time_scale
        delay = send_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        sock.sendto(make_payload(seq, trace_t, size), target)
        sent_bytes += size
    elapsed = max(time.monotonic() - wall_start, 1e-9)
    print(f"sent_packets={len(events)} sent_bytes={sent_bytes} elapsed_s={elapsed:.3f} offered_mbps={sent_bytes * 8 / elapsed / 1e6:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

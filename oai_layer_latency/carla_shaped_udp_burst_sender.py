#!/usr/bin/env python3
"""Send CARLA-shaped UDP uplink bursts over the OAI UE tunnel.

The goal is to mimic only the traffic shape of the split-inference frontend:
roughly one dense split-feature frame every 100 ms, fragmented into large UDP
application chunks. This isolates "burst/backlog shape" from CARLA rendering,
model execution, and back-half inference.
"""

from __future__ import annotations

import argparse
import csv
import math
import socket
import struct
import time
from pathlib import Path


FIELDS = (
    "wall_time_s",
    "elapsed_s",
    "frame_index",
    "chunk_index",
    "chunk_bytes",
    "frame_bytes",
    "period_s",
    "scheduled_frame_time_s",
    "send_lag_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="10.0.0.2")
    parser.add_argument("--remote-host", default="192.168.70.135")
    parser.add_argument("--remote-port", type=int, default=5001)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=1300)
    parser.add_argument(
        "--frame-bytes",
        type=int,
        default=1_079_400,
        help="Bytes per synthetic split-feature frame; default ~=1054.1 KiB.",
    )
    parser.add_argument("--chunk-bytes", type=int, default=60_000)
    parser.add_argument(
        "--inter-chunk-gap-us",
        type=float,
        default=0.0,
        help="Optional gap between chunks inside one frame burst.",
    )
    parser.add_argument("--idle-before-s", type=float, default=5.0)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--socket-sendbuf", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--log-csv", required=True)
    return parser.parse_args()


def make_payload(size: int, frame_index: int, chunk_index: int, chunks_per_frame: int) -> bytes:
    # A tiny recognizable header is useful if we ever tcpdump/pcap this stream.
    # The remainder is deterministic non-zero data to avoid any suspicious all-zero
    # fast paths in lower layers/tools.
    header = struct.pack(
        "!8sIIII",
        b"SSBURST",
        frame_index,
        chunk_index,
        chunks_per_frame,
        size,
    )
    if len(header) >= size:
        return header[:size]
    fill = bytes(((frame_index + chunk_index + i) & 0xFF) for i in range(256))
    repeats = (size - len(header) + len(fill) - 1) // len(fill)
    return header + (fill * repeats)[: size - len(header)]


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.chunk_bytes <= 0 or args.chunk_bytes > 65_507:
        raise SystemExit("--chunk-bytes must be in 1..65507 for UDP")
    if args.frame_bytes <= 0:
        raise SystemExit("--frame-bytes must be positive")

    log_path = Path(args.log_csv).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    period_s = 1.0 / args.fps
    chunks_per_frame = math.ceil(args.frame_bytes / args.chunk_bytes)
    remote = (args.remote_host, args.remote_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(args.socket_sendbuf))
    sock.bind((args.bind_host, 0))

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        time.sleep(max(0.0, args.idle_before_s))
        start = time.perf_counter()
        next_frame_t = start
        sent_bytes = 0
        sent_chunks = 0

        for frame_index in range(args.frames):
            now = time.perf_counter()
            sleep_s = next_frame_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            frame_send_t = time.perf_counter()
            send_lag_ms = (frame_send_t - next_frame_t) * 1000.0

            remaining = args.frame_bytes
            for chunk_index in range(chunks_per_frame):
                chunk_size = min(args.chunk_bytes, remaining)
                payload = make_payload(chunk_size, frame_index, chunk_index, chunks_per_frame)
                sock.sendto(payload, remote)
                sent_bytes += chunk_size
                sent_chunks += 1
                t = time.perf_counter()
                writer.writerow(
                    {
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{t - start:.6f}",
                        "frame_index": frame_index,
                        "chunk_index": chunk_index,
                        "chunk_bytes": chunk_size,
                        "frame_bytes": args.frame_bytes,
                        "period_s": f"{period_s:.6f}",
                        "scheduled_frame_time_s": f"{next_frame_t - start:.6f}",
                        "send_lag_ms": f"{send_lag_ms:.3f}",
                    }
                )
                remaining -= chunk_size
                if args.inter_chunk_gap_us > 0 and chunk_index + 1 < chunks_per_frame:
                    time.sleep(args.inter_chunk_gap_us / 1_000_000.0)

            next_frame_t += period_s

        f.flush()
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)

    duration = max(time.perf_counter() - start, 1e-9)
    offered_mbps = sent_bytes * 8.0 / duration / 1e6
    print(
        f"sent frames={args.frames} chunks={sent_chunks} bytes={sent_bytes} "
        f"duration_s={duration:.3f} offered_mbps={offered_mbps:.3f} "
        f"chunks_per_frame={chunks_per_frame} log={log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

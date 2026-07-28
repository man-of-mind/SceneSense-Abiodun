#!/usr/bin/env python3
"""UDP sink for replay experiments.

Receives UDP packets, prints one-second receive-rate summaries, and optionally
writes a CSV log for post-analysis.
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=55000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout-s", type=float, default=None)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    if args.timeout_s:
        sock.settimeout(args.timeout_s)

    out_f = None
    writer = None
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_f = path.open("w", newline="")
        writer = csv.DictWriter(out_f, fieldnames=["wall_s", "src", "size", "seq", "trace_time", "declared_size"])
        writer.writeheader()

    start = time.monotonic()
    last = start
    packets = 0
    bytes_recv = 0
    interval_packets = 0
    interval_bytes = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            now = time.monotonic()
            packets += 1
            bytes_recv += len(data)
            interval_packets += 1
            interval_bytes += len(data)
            seq = trace_time = declared_size = None
            if len(data) >= 20:
                try:
                    seq, trace_time, declared_size = struct.unpack_from("!QdI", data, 0)
                except struct.error:
                    pass
            if writer:
                writer.writerow(
                    {
                        "wall_s": now - start,
                        "src": f"{addr[0]}:{addr[1]}",
                        "size": len(data),
                        "seq": seq,
                        "trace_time": trace_time,
                        "declared_size": declared_size,
                    }
                )
            if now - last >= 1.0:
                dt = now - last
                print(
                    f"t={now - start:.1f}s rate={interval_bytes * 8 / dt / 1e6:.3f} Mbps "
                    f"pps={interval_packets / dt:.1f} total_mb={bytes_recv / 1e6:.3f}",
                    flush=True,
                )
                last = now
                interval_packets = 0
                interval_bytes = 0
    finally:
        elapsed = max(time.monotonic() - start, 1e-9)
        print(f"done packets={packets} bytes={bytes_recv} elapsed_s={elapsed:.3f} avg_mbps={bytes_recv * 8 / elapsed / 1e6:.3f}")
        if out_f:
            out_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

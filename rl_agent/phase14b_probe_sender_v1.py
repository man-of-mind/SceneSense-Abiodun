#!/usr/bin/env python3
"""Fixed-rate lightweight UDP PUSCH-observation probe sender.

This is measurement instrumentation, not SplitFusion application traffic: it
exists only to distribute independent uplink transmission opportunities across
each 100-ms interval so PUSCH SNR can be observed. Rate, datagram size, and
destination are fixed and identical across every profile and the probe
qualification; they are never tuned in response to a measurement outcome.

Each datagram's first 20 bytes match the wire format already parsed by
`bursty_traffic/udp_sink.py` (`struct.unpack_from("!QdI", data, 0)`): an
unsigned 64-bit sequence number, a double wall-clock send timestamp, and an
unsigned 32-bit declared datagram size. The remainder is deterministic
non-payload filler.
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path

HEADER = struct.Struct("!QdI")
FIELDS = ("seq", "scheduled_send_s", "send_monotonic_s", "send_wall_s", "send_lag_ms", "bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--datagram-bytes", type=int, default=1200)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--log-csv", required=True)
    return parser.parse_args()


def make_payload(seq: int, datagram_bytes: int) -> bytes:
    header = HEADER.pack(seq, time.time(), datagram_bytes)
    if len(header) >= datagram_bytes:
        return header[:datagram_bytes]
    fill = bytes(((seq + i) & 0xFF) for i in range(256))
    repeats = (datagram_bytes - len(header) + len(fill) - 1) // len(fill)
    return header + (fill * repeats)[: datagram_bytes - len(header)]


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be positive")
    if args.datagram_bytes < HEADER.size or args.datagram_bytes > 65_507:
        raise SystemExit(f"--datagram-bytes must be in {HEADER.size}..65507")

    log_path = Path(args.log_csv).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.rate_hz
    remote = (args.remote_host, args.remote_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, 0))

    sent = 0
    sent_bytes = 0
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        start = time.perf_counter()
        next_tick = start
        seq = 0
        try:
            while time.perf_counter() - start < args.duration_s:
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                send_t = time.perf_counter()
                send_lag_ms = (send_t - next_tick) * 1000.0
                payload = make_payload(seq, args.datagram_bytes)
                sock.sendto(payload, remote)
                sent += 1
                sent_bytes += len(payload)
                writer.writerow(
                    {
                        "seq": seq,
                        "scheduled_send_s": f"{next_tick - start:.6f}",
                        "send_monotonic_s": f"{send_t - start:.6f}",
                        "send_wall_s": f"{time.time():.6f}",
                        "send_lag_ms": f"{send_lag_ms:.3f}",
                        "bytes": len(payload),
                    }
                )
                handle.flush()
                seq += 1
                next_tick += period_s
        finally:
            handle.flush()

    duration = max(time.perf_counter() - start, 1e-9)
    print(
        f"sent packets={sent} bytes={sent_bytes} duration_s={duration:.3f} "
        f"rate_hz={args.rate_hz:.1f} log={log_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

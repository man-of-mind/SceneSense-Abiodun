#!/usr/bin/env python3
"""MTU-safe, future-epoch tagged sender for Run-4 queue calibration.

The output schema is explicit and is what ``runner.audit_traffic`` consumes;
the audit never probes guessed aliases such as ``frames_sent`` or
``chunks_sent``.  One monotonic epoch is supplied by the parent runner and is
also used by the channel replay.
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

from rl_agent.ue_n3_structured_udp_receiver import HEADER, MAGIC
from . import contract as C


SUMMARY_SCHEMA = "scenesense.run4_calibration_tagged_sender_summary.v1"
FRAME_SCHEMA = "scenesense.run4_calibration_tagged_sender_frame.v1"
READY_SCHEMA = "scenesense.run4_calibration_tagged_sender_ready.v1"
EPOCH_SCHEMA = "scenesense.run4_calibration_shared_epoch.v1"
FRAME_FIELDS = (
    "schema", "cell_id", "decision_index", "block_index", "tier",
    "action_id", "frame_index_in_block", "is_first_frame_of_block",
    "decisions_since_transition", "previous_tier", "epoch_monotonic_ns",
    "scheduled_monotonic_ns", "decision_monotonic_ns", "decision_wall_ns",
    "first_send_monotonic_ns", "last_send_monotonic_ns", "payload_bytes",
    "chunks_per_frame", "datagrams_handed_to_socket",
    "datagrams_dropped_at_socket", "application_payload_bytes_handed_to_socket",
    "bytes_handed_to_socket", "send_span_ms", "schedule_lag_ms",
    "terminal_reason",
)


class SenderError(RuntimeError):
    """The registered epoch, plan, packetization or create-only gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SenderError(message)


def build_payload(size: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(size)


def chunk_spans(payload_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    while offset < payload_bytes:
        size = min(chunk_bytes, payload_bytes - offset)
        spans.append((offset, size))
        offset += size
    return spans


def _wait_until(epoch_ns: int) -> None:
    while True:
        remaining = epoch_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.01))


def _write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _wait_for_epoch_contract(
    path: Path, *, cell_id: str, period_ns: int,
    sender_ready_sha256: str, timeout_s: float,
) -> dict[str, Any]:
    """Wait for the runner's one-cell epoch contract after reporting READY."""
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
    expected_fields = {
        "schema", "cell_id", "epoch_monotonic_ns", "period_ns",
        "created_monotonic_ns", "sender_ready_sha256",
    }
    require(set(value) == expected_fields, "shared epoch contract fields drifted")
    require(value["schema"] == EPOCH_SCHEMA, "shared epoch contract schema drifted")
    require(value["cell_id"] == cell_id, "shared epoch contract cell drifted")
    require(type(value["epoch_monotonic_ns"]) is int
            and type(value["created_monotonic_ns"]) is int,
            "shared epoch stamps must be integer monotonic nanoseconds")
    require(value["period_ns"] == period_ns, "shared epoch period drifted")
    require(value["sender_ready_sha256"] == sender_ready_sha256,
            "shared epoch was not bound to this sender READY record")
    require(value["created_monotonic_ns"] < value["epoch_monotonic_ns"],
            "shared epoch was not future at publication")
    require(time.monotonic_ns() < value["epoch_monotonic_ns"],
            "sender received the shared epoch after its release time")
    return value


def run(args: argparse.Namespace) -> int:
    require(args.chunk_bytes == C.CHUNK_BYTES,
            f"chunk bytes {args.chunk_bytes} != registered {C.CHUNK_BYTES}")
    require(HEADER.size == C.SSBURST_HEADER_BYTES,
            "SSBURST header size drifted")
    require(args.chunk_bytes + HEADER.size + C.UDP_HEADER_BYTES + C.IPV4_HEADER_BYTES <= C.PATH_MTU_BYTES,
            "full IPv4 packet exceeds the registered path MTU")
    require(args.fps == C.FPS, f"fps {args.fps} != registered {C.FPS}")
    plan_path = Path(args.block_plan)
    blocks = json.loads(plan_path.read_text(encoding="utf-8"))
    require(type(blocks) is list and len(blocks) == C.BLOCKS_PER_CELL,
            "block plan must contain exactly three blocks")
    require([row.get("tier") for row in blocks]
            in [list(value) for value in C.PERMUTATIONS],
            "block sequence is not a registered FIT/VALIDATION permutation")

    payloads: dict[str, bytes] = {}
    spans: dict[str, list[tuple[int, int]]] = {}
    for block in blocks:
        tier = str(block["tier"])
        expected = next(row for row in C.registered_tiers() if row.tier == tier)
        require(int(block["action_id"]) == expected.action_id
                and int(block["payload_bytes"]) == expected.payload_bytes
                and int(block["chunks_per_frame"]) == expected.chunks_per_frame
                and int(block["frames"]) == C.FRAMES_PER_BLOCK,
                f"{tier} block identity drifted")
        payloads[tier] = build_payload(
            expected.payload_bytes, args.payload_seed + expected.action_id)
        spans[tier] = chunk_spans(expected.payload_bytes, args.chunk_bytes)
        require(len(spans[tier]) == expected.chunks_per_frame,
                f"{tier} chunk count drifted")

    log_path = Path(args.log_csv)
    summary_path = Path(args.summary_json)
    ready_path = Path(args.ready_json)
    epoch_contract_path = Path(args.epoch_contract)
    require(not log_path.exists() and not summary_path.exists()
            and not ready_path.exists() and not epoch_contract_path.exists(),
            "sender outputs are create-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.socket_sendbuf)
    if args.bind_host:
        sock.bind((args.bind_host, 0))
    sock.setblocking(False)

    rows: list[dict[str, Any]] = []
    period_ns = int(1e9 / args.fps)
    ready = {
        "schema": READY_SCHEMA, "cell_id": args.cell_id,
        "ready_monotonic_ns": time.monotonic_ns(),
        "bind_host": args.bind_host, "remote_host": args.remote_host,
        "socket_sendbuf": args.socket_sendbuf, "period_ns": period_ns,
        "block_plan_sha256": C.sha256_file(plan_path),
    }
    _write_json_create(ready_path, ready)
    ready_sha256 = C.sha256_file(ready_path)
    epoch_contract = _wait_for_epoch_contract(
        epoch_contract_path, cell_id=args.cell_id, period_ns=period_ns,
        sender_ready_sha256=ready_sha256,
        timeout_s=args.epoch_contract_timeout_s,
    )
    epoch_monotonic_ns = int(epoch_contract["epoch_monotonic_ns"])
    epoch_contract_sha256 = C.sha256_file(epoch_contract_path)
    decision_index = 0
    previous_tier: str | None = None
    _wait_until(epoch_monotonic_ns)
    actual_epoch_release_ns = time.monotonic_ns()
    start_wall_ns = time.time_ns()

    try:
        for block in blocks:
            tier = str(block["tier"])
            payload = payloads[tier]
            block_spans = spans[tier]
            remote = (args.remote_host, int(block["port"]))
            for frame_index in range(C.FRAMES_PER_BLOCK):
                scheduled = epoch_monotonic_ns + decision_index * period_ns
                _wait_until(scheduled)
                decision_mono = time.monotonic_ns()
                decision_wall = time.time_ns()
                handed = dropped = payload_handed = wire_handed = 0
                first_send: int | None = None
                last_send: int | None = None
                for chunk_index, (offset, size) in enumerate(block_spans):
                    datagram = HEADER.pack(
                        MAGIC, frame_index, chunk_index, len(block_spans),
                        HEADER.size + size,
                    ) + payload[offset:offset + size]
                    try:
                        sock.sendto(datagram, remote)
                    except OSError as exc:
                        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN,
                                         errno.ENOBUFS):
                            dropped += 1
                            continue
                        raise
                    stamp = time.monotonic_ns()
                    first_send = stamp if first_send is None else first_send
                    last_send = stamp
                    handed += 1
                    payload_handed += size
                    wire_handed += len(datagram)
                terminal = (
                    "ALL_DATAGRAMS_HANDED_TO_SOCKET" if dropped == 0
                    else "SOCKET_BACKPRESSURE_ALL_DATAGRAMS_DROPPED"
                    if handed == 0 else "SOCKET_BACKPRESSURE_PARTIAL_FRAME"
                )
                rows.append({
                    "schema": FRAME_SCHEMA, "cell_id": args.cell_id,
                    "decision_index": decision_index,
                    "block_index": int(block["block_index"]), "tier": tier,
                    "action_id": int(block["action_id"]),
                    "frame_index_in_block": frame_index,
                    "is_first_frame_of_block": frame_index == 0,
                    "decisions_since_transition": frame_index,
                    "previous_tier": previous_tier if frame_index == 0 else tier,
                    "epoch_monotonic_ns": epoch_monotonic_ns,
                    "scheduled_monotonic_ns": scheduled,
                    "decision_monotonic_ns": decision_mono,
                    "decision_wall_ns": decision_wall,
                    "first_send_monotonic_ns": first_send,
                    "last_send_monotonic_ns": last_send,
                    "payload_bytes": int(block["payload_bytes"]),
                    "chunks_per_frame": len(block_spans),
                    "datagrams_handed_to_socket": handed,
                    "datagrams_dropped_at_socket": dropped,
                    "application_payload_bytes_handed_to_socket": payload_handed,
                    "bytes_handed_to_socket": wire_handed,
                    "send_span_ms": ((last_send - first_send) / 1e6
                                     if first_send is not None else None),
                    "schedule_lag_ms": (decision_mono - scheduled) / 1e6,
                    "terminal_reason": terminal,
                })
                decision_index += 1
            previous_tier = tier
    finally:
        sock.close()
    end_ns = time.monotonic_ns()

    with log_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": SUMMARY_SCHEMA, "cell_id": args.cell_id,
        "decisions": len(rows), "fps": args.fps,
        "epoch_monotonic_ns": epoch_monotonic_ns,
        "sender_ready_sha256": ready_sha256,
        "epoch_contract_sha256": epoch_contract_sha256,
        "actual_epoch_release_monotonic_ns": actual_epoch_release_ns,
        "start_wall_ns": start_wall_ns, "end_monotonic_ns": end_ns,
        "sequence": [row["tier"] for row in blocks],
        "chunk_bytes": args.chunk_bytes, "payload_seed_base": args.payload_seed,
        "wire_contract": {"magic": MAGIC.decode("ascii"),
                          "header_struct": HEADER.format,
                          "header_bytes": HEADER.size},
        "remote_host": args.remote_host, "bind_host": args.bind_host,
        "blocks": [{
            "block_index": int(row["block_index"]), "tier": row["tier"],
            "action_id": int(row["action_id"]),
            "payload_bytes": int(row["payload_bytes"]),
            "chunks_per_frame": int(row["chunks_per_frame"]),
            "frames": int(row["frames"]), "port": int(row["port"]),
            "payload_sha256": hashlib.sha256(payloads[row["tier"]]).hexdigest(),
        } for row in blocks],
        "datagrams_handed_to_socket": sum(
            row["datagrams_handed_to_socket"] for row in rows),
        "datagrams_dropped_at_socket": sum(
            row["datagrams_dropped_at_socket"] for row in rows),
        "application_payload_bytes_handed_to_socket": sum(
            row["application_payload_bytes_handed_to_socket"] for row in rows),
        "bytes_handed_to_socket": sum(
            row["bytes_handed_to_socket"] for row in rows),
        "max_schedule_lag_ms": max(
            (row["schedule_lag_ms"] for row in rows), default=None),
        "per_tier": {tier: {
            "decisions": sum(row["tier"] == tier for row in rows),
            "datagrams_handed_to_socket": sum(
                row["datagrams_handed_to_socket"] for row in rows
                if row["tier"] == tier),
            "datagrams_dropped_at_socket": sum(
                row["datagrams_dropped_at_socket"] for row in rows
                if row["tier"] == tier),
            "application_payload_bytes_handed_to_socket": sum(
                row["application_payload_bytes_handed_to_socket"] for row in rows
                if row["tier"] == tier),
        } for tier in C.TIER_ORDER},
    }
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--block-plan", required=True)
    parser.add_argument("--payload-seed", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--ready-json", required=True)
    parser.add_argument("--epoch-contract", required=True)
    parser.add_argument("--epoch-contract-timeout-s", type=float, required=True)
    parser.add_argument("--socket-sendbuf", type=int, required=True)
    parser.add_argument("--log-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

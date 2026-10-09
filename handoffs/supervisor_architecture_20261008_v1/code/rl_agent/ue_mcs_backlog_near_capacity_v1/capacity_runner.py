#!/usr/bin/env python3
"""Bounded capacity qualification for the exact 273-PRB/4D5U radio.

The module is import-pure.  ``run`` is the only live entry point; ``sink`` and
``sender`` are private subprocess roles launched by that entry point.  The
primary service signal is unique SSBURST application payload delivered at the
ext-DN in fixed 100-ms monotonic windows.  Transport-block sizes are retained
only by the raw trace and never enter the capacity estimate.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import ue_n3_structured_udp_receiver as U3  # noqa: E402
from rl_agent.ue_mcs_backlog_calibration_v1 import runner as V3R  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import authorization as AUTH  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_qualification as CQ  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB  # noqa: E402
from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as NR  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_v1.json"
RESULT_FILENAME = "CAPACITY_QUALIFICATION_RESULT.json"
MANIFEST_FILENAME = "manifest.json"
TERMINAL_FILENAME = "CAPACITY_QUALIFICATION_CAPTURED.json"
STATUS_CAPTURED = "CAPACITY_QUALIFICATION_CAPTURED"
RESULT_SCHEMA = "scenesense.capacity_qualification_result.v1"
MANIFEST_SCHEMA = "scenesense.capacity_qualification_manifest.v1"
TERMINAL_SCHEMA = "scenesense.capacity_qualification_terminal.v1"
STATUS_REFUSED = "CAPACITY_QUALIFICATION_REFUSED"
STATUS_FAILED = "CAPACITY_QUALIFICATION_FAILED"

PERIOD_NS = int(CQ.SAMPLE_PERIOD_S * 1e9)
SETTLE_BINS = int(round(CQ.SETTLE_S / CQ.SAMPLE_PERIOD_S))
MEASURE_BINS = int(round(CQ.MEASURE_S / CQ.SAMPLE_PERIOD_S))
POINT_BINS = SETTLE_BINS + MEASURE_BINS
POINT_FRAMES = POINT_BINS
PROBE_CHUNKS = CQ.PROBE_CHUNKS_PER_FRAME

RLC_BUFFER_FIELDS = (
    "time", "rnti", "ue_id", "frame", "slot", "lcid", "lcgid",
    "bytes_in_buffer", "bj", "pbr", "priority",
)
MONO_SDU_FIELDS = ("time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes")
DEQUEUE_FIELDS = ("time", "mono_sec", "mono_nsec", "ue_id", "lcid", "pdu_bytes")


class CapacityRunError(RuntimeError):
    """A structural or scientific qualification gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityRunError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


TRANSITIVE_REPO_SOURCES: tuple[str, ...] = (
    "rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_runner.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_qualification.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/"
    "CAPACITY_RETRY_AMENDMENT_MTU_SAFE_V1.md",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/runner.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/contract.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/radio_binding.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/authorization.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/protected_evidence.py",
    "rl_agent/ue_mcs_backlog_near_capacity_v1/analysis_spec.py",
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.csv",
    "rl_agent/ue_mcs_backlog_calibration_v1/runner.py",
    "rl_agent/ue_mcs_backlog_calibration_v1/contract.py",
    "rl_agent/ue_n2_oai_ul_calibration_smoke.py",
    "rl_agent/ue_n3_structured_udp_receiver.py",
    "uplink_only_spatial_map_pipeline/run_splitfusion_oai_100mhz_4d5u_v1.sh",
    "rl_agent/splitfusion_phase14a_100mhz_calibration_v1.py",
    "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json",
    "OAI/oai-cn5g/docker-compose.yaml",
    "scripts/ttracer_extract_csv_smoke.sh",
)


def source_inventory(repo_root: Path = ROOT) -> dict[str, Any]:
    """Hash every repository source/binary actually reachable by this stage."""
    relpaths = set(TRANSITIVE_REPO_SOURCES)
    relpaths.update(str(pin["path"]) for pin in RB.PINS.values())
    files: dict[str, dict[str, Any]] = {}
    for relpath in sorted(relpaths):
        path = repo_root / relpath
        if not path.is_file():
            raise CapacityRunError(f"transitive source is missing: {relpath}")
        files[relpath] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    python_path = Path(sys.executable).resolve()
    runtime = {
        "python_executable": str(python_path),
        "python_sha256": sha256_file(python_path),
        "python_version": sys.version,
    }
    body = {"repo_files": files, "runtime": runtime}
    return {**body, "inventory_sha256": canonical_sha256(body)}


def require_inventory_unchanged(
    initial: Mapping[str, Any], *, stage: str, repo_root: Path = ROOT,
) -> dict[str, Any]:
    current = source_inventory(repo_root)
    if current != dict(initial):
        initial_files = initial.get("repo_files", {})
        current_files = current.get("repo_files", {})
        changed = sorted(
            key for key in set(initial_files) | set(current_files)
            if initial_files.get(key) != current_files.get(key)
        )
        raise CapacityRunError(
            f"transitive execution inventory drifted at {stage}: {changed}")
    return {"stage": stage, "utc": utc_now(), "inventory_sha256":
            current["inventory_sha256"], "verified": True}


def runtime_packetization_identity(
    *, frame_payload_bytes: int, chunk_payload_bytes: int,
) -> dict[str, Any]:
    """Describe what sender/sink actually execute, without hiding drift."""

    chunks = math.ceil(frame_payload_bytes / chunk_payload_bytes)
    tail = frame_payload_bytes - (chunks - 1) * chunk_payload_bytes
    udp_payload = U3.HEADER.size + chunk_payload_bytes
    ipv4_packet = CQ.IPV4_HEADER_BYTES + CQ.UDP_HEADER_BYTES + udp_payload
    return {
        "scope": "CAPACITY_QUALIFICATION_ONLY",
        "frame_payload_bytes": frame_payload_bytes,
        "chunk_payload_bytes": chunk_payload_bytes,
        "chunks_per_frame": chunks,
        "last_chunk_payload_bytes": tail,
        "ssburst_header_bytes": U3.HEADER.size,
        "full_udp_payload_bytes": udp_payload,
        "udp_header_bytes": CQ.UDP_HEADER_BYTES,
        "ipv4_header_bytes": CQ.IPV4_HEADER_BYTES,
        "full_ipv4_packet_bytes": ipv4_packet,
        "path_mtu_bytes": CQ.PATH_MTU_BYTES,
        "mtu_safe_without_ipv4_fragmentation": ipv4_packet <= CQ.PATH_MTU_BYTES,
        "capacity_sink_chunk_bound": "DYNAMIC_EXACT_EXPECTED_COUNT",
        "production_scientific_chunk_bytes": C.CHUNK_BYTES,
        "production_scientific_transport_unchanged": True,
    }


@dataclass
class SinkAccounting:
    """Exact unique-payload accounting on fixed monotonic 100-ms bins."""

    epoch_ns: int
    duration_bins: int
    expected_frames: int
    expected_chunks: int
    expected_frame_payload_bytes: int
    expected_chunk_payload_bytes: int

    def __post_init__(self) -> None:
        self.packetization = runtime_packetization_identity(
            frame_payload_bytes=self.expected_frame_payload_bytes,
            chunk_payload_bytes=self.expected_chunk_payload_bytes)
        require(self.expected_chunks == self.packetization["chunks_per_frame"],
                "capacity sink expected chunk count does not derive from payload")
        require(self.expected_chunks < U3.UINT32_LIMIT,
                "capacity sink chunk count does not fit SSBURST uint32")
        self.payload_bytes_per_bin = [0] * self.duration_bins
        self.seen: set[tuple[int, int]] = set()
        self.accepted_unique = 0
        self.duplicate = 0
        self.malformed = 0
        self.packetization_mismatch = 0
        self.outside = 0
        self.pre_epoch = 0
        self.post_window = 0

    def ingest(self, data: bytes, *, monotonic_ns: int) -> dict[str, Any]:
        event: dict[str, Any] = {
            "receiver_monotonic_ns": monotonic_ns,
            "datagram_bytes": len(data),
        }
        try:
            header = U3.parse_ssburst_datagram(
                data, max_chunks_per_frame=self.expected_chunks)
        except U3.PacketContractError as exc:
            self.malformed += 1
            return {**event, "status": "MALFORMED", "reason": exc.reason}
        event.update({
            "frame_index": header.frame_index,
            "chunk_index": header.chunk_index,
            "chunks_per_frame": header.chunks_per_frame,
            "payload_bytes": len(data) - U3.HEADER.size,
        })
        if (header.chunks_per_frame != self.expected_chunks
                or not 0 <= header.frame_index < self.expected_frames):
            self.outside += 1
            return {**event, "status": "OUTSIDE_REGISTERED_PROBE"}
        payload_bytes = len(data) - U3.HEADER.size
        expected_payload = (
            self.packetization["last_chunk_payload_bytes"]
            if header.chunk_index == self.expected_chunks - 1
            else self.expected_chunk_payload_bytes)
        if payload_bytes != expected_payload:
            self.packetization_mismatch += 1
            return {**event, "status": "PACKETIZATION_MISMATCH",
                    "expected_payload_bytes": expected_payload}
        key = (header.frame_index, header.chunk_index)
        if key in self.seen:
            self.duplicate += 1
            return {**event, "status": "DUPLICATE"}
        self.seen.add(key)
        elapsed = monotonic_ns - self.epoch_ns
        if elapsed < 0:
            self.pre_epoch += 1
            return {**event, "status": "UNIQUE_PRE_EPOCH"}
        index = elapsed // PERIOD_NS
        if index >= self.duration_bins:
            self.post_window += 1
            return {**event, "status": "UNIQUE_AFTER_WINDOW"}
        self.payload_bytes_per_bin[int(index)] += payload_bytes
        self.accepted_unique += 1
        return {**event, "status": "ACCEPTED_UNIQUE", "bin_index": int(index)}

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "scenesense.capacity_unique_ssburst_sink.v1",
            "epoch_monotonic_ns": self.epoch_ns,
            "sample_period_ns": PERIOD_NS,
            "duration_bins": self.duration_bins,
            "expected_frames": self.expected_frames,
            "expected_chunks_per_frame": self.expected_chunks,
            "accepted_unique_chunks": self.accepted_unique,
            "duplicate_chunks": self.duplicate,
            "malformed_datagrams": self.malformed,
            "packetization_mismatch_datagrams": self.packetization_mismatch,
            "packetization": self.packetization,
            "outside_registered_probe": self.outside,
            "unique_pre_epoch": self.pre_epoch,
            "unique_after_window": self.post_window,
            "payload_bytes_per_100ms_bin": self.payload_bytes_per_bin,
            "primary_measurement": (
                "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_"
                "100MS_MONOTONIC_WINDOW"),
            "header_bytes_excluded": U3.HEADER.size,
        }


def _wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.01))


def sink_main(args: argparse.Namespace) -> int:
    accounting = SinkAccounting(
        epoch_ns=args.epoch_monotonic_ns,
        duration_bins=args.duration_bins,
        expected_frames=args.expected_frames,
        expected_chunks=args.expected_chunks,
        expected_frame_payload_bytes=args.expected_frame_payload_bytes,
        expected_chunk_payload_bytes=args.expected_chunk_payload_bytes,
    )
    for path in (args.events_jsonl, args.summary_json, args.ready_json):
        if Path(path).exists():
            raise FileExistsError(f"create-only sink output exists: {path}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.receive_buffer_bytes)
    sock.bind((args.bind_host, args.port))
    sock.settimeout(0.1)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    for caught in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(caught, stop)
    write_json_create(Path(args.ready_json), {
        "status": "READY", "port": sock.getsockname()[1],
        "epoch_monotonic_ns": args.epoch_monotonic_ns,
    })
    deadline = args.epoch_monotonic_ns + args.duration_bins * PERIOD_NS
    with Path(args.events_jsonl).open("x", encoding="utf-8", buffering=1) as events:
        while not stopped and time.monotonic_ns() < deadline:
            try:
                data, _address = sock.recvfrom(65_535)
            except socket.timeout:
                continue
            event = accounting.ingest(data, monotonic_ns=time.monotonic_ns())
            events.write(json.dumps(event, sort_keys=True) + "\n")
    sock.close()
    summary = accounting.summary()
    summary["clean_duration_complete"] = not stopped and time.monotonic_ns() >= deadline
    write_json_create(Path(args.summary_json), summary)
    return 0 if summary["clean_duration_complete"] else 1


def _chunk_spans(payload_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    while offset < payload_bytes:
        length = min(chunk_bytes, payload_bytes - offset)
        spans.append((offset, length))
        offset += length
    return spans


def sender_main(args: argparse.Namespace) -> int:
    for path in (args.decisions_csv, args.summary_json):
        if Path(path).exists():
            raise FileExistsError(f"create-only sender output exists: {path}")
    payload = random.Random(args.payload_seed).randbytes(args.payload_bytes)
    spans = _chunk_spans(args.payload_bytes, args.chunk_bytes)
    require(len(spans) == args.expected_chunks,
            f"sender chunk count {len(spans)} != {args.expected_chunks}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.send_buffer_bytes)
    sock.bind((args.bind_host, 0))
    sock.setblocking(False)
    rows: list[dict[str, Any]] = []
    unexpected_errors = 0
    for frame_index in range(args.frames):
        scheduled = args.epoch_monotonic_ns + frame_index * PERIOD_NS
        _wait_until(scheduled)
        opened = time.monotonic_ns()
        handed = dropped = payload_handed = 0
        first_send: int | None = None
        last_send: int | None = None
        for chunk_index, (offset, length) in enumerate(spans):
            header = U3.HEADER.pack(
                U3.MAGIC, frame_index, chunk_index, len(spans),
                U3.HEADER.size + length)
            try:
                sock.sendto(
                    header + payload[offset:offset + length],
                    (args.remote_host, args.port),
                )
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS):
                    dropped += 1
                    continue
                unexpected_errors += 1
                raise
            stamp = time.monotonic_ns()
            first_send = stamp if first_send is None else first_send
            last_send = stamp
            handed += 1
            payload_handed += length
        rows.append({
            "frame_index": frame_index,
            "scheduled_monotonic_ns": scheduled,
            "action_open_monotonic_ns": opened,
            "first_send_monotonic_ns": first_send,
            "last_send_monotonic_ns": last_send,
            "chunks_handed_to_socket": handed,
            "chunks_dropped_by_socket": dropped,
            "payload_bytes_handed_to_socket": payload_handed,
            "schedule_lag_ms": (opened - scheduled) / 1e6,
        })
    sock.close()
    fields = list(rows[0]) if rows else []
    Path(args.decisions_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.decisions_csv).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "scenesense.capacity_saturating_probe_sender.v1",
        "epoch_monotonic_ns": args.epoch_monotonic_ns,
        "frames": len(rows),
        "payload_bytes": args.payload_bytes,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "chunk_bytes": args.chunk_bytes,
        "chunks_per_frame": len(spans),
        "packetization": runtime_packetization_identity(
            frame_payload_bytes=args.payload_bytes,
            chunk_payload_bytes=args.chunk_bytes),
        "chunks_handed_to_socket": sum(row["chunks_handed_to_socket"] for row in rows),
        "chunks_dropped_by_socket": sum(row["chunks_dropped_by_socket"] for row in rows),
        "unexpected_socket_errors": unexpected_errors,
        "max_schedule_lag_ms": max((row["schedule_lag_ms"] for row in rows),
                                   default=None),
        "role": "SATURATING_PROBE_NEVER_A_TIER",
    }
    write_json_create(Path(args.summary_json), summary)
    return 0


def _line_parts(row: tuple[int, int, str]) -> list[str]:
    return row[2].split(",")


def aggregate_rlc_ticks(
    rows: Sequence[tuple[int, int, str]], *, drop_last: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate consecutive per-LCID RLC rows into UE scheduler ticks."""
    ticks: list[dict[str, Any]] = []
    current_key: tuple[int, int, int] | None = None
    current: dict[int, int] = {}
    stamp = 0
    for receipt_wall, receipt_mono, line in rows:
        del receipt_wall
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            key = (int(parts[2]), int(parts[3]), int(parts[4]))
            lcid = int(parts[5])
            backlog = int(parts[7])
        except ValueError:
            continue
        if current_key is not None and key != current_key:
            ticks.append({
                "receipt_monotonic_ns": stamp,
                "ue_id": current_key[0], "frame": current_key[1],
                "slot": current_key[2], "backlog_bytes": sum(current.values()),
            })
            current = {}
        current_key = key
        current[lcid] = backlog
        stamp = receipt_mono
    if current_key is not None and not drop_last:
        ticks.append({
            "receipt_monotonic_ns": stamp,
            "ue_id": current_key[0], "frame": current_key[1],
            "slot": current_key[2], "backlog_bytes": sum(current.values()),
        })
    return ticks


def backlogged_bins(
    rows: Sequence[tuple[int, int, str]], *, start_ns: int, bins: int,
) -> dict[str, Any]:
    grouped: list[list[int]] = [[] for _ in range(bins)]
    for tick in aggregate_rlc_ticks(rows):
        index = (int(tick["receipt_monotonic_ns"]) - start_ns) // PERIOD_NS
        if 0 <= index < bins:
            grouped[int(index)].append(int(tick["backlog_bytes"]))
    flags = [bool(values) and min(values) > 0 for values in grouped]
    return {
        "bin_backlogged": flags,
        "tick_counts_per_bin": [len(values) for values in grouped],
        "minimum_backlog_per_bin": [min(values) if values else None
                                     for values in grouped],
        "backlogged_fraction": sum(flags) / bins,
    }


def trailing_zero_backlog_run(
    ticks: Sequence[Mapping[str, Any]], *, after_ns: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Return the trailing zero-backlog run after a causal boundary.

    ``NRUE_MAC_RLC_BUFFER_STATUS`` is emitted on every connected UE scheduler
    tick, before the grant lookup, so this proof does not depend on another
    payload or grant being issued after the probe stops.  A non-zero tick
    resets the run; an earlier zero interval can therefore never certify the
    current queue state.
    """
    retained = [dict(tick) for tick in ticks
                if int(tick["receipt_monotonic_ns"]) >= after_ns]
    run = 0
    for tick in retained:
        if int(tick["backlog_bytes"]) == 0:
            run += 1
        else:
            run = 0
    return run, retained


def _sum_mono_event_bytes(
    rows: Sequence[tuple[int, int, str]], *, start_ns: int, end_ns: int,
    bytes_index: int,
) -> dict[str, Any]:
    count = total = 0
    for _receipt_wall, _receipt_mono, line in rows:
        parts = line.split(",")
        if len(parts) <= bytes_index:
            continue
        try:
            stamp = int(parts[1]) * 1_000_000_000 + int(parts[2])
            size = int(parts[bytes_index])
        except ValueError:
            continue
        if start_ns <= stamp < end_ns:
            count += 1
            total += size
    return {"events": count, "bytes": total}


def latest_mono_event_ns(
    rows: Sequence[tuple[int, int, str]], *, after_ns: int
) -> tuple[int, int]:
    """Return the latest in-payload monotonic stamp and admitted row count."""

    latest = after_ns
    count = 0
    for _receipt_wall, _receipt_mono, line in rows:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            stamp = int(parts[1]) * 1_000_000_000 + int(parts[2])
        except ValueError:
            continue
        if stamp >= after_ns:
            latest = max(latest, stamp)
            count += 1
    return latest, count


def analyze_point(
    *, label: str, target_snr_db: float, commanded_noise_power_db: float,
    epoch_ns: int, sink_summary: Mapping[str, Any],
    telemetry: Mapping[str, Sequence[tuple[int, int, str]]],
    achieved_pusch_snr_values: Sequence[float],
) -> tuple[CQ.CapacityPoint, list[float], dict[str, Any]]:
    """Pure conversion of one retained capture into a gated point."""
    payload_bins = list(sink_summary["payload_bytes_per_100ms_bin"])
    require(len(payload_bins) >= POINT_BINS,
            f"{label}: sink retained {len(payload_bins)} bins, need {POINT_BINS}")
    measured_payload = payload_bins[SETTLE_BINS:SETTLE_BINS + MEASURE_BINS]
    service = [float(value) * 8.0 / CQ.SAMPLE_PERIOD_S / 1e6
               for value in measured_payload]
    measure_start = epoch_ns + SETTLE_BINS * PERIOD_NS
    measure_end = measure_start + MEASURE_BINS * PERIOD_NS
    backlog = backlogged_bins(
        telemetry["rlc_buffer"], start_ns=measure_start, bins=MEASURE_BINS)
    achieved = tuple(float(value) for value in achieved_pusch_snr_values)
    require(
        all(math.isfinite(value) for value in achieved),
        f"{label}: achieved PUSCH SNR contains a non-finite value",
    )
    point = CQ.CapacityPoint(
        achieved_pusch_snr_db_p50=(statistics.median(achieved) if achieved else math.nan),
        achieved_pusch_snr_samples=len(achieved),
        label=label, target_snr_db=target_snr_db,
        commanded_noise_power_db=commanded_noise_power_db,
        service_mbps_p10=CQ._percentile(service, 0.10),
        service_mbps_p50=CQ._percentile(service, 0.50),
        service_mbps_p90=CQ._percentile(service, 0.90),
        samples=len(service),
        backlogged_fraction=float(backlog["backlogged_fraction"]),
    )
    ingress = _sum_mono_event_bytes(
        telemetry["rlc_sdu"], start_ns=measure_start, end_ns=measure_end,
        bytes_index=5)
    dequeue = _sum_mono_event_bytes(
        telemetry["rlc_dequeue"], start_ns=measure_start, end_ns=measure_end,
        bytes_index=5)
    delivered = _sum_mono_event_bytes(
        telemetry["gnb_pdcp_deliver"], start_ns=measure_start, end_ns=measure_end,
        bytes_index=5)
    ticks = [tick for tick in aggregate_rlc_ticks(telemetry["rlc_buffer"])
             if measure_start <= int(tick["receipt_monotonic_ns"]) < measure_end]
    start_backlog = int(ticks[0]["backlog_bytes"]) if ticks else None
    end_backlog = int(ticks[-1]["backlog_bytes"]) if ticks else None
    recurrence_expected = None
    recurrence_residual = None
    if start_backlog is not None and end_backlog is not None:
        recurrence_expected = max(0, start_backlog + ingress["bytes"] - dequeue["bytes"])
        recurrence_residual = end_backlog - recurrence_expected
    extdn_bytes = sum(measured_payload)
    corroboration_complete = all(item["events"] > 0 for item in (
        ingress, dequeue, delivered)) and bool(ticks)
    corroboration = {
        "measurement_start_monotonic_ns": measure_start,
        "measurement_end_monotonic_ns": measure_end,
        "primary_extdn_unique_application_payload_bytes": extdn_bytes,
        "ue_rlc_tx_sdu": ingress,
        "ue_rlc_tx_dequeue": dequeue,
        "gnb_pdcp_rx_deliver": delivered,
        "backlog": backlog,
        "first_backlog_bytes": start_backlog,
        "last_backlog_bytes": end_backlog,
        "recurrence_expected_end_bytes": recurrence_expected,
        "recurrence_residual_bytes": recurrence_residual,
        "extdn_to_gnb_pdcp_byte_ratio": (
            extdn_bytes / delivered["bytes"] if delivered["bytes"] > 0 else None),
        "corroboration_complete": corroboration_complete,
        "corroboration_role": "DIAGNOSTIC_NOT_PRIMARY_AND_NEVER_RESCALING_SERVICE",
    }
    return point, service, corroboration


def _manifest_files(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and str(path.relative_to(root)) not in excluded_set:
            rows.append({
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapacityRunError(f"cannot open {label} JSON {path}: {exc}") from exc
    require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _safe_manifest_path(root: Path, value: Any) -> tuple[str, Path]:
    require(type(value) is str and bool(value),
            "manifest relative_path must be a non-empty string")
    relative = Path(value)
    require(not relative.is_absolute(),
            f"manifest path must be relative: {value!r}")
    require(value == relative.as_posix() and not any(
        part in ("", ".", "..") for part in relative.parts),
        f"manifest path is not canonical or escapes its root: {value!r}")
    candidate = (root / relative).resolve()
    require(root == candidate.parent or root in candidate.parents,
            f"manifest path escapes its root: {value!r}")
    return value, candidate


def _validate_container_image_bindings(
    value: Any, *, expected_names: Sequence[str],
) -> dict[str, Any]:
    require(type(value) is dict, "container image binding must be an object")
    require(set(value) == set(expected_names),
            "container image binding names differ from registered core containers")
    normalized: dict[str, Any] = {}
    for name in expected_names:
        row = value[name]
        require(type(row) is dict, f"container image binding {name} is not an object")
        image_id = row.get("image_id")
        require(type(image_id) is str and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)),
                f"container {name} has no immutable sha256 image ID")
        require(type(row.get("configured_image")) is str and row["configured_image"],
                f"container {name} has no configured image reference")
        require(type(row.get("repo_digests")) is list and all(
            type(item) is str for item in row["repo_digests"]),
            f"container {name} RepoDigests are malformed")
        normalized[name] = row
    return normalized


def verify_bound_capacity_result(path: Path) -> dict[str, Any]:
    """Reconstruct and verify a captured result before it can site Run-4 tiers.

    A matching result hash is necessary but insufficient: this verifier also
    rejects stale source inventories, incomplete manifests, path traversal,
    unbound container images, missing teardown/cold proofs, and any result whose
    registered audit cannot be reproduced from the retained point samples.
    """
    path = path.resolve()
    require(path.name == RESULT_FILENAME, f"capacity result must be named {RESULT_FILENAME}")
    root = path.parent
    manifest_path = root / MANIFEST_FILENAME
    terminal_path = root / TERMINAL_FILENAME
    require(path.is_file() and manifest_path.is_file() and terminal_path.is_file(),
            "capacity result, manifest and terminal seal must all exist")
    result = _json_object(path, label="capacity result")
    manifest = _json_object(manifest_path, label="capacity manifest")
    terminal = _json_object(terminal_path, label="capacity terminal")
    require(result.get("schema") == RESULT_SCHEMA, "capacity result schema mismatch")
    require(manifest.get("schema") == MANIFEST_SCHEMA,
            "capacity manifest schema mismatch")
    require(terminal.get("schema") == TERMINAL_SCHEMA,
            "capacity terminal schema mismatch")
    require(result.get("stage_id") == CQ.STAGE_ID,
            "capacity result stage identity mismatch")
    require(terminal.get("status") == STATUS_CAPTURED,
            f"capacity terminal is not captured: {terminal.get('status')}")
    require(result.get("status") == STATUS_CAPTURED,
            f"capacity result is not captured: {result.get('status')}")
    require(terminal.get("result_sha256") == sha256_file(path),
            "capacity result hash does not match terminal")
    require(terminal.get("manifest_sha256") == sha256_file(manifest_path),
            "capacity manifest hash does not match terminal")
    require(manifest.get("status") == STATUS_CAPTURED,
            f"capacity manifest is not captured: {manifest.get('status')}")
    require(manifest.get("result_sha256") == sha256_file(path),
            "capacity manifest does not bind the result")

    manifest_files = manifest.get("files")
    require(type(manifest_files) is list, "capacity manifest files must be a list")
    seen: set[str] = set()
    for item in manifest_files:
        require(type(item) is dict, "capacity manifest entry must be an object")
        relative, candidate = _safe_manifest_path(root, item.get("relative_path"))
        require(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        require(candidate.is_file(), f"manifested evidence missing: {candidate}")
        require(type(item.get("size_bytes")) is int and item["size_bytes"] >= 0,
                f"manifested evidence size is invalid: {relative}")
        require(candidate.stat().st_size == item["size_bytes"],
                f"manifested evidence size changed: {candidate}")
        require(type(item.get("sha256")) is str
                and bool(re.fullmatch(r"[0-9a-f]{64}", item["sha256"])),
                f"manifested evidence digest is invalid: {relative}")
        require(sha256_file(candidate) == item["sha256"],
                f"manifested evidence hash changed: {candidate}")
    expected_manifest = _manifest_files(
        root, excluded=(MANIFEST_FILENAME, TERMINAL_FILENAME))
    require(manifest_files == expected_manifest,
            "capacity manifest is not the exact current file inventory")
    expected_evidence = _manifest_files(
        root, excluded=(RESULT_FILENAME, MANIFEST_FILENAME, TERMINAL_FILENAME))
    require(result.get("evidence_files") == expected_evidence,
            "capacity result evidence inventory is incomplete or stale")

    require(result.get("qualified") is True, "capacity result is not qualified")
    require(result.get("failure") is None, "captured capacity result records a failure")
    require(result.get("radio_profile_id") == RB.RADIO_PROFILE_ID,
            "capacity result radio identity mismatch")
    require(result.get("primary_service_measurement") ==
            "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_100MS_MONOTONIC_WINDOW",
            "capacity result uses an unregistered primary service metric")
    require(result.get("pusch_tb_is_primary") is False,
            "PUSCH transport-block bytes cannot be the primary service metric")

    inventory = result.get("source_inventory")
    require(type(inventory) is dict, "capacity result has no source inventory")
    body = {"repo_files": inventory.get("repo_files"),
            "runtime": inventory.get("runtime")}
    require(inventory.get("inventory_sha256") == canonical_sha256(body),
            "capacity source-inventory digest is invalid")
    require(inventory == source_inventory(ROOT),
            "capacity source inventory is not the exact current execution inventory")
    require(manifest.get("source_inventory_sha256") ==
            inventory.get("inventory_sha256"),
            "capacity manifest does not bind the source inventory")
    require(terminal.get("source_inventory_sha256") ==
            inventory.get("inventory_sha256"),
            "capacity terminal does not bind the source inventory")

    checks = result.get("source_verifications")
    require(type(checks) is list and bool(checks),
            "capacity result has no source/radio verification chain")
    required_stages = {
        "before_preflight", "before_point_p25", "before_point_p50",
        "before_point_p75", "final_sealing",
    }
    observed_stages: set[str] = set()
    for check in checks:
        require(type(check) is dict, "source verification row must be an object")
        observed_stages.add(str(check.get("stage")))
        require(check.get("source_inventory", {}).get("verified") is True,
                "source verification did not attest the inventory")
        require(check.get("radio_binding", {}).get("verified") is True,
                "source verification did not attest the radio binding")
        require(check.get("protected_evidence", {}).get("all_unchanged") is True,
                "source verification did not attest protected evidence")
    require(required_stages <= observed_stages,
            "capacity result is missing required source-verification stages")

    probe = CQ.verify_probe_action(ROOT)
    require(result.get("probe_identity") == probe,
            "capacity result does not bind the current largest eligible probe action")
    config = _json_object(DEFAULT_CONFIG, label="registered config")
    expected_packetization = CQ.verify_probe_packetization(
        config["capacity_qualification"]["probe"]["packetization"],
        production_chunk_bytes=C.CHUNK_BYTES,
        ssburst_header_bytes=U3.HEADER.size)
    require(result.get("probe_packetization") == expected_packetization,
            "capacity result does not bind the registered probe packetization")
    expected_containers = tuple(config["radio"]["core_containers"])
    _validate_container_image_bindings(
        result.get("container_images"), expected_names=expected_containers)

    records = result.get("points")
    require(type(records) is list and len(records) == 3,
            "capacity result must retain exactly three point records")
    points: list[CQ.CapacityPoint] = []
    boundary_samples: Sequence[float] | None = None
    labels: list[str] = []
    for record in records:
        require(type(record) is dict, "capacity point record must be an object")
        try:
            point = CQ.CapacityPoint(**record["capacity_point"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CapacityRunError(f"capacity point record is malformed: {exc}") from exc
        require(record.get("label") == point.label
                and record.get("target_snr_db") == point.target_snr_db,
                f"{point.label}: point record identity does not reproduce")
        prime = record.get("prime")
        require(type(prime) is dict
                and prime.get("commanded_noise_power_db")
                    == point.commanded_noise_power_db
                and prime.get("read_back_noise_power_db")
                    == point.commanded_noise_power_db,
                f"{point.label}: commanded/read-back RF state does not reproduce")
        require(record.get("packetization") == expected_packetization,
                f"{point.label}: point packetization does not reproduce")
        sender = record.get("sender")
        require(type(sender) is dict
                and sender.get("frames") == POINT_FRAMES
                and sender.get("payload_bytes") == CQ.PROBE_PAYLOAD_BYTES
                and sender.get("chunk_bytes") == CQ.PROBE_CHUNK_PAYLOAD_BYTES
                and sender.get("chunks_per_frame") == PROBE_CHUNKS
                and sender.get("packetization") == expected_packetization
                and sender.get("unexpected_socket_errors") == 0,
                f"{point.label}: sender packetization summary is invalid")
        require(type(sender.get("chunks_handed_to_socket")) is int
                and type(sender.get("chunks_dropped_by_socket")) is int
                and sender["chunks_handed_to_socket"] >= 0
                and sender["chunks_dropped_by_socket"] >= 0
                and sender["chunks_handed_to_socket"]
                    + sender["chunks_dropped_by_socket"]
                    == POINT_FRAMES * PROBE_CHUNKS,
                f"{point.label}: sender datagram accounting is invalid")
        labels.append(point.label)
        samples = record.get("service_mbps_samples")
        require(type(samples) is list and len(samples) == point.samples,
                f"{point.label}: retained service samples do not match point count")
        require(all(type(value) in (int, float) and math.isfinite(float(value))
                    and float(value) >= 0.0 for value in samples),
                f"{point.label}: retained service samples are invalid")
        sink = record.get("sink")
        require(type(sink) is dict, f"{point.label}: retained sink record is absent")
        require(sink.get("clean_duration_complete") is True
                and sink.get("expected_frames") == POINT_FRAMES
                and sink.get("expected_chunks_per_frame") == PROBE_CHUNKS
                and sink.get("header_bytes_excluded") == U3.HEADER.size
                and sink.get("packetization") == expected_packetization
                and sink.get("malformed_datagrams") == 0
                and sink.get("packetization_mismatch_datagrams") == 0
                and sink.get("outside_registered_probe") == 0,
                f"{point.label}: sink packetization summary is invalid")
        payload_bins = sink.get("payload_bytes_per_100ms_bin")
        require(type(payload_bins) is list and len(payload_bins) >= POINT_BINS
                and all(type(value) is int and value >= 0 for value in payload_bins),
                f"{point.label}: retained sink payload bins are invalid")
        derived_samples = [
            float(value) * 8.0 / CQ.SAMPLE_PERIOD_S / 1e6
            for value in payload_bins[SETTLE_BINS:SETTLE_BINS + MEASURE_BINS]
        ]
        require(samples == derived_samples,
                f"{point.label}: service samples do not derive from ext-DN payload bins")
        expected_percentiles = (
            CQ._percentile(derived_samples, 0.10),
            CQ._percentile(derived_samples, 0.50),
            CQ._percentile(derived_samples, 0.90),
        )
        require((point.service_mbps_p10, point.service_mbps_p50,
                 point.service_mbps_p90) == expected_percentiles,
                f"{point.label}: service percentiles do not reproduce from "
                "ext-DN payload bins")
        achieved = record.get("achieved_pusch_snr_db")
        require(type(achieved) is dict and type(achieved.get("values")) is list,
                f"{point.label}: achieved-PUSCH samples are absent")
        achieved_values = achieved["values"]
        require(len(achieved_values) == point.achieved_pusch_snr_samples
                and all(type(value) in (int, float) and math.isfinite(float(value))
                        for value in achieved_values),
                f"{point.label}: achieved-PUSCH samples are invalid")
        require(achieved.get("samples") == len(achieved_values)
                and achieved.get("p50") == statistics.median(achieved_values)
                and point.achieved_pusch_snr_db_p50
                == statistics.median(achieved_values),
                f"{point.label}: achieved-PUSCH summary does not reproduce")
        corroboration = record.get("corroboration")
        require(type(corroboration) is dict,
                f"{point.label}: corroborating telemetry is absent")
        backlog = corroboration.get("backlog")
        require(type(backlog) is dict,
                f"{point.label}: retained backlog evidence is absent")
        flags = backlog.get("bin_backlogged")
        counts = backlog.get("tick_counts_per_bin")
        minima = backlog.get("minimum_backlog_per_bin")
        require(type(flags) is list and len(flags) == MEASURE_BINS
                and all(type(value) is bool for value in flags)
                and type(counts) is list and len(counts) == MEASURE_BINS
                and all(type(value) is int and value >= 0 for value in counts)
                and type(minima) is list and len(minima) == MEASURE_BINS
                and all(value is None or type(value) is int and value >= 0
                        for value in minima),
                f"{point.label}: retained backlog-bin evidence is malformed")
        reproduced_flags = [
            count > 0 and minimum is not None and minimum > 0
            for count, minimum in zip(counts, minima)
        ]
        reproduced_fraction = sum(reproduced_flags) / MEASURE_BINS
        require(flags == reproduced_flags
                and backlog.get("backlogged_fraction") == reproduced_fraction
                and point.backlogged_fraction == reproduced_fraction,
                f"{point.label}: backlog saturation summary does not reproduce")
        require(corroboration.get(
                    "primary_extdn_unique_application_payload_bytes")
                == sum(payload_bins[SETTLE_BINS:SETTLE_BINS + MEASURE_BINS]),
                f"{point.label}: primary ext-DN byte total does not reproduce")
        sources = tuple(corroboration.get(name) for name in (
            "ue_rlc_tx_sdu", "ue_rlc_tx_dequeue", "gnb_pdcp_rx_deliver"))
        corroboration_complete = all(
            type(source) is dict
            and type(source.get("events")) is int and source["events"] > 0
            and type(source.get("bytes")) is int and source["bytes"] >= 0
            for source in sources
        ) and any(count > 0 for count in counts)
        require(corroboration_complete
                and corroboration.get("corroboration_complete") is True,
                f"{point.label}: corroborating telemetry is incomplete")
        drain = record.get("post_probe_drain")
        require(type(drain) is dict and drain.get("drained") is True
                and drain.get("no_new_pdcp_or_rlc_ingress_during_quiet_interval") is True
                and int(drain.get("quiet_elapsed_ns", -1))
                >= int(drain.get("quiet_interval_ns", 0)) > 0,
                f"{point.label}: post-probe drain/quiet proof is invalid")
        points.append(point)
        if point.label == CQ.BOUNDARY_OPERATING_POINT:
            boundary_samples = [float(value) for value in samples]
    require(labels == ["p25", "p50", "p75"],
            f"capacity point record order/identity is invalid: {labels}")
    recomputed_audit = CQ.audit_points(
        points, boundary_service_samples=boundary_samples, repo_root=ROOT)
    require(recomputed_audit.get("qualified") is True,
            "recomputed capacity audit is not qualified")
    require(result.get("audit") == recomputed_audit,
            "capacity audit does not reproduce from retained point evidence")
    require(result.get("adverse_capacity_mbps") ==
            recomputed_audit.get("adverse_capacity_mbps"),
            "capacity boundary differs from the recomputed audit")

    expected = CQ.select_tiers(float(result["adverse_capacity_mbps"]), repo_root=ROOT)
    require([tier.to_json() for tier in expected] == result.get("selected_tiers"),
            "capacity selected tiers do not reproduce from the pinned catalogue")
    initial_drain = result.get("initial_drain")
    require(type(initial_drain) is dict and initial_drain.get("drained") is True
            and initial_drain.get(
                "no_new_pdcp_or_rlc_ingress_during_quiet_interval") is True,
            "capacity result lacks the initial radio-probe drain/quiet proof")
    cold = result.get("final_cold_state")
    require(type(cold) is dict and cold.get("cold") is True
            and cold.get("schema") == "scenesense.capacity_final_cold_state.v1"
            and cold.get("orphan_processes") == {}
            and cold.get("residual_ue_tunnels") == []
            and cold.get("carla_processes") == []
            and cold.get("probe_errors") == []
            and set(cold.get("core_containers", {})) == set(expected_containers)
            and not any(str(value).startswith("true")
                        for value in cold["core_containers"].values()),
            "capacity result lacks a complete cold final-state proof")
    teardown = result.get("teardown")
    core_teardown = teardown.get("core", {}) if type(teardown) is dict else {}
    require(type(teardown) is dict
            and teardown.get("extract_ttracer_ok") is True
            and teardown.get("ran_notes") == []
            and core_teardown.get("stopped") is True
            and core_teardown.get("returncode") == 0
            and set(core_teardown.get("core_after", {})) == set(expected_containers)
            and not any(str(value).startswith("true")
                        for value in core_teardown["core_after"].values()),
            "capacity result teardown is incomplete or failed")
    require(type(result.get("radio_lineage")) is dict
            and bool(result["radio_lineage"]),
            "capacity result has no authorization/run lineage")
    return {
        **result,
        "binding_verified": True,
        "binding": {
            "result_sha256": sha256_file(path),
            "manifest_sha256": sha256_file(manifest_path),
            "terminal_sha256": sha256_file(terminal_path),
            "root": str(root),
        },
    }


class Runner(NR.Runner):
    """One-RAN, three-point capacity stage with explicit inter-point drains."""

    def __init__(
        self, config_path: Path, output_dir: Path, *,
        initial_inventory: Mapping[str, Any], lineage: Mapping[str, Any],
    ) -> None:
        super().__init__(config_path, output_dir)
        self.initial_inventory = dict(initial_inventory)
        self.lineage = dict(lineage)
        self.capacity_live: dict[str, V3R.n2.LiveCsv] = {}
        self.source_checks: list[dict[str, Any]] = []
        self.point_records: list[dict[str, Any]] = []
        self.container_images: dict[str, Any] = {}
        self.initial_drain: dict[str, Any] = {}
        self.probe_identity: dict[str, Any] = {}
        self.probe_packetization: dict[str, Any] = {}

    def _timeout(self, name: str) -> float:
        raw = self.config["capacity_qualification"]["subprocess_timeouts_s"][name]
        value = float(raw)
        require(math.isfinite(value) and value > 0.0,
                f"subprocess timeout {name!r} must be finite and positive")
        return value

    def _run_external(
        self, argv: Sequence[str], *, timeout_name: str,
        cwd: Path | None = None, env: Mapping[str, str] | None = None,
        stderr: int | None = subprocess.STDOUT,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(argv), cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                text=True, stdout=subprocess.PIPE, stderr=stderr,
                timeout=self._timeout(timeout_name), check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CapacityRunError(
                f"bounded subprocess {argv!r} exceeded "
                f"{self._timeout(timeout_name):.1f}s") from exc

    def _tunnel_interfaces(self) -> list[str]:
        completed = self._run_external(
            ["ip", "-o", "link", "show"], timeout_name="process_probe",
            stderr=subprocess.DEVNULL)
        return sorted(set(re.findall(r"\b(oaitun_ue\d+)\b", completed.stdout)))

    def assert_cold_ran(self, stage: str) -> None:
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            found = self._run_external(
                ["sudo", "-n", "pgrep", "-a", "-x", name],
                timeout_name="process_probe")
            require(not (found.returncode == 0 and found.stdout.strip()),
                    f"{stage}: cold-RAN gate failed: {found.stdout.strip()}")
        stale = self._tunnel_interfaces()
        require(not stale, f"{stage}: stale UE tunnel(s) {stale}")
        tel = self.config["telemetry"]
        ports = [4043, self.config["actuator"]["telnet_port"],
                 tel["gnb_port"], tel["ue_port"],
                 tel["gnb_relay_port"], tel["ue_relay_port"]]
        busy = [port for port in ports if not V3R.n2.port_is_free(int(port))]
        require(not busy, f"{stage}: ports not free: {busy}")

    def start_ran_via_launcher(self, cell_tag: str, cell_dir: Path) -> dict[str, Any]:
        """Run the pinned launcher under an explicit wall-clock timeout."""
        RB.assert_no_forbidden_env(os.environ)
        launcher = ROOT / self.config["paths"]["launcher"]
        state_dir = (ROOT / self.config["paths"]["radio_state_root"]
                     / f"{self.output_dir.name}__{cell_tag}")
        PE.assert_outside_protected_run(state_dir, ROOT)
        require(not state_dir.exists(),
                f"radio state dir already exists (create-only): {state_dir}")
        argv = ["bash", str(launcher), "--execute", RB.EXECUTION_TOKEN,
                "--output", str(state_dir)]
        env = {key: value for key, value in os.environ.items()
               if key not in RB.FORBIDDEN_ENV}
        completed = self._run_external(
            argv, timeout_name="launcher", cwd=ROOT, env=env)
        log = self.out(f"cells/{cell_tag}/logs/launcher.log")
        with log.open("x", encoding="utf-8") as handle:
            handle.write(completed.stdout or "")
        require(completed.returncode == 0,
                f"launcher failed ({completed.returncode}): "
                f"{(completed.stdout or '')[-800:]}")
        require("SPLITFUSION_OAI_100MHZ_4D5U_ATTACHED" in (completed.stdout or ""),
                "launcher did not report the attach token")
        self.radio_state_dir = state_dir
        materialization = _json_object(
            state_dir / "radio_materialization.json", label="radio materialization")
        self.ue_ip = self.config["radio"]["ue_static_ip"]
        record = {
            "launcher": self.config["paths"]["launcher"],
            "launcher_sha256": RB.PINS["launcher"]["sha256"],
            "execution_token": RB.EXECUTION_TOKEN,
            "command": " ".join(shlex.quote(item) for item in argv),
            "radio_state_dir": str(state_dir),
            "effective_gnb_path": materialization.get("effective_gnb_path"),
            "effective_ue_path": materialization.get("effective_ue_path"),
            "radio_profile_id": RB.RADIO_PROFILE_ID,
            "ue_ip": self.ue_ip,
            "launcher_timeout_s": self._timeout("launcher"),
        }
        write_json_create(cell_dir / "radio_attach.json", record)
        return record

    def teardown_ran(self) -> list[str]:
        """Bound every external probe/signal while retaining base cleanup."""
        notes = V3R.Runner.teardown_ran(self)
        for name in ("nr-uesoftmodem", "nr-softmodem"):
            try:
                found = self._run_external(
                    ["sudo", "-n", "pgrep", "-x", name],
                    timeout_name="process_probe")
                if found.returncode != 0 or not found.stdout.strip():
                    continue
                self._run_external(
                    ["sudo", "-n", "pkill", "-INT", "-x", name],
                    timeout_name="signal_command")
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    again = self._run_external(
                        ["sudo", "-n", "pgrep", "-x", name],
                        timeout_name="process_probe")
                    if again.returncode != 0 or not again.stdout.strip():
                        break
                    time.sleep(0.5)
                else:
                    self._run_external(
                        ["sudo", "-n", "pkill", "-TERM", "-x", name],
                        timeout_name="signal_command")
                    term_deadline = time.monotonic() + 10.0
                    while time.monotonic() < term_deadline:
                        again = self._run_external(
                            ["sudo", "-n", "pgrep", "-x", name],
                            timeout_name="process_probe")
                        if again.returncode != 0 or not again.stdout.strip():
                            break
                        time.sleep(0.5)
                    else:
                        self._run_external(
                            ["sudo", "-n", "pkill", "-KILL", "-x", name],
                            timeout_name="signal_command")
                        notes.append(f"{name} required SIGKILL")
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                notes.append(f"{name} bounded teardown failed: {exc}")
        try:
            stale = self._tunnel_interfaces()
            if stale:
                notes.append(f"stale UE tunnel(s) after teardown: {stale}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"tunnel teardown probe failed: {exc}")
        self.radio_state_dir = None
        return notes

    def extract_ttracer(self, cell_tag: str, cell_dir: Path) -> None:
        """Extract retained traces with the preregistered bounded timeout."""
        script = ROOT / self.config["paths"]["extract_script"]
        for source_name in ("gnb", "ue"):
            raw = cell_dir / "ttracer" / source_name / f"{source_name}.raw"
            if not raw.exists() or raw.stat().st_size == 0:
                self.notes.append(f"{cell_tag}: empty raw for {source_name}")
                continue
            argv = [str(script), "--raw", str(raw), "--source", source_name,
                    "--output-root", str(cell_dir), "--clean-output"]
            for event in self.config["telemetry"]["events"][source_name]:
                argv += ["--event", event]
            completed = self._run_external(
                argv, timeout_name="ttracer_extract", cwd=ROOT)
            require(completed.returncode == 0,
                    f"T-tracer extraction failed ({completed.returncode}): "
                    f"{completed.stdout[-1000:]}")

    def verify_all(self, stage: str) -> dict[str, Any]:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage)
        inventory = require_inventory_unchanged(
            self.initial_inventory, stage=stage, repo_root=ROOT)
        report = {
            "stage": stage, "utc": utc_now(),
            "source_inventory": inventory,
            "radio_binding": RB.verify(stage, ROOT),
            "protected_evidence": PE.require_unchanged(stage, ROOT),
        }
        self.source_checks.append(report)
        write_json_create(self.out(f"verification/{safe}.json"), report)
        return report

    def _container_states(self) -> dict[str, str]:
        states: dict[str, str] = {}
        for name in self.config["radio"]["core_containers"]:
            completed = self._run_external(
                ["sudo", "-n", "docker", "inspect", "-f",
                 "{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                 name], timeout_name="docker_inspect")
            states[name] = (completed.stdout.strip()
                            if completed.returncode == 0 else "ABSENT")
        return states

    def _container_image_bindings(self) -> dict[str, Any]:
        """Bind running core containers to immutable image IDs/RepoDigests."""
        bindings: dict[str, Any] = {}
        for name in self.config["radio"]["core_containers"]:
            inspected = self._run_external(
                ["sudo", "-n", "docker", "inspect", name],
                timeout_name="docker_inspect")
            require(inspected.returncode == 0,
                    f"cannot inspect running core container {name}")
            values = json.loads(inspected.stdout)
            require(type(values) is list and len(values) == 1,
                    f"docker inspect for {name} returned an unexpected shape")
            row = values[0]
            image_id = str(row.get("Image", ""))
            configured = str(row.get("Config", {}).get("Image", ""))
            require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)),
                    f"container {name} has no immutable sha256 image ID")
            image = self._run_external(
                ["sudo", "-n", "docker", "image", "inspect", image_id],
                timeout_name="docker_inspect")
            require(image.returncode == 0,
                    f"cannot inspect image {image_id} for {name}")
            image_rows = json.loads(image.stdout)
            require(type(image_rows) is list and len(image_rows) == 1
                    and image_rows[0].get("Id") == image_id,
                    f"image identity did not reconcile for {name}")
            digests = image_rows[0].get("RepoDigests") or []
            require(type(digests) is list and all(type(item) is str for item in digests),
                    f"RepoDigests are malformed for {name}")
            bindings[name] = {
                "configured_image": configured,
                "image_id": image_id,
                "repo_digests": sorted(digests),
                "container_id": str(row.get("Id", "")),
            }
        return _validate_container_image_bindings(
            bindings, expected_names=tuple(self.config["radio"]["core_containers"]))

    def preflight_before_launcher(self) -> dict[str, Any]:
        RB.assert_no_forbidden_env(os.environ)
        privileged = self._run_external(
            ["sudo", "-n", "true"], timeout_name="process_probe")
        require(privileged.returncode == 0, "passwordless sudo preflight failed")
        self.assert_cold_ran("capacity preflight")
        core = self._container_states()
        require(not any(value.startswith("true") for value in core.values()),
                f"capacity stage requires a cold core, found {core}")
        carla = self._run_external(
            ["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
            timeout_name="process_probe", stderr=subprocess.DEVNULL)
        require(not carla.stdout.strip(), "CARLA is running; stage is network-only")
        C.assert_catalog_digests(ROOT)
        self.probe_identity = CQ.verify_probe_action(ROOT)
        registered = self.config["capacity_qualification"]
        self.probe_packetization = CQ.verify_probe_packetization(
            registered["probe"]["packetization"],
            production_chunk_bytes=C.CHUNK_BYTES,
            ssburst_header_bytes=U3.HEADER.size)
        require(PROBE_CHUNKS > U3.MAX_CHUNKS_PER_FRAME_LIMIT,
                "capacity retry no longer exercises the custom >1024-chunk sink")

        require(int(registered["min_pusch_snr_samples"]) == CQ.MIN_PUSCH_SNR_SAMPLES,
                "configured PUSCH sample gate differs from preregistration")
        require(float(registered["max_achieved_target_snr_error_db"])
                == CQ.MAX_ACHIEVED_TARGET_SNR_ERROR_DB,
                "configured achieved-SNR tolerance differs from preregistration")
        require(float(registered["service_measurement"]["drain_quiet_interval_s"])
                == CQ.DRAIN_QUIET_INTERVAL_S,
                "configured ingress-quiet interval differs from preregistration")
        required_timeouts = {
            "launcher", "process_probe", "docker_inspect", "core_down",
            "ttracer_extract", "signal_command", "route_probe",
        }
        require(set(registered["subprocess_timeouts_s"]) == required_timeouts,
                "registered subprocess timeout inventory is incomplete or has drifted")
        for name in sorted(required_timeouts):
            self._timeout(name)
        record = {
            "utc": utc_now(), "radio_profile_id": RB.RADIO_PROFILE_ID,
            "core_before": core, "ran_cold": True, "carla_absent": True,
            "cuda_or_model_started_by_stage": False,
            "probe_identity": self.probe_identity,
            "probe_packetization": self.probe_packetization,
            "subprocess_timeouts_s": dict(registered["subprocess_timeouts_s"]),
        }
        write_json_create(self.out("preflight.json"), record)
        return record

    def bind_edge_context(self) -> dict[str, Any]:
        states = self._container_states()
        require(all(value.startswith("true") and "unhealthy" not in value
                    for value in states.values()),
                f"launcher did not leave a healthy core: {states}")
        self.container_images = self._container_image_bindings()
        container = str(self.config["radio"]["edge_container"])
        edge = self._run_external(
            ["sudo", "-n", "docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container],
            timeout_name="docker_inspect")
        require(edge.returncode == 0, f"cannot inspect ext-DN container {container}")
        edge_host = edge.stdout.strip()
        require(bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", edge_host)),
                f"invalid ext-DN address {edge_host!r}")
        local = self._run_external(
            ["ip", "route", "show", "table", "local"],
            timeout_name="route_probe")
        require(local.returncode == 0, "cannot inspect the host local route table")
        require(f"local {edge_host} " not in local.stdout,
                "ext-DN address is host-local and would bypass the radio")
        self.edge_host = edge_host
        pid = self._run_external(
            ["sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
             container], timeout_name="docker_inspect")
        require(pid.returncode == 0, f"cannot inspect ext-DN PID for {container}")
        self.edge_pid = int(pid.stdout.strip())
        return {
            "core_after_launcher": states, "edge_container": container,
            "edge_host": edge_host, "edge_pid": self.edge_pid,
            "container_images": self.container_images,
        }

    def verify_radio_path(self, cell_dir: Path) -> dict[str, Any]:
        """Bounded form of the two independent radio-routing proofs."""
        radio = self.config["radio"]
        assert self.ue_ip is not None and self.edge_host is not None
        interface = radio["ue_interface"]
        route_result = self._run_external(
            ["ip", "route", "get", self.edge_host, "from", self.ue_ip],
            timeout_name="route_probe")
        route = route_result.stdout.strip()
        via_tunnel = f"dev {interface}" in route
        rules_result = self._run_external(
            ["ip", "rule", "show"], timeout_name="route_probe")
        table = None
        for line in rules_result.stdout.splitlines():
            parts = line.split()
            if "from" in parts and self.ue_ip in parts and "lookup" in parts:
                table = parts[parts.index("lookup") + 1]
                break
        table_routes = ""
        if table is not None:
            table_result = self._run_external(
                ["ip", "route", "show", "table", table],
                timeout_name="route_probe")
            table_routes = table_result.stdout
        rule_via_tunnel = interface in table_routes
        local_result = self._run_external(
            ["ip", "route", "show", "table", "local"],
            timeout_name="route_probe")
        is_local = f"local {self.edge_host} " in local_result.stdout
        outcome = {
            "edge_host": self.edge_host, "ue_ip": self.ue_ip,
            "route_lookup": route, "routes_via_ue_tunnel": via_tunnel,
            "policy_rule_table": table,
            "policy_table_routes_via_tunnel": rule_via_tunnel,
            "policy_table_routes": table_routes.strip().splitlines(),
            "edge_host_is_host_local": is_local,
            "radio_path_verified": (via_tunnel or rule_via_tunnel) and not is_local,
            "subprocess_timeout_s": self._timeout("route_probe"),
        }
        write_json_create(cell_dir / "radio_path_check.json", outcome)
        require(not is_local,
                f"edge host {self.edge_host} is host-local; traffic would bypass the radio")
        require(via_tunnel or rule_via_tunnel,
                f"no route sends {self.ue_ip} -> {self.edge_host} through "
                f"{interface}; route={route!r}; table={table!r}")
        return outcome

    def start_capacity_live_csv(self, cell_tag: str) -> None:
        tel = self.config["telemetry"]
        troot = ROOT / self.config["paths"]["t_tracer_dir"]
        messages = ROOT / self.config["paths"]["t_messages"]
        specs = {
            "rlc_buffer": ("ue_relay_port", "NRUE_MAC_RLC_BUFFER_STATUS",
                           RLC_BUFFER_FIELDS),
            "pdcp_sdu": ("ue_relay_port", "NR_PDCP_TX_SDU",
                         MONO_SDU_FIELDS),
            "rlc_sdu": ("ue_relay_port", "NR_RLC_TX_SDU", MONO_SDU_FIELDS),
            "rlc_dequeue": ("ue_relay_port", "NR_RLC_TX_DEQUEUE", DEQUEUE_FIELDS),
            "gnb_pdcp_deliver": ("gnb_relay_port", "GNB_PDCP_RX_DELIVER",
                                 MONO_SDU_FIELDS),
        }
        for label, (port_key, event, fields) in specs.items():
            command = [
                str(troot / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                "-p", str(tel[port_key]), "-f", "-s", ",", "-t", "time",
                event, *fields,
            ]
            self.capacity_live[label] = V3R.n2.LiveCsv(
                command,
                self.out(f"cells/{cell_tag}/ttracer/live/{label}.csv"))
        self.start_live_pusch(cell_tag)
        time.sleep(1.0)
        require(all(item.process.poll() is None
                    for item in self.capacity_live.values()),
                "a live capacity telemetry extractor exited during startup")

    def stop_capacity_live_csv(self) -> None:
        for label, live in list(self.capacity_live.items()):
            try:
                live.stop()
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"live {label}: {exc}")
        self.capacity_live.clear()

    def telemetry_snapshot(self) -> dict[str, list[tuple[int, int, str]]]:
        return {label: live.snapshot()
                for label, live in self.capacity_live.items()}

    def prove_zero_backlog(self, *, after_ns: int, label: str) -> dict[str, Any]:
        """Require both a drained RLC queue and a stable ingress-quiet window."""
        service = self.config["capacity_qualification"]["service_measurement"]
        required = int(service["drain_zero_consecutive_samples"])
        timeout = float(service["drain_timeout_s"])
        poll = float(service["drain_poll_s"])
        quiet_s = float(service["drain_quiet_interval_s"])
        require(quiet_s == CQ.DRAIN_QUIET_INTERVAL_S,
                "drain quiet interval differs from preregistration")
        quiet_ns = int(quiet_s * 1e9)
        deadline = time.monotonic() + timeout
        best_run = 0
        current_run = 0
        retained: list[dict[str, Any]] = []
        latest_ingress_ns = after_ns
        pdcp_rows = rlc_rows = 0
        quiet_elapsed_ns = 0
        last_signature: tuple[int, int, int, int] | None = None
        last_change_observed_ns = time.monotonic_ns()
        while time.monotonic() < deadline:
            pdcp_latest, pdcp_rows = latest_mono_event_ns(
                self.capacity_live["pdcp_sdu"].snapshot(), after_ns=after_ns)
            rlc_latest, rlc_rows = latest_mono_event_ns(
                self.capacity_live["rlc_sdu"].snapshot(), after_ns=after_ns)
            signature = (pdcp_rows, pdcp_latest, rlc_rows, rlc_latest)
            observed_ns = time.monotonic_ns()
            if signature != last_signature:
                last_signature = signature
                last_change_observed_ns = observed_ns
            latest_ingress_ns = max(after_ns, pdcp_latest, rlc_latest)
            proof_boundary_ns = max(latest_ingress_ns, last_change_observed_ns)
            current_run, ticks = trailing_zero_backlog_run(
                aggregate_rlc_ticks(
                    self.capacity_live["rlc_buffer"].snapshot(), drop_last=True),
                after_ns=proof_boundary_ns)
            best_run = max(best_run, current_run)
            retained = ticks[-max(required * 2, 10):]
            quiet_elapsed_ns = max(0, observed_ns - last_change_observed_ns)
            if current_run >= required and quiet_elapsed_ns >= quiet_ns:
                return {
                    "label": label, "after_monotonic_ns": after_ns,
                    "required_consecutive_zero_ticks": required,
                    "observed_consecutive_zero_ticks": current_run,
                    "best_zero_run": best_run, "drained": True,
                    "quiet_interval_ns": quiet_ns,
                    "quiet_elapsed_ns": quiet_elapsed_ns,
                    "latest_ingress_monotonic_ns": latest_ingress_ns,
                    "ingress_observation_stable_since_monotonic_ns":
                        last_change_observed_ns,
                    "proof_boundary_monotonic_ns": proof_boundary_ns,
                    "post_boundary_pdcp_sdu_rows": pdcp_rows,
                    "post_boundary_rlc_sdu_rows": rlc_rows,
                    "no_new_pdcp_or_rlc_ingress_during_quiet_interval": True,
                    "proof_ticks": retained, "proved_monotonic_ns": observed_ns,
                }
            time.sleep(poll)
        return {
            "label": label, "after_monotonic_ns": after_ns,
            "required_consecutive_zero_ticks": required,
            "observed_consecutive_zero_ticks": current_run,
            "best_zero_run": best_run, "drained": False,
            "quiet_interval_ns": quiet_ns,
            "quiet_elapsed_ns": quiet_elapsed_ns,
            "latest_ingress_monotonic_ns": latest_ingress_ns,
            "ingress_observation_stable_since_monotonic_ns":
                last_change_observed_ns,
            "post_boundary_pdcp_sdu_rows": pdcp_rows,
            "post_boundary_rlc_sdu_rows": rlc_rows,
            "no_new_pdcp_or_rlc_ingress_during_quiet_interval": False,
            "proof_ticks": retained, "proved_monotonic_ns": time.monotonic_ns(),
        }

    def _prime_target(
        self, *, label: str, target_snr_db: float,
        command_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        command, clamped = V3R.inverse_interpolate(target_snr_db, self.anchors)
        require(not clamped, f"{label}: registered target would clamp")
        command = V3R.round_to_granularity(
            command, float(self.config["actuator"]["command_granularity_db"]))
        self.send_noise(
            command, reason="CAPACITY_POINT_PRIME_IDLE", log=command_log,
            operating_point=label, target_snr_db=target_snr_db, clamped=False)
        readback = self.read_back_noise()
        require(abs(readback - command) <= 1e-6,
                f"{label}: RF readback {readback} != command {command}")
        return {"commanded_noise_power_db": command,
                "read_back_noise_power_db": readback,
                "command_event": command_log[-1]}

    def _launch_point_probe(
        self, *, label: str, index: int, epoch_ns: int, point_dir: Path,
    ) -> dict[str, Any]:
        assert self.edge_pid is not None and self.edge_host is not None
        assert self.ue_ip is not None
        service = self.config["capacity_qualification"]["service_measurement"]
        duration_bins = POINT_BINS + int(math.ceil(
            float(service["receiver_tail_s"]) / CQ.SAMPLE_PERIOD_S))
        events = point_dir / "receiver_events.jsonl"
        receiver_summary = point_dir / "receiver_summary.json"
        ready = point_dir / "receiver_ready.json"
        sender_csv = point_dir / "sender_decisions.csv"
        sender_summary = point_dir / "sender_summary.json"
        port = int(service["receiver_port"])
        module = "rl_agent.ue_mcs_backlog_near_capacity_v1.capacity_runner"
        receiver = self.spawn(
            f"capacity_sink_{label}",
            ["sudo", "-n", "nsenter", "-t", str(self.edge_pid), "-n",
             sys.executable, "-m", module, "sink",
             "--bind-host", "0.0.0.0", "--port", str(port),
             "--epoch-monotonic-ns", str(epoch_ns),
             "--duration-bins", str(duration_bins),
             "--expected-frames", str(POINT_FRAMES),
             "--expected-chunks", str(PROBE_CHUNKS),
             "--expected-frame-payload-bytes",
             str(CQ.PROBE_PAYLOAD_BYTES),
             "--expected-chunk-payload-bytes",
             str(CQ.PROBE_CHUNK_PAYLOAD_BYTES),
             "--receive-buffer-bytes",
             str(self.config["traffic"]["receive_buffer_bytes"]),
             "--events-jsonl", str(events), "--summary-json", str(receiver_summary),
             "--ready-json", str(ready)],
            f"cells/capacity/logs/{label}_receiver.log", root_owned=True)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not ready.is_file():
            require(receiver.process.poll() is None,
                    f"{label}: ext-DN sink exited before READY")
            time.sleep(0.1)
        require(ready.is_file(), f"{label}: ext-DN sink did not report READY")
        require(epoch_ns - time.monotonic_ns() >= int(1e9),
                f"{label}: sink startup consumed the future-epoch guard")
        sender = self.spawn(
            f"capacity_sender_{label}",
            [sys.executable, "-m", module, "sender",
             "--bind-host", self.ue_ip, "--remote-host", self.edge_host,
             "--port", str(port), "--epoch-monotonic-ns", str(epoch_ns),
             "--frames", str(POINT_FRAMES),
             "--payload-bytes", str(CQ.PROBE_PAYLOAD_BYTES),
             "--chunk-bytes", str(CQ.PROBE_CHUNK_PAYLOAD_BYTES),
             "--expected-chunks", str(PROBE_CHUNKS),
             "--payload-seed", str(2026092404 + index),
             "--send-buffer-bytes", str(self.config["traffic"]["send_buffer_bytes"]),
             "--decisions-csv", str(sender_csv),
             "--summary-json", str(sender_summary)],
            f"cells/capacity/logs/{label}_sender.log")
        return {"receiver": receiver, "sender": sender, "events": events,
                "receiver_summary": receiver_summary,
                "sender_csv": sender_csv, "sender_summary": sender_summary,
                "epoch_monotonic_ns": epoch_ns, "duration_bins": duration_bins}

    def run_point(
        self, *, label: str, index: int, target_snr_db: float,
        command_log: list[dict[str, Any]],
    ) -> tuple[CQ.CapacityPoint, list[float], dict[str, Any]]:
        self.verify_all(f"before_point_{label}")
        prime = self._prime_target(
            label=label, target_snr_db=target_snr_db, command_log=command_log)
        service = self.config["capacity_qualification"]["service_measurement"]
        epoch_ns = time.monotonic_ns() + int(float(service["future_epoch_lead_s"]) * 1e9)
        point_dir = self.output_dir / "cells" / "capacity" / "points" / label
        point_dir.mkdir(parents=True, exist_ok=False)
        session = self._launch_point_probe(
            label=label, index=index, epoch_ns=epoch_ns, point_dir=point_dir)
        sender_budget = POINT_BINS * CQ.SAMPLE_PERIOD_S + 20.0
        session["sender"].process.wait(timeout=sender_budget)
        require(session["sender"].process.returncode == 0,
                f"{label}: probe sender exited {session['sender'].process.returncode}")
        receiver_budget = (session["duration_bins"] * CQ.SAMPLE_PERIOD_S + 10.0)
        session["receiver"].process.wait(timeout=receiver_budget)
        require(session["receiver"].process.returncode == 0,
                f"{label}: ext-DN sink exited {session['receiver'].process.returncode}")
        sender_summary = json.loads(session["sender_summary"].read_text())
        sink_summary = json.loads(session["receiver_summary"].read_text())
        require(sender_summary["frames"] == POINT_FRAMES,
                f"{label}: sender emitted {sender_summary['frames']} frames")
        require(sender_summary.get("payload_bytes") == CQ.PROBE_PAYLOAD_BYTES
                and sender_summary.get("chunk_bytes")
                    == CQ.PROBE_CHUNK_PAYLOAD_BYTES
                and sender_summary.get("chunks_per_frame") == PROBE_CHUNKS
                and sender_summary.get("packetization")
                    == self.probe_packetization,
                f"{label}: sender packetization differs from registered retry")
        expected_datagrams = POINT_FRAMES * PROBE_CHUNKS
        require(sender_summary.get("chunks_handed_to_socket", 0)
                + sender_summary.get("chunks_dropped_by_socket", 0)
                == expected_datagrams,
                f"{label}: sender datagram accounting is incomplete")
        require(sender_summary["unexpected_socket_errors"] == 0,
                f"{label}: sender reported unexpected socket error")
        require(sink_summary["clean_duration_complete"],
                f"{label}: sink did not finish its bounded duration")
        require(sink_summary.get("expected_frames") == POINT_FRAMES
                and sink_summary.get("expected_chunks_per_frame") == PROBE_CHUNKS
                and sink_summary.get("header_bytes_excluded") == U3.HEADER.size
                and sink_summary.get("packetization") == self.probe_packetization,
                f"{label}: sink packetization differs from registered retry")
        require(sink_summary["malformed_datagrams"] == 0,
                f"{label}: malformed SSBURST datagrams observed")
        require(sink_summary.get("packetization_mismatch_datagrams") == 0,
                f"{label}: sink observed a packetization mismatch")
        require(sink_summary["outside_registered_probe"] == 0,
                f"{label}: datagram outside registered probe identity")
        telemetry = self.telemetry_snapshot()
        measure_start = epoch_ns + SETTLE_BINS * PERIOD_NS
        measure_end = measure_start + MEASURE_BINS * PERIOD_NS
        pusch_values = []
        if self.live_pusch is not None:
            for _wall, receipt_mono, line in self.live_pusch.snapshot():
                if not measure_start <= receipt_mono < measure_end:
                    continue
                parts = line.split(",")
                try:
                    pusch_values.append(int(parts[4]) / 10.0)
                except (IndexError, ValueError):
                    continue
        point, samples, corroboration = analyze_point(
            label=label, target_snr_db=target_snr_db,
            commanded_noise_power_db=float(prime["commanded_noise_power_db"]),
            epoch_ns=epoch_ns, sink_summary=sink_summary, telemetry=telemetry,
            achieved_pusch_snr_values=pusch_values)
        require(corroboration["corroboration_complete"],
                f"{label}: cross-layer corroborating telemetry is incomplete")
        require(point.achieved_pusch_snr_samples >= CQ.MIN_PUSCH_SNR_SAMPLES,
                f"{label}: insufficient achieved-PUSCH evidence")
        require(abs(point.achieved_pusch_snr_db_p50 - target_snr_db)
                <= CQ.MAX_ACHIEVED_TARGET_SNR_ERROR_DB,
                f"{label}: achieved PUSCH SNR missed the target")
        point_end = epoch_ns + POINT_BINS * PERIOD_NS
        drain = self.prove_zero_backlog(after_ns=point_end, label=label)
        require(drain["drained"],
                f"{label}: UE RLC backlog did not return to zero before next point")
        record = {
            "label": label, "target_snr_db": target_snr_db,
            "prime": prime, "epoch_monotonic_ns": epoch_ns,
            "packetization": dict(self.probe_packetization),
            "sender": sender_summary, "sink": sink_summary,
            "capacity_point": point.to_json(), "service_mbps_samples": samples,
            "corroboration": corroboration,
            "achieved_pusch_snr_db": {
                "values": list(pusch_values),
                "samples": len(pusch_values),
                "p50": statistics.median(pusch_values) if pusch_values else None,
            },
            "post_probe_drain": drain,
        }
        write_json_create(point_dir / "point_record.json", record)
        self.point_records.append(record)
        return point, samples, record

    def stop_core(self) -> dict[str, Any]:
        cn_dir = ROOT / "OAI/oai-cn5g"
        try:
            completed = self._run_external(
                ["sudo", "-n", "docker", "compose", "down", "--remove-orphans"],
                timeout_name="core_down", cwd=cn_dir)
            states = self._container_states()
            return {
                "returncode": completed.returncode,
                "output_tail": completed.stdout[-1000:],
                "timeout_s": self._timeout("core_down"),
                "core_after": states,
                "stopped": completed.returncode == 0 and not any(
                    value.startswith("true") for value in states.values()),
            }
        except Exception as exc:  # noqa: BLE001 - preserve a failed teardown record
            return {
                "returncode": None, "output_tail": "", "core_after": {},
                "timeout_s": self._timeout("core_down"), "stopped": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def final_cold_with_core(self) -> dict[str, Any]:
        """Create one immutable, timeout-bounded final cold-state record."""
        orphans: dict[str, list[str]] = {}
        probe_errors: list[str] = []
        for name in ("nr-softmodem", "nr-uesoftmodem"):
            try:
                found = self._run_external(
                    ["sudo", "-n", "pgrep", "-a", "-x", name],
                    timeout_name="process_probe")
                if found.returncode == 0 and found.stdout.strip():
                    orphans[name] = found.stdout.strip().splitlines()
            except Exception as exc:  # noqa: BLE001
                probe_errors.append(f"{name}: {exc}")
        patterns = (
            ("tracer", "T/tracer/(record|multi|csv|replay)"),
            ("sender", "ue_mcs_backlog_calibration_v1.tagged_sender"),
            ("receiver", "ue_n3_structured_udp_receiver"),
            ("capacity", "ue_mcs_backlog_near_capacity_v1.capacity_runner"),
        )
        for label, pattern in patterns:
            try:
                found = self._run_external(
                    ["pgrep", "-af", pattern], timeout_name="process_probe",
                    stderr=subprocess.DEVNULL)
                rows = [row for row in found.stdout.splitlines()
                        if "pgrep" not in row and str(os.getpid()) not in row]
                if rows:
                    orphans[label] = rows
            except Exception as exc:  # noqa: BLE001
                probe_errors.append(f"{label}: {exc}")
        try:
            tunnels = self._tunnel_interfaces()
        except Exception as exc:  # noqa: BLE001
            tunnels = []
            probe_errors.append(f"tunnels: {exc}")
        try:
            carla = self._run_external(
                ["pgrep", "-af", "CarlaUE4|CarlaUnreal"],
                timeout_name="process_probe", stderr=subprocess.DEVNULL)
            carla_rows = [row for row in carla.stdout.splitlines()
                          if "pgrep" not in row]
        except Exception as exc:  # noqa: BLE001
            carla_rows = []
            probe_errors.append(f"carla: {exc}")
        try:
            states = self._container_states()
        except Exception as exc:  # noqa: BLE001
            states = {}
            probe_errors.append(f"core containers: {exc}")
        state = {
            "schema": "scenesense.capacity_final_cold_state.v1",
            "utc": utc_now(),
            "loadavg": Path("/proc/loadavg").read_text().strip(),
            "orphan_processes": orphans,
            "residual_ue_tunnels": tunnels,
            "carla_processes": carla_rows,
            "core_containers": states,
            "probe_errors": probe_errors,
            "cold": (not orphans and not tunnels and not carla_rows
                     and not probe_errors
                     and not any(value.startswith("true")
                                 for value in states.values())),
        }
        write_json_create(self.output_dir / "final_cold_state.json", state)
        return state

    def _write_seals(
        self, *, status: str, failure: str | None, audit: Mapping[str, Any],
        selected: Sequence[CQ.SelectedTier], final_cold: Mapping[str, Any],
        teardown: Mapping[str, Any],
    ) -> None:
        evidence = _manifest_files(
            self.output_dir,
            excluded=(RESULT_FILENAME, MANIFEST_FILENAME, TERMINAL_FILENAME))
        result = {
            "schema": RESULT_SCHEMA,
            "stage_id": CQ.STAGE_ID, "status": status,
            "qualified": status == STATUS_CAPTURED,
            "failure": failure, "radio_profile_id": RB.RADIO_PROFILE_ID,
            "primary_service_measurement": (
                "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_"
                "100MS_MONOTONIC_WINDOW"),
            "pusch_tb_is_primary": False,
            "probe_identity": dict(self.probe_identity),
            "probe_packetization": dict(self.probe_packetization),
            "container_images": dict(self.container_images),
            "initial_drain": dict(self.initial_drain),
            "audit": dict(audit),
            "adverse_capacity_mbps": audit.get("adverse_capacity_mbps"),
            "selected_tiers": [tier.to_json() for tier in selected],
            "points": self.point_records,
            "source_inventory": self.initial_inventory,
            "source_verifications": self.source_checks,
            "radio_lineage": self.lineage,
            "evidence_files": evidence,
            "final_cold_state": dict(final_cold),
            "teardown": dict(teardown),
            "created_utc": utc_now(),
        }
        result_path = self.output_dir / RESULT_FILENAME
        write_json_create(result_path, result)
        manifest_files = _manifest_files(
            self.output_dir, excluded=(MANIFEST_FILENAME, TERMINAL_FILENAME))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": status, "created_utc": utc_now(),
            "result_sha256": sha256_file(result_path),
            "source_inventory_sha256": self.initial_inventory["inventory_sha256"],
            "files": manifest_files,
        }
        manifest_path = self.output_dir / MANIFEST_FILENAME
        write_json_create(manifest_path, manifest)
        terminal_name = (TERMINAL_FILENAME if status == STATUS_CAPTURED
                         else f"{status}.json")
        write_json_create(self.output_dir / terminal_name, {
            "schema": TERMINAL_SCHEMA,
            "status": status, "created_utc": utc_now(),
            "result_sha256": sha256_file(result_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_inventory_sha256": self.initial_inventory["inventory_sha256"],
        })

    def run(self) -> int:
        status = STATUS_FAILED
        failure: str | None = None
        points: list[CQ.CapacityPoint] = []
        samples: dict[str, list[float]] = {}
        selected: tuple[CQ.SelectedTier, ...] = ()
        audit: dict[str, Any] = {"qualified": False, "problems": ["not run"]}
        command_log: list[dict[str, Any]] = []
        final_cold: dict[str, Any] = {}
        teardown: dict[str, Any] = {}
        cell_dir = self.output_dir / "cells" / "capacity"
        cell_dir.mkdir(parents=True, exist_ok=False)

        def terminate(signum: int, _frame: Any) -> None:
            self.aborted = True
            raise CapacityRunError(f"received signal {signum}")

        for caught in (signal.SIGINT, signal.SIGTERM):
            signal.signal(caught, terminate)
        try:
            self.verify_all("before_preflight")
            self.preflight_before_launcher()
            attach = self.start_ran_via_launcher("capacity", cell_dir)
            write_json_create(cell_dir / "radio_attach_capacity.json", attach)
            context = self.bind_edge_context()
            write_json_create(cell_dir / "edge_context.json", context)
            self.verify_radio_path(cell_dir)
            self.start_telemetry("capacity")
            self.open_telnet(cell_dir)
            # Collect RLC state before the path probe so its queue transition
            # and subsequent drain are both observable.
            self.start_capacity_live_csv("capacity")
            self.udp_probe("capacity", cell_dir)
            probe_end_ns = time.monotonic_ns()
            initial_drain = self.prove_zero_backlog(
                after_ns=probe_end_ns, label="initial_after_udp_probe")
            write_json_create(cell_dir / "initial_drain.json", initial_drain)
            require(initial_drain["drained"],
                    "initial UE RLC backlog did not drain after path probe")
            self.initial_drain = initial_drain
            for index, label in enumerate(("p25", "p50", "p75")):
                point, point_samples, _record = self.run_point(
                    label=label, index=index,
                    target_snr_db=CQ.ADVERSE_OPERATING_POINTS_DB[label],
                    command_log=command_log)
                points.append(point)
                samples[label] = point_samples
            audit = CQ.audit_points(
                points, boundary_service_samples=samples["p50"], repo_root=ROOT)
            if audit["qualified"]:
                selected = CQ.select_tiers(
                    float(audit["adverse_capacity_mbps"]), repo_root=ROOT)
                status = STATUS_CAPTURED
            else:
                status = STATUS_REFUSED
                failure = "; ".join(audit["problems"])
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
            status = STATUS_FAILED
        finally:
            if self.telnet is not None:
                try:
                    restored = self.restore_clean(cell_dir, command_log)
                    if not restored:
                        raise CapacityRunError("RF restore/readback did not verify")
                except Exception as exc:  # noqa: BLE001
                    failure = failure or f"RF restore failed: {exc}"
                    status = STATUS_FAILED
            write_json_create(cell_dir / "command_log.json", command_log)
            self.stop_capacity_live_csv()
            teardown_notes = self.teardown_ran()
            if teardown_notes:
                self.notes.extend(teardown_notes)
            try:
                self.extract_ttracer("capacity", cell_dir)
                expected = {
                    "ue": self.config["telemetry"]["events"]["ue"],
                    "gnb": self.config["telemetry"]["events"]["gnb"],
                }
                missing = []
                for source, events in expected.items():
                    for event in events:
                        path = cell_dir / "ttracer" / source / "csv" / f"{event}.csv"
                        if not path.is_file() or path.stat().st_size == 0:
                            missing.append(str(path))
                require(not missing, f"T-tracer extraction missing {missing}")
                extraction_ok = True
            except Exception as exc:  # noqa: BLE001
                extraction_ok = False
                failure = failure or f"T-tracer extraction failed: {exc}"
                status = STATUS_FAILED
            teardown = {
                "ran_notes": list(self.notes), "extract_ttracer_ok": extraction_ok,
                "core": self.stop_core(),
            }
            if self.notes or not teardown["core"]["stopped"]:
                failure = failure or f"teardown failed: {teardown}"
                status = STATUS_FAILED
            final_cold = self.final_cold_with_core()
            if not final_cold.get("cold"):
                failure = failure or f"final host is not cold: {final_cold}"
                status = STATUS_FAILED
            try:
                self.verify_all("final_sealing")
            except Exception as exc:  # noqa: BLE001
                failure = failure or f"final source/radio verification failed: {exc}"
                status = STATUS_FAILED
            if status == STATUS_CAPTURED and failure:
                status = STATUS_FAILED
            self._write_seals(
                status=status, failure=failure, audit=audit, selected=selected,
                final_cold=final_cold, teardown=teardown)
        print(json.dumps({
            "status": status, "failure": failure,
            "adverse_capacity_mbps": audit.get("adverse_capacity_mbps"),
            "selected_action_ids": [tier.action_id for tier in selected],
            "output_dir": str(self.output_dir), "cold": final_cold.get("cold"),
        }, indent=2))
        return 0 if status == STATUS_CAPTURED else 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    run = sub.add_parser("run", help="run the authorized live stage")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output-dir", type=Path, default=None)

    sink = sub.add_parser("sink", help=argparse.SUPPRESS)
    sink.add_argument("--bind-host", required=True)
    sink.add_argument("--port", type=int, required=True)
    sink.add_argument("--epoch-monotonic-ns", type=int, required=True)
    sink.add_argument("--duration-bins", type=_positive_int, required=True)
    sink.add_argument("--expected-frames", type=_positive_int, required=True)
    sink.add_argument("--expected-chunks", type=_positive_int, required=True)
    sink.add_argument("--expected-frame-payload-bytes", type=_positive_int,
                      required=True)
    sink.add_argument("--expected-chunk-payload-bytes", type=_positive_int,
                      required=True)
    sink.add_argument("--receive-buffer-bytes", type=_positive_int, required=True)
    sink.add_argument("--events-jsonl", type=Path, required=True)
    sink.add_argument("--summary-json", type=Path, required=True)
    sink.add_argument("--ready-json", type=Path, required=True)

    sender = sub.add_parser("sender", help=argparse.SUPPRESS)
    sender.add_argument("--bind-host", required=True)
    sender.add_argument("--remote-host", required=True)
    sender.add_argument("--port", type=int, required=True)
    sender.add_argument("--epoch-monotonic-ns", type=int, required=True)
    sender.add_argument("--frames", type=_positive_int, required=True)
    sender.add_argument("--payload-bytes", type=_positive_int, required=True)
    sender.add_argument("--chunk-bytes", type=_positive_int, required=True)
    sender.add_argument("--expected-chunks", type=_positive_int, required=True)
    sender.add_argument("--payload-seed", type=int, required=True)
    sender.add_argument("--send-buffer-bytes", type=_positive_int, required=True)
    sender.add_argument("--decisions-csv", type=Path, required=True)
    sender.add_argument("--summary-json", type=Path, required=True)
    return parser


def run_main(args: argparse.Namespace) -> int:
    require(args.config.resolve() == DEFAULT_CONFIG.resolve(),
            "capacity qualification accepts only the registered config_v1.json")
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["paths"]["capacity_output_root"]
    output = (args.output_dir if args.output_dir is not None
              else output_root / datetime.now().strftime("%Y%m%d_%H%M%S"))
    PE.assert_outside_protected_run(output, ROOT)
    PE.require_unchanged("capacity_before_mkdir", ROOT)
    RB.verify("capacity_before_mkdir", ROOT)
    inventory = source_inventory(ROOT)
    authorization = AUTH.require_authorization(
        AUTH.CAPACITY_STAGE, output_root,
        expected_token=config["authorization"]["capacity_stage_token"],
        repo_root=ROOT)
    require_inventory_unchanged(inventory, stage="after_authorization", repo_root=ROOT)
    output.mkdir(parents=True, exist_ok=False)
    lineage = AUTH.lineage_record(
        AUTH.CAPACITY_STAGE, run_id=output.name, parent=None,
        authorization=authorization)
    write_json_create(output / "lineage.json", lineage)
    write_json_create(output / "source_inventory.json", inventory)
    return Runner(
        args.config, output, initial_inventory=inventory, lineage=lineage).run()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.role == "sink":
        return sink_main(args)
    if args.role == "sender":
        return sender_main(args)
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

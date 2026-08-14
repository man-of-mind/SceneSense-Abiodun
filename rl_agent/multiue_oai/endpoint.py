#!/usr/bin/env python3
"""Production-shaped UDP endpoint for the DG-A OAI contention gate.

The application message keeps the production ``!IHH`` chunk header.  A small
binary metadata header inside chunk zero makes scheduled demand, reassembly,
latency, and checksums independently auditable.  All latency timestamps use
``CLOCK_MONOTONIC_RAW`` because sender and receiver execute on L10319, even
when the receiver joins the ext-DN network namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import socket
import struct
import subprocess
import threading
import time
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


CHUNK_HEADER = struct.Struct("!IHH")
FRAME_HEADER = struct.Struct("!8sBBHQQII")
MAGIC = b"SSDGAV1\0"
VERSION = 1
IP_UDP_OVERHEAD_BYTES = 28
CLOCK_RAW = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
SEND_KINDS = ("equal", "asymmetric", "burst", "controlled", "smoke", "calibration")
CONTROLLERS = ("open_loop", "decentralized_c1", "centralized_observable")

DEMAND_FIELDS = (
    "ue_id",
    "frame_id",
    "scheduled_raw_ns",
    "scheduled_elapsed_s",
    "demand_rate_mbps",
    "controller",
    "status",
    "decision_raw_ns",
    "send_start_raw_ns",
    "send_end_raw_ns",
    "send_lag_ms",
    "payload_bytes",
    "chunks",
    "onwire_bytes",
    "c_hat_mbps",
    "c1_budget_mbps",
    "admit_reason",
    "local_error",
)

CHUNK_FIELDS = (
    "recv_raw_ns",
    "recv_wall_ns",
    "source_ip",
    "ue_id",
    "message_id",
    "chunk_index",
    "total_chunks",
    "udp_payload_bytes",
    "onwire_bytes",
    "duplicate",
)

FRAME_FIELDS = (
    "ue_id",
    "frame_id",
    "message_id",
    "scheduled_raw_ns",
    "first_chunk_raw_ns",
    "complete_raw_ns",
    "complete_latency_ms",
    "payload_bytes",
    "chunks",
    "onwire_bytes",
    "checksum_expected",
    "checksum_actual",
    "checksum_ok",
    "identity_ok",
    "source_ip",
)


def build_ttracer_csv_command(
    csv_binary: str,
    t_messages: str,
    port: int,
    event: str,
    fields: Sequence[str],
) -> List[str]:
    """Build the OAI ``csv`` CLI, which selects its event without record flags."""
    if not event or not fields:
        raise ValueError("T-tracer CSV requires an event and at least one field")
    return [
        csv_binary,
        "-d",
        t_messages,
        "-ip",
        "127.0.0.1",
        "-p",
        str(port),
        "-f",
        "-s",
        ",",
        "-t",
        "time",
        event,
        *fields,
    ]


def parse_ul_new_data_grant(
    line: str, rnti_to_ue: Mapping[int, int]
) -> Optional[tuple[int, int]]:
    """Return UE ID and TBS for a first-transmission uplink grant."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 7:
        return None
    try:
        direction = int(parts[-6])
        rnti = int(parts[-5], 0)
        tbs = int(float(parts[-4]))
        rv = int(float(parts[-2]))
        round_index = int(float(parts[-1]))
    except ValueError:
        return None
    ue_id = rnti_to_ue.get(rnti)
    # nr_ue_procedures.c emits direction=1 for PUSCH and 0 for PDSCH.
    if direction != 1 or ue_id is None or rv > 0 or round_index > 0:
        return None
    return ue_id, tbs


def raw_ns() -> int:
    return time.clock_gettime_ns(CLOCK_RAW)


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def chunks_per_frame(payload_bytes: int, chunk_bytes: int) -> int:
    capacity = chunk_bytes - CHUNK_HEADER.size
    if capacity <= 0:
        raise ValueError("chunk_bytes must exceed the 8-byte production header")
    return math.ceil(payload_bytes / capacity)


def frame_onwire_bytes(payload_bytes: int, chunk_bytes: int) -> int:
    chunks = chunks_per_frame(payload_bytes, chunk_bytes)
    return payload_bytes + chunks * (CHUNK_HEADER.size + IP_UDP_OVERHEAD_BYTES)


def staggered_arrival_credits(ue_ids: Sequence[int], demand_seed: int) -> Dict[int, float]:
    """Return deterministic, evenly staggered phases with a seed-controlled offset."""
    ordered = [int(value) for value in ue_ids]
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("arrival credits require unique UE IDs")
    digest = hashlib.sha256(f"scenesense-dga-phase-{int(demand_seed)}".encode()).digest()
    base = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return {
        ue_id: (base + index / len(ordered)) % 1.0
        for index, ue_id in enumerate(ordered)
    }


def deterministic_body(size: int, ue_id: int) -> bytes:
    seed = hashlib.sha256(f"scenesense-dga-v1-ue{ue_id}".encode()).digest()
    repeats = (size + len(seed) - 1) // len(seed)
    return (seed * repeats)[:size]


def build_frame_blob(payload_bytes: int, ue_id: int, frame_id: int, scheduled_ns: int) -> bytes:
    if payload_bytes < FRAME_HEADER.size:
        raise ValueError("payload is too small for the DG-A metadata header")
    body = deterministic_body(payload_bytes - FRAME_HEADER.size, ue_id)
    checksum = zlib.crc32(body) & 0xFFFFFFFF
    header = FRAME_HEADER.pack(
        MAGIC,
        VERSION,
        ue_id,
        0,
        frame_id,
        scheduled_ns,
        payload_bytes,
        checksum,
    )
    return header + body


@dataclass
class UEConfig:
    ue_id: int
    bind_ip: str
    demand_fraction: float


@dataclass
class Demand:
    ue_id: int
    frame_id: int
    scheduled_raw_ns: int
    scheduled_elapsed_s: float
    demand_rate_mbps: float
    status: str = "pending"
    decision_raw_ns: int = 0
    send_start_raw_ns: int = 0
    send_end_raw_ns: int = 0
    send_lag_ms: float = float("nan")
    c_hat_mbps: float = float("nan")
    c1_budget_mbps: float = float("nan")
    admit_reason: str = ""
    local_error: str = ""


class GrantObserver:
    """Causal one-tick-lag per-UE new-data service estimator."""

    def __init__(
        self,
        *,
        csv_binary: str,
        t_messages: str,
        port: int,
        rnti_to_ue: Mapping[int, int],
        initial_per_ue_mbps: float,
        window_s: float,
        alpha: float,
        conversion: float,
        log_path: Path,
    ) -> None:
        self.rnti_to_ue = dict(rnti_to_ue)
        self.window_ns = int(window_s * 1e9)
        self.alpha = float(alpha)
        self.conversion = float(conversion)
        self.estimates = {ue: float(initial_per_ue_mbps) for ue in self.rnti_to_ue.values()}
        self.previous = dict(self.estimates)
        self.events: Dict[int, Deque[tuple[int, float]]] = defaultdict(deque)
        self.service_event_count = {ue: 0 for ue in self.rnti_to_ue.values()}
        self.latest_event_raw_ns = {ue: 0 for ue in self.rnti_to_ue.values()}
        self.previous_event_raw_ns = dict(self.latest_event_raw_ns)
        self.returned_event_raw_ns = dict(self.latest_event_raw_ns)
        self.last_available_raw_ns = 0
        self.outstanding = defaultdict(float)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.log_handle = log_path.open("w", encoding="utf-8")
        command = build_ttracer_csv_command(
            csv_binary,
            t_messages,
            port,
            "NRUE_MAC_DCI_GRANT",
            ("time", "direction", "rnti", "tbs", "ndi", "rv", "round"),
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, name="grant-observer", daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if self.stop_event.is_set():
                break
            self.log_handle.write(line)
            self.log_handle.flush()
            parsed = parse_ul_new_data_grant(line, self.rnti_to_ue)
            if parsed is None:
                continue
            ue_id, tbs = parsed
            now = raw_ns()
            service_bytes = float(tbs) * self.conversion
            with self.lock:
                self.events[ue_id].append((now, service_bytes))
                self.service_event_count[ue_id] += 1
                self.latest_event_raw_ns[ue_id] = now
                self.outstanding[ue_id] = max(0.0, self.outstanding[ue_id] - service_bytes)

    def add_outstanding(self, ue_id: int, onwire_bytes: int) -> None:
        with self.lock:
            self.outstanding[ue_id] += float(onwire_bytes)

    def tick(self) -> Dict[int, float]:
        """Return the previous tick's estimate, then incorporate causal service."""
        now = raw_ns()
        with self.lock:
            result = dict(self.previous)
            self.returned_event_raw_ns = dict(self.previous_event_raw_ns)
            self.last_available_raw_ns = now
            for ue_id, events in self.events.items():
                while events and events[0][0] < now - self.window_ns:
                    events.popleft()
                if self.outstanding[ue_id] > 0 and events:
                    observed = sum(value for _, value in events) * 8.0 / (self.window_ns / 1e9) / 1e6
                    self.estimates[ue_id] = (
                        (1.0 - self.alpha) * self.estimates[ue_id] + self.alpha * observed
                    )
            self.previous = dict(self.estimates)
            self.previous_event_raw_ns = dict(self.latest_event_raw_ns)
            return result

    def summary(self) -> dict:
        with self.lock:
            return {
                "service_event_count": {
                    str(ue): count for ue, count in sorted(self.service_event_count.items())
                },
                "latest_event_raw_ns": {
                    str(ue): stamp for ue, stamp in sorted(self.latest_event_raw_ns.items())
                },
            }

    def health(self) -> dict:
        return {
            "process_alive": self.process.poll() is None,
            "process_returncode": self.process.returncode,
            "reader_thread_alive": self.thread.is_alive(),
        }

    def close(self) -> None:
        self.stop_event.set()
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        self.log_handle.close()


class TrafficSender:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.payload_bytes = int(args.payload_bytes)
        self.chunk_bytes = int(args.chunk_bytes)
        self.chunk_capacity = self.chunk_bytes - CHUNK_HEADER.size
        self.chunks = chunks_per_frame(self.payload_bytes, self.chunk_bytes)
        self.onwire_bytes = frame_onwire_bytes(self.payload_bytes, self.chunk_bytes)
        self.tick_s = float(args.tick_s)
        self.ues = self._parse_ues(args.ue)
        self.sockets = self._make_sockets()
        self.demands: List[Demand] = []
        self.pending: Dict[int, Optional[Demand]] = {ue.ue_id: None for ue in self.ues}
        self.next_frame_id = {ue.ue_id: (ue.ue_id + 1) * 100_000_000 for ue in self.ues}
        self.initial_arrival_credit = staggered_arrival_credits(
            [ue.ue_id for ue in self.ues], int(args.demand_seed)
        )
        self.arrival_credit = dict(self.initial_arrival_credit)
        self.tokens = {ue.ue_id: float(self.onwire_bytes) for ue in self.ues}
        self.aggregate_tokens = float(self.onwire_bytes)
        self.sent_onwire = defaultdict(int)
        self.local_errors = 0
        self.observer: Optional[GrantObserver] = None
        self.observer_failure: Optional[dict] = None

    @staticmethod
    def _parse_ues(values: Iterable[str]) -> List[UEConfig]:
        result: List[UEConfig] = []
        for raw in values:
            ue_text, ip, fraction_text = raw.split(",", 2)
            result.append(UEConfig(int(ue_text), ip, float(fraction_text)))
        if not result:
            raise ValueError("at least one --ue ue_id,bind_ip,demand_fraction is required")
        ids = [ue.ue_id for ue in result]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate UE IDs are not allowed: {ids}")
        if any(ue.demand_fraction < 0 for ue in result):
            raise ValueError("UE demand fractions must be non-negative")
        return result

    def _make_sockets(self) -> Dict[int, socket.socket]:
        result: Dict[int, socket.socket] = {}
        for ue in self.ues:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(self.args.socket_send_buffer_bytes))
            sock.settimeout(float(self.args.send_timeout_s))
            sock.bind((ue.bind_ip, 0))
            result[ue.ue_id] = sock
        return result

    def _rate_for(self, ue: UEConfig, elapsed_s: float) -> float:
        if self.args.kind == "burst":
            rho = 0.0
            for phase in self.args.phase:
                start, end, phase_rho = (float(value) for value in phase.split(","))
                if start <= elapsed_s < end:
                    rho = phase_rho
                    break
            return rho * float(self.args.mu_hat_mbps) / len(self.ues)
        return ue.demand_fraction * float(self.args.mu_hat_mbps)

    def _new_demand(self, ue: UEConfig, scheduled_ns: int, start_ns: int, rate: float) -> Demand:
        frame_id = self.next_frame_id[ue.ue_id]
        self.next_frame_id[ue.ue_id] += 1
        demand = Demand(
            ue_id=ue.ue_id,
            frame_id=frame_id,
            scheduled_raw_ns=scheduled_ns,
            scheduled_elapsed_s=(scheduled_ns - start_ns) / 1e9,
            demand_rate_mbps=rate,
        )
        self.demands.append(demand)
        return demand

    def _send(self, demand: Demand, c_hat: float, budget: float, reason: str) -> bool:
        demand.decision_raw_ns = raw_ns()
        demand.c_hat_mbps = c_hat
        demand.c1_budget_mbps = budget
        demand.admit_reason = reason
        demand.send_start_raw_ns = raw_ns()
        blob = build_frame_blob(
            self.payload_bytes,
            demand.ue_id,
            demand.frame_id,
            demand.scheduled_raw_ns,
        )
        message_id = ((demand.ue_id + 1) << 28) | (demand.frame_id & 0x0FFFFFFF)
        try:
            for chunk_index in range(self.chunks):
                start = chunk_index * self.chunk_capacity
                piece = blob[start : start + self.chunk_capacity]
                datagram = CHUNK_HEADER.pack(message_id, chunk_index, self.chunks) + piece
                self.sockets[demand.ue_id].sendto(
                    datagram,
                    (self.args.remote_host, int(self.args.remote_port)),
                )
        except OSError as exc:
            demand.status = "local_error"
            demand.local_error = f"{type(exc).__name__}:{exc}"
            self.local_errors += 1
            return False
        demand.send_end_raw_ns = raw_ns()
        demand.send_lag_ms = (demand.send_start_raw_ns - demand.scheduled_raw_ns) / 1e6
        demand.status = "sent"
        self.sent_onwire[demand.ue_id] += self.onwire_bytes
        if self.observer is not None:
            self.observer.add_outstanding(demand.ue_id, self.onwire_bytes)
        return True

    def _emit_arrivals(self, tick_ns: int, start_ns: int, elapsed_s: float) -> None:
        for ue in self.ues:
            rate = self._rate_for(ue, elapsed_s)
            fps = rate * 1e6 / 8.0 / self.onwire_bytes
            self.arrival_credit[ue.ue_id] += fps * self.tick_s
            while self.arrival_credit[ue.ue_id] >= 1.0 - 1e-12:
                self.arrival_credit[ue.ue_id] -= 1.0
                demand = self._new_demand(ue, tick_ns, start_ns, rate)
                if self.args.controller == "open_loop":
                    self._send(demand, float("nan"), float("nan"), "open_loop")
                else:
                    old = self.pending[ue.ue_id]
                    if old is not None and old.status == "pending":
                        old.status = "replaced"
                        old.decision_raw_ns = tick_ns
                        old.admit_reason = "newest_replaces_unsent"
                    self.pending[ue.ue_id] = demand

    def _controller_tick(self, estimates: Mapping[int, float]) -> None:
        factor = float(self.args.pessimism_factor)
        if self.args.controller == "decentralized_c1":
            for ue in self.ues:
                budget = factor * estimates[ue.ue_id]
                self.tokens[ue.ue_id] = min(
                    float(self.onwire_bytes),
                    self.tokens[ue.ue_id] + budget * 1e6 / 8.0 * self.tick_s,
                )
                demand = self.pending[ue.ue_id]
                if demand is not None and self.tokens[ue.ue_id] >= self.onwire_bytes:
                    if self._send(demand, estimates[ue.ue_id], budget, "local_token_available"):
                        self.tokens[ue.ue_id] -= self.onwire_bytes
                    self.pending[ue.ue_id] = None
        elif self.args.controller == "centralized_observable":
            aggregate_hat = sum(estimates.values())
            budget = factor * aggregate_hat
            self.aggregate_tokens = min(
                float(self.onwire_bytes),
                self.aggregate_tokens + budget * 1e6 / 8.0 * self.tick_s,
            )
            choices = [value for value in self.pending.values() if value is not None]
            if choices and self.aggregate_tokens >= self.onwire_bytes:
                demand = min(choices, key=lambda item: (item.scheduled_raw_ns, item.ue_id))
                if self._send(demand, aggregate_hat, budget, "central_oldest_pending"):
                    self.aggregate_tokens -= self.onwire_bytes
                self.pending[demand.ue_id] = None
        else:
            raise ValueError(f"unsupported controller {self.args.controller}")

    def _start_observer(self) -> None:
        if self.args.controller == "open_loop":
            return
        mapping: Dict[int, int] = {}
        for raw in self.args.rnti_map:
            ue_text, rnti_text = raw.split(",", 1)
            mapping[int(rnti_text, 0)] = int(ue_text)
        if set(mapping.values()) != {ue.ue_id for ue in self.ues}:
            raise ValueError("--rnti-map must cover every configured UE")
        self.observer = GrantObserver(
            csv_binary=self.args.ttracer_csv,
            t_messages=self.args.t_messages,
            port=int(self.args.ttracer_port),
            rnti_to_ue=mapping,
            initial_per_ue_mbps=float(self.args.mu_hat_mbps) / len(self.ues),
            window_s=float(self.args.estimator_window_s),
            alpha=float(self.args.estimator_ewma_alpha),
            conversion=float(self.args.service_conversion),
            log_path=self.run_dir / "live_grant_observer.log",
        )

    def run(self) -> int:
        self._start_observer()
        start_ns = raw_ns() + int(float(self.args.start_delay_s) * 1e9)
        end_ns = start_ns + int(float(self.args.duration_s) * 1e9)
        tick_index = 0
        estimate_log: List[Dict[str, object]] = []
        observer_summary: Optional[dict] = None
        while raw_ns() < start_ns:
            time.sleep(min(0.02, max(0.0, (start_ns - raw_ns()) / 1e9)))
        next_tick = start_ns
        estimates = {ue.ue_id: float(self.args.mu_hat_mbps) / len(self.ues) for ue in self.ues}
        while next_tick < end_ns:
            wait_s = (next_tick - raw_ns()) / 1e9
            if wait_s > 0:
                time.sleep(wait_s)
            elapsed_s = (next_tick - start_ns) / 1e9
            if self.observer is not None:
                health = self.observer.health()
                if not health["process_alive"] or not health["reader_thread_alive"]:
                    self.observer_failure = {
                        "detected_raw_ns": raw_ns(),
                        **health,
                    }
                    break
                estimates = self.observer.tick()
            self._emit_arrivals(next_tick, start_ns, elapsed_s)
            if self.args.controller != "open_loop":
                self._controller_tick(estimates)
                estimate_log.append(
                    {
                        "tick": tick_index,
                        "tick_raw_ns": next_tick,
                        "observation_available_raw_ns": self.observer.last_available_raw_ns,
                        **{f"ue{ue}_c_hat_mbps": value for ue, value in estimates.items()},
                        **{
                            f"ue{ue}_source_event_raw_ns": stamp
                            for ue, stamp in self.observer.returned_event_raw_ns.items()
                        },
                    }
                )
            tick_index += 1
            next_tick = start_ns + int(tick_index * self.tick_s * 1e9)

        for demand in self.pending.values():
            if demand is not None and demand.status == "pending":
                demand.status = "skipped_end"
                demand.decision_raw_ns = raw_ns()
                demand.admit_reason = "trial_end"
        if self.observer is not None:
            observer_summary = self.observer.summary()
            health_before_close = self.observer.health()
            observer_summary["health_before_close"] = health_before_close
            observer_summary["unexpected_failure"] = self.observer_failure
            observer_summary["alive_through_sender"] = bool(
                self.observer_failure is None
                and health_before_close["process_alive"]
                and health_before_close["reader_thread_alive"]
            )
            self.observer.close()
        socket_bindings = {
            str(ue.ue_id): {
                "requested_bind_ip": ue.bind_ip,
                "actual_local_ip": str(self.sockets[ue.ue_id].getsockname()[0]),
                "actual_local_port": int(self.sockets[ue.ue_id].getsockname()[1]),
            }
            for ue in self.ues
        }
        for sock in self.sockets.values():
            sock.close()

        demand_path = self.run_dir / "sender_demands.csv"
        with demand_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEMAND_FIELDS)
            writer.writeheader()
            for demand in self.demands:
                writer.writerow(
                    {
                        **demand.__dict__,
                        "controller": self.args.controller,
                        "payload_bytes": self.payload_bytes,
                        "chunks": self.chunks,
                        "onwire_bytes": self.onwire_bytes,
                    }
                )
        if estimate_log:
            fields = list(estimate_log[0])
            with (self.run_dir / "causal_capacity_estimates.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(estimate_log)

        demand_hash = hashlib.sha256()
        for demand in self.demands:
            demand_hash.update(
                f"{demand.ue_id},{demand.frame_id},{demand.scheduled_elapsed_s:.9f},{demand.demand_rate_mbps:.9f}\n".encode()
            )
        elapsed = max(1e-9, (raw_ns() - start_ns) / 1e9)
        summary = {
            "mode": "sender",
            "kind": self.args.kind,
            "controller": self.args.controller,
            "demand_seed": int(self.args.demand_seed),
            "initial_arrival_credit": {
                str(ue_id): value for ue_id, value in sorted(self.initial_arrival_credit.items())
            },
            "start_raw_ns": start_ns,
            "end_raw_ns": raw_ns(),
            "duration_target_s": float(self.args.duration_s),
            "elapsed_from_start_s": elapsed,
            "payload_bytes": self.payload_bytes,
            "chunks_per_frame": self.chunks,
            "onwire_bytes_per_frame": self.onwire_bytes,
            "demand_count": len(self.demands),
            "demand_trace_sha256": demand_hash.hexdigest(),
            "local_errors": self.local_errors,
            "socket_bindings": socket_bindings,
            "observer": observer_summary,
            "per_ue": {
                str(ue.ue_id): {
                    "sent_frames": sum(
                        1 for demand in self.demands if demand.ue_id == ue.ue_id and demand.status == "sent"
                    ),
                    "sent_onwire_bytes": self.sent_onwire[ue.ue_id],
                }
                for ue in self.ues
            },
        }
        atomic_json(self.run_dir / "sender_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if observer_summary is not None and any(
            int(value) <= 0 for value in observer_summary["service_event_count"].values()
        ):
            return 2
        if observer_summary is not None and not observer_summary["alive_through_sender"]:
            return 2
        return 1 if self.local_errors else 0


class TrafficReceiver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = Path(args.stop_file).resolve() if args.stop_file else None
        self.stop_event = threading.Event()
        self.partial: MutableMapping[tuple[str, int], Dict[str, object]] = {}
        self.frames: List[Dict[str, object]] = []
        self.chunk_count = 0
        self.duplicate_count = 0
        self.invalid_count = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(args.socket_receive_buffer_bytes))
        self.sock.settimeout(0.25)
        self.sock.bind((args.bind_host, int(args.port)))

    def stop(self, *_args: object) -> None:
        self.stop_event.set()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        chunk_path = self.run_dir / "receiver_chunks.csv"
        with chunk_path.open("w", newline="", encoding="utf-8") as chunk_handle:
            chunk_writer = csv.DictWriter(chunk_handle, fieldnames=CHUNK_FIELDS)
            chunk_writer.writeheader()
            start_ns = raw_ns()
            last_recv_ns = start_ns
            while not self.stop_event.is_set():
                if self.stop_file is not None and self.stop_file.exists():
                    break
                if float(self.args.max_duration_s) > 0 and raw_ns() - start_ns > float(self.args.max_duration_s) * 1e9:
                    break
                try:
                    datagram, address = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue
                now = raw_ns()
                last_recv_ns = now
                if len(datagram) < CHUNK_HEADER.size:
                    self.invalid_count += 1
                    continue
                message_id, chunk_index, total_chunks = CHUNK_HEADER.unpack_from(datagram)
                source_ip = address[0]
                key = (source_ip, message_id)
                item = self.partial.setdefault(
                    key,
                    {
                        "first_raw_ns": now,
                        "total_chunks": total_chunks,
                        "chunks": {},
                        "source_ip": source_ip,
                    },
                )
                chunks = item["chunks"]
                assert isinstance(chunks, dict)
                duplicate = chunk_index in chunks
                if duplicate:
                    self.duplicate_count += 1
                else:
                    chunks[chunk_index] = datagram[CHUNK_HEADER.size :]
                ue_guess = (message_id >> 28) - 1
                chunk_writer.writerow(
                    {
                        "recv_raw_ns": now,
                        "recv_wall_ns": time.time_ns(),
                        "source_ip": source_ip,
                        "ue_id": ue_guess,
                        "message_id": message_id,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "udp_payload_bytes": len(datagram),
                        "onwire_bytes": len(datagram) + IP_UDP_OVERHEAD_BYTES,
                        "duplicate": int(duplicate),
                    }
                )
                self.chunk_count += 1
                if self.chunk_count % 100 == 0:
                    chunk_handle.flush()
                if len(chunks) != total_chunks:
                    continue
                try:
                    blob = b"".join(chunks[index] for index in range(total_chunks))
                    header = FRAME_HEADER.unpack_from(blob)
                    magic, version, ue_id, _flags, frame_id, scheduled_ns, payload_bytes, expected = header
                    body = blob[FRAME_HEADER.size : payload_bytes]
                    actual = zlib.crc32(body) & 0xFFFFFFFF
                    identity_ok = ue_id == ue_guess
                    self.frames.append(
                        {
                            "ue_id": ue_id,
                            "frame_id": frame_id,
                            "message_id": message_id,
                            "scheduled_raw_ns": scheduled_ns,
                            "first_chunk_raw_ns": item["first_raw_ns"],
                            "complete_raw_ns": now,
                            "complete_latency_ms": (now - scheduled_ns) / 1e6,
                            "payload_bytes": payload_bytes,
                            "chunks": total_chunks,
                            "onwire_bytes": sum(
                                len(CHUNK_HEADER.pack(message_id, index, total_chunks))
                                + len(chunks[index])
                                + IP_UDP_OVERHEAD_BYTES
                                for index in range(total_chunks)
                            ),
                            "checksum_expected": expected,
                            "checksum_actual": actual,
                            "checksum_ok": int(
                                magic == MAGIC and version == VERSION and expected == actual
                            ),
                            "identity_ok": int(identity_ok),
                            "source_ip": source_ip,
                        }
                    )
                except (KeyError, struct.error, ValueError):
                    self.invalid_count += 1
                self.partial.pop(key, None)

        self.sock.close()
        frame_path = self.run_dir / "receiver_frames.csv"
        with frame_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(self.frames)
        latencies = [float(row["complete_latency_ms"]) for row in self.frames]
        summary = {
            "mode": "receiver",
            "start_raw_ns": start_ns,
            "end_raw_ns": raw_ns(),
            "last_receive_raw_ns": last_recv_ns,
            "chunks": self.chunk_count,
            "complete_frames": len(self.frames),
            "partial_frames": len(self.partial),
            "duplicates": self.duplicate_count,
            "invalid": self.invalid_count,
            "checksum_failures": sum(1 for row in self.frames if not int(row["checksum_ok"])),
            "identity_failures": sum(1 for row in self.frames if not int(row["identity_ok"])),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
        }
        atomic_json(self.run_dir / "receiver_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    receive = sub.add_parser("receive", help="receive and reassemble production-shaped frames")
    receive.add_argument("--bind-host", required=True)
    receive.add_argument("--port", type=int, required=True)
    receive.add_argument("--run-dir", required=True)
    receive.add_argument(
        "--stop-file",
        help="graceful-stop request file checked across sudo/nsenter boundaries",
    )
    receive.add_argument("--socket-receive-buffer-bytes", type=int, default=16 * 1024 * 1024)
    receive.add_argument("--max-duration-s", type=float, default=0.0)

    send = sub.add_parser("send", help="generate open-loop or shielded deterministic demand")
    send.add_argument("--remote-host", required=True)
    send.add_argument("--remote-port", type=int, required=True)
    send.add_argument("--run-dir", required=True)
    send.add_argument("--ue", action="append", default=[], metavar="ID,IP,FRACTION")
    send.add_argument("--kind", choices=SEND_KINDS, required=True)
    send.add_argument("--controller", choices=CONTROLLERS, default="open_loop")
    send.add_argument("--mu-hat-mbps", type=float, required=True)
    send.add_argument("--duration-s", type=float, required=True)
    send.add_argument("--start-delay-s", type=float, default=0.5)
    send.add_argument("--payload-bytes", type=int, default=409600)
    send.add_argument("--chunk-bytes", type=int, default=60000)
    send.add_argument("--tick-s", type=float, default=0.05)
    send.add_argument("--phase", action="append", default=[], metavar="START,END,RHO")
    send.add_argument("--socket-send-buffer-bytes", type=int, default=8 * 1024 * 1024)
    send.add_argument("--send-timeout-s", type=float, default=0.05)
    send.add_argument("--pessimism-factor", type=float, default=0.70)
    send.add_argument("--estimator-window-s", type=float, default=1.0)
    send.add_argument("--estimator-ewma-alpha", type=float, default=0.20)
    send.add_argument("--service-conversion", type=float, default=1.0)
    send.add_argument("--rnti-map", action="append", default=[], metavar="UE_ID,RNTI")
    send.add_argument("--ttracer-csv", default="")
    send.add_argument("--t-messages", default="")
    send.add_argument("--ttracer-port", type=int, default=2023)
    send.add_argument("--demand-seed", type=int, default=0)
    return parser


def validate_send_args(args: argparse.Namespace) -> None:
    """Validate sender semantics before sockets, recorders, or OAI traffic start."""
    ues = TrafficSender._parse_ues(args.ue)
    if float(args.mu_hat_mbps) <= 0 or float(args.duration_s) <= 0:
        raise ValueError("sender service estimate and duration must be positive")
    if float(args.start_delay_s) < 0 or float(args.tick_s) <= 0:
        raise ValueError("sender start delay must be non-negative and tick must be positive")
    if int(args.payload_bytes) < FRAME_HEADER.size:
        raise ValueError("sender payload is too small for the metadata header")
    chunks_per_frame(int(args.payload_bytes), int(args.chunk_bytes))
    if int(args.socket_send_buffer_bytes) <= 0 or float(args.send_timeout_s) <= 0:
        raise ValueError("sender socket buffer and timeout must be positive")

    controlled = str(args.controller) != "open_loop"
    if str(args.kind) == "controlled" and not controlled:
        raise ValueError("controlled traffic requires a non-open-loop controller")
    if str(args.kind) != "controlled" and controlled:
        raise ValueError("non-open-loop controllers require kind=controlled")
    if controlled and (not args.ttracer_csv or not args.t_messages):
        raise ValueError("controlled traffic requires --ttracer-csv and --t-messages")

    mappings: Dict[int, int] = {}
    for raw in args.rnti_map:
        try:
            ue_text, rnti_text = raw.split(",", 1)
            ue_id = int(ue_text)
            rnti = int(rnti_text, 0)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"invalid --rnti-map value: {raw}") from exc
        if ue_id in mappings:
            raise ValueError(f"duplicate RNTI mapping for UE{ue_id}")
        mappings[ue_id] = rnti
    expected_ues = {ue.ue_id for ue in ues}
    if controlled and set(mappings) != expected_ues:
        raise ValueError("controlled traffic RNTI map must cover every configured UE")
    if not controlled and mappings:
        raise ValueError("open-loop traffic must not receive an RNTI map")

    if str(args.kind) == "burst":
        phases = []
        for raw in args.phase:
            try:
                start, end, rho = (float(value) for value in raw.split(","))
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"invalid burst phase: {raw}") from exc
            if start < 0 or end <= start or rho < 0:
                raise ValueError(f"invalid burst phase bounds/rho: {raw}")
            phases.append((start, end, rho))
        if not phases:
            raise ValueError("burst traffic requires at least one --phase")
        phases.sort()
        if abs(phases[0][0]) > 1e-9 or abs(phases[-1][1] - float(args.duration_s)) > 1e-9:
            raise ValueError("burst phases must span the complete sender duration")
        if any(abs(left[1] - right[0]) > 1e-9 for left, right in zip(phases, phases[1:])):
            raise ValueError("burst phases must be contiguous")
    elif args.phase:
        raise ValueError("only burst traffic may define --phase")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "receive":
        return TrafficReceiver(args).run()
    try:
        validate_send_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return TrafficSender(args).run()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent low-overhead host/GPU resource sampler (separate process).

Registered target: one sample every ``--interval-ms`` (100 ms) on an absolute
``CLOCK_MONOTONIC_RAW`` schedule.  A tick that starts more than one interval
late does not run twice: every skipped scheduled instant is counted as a
dropped sample, and the measured inter-sample interval is recorded.

Per sample:

* GPU (NVML, read-only): utilization and memory-controller utilization,
  total/used memory, per-process memory (compute and graphics), per-process
  SM/memory utilization samples, SM/memory clocks, power, temperature,
  P-state;
* host: load averages, ``procs_running``/``procs_blocked`` (run queue), total
  CPU busy fraction;
* per role process (CARLA, UE child, edge, map, OAI gNB/UE/core, T-tracer,
  target-SNR actuator, packet capture, runner): CPU %, RSS, thread count,
  voluntary/involuntary context switches; for the Python roles also per-thread
  CPU deltas (and the container-namespace TID for the edge).

The sampler never signals, renices or otherwise touches another process; it
only reads ``/proc`` and NVML.  Its own CPU time, per-sample cost and every
failure are reported in the footer.  It stops on SIGTERM/SIGINT or when the
watched parent exits, and writes its evidence create-only in batches.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

SCHEMA = "scenesense.single_gpu_profiler.sampler.v1"
_RAW = time.CLOCK_MONOTONIC_RAW
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")

# (role, predicate over (comm, cmdline)); first match wins.  Registered rules.
ROLE_RULES: tuple[tuple[str, Callable[[str, str], bool]], ...] = (
    ("ue", lambda c, a: "splitfusion_single_gpu_profiler_v1.child_v1" in a),
    ("edge", lambda c, a: "splitfusion_single_gpu_profiler_v1.edge_profiled_v1" in a),
    ("map", lambda c, a: "map_profiled_v1.py" in a),
    ("carla", lambda c, a: c.startswith("CarlaUnreal") or "CarlaUnreal-Linux" in a),
    ("oai_gnb", lambda c, a: c == "nr-softmodem"),
    ("oai_ue", lambda c, a: c == "nr-uesoftmodem"),
    ("oai_core", lambda c, a: c in ("oai_amf", "oai_smf", "oai_upf", "oai_nrf", "oai_ausf",
                                    "oai_udm", "oai_udr", "amf", "smf", "upf", "nrf",
                                    "ausf", "udm", "udr", "mysqld")),
    ("tracer", lambda c, a: "/T/tracer/" in a),
    ("target_snr", lambda c, a: "target_snr" in a),
    ("capture", lambda c, a: c == "tcpdump"),
    ("docker_edge_shim", lambda c, a: c == "containerd-shim" and "oai-perception-rx" in a),
)
COMM_PREFILTER = ("python", "Carla", "nr-", "oai_", "amf", "smf", "upf", "nrf", "ausf", "udm",
                  "udr", "mysqld", "multi", "record", "csv", "tcpdump", "containerd-shim")
THREAD_ROLES = ("ue", "edge", "map")


def raw_ns() -> int:
    return time.clock_gettime_ns(_RAW)


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return handle.read().decode(errors="replace")
    except OSError:
        return None


def parse_stat(text: str) -> dict[str, Any]:
    """``/proc/<pid>/stat`` fields after the parenthesised comm."""
    head, _, rest = text.rpartition(")")
    fields = rest.split()
    return {"comm": head.split("(", 1)[1], "state": fields[0], "ppid": int(fields[1]),
            "utime": int(fields[11]), "stime": int(fields[12]),
            "threads": int(fields[17]), "rss_pages": int(fields[21]),
            "processor": int(fields[36]) if len(fields) > 36 else None}


def parse_status_ctx(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            out["vcsw"] = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            out["nvcsw"] = int(line.split()[1])
        elif line.startswith("NSpid:"):
            out["nspid"] = [int(v) for v in line.split()[1:]]
    return out


def classify(comm: str, cmdline: str) -> Optional[str]:
    for role, predicate in ROLE_RULES:
        try:
            if predicate(comm, cmdline):
                return role
        except Exception:  # noqa: BLE001
            continue
    return None


class Discovery:
    """Incremental role discovery: at most ``budget`` new PIDs examined per tick."""

    def __init__(self, *, extra: dict[int, str], budget: int = 400,
                 proc: str = "/proc") -> None:
        self.roles: dict[int, str] = dict(extra)
        self._seen: set[int] = set(extra)
        self._queue: "collections.deque[int]" = collections.deque()
        self._budget = int(budget)
        self._proc = proc
        self.scans = 0

    def step(self) -> None:
        if not self._queue:
            try:
                pids = [int(p) for p in os.listdir(self._proc) if p.isdigit()]
            except OSError:
                return
            self._queue.extend(p for p in pids if p not in self._seen)
            self.scans += 1
        for _ in range(min(self._budget, len(self._queue))):
            pid = self._queue.popleft()
            self._seen.add(pid)
            comm = (_read(f"{self._proc}/{pid}/comm") or "").strip()
            if not comm.startswith(COMM_PREFILTER):
                continue
            cmdline = (_read(f"{self._proc}/{pid}/cmdline") or "").replace("\0", " ")
            role = classify(comm, cmdline)
            if role is not None:
                self.roles[pid] = role

    def forget(self, pid: int) -> None:
        self.roles.pop(pid, None)


class SamplerV1:
    def __init__(self, out: Path, *, interval_ms: float = 100.0, watch_pid: Optional[int],
                 extra_roles: dict[int, str], flush_every: int = 10,
                 nvml_factory: Optional[Callable[[], Any]] = None,
                 opener: Callable[[Path], Any] = lambda p: open(p, "x", encoding="utf-8"),
                 proc: str = "/proc") -> None:
        self.out = Path(out)
        self.interval_ns = int(float(interval_ms) * 1e6)
        self.watch_pid = watch_pid
        self.flush_every = int(flush_every)
        self.discovery = Discovery(extra=extra_roles, proc=proc)
        self._proc = proc
        self._handle = opener(self.out)                 # create-only
        self._batch: list[dict] = []
        self._stop = threading.Event()
        self.counters: "collections.Counter[str]" = collections.Counter()
        self.intervals_ns: list[int] = []
        self.sample_cost_ns: list[int] = []
        self._prev_proc: dict[int, tuple[int, int]] = {}
        self._prev_threads: dict[tuple[int, int], int] = {}
        self._nspid: dict[tuple[int, int], Optional[int]] = {}
        self._prev_cpu: Optional[tuple[int, int]] = None
        self.nvml = None
        self.nvml_error = None
        try:
            if nvml_factory is None:
                from .nvml_v1 import NvmlDeviceV1

                nvml_factory = NvmlDeviceV1
            self.nvml = nvml_factory()
        except Exception as exc:  # noqa: BLE001 - GPU sampling degrades, never aborts
            self.nvml_error = f"{type(exc).__name__}: {exc}"

    # -- collection --------------------------------------------------------------
    def _host(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        load = (_read(f"{self._proc}/loadavg") or "").split()
        if len(load) >= 4:
            running, total = load[3].split("/")
            out.update(load1=float(load[0]), load5=float(load[1]), load15=float(load[2]),
                       runnable_entities=int(running), total_entities=int(total))
        stat = _read(f"{self._proc}/stat") or ""
        for line in stat.splitlines():
            if line.startswith("cpu "):
                values = [int(v) for v in line.split()[1:]]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                total = sum(values[:8])
                if self._prev_cpu is not None:
                    d_total = total - self._prev_cpu[0]
                    d_idle = idle - self._prev_cpu[1]
                    out["cpu_busy_pct"] = (100.0 * (d_total - d_idle) / d_total
                                           if d_total > 0 else None)
                self._prev_cpu = (total, idle)
            elif line.startswith("procs_running"):
                out["procs_running"] = int(line.split()[1])
            elif line.startswith("procs_blocked"):
                out["procs_blocked"] = int(line.split()[1])
        return out

    def _process(self, pid: int, role: str, dt_ns: Optional[int]) -> Optional[dict]:
        text = _read(f"{self._proc}/{pid}/stat")
        if text is None:
            return None
        stat = parse_stat(text)
        ticks = stat["utime"] + stat["stime"]
        row = {"pid": pid, "role": role, "comm": stat["comm"], "threads": stat["threads"],
               "rss_bytes": stat["rss_pages"] * PAGE, "cpu_ticks": ticks,
               "processor": stat["processor"], "state": stat["state"]}
        previous = self._prev_proc.get(pid)
        if previous is not None and dt_ns:
            row["cpu_pct"] = 100.0 * ((ticks - previous[0]) / CLK_TCK) / (dt_ns / 1e9)
        self._prev_proc[pid] = (ticks, 0)
        row.update({k: v for k, v in parse_status_ctx(
            _read(f"{self._proc}/{pid}/status") or "").items() if k in ("vcsw", "nvcsw")})
        if role in THREAD_ROLES:
            row["thread_deltas"] = self._threads(pid, role)
        return row

    def _threads(self, pid: int, role: str) -> list[list]:
        """[tid, comm, delta_ticks, ns_tid] for threads that used CPU since last sample."""
        out = []
        try:
            tids = os.listdir(f"{self._proc}/{pid}/task")
        except OSError:
            return out
        for name in tids:
            tid = int(name)
            text = _read(f"{self._proc}/{pid}/task/{tid}/stat")
            if text is None:
                continue
            stat = parse_stat(text)
            ticks = stat["utime"] + stat["stime"]
            key = (pid, tid)
            previous = self._prev_threads.get(key)
            self._prev_threads[key] = ticks
            if key not in self._nspid:
                nspid = parse_status_ctx(_read(f"{self._proc}/{pid}/task/{tid}/status")
                                         or "").get("nspid") or []
                self._nspid[key] = nspid[-1] if len(nspid) > 1 else None
            delta = ticks - (previous if previous is not None else ticks)
            if previous is None or delta:
                out.append([tid, stat["comm"], delta, self._nspid[key]])
        return out

    def sample(self, scheduled_ns: int, previous_ns: Optional[int]) -> dict[str, Any]:
        started = raw_ns()
        cpu0 = time.process_time_ns()
        record: dict[str, Any] = {"r": "sample", "t_raw_ns": started,
                                  "t_wall_ns": time.time_ns(), "late_ns": started - scheduled_ns}
        dt = None if previous_ns is None else started - previous_ns
        try:
            record["host"] = self._host()
        except Exception as exc:  # noqa: BLE001
            self.counters["host_errors"] += 1
            record["host_error"] = repr(exc)[:120]
        if self.nvml is not None:
            try:
                record["gpu"] = self.nvml.device_sample()
                record["gpu_proc_mem"] = self.nvml.process_memory()
                record["gpu_proc_util"] = self.nvml.process_utilization()
            except Exception as exc:  # noqa: BLE001
                self.counters["nvml_errors"] += 1
                record["gpu_error"] = repr(exc)[:120]
        procs = []
        for pid, role in list(self.discovery.roles.items()):
            try:
                row = self._process(pid, role, dt)
            except Exception:  # noqa: BLE001
                self.counters["proc_errors"] += 1
                row = None
            if row is None:
                self.discovery.forget(pid)
                self._prev_proc.pop(pid, None)
                continue
            procs.append(row)
        record["procs"] = procs
        try:
            self.discovery.step()
        except Exception:  # noqa: BLE001
            self.counters["discovery_errors"] += 1
        record["cost_ns"] = raw_ns() - started
        record["cpu_ns"] = time.process_time_ns() - cpu0
        return record

    # -- loop --------------------------------------------------------------------
    def _emit(self, record: dict) -> None:
        self._batch.append(record)
        if len(self._batch) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        try:
            self._handle.write("".join(json.dumps(r, separators=(",", ":")) + "\n"
                                       for r in self._batch))
            self._handle.flush()
        except Exception:  # noqa: BLE001
            self.counters["write_errors"] += 1
            self.counters["dropped_by_write"] += len(self._batch)
        self._batch = []

    def _parent_alive(self) -> bool:
        if self.watch_pid is None:
            return True
        try:
            os.kill(self.watch_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def stop(self, *_args: Any) -> None:
        self._stop.set()

    def run(self, *, max_samples: Optional[int] = None) -> dict[str, Any]:
        cpu_start = time.process_time_ns()
        start = raw_ns()
        self._emit({"r": "header", "schema": SCHEMA, "pid": os.getpid(), "t_raw_ns": start,
                    "t_wall_ns": time.time_ns(), "interval_ns": self.interval_ns,
                    "clock_domain": "CLOCK_MONOTONIC_RAW", "clk_tck": CLK_TCK,
                    "page_bytes": PAGE, "watch_pid": self.watch_pid,
                    "role_rules": [r for r, _ in ROLE_RULES],
                    "nvml": None if self.nvml is None else {
                        "name": self.nvml.name(), "driver": self.nvml.driver_version()},
                    "nvml_error": self.nvml_error})
        self._flush()
        k = 0
        previous = None
        samples = 0
        while not self._stop.is_set():
            scheduled = start + k * self.interval_ns
            now = raw_ns()
            if now < scheduled:
                self._stop.wait((scheduled - now) / 1e9)
                if self._stop.is_set():
                    break
            record = self.sample(scheduled, previous)
            if previous is not None:
                self.intervals_ns.append(record["t_raw_ns"] - previous)
            previous = record["t_raw_ns"]
            self.sample_cost_ns.append(record["cost_ns"])
            samples += 1
            record["seq"] = samples
            self._emit(record)
            after = raw_ns()
            k_next = k + 1
            behind = (after - (start + k_next * self.interval_ns)) // self.interval_ns
            if behind > 0:                              # scheduled instants missed
                self.counters["dropped_samples"] += int(behind)
                k_next += int(behind)
            k = k_next
            if max_samples is not None and samples >= max_samples:
                break
            if samples % 10 == 0 and not self._parent_alive():
                self.counters["stopped_on_parent_exit"] = 1
                break
        summary = self.summary(samples=samples, cpu_ns=time.process_time_ns() - cpu_start,
                               elapsed_ns=raw_ns() - start)
        self._emit({"r": "footer", **summary})
        self._flush()
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        except Exception:  # noqa: BLE001
            self.counters["close_errors"] += 1
        if self.nvml is not None:
            self.nvml.close()
        return summary

    def summary(self, *, samples: int, cpu_ns: int, elapsed_ns: int) -> dict[str, Any]:
        def stats(values: list[int]) -> dict[str, Any]:
            if not values:
                return {"n": 0}
            ordered = sorted(values)

            def rank(p: float) -> float:
                return ordered[max(1, -(-int(p * len(ordered)) // 100)) - 1] / 1e6
            return {"n": len(values), "p50_ms": rank(50), "p95_ms": rank(95),
                    "p99_ms": rank(99), "max_ms": ordered[-1] / 1e6,
                    "mean_ms": statistics.fmean(values) / 1e6}
        return {"samples": samples, "counters": dict(self.counters),
                "dropped_samples": int(self.counters["dropped_samples"]),
                "interval": stats(self.intervals_ns), "sample_cost": stats(self.sample_cost_ns),
                "sampler_cpu_ns": cpu_ns, "elapsed_ns": elapsed_ns,
                "sampler_cpu_pct_of_one_core": (100.0 * cpu_ns / elapsed_ns
                                                if elapsed_ns else None),
                "discovery_scans": self.discovery.scans,
                "nvml_unsupported": sorted(getattr(self.nvml, "unsupported", ()) or ()),
                "nvml_error": self.nvml_error}


def main(argv: Optional[Iterable[str]] = None) -> int:  # pragma: no cover - process
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--watch-pid", type=int, default=None)
    parser.add_argument("--role", action="append", default=[],
                        help="PID=ROLE (e.g. the runner)")
    parser.add_argument("--ready-file", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    extra = {int(p): r for p, r in (item.split("=", 1) for item in args.role)}
    sampler = SamplerV1(args.out, interval_ms=args.interval_ms, watch_pid=args.watch_pid,
                        extra_roles=extra)
    signal.signal(signal.SIGTERM, sampler.stop)
    signal.signal(signal.SIGINT, sampler.stop)
    if args.ready_file is not None:
        with args.ready_file.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "t_raw_ns": raw_ns()}, handle)
    summary = sampler.run()
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

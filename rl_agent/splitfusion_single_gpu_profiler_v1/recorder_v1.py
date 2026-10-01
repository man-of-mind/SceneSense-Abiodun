"""In-process stage recorder: raw-clock spans, optional CUDA events, batched writes.

Design rules (each is tested):

* **Never on the critical path's semantics.**  Every profiler-internal failure
  is counted and swallowed; an exception raised by the *instrumented* code is
  re-raised unchanged.  Nothing here returns a value the caller consumes.
* **No synchronization.**  CUDA durations come from a pair of
  ``torch.cuda.Event(enable_timing=True)`` recorded on the current stream.  A
  background thread resolves them with the non-blocking ``Event.query()``;
  ``synchronize`` is never called, so the live path is not serialized.  CUDA is
  never initialized by the recorder: if the process has not initialized CUDA
  itself, spans are recorded without GPU timing.
* **Batched, create-only evidence.**  Records are appended to an in-memory
  deque and written by one flush thread in batches to a file opened with
  ``"x"``.  A full buffer or a failed write drops records and counts them.
* **One clock.**  Span boundaries are ``CLOCK_MONOTONIC_RAW`` nanoseconds.
  Periodic clock probes (raw, monotonic, realtime) let the analyzer verify
  that every process on the host shares the raw clock.

Record kinds (``"r"``): ``header``, ``span``, ``mark``, ``cuda``, ``gauge``,
``clock``, ``thread``, ``footer``.
"""

from __future__ import annotations

import atexit
import collections
import itertools
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

SCHEMA = "scenesense.single_gpu_profiler.record.v1"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
_RAW = time.CLOCK_MONOTONIC_RAW


def raw_ns() -> int:
    return time.clock_gettime_ns(_RAW)


def clock_probe() -> dict[str, int]:
    """Bracketed (raw, monotonic, realtime) triple; ``raw_ns`` is the midpoint."""
    a = time.clock_gettime_ns(_RAW)
    mono = time.monotonic_ns()
    wall = time.time_ns()
    b = time.clock_gettime_ns(_RAW)
    return {"raw_ns": (a + b) // 2, "mono_ns": mono, "wall_ns": wall,
            "bracket_ns": b - a}


def boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


IDENTITY_KEYS = ("frame_id", "tensor_seq", "session_uuid", "decision_seq", "ticket_seq",
                 "reward_requested", "frame_kind", "mode_id", "q_e4")


class _NullSpan:
    __slots__ = ()

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def annotate(self, **_fields: Any) -> None:
        return None


_NULL_SPAN = _NullSpan()


class NullRecorder:
    """Disabled profiler: every call is a no-op and nothing is written."""

    enabled = False
    role = "disabled"

    def span(self, stage: str, **_fields: Any) -> _NullSpan:
        return _NULL_SPAN

    def emit_span(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def mark(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def bind(self, **_fields: Any) -> None:
        return None

    def unbind(self) -> None:
        return None

    def bound(self) -> bool:
        return False

    def context(self) -> dict[str, Any]:
        return {}

    def register_gauge(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_phase(self, _phase: str) -> None:
        return None

    def note(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self, **_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False}


NULL = NullRecorder()
_ACTIVE: list[Any] = [NULL]


def active() -> Any:
    return _ACTIVE[0]


def activate(recorder: Any) -> Any:
    _ACTIVE[0] = recorder
    return recorder


def deactivate() -> None:
    _ACTIVE[0] = NULL


class _Span:
    __slots__ = ("_rec", "_stage", "_cuda", "_fields", "_seq", "_parent", "_start",
                 "_ev", "_stack")

    def __init__(self, rec: "ProfileRecorderV1", stage: str, cuda: bool,
                 fields: Mapping[str, Any]) -> None:
        self._rec, self._stage, self._cuda, self._fields = rec, stage, cuda, fields
        self._ev = None
        self._stack = None
        self._seq = None

    def annotate(self, **fields: Any) -> None:
        """Attach result fields (e.g. the decision) before the span closes."""
        try:
            self._fields.update(fields)
        except Exception:  # noqa: BLE001
            self._rec._internal_error()

    def __enter__(self) -> "_Span":
        rec = self._rec
        t0 = time.perf_counter_ns()
        try:
            local = rec._thread_state()
            stack = local.stack
            self._seq = next(rec._seq)
            self._parent = stack[-1] if stack else None
            stack.append(self._seq)
            self._stack = stack
            if self._cuda:
                self._ev = rec._cuda_begin()
        except Exception:  # noqa: BLE001 - profiler faults never propagate
            rec._internal_error()
        self._start = time.clock_gettime_ns(_RAW)
        rec._charge(time.perf_counter_ns() - t0)
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> bool:
        end = time.clock_gettime_ns(_RAW)
        rec = self._rec
        t0 = time.perf_counter_ns()
        try:
            if self._seq is not None:
                stack = self._stack
                if stack and stack[-1] == self._seq:
                    stack.pop()
                else:                                # unbalanced use: record, never raise
                    rec._internal_error("unbalanced_span")
                    if self._seq in stack:
                        stack.remove(self._seq)
                record = {"r": "span", "seq": self._seq, "stage": self._stage,
                          "tid": threading.get_native_id(), "start_raw_ns": self._start,
                          "end_raw_ns": end, "parent": self._parent,
                          "depth": len(stack), "ok": exc_type is None,
                          "ph": rec._phase}
                ctx = rec._thread_state().ctx
                if ctx:
                    record.update(ctx)
                if self._fields:
                    record.update(self._fields)
                if self._ev is not None:
                    record["cuda"] = rec._cuda_end(self._seq, self._ev)
                rec._push(record)
        except Exception:  # noqa: BLE001
            rec._internal_error()
        rec._charge(time.perf_counter_ns() - t0)
        return False


class ProfileRecorderV1:
    """One per process.  See the module docstring for the guarantees."""

    enabled = True

    def __init__(self, path: Path, *, role: str, run_label: str = "",
                 identity: Optional[Mapping[str, Any]] = None,
                 flush_interval_s: float = 0.2, gauge_interval_s: float = 0.1,
                 clock_probe_interval_s: float = 1.0, cuda_timeout_s: float = 10.0,
                 max_buffered_records: int = 500_000,
                 opener: Callable[[Path], Any] = lambda p: open(p, "x", encoding="utf-8"),
                 cuda_events: Optional[Callable[[], Any]] = None) -> None:
        self.path = Path(path)
        self.role = str(role)
        self.run_label = str(run_label)
        self._identity = dict(identity or {})
        self._flush_s = float(flush_interval_s)
        self._gauge_s = float(gauge_interval_s)
        self._clock_s = float(clock_probe_interval_s)
        self._cuda_timeout_ns = int(float(cuda_timeout_s) * 1e9)
        self._max = int(max_buffered_records)
        self._opener = opener
        self._cuda_factory = cuda_events          # tests inject fakes; None = torch
        self._buffer: "collections.deque[dict]" = collections.deque()
        self._seq = itertools.count(1)
        self._local = threading.local()
        self._local_states: list[Any] = []
        self._local_lock = threading.Lock()
        self._gauges: dict[str, Callable[[], Any]] = {}
        self._gauge_errors: "collections.Counter[str]" = collections.Counter()
        self._pending_cuda: "collections.deque[tuple]" = collections.deque()
        self._event_pool: "collections.deque[Any]" = collections.deque()
        self._phase = "setup"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._closed = False
        self._close_lock = threading.Lock()
        self.counters: "collections.Counter[str]" = collections.Counter()
        self._writer_cpu_ns = 0
        self._bytes = 0
        self._cuda_state: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "ProfileRecorderV1":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._opener(self.path)       # create-only: raises if present
        probe = clock_probe()
        header = {"r": "header", "schema": SCHEMA, "role": self.role,
                  "run_label": self.run_label, "pid": os.getpid(), "ppid": os.getppid(),
                  "host": socket.gethostname(), "boot_id": boot_id(),
                  "clock_domain": CLOCK_DOMAIN, "clock": probe,
                  "argv": list(sys.argv), "python": sys.version.split()[0],
                  "identity": self._identity,
                  "config": {"flush_interval_s": self._flush_s,
                             "gauge_interval_s": self._gauge_s,
                             "clock_probe_interval_s": self._clock_s,
                             "cuda_timeout_s": self._cuda_timeout_ns / 1e9,
                             "max_buffered_records": self._max}}
        self._write_lines([header])
        self._thread = threading.Thread(target=self._run, name="profiler-flush", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        return self

    def close(self, *, cuda_drain_s: float = 2.0) -> dict[str, Any]:
        with self._close_lock:
            if self._closed:
                return self.summary()
            self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        deadline = time.monotonic() + float(cuda_drain_s)
        while self._pending_cuda and time.monotonic() < deadline:
            self._resolve_cuda(final=False)
            if self._pending_cuda:
                time.sleep(0.01)
        self._resolve_cuda(final=True)
        self._flush()
        summary = self.summary()
        self._write_lines([{"r": "footer", "t_raw_ns": raw_ns(), **summary}])
        try:
            if self._handle is not None:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()
        except Exception:  # noqa: BLE001
            self.counters["close_errors"] += 1
        self._handle = None
        return summary

    def summary(self) -> dict[str, Any]:
        with self._local_lock:
            charged = sum(state.charged_ns for state in self._local_states)
            charged_calls = sum(state.charged_calls for state in self._local_states)
        return {"role": self.role, "pid": os.getpid(), "counters": dict(self.counters),
                "gauge_errors": dict(self._gauge_errors),
                "hot_path_ns_total": charged, "hot_path_calls": charged_calls,
                "writer_cpu_ns": self._writer_cpu_ns, "bytes_written": self._bytes,
                "cuda_state": self._cuda_state, "threads_seen": len(self._local_states),
                "phase": self._phase}

    # -- thread state ------------------------------------------------------------
    def _thread_state(self) -> Any:
        local = self._local
        if not hasattr(local, "stack"):
            local.stack = []
            local.ctx = {}
            local.charged_ns = 0
            local.charged_calls = 0
            with self._local_lock:
                self._local_states.append(local)
            self._push({"r": "thread", "tid": threading.get_native_id(),
                        "name": threading.current_thread().name, "t_raw_ns": raw_ns()})
        return local

    def _charge(self, ns: int) -> None:
        local = self._local
        try:
            local.charged_ns += ns
            local.charged_calls += 1
        except AttributeError:
            pass

    def _internal_error(self, name: str = "internal_errors") -> None:
        self.counters[name] += 1

    # -- public API ----------------------------------------------------------------
    def span(self, stage: str, *, cuda: bool = False, **fields: Any) -> _Span:
        return _Span(self, stage, cuda, fields)

    def emit_span(self, stage: str, start_raw_ns: Optional[int], end_raw_ns: Optional[int],
                  **fields: Any) -> None:
        """A span from authoritative stamps already taken by the instrumented code."""
        t0 = time.perf_counter_ns()
        try:
            if start_raw_ns is None or end_raw_ns is None:
                self.counters["derived_span_missing_stamp"] += 1
                return
            local = self._thread_state()
            record = {"r": "span", "seq": next(self._seq), "stage": stage,
                      "tid": threading.get_native_id(), "start_raw_ns": int(start_raw_ns),
                      "end_raw_ns": int(end_raw_ns),
                      "parent": local.stack[-1] if local.stack else None,
                      "depth": len(local.stack), "ok": True, "derived": True,
                      "ph": self._phase}
            if local.ctx:
                record.update(local.ctx)
            record.update(fields)
            self._push(record)
        except Exception:  # noqa: BLE001
            self._internal_error()
        self._charge(time.perf_counter_ns() - t0)

    def mark(self, event: str, *, t_raw_ns: Optional[int] = None, **fields: Any) -> None:
        t0 = time.perf_counter_ns()
        try:
            local = self._thread_state()
            record = {"r": "mark", "seq": next(self._seq), "stage": event,
                      "tid": threading.get_native_id(),
                      "t_raw_ns": raw_ns() if t_raw_ns is None else int(t_raw_ns),
                      "ph": self._phase}
            if local.ctx:
                record.update(local.ctx)
            record.update(fields)
            self._push(record)
        except Exception:  # noqa: BLE001
            self._internal_error()
        self._charge(time.perf_counter_ns() - t0)

    def bind(self, **fields: Any) -> None:
        """Merge identity fields into this thread's frame context."""
        try:
            ctx = self._thread_state().ctx
            for key, value in fields.items():
                if value is not None:
                    ctx[key] = value
        except Exception:  # noqa: BLE001
            self._internal_error()

    def unbind(self) -> None:
        try:
            self._thread_state().ctx = {}
        except Exception:  # noqa: BLE001
            self._internal_error()

    def bound(self) -> bool:
        local = self._local
        return bool(getattr(local, "ctx", None))

    def context(self) -> dict[str, Any]:
        return dict(getattr(self._local, "ctx", None) or {})

    def register_gauge(self, name: str, fn: Callable[[], Any]) -> None:
        self._gauges[str(name)] = fn

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)
        self.mark("phase", phase=str(phase))

    def note(self, kind: str, **fields: Any) -> None:
        """A free-form, non-timing evidence record (e.g. a runtime identity)."""
        try:
            self._push({"r": "note", "seq": next(self._seq), "kind": str(kind),
                        "t_raw_ns": raw_ns(), **fields})
        except Exception:  # noqa: BLE001
            self._internal_error()

    # -- buffer ------------------------------------------------------------------
    def _push(self, record: dict) -> None:
        if self.counters["write_failed"] or len(self._buffer) >= self._max:
            self.counters["dropped_records"] += 1
            return
        self._buffer.append(record)
        self.counters["records"] += 1

    # -- CUDA --------------------------------------------------------------------
    def _cuda_ready(self) -> bool:
        if self._cuda_factory is not None:
            self._cuda_state = "INJECTED"
            return True
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            self._cuda_state = self._cuda_state or "NOT_INITIALIZED_BY_PROCESS"
            return False
        self._cuda_state = "TORCH_CUDA_EVENTS"
        return True

    def _new_event(self) -> Any:
        try:
            return self._event_pool.pop()
        except IndexError:
            if self._cuda_factory is not None:
                return self._cuda_factory()
            import torch

            self.counters["cuda_events_created"] += 1
            return torch.cuda.Event(enable_timing=True)

    def _cuda_begin(self) -> Optional[tuple]:
        if not self._cuda_ready():
            self.counters["cuda_spans_without_events"] += 1
            return None
        start = self._new_event()
        start.record()
        return (start,)

    def _cuda_end(self, seq: int, pair: tuple) -> Optional[str]:
        end = self._new_event()
        end.record()
        self._pending_cuda.append((seq, pair[0], end, raw_ns()))
        self.counters["cuda_spans"] += 1
        return "PENDING"

    def _resolve_cuda(self, *, final: bool) -> None:
        now = raw_ns()
        remaining = collections.deque()
        resolved = []
        while self._pending_cuda:
            seq, start, end, recorded = self._pending_cuda.popleft()
            try:
                done = bool(end.query())
            except Exception:  # noqa: BLE001
                self.counters["cuda_query_errors"] += 1
                resolved.append({"r": "cuda", "span_seq": seq, "gpu_ms": None,
                                 "status": "QUERY_ERROR"})
                continue
            if done:
                try:
                    gpu_ms = float(start.elapsed_time(end))
                    status = "RESOLVED"
                except Exception:  # noqa: BLE001
                    gpu_ms, status = None, "ELAPSED_ERROR"
                    self.counters["cuda_elapsed_errors"] += 1
                resolved.append({"r": "cuda", "span_seq": seq, "gpu_ms": gpu_ms,
                                 "status": status, "resolved_raw_ns": now})
                self.counters["cuda_resolved"] += 1
                self._event_pool.append(start)
                self._event_pool.append(end)
            elif final or now - recorded > self._cuda_timeout_ns:
                resolved.append({"r": "cuda", "span_seq": seq, "gpu_ms": None,
                                 "status": "UNRESOLVED_AT_CLOSE" if final
                                 else "UNRESOLVED_TIMEOUT"})
                self.counters["cuda_unresolved"] += 1
            else:
                remaining.append((seq, start, end, recorded))
        self._pending_cuda.extendleft(reversed(remaining))
        for record in resolved:
            self._push(record)

    # -- flush thread ------------------------------------------------------------
    def _sample_gauges(self) -> None:
        values = {}
        for name, fn in list(self._gauges.items()):
            try:
                values[name] = fn()
            except Exception:  # noqa: BLE001 - a broken gauge never stops the run
                self._gauge_errors[name] += 1
        if values:
            self._push({"r": "gauge", "t_raw_ns": raw_ns(), "values": values})

    def _run(self) -> None:
        next_flush = next_gauge = next_clock = time.monotonic()
        while not self._stop.is_set():
            cpu0 = time.thread_time_ns()
            now = time.monotonic()
            try:
                if now >= next_gauge:
                    self._sample_gauges()
                    next_gauge = now + self._gauge_s
                if now >= next_clock:
                    self._push({"r": "clock", **clock_probe()})
                    next_clock = now + self._clock_s
                if self._pending_cuda:
                    self._resolve_cuda(final=False)
                if now >= next_flush:
                    self._flush()
                    next_flush = now + self._flush_s
            except Exception:  # noqa: BLE001
                self._internal_error("flush_loop_errors")
            self._writer_cpu_ns += time.thread_time_ns() - cpu0
            self._stop.wait(0.02)

    def _flush(self) -> None:
        batch = []
        while True:
            try:
                batch.append(self._buffer.popleft())
            except IndexError:
                break
        if batch:
            self._write_lines(batch)

    def _write_lines(self, batch: list) -> None:
        if self.counters["write_failed"] or self._handle is None:
            if batch and batch[0].get("r") != "footer":
                self.counters["dropped_records"] += len(batch)
            return
        try:
            text = "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n"
                           for r in batch)
            self._handle.write(text)
            self._handle.flush()
            self._bytes += len(text)
            self.counters["batches_written"] += 1
        except Exception:  # noqa: BLE001 - e.g. disk full: stop writing, keep running
            self.counters["write_failed"] = 1
            self.counters["dropped_records"] += len(batch)


def iter_records(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

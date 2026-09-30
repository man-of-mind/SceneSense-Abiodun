"""Addendum 7: reward-priority object-GT scheduling with per-ticket instrumentation.

:class:`RewardPriorityGtQueueV2` replaces the pinned collector's single
undifferentiated object-GT FIFO, exposing the same ``queue.Queue`` surface the
unchanged ``_evaluation_worker`` uses (``put_nowait``, ``get(timeout)``,
``task_done``, ``empty``, ``unfinished_tasks``, ``None`` sentinel). There are
two bounded classes:

* ``HIGH``: reward-requested policy frames, whose GT the edge evaluator needs;
* ``LOW``: non-reward holds and fallbacks, used only for report metrics.

HIGH is always selected before queued LOW, FIFO holds within each class, and
LOW runs whenever HIGH is empty. Overflow, duplicate or foreign identity, and
unclassifiable items fail explicitly (:class:`GtQueueError`, never
``queue.Full``, so the adapter can never silently drop reward GT). A LOW item
already executing is not pre-empted: a HIGH item arriving mid-LOW waits at most
one LOW computation (the bounded residual). Evaluator-instrumentation
scheduling only; edge latest-only semantics are untouched.

:class:`GtTicketLogV2` records, per ticket: enqueue time, class, worker start,
queue wait, refresh-static time if invoked, object-row construction start/end,
objects-file write start/end, object count, output size/SHA-256/identity, and
completion.

Importing this module performs no I/O.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Mapping, Optional

HIGH, LOW = "HIGH", "LOW"


class GtQueueError(RuntimeError):
    """Explicit object-GT scheduling failure (overflow, duplicate, foreign, missing)."""


class RewardPriorityGtQueueV2:
    def __init__(self, *, classify: Callable[[Mapping[str, Any]], str],
                 maxsize_high: int = 64, maxsize_low: int = 64,
                 clock: Callable[[], int] = time.time_ns,
                 on_enqueue: Optional[Callable[[int, str, int], None]] = None,
                 on_dequeue: Optional[Callable[[int, str, int], None]] = None) -> None:
        self._classify = classify
        self._max = {HIGH: int(maxsize_high), LOW: int(maxsize_low)}
        self._queues: dict[str, collections.deque] = {HIGH: collections.deque(),
                                                      LOW: collections.deque()}
        self._clock = clock
        self._on_enqueue, self._on_dequeue = on_enqueue, on_dequeue
        self._seen: set[int] = set()
        self._sentinel = False
        self._cond = threading.Condition()
        self.unfinished_tasks = 0
        self.counters = collections.Counter()

    def put_nowait(self, item: Optional[Mapping[str, Any]]) -> None:
        with self._cond:
            if item is None:
                self._sentinel = True
                self._cond.notify_all()
                return
            frame_id = int(item["frame_id"])
            if frame_id in self._seen:
                self.counters["duplicate_refused"] += 1
                raise GtQueueError(f"duplicate object-GT ticket for frame {frame_id}")
            klass = self._classify(item)
            if klass not in (HIGH, LOW):
                self.counters["foreign_refused"] += 1
                raise GtQueueError(f"object-GT ticket for frame {frame_id} is unclassifiable")
            if len(self._queues[klass]) >= self._max[klass]:
                self.counters[f"overflow_{klass}"] += 1
                raise GtQueueError(f"object-GT {klass} queue overflow at frame {frame_id}")
            self._seen.add(frame_id)
            now = self._clock()
            self._queues[klass].append((item, klass, now))
            self.unfinished_tasks += 1
            self.counters[f"enqueued_{klass}"] += 1
            self._cond.notify_all()
        if self._on_enqueue is not None:
            self._on_enqueue(frame_id, klass, now)

    put = put_nowait

    def _pop_locked(self):
        for klass in (HIGH, LOW):
            if self._queues[klass]:
                return self._queues[klass].popleft()
        return None

    def get(self, block: bool = True, timeout: Optional[float] = None):
        import queue as _queue

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cond:
            while True:
                entry = self._pop_locked()
                if entry is not None:
                    break
                if self._sentinel:
                    self._sentinel = False
                    self.unfinished_tasks += 1          # the sentinel is task_done'd
                    return None
                if not block:
                    raise _queue.Empty
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise _queue.Empty
                self._cond.wait(timeout=remaining)
        item, klass, _enqueued = entry
        started = self._clock()
        self.counters[f"dequeued_{klass}"] += 1
        if self._on_dequeue is not None:
            self._on_dequeue(int(item["frame_id"]), klass, started)
        return item

    def get_nowait(self):
        return self.get(block=False)

    def task_done(self) -> None:
        with self._cond:
            if self.unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self.unfinished_tasks -= 1

    def empty(self) -> bool:
        with self._cond:
            return not self._queues[HIGH] and not self._queues[LOW]

    def qsize(self) -> int:
        with self._cond:
            return len(self._queues[HIGH]) + len(self._queues[LOW])

    def depth(self) -> dict[str, int]:
        with self._cond:
            return {HIGH: len(self._queues[HIGH]), LOW: len(self._queues[LOW])}


class GtTicketLogV2:
    """Thread-safe per-ticket object-GT timeline (all wall-clock ns)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tickets: dict[int, dict[str, Any]] = {}
        self.refresh_calls: list[dict[str, Any]] = []
        self._current: Optional[int] = None

    def _row(self, frame_id: int) -> dict[str, Any]:
        return self.tickets.setdefault(int(frame_id), {"frame_id": int(frame_id)})

    def enqueued(self, frame_id: int, klass: str, at: int) -> None:
        with self._lock:
            self._row(frame_id).update(enqueue_wall_ns=int(at), queue_class=klass)

    def dequeued(self, frame_id: int, klass: str, at: int) -> None:
        with self._lock:
            row = self._row(frame_id)
            row["worker_start_wall_ns"] = int(at)
            if row.get("enqueue_wall_ns") is not None:
                row["queue_wait_ms"] = (int(at) - row["enqueue_wall_ns"]) / 1e6
            self._current = int(frame_id)

    def refresh(self, started: int, ended: int, refreshed: bool) -> None:
        with self._lock:
            entry = {"start_wall_ns": started, "end_wall_ns": ended,
                     "elapsed_ms": (ended - started) / 1e6, "refreshed": bool(refreshed),
                     "frame_id": self._current}
            self.refresh_calls.append(entry)
            if self._current is not None:
                self._row(self._current)["refresh_static"] = entry

    def rows_started(self, frame_id: int, at: int) -> None:
        with self._lock:
            self._row(frame_id)["object_rows_start_wall_ns"] = int(at)

    def write_started(self, frame_id: int, at: int) -> None:
        with self._lock:
            row = self._row(frame_id)
            row["objects_write_start_wall_ns"] = int(at)
            row.setdefault("object_rows_end_wall_ns", int(at))

    def write_finished(self, frame_id: int, at: int, *, object_count: int,
                       size_bytes: Optional[int], sha256: Optional[str],
                       identity: Mapping[str, Any]) -> None:
        with self._lock:
            self._row(frame_id).update(objects_write_end_wall_ns=int(at),
                                       object_count=int(object_count),
                                       output_size_bytes=size_bytes, output_sha256=sha256,
                                       output_identity=dict(identity))

    def completed(self, frame_id: int, at: int, *, error: Optional[str] = None) -> None:
        with self._lock:
            row = self._row(frame_id)
            row["completion_wall_ns"] = int(at)
            if row.get("object_rows_start_wall_ns") and row.get("object_rows_end_wall_ns"):
                row["object_rows_ms"] = (row["object_rows_end_wall_ns"]
                                         - row["object_rows_start_wall_ns"]) / 1e6
            if error:
                row["error"] = error
            if self._current == int(frame_id):
                self._current = None

    def missing_high_outputs(self) -> list[int]:
        with self._lock:
            return sorted(f for f, r in self.tickets.items()
                          if r.get("queue_class") == HIGH and not r.get("output_sha256"))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"tickets": [dict(r) for _f, r in sorted(self.tickets.items())],
                    "refresh_calls": list(self.refresh_calls)}


def timed_startup_refresh(source: Any, clock: Callable[[], int] = time.time_ns) -> dict[str, Any]:
    """``refresh_static(force=True)`` before any frame; failure refuses admission."""
    started = clock()
    try:
        source.refresh_static(force=True)
    except Exception as exc:
        raise GtQueueError(f"pre-route refresh_static failed: {type(exc).__name__}: {exc}") from exc
    ended = clock()
    return {"start_wall_ns": started, "end_wall_ns": ended,
            "elapsed_ms": (ended - started) / 1e6, "completed": True}


def install_timed_refresh(source: Any, log: GtTicketLogV2,
                          clock: Callable[[], int] = time.time_ns) -> None:
    """Record every per-ticket ``refresh_static`` call (and whether it refreshed)."""
    original = source.refresh_static

    def timed_refresh(*, force: bool = False) -> None:
        before = getattr(source, "_refreshed_at", None)
        begin = clock()
        original(force=force)
        log.refresh(begin, clock(), getattr(source, "_refreshed_at", None) != before)

    source.refresh_static = timed_refresh


__all__ = ["HIGH", "LOW", "GtQueueError", "RewardPriorityGtQueueV2", "GtTicketLogV2",
           "timed_startup_refresh", "install_timed_refresh"]

"""Give the latency-critical direct edge->map threads their own CPUs.

The retained validation showed a publication interval whose *work* is at most
2.3 ms on an idle host but whose live p99 is 58-107 ms, scattered through the
cell, independent of payload size and uncorrelated with the map thread being
busy. That is a run-queue signature, not a work signature: the thread is
runnable and not running.

The OAI launcher is SHA-256 pinned and the radio's timing must not be
perturbed, so nothing here repins the softmodem or CARLA. What it does is
narrow: it lets the heavy edge threads and the latency-critical threads be
placed on disjoint CPU sets so they at least stop queueing behind each other,
and it records exactly what was applied so a run can be read without guessing.

A reservation is always advisory. ``describe_reservation`` reports the
requested and the effective set, and an unavailable CPU is reported rather than
silently dropped.
"""

from __future__ import annotations

import os
import threading
from typing import Any


def parse_cpu_set(value: str) -> tuple[int, ...]:
    """Parse ``"2,3"`` or ``"2-5"`` or ``""`` into a sorted CPU tuple."""

    text = str(value or "").strip()
    if not text:
        return ()
    cpus: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = piece.split("-", 1)
            start, stop = int(low), int(high)
            if stop < start:
                raise ValueError(f"inverted CPU range: {piece!r}")
            cpus.update(range(start, stop + 1))
        else:
            cpus.add(int(piece))
    if any(cpu < 0 for cpu in cpus):
        raise ValueError(f"negative CPU in set: {value!r}")
    return tuple(sorted(cpus))


def apply_thread_reservation(value: str, *, label: str) -> dict[str, Any]:
    """Pin the calling thread; never fail the cell over a placement request."""

    requested = parse_cpu_set(value)
    record: dict[str, Any] = {
        "label": label,
        "thread": threading.current_thread().name,
        "requested_cpus": list(requested),
        "applied": False,
        "effective_cpus": [],
        "unavailable_cpus": [],
        "error": "",
    }
    available = set(os.sched_getaffinity(0))
    if not requested:
        record["effective_cpus"] = sorted(available)
        return record
    record["unavailable_cpus"] = sorted(set(requested) - available)
    usable = sorted(set(requested) & available)
    if not usable:
        record["error"] = "no requested CPU is available to this process"
        record["effective_cpus"] = sorted(available)
        return record
    try:
        os.sched_setaffinity(0, set(usable))
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["effective_cpus"] = sorted(os.sched_getaffinity(0))
        return record
    record["applied"] = True
    record["effective_cpus"] = sorted(os.sched_getaffinity(0))
    return record


def describe_reservation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize applied reservations for the run manifest."""

    return {
        "schema": "splitfusion_direct_cpu_reservation.v1",
        "host_cpu_count": os.cpu_count(),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "threads": list(records),
        "all_requested_applied": all(
            record["applied"] or not record["requested_cpus"] for record in records
        ),
    }

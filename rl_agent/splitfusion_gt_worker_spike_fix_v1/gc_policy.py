"""Bounded, reversible CPython cyclic-GC policy for the GT prep worker process.

Off by default. ``BoundedGcPolicy.install()`` changes only this process's ``gc``
module state; ``uninstall()`` restores enabled flag, thresholds and frozen set.

Mechanism (measured, see experiments/splitfusion_gt_worker_spike_fix_v1/):
* automatic gen0/gen1 collections stay ON with the interpreter defaults;
* the automatic full (gen2) collection is moved out of requests by raising
  threshold2 to ``hard_cap_gen1`` (a HARD CAP, not "disabled": if idle
  maintenance never runs, CPython still forces a full collection after that
  many gen1 collections, so uncollected growth is bounded);
* after ``warmup_decisions`` completed decisions, in an idle window, one young-only
  ``gc.collect(1)`` then ``gc.freeze()`` moves the long-lived import/warm-up heap
  (torch, carla, numpy, producer state) to the permanent generation so later full
  collections only traverse objects allocated since then. No full collection
  precedes the freeze: v1 (full collect + freeze) cost ~28 ms once and delayed the
  next request in 4/4 sessions; replayed full collections in the control worker
  reclaimed 0-20 objects, so the frozen-garbage leak is negligible and one-time;
* at every idle point (after a reply, no request pending, op in ``IDLE_OPS``)
  a full ``gc.collect(2)`` runs when at least ``idle_min_gen1`` gen1 collections
  accumulated since the last full one. Its duration is measured and recorded.
Nothing here changes what any request computes.
"""
from __future__ import annotations

import gc
import os
import select
import time

# Ops after which the worker is normally idle until the next decision's FREEZE.
IDLE_OPS = frozenset(("EVIDENCE_DURABLE", "UNSCORED_DURABLE", "CANCEL_UNBOUND", "CANCEL_BOUND", "VOID"))


def rss_bytes():
    with open("/proc/self/statm") as stream:
        return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class BoundedGcPolicy:
    def __init__(self, *, warmup_decisions=2, idle_min_gen1=1, hard_cap_gen1=200, idle_ops=IDLE_OPS):
        if not (warmup_decisions >= 1 and idle_min_gen1 >= 0 and hard_cap_gen1 >= 10):
            raise ValueError("invalid GC policy bounds")
        self.warmup_decisions, self.idle_min_gen1, self.hard_cap_gen1 = warmup_decisions, idle_min_gen1, hard_cap_gen1
        self.idle_ops = frozenset(idle_ops)
        self.installed = False
        self.saved = None
        self.decisions = 0
        self.frozen = False
        self.events = []          # idle maintenance records (bounded by run length; small dicts)

    # ------------------------------------------------------------- install / uninstall
    def install(self):
        if self.installed:
            raise RuntimeError("policy already installed")
        self.saved = dict(enabled=gc.isenabled(), threshold=gc.get_threshold(), freeze_count=gc.get_freeze_count())
        if self.saved["freeze_count"]:
            raise RuntimeError("process already has a frozen GC generation; refusing to own it")
        t0, t1, _ = self.saved["threshold"]
        gc.set_threshold(t0, t1, self.hard_cap_gen1)
        gc.enable()
        self.installed = True
        return self

    def uninstall(self):
        if not self.installed:
            return
        gc.unfreeze()
        gc.set_threshold(*self.saved["threshold"])
        (gc.enable if self.saved["enabled"] else gc.disable)()
        self.installed = False
        self.frozen = False

    # ------------------------------------------------------------- idle maintenance
    def after_reply(self, op, sock=None):
        """Called after a reply has been sent. Runs bounded maintenance only when idle."""
        if not self.installed or op not in self.idle_ops:
            return None
        self.decisions += 1
        pending = bool(sock is not None and select.select([sock], [], [], 0)[0])
        if pending:
            self.events.append(dict(kind="skipped_pending", op=op, decision=self.decisions,
                                    count=list(gc.get_count())))
            return None
        if not self.frozen and self.decisions >= self.warmup_decisions:
            return self._run("warmup_collect_freeze", op, freeze=True)
        if self.frozen and gc.get_count()[2] >= self.idle_min_gen1:
            return self._run("idle_full_collect", op)
        return None

    def _run(self, kind, op, *, freeze=False):
        count_before = list(gc.get_count())
        rss0 = rss_bytes()
        w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        collected = gc.collect(1 if freeze else 2)
        if freeze:
            gc.freeze()
            self.frozen = True
        w1, c1 = time.perf_counter_ns(), time.thread_time_ns()
        late = bool(self._sock_readable())
        record = dict(kind=kind, op=op, decision=self.decisions, wall_ms=(w1 - w0) / 1e6, cpu_ms=(c1 - c0) / 1e6,
                      collected=collected, count_before=count_before, freeze_count=gc.get_freeze_count(),
                      rss_before=rss0, rss_after=rss_bytes(), start_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                      - (w1 - w0), request_arrived_during=late)
        self.events.append(record)
        return record

    _sock = None

    def _sock_readable(self):
        return self._sock is not None and select.select([self._sock], [], [], 0)[0]

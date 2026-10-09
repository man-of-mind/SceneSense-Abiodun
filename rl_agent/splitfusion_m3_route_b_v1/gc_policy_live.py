"""Registered safety stops around the unchanged ``BoundedGcPolicy`` (candidate prep worker only).

Safety stops are NOT performance criteria. On a breach the policy uninstalls itself (stock CPython GC restored,
frozen generation released), a SAFETY_STOP event is recorded, and the worker continues unchanged; the arm is then a
qualification REJECT. Events are bounded by the run length; overflow is counted.
"""
import time

from rl_agent.splitfusion_gt_worker_spike_fix_v1.gc_policy import BoundedGcPolicy, rss_bytes


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class SafeBoundedGcPolicy(BoundedGcPolicy):
    def __init__(self, *, policy, safety, max_events, rss=rss_bytes):
        super().__init__(warmup_decisions=policy["warmup_decisions"], idle_min_gen1=policy["idle_min_gen1"],
                         hard_cap_gen1=policy["gen2_threshold"], idle_ops=policy["idle_ops"])
        self.safety, self.max_events, self._rss = dict(safety), int(max_events), rss
        self.rss_at_freeze = None
        self.safety_stop = None
        self.events_dropped = 0

    def _record(self, record):
        if len(self.events) < self.max_events:
            self.events.append(record)
        else:
            self.events_dropped += 1

    def _stop(self, reason, **detail):
        if self.safety_stop is None:
            self.uninstall()
            self.safety_stop = dict(reason=reason, raw_ns=raw(), decision=self.decisions, **detail)
            self.events.append(dict(kind="SAFETY_STOP", **self.safety_stop))      # always kept (one record)

    def after_reply(self, op, sock=None):
        if not self.installed:
            return None
        before = len(self.events)
        record = super().after_reply(op, sock)
        # the base class appends to self.events directly; enforce the bound after the fact
        while len(self.events) > self.max_events:
            self.events.pop()
            self.events_dropped += 1
        if record is not None and record["kind"] == "warmup_collect_freeze":
            self.rss_at_freeze = self._rss()          # same reader as the growth check
        if record is not None and record["wall_ms"] > self.safety["idle_collection_wall_max_ms"]:
            self._stop("IDLE_COLLECTION_WALL", wall_ms=record["wall_ms"])
            return record
        if self.installed and op in self.idle_ops:
            rss = self._rss()
            if rss > self.safety["prep_rss_max_bytes"]:
                self._stop("PREP_RSS_MAX", rss=rss)
            elif (self.rss_at_freeze is not None
                  and rss - self.rss_at_freeze > self.safety["prep_rss_growth_after_freeze_max_bytes"]):
                self._stop("PREP_RSS_GROWTH_AFTER_FREEZE", rss=rss, rss_at_freeze=self.rss_at_freeze)
        del before
        return record

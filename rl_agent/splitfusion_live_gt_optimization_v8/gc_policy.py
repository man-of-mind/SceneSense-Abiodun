"""Bounded, observable GC policy for one fresh dedicated CPU GT worker.

Both arms collect once after startup. Only treatment freezes the existing
tracked startup graph. Automatic collection remains enabled at the registered
thresholds, so newly created request cycles remain collectible. This is not a
numerical optimization and does not change any frame validator or deadline.

The caller initializes this after all worker startup/prewarm and archive READY,
before its first request, and closes it after archive drain, including failure
paths. It must never be installed in the parent or edge process. No GC operation
is performed per request. Unfreeze restores collectability, not original
generation membership or allocation counters.
"""
import gc
import os
import threading
import time


SCHEMA = "splitfusion.dedicated_gt_startup_gc_policy.v1"
THRESHOLDS = (700, 10, 10)
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
MAX_EVENTS = 4096


class GCPolicyError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise GCPolicyError(message)


def _raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _rss_bytes():
    with open("/proc/self/statm", "r", encoding="ascii") as stream:
        values = stream.read().split()
    _require(len(values) >= 2, "worker RSS statm is malformed")
    return int(values[1]) * os.sysconf("SC_PAGE_SIZE")


class GCPolicy:
    def __init__(self, mode):
        _require(mode in ("control", "freeze") and type(mode) is str,
                 "registered GC mode must be control or freeze")
        _require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()),
                 "dedicated CPU worker environment is required")
        _require(gc.isenabled(), "automatic GC must already be enabled")
        _require(gc.get_threshold() == THRESHOLDS, "registered GC thresholds differ")
        _require(gc.get_freeze_count() == 0, "pre-existing frozen GC generation is forbidden")
        self.mode = mode
        self.worker_pid = os.getpid()
        self.startup_freeze_count = 0
        self.freeze_calls = self.unfreeze_calls = 0
        self.processed_at_freeze = 0
        self.phase = "startup"
        self.processed = 0
        self.closed = False
        self.events = []
        self.dropped = self.callback_errors = 0
        self._current = None
        self._owns_freeze = False
        self._callback_installed = False
        self._initial_frozen_count = 0
        self.errors = []
        self.startup_collection = None
        self.startup_rss_bytes = _rss_bytes()
        self.close_rss_bytes = None
        self._callback = self._observe_collection
        try:
            gc.callbacks.append(self._callback)
            self._callback_installed = True
            before = gc.get_stats()
            start, cpu = _raw(), time.thread_time_ns()
            collected = gc.collect()
            end, cpu_end = _raw(), time.thread_time_ns()
            after = gc.get_stats()
            self.startup_collection = dict(start_raw_ns=start, end_raw_ns=end,
                wall_ns=end-start, thread_cpu_ns=cpu_end-cpu, collected=collected,
                generation=2, counts_after=list(gc.get_count()),
                stats_delta=[{key: new[key]-old[key] for key in
                    ("collections", "collected", "uncollectable")}
                    for old, new in zip(before, after)])
            _require(gc.isenabled() and gc.get_threshold() == THRESHOLDS
                     and gc.get_freeze_count() == 0,
                     "startup collection changed the registered GC settings")
            if mode == "freeze":
                # Own cleanup before calling the API, including an exception
                # raised by an instrumented API after it changed the process.
                self._owns_freeze = True
                self.freeze_calls += 1
                gc.freeze()
                self._initial_frozen_count = gc.get_freeze_count()
                _require(self._initial_frozen_count > 0, "startup GC graph was not frozen")
            self.phase = "idle"
        except BaseException:
            self._cleanup()
            self.closed = True
            raise

    def _observe_collection(self, phase, info):
        """Constant-size callback state; no formatting, IO or GC inspection."""
        try:
            if phase == "start":
                if self._current is not None:
                    self.callback_errors += 1
                self._current = (_raw(), time.thread_time_ns(), threading.get_native_id(),
                                 self.phase, self.processed, int(info["generation"]))
                return
            if phase != "stop" or self._current is None:
                self.callback_errors += 1
                return
            start, cpu, tid, owner_phase, processed, generation = self._current
            self._current = None
            end, cpu_end = _raw(), time.thread_time_ns()
            if generation != int(info["generation"]) or tid != threading.get_native_id():
                self.callback_errors += 1
                return
            if len(self.events) >= MAX_EVENTS:
                self.dropped += 1
                return
            self.events.append(dict(start_raw_ns=start, end_raw_ns=end,
                wall_ns=end-start, thread_cpu_ns=cpu_end-cpu, native_tid=tid,
                phase=owner_phase, processed=processed, generation=generation,
                collected=int(info["collected"]), uncollectable=int(info["uncollectable"])))
        except BaseException:
            # Never replace the live GT outcome with a callback exception.
            # The bounded diagnostic fault is rejected at the next boundary.
            self.callback_errors += 1
            self._current = None

    def _same_process(self):
        _require(os.getpid() == self.worker_pid, "GC policy belongs to another process")

    def _settings(self):
        return gc.isenabled() and gc.get_threshold() == THRESHOLDS

    def set_phase(self, phase, processed):
        self._same_process()
        _require(not self.closed, "GC policy is already closed")
        _require(type(phase) is str and 0 < len(phase) <= 80,
                 "bounded GC observation phase required")
        _require(type(processed) is int and self.processed <= processed <= 1000,
                 "GC processed count must be causal and bounded")
        _require(self._settings(), "registered GC settings drifted")
        _require(self.callback_errors == 0 and self.dropped == 0,
                 "GC observation failed or exhausted its bounded events")
        self.phase, self.processed = phase, processed

    def snapshot(self, include_events=False):
        """Compact by default; request the full event list once after close."""
        self._same_process()
        _require(type(include_events) is bool, "include_events must be boolean")
        result = dict(schema=SCHEMA, mode=self.mode, worker_pid=self.worker_pid,
            enabled=gc.isenabled(), thresholds=list(gc.get_threshold()),
            startup_freeze_count=self.startup_freeze_count,
            initial_frozen_count=self._initial_frozen_count,
            freeze_count=gc.get_freeze_count(), freeze_calls=self.freeze_calls,
            unfreeze_calls=self.unfreeze_calls, processed_at_freeze=self.processed_at_freeze,
            phase=self.phase, processed=self.processed, closed=self.closed,
            startup_collection=self.startup_collection,
            event_count=len(self.events), max_events=MAX_EVENTS, dropped=self.dropped,
            callback_errors=self.callback_errors,
            collection_in_progress=self._current is not None,
            callback_installed=self._callback_installed,
            startup_rss_bytes=self.startup_rss_bytes, rss_bytes=_rss_bytes(),
            close_rss_bytes=self.close_rss_bytes, errors=list(self.errors),
            settings_restored=self.closed and self._settings()
                and gc.get_freeze_count() == 0 and not self._callback_installed,
            original_generation_membership_restored=False,
            per_request_manual_collection=False)
        if include_events:
            result["events"] = [dict(event) for event in self.events]
        return result

    def _cleanup(self):
        # No other callbacks or independently frozen graphs are removed.
        if self._callback_installed:
            for index, callback in enumerate(gc.callbacks):
                if callback is self._callback:
                    del gc.callbacks[index]
                    break
            self._callback_installed = False
        if self._owns_freeze:
            self.unfreeze_calls += 1
            try:
                gc.unfreeze()
            finally:
                self._owns_freeze = False
        # A settings-drift failure remains a failure, but must not leave this
        # dedicated process with disabled or retuned automatic collection.
        if gc.get_threshold() != THRESHOLDS:
            gc.set_threshold(*THRESHOLDS)
        if not gc.isenabled():
            gc.enable()

    def close(self):
        """Call after native/archive drain; always restore on policy failure."""
        self._same_process()
        if self.closed:
            return self.snapshot()
        if not self._settings():
            self.errors.append("registered GC settings drifted")
        if self.mode == "control" and gc.get_freeze_count() != 0:
            self.errors.append("control acquired an unowned frozen generation")
        if self.mode == "freeze" and gc.get_freeze_count() > self._initial_frozen_count:
            self.errors.append("frozen generation grew after startup")
        if self.callback_errors or self.dropped or self._current is not None:
            self.errors.append("GC observation incomplete or invalid")
        try:
            self.phase = "close_after_archive_drain"
            self._cleanup()
        finally:
            self.closed = True
            self.close_rss_bytes = _rss_bytes()
        result = self.snapshot()
        if self.errors:
            raise GCPolicyError("; ".join(self.errors))
        return result

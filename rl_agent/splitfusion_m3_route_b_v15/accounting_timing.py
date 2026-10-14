"""Bounded per-call timing of the GT prep owner's storage accounting (M3.5 v15, prep worker only).

The diagnosis (``experiments/splitfusion_m3_gt_prep_growth_diag_v1``) fitted the late-route accounting cost; it never
measured it per call.  ``AccountingTimer`` wraps the qualified exact accounting callable -- in v15 the ONE bounded
``splitfusion_gt_storage_accounting_v3.accounting.AccountingSession`` that the qualified switch installed:

* the wrapped callable is called with the identical arguments and its integer is returned unchanged; any exception
  propagates unchanged (it is only counted).  The capacity rule stays in ``ProducerOwner._account``;
* per call it appends, to preallocated-capacity ``array`` buffers: CLOCK_MONOTONIC_RAW begin/end (the clock of the GT
  worker trace ``ops``, so each call nests inside its FREEZE/BIND op), thread CPU ns, the returned byte count and the
  ``ProducerOwner`` method that called ``_account`` (compute / stage_products / set_ready / ...);
* the buffers are bounded by ``capacity``; overflow is counted in ``dropped``, never silent; nothing is written, hashed
  or logged on the request path.  ``document()`` is written once, at worker exit, beside the other trace records.

Cost: two clock pairs, one frame lookup and five array appends (~2-3 us per call against 1-10 ms per call measured).
"""
from array import array
import sys
import time

SCHEMA = "splitfusion.m3_route_b_v15.storage_accounting_timing.v1"
_RAW = time.CLOCK_MONOTONIC_RAW


class AccountingTimer:
    def __init__(self, function, capacity):
        if not callable(function) or int(capacity) < 1:
            raise ValueError("accounting timer needs a callable and a positive capacity")
        self.__wrapped__ = function
        self.capacity = int(capacity)
        self.begin_raw_ns, self.end_raw_ns = array("q"), array("q")
        self.cpu_ns, self.result_bytes, self.caller = array("q"), array("q"), array("h")
        self.caller_names, self.calls, self.dropped, self.errors, self.foreign_frames = [], 0, 0, 0, 0

    def _caller_code(self):
        account = sys._getframe(2)
        if account.f_code.co_name != "_account":
            self.foreign_frames += 1
        name = account.f_back.f_code.co_name if account.f_back is not None else "<none>"
        try:
            return self.caller_names.index(name)
        except ValueError:
            self.caller_names.append(name)
            return len(self.caller_names) - 1

    def __call__(self, *values):
        cpu0 = time.thread_time_ns()
        raw0 = time.clock_gettime_ns(_RAW)
        try:
            result = self.__wrapped__(*values)
        except BaseException:
            self.errors += 1
            raise
        raw1 = time.clock_gettime_ns(_RAW)
        cpu1 = time.thread_time_ns()
        self.calls += 1
        if len(self.cpu_ns) >= self.capacity:
            self.dropped += 1
            return result
        self.begin_raw_ns.append(raw0)
        self.end_raw_ns.append(raw1)
        self.cpu_ns.append(cpu1 - cpu0)
        self.result_bytes.append(result)
        self.caller.append(self._caller_code())
        return result

    def wrapped_identity(self):
        """``module.qualname`` of the wrapped callable; v15 wraps a bounded ``AccountingSession`` INSTANCE, which
        (unlike v10's module-level function) carries no ``__qualname__``, so its class is named instead."""
        target = self.__wrapped__ if hasattr(self.__wrapped__, "__qualname__") else type(self.__wrapped__)
        return f"{target.__module__}.{target.__qualname__}"

    def document(self):
        return dict(schema=SCHEMA, clock="CLOCK_MONOTONIC_RAW (begin/end), thread CPU (cpu_ns)",
                    wrapped=self.wrapped_identity(),
                    capacity=self.capacity, calls=self.calls, recorded=len(self.cpu_ns), dropped=self.dropped,
                    errors=self.errors, foreign_frames=self.foreign_frames, caller_names=list(self.caller_names),
                    begin_raw_ns=self.begin_raw_ns.tolist(), end_raw_ns=self.end_raw_ns.tolist(),
                    cpu_ns=self.cpu_ns.tolist(), result_bytes=self.result_bytes.tolist(),
                    caller=self.caller.tolist())

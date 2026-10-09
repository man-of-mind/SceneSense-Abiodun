"""Trampoline entry for the UNCHANGED GT service worker with optional GC policy and light tracing.

    python -m rl_agent.splitfusion_gt_worker_spike_fix_v1.worker_entry --fd N --parent P [--cpu C]
        env SPIKE_FIX_MODE=control|candidate (default control = no policy), SPIKE_FIX_TRACE=<json path or empty>

It imports ``splitfusion_exact_gt_service_proto_v2.worker`` unchanged and calls its
``serve``. Tracing (both arms identically): gc.callbacks start/stop, per-op wall /
thread-CPU / gc counts / RSS. The candidate additionally installs BoundedGcPolicy
(prep role only) whose idle maintenance runs after the reply is sent.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import gc
import json
import time
import types

from rl_agent.splitfusion_exact_gt_service_proto_v2 import worker as W, channel as CH, prep_core as PC
from .gc_policy import BoundedGcPolicy, rss_bytes


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class Tracer:
    def __init__(self):
        self.gc_events, self.ops = [], []
        self.current = None
        self._start = None
        self.seq = 0

    def gc_callback(self, phase, info):
        if phase == "start":
            self._start = (raw(), time.thread_time_ns())
        elif self._start is not None:
            r0, c0 = self._start
            self.gc_events.append((info["generation"], info["collected"], r0, raw() - r0, time.thread_time_ns() - c0,
                                   self.current))
            self._start = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--cpu", type=int, default=None)
    a = parser.parse_args()
    mode = os.environ.get("SPIKE_FIX_MODE", "control")
    trace_path = os.environ.get("SPIKE_FIX_TRACE") or None
    if mode not in ("control", "candidate"):
        raise SystemExit("unknown SPIKE_FIX_MODE")
    tracer = Tracer()
    policy = BoundedGcPolicy() if mode == "candidate" else None
    original_handle = PC.PrepCore.handle
    state = dict(last_op=None)

    def traced_handle(self, op, args, fd):
        tracer.seq += 1
        tracer.current = tracer.seq
        state["last_op"] = op
        rec = dict(seq=tracer.seq, op=op, begin_raw_ns=raw(), count0=list(gc.get_count()), rss0=rss_bytes())
        c0 = time.thread_time_ns()
        try:
            result = original_handle(self, op, args, fd)
        finally:
            rec.update(end_raw_ns=raw(), cpu_ns=time.thread_time_ns() - c0, count1=list(gc.get_count()),
                       rss1=rss_bytes())
            tracer.current = None
            tracer.ops.append(rec)
        if op == "FREEZE":
            rec["decoded_raw_ns"] = result[0]["decoded_raw_ns"]
            rec["frame_id"] = result[0]["frame_id"]
        return result

    def send(sock, message, fd=None, deadline=None):
        out = CH.send(sock, message, fd=fd, deadline=deadline)
        if policy is not None and message.get("seq") and state["last_op"] is not None:
            policy._sock = sock
            policy.after_reply(state["last_op"], sock)
        return out

    proxy = types.ModuleType("channel_proxy")
    proxy.__dict__.update({k: v for k, v in vars(CH).items() if not k.startswith("__")})
    proxy.send = send
    PC.PrepCore.handle = traced_handle
    W.CH = proxy
    gc.callbacks.append(tracer.gc_callback)
    if policy is not None:
        policy.install()
    try:
        W.serve(a.fd, a.parent, a.cpu)
    finally:
        snapshot = dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()), freeze_count=gc.get_freeze_count())
        if policy is not None:
            policy.uninstall()
        gc.callbacks.remove(tracer.gc_callback)
        PC.PrepCore.handle = original_handle
        W.CH = CH
        if trace_path:
            with open(trace_path, "x") as stream:
                json.dump(dict(mode=mode, pid=os.getpid(), gc_state_at_close=snapshot,
                               gc_state_after_uninstall=dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()),
                                                             freeze_count=gc.get_freeze_count()),
                               gc_events=[dict(gen=g, collected=n, start_raw_ns=r, wall_ns=w, cpu_ns=c, op_seq=s)
                                          for g, n, r, w, c, s in tracer.gc_events],
                               ops=tracer.ops, policy_events=[] if policy is None else policy.events,
                               rss_final=rss_bytes()), stream)


if __name__ == "__main__":
    main()

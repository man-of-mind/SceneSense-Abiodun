"""Trampoline for the UNCHANGED GT service worker (both arms): role enforcement, bounded GC tracing, and the
candidate GC policy in the prep role only.

    python -m rl_agent.splitfusion_m3_route_b_v1.worker_entry --fd N --parent P [--cpu C]
        --role prep|score --gc-policy control|candidate --trace-dir DIR --max-frames N

* ``--gc-policy candidate`` with ``--role`` other than prep is refused before the worker starts.
* The role in the service's INIT message must equal ``--role`` (checked when the role core is constructed, before
  READY); any mismatch fails the worker closed, so the service never becomes READY.
* Tracing (identical in both arms): every gen2 collection individually (wall, CPU, collected, enclosing op or
  between-requests), gen0/gen1 aggregated per op, per-op wall/thread-CPU/RSS, policy idle maintenance, skipped
  maintenance and request arrivals during maintenance. All bounded by the registered run length; overflow counted.
* SIGTERM (parent death) is converted to SystemExit so the trace and GC restoration still run.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time
import types

from rl_agent.splitfusion_gt_worker_spike_fix_v1.gc_policy import rss_bytes

from . import contract as C


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class Tracer:
    def __init__(self, limits):
        self.limits = dict(limits)
        self.ops, self.gen2 = [], []
        self.dropped = dict(ops=0, gen2=0)
        self.current = None           # the op record being executed, else None
        self.last_op = None           # op of the most recent request (independent of the ops bound)
        self.young_between = [0, 0, 0.0]
        self._start = None

    def gc_callback(self, phase, info):
        if phase == "start":
            self._start = (raw(), time.thread_time_ns())
            return
        if self._start is None:
            return
        r0, c0 = self._start
        self._start = None
        wall, cpu = raw() - r0, time.thread_time_ns() - c0
        gen = info["generation"]
        if gen == 2:
            event = dict(start_raw_ns=r0, wall_ns=wall, cpu_ns=cpu, collected=info["collected"],
                         op=None if self.current is None else self.current["op"],
                         op_seq=None if self.current is None else self.current["seq"])
            if len(self.gen2) < self.limits["max_gen2_events"]:
                self.gen2.append(event)
            else:
                self.dropped["gen2"] += 1
        elif self.current is not None:
            self.current["young_n"][gen] += 1
            self.current["young_wall_ns"] += wall
        else:
            self.young_between[gen] += 1
            self.young_between[2] += wall

    def wrap_handle(self, cls):
        original = cls.handle
        tracer = self

        def handle(core, op, args, fd):
            rec = dict(seq=len(tracer.ops) + sum(tracer.dropped.values()) + 1, op=op, begin_raw_ns=raw(),
                       count0=list(gc.get_count()), rss0=rss_bytes(), young_n=[0, 0], young_wall_ns=0)
            c0 = time.thread_time_ns()
            tracer.current = rec
            tracer.last_op = op
            try:
                return original(core, op, args, fd)
            finally:
                tracer.current = None
                rec.update(end_raw_ns=raw(), cpu_ns=time.thread_time_ns() - c0, count1=list(gc.get_count()),
                           rss1=rss_bytes())
                if len(tracer.ops) < tracer.limits["max_ops"]:
                    tracer.ops.append(rec)
                else:
                    tracer.dropped["ops"] += 1
        cls.handle = handle
        return original


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fd", type=int, required=True)
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--cpu", type=int, default=None)
    p.add_argument("--role", choices=("prep", "score"), required=True)
    p.add_argument("--gc-policy", choices=C.ARMS, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, required=True)
    p.add_argument("--producer", choices=C.PRODUCERS, required=True)
    a = p.parse_args(argv)
    if a.gc_policy == "candidate" and a.role != "prep":
        raise SystemExit("REFUSED: the GC policy may only be enabled in the GT prep worker role")
    if a.producer != "v1" and a.role != "prep":
        raise SystemExit("REFUSED: producer v2 may only be selected in the GT prep worker role")
    if a.max_frames not in C.BUDGETS and a.max_frames < 1:
        raise SystemExit("REFUSED: invalid run length")
    if not a.trace_dir.is_dir() or a.trace_dir.is_symlink():
        raise SystemExit("REFUSED: trace directory must exist (created by the UE seam)")
    return a


def main(argv=None, *, serve=None):
    a = parse(argv)
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import worker as W, channel as CH, prep_core as PC
    from .gc_policy_live import SafeBoundedGcPolicy
    limits = C.trace_limits(a.max_frames)
    tracer = Tracer(limits)
    saved_gc = dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()), freeze_count=gc.get_freeze_count())
    state = dict(role_seen=None, policy=None, refusal=None, producer_constructed=None)
    switch = None
    if a.producer == "v2":                     # before the engine/producer is constructed in PrepCore.__init__
        from rl_agent.splitfusion_numeric_gt_producer_v2 import worker_switch as switch
        from rl_agent.splitfusion_numeric_gt_producer_v2.source_v2 import package_sources as producer_sources
        switch.install()
    else:
        producer_sources = None
    capacity = None
    if a.role == "prep":                       # M3 C1/C2: registered frame-ceiling substitutions, before the engine
        from . import capacity as CAP
        capacity = CAP.install(CAP.PREP_WORKER)
    originals = dict(prep_init=PC.PrepCore.__init__, score_init=W.ScoreCore.__init__, CH=W.CH)

    def make_init(role, original):
        def __init__(core, config):
            if role != a.role:
                state["refusal"] = f"INIT role {role!r} differs from registered --role {a.role!r}"
                raise RuntimeError("REFUSED: " + state["refusal"])
            original(core, config)
            state["role_seen"] = role
            if role == "prep":
                built = type(core.engine._owner._producer).__name__
                state["producer_constructed"] = built
                if built != C.PRODUCER_CLASS[a.producer]:
                    state["refusal"] = f"constructed producer {built!r} differs from registered {a.producer!r}"
                    raise RuntimeError("REFUSED: " + state["refusal"])
            if a.gc_policy == "candidate":          # role == prep is guaranteed here (parse + check above)
                policy = SafeBoundedGcPolicy(policy=C.GC_POLICY, safety=C.SAFETY, max_events=limits["max_policy_events"])
                policy.install()
                state["policy"] = policy
        return __init__

    PC.PrepCore.__init__ = make_init("prep", originals["prep_init"])
    W.ScoreCore.__init__ = make_init("score", originals["score_init"])
    handle_original = tracer.wrap_handle(PC.PrepCore if a.role == "prep" else W.ScoreCore)
    traced_cls = PC.PrepCore if a.role == "prep" else W.ScoreCore

    def send(sock, message, fd=None, deadline=None):
        out = CH.send(sock, message, fd=fd, deadline=deadline)
        policy = state["policy"]
        op = tracer.last_op
        if policy is not None and message.get("seq") and op is not None:
            policy._sock = sock
            policy.after_reply(op, sock)
        return out

    proxy = types.ModuleType("channel_proxy")
    proxy.__dict__.update({k: v for k, v in vars(CH).items() if not k.startswith("__")})
    proxy.send = send
    W.CH = proxy
    gc.callbacks.append(tracer.gc_callback)
    previous_term = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    exit_info = dict(exception=None)
    try:
        (serve or W.serve)(a.fd, a.parent, a.cpu)
    except BaseException as error:
        exit_info["exception"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        at_close = dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()), freeze_count=gc.get_freeze_count())
        policy = state["policy"]
        if policy is not None:
            policy.uninstall()
        restored = dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()), freeze_count=gc.get_freeze_count())
        gc.callbacks.remove(tracer.gc_callback)
        signal.signal(signal.SIGTERM, previous_term)
        PC.PrepCore.__init__, W.ScoreCore.__init__, W.CH = originals["prep_init"], originals["score_init"], originals["CH"]
        traced_cls.handle = handle_original
        capacity_document = None
        if capacity is not None:
            capacity.uninstall()
            capacity_document = capacity.document()
        switch_restored = None
        if switch is not None:
            switch.uninstall()
            from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as _D
            from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as _OW
            from rl_agent.splitfusion_numeric_gt_producer_v1 import source as _NS
            switch_restored = (not switch.installed() and _D.NS is _NS
                               and _OW.NumericSourceProducer is _NS.NumericSourceProducer)
        doc = dict(schema="splitfusion.m3_route_b_v1.worker_trace.v1", role=a.role, role_seen=state["role_seen"],
                   gc_policy=a.gc_policy, policy_installed=policy is not None, pid=os.getpid(), limits=limits,
                   refusal=state["refusal"], exit=exit_info, producer=a.producer,
                   producer_constructed=state["producer_constructed"], producer_switch_restored=switch_restored,
                   m3_capacity=capacity_document,
                   producer_v2_sources=None if producer_sources is None else producer_sources(), gc_state_initial=saved_gc, gc_state_at_close=at_close,
                   gc_state_after_restore=restored, restored_equals_initial=restored == saved_gc,
                   ops=tracer.ops, gen2=tracer.gen2, young_between=tracer.young_between, dropped=tracer.dropped,
                   policy_events=[] if policy is None else policy.events,
                   policy_events_dropped=0 if policy is None else policy.events_dropped,
                   safety_stop=None if policy is None else policy.safety_stop,
                   rss_at_freeze=None if policy is None else policy.rss_at_freeze, rss_final=rss_bytes(),
                   written_raw_ns=raw())
        path = a.trace_dir / f"{a.role}_gc_trace.json"
        with open(path, "x") as stream:
            json.dump(doc, stream)


if __name__ == "__main__":
    main()

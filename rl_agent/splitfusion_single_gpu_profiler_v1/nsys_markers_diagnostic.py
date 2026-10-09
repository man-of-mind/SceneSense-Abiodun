"""Opt-in diagnostic NVTX adapter. Does not alter pinned recorder source bytes."""
import atexit
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path

STAGES = {'carla.world_tick', 'ue.worker_frame', 'agent.actor', 'ue.input_7ch',
          'ue.codec.total_ue_preparation', 'ue.codec.front_backbone',
          'edge.process', 'edge.compute', 'edge.codec.frozen_tail',
          'edge.qperc_evaluate'}

def install(output):
    from . import recorder_v1 as R
    enter, leave = R._Span.__enter__, R._Span.__exit__
    active = {}
    costs = collections.Counter()
    anchors = []

    def start(span):
        result = enter(span)
        if span._stage not in STAGES:
            return result
        t0 = time.perf_counter_ns()
        try:
            torch = sys.modules.get('torch')
            if torch is None or not torch.cuda.is_initialized():
                return result
            nv = torch.cuda.nvtx
            ctx = {**span._rec.context(), **span._fields}
            text = json.dumps({'stage': span._stage, 'seq': span._seq,
                               'frame_id': ctx.get('frame_id'),
                               'tensor_seq': ctx.get('tensor_seq'),
                               'ticket_seq': ctx.get('ticket_seq'),
                               'decision_seq': ctx.get('decision_seq'),
                               'session_uuid': ctx.get('session_uuid')}, separators=(',', ':'))
            a = R.raw_ns()
            nv.mark('RAW_ANCHOR|' + str(a))
            b = R.raw_ns()
            anchors.append((a, b, threading.get_native_id()))
            nv.range_push(text)
            active[id(span)] = nv
            costs['ranges'] += 1
        except Exception:
            costs['errors'] += 1
        finally:
            costs['enter_ns'] += time.perf_counter_ns() - t0
        return result

    def end(span, *exc):
        t0 = time.perf_counter_ns()
        nv = active.pop(id(span), None)
        if nv is not None:
            try:
                nv.range_pop()
            except Exception:
                costs['errors'] += 1
            costs['exit_ns'] += time.perf_counter_ns() - t0
        return leave(span, *exc)

    R._Span.__enter__, R._Span.__exit__ = start, end

    def save():
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            json.dump({'pid': os.getpid(), 'costs': dict(costs), 'anchors': anchors,
                       'unclosed_ranges': len(active)}, f)
    atexit.register(save)


"""Production source dispatch; identical core is used by composed CPU checks."""
import json
import threading
import time

from .identity import require
from .transport import SourceIngress
from .edge_components import SharedEdgeProcessor
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class EventLog:
    def __init__(self, path):
        self.handle = path.open('x')
        self.lock = threading.Lock()
    def __call__(self, source, kind, **values):
        row = dict(schema='splitfusion.m4.event.v1', source=source.tag, kind=kind,
                   wall_ns=time.time_ns(), raw_ns=raw(), **values)
        with self.lock:
            self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            self.handle.flush()
        return row
    def close(self):
        self.handle.close()


class _CompleteTap:
    def __init__(self, delegate):
        self.delegate, self.complete = delegate, None
    def ingest(self, *values, **kw):
        self.complete = self.delegate.ingest(*values, **kw)
        return self.complete
    def __getattr__(self, name):
        return getattr(self.delegate, name)


class DataPath:
    def __init__(self, *, sources, contract, evaluators, compute, publish, terminal, log):
        self.sources, self.publish, self.send_terminal, self.log = sources, publish, terminal, log
        self.ingress = SourceIngress(sources)
        self.ingress.reassembly = {k:_CompleteTap(v) for k,v in self.ingress.reassembly.items()}
        self.retired, self.completed = set(), set()
        self.timing = None
        self.compute_lock = threading.Lock()  # one inference owner, not a two-front admission lock

        def measured(**kwargs):
            result = compute(**kwargs)
            context = kwargs['context']
            source = sources.resolve(context.stream_id)
            self.timing.update(tail_complete_wall_s=time.time(), prediction_ready_raw_ns=raw())
            self.log(source, 'prediction_ready', frame_id=context.frame_id,
                     prediction_ready_raw_ns=self.timing['prediction_ready_raw_ns'],
                     prediction_ready_wall_ns=int(self.timing['tail_complete_wall_s']*1e9),
                     compute_ms=(self.timing['tail_complete_wall_s']-self.timing['compute_start_wall_s'])*1000,
                     queue_ms=(self.timing['compute_start_wall_s']-self.timing['reassembly_complete_wall_s'])*1000,
                     capture_timestamp_ns=context.capture_timestamp_ns,
                     mode_id=kwargs['profile'].mode_id, q_e4=kwargs['profile'].q_e4)
            return result
        first = sources.sources[0]
        self.processor = SharedEdgeProcessor(sources=sources, evaluators=evaluators, contract=contract,
            run_id=first.run_id, cell_id=first.cell_id, service_deadline_s=.170, ack_timeout_s=.5, compute=measured)

    def end(self, source):
        require(self.sources.resolve(source.stream_id) == source, 'foreign source retirement')
        self.retired.add(source.stream_id)

    def terminal(self, item, outcome, stage, newer=None):
        source = self.sources.resolve(item['stream_id'])
        value = self.processor.terminal(item['payload'], outcome=outcome, stage=stage,
                                        superseded_by_frame_id=newer)
        self.send_terminal(source, value)
        self.log(source, 'edge_terminal', frame_id=item['frame_id'], outcome=outcome, stage=stage)

    def receive(self, source, datagram, *, monotonic_s=None, wall_s=None):
        require(source.stream_id not in self.retired, 'traffic after source retirement')
        wall = time.time() if wall_s is None else wall_s
        try:
            result = self.ingress.ingest(source, datagram,
                received_at_s=time.monotonic() if monotonic_s is None else monotonic_s, received_wall_s=wall)
        except B.DeadlineExpired:
            complete = self.ingress.reassembly[source.stream_id].complete
            require(complete is not None, 'deadline without completed source frame')
            self.terminal(dict(payload=complete.payload, stream_id=source.stream_id, frame_id=complete.message_id),
                          'STALE_BEFORE_EDGE', B.EDGE_STAGE_AFTER_REASSEMBLY)
            return None
        if result:
            for item in ([result['displaced']] if result['displaced'] else []):
                self.terminal(item, 'SUPERSEDED_PENDING', 'EDGE_PENDING_SLOT', result['offered']['frame_id'])
            if not result['admitted']:
                self.terminal(result['offered'], 'SUPERSEDED_PENDING', 'EDGE_PENDING_SLOT')
            self.log(source, 'reassembled', frame_id=result['offered']['frame_id'],
                     payload_bytes=len(result['offered']['payload']), admitted=result['admitted'])
        return result

    def process_once(self, timeout=.1, *, now_s=None):
        item = self.ingress.pending.take(timeout)
        if item is None:
            return False
        stream, item = item
        source = self.sources.resolve(stream)
        with self.compute_lock:
            try:
                B.check_deadline(B.EDGE_STAGE_BEFORE_DECODE, item['capture_ns'], .5,
                                 now_s=time.time() if now_s is None else now_s)
                self.timing = dict(reassembly_complete_wall_s=item['received_wall_s'],
                                   admission_wall_s=time.time(), compute_start_wall_s=time.time())
                result = self.processor.process(item['payload'], edge_timing=self.timing)
                self.publish(source, result.update)
                self.completed.add((stream, item['frame_id']))
            except B.DeadlineExpired as error:
                self.terminal(item, 'STALE_BEFORE_MAP', error.stage)
            finally:
                self.timing = None
        return True

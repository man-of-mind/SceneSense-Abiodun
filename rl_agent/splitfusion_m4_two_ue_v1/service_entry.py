"""One source, one real containment/accounting/GC service process.

The retained worker substitutions run only INSIDE this isolated process. They
never select or redirect another source. No older module is rebound to M4.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time

from .identity import Source, require
from rl_agent.splitfusion_exact_gt_live_v2.service import Service, verify_sources
from rl_agent.splitfusion_exact_gt_service_proto_v2 import channel as CH


class BoundService(Service):
    def __init__(self, sock, config):
        self.source = Source(**config['m4_source'])
        self.seen = set()
        self.predictions = {}
        self.deferred_faults = {}
        super().__init__(sock, config)

    def enqueue(self, worker, priority, op, args, frame_id, fd=None):
        self.frame(frame_id).setdefault('m4_enqueue', []).append(dict(
            worker=worker, op=op, raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))
        return super().enqueue(worker, priority, op, args, frame_id, fd)

    def on_ue(self, message, fd):
        try:
            message = self.source.unwrap(message)
            op, frame = message['op'], message.get('frame_id')
            if op == 'M4_STATUS':
                require(fd is None and frame in self.frames, 'unknown status frame')
                state = self.frames[frame]
                CH.send(self.ue, self.source.wrap(dict(op='M4_STATUS', frame_id=frame,
                    state={k: state.get(k) for k in ('unbound', 'bound', 'committed', 'outcome', 'no_gt', 'prep_retired')})),
                    deadline=time.monotonic() + 10)
                return
            require(op in ('OPEN', 'FREEZE', 'PLAN', 'SEND', 'TERMINAL', 'CLOSE'), 'unknown UE operation')
            require(('ue', op, frame) not in self.seen, 'duplicate UE operation')
            if op == 'FREEZE':
                descriptor = json.loads(message['descriptor'])
                self.source.capture(descriptor['capture_template'])
                require(descriptor['frame_id'] == frame, 'FREEZE frame mismatch')
            if op == 'PLAN':
                plan = json.loads(message['plan'])
                self.source.capture(plan['capture'])
                require(plan['capture']['frame_id'] == frame == plan['action']['frame_id'], 'PLAN frame mismatch')
                require(message['frame_context']['stream_id'] == self.source.stream_id, 'foreign PLAN context')
            self.seen.add(('ue', op, frame))
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        return super().on_ue(message, fd)

    def accept_edge(self):
        connection, _ = self.listener.accept()
        pid, uid, gid = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                               struct.calcsize('3i')))
        try:
            hello, fd = CH.receive(connection, deadline=time.monotonic() + 10)
            if fd is not None:
                os.close(fd)
            hello = self.source.unwrap(hello)
            require(fd is None and self.edge is None and uid in (0, os.getuid())
                    and hello.get('op') == 'HELLO' and hello.get('nonce') == self.nonce,
                    'edge capability/peer mismatch')
        except BaseException as error:
            connection.close()
            self.errors.append('refused edge connection: ' + str(error))
            return
        self.edge, self.edge_pid = connection, pid
        self.edge_hello = dict(pid=pid, uid=uid, gid=gid, source=self.source.tag)
        CH.send(connection, self.source.wrap(dict(op='WELCOME', session=self.session)),
                deadline=time.monotonic() + 10)

    def on_edge(self, message, fd):
        try:
            message = self.source.unwrap(message)
            op, frame = message['op'], message.get('frame_id')
            require(op in ('PREDICTION', 'EDGE_TERMINAL', 'EMITTED', 'CLOSE'), 'unknown edge operation')
            require(('edge', op, frame) not in self.seen, 'duplicate edge operation')
            if op == 'PREDICTION':
                identity = message['identity']
                self.source.capture(identity['gt_identity'])
                require(frame == identity['action_identity']['frame_id'] == identity['gt_identity']['frame_id'],
                        'prediction frame mismatch')
                require(message['frame_context']['stream_id'] == self.source.stream_id, 'foreign prediction context')
                self.predictions[frame] = identity['action_identity']
            self.seen.add(('edge', op, frame))
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        result = super().on_edge(message, fd)
        if op == 'PREDICTION' and frame in self.deferred_faults:
            self.send_edge(self.deferred_faults.pop(frame))
        return result

    def send_edge(self, value):
        frame = value.get('frame_id')
        # A refusal can precede the prediction. Retain the existing outcome
        # until that source's evaluator owns a ticket; never silently lose it.
        if value['op'] == 'UNSCORED' and frame not in self.predictions:
            require(frame in self.frames, 'fault for unknown source frame')
            self.deferred_faults[frame] = value
            return
        require(frame in self.predictions, 'result lacks a source-bound prediction')
        message = self.source.wrap(dict(value, _m4_action=self.predictions[frame]))
        return super().send_edge(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fd', type=int, required=True)
    p.add_argument('--parent', type=int, required=True)
    p.add_argument('--trace-dir', type=Path, required=True)
    args = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only service required')
    ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)
    require(os.getppid() == args.parent, 'parent changed')
    sock = socket.socket(fileno=args.fd)
    init, fd = CH.receive(sock, deadline=time.monotonic() + 60)
    require(fd is None, 'unexpected initialization fd')
    config = init['config']
    verify_sources(config['sources'])
    require(args.trace_dir.is_dir() and not args.trace_dir.is_symlink(), 'fresh owned trace directory required')
    from rl_agent.splitfusion_gt_geometry_containment_v1.service_entry import WorkerSubstitution
    from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP
    substitution = WorkerSubstitution(arm='candidate', producer='v2', trace_dir=args.trace_dir,
                                      max_frames=config['limits']['max_frames']).install()
    capacity = CAP.install(CAP.GT_SERVICE)
    service, rc = None, 1
    try:
        service = BoundService(sock, config)
        os.sched_setaffinity(0, {config['limits']['service_cpu']})
        CH.send(sock, dict(op='READY', pid=os.getpid(), session=service.session, source=service.source.tag,
                          socket=str(service.socket_path), capability=str(service.capability_path),
                          affinity=sorted(os.sched_getaffinity(0))), deadline=time.monotonic() + 10)
        try:
            service.run()
        except BaseException as error:
            service.errors.append(f'{type(error).__name__}: {error}')
        result = service.close()
        CH.send(sock, dict(op='CLOSED', result={k: result[k] for k in ('errors', 'outcomes', 'decisions', 'refusals')}),
                deadline=time.monotonic() + 30)
        rc = int(bool(result['errors']))
        return rc
    finally:
        substitution.uninstall()
        capacity.uninstall()
        (args.trace_dir / 'm4_service_binding.json').write_text(json.dumps(dict(
            source=config['m4_source'], substitutions=substitution.records, capacity=capacity.document(),
            returncode=rc, pid=os.getpid()), indent=2))
        sock.close()


if __name__ == '__main__':
    raise SystemExit(main())

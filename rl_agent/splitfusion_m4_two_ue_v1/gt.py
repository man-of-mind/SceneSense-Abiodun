"""Source-isolated host services and evaluators; one shared inference callback.

No install_early_gt_edge scope, class _instance, or mutable module binding is
used. Each object's queues, socket, producer/scorer and retirement are owned by
one immutable Source. All qualified GT numerical workers remain unchanged.
"""
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time

from .identity import require
from rl_agent.splitfusion_exact_gt_live_v2 import edge as G
from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff, ROOT, raw
from rl_agent.splitfusion_exact_gt_service_proto_v2 import channel as CH
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W


class SourceHandoff(EarlyGtUeHandoff):
    def __init__(self, *, source, **kwargs):
        self.source = source
        self._status_responses = queue.Queue(maxsize=1)
        super().__init__(**kwargs)

    def _read_service(self):
        while True:
            try:
                message, fd = CH.receive(self.sock, deadline=time.monotonic() + 3600)
                if fd is not None:
                    os.close(fd)
                    raise ValueError('unexpected UE service response fd')
                if message.get('op') == 'M4_STATUS':
                    self._status_responses.put_nowait(self.source.unwrap(message))
                elif message.get('op') == 'CLOSED':
                    self._service_closed = message
                    return
                elif message.get('op') == 'FAILED':
                    raise ValueError(str(message.get('error')))
                else:
                    raise ValueError('unknown UE service response')
            except BaseException as error:
                if self._closed_result is None and not getattr(self, '_closing', False):
                    self._fault(f'service channel: {type(error).__name__}: {error}')
                return

    def status(self, frame):
        """Explicit offline qualification barrier; never part of the live decision path."""
        self._send(dict(op='M4_STATUS', frame_id=frame))
        response = self._status_responses.get(timeout=10)
        require(response['frame_id'] == frame, 'wrong status response')
        return response['state']

    def stage_before_action(self, runtime, **kwargs):
        self.source.capture(dict(run_id=runtime.pipeline.run_id, cell_id=runtime.pipeline.cell_id,
                                 stream_id=runtime.pipeline.stream_id))
        return super().stage_before_action(runtime, **kwargs)

    def _send(self, value, fd=None):
        return super()._send(self.source.wrap(value), fd)

    def start_service(self, engine):
        require(self.process is None, 'service already started')
        reg, limits = self.registration, self.registration['exact_gt_limits']
        trace = self.directory / 'm4_worker_trace'
        trace.mkdir()
        score = dict(run_binding=reg['numeric_run_binding'], producer_binding=reg['numeric_producer_binding'],
            max_jobs=limits['scorer_jobs'], max_bytes=limits['scorer_jobs'] * 8 * 1024**2,
            reservation_bytes=8 * 1024**2, max_waiting=limits['max_waiting'], max_frames=4 * limits['max_frames'])
        config = dict(engine=engine, score=score, limits=limits, sources=reg['exact_gt_sources'],
            m4_source=self.source.tag, evidence_dir=str(self.directory / 'service'),
            socket_path=str(self.evidence_dir / reg['exact_gt_socket_name']),
            capability_path=str(self.evidence_dir / reg['exact_gt_capability_name']),
            binding=dict(run=reg['numeric_run_binding'], scope='exact_gt_live_v2'))
        parent, child = CH.pair()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        env.pop('PYTHONPATH', None)
        self.log = (self.directory / 'service.log').open('xb')
        started = raw()
        try:
            self.process = subprocess.Popen([sys.executable, '-B', '-u', '-m',
                'rl_agent.splitfusion_m4_two_ue_v1.service_entry', '--fd', str(child.fileno()),
                '--parent', str(os.getpid()), '--trace-dir', str(trace)], pass_fds=(child.fileno(),),
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
        finally:
            child.close()
        self.sock = parent
        try:
            CH.send(parent, dict(config=config), deadline=time.monotonic() + 30)
            ready, fd = CH.receive(parent, deadline=time.monotonic() + 120)
            require(fd is None and ready.get('op') == 'READY' and ready.get('pid') == self.process.pid
                    and ready.get('source') == self.source.tag, 'source-bound READY mismatch')
        except BaseException as error:
            self.abort(error)
            raise
        self.ready = dict(ready, started_raw_ns=started, ready_raw_ns=raw())
        self.session = ready['session']
        self._freeze_thread = threading.Thread(target=self._freeze_worker, name=f'gt-freeze-{self.source.ue_id}', daemon=True)
        self._reader = threading.Thread(target=self._read_service, name=f'gt-reader-{self.source.ue_id}', daemon=True)
        self._freeze_thread.start()
        self._reader.start()
        self._write('SERVICE_READY.json', self.ready)


class FeedbackRouter:
    """Compact original R4FB bytes go only to the registered destination.

    sendto is injected (a real bound UDP sender in production, a recording sink
    in CPU qualification). All action identities are checked before delivery.
    """
    def __init__(self, sources, sendto):
        self.sources, self.sendto = sources, sendto
        self.tickets, self.sessions, self.delivered = {}, {}, set()
        self.lock = threading.RLock()

    def reserve(self, source, ticket):
        require(self.sources.resolve(ticket.context.stream_id) == source, 'foreign source ticket')
        source.capture(ticket.gt_identity)
        key = (source.stream_id, ticket.context.frame_id)
        action = MP.run4_identity(ticket.envelope)
        with self.lock:
            require(sum(k[0] == source.stream_id for k in self.tickets) < 3900, 'source frame ceiling exceeded')
            require(key not in self.tickets, 'duplicate source/frame submission')
            owner = self.sessions.get(action['session_uuid'])
            require(owner is None or owner == source, 'decision session belongs to another source')
            self.sessions[action['session_uuid']] = source
            self.tickets[key] = action

    def deliver(self, source, payload, destination):
        require(destination == source.destination and self.sources.resolve(source.stream_id) == source,
                'wrong feedback destination/source')
        feedback, _ = W.decode_feedback(payload)
        key = (source.stream_id, feedback.frame_id)
        with self.lock:
            require(key in self.tickets and key not in self.delivered, 'unknown/duplicate feedback')
            expected = self.tickets[key]
            for name in MP.IDENTITY_FIELDS:
                if name != 'frame_kind':
                    require(getattr(feedback, name) == expected[name], 'feedback action/session mismatch: ' + name)
            self.sendto(payload, destination)
            self.delivered.add(key)


class SourceEvaluator(G.EarlyGtEvaluatorV1):
    def __init__(self, *, source, router, socket_path, capability_path, **kwargs):
        self.source, self.router = source, router
        E.Run4EvaluatorV2.__init__(self, send=lambda data: router.deliver(source, data, source.destination), **kwargs)
        self._b = dict(socket_path=str(socket_path), capability_path=str(capability_path), connect_timeout_s=30.)
        self._materialized, self._pending, self._frames = {}, {}, {}
        self._control = queue.Queue(maxsize=256)
        self._sock = None
        self._send_lock, self._stop = threading.Lock(), threading.Event()
        self._failure = None
        self._closing_source = False
        self._sender = threading.Thread(target=self._send_loop, name=f'gt-send-{source.ue_id}', daemon=True)
        self._receiver = threading.Thread(target=self._receive_loop, name=f'gt-recv-{source.ue_id}', daemon=True)
        self.counters.update(service_results=0, service_unscored=0, terminals_notified=0, control_overflow=0,
                             drained_unanswered=0)

    def submit(self, ticket):
        require(not self._closing_source, 'source is draining/closed')
        self.router.reserve(self.source, ticket)
        self._materialized[ticket.context.frame_id] = raw()
        return super().submit(ticket)

    def _connect(self):
        from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B
        capability = Path(self._b['capability_path'])
        nonce = json.loads(capability.read_bytes())['nonce']
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with B._address(Path(self._b['socket_path'])) as address:
            sock.connect(address)
        CH.send(sock, self.source.wrap(dict(op='HELLO', nonce=nonce, info=dict(pid=os.getpid(), uid=os.getuid()))),
                deadline=time.monotonic() + 10)
        welcome, fd = CH.receive(sock, deadline=time.monotonic() + 10)
        welcome = self.source.unwrap(welcome)
        require(fd is None and welcome.get('op') == 'WELCOME', 'source-bound edge handshake refused')
        self._sock, self.session = sock, welcome['session']

    def _send_service(self, value, fd=None):
        return super()._send_service(self.source.wrap(value), fd)

    def accept_result(self, message):
        """Validate before removing ANY pending work (also a focused test seam)."""
        message = self.source.unwrap(message)
        frame = message.get('frame_id')
        require(message.get('op') in ('RESULT', 'UNSCORED'), 'unknown GT response')
        with self._lock:
            require(frame in self._pending, 'unknown/duplicate source GT response')
            ticket = self._pending[frame]
            require(message.get('_m4_action') == MP.run4_identity(ticket.envelope), 'wrong GT response session/action')
            del self._pending[frame]
        return message, ticket

    def _receive_loop(self):
        try:
            while not self._stop.is_set():
                try:
                    message, fd = CH.receive(self._sock, deadline=time.monotonic() + 3600)
                except (CH.ChannelError, OSError):
                    if self._stop.is_set():
                        return  # source-local shutdown deliberately wakes recvmsg
                    raise
                if fd is not None:
                    os.close(fd)
                    raise ValueError('unexpected result descriptor')
                message, ticket = self.accept_result(message)
                frame, received = ticket.context.frame_id, raw()
                if message['op'] == 'RESULT':
                    feedback, reason = W.quality_feedback(self._spec, message['measurement'], ticket.envelope)
                    require(feedback.q_perc == message['q_perc'], 'unchanged feedback formula differs')
                    self._emit(ticket, feedback, reason, measurement=message['measurement'],
                               timing=dict(service_result_received_raw_ns=received))
                    self.counters['service_results'] += 1
                else:
                    self._emit(ticket, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE,
                               timing=dict(service_unscored_received_raw_ns=received, reason=message.get('reason')))
                    self.counters['service_unscored'] += 1
                emitted, record = raw(), self._record_emission(ticket)
                self._frames.setdefault(frame, {}).update(result_received=received, emitted=emitted,
                                                          op=message['op'], reason=message.get('reason'))
                self._send_service(dict(op='EMITTED', frame_id=frame, raw_ns=emitted,
                    packet_sha256=record['feedback_sha256'], kind=record['kind'], reason=record['reason']))
        except BaseException as error:
            self._failure = f'receiver: {type(error).__name__}: {error}'[:1000]

    def close(self, timeout_s=None):
        self._closing_source = True
        self._queue.put(None)
        self._sender.join(timeout=30)
        deadline = time.monotonic() + (60. if timeout_s is None else float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    break
            time.sleep(.02)
        with self._lock:
            leftover = list(self._pending.values())
            self._pending.clear()
        for item in leftover:
            self._emit(item, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE, timing=dict(drained=True))
            self.counters['drained_unanswered'] += 1
        self._stop.set()
        if self._sock is not None:
            try:
                self._send_service(dict(op='CLOSE'))
            finally:
                # close() alone does not wake another thread blocked in recvmsg
                # on Linux. Wake only this source after its drain/CLOSE, so a
                # 10-s receiver join cannot stall the other source's lifecycle.
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
        self._receiver.join(timeout=10)
        with self._lock:
            return {**self.counters, 'worker_alive': self._sender.is_alive(), 'records': len(self.records),
                    'failure': self._failure, 'frames': dict(self._frames)}


class SharedEdgeProcessor(E.Run4EdgeProcessorV2):
    """One retained compute callback/model, explicit per-source GT dispatch."""
    def __init__(self, *, sources, evaluators, **kwargs):
        self.sources, self.evaluators = sources, evaluators
        require(set(evaluators) == set(sources.by_stream), 'evaluator coverage mismatch')
        super().__init__(**kwargs)

    def process(self, frame, *, edge_timing, segmentation=None):
        envelope, context, profile, identity = self.verify(frame)
        source = self.sources.resolve(context.stream_id)
        require(source.run_id == self.run_id and source.cell_id == self.cell_id, 'edge run binding mismatch')
        records, labels = self._compute(envelope=envelope, context=context, profile=profile, identity=identity)
        if envelope.reward_requested:
            ticket = E.EvaluationTicketV2(envelope=envelope, context=context, records=tuple(records), predicted_mask=labels,
                gt_identity=MP.gt_identity(run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
                    frame_id=context.frame_id, anchor_action_id=profile.action_id, anchor_profile_id=profile.profile_id,
                    capture_timestamp_ns=context.capture_timestamp_ns), enqueued_wall_ns=time.time_ns())
            self.evaluators[source.stream_id].submit(ticket)
        service_at, ack_at = self.deadlines(context.capture_timestamp_ns)
        update = MP.build_run4_map_update(run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
            frame_id=context.frame_id, sequence_id=context.sequence_id, identity=identity,
            decoder_identity=profile.decoder_identity, capture_timestamp_ns=context.capture_timestamp_ns,
            records=records, service_deadline_at=service_at, ack_timeout_at=ack_at, edge_timing=edge_timing,
            segmentation=segmentation or {'available': False})
        return E.ProcessedFrameV2(envelope=envelope, context=context, profile=profile, identity=identity,
                                  update=update, evaluation=None)

    def terminal(self, frame, *, outcome, stage, superseded_by_frame_id=None):
        value = super().terminal(frame, outcome=outcome, stage=stage, superseded_by_frame_id=superseded_by_frame_id)
        source = self.sources.resolve(value['stream_id'])
        if value['run4_identity']['reward_requested']:
            self.evaluators[source.stream_id].notify_terminal(value['frame_id'], outcome, stage)
        return value

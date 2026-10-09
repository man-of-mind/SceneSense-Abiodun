"""Edge-side integration: materialized predictions go to the host GT service right after compute.

``EarlyGtProcessorV1.process`` is the retained ``Run4EdgeProcessorV2.process`` with one
change: for a reward-requested frame the evaluation ticket is created and handed to the
evaluator's (retained, bounded, non-blocking) ``submit`` immediately after ``compute``
returns, BEFORE the map update is built or published. The retained loop therefore never
submits it again (``evaluation=None``). The compute thread only performs a queue put.

``EarlyGtEvaluatorV1`` keeps the retained bounded queue, at-most-once ``_emit`` and the
unchanged ``W.quality_feedback`` formula. A sender thread computes the retained prediction
digest (``edge_live.prediction_for_ticket``), seals the native labels into a memfd and sends
a reference to the host service; a receiver thread turns the service result into the
retained feedback (or the retained GROUND_TRUTH_UNAVAILABLE fault for an explicit no-score
outcome) and reports the emission instant back for the frame clock.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_gt_handoff_v1 import broker as B
from rl_agent.splitfusion_numeric_gt_runtime_v1.edge_live import prediction_for_ticket
from rl_agent.splitfusion_exact_gt_service_proto_v1 import channel as CH


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class EarlyGtProcessorV1(E.Run4EdgeProcessorV2):
    def process(self, frame, *, edge_timing, segmentation=None):
        envelope, context, profile, identity = self.verify(frame)
        records, labels = self._compute(envelope=envelope, context=context, profile=profile, identity=identity)
        if envelope.reward_requested:
            evaluator = EarlyGtEvaluatorV1._instance
            ticket = E.EvaluationTicketV2(envelope=envelope, context=context, records=tuple(records),
                predicted_mask=labels, gt_identity=MP.gt_identity(run_id=self.run_id, cell_id=self.cell_id,
                    stream_id=context.stream_id, frame_id=context.frame_id, anchor_action_id=profile.action_id,
                    anchor_profile_id=profile.profile_id, capture_timestamp_ns=context.capture_timestamp_ns),
                enqueued_wall_ns=time.time_ns())
            evaluator._materialized[int(context.frame_id)] = raw()
            evaluator.submit(ticket)                       # bounded put_nowait; never blocks compute
        service_at, ack_at = self.deadlines(context.capture_timestamp_ns)
        update = MP.build_run4_map_update(run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
            frame_id=context.frame_id, sequence_id=context.sequence_id, identity=identity,
            decoder_identity=profile.decoder_identity, capture_timestamp_ns=context.capture_timestamp_ns,
            records=records, service_deadline_at=service_at, ack_timeout_at=ack_at, edge_timing=edge_timing,
            segmentation=segmentation or {"available": False})
        return E.ProcessedFrameV2(envelope=envelope, context=context, profile=profile, identity=identity,
                                  update=update, evaluation=None)

    def terminal(self, frame, *, outcome, stage, superseded_by_frame_id=None):
        document = super().terminal(frame, outcome=outcome, stage=stage, superseded_by_frame_id=superseded_by_frame_id)
        identity = document.get("run4_identity") or {}
        if identity.get("reward_requested") is True and EarlyGtEvaluatorV1._instance is not None:
            EarlyGtEvaluatorV1._instance.notify_terminal(int(document["frame_id"]), str(outcome), str(stage))
        return document


class EarlyGtEvaluatorV1(E.Run4EvaluatorV2):
    _binding = None
    _instance = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        binding = type(self)._binding
        if type(binding) is not dict:
            raise RuntimeError("early GT edge binding must precede evaluator construction")
        self._b = binding
        self._materialized = {}
        self._pending = {}
        self._frames = {}
        self._control = queue.Queue(maxsize=256)
        self._sock = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._failure = None
        self._sender = threading.Thread(target=self._send_loop, name="gt-edge-sender", daemon=True)
        self._receiver = threading.Thread(target=self._receive_loop, name="gt-edge-receiver", daemon=True)
        self.counters.update(service_results=0, service_unscored=0, terminals_notified=0, control_overflow=0,
                             drained_unanswered=0)
        EarlyGtEvaluatorV1._instance = self

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._sender.start()

    def _connect(self):
        deadline = time.monotonic() + float(self._b["connect_timeout_s"])
        capability = Path(self._b["capability_path"])
        while not capability.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("early GT service capability did not appear")
            time.sleep(0.05)
        nonce = json.loads(capability.read_text())["nonce"]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.settimeout(10)
        with B._address(Path(self._b["socket_path"])) as address:
            sock.connect(address)
        CH.send(sock, dict(op="HELLO", nonce=nonce, info=dict(pid=os.getpid(), uid=os.getuid())),
                deadline=time.monotonic() + 10)
        welcome, _ = CH.receive(sock, deadline=time.monotonic() + 10)
        if welcome.get("op") != "WELCOME":
            raise RuntimeError("early GT service refused the edge")
        self._sock = sock
        self.session = welcome["session"]

    def _send_service(self, value, fd=None):
        with self._send_lock:
            CH.send(self._sock, value, fd=fd, deadline=time.monotonic() + 5)

    def notify_terminal(self, frame_id, outcome, stage):
        try:
            self._control.put_nowait(dict(op="EDGE_TERMINAL", frame_id=frame_id, outcome=outcome, stage=stage, raw_ns=raw()))
        except queue.Full:
            with self._lock:
                self.counters["control_overflow"] += 1

    # ------------------------------------------------------------------ threads
    def _send_loop(self):
        try:
            self._connect()
            self._receiver.start()
            while True:
                try:
                    message = self._control.get_nowait()
                    self._send_service(message)
                    with self._lock:
                        self.counters["terminals_notified"] += 1
                    continue
                except queue.Empty:
                    pass
                try:
                    ticket = self._queue.get(timeout=0.002)
                except queue.Empty:
                    continue
                if ticket is None:
                    while not self._control.empty():
                        self._send_service(self._control.get_nowait())
                    return
                frame_id = int(ticket.context.frame_id)
                dequeued = raw()
                prediction = prediction_for_ticket(ticket)
                fd, size = CH.seal_bytes([prediction["labels"]])
                with self._lock:
                    self._pending[frame_id] = ticket
                    self._frames[frame_id] = dict(materialized=self._materialized.pop(frame_id, None), dequeued=dequeued)
                try:
                    self._send_service(dict(op="PREDICTION", frame_id=frame_id, identity=prediction["identity"],
                        records=prediction["records"], size=size, frame_context=PO.context_document(ticket.context),
                        events=dict(prediction_materialized=self._frames[frame_id]["materialized"],
                                    prediction_dequeued=dequeued, prediction_sent=raw())), fd)
                finally:
                    os.close(fd)
        except BaseException as error:
            self._failure = f"sender: {type(error).__name__}: {error}"[:1000]

    def _record_emission(self, ticket):
        key = MP.run4_identity(ticket.envelope)
        with self._lock:
            for record in reversed(self.records):
                if record["frame_id"] == ticket.envelope.frame_id and record["run4_identity"] == key:
                    return record
        return None

    def _receive_loop(self):
        try:
            while not self._stop.is_set():
                try:
                    message, fd = CH.receive(self._sock, deadline=time.monotonic() + 3600)
                except CH.ChannelError:
                    return
                if fd is not None:
                    os.close(fd)
                frame_id = int(message.get("frame_id", -1))
                with self._lock:
                    ticket = self._pending.pop(frame_id, None)
                if ticket is None:
                    continue
                received = raw()
                if message["op"] == "RESULT":
                    feedback, reason = W.quality_feedback(self._spec, message["measurement"], ticket.envelope)
                    if feedback.q_perc != message["q_perc"]:
                        raise RuntimeError("service quality differs from the unchanged feedback formula")
                    self._emit(ticket, feedback, reason, measurement=message["measurement"],
                               timing=dict(service_result_received_raw_ns=received))
                    with self._lock:
                        self.counters["service_results"] += 1
                else:
                    self._emit(ticket, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE,
                               timing=dict(service_unscored_received_raw_ns=received, reason=message.get("reason")))
                    with self._lock:
                        self.counters["service_unscored"] += 1
                emitted = raw()
                record = self._record_emission(ticket)
                self._frames.setdefault(frame_id, {}).update(result_received=received, emitted=emitted,
                    op=message["op"], reason=message.get("reason"))
                self._send_service(dict(op="EMITTED", frame_id=frame_id, raw_ns=emitted,
                    packet_sha256=None if record is None else record["feedback_sha256"],
                    kind=None if record is None else record["kind"], reason=None if record is None else record["reason"]))
        except BaseException as error:
            self._failure = f"receiver: {type(error).__name__}: {error}"[:1000]

    def close(self, timeout_s=None):
        self._queue.put(None)
        self._sender.join(timeout=30)
        deadline = time.monotonic() + (60.0 if timeout_s is None else float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    break
            time.sleep(0.02)
        with self._lock:
            leftover = list(self._pending.values())
            self._pending.clear()
        for ticket in leftover:            # retained drain semantics: explicit fault, never a score
            self._emit(ticket, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE, timing=dict(drained=True))
            with self._lock:
                self.counters["drained_unanswered"] += 1
        self._stop.set()
        if self._sock is not None:
            try:
                self._send_service(dict(op="CLOSE"))
            except BaseException:
                pass
            self._sock.close()
        self._receiver.join(timeout=10)
        with self._lock:
            return {**self.counters, "worker_alive": self._sender.is_alive(), "records": len(self.records),
                    "failure": self._failure, "frames": dict(self._frames)}


@contextmanager
def install_early_gt_edge(*, socket_path, capability_path, connect_timeout_s=300.0):
    if EarlyGtEvaluatorV1._binding is not None:
        raise RuntimeError("nested early GT edge installation")
    original = (E.Run4EvaluatorV2, E.Run4EdgeProcessorV2)
    EarlyGtEvaluatorV1._binding = dict(socket_path=str(socket_path), capability_path=str(capability_path),
                                       connect_timeout_s=float(connect_timeout_s))
    E.Run4EvaluatorV2, E.Run4EdgeProcessorV2 = EarlyGtEvaluatorV1, EarlyGtProcessorV1
    state = dict(schema="splitfusion.exact_gt_live.edge_scope.v1", installed=True)
    try:
        yield state
    finally:
        E.Run4EvaluatorV2, E.Run4EdgeProcessorV2 = original
        instance = EarlyGtEvaluatorV1._instance
        state["evaluator_failure"] = None if instance is None else instance._failure
        EarlyGtEvaluatorV1._binding = None
        EarlyGtEvaluatorV1._instance = None

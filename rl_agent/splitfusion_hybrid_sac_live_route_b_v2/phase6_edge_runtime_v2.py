"""Phase 6 edge service (runs inside ``oai-perception-rx``): SFD4 -> tail -> map.

Everything scientific is the qualified v3 edge, imported unchanged:
``DetachedOptimizedEdgeV3`` (hash-verified frozen perception + AE decoders,
audited static camera registry, ``SingleFiniteCheckProductionSplitCodec``,
detached contextual tail) and its CPU publication.  What is new:

* frames are **SFD4** (SFD3 continuous action identity + exact
  ``FrameContextV1``); the profile is re-resolved with
  ``DynamicExecutionContract.resolve_q_e4`` and every identity field is
  cross-checked before any decode (:class:`Run4EdgeProcessorV2`);
* the contextual tail receives a Run-4 metadata object carrying exactly the
  fields it reads (context, sequence, capture) -- never ``tail(c2, None)`` and
  never a fabricated ``action_id``;
* map publication uses the Run-4 map schema, immediately on the compute
  thread;
* reward-requested frames only are handed to a bounded evaluator thread that
  reads the privileged CARLA GT, turns it into the authoritative Run-4 Q_perc
  and sends compact R4FB feedback to the UE over the OAI downlink (the
  container routes 10.0.0.0/16 via the UPF).  Map publication never waits
  for evaluation.

Importing this module starts nothing and touches no CUDA device.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from phase2_map_sharing.transport import ChunkReassembler, chunk_payload
from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP
from rl_agent.splitfusion_direct_edge_map_v1.endpoint import audit_publisher_destination
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec
from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import require_inner_agreement

from . import continuous_execution_v2 as X
from . import run4_live_wire_v2 as W
from . import run4_map_protocol_v2 as MP

__all__ = [
    "EDGE_CONFIG_SCHEMA",
    "Run4TailMetadataV2",
    "Run4EdgeProcessorV2",
    "EvaluationTicketV2",
    "Run4EvaluatorV2",
    "Run4MapPublisherV2",
    "run4_compute_on_detached_runtime",
    "main",
]

EDGE_CONFIG_SCHEMA = "scenesense.run4_live_v2.phase6_edge_config.v1"
EDGE_REPORT_SCHEMA = "scenesense.run4_live_v2.phase6_edge_report.v1"
EDGE_CONFIG_NAME = "run4_phase6_edge_config.json"
EDGE_REPORT_NAME = "run4_phase6_edge_report.json"


class EdgeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EdgeError(message)


@dataclasses.dataclass(frozen=True)
class Run4TailMetadataV2:
    """Exactly what the contextual tail reads, plus the Run-4 identity."""

    frame_context: FrameContextV1
    sequence_id: int
    capture_timestamp_ns: int
    run4_identity: Mapping[str, Any]
    protocol_version: int = 4


# ---------------------------------------------------------------------------
# Compute core
# ---------------------------------------------------------------------------


def run4_compute_on_detached_runtime(runtime: Any, *, envelope: X.ExecutionEnvelopeV3,
                                     context: FrameContextV1,
                                     profile: dec.ExecutableDispatchProfile,
                                     identity: Mapping[str, Any]) -> Any:
    """The qualified ``process_compute`` sequence, bound to a dynamic profile.

    Mirrors ``DetachedPreloadedSplitEdgeRuntime.process_compute`` step for
    step (camera registry, context session, inner inspection and agreement,
    AE decoder by family, decode, finiteness/device checks, detached tail) but
    takes the already-verified SFD4 envelope/context and an exact
    ``ExecutableDispatchProfile`` instead of re-resolving a catalog id.
    """
    import torch

    from rl_agent.splitfusion_edge_optimization_v1.detached_runtime import (
        DetachedEdgeComputeResult,
    )

    view = X.codec_view(profile)
    timing = StageRecorder(EDGE_STAGES)
    runtime._counters.frames_attempted += 1
    with timing.stage("total_edge_processing"):
        _require(runtime._camera_registry is not None, "edge camera registry missing")
        runtime._camera_registry.resolve(context.camera_model_sha256,
                                         context.camera_mount_sha256)
        runtime._context_session.accept(context)
        inspected = runtime._codec.inspect(envelope.inner_payload, timing=timing)
        require_inner_agreement(view, inspected.identity)
        decoder = None if profile.family == "noAE" else runtime._ae_decoders.get(profile.family)
        _require(profile.family == "noAE" or decoder is not None,
                 f"preloaded {profile.family} decoder unavailable")
        if decoder is not None:
            runtime._counters.ae_decoder_dispatches += 1
        with torch.inference_mode():
            decoded = runtime._codec.decode(inspected, decoder=decoder,
                                            tail_device=runtime._tail_device,
                                            timing=timing)
        _require(bool(decoded.finite), "reconstructed C2 is non-finite")
        _require(decoded.device == runtime._tail_device, "reconstructed C2 device drift")
        metadata = Run4TailMetadataV2(
            frame_context=context, sequence_id=context.sequence_id,
            capture_timestamp_ns=context.capture_timestamp_ns,
            run4_identity=dict(identity))
        with timing.stage("frozen_tail"):
            runtime._counters.tail_dispatches += 1
            with torch.inference_mode():
                work = runtime._detached_tail.compute_product(decoded.c2, metadata)
    runtime._counters.frames_completed += 1
    return DetachedEdgeComputeResult(
        work=work, metadata=metadata,
        scientific_inner_payload_bytes=len(envelope.inner_payload),
        framing_control_overhead_bytes=0, total_received_bytes=0,
        timing=timing.snapshot())


@dataclasses.dataclass(frozen=True)
class EvaluationTicketV2:
    envelope: X.ExecutionEnvelopeV3
    context: FrameContextV1
    records: tuple
    predicted_mask: Any
    gt_identity: Mapping[str, Any]
    enqueued_wall_ns: int


@dataclasses.dataclass(frozen=True)
class ProcessedFrameV2:
    envelope: X.ExecutionEnvelopeV3
    context: FrameContextV1
    profile: dec.ExecutableDispatchProfile
    identity: Mapping[str, Any]
    update: Mapping[str, Any]
    evaluation: Optional[EvaluationTicketV2]


class Run4EdgeProcessorV2:
    """SFD4 -> verified identity -> injected compute -> map update (+ ticket)."""

    def __init__(self, *, contract: dec.DynamicExecutionContract, run_id: str,
                 cell_id: str, service_deadline_s: float, ack_timeout_s: float,
                 compute: Callable[..., tuple[list, Any]]) -> None:
        self._contract = contract
        self.run_id, self.cell_id = str(run_id), str(cell_id)
        self._service_s, self._ack_s = float(service_deadline_s), float(ack_timeout_s)
        self._compute = compute

    def verify(self, frame: bytes) -> tuple[X.ExecutionEnvelopeV3, FrameContextV1,
                                           dec.ExecutableDispatchProfile, dict]:
        envelope, context = W.unpack_sfd4(frame)
        profile = self._contract.resolve_q_e4(envelope.mode_id, envelope.q_e4)
        _require(envelope.execution_bundle_sha256 == profile.execution_bundle_sha256,
                 "execution-bundle digest mismatch")
        _require(envelope.keep_count == profile.keep_count, "keep_count mismatch")
        _require(envelope.anchor_action_id == profile.action_id,
                 "anchor identity mismatch (fabricated or dropped anchor)")
        X.executed_identity_from_profile(profile, self._contract)
        identity = MP.run4_identity(envelope)
        MP.validate_run4_identity(identity, contract=self._contract)
        return envelope, context, profile, identity

    def deadlines(self, capture_ns: int) -> tuple[float, float]:
        base = capture_ns / 1_000_000_000.0
        return base + self._service_s, base + self._ack_s

    def process(self, frame: bytes, *, edge_timing: Mapping[str, Any],
                segmentation: Mapping[str, Any] | None = None) -> ProcessedFrameV2:
        envelope, context, profile, identity = self.verify(frame)
        records, labels = self._compute(envelope=envelope, context=context,
                                        profile=profile, identity=identity)
        service_at, ack_at = self.deadlines(context.capture_timestamp_ns)
        update = MP.build_run4_map_update(
            run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
            frame_id=context.frame_id, sequence_id=context.sequence_id,
            identity=identity, decoder_identity=profile.decoder_identity,
            capture_timestamp_ns=context.capture_timestamp_ns, records=records,
            service_deadline_at=service_at, ack_timeout_at=ack_at,
            edge_timing=edge_timing, segmentation=segmentation or {"available": False})
        ticket = None
        if envelope.reward_requested:
            ticket = EvaluationTicketV2(
                envelope=envelope, context=context, records=tuple(records),
                predicted_mask=labels,
                gt_identity=MP.gt_identity(
                    run_id=self.run_id, cell_id=self.cell_id,
                    stream_id=context.stream_id, frame_id=context.frame_id,
                    anchor_action_id=profile.action_id,
                    anchor_profile_id=profile.profile_id,
                    capture_timestamp_ns=context.capture_timestamp_ns),
                enqueued_wall_ns=time.time_ns())
        return ProcessedFrameV2(envelope=envelope, context=context, profile=profile,
                                identity=identity, update=update, evaluation=ticket)

    def terminal(self, frame: bytes, *, outcome: str, stage: str,
                 superseded_by_frame_id: Optional[int] = None) -> dict[str, Any]:
        envelope, context, _profile, identity = self.verify(frame)
        service_at, ack_at = self.deadlines(context.capture_timestamp_ns)
        return MP.build_run4_edge_terminal(
            run_id=self.run_id, cell_id=self.cell_id, stream_id=context.stream_id,
            identity=identity, capture_timestamp_ns=context.capture_timestamp_ns,
            service_deadline_at=service_at, ack_timeout_at=ack_at, outcome=outcome,
            stage=stage, age_ms=(time.time() - context.capture_timestamp_ns / 1e9) * 1e3,
            emit_at=time.time(), superseded_by_frame_id=superseded_by_frame_id)


# ---------------------------------------------------------------------------
# Evaluator (reward-requested frames only, asynchronous)
# ---------------------------------------------------------------------------


class Run4EvaluatorV2:
    """Bounded asynchronous exact-quality evaluator emitting R4FB feedback."""

    def __init__(self, *, spec: Any, read_ground_truth: Callable[..., Mapping[str, Any]],
                 send: Callable[[bytes], None], match_distance_m: float,
                 gt_timeout_s: float, queue_depth: int = 64) -> None:
        self._spec = spec
        self._read = read_ground_truth
        self._send = send
        self._match = float(match_distance_m)
        self._gt_timeout = float(gt_timeout_s)
        self._queue: "queue.Queue[Optional[EvaluationTicketV2]]" = queue.Queue(
            maxsize=int(queue_depth))
        self.records: list[dict[str, Any]] = []
        self.counters: dict[str, int] = {"submitted": 0, "emitted": 0,
                                         "queue_overflow": 0, "excluded": 0}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="run4-evaluator",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, ticket: EvaluationTicketV2) -> None:
        _require(ticket.envelope.reward_requested, "only reward frames are evaluated")
        try:
            self._queue.put_nowait(ticket)
        except queue.Full:
            with self._lock:
                self.counters["queue_overflow"] += 1
            # No silent drop: the UE learns it as an excluded infrastructure fault.
            self._emit(ticket, None, W.EvaluatorReason.EVALUATOR_EXCEPTION,
                       infrastructure=True)
            return
        with self._lock:
            self.counters["submitted"] += 1

    def evaluate(self, ticket: EvaluationTicketV2) -> None:
        try:
            gt = self._read(expected_identity=ticket.gt_identity,
                            timeout_s=self._gt_timeout)
        except Exception:  # noqa: BLE001 - GT absence is an evaluator fault
            self._emit(ticket, None, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
            return
        measurement = W.live_measurement(
            frame_id=ticket.context.frame_id, predicted_mask=ticket.predicted_mask,
            ground_truth_mask=gt["semantic"], predictions=list(ticket.records),
            ground_truth_objects=list(gt["objects"]), match_distance_m=self._match)
        feedback, reason = W.quality_feedback(self._spec, measurement, ticket.envelope)
        self._emit(ticket, feedback, reason, measurement=measurement)

    def _emit(self, ticket, feedback, reason, *, measurement=None,
              infrastructure: bool = False) -> None:
        from . import reward_hold_controller_v2 as R

        if feedback is None:
            env = ticket.envelope
            feedback = R.RewardFeedbackV2(
                session_uuid=env.session_uuid,
                controller_lineage_sha256=env.controller_lineage_sha256,
                decision_seq=env.decision_seq, ticket_seq=env.ticket_seq,
                frame_id=env.frame_id, tensor_seq=env.tensor_seq,
                capture_timestamp_ns=env.capture_timestamp_ns, mode_id=env.mode_id,
                q_e4=env.q_e4, execution_bundle_sha256=env.execution_bundle_sha256,
                anchor_action_id=env.anchor_action_id, reward_requested=True,
                kind="INFRASTRUCTURE_FAULT" if infrastructure else "EVALUATOR_FAULT",
                q_perc=None)
        payload = W.encode_feedback(feedback, reason)
        self._send(payload)
        with self._lock:
            self.counters["emitted"] += 1
            if feedback.kind != "DELIVERED_SUCCESS":
                self.counters["excluded"] += 1
            self.records.append({
                "frame_id": ticket.envelope.frame_id,
                "run4_identity": MP.run4_identity(ticket.envelope),
                "kind": feedback.kind, "q_perc": feedback.q_perc,
                "reason": reason.name, "emit_wall_ns": time.time_ns(),
                "enqueued_wall_ns": ticket.enqueued_wall_ns,
                "feedback_sha256": hashlib.sha256(payload).hexdigest(),
                "measurement": measurement,
                "measurement_semantics": W.LIVE_MEASUREMENT_SEMANTICS,
            })

    def _run(self) -> None:
        while True:
            ticket = self._queue.get()
            if ticket is None:
                return
            try:
                self.evaluate(ticket)
            except Exception:  # noqa: BLE001
                self._emit(ticket, None, W.EvaluatorReason.EVALUATOR_EXCEPTION)

    def close(self, timeout_s: float = 10.0) -> dict[str, Any]:
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        with self._lock:
            return {**self.counters, "worker_alive": self._thread.is_alive(),
                    "records": len(self.records)}


class Run4MapPublisherV2:
    """Legacy chunk/zlib publication path for the Run-4 update schema."""

    def __init__(self, *, map_host: str, map_port: int, chunk_bytes: int,
                 socket_buffer_request_bytes: int) -> None:
        import zlib

        audit_publisher_destination(map_host, map_port)
        self._zlib = zlib
        self.remote = (str(map_host), int(map_port))
        self.chunk_bytes = int(chunk_bytes)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                               int(socket_buffer_request_bytes))
        self.published = 0

    def publish(self, document: Mapping[str, Any]) -> dict[str, Any]:
        start = time.time()
        MP.validate_run4_map_update(document)
        stamped = dict(document)
        timing = dict(stamped.get("edge_timing") or {})
        timing["publish_start_wall_s"] = start
        timing["serialization_start_wall_s"] = time.time()
        stamped["edge_timing"] = timing
        payload = self._zlib.compress(DP.encode(stamped), level=1)
        chunks = chunk_payload(payload, message_id=int(stamped["frame_id"]),
                               chunk_bytes=self.chunk_bytes)
        for chunk in chunks:
            self.socket.sendto(chunk, self.remote)
        self.published += 1
        return {"publish_start_wall_s": start, "last_datagram_send_wall_s": time.time(),
                "direct_map_datagrams": len(chunks), "direct_map_payload_bytes": len(payload)}

    def close(self) -> None:
        self.socket.close()


def ready_document(*, action_id: int, direct_map_host: str, direct_map_port: int,
                   ue_control_host: str, ue_control_port: int, tail_device: Any,
                   quality_spec_sha256: str, evidence_dir: Any) -> dict[str, Any]:
    """The edge ready record (pure; the live write site uses exactly this).

    Addendum 4 adds three metadata fields that ``adapter_direct_v1`` requires:
    the dense predicted mask stays at the edge for the asynchronous evaluator
    (never returned to the UE); object records travel edge-to-map directly (only
    compact quality/terminal feedback travels edge-to-UE); and the evaluator's
    evidence directory is the schema-validated ``config["evidence_dir"]``.
    """
    return {"schema": "splitfusion_direct_live_edge_ready.v1",
            "run4_edge": True, "action_id": int(action_id),
            "direct_map_host": str(direct_map_host),
            "direct_map_port": int(direct_map_port),
            "ue_control_host": str(ue_control_host),
            "ue_control_port": int(ue_control_port),
            "tail_device": str(tail_device), "architecture": "DIRECT_EDGE_TO_MAP_V1",
            "quality_spec_sha256": str(quality_spec_sha256),
            "dense_label_map_on_radio": False,
            "object_records_on_radio": False,
            "evaluation_evidence_dir": str(evidence_dir)}


# ---------------------------------------------------------------------------
# Container service
# ---------------------------------------------------------------------------


def run_run4_edge_service(args: argparse.Namespace) -> int:  # pragma: no cover - live
    import torch

    from rl_agent.splitfusion_direct_edge_map_v1.direct_v3_edge import (
        _DeadlineGuardedComputeTail,
    )
    from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
        preload_detached_optimized_edge_v3,
    )
    from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import (
        DetachedOptimizedTailAdapterV3,
    )
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence

    config = json.loads(Path(args.run4_config).read_text(encoding="utf-8"))
    _require(config.get("schema") == EDGE_CONFIG_SCHEMA, "edge config schema drift")
    campaign = base._load_json(args.config)
    runtime_cfg = campaign["runtime"]
    _require(torch.cuda.is_available()
             and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090",
             "edge CUDA device unavailable")
    device = torch.device("cuda:0")
    service_s = base.service_deadline_s(campaign)
    ack_s = base.ack_timeout_s(campaign)
    contract = dec.load_dynamic_execution_contract()
    spec = W.load_run4_quality_spec(Path(__file__).resolve().parents[2])
    _require(str(args.ue_control_host) != str(args.direct_map_host)
             or int(args.ue_control_port) != int(args.direct_map_port),
             "feedback and map endpoints must be distinct")

    def guard(stage: str, capture_ns: int) -> None:
        base.check_deadline(stage, capture_ns, ack_s)

    edge = preload_detached_optimized_edge_v3(device)   # hash-verified, once
    rt = edge.runtime
    rt._detached_tail = _DeadlineGuardedComputeTail(edge.tail, guard)

    def compute(*, envelope, context, profile, identity):
        computed = run4_compute_on_detached_runtime(
            rt, envelope=envelope, context=context, profile=profile, identity=identity)
        published = rt.publish_cpu(computed)
        snapshot = DetachedOptimizedTailAdapterV3.snapshot(published.serialized)
        labels = (snapshot.semantic_labels.detach()
                  .to(device="cpu", dtype=torch.uint8).contiguous().numpy())
        return list(snapshot.records or ()), labels

    processor = Run4EdgeProcessorV2(contract=contract, run_id=args.run_id,
                                    cell_id=args.cell_id, service_deadline_s=service_s,
                                    ack_timeout_s=ack_s, compute=compute)
    publisher = Run4MapPublisherV2(
        map_host=args.direct_map_host, map_port=args.direct_map_port,
        chunk_bytes=int(runtime_cfg["udp_chunk_bytes"]),
        socket_buffer_request_bytes=int(runtime_cfg["socket_buffer_request_bytes"]))
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ue_remote = (str(args.ue_control_host), int(args.ue_control_port))
    evidence_dir = Path(config["evidence_dir"])
    evaluator = Run4EvaluatorV2(
        spec=spec, send=lambda payload: control.sendto(payload, ue_remote),
        read_ground_truth=lambda **kw: gt_evidence.read_ground_truth(evidence_dir, **kw),
        match_distance_m=float(config["match_distance_m"]),
        gt_timeout_s=float(config["gt_timeout_s"]), queue_depth=int(config["queue_depth"]))
    evaluator.start()

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                        int(runtime_cfg["socket_buffer_request_bytes"]))
    receiver.bind(("0.0.0.0", int(args.edge_port)))
    receiver.settimeout(0.25)
    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    pending = base.LatestFramePendingSlot()
    stop = threading.Event()
    counters: dict[str, int] = {}
    failures: list[str] = []

    def bump(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    def send_terminal(frame: bytes, outcome: str, stage: str, superseded=None) -> None:
        try:
            control.sendto(DP.encode(processor.terminal(
                frame, outcome=outcome, stage=stage,
                superseded_by_frame_id=superseded)), ue_remote)
            bump(f"edge_terminal_{outcome}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"terminal: {exc}")

    def receive_loop() -> None:
        while not stop.is_set():
            try:
                datagram, address = receiver.recvfrom(65535)
            except socket.timeout:
                reassembler.expire(time.monotonic())
                continue
            except OSError:
                return
            try:
                complete = reassembler.ingest(str(address), datagram,
                                              received_at_s=time.monotonic())
            except ValueError:
                bump("feature_datagrams_malformed")
                continue
            if complete is None:
                continue
            try:
                envelope, context = W.unpack_sfd4(complete.payload)
            except Exception:  # noqa: BLE001 - corrupt frames fail closed
                bump("sfd4_rejected")
                continue
            if context.frame_id != int(complete.message_id):
                bump("frame_identity_rejected")
                continue
            item = {"payload": complete.payload, "stream_id": context.stream_id,
                    "capture_ns": context.capture_timestamp_ns,
                    "received_wall_s": time.time(), "frame_id": context.frame_id}
            try:
                base.check_deadline(base.EDGE_STAGE_AFTER_REASSEMBLY,
                                    context.capture_timestamp_ns, ack_s,
                                    now_s=item["received_wall_s"])
            except base.DeadlineExpired as expired:
                send_terminal(complete.payload, DP.OUTCOME_STALE_BEFORE_EDGE, expired.stage)
                continue
            admitted, displaced = pending.offer(context.stream_id, item,
                                                sequence=int(context.sequence_id))
            if not admitted:
                send_terminal(complete.payload, DP.OUTCOME_SUPERSEDED_PENDING,
                              "EDGE_ADMISSION_NOT_FRESHEST")
                continue
            if displaced is not None:
                send_terminal(displaced["payload"], DP.OUTCOME_SUPERSEDED_PENDING,
                              "EDGE_PENDING_REPLACED", superseded=context.frame_id)

    def process_loop() -> None:
        while not stop.is_set():
            taken = pending.take(timeout=0.1)
            if taken is None:
                continue
            item = taken[1]
            try:
                base.check_deadline(base.EDGE_STAGE_BEFORE_DECODE, item["capture_ns"], ack_s)
                started = time.time()
                processed = processor.process(item["payload"], edge_timing={
                    "reassembly_complete_wall_s": item["received_wall_s"],
                    "compute_start_wall_s": started})
            except base.DeadlineExpired as expired:
                send_terminal(item["payload"], DP.OUTCOME_STALE_BEFORE_MAP, expired.stage)
                continue
            except Exception as exc:  # noqa: BLE001
                bump("edge_processing_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            try:
                publisher.publish(processed.update)          # immediate map branch
                bump("direct_map_publications")
            except Exception as exc:  # noqa: BLE001
                bump("direct_map_publication_failed")
                failures.append(f"publish: {exc}")
            if processed.evaluation is not None:              # asynchronous quality
                evaluator.submit(processed.evaluation)
                bump("evaluations_submitted")

    ready = Path(args.ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    with ready.open("x", encoding="utf-8") as handle:
        json.dump(ready_document(
            action_id=args.action_id, direct_map_host=args.direct_map_host,
            direct_map_port=args.direct_map_port, ue_control_host=args.ue_control_host,
            ue_control_port=args.ue_control_port, tail_device=device,
            quality_spec_sha256=spec.canonical_sha256(), evidence_dir=evidence_dir),
            handle, sort_keys=True)
    threads = [threading.Thread(target=receive_loop, daemon=True, name="run4-edge-receive"),
               threading.Thread(target=process_loop, daemon=True, name="run4-edge-process")]
    for thread in threads:
        thread.start()
    report_path = Path(config["report_path"])

    def write_report(final: bool) -> None:
        document = {"schema": EDGE_REPORT_SCHEMA, "final": final,
                    "run_id": args.run_id, "cell_id": args.cell_id,
                    "counters": dict(counters), "failures": failures[:32],
                    "evaluator": dict(evaluator.counters),
                    "evaluations": list(evaluator.records),
                    "publications": publisher.published,
                    "source_sha256": {name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("phase6_edge_runtime_v2.py", "run4_live_wire_v2.py",
                                     "run4_map_protocol_v2.py")},
                    "written_at_unix_s": time.time()}
        temporary = report_path.with_name(report_path.name + ".partial")
        temporary.write_text(json.dumps(document, sort_keys=True, default=str),
                             encoding="utf-8")
        temporary.replace(report_path)

    import signal

    def shutdown(_signum, _frame) -> None:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
        evaluator.close()
        write_report(True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    while all(thread.is_alive() for thread in threads):
        time.sleep(1.0)
        write_report(False)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--action-id", type=int)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--direct-map-host", default="")
    parser.add_argument("--direct-map-port", type=int, default=0)
    parser.add_argument("--ue-control-host", default="10.0.0.2")
    parser.add_argument("--ue-control-port", type=int, default=51014)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cell-id", default="")
    parser.add_argument("--run4-config", default=f"/work/torch_cache/{EDGE_CONFIG_NAME}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    args, _ignored = build_parser().parse_known_args(list(argv) if argv is not None else None)
    _require(bool(args.edge) and args.config is not None and args.ready_file is not None
             and args.action_id is not None, "edge mode and bindings are required")
    _require(bool(str(args.direct_map_host).strip()) and int(args.direct_map_port) > 0,
             "the direct edge-to-map endpoint is required")
    return run_run4_edge_service(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

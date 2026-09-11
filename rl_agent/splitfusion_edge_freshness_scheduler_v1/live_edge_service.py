#!/usr/bin/env python3
"""Live edge candidate with latest-only compute and publication scheduling."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from phase2_map_sharing.transport import (
    CHUNK_HEADER,
    ChunkReassembler,
    chunk_payload,
)
from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload import (
    preload_detached_optimized_edge,
)
from rl_agent.splitfusion_edge_optimization_v1.optimized_tail import (
    tree_bitwise_equal,
)
from rl_agent.splitfusion_live_dispatch_v1.envelope import unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    EDGE_RESULT_SCHEMA,
    EDGE_TERMINAL_ACK_SCHEMA,
    OBJECT_MAP_UPDATE_SCHEMA,
    _Counters,
    _finite_tree,
    _trace_ns,
    ack_timeout_s,
    deadline_at_s,
    service_deadline_s,
)
from rl_agent.splitfusion_live_dispatch_v1.timing import TimingTrace
from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common
from rl_agent.splitfusion_timing_diagnostic_v1 import edge_service as diagnostic

from .pipeline import (
    BoundedTwoStagePipeline,
    CandidatePolicy,
    PipelineConfig,
    PipelineWorkerError,
)
from .scheduler import FrameTicket, TerminalFeedback, TerminalReason


WARMUP_PHASE = "WARMUP"
MEASURED_PHASE = "MEASURED"


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _combined_timing(compute: TimingTrace, publication: TimingTrace) -> TimingTrace:
    common.require(
        compute.clock == publication.clock,
        "compute/publication timing clock drift",
    )
    return TimingTrace(
        clock=compute.clock,
        boundaries=compute.boundaries + publication.boundaries,
        latency_published=False,
    )


def _parity_and_warmup(
    edge: Any,
    payloads: list[bytes],
    *,
    action_id: int,
) -> tuple[dict[str, Any], list[float]]:
    c2, metadata = diagnostic._decode_for_equivalence(
        edge, payloads[0], action_id
    )
    with torch.inference_mode():
        reference = edge.reference_tail(c2, metadata)
        reference_bytes = edge.reference_tail.serialize(reference)
        reference_snapshot = edge.reference_tail.take_snapshot()
        candidate = edge.tail.compute_product(c2, metadata)
        candidate_serialized = edge.tail.serialize_product(candidate)
    common.require(
        tree_bitwise_equal(reference, candidate.perception),
        "detached live warm-up perception parity failed",
    )
    common.require(
        reference_bytes == candidate_serialized.serialized_records,
        "detached live warm-up service-record byte parity failed",
    )
    common.require(
        torch.equal(reference_snapshot.original_indices, candidate.original_indices),
        "detached live warm-up p025-index parity failed",
    )
    common.require(
        torch.equal(reference_snapshot.semantic_labels, candidate.semantic_labels),
        "detached live warm-up segmentation parity failed",
    )
    equivalence = {
        "perception_bitwise_identical": True,
        "service_records_byte_identical": True,
        "p025_indices_bitwise_identical": True,
        "segmentation_labels_bitwise_identical": True,
    }
    warmup_ms: list[float] = []
    for payload in payloads:
        started = time.time_ns()
        edge.tail.begin_frame()
        computed = edge.runtime.process_compute(
            payload, transmitted_action_id=action_id
        )
        edge.tail.resolve_frame()
        published = edge.runtime.publish_cpu(computed)
        _finite_tree(computed.work.perception)
        common.require(
            published.serialized.serialized_records,
            "warm-up produced no serialized service records",
        )
        warmup_ms.append((time.time_ns() - started) / 1e6)
    return equivalence, warmup_ms


def run(
    args: argparse.Namespace,
    *,
    policy: CandidatePolicy,
    pipeline_config_factory: Callable[
        [argparse.Namespace, CandidatePolicy, int], PipelineConfig
    ]
    | None = None,
) -> int:
    campaign = common.load_json(Path(args.config).resolve(strict=True))
    runtime = campaign["runtime"]
    common.require(
        int(runtime["sfd1_protocol_version"]) == 2
        and runtime["udp_fragment_header"] == "!IHH",
        "edge protocol binding drift",
    )
    common.require(
        torch.cuda.is_available()
        and torch.cuda.get_device_name(0) == common.DEVICE_NAME,
        "edge CUDA device unavailable or not the bound RTX 5090",
    )
    device = torch.device("cuda:0")
    start_anchor = common.clock_anchor("freshness_edge_process_start")
    allowed = tuple(
        int(value)
        for value in str(args.allowed_action_ids or args.action_id).split(",")
    )
    common.require(
        bool(allowed) and len(allowed) == len(set(allowed)),
        "edge action allowlist is empty or duplicated",
    )
    common.require(
        int(args.action_id) in allowed,
        "edge fixed action is outside its allowlist",
    )

    edge = preload_detached_optimized_edge(device)
    profiles = {value: edge.registry.resolve(value) for value in allowed}
    service_s = service_deadline_s(campaign)
    horizon_s = ack_timeout_s(campaign)
    counters = _Counters()
    terminal_reason = ""
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    outcome_counts: Counter[str] = Counter()
    outcome_class_counts: Counter[str] = Counter()
    outcome_lock = threading.Lock()
    send_lock = threading.Lock()

    records_path = Path(args.records_file)
    summary_path = Path(args.summary_file)
    ready_path = Path(args.ready_file)
    stop_path = Path(args.stop_file) if args.stop_file else None
    warmup_blob = Path(args.warmup_payload)
    warmup_index = common.load_json(warmup_blob.with_name("warmup_index.json"))
    raw = warmup_blob.read_bytes()
    warmup_payloads = [
        raw[offset : offset + size]
        for offset, size in zip(warmup_index["offsets"], warmup_index["sizes"])
    ]
    del raw
    warmup_iterations = int(args.warmup_iterations)
    common.require(
        1 <= warmup_iterations <= len(warmup_payloads)
        and all(warmup_payloads),
        "warm-up payload inventory is invalid",
    )
    equivalence, warmup_wall_ms = _parity_and_warmup(
        edge,
        warmup_payloads[:warmup_iterations],
        action_id=int(args.action_id),
    )
    del warmup_payloads
    torch.cuda.synchronize(device)

    request = int(runtime["socket_buffer_request_bytes"])
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
    receiver.bind(("0.0.0.0", int(args.edge_port)))
    receiver.settimeout(0.25)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    chunk_bytes = int(runtime["udp_chunk_bytes"])
    stop_event = threading.Event()
    first_datagram_wall_ns: dict[tuple[str, int], int] = {}
    expired_seen = 0

    def request_stop(signum: int, _frame: Any) -> None:
        counters.bump(f"shutdown_signal_{int(signum)}")
        stop_event.set()

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, request_stop)

    def send_blob(blob: bytes, *, frame_id: int) -> int:
        chunks = chunk_payload(blob, message_id=frame_id, chunk_bytes=chunk_bytes)
        with send_lock:
            for chunk in chunks:
                sender.sendto(
                    chunk, (str(args.result_host), int(args.result_port))
                )
                counters.bump("result_datagrams_transmitted")
        return len(chunks)

    def feedback_sink(feedback: TerminalFeedback) -> None:
        with outcome_lock:
            outcome_counts[feedback.reason.value] += 1
            outcome_class_counts[feedback.outcome_class.value] += 1
        if feedback.reason is TerminalReason.RESULT_PUBLISHED:
            return
        if feedback.reason is TerminalReason.SUPERSEDED_PENDING:
            counters.bump("edge_pending_replacements")
        if feedback.reason is TerminalReason.SUPERSEDED_PUBLICATION_PENDING:
            counters.bump("edge_publication_pending_replacements")
        payload = feedback.to_dict()
        payload["profile_id"] = profiles[feedback.ticket.action_id].profile_id
        payload["agent_feedback_semantics"] = (
            "INTENTIONAL_SUPERSESSION_IS_NOT_RADIO_FAILURE"
            if feedback.agent_credit().intentional_freshness_drop
            else "EXPLICIT_NON_INSTALL_TERMINAL"
        )
        blob = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        send_blob(blob, frame_id=feedback.ticket.frame_id)
        counters.bump("scheduler_terminal_feedback_transmitted")
        counters.bump("scheduler_terminal_feedback_bytes", len(blob))

    def compute(ticket: FrameTicket, item: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.time_ns()
        counters.bump("edge_process_starts")
        edge.tail.begin_frame()
        result = edge.runtime.process_compute(
            item["payload"], transmitted_action_id=int(item["action_id"])
        )
        finished_ns = time.time_ns()
        stage_timings = edge.tail.resolve_frame()
        _finite_tree(result.work.perception)
        context = result.metadata.frame_context
        common.require(context is not None, "computed result lacks frame context")
        common.require(
            context.stream_id == ticket.stream_id
            and context.frame_id == ticket.frame_id
            and result.metadata.sequence_id == ticket.sequence_id
            and result.total_received_bytes == ticket.feature_bytes,
            "pipeline compute identity/accounting drift",
        )
        counters.bump("tail_completions")
        return {
            "item": item,
            "result": result,
            "worker_start_wall_ns": started_ns,
            "tail_finished_wall_ns": finished_ns,
            "tail_finished_perf_ns": time.perf_counter_ns(),
            "tail_stage": stage_timings,
        }

    def publish(ticket: FrameTicket, bundle: dict[str, Any]) -> None:
        item = bundle["item"]
        computed = bundle["result"]
        published = edge.runtime.publish_cpu(computed)
        snapshot = edge.tail.snapshot(published.serialized)
        context = computed.metadata.frame_context
        assert context is not None
        profile = profiles[int(item["action_id"])]
        common.require(
            profile.profile_id == computed.metadata.profile_id,
            "published profile identity drift",
        )
        service_records = list(snapshot.records or ())
        serialized_bytes = len(snapshot.serialized_records or b"")
        capture_ns = int(item["capture_timestamp_ns"])
        phase = (
            WARMUP_PHASE
            if str(args.warmup_stream_id)
            and str(context.stream_id) == str(args.warmup_stream_id)
            else MEASURED_PHASE
        )
        combined_timing = _combined_timing(computed.timing, published.timing)
        record = {
            "schema": common.EDGE_RECORD_SCHEMA,
            "run_id": str(args.run_id),
            "cell_id": str(args.cell_id),
            "phase": phase,
            "action_id": int(profile.action_id),
            "profile_id": profile.profile_id,
            "family": profile.family,
            "quantizer": profile.quantizer,
            "q_e4": int(profile.q_e4),
            "stream_id": str(context.stream_id),
            "frame_id": int(context.frame_id),
            "sequence_id": int(item["sequence_id"]),
            "message_id": int(item["message_id"]),
            "capture_timestamp_ns": capture_ns,
            "feature_datagrams": int(item["chunk_count"]),
            "duplicate_datagrams": int(item["duplicate_chunks"]),
            "scientific_inner_payload_bytes": int(
                computed.scientific_inner_payload_bytes
            ),
            "framing_control_overhead_bytes": int(
                computed.framing_control_overhead_bytes
            ),
            "total_received_bytes": int(computed.total_received_bytes),
            "edge_first_datagram_wall_ns": int(item["edge_first_datagram_wall_ns"]),
            "edge_complete_reassembly_wall_ns": int(
                item["edge_complete_reassembly_wall_ns"]
            ),
            "edge_admitted_wall_ns": int(item["edge_admitted_wall_ns"]),
            "edge_worker_start_wall_ns": int(bundle["worker_start_wall_ns"]),
            "edge_tail_finished_wall_ns": int(bundle["tail_finished_wall_ns"]),
            "edge_result_finite_check_ms": 0.0,
            "edge_service_wall_ms": (
                int(bundle["tail_finished_wall_ns"])
                - int(bundle["worker_start_wall_ns"])
            ) / 1e6,
            "edge_stage_ns": _trace_ns(combined_timing),
            "tail_stage_wall_ns": bundle["tail_stage"]["wall_ns"],
            "tail_stage_cuda_ms": bundle["tail_stage"]["cuda_ms"],
            "reconstructed_device": str(edge.runtime.tail_device),
            "finite_output_tensor_count": int(snapshot.output_tensor_count),
            "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
            "service_record_count": len(service_records),
            "service_record_bytes": serialized_bytes,
            "semantic_label_shape": [
                int(value) for value in snapshot.semantic_labels.shape
            ],
            "service_deadline_at_s": deadline_at_s(capture_ns, service_s),
            "processing_horizon_at_s": deadline_at_s(capture_ns, horizon_s),
            "service_target_met": False,
            "processing_horizon_met": True,
            "scheduler_policy": policy.value,
        }
        terminal_ack = {
            "schema": EDGE_TERMINAL_ACK_SCHEMA,
            "run_id": str(args.run_id),
            "cell_id": str(args.cell_id),
            "action_id": int(profile.action_id),
            "stream_id": str(context.stream_id),
            "frame_id": int(context.frame_id),
            "capture_timestamp_ns": capture_ns,
            "capture_wall_s": capture_ns / 1e9,
            "edge_receipt_wall_s": float(item["edge_received_wall_s"]),
            "tail_complete_wall_s": int(bundle["tail_finished_wall_ns"]) / 1e9,
            "evidence_install_wall_s": "",
            "service_deadline_at": deadline_at_s(capture_ns, service_s),
            "ack_timeout_at": deadline_at_s(capture_ns, horizon_s),
            "installation_status": "EVALUATION_EVIDENCE_NOT_CONFIGURED",
            "evidence": {},
            "terminal_reason": "EDGE_RESULT_PUBLISHED_AWAITING_MAP_ACK",
        }
        downlink = {
            "schema": EDGE_RESULT_SCHEMA,
            "action_id": profile.action_id,
            "profile_id": profile.profile_id,
            "decoder_identity": profile.decoder_identity,
            "stream_id": context.stream_id,
            "frame_id": context.frame_id,
            "capture_timestamp_ns": capture_ns,
            "finite": True,
            "frame_context_valid": True,
            "reconstructed_device": str(edge.runtime.tail_device),
            "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
            "finite_output_tensor_count": int(snapshot.output_tensor_count),
            "service_record_count": len(service_records),
            "feature_received_datagrams": int(item["chunk_count"]),
            "feature_duplicate_datagrams": int(item["duplicate_chunks"]),
            "edge_call_ledger": edge.ledger.snapshot(),
            "edge_counters": {
                **dict(edge.runtime.counters.__dict__),
                **counters.snapshot(),
            },
            "edge_received_ns": int(item["edge_received_perf_ns"]),
            "tail_finished_ns": int(bundle["tail_finished_perf_ns"]),
            "edge_timing_ns": _trace_ns(combined_timing),
            "object_map_update": {
                "schema": OBJECT_MAP_UPDATE_SCHEMA,
                "stream_id": str(context.stream_id),
                "frame_id": int(context.frame_id),
                "capture_timestamp_ns": capture_ns,
                "action_id": int(profile.action_id),
                "records": service_records,
            },
            "edge_terminal_ack": terminal_ack,
            "scheduler": {
                "policy": policy.value,
                "edge_terminal_state": "RESULT_PUBLISHED_AWAITING_MAP_ACK",
            },
            "diagnostic": {
                "phase": phase,
                "sequence_id": int(item["sequence_id"]),
                "service_record_bytes": serialized_bytes,
                "edge_worker_start_wall_ns": int(bundle["worker_start_wall_ns"]),
                "edge_tail_finished_wall_ns": int(bundle["tail_finished_wall_ns"]),
                "edge_service_wall_ms": record["edge_service_wall_ms"],
                "dense_label_map_on_radio": False,
            },
        }
        blob = json.dumps(
            downlink, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        result_datagrams = send_blob(blob, frame_id=int(context.frame_id))
        published_wall_ns = time.time_ns()
        counters.bump("compact_results_transmitted")
        counters.bump("result_bytes_transmitted", len(blob))
        record["edge_result_published_wall_ns"] = published_wall_ns
        record["deployed_tail_service_ms"] = (
            published_wall_ns - int(bundle["worker_start_wall_ns"])
        ) / 1e6
        record["result_datagrams"] = result_datagrams
        record["service_target_met"] = (
            published_wall_ns / 1e9 <= deadline_at_s(capture_ns, service_s)
        )
        record["processing_horizon_met"] = (
            published_wall_ns / 1e9 <= deadline_at_s(capture_ns, horizon_s)
        )
        if phase == MEASURED_PHASE:
            with records_lock:
                records.append(record)

    processing_horizon_ns = int(round(horizon_s * 1e9))
    pipeline_config = (
        PipelineConfig(
            policy=policy,
            processing_horizon_ns=processing_horizon_ns,
        )
        if pipeline_config_factory is None
        else pipeline_config_factory(args, policy, processing_horizon_ns)
    )
    common.require(
        pipeline_config.policy is policy
        and pipeline_config.processing_horizon_ns == processing_horizon_ns,
        "edge pipeline configuration changed the bound policy or horizon",
    )
    pipeline = BoundedTwoStagePipeline(
        config=pipeline_config,
        compute=compute,
        publish=publish,
        feedback_sink=feedback_sink,
    )
    pipeline.start()

    def reconcile_expiries() -> None:
        nonlocal expired_seen
        observed = int(reassembler.expired_messages)
        if observed != expired_seen:
            counters.bump("incomplete_reassemblies_expired", observed - expired_seen)
            expired_seen = observed

    def receive_loop() -> None:
        while not stop_event.is_set():
            try:
                datagram, address = receiver.recvfrom(65535)
            except socket.timeout:
                reassembler.expire(time.time())
                reconcile_expiries()
                continue
            except OSError:
                return
            datagram_wall_ns = time.time_ns()
            counters.bump("feature_datagrams_received")
            if len(datagram) >= CHUNK_HEADER.size:
                message_id, _index, _total = CHUNK_HEADER.unpack_from(datagram)
                first_datagram_wall_ns.setdefault(
                    (str(address), int(message_id)), datagram_wall_ns
                )
            try:
                complete = reassembler.ingest(
                    str(address), datagram, received_at_s=time.time()
                )
            except ValueError:
                counters.bump("feature_datagrams_malformed")
                continue
            reconcile_expiries()
            counters.set_max(
                "reassembly_pending_high_water", len(reassembler.pending)
            )
            if complete is None:
                continue
            complete_wall_ns = time.time_ns()
            complete_perf_ns = time.perf_counter_ns()
            counters.bump("feature_messages_reassembled")
            counters.bump(
                "feature_datagrams_duplicate", int(complete.duplicate_chunks)
            )
            first_wall_ns = first_datagram_wall_ns.pop(
                (str(address), int(complete.message_id)), datagram_wall_ns
            )
            try:
                outer = unpack_envelope(complete.payload)
            except Exception:
                counters.bump("feature_envelope_rejected")
                continue
            if outer.action_id not in profiles or outer.frame_context is None:
                counters.bump("feature_identity_rejected")
                continue
            context = outer.frame_context
            item = {
                "payload": complete.payload,
                "action_id": int(outer.action_id),
                "message_id": int(complete.message_id),
                "sequence_id": int(outer.sequence_id),
                "capture_timestamp_ns": int(outer.capture_timestamp_ns),
                "chunk_count": int(complete.chunk_count),
                "duplicate_chunks": int(complete.duplicate_chunks),
                "edge_first_datagram_wall_ns": int(first_wall_ns),
                "edge_complete_reassembly_wall_ns": int(complete_wall_ns),
                "edge_received_perf_ns": int(complete_perf_ns),
                "edge_received_wall_s": complete_wall_ns / 1e9,
                "edge_admitted_wall_ns": time.time_ns(),
            }
            ticket = FrameTicket(
                run_id=str(args.run_id),
                cell_id=str(args.cell_id),
                stream_id=str(context.stream_id),
                frame_id=int(context.frame_id),
                sequence_id=int(outer.sequence_id),
                action_id=int(outer.action_id),
                capture_timestamp_ns=int(outer.capture_timestamp_ns),
                edge_arrival_timestamp_ns=int(complete_wall_ns),
                feature_bytes=len(complete.payload),
            )
            try:
                admission = pipeline.offer(ticket, item, now_ns=complete_wall_ns)
            except Exception as exc:
                failures.append(f"pipeline offer: {type(exc).__name__}: {exc}")
                stop_event.set()
                return
            if admission.admitted:
                counters.bump("edge_queue_admissions")
            else:
                counters.bump("edge_admission_refused_not_freshest")

    def publish_records() -> None:
        with records_lock:
            lines = [
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in records
            ]
        _atomic_write_text(
            records_path, "\n".join(lines) + ("\n" if lines else "")
        )

    def summary_payload(final: bool) -> dict[str, Any]:
        snapshot = pipeline.snapshot()
        with outcome_lock:
            reasons = dict(sorted(outcome_counts.items()))
            classes = dict(sorted(outcome_class_counts.items()))
        return {
            "schema": common.EDGE_SUMMARY_SCHEMA,
            "run_id": str(args.run_id),
            "cell_id": str(args.cell_id),
            "action_id": int(args.action_id),
            "profile_id": profiles[int(args.action_id)].profile_id,
            "allowed_action_ids": list(allowed),
            "final": bool(final),
            "tail_device": str(edge.runtime.tail_device),
            "service_deadline_s": service_s,
            "processing_horizon_s": horizon_s,
            "deadline_policy": policy.value,
            "warmup_iterations": warmup_iterations,
            "warmup_wall_ms": warmup_wall_ms,
            "parent_equivalence": equivalence,
            "counters": counters.snapshot(),
            "edge_operation_counters": dict(edge.runtime.counters.__dict__),
            "call_ledger": edge.ledger.snapshot(),
            "incomplete_reassemblies_expired": int(reassembler.expired_messages),
            "reassembly_pending_messages": len(reassembler.pending),
            "pending_depth": snapshot.compute_pending_depth,
            "records_written": len(records),
            "pipeline": dict(snapshot.__dict__),
            "pipeline_terminal_reason_counts": reasons,
            "pipeline_outcome_class_counts": classes,
            "terminal_reason": terminal_reason,
            "failures": failures[:8],
            "start_clock_anchor": start_anchor,
            "clock_anchor": common.clock_anchor("freshness_edge_publish"),
            "updated_at_unix_s": time.time(),
        }

    def publish_summary(final: bool) -> None:
        try:
            _atomic_write_text(
                summary_path,
                json.dumps(summary_payload(final), sort_keys=True, indent=1)
                + "\n",
            )
        except Exception:
            counters.bump("edge_summary_publication_failed")

    ready_path.parent.mkdir(parents=True, exist_ok=True)
    publish_records()
    publish_summary(False)
    receiver_thread = threading.Thread(
        target=receive_loop, name="freshness-edge-receive", daemon=True
    )
    receiver_thread.start()
    with ready_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": common.EDGE_READY_SCHEMA,
                "action_id": int(args.action_id),
                "allowed_action_ids": list(allowed),
                "profiles": {
                    str(key): value.profile_id for key, value in profiles.items()
                },
                "tail_device": str(edge.runtime.tail_device),
                "deadline_policy": policy.value,
                "dense_label_map_on_radio": False,
                "warmup_iterations": warmup_iterations,
                "warmup_wall_ms": warmup_wall_ms,
                "parent_equivalence": equivalence,
                "records_file": str(records_path),
                "summary_file": str(summary_path),
                "first_measured_sequence_id": int(args.first_measured_sequence_id),
                "warmup_stream_id": str(args.warmup_stream_id),
                "edge_receive_reported_bytes": receiver.getsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF
                ),
                "edge_send_reported_bytes": sender.getsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF
                ),
                "start_clock_anchor": start_anchor,
                "ready_clock_anchor": common.clock_anchor("freshness_edge_ready"),
            },
            handle,
            sort_keys=True,
        )
    receiver_error = False
    try:
        while True:
            snapshot = pipeline.snapshot()
            if stop_event.is_set():
                terminal_reason = "STOP_REQUESTED"
                break
            if stop_path is not None and stop_path.exists():
                terminal_reason = "STOP_FILE_OBSERVED"
                break
            if not receiver_thread.is_alive() or snapshot.fatal_error:
                terminal_reason = "WORKER_THREAD_EXITED"
                receiver_error = True
                break
            time.sleep(0.25)
            publish_records()
            publish_summary(False)
        return 1 if receiver_error else 0
    finally:
        stop_event.set()
        receiver.close()
        receiver_thread.join(timeout=3.0)
        try:
            pipeline.close_and_join(timeout_s=30.0)
        except (PipelineWorkerError, TimeoutError) as exc:
            failures.append(f"pipeline drain: {type(exc).__name__}: {exc}")
        publish_records()
        publish_summary(True)
        sender.close()


def build_parser() -> argparse.ArgumentParser:
    return diagnostic.build_parser()


def main(
    argv: list[str] | None = None,
    *,
    forced_policy: CandidatePolicy | None = None,
    pipeline_config_factory: Callable[
        [argparse.Namespace, CandidatePolicy, int], PipelineConfig
    ]
    | None = None,
) -> int:
    args, _ignored = build_parser().parse_known_args(argv)
    common.require(bool(args.diagnostic_edge), "diagnostic edge mode is required")
    policy = forced_policy or CandidatePolicy.LATEST_ONLY_NO_EXPIRY
    try:
        return run(
            args,
            policy=policy,
            pipeline_config_factory=pipeline_config_factory,
        )
    except common.DiagnosticError as exc:
        print(f"freshness edge contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

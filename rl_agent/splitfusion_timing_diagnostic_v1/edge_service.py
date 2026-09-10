#!/usr/bin/env python3
"""Diagnostic edge service: the deployed edge path, fully decomposed in time.

This runs inside the qualified ``oai-perception-rx`` GPU container in place of
``live_pilot_runtime --edge``. It reuses the deployed reassembler, the deployed
bounded latest-frame-first pending slot, the deployed
``PreloadedSplitEdgeRuntime`` and the deployed frozen tail; the only change is
that the tail adapter is the instrumented subclass and that every boundary the
diagnostic needs is recorded with same-host ``time.time_ns()``.

One deliberate difference from the deployed edge is documented in the report:
the 100 ms service target and 500 ms capture-based processing horizon are
*classified and recorded* per frame using the deployed helpers, but a late
frame is still processed instead of dropped. Dropping late frames would delete
exactly the stage decomposition this diagnostic exists to measure. No model,
codec, threshold, action definition or byte on the wire is changed.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from phase2_map_sharing.transport import CHUNK_HEADER, ChunkReassembler, chunk_payload
from rl_agent.splitfusion_live_dispatch_v1.envelope import unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    EDGE_RESULT_SCHEMA,
    EDGE_TERMINAL_ACK_SCHEMA,
    OBJECT_MAP_UPDATE_SCHEMA,
    LatestFramePendingSlot,
    _Counters,
    _finite_tree,
    _trace_ns,
    ack_timeout_s,
    deadline_at_s,
    service_deadline_s,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    ProductionSplitCodec,
    require_inner_agreement,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for

from . import diagnostic_common as common
from .edge_preload import preload_instrumented_edge
from .instrumented_tail import (
    SERIALIZE_STAGE,
    TAIL_STAGES,
    TOTAL_STAGE,
    assert_parent_equivalence,
)


WARMUP_PHASE = "WARMUP"
MEASURED_PHASE = "MEASURED"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _decode_for_equivalence(edge: Any, payload: bytes, action_id: int) -> tuple[Any, Any]:
    """Decode one payload through the production codec, outside any timing."""

    outer = unpack_envelope(payload)
    common.require(int(outer.action_id) == int(action_id), "warm-up payload action drift")
    profile = edge.registry.resolve(int(action_id))
    context = outer.frame_context
    common.require(context is not None, "warm-up payload lacks SFD1 v2 frame context")
    edge.camera_registry.resolve(
        context.camera_model_sha256, context.camera_mount_sha256
    )
    codec = ProductionSplitCodec()
    timing = StageRecorder(EDGE_STAGES)
    inspected = codec.inspect(outer.inner_payload, timing=timing)
    require_inner_agreement(profile, inspected.identity)
    decoder = (
        None if profile.family == "noAE" else edge.runtime._ae_decoders.get(profile.family)
    )
    with torch.inference_mode():
        decoded = codec.decode(
            inspected, decoder=decoder, tail_device=edge.device, timing=timing
        )
    common.require(decoded.finite, "warm-up reconstructed C2 is non-finite")
    metadata = metadata_for(
        profile,
        sequence_id=outer.sequence_id,
        capture_timestamp_ns=outer.capture_timestamp_ns,
        protocol_version=outer.protocol_version,
        frame_context=context,
    )
    return decoded.c2, metadata


def run(args: argparse.Namespace) -> int:
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
    start_anchor = common.clock_anchor("edge_process_start")

    allowed = tuple(
        int(value) for value in str(args.allowed_action_ids or args.action_id).split(",")
    )
    common.require(
        bool(allowed) and len(allowed) == len(set(allowed)),
        "edge action allowlist is empty or duplicated",
    )
    common.require(int(args.action_id) in allowed, "edge fixed action is outside its allowlist")

    edge = preload_instrumented_edge(device)
    profiles = {value: edge.registry.resolve(value) for value in allowed}
    service_s = service_deadline_s(campaign)
    horizon_s = ack_timeout_s(campaign)
    counters = _Counters()
    terminal_reason = ""
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    records_lock = threading.Lock()

    records_path = Path(args.records_file)
    summary_path = Path(args.summary_file)
    ready_path = Path(args.ready_file)
    stop_path = Path(args.stop_file) if args.stop_file else None

    # ---- warm-up and equivalence proof, both before readiness -------------
    warmup_blob = Path(args.warmup_payload)
    warmup_index = common.load_json(warmup_blob.with_name("warmup_index.json"))
    raw = warmup_blob.read_bytes()
    warmup_payloads = [
        raw[offset : offset + size]
        for offset, size in zip(warmup_index["offsets"], warmup_index["sizes"])
    ]
    common.require(
        len(warmup_payloads) == int(warmup_index["count"])
        and all(warmup_payloads)
        and sum(len(item) for item in warmup_payloads) == len(raw),
        "warm-up payload index does not account for the blob exactly",
    )
    del raw
    equivalence = None
    warmup_iterations = int(args.warmup_iterations)
    common.require(
        1 <= warmup_iterations <= len(warmup_payloads),
        "warm-up iterations must be covered by distinct warm-up payloads",
    )
    c2, metadata = _decode_for_equivalence(edge, warmup_payloads[0], int(args.action_id))
    equivalence = assert_parent_equivalence(
        reference=edge.reference_tail,
        instrumented=edge.tail,
        c2=c2,
        metadata=metadata,
    )
    del c2, metadata
    warmup_wall_ms: list[float] = []
    for index in range(warmup_iterations):
        started = time.time_ns()
        edge.tail.begin_frame()
        result = edge.runtime.process(
            warmup_payloads[index], transmitted_action_id=int(args.action_id)
        )
        edge.tail.resolve_frame()
        snapshot = edge.tail.take_snapshot()
        _finite_tree(result.perception)
        del snapshot, result
        warmup_wall_ms.append((time.time_ns() - started) / 1e6)
    del warmup_payloads
    torch.cuda.synchronize(device)

    # ---- sockets, reassembly and the bounded pending slot ------------------
    request = int(runtime["socket_buffer_request_bytes"])
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
    receiver.bind(("0.0.0.0", int(args.edge_port)))
    receiver.settimeout(0.25)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, request)
    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    pending = LatestFramePendingSlot()
    stop_event = threading.Event()
    chunk_bytes = int(runtime["udp_chunk_bytes"])

    # ``docker compose down`` terminates this process; without a handler the
    # default SIGTERM disposition kills the interpreter outright and the final
    # summary below is never published. The stop-file is the deterministic
    # path the host uses; the signal handlers are the backstop.
    def request_stop(signum: int, _frame: Any) -> None:
        counters.bump(f"shutdown_signal_{int(signum)}")
        stop_event.set()

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, request_stop)
    first_datagram_wall_ns: dict[tuple[str, int], int] = {}
    expired_seen = 0

    def publish_summary(final: bool) -> None:
        payload = {
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
            "deadline_policy": "CLASSIFY_AND_RECORD_NEVER_DROP",
            "warmup_iterations": warmup_iterations,
            "warmup_wall_ms": warmup_wall_ms,
            "parent_equivalence": equivalence,
            "counters": counters.snapshot(),
            "edge_operation_counters": dict(edge.runtime.counters.__dict__),
            "call_ledger": edge.ledger.snapshot(),
            "incomplete_reassemblies_expired": int(reassembler.expired_messages),
            "reassembly_pending_messages": len(reassembler.pending),
            "pending_depth": pending.depth(),
            "records_written": len(records),
            "terminal_reason": terminal_reason,
            "failures": failures[:8],
            "start_clock_anchor": start_anchor,
            "clock_anchor": common.clock_anchor("edge_publish"),
            "updated_at_unix_s": time.time(),
        }
        try:
            _atomic_write_text(
                summary_path, json.dumps(payload, sort_keys=True, indent=1) + "\n"
            )
        except Exception:
            counters.bump("edge_summary_publication_failed")

    def publish_records() -> None:
        with records_lock:
            lines = [
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in records
            ]
        try:
            _atomic_write_text(records_path, "\n".join(lines) + ("\n" if lines else ""))
        except Exception:
            counters.bump("edge_record_publication_failed")

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
            datagram_perf_ns = time.perf_counter_ns()
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
            counters.set_max("reassembly_pending_high_water", len(reassembler.pending))
            if complete is None:
                continue
            complete_wall_ns = time.time_ns()
            counters.bump("feature_messages_reassembled")
            counters.bump("feature_datagrams_duplicate", int(complete.duplicate_chunks))
            first_wall_ns = first_datagram_wall_ns.pop(
                (str(address), int(complete.message_id)), datagram_wall_ns
            )
            try:
                outer = unpack_envelope(complete.payload)
            except Exception:
                counters.bump("feature_envelope_rejected")
                continue
            if outer.action_id not in profiles:
                counters.bump("feature_action_outside_allowlist")
                continue
            context = outer.frame_context
            if context is None:
                counters.bump("feature_missing_frame_context")
                continue
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
                "edge_received_perf_ns": int(datagram_perf_ns),
                "edge_received_wall_s": complete_wall_ns / 1e9,
            }
            admitted, displaced = pending.offer(
                str(context.stream_id), item, sequence=int(outer.sequence_id)
            )
            if not admitted:
                counters.bump("edge_admission_refused_not_freshest")
                continue
            item["edge_admitted_wall_ns"] = time.time_ns()
            counters.bump("edge_queue_admissions")
            counters.set_max("edge_pending_depth_high_water", pending.depth())
            if displaced is not None:
                counters.bump("edge_pending_replacements")

    def process_loop() -> None:
        while not stop_event.is_set():
            taken = pending.take(timeout=0.1)
            if taken is None:
                continue
            _stream_id, item = taken
            worker_start_wall_ns = time.time_ns()
            counters.bump("edge_process_starts")
            capture_ns = int(item["capture_timestamp_ns"])
            edge.tail.begin_frame()
            try:
                result = edge.runtime.process(
                    item["payload"], transmitted_action_id=int(item["action_id"])
                )
            except Exception as exc:
                counters.bump("edge_processing_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            tail_finished_wall_ns = time.time_ns()
            tail_finished_perf_ns = time.perf_counter_ns()
            stage_timings = edge.tail.resolve_frame()
            counters.bump("tail_completions")
            try:
                snapshot = edge.tail.take_snapshot()
            except Exception as exc:
                counters.bump("edge_snapshot_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            finite_started_ns = time.time_ns()
            try:
                _finite_tree(result.perception)
            except Exception as exc:
                counters.bump("edge_nonfinite_perception")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            finite_finished_ns = time.time_ns()
            context = result.metadata.frame_context
            if context is None or context.frame_id != int(item["message_id"]):
                counters.bump("edge_frame_identity_rejected")
                failures.append("SFD1/chunk frame identity drift")
                continue
            profile = profiles[int(item["action_id"])]
            common.require(
                str(result.metadata.profile_id) == profile.profile_id,
                "edge action/profile binding drift",
            )
            service_records = list(snapshot.records or ())
            serialized_bytes = len(snapshot.serialized_records or b"")
            semantic_shape = [int(value) for value in snapshot.semantic_labels.shape]
            snapshot_camera_pose_ns = int(snapshot.camera_pose_reconstruct_ns)
            snapshot_tensor_count = int(snapshot.output_tensor_count)
            result_timing = result.timing
            if str(args.warmup_stream_id):
                phase = (
                    WARMUP_PHASE
                    if str(context.stream_id) == str(args.warmup_stream_id)
                    else MEASURED_PHASE
                )
            else:
                phase = (
                    WARMUP_PHASE
                    if int(item["sequence_id"]) < int(args.first_measured_sequence_id)
                    else MEASURED_PHASE
                )
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
                    result.scientific_inner_payload_bytes
                ),
                "framing_control_overhead_bytes": int(
                    result.framing_control_overhead_bytes
                ),
                "total_received_bytes": int(result.total_received_bytes),
                "edge_first_datagram_wall_ns": int(item["edge_first_datagram_wall_ns"]),
                "edge_complete_reassembly_wall_ns": int(
                    item["edge_complete_reassembly_wall_ns"]
                ),
                "edge_admitted_wall_ns": int(item.get("edge_admitted_wall_ns", 0)),
                "edge_worker_start_wall_ns": int(worker_start_wall_ns),
                "edge_tail_finished_wall_ns": int(tail_finished_wall_ns),
                "edge_result_finite_check_ms": (
                    finite_finished_ns - finite_started_ns
                ) / 1e6,
                "edge_service_wall_ms": (tail_finished_wall_ns - worker_start_wall_ns)
                / 1e6,
                "edge_stage_ns": _trace_ns(result.timing),
                "tail_stage_wall_ns": stage_timings["wall_ns"],
                "tail_stage_cuda_ms": stage_timings["cuda_ms"],
                "reconstructed_device": str(edge.runtime.tail_device),
                "finite_output_tensor_count": int(snapshot.output_tensor_count),
                "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
                "service_record_count": len(service_records),
                "service_record_bytes": serialized_bytes,
                "semantic_label_shape": semantic_shape,
                "service_deadline_at_s": deadline_at_s(capture_ns, service_s),
                "processing_horizon_at_s": deadline_at_s(capture_ns, horizon_s),
                "service_target_met": bool(
                    tail_finished_wall_ns / 1e9 <= deadline_at_s(capture_ns, service_s)
                ),
                "processing_horizon_met": bool(
                    tail_finished_wall_ns / 1e9 <= deadline_at_s(capture_ns, horizon_s)
                ),
            }
            del snapshot, result
            if phase != MEASURED_PHASE:
                counters.bump("warmup_frames_processed")
            # The deployed UE result loop consumes this payload and drives map
            # install and install feedback, so it must be the production
            # `splitfusion_edge_result.v2` message. The diagnostic quantities
            # ride alongside under their own key; the dense label map still
            # stays off the radio.
            terminal_ack = {
                "schema": EDGE_TERMINAL_ACK_SCHEMA,
                "run_id": str(args.run_id),
                "cell_id": str(args.cell_id),
                "action_id": int(profile.action_id),
                "stream_id": str(context.stream_id),
                "frame_id": int(context.frame_id),
                "capture_timestamp_ns": capture_ns,
                "capture_wall_s": capture_ns / 1_000_000_000.0,
                "edge_receipt_wall_s": float(item["edge_received_wall_s"]),
                "tail_complete_wall_s": tail_finished_wall_ns / 1e9,
                "evidence_install_wall_s": "",
                "service_deadline_at": deadline_at_s(capture_ns, service_s),
                "ack_timeout_at": deadline_at_s(capture_ns, horizon_s),
                "installation_status": "EVALUATION_EVIDENCE_NOT_CONFIGURED",
                "evidence": {},
                "terminal_reason": "EDGE_SERVICE_COMPLETE",
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
                "camera_pose_reconstruct_ns": int(snapshot_camera_pose_ns),
                "finite_output_tensor_count": int(snapshot_tensor_count),
                "service_record_count": len(service_records),
                "feature_received_datagrams": int(item["chunk_count"]),
                "feature_duplicate_datagrams": int(item["duplicate_chunks"]),
                "edge_call_ledger": edge.ledger.snapshot(),
                "edge_counters": {
                    **dict(edge.runtime.counters.__dict__), **counters.snapshot()
                },
                "edge_received_ns": int(item["edge_received_perf_ns"]),
                "tail_finished_ns": int(tail_finished_perf_ns),
                "edge_timing_ns": _trace_ns(result_timing),
                "object_map_update": {
                    "schema": OBJECT_MAP_UPDATE_SCHEMA,
                    "stream_id": str(context.stream_id),
                    "frame_id": int(context.frame_id),
                    "capture_timestamp_ns": capture_ns,
                    "action_id": int(profile.action_id),
                    "records": service_records,
                },
                "edge_terminal_ack": terminal_ack,
                "diagnostic": {
                    "phase": phase,
                    "sequence_id": int(item["sequence_id"]),
                    "service_record_bytes": serialized_bytes,
                    "edge_worker_start_wall_ns": int(worker_start_wall_ns),
                    "edge_tail_finished_wall_ns": int(tail_finished_wall_ns),
                    "edge_service_wall_ms": record["edge_service_wall_ms"],
                    "dense_label_map_on_radio": False,
                },
            }
            blob = json.dumps(downlink, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
            result_chunks = chunk_payload(
                blob, message_id=int(context.frame_id), chunk_bytes=chunk_bytes
            )
            for chunk in result_chunks:
                sender.sendto(chunk, (str(args.result_host), int(args.result_port)))
                counters.bump("result_datagrams_transmitted")
            published_wall_ns = time.time_ns()
            counters.bump("compact_results_transmitted")
            counters.bump("result_bytes_transmitted", len(blob))
            # Deployed span: worker start through compact-result publication,
            # so it carries serialization, publication and live contention.
            record["edge_result_published_wall_ns"] = int(published_wall_ns)
            record["deployed_tail_service_ms"] = (
                published_wall_ns - worker_start_wall_ns
            ) / 1e6
            record["result_datagrams"] = len(result_chunks)
            if phase == MEASURED_PHASE:
                with records_lock:
                    records.append(record)

    ready_path.parent.mkdir(parents=True, exist_ok=True)
    publish_records()
    publish_summary(final=False)
    receiver_thread = threading.Thread(
        target=receive_loop, name="diag-edge-receive", daemon=True
    )
    processor_thread = threading.Thread(
        target=process_loop, name="diag-edge-process", daemon=True
    )
    receiver_thread.start()
    processor_thread.start()
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
                "deadline_policy": "CLASSIFY_AND_RECORD_NEVER_DROP",
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
                "ready_clock_anchor": common.clock_anchor("edge_ready"),
            },
            handle,
            sort_keys=True,
        )
    try:
        while True:
            if stop_event.is_set():
                terminal_reason = "STOP_REQUESTED"
                break
            if stop_path is not None and stop_path.exists():
                terminal_reason = "STOP_FILE_OBSERVED"
                break
            if not (receiver_thread.is_alive() and processor_thread.is_alive()):
                terminal_reason = "WORKER_THREAD_EXITED"
                break
            time.sleep(0.25)
            publish_records()
            publish_summary(final=False)
        return 0 if terminal_reason != "WORKER_THREAD_EXITED" else 1
    finally:
        stop_event.set()
        for dropped in pending.close():
            counters.bump("edge_pending_dropped_at_shutdown")
            del dropped
        receiver_thread.join(timeout=3.0)
        processor_thread.join(timeout=10.0)
        publish_records()
        publish_summary(final=True)
        receiver.close()
        sender.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SplitFusion timing diagnostic edge")
    parser.add_argument("--diagnostic-edge", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--action-id", type=int, required=True)
    parser.add_argument("--allowed-action-ids")
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--records-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--warmup-payload", required=True)
    parser.add_argument("--stop-file")
    parser.add_argument("--warmup-iterations", type=int, default=12)
    parser.add_argument("--first-measured-sequence-id", type=int, default=1000)
    parser.add_argument("--warmup-stream-id", default="")
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--result-host", default="10.0.0.2")
    parser.add_argument("--result-port", type=int, default=51004)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cell-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, _ignored = build_parser().parse_known_args(argv)
    common.require(bool(args.diagnostic_edge), "diagnostic edge mode is required")
    try:
        return run(args)
    except common.DiagnosticError as exc:
        print(f"diagnostic edge contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

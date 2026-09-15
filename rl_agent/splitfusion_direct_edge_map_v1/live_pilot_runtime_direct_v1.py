"""Direct edge-to-map live runtime: corrected edge service and UE bridge.

The frozen scientific path is untouched and imported, never copied: the front,
ranker, AE/quantizer, split point, SFD1-v2 envelope and fragmentation, the
frozen p025 tail, the latest-frame-first bounded slots, the capture-based
deadline stages and the edge-only dense label-map evidence all come from
``rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime`` unchanged.

What changes is only the post-inference publication path:

before  edge -> (radio) -> UE 10.0.0.2:51004 -> UE re-serialises -> map 127.0.0.1
after   edge -> map on the edge-local CN5G bridge; map installs; map -> UE ACK

The edge never addresses the UE with object records again. Frames the edge ends
before publication produce a compact edge terminal control so a deliberately
superseded frame is credited as replaced work rather than as a radio failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from phase2_map_sharing.transport import ChunkReassembler
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
from rl_agent.splitfusion_live_dispatch_v1.envelope import unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry

from . import protocol
from .cpu_reservation import apply_thread_reservation, describe_reservation
from .direct_v3_edge import EDGE_VARIANT, preload_direct_v3_edge
from .edge_publisher import DirectMapPublisher, UEControlSender
from .protocol import (
    OUTCOME_STALE_BEFORE_EDGE,
    OUTCOME_STALE_BEFORE_MAP,
    OUTCOME_SUPERSEDED_PENDING,
)
from .ue_ledger import DirectTerminalLedger


DIRECT_EDGE_COUNTERS_SCHEMA = "splitfusion_direct_edge_counters.v1"
DIRECT_EDGE_READY_SCHEMA = "splitfusion_direct_live_edge_ready.v1"

# Re-exported so callers bind exactly the frozen deadline semantics.
service_deadline_s = base.service_deadline_s
ack_timeout_s = base.ack_timeout_s
deadline_at_s = base.deadline_at_s
check_deadline = base.check_deadline
DeadlineExpired = base.DeadlineExpired
LivePilotRuntimeError = base.LivePilotRuntimeError
_require = base._require


class _ForbiddenSocket:
    """Stands in for the removed UE map-publication socket.

    Constructing the corrected UE runtime replaces the inherited ``map_socket``
    with this object, so any surviving code path that tried to forward object
    records from the UE to the map fails loudly instead of silently restoring
    the detour.
    """

    def sendto(self, *_args: Any, **_kwargs: Any) -> int:
        raise LivePilotRuntimeError(
            "the UE must not publish object records to the map in the direct "
            "edge-to-map architecture"
        )

    def close(self) -> None:
        return None


PUBLICATION_FIELDS = (
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "action_id",
    "capture_timestamp_ns",
    "reassembly_complete_wall_s",
    "admission_wall_s",
    "compute_start_wall_s",
    "tail_complete_wall_s",
    "evidence_install_wall_s",
    "publication_ticket_ready_wall_s",
    "publisher_worker_start_wall_s",
    "serialization_start_wall_s",
    "serialization_end_wall_s",
    "first_datagram_send_wall_s",
    "last_datagram_send_wall_s",
    "direct_map_datagrams",
    "direct_map_payload_bytes",
    "record_count",
    "publisher_thread",
)


def _write_publication_rows(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Persist the edge-side publication stage stamps beside the counters.

    The send instants cannot travel inside the message they describe, so the
    edge records them itself on the same wall clock the map uses and the
    evaluator joins the two by (stream_id, frame_id).
    """

    path = directory / "direct_edge_publication.csv"
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PUBLICATION_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in PUBLICATION_FIELDS})
    except OSError:
        pass


def run_direct_edge_service(
    *,
    config_path: Path,
    action_id: int,
    allowed_action_ids: tuple[int, ...],
    ready_file: Path,
    edge_port: int,
    map_host: str,
    map_port: int,
    ue_control_host: str,
    ue_control_port: int,
    evidence_dir: Path | None = None,
    run_id: str = "",
    cell_id: str = "",
    compute_cpus: str = "",
    receive_cpus: str = "",
) -> int:
    """Serve one cell's frozen tail, publishing object maps straight to the map."""

    campaign = base._load_json(config_path)
    runtime = campaign["runtime"]
    _require(
        int(runtime["sfd1_protocol_version"]) == 2
        and runtime["udp_fragment_header"] == "!IHH",
        "edge protocol binding drift",
    )
    _require(
        torch.cuda.is_available()
        and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090",
        "edge CUDA device unavailable",
    )
    device = torch.device("cuda:0")
    registry = SplitActionRegistry.from_runtime_binding()
    _require(
        bool(allowed_action_ids)
        and len(allowed_action_ids) == len(set(allowed_action_ids)),
        "edge action allowlist is empty or duplicated",
    )
    profiles = {value: registry.resolve(value) for value in allowed_action_ids}
    _require(int(action_id) in profiles, "edge fixed action is outside its allowlist")
    service_s = service_deadline_s(campaign)
    deadline_s = ack_timeout_s(campaign)
    counters = base._Counters()

    def guard(stage: str, capture_timestamp_ns: int) -> None:
        check_deadline(stage, capture_timestamp_ns, deadline_s)
        if stage == base.EDGE_STAGE_BEFORE_TAIL:
            counters.bump("tail_starts")

    # The repaired overlapped v3 tail, not the serial production tail. Its
    # model, codec, catalog, thresholds and record schema are the frozen ones;
    # only the tail's internal stage overlap differs, and its outputs are
    # qualified bit-identical against the frozen reference adapter.
    edge, tail, ledger, models = preload_direct_v3_edge(device, deadline_guard=guard)
    evidence: base.EdgeEvaluationEvidenceWriter | None = None
    if evidence_dir is not None:
        evidence = base.EdgeEvaluationEvidenceWriter(Path(evidence_dir), counters)

    request = int(runtime["socket_buffer_request_bytes"])
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, request)
    receiver.bind(("0.0.0.0", int(edge_port)))
    receiver.settimeout(0.25)

    chunk_bytes = int(runtime["udp_chunk_bytes"])
    publisher = DirectMapPublisher(
        map_host=map_host,
        map_port=map_port,
        chunk_bytes=chunk_bytes,
        socket_buffer_request_bytes=request,
    )
    control = UEControlSender(ue_host=ue_control_host, ue_port=ue_control_port)

    reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
    pending = base.LatestFramePendingSlot()
    stop_event = threading.Event()
    failures: list[str] = []
    counters_path = Path(ready_file).parent / "direct_edge_counters.json"

    def emit_edge_terminal(
        item: Mapping[str, Any],
        *,
        outcome: str,
        stage: str,
        superseded_by_frame_id: int | None = None,
    ) -> None:
        """Tell the UE the edge ended this obligation, and why."""

        capture_ns = int(item["capture_timestamp_ns"])
        profile = profiles[int(item["action_id"])]
        age_ms = (time.time() - capture_ns / 1_000_000_000.0) * 1000.0
        try:
            message = protocol.build_edge_terminal_control(
                run_id=run_id,
                cell_id=cell_id,
                stream_id=str(item["stream_id"]),
                frame_id=int(item["message_id"]),
                action_id=int(profile.action_id),
                profile_id=str(profile.profile_id),
                capture_timestamp_ns=capture_ns,
                service_deadline_at=deadline_at_s(capture_ns, service_s),
                ack_timeout_at=deadline_at_s(capture_ns, deadline_s),
                outcome=outcome,
                stage=stage,
                age_ms=age_ms,
                emit_at=time.time(),
                superseded_by_frame_id=superseded_by_frame_id,
            )
        except protocol.DirectMapProtocolError as exc:
            counters.bump("edge_terminal_construction_failed")
            failures.append(f"edge terminal: {exc}")
            return
        control.send(message)
        counters.bump(f"edge_terminal_{outcome}")

    publication_rows: list[dict[str, Any]] = []
    reservations: list[dict[str, Any]] = []

    def publish_counters() -> None:
        try:
            base._atomic_write_bytes(
                counters_path,
                json.dumps(
                    {
                        "schema": DIRECT_EDGE_COUNTERS_SCHEMA,
                        "run_id": str(run_id),
                        "cell_id": str(cell_id),
                        "action_id": int(action_id),
                        "service_deadline_s": service_s,
                        "ack_timeout_s": deadline_s,
                        "processing_expiry_s": deadline_s,
                        "pending_depth": pending.depth(),
                        "incomplete_reassemblies_expired": int(
                            reassembler.expired_messages
                        ),
                        "reassembly_pending_messages": len(reassembler.pending),
                        "counters": counters.snapshot(),
                        "edge_tail_variant": str(edge.variant),
                        "asynchronous_verdict_corrections": int(
                            edge.asynchronous_verdict_corrections
                        ),
                        "cpu_reservation": describe_reservation(reservations),
                        "direct_map_publisher": publisher.snapshot(),
                        "ue_control": control.snapshot(),
                        "edge_operation_counters": dict(edge.counters.__dict__),
                        "call_ledger": ledger.snapshot(),
                        "failures": failures[:8],
                        "updated_at_unix_s": time.time(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        except Exception:
            counters.bump("edge_counter_publication_failed")

    expired_seen = 0

    def reconcile_expiries() -> None:
        nonlocal expired_seen
        observed = int(reassembler.expired_messages)
        if observed != expired_seen:
            delta = observed - expired_seen
            counters.bump("incomplete_reassemblies_expired", delta)
            counters.bump("reassembly_buffer_evictions", delta)
            expired_seen = observed

    def receive_loop() -> None:
        reservations.append(
            apply_thread_reservation(receive_cpus, label="direct-edge-feature-receive")
        )
        while not stop_event.is_set():
            try:
                datagram, address = receiver.recvfrom(65535)
            except socket.timeout:
                reassembler.expire(time.monotonic())
                reconcile_expiries()
                continue
            except OSError:
                return
            counters.bump("feature_datagrams_received")
            try:
                complete = reassembler.ingest(
                    str(address), datagram, received_at_s=time.monotonic()
                )
            except ValueError:
                counters.bump("feature_datagrams_malformed")
                continue
            reconcile_expiries()
            counters.set_max("reassembly_pending_high_water", len(reassembler.pending))
            if complete is None:
                continue
            received_wall = time.time()
            received_ns = time.perf_counter_ns()
            counters.bump("feature_messages_reassembled")
            counters.bump("feature_datagrams_duplicate", int(complete.duplicate_chunks))
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
                "stream_id": str(context.stream_id),
                "capture_timestamp_ns": int(outer.capture_timestamp_ns),
                "chunk_count": int(complete.chunk_count),
                "duplicate_chunks": int(complete.duplicate_chunks),
                "edge_received_ns": received_ns,
                "reassembly_complete_wall_s": received_wall,
            }
            try:
                check_deadline(
                    base.EDGE_STAGE_AFTER_REASSEMBLY,
                    outer.capture_timestamp_ns,
                    deadline_s,
                    now_s=received_wall,
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                emit_edge_terminal(
                    item, outcome=OUTCOME_STALE_BEFORE_EDGE, stage=expired.stage
                )
                continue
            admitted, displaced = pending.offer(
                str(context.stream_id), item, sequence=int(outer.sequence_id)
            )
            if not admitted:
                counters.bump("edge_admission_refused_not_freshest")
                emit_edge_terminal(
                    item,
                    outcome=OUTCOME_SUPERSEDED_PENDING,
                    stage="EDGE_ADMISSION_NOT_FRESHEST",
                )
                continue
            item["admission_wall_s"] = time.time()
            counters.bump("edge_queue_admissions")
            counters.set_max("edge_pending_depth_high_water", pending.depth())
            if displaced is not None:
                counters.bump("edge_pending_replacements")
                emit_edge_terminal(
                    displaced,
                    outcome=OUTCOME_SUPERSEDED_PENDING,
                    stage="EDGE_PENDING_REPLACED",
                    superseded_by_frame_id=int(item["message_id"]),
                )

    def process_loop() -> None:
        reservations.append(
            apply_thread_reservation(compute_cpus, label="direct-edge-tail-process")
        )
        while not stop_event.is_set():
            taken = pending.take(timeout=0.1)
            if taken is None:
                continue
            _stream_id, item = taken
            capture_timestamp_ns = int(item["capture_timestamp_ns"])
            try:
                check_deadline(
                    base.EDGE_STAGE_BEFORE_DECODE, capture_timestamp_ns, deadline_s
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                emit_edge_terminal(
                    item, outcome=OUTCOME_STALE_BEFORE_MAP, stage=expired.stage
                )
                continue
            counters.bump("edge_process_starts")
            compute_start_wall_s = time.time()
            compute_start_ns = time.perf_counter_ns()
            try:
                result = edge.process(
                    item["payload"], transmitted_action_id=int(item["action_id"])
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                emit_edge_terminal(
                    item, outcome=OUTCOME_STALE_BEFORE_MAP, stage=expired.stage
                )
                continue
            except Exception as exc:
                counters.bump("edge_processing_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            compute_finish_ns = time.perf_counter_ns()
            tail_finished_wall = time.time()
            counters.bump("tail_completions")
            try:
                snapshot = tail.take_snapshot()
            except Exception as exc:
                counters.bump("edge_snapshot_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            try:
                base._finite_tree(result.perception)
            except Exception as exc:
                counters.bump("edge_nonfinite_perception")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            context = result.metadata.frame_context
            if context is None or context.frame_id != int(item["message_id"]):
                counters.bump("edge_frame_identity_rejected")
                failures.append("SFD1/chunk frame identity drift")
                continue
            profile = profiles[int(item["action_id"])]
            try:
                check_deadline(
                    base.EDGE_STAGE_BEFORE_PUBLICATION,
                    capture_timestamp_ns,
                    deadline_s,
                )
            except DeadlineExpired as expired:
                counters.bump(f"deadline_drop_{expired.stage}")
                emit_edge_terminal(
                    item, outcome=OUTCOME_STALE_BEFORE_MAP, stage=expired.stage
                )
                continue

            records = list(snapshot.records or ())
            installation_status = "EVALUATION_EVIDENCE_NOT_CONFIGURED"
            evidence_meta: dict[str, Any] = {}
            evidence_install_wall: Any = ""
            if evidence is not None:
                # The dense 720x1280 label map stays on the edge's own mount and
                # never enters the direct map update or the UE control message.
                labels = (
                    snapshot.semantic_labels.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                    .numpy()
                )
                digest = hashlib.sha256(labels.tobytes()).hexdigest()
                evidence_meta = {
                    "evidence_name": base.segmentation_evidence_name(
                        context.stream_id, context.frame_id
                    ),
                    "schema": base.EVIDENCE_SIDECAR_SCHEMA,
                    "run_id": str(run_id),
                    "cell_id": str(cell_id),
                    "action_id": int(profile.action_id),
                    "profile_id": str(profile.profile_id),
                    "stream_id": str(context.stream_id),
                    "frame_id": int(context.frame_id),
                    "capture_timestamp_ns": int(context.capture_timestamp_ns),
                    "shape": [int(value) for value in labels.shape],
                    "dtype": str(labels.dtype),
                    "bytes": int(labels.nbytes),
                    "sha256": digest,
                }
                installation_status = evidence.submit(labels, evidence_meta)
                evidence_install_wall = time.time()

            edge_timing = {
                "reassembly_complete_wall_s": float(item["reassembly_complete_wall_s"]),
                "admission_wall_s": float(item.get("admission_wall_s") or 0.0),
                "compute_start_wall_s": compute_start_wall_s,
                "compute_finish_wall_s": tail_finished_wall,
                "tail_complete_wall_s": tail_finished_wall,
                "evidence_install_wall_s": evidence_install_wall,
                "compute_duration_ms": (compute_finish_ns - compute_start_ns) / 1e6,
                "edge_received_ns": int(item["edge_received_ns"]),
                "feature_received_datagrams": int(item["chunk_count"]),
                "feature_duplicate_datagrams": int(item["duplicate_chunks"]),
                "decoder_identity": str(profile.decoder_identity),
                "reconstructed_device": str(edge.tail_device),
                "camera_pose_reconstruct_ns": int(snapshot.camera_pose_reconstruct_ns),
                "finite_output_tensor_count": int(snapshot.output_tensor_count),
                "edge_timing_ns": base._trace_ns(result.timing),
            }
            try:
                update = protocol.build_object_map_update(
                    run_id=run_id,
                    cell_id=cell_id,
                    stream_id=str(context.stream_id),
                    frame_id=int(context.frame_id),
                    sequence_id=int(context.sequence_id),
                    action_id=int(profile.action_id),
                    profile_id=str(profile.profile_id),
                    decoder_identity=str(profile.decoder_identity),
                    capture_timestamp_ns=int(context.capture_timestamp_ns),
                    carla_timestamp=0.0,
                    records=records,
                    service_deadline_at=deadline_at_s(capture_timestamp_ns, service_s),
                    ack_timeout_at=deadline_at_s(capture_timestamp_ns, deadline_s),
                    edge_timing=edge_timing,
                    segmentation={
                        "available": True,
                        "evidence": evidence_meta,
                        "installation_status": installation_status,
                    },
                )
            except protocol.DirectMapProtocolError as exc:
                counters.bump("direct_update_construction_failed")
                failures.append(f"direct update: {exc}")
                continue
            # The update document is complete: this is the instant a
            # publication ticket becomes available to the publishing owner.
            publication_ticket_ready_wall_s = time.time()
            try:
                accounting = publisher.publish(update)
            except Exception as exc:
                counters.bump("direct_map_publication_failed")
                failures.append(f"{type(exc).__name__}: {exc}")
                continue
            publication_rows.append(
                {
                    "run_id": str(run_id),
                    "cell_id": str(cell_id),
                    "stream_id": str(context.stream_id),
                    "frame_id": int(context.frame_id),
                    "action_id": int(profile.action_id),
                    "capture_timestamp_ns": int(context.capture_timestamp_ns),
                    "reassembly_complete_wall_s": edge_timing[
                        "reassembly_complete_wall_s"
                    ],
                    "admission_wall_s": edge_timing["admission_wall_s"],
                    "compute_start_wall_s": edge_timing["compute_start_wall_s"],
                    "tail_complete_wall_s": edge_timing["tail_complete_wall_s"],
                    "evidence_install_wall_s": evidence_install_wall,
                    "publication_ticket_ready_wall_s": (
                        publication_ticket_ready_wall_s
                    ),
                    # Publication is performed by the compute owner in this
                    # boundary, so the worker start is the call instant.
                    "publisher_worker_start_wall_s": accounting[
                        "publish_start_wall_s"
                    ],
                    "serialization_start_wall_s": accounting[
                        "serialization_start_wall_s"
                    ],
                    "serialization_end_wall_s": accounting[
                        "serialization_end_wall_s"
                    ],
                    "first_datagram_send_wall_s": accounting[
                        "first_datagram_send_wall_s"
                    ],
                    "last_datagram_send_wall_s": accounting[
                        "last_datagram_send_wall_s"
                    ],
                    "direct_map_datagrams": int(accounting["direct_map_datagrams"]),
                    "direct_map_payload_bytes": int(
                        accounting["direct_map_payload_bytes"]
                    ),
                    "record_count": int(update["record_count"]),
                    "publisher_thread": threading.current_thread().name,
                }
            )
            counters.bump("direct_map_publications")
            counters.bump(
                "direct_map_payload_bytes", int(accounting["direct_map_payload_bytes"])
            )
            counters.set_max(
                "direct_map_datagrams_high_water", int(accounting["direct_map_datagrams"])
            )

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    with ready_file.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": DIRECT_EDGE_READY_SCHEMA,
                "action_id": int(action_id),
                "allowed_action_ids": list(allowed_action_ids),
                "profiles": {
                    str(key): value.profile_id for key, value in profiles.items()
                },
                "tail_device": str(edge.tail_device),
                "state_root": str(ready_file.parent),
                "state_root_writable": True,
                "service_deadline_s": service_s,
                "ack_timeout_s": deadline_s,
                "processing_expiry_s": deadline_s,
                "dense_label_map_on_radio": False,
                "object_records_on_radio": False,
                "direct_map_host": str(map_host),
                "direct_map_port": int(map_port),
                "ue_control_host": str(ue_control_host),
                "ue_control_port": int(ue_control_port),
                "architecture": "DIRECT_EDGE_TO_MAP_V1",
                "evaluation_evidence_dir": str(evidence_dir) if evidence_dir else "",
                "edge_receive_reported_bytes": receiver.getsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF
                ),
            },
            handle,
            sort_keys=True,
        )
    publish_counters()
    receiver_thread = threading.Thread(
        target=receive_loop, name="direct-edge-feature-receive", daemon=True
    )
    processor_thread = threading.Thread(
        target=process_loop, name="direct-edge-tail-process", daemon=True
    )
    receiver_thread.start()
    processor_thread.start()
    try:
        while receiver_thread.is_alive() and processor_thread.is_alive():
            time.sleep(1.0)
            publish_counters()
        return 1
    finally:
        stop_event.set()
        for dropped in pending.close():
            counters.bump("edge_pending_dropped_at_shutdown")
            del dropped
        receiver_thread.join(timeout=2.0)
        processor_thread.join(timeout=5.0)
        if evidence is not None:
            evidence.close()
        _write_publication_rows(counters_path.parent, publication_rows)
        publish_counters()
        receiver.close()
        publisher.close()
        control.close()
        del models, ledger


class DirectLivePilotCellRuntime(base.LivePilotCellRuntime):
    """UE half of one cell under the direct edge-to-map architecture.

    Inherits the frozen preparation and transmission path unchanged. The
    inherited result loop -- which received object records over the radio and
    forwarded them to the map -- is replaced by a control loop that only ever
    receives compact, record-free terminals.
    """

    def __init__(
        self,
        *,
        campaign: Mapping[str, Any],
        cell: Mapping[str, Any],
        attempt_dir: Path,
        evidence_dir: Path,
        ue_control_port: int,
        ledger: DirectTerminalLedger,
    ) -> None:
        # The inherited constructor binds its receive socket to
        # ``camera_result_port``. Point that at the UE control port so the
        # superseded result port is never bound at all.
        direct_campaign = dict(campaign)
        direct_runtime = dict(campaign["runtime"])
        direct_runtime["camera_result_port"] = int(ue_control_port)
        direct_campaign["runtime"] = direct_runtime
        self.ledger = ledger
        self.control_counters = base._Counters()
        self._terminal_frames: dict[int, str] = {}
        super().__init__(
            campaign=direct_campaign,
            cell=cell,
            attempt_dir=attempt_dir,
            # Deliberately unreachable: replaced below and never used.
            map_host="127.0.0.1",
            map_port=1,
            evidence_dir=evidence_dir,
        )
        try:
            self.map_socket.close()
        except OSError:
            pass
        self.map_socket = _ForbiddenSocket()
        self.map_remote = None
        self.ue_control_port = int(ue_control_port)

    def _result_loop(self) -> None:
        """Receive compact map feedback and edge terminal control messages."""

        while not self.stop_event.is_set():
            try:
                datagram, address = self.receiver.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            received_at = time.time()
            self.control_counters.bump("control_datagrams_received")
            self.control_counters.bump("control_bytes_received", len(datagram))
            try:
                message = protocol.decode(datagram)
                protocol.assert_no_object_records(message)
                message["_feedback_bytes"] = len(datagram)
                message["_source_address"] = f"{address[0]}:{address[1]}"
                schema = str(message.get("schema") or "")
                if schema == protocol.DIRECT_MAP_FEEDBACK_SCHEMA:
                    self.control_counters.bump("map_feedback_messages")
                elif schema == protocol.EDGE_TERMINAL_CONTROL_SCHEMA:
                    self.control_counters.bump("edge_terminal_messages")
                else:
                    raise LivePilotRuntimeError(f"unknown control schema {schema!r}")
                frame_id = int(message["frame_id"])
                metric = self.metrics.get(frame_id)
                _require(metric is not None, "control message has no transmitted frame")
                _require(
                    int(message["action_id"]) == int(metric["action_id"]),
                    "control action identity drift",
                )
                _require(
                    str(message.get("stream_id") or "") == str(metric["stream_id"]),
                    "control stream identity drift",
                )
                # The ledger write is deliberately performed by the consumer
                # thread, not here: the supervising feedback worker snapshots
                # ``pending`` around its own ``receive_once`` call, so the
                # obligation must close inside that call rather than
                # asynchronously underneath it.
                self.ledger.enqueue(message, received_at)
                terminal = bool(message.get("terminal"))
                with self.lock:
                    if terminal:
                        if frame_id in self._terminal_frames:
                            self.control_counters.bump("duplicate_terminal_suppressed")
                        else:
                            self._terminal_frames[frame_id] = str(
                                message.get("outcome") or ""
                            )
                            self._published_frames.add(frame_id)
                            self.completed += 1
                    else:
                        self.control_counters.bump("late_nonterminal_messages")
                    metric.update(
                        self._control_fields(message, {}, received_at)
                    )
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                return

    @staticmethod
    def _control_fields(
        message: Mapping[str, Any], row: Mapping[str, Any], received_at: float
    ) -> dict[str, Any]:
        install_at = message.get("install_timestamp", "")
        outcome = str(message.get("outcome") or "")
        emit_at = message.get("feedback_emit_at", message.get("emit_at", ""))
        observation_delay_ms = (
            (float(received_at) - float(emit_at)) * 1000.0
            if emit_at not in (None, "")
            else ""
        )
        return {
            "terminal_source": str(message.get("source") or ""),
            "terminal_outcome": outcome,
            "agent_credit": str(message.get("agent_credit") or ""),
            "map_publication_status": (
                "INSTALLED"
                if outcome == protocol.OUTCOME_RESULT_INSTALLED
                else outcome
            ),
            "map_installed_at": install_at,
            "map_ingest_at": message.get("map_ingest_at", ""),
            "feedback_emit_at": emit_at,
            "feedback_received_at": received_at,
            "map_age_at_install_ms": message.get("map_age_at_install_ms", ""),
            "feedback_observation_delay_ms": observation_delay_ms,
            "direct_update_bytes": message.get("direct_update_bytes", ""),
            "direct_update_datagrams": message.get("direct_update_datagrams", ""),
            "control_message_bytes": int(message.get("_feedback_bytes") or 0),
            "control_source_address": str(message.get("_source_address") or ""),
            "superseded_by_frame_id": message.get("superseded_by_frame_id", ""),
            "edge_timing": message.get("edge_timing", ""),
            "rejection_reason": str(message.get("rejection_reason") or ""),
        }

    def terminal_frames(self) -> dict[int, str]:
        with self.lock:
            return dict(self._terminal_frames)

    def close(self) -> dict[str, Any]:
        report = super().close()
        report["architecture"] = "DIRECT_EDGE_TO_MAP_V1"
        report["control_counters"] = self.control_counters.snapshot()
        report["terminal_frames"] = len(self.terminal_frames())
        report["ledger"] = self.ledger.summary()
        report["ue_published_object_updates"] = 0
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="direct edge-to-map live edge")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--action-id", type=int)
    parser.add_argument("--allowed-action-ids")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--direct-map-host", required=False, default="")
    parser.add_argument("--direct-map-port", type=int, default=0)
    parser.add_argument("--ue-control-host", default="10.0.0.2")
    parser.add_argument("--ue-control-port", type=int, default=51014)
    parser.add_argument("--edge-segmentation-evidence-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cell-id", default="")
    parser.add_argument(
        "--edge-compute-cpus",
        default="",
        help="CPU set for the tail/compute owner, e.g. '0-15'. Advisory.",
    )
    parser.add_argument(
        "--edge-receive-cpus",
        default="",
        help="CPU set for the feature receive owner, e.g. '16-17'. Advisory.",
    )
    args, _ignored = parser.parse_known_args(list(argv) if argv is not None else None)
    _require(
        bool(args.edge)
        and args.config is not None
        and args.action_id is not None
        and args.ready_file is not None,
        "edge mode and all qualified bindings are required",
    )
    _require(
        bool(str(args.direct_map_host).strip()) and int(args.direct_map_port) > 0,
        "the direct edge-to-map endpoint is required",
    )
    allowed_action_ids = tuple(
        int(value)
        for value in str(args.allowed_action_ids or args.action_id).split(",")
    )
    return run_direct_edge_service(
        config_path=args.config.resolve(strict=True),
        action_id=int(args.action_id),
        allowed_action_ids=allowed_action_ids,
        ready_file=args.ready_file,
        edge_port=int(args.edge_port),
        map_host=str(args.direct_map_host),
        map_port=int(args.direct_map_port),
        ue_control_host=str(args.ue_control_host),
        ue_control_port=int(args.ue_control_port),
        evidence_dir=args.edge_segmentation_evidence_dir,
        run_id=str(args.run_id),
        cell_id=str(args.cell_id),
        compute_cpus=str(args.edge_compute_cpus),
        receive_cpus=str(args.edge_receive_cpus),
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Executable GT-free Run-4B/Run-5B edge runtime.

This additive runtime owns the feature UDP listener, SFD4 reassembly,
latest-only pending slot, the qualified Phase-6 processor/decoder/tail, the
compact operational-ACK UDP sender, independent direct-map publication and
create-only prediction evidence.  The synchronous ordering is:

``tail usable -> operational ACK -> map/prediction queue offers``.

There is deliberately no TCP listener: the retired TCP input was simulator
ground truth.  This runtime accepts only feature UDP and emits only compact
ACK UDP plus the already-qualified direct-map UDP publication.

Importing this module performs no I/O and imports neither torch nor any CUDA
runtime.  Live-only authorities are imported inside :func:`real_factory`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from phase2_map_sharing.transport import ChunkReassembler

from . import b_edge_engineering_request_v2 as Q
from . import b_edge_process_v1 as E
from . import b_edge_service_v2 as S
from . import branch_evidence_v1 as B
from . import live_adapters_v1 as L
from . import tail_evidence_codec_v2 as C


READY_SCHEMA = "scenesense.splitfusion.run4b5b.gt_free_edge_ready.v2"
REPORT_SCHEMA = "scenesense.splitfusion.run4b5b.gt_free_edge_report.v2"
RUNTIME_PREFLIGHT_SCHEMA = (
    "scenesense.splitfusion.run4b5b.gt_free_edge_runtime_preflight.v2"
)
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_GT_FREE_EDGE_V2_EXECUTE"


class BEdgeRuntimeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BEdgeRuntimeError(message)


def _create_only_json(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(document), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace_json(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(document), sort_keys=True,
                                    separators=(",", ":"), ensure_ascii=True,
                                    allow_nan=False), encoding="ascii")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class RuntimeArgumentsV2:
    request_b64: str
    campaign_config: Path
    ready_file: Path
    cell_id: str
    edge_port: int
    direct_map_host: str
    direct_map_port: int
    queue_depth: int

    def __post_init__(self) -> None:
        _require(type(self.request_b64) is str and bool(self.request_b64),
                 "request is empty")
        _require(type(self.cell_id) is str and bool(self.cell_id),
                 "cell_id is empty")
        for name, value in (("edge_port", self.edge_port),
                            ("direct_map_port", self.direct_map_port)):
            _require(type(value) is int and 1024 <= value <= 65535,
                     f"{name} is invalid")
        _require(type(self.direct_map_host) is str
                 and bool(self.direct_map_host.strip()),
                 "direct-map host is empty")
        _require(type(self.queue_depth) is int and self.queue_depth > 0,
                 "queue depth is invalid")


@dataclass(frozen=True, slots=True)
class RuntimeAuthoritiesV2:
    """Injected construction authorities; production values are loaded lazily."""

    torch: Any
    load_campaign: Callable[[Path], Mapping[str, Any]]
    service_deadline_s: Callable[[Mapping[str, Any]], float]
    ack_timeout_s: Callable[[Mapping[str, Any]], float]
    load_contract: Callable[[], Any]
    preload_edge: Callable[[Any], Any]
    snapshot: Callable[[Any], Any]
    processor_type: Any
    map_publisher_type: Any
    compute: Callable[..., Any]
    pending_type: Any
    warm_edge: Callable[..., Mapping[str, Any]]
    socket_factory: Callable[..., Any]
    raw_clock: Callable[[], int]


def real_factory() -> RuntimeAuthoritiesV2:  # pragma: no cover - live imports
    import torch

    from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
        preload_detached_optimized_edge_v3,
    )
    from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import (
        DetachedOptimizedTailAdapterV3,
    )
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_edge_runtime_v2 as qualified,
        phase6_prewarm_v2 as prewarm,
    )
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base
    from rl_agent.splitfusion_live_dispatch_v1.dynamic_execution_contract import (
        load_dynamic_execution_contract,
    )

    return RuntimeAuthoritiesV2(
        torch=torch,
        load_campaign=base._load_json,
        service_deadline_s=base.service_deadline_s,
        ack_timeout_s=base.ack_timeout_s,
        load_contract=load_dynamic_execution_contract,
        preload_edge=preload_detached_optimized_edge_v3,
        snapshot=DetachedOptimizedTailAdapterV3.snapshot,
        processor_type=qualified.Run4EdgeProcessorV2,
        map_publisher_type=qualified.Run4MapPublisherV2,
        compute=qualified.run4_compute_on_detached_runtime,
        pending_type=base.LatestFramePendingSlot,
        warm_edge=prewarm.warm_edge,
        socket_factory=socket.socket,
        raw_clock=lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
    )


@dataclass(slots=True)
class BuiltRuntimeV2:
    service: S.BEdgeDatagramServiceV2
    dispatcher: L.TailOutputDispatchV1
    publisher: Any
    control_socket: Any
    receiver_socket: Any
    prediction_store: B.PredictionEvidenceStoreV1
    prewarm_report: Mapping[str, Any]
    ready_document: Mapping[str, Any]
    report_file: Path
    _closed: bool = False

    def start(self) -> None:
        self.service.start()

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            return {"already_closed": True}
        result = self.service.close()
        self._closed = True
        return result
def _validated_request(encoded: str, *,
                       find_module: Callable[[str], Any] | None = None,
                       ) -> dict[str, Any]:
    """Select one exact schema without weakening either validator."""
    try:
        padding = "=" * (-len(encoded) % 4)
        preview = json.loads(base64.b64decode(
            encoded + padding, altchars=b"-_", validate=True).decode("ascii"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise BEdgeRuntimeError("runtime request cannot be decoded") from exc
    _require(type(preview) is dict, "runtime request is not an object")
    schema = preview.get("schema")
    if schema == Q.SCHEMA:
        raw = Q.decode_and_validate(encoded)
        if find_module is not None:
            for module in (*E.REQUIRED_AUTHORITIES, E.PROVEN_EDGE_MODULE):
                _require(find_module(module) is not None,
                         f"runtime authority is missing: {module}")
        return raw
    if schema == E.REQUEST_SCHEMA:
        if find_module is not None:
            E.preflight_request(encoded, find_module=find_module)
        raw = E._validate_request(E._decode_request(encoded))
        return {**raw,
                "purpose": "FROZEN_300_FRAME_QUALIFICATION",
                "claim_scope": "REGISTERED_QUALIFICATION",
                "transmitted_budget": 300}
    raise BEdgeRuntimeError("runtime request schema is unknown")



def runtime_preflight(args: RuntimeArgumentsV2, *,
                      find_module: Callable[[str], Any] = E._find_module,
                      ) -> Mapping[str, Any]:
    request = _validated_request(args.request_b64, find_module=find_module)
    _require(Path(args.campaign_config).is_file(),
             "campaign config is absent")
    _require(not Path(args.ready_file).exists(),
             "ready file must be create-only")
    attempt = Path(request["remote_attempt_root"])
    _require(not attempt.exists(), "remote attempt root must be create-only")
    return {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "run_id": request["run_id"],
        "variant": request["variant"],
        "cell_id": args.cell_id,
        "purpose": request["purpose"],
        "claim_scope": request["claim_scope"],
        "transmitted_budget": request["transmitted_budget"],
        "feature_transport": "UDP_SFD4_CHUNKED",
        "operational_ack_transport": "UDP_COMPACT_ACK",
        "direct_map_transport": "UDP_DIRECT_MAP",
        "tcp_listener": False,
        "gt_ingress": False,
        "qperc_reward_evaluator": False,
        "map_wait": False,
    }


def _safe_prediction_callback(store: B.PredictionEvidenceStoreV1,
                              raw_clock: Callable[[], int],
                              ) -> Callable[[L.ImmutableTailWorkV1],
                                            L.BranchCallbackResultV1]:
    retain = L.prediction_store_callback(store, clock=raw_clock)

    def callback(work: L.ImmutableTailWorkV1) -> L.BranchCallbackResultV1:
        C.decode(work.tail_output, expected_identity=work.identity)
        return retain(work)

    return callback


def build_runtime(args: RuntimeArgumentsV2, authorities: RuntimeAuthoritiesV2,
                  ) -> BuiltRuntimeV2:
    """Construct, bind and warm the real edge without starting worker loops."""
    request = _validated_request(args.request_b64)
    campaign = authorities.load_campaign(Path(args.campaign_config))
    _require(isinstance(campaign, Mapping)
             and isinstance(campaign.get("runtime"), Mapping),
             "campaign runtime binding is absent")
    runtime_cfg = campaign["runtime"]
    for field in ("udp_chunk_bytes", "socket_buffer_request_bytes"):
        _require(type(runtime_cfg.get(field)) is int
                 and int(runtime_cfg[field]) > 0,
                 f"campaign runtime field {field} is invalid")
    ack_remote = (request["split_host"]["ack_receiver_host"],
                  request["split_host"]["ack_receiver_port"])
    _require(ack_remote != (args.direct_map_host, args.direct_map_port),
             "ACK and map endpoints must be distinct")
    attempt = Path(request["remote_attempt_root"])
    _require(not attempt.exists(), "remote attempt root must be create-only")

    _require(authorities.torch.cuda.is_available(),
             "qualified CUDA edge is unavailable")
    device = authorities.torch.device("cuda:0")
    contract = authorities.load_contract()
    edge = authorities.preload_edge(device)
    rt = edge.runtime

    def compute(*, envelope: Any, context: Any, profile: Any,
                identity: Any) -> tuple[list[Any], Any]:
        computed = authorities.compute(
            rt, envelope=envelope, context=context,
            profile=profile, identity=identity)
        published = rt.publish_cpu(computed)
        snapshot = authorities.snapshot(published.serialized)
        labels = (snapshot.semantic_labels.detach()
                  .to(device="cpu", dtype=authorities.torch.uint8)
                  .contiguous().numpy())
        return list(snapshot.records or ()), labels

    processor = authorities.processor_type(
        contract=contract, run_id=request["run_id"], cell_id=args.cell_id,
        service_deadline_s=float(authorities.service_deadline_s(campaign)),
        ack_timeout_s=float(authorities.ack_timeout_s(campaign)),
        compute=compute)
    publisher = authorities.map_publisher_type(
        map_host=args.direct_map_host, map_port=args.direct_map_port,
        chunk_bytes=int(runtime_cfg["udp_chunk_bytes"]),
        socket_buffer_request_bytes=int(
            runtime_cfg["socket_buffer_request_bytes"]))
    control = authorities.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = authorities.socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                        int(runtime_cfg["socket_buffer_request_bytes"]))
    receiver.bind(("0.0.0.0", args.edge_port))
    receiver.settimeout(0.25)
    attempt.mkdir(parents=True)
    store = B.PredictionEvidenceStoreV1.create(attempt / "prediction")
    registry = E.MapDocumentRegistryV1(publisher.publish)
    dispatcher = L.TailOutputDispatchV1(
        ack_sender=lambda payload: control.sendto(payload, ack_remote),
        queue_depth=args.queue_depth,
        map_callback=registry.callback,
        prediction_callback=_safe_prediction_callback(
            store, authorities.raw_clock))
    seam = S.GTFreeEdgePostComputeSeamV2(
        dispatcher=dispatcher, register_map_document=registry.register)
    adapter = S.BEdgePostComputeAdapterV2(
        processor=processor, seam=seam, run_id=request["run_id"],
        cell_id=args.cell_id, raw_clock=authorities.raw_clock)

    closed = False
    dispatcher_started = False

    def close_components() -> None:
        nonlocal closed, dispatcher_started
        if closed:
            return
        # Service has stopped ingress/compute. Drain both independent branches
        # before closing their publisher/socket ownership.
        if dispatcher_started:
            dispatcher.stop()
            dispatcher_started = False
        publisher.close()
        control.close()
        closed = True

    service = S.BEdgeDatagramServiceV2(
        receiver=receiver,
        reassembler=ChunkReassembler(timeout_s=2.0, max_chunks=4096),
        pending=authorities.pending_type(), adapter=adapter,
        close_components=close_components)
    try:
        prewarm = authorities.warm_edge(
            processor, rt, contract, device=device,
            encoders=edge.autoencoders, codec=rt._codec,
            unguarded_tail=edge.tail)
        _require(prewarm.get("completed") is True,
                 "qualified edge prewarm did not complete")
        dispatcher.start()
        dispatcher_started = True
        ready = {
            "schema": READY_SCHEMA,
            "run_id": request["run_id"], "cell_id": args.cell_id,
            "variant": request["variant"],
            "purpose": request["purpose"],
            "claim_scope": request["claim_scope"],
            "transmitted_budget": request["transmitted_budget"],
            "edge_port": args.edge_port,
            "direct_map_host": args.direct_map_host,
            "direct_map_port": args.direct_map_port,
            "ack_receiver_host": ack_remote[0],
            "ack_receiver_port": ack_remote[1],
            "tail_device": str(device),
            "architecture": "GT_FREE_OPERATIONAL_ACK_BEFORE_ASYNC_BRANCHES_V2",
            "feature_transport": "UDP_SFD4_CHUNKED",
            "operational_ack_transport": "UDP_COMPACT_ACK",
            "tcp_listener": False,
            "gt_ingress": False, "qperc_reward_evaluator": False,
            "legacy_quality_feedback": False, "map_wait": False,
            "prediction_evidence_dir": str(attempt / "prediction"),
            "prewarm_completed": True,
        }
        return BuiltRuntimeV2(
            service=service, dispatcher=dispatcher, publisher=publisher,
            control_socket=control, receiver_socket=receiver,
            prediction_store=store, prewarm_report=prewarm,
            ready_document=ready, report_file=attempt / "B_EDGE_REPORT.json")
    except BaseException:
        receiver.close()
        close_components()
        raise


def run_live(args: RuntimeArgumentsV2, *,
             factory: Callable[[], RuntimeAuthoritiesV2] = real_factory,
             sleep: Callable[[float], None] = time.sleep) -> int:
    runtime_preflight(args)
    runtime = build_runtime(args, factory())
    _create_only_json(args.ready_file, runtime.ready_document)
    _create_only_json(runtime.report_file.with_name("B_EDGE_PREWARM.json"),
                      runtime.prewarm_report)
    stopping = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopping.set()

    prior_int = signal.signal(signal.SIGINT, stop)
    prior_term = signal.signal(signal.SIGTERM, stop)
    result: Mapping[str, Any] = {}
    exit_code = 0
    try:
        runtime.start()
        while (not stopping.is_set()
               and not runtime.service.stop_event.is_set()
               and all(thread.is_alive()
                       for thread in runtime.service.threads)):
            sleep(0.25)
        if (not stopping.is_set() and not runtime.service.failures
                and any(not thread.is_alive()
                        for thread in runtime.service.threads)):
            runtime.service.failures.append("runtime worker exited unexpectedly")
        exit_code = 0 if not runtime.service.failures else 1
    finally:
        result = runtime.close()
        signal.signal(signal.SIGINT, prior_int)
        signal.signal(signal.SIGTERM, prior_term)
        _replace_json(runtime.report_file, {
            "schema": REPORT_SCHEMA, "final": True,
            "run_id": runtime.ready_document["run_id"],
            "cell_id": runtime.ready_document["cell_id"],
            "counters": result.get("counters", {}),
            "failures": list(result.get("failures", ())),
            "pending_dropped_at_close": result.get(
                "pending_dropped_at_close", 0),
            "map_publications": getattr(runtime.publisher, "published", None),
            "branch_worker_errors": list(runtime.dispatcher.worker_errors),
            "gt_qperc_reward_used": False,
            "written_at_unix_s": time.time(),
        })
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "start"))
    parser.add_argument("--request-b64", required=True)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--edge-port", type=int, default=51002)
    parser.add_argument("--direct-map-host", required=True)
    parser.add_argument("--direct-map-port", type=int, required=True)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--execute")
    return parser


def _arguments(namespace: argparse.Namespace) -> RuntimeArgumentsV2:
    return RuntimeArgumentsV2(
        request_b64=namespace.request_b64,
        campaign_config=namespace.campaign_config,
        ready_file=namespace.ready_file, cell_id=namespace.cell_id,
        edge_port=namespace.edge_port,
        direct_map_host=namespace.direct_map_host,
        direct_map_port=namespace.direct_map_port,
        queue_depth=namespace.queue_depth)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(
        list(argv) if argv is not None else None)
    args = _arguments(namespace)
    if namespace.command == "preflight":
        print(json.dumps(runtime_preflight(args), sort_keys=True))
        return 0
    _require(namespace.execute == EXECUTE_TOKEN,
             "live start requires the exact execution token")
    return run_live(args)


if __name__ == "__main__":  # pragma: no cover - live entry
    try:
        raise SystemExit(main())
    except (BEdgeRuntimeError, E.BEdgeProcessError) as exc:
        print(f"B_EDGE_RUNTIME_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)

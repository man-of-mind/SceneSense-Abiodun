"""Additive GT-free edge service for Run-4B/Run-5B.

Qualified Phase-6 symbols remain authoritative for SFD4 verification,
continuous-q resolution, decode and tail compute.  This file owns only the
latest-only ingress loop and the post-tail ordering:

``tail ready -> synchronous operational ACK -> map/prediction queues``.

There is no GT ingress, Q_perc, reward, map wait, or legacy quality feedback.
Importing this file performs no I/O and imports neither torch nor Phase-6.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from phase2_map_sharing.transport import ChunkReassembler

from . import b_edge_process_v1 as E
from . import branch_evidence_v1 as B
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A
from . import tail_evidence_codec_v2 as C


PROVEN_EDGE_MODULE = (
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2."
    "phase6_edge_runtime_v2"
)
FORBIDDEN_RUNTIME_PREFIXES = (
    "rl_agent.splitfusion_quality_feedback_probe_v1.gt_evidence",
    "rl_agent.splitfusion_quality_feedback_probe_v1.scoring",
    "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality",
)
PREFLIGHT_SCHEMA = "scenesense.splitfusion.run4b5b.b_edge_service_preflight.v2"
OFFLINE_FAKE_SCHEMA = "scenesense.splitfusion.run4b5b.b_edge_service_fake.v2"


class BEdgeServiceError(RuntimeError):
    pass


class IngressIdentityConflict(BEdgeServiceError):
    pass


class IngressPayloadConflict(BEdgeServiceError):
    pass


def _require(condition: bool, message: str,
             error: type[BEdgeServiceError] = BEdgeServiceError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BEdgeServiceError("value is not canonicalizable") from exc


def assert_import_purity() -> None:
    loaded = tuple(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in FORBIDDEN_RUNTIME_PREFIXES)
    )
    _require(not loaded, "GT/Qperc module imported: "
             + ",".join(sorted(loaded)))


@dataclass(frozen=True, slots=True)
class QualifiedSymbolsV2:
    processor_type: Any
    map_publisher_type: Any
    compute: Callable[..., Any]


def load_qualified_symbols(
        importer: Callable[[str], Any] = importlib.import_module,
        ) -> QualifiedSymbolsV2:
    module = importer(PROVEN_EDGE_MODULE)
    result = QualifiedSymbolsV2(
        processor_type=getattr(module, "Run4EdgeProcessorV2", None),
        map_publisher_type=getattr(module, "Run4MapPublisherV2", None),
        compute=getattr(module, "run4_compute_on_detached_runtime", None),
    )
    _require(result.processor_type is not None,
             "qualified processor is unavailable")
    _require(result.map_publisher_type is not None,
             "qualified map publisher is unavailable")
    _require(callable(result.compute), "qualified compute is unavailable")
    return result


@dataclass(frozen=True, slots=True)
class VerifiedIngressV2:
    payload: bytes
    payload_sha256: str
    identity: A.FrameActionIdentityV1
    stream_id: str
    sequence_id: int
    frame_id: int
    received_wall_s: float


class LifetimeIngressIndexV2:
    """Lifetime logical-identity and exact-payload conflict index."""

    def __init__(self) -> None:
        self._rows: dict[
            tuple[Any, ...], tuple[A.FrameActionIdentityV1, str]
        ] = {}

    def admit(self, identity: A.FrameActionIdentityV1,
              payload_sha256: str) -> bool:
        _require(type(identity) is A.FrameActionIdentityV1,
                 "ingress identity has a foreign type")
        _require(type(payload_sha256) is str and len(payload_sha256) == 64
                 and all(ch in "0123456789abcdef" for ch in payload_sha256),
                 "payload digest is invalid")
        key = identity.decision_key()
        prior = self._rows.get(key)
        if prior is None:
            self._rows[key] = identity, payload_sha256
            return True
        if prior[0] != identity:
            raise IngressIdentityConflict(
                "logical decision has a different exact identity")
        if prior[1] != payload_sha256:
            raise IngressPayloadConflict(
                "exact identity has different SFD4 bytes")
        return False


class GTFreeEdgePostComputeSeamV2:
    """Neutral evidence codec plus the existing synchronous dispatcher."""

    def __init__(self, *, dispatcher: L.TailOutputDispatchV1,
                 register_map_document: Callable[
                     [A.FrameActionIdentityV1, Mapping[str, Any]], None]) -> None:
        _require(type(dispatcher) is L.TailOutputDispatchV1,
                 "dispatcher must be exactly TailOutputDispatchV1")
        _require(callable(register_map_document),
                 "map registrar is not callable")
        self._dispatcher = dispatcher
        self._register_map_document = register_map_document

    def emit(self, output: E.UsableTailOutputV1
             ) -> L.DispatchReceiptV1 | bool:
        _require(type(output) is E.UsableTailOutputV1,
                 "usable output has a foreign type")
        self._register_map_document(output.identity,
                                    dict(output.map_document))
        if not output.reward_requested:
            marker = _canonical({
                "identity_sha256": output.identity.exact_sha256(),
                "map_only": True,
            })
            return self._dispatcher.dispatch_map_only(
                output.identity, marker,
                tail_ready_monotonic_raw_ns=(
                    output.tail_ready_monotonic_raw_ns))
        prediction = C.encode(
            identity=output.identity,
            objects=output.object_records,
            semantic_mask=output.semantic_mask,
        )
        # TailOutputDispatchV1 sends the ACK synchronously before either offer.
        return self._dispatcher.dispatch_decision(
            output.identity, prediction,
            tail_ready_monotonic_raw_ns=output.tail_ready_monotonic_raw_ns)


class BEdgePostComputeAdapterV2:
    def __init__(self, *, processor: Any,
                 seam: GTFreeEdgePostComputeSeamV2,
                 run_id: str, cell_id: str,
                 raw_clock: Callable[[], int]) -> None:
        _require(callable(getattr(processor, "verify", None))
                 and callable(getattr(processor, "process", None)),
                 "processor lacks verify/process")
        _require(type(seam) is GTFreeEdgePostComputeSeamV2,
                 "post-compute seam has a foreign type")
        _require(type(run_id) is str and bool(run_id)
                 and type(cell_id) is str and bool(cell_id),
                 "run/cell identity is empty")
        _require(callable(raw_clock), "raw clock is not callable")
        self.processor = processor
        self.seam = seam
        self.run_id = run_id
        self.cell_id = cell_id
        self.raw_clock = raw_clock

    def _identity(self, envelope: Any, context: Any,
                  profile: Any) -> A.FrameActionIdentityV1:
        return L.identity_from_phase6(
            run_id=self.run_id, cell_id=self.cell_id,
            envelope=envelope, context=context, profile=profile,
            require_operational_ack=False)

    def verify(self, payload: bytes, *,
               received_wall_s: float) -> VerifiedIngressV2:
        _require(type(payload) is bytes and bool(payload),
                 "complete SFD4 payload is empty")
        envelope, context, profile, _identity = self.processor.verify(payload)
        identity = self._identity(envelope, context, profile)
        return VerifiedIngressV2(
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            identity=identity,
            stream_id=identity.stream_id,
            sequence_id=int(context.sequence_id),
            frame_id=identity.frame_id,
            received_wall_s=float(received_wall_s),
        )

    def process(self, item: VerifiedIngressV2
                ) -> L.DispatchReceiptV1 | bool:
        _require(type(item) is VerifiedIngressV2,
                 "pending ingress has a foreign type")
        _require(hashlib.sha256(item.payload).hexdigest()
                 == item.payload_sha256,
                 "payload changed after admission")
        processed = self.processor.process(
            item.payload,
            edge_timing={
                "reassembly_complete_wall_s": item.received_wall_s,
                "compute_start_wall_s": time.time(),
            })
        identity = self._identity(processed.envelope, processed.context,
                                  processed.profile)
        _require(identity == item.identity,
                 "identity changed between admission and compute")
        if processed.envelope.reward_requested:
            _require(processed.evaluation is not None,
                     "decision output lacks usable records/mask")
            records = tuple(processed.evaluation.records)
            mask = processed.evaluation.predicted_mask
        else:
            _require(processed.evaluation is None,
                     "hold produced a decision-only output")
            records, mask = (), None
        result = self.seam.emit(E.UsableTailOutputV1(
            identity=identity,
            reward_requested=processed.envelope.reward_requested,
            object_records=records,
            semantic_mask=mask,
            map_document=dict(processed.update),
            tail_ready_monotonic_raw_ns=int(self.raw_clock()),
        ))
        return result


class BEdgeDatagramServiceV2:
    """Own receiver/pending threads; caller-owned components close explicitly."""

    def __init__(self, *, receiver: Any, reassembler: ChunkReassembler,
                 pending: Any, adapter: BEdgePostComputeAdapterV2,
                 close_components: Callable[[], None],
                 wall_clock: Callable[[], float] = time.time,
                 monotonic_clock: Callable[[], float] = time.monotonic) -> None:
        _require(callable(getattr(receiver, "recvfrom", None))
                 and callable(getattr(receiver, "close", None)),
                 "receiver lacks recvfrom/close")
        _require(type(reassembler) is ChunkReassembler,
                 "reassembler has a foreign type")
        _require(all(callable(getattr(pending, name, None))
                     for name in ("offer", "take", "close")),
                 "pending slot lacks latest-only operations")
        _require(type(adapter) is BEdgePostComputeAdapterV2,
                 "adapter has a foreign type")
        _require(callable(close_components),
                 "component closer is not callable")
        self.receiver, self.reassembler = receiver, reassembler
        self.pending, self.adapter = pending, adapter
        self.close_components = close_components
        self.wall_clock, self.monotonic_clock = wall_clock, monotonic_clock
        self.index = LifetimeIngressIndexV2()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.counters: dict[str, int] = {}
        self.failures: list[str] = []
        self._closed = False

    def _bump(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def ingest_datagram(self, source: Any, datagram: bytes) -> None:
        complete = self.reassembler.ingest(
            str(source), datagram,
            received_at_s=float(self.monotonic_clock()))
        if complete is None:
            return
        item = self.adapter.verify(
            complete.payload, received_wall_s=float(self.wall_clock()))
        _require(item.frame_id == int(complete.message_id),
                 "chunk message ID differs from verified frame")
        if not self.index.admit(item.identity, item.payload_sha256):
            self._bump("exact_duplicates")
            return
        admitted, displaced = self.pending.offer(
            item.stream_id, item, sequence=item.sequence_id)
        if not admitted:
            self._bump("not_freshest")
            return
        self._bump("admitted")
        if displaced is not None:
            self._bump("superseded_pending")

    def process_one(self, timeout: float = 0.1) -> bool:
        taken = self.pending.take(timeout=float(timeout))
        if taken is None:
            return False
        self.adapter.process(taken[1])
        self._bump("processed")
        return True

    def _receive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                datagram, source = self.receiver.recvfrom(65535)
            except socket.timeout:
                self.reassembler.expire(float(self.monotonic_clock()))
                continue
            except OSError:
                return
            try:
                self.ingest_datagram(source, datagram)
            except Exception as exc:
                self.failures.append(f"receive:{type(exc).__name__}:{exc}")
                self.stop_event.set()
                self.pending.close()
                return

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.process_one()
            except Exception as exc:
                self.failures.append(f"process:{type(exc).__name__}:{exc}")
                self.stop_event.set()
                self.pending.close()
                return

    def start(self) -> None:
        _require(not self.threads and not self._closed,
                 "service was already started/closed")
        self.threads = [
            threading.Thread(target=self._receive_loop,
                             name="run4b5b-edge-receive", daemon=True),
            threading.Thread(target=self._process_loop,
                             name="run4b5b-edge-process", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            return {"counters": dict(self.counters),
                    "failures": tuple(self.failures),
                    "pending_dropped_at_close": 0}
        self.stop_event.set()
        dropped = self.pending.close()
        self.receiver.close()
        for thread in self.threads:
            thread.join(timeout=5.0)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        try:
            self.close_components()
        finally:
            self._closed = True
        _require(not alive, "service threads did not stop: " + ",".join(alive))
        return {"counters": dict(self.counters),
                "failures": tuple(self.failures),
                "pending_dropped_at_close": len(dropped)}


def preflight(request_b64: str, *,
              importer: Callable[[str], Any] = importlib.import_module,
              find_module: Callable[[str], Any] = E._find_module,
              ) -> Mapping[str, Any]:
    base = E.preflight_request(request_b64, find_module=find_module)
    symbols = load_qualified_symbols(importer)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": base["run_id"],
        "variant": base["variant"],
        "ack_receiver_host": base["ack_receiver_host"],
        "ack_receiver_port": base["ack_receiver_port"],
        "processor": symbols.processor_type.__name__,
        "map_publisher": symbols.map_publisher_type.__name__,
        "compute": symbols.compute.__name__,
        "gt_ingress": False,
        "qperc_or_reward": False,
        "legacy_quality_feedback": False,
    }


def offline_fake(request_b64: str, *, root: Path) -> Mapping[str, Any]:
    """Execute ACK-first neutral dispatch without Phase6/CUDA/network."""
    E.preflight_request(request_b64)
    root = Path(root)
    _require(not root.exists(), "offline fake root must be create-only")
    root.mkdir(parents=True)
    events: list[str] = []
    packets: list[bytes] = []
    registry = E.MapDocumentRegistryV1(
        lambda document: events.append("MAP") or {
            "frame_id": document["frame_id"]})
    store = B.PredictionEvidenceStoreV1.create(root / "prediction")
    retain = L.prediction_store_callback(store, clock=lambda: 7)

    def prediction(work: L.ImmutableTailWorkV1) -> L.BranchCallbackResultV1:
        C.decode(work.tail_output, expected_identity=work.identity)
        events.append("PREDICTION")
        return retain(work)

    dispatcher = L.TailOutputDispatchV1(
        ack_sender=lambda packet: (
            events.append("ACK"), packets.append(bytes(packet))),
        queue_depth=4, map_callback=registry.callback,
        prediction_callback=prediction)
    dispatcher.start()
    try:
        identity = E._fake_identity()
        seam = GTFreeEdgePostComputeSeamV2(
            dispatcher=dispatcher,
            register_map_document=registry.register)
        import numpy as np
        receipt = seam.emit(E.UsableTailOutputV1(
            identity=identity, reward_requested=True,
            object_records=({"class_name": "person"},),
            semantic_mask=np.asarray([[0, 1]], dtype=np.uint8),
            map_document={"frame_id": identity.frame_id},
            tail_ready_monotonic_raw_ns=1))
    finally:
        dispatcher.stop()
    _require(events and events[0] == "ACK", "ACK was not synchronous first")
    return {"schema": OFFLINE_FAKE_SCHEMA, "events": events,
            "identity_sha256": identity.exact_sha256(),
            "ack_packet_sha256": receipt.ack_packet_sha256,
            "map_enqueued": receipt.map_enqueued,
            "prediction_enqueued": receipt.prediction_enqueued,
            "gt_qperc_reward_used": False,
            "prediction_records": len(tuple(store.records.glob("*.json")))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "offline-fake"))
    parser.add_argument("--request-b64", required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "preflight":
        result = preflight(args.request_b64)
    else:
        _require(args.output_root is not None,
                 "offline-fake requires --output-root")

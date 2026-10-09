"""Production dependency factory for the W10275 B UE process.

This module constructs only already-proven components: Phase-6 live telemetry
readers, the registered dynamic execution contract, the frozen UE
front/ranker/AE/codec preload, the existing 7-channel input builder and the
UDP sender.  Run-5B additionally receives the frozen causal RFsim SNR lease
adapter; Run-4B refuses any SNR adapter.  Importing this module starts nothing
and does not import torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable, Mapping, Optional

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as R5B
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    phase6_prewarm_v2 as PW,
    continuous_execution_v2 as X,
    phase6_ue_runtime_v2 as P6,
    ue_telemetry_provider_v2 as T,
)
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as DEC

from . import live_adapters_v1 as L


class ProductionDependencyError(RuntimeError): pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise ProductionDependencyError(message)


class CausalSnrLeaseReaderV1:
    """Exact Run-5B live lease admission exposed as the processor protocol."""

    def __init__(self, adapter: SNR.RfsimLeaseSnrAdapterV1) -> None:
        _require(isinstance(adapter, SNR.RfsimLeaseSnrAdapterV1),
                 "SNR adapter is not the frozen live adapter")
        self.adapter = adapter
        self.lease = R5B.lease_policy()

    def observe_db(self, *, cutoff_raw_ns: int, session_uuid: str,
                   decision_seq: int) -> float:
        _require(session_uuid == self.adapter.session_uuid,
                 "SNR/telemetry session mismatch")
        _require(type(cutoff_raw_ns) is int and cutoff_raw_ns > 0,
                 "SNR cutoff is invalid")
        boundary = R4.DecisionBoundaryV1(
            identity=R4.DecisionIdentityV1(
                session_uuid, self.adapter.ue_id, decision_seq),
            state_commit_timestamp_ns=cutoff_raw_ns - 1,
            action_open_timestamp_ns=cutoff_raw_ns,
            clock_domain=SNR.LIVE_CLOCK_DOMAIN)
        observation = self.adapter.observe(boundary)
        return R5B.admit_snr(observation, boundary, self.lease)


class UplinkKeepaliveV1:
    """Small periodic UE uplink datagrams to the external DN (never the edge).

    The causal radio guard needs a fresh (<= one tensor period) new-data UL
    grant before a decision.  An idle UE gets none, so until the first
    decision has been processed this keeps grants fresh.  ``request_stop`` only
    signals (it never blocks the decision path); ``stop`` joins.
    """

    PERIOD_S = 0.05
    PAYLOAD = b"\0" * 64

    def __init__(self, sock: Any, destination: tuple[str, int]) -> None:
        self._socket, self._destination = sock, destination
        self._stop = threading.Event()
        self.sent = 0
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="b-uplink-keepalive")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.PERIOD_S):
                self._socket.sendto(self.PAYLOAD, self._destination)
                self.sent += 1
        except BaseException as exc:  # surfaced by stop()
            self.error = exc

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._socket.close()
        _require(not self._thread.is_alive(), "uplink keepalive did not stop")
        if self.error is not None:
            raise ProductionDependencyError(
                f"uplink keepalive failed: {self.error}")


@dataclass(slots=True)
class ProductionDependenciesV1:
    variant: L.ActorVariant
    telemetry: T.UeTelemetryProviderV2
    telemetry_readers: tuple[T.LiveEventReaderV2, ...]
    telemetry_audit: T.AuditWriterV2
    dynamic_contract: DEC.DynamicExecutionContract
    continuous_ue: X.ContinuousUERuntimeV2
    sender: Any
    remote: tuple[str, int]
    chunk_bytes: int
    input_builder: Callable[[Any, Any], Any]
    models: tuple[Any, ...]
    snr_controller_adapter: Optional[SNR.RfsimLeaseSnrAdapterV1]
    snr_reader: Optional[CausalSnrLeaseReaderV1]
    uplink_keepalive: Optional[UplinkKeepaliveV1] = None

    def close(self) -> None:
        errors = []
        if self.uplink_keepalive is not None:
            try: self.uplink_keepalive.stop()
            except BaseException as exc: errors.append(exc)
        for reader in self.telemetry_readers:
            try: reader.stop()
            except BaseException as exc: errors.append(exc)
        try: self.telemetry_audit.stop()
        except BaseException as exc: errors.append(exc)
        try: self.sender.close()
        except BaseException as exc: errors.append(exc)
        if errors:
            raise ProductionDependencyError(
                "dependency cleanup failed: " + "; ".join(
                    f"{type(exc).__name__}: {exc}" for exc in errors))


def build_production_dependencies_v1(
        *, variant: L.ActorVariant, tracer_dir: Path, t_messages: Path,
        ue_relay_port: int, telemetry_root: Path,
        ue_bind_host: str, edge_remote_host: str, edge_receive_port: int,
        udp_chunk_bytes: int, socket_buffer_request_bytes: int,
        wait_timeout_s: float = 20.0,
        telemetry_ready_timeout_s: float = 30.0,
        warmup_destination: Optional[tuple[str, int]] = None,
        ) -> ProductionDependenciesV1:
    """Start the exact UE-side dependencies; no CARLA/route is started here.

    ``telemetry_root`` is owned exclusively by these dependencies: it must not
    exist, its parent must, and only ``telemetry_root/telemetry_live`` is
    written.  It must never be (or contain) the UE output/evidence roots,
    which the UE execution loop alone creates.
    """
    _require(type(variant) is L.ActorVariant, "actor variant is foreign")
    telemetry_root = Path(telemetry_root)
    _require(telemetry_root.parent.is_dir(),
             "telemetry root parent (the attempt root) is absent")
    _require(not telemetry_root.exists() and not telemetry_root.is_symlink(),
             "telemetry root must be create-only")
    for path, label in ((Path(tracer_dir), "tracer directory"),
                        (Path(t_messages), "T_messages")):
        _require(path.exists(), f"{label} is absent")
    _require(type(ue_relay_port) is int and 1024 <= ue_relay_port <= 65535,
             "UE relay port is invalid")
    _require(type(edge_receive_port) is int and 1024 <= edge_receive_port <= 65535,
             "edge receive port is invalid")
    _require(type(udp_chunk_bytes) is int and udp_chunk_bytes > 0,
             "UDP chunk size is invalid")
    _require(type(socket_buffer_request_bytes) is int
             and socket_buffer_request_bytes > 0,
             "socket-buffer request is invalid")

    import torch
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as BASE

    _require(torch.cuda.is_available(), "CUDA is unavailable for the UE front")
    device = torch.device("cuda:0")
    ue, _ledger, models = BASE._preload_ue(device)  # exact proven preload
    dynamic = DEC.load_dynamic_execution_contract()
    continuous = X.ContinuousUERuntimeV2(
        dynamic, front=P6._SqueezedFront(ue._front, device),
        ranker=ue._ranker, ae_encoders=dict(ue._ae_encoders), codec=ue._codec)
    # Warm every registered UE path (12 modes x lower/mid/upper q, then a hot
    # repeat) with the frozen Phase-6 pre-warm before any frame is admitted:
    # a cold first CUDA front/codec call otherwise consumes the ACK budget.
    prewarm = PW.warm_ue(
        continuous, dynamic,
        prepare_input=lambda frame, radar: BASE._prepare_live_input(
            frame, radar, device))
    _require(bool(prewarm.get("completed"))
             and prewarm.get("modes_warmed") == list(range(12)),
             "UE pre-warm incomplete; no decision frame may be admitted")

    telemetry = T.UeTelemetryProviderV2(bridge=T.CausalClockBridgeV2())
    handlers = {"NRUE_MAC_DCI_GRANT": telemetry.on_dci,
                "NRUE_MAC_RLC_BUFFER_STATUS": telemetry.on_rlc,
                "NR_PDCP_TX_SDU": telemetry.on_pdcp}
    readers = []
    audit = None
    sender = None
    warm = None
    keepalive = None
    try:
        # Every started process/writer is inside the protected region, so a
        # failure at any step stops whatever was already started.
        for event, handler in handlers.items():
            reader = T.LiveEventReaderV2(event, handler, telemetry)
            readers.append(reader)
            reader.start(T.csv_reader_argv(Path(tracer_dir), Path(t_messages),
                                           ue_relay_port, event),
                         cwd=Path(tracer_dir))
        telemetry_root.mkdir(parents=False, exist_ok=False)
        PW.write_report_create_only(telemetry_root / "prewarm_ue.json", prewarm)
        audit = T.AuditWriterV2(readers, telemetry_root / "telemetry_live")
        audit.start()
        deadline = time.monotonic() + float(wait_timeout_s)
        while time.monotonic() < deadline and not (
                telemetry.snapshot().all_readers_alive
                and len(telemetry.unbound_ue_candidates) == 1):
            time.sleep(0.05)
        _require(len(telemetry.unbound_ue_candidates) == 1,
                 "exactly one UE must be visible in live telemetry")
        rnti, ue_id = sorted(telemetry.unbound_ue_candidates)[0]
        telemetry.bind_ue(rnti=rnti, oai_ue_id=ue_id)
        # Decision readiness: the causal radio guard refuses any decision until
        # the clock bridge is warm and UL-grant/RLC samples exist.  Wait for
        # that here (bounded, fail-closed) so the first route opportunity is
        # not consumed by a warming telemetry pipeline.
        # The clock bridge is anchored only by UE PDCP TX SDUs, i.e. uplink
        # traffic, which an idle UE does not send.  Small pre-decision
        # datagrams from the UE tunnel to the external DN (never the edge)
        # provide those anchors; they stop as soon as telemetry is ready.
        if warmup_destination is not None:
            warm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            warm.bind((str(ue_bind_host), 0))
        warm_datagrams = 0
        ready_deadline = time.monotonic() + float(telemetry_ready_timeout_s)
        while True:
            snapshot = telemetry.snapshot()
            if (snapshot.all_readers_alive and snapshot.bridge_warm
                    and len(snapshot.dci) > 0 and len(snapshot.rlc) > 0):
                break
            _require(time.monotonic() < ready_deadline,
                     "UE telemetry not decision-ready: "
                     f"readers_alive={snapshot.all_readers_alive} "
                     f"bridge_warm={snapshot.bridge_warm} "
                     f"dci={len(snapshot.dci)} rlc={len(snapshot.rlc)} "
                     f"warm_datagrams={warm_datagrams}")
            if warm is not None:
                warm.sendto(b"\0" * 64, (str(warmup_destination[0]),
                                          int(warmup_destination[1])))
                warm_datagrams += 1
            time.sleep(0.1)
        keepalive = None
        if warm is not None:
            # Ownership of the warm socket passes to the keepalive, which
            # keeps UL grants fresh until the first decision is processed.
            keepalive = UplinkKeepaliveV1(
                warm, (str(warmup_destination[0]), int(warmup_destination[1])))
            warm = None
            keepalive.start()
        print(f"[b-deps] telemetry decision-ready after {warm_datagrams} "
              "warm-up datagram(s); uplink keepalive "
              f"{'running' if keepalive is not None else 'off'}", flush=True)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                          socket_buffer_request_bytes)
        sender.bind((str(ue_bind_host), 0))
        snr_adapter = snr_reader = None
        if variant is L.ActorVariant.RUN5B:
            assert telemetry.ue_label is not None
            snr_adapter = SNR.RfsimLeaseSnrAdapterV1(
                provider_id="splitfusion.run5b.live_effective_snr.v1",
                session_uuid=telemetry.session_uuid, ue_id=telemetry.ue_label)
            snr_reader = CausalSnrLeaseReaderV1(snr_adapter)
        return ProductionDependenciesV1(
            variant=variant, telemetry=telemetry,
            telemetry_readers=tuple(readers), telemetry_audit=audit,
            dynamic_contract=dynamic, continuous_ue=continuous, sender=sender,
            remote=(str(edge_remote_host), edge_receive_port),
            chunk_bytes=udp_chunk_bytes,
            input_builder=lambda frame, radar: BASE._prepare_live_input(
                frame, radar, device),
            models=tuple(models), snr_controller_adapter=snr_adapter,
            snr_reader=snr_reader, uplink_keepalive=keepalive)
    except BaseException:
        for reader in readers:
            try: reader.stop()
            except BaseException: pass
        if audit is not None:
            try: audit.stop()
            except BaseException: pass
        if sender is not None:
            try: sender.close()
            except BaseException: pass
        if warm is not None:
            try: warm.close()
            except BaseException: pass
        if keepalive is not None:
            try: keepalive.stop()
            except BaseException: pass
        raise


__all__ = [
    "ProductionDependencyError", "CausalSnrLeaseReaderV1",
    "ProductionDependenciesV1", "build_production_dependencies_v1",
]

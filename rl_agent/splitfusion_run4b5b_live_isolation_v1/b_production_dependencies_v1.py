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
import time
from typing import Any, Callable, Mapping, Optional

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as R5B
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
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

    def close(self) -> None:
        errors = []
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
        ue_relay_port: int, evidence_dir: Path,
        ue_bind_host: str, edge_remote_host: str, edge_receive_port: int,
        udp_chunk_bytes: int, socket_buffer_request_bytes: int,
        wait_timeout_s: float = 20.0,
        ) -> ProductionDependenciesV1:
    """Start the exact UE-side dependencies; no CARLA/route is started here."""
    _require(type(variant) is L.ActorVariant, "actor variant is foreign")
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

    telemetry = T.UeTelemetryProviderV2(bridge=T.CausalClockBridgeV2())
    handlers = {"NRUE_MAC_DCI_GRANT": telemetry.on_dci,
                "NRUE_MAC_RLC_BUFFER_STATUS": telemetry.on_rlc,
                "NR_PDCP_TX_SDU": telemetry.on_pdcp}
    readers = []
    for event, handler in handlers.items():
        reader = T.LiveEventReaderV2(event, handler, telemetry)
        reader.start(T.csv_reader_argv(Path(tracer_dir), Path(t_messages),
                                       ue_relay_port, event),
                     cwd=Path(tracer_dir))
        readers.append(reader)
    audit = T.AuditWriterV2(readers, Path(evidence_dir) / "telemetry_live")
    audit.start()
    sender = None
    try:
        deadline = time.monotonic() + float(wait_timeout_s)
        while time.monotonic() < deadline and not (
                telemetry.snapshot().all_readers_alive
                and len(telemetry.unbound_ue_candidates) == 1):
            time.sleep(0.05)
        _require(len(telemetry.unbound_ue_candidates) == 1,
                 "exactly one UE must be visible in live telemetry")
        rnti, ue_id = sorted(telemetry.unbound_ue_candidates)[0]
        telemetry.bind_ue(rnti=rnti, oai_ue_id=ue_id)
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
            snr_reader=snr_reader)
    except BaseException:
        for reader in readers:
            try: reader.stop()
            except BaseException: pass
        try: audit.stop()
        except BaseException: pass
        if sender is not None:
            try: sender.close()
            except BaseException: pass
        raise


__all__ = [
    "ProductionDependencyError", "CausalSnrLeaseReaderV1",
    "ProductionDependenciesV1", "build_production_dependencies_v1",
]

"""Preloaded edge-side catalog validation, decode dispatch and frozen-tail call."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from .envelope import EnvelopeError, SplitEnvelope, unpack_envelope
from .registry import DispatchContractError, SplitActionRegistry
from .runtime_support import OperationCounters, OperationSnapshot, prepare_preloaded_module
from .timing import EDGE_STAGES, StageRecorder, TimingTrace
from .transport import (
    DecodedC2,
    EdgeSplitCodec,
    ProductionSplitCodec,
    require_inner_agreement,
)
from .ue_runtime import AE_FAMILIES, DispatchMetadata, _validate_preloaded_ae, metadata_for


@dataclass(frozen=True)
class EdgeDispatchResult:
    perception: Any
    serialized_output: Any
    metadata: DispatchMetadata
    scientific_inner_payload_bytes: int
    framing_control_overhead_bytes: int
    total_received_bytes: int
    timing: TimingTrace


class PreloadedSplitEdgeRuntime:
    """Own one frozen p025 tail path and three already-resident AE decoders."""

    def __init__(
        self,
        registry: SplitActionRegistry,
        *,
        frozen_p025_tail: Callable[[Any, DispatchMetadata], Any],
        ae_decoders: Mapping[str, Any],
        tail_device: torch.device,
        codec: EdgeSplitCodec | None = None,
        output_serializer: Callable[[Any], Any] | None = None,
        prepare_modules: bool = True,
        startup_model_load_operations: int = 0,
        startup_model_construction_operations: int = 0,
    ) -> None:
        if not isinstance(tail_device, torch.device):
            raise DispatchContractError("tail_device must be a torch.device")
        if not callable(frozen_p025_tail):
            raise DispatchContractError("preloaded frozen p025 tail must be callable")
        if set(ae_decoders) != set(AE_FAMILIES):
            raise DispatchContractError("edge requires exactly the AE128/AE64/AE32 decoders")
        self._registry = registry
        self._tail = frozen_p025_tail
        self._ae_decoders = MappingProxyType(dict(ae_decoders))
        self._tail_device = tail_device
        self._codec = codec if codec is not None else ProductionSplitCodec()
        self._output_serializer = output_serializer or (lambda _output: None)
        self._counters = OperationCounters(
            startup_model_load_operations=startup_model_load_operations,
            startup_model_construction_operations=startup_model_construction_operations,
            startup_preloaded_objects=4,
        )
        for family, decoder in self._ae_decoders.items():
            _validate_preloaded_ae(registry, family, decoder, "decoder")
        if prepare_modules:
            prepare_preloaded_module(self._tail, tail_device, self._counters)
            for decoder in self._ae_decoders.values():
                prepare_preloaded_module(decoder, tail_device, self._counters)

    @property
    def tail_device(self) -> torch.device:
        return self._tail_device

    @property
    def counters(self) -> OperationSnapshot:
        return self._counters.snapshot()

    @staticmethod
    def _outer(frame_bytes: bytes | bytearray | memoryview) -> SplitEnvelope:
        try:
            return unpack_envelope(frame_bytes)
        except EnvelopeError as exc:
            raise DispatchContractError(f"outer envelope rejected: {exc}") from exc

    def process(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        transmitted_action_id: int,
    ) -> EdgeDispatchResult:
        timing = StageRecorder(EDGE_STAGES)
        self._counters.frames_attempted += 1
        with timing.stage("total_edge_processing"):
            outer = self._outer(frame_bytes)
            if outer.action_id != transmitted_action_id:
                raise DispatchContractError(
                    "control-plane action_id disagrees with outer envelope"
                )
            profile = self._registry.resolve(transmitted_action_id)
            inspected = self._codec.inspect(outer.inner_payload, timing=timing)
            require_inner_agreement(profile, inspected.identity)
            decoder = (
                None
                if profile.family == "noAE"
                else self._ae_decoders.get(profile.family)
            )
            if profile.family != "noAE" and decoder is None:
                raise DispatchContractError(
                    f"preloaded {profile.family} decoder unavailable"
                )
            if decoder is not None:
                self._counters.ae_decoder_dispatches += 1
            with torch.inference_mode():
                decoded: DecodedC2 = self._codec.decode(
                    inspected,
                    decoder=decoder,
                    tail_device=self._tail_device,
                    timing=timing,
                )
            if not decoded.finite:
                raise DispatchContractError("reconstructed C2 is non-finite")
            if decoded.device != self._tail_device:
                raise DispatchContractError(
                    f"reconstructed C2 device {decoded.device} != tail device {self._tail_device}"
                )
            metadata = metadata_for(
                profile,
                sequence_id=outer.sequence_id,
                capture_timestamp_ns=outer.capture_timestamp_ns,
            )
            with timing.stage("frozen_tail"):
                self._counters.tail_dispatches += 1
                with torch.inference_mode():
                    perception = self._tail(decoded.c2, metadata)
            with timing.stage("output_serialization"):
                serialized = self._output_serializer(perception)
        self._counters.frames_completed += 1
        return EdgeDispatchResult(
            perception=perception,
            serialized_output=serialized,
            metadata=metadata,
            scientific_inner_payload_bytes=outer.inner_payload_length,
            framing_control_overhead_bytes=outer.control_overhead_bytes,
            total_received_bytes=outer.total_transmitted_bytes,
            timing=timing.snapshot(),
        )

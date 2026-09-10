"""Split the qualified edge runtime at its CPU publication boundary.

The deployed runtime deliberately keeps decode, the frozen tail and output
serialization inside one synchronous call.  The freshness scheduler needs a
different ownership boundary: one thread owns all decoder/tail/CUDA work and
a second thread owns CPU-only service-record serialization.  This additive
candidate preserves the deployed validation and codec sequence byte for byte;
it changes only where the already-qualified detached tail product is handed
off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_agent.splitfusion_live_dispatch_v1.edge_runtime import (
    PreloadedSplitEdgeRuntime,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
from rl_agent.splitfusion_live_dispatch_v1.timing import (
    EDGE_STAGES,
    StageRecorder,
    TimingTrace,
)
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    DecodedC2,
    require_inner_agreement,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for

from .detached_tail import (
    DetachedOptimizedTailAdapter,
    DetachedSerializedTailProduct,
    DetachedTailWorkProduct,
)


@dataclass(frozen=True)
class DetachedEdgeComputeResult:
    work: DetachedTailWorkProduct
    metadata: Any
    scientific_inner_payload_bytes: int
    framing_control_overhead_bytes: int
    total_received_bytes: int
    timing: TimingTrace


@dataclass(frozen=True)
class DetachedEdgePublicationResult:
    computed: DetachedEdgeComputeResult
    serialized: DetachedSerializedTailProduct
    timing: TimingTrace


class DetachedPreloadedSplitEdgeRuntime(PreloadedSplitEdgeRuntime):
    """Production validation/decode with detached CPU-only publication."""

    def __init__(self, *args: Any, detached_tail: DetachedOptimizedTailAdapter,
                 **kwargs: Any) -> None:
        if not isinstance(detached_tail, DetachedOptimizedTailAdapter):
            raise DispatchContractError("detached optimized tail is required")
        self._detached_tail = detached_tail
        super().__init__(
            *args,
            frozen_p025_tail=detached_tail,
            output_serializer=lambda _output: None,
            **kwargs,
        )

    def process_compute(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        transmitted_action_id: int,
    ) -> DetachedEdgeComputeResult:
        """Run through the frozen tail on the sole compute/CUDA owner."""

        timing = StageRecorder(EDGE_STAGES)
        self._counters.frames_attempted += 1
        with timing.stage("total_edge_processing"):
            outer = self._outer(frame_bytes)
            if outer.action_id != transmitted_action_id:
                raise DispatchContractError(
                    "control-plane action_id disagrees with outer envelope"
                )
            profile = self._registry.resolve(transmitted_action_id)
            context = outer.frame_context
            if self._require_frame_context and context is None:
                raise DispatchContractError("SFD1 v2 frame context is required")
            if context is not None:
                if self._camera_registry is None:
                    raise DispatchContractError(
                        "SFD1 v2 frame context has no edge camera registry"
                    )
                self._camera_registry.resolve(
                    context.camera_model_sha256, context.camera_mount_sha256
                )
                self._context_session.accept(context)
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
                    f"reconstructed C2 device {decoded.device} != "
                    f"tail device {self._tail_device}"
                )
            metadata = metadata_for(
                profile,
                sequence_id=outer.sequence_id,
                capture_timestamp_ns=outer.capture_timestamp_ns,
                protocol_version=outer.protocol_version,
                frame_context=context,
            )
            with timing.stage("frozen_tail"):
                self._counters.tail_dispatches += 1
                with torch.inference_mode():
                    work = self._detached_tail.compute_product(decoded.c2, metadata)
        self._counters.frames_completed += 1
        return DetachedEdgeComputeResult(
            work=work,
            metadata=metadata,
            scientific_inner_payload_bytes=outer.inner_payload_length,
            framing_control_overhead_bytes=outer.control_overhead_bytes,
            total_received_bytes=outer.total_transmitted_bytes,
            timing=timing.snapshot(),
        )

    def publish_cpu(
        self, computed: DetachedEdgeComputeResult
    ) -> DetachedEdgePublicationResult:
        """Serialize one detached product without accessing CUDA tensors."""

        timing = StageRecorder(EDGE_STAGES)
        with timing.stage("output_serialization"):
            serialized = self._detached_tail.serialize_product(computed.work)
        return DetachedEdgePublicationResult(
            computed=computed,
            serialized=serialized,
            timing=timing.snapshot(),
        )

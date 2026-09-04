"""Preloaded UE-side dispatcher for the 72 locked SPLIT actions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .envelope import HEADER_BYTES, PROTOCOL_VERSION, pack_envelope
from .registry import ActionProfile, DispatchContractError, SplitActionRegistry
from .runtime_support import OperationCounters, OperationSnapshot, prepare_preloaded_module
from .timing import StageRecorder, TimingTrace, UE_STAGES
from .transport import ProductionSplitCodec, UESplitCodec


AE_FAMILIES = ("AE128", "AE64", "AE32")


@dataclass(frozen=True)
class DispatchMetadata:
    protocol_version: int
    action_id: int
    profile_id: str
    sequence_id: int
    capture_timestamp_ns: int
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    q_e4: int
    keep_count: int
    routing_tag: int
    latent_width: int | None
    wire_codec_id: int
    wire_version: int
    segmentation_installable: bool
    segmentation_behavior: str


@dataclass(frozen=True)
class EncodedSplitFrame:
    wire_bytes: bytes
    inner_payload_bytes: int
    outer_envelope_bytes: int
    total_transmitted_bytes: int
    metadata: DispatchMetadata
    timing: TimingTrace


def metadata_for(
    profile: ActionProfile,
    *,
    sequence_id: int,
    capture_timestamp_ns: int,
) -> DispatchMetadata:
    return DispatchMetadata(
        protocol_version=PROTOCOL_VERSION,
        action_id=profile.action_id,
        profile_id=profile.profile_id,
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        family=profile.family,
        family_id=profile.family_id,
        quantizer=profile.quantizer,
        bit_width=profile.bit_width,
        q_e4=profile.q_e4,
        keep_count=profile.keep_count,
        routing_tag=profile.routing_tag,
        latent_width=profile.latent_width,
        wire_codec_id=profile.wire.codec_id,
        wire_version=profile.wire.version,
        segmentation_installable=profile.segmentation_installable,
        segmentation_behavior=profile.segmentation_behavior,
    )


def _validate_preloaded_ae(
    registry: SplitActionRegistry,
    family: str,
    module: Any,
    role: str,
) -> None:
    reference = registry.find(family, "UINT8", 0)
    if (
        int(getattr(module, "family_id", -1)) != reference.family_id
        or int(getattr(module, "bottleneck", -1)) != reference.transported_channels
        or int(getattr(module, "routing_tag", -1)) != reference.routing_tag
    ):
        raise DispatchContractError(f"preloaded {family} {role} identity disagrees with catalog")


class PreloadedSplitUERuntime:
    """Own references to one front, one ranker and three resident AE encoders."""

    def __init__(
        self,
        registry: SplitActionRegistry,
        *,
        front: Any,
        ranker: Any,
        ae_encoders: Mapping[str, Any],
        device: torch.device,
        codec: UESplitCodec | None = None,
        prepare_modules: bool = True,
        startup_model_load_operations: int = 0,
        startup_model_construction_operations: int = 0,
    ) -> None:
        if not isinstance(device, torch.device):
            raise DispatchContractError("UE device must be a torch.device")
        if not callable(front):
            raise DispatchContractError("preloaded UE front must be callable")
        if not hasattr(ranker, "score_cells"):
            raise DispatchContractError("preloaded ranker must expose score_cells")
        if set(ae_encoders) != set(AE_FAMILIES):
            raise DispatchContractError("UE requires exactly the AE128/AE64/AE32 encoders")
        self._registry = registry
        self._device = device
        self._front = front
        self._ranker = ranker
        self._ae_encoders = MappingProxyType(dict(ae_encoders))
        self._codec = codec if codec is not None else ProductionSplitCodec()
        self._counters = OperationCounters(
            startup_model_load_operations=startup_model_load_operations,
            startup_model_construction_operations=startup_model_construction_operations,
            startup_preloaded_objects=5,
        )
        for family, encoder in self._ae_encoders.items():
            _validate_preloaded_ae(registry, family, encoder, "encoder")
        if prepare_modules:
            prepare_preloaded_module(self._front, device, self._counters)
            prepare_preloaded_module(self._ranker, device, self._counters)
            for encoder in self._ae_encoders.values():
                prepare_preloaded_module(encoder, device, self._counters)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def counters(self) -> OperationSnapshot:
        return self._counters.snapshot()

    def prepare(
        self,
        action_id: int,
        input_7ch: Any,
        *,
        sequence_id: int,
        capture_timestamp_ns: int,
    ) -> EncodedSplitFrame:
        profile = self._registry.resolve(action_id)
        timing = StageRecorder(UE_STAGES)
        self._counters.frames_attempted += 1
        with timing.stage("total_ue_preparation"):
            with timing.stage("front_backbone"):
                with torch.inference_mode():
                    c2 = self._front(input_7ch)
                if isinstance(c2, torch.Tensor):
                    if c2.ndim == 4 and int(c2.shape[0]) == 1:
                        c2 = c2[0]
                    elif c2.ndim != 3:
                        raise DispatchContractError("UE front must return one C2 frame")
                    if c2.device != self._device:
                        raise DispatchContractError(
                            f"UE front returned C2 on {c2.device}, expected {self._device}"
                        )
            ranker = None if profile.q_e4 == 0 else self._ranker
            encoder = None if profile.family == "noAE" else self._ae_encoders[profile.family]
            if ranker is not None:
                self._counters.ranker_dispatches += 1
            if encoder is not None:
                self._counters.ae_encoder_dispatches += 1
            with torch.inference_mode():
                inner = self._codec.encode(
                    profile,
                    c2,
                    ranker=ranker,
                    ae_encoder=encoder,
                    timing=timing,
                )
            if not isinstance(inner, bytes) or not inner:
                raise DispatchContractError("UE codec returned an empty or non-bytes payload")
            wire = pack_envelope(
                inner,
                action_id=profile.action_id,
                sequence_id=sequence_id,
                capture_timestamp_ns=capture_timestamp_ns,
            )
        self._counters.frames_completed += 1
        return EncodedSplitFrame(
            wire_bytes=wire,
            inner_payload_bytes=len(inner),
            outer_envelope_bytes=HEADER_BYTES,
            total_transmitted_bytes=len(wire),
            metadata=metadata_for(
                profile,
                sequence_id=sequence_id,
                capture_timestamp_ns=capture_timestamp_ns,
            ),
            timing=timing.snapshot(),
        )

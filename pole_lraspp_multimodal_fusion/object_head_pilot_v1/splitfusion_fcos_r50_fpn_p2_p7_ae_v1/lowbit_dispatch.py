"""Per-packet receive and decoder dispatch for UINT6/UINT4 feature frames.

Only compressed bytes cross this interface.  The inner low-bit header decides
whether the frame is a noAE C2 tensor or which already-loaded AE decoder may
reconstruct it.  The zstd frame is decompressed exactly once; profile identity
is never taken from mutable caller state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import ZstdWireCodec
from . import ae_contract, ae_uint8_transport, lowbit_transport
from .ae_model import SplitFeatureAE


@dataclass(frozen=True)
class LowBitWireFamily:
    family_id: int
    family_name: str
    transported_channels: int
    routing_tag: int
    bit_width: int


@dataclass(frozen=True)
class LowBitReceiveDiagnostics:
    parsed: lowbit_transport.InspectedLowBitPayload
    decoded_feature: torch.Tensor
    keep_mask: torch.Tensor
    decoder: SplitFeatureAE | None


@dataclass(frozen=True)
class ReceivedLowBitFrame:
    c2: torch.Tensor
    q: float
    family: LowBitWireFamily
    keep_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    diagnostics: LowBitReceiveDiagnostics | None = None


class PreloadedLowBitDecoders:
    """Read-only view over AE decoders constructed by the existing runtime."""

    def __init__(self, autoencoders: Iterable[SplitFeatureAE]) -> None:
        by_family: dict[int, SplitFeatureAE] = {}
        for autoencoder in autoencoders:
            if not isinstance(autoencoder, SplitFeatureAE):
                raise guards.HybridQConfigError(
                    "preloaded low-bit decoders must be SplitFeatureAE instances"
                )
            family = ae_contract.family_for_bottleneck(autoencoder.bottleneck)
            if family != autoencoder.family_id:
                raise guards.HybridQConfigError(
                    "preloaded AE family disagrees with its latent width"
                )
            if family in by_family:
                raise guards.HybridQConfigError("an AE family was supplied twice")
            ae_contract.require_bound_routing_tag(
                autoencoder.routing_tag,
                what=f"preloaded {autoencoder.family_name} routing tag",
            )
            by_family[family] = autoencoder
        self._by_family = by_family

    @property
    def families(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_family))

    def _select(
        self, parsed: lowbit_transport.InspectedLowBitPayload
    ) -> SplitFeatureAE:
        if parsed.family_id == ae_contract.AE_FAMILY_NOAE:
            raise guards.HybridQPayloadError("noAE frames do not use an AE decoder")
        autoencoder = self._by_family.get(parsed.family_id)
        if autoencoder is None:
            raise guards.HybridQPayloadError(
                f"no preloaded decoder for {ae_contract.family_name(parsed.family_id)}"
            )
        if parsed.channels != autoencoder.bottleneck:
            raise guards.HybridQPayloadError(
                "low-bit latent width disagrees with the selected decoder"
            )
        if parsed.routing_tag != autoencoder.routing_tag:
            raise guards.HybridQPayloadError(
                "low-bit routing tag disagrees with the selected decoder"
            )
        return autoencoder

    def receive(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        wire_codec: ZstdWireCodec | None = None,
        expected_packet: lowbit_transport.LowBitZstdPacket | None = None,
        diagnostics: bool = False,
    ) -> ReceivedLowBitFrame:
        if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            raise guards.HybridQPayloadError("receive requires compressed bytes")
        compressed = bytes(frame_bytes)
        if not compressed:
            raise guards.HybridQPayloadError("received an empty low-bit frame")

        codec = wire_codec if wire_codec is not None else ZstdWireCodec()
        sparse = codec.decompress_bytes(compressed)
        parsed = lowbit_transport.inspect(sparse)
        decoded, keep_mask, q = lowbit_transport.decode_inspected(parsed)

        if expected_packet is not None:
            if not isinstance(expected_packet, lowbit_transport.LowBitZstdPacket):
                raise guards.HybridQPayloadError(
                    "expected_packet must be a LowBitZstdPacket"
                )
            if expected_packet.uncompressed_bytes != len(sparse):
                raise guards.HybridQPayloadError(
                    "packet metadata disagrees with decompressed length"
                )
            expected = (
                expected_packet.bit_width,
                expected_packet.family_id,
                expected_packet.routing_tag,
                expected_packet.channels,
            )
            observed = (
                parsed.bit_width,
                parsed.family_id,
                parsed.routing_tag,
                parsed.channels,
            )
            if observed != expected:
                raise guards.HybridQPayloadError(
                    "inner low-bit identity disagrees with packet metadata"
                )

        decoder: SplitFeatureAE | None
        if parsed.family_id == ae_contract.AE_FAMILY_NOAE:
            decoder = None
            guards.require_frozen_c2(decoded, what="received noAE low-bit C2")
            reconstructed = decoded
        else:
            decoder = self._select(parsed)
            device = ae_uint8_transport.decoder_device(decoder)
            reconstructed = decoder.decode(decoded.to(device), keep_mask.to(device))
            guards.require_frozen_c2(reconstructed, what="reconstructed low-bit C2")

        family = LowBitWireFamily(
            family_id=parsed.family_id,
            family_name=ae_contract.family_name(parsed.family_id),
            transported_channels=parsed.channels,
            routing_tag=parsed.routing_tag,
            bit_width=parsed.bit_width,
        )
        return ReceivedLowBitFrame(
            c2=reconstructed,
            q=q,
            family=family,
            keep_count=int(parsed.header.keep_count),
            compressed_bytes=len(compressed),
            uncompressed_bytes=len(sparse),
            diagnostics=(
                LowBitReceiveDiagnostics(
                    parsed=parsed,
                    decoded_feature=decoded,
                    keep_mask=keep_mask,
                    decoder=decoder,
                )
                if diagnostics
                else None
            ),
        )


def require_received_boundary(frame: ReceivedLowBitFrame) -> ReceivedLowBitFrame:
    """Small deployment guard shared by later qualification/validation runners."""
    if not isinstance(frame, ReceivedLowBitFrame):
        raise guards.HybridQPayloadError("expected a received low-bit frame")
    guards.require_frozen_c2(frame.c2, what="received low-bit tail input")
    plan = continuous_q.quantize_q(frame.q)
    guards.require_keep_cardinality(frame.keep_count, plan.keep_count)
    if frame.family.transported_channels not in (
        contract.SPLIT_CHANNELS,
        *ae_contract.AE_BOTTLENECKS,
    ):
        raise guards.HybridQPayloadError("received channel width is unregistered")
    return frame

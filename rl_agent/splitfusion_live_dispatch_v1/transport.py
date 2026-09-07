"""Adapters over the existing SplitFusion codecs; no numerical logic is copied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_contract,
    ae_uint8_transport,
    lowbit_transport,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    continuous_q,
    guards,
    uint8_codec,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
    frame_content_size,
)

from .registry import ActionProfile, DispatchContractError
from .timing import StageRecorder


def _magic_ascii(value: bytes) -> str:
    return value.decode("ascii").replace("\x00", "\\0")


@dataclass(frozen=True)
class InnerIdentity:
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    q_e4: int
    keep_count: int
    routing_tag: int
    transported_channels: int
    latent_width: int | None
    wire_magic_ascii: str
    wire_codec_id: int
    wire_version: int


@dataclass(frozen=True)
class InspectedInnerPayload:
    identity: InnerIdentity
    compressed_bytes: int
    uncompressed_bytes: int
    kind: str
    sparse_bytes: bytes
    parsed: Any


@dataclass(frozen=True)
class DecodedC2:
    c2: Any
    finite: bool
    device: torch.device


class UESplitCodec(Protocol):
    def encode(
        self,
        profile: ActionProfile,
        c2: Any,
        *,
        ranker: Any | None,
        ae_encoder: Any | None,
        timing: StageRecorder,
    ) -> bytes: ...


class EdgeSplitCodec(Protocol):
    def inspect(self, payload: bytes, *, timing: StageRecorder) -> InspectedInnerPayload: ...

    def decode(
        self,
        inspected: InspectedInnerPayload,
        *,
        decoder: Any | None,
        tail_device: torch.device,
        timing: StageRecorder,
    ) -> DecodedC2: ...


def expected_inner_identity(profile: ActionProfile) -> InnerIdentity:
    return InnerIdentity(
        family=profile.family,
        family_id=profile.family_id,
        quantizer=profile.quantizer,
        bit_width=profile.bit_width,
        q_e4=profile.q_e4,
        keep_count=profile.keep_count,
        routing_tag=profile.routing_tag,
        transported_channels=profile.transported_channels,
        latent_width=profile.latent_width,
        wire_magic_ascii=profile.wire.magic_ascii,
        wire_codec_id=profile.wire.codec_id,
        wire_version=profile.wire.version,
    )


def require_inner_agreement(profile: ActionProfile, observed: InnerIdentity) -> None:
    expected = expected_inner_identity(profile)
    for field in expected.__dataclass_fields__:
        wanted = getattr(expected, field)
        actual = getattr(observed, field)
        if actual != wanted:
            raise DispatchContractError(
                f"packet/catalog {field} mismatch for action {profile.action_id}: "
                f"packet={actual!r}, catalog={wanted!r}"
            )


class ProductionSplitCodec:
    """One startup-created zstd context over the existing public codec APIs."""

    def __init__(self, wire_codec: ZstdWireCodec | None = None) -> None:
        self._wire = wire_codec if wire_codec is not None else ZstdWireCodec()

    @staticmethod
    def _selection(profile: ActionProfile, c2: torch.Tensor, ranker: Any | None):
        plan = continuous_q.quantize_q(profile.q)
        if plan.q_e4 != profile.q_e4 or plan.keep_count != profile.keep_count:
            raise DispatchContractError("catalog q plan disagrees with frozen q semantics")
        if plan.is_bypass:
            if ranker is not None:
                raise DispatchContractError("q=0 must receive the ranker-bypass route")
            return plan, None
        if ranker is None:
            raise DispatchContractError("q>0 requires the preloaded stable ranker")
        scores = ranker.score_cells(c2)
        return plan, continuous_q.select_cells(scores, plan.wire_q)

    def encode(
        self,
        profile: ActionProfile,
        c2: torch.Tensor,
        *,
        ranker: Any | None,
        ae_encoder: Any | None,
        timing: StageRecorder,
    ) -> bytes:
        guards.require_frozen_c2(c2, what="preloaded UE C2")
        with timing.stage("ranker_selection"):
            plan, selection = self._selection(profile, c2, ranker)
        latent = None
        with timing.stage("ae_encode"):
            if profile.family != "noAE":
                if ae_encoder is None:
                    raise DispatchContractError(f"preloaded {profile.family} encoder unavailable")
                latent = ae_encoder.encode(c2)
                ae_contract.require_latent(
                    latent,
                    profile.transported_channels,
                    what=f"preloaded {profile.family} encoder output",
                )
        with timing.stage("quantize_pack"):
            if profile.quantizer == "UINT8" and profile.family == "noAE":
                sparse = uint8_codec.encode(uint8_codec.prepare(c2), plan.wire_q, selection)
            elif profile.quantizer == "UINT8":
                sparse = ae_uint8_transport.encode_sparse(
                    ae_uint8_transport.prepare(latent),
                    plan.wire_q,
                    selection,
                    routing_tag=profile.routing_tag,
                )
            else:
                source = c2 if profile.family == "noAE" else latent
                sparse = lowbit_transport.encode_sparse(
                    lowbit_transport.prepare_feature(
                        source,
                        family_id=profile.family_id,
                        routing_tag=profile.routing_tag,
                    ),
                    plan.wire_q,
                    profile.bit_width,
                    selection,
                )
        with timing.stage("zstd_compression"):
            compressed = self._wire.compress(sparse.data)
            if frame_content_size(compressed.data) != sparse.total_bytes:
                raise DispatchContractError("zstd content size does not bind inner payload")
        return compressed.data

    def inspect(
        self,
        payload: bytes,
        *,
        timing: StageRecorder,
    ) -> InspectedInnerPayload:
        if not isinstance(payload, bytes) or not payload:
            raise DispatchContractError("inner payload must be non-empty bytes")
        with timing.stage("zstd_decompression"):
            sparse = self._wire.decompress_bytes(payload)
        magic = sparse[:4]
        if magic == uint8_codec.MAGIC:
            parsed = uint8_codec.inspect(sparse)
            identity = InnerIdentity(
                family="noAE",
                family_id=ae_contract.AE_FAMILY_NOAE,
                quantizer="UINT8",
                bit_width=8,
                q_e4=int(parsed.header.q_e4),
                keep_count=int(parsed.header.keep_count),
                routing_tag=ae_contract.AE_UNBOUND_ROUTING_TAG,
                transported_channels=int(parsed.header.channels),
                latent_width=None,
                wire_magic_ascii=_magic_ascii(parsed.header.magic),
                wire_codec_id=int(parsed.header.codec_id),
                wire_version=int(parsed.header.version),
            )
            kind = "noae_uint8"
        elif magic == ae_uint8_transport.MAGIC:
            parsed = ae_uint8_transport.inspect(sparse)
            identity = InnerIdentity(
                family=ae_contract.family_name(parsed.family_id),
                family_id=int(parsed.family_id),
                quantizer="UINT8",
                bit_width=8,
                q_e4=int(parsed.header.q_e4),
                keep_count=int(parsed.header.keep_count),
                routing_tag=int(parsed.routing_tag),
                transported_channels=int(parsed.bottleneck),
                latent_width=int(parsed.bottleneck),
                wire_magic_ascii=_magic_ascii(parsed.header.magic),
                wire_codec_id=int(parsed.header.codec_id),
                wire_version=int(parsed.header.version),
            )
            kind = "ae_uint8"
        elif magic == lowbit_transport.MAGIC:
            parsed = lowbit_transport.inspect(sparse)
            family = ae_contract.family_name(parsed.family_id)
            identity = InnerIdentity(
                family=family,
                family_id=int(parsed.family_id),
                quantizer=f"UINT{int(parsed.bit_width)}",
                bit_width=int(parsed.bit_width),
                q_e4=int(parsed.header.q_e4),
                keep_count=int(parsed.header.keep_count),
                routing_tag=int(parsed.routing_tag),
                transported_channels=int(parsed.channels),
                latent_width=(None if family == "noAE" else int(parsed.channels)),
                wire_magic_ascii=_magic_ascii(parsed.header.magic),
                wire_codec_id=int(parsed.header.codec_id),
                wire_version=int(parsed.header.version),
            )
            kind = "lowbit"
        else:
            raise DispatchContractError("inner payload has no supported SplitFusion magic")
        return InspectedInnerPayload(
            identity=identity,
            compressed_bytes=len(payload),
            uncompressed_bytes=len(sparse),
            kind=kind,
            sparse_bytes=sparse,
            parsed=parsed,
        )

    def decode(
        self,
        inspected: InspectedInnerPayload,
        *,
        decoder: Any | None,
        tail_device: torch.device,
        timing: StageRecorder,
    ) -> DecodedC2:
        identity = inspected.identity
        with timing.stage("unpack_dequantize"):
            if inspected.kind == "noae_uint8":
                decoded, q = uint8_codec.decode(inspected.sparse_bytes)
                keep_mask = None
            elif inspected.kind == "ae_uint8":
                decoded, keep_mask, q, _ = ae_uint8_transport.decode_sparse(
                    inspected.sparse_bytes
                )
            elif inspected.kind == "lowbit":
                decoded, keep_mask, q = lowbit_transport.decode_inspected(
                    inspected.parsed
                )
            else:
                raise DispatchContractError("inspected inner payload kind is unsupported")
            if continuous_q.quantize_q(q).q_e4 != identity.q_e4:
                raise DispatchContractError("decoded q disagrees with inspected inner header")
        with timing.stage("ae_decode"):
            if identity.family == "noAE":
                if decoder is not None:
                    raise DispatchContractError("noAE action must not select an AE decoder")
                reconstructed = decoded.to(tail_device)
            else:
                if decoder is None:
                    raise DispatchContractError(f"preloaded {identity.family} decoder unavailable")
                if (
                    int(getattr(decoder, "family_id", -1)) != identity.family_id
                    or int(getattr(decoder, "bottleneck", -1)) != identity.transported_channels
                    or int(getattr(decoder, "routing_tag", -1)) != identity.routing_tag
                ):
                    raise DispatchContractError("selected decoder identity disagrees with inner header")
                reconstructed = decoder.decode(
                    decoded.to(tail_device), keep_mask.to(tail_device)
                )
        guards.require_frozen_c2(reconstructed, what="preloaded edge reconstructed C2")
        finite = bool(torch.isfinite(reconstructed).all())
        if not finite:
            raise DispatchContractError("reconstructed C2 is non-finite")
        if reconstructed.device != tail_device:
            raise DispatchContractError(
                f"reconstructed C2 device {reconstructed.device} != tail device {tail_device}"
            )
        return DecodedC2(c2=reconstructed, finite=True, device=reconstructed.device)

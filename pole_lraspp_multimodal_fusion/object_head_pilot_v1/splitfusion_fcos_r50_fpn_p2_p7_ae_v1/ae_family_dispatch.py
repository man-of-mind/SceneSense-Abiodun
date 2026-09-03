"""Minimal per-packet decoder-selection adapter for the preloaded profile system.

This is an interface and provenance safeguard, not a model registry. It does not
load checkpoints, construct profiles, switch profiles or copy the frozen
perception tail. It takes AE encoder-decoder pairs the *existing* preloaded
profile mechanism already holds, and answers exactly one question per packet:
**which already-loaded decoder is this frame allowed to use?**

The deployed runtime keeps one shared frozen front/tail, the noAE direct path
and preloaded AE128/AE64/AE32 pairs, and selects q and quantizer settings at
runtime. Because a packet can be delayed or reordered across a profile switch,
selection is driven by the frame's own envelope, never by the currently selected
profile. Any disagreement between the frame's family id, its transported latent
channel count and the selected decoder's routing tag fails closed.

Only the compressed zstd byte string is guaranteed to cross the network, so
`receive` is the deployable entry point: it takes those raw bytes, decompresses
them exactly once, and discovers the decoder from the authoritative inner AE
header. Python dataclass fields are not wire provenance; an `AeZstdPacket` may
be cross-checked when it happens to be available locally, but it is never
required to find the decoder.

Training never uses this: a training run selects one bottleneck family
explicitly and keeps it for the whole run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import guards
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import uint8_codec
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import ZstdWireCodec
from . import ae_contract, ae_uint8_transport
from .ae_model import SplitFeatureAE


@dataclass(frozen=True)
class WireFamily:
    """The model family one framed packet declares, from the packet alone."""

    family_id: int
    family_name: str
    transported_channels: int
    routing_tag: int
    codec: str


@dataclass(frozen=True)
class ReceivedFrame:
    """One received wire frame, reconstructed by the decoder it declared."""

    c2: torch.Tensor  # [256,112,192] FP32
    q: float
    family: WireFamily
    keep_count: int
    compressed_bytes: int
    uncompressed_bytes: int


def identify_sparse_frame(data: bytes) -> WireFamily:
    """Identify the model family of one *uncompressed* framed payload.

    The frozen noAE wire is recognized by its own unchanged envelope and is
    never re-framed here; the AE wire is recognized by its explicit family
    field. Anything else is refused rather than guessed.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise guards.HybridQPayloadError("frame identification requires bytes")
    payload = bytes(data)
    if payload[:4] == ae_uint8_transport.MAGIC:
        parsed = ae_uint8_transport.inspect(payload)
        return WireFamily(
            family_id=parsed.family_id,
            family_name=ae_contract.family_name(parsed.family_id),
            transported_channels=parsed.bottleneck,
            routing_tag=parsed.routing_tag,
            codec="ae_latent_uint8",
        )
    if payload[:4] == uint8_codec.MAGIC:
        # Validated noAE wire, unchanged: 256 C2 channels, no AE decoder.
        parsed_noae = uint8_codec.inspect(payload)
        return WireFamily(
            family_id=ae_contract.AE_FAMILY_NOAE,
            family_name=ae_contract.family_name(ae_contract.AE_FAMILY_NOAE),
            transported_channels=int(parsed_noae.header.channels),
            routing_tag=ae_contract.AE_UNBOUND_ROUTING_TAG,  # noAE has no AE pair
            codec="per_channel_uint8",
        )
    raise guards.HybridQPayloadError("frame carries no recognized wire identity")


class PreloadedAeDecoders:
    """Read-only view over AE decoders the runtime has *already* constructed.

    Construction takes existing `SplitFeatureAE` objects. It builds nothing,
    reads no file and holds no frozen perception state; the shared frozen
    front/tail stays exactly where the existing runtime keeps it.
    """

    def __init__(self, autoencoders: Iterable[SplitFeatureAE]) -> None:
        by_family: dict[int, SplitFeatureAE] = {}
        for autoencoder in autoencoders:
            if not isinstance(autoencoder, SplitFeatureAE):
                raise guards.HybridQConfigError(
                    "preloaded decoders must be SplitFeatureAE instances"
                )
            family = ae_contract.family_for_bottleneck(autoencoder.bottleneck)
            if family != autoencoder.family_id:
                raise guards.HybridQConfigError(
                    "AE family id disagrees with its own latent channel count"
                )
            if family in by_family:
                raise guards.HybridQConfigError(
                    f"family {ae_contract.family_name(family)} was supplied twice"
                )
            # An unbound decoder cannot be routed to, so it is refused here
            # rather than at the first frame that needed it.
            ae_contract.require_bound_routing_tag(
                autoencoder.routing_tag,
                what=f"preloaded {autoencoder.family_name} routing tag",
            )
            by_family[family] = autoencoder
        if not by_family:
            raise guards.HybridQConfigError("no preloaded AE decoder was supplied")
        self._by_family = by_family

    @property
    def families(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_family))

    def select_for_packet(
        self, packet: ae_uint8_transport.AeZstdPacket
    ) -> SplitFeatureAE:
        """Pick the decoder this individual packet declares, or fail closed."""
        if not isinstance(packet, ae_uint8_transport.AeZstdPacket):
            raise guards.HybridQPayloadError("decoder selection requires an AeZstdPacket")
        autoencoder = self._by_family.get(int(packet.family_id))
        if autoencoder is None:
            raise guards.HybridQPayloadError(
                f"no preloaded decoder for packet family "
                f"{ae_contract.family_name(packet.family_id)}"
            )
        return ae_uint8_transport.require_family_agreement(packet, autoencoder)

    def select_for_header(
        self, parsed: ae_uint8_transport.InspectedAePayload
    ) -> SplitFeatureAE:
        """Pick the decoder the *inner AE header* declares, or fail closed."""
        if not isinstance(parsed, ae_uint8_transport.InspectedAePayload):
            raise guards.HybridQPayloadError(
                "decoder selection requires an inspected AE header"
            )
        autoencoder = self._by_family.get(int(parsed.family_id))
        if autoencoder is None:
            raise guards.HybridQPayloadError(
                f"no preloaded decoder for frame family "
                f"{ae_contract.family_name(parsed.family_id)}"
            )
        return ae_uint8_transport.require_family_agreement(parsed, autoencoder)

    def receive(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        wire_codec: ZstdWireCodec | None = None,
        expected_packet: ae_uint8_transport.AeZstdPacket | None = None,
    ) -> ReceivedFrame:
        """Deployable edge path: received zstd bytes in, reconstructed C2 out.

        The bytes are decompressed exactly once. Everything after that -- which
        decoder to use, which q, which keep set -- is read from the inner AE
        header, so a delayed or reordered frame is routed by what it actually
        carries. `expected_packet` is an optional local cross-check and is never
        needed to discover the decoder.
        """
        if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            raise guards.HybridQPayloadError("receive requires the raw wire bytes")
        compressed = bytes(frame_bytes)
        if not compressed:
            raise guards.HybridQPayloadError("received an empty wire frame")

        decompressor = wire_codec if wire_codec is not None else ZstdWireCodec()
        # Exactly one decompression for the whole receive path.
        sparse = decompressor.decompress_bytes(compressed)

        parsed = ae_uint8_transport.inspect(sparse)
        autoencoder = self.select_for_header(parsed)
        if expected_packet is not None:
            if not isinstance(expected_packet, ae_uint8_transport.AeZstdPacket):
                raise guards.HybridQPayloadError(
                    "the optional cross-check must be an AeZstdPacket"
                )
            ae_uint8_transport.require_family_agreement(expected_packet, autoencoder)
            if expected_packet.uncompressed_bytes != len(sparse):
                raise guards.HybridQPayloadError(
                    "local packet metadata disagrees with the received frame length"
                )

        # Reuse the already-decompressed bytes: no second zstd pass.
        latent, keep_mask, q, _ = ae_uint8_transport.decode_sparse(sparse)
        # The wire is host bytes, so the latent is rebuilt on CPU; hand it to
        # the selected decoder on whichever device that decoder actually lives.
        device = ae_uint8_transport.decoder_device(autoencoder)
        reconstructed = autoencoder.decode(latent.to(device), keep_mask.to(device))
        guards.require_frozen_c2(reconstructed, what="reconstructed C2")
        return ReceivedFrame(
            c2=reconstructed,
            q=q,
            family=WireFamily(
                family_id=parsed.family_id,
                family_name=ae_contract.family_name(parsed.family_id),
                transported_channels=parsed.bottleneck,
                routing_tag=parsed.routing_tag,
                codec="ae_latent_uint8",
            ),
            keep_count=int(parsed.header.keep_count),
            compressed_bytes=len(compressed),
            uncompressed_bytes=len(sparse),
        )

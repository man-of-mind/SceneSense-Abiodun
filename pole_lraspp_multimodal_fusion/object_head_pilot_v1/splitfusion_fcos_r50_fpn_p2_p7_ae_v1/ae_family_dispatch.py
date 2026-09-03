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
channel count and the selected decoder's registered checkpoint binding fails
closed.

Training never uses this: a training run selects one bottleneck family
explicitly and keeps it for the whole run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import guards
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import uint8_codec
from . import ae_contract, ae_uint8_transport
from .ae_model import SplitFeatureAE


@dataclass(frozen=True)
class WireFamily:
    """The model family one framed packet declares, from the packet alone."""

    family_id: int
    family_name: str
    transported_channels: int
    checkpoint_binding: int
    codec: str


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
            checkpoint_binding=parsed.checkpoint_binding,
            codec="ae_latent_uint8",
        )
    if payload[:4] == uint8_codec.MAGIC:
        # Validated noAE wire, unchanged: 256 C2 channels, no AE decoder.
        parsed_noae = uint8_codec.inspect(payload)
        return WireFamily(
            family_id=ae_contract.AE_FAMILY_NOAE,
            family_name=ae_contract.family_name(ae_contract.AE_FAMILY_NOAE),
            transported_channels=int(parsed_noae.header.channels),
            checkpoint_binding=ae_contract.AE_UNBOUND_CHECKPOINT_BINDING,
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

    def reconstruct(
        self, packet: ae_uint8_transport.AeZstdPacket, *, wire_codec=None
    ) -> tuple[object, float]:
        """Select this packet's decoder, then reconstruct C2 with exactly it."""
        autoencoder = self.select_for_packet(packet)
        return ae_uint8_transport.reconstruct_c2(
            packet, autoencoder, wire_codec=wire_codec
        )

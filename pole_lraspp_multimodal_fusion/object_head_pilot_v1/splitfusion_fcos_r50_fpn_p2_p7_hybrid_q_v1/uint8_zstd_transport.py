"""Mandatory Phase-7 zstd wrapper for the per-channel UINT8 Hybrid-q codec."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import continuous_q, guards, uint8_codec
from .selection import CellSelection
from .zstd_transport import ZstdWireCodec, frame_content_size


@dataclass(frozen=True)
class Uint8ZstdPacket:
    """Deployable wire object: exactly one zstd frame, never raw sparse bytes."""

    data: bytes
    uncompressed_bytes: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class Uint8ZstdTransport:
    plan: continuous_q.ContinuousQ
    selection: CellSelection | None
    packet: Uint8ZstdPacket


def prepare_frame(c2: torch.Tensor) -> uint8_codec.PreparedUint8Frame:
    """Bind one original FP32 frame to its once-computed full-C2 ranges."""
    return uint8_codec.prepare(c2)


def encode(
    prepared: uint8_codec.PreparedUint8Frame,
    ranker,
    q: float,
    *,
    wire_codec: ZstdWireCodec | None = None,
) -> Uint8ZstdTransport:
    """Rank/select on FP32, UINT8-frame retained values, then always zstd-wrap."""
    plan = continuous_q.quantize_q(q)
    if plan.is_bypass:
        selection = None
    else:
        # The stable ranker sees the original FP32 tensor, before quantization.
        scores = ranker.score_cells(prepared.c2)
        selection = continuous_q.select_cells(scores, plan.wire_q)

    sparse = uint8_codec.encode(prepared, plan.wire_q, selection)
    compressor = wire_codec if wire_codec is not None else ZstdWireCodec()
    compressed = compressor.compress(sparse.data)
    if frame_content_size(compressed.data) != sparse.total_bytes:
        raise guards.HybridQPayloadError("zstd content size does not bind sparse payload")
    return Uint8ZstdTransport(
        plan=plan,
        selection=selection,
        packet=Uint8ZstdPacket(
            data=compressed.data,
            uncompressed_bytes=compressed.uncompressed_bytes,
        ),
    )


def decompress_payload(
    packet: Uint8ZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> bytes:
    """Diagnostic exact sparse bytes after the mandatory zstd decode step."""
    if not isinstance(packet, Uint8ZstdPacket):
        raise guards.HybridQPayloadError("decode requires a Uint8ZstdPacket")
    if not packet.data:
        raise guards.HybridQPayloadError("zstd packet is empty")
    decompressor = wire_codec if wire_codec is not None else ZstdWireCodec()
    return decompressor.decompress(
        packet.data, expected_bytes=packet.uncompressed_bytes
    )


def decode(
    packet: Uint8ZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> tuple[torch.Tensor, float]:
    """zstd decompress -> UINT8 dequantize -> scatter -> dense FP32 C2."""
    sparse_bytes = decompress_payload(packet, wire_codec=wire_codec)
    return uint8_codec.decode(sparse_bytes)

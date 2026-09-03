"""AE-latent per-channel UINT8 wire plus the mandatory zstd wrapper.

This is a **separate** codec from the validated noAE `uint8_codec`: that wire
carries 256 C2 channels and is unchanged here, magic `HQ8\\0`, codec id 1. The
AE wire carries B latent channels, magic `AE8\\0`, codec id 2, and only accepts
B in {128, 64, 32}.

Per frame the ranges are computed once from the **complete** latent, before any
q is applied, and then reused for every q of that frame. So the same cell
quantizes to the same code at every q, and a q sweep over one frame is a pure
subset relation on the value block.

Deterministic value order, cell-major: retained cells in ascending row-major
cell index, and within one cell the B latent channels in ascending channel
index. All B channels of a retained cell therefore stay contiguous on the wire.

Every frame carries its own model-family identity: an explicit family id
(AE128/AE64/AE32) and a 32-bit checkpoint-binding word, alongside the latent
channel count. The edge therefore selects a decoder per packet instead of
trusting whichever profile is currently selected, and a packet that was delayed
across a profile switch is refused rather than decoded by the wrong AE. The
frozen noAE wire is not re-framed: its own envelope (magic HQ8\0, codec id 1,
256 channels) already identifies family 0.

Nothing raw ever leaves this module. `encode` returns a zstd packet, and the
decoder accepts only that packet type, so the mandatory Phase-7 level-1 wrapper
cannot be skipped. The entropy level is unchanged and is not tuned here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    uint8_codec,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.codec import _pack_bitmask, _unpack_bitmask
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.selection import CellSelection
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
    frame_content_size,
)
from . import ae_composition, ae_contract
from .ae_model import SplitFeatureAE


MAGIC = b"AE8\x00"
FORMAT_VERSION = 1
CODEC_ID_AE_LATENT_UINT8 = 2

# magic, version, codec id, family id, checkpoint binding, B, H, W, q_e4,
# keep count, mask bytes, range bytes, value bytes. All little endian, packed.
HEADER_FORMAT = "<4sHHHIIIIIIIIQ"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)  # 50

# Locked to the validated noAE value so the two wires cannot drift apart.
CONSTANT_SPAN_EPSILON = uint8_codec.CONSTANT_SPAN_EPSILON


def range_byte_count(bottleneck: int) -> int:
    """One little-endian FP32 min/max pair per latent channel."""
    return ae_contract.require_bottleneck(bottleneck) * 2 * 4


@dataclass(frozen=True)
class PreparedAeLatent:
    """One complete dense latent bound to its once-computed channel ranges."""

    latent: torch.Tensor
    bottleneck: int
    family_id: int
    channel_ranges: torch.Tensor  # [B, 2], min then max, FP32
    latent_version: int | None
    ranges_version: int | None


@dataclass(frozen=True)
class AeUint8Header:
    magic: bytes
    version: int
    codec_id: int
    family_id: int
    checkpoint_binding: int
    bottleneck: int
    height: int
    width: int
    q_e4: int
    keep_count: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int


@dataclass(frozen=True)
class AeSparsePayload:
    """Uncompressed diagnostic payload; deployment always wraps this in zstd."""

    data: bytes
    q: float
    family_id: int
    checkpoint_binding: int
    bottleneck: int
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class InspectedAePayload:
    """Strictly validated sparse contents, before dequantization and scatter."""

    header: AeUint8Header
    q: float
    family_id: int
    checkpoint_binding: int
    bottleneck: int
    keep_indices: torch.Tensor  # ascending row-major int64
    keep_mask: torch.Tensor  # bool [112,192], reconstructed for the decoder
    channel_ranges: torch.Tensor  # [B, 2] FP32
    values: torch.Tensor  # [keep, B] UINT8, cell-major


@dataclass(frozen=True)
class AeAnalyticalSize:
    """Exact pre-zstd byte accounting. No measured payload is claimed."""

    q: float
    q_e4: int
    family_id: int
    bottleneck: int
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class AeZstdPacket:
    """Deployable wire object: exactly one zstd frame, never raw sparse bytes."""

    data: bytes
    uncompressed_bytes: int
    family_id: int
    checkpoint_binding: int
    bottleneck: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)

    @property
    def family_name(self) -> str:
        return ae_contract.family_name(self.family_id)


@dataclass(frozen=True)
class AeZstdTransport:
    plan: continuous_q.ContinuousQ
    family_id: int
    bottleneck: int
    selection: CellSelection | None
    keep_mask: torch.Tensor
    packet: AeZstdPacket


# ---------------------------------------------------------------------------
# Range preparation
# ---------------------------------------------------------------------------


def _require_valid_ranges(
    ranges: torch.Tensor, bottleneck: int, *, what: str
) -> torch.Tensor:
    size = ae_contract.require_bottleneck(bottleneck)
    if not isinstance(ranges, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(ranges.shape) != (size, 2):
        raise guards.HybridQPayloadError(
            f"{what} must be [{size}, 2], got {list(ranges.shape)}"
        )
    if ranges.dtype is not torch.float32:
        raise guards.HybridQPayloadError(f"{what} must be float32, got {ranges.dtype}")
    guards.require_finite(ranges, what)
    if bool((ranges[:, 0] > ranges[:, 1]).any()):
        raise guards.HybridQPayloadError(
            "channel range ordering must be [min, max] with min <= max"
        )
    if not bool(torch.isfinite(ranges[:, 1] - ranges[:, 0]).all()):
        raise guards.HybridQPayloadError("channel range span is non-finite")
    return ranges


def prepare(latent: torch.Tensor) -> PreparedAeLatent:
    """Compute all per-channel ranges once from one complete dense latent."""
    if not isinstance(latent, torch.Tensor) or latent.dim() != 3:
        raise guards.HybridQPayloadError("prepare expects one dense [B,112,192] latent")
    bottleneck = ae_contract.require_bottleneck(int(latent.shape[0]))
    ae_contract.require_latent(latent, bottleneck, what="latent to prepare")
    with torch.no_grad():
        flat = latent.detach().reshape(bottleneck, ae_contract.AE_LATENT_CELLS)
        ranges = torch.stack((flat.amin(dim=1), flat.amax(dim=1)), dim=1).contiguous()
        _require_valid_ranges(ranges, bottleneck, what="computed channel ranges")
    return PreparedAeLatent(
        latent=latent,
        bottleneck=bottleneck,
        family_id=ae_contract.family_for_bottleneck(bottleneck),
        channel_ranges=ranges,
        latent_version=uint8_codec._tensor_version(latent),
        ranges_version=uint8_codec._tensor_version(ranges),
    )


def _require_prepared(prepared: PreparedAeLatent) -> PreparedAeLatent:
    if not isinstance(prepared, PreparedAeLatent):
        raise guards.HybridQPayloadError("encode requires a PreparedAeLatent")
    ae_contract.require_latent(
        prepared.latent, prepared.bottleneck, what="prepared latent"
    )
    if uint8_codec._tensor_version(prepared.latent) != prepared.latent_version:
        raise guards.HybridQPayloadError("prepared latent changed after range analysis")
    if uint8_codec._tensor_version(prepared.channel_ranges) != prepared.ranges_version:
        raise guards.HybridQPayloadError("prepared channel ranges changed after analysis")
    if prepared.family_id != ae_contract.family_for_bottleneck(prepared.bottleneck):
        raise guards.HybridQPayloadError(
            "prepared latent family id disagrees with its latent channel count"
        )
    _require_valid_ranges(
        prepared.channel_ranges, prepared.bottleneck, what="prepared channel ranges"
    )
    return prepared


def _quantize_retained(
    retained: torch.Tensor, channel_ranges: torch.Tensor
) -> torch.Tensor:
    """Affine per-channel quantization of retained cells only, on CPU."""
    values = retained.detach().to(device="cpu", dtype=torch.float32).contiguous()
    ranges = channel_ranges.detach().to(device="cpu", dtype=torch.float32)
    minima = ranges[:, 0]
    spans = ranges[:, 1] - minima
    constant = spans <= CONSTANT_SPAN_EPSILON
    safe_spans = torch.where(constant, torch.ones_like(spans), spans)
    normalized = torch.clamp((values - minima) / safe_spans, 0.0, 1.0)
    codes = torch.round(normalized * 255.0).to(torch.uint8)
    if bool(constant.any()):
        codes[:, constant] = 0
    return codes.contiguous()


# ---------------------------------------------------------------------------
# Sparse framing
# ---------------------------------------------------------------------------


def analytical_size(q: float, bottleneck: int) -> AeAnalyticalSize:
    """Exact pre-zstd bytes for one AE-latent payload at continuous q."""
    size = ae_contract.require_bottleneck(bottleneck)
    plan = continuous_q.quantize_q(q, cells=ae_contract.AE_LATENT_CELLS)
    mask_bytes = 0 if plan.is_bypass else contract.mask_byte_count(
        ae_contract.AE_LATENT_CELLS
    )
    range_bytes = range_byte_count(size)
    value_bytes = plan.keep_count * size
    return AeAnalyticalSize(
        q=plan.wire_q,
        q_e4=plan.q_e4,
        family_id=ae_contract.family_for_bottleneck(size),
        bottleneck=size,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=mask_bytes,
        range_bytes=range_bytes,
        value_bytes=value_bytes,
        total_bytes=HEADER_BYTES + mask_bytes + range_bytes + value_bytes,
    )


def encode_sparse(
    prepared: PreparedAeLatent,
    q: float,
    selection: CellSelection | None = None,
    *,
    checkpoint_binding: int = ae_contract.AE_UNBOUND_CHECKPOINT_BINDING,
) -> AeSparsePayload:
    """Frame retained latent cells as UINT8 using the once-computed ranges.

    q=0 carries no bitmask and no selection, but its complete value block is
    still quantized. q>0 requires the frozen continuous-q selection.
    """
    prepared = _require_prepared(prepared)
    size = prepared.bottleneck
    binding = ae_contract.require_checkpoint_binding(checkpoint_binding)
    plan = continuous_q.quantize_q(q, cells=ae_contract.AE_LATENT_CELLS)
    cell_major = (
        prepared.latent.detach()
        .reshape(size, ae_contract.AE_LATENT_CELLS)
        .transpose(0, 1)
    )

    if plan.is_bypass:
        if selection is not None:
            raise guards.HybridQConfigError("q=0 must not carry a sparse selection")
        mask = b""
        retained = cell_major
    else:
        if selection is None:
            raise guards.HybridQConfigError("q>0 requires a cell selection")
        guards.require_selection_integrity(
            selection,
            plan.wire_q,
            cells=ae_contract.AE_LATENT_CELLS,
            spatial_shape=ae_contract.AE_LATENT_SPATIAL_SHAPE,
        )
        indices = selection.keep_indices.to(torch.int64).cpu()
        mask = _pack_bitmask(indices, ae_contract.AE_LATENT_CELLS)
        retained = cell_major.index_select(0, indices.to(cell_major.device))

    # Selection precedes this call: dropped latent cells are never quantized.
    codes = _quantize_retained(retained, prepared.channel_ranges)
    values = codes.numpy().tobytes(order="C")
    ranges = (
        prepared.channel_ranges.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    if len(ranges) != range_byte_count(size):
        raise guards.HybridQPayloadError("computed channel range block has wrong length")

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        CODEC_ID_AE_LATENT_UINT8,
        prepared.family_id,
        binding,
        size,
        ae_contract.AE_LATENT_HEIGHT,
        ae_contract.AE_LATENT_WIDTH,
        plan.q_e4,
        plan.keep_count,
        len(mask),
        len(ranges),
        len(values),
    )
    payload = AeSparsePayload(
        data=header + mask + ranges + values,
        q=plan.wire_q,
        family_id=prepared.family_id,
        checkpoint_binding=binding,
        bottleneck=size,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=len(mask),
        range_bytes=len(ranges),
        value_bytes=len(values),
    )
    expected = analytical_size(plan.wire_q, size)
    if payload.total_bytes != expected.total_bytes:
        raise guards.HybridQPayloadError("encoded payload length violates analytical size")
    return payload


def _payload_bytes(payload: bytes | bytearray | memoryview | AeSparsePayload) -> bytes:
    if isinstance(payload, AeSparsePayload):
        return payload.data
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    raise guards.HybridQPayloadError("AE payload must be a bytes-like object")


def inspect(
    payload: bytes | bytearray | memoryview | AeSparsePayload,
) -> InspectedAePayload:
    """Parse and fully validate one uncompressed AE-latent payload."""
    data = _payload_bytes(payload)
    if len(data) < HEADER_BYTES:
        raise guards.HybridQPayloadError("payload shorter than the AE header")

    header = AeUint8Header(*struct.unpack(HEADER_FORMAT, data[:HEADER_BYTES]))
    if header.magic != MAGIC:
        raise guards.HybridQPayloadError("AE payload magic mismatch")
    if header.version != FORMAT_VERSION:
        raise guards.HybridQPayloadError(
            f"unsupported AE format version {header.version}"
        )
    if header.codec_id != CODEC_ID_AE_LATENT_UINT8:
        raise guards.HybridQPayloadError("AE codec identity mismatch")
    if header.bottleneck not in ae_contract.AE_BOTTLENECKS:
        raise guards.HybridQPayloadError(
            f"AE header bottleneck {header.bottleneck} is not a registered family"
        )
    size = int(header.bottleneck)
    # Family id and transported latent channel count must agree in the frame
    # itself, so a truncated or spoofed identity cannot pick a decoder.
    if ae_contract.bottleneck_for_family(header.family_id) != size:
        raise guards.HybridQPayloadError(
            f"AE header family {ae_contract.family_name(header.family_id)} does not "
            f"transport {size} latent channels"
        )
    ae_contract.require_checkpoint_binding(header.checkpoint_binding)
    if (header.height, header.width) != ae_contract.AE_LATENT_SPATIAL_SHAPE:
        raise guards.HybridQPayloadError(
            "AE header spatial dimensions do not match the frozen 112x192 split"
        )

    q = header.q_e4 / 10000.0
    try:
        plan = continuous_q.quantize_q(q, cells=ae_contract.AE_LATENT_CELLS)
    except guards.HybridQConfigError as exc:
        raise guards.HybridQPayloadError("AE header q is invalid") from exc
    if plan.q_e4 != header.q_e4:
        raise guards.HybridQPayloadError("AE header q is off the 1e-4 wire grid")
    guards.require_keep_cardinality(header.keep_count, plan.keep_count)

    expected_mask = 0 if plan.is_bypass else contract.mask_byte_count(
        ae_contract.AE_LATENT_CELLS
    )
    if header.mask_bytes != expected_mask:
        raise guards.HybridQPayloadError(
            f"mask length {header.mask_bytes} != expected {expected_mask}"
        )
    if header.range_bytes != range_byte_count(size):
        raise guards.HybridQPayloadError(
            f"range length {header.range_bytes} != expected {range_byte_count(size)}"
        )
    expected_values = plan.keep_count * size
    if header.value_bytes != expected_values:
        raise guards.HybridQPayloadError(
            f"value length {header.value_bytes} != expected {expected_values}"
        )
    expected_total = (
        HEADER_BYTES + header.mask_bytes + header.range_bytes + header.value_bytes
    )
    if len(data) != expected_total:
        raise guards.HybridQPayloadError("payload length disagrees with AE header")

    mask_end = HEADER_BYTES + header.mask_bytes
    range_end = mask_end + header.range_bytes
    if plan.is_bypass:
        keep_indices = torch.arange(ae_contract.AE_LATENT_CELLS, dtype=torch.int64)
        keep_mask = ae_composition.all_keep_mask()
    else:
        unpacked = _unpack_bitmask(data[HEADER_BYTES:mask_end], ae_contract.AE_LATENT_CELLS)
        if int(unpacked.size) != plan.keep_count:
            raise guards.HybridQPayloadError(
                f"bitmask retains {unpacked.size} cells, header declares {plan.keep_count}"
            )
        keep_indices = guards.require_sorted_unique_indices(
            torch.from_numpy(unpacked), ae_contract.AE_LATENT_CELLS
        )
        flat_mask = torch.zeros(ae_contract.AE_LATENT_CELLS, dtype=torch.bool)
        flat_mask[keep_indices] = True
        keep_mask = flat_mask.reshape(ae_contract.AE_LATENT_SPATIAL_SHAPE)
    ae_contract.require_keep_mask(
        keep_mask, what="reconstructed keep mask", expect_keep=plan.keep_count
    )

    range_array = np.frombuffer(data[mask_end:range_end], dtype="<f4")
    if int(range_array.size) != size * 2:
        raise guards.HybridQPayloadError(
            f"range block does not contain {size} min/max pairs"
        )
    channel_ranges = torch.from_numpy(range_array.reshape(size, 2).copy()).to(
        torch.float32
    )
    _require_valid_ranges(channel_ranges, size, what="decoded channel ranges")

    value_array = np.frombuffer(data[range_end:], dtype=np.uint8)
    if int(value_array.size) != expected_values:
        raise guards.HybridQPayloadError("AE value block element count mismatch")
    values = torch.from_numpy(
        value_array.reshape(plan.keep_count, size).copy()
    ).to(torch.uint8)
    return InspectedAePayload(
        header=header,
        q=plan.wire_q,
        family_id=int(header.family_id),
        checkpoint_binding=int(header.checkpoint_binding),
        bottleneck=size,
        keep_indices=keep_indices,
        keep_mask=keep_mask,
        channel_ranges=channel_ranges,
        values=values,
    )


def decode_sparse(
    payload: bytes | bytearray | memoryview | AeSparsePayload,
) -> tuple[torch.Tensor, torch.Tensor, float, InspectedAePayload]:
    """Dequantize and zero-scatter: dense FP32 latent, keep mask, wire q.

    Dropped cells are exact zeros. The returned mask is what the AE decoder
    consumes as its extra input channel; it is read back off the wire, not
    supplied out of band.
    """
    parsed = inspect(payload)
    size = parsed.bottleneck
    minima = parsed.channel_ranges[:, 0]
    spans = parsed.channel_ranges[:, 1] - minima
    constant = spans <= CONSTANT_SPAN_EPSILON

    retained = parsed.values.to(torch.float32) / 255.0
    retained = retained * spans + minima
    if bool(constant.any()):
        retained[:, constant] = minima[constant]
    guards.require_finite(retained, "dequantized retained latent values")

    cell_major = torch.zeros(
        ae_contract.AE_LATENT_CELLS, size, dtype=torch.float32
    )
    cell_major.index_copy_(0, parsed.keep_indices, retained)
    dense = (
        cell_major.transpose(0, 1)
        .reshape(size, ae_contract.AE_LATENT_HEIGHT, ae_contract.AE_LATENT_WIDTH)
        .contiguous()
    )
    ae_contract.require_latent(dense, size, what="decoded AE latent")
    return dense, parsed.keep_mask, parsed.q, parsed


# ---------------------------------------------------------------------------
# Per-frame family agreement
# ---------------------------------------------------------------------------


def require_family_agreement(
    parsed: InspectedAePayload | AeZstdPacket, autoencoder: SplitFeatureAE
) -> SplitFeatureAE:
    """Refuse to decode a frame with an AE that did not produce it.

    Three independent facts must agree: the family id, the transported latent
    channel count and the registered checkpoint binding. A packet that was
    delayed or reordered across a runtime profile switch therefore fails closed
    instead of being reconstructed by whichever AE is currently selected.
    """
    if not isinstance(autoencoder, SplitFeatureAE):
        raise guards.HybridQConfigError("family agreement requires a SplitFeatureAE")
    if int(parsed.family_id) != int(autoencoder.family_id):
        raise guards.HybridQPayloadError(
            f"packet family {ae_contract.family_name(parsed.family_id)} does not match "
            f"the selected decoder {autoencoder.family_name}"
        )
    if int(parsed.bottleneck) != int(autoencoder.bottleneck):
        raise guards.HybridQPayloadError(
            f"packet transports {int(parsed.bottleneck)} latent channels, selected "
            f"decoder has {autoencoder.bottleneck}"
        )
    if int(parsed.checkpoint_binding) != int(autoencoder.checkpoint_binding):
        raise guards.HybridQPayloadError(
            f"packet checkpoint binding {int(parsed.checkpoint_binding)} does not "
            f"match the selected decoder binding {autoencoder.checkpoint_binding}"
        )
    return autoencoder


# ---------------------------------------------------------------------------
# Mandatory zstd wire
# ---------------------------------------------------------------------------


def _compress(payload: AeSparsePayload, wire_codec: ZstdWireCodec | None) -> AeZstdPacket:
    compressor = wire_codec if wire_codec is not None else ZstdWireCodec()
    compressed = compressor.compress(payload.data)
    if frame_content_size(compressed.data) != payload.total_bytes:
        raise guards.HybridQPayloadError("zstd content size does not bind AE payload")
    return AeZstdPacket(
        data=compressed.data,
        uncompressed_bytes=compressed.uncompressed_bytes,
        family_id=payload.family_id,
        checkpoint_binding=payload.checkpoint_binding,
        bottleneck=payload.bottleneck,
    )


def encode(
    prepared: PreparedAeLatent,
    q: float,
    selection: CellSelection | None = None,
    *,
    checkpoint_binding: int = ae_contract.AE_UNBOUND_CHECKPOINT_BINDING,
    wire_codec: ZstdWireCodec | None = None,
) -> AeZstdPacket:
    """Frame one prepared latent and always wrap it in the existing zstd codec."""
    payload = encode_sparse(
        prepared, q, selection, checkpoint_binding=checkpoint_binding
    )
    return _compress(payload, wire_codec)


def encode_frame(
    c2: torch.Tensor,
    autoencoder: SplitFeatureAE,
    ranker,
    q: float,
    *,
    wire_codec: ZstdWireCodec | None = None,
) -> AeZstdTransport:
    """Full transmit path: rank original C2, encode, drop, quantize, zstd.

    The ranker reads the original FP32 C2 and the AE encoder runs on the
    complete frame; only then are cells dropped. Ranges are prepared from that
    complete latent, so every q of this frame shares one set of ranges and a
    retained cell quantizes to the same code at every q.
    """
    composition = ae_composition.compose(c2, autoencoder, ranker, q)
    prepared = prepare(composition.latent.detach())
    payload = encode_sparse(
        prepared,
        composition.plan.wire_q,
        composition.selection,
        checkpoint_binding=autoencoder.checkpoint_binding,
    )
    if payload.family_id != autoencoder.family_id:
        raise guards.HybridQPayloadError(
            "framed family id disagrees with the encoding AE family"
        )
    return AeZstdTransport(
        plan=composition.plan,
        family_id=autoencoder.family_id,
        bottleneck=composition.bottleneck,
        selection=composition.selection,
        keep_mask=composition.keep_mask,
        packet=_compress(payload, wire_codec),
    )


def decompress_payload(
    packet: AeZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> bytes:
    """Diagnostic exact sparse bytes after the mandatory zstd decode step."""
    if not isinstance(packet, AeZstdPacket):
        raise guards.HybridQPayloadError("decode requires an AeZstdPacket")
    if not packet.data:
        raise guards.HybridQPayloadError("zstd packet is empty")
    decompressor = wire_codec if wire_codec is not None else ZstdWireCodec()
    return decompressor.decompress(
        packet.data, expected_bytes=packet.uncompressed_bytes
    )


def decode(
    packet: AeZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> tuple[torch.Tensor, torch.Tensor, float, InspectedAePayload]:
    """zstd decompress -> UINT8 dequantize -> zero-scatter latent + keep mask.

    The returned inspection carries the per-frame family identity the caller
    must use to select a decoder.
    """
    sparse_bytes = decompress_payload(packet, wire_codec=wire_codec)
    latent, keep_mask, q, parsed = decode_sparse(sparse_bytes)
    if latent.shape[0] != packet.bottleneck:
        raise guards.HybridQPayloadError(
            "decoded latent bottleneck disagrees with the packet family"
        )
    if parsed.family_id != packet.family_id:
        raise guards.HybridQPayloadError(
            "decoded family id disagrees with the packet envelope"
        )
    if parsed.checkpoint_binding != packet.checkpoint_binding:
        raise guards.HybridQPayloadError(
            "decoded checkpoint binding disagrees with the packet envelope"
        )
    return latent, keep_mask, q, parsed


def reconstruct_c2(
    packet: AeZstdPacket,
    autoencoder: SplitFeatureAE,
    *,
    wire_codec: ZstdWireCodec | None = None,
) -> tuple[torch.Tensor, float]:
    """Full receive path up to the unchanged frozen perception tail input.

    Returns reconstructed FP32 C2 of exactly [256,112,192]. This is a
    reconstruction, never an identity: at q=0 the channel compression alone is
    lossy.
    """
    if not isinstance(autoencoder, SplitFeatureAE):
        raise guards.HybridQConfigError("reconstruct_c2 requires a SplitFeatureAE")
    latent, keep_mask, q, parsed = decode(packet, wire_codec=wire_codec)
    require_family_agreement(parsed, autoencoder)
    reconstructed = autoencoder.decode(latent, keep_mask)
    guards.require_frozen_c2(reconstructed, what="reconstructed C2")
    return reconstructed, q

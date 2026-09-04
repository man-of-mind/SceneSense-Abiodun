"""Shared per-channel UINT6/UINT4 transport for noAE and AE feature families.

This module adds the two missing quantizers for the 72-profile experiment.  It
does not modify the validated UINT8 wires.  One low-bit envelope covers all
four representation families:

* noAE transports the frozen 256-channel C2 tensor and uses family id 0;
* AE128/AE64/AE32 transport 128/64/32 latent channels and carry the selected
  checkpoint's non-zero routing tag.

The quantizer is the same affine per-channel min/max rule used by the existing
UINT8 codecs, with ``2**bit_width - 1`` levels.  Ranges are computed from the
complete feature map before q drops any spatial cells.  Retained values are
ordered cell-major and packed MSB-first.  UINT4 stores the first code in the
high nibble; UINT6 stores four codes in three bytes.  Any unused low bits in
the last byte must be zero.

Deployment packets are always one independent zstd level-1 frame per camera
frame through the existing frozen :class:`ZstdWireCodec`.  Accuracy, payload
and latency are deliberately unmeasured in this implementation phase.
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
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.codec import (
    _pack_bitmask,
    _unpack_bitmask,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.selection import CellSelection
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
    frame_content_size,
)
from . import ae_composition, ae_contract
from .ae_model import SplitFeatureAE


MAGIC = b"HQLB"
FORMAT_VERSION = 1
CODEC_ID_PER_CHANNEL_LOWBIT = 3
SUPPORTED_BIT_WIDTHS = (6, 4)

# magic, version, codec id, bit width, family id, routing tag, channels,
# H, W, q_e4, keep count, mask bytes, range bytes, value bytes.
HEADER_FORMAT = "<4sHHHHIIIIIIIIQ"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)  # 52

CONSTANT_SPAN_EPSILON = uint8_codec.CONSTANT_SPAN_EPSILON


def require_bit_width(bit_width: int) -> int:
    if isinstance(bit_width, bool) or not isinstance(bit_width, int):
        raise guards.HybridQConfigError("bit width must be an integer")
    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise guards.HybridQConfigError(
            f"bit width must be one of {SUPPORTED_BIT_WIDTHS}, got {bit_width}"
        )
    return int(bit_width)


def _channels_for_family(family_id: int) -> int:
    if family_id == ae_contract.AE_FAMILY_NOAE:
        return contract.SPLIT_CHANNELS
    return ae_contract.bottleneck_for_family(family_id)


def _require_family_wire_identity(
    family_id: int, channels: int, routing_tag: int
) -> tuple[int, int, int]:
    if isinstance(family_id, bool) or not isinstance(family_id, int):
        raise guards.HybridQConfigError("family id must be an integer")
    family = int(family_id)
    expected_channels = _channels_for_family(family)
    if int(channels) != expected_channels:
        raise guards.HybridQPayloadError(
            f"family {ae_contract.family_name(family)} requires "
            f"{expected_channels} channels, got {channels}"
        )
    if family == ae_contract.AE_FAMILY_NOAE:
        tag = ae_contract.require_routing_tag(routing_tag)
        if tag != ae_contract.AE_UNBOUND_ROUTING_TAG:
            raise guards.HybridQPayloadError("noAE packets must carry routing tag 0")
    else:
        tag = ae_contract.require_bound_routing_tag(
            routing_tag, what=f"{ae_contract.family_name(family)} routing tag"
        )
    return family, expected_channels, tag


def range_byte_count(channels: int) -> int:
    count = int(channels)
    if count <= 0:
        raise guards.HybridQConfigError("channel count must be positive")
    return count * 2 * 4


def packed_value_byte_count(code_count: int, bit_width: int) -> int:
    count = int(code_count)
    if count < 0:
        raise guards.HybridQConfigError("code count must be non-negative")
    bits = require_bit_width(bit_width)
    return (count * bits + 7) // 8


def _pack_codes(codes: torch.Tensor, bit_width: int) -> bytes:
    """Pack a flat UINT code stream MSB-first with zero low-bit padding."""
    bits = require_bit_width(bit_width)
    if not isinstance(codes, torch.Tensor) or codes.dtype is not torch.uint8:
        raise guards.HybridQPayloadError("codes must be a UINT8 tensor")
    flat = codes.detach().to(device="cpu").contiguous().reshape(-1).numpy()
    levels = (1 << bits) - 1
    if flat.size and int(flat.max()) > levels:
        raise guards.HybridQPayloadError(
            f"a code exceeds the UINT{bits} maximum {levels}"
        )

    if bits == 4:
        padded = np.zeros((flat.size + 1) // 2 * 2, dtype=np.uint8)
        padded[: flat.size] = flat
        out = ((padded[0::2] << 4) | padded[1::2]).astype(np.uint8, copy=False)
    else:
        padded = np.zeros((flat.size + 3) // 4 * 4, dtype=np.uint8)
        padded[: flat.size] = flat
        groups = padded.astype(np.uint16, copy=False).reshape(-1, 4)
        out = np.empty(groups.shape[0] * 3, dtype=np.uint8)
        out[0::3] = ((groups[:, 0] << 2) | (groups[:, 1] >> 4)).astype(np.uint8)
        out[1::3] = (
            ((groups[:, 1] & 0x0F) << 4) | (groups[:, 2] >> 2)
        ).astype(np.uint8)
        out[2::3] = (
            ((groups[:, 2] & 0x03) << 6) | groups[:, 3]
        ).astype(np.uint8)
        out = out[: packed_value_byte_count(int(flat.size), bits)]

    packed = out.tobytes(order="C")
    if len(packed) != packed_value_byte_count(int(flat.size), bits):
        raise guards.HybridQPayloadError("packed value length is inconsistent")
    return packed


def _unpack_codes(data: bytes, code_count: int, bit_width: int) -> torch.Tensor:
    """Inverse of :func:`_pack_codes`, including strict padding validation."""
    bits = require_bit_width(bit_width)
    count = int(code_count)
    expected = packed_value_byte_count(count, bits)
    if len(data) != expected:
        raise guards.HybridQPayloadError(
            f"packed value length {len(data)} != expected {expected}"
        )
    if count == 0:
        return torch.empty(0, dtype=torch.uint8)
    padding_bits = expected * 8 - count * bits
    if padding_bits and (data[-1] & ((1 << padding_bits) - 1)):
        raise guards.HybridQPayloadError("non-zero padding bits in low-bit value block")

    raw = np.frombuffer(data, dtype=np.uint8)
    if bits == 4:
        unpacked = np.empty(raw.size * 2, dtype=np.uint8)
        unpacked[0::2] = raw >> 4
        unpacked[1::2] = raw & 0x0F
    else:
        padded = np.zeros((raw.size + 2) // 3 * 3, dtype=np.uint8)
        padded[: raw.size] = raw
        groups = padded.astype(np.uint16, copy=False).reshape(-1, 3)
        unpacked = np.empty(groups.shape[0] * 4, dtype=np.uint8)
        unpacked[0::4] = (groups[:, 0] >> 2).astype(np.uint8)
        unpacked[1::4] = (
            ((groups[:, 0] & 0x03) << 4) | (groups[:, 1] >> 4)
        ).astype(np.uint8)
        unpacked[2::4] = (
            ((groups[:, 1] & 0x0F) << 2) | (groups[:, 2] >> 6)
        ).astype(np.uint8)
        unpacked[3::4] = (groups[:, 2] & 0x3F).astype(np.uint8)
    return torch.from_numpy(unpacked[:count].copy()).to(torch.uint8)


@dataclass(frozen=True)
class PreparedLowBitFeature:
    feature: torch.Tensor
    family_id: int
    routing_tag: int
    channels: int
    channel_ranges: torch.Tensor
    feature_version: int | None
    ranges_version: int | None


@dataclass(frozen=True)
class LowBitHeader:
    magic: bytes
    version: int
    codec_id: int
    bit_width: int
    family_id: int
    routing_tag: int
    channels: int
    height: int
    width: int
    q_e4: int
    keep_count: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int


@dataclass(frozen=True)
class LowBitSparsePayload:
    data: bytes
    q: float
    bit_width: int
    family_id: int
    routing_tag: int
    channels: int
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class InspectedLowBitPayload:
    header: LowBitHeader
    q: float
    bit_width: int
    family_id: int
    routing_tag: int
    channels: int
    keep_indices: torch.Tensor
    keep_mask: torch.Tensor
    channel_ranges: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class LowBitAnalyticalSize:
    q: float
    q_e4: int
    bit_width: int
    family_id: int
    channels: int
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class LowBitZstdPacket:
    data: bytes
    uncompressed_bytes: int
    bit_width: int
    family_id: int
    routing_tag: int
    channels: int

    @property
    def compressed_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class LowBitTransport:
    plan: continuous_q.ContinuousQ
    bit_width: int
    family_id: int
    selection: CellSelection | None
    keep_mask: torch.Tensor
    packet: LowBitZstdPacket


def _require_feature(feature: torch.Tensor, channels: int, *, what: str) -> torch.Tensor:
    if not isinstance(feature, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    expected = (int(channels), contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH)
    if tuple(feature.shape) != expected:
        raise guards.HybridQPayloadError(
            f"{what} must be {list(expected)}, got {list(feature.shape)}"
        )
    if feature.dtype is not torch.float32:
        raise guards.HybridQPayloadError(f"{what} must be float32")
    guards.require_finite(feature, what)
    return feature


def _require_ranges(ranges: torch.Tensor, channels: int, *, what: str) -> torch.Tensor:
    if not isinstance(ranges, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(ranges.shape) != (int(channels), 2) or ranges.dtype is not torch.float32:
        raise guards.HybridQPayloadError(
            f"{what} must be float32 [{int(channels)},2]"
        )
    guards.require_finite(ranges, what)
    if bool((ranges[:, 0] > ranges[:, 1]).any()):
        raise guards.HybridQPayloadError("channel ranges require min <= max")
    if not bool(torch.isfinite(ranges[:, 1] - ranges[:, 0]).all()):
        raise guards.HybridQPayloadError("channel range span is non-finite")
    return ranges


def prepare_feature(
    feature: torch.Tensor, *, family_id: int, routing_tag: int
) -> PreparedLowBitFeature:
    channels = int(feature.shape[0]) if isinstance(feature, torch.Tensor) and feature.dim() else -1
    family, channels, tag = _require_family_wire_identity(
        family_id, channels, routing_tag
    )
    _require_feature(feature, channels, what="feature to prepare")
    with torch.no_grad():
        flat = feature.detach().reshape(channels, contract.SPLIT_CELLS)
        ranges = torch.stack((flat.amin(dim=1), flat.amax(dim=1)), dim=1).contiguous()
        _require_ranges(ranges, channels, what="computed channel ranges")
    return PreparedLowBitFeature(
        feature=feature,
        family_id=family,
        routing_tag=tag,
        channels=channels,
        channel_ranges=ranges,
        feature_version=uint8_codec._tensor_version(feature),
        ranges_version=uint8_codec._tensor_version(ranges),
    )


def _require_prepared(prepared: PreparedLowBitFeature) -> PreparedLowBitFeature:
    if not isinstance(prepared, PreparedLowBitFeature):
        raise guards.HybridQPayloadError("encode requires PreparedLowBitFeature")
    _require_family_wire_identity(
        prepared.family_id, prepared.channels, prepared.routing_tag
    )
    _require_feature(prepared.feature, prepared.channels, what="prepared feature")
    if uint8_codec._tensor_version(prepared.feature) != prepared.feature_version:
        raise guards.HybridQPayloadError("prepared feature changed after range analysis")
    if uint8_codec._tensor_version(prepared.channel_ranges) != prepared.ranges_version:
        raise guards.HybridQPayloadError("prepared ranges changed after range analysis")
    _require_ranges(
        prepared.channel_ranges, prepared.channels, what="prepared channel ranges"
    )
    return prepared


def _quantize_retained(
    retained: torch.Tensor, channel_ranges: torch.Tensor, bit_width: int
) -> torch.Tensor:
    bits = require_bit_width(bit_width)
    levels = float((1 << bits) - 1)
    values = retained.detach().to(device="cpu", dtype=torch.float32).contiguous()
    ranges = channel_ranges.detach().to(device="cpu", dtype=torch.float32)
    minima = ranges[:, 0]
    spans = ranges[:, 1] - minima
    constant = spans <= CONSTANT_SPAN_EPSILON
    safe_spans = torch.where(constant, torch.ones_like(spans), spans)
    normalized = torch.clamp((values - minima) / safe_spans, 0.0, 1.0)
    codes = torch.round(normalized * levels).to(torch.uint8)
    if bool(constant.any()):
        codes[:, constant] = 0
    return codes.contiguous()


def analytical_size(q: float, family_id: int, bit_width: int) -> LowBitAnalyticalSize:
    bits = require_bit_width(bit_width)
    channels = _channels_for_family(int(family_id))
    plan = continuous_q.quantize_q(q)
    mask_bytes = 0 if plan.is_bypass else contract.mask_byte_count()
    ranges = range_byte_count(channels)
    values = packed_value_byte_count(plan.keep_count * channels, bits)
    return LowBitAnalyticalSize(
        q=plan.wire_q,
        q_e4=plan.q_e4,
        bit_width=bits,
        family_id=int(family_id),
        channels=channels,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=mask_bytes,
        range_bytes=ranges,
        value_bytes=values,
        total_bytes=HEADER_BYTES + mask_bytes + ranges + values,
    )


def encode_sparse(
    prepared: PreparedLowBitFeature,
    q: float,
    bit_width: int,
    selection: CellSelection | None = None,
) -> LowBitSparsePayload:
    prepared = _require_prepared(prepared)
    bits = require_bit_width(bit_width)
    plan = continuous_q.quantize_q(q)
    cell_major = prepared.feature.detach().reshape(
        prepared.channels, contract.SPLIT_CELLS
    ).transpose(0, 1)
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
            cells=contract.SPLIT_CELLS,
            spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
        )
        indices = selection.keep_indices.to(torch.int64).cpu()
        mask = _pack_bitmask(indices, contract.SPLIT_CELLS)
        retained = cell_major.index_select(0, indices.to(cell_major.device))

    codes = _quantize_retained(retained, prepared.channel_ranges, bits)
    values = _pack_codes(codes, bits)
    ranges = (
        prepared.channel_ranges.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        CODEC_ID_PER_CHANNEL_LOWBIT,
        bits,
        prepared.family_id,
        prepared.routing_tag,
        prepared.channels,
        contract.SPLIT_HEIGHT,
        contract.SPLIT_WIDTH,
        plan.q_e4,
        plan.keep_count,
        len(mask),
        len(ranges),
        len(values),
    )
    payload = LowBitSparsePayload(
        data=header + mask + ranges + values,
        q=plan.wire_q,
        bit_width=bits,
        family_id=prepared.family_id,
        routing_tag=prepared.routing_tag,
        channels=prepared.channels,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=len(mask),
        range_bytes=len(ranges),
        value_bytes=len(values),
    )
    expected = analytical_size(plan.wire_q, prepared.family_id, bits)
    if payload.total_bytes != expected.total_bytes:
        raise guards.HybridQPayloadError("low-bit payload violates analytical size")
    return payload


def _payload_bytes(payload: bytes | bytearray | memoryview | LowBitSparsePayload) -> bytes:
    if isinstance(payload, LowBitSparsePayload):
        return payload.data
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    raise guards.HybridQPayloadError("low-bit payload must be bytes-like")


def inspect(
    payload: bytes | bytearray | memoryview | LowBitSparsePayload,
) -> InspectedLowBitPayload:
    data = _payload_bytes(payload)
    if len(data) < HEADER_BYTES:
        raise guards.HybridQPayloadError("payload is shorter than the low-bit header")
    header = LowBitHeader(*struct.unpack(HEADER_FORMAT, data[:HEADER_BYTES]))
    if header.magic != MAGIC or header.version != FORMAT_VERSION:
        raise guards.HybridQPayloadError("low-bit magic or version mismatch")
    if header.codec_id != CODEC_ID_PER_CHANNEL_LOWBIT:
        raise guards.HybridQPayloadError("low-bit codec identity mismatch")
    try:
        bits = require_bit_width(int(header.bit_width))
    except guards.HybridQConfigError as exc:
        raise guards.HybridQPayloadError(
            f"unsupported low-bit wire width {header.bit_width}"
        ) from exc
    family, channels, tag = _require_family_wire_identity(
        int(header.family_id), int(header.channels), int(header.routing_tag)
    )
    if (header.height, header.width) != contract.SPLIT_SPATIAL_SHAPE:
        raise guards.HybridQPayloadError("low-bit spatial dimensions are not 112x192")
    q = int(header.q_e4) / 10000.0
    try:
        plan = continuous_q.quantize_q(q)
    except guards.HybridQConfigError as exc:
        raise guards.HybridQPayloadError("low-bit header q is invalid") from exc
    if plan.q_e4 != int(header.q_e4):
        raise guards.HybridQPayloadError("low-bit q is off the wire grid")
    guards.require_keep_cardinality(int(header.keep_count), plan.keep_count)

    expected_mask = 0 if plan.is_bypass else contract.mask_byte_count()
    expected_ranges = range_byte_count(channels)
    expected_values = packed_value_byte_count(plan.keep_count * channels, bits)
    if int(header.mask_bytes) != expected_mask:
        raise guards.HybridQPayloadError("low-bit mask byte count mismatch")
    if int(header.range_bytes) != expected_ranges:
        raise guards.HybridQPayloadError("low-bit range byte count mismatch")
    if int(header.value_bytes) != expected_values:
        raise guards.HybridQPayloadError("low-bit value byte count mismatch")
    if len(data) != HEADER_BYTES + expected_mask + expected_ranges + expected_values:
        raise guards.HybridQPayloadError("low-bit framed length mismatch")

    mask_end = HEADER_BYTES + expected_mask
    ranges_end = mask_end + expected_ranges
    if plan.is_bypass:
        keep_indices = torch.arange(contract.SPLIT_CELLS, dtype=torch.int64)
        keep_mask = torch.ones(contract.SPLIT_SPATIAL_SHAPE, dtype=torch.bool)
    else:
        raw_indices = _unpack_bitmask(data[HEADER_BYTES:mask_end], contract.SPLIT_CELLS)
        if int(raw_indices.size) != plan.keep_count:
            raise guards.HybridQPayloadError("low-bit mask cardinality mismatch")
        keep_indices = guards.require_sorted_unique_indices(
            torch.from_numpy(raw_indices), contract.SPLIT_CELLS
        )
        flat_mask = torch.zeros(contract.SPLIT_CELLS, dtype=torch.bool)
        flat_mask[keep_indices] = True
        keep_mask = flat_mask.reshape(contract.SPLIT_SPATIAL_SHAPE)
    ae_contract.require_keep_mask(keep_mask, expect_keep=plan.keep_count)

    range_array = np.frombuffer(data[mask_end:ranges_end], dtype="<f4")
    if int(range_array.size) != channels * 2:
        raise guards.HybridQPayloadError("low-bit range element count mismatch")
    channel_ranges = torch.from_numpy(range_array.reshape(channels, 2).copy()).to(
        torch.float32
    )
    _require_ranges(channel_ranges, channels, what="decoded channel ranges")

    code_count = plan.keep_count * channels
    flat_codes = _unpack_codes(data[ranges_end:], code_count, bits)
    values = flat_codes.reshape(plan.keep_count, channels).contiguous()
    return InspectedLowBitPayload(
        header=header,
        q=plan.wire_q,
        bit_width=bits,
        family_id=family,
        routing_tag=tag,
        channels=channels,
        keep_indices=keep_indices,
        keep_mask=keep_mask,
        channel_ranges=channel_ranges,
        values=values,
    )


def decode_inspected(
    parsed: InspectedLowBitPayload,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    levels = float((1 << parsed.bit_width) - 1)
    minima = parsed.channel_ranges[:, 0]
    spans = parsed.channel_ranges[:, 1] - minima
    constant = spans <= CONSTANT_SPAN_EPSILON
    retained = parsed.values.to(torch.float32) / levels
    retained = retained * spans + minima
    if bool(constant.any()):
        retained[:, constant] = minima[constant]
    guards.require_finite(retained, "dequantized low-bit values")

    cell_major = torch.zeros(
        contract.SPLIT_CELLS, parsed.channels, dtype=torch.float32
    )
    cell_major.index_copy_(0, parsed.keep_indices, retained)
    dense = cell_major.transpose(0, 1).reshape(
        parsed.channels, contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH
    ).contiguous()
    _require_feature(dense, parsed.channels, what="decoded low-bit feature")
    return dense, parsed.keep_mask, parsed.q


def decode_sparse(
    payload: bytes | bytearray | memoryview | LowBitSparsePayload,
) -> tuple[torch.Tensor, torch.Tensor, float, InspectedLowBitPayload]:
    parsed = inspect(payload)
    dense, mask, q = decode_inspected(parsed)
    return dense, mask, q, parsed


def _compress(
    payload: LowBitSparsePayload, wire_codec: ZstdWireCodec | None
) -> LowBitZstdPacket:
    codec = wire_codec if wire_codec is not None else ZstdWireCodec()
    compressed = codec.compress(payload.data)
    if frame_content_size(compressed.data) != payload.total_bytes:
        raise guards.HybridQPayloadError("zstd content size does not bind low-bit payload")
    return LowBitZstdPacket(
        data=compressed.data,
        uncompressed_bytes=payload.total_bytes,
        bit_width=payload.bit_width,
        family_id=payload.family_id,
        routing_tag=payload.routing_tag,
        channels=payload.channels,
    )


def encode_noae_frame(
    c2: torch.Tensor,
    ranker,
    q: float,
    bit_width: int,
    *,
    wire_codec: ZstdWireCodec | None = None,
) -> LowBitTransport:
    guards.require_frozen_c2(c2)
    plan = continuous_q.quantize_q(q)
    if plan.is_bypass:
        selection = None
        keep_mask = torch.ones(
            contract.SPLIT_SPATIAL_SHAPE, dtype=torch.bool, device=c2.device
        )
    else:
        selection = continuous_q.select_cells(ranker.score_cells(c2), plan.wire_q)
        keep_mask = selection.keep_mask
    prepared = prepare_feature(
        c2,
        family_id=ae_contract.AE_FAMILY_NOAE,
        routing_tag=ae_contract.AE_UNBOUND_ROUTING_TAG,
    )
    payload = encode_sparse(prepared, plan.wire_q, bit_width, selection)
    return LowBitTransport(
        plan=plan,
        bit_width=require_bit_width(bit_width),
        family_id=prepared.family_id,
        selection=selection,
        keep_mask=keep_mask,
        packet=_compress(payload, wire_codec),
    )


def encode_ae_frame(
    c2: torch.Tensor,
    autoencoder: SplitFeatureAE,
    ranker,
    q: float,
    bit_width: int,
    *,
    wire_codec: ZstdWireCodec | None = None,
) -> LowBitTransport:
    ae_contract.require_bound_routing_tag(
        autoencoder.routing_tag, what=f"{autoencoder.family_name} routing tag"
    )
    composition = ae_composition.compose(c2, autoencoder, ranker, q)
    prepared = prepare_feature(
        composition.latent.detach(),
        family_id=autoencoder.family_id,
        routing_tag=autoencoder.routing_tag,
    )
    payload = encode_sparse(
        prepared,
        composition.plan.wire_q,
        bit_width,
        composition.selection,
    )
    return LowBitTransport(
        plan=composition.plan,
        bit_width=require_bit_width(bit_width),
        family_id=prepared.family_id,
        selection=composition.selection,
        keep_mask=composition.keep_mask,
        packet=_compress(payload, wire_codec),
    )


def decompress_packet(
    packet: LowBitZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> bytes:
    if not isinstance(packet, LowBitZstdPacket) or not packet.data:
        raise guards.HybridQPayloadError("decode requires a non-empty LowBitZstdPacket")
    codec = wire_codec if wire_codec is not None else ZstdWireCodec()
    return codec.decompress(packet.data, expected_bytes=packet.uncompressed_bytes)


def decode(
    packet: LowBitZstdPacket, *, wire_codec: ZstdWireCodec | None = None
) -> tuple[torch.Tensor, torch.Tensor, float, InspectedLowBitPayload]:
    sparse = decompress_packet(packet, wire_codec=wire_codec)
    dense, mask, q, parsed = decode_sparse(sparse)
    if (
        parsed.bit_width != packet.bit_width
        or parsed.family_id != packet.family_id
        or parsed.routing_tag != packet.routing_tag
        or parsed.channels != packet.channels
    ):
        raise guards.HybridQPayloadError("inner low-bit header disagrees with packet metadata")
    return dense, mask, q, parsed

"""Per-channel UINT8 sparse wire codec for the frozen Hybrid-q C2 boundary.

This is a separate codec from :mod:`codec`; the existing FP32 wire is not
changed.  A frame is prepared once so its 256 full-C2 channel ranges can be
reused for every q.  Selection still happens on the original FP32 C2, and only
the selected cell values pass through the affine quantizer.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

from . import contract, continuous_q, guards
from .codec import _pack_bitmask, _unpack_bitmask
from .selection import CellSelection


MAGIC = b"HQ8\x00"
FORMAT_VERSION = 1
CODEC_ID_PER_CHANNEL_UINT8 = 1

# magic, version, codec id, C, H, W, q_e4, keep count, mask bytes,
# range bytes, value bytes.  All integers are little endian.
HEADER_FORMAT = "<4sHHIIIIIIIQ"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)  # 44

CHANNEL_RANGE_PAIRS = contract.SPLIT_CHANNELS
RANGE_BYTES = CHANNEL_RANGE_PAIRS * 2 * 4  # 2,048
CONSTANT_SPAN_EPSILON = 1.0e-12


@dataclass(frozen=True)
class PreparedUint8Frame:
    """Original FP32 C2 plus ranges computed once from its complete contents."""

    c2: torch.Tensor
    channel_ranges: torch.Tensor  # [256, 2], min then max, FP32
    c2_version: int
    ranges_version: int


@dataclass(frozen=True)
class Uint8Header:
    magic: bytes
    version: int
    codec_id: int
    channels: int
    height: int
    width: int
    q_e4: int
    keep_count: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int


@dataclass(frozen=True)
class Uint8SparsePayload:
    """One uncompressed diagnostic payload; deployment wraps ``data`` in zstd."""

    data: bytes
    q: float
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        return len(self.data)

    @property
    def framed_fp32_q0_ratio(self) -> float:
        return self.total_bytes / contract.FRAMED_Q0_PAYLOAD_BYTES


@dataclass(frozen=True)
class InspectedUint8Payload:
    """Strictly validated sparse contents, before dequantization/scatter."""

    header: Uint8Header
    q: float
    keep_indices: torch.Tensor  # ascending row-major int64
    channel_ranges: torch.Tensor  # [256, 2] FP32
    values: torch.Tensor  # [keep, 256] UINT8, cell-major


@dataclass(frozen=True)
class AnalyticalPayloadSize:
    q: float
    q_e4: int
    keep_count: int
    header_bytes: int
    mask_bytes: int
    range_bytes: int
    value_bytes: int
    total_bytes: int

    @property
    def framed_fp32_q0_ratio(self) -> float:
        return self.total_bytes / contract.FRAMED_Q0_PAYLOAD_BYTES


def _require_valid_ranges(ranges: torch.Tensor, *, what: str) -> torch.Tensor:
    if not isinstance(ranges, torch.Tensor):
        raise guards.HybridQPayloadError(f"{what} must be a torch.Tensor")
    if tuple(ranges.shape) != (contract.SPLIT_CHANNELS, 2):
        raise guards.HybridQPayloadError(
            f"{what} must be [{contract.SPLIT_CHANNELS}, 2], got {list(ranges.shape)}"
        )
    if ranges.dtype is not torch.float32:
        raise guards.HybridQPayloadError(f"{what} must be float32, got {ranges.dtype}")
    guards.require_finite(ranges, what)
    minima = ranges[:, 0]
    maxima = ranges[:, 1]
    if bool((minima > maxima).any()):
        raise guards.HybridQPayloadError(
            "channel range ordering must be [min, max] with min <= max"
        )
    spans = maxima - minima
    if not bool(torch.isfinite(spans).all()):
        raise guards.HybridQPayloadError("channel range span is non-finite")
    return ranges


def prepare(c2: torch.Tensor) -> PreparedUint8Frame:
    """Compute all per-channel ranges once from one complete original FP32 C2."""
    guards.require_frozen_c2(c2)
    with torch.no_grad():
        flat = c2.detach().reshape(contract.SPLIT_CHANNELS, contract.SPLIT_CELLS)
        ranges = torch.stack((flat.amin(dim=1), flat.amax(dim=1)), dim=1).contiguous()
        _require_valid_ranges(ranges, what="computed channel ranges")
    return PreparedUint8Frame(
        c2=c2,
        channel_ranges=ranges,
        c2_version=int(c2._version),
        ranges_version=int(ranges._version),
    )


def _require_prepared(prepared: PreparedUint8Frame) -> PreparedUint8Frame:
    if not isinstance(prepared, PreparedUint8Frame):
        raise guards.HybridQPayloadError("encode requires a PreparedUint8Frame")
    guards.require_frozen_c2(prepared.c2, what="prepared original C2")
    if int(prepared.c2._version) != int(prepared.c2_version):
        raise guards.HybridQPayloadError("prepared original C2 changed after range analysis")
    if int(prepared.channel_ranges._version) != int(prepared.ranges_version):
        raise guards.HybridQPayloadError("prepared channel ranges changed after analysis")
    _require_valid_ranges(prepared.channel_ranges, what="prepared channel ranges")
    return prepared


def _quantize_retained(
    retained: torch.Tensor, channel_ranges: torch.Tensor
) -> torch.Tensor:
    """Apply the specified affine rule on CPU to selected FP32 values only."""
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


def encode(
    prepared: PreparedUint8Frame,
    q: float,
    selection: CellSelection | None = None,
) -> Uint8SparsePayload:
    """Frame selected UINT8 values using full-C2 ranges from ``prepare``.

    q=0 has no mask and bypasses selection, but its complete value block is
    still quantized.  q>0 requires the existing continuous-q selection.
    """
    prepared = _require_prepared(prepared)
    plan = continuous_q.quantize_q(q)
    c2 = prepared.c2
    cell_major = c2.detach().reshape(
        contract.SPLIT_CHANNELS, contract.SPLIT_CELLS
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

    # Selection/indexing precedes this call: dropped values are never quantized.
    codes = _quantize_retained(retained, prepared.channel_ranges)
    values = codes.numpy().tobytes(order="C")
    ranges = (
        prepared.channel_ranges.detach()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    if len(ranges) != RANGE_BYTES:
        raise guards.HybridQPayloadError("computed channel range block has wrong length")

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        CODEC_ID_PER_CHANNEL_UINT8,
        contract.SPLIT_CHANNELS,
        contract.SPLIT_HEIGHT,
        contract.SPLIT_WIDTH,
        plan.q_e4,
        plan.keep_count,
        len(mask),
        len(ranges),
        len(values),
    )
    payload = Uint8SparsePayload(
        data=header + mask + ranges + values,
        q=plan.wire_q,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=len(mask),
        range_bytes=len(ranges),
        value_bytes=len(values),
    )
    expected = analytical_size(plan.wire_q)
    if payload.total_bytes != expected.total_bytes:
        raise guards.HybridQPayloadError("encoded payload length violates analytical size")
    return payload


def _payload_bytes(payload: bytes | bytearray | memoryview | Uint8SparsePayload) -> bytes:
    if isinstance(payload, Uint8SparsePayload):
        return payload.data
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    raise guards.HybridQPayloadError("UINT8 payload must be a bytes-like object")


def inspect(
    payload: bytes | bytearray | memoryview | Uint8SparsePayload,
) -> InspectedUint8Payload:
    """Parse and fully validate one uncompressed UINT8 sparse payload."""
    data = _payload_bytes(payload)
    if len(data) < HEADER_BYTES:
        raise guards.HybridQPayloadError("payload shorter than the UINT8 header")

    fields = struct.unpack(HEADER_FORMAT, data[:HEADER_BYTES])
    header = Uint8Header(*fields)
    if header.magic != MAGIC:
        raise guards.HybridQPayloadError("UINT8 payload magic mismatch")
    if header.version != FORMAT_VERSION:
        raise guards.HybridQPayloadError(
            f"unsupported UINT8 format version {header.version}"
        )
    if header.codec_id != CODEC_ID_PER_CHANNEL_UINT8:
        raise guards.HybridQPayloadError("UINT8 codec identity mismatch")
    if (header.channels, header.height, header.width) != contract.SPLIT_SHAPE:
        raise guards.HybridQPayloadError(
            "UINT8 header dimensions do not match the frozen C2 contract"
        )

    q = header.q_e4 / 10000.0
    try:
        plan = continuous_q.quantize_q(q)
    except guards.HybridQConfigError as exc:
        raise guards.HybridQPayloadError("UINT8 header q is invalid") from exc
    if plan.q_e4 != header.q_e4:
        raise guards.HybridQPayloadError("UINT8 header q is off the 1e-4 wire grid")
    guards.require_keep_cardinality(header.keep_count, plan.keep_count)

    expected_mask = 0 if plan.is_bypass else contract.mask_byte_count()
    if header.mask_bytes != expected_mask:
        raise guards.HybridQPayloadError(
            f"mask length {header.mask_bytes} != expected {expected_mask}"
        )
    if header.range_bytes != RANGE_BYTES:
        raise guards.HybridQPayloadError(
            f"range length {header.range_bytes} != expected {RANGE_BYTES}"
        )
    expected_values = plan.keep_count * contract.SPLIT_CHANNELS
    if header.value_bytes != expected_values:
        raise guards.HybridQPayloadError(
            f"value length {header.value_bytes} != expected {expected_values}"
        )
    expected_total = HEADER_BYTES + header.mask_bytes + RANGE_BYTES + header.value_bytes
    if len(data) != expected_total:
        raise guards.HybridQPayloadError("payload length disagrees with UINT8 header")

    mask_end = HEADER_BYTES + header.mask_bytes
    range_end = mask_end + RANGE_BYTES
    if plan.is_bypass:
        keep_indices = torch.arange(contract.SPLIT_CELLS, dtype=torch.int64)
    else:
        unpacked = _unpack_bitmask(data[HEADER_BYTES:mask_end], contract.SPLIT_CELLS)
        if int(unpacked.size) != plan.keep_count:
            raise guards.HybridQPayloadError(
                f"bitmask retains {unpacked.size} cells, header declares {plan.keep_count}"
            )
        keep_indices = guards.require_sorted_unique_indices(
            torch.from_numpy(unpacked), contract.SPLIT_CELLS
        )

    range_array = np.frombuffer(data[mask_end:range_end], dtype="<f4")
    if int(range_array.size) != CHANNEL_RANGE_PAIRS * 2:
        raise guards.HybridQPayloadError("range block does not contain 256 min/max pairs")
    channel_ranges = torch.from_numpy(
        range_array.reshape(CHANNEL_RANGE_PAIRS, 2).copy()
    ).to(torch.float32)
    _require_valid_ranges(channel_ranges, what="decoded channel ranges")

    value_array = np.frombuffer(data[range_end:], dtype=np.uint8)
    if int(value_array.size) != expected_values:
        raise guards.HybridQPayloadError("UINT8 value block element count mismatch")
    values = torch.from_numpy(
        value_array.reshape(plan.keep_count, contract.SPLIT_CHANNELS).copy()
    ).to(torch.uint8)
    return InspectedUint8Payload(
        header=header,
        q=plan.wire_q,
        keep_indices=keep_indices,
        channel_ranges=channel_ranges,
        values=values,
    )


def decode(
    payload: bytes | bytearray | memoryview | Uint8SparsePayload,
) -> tuple[torch.Tensor, float]:
    """Dequantize, spatially scatter, and return dense frozen-shape FP32 C2."""
    parsed = inspect(payload)
    minima = parsed.channel_ranges[:, 0]
    spans = parsed.channel_ranges[:, 1] - minima
    constant = spans <= CONSTANT_SPAN_EPSILON

    retained = parsed.values.to(torch.float32) / 255.0
    retained = retained * spans + minima
    if bool(constant.any()):
        retained[:, constant] = minima[constant]
    guards.require_finite(retained, "dequantized retained values")

    cell_major = torch.zeros(
        contract.SPLIT_CELLS, contract.SPLIT_CHANNELS, dtype=torch.float32
    )
    cell_major.index_copy_(0, parsed.keep_indices, retained)
    dense = cell_major.transpose(0, 1).reshape(contract.SPLIT_SHAPE).contiguous()
    guards.require_frozen_c2(dense, what="decoded UINT8 C2 tensor")
    return dense, parsed.q


def analytical_size(q: float) -> AnalyticalPayloadSize:
    """Exact pre-zstd bytes for one frozen-shape UINT8 payload at continuous q."""
    plan = continuous_q.quantize_q(q)
    mask_bytes = 0 if plan.is_bypass else contract.mask_byte_count()
    value_bytes = plan.keep_count * contract.SPLIT_CHANNELS
    total = HEADER_BYTES + mask_bytes + RANGE_BYTES + value_bytes
    return AnalyticalPayloadSize(
        q=plan.wire_q,
        q_e4=plan.q_e4,
        keep_count=plan.keep_count,
        header_bytes=HEADER_BYTES,
        mask_bytes=mask_bytes,
        range_bytes=RANGE_BYTES,
        value_bytes=value_bytes,
        total_bytes=total,
    )

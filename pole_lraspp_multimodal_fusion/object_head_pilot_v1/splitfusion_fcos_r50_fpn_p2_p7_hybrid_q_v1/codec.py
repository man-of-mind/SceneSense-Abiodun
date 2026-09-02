from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

from . import contract, guards
from .selection import CellSelection


MAGIC = b"HQ1\x00"
HEADER_FORMAT = "<4sHHHHIIIIIIQ"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)  # 44

DTYPE_FP32 = 1
_DTYPE_CODES = {torch.float32: DTYPE_FP32}
_CODE_DTYPES = {DTYPE_FP32: torch.float32}
_CODE_ITEMSIZE = {DTYPE_FP32: 4}
_CODE_NUMPY = {DTYPE_FP32: "<f4"}

FLAG_MASK_PRESENT = 1 << 0


@dataclass(frozen=True)
class SparsePayload:
    """A serialized hybrid-q wire payload and its measured length."""

    data: bytes
    q: float
    keep_count: int
    header_bytes: int
    mask_bytes: int
    value_bytes: int

    @property
    def total_bytes(self) -> int:
        """Actual serialized length of the framed wire payload."""
        return len(self.data)

    @property
    def framed_ratio(self) -> float:
        """Primary hybrid-q ratio: this payload / the framed q=0 payload."""
        return contract.framed_payload_ratio(self.total_bytes)


def _pack_bitmask(keep_indices: torch.Tensor, cells: int) -> bytes:
    """Fixed-order bitmask: bit 1 = retained, cell i at byte i//8, bit 7-(i%8).

    Byte order is ascending cell index; within a byte the most significant bit
    is the lowest cell index. Padding bits past cell N-1 are always zero.
    """
    bits = np.zeros(contract.mask_byte_count(cells) * 8, dtype=np.uint8)
    bits[keep_indices.detach().cpu().numpy().astype(np.int64)] = 1
    return np.packbits(bits, bitorder="big").tobytes()


def _unpack_bitmask(mask: bytes, cells: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(mask, dtype=np.uint8), bitorder="big")
    if bits[cells:].any():
        raise guards.HybridQPayloadError("bitmask padding bits past the last cell are set")
    return np.flatnonzero(bits[:cells]).astype(np.int64)


def _encode(
    c2: torch.Tensor,
    q: float,
    selection: CellSelection | None = None,
    *,
    registered_only: bool = True,
) -> SparsePayload:
    """Private generic encoder over any [C,H,W] FP32 tensor (tests and internals)."""
    value = guards.require_valid_q(q, registered_only=registered_only)
    guards.require_generic_c2(c2, channels=int(c2.shape[0]), what="C2 tensor")
    guards.require_finite(c2, "C2 tensor")
    dtype_code = _DTYPE_CODES.get(c2.dtype)
    if dtype_code is None:
        raise guards.HybridQPayloadError(f"unsupported transport dtype {c2.dtype}")

    channels, height, width = (int(size) for size in c2.shape)
    cells = height * width
    keep = contract.keep_count(value, cells)

    # [cells, channels]: all channels of a cell are contiguous on the wire.
    cell_major = c2.detach().reshape(channels, cells).transpose(0, 1).contiguous()

    if keep == cells:
        if selection is not None:
            raise guards.HybridQConfigError("q=0 must not carry a sparse selection")
        flags = 0
        mask = b""
        retained = cell_major
    else:
        if selection is None:
            raise guards.HybridQConfigError("q>0 requires a cell selection")
        guards.require_selection_integrity(
            selection, value, cells=cells, spatial_shape=(height, width)
        )
        indices = selection.keep_indices.to(torch.int64).cpu()
        flags = FLAG_MASK_PRESENT
        mask = _pack_bitmask(indices, cells)
        retained = cell_major.index_select(0, indices.to(cell_major.device))

    values = retained.cpu().numpy().astype(_CODE_NUMPY[dtype_code], copy=False).tobytes()
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        contract.FORMAT_VERSION,
        dtype_code,
        flags,
        0,
        channels,
        height,
        width,
        contract._q_to_e4(value),
        keep,
        len(mask),
        len(values),
    )
    return SparsePayload(
        data=header + mask + values,
        q=value,
        keep_count=keep,
        header_bytes=HEADER_BYTES,
        mask_bytes=len(mask),
        value_bytes=len(values),
    )


def encode(
    c2: torch.Tensor, q: float, selection: CellSelection | None = None
) -> SparsePayload:
    """Serialize one frozen C2 frame for transport.

    q=0 emits the dense form (no bitmask, every cell present). q>0 emits the
    bitmask plus the retained values in ascending row-major cell order, each
    retained cell holding all 256 FP32 channels contiguously. For q>0 the
    supplied selection is cross-checked against the requested q before framing.
    """
    guards.require_frozen_c2(c2)
    value = guards.require_valid_q(q)
    if selection is not None:
        guards.require_selection_integrity(
            selection,
            value,
            cells=contract.SPLIT_CELLS,
            spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
        )
    return _encode(c2, value, selection)


def _decode(
    payload: bytes | SparsePayload, *, require_frozen: bool = True
) -> tuple[torch.Tensor, float]:
    """Reconstruct the dense tensor, filling dropped cells with exact zeros."""
    data = payload.data if isinstance(payload, SparsePayload) else payload
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise guards.HybridQPayloadError("payload must be a bytes-like object")
    data = bytes(data)
    if len(data) < HEADER_BYTES:
        raise guards.HybridQPayloadError("payload shorter than the fixed header")

    (
        magic,
        version,
        dtype_code,
        flags,
        reserved,
        channels,
        height,
        width,
        q_e4,
        keep,
        mask_bytes,
        value_bytes,
    ) = struct.unpack(HEADER_FORMAT, data[:HEADER_BYTES])

    if magic != MAGIC:
        raise guards.HybridQPayloadError("payload magic mismatch")
    if version != contract.FORMAT_VERSION:
        raise guards.HybridQPayloadError(f"unsupported format version {version}")
    if reserved != 0:
        raise guards.HybridQPayloadError("reserved header field must be zero")
    if dtype_code not in _CODE_DTYPES:
        raise guards.HybridQPayloadError(f"unsupported dtype code {dtype_code}")
    if flags & ~FLAG_MASK_PRESENT:
        raise guards.HybridQPayloadError("unknown header flag bits set")
    if channels <= 0 or height <= 0 or width <= 0:
        raise guards.HybridQPayloadError("non-positive tensor dimension in header")
    if require_frozen:
        guards.require_frozen_header_dims(
            channels, height, width, dtype_code, fp32_code=DTYPE_FP32
        )

    cells = height * width
    q = q_e4 / 10000.0
    guards.require_valid_q(q, registered_only=require_frozen)
    guards.require_keep_cardinality(keep, contract.keep_count(q, cells))

    has_mask = bool(flags & FLAG_MASK_PRESENT)
    if has_mask == (keep == cells):
        raise guards.HybridQPayloadError("mask presence disagrees with keep cardinality")
    expected_mask = contract.mask_byte_count(cells) if has_mask else 0
    if mask_bytes != expected_mask:
        raise guards.HybridQPayloadError(
            f"bitmask length {mask_bytes} != expected {expected_mask}"
        )
    expected_values = keep * channels * _CODE_ITEMSIZE[dtype_code]
    if value_bytes != expected_values:
        raise guards.HybridQPayloadError(
            f"value block length {value_bytes} != expected {expected_values}"
        )
    if len(data) != HEADER_BYTES + mask_bytes + value_bytes:
        raise guards.HybridQPayloadError("payload length disagrees with header")

    mask_end = HEADER_BYTES + mask_bytes
    if has_mask:
        indices = _unpack_bitmask(data[HEADER_BYTES:mask_end], cells)
        if indices.size != keep:
            raise guards.HybridQPayloadError(
                f"bitmask retains {indices.size} cells, header declares {keep}"
            )
        index_tensor = guards.require_sorted_unique_indices(
            torch.from_numpy(indices), cells
        )
    else:
        index_tensor = None

    flat = np.frombuffer(data[mask_end:], dtype=_CODE_NUMPY[dtype_code])
    values = torch.from_numpy(flat.reshape(keep, channels).copy()).to(
        _CODE_DTYPES[dtype_code]
    )
    guards.require_finite(values, "decoded values")

    if index_tensor is None:
        cell_major = values
    else:
        cell_major = torch.zeros(cells, channels, dtype=values.dtype)
        cell_major.index_copy_(0, index_tensor, values)

    dense = cell_major.transpose(0, 1).reshape(channels, height, width).contiguous()
    return dense, q


def decode(payload: bytes | SparsePayload) -> tuple[torch.Tensor, float]:
    """Decode a frozen-contract payload to dense [256,112,192] FP32.

    Fails closed unless the header specifies exactly 256 channels, height 112,
    width 192 and FP32. Dropped cells decode to exact zeros.
    """
    dense, q = _decode(payload, require_frozen=True)
    guards.require_frozen_c2(dense, what="decoded C2 tensor")
    return dense, q


def raw_fp32_reference_bytes(c2: torch.Tensor) -> int:
    """Unframed raw FP32 tensor size. Reported separately from framed payloads.

    This is NOT the wire representation and the framed q=0 payload is not
    byte-identical to it: framing adds the 44-byte versioned header.
    """
    return int(c2.numel()) * 4

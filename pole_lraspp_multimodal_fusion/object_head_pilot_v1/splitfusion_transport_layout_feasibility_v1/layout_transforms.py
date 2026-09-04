"""Reversible external value-layout transforms for existing inner payloads.

These helpers intentionally know neither a production wire version nor a
decoder.  A caller supplies a value-block boundary and logical ``[K, C]`` code
shape obtained from the existing codec inspector.  UINT6/UINT4 packing and
unpacking are delegated to the existing low-bit codec helpers; this module only
permutes already-quantized symbols or applies the registered modular delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import lowbit_transport


class Layout(str, Enum):
    CURRENT_CELL_MAJOR = "CURRENT_CELL_MAJOR"
    CHANNEL_MAJOR = "CHANNEL_MAJOR"
    CHANNEL_MAJOR_MODULAR_DELTA = "CHANNEL_MAJOR_MODULAR_DELTA"


@dataclass(frozen=True)
class ValueBlockPlan:
    """External metadata; no production header is changed or interpreted here."""

    value_offset: int
    keep_count: int
    channels: int
    bit_width: int

    @property
    def symbol_count(self) -> int:
        return self.keep_count * self.channels


def _require_plan(plan: ValueBlockPlan, inner: bytes) -> ValueBlockPlan:
    if not isinstance(plan, ValueBlockPlan) or plan.value_offset < 0:
        raise ValueError("invalid external value-block plan")
    if plan.keep_count <= 0 or plan.channels <= 0 or plan.bit_width not in (4, 6, 8):
        raise ValueError("invalid value-block dimensions or bit width")
    if plan.value_offset > len(inner):
        raise ValueError("value block starts beyond inner payload")
    expected = (plan.symbol_count * plan.bit_width + 7) // 8
    if len(inner) - plan.value_offset != expected:
        raise ValueError("value-block byte count disagrees with external plan")
    return plan


def _unpack_symbols(block: bytes, plan: ValueBlockPlan) -> np.ndarray:
    if plan.bit_width == 8:
        values = np.frombuffer(block, dtype=np.uint8).copy()
    else:
        # Existing implementation owns all UINT6/UINT4 bit order semantics.
        values = lowbit_transport._unpack_codes(block, plan.symbol_count, plan.bit_width).numpy()
    if int(values.size) != plan.symbol_count:
        raise ValueError("unpacked symbol count drift")
    return values


def _unpack(block: bytes, plan: ValueBlockPlan) -> np.ndarray:
    return _unpack_symbols(block, plan).reshape(plan.keep_count, plan.channels)


def _pack(values: np.ndarray, plan: ValueBlockPlan) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if int(flat.size) != plan.symbol_count:
        raise ValueError("packed symbol count drift")
    if plan.bit_width == 8:
        return flat.tobytes()
    # Existing implementation owns all UINT6/UINT4 packing order semantics.
    return lowbit_transport._pack_codes(torch.from_numpy(flat.copy()), plan.bit_width)


def transform(inner: bytes, plan: ValueBlockPlan, layout: Layout) -> bytes:
    """Return an external-layout payload with the non-value prefix unchanged."""
    plan = _require_plan(plan, inner)
    if layout is Layout.CURRENT_CELL_MAJOR:
        return bytes(inner)
    prefix, block = inner[: plan.value_offset], inner[plan.value_offset :]
    cell_major = _unpack(block, plan)
    channel_major = cell_major.T.copy()
    if layout is Layout.CHANNEL_MAJOR_MODULAR_DELTA:
        modulus = 1 << plan.bit_width
        encoded = np.empty_like(channel_major)
        encoded[:, 0] = channel_major[:, 0]
        encoded[:, 1:] = (channel_major[:, 1:].astype(np.int16) - channel_major[:, :-1].astype(np.int16)) % modulus
        channel_major = encoded
    elif layout is not Layout.CHANNEL_MAJOR:
        raise ValueError(f"unrecognized layout {layout!r}")
    transformed = prefix + _pack(channel_major, plan)
    if transformed[: plan.value_offset] != prefix:
        raise ValueError("layout changed header/mask/range bytes")
    return transformed


def inverse(transformed: bytes, plan: ValueBlockPlan, layout: Layout) -> bytes:
    """Restore the original complete inner payload byte-for-byte."""
    plan = _require_plan(plan, transformed)
    if layout is Layout.CURRENT_CELL_MAJOR:
        return bytes(transformed)
    prefix, block = transformed[: plan.value_offset], transformed[plan.value_offset :]
    channel_major = _unpack_symbols(block, plan).reshape(plan.channels, plan.keep_count).copy()
    if layout is Layout.CHANNEL_MAJOR_MODULAR_DELTA:
        modulus = 1 << plan.bit_width
        decoded = np.empty_like(channel_major)
        decoded[:, 0] = channel_major[:, 0]
        for index in range(1, plan.keep_count):
            decoded[:, index] = (decoded[:, index - 1].astype(np.int16) + channel_major[:, index].astype(np.int16)) % modulus
        channel_major = decoded
    elif layout is not Layout.CHANNEL_MAJOR:
        raise ValueError(f"unrecognized layout {layout!r}")
    restored = prefix + _pack(channel_major.T, plan)
    if restored[: plan.value_offset] != prefix:
        raise ValueError("inverse changed header/mask/range bytes")
    return restored

"""Continuous-q execution interface over the frozen hybrid-q transport path.

Execution only. This module lets the already-trained stable ranker and the
existing v1 sparse codec serve *any* finite q in [0, 0.98] at the wire
resolution the header already carries, instead of only the six registered
measurement anchors. It trains nothing, loads nothing, measures nothing and
changes no frozen behaviour: ranker, selection, codec, guards and the locked
config are untouched, and every registered q still produces byte-identical
bytes through this path (see `tests/test_continuous_q.py`).

Why an arbitrary q is constructible at all: the ranker never sees q and reads
detached fused C2 only, so one q-independent per-cell ordering induces every
mask, and the registered masks are nested prefixes of it. Cutting that ordering
at an unregistered cardinality needs no retraining.

Scope caveat, unchanged by this module: *executability is not measured
accuracy*. Phase 6 measured accuracy at the six registered anchors only.
Accuracy at an unmeasured q is neither interpolated nor validated here. Callers
that need a measured accuracy guarantee should keep using the registered
anchors, or snap a request down with `contract.snap_continuous_q` before
calling in. This interface deliberately does **not** snap: it serves exactly
the q it was asked for, so that snapping stays a caller-side policy choice
rather than a hidden transport behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import contract, guards
from .codec import SparsePayload, _decode, _encode
from .selection import CellSelection, _select_cells, apply_selection


# The header already encodes q in ten-thousandths (`contract._q_to_e4`), so
# 1e-4 is the exact wire resolution and no format change is needed.
WIRE_Q_SCALE = 10000
WIRE_Q_RESOLUTION = 1.0 / WIRE_Q_SCALE

# Supported continuous range: q=0 is the dense identity, q=0.98 is the most
# aggressive registered rung and remains the maximum drop the wire may carry.
CONTINUOUS_Q_MIN = 0.0
CONTINUOUS_Q_MAX = 0.98


@dataclass(frozen=True)
class ContinuousQ:
    """One resolved continuous-q request: what the wire will actually carry."""

    requested_q: float
    wire_q: float
    q_e4: int
    cells: int
    keep_count: int
    drop_count: int
    is_registered: bool
    is_bypass: bool

    @property
    def snapped(self) -> bool:
        """Always False: a continuous request is never moved to an anchor."""
        return False


@dataclass(frozen=True)
class ContinuousTransport:
    """End-to-end result of executing one continuous q on one frozen frame."""

    plan: ContinuousQ
    selection: CellSelection | None
    masked: torch.Tensor
    payload: SparsePayload


def quantize_q(q: float, *, cells: int = contract.SPLIT_CELLS) -> ContinuousQ:
    """Resolve any real finite q in [0, 0.98] onto the 1e-4 wire grid.

    Quantization is half-up to ten-thousandths, matching `contract._q_to_e4`
    and `contract.drop_count`, and it is the *only* movement applied: the result
    is never nudged onto a registered anchor. Keep/drop counts come from the
    existing `contract.keep_count`/`contract.drop_count`, so a registered q
    resolves to exactly its registered cardinality.
    """
    # Reuses the type/finiteness checks; CONTINUOUS_Q_MAX < 1.0, so anything
    # this interface accepts is also inside the generic 0 <= q < 1 contract.
    value = guards.require_valid_q(q, registered_only=False)
    if not CONTINUOUS_Q_MIN <= value <= CONTINUOUS_Q_MAX:
        raise guards.HybridQConfigError(
            f"continuous q must satisfy {CONTINUOUS_Q_MIN} <= q <= {CONTINUOUS_Q_MAX}, "
            f"got {value!r}"
        )

    q_e4 = int(math.floor(value * WIRE_Q_SCALE + 0.5))
    wire_q = q_e4 / WIRE_Q_SCALE
    if contract._q_to_e4(wire_q) != q_e4:
        raise guards.HybridQConfigError(
            f"q={value!r} did not quantize onto the {WIRE_Q_RESOLUTION} wire grid"
        )

    total = int(cells)
    keep = contract.keep_count(wire_q, total)
    drop = contract.drop_count(wire_q, total)
    if keep + drop != total or not 1 <= keep <= total:
        raise guards.HybridQConfigError(
            f"q={wire_q!r} yields an unusable keep count {keep} of {total} cells"
        )
    return ContinuousQ(
        requested_q=value,
        wire_q=wire_q,
        q_e4=q_e4,
        cells=total,
        keep_count=keep,
        drop_count=drop,
        is_registered=contract.is_registered_q(wire_q),
        is_bypass=drop == 0,
    )


def select_cells(scores: torch.Tensor, q: float) -> CellSelection:
    """Continuous-q selection over one frozen [112,192] score map.

    Identical ordering to `selection.select_cells`: the same private stable
    descending sort, the same lower-row-major-index tie preference, the same
    integrity cross-check. Only the registered-q admission is relaxed.
    """
    guards.require_frozen_scores(scores)
    plan = quantize_q(q)
    selection = _select_cells(scores, plan.wire_q, registered_only=False)
    guards.require_selection_integrity(
        selection,
        plan.wire_q,
        cells=contract.SPLIT_CELLS,
        spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
    )
    return selection


def select_and_apply(
    c2: torch.Tensor, ranker, q: float
) -> tuple[torch.Tensor, CellSelection | None]:
    """Continuous-q masking at the frozen boundary.

    Preserves the exact q=0 bypass: when the resolved wire q drops no cells the
    ranker is never invoked and the input tensor object itself is returned, so
    dense identity stays exact by construction. Callers must not mutate it.
    """
    guards.require_frozen_c2(c2)
    plan = quantize_q(q)
    if plan.is_bypass:
        return c2, None
    scores = ranker.score_cells(c2)
    selection = select_cells(scores, plan.wire_q)
    return apply_selection(c2, selection), selection


def encode(
    c2: torch.Tensor, q: float, selection: CellSelection | None = None
) -> SparsePayload:
    """Serialize one frozen frame at a continuous q with the existing v1 wire.

    Same 44-byte header, same fixed-order bitmask, same cell-major retained
    value block as `codec.encode`; the header simply carries an unregistered
    q_e4. The supplied selection is cross-checked against the resolved wire q
    before framing.
    """
    guards.require_frozen_c2(c2)
    plan = quantize_q(q)
    if selection is not None:
        guards.require_selection_integrity(
            selection,
            plan.wire_q,
            cells=contract.SPLIT_CELLS,
            spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
        )
    return _encode(c2, plan.wire_q, selection, registered_only=False)


def decode(payload: bytes | SparsePayload) -> tuple[torch.Tensor, float]:
    """Decode a continuous-q payload to dense [256,112,192] FP32.

    Runs the existing structural validation (magic, version, reserved word,
    flags, dimensions, keep cardinality against the header q, bitmask length,
    padding bits, value-block length, framed length, index ordering), then
    fails closed unless the payload describes exactly the frozen boundary and
    its q lies on the supported continuous grid. Dropped cells decode to exact
    zeros; retained values are bit-exact.
    """
    dense, q = _decode(payload, require_frozen=False)
    guards.require_frozen_c2(dense, what="decoded C2 tensor")
    plan = quantize_q(q)
    if contract._q_to_e4(plan.wire_q) != contract._q_to_e4(q):
        raise guards.HybridQPayloadError(f"payload q={q!r} is off the wire grid")
    return dense, plan.wire_q


def transport(c2: torch.Tensor, ranker, q: float) -> ContinuousTransport:
    """Execute one continuous q end to end: rank, select, mask, frame.

    Mirrors the production call order used by the registered runners
    (`codec.encode(apply_selection(frame, selection), q, selection)`), so at a
    registered q the emitted bytes are identical to the discrete path.
    """
    plan = quantize_q(q)
    masked, selection = select_and_apply(c2, ranker, plan.wire_q)
    payload = encode(masked, plan.wire_q, selection)
    guards.require_keep_cardinality(int(payload.keep_count), plan.keep_count)
    return ContinuousTransport(
        plan=plan, selection=selection, masked=masked, payload=payload
    )

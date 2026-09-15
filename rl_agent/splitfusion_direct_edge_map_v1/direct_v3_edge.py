"""Serve the direct edge->map architecture from the repaired v3 edge tail.

The direct runtime asks its edge for two things: ``process(payload,
transmitted_action_id=...)`` returning an ``EdgeDispatchResult``-shaped value,
and a ``take_snapshot()`` carrying the service records, the dense segmentation
labels and the tail's own provenance counts. The optimized v3 edge exposes the
same work through the detached compute/publication pair instead, because its
publication product is built from CPU tensors only.

This module is the adapter between those two surfaces. It adds no science: the
frozen model, codec, action catalog, thresholds, record schema and byte
accounting are the ones ``DetachedOptimizedEdgeV3`` already preloads, and the
per-frame product is the one the v3 tail already produced. The pre-tail
deadline gate is installed exactly where the production runtime installs it —
around inference only — so a capture that can no longer be timely is refused
before the model runs, not after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
    DetachedOptimizedEdgeV3,
    preload_detached_optimized_edge_v3,
)
from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import (
    DetachedOptimizedTailAdapterV3,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import ContextTailSnapshot
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    EDGE_STAGE_BEFORE_TAIL,
)
from rl_agent.splitfusion_live_dispatch_v1.timing import TimingTrace

from .protocol import _require

EDGE_VARIANT = "OVERLAPPED_OUTPUT_PRESERVING_V3_REPAIRED"


@dataclass(frozen=True)
class DirectEdgeDispatchResult:
    """The fields the direct runtime reads off a completed edge call."""

    perception: Any
    serialized_output: bytes
    metadata: Any
    scientific_inner_payload_bytes: int
    framing_control_overhead_bytes: int
    total_received_bytes: int
    timing: TimingTrace


class _DeadlineGuardedComputeTail:
    """Refuse v3 inference for a capture that can no longer be timely.

    The guard wraps ``compute_product`` rather than ``__call__`` because the
    detached tail's inference entry point is ``compute_product``; the guard
    instant and the stage label are the production ones.
    """

    def __init__(
        self,
        tail: DetachedOptimizedTailAdapterV3,
        guard: Callable[[str, int], None],
    ) -> None:
        self._tail = tail
        self._guard = guard

    def compute_product(self, c2: Any, metadata: Any) -> Any:
        self._guard(EDGE_STAGE_BEFORE_TAIL, int(metadata.capture_timestamp_ns))
        return self._tail.compute_product(c2, metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tail, name)


class DirectV3Edge:
    """Present the repaired v3 detached edge as one synchronous edge call."""

    def __init__(
        self,
        edge: DetachedOptimizedEdgeV3,
        *,
        deadline_guard: Callable[[str, int], None] | None = None,
    ) -> None:
        self._edge = edge
        self._runtime = edge.runtime
        self._tail = edge.tail
        self._guard = deadline_guard
        self._snapshot: ContextTailSnapshot | None = None
        if deadline_guard is not None:
            self._runtime._detached_tail = _DeadlineGuardedComputeTail(
                edge.tail, deadline_guard
            )

    @property
    def tail_device(self) -> torch.device:
        return self._edge.device

    @property
    def variant(self) -> str:
        return EDGE_VARIANT

    @property
    def counters(self) -> Any:
        return self._runtime._counters

    @property
    def asynchronous_verdict_corrections(self) -> int:
        """Deferred finite verdicts the synchronous authority overturned."""

        return int(self._tail.asynchronous_verdict_corrections)

    def process(
        self,
        frame_bytes: bytes | bytearray | memoryview,
        *,
        transmitted_action_id: int,
    ) -> DirectEdgeDispatchResult:
        """Run compute then CPU publication, leaving one consumable snapshot."""

        _require(
            self._snapshot is None,
            "previous direct v3 edge snapshot was not consumed",
        )
        computed = self._runtime.process_compute(
            frame_bytes, transmitted_action_id=transmitted_action_id
        )
        published = self._runtime.publish_cpu(computed)
        self._snapshot = DetachedOptimizedTailAdapterV3.snapshot(published.serialized)
        boundaries = computed.timing.boundaries + published.timing.boundaries
        timing = TimingTrace(
            clock=computed.timing.clock,
            boundaries=boundaries,
            latency_published=False,
        )
        return DirectEdgeDispatchResult(
            perception=computed.work.perception,
            serialized_output=published.serialized.serialized_records,
            metadata=computed.metadata,
            scientific_inner_payload_bytes=computed.scientific_inner_payload_bytes,
            framing_control_overhead_bytes=computed.framing_control_overhead_bytes,
            total_received_bytes=computed.total_received_bytes,
            timing=timing,
        )

    def take_snapshot(self) -> ContextTailSnapshot:
        snapshot, self._snapshot = self._snapshot, None
        _require(snapshot is not None, "direct v3 edge produced no snapshot")
        return snapshot


def preload_direct_v3_edge(
    device: torch.device,
    *,
    deadline_guard: Callable[[str, int], None] | None = None,
) -> tuple[DirectV3Edge, DirectV3Edge, Any, list[Any]]:
    """Match ``live_pilot_runtime._preload_edge``'s four-value contract.

    The edge and the tail surface are the same object here: the direct runtime
    only ever asks the edge to ``process`` and the tail to ``take_snapshot``.
    """

    edge = preload_detached_optimized_edge_v3(device)
    direct = DirectV3Edge(edge, deadline_guard=deadline_guard)
    models = [edge.model, *edge.autoencoders.values()]
    return direct, direct, edge.ledger, models

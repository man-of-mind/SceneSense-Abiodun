"""Per-frame detached handoff for the qualified optimized FCOS tail.

The existing candidate stores its last result in a singleton ``_last`` slot
until serialization consumes it.  That is correct for serial execution but is
not safe when publication overlaps the next model call.  This adapter drains
that slot immediately into an exclusive per-frame product and crosses the
CUDA/CPU boundary on the sole compute owner.  The publication owner consumes
CPU tensors only.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextTailSnapshot,
    bind_context_service_record_identity,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import FrameContextV1
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata

from .optimized_tail import OptimizedFrozenP025TailAdapter, _require


@dataclass(frozen=True)
class DetachedTailWorkProduct:
    """Exclusive result of one tail call with publication-ready CPU tensors."""

    perception: Mapping[str, torch.Tensor]
    perception_cpu: Mapping[str, torch.Tensor]
    original_indices: torch.Tensor
    original_indices_cpu: torch.Tensor
    semantic_logits: torch.Tensor
    semantic_labels: torch.Tensor
    outputs: Mapping[str, Any]
    camera_world: torch.Tensor
    frame_context: FrameContextV1
    camera_pose_reconstruct_ns: int
    output_tensor_count: int


@dataclass(frozen=True)
class DetachedSerializedTailProduct:
    work: DetachedTailWorkProduct
    serialized_records: bytes
    record_count: int
    serialized_sha256: str


class DetachedOptimizedTailAdapter(OptimizedFrozenP025TailAdapter):
    """Optimized tail with no cross-frame singleton serialization dependency."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._compute_call_lock = threading.Lock()
        self._publication_call_lock = threading.Lock()

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        raise RuntimeError("detached tail requires compute_product()")

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        raise RuntimeError("detached tail requires serialize_product()")

    def take_snapshot(self) -> ContextTailSnapshot:
        raise RuntimeError("detached tail snapshot belongs to a serialized product")

    def compute_product(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> DetachedTailWorkProduct:
        _require(
            self._compute_call_lock.acquire(blocking=False),
            "concurrent calls into the frozen tail are forbidden",
        )
        try:
            perception = super().__call__(c2, metadata)
            snapshot = self._last
            _require(snapshot is not None, "optimized tail did not create a snapshot")
            try:
                perception_cpu = MappingProxyType(
                    {
                        name: value.detach().cpu()
                        for name, value in snapshot.perception.items()
                    }
                )
                original_indices_cpu = snapshot.original_indices.detach().cpu()
                _require(
                    all(value.device.type == "cpu" for value in perception_cpu.values()),
                    "publication perception tensors did not reach CPU",
                )
                _require(
                    original_indices_cpu.device.type == "cpu",
                    "publication indices did not reach CPU",
                )
                return DetachedTailWorkProduct(
                    perception=MappingProxyType(dict(snapshot.perception)),
                    perception_cpu=perception_cpu,
                    original_indices=snapshot.original_indices,
                    original_indices_cpu=original_indices_cpu,
                    semantic_logits=snapshot.semantic_logits,
                    semantic_labels=snapshot.semantic_labels,
                    outputs=MappingProxyType(dict(snapshot.outputs)),
                    camera_world=snapshot.camera_world,
                    frame_context=snapshot.frame_context,
                    camera_pose_reconstruct_ns=snapshot.camera_pose_reconstruct_ns,
                    output_tensor_count=snapshot.output_tensor_count,
                )
            finally:
                # The work product, not mutable adapter state, now owns this frame.
                self._last = None
        finally:
            self._compute_call_lock.release()

    def serialize_product(
        self, work: DetachedTailWorkProduct
    ) -> DetachedSerializedTailProduct:
        _require(
            self._publication_call_lock.acquire(blocking=False),
            "concurrent publication calls are forbidden",
        )
        try:
            self._ledger.bump("service_record_serialization")
            _require(
                all(value.device.type == "cpu" for value in work.perception_cpu.values()),
                "publication received a non-CPU perception tensor",
            )
            _require(
                work.original_indices_cpu.device.type == "cpu",
                "publication received non-CPU indices",
            )
            context = work.frame_context
            identity = f"{context.stream_id}:{context.frame_id}"
            rows = combined_records(
                self._base,
                {"sample_id": identity, "frame_id": context.frame_id},
                work.perception_cpu,
                work.original_indices_cpu,
            )
            records = tuple(
                bind_context_service_record_identity(record, context)
                for record in rows
            )
            for record in records:
                _require(
                    tuple(record)
                    == ("stream_id", "capture_timestamp_ns", *self._record_fields),
                    "detached service-record schema drift",
                )
                _require(
                    record["stream_id"] == context.stream_id,
                    "detached service stream drift",
                )
                _require(
                    record["frame_id"] == context.frame_id,
                    "detached service frame drift",
                )
                _require(
                    record["capture_timestamp_ns"] == context.capture_timestamp_ns,
                    "detached service timestamp drift",
                )
                for value in record.values():
                    if isinstance(value, (int, float)):
                        _require(
                            math.isfinite(float(value)),
                            "non-finite detached service-record scalar",
                        )
            serialized = json.dumps(
                records, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            return DetachedSerializedTailProduct(
                work=work,
                serialized_records=serialized,
                record_count=len(records),
                serialized_sha256=hashlib.sha256(serialized).hexdigest(),
            )
        finally:
            self._publication_call_lock.release()

    @staticmethod
    def snapshot(
        serialized: DetachedSerializedTailProduct,
    ) -> ContextTailSnapshot:
        """Materialize the established evidence view after publication."""

        work = serialized.work
        records = tuple(json.loads(serialized.serialized_records.decode("utf-8")))
        return ContextTailSnapshot(
            perception=work.perception,
            original_indices=work.original_indices,
            semantic_logits=work.semantic_logits,
            semantic_labels=work.semantic_labels,
            outputs=work.outputs,
            camera_world=work.camera_world,
            frame_context=work.frame_context,
            camera_pose_reconstruct_ns=work.camera_pose_reconstruct_ns,
            output_tensor_count=work.output_tensor_count,
            records=records,
            serialized_records=serialized.serialized_records,
        )

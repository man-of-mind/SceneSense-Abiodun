"""Detached publication handoff for the overlapped v3 edge candidate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextTailSnapshot,
    bind_context_service_record_identity,
)

from .detached_tail import (
    DetachedOptimizedTailAdapter,
    DetachedSerializedTailProduct,
    DetachedTailWorkProduct,
)
from .optimized_tail import _require
from .optimized_tail_v3 import OptimizedFrozenP025TailAdapterV3


@dataclass(frozen=True)
class DetachedSerializedTailProductV3(DetachedSerializedTailProduct):
    """Carry the records already used to produce ``serialized_records``."""

    records: tuple[dict[str, Any], ...]


class DetachedOptimizedTailAdapterV3(
    DetachedOptimizedTailAdapter, OptimizedFrozenP025TailAdapterV3
):
    """V3 compute plus a no-reparse CPU publication product."""

    def serialize_product(
        self, work: DetachedTailWorkProduct
    ) -> DetachedSerializedTailProductV3:
        _require(
            self._publication_call_lock.acquire(blocking=False),
            "concurrent publication calls are forbidden",
        )
        try:
            self._ledger.bump("service_record_serialization")
            _require(
                all(
                    value.device.type == "cpu"
                    for value in work.perception_cpu.values()
                ),
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
                    "detached v3 service-record schema drift",
                )
                _require(
                    record["stream_id"] == context.stream_id
                    and record["frame_id"] == context.frame_id
                    and record["capture_timestamp_ns"]
                    == context.capture_timestamp_ns,
                    "detached v3 service-record identity drift",
                )
                for value in record.values():
                    if isinstance(value, (int, float)):
                        _require(
                            math.isfinite(float(value)),
                            "non-finite detached v3 service-record scalar",
                        )
            serialized = json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            return DetachedSerializedTailProductV3(
                work=work,
                serialized_records=serialized,
                record_count=len(records),
                serialized_sha256=hashlib.sha256(serialized).hexdigest(),
                records=records,
            )
        finally:
            self._publication_call_lock.release()

    @staticmethod
    def snapshot(
        serialized: DetachedSerializedTailProductV3,
    ) -> ContextTailSnapshot:
        work = serialized.work
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
            records=serialized.records,
            serialized_records=serialized.serialized_records,
        )


def require_v3_mro() -> tuple[str, ...]:
    names = tuple(cls.__name__ for cls in DetachedOptimizedTailAdapterV3.__mro__)
    required = (
        "DetachedOptimizedTailAdapterV3",
        "DetachedOptimizedTailAdapter",
        "OptimizedFrozenP025TailAdapterV3",
        "OptimizedFrozenP025TailAdapterV2",
        "OptimizedFrozenP025TailAdapter",
    )
    if names[: len(required)] != required:
        raise RuntimeError(f"detached v3 method-resolution drift: {names}")
    return names


require_v3_mro()

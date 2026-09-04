"""Frozen p025 tail bound only to validated SFD1 v2 frame context."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from .frame_context import (
    FrameContextV1,
    StaticCameraRegistry,
    camera_world_matrix,
)
from .registry import DispatchContractError
from .ue_runtime import DispatchMetadata


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchContractError(message)


def _require_tree_finite(value: Any, label: str) -> int:
    count = 0
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), f"{label} contains non-finite values")
        return 1
    if isinstance(value, Mapping):
        for child in value.values():
            count += _require_tree_finite(child, label)
    elif isinstance(value, (tuple, list)):
        for child in value:
            count += _require_tree_finite(child, label)
    return count


@dataclass
class ContextTailSnapshot:
    perception: Mapping[str, torch.Tensor]
    original_indices: torch.Tensor
    semantic_logits: torch.Tensor
    semantic_labels: torch.Tensor
    outputs: Mapping[str, Any]
    camera_world: torch.Tensor
    frame_context: FrameContextV1
    output_tensor_count: int
    records: tuple[dict[str, Any], ...] | None = None
    serialized_records: bytes | None = None


class ContextualFrozenP025TailAdapter:
    """One frozen tail with static edge calibration and dynamic frame pose."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base: Any,
        camera_registry: StaticCameraRegistry,
        device: torch.device,
        ledger: Any,
    ) -> None:
        _require(isinstance(device, torch.device), "context tail device must be torch.device")
        self._model = model
        self._base = base
        self._registry = camera_registry
        self._device = device
        self._ledger = ledger
        self._last: ContextTailSnapshot | None = None
        fields = tuple(base.infer.FIELDS)
        forbidden = ("gt", "ground_truth", "target", "label_path")
        _require(
            not any(
                any(token in field.lower() for token in forbidden)
                for field in fields
            ),
            "contextual service-record schema requires evaluator-only fields",
        )
        self._record_fields = fields
        static: dict[tuple[str, str], Mapping[str, Any]] = {}
        for camera_hash, mount_hash in camera_registry.identities:
            resolved = camera_registry.resolve(camera_hash, mount_hash)
            static[(camera_hash, mount_hash)] = MappingProxyType(
                {
                    "spec": resolved,
                    "intrinsic": resolved.intrinsic(device),
                    "camera_to_ego": resolved.camera_to_ego(device),
                }
            )
        self._static = MappingProxyType(static)

    @property
    def device(self) -> torch.device:
        return self._device

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        self._ledger.bump("tail")
        _require(self._last is None, "previous contextual tail snapshot was not consumed")
        context = metadata.frame_context
        _require(context is not None, "frozen context tail requires frame context")
        _require(
            metadata.sequence_id == context.sequence_id
            and metadata.capture_timestamp_ns == context.capture_timestamp_ns,
            "tail metadata/frame-context identity mismatch",
        )
        key = (context.camera_model_sha256, context.camera_mount_sha256)
        _require(key in self._static, "tail static calibration is not preloaded")
        static = self._static[key]
        camera_world_cpu = camera_world_matrix(context, static["spec"])
        camera_world = torch.tensor(
            camera_world_cpu, dtype=torch.float64, device=self._device
        )
        calibration = {
            "intrinsic": static["intrinsic"],
            "extrinsic": camera_world,
        }
        batch = c2.unsqueeze(0) if c2.ndim == 3 else c2
        outputs = self._model.decode_tail(batch, dense=False)
        tensor_count = _require_tree_finite(outputs, "frozen tail output")
        postprocessed = self._model.postprocess(outputs, [calibration])[0]
        tensor_count += _require_tree_finite(
            postprocessed, "camera-aware postprocess output"
        )
        perception, original_indices = apply_p025_service_policy(
            {"semantic_logits": outputs["semantic_logits"]}, postprocessed
        )
        tensor_count += _require_tree_finite(perception, "p025 service output")
        source_hw = (static["spec"].source_height, static["spec"].source_width)
        semantic_logits = outputs["semantic_logits"]
        semantic_labels = F.interpolate(
            semantic_logits.float(),
            size=source_hw,
            mode="bilinear",
            align_corners=False,
        ).argmax(1)[0]
        self._last = ContextTailSnapshot(
            perception=perception,
            original_indices=original_indices,
            semantic_logits=semantic_logits,
            semantic_labels=semantic_labels,
            outputs=outputs,
            camera_world=camera_world,
            frame_context=context,
            output_tensor_count=tensor_count,
        )
        return perception

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        self._ledger.bump("service_record_serialization")
        snapshot = self._last
        _require(
            snapshot is not None and perception is snapshot.perception,
            "context tail/serializer handoff drift",
        )
        context = snapshot.frame_context
        identity = f"{context.stream_id}:{context.frame_id}"
        rows = combined_records(
            self._base,
            {"sample_id": identity, "frame_id": context.frame_id},
            snapshot.perception,
            snapshot.original_indices,
        )
        records = tuple(
            {
                "stream_id": context.stream_id,
                "capture_timestamp_ns": context.capture_timestamp_ns,
                **record,
            }
            for record in rows
        )
        for record in records:
            _require(
                tuple(record)
                == ("stream_id", "capture_timestamp_ns", *self._record_fields),
                "contextual service-record schema drift",
            )
            _require(record["stream_id"] == context.stream_id, "service stream drift")
            _require(record["frame_id"] == context.frame_id, "service frame drift")
            _require(
                record["capture_timestamp_ns"] == context.capture_timestamp_ns,
                "service timestamp drift",
            )
            for value in record.values():
                if isinstance(value, (int, float)):
                    _require(
                        math.isfinite(float(value)),
                        "non-finite contextual service-record scalar",
                    )
        serialized = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        snapshot.records = records
        snapshot.serialized_records = serialized
        return serialized

    def take_snapshot(self) -> ContextTailSnapshot:
        _require(
            self._last is not None
            and self._last.records is not None
            and self._last.serialized_records is not None,
            "context tail snapshot was not serialized",
        )
        value = self._last
        self._last = None
        return value

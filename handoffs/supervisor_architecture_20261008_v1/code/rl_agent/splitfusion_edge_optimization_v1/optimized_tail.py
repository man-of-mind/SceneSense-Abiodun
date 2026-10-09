"""Semantics-preserving optimization candidate for the frozen FCOS edge tail.

The scientific model and its original postprocessor remain untouched.  This
module supplies an independently qualified adapter which changes only two
execution details:

* camera/world geometry is decoded for post-NMS survivors rather than every
  pre-NMS candidate; and
* retained tensors cross the CUDA/CPU boundary once before service-record
  construction rather than once per scalar field.

Promotion requires bit-identical tensors, p025 indices, segmentation labels,
and serialized service bytes against the frozen production adapter.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.policy import (
    filter_consolidated_person_outputs,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    CANONICAL_SCORE_THRESHOLD,
    PERSON_INTERNAL_CLASS,
    WORLD_MATCH_RADIUS_M,
    _box_iou,
    _box_pixel_bounds,
    connected_person_components,
    person_mask_from_logits,
    validate_configuration,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    calibrate_vehicle_scores,
    combined_records,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    PERSON_RULE,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextTailSnapshot,
    ContextualFrozenP025TailAdapter,
    _require_tree_finite,
    bind_context_service_record_identity,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import camera_world_matrix
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata
from rl_agent.splitfusion_timing_diagnostic_v1.instrumented_tail import (
    CUDA_STAGES,
    SERIALIZE_STAGE,
    TAIL_STAGES,
    TOTAL_STAGE,
    DeferredCudaStageTimers,
    _WallStages,
)


# Frozen SplitFusion detector geometry. Importing the historical model module
# directly is intentionally avoided because it uses script-local imports; the
# qualified loader owns that compatibility boundary.
CONTENT_H = 432
CONTENT_W = 768
LEVELS = ("p2", "p3", "p4", "p5", "p6", "p7")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchContractError(message)


def _assign_components_numpy(
    component_labels: torch.Tensor, boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact NumPy equivalent of the frozen per-box CPU assignment loop."""

    labels = np.ascontiguousarray(
        component_labels.detach().long().cpu().numpy(), dtype=np.int64
    )
    boxes_np = np.ascontiguousarray(
        boxes.detach().double().cpu().numpy(), dtype=np.float64
    )
    if labels.ndim != 2 or boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError("component-label/box shape drift")
    height, width = labels.shape
    component_count = int(labels.max()) if labels.size else 0
    areas = np.bincount(
        labels.reshape(-1), minlength=component_count + 1
    ).astype(np.float64, copy=False)
    assignments = np.full((boxes_np.shape[0],), -1, dtype=np.int32)
    support = np.zeros((boxes_np.shape[0],), dtype=np.float32)
    for index, box_np in enumerate(boxes_np):
        box = torch.from_numpy(box_np)
        left, top, right, bottom = _box_pixel_bounds(box, height, width)
        box_area = (right - left) * (bottom - top)
        if box_area <= 0 or component_count == 0:
            continue
        intersections = np.bincount(
            labels[top:bottom, left:right].reshape(-1),
            minlength=component_count + 1,
        ).astype(np.float64, copy=False)
        intersections[0] = 0.0
        unions = float(box_area) + areas - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(unions),
            where=unions > 0,
        )
        ious[0] = 0.0
        best = int(np.argmax(ious))
        if best > 0 and float(intersections[best]) > 0.0:
            assignments[index] = best
            support[index] = np.float32(ious[best])
    return torch.from_numpy(assignments), torch.from_numpy(support)


def _consolidate_person_candidates_cpu(
    *,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    world_xy: torch.Tensor,
    component_ids: torch.Tensor,
    semantic_support: torch.Tensor,
    original_indices: torch.Tensor,
) -> torch.Tensor:
    """Frozen grid-27 consolidation over already materialized CPU tensors."""

    validate_configuration(PERSON_RULE)
    scores = scores.float()
    boxes = boxes.double()
    world_xy = world_xy.double()
    component_ids = component_ids.long()
    semantic_support = semantic_support.float()
    original = original_indices.long()
    count = scores.numel()
    if (
        boxes.shape != (count, 4)
        or world_xy.shape != (count, 2)
        or component_ids.shape != (count,)
        or semantic_support.shape != (count,)
        or original.shape != (count,)
    ):
        raise ValueError("optimized person consolidation input shape drift")
    if not bool(
        torch.isfinite(scores).all()
        and torch.isfinite(boxes).all()
        and torch.isfinite(world_xy).all()
        and torch.isfinite(semantic_support).all()
    ):
        raise FloatingPointError("non-finite optimized person consolidation input")
    if len(set(original.tolist())) != count:
        raise ValueError("optimized original indices must be unique")
    eligible = scores >= CANONICAL_SCORE_THRESHOLD
    eligible &= semantic_support >= float(PERSON_RULE["semantic_support_threshold"])
    positions = torch.where(eligible)[0].tolist()
    if len(positions) < 2:
        return torch.tensor(
            sorted(positions, key=lambda index: int(original[index])),
            dtype=torch.long,
        )
    parent = {index: index for index in positions}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for offset, left in enumerate(positions):
        for right in positions[offset + 1 :]:
            if (
                int(component_ids[left]) < 0
                or int(component_ids[left]) != int(component_ids[right])
            ):
                continue
            if (
                float(torch.linalg.vector_norm(world_xy[left] - world_xy[right]))
                > WORLD_MATCH_RADIUS_M
            ):
                continue
            if _box_iou(boxes[left], boxes[right]) >= float(
                PERSON_RULE["group_box_iou_threshold"]
            ):
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in positions:
        groups[find(index)].append(index)
    winners = [
        min(members, key=lambda index: (-float(scores[index]), int(original[index])))
        for members in groups.values()
    ]
    return torch.tensor(
        sorted(winners, key=lambda index: int(original[index])), dtype=torch.long
    )


def apply_p025_service_policy_optimized(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Exact frozen p025 policy with one consolidated GPU-to-CPU boundary."""

    scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = scores.numel()
    if scores.dtype != torch.float32 or classes.shape != (count,):
        raise ValueError("optimized post-NMS score/class contract drift")
    if any(
        not isinstance(value, torch.Tensor) or value.shape[0] != count
        for value in detections.values()
    ):
        raise ValueError("optimized post-NMS detection field alignment drift")

    person_indices = torch.where(classes == PERSON_INTERNAL_CLASS)[0]
    person_mask = person_mask_from_logits(outputs["semantic_logits"])
    components, _component_count = connected_person_components(person_mask)
    # Transfer aligned person inputs once before all classical processing.
    person_indices_cpu = person_indices.detach().cpu()
    person_scores_cpu = scores.index_select(0, person_indices).detach().cpu()
    person_boxes_cpu = (
        detections["boxes"].index_select(0, person_indices).detach().cpu()
    )
    person_world_cpu = (
        detections["world_xyz"]
        .index_select(0, person_indices)[:, :2]
        .detach()
        .cpu()
    )
    component_ids, support = _assign_components_numpy(
        components, person_boxes_cpu
    )
    retained_positions = _consolidate_person_candidates_cpu(
        scores=person_scores_cpu,
        boxes=person_boxes_cpu,
        world_xy=person_world_cpu,
        component_ids=component_ids,
        semantic_support=support,
        original_indices=person_indices_cpu,
    )
    retained_person = person_indices.index_select(
        0, retained_positions.to(person_indices.device)
    )
    vehicle = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    keep = torch.cat((vehicle, retained_person)).sort().values
    if (
        keep.ndim != 1
        or keep.dtype != torch.long
        or (keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()))
    ):
        raise RuntimeError("optimized person consolidation changed candidate ordering")

    result = {name: value.index_select(0, keep) for name, value in detections.items()}
    retained_classes = result["labels_internal"].long()
    vehicle_positions = torch.where(retained_classes != PERSON_INTERNAL_CLASS)[0]
    person_positions = torch.where(retained_classes == PERSON_INTERNAL_CLASS)[0]
    original_vehicle_indices = torch.where(classes != PERSON_INTERNAL_CLASS)[0]
    retained_vehicle_indices = keep.index_select(0, vehicle_positions)
    if not torch.equal(retained_vehicle_indices, original_vehicle_indices):
        raise RuntimeError("optimized policy filtered or reordered a vehicle")
    original_selected_scores = scores.index_select(0, keep)
    combined_scores = original_selected_scores.clone()
    combined_scores[vehicle_positions] = calibrate_vehicle_scores(
        original_selected_scores.index_select(0, vehicle_positions)
    )
    result["scores"] = combined_scores
    for name, value in detections.items():
        if name != "scores" and not torch.equal(
            result[name], value.index_select(0, keep)
        ):
            raise RuntimeError(f"optimized policy changed detection field: {name}")
    if not torch.equal(
        result["scores"].index_select(0, person_positions),
        original_selected_scores.index_select(0, person_positions),
    ):
        raise RuntimeError("optimized policy changed a retained person score")

    filtered, positions = filter_consolidated_person_outputs(result)
    p025_indices = keep.index_select(0, positions.to(keep.device))
    if not torch.equal(p025_indices, keep.index_select(0, positions)):
        raise RuntimeError("optimized p025 original-index subset drift")
    return filtered, p025_indices


def postprocess_geometry_after_nms(
    model: Any,
    outputs: Mapping[str, Any],
    calibrations: Sequence[Mapping[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Reproduce ``SplitFusionFCOS.postprocess`` but defer geometry to survivors.

    Scores, thresholding, top-k selection, box decoding and NMS execute in the
    original order. Geometry does not participate in any of those decisions,
    so it is evaluated only for the final, ordered NMS survivors.
    """

    per = outputs["detection"]["per_level"]
    anchors_by_image = outputs["anchors"]
    geometry_levels = outputs["geometry"]
    results: list[dict[str, torch.Tensor]] = []
    for image_index, calibration in enumerate(calibrations):
        image_boxes: list[torch.Tensor] = []
        image_scores: list[torch.Tensor] = []
        image_labels: list[torch.Tensor] = []
        image_level: list[torch.Tensor] = []
        image_point: list[torch.Tensor] = []
        level_offsets: list[int] = []
        level_candidate_counts: list[int] = []
        split_anchors = list(
            anchors_by_image[image_index].split(
                [level.shape[1] for level in per["cls_logits"]]
            )
        )
        for level_index, _level_name in enumerate(LEVELS):
            cls = per["cls_logits"][level_index][image_index].float()
            ctr = per["bbox_ctrness"][level_index][image_index].float()
            scores = torch.sqrt(torch.sigmoid(cls) * torch.sigmoid(ctr)).flatten()
            keep_score = scores > model.score_thresh
            candidate = torch.where(keep_score)[0]
            scores = scores[keep_score]
            if candidate.numel() > model.topk_candidates:
                scores, order = scores.topk(model.topk_candidates)
                candidate = candidate[order]
            point = torch.div(candidate, 2, rounding_mode="floor")
            label = candidate % 2
            anchors = split_anchors[level_index]
            boxes = model.box_coder.decode(
                per["bbox_regression"][level_index][image_index][point].float(),
                anchors[point].float(),
            )
            boxes = box_ops.clip_boxes_to_image(boxes, (CONTENT_H, CONTENT_W))
            level_offsets.append(sum(level_candidate_counts))
            level_candidate_counts.append(int(point.numel()))
            image_boxes.append(boxes)
            image_scores.append(scores)
            image_labels.append(label)
            image_level.append(torch.full_like(point, level_index))
            image_point.append(point)

        boxes, scores, labels = map(
            torch.cat, (image_boxes, image_scores, image_labels)
        )
        levels, points = torch.cat(image_level), torch.cat(image_point)
        keep = box_ops.batched_nms(
            boxes, scores, labels, model.nms_thresh
        )[: model.detections_per_img]
        selected_levels = levels.index_select(0, keep)
        selected_points = points.index_select(0, keep)
        selected_labels = labels.index_select(0, keep)

        position_parts: list[torch.Tensor] = []
        geometry_parts: dict[str, list[torch.Tensor]] = {}
        for level_index, _level_name in enumerate(LEVELS):
            positions = torch.where(selected_levels == level_index)[0]
            if positions.numel() == 0:
                continue
            point = selected_points.index_select(0, positions)
            label = selected_labels.index_select(0, positions)
            raw_geometry = {
                name: value[image_index]
                for name, value in geometry_levels[level_index].items()
            }
            decoded = model._decode_geometry(
                raw_geometry,
                split_anchors[level_index],
                point,
                label,
                calibration["intrinsic"],
                calibration["extrinsic"],
            )
            # The original implementation performs its FP64 world transform
            # with the complete pre-NMS level batch. CUDA's GEMM kernel choice
            # depends on that row count, causing a few last-bit differences if
            # it is run only on survivors. Preserve the original matrix shape
            # for this small transform while deferring every other geometry
            # operation. Unselected rows cannot influence selected rows.
            local_positions = keep.index_select(0, positions) - level_offsets[level_index]
            full_local = torch.zeros(
                (level_candidate_counts[level_index], 3),
                dtype=decoded["local_xyz"].dtype,
                device=decoded["local_xyz"].device,
            )
            full_local[local_positions] = decoded["local_xyz"]
            homogeneous = torch.cat(
                (
                    full_local.double(),
                    torch.ones(
                        len(full_local),
                        1,
                        device=full_local.device,
                        dtype=torch.float64,
                    ),
                ),
                dim=1,
            )
            world_full = (
                homogeneous
                @ calibration["extrinsic"].to(
                    device=full_local.device, dtype=torch.float64
                ).T
            )[:, :3]
            decoded["world_xyz"] = world_full.index_select(0, local_positions)
            position_parts.append(positions)
            for name, value in decoded.items():
                geometry_parts.setdefault(name, []).append(value)

        if position_parts:
            concatenated_positions = torch.cat(position_parts)
            restore_order = torch.argsort(concatenated_positions)
            geometry = {
                name: torch.cat(parts).index_select(0, restore_order)
                for name, parts in geometry_parts.items()
            }
        else:
            empty = selected_points
            raw_geometry = {
                name: value[image_index]
                for name, value in geometry_levels[0].items()
            }
            geometry = model._decode_geometry(
                raw_geometry,
                split_anchors[0],
                empty,
                selected_labels,
                calibration["intrinsic"],
                calibration["extrinsic"],
            )

        result = {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels_internal": selected_labels,
            "labels_canonical": selected_labels + 1,
            "level_indices": selected_levels,
            "point_indices": selected_points,
            "candidate_identity": torch.stack(
                (
                    torch.full_like(selected_levels, image_index),
                    selected_levels,
                    selected_points,
                    selected_labels,
                ),
                dim=1,
            ),
        }
        result.update(geometry)
        results.append(result)
    return results


class OptimizedFrozenP025TailAdapter(ContextualFrozenP025TailAdapter):
    """Candidate adapter; never promote without reference equivalence proof."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cuda_timers = DeferredCudaStageTimers(self._device, CUDA_STAGES)
        self._wall_timers = _WallStages()
        self._timing_active = False

    def begin_frame(self) -> None:
        self._cuda_timers.reset()
        self._wall_timers.reset()
        self._timing_active = True

    def resolve_frame(self) -> dict[str, Any]:
        if not self._timing_active:
            return {"wall_ns": {}, "cuda_ms": {}, "wall_stage_order": []}
        cuda_ms = self._cuda_timers.resolve()
        record = {
            "wall_ns": self._wall_timers.elapsed_ns,
            "cuda_ms": cuda_ms,
            "wall_stage_order": list(self._wall_timers.order),
        }
        self._timing_active = False
        return record

    def _timing_start(self, name: str) -> None:
        if self._timing_active:
            self._wall_timers.start(name)
            self._cuda_timers.start(name)

    def _timing_finish(self, name: str) -> None:
        if self._timing_active:
            self._cuda_timers.finish(name)
            self._wall_timers.finish(name)

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        self._ledger.bump("tail")
        _require(self._last is None, "previous optimized tail snapshot was not consumed")
        context = metadata.frame_context
        _require(context is not None, "optimized context tail requires frame context")
        _require(
            metadata.sequence_id == context.sequence_id
            and metadata.capture_timestamp_ns == context.capture_timestamp_ns,
            "optimized tail metadata/frame-context identity mismatch",
        )
        key = (context.camera_model_sha256, context.camera_mount_sha256)
        _require(key in self._static, "optimized tail calibration is not preloaded")
        static = self._static[key]
        self._timing_start(TOTAL_STAGE)
        self._timing_start("camera_pose_reconstruct")
        pose_started = time.perf_counter_ns()
        camera_world_cpu = camera_world_matrix(context, static["spec"])
        camera_world = torch.tensor(
            camera_world_cpu, dtype=torch.float64, device=self._device
        )
        pose_ns = time.perf_counter_ns() - pose_started
        self._timing_finish("camera_pose_reconstruct")
        calibration = {
            "intrinsic": static["intrinsic"],
            "extrinsic": camera_world,
        }
        batch = c2.unsqueeze(0) if c2.ndim == 3 else c2
        self._timing_start("decode_tail_launch")
        if self._timing_active:
            self._cuda_timers.start("decode_tail_cuda")
        outputs = self._model.decode_tail(batch, dense=False)
        if self._timing_active:
            self._cuda_timers.finish("decode_tail_cuda")
        self._timing_finish("decode_tail_launch")
        self._timing_start("finite_check_outputs")
        tensor_count = _require_tree_finite(outputs, "optimized tail output")
        self._timing_finish("finite_check_outputs")
        self._timing_start("camera_aware_postprocess")
        postprocessed = postprocess_geometry_after_nms(
            self._model, outputs, [calibration]
        )[0]
        self._timing_finish("camera_aware_postprocess")
        self._timing_start("finite_check_postprocess")
        tensor_count += _require_tree_finite(
            postprocessed, "optimized camera-aware postprocess output"
        )
        self._timing_finish("finite_check_postprocess")
        self._timing_start("p025_service_filter")
        perception, original_indices = apply_p025_service_policy_optimized(
            {"semantic_logits": outputs["semantic_logits"]}, postprocessed
        )
        self._timing_finish("p025_service_filter")
        self._timing_start("finite_check_p025")
        tensor_count += _require_tree_finite(perception, "optimized p025 service output")
        self._timing_finish("finite_check_p025")
        source_hw = (static["spec"].source_height, static["spec"].source_width)
        semantic_logits = outputs["semantic_logits"]
        self._timing_start("segmentation_upsample_argmax")
        semantic_labels = F.interpolate(
            semantic_logits.float(),
            size=source_hw,
            mode="bilinear",
            align_corners=False,
        ).argmax(1)[0]
        self._timing_finish("segmentation_upsample_argmax")
        self._last = ContextTailSnapshot(
            perception=perception,
            original_indices=original_indices,
            semantic_logits=semantic_logits,
            semantic_labels=semantic_labels,
            outputs=outputs,
            camera_world=camera_world,
            frame_context=context,
            camera_pose_reconstruct_ns=pose_ns,
            output_tensor_count=tensor_count,
        )
        self._timing_finish(TOTAL_STAGE)
        return perception

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        """Cross the CUDA/CPU boundary once, then use the frozen row builder."""

        self._timing_start(SERIALIZE_STAGE)
        self._ledger.bump("service_record_serialization")
        try:
            snapshot = self._last
            _require(
                snapshot is not None and perception is snapshot.perception,
                "optimized tail/serializer handoff drift",
            )
            context = snapshot.frame_context
            identity = f"{context.stream_id}:{context.frame_id}"
            cpu_perception = {
                name: value.detach().cpu()
                for name, value in snapshot.perception.items()
            }
            cpu_indices = snapshot.original_indices.detach().cpu()
            rows = combined_records(
                self._base,
                {"sample_id": identity, "frame_id": context.frame_id},
                cpu_perception,
                cpu_indices,
            )
            records = tuple(
                bind_context_service_record_identity(record, context) for record in rows
            )
            for record in records:
                _require(
                    tuple(record)
                    == ("stream_id", "capture_timestamp_ns", *self._record_fields),
                    "optimized service-record schema drift",
                )
                for value in record.values():
                    if isinstance(value, (int, float)):
                        _require(
                            math.isfinite(float(value)),
                            "non-finite optimized service-record scalar",
                        )
            serialized = json.dumps(
                records, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            snapshot.records = records
            snapshot.serialized_records = serialized
            return serialized
        finally:
            self._timing_finish(SERIALIZE_STAGE)


def tree_bitwise_equal(left: Any, right: Any) -> bool:
    """Strict recursive equality used by qualification code and tests."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(tree_bitwise_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(tree_bitwise_equal(a, b) for a, b in zip(left, right))
        )
    return left == right

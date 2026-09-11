"""Synchronization-light, output-preserving FCOS edge adapter candidate.

The qualified v1 adapter moved geometry behind NMS. This additive v2 candidate
keeps that algorithm and removes avoidable device/host synchronization from
the numerical-recovery checks and p025 self-audits. The frozen model, weights,
thresholds, NMS, geometry equations, ordering and service schema are unchanged.
Promotion still requires exact equality against v1 on every output surface.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    PERSON_INTERNAL_CLASS,
    connected_person_components,
    person_mask_from_logits,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.policy import (
    PERSON_SCORE_THRESHOLD,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    PERSON_RULE,
    VEHICLE_CLAMP_EPSILON,
    VEHICLE_LOGIT_BIAS,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import ContextTailSnapshot
from rl_agent.splitfusion_live_dispatch_v1.frame_context import camera_world_matrix
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata
from rl_agent.splitfusion_timing_diagnostic_v1.instrumented_tail import TOTAL_STAGE

from .optimized_tail import (
    CONTENT_H,
    CONTENT_W,
    LEVELS,
    OptimizedFrozenP025TailAdapter,
    _assign_components_numpy,
    _consolidate_person_candidates_cpu,
    _require,
)


def _tree_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            tensors.extend(_tree_tensors(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            tensors.extend(_tree_tensors(child))
    return tensors


def require_tree_finite_batched(
    value: Any,
    label: str,
    *,
    extra_checks: Sequence[torch.Tensor] = (),
) -> int:
    """Validate a tensor tree with one host synchronization per device.

    ``extra_checks`` lets callers retain non-finite structural predicates in
    the same synchronization. It does not contribute to the returned tensor
    inventory count.
    """

    tensors = _tree_tensors(value)
    checks_by_device: dict[torch.device, list[torch.Tensor]] = defaultdict(list)
    for tensor in tensors:
        checks_by_device[tensor.device].append(torch.isfinite(tensor).all())
    for check in extra_checks:
        if not isinstance(check, torch.Tensor) or check.numel() != 1:
            raise ValueError("batched finite extra check must be a scalar tensor")
        checks_by_device[check.device].append(check.bool())
    for checks in checks_by_device.values():
        if checks and not bool(torch.stack(checks).all()):
            raise DispatchContractError(f"{label} contains non-finite values")
    return len(tensors)


def _decode_recovery_geometry_without_diagnostics(
    model: Any,
    raw: Mapping[str, torch.Tensor],
    anchors: torch.Tensor,
    point_indices: torch.Tensor,
    labels: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reproduce frozen recovery arithmetic without discarded audit records.

    The complete model-output tree is checked before this function and the
    decoded tree afterwards. The historical decoder additionally computed
    per-intermediate min/max/mean diagnostics and discarded them; those reads
    forced repeated CUDA synchronization without influencing an output.
    """

    device = anchors.device
    row = torch.arange(len(point_indices), device=device)
    gathered = {name: value[point_indices, labels] for name, value in raw.items()}
    _require(
        gathered["log_dimensions"].shape[-1] == 3,
        "v2 log dimensions must have final dimension three",
    )
    _require(
        gathered["yaw"].shape[-1] == 2,
        "v2 raw yaw must have final dimension two",
    )
    with torch.autocast(device_type=device.type, enabled=False):
        logits = gathered["depth_bin_logits"].float()
        probabilities = torch.softmax(logits, dim=1)
        depth_bins = int(gathered["depth_bin_residuals"].shape[1])
        _require(
            logits.shape[1] == depth_bins + 1,
            "optimized recovery depth-bin contract drift",
        )
        bins = logits.argmax(dim=1)
        in_range = bins < depth_bins
        safe_bins = bins.clamp(max=depth_bins - 1)
        edges = model.depth_edges_m.float().to(device)
        _require(
            edges.numel() == depth_bins + 1,
            "optimized recovery depth-edge contract drift",
        )
        lower = edges[safe_bins]
        upper = edges[safe_bins + 1]
        residuals = 0.5 * torch.tanh(gathered["depth_bin_residuals"].float())
        selected_residual = residuals[row, safe_bins]
        zl, zu = torch.log1p(lower), torch.log1p(upper)
        log_depth = 0.5 * (zl + zu) + selected_residual * (zu - zl)
        depth = torch.where(
            in_range,
            torch.expm1(log_depth).clamp(0.0, 40.0),
            torch.full_like(log_depth, 40.0),
        )
        sizes = anchors[point_indices, 2] - anchors[point_indices, 0]
        centers = (anchors[point_indices, :2] + anchors[point_indices, 2:]) / 2
        physical_ray = gathered["physical_ray"].float()
        uv = centers + sizes[:, None] * physical_ray
        k = intrinsic.float().to(device)
        local = torch.stack(
            (
                depth,
                depth * (uv[:, 0] - k[0, 2]) / k[0, 0],
                depth * (k[1, 2] - uv[:, 1]) / k[1, 1],
            ),
            dim=1,
        )
        homogeneous = torch.cat(
            (
                local.double(),
                torch.ones(len(local), 1, device=device, dtype=torch.float64),
            ),
            dim=1,
        )
        world = (homogeneous @ extrinsic.to(device=device, dtype=torch.float64).T)[
            :, :3
        ]
        log_dimensions = gathered["log_dimensions"].double()
        dimensions = torch.exp(log_dimensions)
        raw_yaw = gathered["yaw"].float()
        scale = raw_yaw.abs().amax(dim=-1, keepdim=True)
        zero = scale == 0
        scaled = raw_yaw / torch.where(zero, torch.ones_like(scale), scale)
        norm = scale * torch.sqrt(
            scaled.square().sum(dim=-1, keepdim=True) + zero.to(raw_yaw.dtype)
        )
        tau = float(model._recovery_tau)
        yaw = raw_yaw / norm.clamp_min(tau)
        below_tau = norm < tau
    return {
        "local_xyz": local,
        "world_xyz": world,
        "dimensions": dimensions,
        "yaw": yaw,
        "yaw_raw_norm": norm.squeeze(-1),
        "yaw_below_tau": below_tau.squeeze(-1),
        "physical_uv": uv,
        "depth_bin": bins,
        "depth_residual": selected_residual,
        "depth": depth,
        "depth_bin_logits": logits,
        "depth_bin_probabilities": probabilities,
        "bounded_depth_residuals": residuals,
        "physical_ray_offsets": physical_ray,
        "log_dimensions": log_dimensions,
        "raw_yaw": raw_yaw,
    }


def postprocess_geometry_after_nms_v2(
    model: Any,
    outputs: Mapping[str, Any],
    calibrations: Sequence[Mapping[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """The v1 post-NMS geometry order with synchronization-light recovery."""

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
        keep = box_ops.batched_nms(boxes, scores, labels, model.nms_thresh)[
            : model.detections_per_img
        ]
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
            decoded = _decode_recovery_geometry_without_diagnostics(
                model,
                raw_geometry,
                split_anchors[level_index],
                point,
                label,
                calibration["intrinsic"],
                calibration["extrinsic"],
            )
            local_positions = (
                keep.index_select(0, positions) - level_offsets[level_index]
            )
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
            geometry = _decode_recovery_geometry_without_diagnostics(
                model,
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


def apply_p025_service_policy_v2(
    outputs: Mapping[str, Any], detections: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the frozen p025 policy without redundant tensor self-comparisons.

    Every selection and arithmetic operation is the same as v1. V1 then
    repeatedly compares each freshly indexed tensor with an identical second
    ``index_select``. Those comparisons are valuable qualification assertions,
    but they synchronize CUDA once per field when executed on every live frame.
    V2 moves that proof to the mandatory v1/v2 parity qualification.
    """

    scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = scores.numel()
    if scores.dtype != torch.float32 or classes.shape != (count,):
        raise ValueError("v2 post-NMS score/class contract drift")
    if any(
        not isinstance(value, torch.Tensor) or value.shape[0] != count
        for value in detections.values()
    ):
        raise ValueError("v2 post-NMS detection field alignment drift")

    person_indices = torch.where(classes == PERSON_INTERNAL_CLASS)[0]
    person_mask = person_mask_from_logits(outputs["semantic_logits"])
    components, _component_count = connected_person_components(person_mask)
    person_indices_cpu = person_indices.detach().cpu()
    person_scores_cpu = scores.index_select(0, person_indices).detach().cpu()
    person_boxes_cpu = detections["boxes"].index_select(0, person_indices).detach().cpu()
    person_world_cpu = (
        detections["world_xyz"]
        .index_select(0, person_indices)[:, :2]
        .detach()
        .cpu()
    )
    component_ids, support = _assign_components_numpy(components, person_boxes_cpu)
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
    result = {name: value.index_select(0, keep) for name, value in detections.items()}

    retained_classes = result["labels_internal"].long()
    vehicle_positions = torch.where(retained_classes != PERSON_INTERNAL_CLASS)[0]
    selected_scores = scores.index_select(0, keep)
    combined_scores = selected_scores.clone()
    base_vehicle_scores = selected_scores.index_select(0, vehicle_positions)
    bias = torch.tensor(
        VEHICLE_LOGIT_BIAS,
        dtype=torch.float32,
        device=base_vehicle_scores.device,
    )
    combined_scores[vehicle_positions] = torch.sigmoid(
        torch.logit(
            base_vehicle_scores.clamp(
                min=VEHICLE_CLAMP_EPSILON,
                max=1.0 - VEHICLE_CLAMP_EPSILON,
            )
        )
        + bias
    )
    result["scores"] = combined_scores

    person = retained_classes == PERSON_INTERNAL_CLASS
    p025_positions = torch.where(
        ~person | (combined_scores >= PERSON_SCORE_THRESHOLD)
    )[0]
    filtered = {
        name: value.index_select(0, p025_positions) for name, value in result.items()
    }
    return filtered, keep.index_select(0, p025_positions)


class OptimizedFrozenP025TailAdapterV2(OptimizedFrozenP025TailAdapter):
    """Synchronization-light candidate layered on the qualified v1 adapter."""

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        self._ledger.bump("tail")
        _require(self._last is None, "previous v2 tail snapshot was not consumed")
        context = metadata.frame_context
        _require(context is not None, "v2 context tail requires frame context")
        _require(
            metadata.sequence_id == context.sequence_id
            and metadata.capture_timestamp_ns == context.capture_timestamp_ns,
            "v2 tail metadata/frame-context identity mismatch",
        )
        key = (context.camera_model_sha256, context.camera_mount_sha256)
        _require(key in self._static, "v2 tail calibration is not preloaded")
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
        calibration = {"intrinsic": static["intrinsic"], "extrinsic": camera_world}
        batch = c2.unsqueeze(0) if c2.ndim == 3 else c2

        self._timing_start("decode_tail_launch")
        if self._timing_active:
            self._cuda_timers.start("decode_tail_cuda")
        outputs = self._model.decode_tail(batch, dense=False)
        if self._timing_active:
            self._cuda_timers.finish("decode_tail_cuda")
        self._timing_finish("decode_tail_launch")
        self._timing_start("finite_check_outputs")
        valid_calibration = (
            torch.isfinite(calibration["intrinsic"]).all()
            & torch.isfinite(calibration["extrinsic"]).all()
            & (calibration["intrinsic"][0, 0] != 0)
            & (calibration["intrinsic"][1, 1] != 0)
        )
        tensor_count = require_tree_finite_batched(
            outputs,
            "v2 tail output or calibration",
            extra_checks=(valid_calibration,),
        )
        self._timing_finish("finite_check_outputs")

        self._timing_start("camera_aware_postprocess")
        postprocessed = postprocess_geometry_after_nms_v2(
            self._model, outputs, [calibration]
        )[0]
        self._timing_finish("camera_aware_postprocess")
        self._timing_start("finite_check_postprocess")
        tensor_count += require_tree_finite_batched(
            postprocessed,
            "v2 camera-aware postprocess output",
            extra_checks=((postprocessed["dimensions"] > 0).all(),),
        )
        self._timing_finish("finite_check_postprocess")

        self._timing_start("p025_service_filter")
        perception, original_indices = apply_p025_service_policy_v2(
            {"semantic_logits": outputs["semantic_logits"]}, postprocessed
        )
        self._timing_finish("p025_service_filter")
        self._timing_start("finite_check_p025")
        tensor_count += require_tree_finite_batched(perception, "v2 p025 output")
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

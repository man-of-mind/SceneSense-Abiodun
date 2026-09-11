"""Output-preserving overlap candidate for the FCOS edge tail.

The v2 adapter still performs the classical person-mask connected-component
labelling serially after camera-aware detection post-processing.  Those two
operations are independent once ``decode_tail`` has produced its outputs:

* connected components depend only on the semantic logits; and
* camera-aware post-processing depends on the detection/geometry tensors.

V3 therefore prepares the semantic mask on a dedicated CUDA stream, transfers
it into a reusable pinned host buffer, and runs the unchanged deterministic
OpenCV component labeller on one CPU worker while the sole edge compute owner
continues the v2 post-processing path.  The worker is joined before p025 uses
the component labels.  No model call is concurrent and no running CUDA kernel
is preempted.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    PERSON_INTERNAL_CLASS,
    PERSON_SEMANTIC_CHANNEL,
    _box_pixel_bounds,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.policy import (
    PERSON_SCORE_THRESHOLD,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    VEHICLE_CLAMP_EPSILON,
    VEHICLE_LOGIT_BIAS,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import ContextTailSnapshot
from rl_agent.splitfusion_live_dispatch_v1.frame_context import camera_world_matrix
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import DispatchMetadata
from rl_agent.splitfusion_timing_diagnostic_v1.instrumented_tail import TOTAL_STAGE

from .optimized_tail import (
    CONTENT_H,
    CONTENT_W,
    _consolidate_person_candidates_cpu,
    _require,
)
from .optimized_tail_v2 import (
    OptimizedFrozenP025TailAdapterV2,
    _tree_tensors,
    postprocess_geometry_after_nms_v2,
)


def _connected_person_components_array(mask_cpu: torch.Tensor) -> tuple[np.ndarray, int]:
    """Return component labels without a NumPy->Torch->NumPy round trip.

    Component *numbers* are internal: downstream policy uses only equality and
    foreground identity.  SAUF already emits deterministic row-major labels,
    while the fallback may number components differently without changing any
    grouping or selected detection.
    """

    mask = np.ascontiguousarray(mask_cpu.numpy(), dtype=np.uint8)
    if hasattr(cv2, "connectedComponentsWithAlgorithm"):
        count, labels = cv2.connectedComponentsWithAlgorithm(
            mask, 8, cv2.CV_32S, cv2.CCL_SAUF
        )
    else:
        count, labels = cv2.connectedComponents(
            mask, connectivity=8, ltype=cv2.CV_32S
        )
    return np.ascontiguousarray(labels, dtype=np.int32), int(count) - 1


def _assign_components_array(
    labels: np.ndarray, boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact box/component assignment directly over the OpenCV label array."""

    boxes_np = np.ascontiguousarray(
        boxes.detach().double().cpu().numpy(), dtype=np.float64
    )
    if labels.ndim != 2 or boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError("v3 component-label/box shape drift")
    height, width = labels.shape
    component_count = int(labels.max()) if labels.size else 0
    areas = np.bincount(
        labels.reshape(-1), minlength=component_count + 1
    ).astype(np.float64, copy=False)
    assignments = np.full((boxes_np.shape[0],), -1, dtype=np.int32)
    support = np.zeros((boxes_np.shape[0],), dtype=np.float32)
    for index, box_np in enumerate(boxes_np):
        left, top, right, bottom = _box_pixel_bounds(
            torch.from_numpy(box_np), height, width
        )
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


def _finish_person_components(
    ready: torch.cuda.Event,
    mask_cpu: torch.Tensor,
    _mask_gpu_keepalive: torch.Tensor,
) -> tuple[np.ndarray, int]:
    """Wait for the asynchronous mask copy and run the frozen CPU labeller."""

    ready.synchronize()
    return _connected_person_components_array(mask_cpu)


class _DeferredFiniteValidator:
    """Run fail-closed tensor validation on one non-model CUDA stream."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._pending: list[tuple[str, torch.Tensor]] = []

    def reset(self) -> None:
        if self._pending:
            raise RuntimeError("v3 finite validation was not resolved")

    def launch(
        self,
        value: Any,
        label: str,
        *,
        extra_checks: tuple[torch.Tensor, ...] = (),
    ) -> int:
        tensors = _tree_tensors(value)
        if any(tensor.device != self._device for tensor in tensors):
            raise ValueError("v3 finite validation received a foreign device")
        current = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            checks = [torch.isfinite(tensor).all() for tensor in tensors]
            for check in extra_checks:
                if check.device != self._device or check.numel() != 1:
                    raise ValueError("v3 finite extra check contract drift")
                checks.append(check.bool())
            combined = torch.stack(checks).all()
        self._pending.append((label, combined))
        return len(tensors)

    def resolve(self) -> None:
        self._stream.synchronize()
        pending, self._pending = self._pending, []
        for label, combined in pending:
            if not bool(combined):
                raise RuntimeError(f"{label} contains non-finite values")


def apply_p025_service_policy_v3(
    detections: Mapping[str, torch.Tensor],
    components: np.ndarray,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the exact v2 p025 policy using already-computed components."""

    scores = detections["scores"]
    classes = detections["labels_internal"].long()
    count = scores.numel()
    if scores.dtype != torch.float32 or classes.shape != (count,):
        raise ValueError("v3 post-NMS score/class contract drift")
    if any(
        not isinstance(value, torch.Tensor) or value.shape[0] != count
        for value in detections.values()
    ):
        raise ValueError("v3 post-NMS detection field alignment drift")
    if components.shape != (CONTENT_H, CONTENT_W) or components.dtype != np.int32:
        raise ValueError("v3 person-component contract drift")

    person_indices = torch.where(classes == PERSON_INTERNAL_CLASS)[0]
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
    component_ids, support = _assign_components_array(components, person_boxes_cpu)
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


class OptimizedFrozenP025TailAdapterV3(OptimizedFrozenP025TailAdapterV2):
    """Overlap semantic connected components with GPU post-processing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require(self._device.type == "cuda", "v3 overlap requires a CUDA tail")
        self._component_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="splitfusion-p025-components",
        )
        self._component_stream = torch.cuda.Stream(device=self._device)
        self._component_ready = torch.cuda.Event(blocking=True)
        self._finite_validator = _DeferredFiniteValidator(self._device)
        self._person_mask_cpu = torch.empty(
            (CONTENT_H, CONTENT_W),
            dtype=torch.bool,
            device="cpu",
            pin_memory=True,
        )

    def _launch_person_components(
        self, semantic_logits: torch.Tensor
    ) -> Future[tuple[np.ndarray, int]]:
        _require(
            semantic_logits.ndim == 4
            and semantic_logits.shape[0] == 1
            and semantic_logits.shape[1] > PERSON_SEMANTIC_CHANNEL
            and tuple(semantic_logits.shape[-2:]) == (CONTENT_H, CONTENT_W),
            "v3 semantic-logit shape drift",
        )
        current = torch.cuda.current_stream(self._device)
        self._component_stream.wait_stream(current)
        with torch.cuda.stream(self._component_stream):
            mask_gpu = semantic_logits.argmax(dim=1)[0].eq(
                PERSON_SEMANTIC_CHANNEL
            )
            self._person_mask_cpu.copy_(mask_gpu, non_blocking=True)
            self._component_ready.record(self._component_stream)
        return self._component_executor.submit(
            _finish_person_components,
            self._component_ready,
            self._person_mask_cpu,
            mask_gpu,
        )

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        self._ledger.bump("tail")
        _require(self._last is None, "previous v3 tail snapshot was not consumed")
        context = metadata.frame_context
        _require(context is not None, "v3 context tail requires frame context")
        _require(
            metadata.sequence_id == context.sequence_id
            and metadata.capture_timestamp_ns == context.capture_timestamp_ns,
            "v3 tail metadata/frame-context identity mismatch",
        )
        key = (context.camera_model_sha256, context.camera_mount_sha256)
        _require(key in self._static, "v3 tail calibration is not preloaded")
        static = self._static[key]
        self._finite_validator.reset()
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
        components_future = self._launch_person_components(
            outputs["semantic_logits"]
        )
        valid_calibration = (
            torch.isfinite(calibration["intrinsic"]).all()
            & torch.isfinite(calibration["extrinsic"]).all()
            & (calibration["intrinsic"][0, 0] != 0)
            & (calibration["intrinsic"][1, 1] != 0)
        )
        tensor_count = self._finite_validator.launch(
            outputs,
            "v3 tail output or calibration",
            extra_checks=(valid_calibration,),
        )
        self._timing_finish("finite_check_outputs")

        self._timing_start("camera_aware_postprocess")
        postprocessed = postprocess_geometry_after_nms_v2(
            self._model, outputs, [calibration]
        )[0]
        self._timing_finish("camera_aware_postprocess")
        self._timing_start("finite_check_postprocess")
        tensor_count += self._finite_validator.launch(
            postprocessed,
            "v3 camera-aware postprocess output",
            extra_checks=((postprocessed["dimensions"] > 0).all(),),
        )
        self._timing_finish("finite_check_postprocess")

        self._timing_start("p025_service_filter")
        components, _component_count = components_future.result()
        perception, original_indices = apply_p025_service_policy_v3(
            postprocessed, components
        )
        self._timing_finish("p025_service_filter")
        self._timing_start("finite_check_p025")
        tensor_count += self._finite_validator.launch(
            perception, "v3 p025 output"
        )
        self._finite_validator.resolve()
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
